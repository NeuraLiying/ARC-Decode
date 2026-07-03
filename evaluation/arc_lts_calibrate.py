#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("PYTHONPYCACHEPREFIX", "/tmp/eagle_pycache")

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ARC_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ARC_ROOT / "data"

MODEL_SPECS = {
    "llama": {
        "base": os.environ.get("ARC_LLAMA_BASE", "/path-to-llama"),
        "out": ARC_ROOT / "evaluation" / "calibration" / "lts_llama_params_t_1.pt",
        "dtype": "float16",
    },
    "qwen": {
        "base": os.environ.get("ARC_QWEN_BASE", "/path-to-qwen"),
        "out": ARC_ROOT / "evaluation" / "calibration" / "lts_qwen_params_t_1.pt",
        "dtype": "float16",
    },
    "vicuna": {
        "base": os.environ.get("ARC_VICUNA_BASE", "/path-to-vicuna"),
        "out": ARC_ROOT / "evaluation" / "calibration" / "lts_vicuna_params_t_1.pt",
        "dtype": "float16",
    },
}

EPS = 1e-12


@dataclass
class LTSFittedParams:
    c_s_prime: float
    alpha_kappa: float
    tau_delta: float
    margin_bound: float
    logit_bound: float
    budget_table: Optional[List[float]]
    delta: float
    topk_js: int
    lts_theta: float
    parent_topk: int
    parent_min_keep: int
    margin_relax: float
    path_margin_scale: float
    path_logit_scale: float


def dtype_from_name(name: str):
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    return torch.float16


def read_jsonl(path: Path, limit: int = 0) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def pick_real_device(model) -> torch.device:
    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def prepare_logits(logits: torch.Tensor, temperature: float, top_p: float, top_k: int) -> torch.Tensor:
    logits = logits.float()
    if temperature > 1e-5 and temperature != 1.0:
        logits = logits / temperature
    if top_k > 0 and top_k < logits.numel():
        vals, idx = torch.topk(logits, top_k)
        masked = torch.full_like(logits, torch.finfo(logits.dtype).min)
        logits = masked.scatter(0, idx, vals)
    if 1e-8 <= top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(sorted_probs, dim=-1)
        remove = cum > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        logits[sorted_idx[remove]] = torch.finfo(logits.dtype).min
    return logits


def with_processor_softmax(logits: torch.Tensor, temperature: float, top_p: float, top_k: int) -> torch.Tensor:
    logits = prepare_logits(logits, temperature, top_p, top_k)
    return torch.softmax(logits, dim=-1)


def topk_union_normalize(p: torch.Tensor, q: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if k <= 0 or k >= p.numel():
        return p / (p.sum() + EPS), q / (q.sum() + EPS)
    pa = torch.topk(p, k=min(k, p.numel())).indices
    qa = torch.topk(q, k=min(k, q.numel())).indices
    idx = torch.unique(torch.cat([pa, qa], dim=0))
    p2 = p[idx]
    q2 = q[idx]
    return p2 / (p2.sum() + EPS), q2 / (q2.sum() + EPS)


def js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    m = 0.5 * (p + q)
    kl_pm = torch.sum(p * (torch.log(torch.clamp(p, min=EPS)) - torch.log(torch.clamp(m, min=EPS))))
    kl_qm = torch.sum(q * (torch.log(torch.clamp(q, min=EPS)) - torch.log(torch.clamp(m, min=EPS))))
    return 0.5 * (kl_pm + kl_qm)


def compute_inv_std_cpu(emb_table: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    std = emb_table.detach().float().cpu().std(dim=0)
    return 1.0 / (std + eps)


def pick_t_d(target_probs: torch.Tensor, forbid_id: int, strategy: str) -> int:
    if strategy == "sample":
        probs = target_probs.clone()
        probs[forbid_id] = 0.0
        denom = probs.sum()
        if denom <= 0:
            return int(torch.argmax(target_probs).item())
        return int(torch.multinomial(probs / denom, 1).item())
    vals, idx = torch.topk(target_probs, k=min(4, target_probs.numel()))
    for token_id in idx.tolist():
        if int(token_id) != int(forbid_id):
            return int(token_id)
    return int(idx[-1].item())


def sample_next_token(step_probs: torch.Tensor, temperature: float) -> int:
    if temperature <= 1e-5:
        return int(torch.argmax(step_probs).item())
    return int(torch.multinomial(step_probs, 1).item())


def build_messages(family: str, question: str) -> List[Dict[str, str]]:
    if family == "qwen":
        system = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    else:
        system = "You are a helpful, honest, and harmless assistant."
    return [{"role": "system", "content": system}, {"role": "user", "content": question}]


def apply_chat(tokenizer, family: str, messages: List[Dict[str, str]]) -> str:
    if family == "vicuna":
        parts = []
        for msg in messages:
            if msg["role"] == "system":
                if msg["content"].strip():
                    parts.append(msg["content"].strip())
            elif msg["role"] == "user":
                parts.append(f"USER: {msg['content'].strip()}")
            else:
                parts.append(f"ASSISTANT: {msg['content'].strip()}</s>")
        parts.append("ASSISTANT:")
        return " ".join(parts)
    try:
        if family == "qwen":
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def collect_samples(
    model,
    tokenizer,
    family: str,
    questions: List[dict],
    temperature: float,
    top_p: float,
    top_k: int,
    max_positions_per_turn: int,
    gen_steps_per_turn: int,
    max_prompt_tokens: int,
    topk_js: int,
    td_strategy: str,
) -> Dict[str, List[float]]:
    device = pick_real_device(model)
    emb_table = model.get_input_embeddings().weight
    inv_std = compute_inv_std_cpu(emb_table)
    inv_std_dev = inv_std.to(device=emb_table.device, dtype=emb_table.dtype)
    eos_id = tokenizer.eos_token_id

    diffs, u_emb_raws, u_logit_raws, delta_logs = [], [], [], []

    for q_idx, row in enumerate(questions):
        turns = row.get("turns", [])
        if not turns:
            continue
        prompt = apply_chat(tokenizer, family, build_messages(family, turns[0]))
        ids = tokenizer([prompt], add_special_tokens=False).input_ids[0]
        if max_prompt_tokens > 0 and len(ids) > max_prompt_tokens:
            ids = ids[-max_prompt_tokens:]
        cur_ids = torch.as_tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        collected = 0

        for _ in range(gen_steps_per_turn):
            out = model(input_ids=cur_ids, use_cache=False)
            step_probs = with_processor_softmax(out.logits[0, -1], temperature, top_p, top_k)
            t_m = int(torch.argmax(step_probs).item())
            t_d = pick_t_d(step_probs, t_m, td_strategy)

            if t_d != t_m:
                log_tm = torch.log(torch.clamp(step_probs[t_m], min=EPS))
                log_td = torch.log(torch.clamp(step_probs[t_d], min=EPS))
                delta_log = float((log_tm - log_td).item())
                u_logit_raw = float(delta_log * delta_log)

                e_tm = emb_table[t_m]
                e_td = emb_table[t_d]
                diff = (e_td - e_tm) * inv_std_dev
                u_emb_raw = float(torch.sum(diff * diff).item())

                ids_tm = torch.cat([cur_ids[0], torch.tensor([t_m], device=device)], dim=0).unsqueeze(0)
                ids_td = torch.cat([cur_ids[0], torch.tensor([t_d], device=device)], dim=0).unsqueeze(0)
                out_tm = model(input_ids=ids_tm, use_cache=False)
                out_td = model(input_ids=ids_td, use_cache=False)
                p_tm = with_processor_softmax(out_tm.logits[0, -1], temperature, top_p, top_k)
                p_td = with_processor_softmax(out_td.logits[0, -1], temperature, top_p, top_k)
                pa, pb = topk_union_normalize(p_tm, p_td, topk_js)
                diff_js = float(js_divergence(pa, pb).item())

                if all(math.isfinite(x) for x in [diff_js, u_emb_raw, u_logit_raw, delta_log]):
                    diffs.append(diff_js)
                    u_emb_raws.append(u_emb_raw)
                    u_logit_raws.append(u_logit_raw)
                    delta_logs.append(delta_log)
                    collected += 1

            next_tok = sample_next_token(step_probs, temperature)
            cur_ids = torch.cat([cur_ids, torch.tensor([[next_tok]], dtype=torch.long, device=device)], dim=1)
            if eos_id is not None and next_tok == eos_id:
                break
            if collected >= max_positions_per_turn:
                break

        print(f"[collect] {family} question {q_idx + 1}/{len(questions)} total_samples={len(diffs)}", flush=True)

    return {
        "Diff": diffs,
        "U_emb_raw": u_emb_raws,
        "U_logit_raw": u_logit_raws,
        "delta_log": delta_logs,
    }


def quantile_robust(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0
    values.sort()
    hi = int(max(0, min(values.size - 1, round(0.995 * (values.size - 1)))))
    return float(np.quantile(values[: hi + 1], q))


def fit_scales_and_bounds(samples: Dict[str, List[float]], delta: float, lts_theta: float):
    diff = np.asarray(samples["Diff"], dtype=np.float64)
    ue = np.asarray(samples["U_emb_raw"], dtype=np.float64)
    ul = np.asarray(samples["U_logit_raw"], dtype=np.float64)
    dlog = np.asarray(samples["delta_log"], dtype=np.float64)

    re = diff[ue > 1e-20] / ue[ue > 1e-20]
    rl = diff[ul > 1e-20] / ul[ul > 1e-20]
    c_s_prime = quantile_robust(re, 1.0 - delta)
    alpha_kappa = quantile_robust(rl, 1.0 - delta)
    u_min = np.minimum(c_s_prime * ue, alpha_kappa * ul)
    u_bound = quantile_robust(u_min, 1.0 - delta)
    tau_delta = float(u_bound / max(1.0 - float(lts_theta), 1e-8))
    margin_bound = quantile_robust(dlog, 1.0 - delta)
    logit_bound = quantile_robust(alpha_kappa * ul, 1.0 - delta)
    return c_s_prime, alpha_kappa, tau_delta, margin_bound, logit_bound


def build_budget_table(samples, c_s_prime, alpha_kappa, delta, max_l, bootstrap_n, seed):
    u = np.minimum(
        np.asarray(samples["U_emb_raw"], dtype=np.float64) * c_s_prime,
        np.asarray(samples["U_logit_raw"], dtype=np.float64) * alpha_kappa,
    )
    u = u[np.isfinite(u)]
    u = u[u >= 0]
    if u.size == 0:
        return [0.0] * (max_l + 1)
    rng = np.random.default_rng(seed)
    table = [0.0] * (max_l + 1)
    for length in range(1, max_l + 1):
        draws = rng.choice(u, size=(bootstrap_n, length), replace=True)
        table[length] = float(np.quantile(draws.sum(axis=1), 1.0 - delta))
    return table


def save_params(out_path: Path, params: LTSFittedParams, inv_std: torch.Tensor, emb_table: torch.Tensor, n_samples: int):
    obj = {
        "params": {
            "c_s_prime": params.c_s_prime,
            "alpha_kappa": params.alpha_kappa,
            "tau_delta": params.tau_delta,
            "margin_bound": params.margin_bound,
            "logit_bound": params.logit_bound,
            "budget_table": params.budget_table,
            "delta": params.delta,
            "topk_js": params.topk_js,
            "lts_theta": params.lts_theta,
            "parent_topk": params.parent_topk,
            "parent_min_keep": params.parent_min_keep,
            "margin_relax": params.margin_relax,
            "path_margin_scale": params.path_margin_scale,
            "path_logit_scale": params.path_logit_scale,
        },
        "inv_std": inv_std.cpu(),
        "meta": {
            "temperature": 1.0,
            "n_samples": n_samples,
            "emb_dtype": str(emb_table.dtype),
            "emb_device_hint": str(emb_table.device),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--topk-js", type=int, default=128)
    parser.add_argument("--max-questions", type=int, default=50)
    parser.add_argument("--max-positions-per-turn", type=int, default=32)
    parser.add_argument("--gen-steps-per-turn", type=int, default=64)
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    parser.add_argument("--td-strategy", choices=["top2", "sample"], default="top2")
    parser.add_argument("--max-budget-l", type=int, default=32)
    parser.add_argument("--bootstrap-n", type=int, default=20000)
    parser.add_argument("--no-budget", action="store_true")
    parser.add_argument("--lts-theta", type=float, default=0.3)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="")
    parser.add_argument("--device-map", default="auto")
    args = parser.parse_args()

    if abs(args.temperature - 1.0) > 1e-8:
        raise SystemExit("This calibration is for T=1.0; pass --temperature 1.0.")

    spec = MODEL_SPECS[args.model]
    dtype = dtype_from_name(args.dtype or spec["dtype"])
    out_path = Path(args.output).resolve() if args.output else Path(spec["out"]).resolve()

    print(f"[load] {args.model}: {spec['base']}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(spec["base"], use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        spec["base"],
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=args.device_map,
    )
    model.eval()

    questions = read_jsonl(DATA_ROOT / "mt_bench" / "question.jsonl", args.max_questions)
    samples = collect_samples(
        model,
        tokenizer,
        args.model,
        questions,
        args.temperature,
        args.top_p,
        args.top_k,
        args.max_positions_per_turn,
        args.gen_steps_per_turn,
        args.max_prompt_tokens,
        args.topk_js,
        args.td_strategy,
    )
    n = len(samples["Diff"])
    print(f"[fit] samples={n}", flush=True)
    if n == 0:
        raise SystemExit("No calibration samples collected.")

    c_s_prime, alpha_kappa, tau_delta, margin_bound, logit_bound = fit_scales_and_bounds(
        samples, args.delta, args.lts_theta
    )
    budget = None
    if not args.no_budget:
        budget = build_budget_table(
            samples, c_s_prime, alpha_kappa, args.delta, args.max_budget_l, args.bootstrap_n, seed=123
        )

    emb_table = model.get_input_embeddings().weight
    inv_std = compute_inv_std_cpu(emb_table)
    params = LTSFittedParams(
        c_s_prime=c_s_prime,
        alpha_kappa=alpha_kappa,
        tau_delta=tau_delta,
        margin_bound=margin_bound,
        logit_bound=logit_bound,
        budget_table=budget,
        delta=args.delta,
        topk_js=args.topk_js,
        lts_theta=args.lts_theta,
        parent_topk=8,
        parent_min_keep=1,
        margin_relax=1.25,
        path_margin_scale=1.0,
        path_logit_scale=1.0,
    )
    save_params(out_path, params, inv_std, emb_table, n)
    print(
        json.dumps(
            {
                "output": str(out_path),
                "samples": n,
                "c_s_prime": c_s_prime,
                "alpha_kappa": alpha_kappa,
                "tau_delta": tau_delta,
                "margin_bound": margin_bound,
                "logit_bound": logit_bound,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
