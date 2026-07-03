#!/usr/bin/env python3
import argparse
import json
import math
import multiprocessing as mp
import os
import random
import re
import sys
import textwrap
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("PYTHONPYCACHEPREFIX", "/tmp/eagle_pycache")

ARC_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ARC_ROOT / "model"
for path in (str(ARC_ROOT), str(MODEL_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import torch

from model.ea_model import EaModel


MODEL_SPECS = {
    "llama": {
        "base": os.environ.get("ARC_LLAMA_BASE", "/path-to-llama"),
        "draft": os.environ.get("ARC_LLAMA_DRAFT", "/path-to-llama-draft"),
        "model_id": "llama3.1-8b-instruct-arc-decode",
        "dtype": "float16",
        "lts": ARC_ROOT / "evaluation" / "calibration" / "lts_llama_params_t_1.pt",
    },
    "qwen": {
        "base": os.environ.get("ARC_QWEN_BASE", "/path-to-qwen"),
        "draft": os.environ.get("ARC_QWEN_DRAFT", "/path-to-qwen-draft"),
        "model_id": "qwen3-8b-arc-decode",
        "dtype": "float16",
        "lts": ARC_ROOT / "evaluation" / "calibration" / "lts_qwen_params_t_1.pt",
    },
    "vicuna": {
        "base": os.environ.get("ARC_VICUNA_BASE", "/path-to-vicuna"),
        "draft": os.environ.get("ARC_VICUNA_DRAFT", "/path-to-vicuna-draft"),
        "model_id": "vicuna-13b-v1.3-arc-decode",
        "dtype": "float16",
        "lts": ARC_ROOT / "evaluation" / "calibration" / "lts_vicuna_params_t_1.pt",
    },
}

DATA_ROOT = ARC_ROOT / "data"
HUMANEVAL_OFFICIAL = DATA_ROOT / "humaneval" / "humaneval_official.jsonl"
NUM_RE = re.compile(r"-?\d+\.\d+|-?\d+")


def read_jsonl(path: Path, limit: int = 0) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def dtype_from_name(name: str):
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    return torch.float16


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def first_param_device(model: EaModel) -> torch.device:
    for p in model.base_model.parameters():
        if p.device.type != "meta":
            return p.device
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def move_ea_layer_to_real_device(model: EaModel) -> None:
    real_dev = first_param_device(model)
    model.ea_layer = model.ea_layer.to(model.base_model.dtype).to(real_dev)
    model.ea_layer.init_tree()


def reset_tree_budget(model: EaModel, total_tokens: int, depth: int) -> None:
    model.ea_layer.total_tokens = int(total_tokens) - 1
    model.ea_layer.depth = int(depth)
    model.ea_layer.init_tree()


def clean_special(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    for token in [
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|eot_id|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|im_start|>",
        "<|im_end|>",
    ]:
        text = text.replace(token, "")
    text = re.sub(r"<\|[a-zA-Z0-9_]+?\|>", "", text)
    return text.replace("```python", "").replace("```", "").replace("\r\n", "\n").strip()


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
        if parts and parts[-1].startswith("USER:"):
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


def stop_ids_for(tokenizer, family: str) -> List[int]:
    ids = []
    if tokenizer.eos_token_id is not None:
        ids.append(int(tokenizer.eos_token_id))
    if family == "llama":
        try:
            eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if isinstance(eot, int) and eot >= 0:
                ids.append(eot)
        except Exception:
            pass
    return list(dict.fromkeys(ids))


def strip_stop_tokens(output_ids: torch.Tensor, stop_ids: List[int]) -> torch.Tensor:
    if not stop_ids:
        return output_ids
    values = output_ids.tolist()
    for idx, token_id in enumerate(values):
        if int(token_id) in stop_ids:
            return output_ids[:idx]
    return output_ids


def normalize_number(s: str) -> str:
    s = s.replace(",", "").strip()
    try:
        d = Decimal(s)
        s = format(d.normalize(), "f")
        if s.endswith("."):
            s = s[:-1]
    except InvalidOperation:
        pass
    return s


def extract_last_number(text: str) -> str:
    nums = NUM_RE.findall(clean_special(text))
    return normalize_number(nums[-1]) if nums else ""


def extract_gsm_answer(reference: str) -> str:
    if "####" in reference:
        return normalize_number(reference.split("####")[-1])
    return extract_last_number(reference)


def build_messages(family: str, benchmark: str, prompt: str) -> List[Dict[str, str]]:
    if family == "qwen":
        system = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    elif benchmark == "humaneval":
        system = (
            "You are a helpful coding assistant. Write only the Python function body. "
            "Do not repeat the def line. Do not include markdown."
        )
    else:
        system = "You are a helpful, respectful and honest assistant."

    if benchmark == "gsm8k":
        prompt = f"Question: {prompt}\nLet's think step by step\nAnswer:"
    elif benchmark == "humaneval":
        prompt = "Complete the following Python function. Return only valid Python code for the function body.\n\n" + prompt
    return [{"role": "system", "content": system}, {"role": "user", "content": prompt}]


def build_mt_messages(family: str, question: Dict[str, Any], prior_turns: List[str], turn_idx: int):
    system = (
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        if family == "qwen"
        else "You are a helpful, respectful and honest assistant."
    )
    messages = [{"role": "system", "content": system}]
    for idx in range(turn_idx + 1):
        messages.append({"role": "user", "content": question["turns"][idx]})
        if idx < turn_idx:
            messages.append({"role": "assistant", "content": prior_turns[idx]})
    return messages


def encode_prompt(tokenizer, prompt: str, device: torch.device) -> torch.Tensor:
    ids = tokenizer([prompt], add_special_tokens=False).input_ids
    return torch.as_tensor(ids, dtype=torch.long, device=device)


def strip_leading_docstring(code: str) -> str:
    code = code.lstrip()
    match = re.match(r"^[rRuUfFbB]*(\"\"\"|''')", code)
    if not match:
        return code
    quote = match.group(1)
    end = code.find(quote, match.end())
    return code[end + len(quote):].lstrip() if end >= 0 else code


def cut_trailing_prose(lines: List[str]) -> List[str]:
    prose = re.compile(
        r"^(#{1,6}\s*)?"
        r"(explanation|note|notes|this function|the function|here is|here's|"
        r"in this|we use|finally,|overall,|time complexity|space complexity)\b",
        re.I,
    )
    kept = []
    for line in lines:
        stripped = line.strip()
        if stripped and (prose.search(stripped) or stripped.startswith("---")):
            break
        kept.append(line)
    return kept


def normalize_body_indentation(body: str) -> str:
    lines = body.replace("\t", "    ").splitlines()
    nonempty = [line for line in lines if line.strip()]
    if nonempty:
        indents = [len(line) - len(line.lstrip(" ")) for line in nonempty]
        positive = [indent for indent in indents if indent > 0]
        if positive and len(positive) >= max(1, len(nonempty) - 1):
            shift = min(positive)
            lines = [line[shift:] if line.startswith(" " * shift) else line for line in lines]
    return textwrap.dedent("\n".join(lines)).strip("\n")


def extract_func_body(generated: str, entry_point: str) -> str:
    code = clean_special(generated)
    lines = code.splitlines()
    header_pat = rf"^\s*def\s+{re.escape(entry_point)}\s*\([^)]*\)\s*(?:->\s*[^:]+)?\s*:\s*$"
    header_idx = next((idx for idx, line in enumerate(lines) if re.match(header_pat, line)), None)

    if header_idx is not None:
        body_lines = []
        for line in lines[header_idx + 1:]:
            if not line.strip():
                body_lines.append(line)
                continue
            if not line.startswith((" ", "\t")):
                break
            body_lines.append(line)
        body = strip_leading_docstring("\n".join(body_lines))
    else:
        body_lines = lines
        code_start = re.compile(
            r"^\s*(from\s+\S+\s+import\s+|import\s+|return\b|if\b|for\b|while\b|"
            r"try\b|with\b|raise\b|assert\b|[A-Za-z_]\w*\s*=|[A-Za-z_]\w*\()"
        )
        first_code = next((idx for idx, line in enumerate(body_lines) if code_start.match(line)), 0)
        body = "\n".join(body_lines[first_code:])
        body = strip_leading_docstring(body)

    pruned = []
    seen_code = False
    for line in body.splitlines():
        stripped = line.strip()
        if not seen_code and (stripped.startswith("import ") or stripped.startswith("from ")):
            continue
        if stripped:
            seen_code = True
        pruned.append(line)
    body = normalize_body_indentation("\n".join(cut_trailing_prose(pruned)))
    return body if body.strip() else "pass"


def build_humaneval_program(prompt: str, body: str) -> str:
    body = textwrap.dedent(body or "").strip("\n") or "pass"
    return prompt.rstrip() + "\n" + textwrap.indent(body, "    ") + "\n"


def humaneval_worker(program: str, test: str, entry_point: str, queue) -> None:
    try:
        ns: Dict[str, Any] = {}
        exec(program + "\n" + test + f"\ncheck({entry_point})\n", ns)
        queue.put(True)
    except BaseException:
        queue.put(False)


def run_humaneval_test(program: str, test: str, entry_point: str, timeout_s: float) -> bool:
    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    proc = ctx.Process(target=humaneval_worker, args=(program, test, entry_point, queue))
    proc.start()
    proc.join(timeout_s)
    if proc.is_alive():
        proc.kill()
        proc.join()
        return False
    try:
        return bool(queue.get_nowait())
    except Exception:
        return False


def eval_data_path(benchmark: str, full_data: bool) -> Tuple[Path, str]:
    if full_data and benchmark in {"gsm8k", "alpaca"}:
        return DATA_ROOT / benchmark / "full_question.jsonl", "full"
    return DATA_ROOT / benchmark / "question.jsonl", "subset" if benchmark in {"gsm8k", "alpaca"} else "full"


def load_eval_rows(benchmark: str, limit: int, full_data: bool) -> Tuple[List[Dict[str, Any]], Path, str]:
    if benchmark == "humaneval":
        rows = read_jsonl(HUMANEVAL_OFFICIAL, limit)
        for row in rows:
            row["turns"] = [row["prompt"]]
        return rows, HUMANEVAL_OFFICIAL, "full"
    path, scope = eval_data_path(benchmark, full_data)
    return read_jsonl(path, limit), path, scope


def load_lts_ckpt(model: EaModel, ckpt_path: Path, args) -> Dict[str, Any]:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"LTS ckpt not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    emb_w = model.base_model.model.embed_tokens.weight
    inv_std = ckpt["inv_std"].to(device=emb_w.device, dtype=emb_w.dtype)
    params = ckpt.get("params", ckpt)
    if args.lts_param_mode == "defaults":
        c_s_prime = 1.0
        alpha_kappa = 1.0
        tau_delta = 1.0
        budget_table = None
    else:
        c_s_prime = float(params.get("c_s_prime", 1.0))
        alpha_kappa = float(params.get("alpha_kappa", 1.0))
        tau_delta = float(params.get("tau_delta", 1.0))
        budget_table = params.get("budget_table")
    use_budget = (budget_table is not None) if args.lts_use_budget is None else bool(args.lts_use_budget)
    model.set_lts_params(
        c_s_prime=c_s_prime,
        alpha_kappa=alpha_kappa,
        tau_delta=tau_delta,
        inv_std=inv_std,
        theta=args.lts_theta,
        budget_table=budget_table,
        deny_token_strs=args.lts_deny_token_strs,
        use_budget=use_budget,
        parent_topk=int(params.get("parent_topk", 8)),
        parent_min_keep=int(params.get("parent_min_keep", 1)),
        margin_relax=float(params.get("margin_relax", 1.25)),
        path_margin_scale=float(params.get("path_margin_scale", 1.0)),
        path_logit_scale=float(params.get("path_logit_scale", 1.0)),
    )
    model.lts_cfg.lts_lambda = float(args.lts_lambda)
    model.lts_cfg.min_boost_prob = float(args.lts_min_boost_prob)
    model.lts_cfg.pf_lts_record_stats = bool(args.pf_lts_record_stats)
    model.lts_cfg.pf_lts_selective = bool(args.pf_lts_selective)
    model.lts_cfg.pf_lts_min_acp_default = float(args.pf_lts_min_acp_default)
    model.lts_cfg.pf_lts_min_acp_code = float(args.pf_lts_min_acp_code)
    model.lts_cfg.pf_lts_min_acp_math = float(args.pf_lts_min_acp_math)
    model.lts_cfg.pf_lts_max_u_ratio_default = float(args.pf_lts_max_u_ratio_default)
    model.lts_cfg.pf_lts_max_u_ratio_code = float(args.pf_lts_max_u_ratio_code)
    model.lts_cfg.pf_lts_max_u_ratio_math = float(args.pf_lts_max_u_ratio_math)
    model._eval_mix_ratio = float(args.mix_ratio)
    model._use_pf_lts = bool(args.pf_lts)
    return {
        "ckpt": str(ckpt_path),
        "param_mode": args.lts_param_mode,
        "c_s_prime": c_s_prime,
        "alpha_kappa": alpha_kappa,
        "tau_delta": tau_delta,
        "theta": args.lts_theta,
        "lts_lambda": args.lts_lambda,
        "min_boost_prob": args.lts_min_boost_prob,
        "pf_lts_record_stats": bool(args.pf_lts_record_stats),
        "pf_lts_selective": bool(args.pf_lts_selective),
        "pf_lts_min_acp": {
            "default": args.pf_lts_min_acp_default,
            "code": args.pf_lts_min_acp_code,
            "math": args.pf_lts_min_acp_math,
        },
        "pf_lts_max_u_ratio": {
            "default": args.pf_lts_max_u_ratio_default,
            "code": args.pf_lts_max_u_ratio_code,
            "math": args.pf_lts_max_u_ratio_math,
        },
        "mix_ratio": args.mix_ratio,
        "pf_lts": bool(args.pf_lts),
        "use_budget": use_budget,
    }


def scaled_adaptive_templates(total_tokens: int, depth: int) -> Dict[str, Dict[str, float]]:
    return {
        "deep": {"depth": int(depth), "total_token": int(total_tokens), "h_max": 3.0},
        "base": {"depth": int(depth), "total_token": max(2, int(round(total_tokens * 0.75))), "h_max": 5.5},
        "shallow": {
            "depth": max(1, int(round(depth * 2.0 / 3.0))),
            "total_token": max(2, int(round(total_tokens * 0.5))),
            "h_max": 1.0e9,
        },
    }


def apply_adaptive_lts_lambdas(templates: Dict[str, Dict[str, float]], args) -> None:
    for name in ("deep", "base", "shallow"):
        value = getattr(args, f"adaptive_lts_lambda_{name}", None)
        if value is not None and name in templates:
            templates[name]["lts_lambda"] = float(value)
        code_value = getattr(args, f"adaptive_lts_code_lambda_{name}", None)
        if code_value is not None and name in templates:
            templates[name]["code_lts_lambda"] = float(code_value)
        math_value = getattr(args, f"adaptive_lts_math_lambda_{name}", None)
        if math_value is not None and name in templates:
            templates[name]["math_lts_lambda"] = float(math_value)


def looks_code_like(prompt: str) -> bool:
    lowered = prompt.lower()
    markers = (
        "```",
        "\ndef ",
        "\nclass ",
        "\nimport ",
        "\nfrom ",
        "assert ",
        "write a function",
        "complete the function",
    )
    return any(marker in lowered for marker in markers)


def looks_strict_code_completion(prompt: str) -> bool:
    lowered = prompt.lower()
    markers = (
        "complete the following python function",
        "return only valid python code for the function body",
        "write only the python function body",
        "do not repeat the def line",
    )
    return any(marker in lowered for marker in markers)


def looks_math_like(prompt: str) -> bool:
    lowered = prompt.lower()
    if looks_strict_math_prompt(prompt):
        return True
    math_markers = (
        "calculate",
        "how many",
        "how much",
        "what is the total",
        "percent",
        "percentage",
        "ratio",
        "average",
        "probability",
        "dollars",
        "minutes",
        "hours",
    )
    return len(NUM_RE.findall(prompt)) >= 3 and any(marker in lowered for marker in math_markers)


def looks_strict_math_prompt(prompt: str) -> bool:
    lowered = prompt.lower()
    return "let's think step by step" in lowered and "\nanswer:" in lowered


def prompt_risk_profile(prompt: str) -> str:
    if looks_code_like(prompt):
        return "code"
    if looks_math_like(prompt):
        return "math"
    return "default"


def configure_arc_features(model: EaModel, args, spec: Dict[str, Any]) -> Dict[str, Any]:
    if args.mode != "arc":
        return {}
    if args.lts:
        lts_path = Path(args.lts_ckpt).resolve() if args.lts_ckpt else Path(spec["lts"]).resolve()
        lts_info = load_lts_ckpt(model, lts_path, args)
        model._base_lts_lambda = float(args.lts_lambda)
    else:
        model.lts_cfg = None
        model._eval_mix_ratio = 0.0
        model._use_pf_lts = False
        model._base_lts_lambda = float(args.lts_lambda)
        lts_info = {"enabled": False}

    if args.pct:
        model.enable_pct(theta=args.pct_theta, leaf_thresh=args.pct_leaf_thresh)
        model._pct_min_tree = int(args.pct_min_tree)
        model._pct_adaptive_gate = bool(args.pct_adaptive_gate)
        model._pct_gate_min_calls = int(args.pct_gate_min_calls)
        model._pct_gate_min_prune_ratio = float(args.pct_gate_min_prune_ratio)
        model._pct_gate_entropy_min = float(args.pct_gate_entropy_min)
        model._pct_sync_timing = not bool(args.no_pct_sync_timing)
        model._pct_skip_low_entropy = bool(args.pct_skip_low_entropy)
    else:
        model.disable_pct()

    adaptive_info: Dict[str, Any] = {"mode": args.adaptive_templates}
    if args.adaptive_templates == "off":
        model._adaptive_templates = None
    elif args.adaptive_templates == "current":
        model._adaptive_templates = None
        from model import ea_model as ea_model_mod

        model._adaptive_templates = ea_model_mod._TREE_TEMPLATES
        apply_adaptive_lts_lambdas(model._adaptive_templates, args)
        adaptive_info["templates"] = model._adaptive_templates
    elif args.adaptive_templates == "scaled":
        model._adaptive_templates = scaled_adaptive_templates(args.total_tokens, args.depth)
        apply_adaptive_lts_lambdas(model._adaptive_templates, args)
        adaptive_info["templates"] = model._adaptive_templates
    else:
        raise ValueError(f"Unknown adaptive template mode: {args.adaptive_templates}")
    model._entropy_lambda = float(args.entropy_lambda)
    adaptive_info["code_guard_adaptive_lts"] = bool(args.code_guard_adaptive_lts)
    adaptive_info["strict_code_guard_adaptive_lts"] = bool(args.strict_code_guard_adaptive_lts)
    adaptive_info["prompt_risk_guard_adaptive_lts"] = bool(args.prompt_risk_guard_adaptive_lts)
    adaptive_info["calibrated_code_scale"] = bool(args.adaptive_lts_calibrated_code_scale)
    adaptive_info["code_disable_pct_gate"] = bool(args.code_disable_pct_gate)
    adaptive_info["code_calibrated_pct_gate"] = bool(args.code_calibrated_pct_gate)
    adaptive_info["code_pct_gate_alpha_thresh"] = float(args.code_pct_gate_alpha_thresh)
    adaptive_info["code_calibrated_pct_sync"] = bool(args.code_calibrated_pct_sync)
    adaptive_info["code_pct_sync_alpha_thresh"] = float(args.code_pct_sync_alpha_thresh)
    adaptive_info["math_guard_adaptive_lts"] = bool(args.math_guard_adaptive_lts)
    adaptive_info["strict_math_guard_adaptive_lts"] = bool(args.strict_math_guard_adaptive_lts)
    adaptive_info["math_disable_pct_gate"] = bool(args.math_disable_pct_gate)
    adaptive_info["math_sync_pct_timing"] = bool(args.math_sync_pct_timing)
    model._adaptive_lts_calibrated_code_scale = bool(args.adaptive_lts_calibrated_code_scale)
    model._adaptive_lts_code_alpha_lo = float(args.adaptive_lts_code_alpha_lo)
    model._adaptive_lts_code_alpha_hi = float(args.adaptive_lts_code_alpha_hi)
    model._qx_mode = str(args.qx_mode)
    model._qx_alpha = float(args.qx_alpha)
    model._posterior_mode = str(args.posterior_mode)

    return {
        "lts": lts_info,
        "pct": {
            "enabled": bool(args.pct),
            "theta": args.pct_theta,
            "leaf_thresh": args.pct_leaf_thresh,
            "min_tree": args.pct_min_tree,
            "adaptive_gate": bool(args.pct_adaptive_gate),
            "gate_min_calls": args.pct_gate_min_calls,
            "gate_min_prune_ratio": args.pct_gate_min_prune_ratio,
            "gate_entropy_min": args.pct_gate_entropy_min,
            "sync_timing": not bool(args.no_pct_sync_timing),
            "skip_low_entropy": bool(args.pct_skip_low_entropy),
        },
        "adaptive": adaptive_info,
        "qx": "draft_node_logprob_when_available" if args.qx_mode == "actual" else "one",
        "qx_alpha": args.qx_alpha,
        "posterior_mode": args.posterior_mode,
    }


@torch.no_grad()
def model_generate_one(
    model: EaModel,
    tokenizer,
    family: str,
    messages: List[Dict[str, str]],
    seed: int,
    args,
) -> Tuple[str, Dict[str, Any], float, int, int]:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    reset_tree_budget(model, args.total_tokens, args.depth)

    prompt = apply_chat(tokenizer, family, messages)
    if args.strict_code_guard_adaptive_lts and looks_strict_code_completion(prompt):
        profile = "code"
    elif args.strict_math_guard_adaptive_lts and looks_strict_math_prompt(prompt):
        profile = "math"
    elif args.math_guard_adaptive_lts and looks_math_like(prompt):
        profile = "math"
    elif args.prompt_risk_guard_adaptive_lts:
        profile = prompt_risk_profile(prompt)
    elif args.code_guard_adaptive_lts and looks_code_like(prompt):
        profile = "code"
    else:
        profile = "default"
    model._adaptive_lts_prompt_profile = profile
    model._adaptive_lts_code_prompt = profile == "code"
    if getattr(model, "lts_cfg", None) is not None:
        model.lts_cfg.pf_lts_prompt_profile = profile
    disable_pct_gate = bool(args.code_disable_pct_gate and profile == "code")
    if args.code_calibrated_pct_gate and profile == "code":
        alpha = float(getattr(getattr(model, "lts_cfg", None), "alpha_kappa", 0.0) or 0.0)
        disable_pct_gate = alpha >= float(args.code_pct_gate_alpha_thresh)
    if args.math_disable_pct_gate and profile == "math":
        disable_pct_gate = True
    model._pct_disable_gate_for_prompt = disable_pct_gate
    sync_override = None
    if args.code_calibrated_pct_sync and profile == "code":
        alpha = float(getattr(getattr(model, "lts_cfg", None), "alpha_kappa", 0.0) or 0.0)
        if alpha >= float(args.code_pct_sync_alpha_thresh):
            sync_override = True
    if args.math_sync_pct_timing and profile == "math":
        sync_override = True
    model._pct_sync_timing_prompt_override = sync_override
    input_ids = encode_prompt(tokenizer, prompt, first_param_device(model))
    is_llama3 = family == "llama"
    dynamic_max_length = max(args.max_length, input_ids.shape[1] + args.max_new_tokens + args.total_tokens + 32)

    cuda_sync()
    start = time.perf_counter()
    if args.mode == "baseline":
        output_ids, returned_new_token, idx = model.naivegenerate(
            input_ids,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.sample_top_k,
            max_new_tokens=args.max_new_tokens,
            max_length=dynamic_max_length,
            log=True,
            is_llama3=is_llama3,
        )
        acceptance_info: Dict[str, Any] = {}
        prune_stats: Dict[str, Any] = {}
        lts_stats: Dict[str, Any] = {}
        time_breakdown: Dict[str, Any] = {}
    else:
        output_ids, returned_new_token, last_accept = model.eagenerate(
            input_ids,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.sample_top_k,
            max_new_tokens=args.max_new_tokens,
            max_length=dynamic_max_length,
            log=True,
            is_llama3=is_llama3,
            log_rejections=False,
        )
        acceptance_info = model.get_last_acceptance_info()
        prune_stats = model.get_prune_stats() if hasattr(model, "get_prune_stats") else {}
        lts_stats = model.get_lts_stats() if hasattr(model, "get_lts_stats") else {}
        time_breakdown = model.get_time_breakdown() if hasattr(model, "get_time_breakdown") else {}
        idx = int(acceptance_info.get("num_steps", 0)) - 1
    cuda_sync()
    elapsed = time.perf_counter() - start

    raw_gen = output_ids[0][input_ids.shape[1]:]
    visible_gen = strip_stop_tokens(raw_gen, stop_ids_for(tokenizer, family))
    text = clean_special(tokenizer.decode(visible_gen, spaces_between_special_tokens=False))
    raw_new_tokens = int(raw_gen.numel())
    visible_new_tokens = int(visible_gen.numel())
    num_steps = int(acceptance_info.get("num_steps", max(1, idx + 1)))
    stats = {
        "returned_new_token": int(returned_new_token),
        "model_new_tokens": raw_new_tokens,
        "visible_new_tokens": visible_new_tokens,
        "num_steps": num_steps,
        "acceptance_info": acceptance_info,
        "prune_stats": prune_stats,
        "lts_stats": lts_stats,
        "time_breakdown": time_breakdown,
    }
    return text, stats, elapsed, raw_new_tokens, visible_new_tokens


def merge_counter(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    for key, value in (src or {}).items():
        if isinstance(value, dict):
            child = dst.setdefault(key, {})
            if isinstance(child, dict):
                merge_counter(child, value)
        elif isinstance(value, (int, float)):
            dst[key] = dst.get(key, 0) + value


def run(args) -> Dict[str, Any]:
    if abs(args.temperature - 1.0) > 1e-8:
        raise SystemExit("This experiment must run with T=1.0.")

    spec = MODEL_SPECS[args.model]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gen_path = out_dir / "generations.jsonl"
    stats_path = out_dir / "stats.json"

    model = EaModel.from_pretrained(
        base_model_path=spec["base"],
        ea_model_path=spec["draft"],
        total_token=args.total_tokens,
        depth=args.depth,
        top_k=args.tree_top_k,
        qwen3_kv_mode=args.qwen3_kv_mode,
        torch_dtype=dtype_from_name(spec["dtype"]),
        low_cpu_mem_usage=True,
        device_map="auto",
        use_eagle3=True,
    )
    move_ea_layer_to_real_device(model)
    model._disable_internal_timing_sync = bool(args.no_internal_timing_sync)
    model.eval()
    tokenizer = model.get_tokenizer()
    feature_config = configure_arc_features(model, args, spec)

    rows, data_file, data_scope = load_eval_rows(args.benchmark, args.limit, args.full_data)
    choices_per_problem = args.humaneval_samples if args.benchmark == "humaneval" else 1

    total_visible_new_tokens = 0
    total_model_new_tokens = 0
    total_verify_steps = 0
    total_latency = 0.0
    aggregate_lts_stats: Dict[str, Any] = {}
    aggregate_prune_stats: Dict[str, Any] = {}
    correct = 0
    humaneval_passed_any = 0

    with gen_path.open("w", encoding="utf-8") as fout:
        for q_idx, row in enumerate(rows):
            if args.benchmark == "mt_bench":
                prior_turns: List[str] = []
                turn_records = []
                for turn_idx in range(len(row["turns"])):
                    messages = build_mt_messages(args.model, row, prior_turns, turn_idx)
                    text, step_stats, elapsed, raw_new, visible_new = model_generate_one(
                        model, tokenizer, args.model, messages, args.seed + q_idx * 100 + turn_idx, args
                    )
                    prior_turns.append(text)
                    turn_records.append(
                        {
                            "turn": turn_idx,
                            "output": text,
                            "model_new_tokens": raw_new,
                            "visible_new_tokens": visible_new,
                            "latency": elapsed,
                            "stats": step_stats,
                        }
                    )
                    total_model_new_tokens += raw_new
                    total_visible_new_tokens += visible_new
                    total_verify_steps += int(step_stats.get("num_steps", max(1, raw_new)))
                    total_latency += elapsed
                    merge_counter(aggregate_lts_stats, step_stats.get("lts_stats", {}))
                    merge_counter(aggregate_prune_stats, step_stats.get("prune_stats", {}))
                fout.write(json.dumps({"question_id": row.get("question_id", q_idx), "turns": turn_records}) + "\n")
                fout.flush()
                continue

            completions = []
            problem_passed = False
            for cidx in range(choices_per_problem):
                messages = build_messages(args.model, args.benchmark, row["turns"][0])
                text, step_stats, elapsed, raw_new, visible_new = model_generate_one(
                    model, tokenizer, args.model, messages, args.seed + q_idx * 1000 + cidx, args
                )
                rec = {
                    "choice": cidx,
                    "output": text,
                    "model_new_tokens": raw_new,
                    "visible_new_tokens": visible_new,
                    "latency": elapsed,
                    "stats": step_stats,
                }
                if args.benchmark == "gsm8k":
                    pred = extract_last_number(text)
                    target = extract_gsm_answer(row["reference"][0])
                    rec["prediction"] = pred
                    rec["target"] = target
                    rec["correct"] = pred == target
                    correct += int(rec["correct"])
                elif args.benchmark == "humaneval":
                    body = extract_func_body(text, row["entry_point"])
                    program = build_humaneval_program(row["prompt"], body)
                    ok = run_humaneval_test(program, row["test"], row["entry_point"], args.humaneval_timeout)
                    rec["passed"] = ok
                    rec["completion_body"] = body
                    problem_passed = problem_passed or ok
                completions.append(rec)
                total_model_new_tokens += raw_new
                total_visible_new_tokens += visible_new
                total_verify_steps += int(step_stats.get("num_steps", max(1, raw_new)))
                total_latency += elapsed
                merge_counter(aggregate_lts_stats, step_stats.get("lts_stats", {}))
                merge_counter(aggregate_prune_stats, step_stats.get("prune_stats", {}))
            if args.benchmark == "humaneval":
                humaneval_passed_any += int(problem_passed)
            fout.write(json.dumps({"question_id": row.get("question_id", q_idx), "choices": completions}) + "\n")
            fout.flush()

    throughput = total_model_new_tokens / total_latency if total_latency > 0 else 0.0
    accept_length = (
        total_model_new_tokens / total_verify_steps
        if args.mode == "arc" and total_verify_steps > 0
        else 1.0
    )
    stats = {
        "model": args.model,
        "model_id": spec["model_id"],
        "benchmark": args.benchmark,
        "mode": args.mode,
        "temperature": args.temperature,
        "total_tokens": args.total_tokens,
        "depth": args.depth,
        "tree_top_k": args.tree_top_k,
        "top_p": args.top_p,
        "sample_top_k": args.sample_top_k,
        "qwen3_kv_mode": args.qwen3_kv_mode,
        "no_internal_timing_sync": bool(args.no_internal_timing_sync),
        "prompt_config": {
            "qwen_enable_thinking": False if args.model == "qwen" else None,
        },
        "max_new_tokens": args.max_new_tokens,
        "data_file": str(data_file),
        "data_scope": data_scope,
        "num_questions": len(rows),
        "choices_per_problem": choices_per_problem,
        "total_visible_new_tokens": total_visible_new_tokens,
        "total_model_new_tokens": total_model_new_tokens,
        "total_verify_steps": total_verify_steps,
        "total_latency": total_latency,
        "throughput": throughput,
        "accept_length": accept_length,
        "gsm8k_em": correct / max(1, len(rows)) if args.benchmark == "gsm8k" else None,
        "humaneval_pass_at_10": humaneval_passed_any / max(1, len(rows)) if args.benchmark == "humaneval" else None,
        "feature_config": feature_config,
        "aggregate_lts_stats": aggregate_lts_stats,
        "aggregate_prune_stats": aggregate_prune_stats,
        "generations_file": str(gen_path),
    }
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2), flush=True)
    return stats


def optional_bool(value: str):
    if value.lower() == "auto":
        return None
    return value.lower() in {"1", "true", "yes", "y"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--benchmark", choices=["mt_bench", "humaneval", "gsm8k", "alpaca"], required=True)
    parser.add_argument("--mode", choices=["baseline", "arc"], required=True)
    parser.add_argument("--total-tokens", type=int, default=60)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--tree-top-k", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--sample-top-k", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--humaneval-samples", type=int, default=10)
    parser.add_argument("--humaneval-timeout", type=float, default=10.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--full-data", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--lts-ckpt", default="")
    parser.add_argument("--lts-param-mode", choices=["calibrated", "defaults"], default="calibrated")
    parser.add_argument("--lts-theta", type=float, default=0.3)
    parser.add_argument("--lts-lambda", type=float, default=0.25)
    parser.add_argument("--lts-min-boost-prob", type=float, default=0.05)
    parser.add_argument("--pf-lts-record-stats", action="store_true")
    parser.add_argument("--pf-lts-selective", action="store_true")
    parser.add_argument("--pf-lts-min-acp-default", type=float, default=0.0)
    parser.add_argument("--pf-lts-min-acp-code", type=float, default=0.0)
    parser.add_argument("--pf-lts-min-acp-math", type=float, default=0.0)
    parser.add_argument("--pf-lts-max-u-ratio-default", type=float, default=1.0)
    parser.add_argument("--pf-lts-max-u-ratio-code", type=float, default=1.0)
    parser.add_argument("--pf-lts-max-u-ratio-math", type=float, default=1.0)
    parser.add_argument("--lts-use-budget", type=optional_bool, default=None)
    parser.add_argument("--lts-deny-token-strs", nargs="*", default=["<pad>", "<unk>", "<|eot_id|>"])
    parser.add_argument("--lts", action="store_true", default=True)
    parser.add_argument("--no-lts", action="store_false", dest="lts")
    parser.add_argument("--pf-lts", action="store_true", default=True)
    parser.add_argument("--no-pf-lts", action="store_false", dest="pf_lts")
    parser.add_argument("--mix-ratio", type=float, default=0.0)

    parser.add_argument("--pct", action="store_true", default=True)
    parser.add_argument("--no-pct", action="store_false", dest="pct")
    parser.add_argument("--pct-theta", type=float, default=0.005)
    parser.add_argument("--pct-leaf-thresh", type=float, default=0.001)
    parser.add_argument("--pct-min-tree", type=int, default=48)
    parser.add_argument("--pct-adaptive-gate", action="store_true", default=True)
    parser.add_argument("--pct-gate-min-calls", type=int, default=4)
    parser.add_argument("--pct-gate-min-prune-ratio", type=float, default=0.08)
    parser.add_argument("--pct-gate-entropy-min", type=float, default=5.5)
    parser.add_argument("--pct-skip-low-entropy", action="store_true", default=True)
    parser.add_argument("--no-pct-sync-timing", action="store_true", default=True)

    parser.add_argument("--adaptive-templates", choices=["off", "current", "scaled"], default="scaled")
    parser.add_argument("--adaptive-lts-lambda-deep", type=float, default=None)
    parser.add_argument("--adaptive-lts-lambda-base", type=float, default=None)
    parser.add_argument("--adaptive-lts-lambda-shallow", type=float, default=0.50)
    parser.add_argument("--adaptive-lts-code-lambda-deep", type=float, default=None)
    parser.add_argument("--adaptive-lts-code-lambda-base", type=float, default=None)
    parser.add_argument("--adaptive-lts-code-lambda-shallow", type=float, default=None)
    parser.add_argument("--adaptive-lts-math-lambda-deep", type=float, default=None)
    parser.add_argument("--adaptive-lts-math-lambda-base", type=float, default=None)
    parser.add_argument("--adaptive-lts-math-lambda-shallow", type=float, default=None)
    parser.add_argument("--code-guard-adaptive-lts", action="store_true")
    parser.add_argument("--strict-code-guard-adaptive-lts", action="store_true", default=True)
    parser.add_argument("--prompt-risk-guard-adaptive-lts", action="store_true")
    parser.add_argument("--adaptive-lts-calibrated-code-scale", action="store_true")
    parser.add_argument("--adaptive-lts-code-alpha-lo", type=float, default=4.0)
    parser.add_argument("--adaptive-lts-code-alpha-hi", type=float, default=20.0)
    parser.add_argument("--code-disable-pct-gate", action="store_true")
    parser.add_argument("--code-calibrated-pct-gate", action="store_true", default=True)
    parser.add_argument("--code-pct-gate-alpha-thresh", type=float, default=10.0)
    parser.add_argument("--code-calibrated-pct-sync", action="store_true", default=True)
    parser.add_argument("--code-pct-sync-alpha-thresh", type=float, default=10.0)
    parser.add_argument("--math-guard-adaptive-lts", action="store_true")
    parser.add_argument("--strict-math-guard-adaptive-lts", action="store_true", default=True)
    parser.add_argument("--math-disable-pct-gate", action="store_true", default=True)
    parser.add_argument("--math-sync-pct-timing", action="store_true", default=True)
    parser.add_argument("--entropy-lambda", type=float, default=0.0)
    parser.add_argument("--qx-mode", choices=["actual", "one"], default="actual")
    parser.add_argument("--qx-alpha", type=float, default=1.0)
    parser.add_argument("--posterior-mode", choices=["arc", "eagle"], default="arc")
    parser.add_argument("--qwen3-kv-mode", choices=["arc", "eagle"], default="eagle")
    parser.add_argument("--no-internal-timing-sync", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
