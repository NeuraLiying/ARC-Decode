"""ARC-Decode helper."""
import copy
import json
import math
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import random

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import os
from transformers import PreTrainedModel, PretrainedConfig, AutoConfig

from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .modeling_mixtral_kv import MixtralForCausalLM as KVMixtralForCausalLM
from .modeling_qwen2_kv import Qwen2ForCausalLM as KVQwen2ForCausalLM
from .modeling_qwen3_kv import Qwen3ForCausalLM as KVQwen3ForCausalLM
from .modeling_qwen3_kv_eagle import Qwen3ForCausalLM as KVEagleQwen3ForCausalLM
from .utils import *
from .utils import _compute_entropy_from_logits
from .kv_cache import initialize_past_key_values

from .cnets import Model
from .cnets1 import Model as Model1
from .configs import EConfig




# === Entropy-conditioned tree template selection ===
_TREE_TEMPLATES = {
    'deep':    {'depth': 6, 'total_token': 64, 'h_max': 3.0},
    'base':    {'depth': 6, 'total_token': 48, 'h_max': 5.5},
    'shallow': {'depth': 4, 'total_token': 32, 'h_max': float('inf')},
}

def _select_tree_template(entropy, templates=None):
    """Select tree template based on target entropy."""
    if templates is None:
        templates = _TREE_TEMPLATES
    if entropy is None:
        return templates.get('base', {'depth': 6, 'total_token': 48})
    for name in ['deep', 'base', 'shallow']:
        if name not in templates:
            continue
        t = templates[name]
        if entropy < t['h_max']:
            return t
    return min(templates.values(), key=lambda t: t['depth'])


def _clip01(value):
    return max(0.0, min(1.0, float(value)))

@dataclass
class EAGLERunStats:
    """ARC-Decode helper."""
    accept_lengths: list = field(default_factory=list)          
    verified_positions: list = field(default_factory=list)      
    appended_per_step: list = field(default_factory=list)       
    token_sources: list = field(default_factory=list)           
    accepted_total: int = 0                                      
    verified_total: int = 0                                      


@dataclass
class EAGLETimeStats:
    """ARC-Decode helper."""
    
    prefill_time: float = 0.0              
    main_generation_time: float = 0.0      
    draft_generation_time: float = 0.0     
    verify_forward_time: float = 0.0       
    verify_post_time: float = 0.0          
    
    total_time: float = 0.0
    num_steps: int = 0
    
    
    position_prep_time: float = 0.0
    transformer_forward_time: float = 0.0
    lm_head_time: float = 0.0
    device_transfer_time: float = 0.0
    retrieve_index_time: float = 0.0
    verify_forward_token_cnt: int = 0
    prune_time: float = 0.0

    
    pruned_tokens_total: int = 0
    pruned_tokens_pct: int = 0
    pruned_tokens_pct_init: int = 0
    pruned_tokens_pct_loop: int = 0
    prune_calls_pct_init: int = 0
    prune_calls_pct_loop: int = 0
    

    @property
    def verification_time(self):
        """ARC-Decode helper."""
        return self.verify_forward_time + self.verify_post_time
    
    def get_stage_breakdown(self):
        """ARC-Decode helper."""
        if self.total_time == 0:
            return {}
        return {
            "prefill": (self.prefill_time / self.total_time) * 100,
            "main_generation": (self.main_generation_time / self.total_time) * 100,
            "draft_generation": (self.draft_generation_time / self.total_time) * 100,
            "verify_forward": (self.verify_forward_time / self.total_time) * 100,
            "verify_post": (self.verify_post_time / self.total_time) * 100,
            "prune": (getattr(self, "prune_time", 0.0) / self.total_time) * 100,  
        }



class EaModel(nn.Module):

    def __init__(
            self,
            use_eagle3,
            base_model,
            base_model_name_or_path,
            ea_model_path,
            total_token,
            depth,
            top_k,
            threshold,
            ea_layer_state_dict,
    ):

        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path, use_fast=False)
        self.use_eagle3 = use_eagle3
        config = EConfig.from_pretrained(ea_model_path)
        with open(ea_model_path, "r") as f:
            con = json.loads(f.read())
        try:
            bias = con["bias"]
        except:
            bias = True
        if use_eagle3:
            self.ea_layer = Model(config, bias=bias, total_tokens=total_token, depth=depth, top_k=top_k,
                                  threshold=threshold, path=base_model_name_or_path,load_emb=True)
        else:
            self.ea_layer = Model1(config, bias=bias, total_tokens=total_token, depth=depth, top_k=top_k,
                                  threshold=threshold, path=base_model_name_or_path,load_emb=True)

        low_memory = False

        device = base_model.model.layers[-1].self_attn.q_proj.weight.device
        if device != base_model.lm_head.weight.device:
            self.ea_layer.diff_device = True
            if not low_memory:
                self.ea_layer.headweight = base_model.lm_head.weight.clone().to(device)
            else:
                self.ea_layer.layer_device = device

        else:
            self.ea_layer.diff_device = False
        if self.use_eagle3 and config.vocab_size==config.draft_vocab_size:
            del self.ea_layer.d2t,self.ea_layer.t2d
        load_=self.ea_layer.load_state_dict(ea_layer_state_dict, strict=False)
        self.ea_layer.to(self.base_model.dtype).to(device)
        self.ea_layer.init_tree()

    def _trace_begin(self):
        """ARC-Decode helper."""
        self._run_stats = EAGLERunStats()
        
        self._rejtrace = RejectionTrace()

    def _trace_step(self, eval_result, appended_n):
        """ARC-Decode helper."""
        if not hasattr(self, "_run_stats"):
            return
        
        stats = self._run_stats
        
       
        stats.accepted_total += appended_n  
        
        
        stats.accept_lengths.append(eval_result.accept_length)
        
        stats.appended_per_step.append(appended_n)
        
        
        eagle_tokens = min(eval_result.accept_length, appended_n)
        target_tokens = max(0, appended_n - eagle_tokens)
        step_sources = ["eagle"] * eagle_tokens + ["target"] * target_tokens
        stats.token_sources.extend(step_sources)

    def get_last_acceptance_info(self):
        """ARC-Decode helper."""
        if not hasattr(self, "_run_stats"):
            return {}
        
        s = self._run_stats
      
        
        return {
            "accept_lengths": s.accept_lengths,
         
            "appended_per_step": s.appended_per_step,
            "token_sources": s.token_sources,
            "accepted_total": s.accepted_total,      
         
            "num_steps": len(s.accept_lengths),    
        }

    def get_rejection_trace(self) -> RejectionTrace:
        """ARC-Decode helper."""
        if not hasattr(self, "_rejtrace"):
            return RejectionTrace()
        
        
        import copy
        return copy.deepcopy(self._rejtrace)

    def _trace_rejection(self, eval_result, candidates, best_candidate, accept_length, 
                        sample_token, prev_input_len, input_ids):
        """ARC-Decode helper."""
        if not hasattr(self, "_rejtrace"):
            return
        
        
        for i in range(accept_length):
            abs_position = prev_input_len + i  
            
            
            rejected_token_ids = eval_result.rejected_map.get(i, [])
            
            
            self._rejtrace.record_rejection(abs_position, rejected_token_ids)
        
        
        if sample_token is not None and accept_length < candidates.shape[1]:
            abs_position = prev_input_len + accept_length
            final_rejected_ids = eval_result.rejected_map.get(accept_length, [])
            
            
            self._rejtrace.record_rejection(abs_position, final_rejected_ids)

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer

    def _kv_len(self) -> int:
        cl = self.current_length_data
        if isinstance(cl, torch.Tensor):
            if cl.numel() == 1:
                return int(cl.item())
            
            return int(cl.view(-1)[0].item())  
        return int(cl)

    def _kv_fill(self, new_len_minus_one: int):
        
        self.current_length_data.fill_(int(new_len_minus_one))

    def enable_pct(self, theta: float = 0.05, leaf_thresh: float = 0.05):
        self._enable_pct = True
        self._pct_theta = float(theta)
        self._pct_leaf = float(leaf_thresh)

    def disable_pct(self):
        self._enable_pct = False

    def _timing_sync(self):
        if getattr(self, "_disable_internal_timing_sync", False):
            return
        torch.cuda.synchronize()

    def _reset_pct_gate_state(self):
        self._pct_gate_state = {
            "calls": 0,
            "pre_tokens": 0,
            "pruned_tokens": 0,
            "disabled": False,
        }

    def _pct_gate_entropy_active(self, entropy: Optional[float]) -> bool:
        min_entropy = float(getattr(self, "_pct_gate_entropy_min", -1.0))
        if min_entropy < 0.0:
            return True
        if entropy is None:
            return False
        return float(entropy) >= min_entropy

    def _pct_adaptive_gate_active_for_prompt(self) -> bool:
        return bool(getattr(self, "_pct_adaptive_gate", False)) and not bool(
            getattr(self, "_pct_disable_gate_for_prompt", False)
        )

    def _pct_sync_timing_active_for_prompt(self) -> bool:
        override = getattr(self, "_pct_sync_timing_prompt_override", None)
        if override is not None:
            return bool(override)
        return bool(getattr(self, "_pct_sync_timing", True))

    def _pct_should_run(self, pre_S: int, entropy: Optional[float] = None) -> bool:
        if not getattr(self, "_enable_pct", False):
            return False
        if pre_S <= int(getattr(self, "_pct_min_tree", 48)):
            return False
        if not self._pct_adaptive_gate_active_for_prompt():
            return True
        if not self._pct_gate_entropy_active(entropy):
            return not bool(getattr(self, "_pct_skip_low_entropy", False))
        state = getattr(self, "_pct_gate_state", None)
        if state is None:
            self._reset_pct_gate_state()
            state = self._pct_gate_state
        return not bool(state.get("disabled", False))

    def _pct_gate_update(self, pre_S: int, post_S: int, entropy: Optional[float] = None):
        if not self._pct_adaptive_gate_active_for_prompt():
            return
        if not self._pct_gate_entropy_active(entropy):
            return
        state = getattr(self, "_pct_gate_state", None)
        if state is None:
            self._reset_pct_gate_state()
            state = self._pct_gate_state
        state["calls"] += 1
        state["pre_tokens"] += max(0, int(pre_S))
        state["pruned_tokens"] += max(0, int(pre_S) - int(post_S))
        min_calls = int(getattr(self, "_pct_gate_min_calls", 4))
        min_ratio = float(getattr(self, "_pct_gate_min_prune_ratio", 0.08))
        if state["calls"] >= min_calls and state["pre_tokens"] > 0:
            ratio = state["pruned_tokens"] / max(1, state["pre_tokens"])
            if ratio < min_ratio:
                state["disabled"] = True

    def _adaptive_lts_lambda_for_template(self, template: Dict[str, float]) -> float:
        base_lts_lambda = float(getattr(self, "_base_lts_lambda", self.lts_cfg.lts_lambda))
        profile = str(getattr(self, "_adaptive_lts_prompt_profile", "default"))
        if profile == "code":
            if "code_lts_lambda" not in template:
                return base_lts_lambda
            target = float(template["code_lts_lambda"])
            if getattr(self, "_adaptive_lts_calibrated_code_scale", False):
                alpha = max(float(getattr(self.lts_cfg, "alpha_kappa", 1.0)), 1.0e-6)
                lo = float(getattr(self, "_adaptive_lts_code_alpha_lo", 4.0))
                hi = max(float(getattr(self, "_adaptive_lts_code_alpha_hi", 20.0)), lo + 1.0e-6)
                scale = _clip01((math.log(alpha) - math.log(lo)) / (math.log(hi) - math.log(lo)))
                return base_lts_lambda + (target - base_lts_lambda) * scale
            return target
        if profile == "math":
            if "math_lts_lambda" in template:
                return float(template["math_lts_lambda"])
            return base_lts_lambda
        if "lts_lambda" in template:
            return float(template["lts_lambda"])
        return base_lts_lambda

    def _record_prune_tokens(self, *, pre_S: int, post_S: int, kind: str, where: str, pre_R: Optional[int] = None, post_R: Optional[int] = None):
        """ARC-Decode helper."""
        if not hasattr(self, "_time_stats"):
            return
        s = self._time_stats

        
        dS = max(0, int(pre_S) - int(post_S))
        s.pruned_tokens_total += dS
        if kind == "pct":
            s.pruned_tokens_pct += dS
            if where == "init":
                s.pruned_tokens_pct_init += dS
                s.prune_calls_pct_init += 1
            else:
                s.pruned_tokens_pct_loop += dS
                s.prune_calls_pct_loop += 1


    def clear_prune_stats(self):
        """ARC-Decode helper."""
        if not hasattr(self, "_time_stats"):
            self._time_stats = EAGLETimeStats()
        s = self._time_stats
        s.pruned_tokens_total = 0
        s.pruned_tokens_pct = 0
        s.pruned_tokens_pct_init = 0
        s.pruned_tokens_pct_loop = 0
        s.prune_calls_pct_init = 0
        s.prune_calls_pct_loop = 0

    def get_prune_stats(self):
        """ARC-Decode helper."""
        if not hasattr(self, "_time_stats"):
            return {}
        s = self._time_stats
        return {
            "tokens": {
                "total": s.pruned_tokens_total,
                "pct": {
                    "total": s.pruned_tokens_pct,
                    "init": s.pruned_tokens_pct_init,
                    "loop": s.pruned_tokens_pct_loop,
                    "calls": {
                        "init": s.prune_calls_pct_init,
                        "loop": s.prune_calls_pct_loop,
                    }
                },
            },
        }

    def get_lts_stats(self):
        cfg = getattr(self, "lts_cfg", None)
        stats = getattr(cfg, "_stats", None) if cfg is not None else None
        return dict(stats) if isinstance(stats, dict) else {}

    def set_lts_params(
        self,
        *,
        c_s_prime: float,
        alpha_kappa: float,
        tau_delta: float,
        inv_std: torch.Tensor,
        theta: float = 0.0,
        budget_table: Optional[List[float]] = None,
        deny_token_strs: Optional[List[str]] = None,
        use_budget: bool = False,
        margin_bound: Optional[float] = None,
        logit_bound: Optional[float] = None,
        parent_topk: int = 8,
        parent_min_keep: int = 1,
        margin_relax: float = 1.25,
        path_margin_scale: float = 1.0,
        path_logit_scale: float = 1.0,
        max_alive_parents: int = 0,
        depth_margin_bounds: Optional[List[float]] = None,
        depth_logit_bounds: Optional[List[float]] = None,
    ):
        """ARC-Decode helper."""
        deny_ids = set()
        if deny_token_strs is not None:
            tok = self.get_tokenizer()
            for s in deny_token_strs:
                try:
                    tid = tok.convert_tokens_to_ids(s)
                    if isinstance(tid, list):  
                        for x in tid:
                            if isinstance(x, int) and x >= 0:
                                deny_ids.add(int(x))
                    elif isinstance(tid, int) and tid >= 0:
                        deny_ids.add(int(tid))
                except Exception:
                    pass

        self.lts_cfg = LTSConfig(
            c_s_prime=c_s_prime, alpha_kappa=alpha_kappa, tau_delta=tau_delta,
            theta=theta, inv_std=inv_std.detach().clone(),
            budget_table=budget_table, deny_token_ids=deny_ids, use_budget=use_budget,
            margin_bound=margin_bound,
            logit_bound=logit_bound,
            parent_topk=int(parent_topk),
            parent_min_keep=int(parent_min_keep),
            margin_relax=float(margin_relax),
            path_margin_scale=float(path_margin_scale),
            path_logit_scale=float(path_logit_scale),
            max_alive_parents=int(max_alive_parents),
            depth_margin_bounds=depth_margin_bounds,
            depth_logit_bounds=depth_logit_bounds,
        )
        self.lts_cfg.soft_window_size = getattr(self.lts_cfg, "soft_W", 64)
        
        if not hasattr(self, "_eval_mix_ratio"):
            self._eval_mix_ratio = 0.0

    @classmethod
    def from_pretrained(
            cls,
            use_eagle3=True,
            base_model_path=None,
            ea_model_path=None,
            total_token=60,
            depth=7,
            top_k=10,
            threshold=1.0,
            qwen3_kv_mode="arc",
            **kwargs,
    ):
        # assert Type=="LLaMA" or "Mixtral"
        Type = AutoConfig.from_pretrained(base_model_path).architectures[0]

        if Type == 'LlamaForCausalLM':
            base_model = KVLlamaForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        elif Type == 'Qwen2ForCausalLM':
            base_model = KVQwen2ForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        elif Type == 'Qwen3ForCausalLM':
            if qwen3_kv_mode not in {"arc", "eagle"}:
                raise ValueError(f"Unknown qwen3_kv_mode: {qwen3_kv_mode}")
            qwen3_cls = KVEagleQwen3ForCausalLM if qwen3_kv_mode == "eagle" else KVQwen3ForCausalLM
            base_model = qwen3_cls.from_pretrained(
                base_model_path, **kwargs
            )
        else:
            base_model = KVMixtralForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )

        configpath = os.path.join(ea_model_path, "config.json")
        if not os.path.exists(configpath):
            configpath = hf_hub_download(ea_model_path, "config.json")

        try:
            load_model_path = os.path.join(ea_model_path, "pytorch_model.bin")
            if not os.path.exists(load_model_path):
                load_model_path = hf_hub_download(ea_model_path, "pytorch_model.bin")
            ea_layer_state_dict = torch.load(load_model_path,
                                             map_location=base_model.device)
        except:
            from safetensors.torch import load_file
            load_model_path = os.path.join(ea_model_path, "model.safetensors")
            if not os.path.exists(load_model_path):
                load_model_path = hf_hub_download(ea_model_path, "model.safetensors")
            ea_layer_state_dict = load_file(load_model_path)
        model = cls(
            use_eagle3,
            base_model,
            base_model_path,
            configpath,
            total_token,
            depth,
            top_k,
            threshold,
            ea_layer_state_dict
        )

        if total_token == -1:
            device = model.base_model.model.layers[0].self_attn.q_proj.weight.device
            cans = [40, 48, 50, 56, 60]
            x = [1, 1.05, 1.07, 1.1, 1.13]
            times = []

            for i in range(len(cans)):
                length = cans[i]
                input_ids = torch.randint(0, model.config.vocab_size - 200, (1, length)).to(device)
                torch.cuda.synchronize()
                start_time = time.time()
                for _ in range(20):
                    torch.cuda.synchronize()
                    with torch.no_grad():
                        outputs = model.base_model(input_ids)
                    torch.cuda.synchronize()
                torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) / x[i])
            total_token = cans[times.index(min(times))]
            model.ea_layer.total_tokens = total_token - 1

        return model

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            past_key_values=None,
            output_orig=False,
            position_ids=None,
    ):
        with torch.inference_mode():
            
            _t0 = time.time() if hasattr(self, '_time_stats') else None
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
            if hasattr(self, '_time_stats'):
                self._timing_sync()
                self._time_stats.transformer_forward_time += time.time() - _t0

            # LM Head
            if output_orig:
                _t1 = time.time() if hasattr(self, '_time_stats') else None
                orig = self.base_model.lm_head(outputs[0])
                if hasattr(self, '_time_stats'):
                    self._timing_sync()
                    self._time_stats.lm_head_time += time.time() - _t1
            hidden_states = outputs[0]

        if output_orig:
            return outputs, orig, hidden_states
        else:
            return outputs, hidden_states

    @torch.no_grad()
    def eagenerate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=256,
            max_length=2048,
            log=False,
            is_llama3=False,
            log_rejections=False,
    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")

        # logits processor
        logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)\
            if temperature > 1e-5 else None

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        
        self._time_stats = EAGLETimeStats()                          
        self._reset_pct_gate_state()
        self._trace_begin()
        overall_start = time.time()

        prev_len = input_ids.shape[1]

        # KV cache
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data

            self._timing_sync()
            _t0 = time.time()
            current_length_data.zero_()
            self._timing_sync()
            self._time_stats.prefill_time += time.time() - _t0
        else:
            self._timing_sync()
            _t0 = time.time()
            (past_key_values, past_key_values_data, current_length_data) =\
                initialize_past_key_values(self.base_model, max_length=max_length)
            self._timing_sync()
            self._time_stats.prefill_time += time.time() - _t0

            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)

        
        if getattr(self, "lts_cfg", None) is not None:               # <<<
            from collections import deque                             # <<<
            self.lts_cfg._state = {                                   # <<<
                "win": deque(),
                "win_soft": 0,
                "recent_q": deque(),
                "recent_cnt": {},
                "no_accept_steps": 0,
                "rescue_left": 0,
                "cooldown_left": 0,
            }                                                          # <<<
            if bool(getattr(self.lts_cfg, "pf_lts_record_stats", False)):
                self.lts_cfg._stats = {
                    "positions": 0,
                    "posterior_accept": 0,
                    "posterior_reject": 0,
                    "boost_evaluated": 0,
                    "boost_candidates": 0,
                    "boost_accept": 0,
                    "blocked_min_prob": 0,
                    "blocked_acp": 0,
                    "blocked_u": 0,
                }
            else:
                self.lts_cfg._stats = None

            
            try:                                                     # <<<
                emb_table = self.base_model.model.embed_tokens.weight
            except Exception:
                emb_table = self.base_model.lm_head.weight           # <<<

        
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token, node_logprob = initialize_tree(
            input_ids, self, past_key_values, logits_processor
        )

        
        if getattr(self, "_enable_pct", False):
            theta = getattr(self, "_pct_theta", 0.05)
            leaf_t = getattr(self, "_pct_leaf", 0.05)
            _t_pct = time.time() if hasattr(self, "_time_stats") else None

            _pre_S = int(draft_tokens.size(1))

            if (not getattr(self, "_pct_adaptive_gate", False)) or self._pct_should_run(_pre_S, None):
                draft_tokens, retrieve_indices, tree_mask, tree_position_ids, node_logprob = pct_prune_by_columns(
                    draft_tokens, retrieve_indices, tree_mask, tree_position_ids,
                    node_logprob=node_logprob,
                    theta=theta, leaf_thresh=leaf_t
                )

                _post_S = int(draft_tokens.size(1))
                self._record_prune_tokens(pre_S=_pre_S, post_S=_post_S, kind="pct", where="init")
                self._pct_gate_update(_pre_S, _post_S, None)

                if hasattr(self, "_time_stats"):
                    if self._pct_sync_timing_active_for_prompt():
                        self._timing_sync()
                    self._time_stats.prune_time += time.time() - _t_pct

        

        new_token = 0
        total_new_tokens = 0
        max_length = max_length - self.ea_layer.total_tokens - 10

        for idx in range(max_length):
            
            vf_start = time.time()

            self.base_model.model.tree_mask = tree_mask
            draft_tokens = draft_tokens.to(input_ids.device)

            logits, hidden_state_new, outputs = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
            )

            self._timing_sync()
            self._time_stats.verify_forward_time += time.time() - vf_start

            
            vp_start = time.time()

            draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens[0, retrieve_indices]

            
            step_verify_tokens = max(1, candidates.shape[1] - 1)
            self._time_stats.verify_forward_token_cnt += int(step_verify_tokens)
            self._time_stats.num_steps += 1

            cur_len_before = input_ids.shape[1]

            
            use_lts = getattr(self, "lts_cfg", None) is not None
            mix_ratio = float(getattr(self, "_eval_mix_ratio", 0.0))
            do_posterior = False
            if use_lts:
                
                if torch.rand((), device=logits.device).item() < mix_ratio:
                    do_posterior = True

            if use_lts and not do_posterior:
                posterior_node_logprob = None if getattr(self, "_qx_mode", "actual") == "one" else node_logprob
                if posterior_node_logprob is not None:
                    posterior_node_logprob = posterior_node_logprob * float(getattr(self, "_qx_alpha", 1.0))
                if getattr(self, '_use_pf_lts', False):
                    # PF-LTS: posterior-floor + LTS rescue
                    eval_result = evaluate_pf_lts(
                        logits=logits,
                        candidates=candidates,
                        logits_processor=logits_processor,
                        node_logprob=posterior_node_logprob,
                        retrieve_indices=retrieve_indices,
                        emb_table=emb_table,
                        lts_cfg=self.lts_cfg,
                        log_rejections=log_rejections,
                    )
                else:
                    # Original LTS verification
                    eval_result = evaluate_lts(
                        logits=logits,
                        candidates=candidates,
                        logits_processor=logits_processor,
                        emb_table=emb_table,
                        lts_cfg=self.lts_cfg,
                        log_rejections=log_rejections
                    )
            else:
                
                posterior_node_logprob = None if getattr(self, "_qx_mode", "actual") == "one" else node_logprob
                if posterior_node_logprob is not None:
                    posterior_node_logprob = posterior_node_logprob * float(getattr(self, "_qx_alpha", 1.0))
                if getattr(self, "_posterior_mode", "arc") == "eagle":
                    eval_result = evaluate_posterior_eagle_compat(
                        logits, candidates, logits_processor, log_rejections=log_rejections,
                        node_logprob=posterior_node_logprob, retrieve_indices=retrieve_indices
                    )
                else:
                    eval_result = evaluate_posterior(
                        logits, candidates, logits_processor, log_rejections=log_rejections,
                        node_logprob=posterior_node_logprob, retrieve_indices=retrieve_indices
                    )
                
                if use_lts:
                    from .utils import lts_sync_after_posterior
                    bc = int(eval_result.best_candidate.item() if hasattr(eval_result.best_candidate, "item") else int(eval_result.best_candidate))
                    al = int(eval_result.accept_length)
                    lts_sync_after_posterior(
                        logits=logits,
                        candidates=candidates,
                        best_candidate=bc,
                        accept_length=al,
                        logits_processor=logits_processor,
                        lts_cfg=self.lts_cfg,
                    )

            best_candidate = eval_result.best_candidate
            accept_length = eval_result.accept_length
            sample_p = eval_result.sample_p

            # === Entropy-conditioned adaptive template selection ===
            _loop_entropy = None
            _entropy_lambda = getattr(self, '_entropy_lambda', 0.0)
            _adaptive_templates = getattr(self, '_adaptive_templates', None)
            if (_entropy_lambda > 0.0 or getattr(self, "_adaptive_tree", None) is not None
                    or _adaptive_templates is not None) and accept_length < logits.shape[1]:
                _loop_entropy = _compute_entropy_from_logits(
                    logits[best_candidate, accept_length]
                ).item()

            if _adaptive_templates is not None:
                _tpl = _select_tree_template(_loop_entropy, _adaptive_templates)
                _remaining = max_length - input_ids.shape[1]
                if _tpl['total_token'] > _remaining:
                    _fit = {k: v for k, v in _adaptive_templates.items() if v['total_token'] <= _remaining}
                    if _fit:
                        _tpl = _select_tree_template(_loop_entropy, _fit)
                    else:
                        _tpl = min(_adaptive_templates.values(), key=lambda t: t['total_token'])
                self.ea_layer.depth = _tpl['depth']
                self.ea_layer.total_tokens = _tpl['total_token'] - 1
                if getattr(self, 'lts_cfg', None) is not None:
                    self.lts_cfg.lts_lambda = self._adaptive_lts_lambda_for_template(_tpl)

            self._timing_sync()
            self._time_stats.verify_post_time += time.time() - vp_start

            
            input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids,\
                new_token, hidden_state, sample_token, node_logprob = update_inference_inputs(
                    input_ids,
                    candidates,
                    best_candidate,
                    accept_length,
                    retrieve_indices,
                    logits_processor,
                    new_token,
                    past_key_values_data,
                    current_length_data,
                    self,
                    hidden_state_new,
                    sample_p,
                    entropy=_loop_entropy,
                    entropy_lambda=_entropy_lambda,
                    adaptive_cfg=getattr(self, '_adaptive_tree', None),
                )

            
            if getattr(self, "_enable_pct", False):
                _t_pct = time.time() if hasattr(self, "_time_stats") else None
                _pre_S = int(draft_tokens.size(1))

                if self._pct_should_run(_pre_S, _loop_entropy):
                  draft_tokens, retrieve_indices, tree_mask, tree_position_ids, node_logprob = pct_prune_by_columns(
                    draft_tokens, retrieve_indices, tree_mask, tree_position_ids,
                    node_logprob=node_logprob,          
                    theta=getattr(self, "_pct_theta", 0.05),
                    leaf_thresh=getattr(self, "_pct_leaf", 0.05),
                )
                  _post_S = int(draft_tokens.size(1))
                  self._record_prune_tokens(pre_S=_pre_S, post_S=_post_S, kind="pct", where="loop")
                  self._pct_gate_update(_pre_S, _post_S, _loop_entropy)

                  
                  if hasattr(self, "_time_stats"):
                      if self._pct_sync_timing_active_for_prompt():
                          self._timing_sync()
                      self._time_stats.prune_time += time.time() - _t_pct

            total_new_tokens += (len(input_ids[0]) - cur_len_before)

            if log_rejections:
                self._trace_rejection(
                    eval_result, candidates, best_candidate, accept_length,
                    sample_token, cur_len_before, input_ids
                )

            
            appended_n = input_ids.shape[1] - cur_len_before
            class _S:  
                def __init__(self, a): self.accept_length = int(a)
            self._trace_step(_S(accept_length), appended_n)

            
            if is_llama3 and (stop_token_id in input_ids[0, input_len:].tolist()):
                break
            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if total_new_tokens >= max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break

        self._timing_sync()
        self._time_stats.total_time = time.time() - overall_start

        if log_rejections and hasattr(self, "_rejtrace"):
            self._rejtrace.set_final_output(input_ids[0].tolist())

        if not log:
            return input_ids
        else:
            
            return input_ids, new_token, int(accept_length)          # <<<

    def get_time_breakdown(self):
        if not hasattr(self, '_time_stats'):
            return {}

        stats = self._time_stats
        vf_tokens = max(1, stats.verify_forward_token_cnt)
        return {
            "stage_breakdown": stats.get_stage_breakdown(),
            "absolute_times": {
                "prefill": stats.prefill_time,
                "main_generation": stats.main_generation_time,
                "draft_generation": stats.draft_generation_time,
                "verify_forward": stats.verify_forward_time,
                "verify_post": stats.verify_post_time,
                "prune": getattr(stats, 'prune_time', 0.0),   
                "verification": stats.verification_time,
                "total": stats.total_time,
            },
            "verify_forward_breakdown": {
                "position_prep": stats.position_prep_time,
                "transformer_forward": stats.transformer_forward_time,
                "lm_head": stats.lm_head_time,
                "device_transfer": stats.device_transfer_time,
                "retrieve_index": stats.retrieve_index_time,
                "tokens": stats.verify_forward_token_cnt,
                "ms_per_draft_token": 1000.0 * stats.verify_forward_time / vf_tokens,
            },
            "num_steps": stats.num_steps,
            "pruning_counters": {  
                "tokens": {
                    "total": stats.pruned_tokens_total,
                    "pct": {
                        "total": stats.pruned_tokens_pct,
                        "init": stats.pruned_tokens_pct_init,
                        "loop": stats.pruned_tokens_pct_loop,
                        "calls": {
                            "init": stats.prune_calls_pct_init,
                            "loop": stats.prune_calls_pct_loop,
                        }
                    },
                },
            }
        }



#########################################################
    @torch.no_grad()
    def naivegenerate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=256,
            max_length=2048,
            log=False,
            is_llama3=False,

    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")


        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model,max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        outputs = self.base_model(input_ids, past_key_values=past_key_values, use_cache=True)
        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        for idx in range(max_length):
            if logits_processor is not None:
                logits = outputs.logits[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                input_id = torch.multinomial(probabilities, 1)
            else:
                input_id = outputs.logits[:, -1:].argmax(dim=-1)
            outputs = self.base_model(input_id, use_cache=True, past_key_values=past_key_values)
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            new_token += 1

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
        if not log:
            return input_ids
        else:
            return input_ids, new_token, idx

    @torch.no_grad()
    def ea_generate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=256,
            max_length=2048,
            log=False,
            is_llama3=False,

    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")


        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model,max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token, node_logprob = initialize_tree(
            input_ids, self, past_key_values, logits_processor
        )
        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        for idx in range(max_length):
            # with Timer("all"):
            self.base_model.model.tree_mask = tree_mask

            draft_tokens = draft_tokens.to(input_ids.device)
            
            
            cur_len_before = input_ids.shape[1]
            
            # with Timer("tree_decoding"):
            logits, hidden_state_new, outputs = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
            )
            # retrieve_indices=tree_buffers["retrieve_indices"]
            # logits = logits[0, retrieve_indices]
            draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens[0, retrieve_indices]
            eval_result = evaluate_posterior(
                logits, candidates, logits_processor, 
                log_rejections=True
            )
            # print(accept_length)
            # with Timer("update_inference_inputs"):
            input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token, hidden_state, sample_token = update_inference_inputs(
                input_ids,
                candidates,
                eval_result.best_candidate,
                eval_result.accept_length,
                retrieve_indices,
                logits_processor,
                new_token,
                past_key_values_data,
                current_length_data,
                self,
                hidden_state_new,
                sample_token
            )

            yield input_ids

            
            self._trace_rejection(eval_result, candidates, eval_result.best_candidate, eval_result.accept_length, 
                                sample_token, cur_len_before, input_ids)

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break

    @torch.no_grad()
    def naive_generate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=128,
            max_length=2048,
            log=False,
            is_llama3=False,

    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")


        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model,max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        outputs = self.base_model(input_ids, past_key_values=past_key_values, use_cache=True)
        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        for idx in range(max_length):
            if logits_processor is not None:
                logits = outputs.logits[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                input_id = torch.multinomial(probabilities, 1)
            else:
                input_id = outputs.logits[:, -1:].argmax(dim=-1)

            outputs = self.base_model(input_id, use_cache=True, past_key_values=past_key_values)
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            new_token += 1

            yield input_ids


            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break

        if not log:
            return input_ids
        else:
            return input_ids, new_token, idx

    
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")


        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model,max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token, node_logprob = initialize_tree(
            input_ids, self, past_key_values, logits_processor
        )
        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        for idx in range(max_length):
            # with Timer("all"):
            self.base_model.model.tree_mask = tree_mask

            draft_tokens = draft_tokens.to(input_ids.device)
            
            
            cur_len_before = input_ids.shape[1]
            
            # with Timer("tree_decoding"):
            logits, hidden_state_new, outputs = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
            )
            # retrieve_indices=tree_buffers["retrieve_indices"]
            # logits = logits[0, retrieve_indices]
            draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens[0, retrieve_indices]
            eval_result = evaluate_posterior(
                logits, candidates, logits_processor, 
                log_rejections=True
            )
            # print(accept_length)
            # with Timer("update_inference_inputs"):
            input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token, hidden_state, sample_token = update_inference_inputs(
                input_ids,
                candidates,
                eval_result.best_candidate,
                eval_result.accept_length,
                retrieve_indices,
                logits_processor,
                new_token,
                past_key_values_data,
                current_length_data,
                self,
                hidden_state_new,
                sample_token
            )

            yield input_ids

            
            self._trace_rejection(eval_result, candidates, eval_result.best_candidate, eval_result.accept_length, 
                                sample_token, cur_len_before, input_ids)

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
