"""ARC-Decode helper."""


import copy
import math
import random
from dataclasses import dataclass, field

# typing 
from typing import List, Tuple, Optional, Dict
import time
import torch


TOPK = 10  # topk for sparse tree

from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)


class Timer:
    def __init__(self,name):
        self.name = name
    def __enter__(self):
        torch.cuda.synchronize()
        self.start = time.perf_counter()


    def __exit__(self, exc_type, exc_value, traceback):
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.start
        print(f'{self.name} took {elapsed} seconds')


def _maybe_timing_sync(model) -> None:
    if getattr(model, "_disable_internal_timing_sync", False):
        return
    torch.cuda.synchronize()




def _compute_entropy_from_logits(logits):
    """H = -sum(p * log(p)) from raw logits (base model output)."""
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    probs = torch.exp(log_probs)
    return -(probs * log_probs).sum(dim=-1)

def prepare_logits_processor(
        temperature: float = 0.0,
        repetition_penalty: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 0
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    if temperature > 1e-5:
        if temperature >= 1e-5 and temperature != 1.0:
            processor_list.append(TemperatureLogitsWarper(temperature))
        if repetition_penalty > 1.0:
            processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
        if 1e-8 <= top_p < 1.0:
            processor_list.append(TopPLogitsWarper(top_p))
        if top_k > 0:
            processor_list.append(TopKLogitsWarper(top_k))
    return processor_list


def pad_path(path: List[int], length: int, pad_value: int = -2) -> List[int]:
    """
    Pad the given path list with a specific value up to a specified length.

    Parameters:
    - path (list): The original list that needs padding.
    - length (int): The desired length of the padded list.
    - pad_value (optional, default=-2): The value to use for padding.

    Returns:
    - list: A new list based on the original path but padded to the desired length.

    Example:
    >>> pad_path([1,2,3], 5)
    [1, 2, 3, -2, -2]

    Note:
    If the given path is already longer than the specified length,
    then no padding occurs, and the original path is returned.
    """

    # Calculate the number of padding values needed by subtracting the length
    # of the path from the desired length.
    # Append the padding values to the original path and return the new list.
    return path + [pad_value] * (length - len(path))


def generate_tree_buffers(tree_choices, device="cuda"):
    def custom_sort(lst):
        # sort_keys=[len(list)]
        sort_keys = []
        for i in range(len(lst)):
            sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
        return sort_keys
    with Timer("sort"):

        sorted_tree_choices = sorted(tree_choices, key=lambda x: (len(x), x))
        tree_len = len(sorted_tree_choices) + 1

    # Initialize depth_counts to keep track of how many choices have a particular depth
        depth_counts = []
        prev_depth = 0
        for path in sorted_tree_choices:
            depth = len(path)
            if depth != prev_depth:
                depth_counts.append(0)
            depth_counts[depth - 1] += 1
            prev_depth = depth

        tree_attn_mask = torch.eye(tree_len, tree_len)
        tree_attn_mask[:, 0] = 1
        start = 0
        for i in range(len(depth_counts)):
            for j in range(depth_counts[i]):
                cur_tree_choice = sorted_tree_choices[start + j]
                # retrieve ancestor position
                if len(cur_tree_choice) == 1:
                    continue
                ancestor_idx = []
                for c in range(len(cur_tree_choice) - 1):
                    ancestor_idx.append(sorted_tree_choices.index(cur_tree_choice[:c + 1]) + 1)
                tree_attn_mask[j + start + 1, ancestor_idx] = 1
            start += depth_counts[i]

        tree_indices = torch.zeros(tree_len, dtype=torch.long)
        p_indices = [0 for _ in range(tree_len - 1)]
        b_indices = [[] for _ in range(tree_len - 1)]
        tree_indices[0] = 0
        start = 0
        bias = 0
        for i in range(len(depth_counts)):
            inlayer_bias = 0
            b = []
            for j in range(depth_counts[i]):
                cur_tree_choice = sorted_tree_choices[start + j]
                cur_parent = cur_tree_choice[:-1]
                if j != 0:
                    if cur_parent != parent:
                        bias += 1
                        inlayer_bias += 1
                        parent = cur_parent
                        b = []
                else:
                    parent = cur_parent
                tree_indices[start + j + 1] = cur_tree_choice[-1] + TOPK * (i + bias) + 1
                p_indices[start + j] = inlayer_bias
                if len(b) > 0:
                    b_indices[start + j] = copy.deepcopy(b)
                else:
                    b_indices[start + j] = []
                b.append(cur_tree_choice[-1] + TOPK * (i + bias) + 1)
            start += depth_counts[i]

        p_indices = [-1] + p_indices
        tree_position_ids = torch.zeros(tree_len, dtype=torch.long)
        start = 0
        for i in range(len(depth_counts)):
            tree_position_ids[start + 1: start + depth_counts[i] + 1] = i + 1
            start += depth_counts[i]

        retrieve_indices_nest = []
        retrieve_paths = []
        for i in range(len(sorted_tree_choices)):
            cur_tree_choice = sorted_tree_choices[-i - 1]
            retrieve_indice = []
            if cur_tree_choice in retrieve_paths:
                continue
            else:
                for c in range(len(cur_tree_choice)):
                    retrieve_indice.append(sorted_tree_choices.index(cur_tree_choice[:c + 1]))
                    retrieve_paths.append(cur_tree_choice[:c + 1])
            retrieve_indices_nest.append(retrieve_indice)
        max_length = max([len(x) for x in retrieve_indices_nest])
        retrieve_indices = [pad_path(path, max_length) for path in retrieve_indices_nest]
        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
        retrieve_indices = retrieve_indices + 1
        retrieve_indices = torch.cat([torch.zeros((retrieve_indices.shape[0], 1), dtype=torch.long), retrieve_indices],
                                     dim=1)

        maxitem = retrieve_indices.max().item() + 5



        retrieve_indices = retrieve_indices.tolist()
        retrieve_indices = sorted(retrieve_indices, key=custom_sort)
        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)



    # Aggregate the generated buffers into a dictionary
    tree_buffers = {
        "tree_attn_mask": tree_attn_mask.unsqueeze(0).unsqueeze(0),
        "tree_indices": tree_indices,
        "tree_position_ids": tree_position_ids,
        "retrieve_indices": retrieve_indices,
    }

    # Move the tensors in the dictionary to the specified device
    tree_buffers = {
        k: v.clone().to(device)
        if isinstance(v, torch.Tensor)
        else torch.tensor(v, device=device)
        for k, v in tree_buffers.items()
    }

    return tree_buffers


def initialize_tree0(input_ids, model, past_key_values, logits_processor):
    draft_tokens, retrieve_indices,tree_mask,tree_position_ids, outputs, logits, hidden_state, sample_token = model(
        input_ids, past_key_values=past_key_values, output_orig=True, logits_processor=logits_processor
    )
    return draft_tokens, retrieve_indices,tree_mask,tree_position_ids, logits, hidden_state, sample_token

def initialize_tree(input_ids, model, past_key_values, logits_processor):
    
    if hasattr(model, '_time_stats'):
        prefill_start = time.time()  
    
    
    outputs, orig, hidden_states = model(
        input_ids, past_key_values=past_key_values, output_orig=True
    )

    if logits_processor is not None:
        logits = orig[:, -1] #torch.Size([1, 51, 151936]) -> torch.Size([1, 151936])
        logits = logits_processor(None, logits)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        token = torch.multinomial(probabilities, 1) #torch.Size([1, 1])
    else:
        token = torch.argmax(orig[:, -1])
        token = token[None, None]
    input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)
    
    if hasattr(model, '_time_stats'):
        _maybe_timing_sync(model)
        model._time_stats.prefill_time += time.time() - prefill_start  

    # Clone the output hidden states
    if model.use_eagle3: 
        ea_device = model.ea_layer.lm_head.weight.device
        if outputs["hidden_states"][0].device != ea_device:
            outputs["hidden_states"] = [x.to(ea_device) for x in outputs["hidden_states"]]
        hidden_states=torch.cat(outputs["hidden_states"],dim=-1) #torch.Size([1, 51, 12288])
        
    
    if hasattr(model, '_time_stats'):
        draft_start = time.time()
    
    # Entropy computation for adaptive template selection
    _entropy = None
    _entropy_lambda = getattr(model, '_entropy_lambda', 0.0)
    if _entropy_lambda > 0.0 or getattr(model, '_adaptive_tree', None) is not None:
        _entropy = _compute_entropy_from_logits(orig[:, -1]).item()

    draft_tokens, retrieve_indices, tree_mask, tree_position_ids, node_logprob = model.ea_layer.topK_genrate(
        hidden_states, 
        input_ids, 
        model.base_model.lm_head,
        logits_processor,
        entropy=_entropy,
        entropy_lambda=_entropy_lambda,
    )
    
    if hasattr(model, '_time_stats'):
        _maybe_timing_sync(model)
        model._time_stats.draft_generation_time += time.time() - draft_start
    
    return draft_tokens, retrieve_indices, tree_mask, tree_position_ids, orig, hidden_states, token, node_logprob
    


def reset_tree_mode(
        model,
):
    model.base_model.model.tree_mask = None
    model.base_model.model.tree_mode = None


def reset_past_key_values(passed_key_values: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Resets the current lengths in the passed key-values to zero.

    This function is designed to be used during the evaluation of a baseline model.
    It iterates through each layer's key-values and sets their current lengths to zero,
    effectively resetting their state.

    Args:
    - passed_key_values (list of torch.Tensor): Contains past hidden states and past attention values for each layer.

    Returns:
    - passed_key_values (list of torch.Tensor): Updated past hidden states and past attention values with reset lengths.
    """
    for i in range(len(passed_key_values)):
        for j in range(2):
            passed_key_values[i][j].current_length.fill_(0)
    return passed_key_values


def generate_candidates(tree_logits, tree_indices, retrieve_indices, sample_token, logits_processor):
    sample_token = sample_token.to(tree_indices.device)

    candidates_logit = sample_token[0]

    candidates_tree_logits = tree_logits

    candidates = torch.cat([candidates_logit, candidates_tree_logits.view(-1)], dim=-1)

    tree_candidates = candidates[tree_indices]

    tree_candidates_ext = torch.cat(
        [tree_candidates, torch.zeros((1), dtype=torch.long, device=tree_candidates.device) - 1], dim=0)

    cart_candidates = tree_candidates_ext[retrieve_indices]


    # Unsqueeze the tree candidates for dimension consistency.
    tree_candidates = tree_candidates.unsqueeze(0)
    return cart_candidates,  tree_candidates


def tree_decoding(
        model,
        tree_candidates,
        past_key_values,
        tree_position_ids,
        input_ids,
        retrieve_indices,
):
    
    if hasattr(model, '_time_stats'):
        _tp = time.time()
    position_ids = tree_position_ids + input_ids.shape[1]
    if position_ids is not None and position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0)
    if hasattr(model, '_time_stats'):
        _maybe_timing_sync(model)
        model._time_stats.position_prep_time += time.time() - _tp

    
    outputs, tree_logits, hidden_state = model(
        tree_candidates,
        output_orig=True,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )

    
    if hasattr(model, '_time_stats'):
        _td = time.time()
    if model.use_eagle3:
        ea_device = model.ea_layer.lm_head.weight.device
        if outputs["hidden_states"][0].device != ea_device:
            outputs["hidden_states"] = [x.to(ea_device) for x in outputs["hidden_states"]]
        hidden_state = torch.cat(outputs["hidden_states"], dim=-1)
    if hasattr(model, '_time_stats'):
        _maybe_timing_sync(model)
        model._time_stats.device_transfer_time += time.time() - _td

    
    if hasattr(model, '_time_stats'):
        _tr = time.time()
    if retrieve_indices.device != tree_logits.device:
        retrieve_indices = retrieve_indices.to(tree_logits.device, non_blocking=True)
    logits = tree_logits[0, retrieve_indices]
    if hasattr(model, '_time_stats'):
        _maybe_timing_sync(model)
        model._time_stats.retrieve_index_time += time.time() - _tr
        
        model._time_stats.verify_forward_token_cnt += int(tree_candidates.shape[1])

    return logits, hidden_state, outputs


@dataclass
class RejectionTrace:
    """ARC-Decode helper."""
    position_rejections: Dict[int, List[int]] = field(default_factory=dict)  
    final_output_ids: List[int] = field(default_factory=list)                
    meta: dict = field(default_factory=dict)                                 
    
    def record_rejection(self, abs_position: int, rejected_token_ids: List[int]):
        """ARC-Decode helper."""
        if rejected_token_ids:
            
            seen = set()
            unique_rejected = [tid for tid in rejected_token_ids if not (tid in seen or seen.add(tid))]
            self.position_rejections[abs_position] = unique_rejected
    
    def set_final_output(self, output_ids: List[int]):
        """ARC-Decode helper."""
        self.final_output_ids = output_ids.copy()
    
    def render_bracketed_text(self, tokenizer, prefix_len: int = 0, format_style: str = "json") -> str:
        """ARC-Decode helper."""
        if not self.final_output_ids:
            return ""
        
        
        new_tokens = self.final_output_ids[prefix_len:]
        result_parts = []
        
        for i, token_id in enumerate(new_tokens):
            abs_pos = prefix_len + i  
            token_text = tokenizer.decode([token_id], skip_special_tokens=True)
            
            rejected_ids = self.position_rejections.get(abs_pos, [])
            if rejected_ids:
                
                rejected_texts = [tokenizer.decode([rid], skip_special_tokens=True) for rid in rejected_ids]
                
                if format_style == "json":
                    
                    import json
                    rejected_json = json.dumps(rejected_texts, ensure_ascii=False)
                    result_parts.append(f"{token_text}{rejected_json}")
                else:
                    
                    rejected_str = ",".join(rejected_texts)
                    result_parts.append(f"{token_text}[{rejected_str}]")
            else:
                result_parts.append(token_text)
        
        return ''.join(result_parts)

@dataclass
class EvaluationResult:
    """ARC-Decode helper."""
    best_candidate: torch.Tensor                                        
    accept_length: int                                                  
    verified_positions: int                                             
    sample_p: torch.Tensor                                             
    rejected_map: Dict[int, List[int]] = field(default_factory=dict)   
    target_probs_map: Dict[int, torch.Tensor] = field(default_factory=dict)  

    
    @property
    def posterior_mask(self):
        """ARC-Decode helper."""
        return None

    @property 
    def candidates_accept_length(self):
        """ARC-Decode helper."""
        return self.accept_length
    
    @property
    def sample_token(self):
        """ARC-Decode helper."""
        if self.sample_p is not None and len(self.sample_p.shape) > 0:
            return torch.argmax(self.sample_p).unsqueeze(0)
        return None

"""ARC-Decode helper."""
def evaluate_posterior(
        logits: torch.Tensor,
        candidates: torch.Tensor,
        logits_processor,
        log_rejections: bool = False,
        node_logprob: Optional[torch.Tensor] = None,
        retrieve_indices: Optional[torch.Tensor] = None,
):
    """ARC-Decode helper."""
    
    rejected_map = {}        # {position: [token_id1, token_id2, ...]}
    target_probs_map = {}    # {position: target_probs_tensor}
    
    
    if logits_processor is None:
        
        posterior_mask = (
                candidates[:, 1:].to(logits.device) == torch.argmax(logits[:, :-1], dim=-1)
        ).int()
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        accept_length = candidates_accept_length.max()
        if accept_length == 0:
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
        
        
        if log_rejections:
            
            argmax_tokens = torch.argmax(logits[:, :-1], dim=-1)  # [num_candidates, seq_len-1]
            
            
            for i in range(min(accept_length + 1, candidates.shape[1] - 1)):
                
                target_token_id = argmax_tokens[best_candidate, i].item()
                
                
                position_candidates = []
                for j in range(candidates.shape[0]):
                    candidate_token_id = candidates[j, i + 1].item()
                    if candidate_token_id != -1 and candidate_token_id not in [c[0] for c in position_candidates]:
                        position_candidates.append((candidate_token_id, j))
                
                
                rejected_ids = []
                for candidate_token_id, cand_idx in position_candidates:
                    if candidate_token_id != target_token_id:
                        rejected_ids.append(candidate_token_id)  
                
                if rejected_ids:
                    rejected_map[i] = rejected_ids
        
        
        result = EvaluationResult(
            best_candidate=best_candidate,
            accept_length=accept_length,
            verified_positions=accept_length,
            sample_p=logits[best_candidate, accept_length],
            rejected_map=rejected_map,
            target_probs_map=target_probs_map
        )
        return result

    else:
        
        accept_length = 1
        accept_cand = candidates[0][:1]
        best_candidate = 0
        
        for i in range(1, candidates.shape[1]):
            if i != accept_length:
                break
                
            
            rejected_map[i-1] = []
            
            
            draft_probs = torch.zeros_like(logits[0, 0])
                
            
            is_eq = (candidates[:, :accept_length] == accept_cand).all(dim=1)
            fi = torch.nonzero(is_eq, as_tuple=True)[0][0]
            
            
            gt_logits = logits[fi, i - 1][None]
            gt_logits = logits_processor(None, gt_logits)[0]
            target_probs = torch.softmax(gt_logits, dim=0)
            
            
            target_probs_map[i-1] = target_probs.clone()
            
            seen = set()
            position_accepted = False
            
            
            for j in range(candidates.shape[0]):
                if is_eq[j]:
                    xi = candidates[j, i].item()
                    
                    if xi == -1 or xi in seen:
                        continue
                    seen.add(xi)
                    
                    
                    r = random.random()
                    px = target_probs[xi].item()
                    
                    # Compute draft probability from node_logprob
                    if node_logprob is not None and retrieve_indices is not None:
                        child_idx = int(retrieve_indices[j, i].item())
                        parent_idx = int(retrieve_indices[j, i-1].item())
                        if (child_idx >= 0 and parent_idx >= 0
                                and child_idx < node_logprob.numel()
                                and parent_idx < node_logprob.numel()):
                            log_q = node_logprob[child_idx].item() - node_logprob[parent_idx].item()
                            qx = max(math.exp(log_q), 1e-10)
                        else:
                            qx = 1.0
                    else:
                        qx = 1.0
                    
                    acp = min(1.0, px / qx)
                    
                    if r <= acp:  
                        accept_cand = torch.cat((accept_cand, candidates[j, i][None]), dim=0)
                        accept_length += 1
                        best_candidate = j
                        position_accepted = True
                        break
                    else:  
                        # Record actual draft probability for corrected sampling
                        if node_logprob is not None and retrieve_indices is not None:
                            draft_probs[xi] = qx
                        else:
                            draft_probs[xi] = target_probs[xi]
                        
                        if log_rejections:
                            rejected_map[i-1].append(xi)
            
            
            if not position_accepted:
                
                final_gt_logits = logits[fi, i - 1][None]
                final_gt_logits = logits_processor(None, final_gt_logits)[0]
                final_target_probs = torch.softmax(final_gt_logits, dim=0)
                
                
                target_probs_map[i-1] = final_target_probs.clone()
                
                
                corrected_probs = torch.relu(final_target_probs - draft_probs)
                
                
                if corrected_probs.sum() > 1e-8:
                    sample_p = corrected_probs / corrected_probs.sum()
                else:
                    
                    sample_p = final_target_probs
                
                
                result = EvaluationResult(
                    best_candidate=torch.tensor(best_candidate),
                    accept_length=accept_length - 1,
                    verified_positions=accept_length - 1,
                    sample_p=sample_p,
                    rejected_map=rejected_map,
                    target_probs_map=target_probs_map
                )
                return result
        
        
        final_gt_logits = logits[best_candidate, accept_length - 1][None]
        final_gt_logits = logits_processor(None, final_gt_logits)[0]
        final_target_probs = torch.softmax(final_gt_logits, dim=0)
        
        
        target_probs_map[accept_length - 1] = final_target_probs.clone()
        
        
        result = EvaluationResult(
            best_candidate=torch.tensor(best_candidate),
            accept_length=accept_length - 1,
            verified_positions=accept_length - 1,
            sample_p=final_target_probs,  
            rejected_map=rejected_map,
            target_probs_map=target_probs_map
        )
        return result


def evaluate_posterior_eagle_compat(
        logits: torch.Tensor,
        candidates: torch.Tensor,
        logits_processor,
        log_rejections: bool = False,
        node_logprob: Optional[torch.Tensor] = None,
        retrieve_indices: Optional[torch.Tensor] = None,
):
    """Original EAGLE-3 posterior logic wrapped in EvaluationResult."""
    rejected_map = {}
    target_probs_map = {}

    if logits_processor is None:
        posterior_mask = (
                candidates[:, 1:].to(logits.device) == torch.argmax(logits[:, :-1], dim=-1)
        ).int()
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        accept_length = candidates_accept_length.max()
        if accept_length == 0:
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
        return EvaluationResult(
            best_candidate=best_candidate,
            accept_length=accept_length,
            verified_positions=accept_length,
            sample_p=logits[best_candidate, accept_length],
            rejected_map=rejected_map,
            target_probs_map=target_probs_map,
        )

    accept_length = 1
    accept_cand = candidates[0][:1]
    best_candidate = 0
    adjustflag = False
    gtp = None
    for i in range(1, candidates.shape[1]):
        if i != accept_length:
            break
        adjustflag = False
        rejected_map[i - 1] = []
        is_eq = (candidates[:, :accept_length] == accept_cand).all(dim=1)
        fi = torch.nonzero(is_eq, as_tuple=True)[0][0]
        gt_logits = logits[fi, i - 1][None]
        gt_logits = logits_processor(None, gt_logits)[0]
        gtp = torch.softmax(gt_logits, dim=0)
        target_probs_map[i - 1] = gtp.clone()
        candidates_set = []
        for j in range(candidates.shape[0]):
            if is_eq[j]:
                x = candidates[j, i]
                xi = x.item()
                if xi in candidates_set or xi == -1:
                    continue
                candidates_set.append(xi)
                r = random.random()
                px = gtp[xi]
                qx = 1.0
                acp = px / qx
                if r <= acp:
                    accept_cand = torch.cat((accept_cand, x[None]), dim=0)
                    accept_length += 1
                    best_candidate = j
                    break
                else:
                    gtp[xi] = 0
                    gtp = gtp / gtp.sum()
                    target_probs_map[i - 1] = gtp.clone()
                    adjustflag = True
                    if log_rejections:
                        rejected_map[i - 1].append(xi)

    if adjustflag and accept_length != candidates.shape[1]:
        sample_p = gtp
    else:
        gt_logits = logits[best_candidate, accept_length - 1][None]
        gt_logits = logits_processor(None, gt_logits)[0]
        sample_p = torch.softmax(gt_logits, dim=0)
        target_probs_map[accept_length - 1] = sample_p.clone()

    return EvaluationResult(
        best_candidate=torch.tensor(best_candidate, device=candidates.device),
        accept_length=accept_length - 1,
        verified_positions=accept_length - 1,
        sample_p=sample_p,
        rejected_map=rejected_map,
        target_probs_map=target_probs_map,
    )

def render_bracketed(trace: RejectionTrace, tokenizer, prefix_len: int = 0, with_probs: bool = False) -> str:
    """ARC-Decode helper."""
    if not trace.final_output_ids:
        return ""
    
    return trace.render_bracketed_text(tokenizer, prefix_len)

def save_rejection_trace_jsonl(trace: RejectionTrace, filepath: str, tokenizer, prefix_len: int = 0):
    """ARC-Decode helper."""
    import json
    
    
    serializable_data = {
        "position_rejections": trace.position_rejections,  
        "final_output_ids": trace.final_output_ids,        
        "bracketed_text": trace.render_bracketed_text(tokenizer, prefix_len),  
        "meta": trace.meta
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, ensure_ascii=False, indent=2)


@torch.no_grad()
def update_inference_inputs(
        input_ids,
        candidates,
        best_candidate,
        accept_length,
        retrieve_indices,
        logits_processor,
        new_token,
        past_key_values_data_list,
        current_length_data,
        model,
        hidden_state_new,
        sample_p,
        entropy=None,
        entropy_lambda=0.0,
        adaptive_cfg=None,
):
    prev_input_len = input_ids.shape[1]
    # Map the best candidate indices to the original indices in the sequence
    select_indices = (
            retrieve_indices[best_candidate, : accept_length + 1] + prev_input_len
    )
    # Append the tokens from the best candidate to the input sequence
    input_ids = torch.cat(
        [input_ids, candidates[None, best_candidate, : accept_length + 1].to(input_ids.device)], dim=-1
    )
    # Update the past key values based on the selected tokens
    # Source tensor that contains relevant past information based on the selected candidate
    for past_key_values_data in past_key_values_data_list:
        tgt = past_key_values_data[..., select_indices.to(past_key_values_data.device), :]
        # Destination tensor where the relevant past information will be stored
        dst = past_key_values_data[..., prev_input_len: prev_input_len + tgt.shape[-2], :]
        # Copy relevant past information from the source to the destination
        dst.copy_(tgt, non_blocking=True)

    # Update the current length tensor (currently only support batch size is 1)
    current_length_data.fill_(prev_input_len + tgt.shape[-2])

    retrieve_hidden_state_new = hidden_state_new[:, retrieve_indices]
    accept_hidden_state_new = retrieve_hidden_state_new[:, best_candidate, : accept_length + 1]
    # token=model.base_model.lm_head(accept_hidden_state_new[:,-1]).argmax()
    # token=token[None,None]
    prob = sample_p
    if logits_processor is not None:
        token = torch.multinomial(prob, 1)
        token = token[None]
    else:
        token = torch.argmax(prob)
        token = token[None, None]
    # hidden_state = torch.cat((hidden_state, accept_hidden_state_new), dim=1)
    
    
    if hasattr(model, '_time_stats'):
        draft_start = time.time()
    
    
    draft_tokens, retrieve_indices, tree_mask, tree_position_ids, node_logprob = model.ea_layer.topK_genrate(
        accept_hidden_state_new,
        input_ids=torch.cat((input_ids, token.to(input_ids.device)), dim=1),
        head=model.base_model.lm_head,
        logits_processor=logits_processor,
        entropy=entropy,
        entropy_lambda=entropy_lambda,
        adaptive_cfg=getattr(model, "_adaptive_tree", None),
    )
    
    if hasattr(model, '_time_stats'):
        _maybe_timing_sync(model)
        model._time_stats.draft_generation_time += time.time() - draft_start

    new_token += accept_length + 1

    return input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token, None, token, node_logprob


#======================PCT Pruning======================

@torch.no_grad()
def pct_prune_by_columns(
    draft_tokens: torch.Tensor,               # [1, S]
    retrieve_indices: torch.Tensor,           
    tree_mask: torch.Tensor,                  
    tree_position_ids: torch.Tensor,          
    node_logprob: Optional[torch.Tensor],     
    theta: float,
    leaf_thresh: float = 1e-2,                
):
    device = draft_tokens.device
    S = int(draft_tokens.size(1))

    
    retrieve_indices = retrieve_indices.to(device=device, dtype=torch.long, non_blocking=True)
    tree_position_ids = tree_position_ids.to(device=device, dtype=torch.long, non_blocking=True)
    tree_mask = tree_mask.to(device=device, non_blocking=True)

    
    if (node_logprob is None) or (node_logprob.numel() != S):
        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids, node_logprob

    
    nl = node_logprob.to(device).float()
    overall_conf = torch.exp(nl).clamp_(min=0.0, max=1.0)  # [S]

    
    parent = torch.full((S,), -1, device=device, dtype=torch.long)
    if retrieve_indices.dim() == 2 and retrieve_indices.size(1) > 1:
        pa = retrieve_indices[:, :-1]; ch = retrieve_indices[:, 1:]
        m = (pa >= 0) & (ch >= 0)
        if m.any():
            
            parent.scatter_(0, ch[m], pa[m])

    depth_max = int(tree_position_ids.max().item()) if S > 0 else 0

    
    expandable = overall_conf >= float(theta)
    keep = torch.zeros(S, dtype=torch.bool, device=device)
    keep[0] = True  

    for d in range(1, depth_max + 1):
        cols = (tree_position_ids == d).nonzero(as_tuple=True)[0]
        if cols.numel() == 0:
            continue
        par = parent.index_select(0, cols)
        valid_par = par >= 0
        if not valid_par.any():
            continue

        
        par_kept = keep.index_select(0, par[valid_par])
        cond = par_kept\
             & expandable.index_select(0, par[valid_par])\
             & expandable.index_select(0, cols[valid_par])
        if cond.any():
            keep[cols[valid_par][cond]] = True

    
    has_child = torch.zeros(S, dtype=torch.bool, device=device)
    kept_cols = keep.nonzero(as_tuple=True)[0]
    if kept_cols.numel() > 0:
        kept_par = parent.index_select(0, kept_cols)
        m = kept_par >= 0
        if m.any():
            has_child[kept_par[m]] = True
    leaves = keep & (~has_child)
    drop = leaves & (overall_conf < float(leaf_thresh))
    if drop.any():
        keep[drop] = False

    
    keep_pos = keep.nonzero(as_tuple=True)[0]
    if keep_pos.numel() == 0 or keep_pos[0].item() != 0:
        keep_pos = torch.cat([torch.tensor([0], device=device, dtype=torch.long), keep_pos], dim=0)
    keep_pos = torch.unique(keep_pos, sorted=True)

    old2new = torch.full((S,), -1, device=device, dtype=torch.long)
    old2new.index_copy_(0, keep_pos, torch.arange(keep_pos.numel(), device=device, dtype=torch.long))

    new_draft = draft_tokens.index_select(1, keep_pos)
    if tree_mask.dim() == 4:
        tm = tree_mask.index_select(2, keep_pos); new_mask = tm.index_select(3, keep_pos)
    else:
        tm = tree_mask.index_select(0, keep_pos); new_mask = tm.index_select(1, keep_pos)
    new_pos = tree_position_ids.index_select(0, keep_pos)

    ri = retrieve_indices.clone()
    nonneg = (ri >= 0)
    ri[nonneg] = old2new.index_select(0, ri[nonneg])
    ri[ri < 0] = -1
    valid_prefix = torch.cumprod((ri >= 0).long(), dim=1).bool()
    new_ri = torch.where(valid_prefix, ri, torch.full_like(ri, -1))

    
    new_node_logprob = nl.index_select(0, keep_pos)

    return new_draft, new_ri, new_mask, new_pos, new_node_logprob
#======================PCT Pruning======================


#====== LTS: near-lossless soft-acceptance ======

from typing import Optional, List, Set
@dataclass
class LTSConfig:
    
    c_s_prime: float
    alpha_kappa: float
    tau_delta: float
    theta: float = 0.0
    inv_std: Optional[torch.Tensor] = None
    budget_table: Optional[List[float]] = None
    deny_token_ids: Optional[Set[int]] = None
    eps: float = 1e-12
    use_budget: bool = False
    debug: bool = False

    
    theta_lts: float = 0.98       
    m_max: float = 0.08           
    p_top1_max: float = 0.9        

    
    margin_bound: Optional[float] = None        
    logit_bound: Optional[float] = None         
    path_margin_scale: float = 1.0              
    path_logit_scale: float = 1.0               
    depth_margin_bounds: Optional[List[float]] = None
    depth_logit_bounds: Optional[List[float]] = None

    
    parent_topk: int = 8                        
    parent_min_keep: int = 1                    
    margin_relax: float = 1.25                  
    max_alive_parents: int = 0                  

    
    ngram_n: int = 3                
    ngram_window: int = 64          

    
    soft_k: int = 3                
    soft_W: int = 64               

    
    rescue_streak: int = 2
    rescue_steps: int = 2
    rescue_cooldown: int = 16

    # LTS-boost (v3): confidence-boosted acceptance
    rescue_alpha: float = 0.8
    lts_boost_lambda: float = 0.0
    boost_gate_pq: float = 0.0
    conf_floor: float = 0.0

    # PF-LTS selective boost gates. Defaults preserve the existing behavior.
    lts_lambda: float = 0.5
    min_boost_prob: float = 0.01
    pf_lts_record_stats: bool = False
    pf_lts_selective: bool = False
    pf_lts_min_acp_default: float = 0.0
    pf_lts_min_acp_code: float = 0.0
    pf_lts_min_acp_math: float = 0.0
    pf_lts_max_u_ratio_default: float = 1.0
    pf_lts_max_u_ratio_code: float = 1.0
    pf_lts_max_u_ratio_math: float = 1.0
    pf_lts_prompt_profile: str = "default"

    
    _state: Optional[dict] = None
    _stats: Optional[dict] = None
    history_ids: Optional[torch.Tensor] = None  

def _recent_token_update(st: dict, token_id: int, quota_W: int) -> None:
    """ARC-Decode helper."""
    if quota_W <= 0:
        return

    
    rq = st.get("recent_q", None)
    rc = st.get("recent_cnt", None)
    if (rq is None) or (rc is None) or (not isinstance(rq, deque)):
        st["recent_q"] = deque()
        st["recent_cnt"] = {}
    q: deque = st["recent_q"]
    cnt: dict = st["recent_cnt"]

    
    if isinstance(token_id, torch.Tensor):
        token_id = int(token_id.item())
    else:
        token_id = int(token_id)

    
    q.append(token_id)
    cnt[token_id] = cnt.get(token_id, 0) + 1

    
    while len(q) > quota_W:
        old = q.popleft()
        c = cnt.get(old, 0)
        if c <= 1:
            cnt.pop(old, None)
        else:
            cnt[old] = c - 1


def _lts_single_step_score(
    target_logits: torch.Tensor,   # [vocab]
    logits_processor,
    t_d: int,
    emb_table: torch.Tensor,       # [vocab, dim]
    inv_std: torch.Tensor,         # [dim]
    c_s_prime: float, alpha_kappa: float, tau_delta: float, eps: float
):
    if logits_processor is not None:
        target_logits = logits_processor(None, target_logits[None])[0]
    target_probs = torch.softmax(target_logits, dim=-1)
    t_m = torch.argmax(target_probs).item()

    p_tm = torch.clamp(target_probs[t_m], min=eps)
    p_td = torch.clamp(target_probs[t_d], min=eps)
    delta_logit = torch.log(p_tm) - torch.log(p_td)
    U_logit = alpha_kappa * (delta_logit * delta_logit)

    
    e_tm = emb_table[t_m]
    e_td = emb_table[t_d]
    diff = (e_td - e_tm) * inv_std
    U_emb = c_s_prime * (diff * diff).sum()

    LTS_logit = 1.0 - (U_logit / tau_delta)
    LTS_emb   = 1.0 - (U_emb   / tau_delta)
    return LTS_logit, LTS_emb, U_logit, U_emb, t_m, target_probs



def _recent_token_quota_ok(st: dict, token_id: int, quota_W: int = 32, max_count: int = 2) -> bool:
    """ARC-Decode helper."""
    if "recent_q" not in st:
        st["recent_q"] = deque()
        st["recent_cnt"] = {}
    q: deque = st["recent_q"]
    cnt: dict = st["recent_cnt"]

    
    return (cnt.get(int(token_id), 0) < max_count)

import math
from collections import deque

@torch.no_grad()
def evaluate_lts(
    logits: torch.Tensor,               # [n_paths, seq_len, vocab]
    candidates: torch.Tensor,           # [n_cands, seq_len]
    logits_processor,
    emb_table: torch.Tensor,            # model.model.embed_tokens.weight
    lts_cfg: LTSConfig,
    log_rejections: bool = False,
):
    """ARC-Decode helper."""
    device = logits.device
    emb_table = emb_table.to(device)
    if lts_cfg.inv_std is None:
        raise ValueError("lts_cfg.inv_std is required for evaluate_lts")
    inv_std = lts_cfg.inv_std.to(device=emb_table.device, dtype=emb_table.dtype)
    cand = candidates.to(device=device, dtype=torch.long)

    st = getattr(lts_cfg, "_state", None)
    if st is None:
        st = {
            "win": deque(), "win_soft": 0,
            "no_accept_steps": 0, "rescue_left": 0, "cooldown_left": 0,
            "recent_q": deque(), "recent_cnt": {},
        }
        lts_cfg._state = st
    else:
        if "win" not in st or not isinstance(st["win"], deque):
            st["win"] = deque()
        else:
            if st["win"].maxlen is not None:
                st["win"] = deque(list(st["win"]))
        st.setdefault("win_soft", 0)
        st.setdefault("no_accept_steps", 0)
        st.setdefault("rescue_left", 0)
        st.setdefault("cooldown_left", 0)
        st.setdefault("recent_q", deque())
        st.setdefault("recent_cnt", {})

    if st.get("cooldown_left", 0) > 0:
        st["cooldown_left"] -= 1

    eps_f = float(lts_cfg.eps)
    TWO_GRAM_ALLOW_NEAR_TOP1 = 0.98
    NGRAM_MAX_COUNT = 2

    n_cands = int(cand.shape[0])
    seq_len = int(cand.shape[1])
    max_depth = max(0, seq_len - 1)

    rejected_map = {} if log_rejections else {}
    target_probs_map = {} if log_rejections else {}

    if max_depth == 0:
        final_logits = logits[0, 0]
        if logits_processor is not None:
            final_logits = logits_processor(None, final_logits[None])[0]
        sample_p = torch.softmax(final_logits.float(), dim=0)
        return EvaluationResult(
            best_candidate=torch.tensor(0, device=candidates.device, dtype=torch.long),
            accept_length=0,
            verified_positions=0,
            sample_p=sample_p,
            rejected_map=rejected_map,
            target_probs_map=target_probs_map,
        )

    def _depth_bound(per_depth_bounds, depth, default_val):
        if per_depth_bounds is not None and depth - 1 < len(per_depth_bounds):
            return float(per_depth_bounds[depth - 1])
        return float(default_val)

    def _state_accept(accepted_id: int, top1_id: int):
        st["win"].append(1 if accepted_id != top1_id else 0)
        if st["win"][-1] == 1:
            st["win_soft"] += 1
        if len(st["win"]) > lts_cfg.soft_W:
            dropped = st["win"].popleft()
            if dropped == 1:
                st["win_soft"] -= 1
        _recent_token_update(st, accepted_id, quota_W=lts_cfg.ngram_window)
        st["no_accept_steps"] = 0

    def _state_reject():
        st["win"].append(0)
        if len(st["win"]) > lts_cfg.soft_W:
            dropped = st["win"].popleft()
            if dropped == 1:
                st["win_soft"] -= 1
        st["no_accept_steps"] += 1
        if (
            st["rescue_left"] == 0
            and st["cooldown_left"] == 0
            and st["no_accept_steps"] >= lts_cfg.rescue_streak
        ):
            st["rescue_left"] = lts_cfg.rescue_steps
            st["no_accept_steps"] = 0
        if st["rescue_left"] > 0:
            st["rescue_left"] -= 1
            if st["rescue_left"] == 0:
                st["cooldown_left"] = lts_cfg.rescue_cooldown

    base_margin_bound = (
        float(lts_cfg.margin_bound)
        if lts_cfg.margin_bound is not None
        else float(lts_cfg.m_max)
    )
    base_logit_bound = (
        float(lts_cfg.logit_bound)
        if lts_cfg.logit_bound is not None
        else float(lts_cfg.tau_delta) * (1.0 - float(lts_cfg.theta))
    )
    parent_topk = max(int(getattr(lts_cfg, "parent_topk", 8)), 0)
    parent_min_keep = max(int(getattr(lts_cfg, "parent_min_keep", 1)), 1)
    margin_relax = max(float(getattr(lts_cfg, "margin_relax", 1.25)), 1.0)
    path_margin_scale = max(float(getattr(lts_cfg, "path_margin_scale", 1.0)), 0.0)
    path_logit_scale = max(float(getattr(lts_cfg, "path_logit_scale", 1.0)), 0.0)
    max_alive_parents = max(int(getattr(lts_cfg, "max_alive_parents", 0)), 0)
    u_bound = float(lts_cfg.tau_delta) * (1.0 - float(lts_cfg.theta))

    use_budget = bool(getattr(lts_cfg, "use_budget", False))
    budget_table = getattr(lts_cfg, "budget_table", None)

    
    cand_cpu = cand.detach().to("cpu")
    prefix_token_rows = [dict() for _ in range(max_depth + 1)]
    row_prefix_keys = [[None] * (max_depth + 1) for _ in range(n_cands)]
    for row in range(n_cands):
        for depth in range(1, max_depth + 1):
            if int(cand_cpu[row, depth - 1].item()) == -1:
                break
            key = tuple(int(x) for x in cand_cpu[row, :depth].tolist())
            row_prefix_keys[row][depth] = key
            tok = int(cand_cpu[row, depth].item())
            if tok == -1:
                continue
            slot = prefix_token_rows[depth].setdefault(key, {})
            if tok not in slot:
                slot[tok] = row
    prefix_tensor_cache = [dict() for _ in range(max_depth + 1)]

    def _get_parent_unique(depth: int, parent_row: int):
        key = row_prefix_keys[parent_row][depth]
        if key is None:
            return None, None, None
        cached = prefix_tensor_cache[depth].get(key, None)
        if cached is not None:
            return cached[0], cached[1], key
        token_row = prefix_token_rows[depth].get(key, None)
        if not token_row:
            prefix_tensor_cache[depth][key] = (None, None)
            return None, None, key
        uniq_tokens = torch.as_tensor(
            list(token_row.keys()), device=device, dtype=torch.long
        )
        uniq_rows = torch.as_tensor(
            list(token_row.values()), device=device, dtype=torch.long
        )
        prefix_tensor_cache[depth][key] = (uniq_tokens, uniq_rows)
        return uniq_tokens, uniq_rows, key

    # alive[row_id] keeps best state for that prefix represented by row_id.
    alive = {
        0: {
            "logp": 0.0,
            "cum_margin": 0.0,
            "cum_logit": 0.0,
            "cum_u": 0.0,
        }
    }
    best_row = 0
    accepted_tokens = 0
    fallback_sample_p = None
    failed = False

    for depth in range(1, max_depth + 1):
        if max_alive_parents > 0 and len(alive) > max_alive_parents:
            top_alive = sorted(
                alive.items(), key=lambda kv: kv[1]["logp"], reverse=True
            )[:max_alive_parents]
            alive = dict(top_alive)

        next_alive = {}
        parent_fallback = {}
        parent_rejected = {}
        parent_step_logits = {} if log_rejections else None
        prev_best_row = max(alive.items(), key=lambda kv: kv[1]["logp"])[0]

        for parent_row, parent_state in alive.items():
            step_logits = logits[parent_row, depth - 1]
            if logits_processor is not None:
                step_logits = logits_processor(None, step_logits[None])[0]
            step_logits = step_logits.float()
            if parent_step_logits is not None:
                parent_step_logits[parent_row] = step_logits

            uniq_tokens, uniq_rows, _parent_key = _get_parent_unique(depth, parent_row)
            if uniq_tokens is None or uniq_tokens.numel() == 0:
                parent_fallback[parent_row] = torch.softmax(step_logits, dim=0)
                parent_rejected[parent_row] = []
                continue

            t_m = int(torch.argmax(step_logits).item())
            log_z = torch.logsumexp(step_logits, dim=0)
            log_p_tm = step_logits[t_m] - log_z
            p_tm = max(math.exp(float(log_p_tm.item())), eps_f)

            log_p_x = step_logits.index_select(0, uniq_tokens) - log_z
            p_x = torch.exp(log_p_x).clamp(min=eps_f)
            delta = log_p_tm - log_p_x
            u_logit = float(lts_cfg.alpha_kappa) * (delta * delta)

            margin_bound_d = _depth_bound(lts_cfg.depth_margin_bounds, depth, base_margin_bound)
            logit_bound_d = _depth_bound(lts_cfg.depth_logit_bounds, depth, base_logit_bound)
            path_margin_budget_d = margin_bound_d * depth * path_margin_scale
            path_logit_budget_d = logit_bound_d * depth * path_logit_scale

            accepted_mask = torch.ones_like(delta, dtype=torch.bool)
            rejected_ids_set = set()

            uniq_list = [int(x) for x in uniq_tokens.tolist()]
            if lts_cfg.deny_token_ids:
                deny_mask = torch.tensor(
                    [(xi in lts_cfg.deny_token_ids) and (xi != t_m) for xi in uniq_list],
                    device=delta.device,
                    dtype=torch.bool,
                )
                accepted_mask &= ~deny_mask
                for idx in torch.nonzero(deny_mask, as_tuple=True)[0].tolist():
                    rejected_ids_set.add(uniq_list[idx])

            last1 = st["recent_q"][-1] if len(st["recent_q"]) >= 1 else None
            last2 = st["recent_q"][-2] if len(st["recent_q"]) >= 2 else None
            for idx, xi in enumerate(uniq_list):
                if not bool(accepted_mask[idx].item()):
                    continue
                if xi == t_m:
                    continue
                if st.get("win_soft", 0) >= lts_cfg.soft_k:
                    accepted_mask[idx] = False
                    rejected_ids_set.add(xi)
                    continue
                if not _recent_token_quota_ok(
                    st, xi, quota_W=lts_cfg.ngram_window, max_count=NGRAM_MAX_COUNT
                ):
                    accepted_mask[idx] = False
                    rejected_ids_set.add(xi)
                    continue
                if last1 == xi and last2 == xi:
                    p_xi = float(p_x[idx].item())
                    if (p_xi / max(p_tm, eps_f)) < TWO_GRAM_ALLOW_NEAR_TOP1:
                        accepted_mask[idx] = False
                        rejected_ids_set.add(xi)

            pre_mask = accepted_mask & (delta <= (margin_bound_d * margin_relax))
            pre_idx = torch.nonzero(pre_mask, as_tuple=True)[0]
            if pre_idx.numel() == 0:
                pre_idx = torch.nonzero(accepted_mask, as_tuple=True)[0]
            if pre_idx.numel() == 0:
                base_probs = torch.softmax(step_logits, dim=0)
                sample_p = base_probs.clone()
                if len(rejected_ids_set) > 0:
                    rej_idx = torch.as_tensor(
                        sorted(rejected_ids_set), device=sample_p.device, dtype=torch.long
                    )
                    sample_p.index_fill_(0, rej_idx, 0.0)
                    s = sample_p.sum()
                    sample_p = sample_p / s if float(s.item()) > 1e-8 else base_probs
                parent_fallback[parent_row] = sample_p
                parent_rejected[parent_row] = sorted(rejected_ids_set)
                continue

            if parent_topk > 0 and pre_idx.numel() > parent_topk:
                _, top_rel = torch.topk(
                    p_x.index_select(0, pre_idx), k=parent_topk, dim=0
                )
                pre_idx = pre_idx[top_rel]

            if pre_idx.numel() < parent_min_keep:
                need = min(parent_min_keep, int(uniq_tokens.numel()))
                _, top_all = torch.topk(p_x, k=need, dim=0)
                pre_idx = torch.unique(torch.cat([pre_idx, top_all], dim=0), sorted=False)

            sel_ids = uniq_tokens.index_select(0, pre_idx)
            sel_rows = uniq_rows.index_select(0, pre_idx)
            sel_delta = delta.index_select(0, pre_idx)
            sel_u_logit = u_logit.index_select(0, pre_idx)
            sel_log_p = log_p_x.index_select(0, pre_idx)

            e_tm_w = (emb_table[t_m] * inv_std).float()
            e_x_w = (emb_table.index_select(0, sel_ids) * inv_std).float()
            diff_w = e_x_w - e_tm_w.unsqueeze(0)
            u_emb = float(lts_cfg.c_s_prime) * (diff_w * diff_w).sum(dim=-1)
            u_mix = torch.minimum(u_emb, sel_u_logit)

            final_accept = (
                (u_mix <= u_bound)
                & (sel_delta <= margin_bound_d)
                & (sel_u_logit <= logit_bound_d)
                & (parent_state["cum_margin"] + sel_delta <= path_margin_budget_d)
                & (parent_state["cum_logit"] + sel_u_logit <= path_logit_budget_d)
            )

            if use_budget and (budget_table is not None) and (len(budget_table) > 0):
                b_idx = min(depth, len(budget_table) - 1)
                b_d = float(budget_table[b_idx])
                final_accept &= (parent_state["cum_u"] + u_mix <= b_d)

            sel_ids_list = [int(x) for x in sel_ids.tolist()]
            sel_rows_list = [int(x) for x in sel_rows.tolist()]
            sel_delta_list = [float(x) for x in sel_delta.tolist()]
            sel_u_logit_list = [float(x) for x in sel_u_logit.tolist()]
            sel_u_mix_list = [float(x) for x in u_mix.tolist()]
            sel_log_p_list = [float(x) for x in sel_log_p.tolist()]
            keep_list = final_accept.tolist()

            any_keep = False
            for k, keep in enumerate(keep_list):
                xi = sel_ids_list[k]
                if not keep:
                    rejected_ids_set.add(xi)
                    continue
                any_keep = True
                child_row = sel_rows_list[k]
                child_logp = parent_state["logp"] + sel_log_p_list[k]
                child_cum_margin = parent_state["cum_margin"] + sel_delta_list[k]
                child_cum_logit = parent_state["cum_logit"] + sel_u_logit_list[k]
                child_cum_u = parent_state["cum_u"] + sel_u_mix_list[k]

                cur = next_alive.get(child_row)
                if (cur is None) or (child_logp > cur["logp"]):
                    next_alive[child_row] = {
                        "logp": child_logp,
                        "cum_margin": child_cum_margin,
                        "cum_logit": child_cum_logit,
                        "cum_u": child_cum_u,
                    }

            if not any_keep:
                base_probs = torch.softmax(step_logits, dim=0)
                sample_p = base_probs.clone()
                if len(rejected_ids_set) > 0:
                    rej_idx = torch.as_tensor(
                        sorted(rejected_ids_set), device=sample_p.device, dtype=torch.long
                    )
                    sample_p.index_fill_(0, rej_idx, 0.0)
                    s = sample_p.sum()
                    sample_p = sample_p / s if float(s.item()) > 1e-8 else base_probs
                parent_fallback[parent_row] = sample_p

            parent_rejected[parent_row] = sorted(rejected_ids_set)

        if len(next_alive) == 0:
            failed = True
            fallback_parent = prev_best_row
            fallback_sample_p = parent_fallback.get(fallback_parent, None)
            if fallback_sample_p is None:
                fallback_logits = logits[fallback_parent, depth - 1]
                if logits_processor is not None:
                    fallback_logits = logits_processor(None, fallback_logits[None])[0]
                fallback_sample_p = torch.softmax(fallback_logits.float(), dim=0)

            if log_rejections:
                rejected_ids = parent_rejected.get(fallback_parent, [])
                if len(rejected_ids) > 0:
                    rejected_map[depth - 1] = rejected_ids
                if parent_step_logits is not None and fallback_parent in parent_step_logits:
                    tp = torch.softmax(parent_step_logits[fallback_parent], dim=0)
                else:
                    tp_logits = logits[fallback_parent, depth - 1]
                    if logits_processor is not None:
                        tp_logits = logits_processor(None, tp_logits[None])[0]
                    tp = torch.softmax(tp_logits.float(), dim=0)
                target_probs_map[depth - 1] = tp.detach().clone()
            break

        alive = next_alive
        accepted_tokens = depth
        best_row, _best_state = max(alive.items(), key=lambda kv: kv[1]["logp"])

        if log_rejections:
            best_key = row_prefix_keys[best_row][depth]
            for parent_row, rej in parent_rejected.items():
                if row_prefix_keys[parent_row][depth] == best_key:
                    if len(rej) > 0:
                        rejected_map[depth - 1] = rej
                    if parent_step_logits is not None and parent_row in parent_step_logits:
                        tp = torch.softmax(parent_step_logits[parent_row], dim=0)
                    else:
                        tp_logits = logits[parent_row, depth - 1]
                        if logits_processor is not None:
                            tp_logits = logits_processor(None, tp_logits[None])[0]
                        tp = torch.softmax(tp_logits.float(), dim=0)
                    target_probs_map[depth - 1] = tp.detach().clone()
                    break

    if not failed:
        final_logits = logits[best_row, accepted_tokens]
        if logits_processor is not None:
            final_logits = logits_processor(None, final_logits[None])[0]
        fallback_sample_p = torch.softmax(final_logits.float(), dim=0)
        if log_rejections:
            target_probs_map[accepted_tokens] = fallback_sample_p.detach().clone()

    for d in range(1, accepted_tokens + 1):
        step_logits = logits[best_row, d - 1]
        if logits_processor is not None:
            step_logits = logits_processor(None, step_logits[None])[0]
        step_probs = torch.softmax(step_logits.float(), dim=0)
        top1_id = int(torch.argmax(step_probs).item())
        accepted_id = int(cand[best_row, d].item())
        _state_accept(accepted_id, top1_id)

    if accepted_tokens < max_depth:
        _state_reject()
    else:
        st["no_accept_steps"] = 0
        if st["rescue_left"] > 0:
            st["rescue_left"] -= 1
            if st["rescue_left"] == 0:
                st["cooldown_left"] = lts_cfg.rescue_cooldown

    return EvaluationResult(
        best_candidate=torch.tensor(best_row, device=candidates.device, dtype=torch.long),
        accept_length=accepted_tokens,
        verified_positions=accepted_tokens,
        sample_p=fallback_sample_p,
        rejected_map=rejected_map,
        target_probs_map=target_probs_map,
    )

@torch.no_grad()
def lts_sync_after_posterior(
    *,
    logits: torch.Tensor,                # [n_paths, seq_len, vocab]
    candidates: torch.Tensor,            # [n_cands, seq_len]
    best_candidate: int,
    accept_length: int,                  
    logits_processor,
    lts_cfg: LTSConfig,
):
    """ARC-Decode helper."""
    st = getattr(lts_cfg, "_state", None)
    if st is None:
        return  

    
    from collections import deque
    st.setdefault("win", deque())
    st.setdefault("win_soft", 0)
    st.setdefault("recent_q", deque())
    st.setdefault("recent_cnt", {})

    
    for i in range(1, accept_length + 1):
        step_logits = logits[best_candidate, i - 1]
        if logits_processor is not None:
            step_logits = logits_processor(None, step_logits[None])[0]
        target_probs = torch.softmax(step_logits, dim=0)
        t_m = int(torch.argmax(target_probs).item())

        accepted_id = int(candidates[best_candidate, i].item())
        st["win"].append(1 if accepted_id != t_m else 0)
        if st["win"][-1] == 1:
            st["win_soft"] += 1
        if len(st["win"]) > lts_cfg.soft_W:
            dropped = st["win"].popleft()
            if dropped == 1:
                st["win_soft"] -= 1

        _recent_token_update(st, accepted_id, quota_W=lts_cfg.ngram_window)

    
    st["win"].append(0)
    if len(st["win"]) > lts_cfg.soft_W:
        dropped = st["win"].popleft()
        if dropped == 1:
            st["win_soft"] -= 1


def _pf_lts_stats(lts_cfg: Optional['LTSConfig']) -> Optional[dict]:
    if lts_cfg is None:
        return None
    if not bool(getattr(lts_cfg, "pf_lts_record_stats", False)):
        return None
    stats = getattr(lts_cfg, "_stats", None)
    if stats is None:
        stats = {
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
        lts_cfg._stats = stats
    return stats


def _pf_lts_stat_inc(stats: Optional[dict], key: str, value: int = 1) -> None:
    if stats is not None:
        stats[key] = int(stats.get(key, 0)) + int(value)


def _pf_lts_profile_value(lts_cfg: 'LTSConfig', name: str, default: float) -> float:
    profile = str(getattr(lts_cfg, "pf_lts_prompt_profile", "default"))
    if profile not in {"default", "code", "math"}:
        profile = "default"
    return float(getattr(lts_cfg, f"{name}_{profile}", getattr(lts_cfg, f"{name}_default", default)))


@torch.no_grad()
def evaluate_pf_lts(
        logits: torch.Tensor,
        candidates: torch.Tensor,
        logits_processor,
        node_logprob: Optional[torch.Tensor] = None,
        retrieve_indices: Optional[torch.Tensor] = None,
        emb_table: Optional[torch.Tensor] = None,
        lts_cfg: Optional['LTSConfig'] = None,
        log_rejections: bool = False,
):
    """ARC-Decode helper."""
    rejected_map = {}
    target_probs_map = {}

    if logits_processor is None:
        posterior_mask = (
                candidates[:, 1:].to(logits.device) == torch.argmax(logits[:, :-1], dim=-1)
        ).int()
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        accept_length = candidates_accept_length.max()
        if accept_length == 0:
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
        result = EvaluationResult(
            best_candidate=best_candidate,
            accept_length=accept_length,
            verified_positions=accept_length,
            sample_p=logits[best_candidate, accept_length],
            rejected_map=rejected_map,
            target_probs_map=target_probs_map,
        )
        return result

    has_lts = (lts_cfg is not None and lts_cfg.inv_std is not None and emb_table is not None)
    if has_lts:
        _u_bound = float(lts_cfg.tau_delta) * (1.0 - float(lts_cfg.theta))
        _alpha_kappa = float(lts_cfg.alpha_kappa)
        _c_s_prime = float(lts_cfg.c_s_prime)
        _inv_std = lts_cfg.inv_std.to(device=logits.device, dtype=logits.dtype)
        _emb_table = emb_table.to(device=logits.device)
        _lam = float(getattr(lts_cfg, 'lts_lambda', 0.5))
        _min_p = float(getattr(lts_cfg, 'min_boost_prob', 0.01))
        _selective = bool(getattr(lts_cfg, 'pf_lts_selective', False))
        _min_acp = _pf_lts_profile_value(lts_cfg, "pf_lts_min_acp", 0.0)
        _max_u_ratio = _pf_lts_profile_value(lts_cfg, "pf_lts_max_u_ratio", 1.0)
        _stats = _pf_lts_stats(lts_cfg)
    else:
        _stats = None

    accept_length = 1
    accept_cand = candidates[0][:1]
    best_candidate = 0

    for i in range(1, candidates.shape[1]):
        if i != accept_length:
            break

        rejected_map[i - 1] = []
        draft_probs = torch.zeros_like(logits[0, 0])

        is_eq = (candidates[:, :accept_length] == accept_cand).all(dim=1)
        fi = torch.nonzero(is_eq, as_tuple=True)[0][0]

        gt_logits = logits[fi, i - 1][None]
        gt_logits = logits_processor(None, gt_logits)[0]
        target_probs = torch.softmax(gt_logits, dim=0)
        target_probs_map[i - 1] = target_probs.clone()
        _pf_lts_stat_inc(_stats, "positions")

        seen = set()
        position_accepted = False
        lts_top1_id = None
        lts_log_p_top1 = None
        lts_e_top1_w = None

        for j in range(candidates.shape[0]):
            if is_eq[j]:
                xi = candidates[j, i].item()

                if xi == -1 or xi in seen:
                    continue
                seen.add(xi)

                r = random.random()
                px = target_probs[xi].item()

                if node_logprob is not None and retrieve_indices is not None:
                    child_idx = int(retrieve_indices[j, i].item())
                    parent_idx = int(retrieve_indices[j, i - 1].item())
                    if (child_idx >= 0 and parent_idx >= 0
                            and child_idx < node_logprob.numel()
                            and parent_idx < node_logprob.numel()):
                        log_q = node_logprob[child_idx].item() - node_logprob[parent_idx].item()
                        qx = max(math.exp(log_q), 1e-10)
                    else:
                        qx = 1.0
                else:
                    qx = 1.0

                acp = min(1.0, px / qx)

                # Standard posterior check (fast path)
                if r <= acp:
                    _pf_lts_stat_inc(_stats, "posterior_accept")
                    accept_cand = torch.cat((accept_cand, candidates[j, i][None]), dim=0)
                    accept_length += 1
                    best_candidate = j
                    position_accepted = True
                    break
                _pf_lts_stat_inc(_stats, "posterior_reject")

                
                if has_lts and _lam > 0:
                    if px < _min_p:
                        _pf_lts_stat_inc(_stats, "blocked_min_prob")
                        if node_logprob is not None and retrieve_indices is not None:
                            draft_probs[xi] = qx
                        else:
                            draft_probs[xi] = target_probs[xi]
                        if log_rejections:
                            rejected_map[i - 1].append(xi)
                        continue
                    _pf_lts_stat_inc(_stats, "boost_evaluated")
                    if _selective and acp < _min_acp:
                        _pf_lts_stat_inc(_stats, "blocked_acp")
                        if node_logprob is not None and retrieve_indices is not None:
                            draft_probs[xi] = qx
                        else:
                            draft_probs[xi] = target_probs[xi]
                        if log_rejections:
                            rejected_map[i - 1].append(xi)
                        continue
                    # Lazy compute top-1 LTS variables (once per position)
                    if lts_top1_id is None:
                        lts_top1_id = int(torch.argmax(target_probs).item())
                        lts_log_p_top1 = math.log(max(float(target_probs[lts_top1_id].item()), 1e-12))
                        lts_e_top1_w = (_emb_table[lts_top1_id] * _inv_std).float()

                    # U_logit: logit-space JS bound
                    log_px = math.log(max(px, 1e-12))
                    delta = lts_log_p_top1 - log_px
                    u_logit = _alpha_kappa * delta * delta

                    # U_emb: embedding-space JS bound
                    e_x_w = (_emb_table[xi] * _inv_std).float()
                    diff_w = e_x_w - lts_e_top1_w
                    u_emb = _c_s_prime * float((diff_w * diff_w).sum().item())

                    # Combined LTS score
                    u_mix = min(u_emb, u_logit)

                    # Smooth boost: proportional to LTS quality
                    if u_mix < _u_bound:
                        u_ratio = u_mix / max(_u_bound, 1e-12)
                        if _selective and u_ratio > _max_u_ratio:
                            _pf_lts_stat_inc(_stats, "blocked_u")
                            if node_logprob is not None and retrieve_indices is not None:
                                draft_probs[xi] = qx
                            else:
                                draft_probs[xi] = target_probs[xi]
                            if log_rejections:
                                rejected_map[i - 1].append(xi)
                            continue
                        _pf_lts_stat_inc(_stats, "boost_candidates")
                        boost = 1.0 - u_mix / _u_bound
                        acp_boosted = min(1.0, acp + _lam * (1.0 - acp) * boost)

                        if r <= acp_boosted:
                            _pf_lts_stat_inc(_stats, "boost_accept")
                            accept_cand = torch.cat((accept_cand, candidates[j, i][None]), dim=0)
                            accept_length += 1
                            best_candidate = j
                            position_accepted = True
                            break
                

                if node_logprob is not None and retrieve_indices is not None:
                    draft_probs[xi] = qx
                else:
                    draft_probs[xi] = target_probs[xi]
                if log_rejections:
                    rejected_map[i - 1].append(xi)

        if not position_accepted:
            final_gt_logits = logits[fi, i - 1][None]
            final_gt_logits = logits_processor(None, final_gt_logits)[0]
            final_target_probs = torch.softmax(final_gt_logits, dim=0)
            target_probs_map[i - 1] = final_target_probs.clone()

            corrected_probs = torch.relu(final_target_probs - draft_probs)
            if corrected_probs.sum() > 1e-8:
                sample_p = corrected_probs / corrected_probs.sum()
            else:
                sample_p = final_target_probs

            result = EvaluationResult(
                best_candidate=torch.tensor(best_candidate),
                accept_length=accept_length - 1,
                verified_positions=accept_length - 1,
                sample_p=sample_p,
                rejected_map=rejected_map,
                target_probs_map=target_probs_map,
            )
            return result

    final_gt_logits = logits[best_candidate, accept_length - 1][None]
    final_gt_logits = logits_processor(None, final_gt_logits)[0]
    final_target_probs = torch.softmax(final_gt_logits, dim=0)
    target_probs_map[accept_length - 1] = final_target_probs.clone()

    result = EvaluationResult(
        best_candidate=torch.tensor(best_candidate),
        accept_length=accept_length - 1,
        verified_positions=accept_length - 1,
        sample_p=final_target_probs,
        rejected_map=rejected_map,
        target_probs_map=target_probs_map,
    )
    return result
