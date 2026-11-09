from collections import defaultdict
from math import comb
import gc
import time
from typing import Any, Dict, Literal, Tuple, Union

from datasets import Dataset
import numpy as np
import torch
import torch.distributed as dist
from tqdm import trange
from transformers import AutoModelForCausalLM
from vllm import LLM, SamplingParams, TokensPrompt
from ..behavior import PATTERNS, annotate
from .utils import compute_token_log_probs, masked_mean, combine_and_pad, close_to_zero
from .fsdp_checkpoint import find_fsdp_checkpoint

EPSODE_QUERY_TEXTS_KEY = "all_prompt_texts"
EPSODE_RESPONSE_TEXTS_KEY = "all_response_texts"
EPSODE_QUERY_IDS_KEY = "all_query_token_ids"
EPSODE_RESPONSE_IDS_KEY = "all_response_token_ids"
EPSODE_ADVANTAGES_KEY = "advantages"
EPSODE_REWARDS_KEY = "all_rewards"

def compute_group_normalized_rewards(
    group_size,
    rewards,
    advantage_eps,
    normalize_by_std,
):
    rewards = np.asarray(rewards, dtype=np.float64)
    grouped_rewards = rewards.reshape((-1, group_size))
    group_means = grouped_rewards.mean(axis=1, keepdims=True)
    group_advantages = grouped_rewards - group_means
    if normalize_by_std:
        assert group_size > 1
        group_std = grouped_rewards.std(axis=1, ddof=1, keepdims=True)
        group_advantages = group_advantages / (group_std + advantage_eps)
    advantages = group_advantages.reshape(-1)
    return advantages

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute the policy-gradient loss at every token, where raw_rewards_or_advantages is either the raw reward or an already-normalized advantage.

    Args:
        raw_rewards_or_advantages (torch.Tensor): Shape (batch_size, 1), scalar
            reward/advantage for each rollout response.
        policy_log_probs (torch.Tensor): Shape (batch_size, sequence_length), logprobs
            for each token.

    Returns:
        torch.Tensor: Shape (batch_size, sequence_length), the per-token policy-gradient
            loss (to be aggregated across the batch and sequence dimensions in the
            training loop).
    """
    return -raw_rewards_or_advantages * policy_log_probs

def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    
    prob_ratio = torch.exp(policy_log_probs - old_log_probs)
    noclip_values = prob_ratio * advantages
    clipping_values = torch.clamp(
        prob_ratio, 
        min=1 - cliprange,
        max=1 + cliprange
    ) * advantages
    loss = -torch.minimum(clipping_values, noclip_values)
    with torch.no_grad():
        # Tokens where the ratio itself exceeded [1 - cliprange, 1 + cliprange]
        is_clipped = (prob_ratio < (1.0 - cliprange)) | (prob_ratio > (1.0 + cliprange))
        # Tokens where the clipped objective was smaller than the unclipped objective (active clip in min)
        is_clipped_taken = clipping_values < noclip_values

        metadata = {
            "clip_fraction": is_clipped.float().mean(),
            "min_take_fraction": is_clipped_taken.float().mean(),
            # "ratio": prob_ratio.detach(),
            # "is_clipped": is_clipped,
            # "is_clipped_taken": is_clipped_taken,
        }

    return loss, metadata


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None= None,
    advantages: torch.Tensor | None= None,
    old_log_probs: torch.Tensor | None= None,
    cliprange: float | None= None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Select and compute the desired policy-gradient loss.
    Args:
        policy_log_probs (batch_size, sequence_length): per-token log-probabilities from the
        policy being trained.
        loss_type One of "no_baseline", "reinforce_with_baseline", or "grpo_clip".
        raw_rewards Required if loss_type == "no_baseline"; shape (batch_size, 1).
        advantages Required for"reinforce_with_baseline" and "grpo_clip"; shape
        (batch_size, 1).
        old_log_probs Required for"grpo_clip"; shape (batch_size, sequence_length).
        cliprange Required for"grpo_clip"; scalar ϵ used for clipping.
    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
        loss (batch_size, sequence_length), per-token loss.
        metadata dict, statistics from the underlying routine (e.g., clip fraction for GRPO-Clip).
    """
    if loss_type == "no_baseline":
        assert raw_rewards is not None
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        metadata = {}
    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        metadata = {}
    elif loss_type == "grpo_clip":
        assert advantages is not None
        loss, metadata = compute_grpo_clip_loss(
            advantages, policy_log_probs, old_log_probs, cliprange
        )
    else:
        raise ValueError
    return loss, metadata

def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None= None,
    advantages: torch.Tensor | None= None,
    old_log_probs: torch.Tensor | None= None,
    cliprange: float | None= None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pertoken_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        old_log_probs=old_log_probs,
        raw_rewards=raw_rewards,
        advantages=advantages,
        loss_type=loss_type,
        cliprange=cliprange,
    )
    loss = masked_mean(
        pertoken_loss,
        response_mask,
        dim=1
    )
    loss = loss.mean() / gradient_accumulation_steps
    loss.backward()

    return loss.detach(), metadata

def generate_and_score(
    samples,
    inference_engine,
    sampling_params: SamplingParams,
    reward_fn,
    metadata,  
):
    """Generate, score, and record numeric metadata in prompt/completion order.

    ``behavior/<label>_count`` lists count sentences containing each heuristic
    behavior in the decoded completion (not the input prompt).
    ``pass@k`` (1 <= k <= n) contains one estimate per prompt, using binary
    correctness from answer_reward, equation_reward, or accuracy (in that order).
    If any completion lacks a correctness metric, pass@k is omitted. Total
    reward is never treated as correctness. The estimator assumes independently
    sampled completions; correlated decoding can affect its interpretation.

    Returns a dict with keys ``EPSODE_QUERY_IDS_KEY``,
    ``EPSODE_RESPONSE_IDS_KEY``, ``EPSODE_QUERY_TEXTS_KEY``,
    ``EPSODE_RESPONSE_TEXTS_KEY``, and ``EPSODE_REWARDS_KEY``. The text lists are
    per completion (prompt text repeated for each of a prompt's generations).
    Prompt text comes from the vLLM output when set, otherwise from the
    samples' ``prompt`` column when present.
    """
    if len(samples) == 0:
        raise ValueError("generate_and_score requires at least one sample")
    generations_per_sample = sampling_params.n
    formatted_inputs = [TokensPrompt(prompt_token_ids=samples["input_ids"][i]) for i in range(len(samples))]
    start_gen_time = time.time()
    outputs = inference_engine.generate(
        formatted_inputs,
        sampling_params=sampling_params,
    )
    metadata['gen_time'] = [time.time() - start_gen_time]
    if len(outputs) != len(samples):
        raise ValueError(f"Expected {len(samples)} request outputs, received {len(outputs)}")
    for index, output in enumerate(outputs):
        if len(output.outputs) != generations_per_sample:
            raise ValueError(
                f"Prompt {index}: expected {generations_per_sample} completions, "
                f"received {len(output.outputs)}"
            )

    all_query_token_ids = []
    all_responses_token_ids = []
    all_prompt_texts = []
    all_response_texts = []
    all_finish_reasons = []
    all_rewards = []
    metadata["response_lengths"] = []
    metadata["non_stop_rate"] = []
    metadata["rewards"] = []
    # Metadata may be reused with a different number of completions.
    for key in list(metadata):
        if key.startswith("pass@"):
            del metadata[key]
    correct_counts = []
    has_correctness = True
    for name in PATTERNS:
        metadata[f"behavior/{name}_count"] = []
    for sample, outs in zip(samples, outputs):
        prompt_text = outs.prompt if getattr(outs, "prompt", None) is not None else sample.get("prompt")
        correct_count = 0
        for ep in outs.outputs:
            all_query_token_ids.append(list(outs.prompt_token_ids))
            all_responses_token_ids.append(list(ep.token_ids))
            all_prompt_texts.append(prompt_text)
            all_response_texts.append(ep.text)
            all_finish_reasons.append(ep.finish_reason)

            resp_text = ep.text
            reward, reward_metrics = reward_fn(resp_text, sample)
            all_rewards.append(reward)
            correctness = next(
                (reward_metrics[key] for key in ("answer_reward", "equation_reward", "accuracy")
                 if key in reward_metrics), None,
            )
            if correctness is None:
                has_correctness = False
            else:
                if correctness not in (0, 1):
                    raise ValueError("pass@k requires binary correctness metrics (0 or 1)")
                correct_count += int(correctness)
            
            for k, v in reward_metrics.items():
                metadata.setdefault(f"reward_metrics/{k}", []).append(v)
            metadata["response_lengths"].append(len(ep.token_ids))
            metadata["non_stop_rate"].append(ep.finish_reason != "stop")
            metadata["rewards"].append(reward)
            for name, annotation in annotate(resp_text).items():
                metadata[f"behavior/{name}_count"].append(annotation["count"])
        correct_counts.append(correct_count)

    if has_correctness:
        n = generations_per_sample
        for logk in range(0, n + 1):
            k = 2**logk
            if k > n:
                break
            metadata[f"pass@{k}"] = [
                1.0 - comb(n - c, k) / comb(n, k) if n - c >= k else 1.0
                for c in correct_counts
            ]

    assert len(all_responses_token_ids) == len(all_finish_reasons)
    assert len(all_responses_token_ids) == len(samples) * generations_per_sample

    # metadata["reward_mean"] = float(np.mean(all_rewards))
    # metadata["reward_max"] = float(np.max(all_rewards))
    # metadata["reward_min"] = float(np.min(all_rewards))

    return {
        EPSODE_QUERY_IDS_KEY: all_query_token_ids,
        EPSODE_RESPONSE_IDS_KEY: all_responses_token_ids,
        EPSODE_QUERY_TEXTS_KEY: all_prompt_texts,
        EPSODE_RESPONSE_TEXTS_KEY: all_response_texts,
        EPSODE_REWARDS_KEY: all_rewards,
    }

def create_training_episodes(
    config,
    samples,
    inference_engine: LLM,
    sampling_params,
    reward_fn,
    normalize_by_std: bool,
    advantage_eps: float = 1e-6,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if sampling_params.n != config.generations_per_sample:
        raise ValueError("sampling_params.n must match config.generations_per_sample")
    metadata = {}
    # indices = sampler_rng.choice(len(train_dataset), size=config.samples_per_rank, replace=False)
    # samples = train_dataset.select(indices)
    
    generated = generate_and_score(
        samples, inference_engine, sampling_params,
        reward_fn = reward_fn,
        metadata = metadata,
    )
    all_query_token_ids = generated[EPSODE_QUERY_IDS_KEY]
    all_responses_token_ids = generated[EPSODE_RESPONSE_IDS_KEY]
    all_rewards = generated[EPSODE_REWARDS_KEY]

    advantages = compute_group_normalized_rewards(
        group_size=config.generations_per_sample,
        rewards=all_rewards,
        normalize_by_std=normalize_by_std,
        advantage_eps=advantage_eps,
    )[:, None]

    # episodes = {
    #     EPSODE_QUERY_IDS_KEY: all_query_token_ids,
    #     EPSODE_RESPONSE_IDS_KEY: all_responses_token_ids,
    #     EPSODE_ADVANTAGES_KEY: advantages,
    # }
    generated[EPSODE_ADVANTAGES_KEY] = advantages
    episodes = generated

    return episodes, metadata

def create_test_episodes(
    test_dataset: Dataset,
    inference_engine: LLM,
    sampling_params,
    reward_fn,
):
    metadata = {}    
    generated = generate_and_score(
        test_dataset, inference_engine, sampling_params,
        reward_fn=reward_fn,
        metadata=metadata,
    )
    # all_query_token_ids = generated[EPSODE_QUERY_IDS_KEY]
    # all_responses_token_ids = generated[EPSODE_RESPONSE_IDS_KEY]
    # all_rewards = generated[EPSODE_REWARDS_KEY]

    # episodes = {
    #     EPSODE_QUERY_IDS_KEY: all_query_token_ids,
    #     EPSODE_RESPONSE_IDS_KEY: all_responses_token_ids,
    # }
    episodes = generated
    
    return episodes, metadata


def grpo_step(config, tokenizer, reference_model, policy_model, optimizer, episodes):
    batches = []
    metadata = {}
    curr_cuda_device = torch.device("cuda")

    padding_fractions = []
    total_response_len = torch.zeros(1, device=torch.cuda.current_device())
    for i in range(0, config.episodes_per_rank, config.per_device_batch_size):
        batch = {k: v[i : i + config.per_device_batch_size] for k, v in episodes.items()}
        batch.update(combine_and_pad(batch[EPSODE_QUERY_IDS_KEY], 
                                     batch[EPSODE_RESPONSE_IDS_KEY], 
                                     tokenizer))
        batch[EPSODE_ADVANTAGES_KEY] = torch.tensor(batch[EPSODE_ADVANTAGES_KEY], dtype=torch.float)

        # Compute metadata on CPU before moving tensors to CUDA
        total_tokens = batch['attention_mask'].numel()
        padding_fractions.append(
            1.0 - batch['attention_mask'].sum().item() / total_tokens
            if total_tokens else 0.0
        )
        total_response_len += batch['response_mask'].sum().float()

        # Move to current device
        for k,v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(curr_cuda_device)
        batches.append(batch)


    metadata['padding_fractions'] = padding_fractions
    
    dist.all_reduce(total_response_len, op=dist.ReduceOp.SUM)
    metadata['total_response_len'] = total_response_len.item()
    if total_response_len.item() == 0:
        raise ValueError("Generated batch has no response tokens")
    
    reference_model.eval()
    with torch.no_grad():
        for i in trange(
            0,
            len(batches),
            desc="Computing reference logprobs",
            disable=dist.get_rank() != 0,
        ):
            batch = batches[i]
            # FSDP CPUOffload will seamlessly handle the to/from GPU swaps during this computation
            batch_ref_log_probs = compute_token_log_probs(
                model=reference_model,
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                label_ids=batch['label_ids'],
                temperature=config.temperature,
            )
            batches[i]['ref_log_probs'] = batch_ref_log_probs
        del batch_ref_log_probs
    
    # gc.collect()
    # torch.cuda.empty_cache()
    # time.sleep(config.cleanup_delay_seconds)

    metrics = defaultdict(list)
    policy_model.train()
    train_time = time.time()
    # Compensate for FSDP gradient averaging to normalize by global tokens.
    # total_response_len = total_response_len / dist.get_world_size()
    # Zero gradients explicitly prior to our accumulation loop
    optimizer.zero_grad()
    for i in trange(
        0,
        len(batches),
        desc="Computing PG loss and backward",
        disable=dist.get_rank() != 0,
    ):
        batch = batches[i]

        advantages = batch[EPSODE_ADVANTAGES_KEY]

        policy_log_probs = compute_token_log_probs(
            model=policy_model,
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            label_ids=batch['label_ids'],
            temperature=config.temperature,
        )
        pg_loss, loss_metadata = compute_policy_gradient_loss(
            policy_log_probs=policy_log_probs,
            advantages=advantages,
            loss_type="reinforce_with_baseline",
        )
        pg_loss = pg_loss.masked_fill(~batch["response_mask"], 0.0)
        # pg_loss = masked_mean(
        #     pertoken_loss,
        #     batch['response_mask'],
        #     dim=1
        # )
        # pg_loss = pg_loss.mean() / gradient_accumulation_steps

        ref_logratio = batch['ref_log_probs'].float() - policy_log_probs.float()
        k3_kl_penalty = torch.expm1(ref_logratio) - ref_logratio 
        k3_kl_penalty = k3_kl_penalty.masked_fill(~batch["response_mask"], 0.0)

        loss = (pg_loss + config.kl_coeff * k3_kl_penalty).sum() / (total_response_len / dist.get_world_size())

        with torch.no_grad():
            # The .item() calls used to append loss, policy-loss, KL, and entropy 
            # metrics force the GPU to finish work before continuing. 
            # Accumulate detached metric tensors on-device and convert them 
            # to Python values once after the loop. 
            response_log_probs = policy_log_probs[batch["response_mask"]]
            metrics['loss'].append(loss.detach())

            metrics['policy_loss'].append(pg_loss.sum().detach() / (total_response_len / dist.get_world_size()))
            metrics['kl_penalty'].append(k3_kl_penalty.sum().detach() / (total_response_len / dist.get_world_size()))
            metrics['entropy'].append(-response_log_probs.detach().sum() / (total_response_len / dist.get_world_size()))
            # "zero_advantages_ratio": zero_advantages.item() / total_response_len.item(),
        loss.backward()
        del loss

    for k,v in metrics.items():
        metrics[k] = [t.item() for t in v]
    
    # FSDP Gradient Clipping & Step
    grad_norm = policy_model.clip_grad_norm_(config.max_grad_norm)
    optimizer.step()
    metadata['train_time'] = time.time() - train_time

    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(config.cleanup_delay_seconds)
    metadata['grad_norm'] = grad_norm.item()
    metadata['metrics'] = metrics
    return metadata


# class VllmInferenceEngine:
#     def __init__(self, config, seed, EOS_TOKEN_ID):
#         self.config = config
#         self.inference_engine = LLM(
#             model=config.model_name,
#             skip_tokenizer_init=False,
#             gpu_memory_utilization=config.gpu_memory_utilization,
#             enable_prefix_caching=config.enable_prefix_caching,
#             swap_space=config.swap_space,
#             scheduling_policy=config.scheduling_policy,
#             dtype=getattr(torch, config.model_dtype),
#             max_model_len=config.max_model_len,
#             tensor_parallel_size=config.tensor_parallel_size,
#             distributed_executor_backend=config.distributed_executor_backend,
#             seed=seed,
#             max_num_batched_tokens=config.max_num_batched_tokens,
#             enable_sleep_mode=config.enable_sleep_mode,
#             logprobs_mode=config.logprobs_mode,
#         )
#         generation_kwargs = {
#             "n": config.generations_per_sample,
#             "repetition_penalty": config.repetition_penalty,
#             "temperature": config.temperature,
#             "top_p": config.top_p,
#             "top_k": config.top_k,
#             "min_p": config.min_p,
#             "max_tokens": config.max_response_tokens,
#             "logprobs": 0,  
#             "detokenize": False,
#             "stop_token_ids": [EOS_TOKEN_ID],
#         }
#         self.sampling_params = SamplingParams(**generation_kwargs)
#         self.last_weight_loaded_iteration = None

#     def wakeup_and_load_weight(self):
#         if self.config.enable_sleep_mode:
#             torch.cuda.empty_cache()
#             self.inference_engine.wake_up(tags=['weights'])
#             self.inference_engine.collective_rpc("reload_weights")

#         if config.enable_sleep_mode:
#             inference_engine.wake_up(tags=['kv_cache'])
        
