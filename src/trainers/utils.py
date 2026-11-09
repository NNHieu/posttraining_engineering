from datetime import timedelta
import json
import os
import shutil
import socket
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoTokenizer, PreTrainedModel

def log_softmax_and_gather(
        logits: torch.Tensor, 
        index: torch.Tensor) -> torch.Tensor:
    return torch.gather(logits.log_softmax(dim=-1), dim=-1, 
                        index=index.unsqueeze(-1)).squeeze(-1)

def compute_token_log_probs(
    model: Union[PreTrainedModel],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    label_ids: torch.Tensor,
    temperature: float = None,
) -> torch.Tensor:
    """
    Compute log probabilities for each token in the sequence.
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
        use_cache=False,
    )

    logits = outputs.logits # Shape: [batch_size, seq_len, vocab_size]
    if temperature is not None:
        logits = logits / temperature  

    # Calculate log probabilities
    # log_probs = log_softmax_and_gather(logits, label_ids)  # Shape: [batch_size, seq_len]
    
    vocab_size = logits.shape[-1]
    log_probs = -torch.nn.functional.cross_entropy(
        logits.reshape(-1, vocab_size), 
        label_ids.reshape(-1),
        reduction="none",
    ).view_as(label_ids)
    return log_probs

def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None= None,
) -> torch.Tensor:
    return tensor.masked_fill(mask != 1, 0.0).sum(dim=dim) / mask.sum(dim=dim)

def combine_and_pad_prompt_and_response(prompt_token_ids, responses_token_ids, tokenizer) -> dict[str, torch.Tensor]:
    combined_ids = []
    response_lengths = []

    for row, (prompt, response) in enumerate(zip(
        prompt_token_ids, responses_token_ids, strict=True
    )):
        if not isinstance(prompt, (list)):
            raise TypeError(
                f"Prompt at row {row} must be a list, "
                f"got {type(prompt).__name__}"
            )
        if not isinstance(response, list):
            raise TypeError(
                f"Response at row {row} must be a list, "
                f"got {type(response).__name__}"
            )
        combined_ids.append(prompt + response)
        response_lengths.append(len(response))

    padded = tokenizer.pad(
        {"input_ids": combined_ids},
        padding=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    ids = padded["input_ids"]
    attention = padded["attention_mask"].bool()
    device = ids.device
    width = ids.shape[1]

    response_lengths = torch.tensor(response_lengths, device=device)

    if tokenizer.padding_side == "left":
        ends = torch.full_like(response_lengths, width)
    else:
        ends = torch.tensor(
            [len(sequence) for sequence in combined_ids],
            device=device,
        )

    starts = ends - response_lengths

    # Positions of the labels in the original padded sequences.
    positions = torch.arange(1, width, device=device)[None, :]
    response_mask = (
        (positions >= starts[:, None])
        & (positions < ends[:, None])
        & attention[:, 1:]
    )

    return {
        "input_ids": ids[:, :-1],
        "label_ids": ids[:, 1:],
        "response_mask": response_mask,
        "attention_mask": attention[:, :-1],
    }

def close_to_zero(tensor: torch.Tensor, mask: torch.Tensor, threshold: float = 1e-8) -> torch.Tensor:
    """
    Computes the number of values in the tensor that are close to zero.
    """
    close_to_zero_mask = torch.abs(tensor) < threshold
    num_close_to_zero = (close_to_zero_mask * mask).sum()
    return num_close_to_zero