"""Configuration for the FSDP GRPO runner (no GPU dependencies)."""
import argparse
import math
from dataclasses import MISSING, asdict, dataclass, fields
from pathlib import Path
from typing import get_args, get_type_hints

@dataclass(frozen=True)
class Config:
    num_fsdp_workers: int
    model_name: str
    train_data: str
    output_dir: str
    eval_data: str = ""
    tokenizer: str = ""
    num_vllm_workers: int = 1
    dataset_cache_dir: str = ""
    epochs: int = 3
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_length: int = 2048
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    eval_fraction: float = 0.05
    seed: int = 42
    precision: str = "bfloat16"
    attn_implementation: str = "sdpa"
    gradient_checkpointing: bool = True
    fsdp_min_num_params: int = 1000000
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 10
    wandb_project: str = ""
    run_name: str = "sft"
    resume: bool = False
    last_checkpoint_dir: str = ""
    profile: bool = False
    profile_warmup_steps: int = 2
    profile_steps: int = 3

    def __post_init__(self):
        if self.profile_steps <= 0 or self.profile_warmup_steps < 0:
            raise ValueError("profile_steps must be positive and profile_warmup_steps nonnegative")
        for name in ("epochs", "batch_size", "gradient_accumulation_steps", "max_length",
                     "fsdp_min_num_params", "save_steps", "eval_steps", "logging_steps"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("learning_rate", "max_grad_norm"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if not 0 <= self.eval_fraction < 1 or not 0 <= self.warmup_ratio < 1:
            raise ValueError("eval_fraction and warmup_ratio must be in [0, 1)")
        if self.precision not in {"bfloat16", "float32"} or self.seed < 0:
            raise ValueError("precision must be bfloat16 or float32, and seed must be nonnegative")
        if self.max_length < 2:
            raise ValueError("max_length must be at least two")

    @classmethod
    def from_cli(cls):
        parser = argparse.ArgumentParser(description=__doc__)
        for field in fields(cls):
            options = {"help": field.name.replace("_", " ")}
            if field.name in {"model_name", "train_data", "output_dir"}:
                options.update(required=True, type=str)
            else:
                options["default"] = field.default
                if field.type is bool:
                    options["action"] = argparse.BooleanOptionalAction
                else:
                    options["type"] = field.type
            parser.add_argument("--" + field.name.replace("_", "-"), **options)
        try:
            return cls(**vars(parser.parse_args()))
        except ValueError as error:
            parser.error(str(error))

    def to_dict(self):
        return asdict(self)
