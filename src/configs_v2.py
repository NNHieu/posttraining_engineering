"""Configuration dataclasses for training.

Training knobs use the same names as ``transformers.TrainingArguments``
(``num_train_epochs``, ``per_device_train_batch_size``, ``dataloader_num_workers``,
``bf16``, ...) so the values can be mapped directly onto a
``TrainingArguments`` instance.

The settings are split into mixins so each training script can combine the
parts it needs::

    @dataclass(frozen=True, kw_only=True)
    class SFTConfig(ModelConfig, SFTDatasetConfig, TrainerConfig):
        pass

``BaseConfig.validate`` uses cooperative ``super()`` calls, so every mixin's
validation runs for combined classes.  Fields are keyword-only to make
multiple inheritance work regardless of which mixin declares required fields.
"""

import argparse
import math
from dataclasses import MISSING, asdict, dataclass, fields, replace


@dataclass(frozen=True, kw_only=True)
class BaseConfig:
    def to_dict(self):
        return asdict(self)

    def replace(self, **kwargs):
        return replace(self, **kwargs)

    def __post_init__(self):
        self.validate()

    def validate(self):
        """Validate this config. Subclasses call ``super().validate()`` first."""

    @classmethod
    def from_cli(cls):
        parser = argparse.ArgumentParser(description=__doc__)
        for field in fields(cls):
            options = {"help": field.name.replace("_", " ")}
            if field.default is MISSING and field.default_factory is MISSING:
                options["required"] = True
            else:
                options["default"] = (field.default if field.default is not MISSING
                                      else field.default_factory())
            if field.type is bool:
                options["action"] = argparse.BooleanOptionalAction
            else:
                options["type"] = field.type
            parser.add_argument("--" + field.name.replace("_", "-"), **options)
        try:
            return cls(**vars(parser.parse_args()))
        except ValueError as error:
            parser.error(str(error))


@dataclass(frozen=True, kw_only=True)
class ModelConfig(BaseConfig):
    model_name: str
    tokenizer: str = ""
    attn_implementation: str = "flash_attention_2"
    max_length: int = 2048

    def validate(self):
        super().validate()
        if self.max_length < 2:
            raise ValueError("max_length must be at least two")


@dataclass(frozen=True, kw_only=True)
class SFTDatasetConfig(BaseConfig):
    train_data: str
    eval_data: str = ""
    dataset_cache_dir: str = ""
    eval_fraction: float = 0.05

    def validate(self):
        super().validate()
        if not 0 <= self.eval_fraction < 1:
            raise ValueError("eval_fraction must be in [0, 1)")


@dataclass(frozen=True, kw_only=True)
class TrainerConfig(BaseConfig):
    num_fsdp_workers: int
    output_dir: str
    num_vllm_workers: int = 1
    dataloader_num_workers: int = 1
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    seed: int = 42
    bf16: bool = True
    gradient_checkpointing: bool = True
    fsdp_min_num_params: int = 1000000
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 10
    wandb_project: str = ""
    run_name: str = "sft"
    resume: bool = False
    last_checkpoint_dir: str = ""

    def validate(self):
        super().validate()
        for name in ("num_train_epochs", "per_device_train_batch_size",
                     "gradient_accumulation_steps", "fsdp_min_num_params",
                     "save_steps", "eval_steps", "logging_steps"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.dataloader_num_workers < 0:
            raise ValueError("dataloader_num_workers must be nonnegative")
        for name in ("learning_rate", "max_grad_norm"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")


@dataclass(frozen=True, kw_only=True)
class Config(ModelConfig, SFTDatasetConfig, TrainerConfig):
    """Default combination used by the SFT training script."""
