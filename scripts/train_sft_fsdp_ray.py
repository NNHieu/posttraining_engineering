"""Full-shard SFT for prompt/completion data."""

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields, replace
from functools import partial
import itertools
import json
import logging
import time
import math
import sys
import os
os.environ["RAY_TRAIN_V2_ENABLED"] = "1"
from pathlib import Path
import random
import uuid
import ray

import ray.train
import ray.train.torch

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
# FSDP2 sharding imports 
from torch.distributed.fsdp import (
    StateDictType,
    fully_shard,
    CPUOffloadPolicy,
    OffloadPolicy,
    MixedPrecisionPolicy,
)
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)

from transformers import AutoModelForCausalLM, AutoTokenizer
# from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
# from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy, size_based_auto_wrap_policy
from torch.distributed.device_mesh import init_device_mesh 

from src.configs import Config
from src.distributed_checkpointing import load_fsdp_checkpoint, report_metrics_and_save_fsdp_checkpoint
from src.experiment_utils import ExperimentPaths, update_manifest, write_json, append_jsonl
from src.logging_utils import setup_logging
from src.workers.vllm import VLLMRolloutWorker
from vllm.distributed.weight_transfer import (
    ModuleSource,
    HTTPVLLMWeightSyncClient,
    WeightTransferTrainerFactory,
)
from vllm.distributed.weight_transfer.nccl_engine import NCCLTrainerInitInfo

logger = logging.getLogger(__name__)

PROMPT_KEY = "question"
COMPLETION_KEY = "answer"

# @dataclass(frozen=True)
# class Config:
#     model_name: str
#     train_data: str
#     output_dir: str
#     eval_data: str = ""
#     tokenizer: str = ""
#     dataset_cache_dir: str = ""
#     num_workers: int = 1
#     epochs: int = 3
#     batch_size: int = 2
#     gradient_accumulation_steps: int = 8
#     max_length: int = 2048
#     learning_rate: float = 2e-5
#     weight_decay: float = 0.01
#     warmup_ratio: float = 0.03
#     max_grad_norm: float = 1.0
#     eval_fraction: float = 0.05
#     seed: int = 42
#     precision: str = "bfloat16"
#     attn_implementation: str = "sdpa"
#     gradient_checkpointing: bool = True
#     fsdp_min_num_params: int = 1000000
#     save_steps: int = 500
#     eval_steps: int = 100
#     logging_steps: int = 10
#     wandb_project: str = ""
#     run_name: str = "sft"
#     resume: bool = False
#     last_checkpoint_dir: str = ""

#     def __post_init__(self):
#         for name in ("epochs", "batch_size", "gradient_accumulation_steps", "max_length",
#                      "fsdp_min_num_params", "save_steps", "eval_steps", "logging_steps"):
#             if getattr(self, name) <= 0:
#                 raise ValueError(f"{name} must be positive")
#         for name in ("learning_rate", "max_grad_norm"):
#             if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
#                 raise ValueError(f"{name} must be finite and positive")
#         if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
#             raise ValueError("weight_decay must be finite and nonnegative")
#         if not 0 <= self.eval_fraction < 1 or not 0 <= self.warmup_ratio < 1:
#             raise ValueError("eval_fraction and warmup_ratio must be in [0, 1)")
#         if self.precision not in {"bfloat16", "float32"} or self.seed < 0:
#             raise ValueError("precision must be bfloat16 or float32, and seed must be nonnegative")
#         if self.max_length < 2:
#             raise ValueError("max_length must be at least two")

#     @classmethod
#     def from_cli(cls):
#         parser = argparse.ArgumentParser(description=__doc__)
#         for field in fields(cls):
#             options = {"help": field.name.replace("_", " ")}
#             if field.name in {"model_name", "train_data", "output_dir"}:
#                 options.update(required=True, type=str)
#             else:
#                 options["default"] = field.default
#                 if field.type is bool:
#                     options["action"] = argparse.BooleanOptionalAction
#                 else:
#                     options["type"] = field.type
#             parser.add_argument("--" + field.name.replace("_", "-"), **options)
#         try:
#             return cls(**vars(parser.parse_args()))
#         except ValueError as error:
#             parser.error(str(error))


def encode_example(example, tokenizer, max_length):
    """Mask prompt tokens using offsets in the jointly tokenized full text.

    Drop overlength examples rather than truncating a verified answer. Tokens
    crossing the prompt boundary are masked to avoid supervising prompt characters.
    """
    prompt, completion = example.get(PROMPT_KEY), example.get(COMPLETION_KEY)
    if not isinstance(prompt, str) or not isinstance(completion, str) or not completion.strip():
        raise ValueError("Every SFT row needs string prompt and nonempty completion fields")
    encoded = tokenizer(prompt + completion, add_special_tokens=True, return_offsets_mapping=True)
    ids = [int(token) for token in encoded["input_ids"]]
    labels = [token if end > start and start >= len(prompt) else -100
              for token, (start, end) in zip(ids, encoded["offset_mapping"])]
    # EOS is supervised; a preexisting terminal EOS must not be appended twice.
    if not ids or ids[-1] != tokenizer.eos_token_id:
        ids.append(tokenizer.eos_token_id)
        labels.append(tokenizer.eos_token_id)
    else:
        labels[-1] = tokenizer.eos_token_id
    if len(ids) > max_length or not any(label != -100 for label in labels[1:-1]):
        return None
    return {"input_ids": ids, "labels": labels}


def collate_examples(examples, pad_token_id):
    length = max(len(example["input_ids"]) for example in examples)
    return {
        "input_ids": torch.tensor([e["input_ids"] + [pad_token_id] * (length - len(e["input_ids"]))
                                   for e in examples], dtype=torch.long),
        "attention_mask": torch.tensor([[1] * len(e["input_ids"]) + [0] * (length - len(e["input_ids"]))
                                       for e in examples], dtype=torch.long),
        "labels": torch.tensor([e["labels"] + [-100] * (length - len(e["labels"]))
                                for e in examples], dtype=torch.long),
    }


def split_by_prompt(dataset, fraction, seed):
    if fraction == 0:
        return dataset, None
    prompts = sorted(set(dataset[PROMPT_KEY]))
    if len(prompts) < 2:
        raise ValueError("Automatic evaluation split needs two unique prompts; use --eval-fraction 0")
    random.Random(seed).shuffle(prompts)
    held_out = set(prompts[:max(1, int(len(prompts) * fraction))])
    train = dataset.select([i for i, prompt in enumerate(dataset[PROMPT_KEY]) if prompt not in held_out])
    evaluation = dataset.select([i for i, prompt in enumerate(dataset[PROMPT_KEY]) if prompt in held_out])
    return train, evaluation


def prepare_data(config, tokenizer):
    from datasets import load_dataset

    def load(path):
        kind = "parquet" if Path(path).suffix == ".parquet" else "json"
        dataset = load_dataset(kind, data_files=path, split="train", cache_dir=config.dataset_cache_dir or None)
        if not {PROMPT_KEY, COMPLETION_KEY}.issubset(dataset.column_names):
            raise ValueError("SFT data must have prompt and completion columns")
        return dataset

    training = load(config.train_data)
    if config.eval_data:
        evaluation = load(config.eval_data)
        if set(training[PROMPT_KEY]) & set(evaluation[PROMPT_KEY]):
            raise ValueError("Training and evaluation data contain overlapping prompts")
    else:
        training, evaluation = split_by_prompt(training, config.eval_fraction, config.seed)

    def encode(dataset, name):
        if dataset is None:
            return None
        rows = []
        for example in dataset:
            encoded = encode_example(example, tokenizer, config.max_length)
            if encoded is not None:
                rows.append(encoded)
        logger.info("%s: kept %d/%d rows within max_length and with supervised tokens", name, len(rows), len(dataset))
        if not rows:
            raise ValueError(f"No usable {name} examples remain; check max_length and prompt/completion boundaries")
        return rows

    return encode(training, "training"), encode(evaluation, "evaluation")


def causal_loss_sum(logits, labels):
    return F.cross_entropy(logits[:, :-1].contiguous().float().view(-1, logits.size(-1)),
                           labels[:, 1:].contiguous().view(-1), ignore_index=-100, reduction="sum")


def accumulation_windows(loader, steps):
    iterator = iter(loader)
    while window := list(itertools.islice(iterator, steps)):
        yield window


def evaluate(model, loader, device, precision):
    model.eval()
    totals = torch.zeros(2, device=device, dtype=torch.float64)
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            with autocast(precision):
                logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits
                loss = causal_loss_sum(logits, batch["labels"])
            totals[0] += loss.detach()
            totals[1] += (batch["labels"][:, 1:] != -100).sum()
            del logits, loss, batch

    dist.all_reduce(totals)
    model.train()
    return (totals[0] / totals[1]).item()


def autocast(precision):
    return torch.autocast("cuda", dtype=torch.bfloat16) if precision == "bfloat16" else nullcontext()

def shard_model(
        model: torch.nn.Module,
        *,
        world_size: int,
        precision: str,
        cpu_offload: bool) -> None: 
    """Apply FSDP2 sharding to the model with optimized configuration.
    
    Args:
        model: The PyTorch model to shard
    """
    logger.info("Applying FSDP2 sharding to model...")

    # Step 1: Identify transformer blocks in the model
    block_names = set(getattr(model, "_no_split_modules", None) or ())
    blocks = [
        module for module in model.modules()
        if type(module).__name__ in block_names
    ]
    if not blocks:
        raise ValueError(
            f"Could not identify transformer blocks for {type(model).__name__}; "
            "check the model's _no_split_modules metadata"
        )

    # Step 2: Create 1D device mesh for data parallel sharding
    world_size = ray.train.get_context().get_world_size()
    mesh = init_device_mesh(
        device_type="cuda", 
        mesh_shape=(world_size,), 
        mesh_dim_names=("data_parallel",)
    )

    # Step 3: Configure CPU offloading policy (optional)
    offload_policy = CPUOffloadPolicy() if cpu_offload else OffloadPolicy()

    # Step 4: Configure mixed precision policy (optional)
    dtype = torch.bfloat16 if precision == "bfloat16" else None
    mp_policy = MixedPrecisionPolicy(
        param_dtype=dtype,
        reduce_dtype=torch.float32 if dtype is not None else None,
    )

    # Step 5: Apply sharding to each transformer encoder block
    for block in blocks:
        fully_shard(
            block, 
            mesh=mesh, 
            reshard_after_forward=False,   # Free memory after forward pass
            offload_policy=offload_policy, 
            mp_policy=mp_policy
        )

    # Step 6: Apply sharding to the root model
    # This wraps the entire model and enables top-level FSDP2 functionality
    fully_shard(
        model, 
        mesh=mesh, 
        reshard_after_forward=False,   # Free memory after forward pass
        offload_policy=offload_policy, 
        mp_policy=mp_policy
    )

    # model.config.use_cache = False
    # block_names = set(getattr(model, "_no_split_modules", None) or [])
    # blocks = {type(module) for module in model.modules() if type(module).__name__ in block_names}
    # if config.gradient_checkpointing and not blocks:
    #     raise ValueError("Cannot identify transformer blocks; use --no-gradient-checkpointing or a supported model")
    # policy = (partial(transformer_auto_wrap_policy, transformer_layer_cls=blocks) if blocks else
    #           partial(size_based_auto_wrap_policy, min_num_params=config.fsdp_min_num_params))
    # mixed = (MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.bfloat16)
    #          if config.precision == "bfloat16" else None)
    # model = FSDP(model, auto_wrap_policy=policy, sharding_strategy=ShardingStrategy.FULL_SHARD,
    #              mixed_precision=mixed, device_id=device, use_orig_params=True,
    #              limit_all_gathers=True, sync_module_states=True)
    # if config.gradient_checkpointing:
    #     from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    #         apply_activation_checkpointing, checkpoint_wrapper, CheckpointImpl,
    #     )
    #     apply_activation_checkpointing(
    #         model, checkpoint_wrapper_fn=partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
    #         check_fn=lambda module: type(module) in blocks,
    #     )

def train(train_loop_config):
    train_loop_config = dict(train_loop_config)
    vllm_actor = train_loop_config.pop("vllm_rollout_actor")
    config = Config(**train_loop_config)
    device = ray.train.torch.get_device()
    torch.cuda.set_device(device)
    world_rank = ray.train.get_context().get_world_rank()
    world_size = ray.train.get_context().get_world_size()
    # logging.basicConfig(level=logging.INFO if world_rank == 0 else logging.WARNING,
    #                     format="%(asctime)s %(levelname)s %(message)s")
    wandb_run = None
    try:
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        experiment_paths = ExperimentPaths.from_root(config.output_dir)
        logger = setup_logging(rank=world_rank, log_dir=experiment_paths.logs, level=logging.INFO)

        if config.precision == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise ValueError("This GPU does not support BF16; use --precision float32")
        model_path = Path(config.model_name)
        model_name = str(model_path / "hf_model") if (model_path / "hf_model").is_dir() else config.model_name
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer or model_name, use_fast=True)
        if not tokenizer.is_fast or tokenizer.eos_token_id is None:
            raise ValueError("A fast tokenizer with EOS is required for prompt boundary masking")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        training, evaluation = prepare_data(config, tokenizer)
        # sampler = DistributedSampler(training, num_replicas=world_size, rank=rank, seed=config.seed, shuffle=True)
        collator = partial(collate_examples, pad_token_id=tokenizer.pad_token_id)
        loader_generator = torch.Generator().manual_seed(config.seed)
        loader = DataLoader(training, 
                            batch_size=config.batch_size, 
                            # sampler=sampler, 
                            collate_fn=collator,
                            shuffle=True,
                            generator=loader_generator)

        # Pre-loading entire gradient accumulation windows onto GPU
        loader = ray.train.torch.prepare_data_loader(loader, 
                                                     move_to_device=False)

        eval_loader = None
        if evaluation is not None:
            eval_loader = DataLoader(evaluation, batch_size=config.batch_size, collate_fn=collator)
            # Duplicating evaluation samples across ranks: 
            # ray.train.torch.prepare_data_loader(eval_loader) attaches a DistributedSampler(drop_last=False) 
            # that pads the dataset by duplicating examples when len(evaluation) % world_size != 0, skewing eval/loss.
            eval_loader = ray.train.torch.prepare_data_loader(eval_loader)
        
        # Load in bfloat16 first so flash_attention_2 initialization succeeds,
        # then cast to FP32 master weights for FSDP2 mixed precision.
        init_dtype = torch.bfloat16 if config.precision == "bfloat16" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=init_dtype,
            attn_implementation=config.attn_implementation,
        )
        if config.precision == "bfloat16":
            # FP32 master weights; FSDP casts compute parameters to BF16 when enabled.
            model = model.to(dtype=torch.float32)

        if config.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            model.config.use_cache = False
        shard_model(model, 
                    world_size=world_size, 
                    precision=config.precision, 
                    cpu_offload=False)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        windows_per_epoch = math.ceil(len(loader) / config.gradient_accumulation_steps)
        total_steps = config.epochs * windows_per_epoch
        warmup = int(total_steps * config.warmup_ratio)

        def lr_scale(step):
            if warmup and step < warmup:
                return (step + 1) / warmup
            progress = (step - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
        start_epoch = 0
        start_window = 0
        step = 0
        if config.last_checkpoint_dir:
            checkpoint = ray.train.Checkpoint.from_directory(config.last_checkpoint_dir)
            start_epoch, start_window, step = load_fsdp_checkpoint(
                model,
                optimizer,
                checkpoint,
                logger=logger,
                scheduler=scheduler,
            )
        if world_rank == 0:
            logger.info("Training %d rows for %d updates on %d GPUs", len(training), total_steps, world_size)
            if config.wandb_project:
                import wandb
                wandb_run = wandb.init(project=config.wandb_project, name=config.run_name or None, config=asdict(config))
        dist.barrier()
        model.train()
        for epoch in range(start_epoch, config.epochs):
            if hasattr(loader.sampler, "set_epoch"):
                loader.sampler.set_epoch(epoch)
            else:
                loader.generator.manual_seed(config.seed + epoch)

            for window_index, window in enumerate(
                accumulation_windows(loader, config.gradient_accumulation_steps)
            ):
                if epoch == start_epoch and window_index < start_window:
                    continue
                train_step_start = time.time()
                optimizer.zero_grad(set_to_none=True)
                token_count = torch.tensor(sum((batch["labels"][:, 1:] != -100).sum().item() for batch in window),
                                           device=device, dtype=torch.float64)
                dist.all_reduce(token_count)
                if token_count.item() == 0:
                    raise ValueError("Accumulation window has no supervised causal tokens")
                loss_total = torch.zeros((), device=device, dtype=torch.float64)
                num_microbatches = len(window)
                for batch_idx_in_window, batch in enumerate(window):
                    batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                    window[batch_idx_in_window] = None  # Clear reference to free memory early

                    # Only synchronize gradients on the final microbatch
                    model.set_requires_gradient_sync(batch_idx_in_window == num_microbatches - 1)

                    with autocast(config.precision):
                        logits = model(input_ids=batch["input_ids"], 
                                       attention_mask=batch["attention_mask"], 
                                       use_cache=False).logits
                        loss_sum = causal_loss_sum(logits, batch["labels"])
                        # FSDP averages gradients across ranks. Account for variable
                        # response lengths and a partial final accumulation window.
                        loss = loss_sum * (world_size / token_count.item())
                    
                    # Syncing gradients on every microbatch during gradient accumulation: 
                    # loss.backward() triggers a full FSDP2 reduce_scatter across ranks on every microbatch 
                    # instead of accumulating locally and syncing once on the final microbatch.
                    loss.backward()

                    loss_total += loss_sum.detach()
                    del logits, loss, loss_sum, batch

                # PyTorch’s FSDP2-compatible utility
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=config.max_grad_norm,
                )
                if hasattr(grad_norm, "full_tensor"):
                    grad_norm = grad_norm.to(device).full_tensor().item()
                else:
                    grad_norm = grad_norm.item()

                learning_rate = optimizer.param_groups[0]["lr"]
                optimizer.step()
                scheduler.step()
                step += 1
                dist.all_reduce(loss_total)
                train_step_end = time.time()
                metrics = {"train/loss": (loss_total / token_count).item(), 
                           "train/learning_rate": learning_rate,
                           "train/grad_norm": float(grad_norm), 
                           "epoch": epoch + 1, 
                           "step": step,
                           "train/tokens_per_second": token_count.item() / (train_step_end - train_step_start)}
                if eval_loader is not None and (step % config.eval_steps == 0 or step == total_steps):
                    metrics["eval/loss"] = evaluate(model, eval_loader, device, config.precision)

                if world_rank == 0:
                    # app(metrics, experiment_paths.metrics / f"step-{step:06d}.json")
                    append_jsonl(experiment_paths.metrics / "metrics.jsonl", metrics)
                    # Gather weights on Rank 0 and push to vLLM
                
                # Run on every rank: full-state gathering is distributed.
                full_state_dict = get_model_state_dict(
                    model,
                    options=StateDictOptions(full_state_dict=True, cpu_offload=True),
                )

                if world_rank == 0:
                    for name, tensor in full_state_dict.items():
                        full_state_dict[name] = tensor.to(dtype=torch.bfloat16)
                    ray.get(vllm_actor.update_weights.remote(full_state_dict))
                    responses = ray.get(
                        vllm_actor.generate.remote(["Evaluate training progress:"])
                    )
                    logger.info("Sampling output: %s", responses)
                
                dist.barrier()
                del full_state_dict
                
                if world_rank == 0 and (step % config.logging_steps == 0 or step == 1 or "eval/loss" in metrics or step == total_steps):
                    logger.info("step=%d/%d %s", step, total_steps, metrics)
                    if wandb_run:
                        wandb_run.log(metrics, step=step)
                if step % config.save_steps == 0 or step == total_steps:
                    logger.info("Saving checkpoint and reporting metrics...")
                    next_epoch = epoch
                    next_window = window_index + 1
                    if next_window == windows_per_epoch:
                        next_epoch += 1
                        next_window = 0
                    checkpoint_dir = report_metrics_and_save_fsdp_checkpoint(
                        model, 
                        optimizer, 
                        experiment_paths.get_checkpoint_path(step),
                        metrics, 
                        scheduler=scheduler,
                        step=step,
                        epoch=next_epoch,
                        next_window=next_window,
                    )
                    if world_rank == 0:
                        update_manifest(
                            experiment_paths.manifest,
                            "running",
                            last_checkpoint_dir=str(checkpoint_dir),
                        )
                    dist.barrier()
                    logger.info(f"Checkpoint saved successfully. Metrics: {metrics}")
    finally:
        if wandb_run:
            wandb_run.finish()


if __name__ == "__main__":
    config = Config.from_cli()

    # Always resolve to an absolute path so Ray workers and RunConfig
    # point to the exact same shared filesystem location.
    destination = Path(config.output_dir).expanduser().resolve()
    config = replace(config, output_dir=str(destination))

    # Avoid silently overwriting an earlier training run.
    if destination.exists() and any(destination.iterdir()):
        # raise FileExistsError(f"Output directory {destination} must be empty")
        if not config.resume:
            raise FileExistsError(f"Output directory {destination} must be empty or use --resume to continue training")
        else:
            logger.info(f"Output directory {destination} already exists and is not empty. Continuing training may overwrite existing files.")

    
    experiment_paths = ExperimentPaths.from_root(destination)
    experiment_paths.mkdir()
    manifest = {}
    if experiment_paths.manifest.exists():
        manifest = json.loads(experiment_paths.manifest.read_text(encoding="utf-8"))
    if config.resume:
        checkpoint_dir = manifest.get("last_checkpoint_dir")
        if not checkpoint_dir or not Path(checkpoint_dir).is_dir():
            raise FileNotFoundError(
                f"No resumable checkpoint recorded in {experiment_paths.manifest}"
            )
        config = replace(config, last_checkpoint_dir=checkpoint_dir)
    (experiment_paths.config / "config.json").write_text(json.dumps(asdict(config), indent=2))
    
    # Update the run manifest to indicate that training is starting, including the run name and command-line arguments.
    manifest = update_manifest(
        experiment_paths.manifest,
        "running",
        run_name=config.run_name,
        script_args=" ".join(sys.argv),
    )
    if config.last_checkpoint_dir:
        logger.info("Resuming training from checkpoint: %s", config.last_checkpoint_dir)

    ray.init(
        ignore_reinit_error=True,
    )

    scaling_config = ray.train.ScalingConfig(
        num_workers=config.num_workers,      # Number of distributed workers
        use_gpu=True        # Enable GPU training
    )

    # Ray Train writes to <storage_path>/<name>.
    # Setting storage_path=destination.parent and name=destination.name
    # makes Ray Train use `destination` directly as the experiment folder.
    storage_path = str(destination.parent)
    experiment_name = config.run_name or destination.name
    # experiment_name=f"fsdp_ray_{uuid.uuid4().hex[:8]}"

    # Configure run settings and storage
    run_config = ray.train.RunConfig(
        # Persistent storage path accessible across all worker nodes
        storage_path=str(destination / "ray_runs"),
        # Unique experiment name (use consistent name to resume from checkpoints)
        name=experiment_name,
        # Fault tolerance configuration
        failure_config=ray.train.FailureConfig(max_failures=0),
    )

    # A. Instantiate vLLM actor on GPU 4
    vllm_rollout = VLLMRolloutWorker.remote(config.model_name)

    train_loop_config=asdict(config)
    train_loop_config["vllm_rollout_actor"] = vllm_rollout

    # Initialize and launch the distributed training job
    trainer = ray.train.torch.TorchTrainer(
        train_loop_per_worker=train,
        scaling_config=scaling_config,
        train_loop_config=train_loop_config,
        run_config=run_config,
    )

    print(f"Starting FSDP2 training job in {Path(storage_path) / experiment_name}...")
    result = trainer.fit()
    print("Training completed successfully!")

