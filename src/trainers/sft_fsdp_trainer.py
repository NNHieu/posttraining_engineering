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
from pathlib import Path
import random
import uuid
import ray
import ray.train

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
# FSDP2 sharding imports 
from torch.distributed.fsdp import (
    fully_shard,
    CPUOffloadPolicy,
    OffloadPolicy,
    MixedPrecisionPolicy,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.distributed.device_mesh import init_device_mesh 

from src.configs import Config
from src.distributed_checkpointing import load_fsdp_checkpoint, report_metrics_and_save_fsdp_checkpoint
from src.experiment_utils import ExperimentPaths, update_manifest, write_json, append_jsonl
from src.logging_utils import setup_logging
from .utils import combine_and_pad_prompt_and_response

# from vllm.distributed.weight_transfer import (
#     HTTPVLLMWeightSyncClient,
#     ModuleSource,
#     WeightTransferTrainerFactory,
# )
# from vllm.distributed.weight_transfer.nccl_engine import NCCLTrainerInitInfo
# from vllm.utils.network_utils import get_ip, get_open_port

logger = logging.getLogger(__name__)

PROMPT_KEY = "question"
COMPLETION_KEY = "answer"

def collate_examples(examples, tokenizer):
    batch = combine_and_pad_prompt_and_response(
        [e["prompt_token_ids"] for e in examples],
        [e["responses_token_ids"] for e in examples],
        tokenizer,
    )
    # Labels share storage with input_ids; mask a copy to preserve model inputs.
    batch["label_ids"] = batch["label_ids"].masked_fill(~batch["response_mask"], -100)
    return batch

def split_by_prompt(dataset, fraction, seed, prompt_key="prompt"):
    if fraction == 0:
        return dataset, None
    prompts = sorted(set(dataset[prompt_key]))
    if len(prompts) < 2:
        raise ValueError("Automatic evaluation split needs two unique prompts; use --eval-fraction 0")
    random.Random(seed).shuffle(prompts)
    held_out = set(prompts[:max(1, int(len(prompts) * fraction))])
    train = dataset.select([i for i, prompt in enumerate(dataset[prompt_key]) if prompt not in held_out])
    evaluation = dataset.select([i for i, prompt in enumerate(dataset[prompt_key]) if prompt in held_out])
    return train, evaluation

def prepare_data(config, tokenizer: AutoTokenizer):
    from datasets import load_dataset

    prompt_key = getattr(config, "prompt_key", "") or PROMPT_KEY
    completion_key = getattr(config, "completion_key", "") or COMPLETION_KEY

    def load(path):
        kind = "parquet" if Path(path).suffix == ".parquet" else "json"
        dataset = load_dataset(kind, data_files=path, split="train", cache_dir=config.dataset_cache_dir or None)
        if not {prompt_key, completion_key}.issubset(dataset.column_names):
            raise ValueError("SFT data must have prompt and completion columns")
        return dataset

    training = load(config.train_data)
    if config.eval_data:
        evaluation = load(config.eval_data)
        if set(training[prompt_key]) & set(evaluation[prompt_key]):
            raise ValueError("Training and evaluation data contain overlapping prompts")
    else:
        training, evaluation = split_by_prompt(
            training, config.eval_fraction, config.seed, prompt_key=prompt_key
        )

    def encode_dataset(dataset, name):
        if dataset is None:
            return None
        
        def batch_encode(examples):
            prompt_token_ids = tokenizer(examples[prompt_key], add_special_tokens=True)["input_ids"]
            responses_token_ids = tokenizer(examples[completion_key], add_special_tokens=False)["input_ids"]

            kept_prompt_token_ids = []
            kept_responses_token_ids = []

            for i, (prompt, response) in enumerate(zip(prompt_token_ids, responses_token_ids, strict=True)):
                # 1. Drop empty completions
                if len(response) == 0:
                    logger.warning(f"Row {i} in {name} has empty completion and will be dropped")
                    continue

                # 2. Strip trailing EOS from prompt if the tokenizer added one
                if prompt and prompt[-1] == tokenizer.eos_token_id:
                    prompt = prompt[:-1]

                # 3. Append EOS to response first so length calculation is exact
                if response[-1] != tokenizer.eos_token_id:
                    response = response + [tokenizer.eos_token_id]

                # 4. Drop if total length exceeds max_length
                if len(prompt) + len(response) > config.max_length:
                    logger.warning(
                        f"Row {i} in {name} exceeds max_length {config.max_length} and will be dropped"
                    )
                    continue

                kept_prompt_token_ids.append(prompt)
                kept_responses_token_ids.append(response)

            # 5. Return the filtered lists, not the original ones
            return {
                "prompt_token_ids": kept_prompt_token_ids,
                "responses_token_ids": kept_responses_token_ids,
            }
        filtered_dataset = dataset.map(batch_encode, batched=True, remove_columns=dataset.column_names)

        
        # rows = []
        # for example in dataset:
        #     encoded = encode_example(example, tokenizer, config.max_length)
        #     if encoded is not None:
        #         rows.append(encoded)
        logger.info("%s: kept %d/%d rows within max_length and with supervised tokens", name, len(filtered_dataset), len(dataset))
        if len(filtered_dataset) == 0:
            raise ValueError(f"No usable {name} examples remain; check max_length and prompt/completion boundaries")
        return filtered_dataset

    return encode_dataset(training, "training"), encode_dataset(evaluation, "evaluation")

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
        and any(parameter.requires_grad for parameter in module.parameters())
    ]
    if not blocks:
        raise ValueError(
            f"Could not identify transformer blocks for {type(model).__name__}; "
            "check the model's _no_split_modules metadata"
        )

    # Step 2: Create 1D device mesh for data parallel sharding
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
            reshard_after_forward=True,   # Free memory after forward pass
            offload_policy=offload_policy, 
            mp_policy=mp_policy
        )

    # Step 6: Apply sharding to the root model
    # This wraps the entire model and enables top-level FSDP2 functionality
    fully_shard(
        model, 
        mesh=mesh, 
        reshard_after_forward=True,   # Free memory after forward pass
        offload_policy=offload_policy, 
        mp_policy=mp_policy
    )

def accumulation_windows(loader, steps):
    iterator = iter(loader)
    while window := list(itertools.islice(iterator, steps)):
        yield window

@ray.remote(num_gpus=1)
class SFT_FSDP_TrainerWorker:
    def __init__(self, 
                 config: dict, 
                 rank: int,
                 fsdp_world_size: int,
                 fsdp_master_addr: str,
                 fsdp_master_port: int,):
        config = Config(**config)
        self.config = config
        self.experiment_paths = ExperimentPaths.from_root(config.output_dir)
        self.world_rank = rank
        self.fsdp_world_size = fsdp_world_size
        self.logger = setup_logging(rank=self.world_rank, log_dir=self.experiment_paths.logs, level=logging.INFO)
        
        os.environ["MASTER_ADDR"] = fsdp_master_addr
        os.environ["MASTER_PORT"] = str(fsdp_master_port)

        # torch.accelerator.set_device_index(0)
        # self.device = ray.train.torch.get_device()
        # self.world_rank = ray.train.get_context().get_world_rank()
        # self.world_size = ray.train.get_context().get_world_size()
        # Additional initialization for FSDP and other components can be added here

        if config.precision == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise ValueError("This GPU does not support BF16; use --precision float32")

    def get_rank(self):
        return self.world_rank
    
    def get_gpu_ids(self):
        """Physical GPU id(s) Ray assigned to this worker (for server/train split)."""
        return ray.get_gpu_ids()
    
    def setup(self):
        # Keep model and FSDP dependencies out of Ray's actor serialization.
        import math
        import random
        from functools import partial
        from pathlib import Path
        import ray.train
        import torch
        import torch.distributed as dist
        from torch.utils.data import DataLoader, DistributedSampler
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from src.distributed_checkpointing import load_fsdp_checkpoint
        from src.trainers.sft_fsdp_trainer import prepare_data, collate_examples, shard_model

        config = self.config
        world_rank = self.world_rank
        world_size = self.fsdp_world_size

        torch.cuda.set_device(0)
        dist.init_process_group(backend="nccl", rank=world_rank, world_size=world_size)
        random.seed(config.seed)
        torch.manual_seed(config.seed)

        model_path = Path(config.model_name)
        model_name = str(model_path / "hf_model") if (model_path / "hf_model").is_dir() else config.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer or model_name, use_fast=True)
        if not self.tokenizer.is_fast or self.tokenizer.eos_token_id is None:
            raise ValueError("A fast tokenizer with EOS is required for prompt boundary masking")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        training, evaluation = prepare_data(config, self.tokenizer)
        collator = partial(collate_examples, tokenizer=self.tokenizer)
        sampler = DistributedSampler(training, num_replicas=world_size, rank=world_rank,
                                     seed=config.seed, shuffle=True)
        self.loader = DataLoader(training, batch_size=config.batch_size, sampler=sampler,
                                 collate_fn=collator)
        self.eval_loader = None
        if evaluation is not None:
            eval_sampler = DistributedSampler(evaluation, num_replicas=world_size,
                                              rank=world_rank, shuffle=False, drop_last=True)
            self.eval_loader = DataLoader(evaluation, batch_size=config.batch_size,
                                          sampler=eval_sampler, collate_fn=collator)

        init_dtype = torch.bfloat16 if config.precision == "bfloat16" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=init_dtype,
            attn_implementation=config.attn_implementation,
        )
        # This trainer supplies text only, so Gemma's vision branch is unused.
        backbone = getattr(model, "model", model)
        for name in ("vision_tower", "multi_modal_projector"):
            component = getattr(backbone, name, None)
            if component is not None:
                component.requires_grad_(False)
        if config.precision == "bfloat16":
            model = model.to(dtype=torch.float32)
        if config.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            model.config.use_cache = False
        shard_model(model, world_size=world_size, precision=config.precision,
                    cpu_offload=False)

        self.model = model
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
            foreach=False,
        )
        self.windows_per_epoch = math.ceil(len(self.loader) / config.gradient_accumulation_steps)
        self.total_steps = config.epochs * self.windows_per_epoch
        warmup = int(self.total_steps * config.warmup_ratio)

        def lr_scale(step):
            if warmup and step < warmup:
                return (step + 1) / warmup
            progress = (step - warmup) / max(1, self.total_steps - warmup)
            return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_scale)
        if config.last_checkpoint_dir:
            self.start_epoch, self.start_window, self.step = load_fsdp_checkpoint(
                model, self.optimizer,
                ray.train.Checkpoint.from_directory(config.last_checkpoint_dir),
                logger=self.logger, scheduler=self.scheduler,
            )
        else:
            self.start_epoch = self.start_window = self.step = 0
        if world_rank == 0:
            self.logger.info("Initialized %d training rows for %d updates on %d GPUs",
                             len(training), self.total_steps, world_size)

    def profile_training(self):
        """Profile real optimizer updates; all FSDP workers must call together."""
        import json
        import time
        from pathlib import Path
        import torch
        import torch.distributed as dist
        from torch.profiler import profile, ProfilerActivity, record_function

        config = self.config
        required = config.profile_warmup_steps + config.profile_steps
        if self.windows_per_epoch < required:
            raise ValueError(
                f"Profiling needs {required} accumulation windows, but this epoch has "
                f"{self.windows_per_epoch}; reduce profile steps or use more data"
            )
        output = Path(config.output_dir) / "profiles" / f"rank-{self.world_rank}"
        output.mkdir(parents=True, exist_ok=True)
        for _ in range(config.profile_warmup_steps):
            self.one_train_step()
        torch.cuda.synchronize()
        dist.barrier()

        measurements = []
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as profiler:
            for _ in range(config.profile_steps):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                with record_function("optimizer_update"):
                    metrics = self.one_train_step()
                torch.cuda.synchronize()
                metrics.update({
                    "profile/step_seconds": time.perf_counter() - started,
                    "profile/peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "profile/peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                    "profile/allocated_gib": torch.cuda.memory_allocated() / 2**30,
                    "profile/reserved_gib": torch.cuda.memory_reserved() / 2**30,
                })
                measurements.append(metrics)
                profiler.step()

        profiler.export_chrome_trace(str(output / "trace.json"))
        averages = profiler.key_averages()
        (output / "operators.txt").write_text(
            averages.table(sort_by="self_cuda_time_total", row_limit=50), encoding="utf-8"
        )
        (output / "memory.txt").write_text(torch.cuda.memory_summary(), encoding="utf-8")
        report = {
            "rank": self.world_rank,
            "gpu_ids": self.get_gpu_ids(),
            "warmup_steps": config.profile_warmup_steps,
            "profile_steps": config.profile_steps,
            "mean_step_seconds": sum(m["profile/step_seconds"] for m in measurements) / len(measurements),
            "measurements": measurements,
            "notes": "Timings include profiler overhead and data loading; CUDA memory excludes allocations outside PyTorch.",
        }
        (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return {"rank": self.world_rank, "profile_dir": str(output),
                "mean_step_seconds": report["mean_step_seconds"]}

    def one_train_step(self, window=None):
        import time
        import torch
        import torch.distributed as dist
        import torch.nn.functional as F
        from torch.profiler import record_function
        from src.trainers.sft_fsdp_trainer import autocast, accumulation_windows

        with record_function("data_loading"):
            if window is None:
                if not hasattr(self, "_windows"):
                    self._windows = accumulation_windows(self.loader, self.config.gradient_accumulation_steps)
                window = next(self._windows)
        model = self.model
        model.train()
        optimizer = self.optimizer
        scheduler = self.scheduler
        config = self.config
        device = torch.device("cuda", 0)
        world_size = self.fsdp_world_size

        
        train_step_start = time.time()
        optimizer.zero_grad(set_to_none=True)
        token_count = torch.tensor(sum((batch["label_ids"] != -100).sum().item() for batch in window),
                                    device=device, dtype=torch.float64)
        dist.all_reduce(token_count)
        if token_count.item() == 0:
            raise ValueError("Accumulation window has no supervised causal tokens")
        loss_total = torch.zeros((), device=device, dtype=torch.float64)
        # Reduce each microbatch so accumulated gradients stay sharded. Disabling
        # synchronization retains full gradients and exhausts two 48 GB GPUs.
        model.set_requires_gradient_sync(True)
        for batch_idx_in_window, batch in enumerate(window):
            with record_function("host_to_device"):
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            window[batch_idx_in_window] = None  # Clear reference to free memory early

            with autocast(config.precision):
                with record_function("forward"):
                    logits = model(input_ids=batch["input_ids"], 
                                    attention_mask=batch["attention_mask"], 
                                    use_cache=False).logits
                with record_function("loss"):
                    labels = batch["label_ids"]
                    loss_sum = F.cross_entropy(logits.contiguous().float().view(-1, logits.size(-1)),
                                                labels.contiguous().view(-1), 
                                                ignore_index=-100, reduction="sum")

                    # FSDP averages gradients across ranks. Account for variable
                    # response lengths and a partial final accumulation window.
                    loss = loss_sum * (world_size / token_count.item())
            
            with record_function("backward"):
                loss.backward()
            loss_total += loss_sum.detach()
            del logits, loss, loss_sum, batch

        # PyTorch’s FSDP2-compatible utility
        with record_function("gradient_clipping"):
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=config.max_grad_norm,
            )
            if hasattr(grad_norm, "full_tensor"):
                grad_norm = grad_norm.to(device).full_tensor().item()
            else:
                grad_norm = grad_norm.item()
        learning_rate = optimizer.param_groups[0]["lr"]
        with record_function("optimizer"):
            optimizer.step()
            scheduler.step()
        self.step += 1
        dist.all_reduce(loss_total)
        train_step_end = time.time()
        metrics = {"train/loss": (loss_total / token_count).item(), 
                    "train/learning_rate": learning_rate,
                    "train/grad_norm": float(grad_norm), 
                    "epoch": self.start_epoch + 1, 
                    "step": self.step,
                    "train/tokens_per_second": token_count.item() / (train_step_end - train_step_start)}
        return metrics

    # def train(self):
    #     config = self.config
    #     device = ray.train.torch.get_device()
    #     torch.cuda.set_device(self.device)
    #     world_rank = ray.train.get_context().get_world_rank()
    #     world_size = ray.train.get_context().get_world_size()

    #     # Implement the training loop here
    #     wandb_run = None
    #     try:
    #         self.model.train()
    #         for epoch in range(self.start_epoch, config.epochs):
    #             if hasattr(self.loader.sampler, "set_epoch"):
    #                 self.loader.sampler.set_epoch(epoch)
    #             else:
    #                 self.loader.generator.manual_seed(config.seed + epoch)
                
    #             for window_index, window in enumerate(
    #                 accumulation_windows(self.loader, config.gradient_accumulation_steps)
    #             ):
    #                 if epoch == self.start_epoch and window_index < self.start_window:
    #                     continue
    #                 metrics = self.one_train_step(window)

    #                 if self.eval_loader is not None and (self.step % config.eval_steps == 0 or self.step == self.self.total_steps):
    #                     metrics["eval/loss"] = self.evaluate()
            
    #                 if world_rank == 0:
    #                     # app(metrics, experiment_paths.metrics / f"step-{step:06d}.json")
    #                     append_jsonl(self.experiment_paths.metrics / "metrics.jsonl", metrics)
    #                     # Gather weights on Rank 0 and push to vLLM
                    
    #                 if world_rank == 0 and (self.step % config.logging_steps == 0 or self.step == 1 or "eval/loss" in metrics or self.step == self.total_steps):
    #                     logger.info("step=%d/%d %s", self.step, self.total_steps, metrics)
    #                     if wandb_run:
    #                         wandb_run.log(metrics, step=self.step)
    #                 if self.step % config.save_steps == 0 or self.step == self.total_steps:
    #                     self.save_checkpoint(epoch, window_index, self.windows_per_epoch, metrics)
    #     finally:
    #         if wandb_run:
    #             wandb_run.finish()

    # def evaluate(self):
    #     # Implement the evaluation logic here
    #     device = ray.train.torch.get_device()
    #     self.model.eval()
    #     totals = torch.zeros(2, device=device, dtype=torch.float64)
    #     with torch.no_grad():
    #         for batch in self.eval_loader:
    #             batch = {key: value.to(device) for key, value in batch.items()}
    #             with autocast(self.config.precision):
    #                 logits = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits
    #                 loss = F.cross_entropy(logits.contiguous().float().view(-1, logits.size(-1)),
    #                                         batch["label_ids"].contiguous().view(-1), 
    #                                         ignore_index=-100, reduction="sum")
    #             totals[0] += loss.detach()
    #             totals[1] += (batch["label_ids"] != -100).sum()
    #             del logits, loss, batch
    #     dist.all_reduce(totals)
    #     return (totals[0] / totals[1]).item()

    # def save_checkpoint(self, epoch, window_index, windows_per_epoch, metrics):

    #     logger.info("Saving checkpoint and reporting metrics...")
    #     next_epoch = epoch
    #     next_window = window_index + 1
    #     if next_window == windows_per_epoch:
    #         next_epoch += 1
    #         next_window = 0
    #     checkpoint_dir = report_metrics_and_save_fsdp_checkpoint(
    #         self.model, 
    #         self.optimizer, 
    #         self.experiment_paths.get_checkpoint_path(self.step),
    #         metrics, 
    #         scheduler=self.scheduler,
    #         step=self.step,
    #         epoch=next_epoch,
    #         next_window=next_window,
    #     )
    #     if self.world_rank == 0:
    #         update_manifest(
    #             self.experiment_paths.manifest,
    #             "running",
    #             last_checkpoint_dir=str(checkpoint_dir),
    #         )
    #     dist.barrier()
    #     logger.info(f"Checkpoint saved successfully. Metrics: {metrics}")

    # # ---- weight-transfer setup (rank 0 only) ----

    # # def setup_transfer_endpoint(self):
    # #     """Create the NCCL rendezvous endpoint for weight transfer."""
    # #     assert self.rank == 0
    # #     self.transfer_port = get_open_port()
    # #     self.transfer_master_address = get_ip()
    # #     return self.transfer_master_address, self.transfer_port

    # # def setup_engine(
    # #     self,
    # #     base_url: str,
    # #     transfer_master_address: str,
    # #     transfer_port: int,
    # #     transfer_world_size: int,
    # # ):
    # #     """Build the trainer engine on every FSDP rank.

    # #     Called on all ranks with the shared rendezvous endpoint. Rank 0 is the
    # #     sender: `trainer_init` opens its rank-0 NCCL endpoint and, on a worker
    # #     thread, calls the server's `init_weight_transfer_engine` over HTTP so
    # #     both ends rendezvous together. The other ranks skip the rendezvous and
    # #     only join the FSDP all-gather during send_weights.
    # #     """
    # #     self.engine = WeightTransferTrainerFactory.trainer_init(
    # #         init_info=NCCLTrainerInitInfo(
    # #             master_address=transfer_master_address,
    # #             master_port=transfer_port,
    # #             world_size=transfer_world_size,
    # #             rank=self.rank,  # FSDP rank; sender is rank 0
    # #             packed=True,
    # #         ),
    # #         client=HTTPVLLMWeightSyncClient(base_url),
    # #         # Yields sharded DTensors; the engine reads global shape/dtype for
    # #         # metadata (no gather) and calls full_tensor() at broadcast time.
    # #         source=ModuleSource(self.model),
    # #     )

    # # # ---- collective ops (ALL FSDP ranks must call concurrently) ----

    # # def gather_and_broadcast_weights(self):
    # #     """All-gather full parameters and broadcast them to the vLLM server.

    # #     Called on all FSDP ranks. `send_weights` gathers each param via
    # #     `full_tensor()` (a collective every rank must enter in the same order);
    # #     only rank 0 (the sender) drives the server-side update_weights
    # #     concurrently with the NCCL broadcast — the other ranks only gather.
    # #     """
    # #     self.engine.send_weights()
