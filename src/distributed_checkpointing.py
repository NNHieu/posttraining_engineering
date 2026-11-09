from pathlib import Path
from typing import Optional

import torch
import ray.train
from torch.distributed.fsdp import (
    FSDPModule,
)
import torch.distributed as dist
# PyTorch Distributed Checkpoint (DCP) Core import
import torch.distributed.checkpoint as dcp
# PyTorch Distributed Checkpoint (DCP) imports
from torch.distributed.checkpoint.state_dict import (
    get_state_dict,
    set_state_dict,
    get_model_state_dict,
    StateDictOptions
)
from torch.distributed.checkpoint.stateful import Stateful
from logging import Logger

class AppState(Stateful):
    """This is a useful wrapper for checkpointing the Application State. Because this object is compliant
    with the Stateful protocol, PyTorch DCP automatically calls state_dict/load_state_dict as needed in the
    dcp.save/load APIs.

    Note: This wrapper is used to handle calling distributed state dict methods on the model
    and optimizer.
    """

    def __init__(self, model, optimizer=None, scheduler=None, epoch=0, next_window=0, step=0):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epoch = epoch
        self.next_window = next_window
        self.step = step
        self.world_size = dist.get_world_size()

    def state_dict(self):
        # this line automatically manages FSDP2 FQN's (Fully Qualified Name), as well as sets the default state dict type to FSDP.SHARDED_STATE_DICT
        model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizer)
        return {
            "model": model_state_dict,
            "optim": optimizer_state_dict,
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "epoch": self.epoch,
            "next_window": self.next_window,
            "step": self.step,
            "world_size": self.world_size,
        }

    def load_state_dict(self, state_dict):
        # sets our state dicts on the model and optimizer, now that loading is complete
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"],
        )
        saved_world_size = state_dict.get("world_size")
        if saved_world_size is not None and saved_world_size != dist.get_world_size():
            raise ValueError(
                f"Checkpoint world size {saved_world_size} does not match current world size "
                f"{dist.get_world_size()}"
            )
        self.epoch = state_dict.get("epoch", 0)
        # Older checkpoints stored a one-based completed epoch, which is the next
        # epoch index when no within-epoch cursor is present.
        self.next_window = state_dict.get("next_window", 0)
        self.step = state_dict.get("step", 0)
        if self.scheduler is not None and state_dict.get("scheduler") is not None:
            self.scheduler.load_state_dict(state_dict["scheduler"])


class OptimizerPartialLoadPlanner(dcp.DefaultLoadPlanner):
    def create_local_plan(self):
        if self.metadata is None:
            raise RuntimeError("DCP load planner was not initialized with checkpoint metadata")

        expected_keys = set(self.state_dict)
        saved_keys = set(self.metadata.state_dict_metadata)
        missing_keys = expected_keys - saved_keys
        allowed_missing = {
            "app.scheduler",
            "app.next_window",
            "app.world_size",
        }
        invalid_missing = {
            key for key in missing_keys
            if not key.startswith("app.optim.state.") and key not in allowed_missing
        }
        if invalid_missing:
            raise RuntimeError(
                "Checkpoint is missing non-optimizer state: "
                + ", ".join(sorted(invalid_missing)[:10])
            )

        return super().create_local_plan()


def _rng_state_path(checkpoint_dir: Path, rank: int) -> Path:
    return checkpoint_dir / f"rng-rank-{rank:05d}.pt"


def _save_rng_state(checkpoint_dir: Path, rank: int) -> None:
    path = _rng_state_path(checkpoint_dir, rank)
    temporary_path = path.with_suffix(".tmp")
    torch.save(
        {
            "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(),
        },
        temporary_path,
    )
    temporary_path.replace(path)


def _load_rng_state(checkpoint_dir: Path, rank: int) -> bool:
    path = _rng_state_path(checkpoint_dir, rank)
    if not path.exists():
        return False
    state = torch.load(path, map_location="cpu", weights_only=True)
    torch.set_rng_state(state["cpu"])
    torch.cuda.set_rng_state(state["cuda"])
    return True

def load_fsdp_checkpoint(model: FSDPModule, 
                         optimizer: torch.optim.Optimizer, 
                         ckpt: ray.train.Checkpoint,
                         logger: Optional[Logger] = None,
                         scheduler=None
                         ) -> tuple[int, int, int]:
    """Load an FSDP checkpoint into the model and optimizer.
    
    This function handles distributed checkpoint loading with automatic resharding
    support. It can restore checkpoints even when the number of workers differs
    from the original training run.
    
    Args:
        model: The FSDP-wrapped model to load state into
        optimizer: The optimizer to load state into
        ckpt: Ray Train checkpoint containing the saved state

    Returns:
        A tuple of next epoch index, next accumulation-window index, and completed step.
    """
    if logger: logger.info("Loading distributed checkpoint for resuming training...")
    
    try:
        with ckpt.as_directory() as checkpoint_dir:
            # Create state wrapper for DCP loading
            app_state = AppState(model, optimizer, scheduler)
            state_dict = {"app": app_state}
            
            # Load the distributed checkpoint
            dcp.load(
                state_dict=state_dict,
                checkpoint_id=checkpoint_dir,
                planner=OptimizerPartialLoadPlanner(allow_partial_load=True),
            )

            rank = ray.train.get_context().get_world_rank()
            if not _load_rng_state(Path(checkpoint_dir), rank):
                if logger:
                    logger.warning("Checkpoint has no per-rank RNG state; continuing with current RNG state")
            
        if logger:
            logger.info(
                "Successfully loaded distributed checkpoint at epoch %d, window %d, step %d",
                app_state.epoch,
                app_state.next_window,
                app_state.step,
            )
        return app_state.epoch, app_state.next_window, app_state.step
    except Exception as e:
        if logger: logger.error(f"Failed to load checkpoint: {e}")
        raise RuntimeError(f"Checkpoint loading failed: {e}") from e

def report_metrics_and_save_fsdp_checkpoint(
    model: FSDPModule, optimizer: torch.optim.Optimizer,
    checkpoint_dir: Path,
    metrics: dict, 
    scheduler=None,
    step: Optional[int] = None,
    epoch: int = 0,
    next_window: int = 0,
) -> Path:
    """Report training metrics and save an FSDP checkpoint.
    
    This function performs two critical operations:
    1. Saves the current model and optimizer state using distributed checkpointing
    2. Reports metrics to Ray Train for tracking
    
    Args:
        model: The FSDP-wrapped model to checkpoint
        optimizer: The optimizer to checkpoint
        metrics: Dictionary of metrics to report (e.g., loss, accuracy)
        epoch: The current epoch to be saved
    """
    checkpoint_dir = Path(checkpoint_dir)
    rank = ray.train.get_context().get_world_rank()

    if rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    
    # Perform a distributed checkpoint with DCP
    dcp.save(state_dict={"app": AppState(model, optimizer, scheduler, epoch, next_window, step)},
             checkpoint_id=checkpoint_dir)
    dist.barrier()

    _save_rng_state(checkpoint_dir, rank)
    dist.barrier()

    # Report each checkpoint shard from all workers
    # This saves the checkpoint to shared cluster storage for persistence
    # checkpoint = ray.train.Checkpoint.from_directory(checkpoint_dir) if rank == 0 else None
    ray.train.report(metrics,)
    return checkpoint_dir