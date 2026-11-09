"""Full-shard SFT for prompt/completion data."""

from dataclasses import asdict, replace
import json
import logging
import sys
import os
os.environ["RAY_TRAIN_V2_ENABLED"] = "1"
from pathlib import Path
import ray
import torch
# FSDP2 sharding imports 

from src.configs import Config
from src.logging_utils import setup_logging

from src.trainers.sft_fsdp_trainer import SFT_FSDP_TrainerWorker
from src.experiment_utils import ExperimentPaths, update_manifest
from vllm.utils.network_utils import get_ip, get_open_port

logger = logging.getLogger(__name__)

# def start_vllm_server(server_gpus: str) -> subprocess.Popen:
#     """Spawn a `vllm serve` HTTP server (EP+DP) on `server_gpus` and wait for it."""
#     serve_args = [
#         "vllm",
#         "serve",
#         MODEL_NAME,
#         "--served-model-name",
#         SERVED_MODEL_NAME,
#         "--tensor-parallel-size",
#         str(INFERENCE_TP_SIZE),
#         "--data-parallel-size",
#         str(INFERENCE_DP_SIZE),
#         "--enable-expert-parallel",
#         "--enforce-eager",
#         "--load-format",
#         "dummy",
#         "--gpu-memory-utilization",
#         "0.7",
#         "--port",
#         str(SERVER_PORT),
#         "--weight-transfer-config",
#         json.dumps({"backend": "nccl"}),
#     ]
#     env = os.environ.copy()
#     env["CUDA_VISIBLE_DEVICES"] = server_gpus
#     env["VLLM_SERVER_DEV_MODE"] = "1"  # exposes the weight-transfer endpoints
#     env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
#     print(f"[server] Launching: {' '.join(serve_args)} (GPUs {server_gpus})")
#     proc = subprocess.Popen(
#         serve_args,
#         env=env,
#         stdout=sys.stdout,
#         stderr=sys.stderr,
#         start_new_session=True,
#     )

#     # Wait for the server to come up (model load can take a while).
#     deadline = time.monotonic() + 1800
#     while True:
#         if proc.poll() is not None:
#             raise RuntimeError("vLLM server exited before becoming ready.")
#         try:
#             if requests.get(f"{BASE_URL}/health", timeout=5).status_code == 200:
#                 break
#         except requests.RequestException:
#             pass
#         if time.monotonic() > deadline:
#             raise RuntimeError("vLLM server failed to start in time.")
#         time.sleep(2)
#     print("[server] Ready.")
#     return proc

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
    logger = setup_logging(rank=0, log_dir=experiment_paths.logs)
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
    # FSDP rendezvous address (single-node).
    fsdp_master_addr = get_ip()
    fsdp_master_port = get_open_port()

    # Launch the FSDP training workers first so Ray reserves their GPUs, then
    # place the inference server on the GPUs Ray did NOT use. This keeps the two
    # on disjoint physical GPUs whether ray.init() started a fresh cluster or
    # connected to an existing one.
    fsdp_workers = [
        SFT_FSDP_TrainerWorker.remote(
            config.to_dict(),
            rank,
            config.num_fsdp_workers,
            fsdp_master_addr,
            fsdp_master_port,
        )
        for rank in range(config.num_fsdp_workers)
    ]
    ray.get([w.get_rank.remote() for w in fsdp_workers])
    logger.info(f"[init] {config.num_fsdp_workers} FSDP training workers ready.")
    logger.info("[init] Assigned GPU IDs: %s", ray.get([w.get_gpu_ids.remote() for w in fsdp_workers]))

    try:
        logger.info("[setup] Initializing data, models, optimizer, and FSDP shard...")
        ray.get([w.setup.remote() for w in fsdp_workers])
        logger.info("[setup] Completed.")

        logger.info("[training] Start training")
        if config.profile:
            metrics = ray.get([w.profile_training.remote() for w in fsdp_workers])
        else:
            metrics = ray.get([w.one_train_step.remote() for w in fsdp_workers])
        logger.info(metrics)
        logger.info("[training] Completed.")
    finally:
        for worker in fsdp_workers:
            ray.kill(worker)
        ray.shutdown()

    # logger.info("[train] Initializing data, models, optimizer, and FSDP shard...")
    # ray.get([w.train.remote() for w in fsdp_workers])
    # logger.info("[train] Completed.")

    # # Discover the physical GPUs Ray assigned to training; run the server on the
    # # complementary GPUs.
    # training_gpus = {
    #     int(g)
    #     for ids in ray.get([w.get_gpu_ids.remote() for w in fsdp_workers])
    #     for g in ids
    # }
    # num_gpus = int(ray.cluster_resources().get("GPU", 0))
    # num_server_gpus = config.num_vllm_workers
    # server_gpu_ids = [g for g in range(num_gpus) if g not in training_gpus][
    #     :num_server_gpus
    # ]
    # if len(server_gpu_ids) < num_server_gpus:
    #     raise RuntimeError(
    #         f"Need {num_server_gpus} free GPUs for the inference server but only "
    #         f"found {server_gpu_ids} (training uses {sorted(training_gpus)} of "
    #         f"{num_gpus} cluster GPUs)."
    #     )
    # server_gpus = ",".join(str(g) for g in server_gpu_ids)
    # print(f"[init] Training GPUs {sorted(training_gpus)}; server GPUs [{server_gpus}].")

    # scaling_config = ray.train.ScalingConfig(
    #     num_workers=config.num_workers,      # Number of distributed workers
    #     use_gpu=True        # Enable GPU training
    # )

    # # Ray Train writes to <storage_path>/<name>.
    # # Setting storage_path=destination.parent and name=destination.name
    # # makes Ray Train use `destination` directly as the experiment folder.
    # storage_path = str(destination.parent)
    # experiment_name = config.run_name or destination.name
    # # experiment_name=f"fsdp_ray_{uuid.uuid4().hex[:8]}"

    # # Configure run settings and storage
    # run_config = ray.train.RunConfig(
    #     # Persistent storage path accessible across all worker nodes
    #     storage_path=str(destination / "ray_runs"),
    #     # Unique experiment name (use consistent name to resume from checkpoints)
    #     name=experiment_name,
    #     # Fault tolerance configuration
    #     failure_config=ray.train.FailureConfig(max_failures=0),
    # )

    # A. Instantiate vLLM actor on GPU 4
    # vllm_rollout = VLLMRolloutWorker.remote(config.model_name)

    # train_loop_config=asdict(config)
    # train_loop_config["vllm_rollout_actor"] = vllm_rollout

    # Initialize and launch the distributed training job
    # trainer = ray.train.torch.TorchTrainer(
    #     train_loop_per_worker=train,
    #     scaling_config=scaling_config,
    #     train_loop_config=train_loop_config,
    #     run_config=run_config,
    # )

    # print(f"Starting FSDP2 training job in {Path(storage_path) / experiment_name}...")
    # result = trainer.fit()
    # print("Training completed successfully!")

