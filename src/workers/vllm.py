import os
import ray
import torch
import torch.nn as nn
from vllm import LLM, SamplingParams
from vllm.config import WeightTransferConfig

# ---------------------------------------------------------
# 1. Define the vLLM Rollout Worker Actor (Runs on GPU 4)
# ---------------------------------------------------------
@ray.remote(num_gpus=1)
class VLLMRolloutWorker:
    def __init__(self, base_model: str):
        print("Initializing vLLM Rollout Worker...")
        # Set before creating the LLM; setting it before importing vllm is safest.
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        self.llm = LLM(
            model=base_model,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.7,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            weight_transfer_config=WeightTransferConfig(backend="nccl")
        )

    def update_weights(self, state_dict):
        def load_weights(model):
            def remap():
                for name, tensor in state_dict.items():
                    if name.startswith(("model.", "lm_head.")):
                        name = "language_model." + name
                    yield name, tensor

            with torch.no_grad():
                return model.load_weights(remap())

        self.llm.apply_model(load_weights)
        self.llm.reset_prefix_cache()

    def generate(self, prompts: list[str]):
        sampling_params = SamplingParams(max_tokens=64, temperature=0.7)
        outputs = self.llm.generate(prompts, sampling_params)
        return [o.outputs[0].text for o in outputs]
