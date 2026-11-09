#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
PYTHON="$VENV_DIR/bin/python"
VLLM_WHEEL="https://github.com/vllm-project/vllm/releases/download/v0.32.0/vllm-0.32.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required. Install uv, then rerun this script." >&2
    exit 1
fi

uv venv --python 3.12 "$VENV_DIR"

# Use vLLM's CUDA 12.9 wheel and its matching PyTorch/CUDA dependencies.
uv pip install --python "$PYTHON" \
    "$VLLM_WHEEL" \
    --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu129

# vLLM 0.32's unpinned TorchCodec dependency can resolve to a CUDA 13 wheel,
# which fails to load in the CUDA 12.9 environment. Use the CPU TorchCodec wheel.
uv pip uninstall --python "$PYTHON" torchcodec >/dev/null 2>&1 || true
uv pip install --python "$PYTHON" \
    --index-url https://download.pytorch.org/whl/cpu \
    'torchcodec==0.17.0'

# Ray is used by Ray-hosted vLLM weight-sync clients.
uv pip install --python "$PYTHON" 'ray==2.52.0'

# Make the user-installed CUDA 12.9 compiler visible to FlashInfer JIT builds
# and flash-attn.
source "$SCRIPT_DIR/use_cuda129.sh"

# Additional data-loading dependency used by scripts/train_sft_fsdp_ray.py.
uv pip install --python "$PYTHON" 'datasets'

# Build flash-attn for the local CUDA toolkit. The Conda toolkit stores headers
# under targets/x86_64-linux/include; exposing them lets the C++ extension find
# cuda_runtime.h. Override FLASH_ATTN_CUDA_ARCHS and MAX_JOBS for other systems.
CPATH="$CUDA_HOME/targets/x86_64-linux/include${CPATH:+:$CPATH}" \
MAX_JOBS="${MAX_JOBS:-8}" \
FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-80}" \
    uv pip install --python "$PYTHON" 'flash-attn' --no-build-isolation

echo "Installed vLLM: $("$PYTHON" -c 'import vllm; print(vllm.__version__)')"
echo "PyTorch CUDA: $("$PYTHON" -c 'import torch; print(torch.version.cuda)')"
echo "CUDA compiler: $("$CUDACXX" --version | tail -n 1)"
echo "Environment ready at $VENV_DIR"
