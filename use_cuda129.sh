# Source this file to use the CUDA 12.9 toolkit in the current shell.

_cuda129_prefix=/home/grads/hnn5071/.conda/envs/cuda129-toolkit

if [[ ! -x "$_cuda129_prefix/bin/nvcc" ]]; then
    echo "CUDA 12.9 nvcc not found at $_cuda129_prefix/bin/nvcc" >&2
    unset _cuda129_prefix
    return 1 2>/dev/null || exit 1
fi

export CUDA_HOME="$_cuda129_prefix"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"

unset _cuda129_prefix
source .venv/bin/activate