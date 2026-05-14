# Build Environment Summary

## Hardware
- GPU: NVIDIA GeForce RTX 5080 (16GB VRAM)
- Platform: WSL2 on Windows (Ubuntu 24.04.4 LTS)
- CUDA Driver: 13.2 (NVIDIA-SMI 595.54, Driver 595.79)

## Environment Setup
- Conda env name: `vllm`
- Python: 3.12.13
- PyTorch: 2.11.0+cu130
- CUDA Toolkit: 13.2 (installed at `/usr/local/cuda-13.2`)
- GCC: 13.3.0
- CMake: 4.3.0
- Build tools installed via: `conda install -c conda-forge cmake ninja gcc_linux-64 gxx_linux-64 ccache uv -y`

## Activation
```bash
conda activate vllm
export CUDA_HOME=/usr/local/cuda-13.2
export PATH=$CUDA_HOME/bin:$PATH
```

## vLLM Build Info
- Version: 0.20.2rc1.dev306+g3b1ef03be.d20260514.cu132
- Install type: Editable (`-e .`), full C++/CUDA source build
- Build command used:
  ```bash
  MAX_JOBS=4 NVCC_THREADS=1 CCACHE_NOHASHDIR=true uv pip install --no-build-isolation -e . --torch-backend=auto
  ```

## Incremental Rebuild (after editing csrc/ files)
```bash
MAX_JOBS=4 NVCC_THREADS=1 CCACHE_NOHASHDIR=true uv pip install --no-build-isolation -e . --torch-backend=auto
```
Python-only changes need no recompilation (editable install picks them up immediately).

## WSL Memory Note
If builds OOM, reduce MAX_JOBS to 1-2 or increase WSL memory in `C:\Users\<user>\.wslconfig`:
```ini
[wsl2]
memory=24GB
```
