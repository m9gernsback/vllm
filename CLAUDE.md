# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@AGENTS.md

## Architecture Overview

vLLM is a high-throughput LLM inference and serving engine. The codebase has two engine generations: the legacy engine (`vllm/engine/`) and the active **V1 engine** (`vllm/v1/`). All new development targets V1.

### Request Lifecycle (V1)

1. **Entrypoints** (`vllm/entrypoints/`) — User-facing APIs:
   - `llm.py`: Offline batch inference (`LLM` class)
   - `openai/api_server.py`: OpenAI-compatible HTTP server (production)
   - `cli/`: `vllm serve`, `vllm bench`, etc.

2. **AsyncLLM / LLMEngine** (`vllm/v1/engine/async_llm.py`, `llm_engine.py`) — Front-end engine that preprocesses inputs, manages output streams, and communicates with the core engine process via ZMQ.

3. **EngineCore** (`vllm/v1/engine/core.py`) — Runs in a separate process. Owns the scheduler and orchestrates workers. Communicates via ZMQ with msgspec serialization.

4. **Scheduler** (`vllm/v1/core/sched/scheduler.py`) — Decides which requests to run each iteration. Manages KV cache allocation via `KVCacheManager` (`vllm/v1/core/kv_cache_manager.py`) and `BlockPool` (`vllm/v1/core/block_pool.py`).

5. **Executor** (`vllm/v1/executor/`) — Launches and manages workers. Variants: `uniproc_executor.py` (single GPU), `multiproc_executor.py` (tensor parallelism), `ray_executor.py` (multi-node).

6. **Worker** (`vllm/v1/worker/gpu_worker.py`) — One per GPU. Manages the GPU model runner and KV cache memory.

7. **GPUModelRunner** (`vllm/v1/worker/gpu_model_runner.py`) — Prepares input tensors, runs the model forward pass, handles CUDA graphs, and manages attention metadata.

### Key Subsystems

- **Config** (`vllm/config/`) — All configuration dataclasses (`VllmConfig` is the top-level container). Passed through the entire stack.
- **Model implementations** (`vllm/model_executor/models/`) — 200+ model architectures. Each file implements one model family using layers from `vllm/model_executor/layers/`.
- **Layers** (`vllm/model_executor/layers/`) — Reusable building blocks: `linear.py` (parallelized linear), `attention/` (attention backends), `fused_moe/` (mixture-of-experts), `rotary_embedding/`, `layernorm.py`, `quantization/`.
- **Attention backends** (`vllm/v1/attention/`) — FlashAttention, FlashInfer, Triton, etc. Selected at runtime based on hardware/config.
- **Kernels** (`vllm/kernels/`, `csrc/`) — Custom CUDA/Triton kernels. Python wrappers in `vllm/kernels/`, C++/CUDA sources in `csrc/`.
- **Distributed** (`vllm/distributed/`) — Tensor/pipeline/expert parallelism, KV cache transfer (disaggregated serving), collective communication.
- **Compilation** (`vllm/compilation/`) — `torch.compile` integration with piecewise CUDA graph capture.
- **Speculative decoding** (`vllm/v1/spec_decode/`) — Draft model, EAGLE, n-gram, suffix decoding proposers.
- **Multimodal** (`vllm/multimodal/`) — Image/audio/video input processing and encoder management.
- **Platforms** (`vllm/platforms/`) — Hardware abstraction (CUDA, ROCm, TPU, CPU, XPU).
- **LoRA** (`vllm/lora/`) — Multi-LoRA adapter serving.

### Important Patterns

- `VllmConfig` is the single source of truth for all configuration; it flows everywhere.
- Models are registered via `@MULTIMODAL_MODEL_REGISTRY` or discovered through HuggingFace architecture names in `vllm/model_executor/models/__init__.py`.
- The V1 engine uses multiprocessing with ZMQ for engine-core communication and shared-memory/IPC for tensor transfer.
- Environment variables are centralized in `vllm/envs.py`.
