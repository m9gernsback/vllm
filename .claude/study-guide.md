# vLLM Study Guide

A structured study plan for learning vLLM from the ground up. Designed for someone with Python proficiency, ML/Transformer conceptual knowledge, and GPU graphics pipeline experience (but not CUDA compute or LLM inference background).

---

## Phase 0: Foundational Concepts (Before Reading Code)

Before diving into vLLM, you need to understand the LLM inference problem it solves.

### Key Concepts to Learn

| Concept | What It Is | GPU Graphics Analogy |
|---------|-----------|---------------------|
| **Tokenization** | Converting text → integer IDs (and back). Each model has a vocabulary. | Like vertex indexing in a mesh |
| **Autoregressive Generation** | LLM generates one token at a time; each token depends on all previous tokens | Like sequential frame rendering where each frame depends on the last |
| **KV Cache** | Stores computed Key/Value attention vectors so they aren't recomputed each step | Like a texture cache — avoid redundant computation |
| **PagedAttention** | vLLM's core innovation: manages KV cache in fixed-size blocks (like OS virtual memory) | **Exactly** like GPU virtual memory paging / sparse texture binding |
| **Continuous Batching** | Dynamically adding/removing requests from a batch as they finish (vs. static batching) | Like dynamic draw call batching in a scene graph |
| **Prefill vs Decode** | Prefill: process all input tokens at once (compute-bound). Decode: generate one token at a time (memory-bound) | Prefill ≈ initial scene rasterization. Decode ≈ incremental updates |
| **CUDA Graphs** | Record a sequence of GPU operations and replay them without CPU overhead | Like recording command buffers in Vulkan/DX12 |

### Required Reading (External)

1. **vLLM Blog Post** — https://blog.vllm.ai/2023/06/20/vllm.html
   - Best first introduction. Explains PagedAttention visually.

2. **vLLM Paper** (SOSP 2023) — https://arxiv.org/abs/2309.06180
   - Read Sections 1-4 (skip the evaluation). Focus on the PagedAttention mechanism.

3. **The Illustrated Transformer** — https://jalammar.github.io/illustrated-transformer/
   - If you need a refresher on attention/KV in transformers.

---

## Phase 1: User-Level Understanding (Week 1)

**Goal**: Understand what vLLM does from a user's perspective.

### Reading Order

| # | File | Lines | What You Learn |
|---|------|-------|---------------|
| 1 | `README.md` | 111 | Features, capabilities, supported models |
| 2 | `docs/design/arch_overview.md` | ~314 | **ESSENTIAL** — Full system architecture with diagrams |
| 3 | `examples/basic/offline_inference/basic.py` | 35 | Minimal working example (LLM class usage) |
| 4 | `vllm/entrypoints/llm.py` | ~600 | The user-facing `LLM` class — read the docstrings and `generate()` method |
| 5 | `vllm/sampling_params.py` | — | All generation parameters (temperature, top_p, etc.) |

### Hands-On Exercise

```bash
conda activate vllm
export CUDA_HOME=/usr/local/cuda-13.2 && export PATH=$CUDA_HOME/bin:$PATH
python examples/basic/offline_inference/basic.py
```

Then modify the example: change the model, adjust temperature, add more prompts. Observe how batching works.

---

## Phase 2: Core Engine Architecture (Week 2)

**Goal**: Understand the request lifecycle — how a prompt becomes generated text.

### The Data Flow

```
User prompt (string)
  → LLM.generate()                          [vllm/entrypoints/llm.py]
  → LLMEngine                               [vllm/v1/engine/llm_engine.py]
  → Tokenization + InputProcessor           [vllm/v1/engine/input_processor.py]
  → EngineCore (separate process, via ZMQ)  [vllm/v1/engine/core.py]
  → Scheduler (decides what to run)         [vllm/v1/core/sched/scheduler.py]
  → Executor → Worker                       [vllm/v1/executor/, worker/gpu_worker.py]
  → GPUModelRunner (forward pass)           [vllm/v1/worker/gpu_model_runner.py]
  → Model (attention + MLP layers)          [vllm/model_executor/models/]
  → Logits → Sampling → Token              [vllm/v1/sample/]
  → OutputProcessor → Detokenize → User    [vllm/v1/engine/output_processor.py]
```

### Reading Order

| # | File | Focus On |
|---|------|----------|
| 1 | `docs/design/paged_attention.md` | vLLM's core innovation — KV cache as virtual memory |
| 2 | `vllm/v1/engine/core.py` | Top ~100 lines — how EngineCore orchestrates everything |
| 3 | `vllm/v1/core/sched/scheduler.py` | Top ~150 lines — how requests are scheduled |
| 4 | `vllm/v1/core/kv_cache_manager.py` | How KV cache blocks are allocated/freed |
| 5 | `vllm/v1/core/block_pool.py` | The block allocator (like a GPU memory pool) |
| 6 | `docs/design/multiprocessing.md` | Why engine uses separate processes + ZMQ |

### Key Insight for GPU Developers

The Scheduler + KVCacheManager is conceptually identical to a GPU memory manager:
- `BlockPool` = free list of fixed-size memory blocks
- `KVCacheManager` = virtual-to-physical page table
- Scheduling = deciding which "draw calls" (requests) fit in available VRAM

---

## Phase 3: Model Execution Layer (Week 3)

**Goal**: Understand how models run on GPU — layers, attention, and kernels.

### Reading Order

| # | File | What You Learn |
|---|------|---------------|
| 1 | `vllm/model_executor/models/phi3.py` | Simplest model (18 lines! just inherits from Llama) |
| 2 | `vllm/model_executor/models/llama.py` | Reference model implementation (~500 lines) — study this carefully |
| 3 | `vllm/model_executor/layers/linear.py` | Parallelized linear layers (tensor parallelism) |
| 4 | `vllm/model_executor/layers/attention/` | Attention layer abstraction |
| 5 | `vllm/model_executor/layers/rotary_embedding/` | Position embeddings |
| 6 | `vllm/model_executor/layers/layernorm.py` | Layer normalization |
| 7 | `docs/design/attention_backends.md` | How different attention kernels are swapped |
| 8 | `docs/contributing/model/basic.md` | Guide on implementing a new model |

### Understanding the Model Pattern

Every model in vLLM follows this pattern:
```python
class LlamaForCausalLM(nn.Module):
    def __init__(self, ...):
        self.model = LlamaModel(...)       # Transformer stack
        self.lm_head = ...                  # Final projection to vocabulary

    def forward(self, input_ids, positions, intermediate_tensors, ...):
        hidden_states = self.model(input_ids, positions, ...)
        return hidden_states               # Logits computed separately
```

Layers use **custom CUDA kernels** (not standard PyTorch) for performance. The `layers/` directory wraps these kernels in familiar PyTorch `nn.Module` interfaces.

---

## Phase 4: GPU Optimization & Kernels (Week 4)

**Goal**: Understand the performance-critical GPU code.

### Reading Order

| # | File | What You Learn |
|---|------|---------------|
| 1 | `docs/design/cuda_graphs.md` | How CUDA graphs eliminate CPU overhead |
| 2 | `docs/design/torch_compile.md` | torch.compile integration with piecewise graphs |
| 3 | `vllm/v1/worker/gpu_model_runner.py` | Top ~200 lines — how forward passes are orchestrated |
| 4 | `vllm/compilation/` | torch.compile + CUDA graph capture system |
| 5 | `csrc/` | Browse C++/CUDA kernel sources (attention, cache, quantization) |
| 6 | `vllm/kernels/` | Python wrappers for custom kernels |
| 7 | `docs/design/fused_moe_modular_kernel.md` | Mixture-of-experts kernel design |

### GPU Graphics → CUDA Compute Mental Model

| Graphics Concept | vLLM Equivalent |
|-----------------|-----------------|
| Command buffer recording/replay | CUDA Graphs |
| Shader programs | CUDA kernels in `csrc/` |
| Texture memory binding | KV cache block table mapping |
| Draw call batching | Continuous batching of requests |
| Compute shaders | Attention/MoE kernels (Triton or CUDA) |
| Pipeline state objects | `torch.compile` compiled graphs |
| Virtual textures / sparse binding | PagedAttention block mapping |
| Multi-pass rendering | Prefill pass + Decode passes |

---

## Phase 5: Advanced Topics (Week 5+)

Pick based on interest:

### Distributed Inference
- `vllm/distributed/` — tensor/pipeline/expert parallelism
- `vllm/v1/executor/multiproc_executor.py` — multi-GPU
- `vllm/v1/executor/ray_executor.py` — multi-node

### Speculative Decoding
- `vllm/v1/spec_decode/` — draft models, EAGLE, n-gram proposers
- Concept: use a small fast model to "guess" multiple tokens, verify with the big model

### Quantization
- `vllm/model_executor/layers/quantization/` — FP8, INT8, INT4, GPTQ, AWQ
- Concept: reduce model precision to save memory and increase throughput

### Multimodal (Vision/Audio)
- `vllm/multimodal/` — image/video/audio input processing
- Models like LLaVA, Qwen-VL in `vllm/model_executor/models/`

### Configuration Deep Dive
- `vllm/config/vllm.py` — the master `VllmConfig` object (~2000 lines)
- `vllm/envs.py` — all environment variables

---

## Quick Reference: File Map

```
vllm/
├── entrypoints/          # User APIs (LLM class, OpenAI server, CLI)
├── v1/
│   ├── engine/           # Front-end engine (AsyncLLM, EngineCore, ZMQ IPC)
│   ├── core/sched/       # Scheduler (what to run each step)
│   ├── core/             # KV cache manager, block pool
│   ├── executor/         # Process management (single/multi-GPU/Ray)
│   ├── worker/           # GPU worker + model runner
│   ├── attention/        # Attention backend implementations
│   └── spec_decode/      # Speculative decoding
├── model_executor/
│   ├── models/           # 200+ model architectures
│   └── layers/           # Reusable layers (linear, attention, MoE, etc.)
├── config/               # All configuration dataclasses
├── distributed/          # Parallelism + communication
├── compilation/          # torch.compile + CUDA graph management
├── kernels/              # Python wrappers for custom kernels
├── platforms/            # Hardware abstraction (CUDA, ROCm, TPU, CPU)
└── multimodal/           # Vision/audio/video processing
csrc/                     # C++/CUDA kernel source code
docs/design/              # Architecture and design documents
examples/                 # Runnable examples
```

---

## Study Tips

1. **Start from the user API and trace inward** — Don't start with kernels. Start with `LLM.generate()` and follow the call stack.

2. **Read `docs/design/` liberally** — These 27 design docs explain the "why" behind the code.

3. **Use your GPU intuition** — You already understand memory management, batching, and command recording. vLLM applies these same patterns to LLM inference.

4. **Run the code with a small model** — `facebook/opt-125m` (250MB) is perfect for experimentation. It runs fast even on CPU.

5. **The V1 engine is the only one that matters** — Ignore `vllm/engine/` (legacy). All active development is in `vllm/v1/`.
