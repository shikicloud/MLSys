---
title: "vLLM: High-Throughput LLM Serving Engine"
category: llm-inference
tags: [vllm, serving, paged-attention, inference-engine, v1-architecture, continuous-batching]
created: 2026-04-13
updated: 2026-05-20
status: mature
---

# vLLM: High-Throughput LLM Serving Engine

> [!abstract]+ TL;DR
> vLLM is an open-source, high-throughput, memory-efficient LLM inference and serving engine, released by UC Berkeley's Sky Computing Lab in 2023. The central innovation is [[paged-attention|PagedAttention]] — it borrows the OS virtual-memory paging idea to manage the KV cache and drives fragmentation to nearly zero. As of April 2026 (v0.19.0) it is one of the most widely deployed serving frameworks (50k+ GitHub stars), with the broadest hardware coverage (NVIDIA / AMD / TPU / Gaudi / Trainium); the V1 rewrite delivers **up to 1.7× throughput**.

> [!info] Links
> [GitHub](https://github.com/vllm-project/vllm) · [Docs](https://docs.vllm.ai/) · [vLLM Blog](https://vllm.ai/blog)

**Core strengths**:

- **Extreme memory efficiency**: PagedAttention drives KV-cache waste to near zero
- **High throughput**: [[continuous-batching|continuous batching]] + prefix caching + chunked prefill
- **Broadest hardware support**: NVIDIA (Ampere/Hopper/Blackwell), AMD, Intel Gaudi, Google TPU, AWS Trainium
- **OpenAI-compatible API**: drop-in replacement for the OpenAI backend
- **Rich model ecosystem**: Transformer, MoE, multi-modal, multi-LoRA
- **Active community**: bi-weekly releases, latest v0.19.0 (April 2026)

---

## Architecture evolution

### V0 architecture (2023-2024)

The defining V0 problem is that the **Scheduler and Worker 0 share a process**, which causes asymmetric architecture, CPU tasks blocking the GPU, full-state transfer, separated prefill/decode scheduling paths, and limited scalability.

### V1 architecture (default since v0.8.0, January 2025)

V1 is a ground-up rewrite of the engine, designed to fix every architectural problem of V0. V1 reaches **up to 1.7×** higher throughput than V0 while being cleaner and more extensible.

**Summary of V0 → V1 changes**:

| Dimension | V0 | V1 |
|------|----|----|
| Worker architecture | Asymmetric (W0 is special) | Symmetric (all Workers identical) |
| Scheduler | Prefill/Decode split | Unified `{req_id: n_tokens}` |
| State transfer | Full transfer | Incremental diffs |
| Batching | Rebuild inputs each step | Persistent batching (cache + diff) |
| CPU tasks | Same process as scheduler | Isolated to a separate process |
| IPC | Direct calls | ZeroMQ IPC |
| CUDA Graph | Traditional full-graph capture | Piecewise CUDA Graph |

---

## V1 architecture deep dive

### Overall diagram

```
┌── API Server process ───────────────────────────────────┐
│ HTTP → OpenAI API → Tokenization → AsyncLLM             │
│                        │ ZMQ IPC ▲                      │
│  ┌─ EngineCore process ▼─────────┴──────────────────┐   │
│  │  Input Queue → Scheduler{req_id:n} → KV Cache Mgr│   │
│  │                    → MultiProcExecutor            │   │
│  └────────────────────────┬──────────────────────────┘   │
│ Detokenization ◀── Output Queue ◀── Results              │
└───────────────────────────┼──────────────────────────────┘
       shared memory        │ rpc_broadcast_mq
    ┌───────────┬───────────┼───────────┐
    ▼           ▼           ▼           ▼
 Worker 0   Worker 1    Worker 2    Worker N
 GPU 0      GPU 1       GPU 2       GPU N
 ModelRunner + KV Cache(PagedAtt) + local state cache
    └─────── NCCL AllReduce ────────────┘
```

### EngineCore isolation

EngineCore runs in its own process and talks to the API Server via ZeroMQ. Key benefits:

1. **Bypass the GIL**: Python's GIL is no longer a bottleneck. CPU-heavy work (tokenization, multi-modal preprocessing, detokenization, response streaming) runs in the API Server process, in full parallel with GPU execution.
2. **Scheduling never blocks**: EngineCore runs a busy loop that keeps scheduling and dispatching forwards, so the GPU never idles waiting on CPU work.
3. **Clean separation of concerns**: the API layer handles HTTP, auth, format conversion; EngineCore focuses on scheduling and model execution.

```python
# EngineCore main loop (simplified)
while True:
    new_reqs = input_queue.get_nowait()     # 1. pull new requests
    scheduler.add_requests(new_reqs)
    schedule = scheduler.schedule()          # 2. {req_id: num_tokens}
    output = executor.execute_model(schedule)# 3. GPU forward pass
    output_queue.put(output)                 # 4. enqueue results
```

### Unified scheduler

The core innovation of the V1 scheduler is to collapse every scheduling decision into one simple dictionary:

```
{request_id: num_tokens}
```

This single abstraction unifies all of the following cases:

| Case | Schedule entry | Note |
|------|----------|------|
| Normal prefill | `{req_1: 512}` | Process all prompt tokens in one shot |
| Chunked prefill | `{req_1: 256}` | Long prompt, chunked |
| Normal decode | `{req_2: 1}` | Token-by-token autoregressive decoding |
| Speculative decoding | `{req_3: 5}` | Draft model proposes 5 tokens |
| Prefix cache hit | `{req_4: 128}` | Skip the cached prefix, only process the rest |

Scheduling flow: (1) allocate decode tokens for all running requests → (2) compute remaining budget → (3) pull prefill requests from the waiting queue (check prefix cache, chunk if needed, allocate KV blocks) → (4) emit `{req_id: num_tokens}`.

### Persistent batch

V0 rebuilds the full input tensor every step, wasting CPU time. V1 introduces a persistent batch:

```
step t:    batch = [req_1, req_2, req_3, req_4]
                      ↓ req_2 finishes, req_5 arrives
step t+1:  batch = [req_1, -----, req_3, req_4, req_5]
                           apply diff only!

Specifically:
  - drop req_2's slot (mark available)
  - place req_5's tokens into the free slot
  - leave the cached tensors at other slots untouched
```

NumPy ops replace native Python ops to apply diffs efficiently, sharply reducing per-step CPU overhead.

### Symmetric worker architecture

In V1 all Workers are identical (in V0 Worker 0 had special responsibilities). Each Worker caches request state locally and only receives incremental updates (new requests / completions / preemptions); instructions arrive through shared-memory `rpc_broadcast_mq` and results return via `worker_response_mq`.

### Request lifecycle

```
Client POST /v1/chat/completions
  → API Server (HTTP validation) → AsyncLLM (tokenization)
  → [ZMQ IPC] → EngineCore Input Queue
  → Scheduler ({req_id: num_tokens} + KV block allocation)
  → MultiProcExecutor → Workers (forward + sampling)
  → [ZMQ IPC] → AsyncLLM (detokenization)
  → API Server → SSE streaming response
```

---

## Core technology stack

### PagedAttention integration

The foundation of vLLM is [[paged-attention|PagedAttention]] (Kwon et al., SOSP 2023), which borrows the OS virtual-memory paging mechanism:

```
Traditional allocation: pre-allocate max length per request → ~40% average waste
PagedAttention: physical block pool [B0][B1][B2]... + page-table mapping
  Req1: B0→B3→B5 | Req2: B1→B4 | Req3: B2→B6→B7
  → only the last block has fragmentation (<4% waste)
```

Key parameters:
- `block_size`: tokens per physical block (default 16)
- `gpu_memory_utilization`: fraction of GPU memory used for KV cache (default 0.9)
- KV cache memory = total GPU memory × `gpu_memory_utilization` − model weights − activations

### Automatic prefix caching

Prefix caching is on by default in V1 and uses hash-based block-level caching. The idea: if multiple requests share the same prefix (e.g. a system prompt), they can reuse the already-computed KV cache.

**Hash computation**:

```python
# Each KV block's hash is computed from
block_hash = hash(
    parent_block_hash,     # parent block's hash (chain dependency)
    tuple(block_tokens),   # token sequence in this block
    extra_hashes           # LoRA ID / multi-modal input hash / cache salt
)
```

**BlockPool data structure**:

```
BlockPool
├── blocks[]: array of KVCacheBlock objects
│     each block contains:
│     - block_id: physical block ID
│     - ref_count: reference count (# requests using this block)
│     - last_access_time: last access timestamp
│     - block_hash: content hash
│
├── free_block_queue: doubly linked list (LRU order)
│     free blocks ordered by most-recent-use time
│
└── cached_block_map: {hash → KVCacheBlock}
      global hash table for fast cached-block lookup
```

**LRU eviction policy**:

```
Eviction priority (high to low):
1. ref_count == 0 (no request is using the block)
2. among ref_count == 0 blocks, evict the least-recently-used (LRU)
3. if last_access_time ties, prefer evicting blocks at the end of the
   longest prefix (i.e. evict from the tail, keep the head)
```

**Configurable hash algorithms**:

| Algorithm | Serialization | Trait |
|------|-----------|------|
| `sha256` (default) | Python pickle | Secure, general-purpose |
| `sha256_cbor` | cbor2 | Reproducible hashes |
| `xxhash` | pickle + xxHash 128-bit | Faster non-cryptographic hash |

**Performance characteristics**:
- 0% hit rate: overhead < 1% (effectively free)
- High hit rate: multiplicative throughput gain (skip computation for cached prefixes)
- Typical settings (multi-turn chat, shared system prompt): 60-90% hit rate

### Continuous batching

vLLM implements [[continuous-batching|continuous batching]] (also called iteration-level scheduling):

In static batching, finished requests leave empty slots until the whole batch ends. Continuous batching fills slots with new requests immediately (e.g. as soon as Req2 finishes, Req4 is scheduled into its place), keeping the GPU fully loaded.

### Chunked prefill

Chunked prefill is enabled by default in V1; it splits long prompts into smaller chunks and interleaves them with decode requests:

```
No chunked prefill:  Step1: Prefill_A(4096) | Step2: Decode_B,C,D  ← long prefill blocks decode
With chunked prefill: Step1: Prefill_A_chunk1(512)+Decode_B,C | Step2: chunk2+Decode_B,C | ...
→ Prefill/Decode interleaved, decode latency stable
```

**Why it works**: Prefill is compute-bound, decode is memory-bound. Mixing them in one batch saturates both compute and memory bandwidth at once. The scheduler first allocates decode tokens for all running requests (1 each), then fills the remaining `max_num_batched_tokens` budget with prefill chunks.

### CUDA Graph integration

V1 uses **piecewise CUDA Graphs**: split the compute graph at attention ops, capture the non-attention parts as CUDA Graphs, and run attention in eager mode (FlashAttention 3). Modes: `FULL_AND_PIECEWISE` (default, best perf, highest memory), `PIECEWISE`, `NONE`.

---

## Parallelism strategies

vLLM supports multiple parallelism strategies that can be flexibly combined for different model sizes and hardware configurations. For the deeper theory see [[parallelism-strategies-deep-dive]].

### Supported parallelism

| Strategy | Principle | Comm | Effect |
|------|------|------|------|
| **TP** | Shard each layer across GPUs | AllReduce | Lower single-request latency |
| **PP** | Layers split across GPUs | Pipeline | Cross-node, supports non-2^n GPU counts |
| **DP** | Independent replicas serve different requests | None | Boost concurrent throughput |
| **EP** | Experts distributed across GPUs (MoE only) | AllToAll/AllReduce | Must combine with TP or DP |

### DP Attention + EP (the core strategy for MoE)

DP Attention is a data-parallel variant designed specifically for MoE models and differs fundamentally from classic DP:

```
Classic DP: one full replica per GPU (KV cache fully duplicated)
DP Attention + EP: a single logical replica, independent attention + KV cache partitioned by request
┌──────────┐  ┌──────────┐  ┌──────────┐
│  GPU 0    │  │  GPU 1    │  │  GPU 2    │
│Attn(indep)│  │Attn(indep)│  │Attn(indep)│
│KV(part)   │  │KV(part)   │  │KV(part)   │
│ Expert0-2 │  │ Expert3-5 │  │ Expert6-8 │
└─────┬─────┘  └─────┬─────┘  └─────┬─────┘
      └──── AllToAll (expert comm) ──┘
```

**Why does MoE need DP Attention?**

DeepSeek-V3/R1 uses Multi-Head Latent Attention (MLA), which has only a single KV head. Under classic TP the KV cache cannot be split along the head dimension and must be fully replicated across all TP ranks. DP Attention solves this by partitioning the KV cache by request.

**Expert distribution modes**:

| Mode | Description | Comm |
|------|------|------|
| Sharded Expert (no EP flag) | All experts present on every GPU, weights sharded | AllReduce |
| Split Expert (`--enable-expert-parallel`) | Each GPU holds full weights for a different subset of experts | AllToAll (DP>1) |

Formula: experts per GPU = total experts / (TP_SIZE × DP_SIZE)

### Configuration examples

**Basic Tensor Parallel (single node, 4 GPUs)**:

```bash
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4
```

**Pipeline + Tensor Parallel (2 nodes, 4 GPUs each)**:

```bash
vllm serve meta-llama/Llama-3.1-405B-Instruct \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 2
```

**DeepSeek-R1 low-concurrency (TP+EP, 8 GPUs)**:

```bash
# For ≤128 concurrent requests
# 52% higher throughput, 80% lower TTFT vs. the DP variant
vllm serve deepseek-ai/DeepSeek-R1 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel
```

**DeepSeek-R1 high-concurrency (DP+EP, 8 GPUs)**:

```bash
# For ≥512 concurrent requests
# 47% higher throughput at high concurrency
vllm serve deepseek-ai/DeepSeek-R1 \
    --tensor-parallel-size 1 \
    --data-parallel-size 8 \
    --enable-expert-parallel
```

### Choosing a parallelism strategy

- **Dense models**: fits one GPU → single GPU; otherwise → TP+PP
- **MoE models (expert activation >3%)**: low concurrency (≤128) → TP+EP; high concurrency (≥512) → DP+EP
- **MoE models (expert activation <1%)**: don't use EP (AllToAll overhead exceeds the gain)
- **MLA models (DeepSeek)**: must use DP+EP (KV cache cannot be split along heads)

---

## Speculative decoding

vLLM supports multiple [[speculative-decoding|speculative decoding]] methods that accelerate autoregressive generation via "guess then verify". For the full theory see [[speculative-decoding]].

### Supported methods

| Method | Description | Speedup | When to use |
|------|------|--------|----------|
| Draft Model | Small model proposes candidate tokens, large model verifies | 1.5-2.5x | When a matching small model exists |
| EAGLE-1/3 | Lightweight draft head based on feature extrapolation | 2-3x | Recommended, high acceptance |
| Medusa | Multi-head parallel prediction | 1.5-2x | No extra model needed |
| N-gram | Predict based on input n-gram matches | 1.2-1.5x | Repetitive tasks (translation/summarization) |
| MLP Speculator | Lightweight MLP prediction head | 1.5-2x | Low-overhead scenarios |

### Configuration examples

```bash
# Draft Model speculative decoding
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model meta-llama/Llama-3.1-8B-Instruct \
    --num-speculative-tokens 5 \
    --speculative-draft-tensor-parallel-size 1

# EAGLE-3 speculative decoding
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model yuhuili/EAGLE3-LLaMA3.1-Instruct-70B \
    --speculative-method eagle \
    --num-speculative-tokens 5

# N-gram speculative decoding (no extra model needed)
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model [ngram] \
    --num-speculative-tokens 5 \
    --ngram-prompt-lookup-max 4
```

**Reference numbers** (EAGLE-3 on Llama 3.1 70B, H100):
- Draft acceptance rate: ~70-85%
- Average accepted length: 3.2-4.1 tokens
- End-to-end speedup: 2.0-2.8x (task-dependent)

In V1 the schedule representation for speculative decoding is identical to every other case: `{req_id: num_speculative_tokens}`. The scheduler does not have to distinguish between speculative and normal decode — one of the advantages of the unified scheduler.

---

## Quantization support

vLLM reduces memory footprint and accelerates inference via [[quantization|quantization]]. For the theory see [[quantization]].

### Supported quantization formats

| Method | Precision | Kernel support | Recommended use |
|----------|------|----------|----------|
| FP8 (E4M3) | W8A8 | vLLM native | **Recommended**: minimal accuracy loss, clear perf gain |
| FP4 (NV) | W4A4 | Blackwell SM120+ | NVIDIA Blackwell exclusive |
| AWQ | W4A16 | Marlin / GEMM | Best inference perf among weight-only methods |
| GPTQ | W4A16 | Marlin / Machete / Exllama | Optimized for Ampere+ |
| SmoothQuant | W8A8 | Native | Joint weight + activation quantization |
| INT4 | W4A16 | Compute Cap > 8.0 | Maximum memory savings |
| GGUF | Mixed precision | llama.cpp compatible | llama.cpp ecosystem |
| bitsandbytes | W4/W8 | HuggingFace compatible | Quick experiments |

### Usage examples

```bash
# Use an FP8-quantized model
vllm serve neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8 \
    --tensor-parallel-size 4

# Use an AWQ-quantized model
vllm serve TheBloke/Llama-2-70B-Chat-AWQ \
    --quantization awq \
    --tensor-parallel-size 4

# KV cache quantization (FP8)
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --kv-cache-dtype fp8
```

### Impact of quantization on performance

Taking Llama 3.1 70B on H100 as an example:

| Config | GPUs needed | Throughput (tok/s) | vs BF16 |
|------|-----------|----------------|-----------|
| BF16 (baseline) | 4x H100 | ~4,800 | 1.0x |
| FP8 | 2x H100 | ~5,200 | 1.08x |
| AWQ W4 | 2x H100 | ~4,600 | 0.96x |
| GPTQ W4 | 2x H100 | ~4,500 | 0.94x |

FP8 is the currently recommended scheme: negligible accuracy loss (<0.5% on most benchmarks), half the memory, and a small throughput uplift.

---

## Code examples

### Basic offline inference

```python
from vllm import LLM, SamplingParams

# Initialize the model
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    dtype="bfloat16",
    gpu_memory_utilization=0.9,      # GPU memory utilization
    max_model_len=8192,              # Max context length
    enable_prefix_caching=True,      # Enable prefix caching (on by default in V1)
)

# Sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    max_tokens=512,
    repetition_penalty=1.1,
    stop=["<|eot_id|>"],
)

# Build chat messages
messages_list = [
    [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Explain the attention mechanism in Transformer."},
    ],
    [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What is the KV Cache?"},
    ],
]

# Chat inference (automatically applies the chat template)
outputs = llm.chat(messages_list, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt[:50]}...")
    print(f"Output: {generated_text}\n")
```

### OpenAI-compatible API server

**Start the server**:

```bash
# Basic launch
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --api-key my-secret-key

# Production config
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.92 \
    --enable-prefix-caching \
    --host 0.0.0.0 \
    --port 8000
```

**Client**:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="my-secret-key",
)

# Streaming output
stream = client.chat.completions.create(
    model="meta-llama/Llama-3.1-70B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Implement quicksort in Python"},
    ],
    temperature=0.7,
    max_tokens=1024,
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### Multi-GPU server configurations

```bash
# 4 GPU Tensor Parallel
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --host 0.0.0.0 --port 8000

# 8 GPU TP + PP (cross-node)
vllm serve meta-llama/Llama-3.1-405B-Instruct \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 2 \
    --host 0.0.0.0 --port 8000

# DeepSeek-R1 with DP Attention + EP
vllm serve deepseek-ai/DeepSeek-R1 \
    --data-parallel-size 8 \
    --enable-expert-parallel \
    --enable-dbo \
    --host 0.0.0.0 --port 8000
```

### Common sampling parameters

```python
from vllm import SamplingParams

greedy   = SamplingParams(temperature=0, max_tokens=256)                              # Deterministic
creative = SamplingParams(temperature=1.2, top_p=0.95, top_k=100, max_tokens=2048)    # Creative
code_gen = SamplingParams(temperature=0.2, top_p=0.9, max_tokens=4096,
                          stop=["```\n"], repetition_penalty=1.05)                     # Code
beam     = SamplingParams(use_beam_search=True, best_of=4, temperature=0)              # Beam search
```

---

## Performance analysis

### Benchmark data

The numbers below are measured on NVIDIA H100 SXM5 80GB with Llama-family models:

**Single-GPU throughput (Llama 3.1 8B, BF16)**:

| Engine version | Throughput (tok/s) | Relative |
|----------|----------------|----------|
| vLLM V0 | ~7,500 | Baseline |
| vLLM V1 | ~12,500 | **1.67x** |

**Multi-GPU throughput (Llama 3.3 70B, FP8, H100)**:

| Concurrent requests | vLLM (tok/s) | SGLang (tok/s) | TensorRT-LLM (tok/s) |
|-----------|-------------|----------------|----------------------|
| 1 | 120 | 125 | 130 |
| 10 | 650 | 680 | 710 |
| 50 | 1,850 | 1,920 | 2,100 |
| 100 | 2,400 | 2,460 | 2,780 |

> Conditions: 512 avg input tokens, 256 avg output tokens, 200 prompts

**Latency (Llama 3.3 70B, FP8, single H100)**:

| Concurrency | vLLM TTFT p50/p95 (ms) | TensorRT-LLM TTFT p50/p95 (ms) |
|------|------------------------|----------------------------------|
| 1 | 45 / 68 | 38 / 55 |
| 10 | 120 / 195 | 105 / 170 |
| 50 | 380 / 720 | 340 / 620 |
| 100 | 740 / 1,450 | 680 / 1,280 |

**Large-scale MoE deployment (DeepSeek-R1, 671B)**:

| Setup | Hardware | Throughput |
|----------|------|--------|
| Wide-EP (H200 cluster) | Multi-node H200 | **2,200 tok/s/GPU** |
| Wide-EP (GB200) Prefill | 4 prefill + 1 decode | 26,200 prefill TPGS |
| Wide-EP (GB200) Decode | 4 prefill + 1 decode | 10,100 decode TPGS |

### VRAM usage

| State | vLLM | TensorRT-LLM | SGLang |
|------|------|-------------|--------|
| Model loaded (idle) | 71 GB | 74 GB | 72 GB |
| Peak at 50 concurrent | 76 GB | 77 GB | 75 GB |
| Peak at 100 concurrent | 78 GB | 79 GB | 78 GB |

### Cold start

| Engine | Cold start | Note |
|------|-----------|------|
| vLLM | ~62 s | Weight load + CUDA Graph capture |
| SGLang | ~58 s | Comparable to vLLM |
| TensorRT-LLM | ~28 min | One-time engine build (later starts are fast) |

---

## vLLM vs SGLang vs TensorRT-LLM

Detailed comparison of the three mainstream inference engines:

| Dimension | vLLM | [[sglang\|SGLang]] | [[tensorrt-llm\|TensorRT-LLM]] |
|------|------|--------|---------------|
| **Developer** | UC Berkeley + community | UC Berkeley | NVIDIA |
| **License** | Apache 2.0 | Apache 2.0 | Apache 2.0 |
| **Core advantage** | Broad compatibility, rich ecosystem | Prefix reuse + structured output | Peak performance |
| **Prefix cache** | Hash-based LRU (block level) | RadixAttention (token-level radix tree) | Limited |
| **Prefix-heavy workloads** | Good | **29% faster** | Average |
| **Hardware support** | **Broadest** (NVIDIA/AMD/TPU/Gaudi/Trainium) | Mainly NVIDIA + AMD | NVIDIA only |
| **Quantization** | FP4/FP8/AWQ/GPTQ/INT4/GGUF | FP8/AWQ/GPTQ | FP8/INT4/INT8 (TensorRT) |
| **Speculative decoding** | EAGLE-3, Draft Model, Medusa, N-gram | EAGLE, Draft Model | Draft Model |
| **Structured output** | XGrammar | **SGLang Grammar (better)** | Limited |
| **Multi-modal** | Full support | Supported | Supported |
| **LoRA** | Multi-LoRA batching | Supported | Limited |
| **Deployment complexity** | Low (pip install) | Low (pip install) | High (engine compile) |
| **Cold start** | ~60s | ~60s | ~28 min (first build) |
| **Community size** | ~50k stars | ~20k stars (fast growth) | ~10k stars |
| **Production maturity** | **Most mature** | Catching up fast | Mature (NVIDIA ecosystem) |
| **Best fit** | Diverse hardware, generic deployments, batch | Multi-turn chat, agents, structured output | Fixed models, max perf |

**Selection guide**:

```
Need AMD/TPU/Gaudi support?           → vLLM
Multi-turn chat, shared prefixes?     → SGLang (clear RadixAttention edge)
Fixed model, chasing max throughput?  → TensorRT-LLM
General use, fast onboarding?         → vLLM (richest ecosystem, best docs)
Large-scale MoE serving?              → vLLM (DP Attention + EP most mature)
```

---

## Deployment in practice

### Docker deployment

```bash
# Basic Docker launch
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model meta-llama/Llama-3.1-8B-Instruct

# Production Docker Compose
```

```yaml
# docker-compose.yml (key configuration)
services:
  vllm:
    image: vllm/vllm-openai:latest
    runtime: nvidia
    ports: ["8000:8000"]
    volumes: [model-cache:/root/.cache/huggingface]
    environment: [HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}]
    deploy:
      resources:
        reservations:
          devices: [{driver: nvidia, count: 4, capabilities: [gpu]}]
    command: >
      --model meta-llama/Llama-3.1-70B-Instruct
      --tensor-parallel-size 4 --gpu-memory-utilization 0.92
      --max-model-len 32768 --enable-prefix-caching
      --host 0.0.0.0 --port 8000
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      start_period: 120s    # model load takes time
    ipc: host               # NCCL shared memory
```

### Kubernetes deployment

The recommended path is the [vLLM Production Stack](https://github.com/vllm-project/production-stack) Helm chart. Key configuration points:

1. **Shared memory**: mount `emptyDir{medium: Memory}` to `/dev/shm` (required for NCCL)
2. **Startup probe**: `failureThreshold × periodSeconds > 120s` (weight load takes 30-120s)
3. **Model persistence**: PVC-backed weights so pod restarts don't re-download; multi-replica needs ReadWriteMany
4. **GPU resources**: `nvidia.com/gpu: N` limit, paired with `--tensor-parallel-size N`
5. **Health checks**: `/health` endpoint for startup/readiness/liveness probes

### Production monitoring metrics

| Metric | Description | Suggested alert |
|------|------|-------------|
| `vllm:num_requests_running` | Currently running requests | Near `max_num_seqs` |
| `vllm:num_requests_waiting` | Waiting queue length | >100 for 5 min |
| `vllm:gpu_cache_usage_perc` | KV cache utilization | >95% |
| `vllm:avg_prompt_throughput_toks_per_s` | Prefill throughput | 20% below baseline |
| `vllm:avg_generation_throughput_toks_per_s` | Decode throughput | 20% below baseline |
| `vllm:e2e_request_latency_seconds` | End-to-end latency | p99 > SLA |
| `vllm:time_to_first_token_seconds` | TTFT | p95 > target |
| `vllm:prefix_cache_hit_rate` | Prefix cache hit rate | Trend monitoring |

---

## Limitations

### Known limitations

| Limitation | Detail |
|------|----------|
| **CPU/edge deployment** | vLLM is GPU-only; CPU-only or edge scenarios should use llama.cpp / Ollama |
| **Low-concurrency settings** | At single-user or very low concurrency, vLLM's scheduling overhead may lose to lighter-weight alternatives |
| **Prefix-cache granularity** | Block-level hashing is less flexible than SGLang's token-level RadixAttention |
| **CUDA Graph memory** | The default `FULL_AND_PIECEWISE` mode consumes extra GPU memory; smaller GPUs may need a downgrade |
| **PCIe topology** | Without NVLink, PCIe machines can hit peer-access errors |
| **Very long sequences** | Extreme sequences (>128K tokens) put pressure on the KV cache and need careful tuning |
| **Engine compile latency** | First start time grows because of CUDA Graph capture and torch.compile |

### Inappropriate scenarios

- **Single-user desktop apps**: Ollama / llama.cpp are better
- **Embedded/mobile**: prefer lightweight MLC-LLM, llama.cpp etc.
- **Lowest latency on a fixed model**: TensorRT-LLM's compiled engine is lower latency
- **Severely memory-constrained (<16GB VRAM)**: consider GGUF + llama.cpp

### Security notes

- vLLM Completions API has had insecure-deserialization issues — always enable API key auth in production
- Do not expose it directly on the public internet; deploy behind a reverse proxy (Nginx/Envoy) + API gateway
- Sensitive-data scenarios require additional security review

---

## Roadmap

### Recent releases

| Version | Date | Highlights |
|------|------|----------|
| v0.19.0 | 2026-04 | Gemma 4 MoE support, zero-bubble async scheduler + speculative decoding, Vision ViT CUDA Graph |
| v0.18.x | 2026-03 | Model Runner V2, piecewise CUDA Graph for PP |
| v0.15.1 | 2026-02 | NVIDIA Blackwell SM120 + GB200 support, Wide-EP maturation |
| v0.9.0 | 2025-Q2 | DP Attention + EP for MoE (first introduction) |
| v0.8.5 | 2025-Q1 | EAGLE-1/3 integration |
| v0.8.0 | 2025-01 | V1 architecture enabled by default |

### Q1 2026 roadmap

- **EngineCore optimizations**: data-structure efficiency for the scheduler and KV cache manager
- **PyTorch compile integration**: custom compile + fusion passes, vLLM IR for kernel registration
- **Hardware support**: GB300 nightly wheels, day-zero accuracy validation for all frontier models
- **Bi-weekly release cadence**: 6 releases planned in Q1
- **vLLM-Omni**: TTS, diffusion, world models, VLA model support

### Long-term directions

- [[prefill-decode-disaggregation|Prefill-decode disaggregation]] (Mooncake Transfer Engine integration)
- KV cache offloading (CPU/SSD)
- vLLM Semantic Router (intelligent routing layer, v0.1 Iris released)
- Deeper [[speculative-decoding|speculative decoding]] optimization (zero-bubble overlap)
- Multi-modal serving (Omni models)

---

## References

### Core paper

- Kwon et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention" (SOSP 2023) — [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)

### Official resources

- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM Blog](https://vllm.ai/blog)
- [vLLM V1 Alpha Release Blog](https://vllm.ai/blog/v1-alpha-release)
- [vLLM Large Scale Serving Blog](https://vllm.ai/blog/large-scale-serving)
- [vLLM Production Stack](https://github.com/vllm-project/production-stack)

### Community resources

- [Life of an Inference Request (vLLM V1) — Ubicloud](https://www.ubicloud.com/blog/life-of-an-inference-request-vllm-v1)
- [The vLLM MoE Playbook — AMD ROCm Blog](https://rocm.blogs.amd.com/software-tools-optimization/vllm-moe-guide/README.html)
- [vLLM vs TensorRT-LLM vs SGLang Benchmarks — Spheron](https://www.spheron.network/blog/vllm-vs-tensorrt-llm-vs-sglang-benchmarks/)
- [Speculators v0.3.0 — vLLM Blog](https://vllm.ai/blog/speculators-v030)
- [EAGLE-3 with vLLM — Red Hat Developer](https://developers.redhat.com/articles/2025/07/01/fly-eagle3-fly-faster-inference-vllm-speculative-decoding)

---

## Related pages

- [[paged-attention]] — Core memory-management algorithm
- [[continuous-batching]] — Continuous batching scheduling
- [[speculative-decoding]] — Speculative decoding acceleration
- [[kv-cache-optimization]] — KV cache optimization techniques
- [[quantization]] — Model quantization methods
- [[model-parallelism]] — Model parallelism fundamentals
- [[parallelism-strategies-deep-dive]] — Deeper look at parallelism strategies
- [[prefill-decode-disaggregation]] — Prefill-decode disaggregation architecture
- [[sglang]] — Alternative inference engine (RadixAttention)
- [[tensorrt-llm]] — NVIDIA TensorRT-LLM inference engine
