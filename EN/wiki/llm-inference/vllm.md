---
title: "vLLM: High-Throughput LLM Serving Engine"
category: llm-inference
tags: [vllm, serving, paged-attention, inference-engine, v1-architecture, continuous-batching]
created: 2026-04-13
updated: 2026-04-14
status: mature
---

# vLLM: High-Throughput LLM Serving Engine

> [!abstract]+ TL;DR
> vLLM is an open-source, high-throughput, memory-efficient LLM inference and serving engine from UC Berkeley's Sky Computing Lab (2023). Its core innovation is [[paged-attention|PagedAttention]] — borrowing OS virtual memory paging to manage KV cache and virtually eliminating memory fragmentation. As of April 2026 (v0.19.0), it's one of the most widely deployed serving frameworks (50k+ GitHub stars), with broad hardware support (NVIDIA/AMD/TPU/Gaudi/Trainium) and the V1 rewrite delivering **up to 1.7× throughput** over V0.

> [!info] Links
> [GitHub](https://github.com/vllm-project/vllm) · [Docs](https://docs.vllm.ai/) · [Blog](https://vllm.ai/blog)

**Key strengths**: near-zero KV cache waste via PagedAttention; [[continuous-batching]] + prefix caching + chunked prefill for maximum GPU utilization; broadest hardware support; OpenAI-compatible API; rich model ecosystem (Transformer, MoE, multimodal, multi-LoRA).

---

## Architecture Evolution: V0 to V1

V1 (default since v0.8.0, Jan 2025) is a ground-up rewrite achieving **up to 1.7x throughput** over V0.

| Dimension | V0 | V1 |
|-----------|----|----|
| Worker design | Asymmetric (Worker 0 special) | Symmetric (all identical) |
| Scheduler | Separate prefill/decode paths | Unified `{req_id: num_tokens}` |
| State transfer | Full state each step | Incremental diffs only |
| Batching | Rebuild inputs per step | Persistent batch (cache + diffs) |
| CPU tasks | Co-located with scheduler | Isolated in separate process |
| IPC | Direct calls | ZeroMQ |
| CUDA Graphs | Full capture | Piecewise (split around attention) |

---

## V1 Architecture Deep Dive

### Multi-Process Design

```
┌──────────────────────────────────────────────────────┐
│  API Server Process                                  │
│  HTTP → OpenAI API → Tokenization → AsyncLLM         │
│                         │         ▲                  │
│                    ZMQ IPC     ZMQ IPC               │
│                         │         │                  │
│  ┌──────────────────────▼─────────┴──────────────┐   │
│  │  EngineCore Process (isolated)                │   │
│  │  ┌───────────────────────────────────────┐    │   │
│  │  │ Scheduler → KV Cache Mgr → Executor   │    │   │
│  │  └───────────────┬───────────────────────┘    │   │
│  └──────────────────┼────────────────────────────┘   │
│  Detokenization ◀── Output Queue ◀── Results          │
└──────────────────────┼───────────────────────────────┘
          Shared Memory │ (rpc_broadcast_mq)
    ┌─────────┬─────────┼─────────┐
    ▼         ▼         ▼         ▼
 Worker 0  Worker 1  Worker 2  Worker N
 (GPU 0)   (GPU 1)   (GPU 2)   (GPU N)
    └─────── NCCL AllReduce ──────┘
```

**EngineCore isolation** bypasses Python's GIL: tokenization, multimodal processing, and detokenization run in the API Server process, overlapping with GPU execution. The EngineCore runs a tight busy loop: schedule -> execute -> output.

**Unified Scheduler** represents all decisions as `{request_id: num_tokens}`, unifying regular prefill, chunked prefill, decode, speculative decode, and prefix-cache hits under one abstraction. It prioritizes decode requests first, then fills remaining budget with prefill chunks.

**Persistent Batch** caches input tensors across steps and applies only diffs (via NumPy), significantly reducing per-step CPU overhead.

**Symmetric Workers** cache request states locally and receive only incremental updates, eliminating V0's asymmetric Worker 0 problem.

---

## Core Technology Stack

### PagedAttention

[[paged-attention|PagedAttention]] (Kwon et al., SOSP 2023) manages KV cache as fixed-size blocks with a page table per request, reducing memory waste from ~40% to <4%. Key params: `block_size` (default 16), `gpu_memory_utilization` (default 0.9).

### Prefix Caching (Hash-Based LRU)

Enabled by default in V1. Multiple requests sharing the same prefix reuse computed KV cache blocks.

- **Hash computation**: `hash(parent_hash, tuple(block_tokens), extra_hashes)` -- chain-dependent, includes LoRA IDs and cache salts
- **Data structures**: BlockPool with KVCacheBlock array, free block queue (doubly linked list, LRU order), and cached block map (hash -> block)
- **Eviction**: LRU among blocks with ref_count=0; ties broken by evicting blocks at the end of longest prefixes first
- **Overhead**: <1% at 0% hit rate; multiplicative throughput gains at high hit rates (typical 60-90% for multi-turn chat)
- **Hash algorithms**: sha256 (default), sha256_cbor (reproducible), xxhash (faster, non-cryptographic)

### Continuous Batching & Chunked Prefill

[[continuous-batching|Continuous batching]] inserts new requests into completed slots immediately. **Chunked prefill** (always-on in V1) splits long prompts into chunks interleaved with decode tokens, exploiting the compute-bound (prefill) + memory-bound (decode) complementarity.

### Piecewise CUDA Graphs

V1 splits the computation graph around attention operations: non-attention ops are captured as CUDA graphs, while attention runs in eager mode (FlashAttention 3). Modes: `FULL_AND_PIECEWISE` (default, best perf, most memory), `PIECEWISE`, `NONE`.

---

## Parallelism Strategies

See [[parallelism-strategies-deep-dive]] for general theory. vLLM supports TP, PP, DP, and EP.

### DP Attention + EP for MoE

For MoE models (DeepSeek-V3/R1), DP Attention differs fundamentally from traditional DP: it creates a single logical model with replicated attention layers but distributed experts via AllToAll. KV cache is partitioned by request (not duplicated), critical for MLA models with a single KV head.

| Config | Communication | Best For |
|--------|--------------|----------|
| TP+EP | AllReduce | Low concurrency (<=128 req), 52% higher throughput |
| DP+EP | AllToAll | High concurrency (>=512 req), 47% higher throughput |

**Wide-EP production result**: 2,200 tok/s/GPU on H200 cluster (DeepSeek-R1).

```bash
# Low concurrency: TP+EP
vllm serve deepseek-ai/DeepSeek-R1 --tensor-parallel-size 8 --enable-expert-parallel

# High concurrency: DP+EP
vllm serve deepseek-ai/DeepSeek-R1 --data-parallel-size 8 --enable-expert-parallel --enable-dbo
```

---

## Speculative Decoding

Supports multiple [[speculative-decoding]] methods:

| Method | Speedup | Notes |
|--------|---------|-------|
| Draft Model | 1.5-2.5x | Requires compatible smaller model |
| EAGLE-1/3 | 2-3x | **Recommended** -- feature extrapolation draft head |
| Medusa | 1.5-2x | Multi-head parallel prediction |
| N-gram | 1.2-1.5x | Good for repetitive tasks |

```bash
# EAGLE-3
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model yuhuili/EAGLE3-LLaMA3.1-Instruct-70B \
    --speculative-method eagle --num-speculative-tokens 5
```

---

## Quantization

See [[quantization]] for fundamentals. Key formats:

| Method | Precision | Recommendation |
|--------|-----------|---------------|
| FP8 (E4M3) | W8A8 | **Best overall** -- minimal quality loss, significant speedup |
| AWQ | W4A16 | Best weight-only inference performance |
| GPTQ | W4A16 | Marlin/Machete kernels on Ampere+ |
| FP4 (NV) | W4A4 | Blackwell SM120+ only |
| INT4 | W4A16 | Maximum memory savings |

FP8 on Llama 3.1 70B: halves GPU count (4->2 H100s) while maintaining ~1.08x throughput vs BF16.

---

## Code Examples

### Offline Inference

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", dtype="bfloat16",
           gpu_memory_utilization=0.9, max_model_len=8192)

sampling = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512)

messages = [[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain the attention mechanism in Transformers."},
]]

outputs = llm.chat(messages, sampling)
for out in outputs:
    print(out.outputs[0].text)
```

### Online Serving

```bash
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 --gpu-memory-utilization 0.92 \
    --enable-prefix-caching --host 0.0.0.0 --port 8000
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="token")
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-70B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
)
for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")
```

---

## Performance Benchmarks

**Test config**: H100 SXM5 80GB, Llama 3.3 70B FP8, 512 avg input / 256 avg output tokens.

| Concurrency | vLLM (tok/s) | SGLang (tok/s) | TensorRT-LLM (tok/s) |
|-------------|-------------|----------------|----------------------|
| 1 | 120 | 125 | 130 |
| 10 | 650 | 680 | 710 |
| 50 | 1,850 | 1,920 | 2,100 |
| 100 | 2,400 | 2,460 | 2,780 |

**Cold start**: vLLM ~62s, SGLang ~58s, TensorRT-LLM ~28min (one-time compilation).

**V1 vs V0**: Llama 3.1 8B single H100 -- 12,500 vs 7,500 tok/s (1.67x improvement).

---

## vLLM vs SGLang vs TensorRT-LLM

| Aspect | vLLM | [[sglang\|SGLang]] | [[tensorrt-llm\|TensorRT-LLM]] |
|--------|------|--------|---------------|
| Prefix caching | Hash-based LRU (block-level) | RadixAttention (token-level) | Limited |
| Prefix-heavy perf | Good | **29% faster** | Baseline |
| Hardware | **Broadest** (NVIDIA/AMD/TPU/Gaudi/Trainium) | NVIDIA + AMD | NVIDIA only |
| Deployment complexity | Low (pip install) | Low | High (engine compilation) |
| Best for | Diverse hardware, general production | Multi-turn chat, structured output | Fixed model, max throughput |
| Community | ~50k stars | ~20k stars | ~10k stars |

---

## Production Deployment

**Docker**: Use `vllm/vllm-openai` image. Mount `/dev/shm` with sufficient shared memory for NCCL. Set health check `start_period` to 120s+ for model loading.

**Kubernetes**: Use [vLLM Production Stack](https://github.com/vllm-project/production-stack) Helm chart. Key requirements:
- Memory-backed emptyDir for `/dev/shm` (NCCL communication)
- Startup probe with sufficient `failureThreshold` (model loading time)
- PVC for model weights (avoid re-downloading on pod restart)
- ReadWriteMany PVC for multi-replica scaling

**Monitoring**: Track `gpu_cache_usage_perc`, `avg_generation_throughput_toks_per_s`, `time_to_first_token_seconds`, `prefix_cache_hit_rate`, and `num_requests_waiting`.

---

## Limitations

- **Not for CPU/edge**: Use llama.cpp/Ollama for single-user or resource-constrained scenarios
- **Prefix caching granularity**: Block-level hashing less flexible than SGLang's token-level RadixAttention
- **CUDA Graph memory**: Default mode consumes extra VRAM; may need to downgrade on smaller GPUs
- **Cold start**: CUDA Graph capture + torch.compile adds ~60s startup overhead
- **Security**: Past deserialization vulnerability in Completions API; always use API key auth + reverse proxy in production

---

## Roadmap (2026)

- **v0.19.0** (Apr 2026): Gemma 4 MoE, zero-bubble async scheduling with speculative decode
- **Q1 2026 focus**: EngineCore efficiency, PyTorch compilation integration, GB300 support, two-week release cadence
- **vLLM-Omni**: TTS, diffusion, world models, VLA model support
- **Long-term**: [[prefill-decode-disaggregation]], KV cache offloading, Semantic Router, deeper speculative decode optimization

---

## References

- Kwon et al. "Efficient Memory Management for LLM Serving with PagedAttention" (SOSP 2023) -- [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
- [vLLM V1 Alpha Release Blog](https://vllm.ai/blog/v1-alpha-release)
- [Life of an Inference Request (vLLM V1)](https://www.ubicloud.com/blog/life-of-an-inference-request-vllm-v1)
- [The vLLM MoE Playbook -- AMD ROCm](https://rocm.blogs.amd.com/software-tools-optimization/vllm-moe-guide/README.html)
- [vLLM Large Scale Serving Blog](https://vllm.ai/blog/large-scale-serving)
- [vLLM vs TensorRT-LLM vs SGLang Benchmarks](https://www.spheron.network/blog/vllm-vs-tensorrt-llm-vs-sglang-benchmarks/)

## Related Pages

- [[paged-attention]] -- Core memory management algorithm
- [[continuous-batching]] -- Scheduling strategy
- [[speculative-decoding]] -- Speculative decoding acceleration
- [[kv-cache-optimization]] -- KV cache optimization techniques
- [[quantization]] -- Model quantization methods
- [[model-parallelism]] -- Model parallelism fundamentals
- [[parallelism-strategies-deep-dive]] -- Parallelism strategies analysis
- [[prefill-decode-disaggregation]] -- Phase separation architecture
- [[sglang]] -- Alternative serving engine (RadixAttention)
- [[tensorrt-llm]] -- NVIDIA TensorRT-LLM
