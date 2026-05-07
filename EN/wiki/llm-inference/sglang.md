---
title: "SGLang: Fast Structured Generation and Serving"
category: llm-inference
tags: [sglang, radix-attention, structured-generation, inference-engine, constrained-decoding]
created: 2026-04-13
updated: 2026-04-14
status: mature
---

# SGLang: Fast Structured Generation and Serving

> [!abstract]+ TL;DR
> SGLang (Structured Generation Language) from LMSYS (UC Berkeley) combines a **frontend DSL** for programming LLM applications with a **backend runtime (SRT)** featuring novel optimizations: **RadixAttention** (cross-request KV reuse via radix tree), compressed FSM (jump-forward constrained decoding), XGrammar integration. First published Dec 2023 (NeurIPS 2024). As of 2026 powers xAI (Grok 3), Microsoft Azure, LinkedIn, Cursor — 400,000+ GPUs in production. Delivers **29 % higher throughput than [[vllm|vLLM]]** on H100 (16,200 vs 12,500 tok/s) and up to **6.4× gains** on prefix-heavy workloads. Excels at multi-turn conversations, structured output, and agent workloads.

[GitHub](https://github.com/sgl-project/sglang) | [Paper](https://arxiv.org/abs/2312.07104)

---

## Architecture

SGLang uses a multi-process architecture to avoid Python GIL contention:

```
+-------------------------------------------------------+
|                  User Applications                     |
|   @sgl.function programs / OpenAI-compatible API       |
+-------------------------------------------------------+
         |                              |
         v                              v
+------------------+     +---------------------------+
| SGLang Frontend  |     |   HTTP/gRPC API Gateway   |
| DSL (gen/select/ |     |   (FastAPI, OpenAI compat) |
| fork/extend)     |     +---------------------------+
+------------------+                |
         |                          v
         |            +---------------------------+
         |            |    TokenizerManager       |
         |            |    (separate process)      |
         |            +---------------------------+
         |                     | ZeroMQ IPC
         v                     v
+------------------------------------------------+
|              Scheduler                          |
|  - Cache-aware scheduling (LPM policy)          |
|  - RadixCache management                        |
|  - Memory budget control                        |
|  - Batch formation (ScheduleBatch)              |
+------------------------------------------------+
                       |
                       v
+------------------------------------------------+
|           ModelRunner / Worker                   |
|  - FlashInfer / Triton attention kernels         |
|  - CUDA Graph optimization                       |
|  - ForwardBatch GPU execution                    |
+------------------------------------------------+
                       |
                       v
+------------------------------------------------+
|              GPU Cluster                         |
|     TP / PP / DP / EP / CP parallelism           |
+------------------------------------------------+
```

**Request flow**: User request --> FastAPI --> TokenizerManager (tokenize) --> Scheduler (prefix match, batch formation) --> ModelRunner (GPU forward) --> Scheduler (sample token, update RadixCache) --> DetokenizerManager (detokenize, stream back).

Data structures evolve through the pipeline: **Req** --> **ScheduleBatch** (CPU) --> **ModelWorkerBatch** (Worker) --> **ForwardBatch** (GPU).

---

## RadixAttention

The core innovation enabling **automatic cross-request KV cache reuse** with zero configuration.

### How It Works

A radix tree maps token sequences to their KV cache tensors in paged GPU memory (1 token = 1 page):

```
                    [root]
                   /      \
             [system       [system
              prompt A]     prompt B]
             /    \              \
       [user      [user         [user
        msg 1]     msg 2]        msg X]
       /    \
 [asst      [asst
  rsp 1]     rsp 1']

Each node stores:
  - Token sequence fragment (edge label)
  - KV cache tensor reference (paged GPU memory)
  - Reference count + LRU timestamp
```

1. **Prefix matching**: Walk the tree to find the longest cached prefix
2. **Incremental compute**: Prefill only from the match point onward
3. **LRU eviction**: Recursively evict zero-refcount leaf nodes when memory is full
4. **Cache-aware scheduling**: Prioritize requests with longest prefix match (approximate DFS order)

### vs vLLM Hash-Based Prefix Caching

| Aspect | SGLang RadixAttention | vLLM Prefix Caching |
|--------|----------------------|---------------------|
| Granularity | Token-level | Block-level (16 tokens) |
| Configuration | Zero-config, automatic | Manual `--enable-prefix-caching` |
| Scheduling | Cache-aware (native) | Independent from cache |
| Cache hit rates | 50-99% across benchmarks | Lower (block boundary mismatch) |

**Four reuse patterns**: few-shot examples, self-consistency sampling, multi-turn chat history, tree-of-thought search branches.

---

## SGLang DSL (Frontend Language)

A Python-embedded DSL with primitives for declarative LLM programming.

### Primitives

| Primitive | Syntax | Purpose |
|-----------|--------|---------|
| **extend** | `s += "text"` | Append text to prompt |
| **gen** | `gen(name, max_tokens, stop, regex, choices)` | LLM generation (non-blocking) |
| **select** | `gen(name, choices=[...])` | Choose highest-probability option |
| **fork** | `s.fork(n)` | Create n parallel branches |
| **roles** | `s.system()` / `s.user()` / `s.assistant()` | Chat template wrappers |

### Key Examples

```python
from sglang import function, gen, system, user, assistant

@function
def multi_turn(s):
    s += system("You are a helpful assistant.")
    s += user("List 3 countries and capitals.")
    s += assistant(gen("a1", max_tokens=256))
    s += user("List 3 more, different from above.")
    s += assistant(gen("a2", max_tokens=256))  # Auto KV cache reuse

@function
def tool_use(s, question):
    s += assistant("I need a " +
        gen("tool", choices=["calculator", "search engine"]) + ". ")
    if s["tool"] == "calculator":
        s += assistant("Expression: " + gen("expr"))

@function
def parallel_eval(s, essay):
    forks = s.fork(3)
    for f, dim in zip(forks, ["clarity", "argument", "creativity"]):
        f += assistant(gen(f"{dim}_eval", max_tokens=256))
```

---

## Constrained Decoding

Two core technologies: **Compressed FSM** and **XGrammar**.

### Compressed FSM + Jump-Forward

Analyzes the FSM for consecutive singular-transition edges and compresses them:

```
Original FSM:  S0 --{--> S1 --"--> S2 --n--> S3 --a--> S4 --m--> S5 --e--> S6 --"--> S7
Compressed:    S0 ------{"name"------> S7  (one step instead of seven)
```

**Jump-forward optimization**: When entering a deterministic path, prefill the entire token sequence at once instead of decoding token-by-token. RadixAttention automatically reuses the KV cache.

### XGrammar Integration

Default structured output backend since v0.4. Supports JSON schema, regex, EBNF grammar, and structural tags. Overlaps grammar mask generation with the LLM forward pass, eliminating constraint overhead.

**Performance**: 2x latency reduction, 2.5x throughput improvement for JSON. Constrained decoding becomes *faster* than unconstrained decoding. 99.8% JSON compliance rate.

---

## FlashInfer Integration

SGLang uses FlashInfer as its default attention kernel backend:
- Optimized PagedAttention kernels (block-sparse attention)
- GPU shared memory prefetching for page indices
- JIT-compiled custom attention variants
- Native FP8 KV cache support
- **29-69% ITL reduction** and up to **21% TTFT improvement** vs Triton backend

---

## Parallelism Strategies

Five orthogonal strategies: `world_size = TP x PP x EP x DP`

| Strategy | Flag | Use Case |
|----------|------|----------|
| Tensor Parallelism | `--tp N` | Model too large for one GPU |
| Pipeline Parallelism | `--pp N` | Cross-node large models |
| Data Parallelism | `--dp N` | Maximize throughput |
| Expert Parallelism | `--ep N` | MoE models (DeepSeek V3/R1) |
| Context Parallelism | `--cp N` | Ultra-long sequences |

---

## Performance

### Throughput (H100)

| Engine | Standard (tok/s) | Prefix-Heavy | DeepSeek V3 |
|--------|-------------------|-------------|-------------|
| **SGLang** | **16,200** | **up to 6.4x** | **3.1x vs vLLM** |
| vLLM | 12,500 | baseline | baseline |

### Latency (Llama 3.1 8B, H100)

| Metric | SGLang | vLLM |
|--------|--------|------|
| TTFT | **79 ms** | 103 ms |
| ITL | **6.0 ms** | 7.1 ms |
| Output throughput | **894 tok/s** | 413 tok/s |

### Large-Scale PD Disaggregation (96 H100s)

- Input: 52,300 tok/s per node
- Output: 22,300 tok/s per node
- 5x improvement over vanilla TP

### GPU Utilization (v0.4+)

Zero-overhead batch scheduler: **95-98% GPU utilization** (vs 70-80% traditional), CPU overhead <2%.

---

## SGLang vs vLLM

| Dimension | SGLang | vLLM |
|-----------|--------|------|
| Core optimization | RadixAttention (radix tree) | PagedAttention (paged memory) |
| KV cache reuse | Cross-request, automatic | Manual enable, block-level |
| Structured output | Compressed FSM + XGrammar (~3x faster) | XGrammar / Outlines |
| Frontend DSL | Yes (gen/select/fork) | No (API only) |
| H100 throughput | **16,200 tok/s** | 12,500 tok/s |
| Hardware support | NVIDIA, AMD (limited) | NVIDIA, AMD, TPU, Trainium, Gaudi |
| Model coverage | Decoder, multimodal, MoE | Decoder, encoder-decoder, multimodal, MoE |
| Community | ~25K stars, 600 contributors | ~75K stars, 2,400 contributors |
| PD disaggregation | First-class API | Experimental |

**Choose SGLang** for: multi-turn chat, RAG, structured JSON output, DeepSeek deployments, agent workloads, maximum throughput.

**Choose vLLM** for: batch processing (unique prompts), non-NVIDIA hardware, encoder-decoder models (T5/BART), broadest model support, larger community.

---

## Deployment

### Server Launch

```bash
# Basic
python -m sglang.launch_server \
    --model meta-llama/Llama-3.1-8B-Instruct --port 30000

# Multi-GPU with DP
python -m sglang.launch_server \
    --model meta-llama/Llama-3.1-70B-Instruct --tp 4 --dp 2

# DeepSeek with EP
python -m sglang.launch_server \
    --model deepseek-ai/DeepSeek-V3 --tp 4 --ep 4
```

### Key Tuning Parameters

| Parameter | Recommendation |
|-----------|----------------|
| `mem_fraction_static` | 0.80-0.90 |
| `schedule_policy` | `lpm` (longest prefix match) |
| `chunked_prefill_size` | 8192 for long inputs |
| `quantization` | fp8 for memory savings |

---

## Limitations

- **Hardware**: Primarily NVIDIA; limited AMD; no TPU/Trainium/Gaudi (vLLM supports all)
- **Model coverage**: No encoder-decoder models (T5, BART)
- **Community**: Smaller than vLLM (~25K vs ~75K stars, slower issue response)
- **Python GIL**: CPU routing bottleneck under extreme concurrency
- **RadixAttention overhead**: Cache memory cost counterproductive when prefix overlap is low
- **Maturity**: Younger project, API still evolving; some features (PD disagg, CP) in rapid iteration

---

## References

- Zheng, L., et al. **"SGLang: Efficient Execution of Structured Language Model Programs."** NeurIPS 2024. [arXiv:2312.07104](https://arxiv.org/abs/2312.07104)
- LMSYS Blog. **"Fast and Expressive LLM Inference with RadixAttention and SGLang."** [Link](https://www.lmsys.org/blog/2024-01-17-sglang/)
- LMSYS Blog. **"Fast JSON Decoding with Compressed FSM."** [Link](https://www.lmsys.org/blog/2024-02-05-compressed-fsm/)
- LMSYS Blog. **"SGLang v0.4."** [Link](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/)
- LMSYS Blog. **"DeepSeek PD Disaggregation on 96 H100s."** [Link](https://www.lmsys.org/blog/2025-05-05-large-scale-ep/)
- Dong, Y., et al. **"XGrammar."** [arXiv:2411.15100](https://arxiv.org/abs/2411.15100)
- Ye, Z., et al. **"FlashInfer."** [arXiv:2501.01005](https://arxiv.org/abs/2501.01005)

---

## Related Pages

- [[vllm]] — Alternative serving engine (broader hardware support)
- [[continuous-batching]] — Continuous batching technique used by SGLang
- [[kv-cache-optimization]] — KV cache techniques (PagedAttention, prefix caching)
- [[structured-output-serving]] — Constrained decoding and structured output
- [[multi-turn-optimization]] — Multi-turn serving where SGLang excels
- [[prefill-decode-disaggregation]] — PD disaggregation deployment
- [[speculative-decoding]] — Speculative decoding (EAGLE)
- [[flashinfer]] — FlashInfer attention kernel library
