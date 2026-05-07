---
title: "TensorRT-LLM: NVIDIA's Optimized Inference Engine"
category: llm-inference
tags: [tensorrt-llm, nvidia, inference-optimization, compilation, quantization, inflight-batching]
created: 2026-04-13
updated: 2026-05-07
status: mature
---

# TensorRT-LLM: NVIDIA's Optimized Inference Engine

> [!abstract]+ TL;DR
> NVIDIA's purpose-built library for high-performance LLM inference: Python model-definition API + highly optimized C++/CUDA runtime, achieving the **highest raw kernel performance** on NVIDIA GPUs via graph-level optimization, operator fusion, and custom CUDA kernels. Key capabilities: native **FP8/NVFP4** quantization, in-flight batching, paged KV cache, TP/PP/EP parallelism, **EAGLE-3 / MTP / N-gram** speculative decoding. Trade-offs vs. [[vllm|vLLM]]/[[sglang|SGLang]]: highest single-GPU perf but NVIDIA-only and harder to customize.

> [!info] Links
> [GitHub](https://github.com/NVIDIA/TensorRT-LLM) · [Docs](https://nvidia.github.io/TensorRT-LLM/)

## Architecture

TensorRT-LLM operates in two phases: **Build** (compile) and **Runtime** (serve).

```
 ┌─────────────────── Build Phase ───────────────────┐
 │  Model Weights (HF) ──┐                          │
 │  Model Definition ─────┼──→ Graph Optimization    │
 │  Quantization Config ──┘    Kernel Selection      │
 │                             Compilation           │
 │                                 │                 │
 │                                 ▼                 │
 │                        .engine file               │
 │                  (GPU-architecture-specific)       │
 └───────────────────────────┬───────────────────────┘
                             │
 ┌─────────────────── Runtime Phase ─────────────────┐
 │  In-Flight Batcher → KV Cache Manager →           │
 │  Attention Kernels → Sampling → Output            │
 │  (C++/CUDA runtime, dynamic request management)   │
 └───────────────────────────────────────────────────┘
```

### Graph Optimization

The build phase performs extensive optimizations:
- **Operator fusion**: QKV projections merged into 1 GEMM, attention + softmax fused, LayerNorm + bias fused
- **Constant folding**: Pre-compute input-independent expressions
- **Memory optimization**: Tensor reuse, optimal memory layouts
- **AutoTuner**: Tries multiple kernel implementations per op, selects fastest for target GPU

## Core Optimizations

**In-Flight Batching**: Dynamic request scheduling (equivalent to [[continuous-batching]]). New requests join immediately as others finish -- GPU stays fully utilized.

**Paged KV Cache**: Block-based allocation (similar to [[paged-attention]]) with KV Cache Connector API for cross-request reuse (prefix caching) and cross-node transfer (for [[prefill-decode-disaggregation]]).

**Quantization** (best-in-class on NVIDIA):

| Method | Bits | Hardware | Notes |
|--------|------|----------|-------|
| FP8 | 8 | Hopper+ | Near-lossless, default choice |
| NVFP4 | 4 | Blackwell | Two-level scaling, native |
| Block-scale FP8 | 8 | Hopper+ | For DeepSeek V3 |
| INT4 AWQ/GPTQ | 4 | All | Weight-only |
| INT8 SmoothQuant | 8 | All | W8A8 |

**Custom CUDA Kernels**: Flash Attention variants, fused MoE, paged context FMHA, native FP8 GEMM.

## Comparison with vLLM / SGLang

| Dimension | TensorRT-LLM | vLLM | SGLang |
|-----------|-------------|------|--------|
| Developer | NVIDIA | UC Berkeley + community | LMSYS + community |
| Open source | Partial (closed kernels) | Fully open | Fully open |
| Hardware | NVIDIA only | NVIDIA, AMD, TPU, CPU | NVIDIA, AMD, TPU |
| Setup complexity | High (engine compilation) | Low (pip install) | Low (pip install) |
| Raw kernel perf | Highest | High | High |
| Model support | Lags behind | Broadest | Broad |
| Structured output | Basic | Supported | Best (RadixAttention) |
| Community | Moderate | Most active | Active |

**Performance gaps** (approximate): TRT-LLM is ~10-20% faster in raw throughput and latency. Gaps are narrowing as vLLM/SGLang iterate rapidly.

**Choose TRT-LLM** when: pure NVIDIA environment, need last 10-20% performance, dedicated MLOps team, fixed model, need NVFP4/Block-scale FP8.
**Choose vLLM/SGLang** when: multi-hardware, rapid iteration, community support, model variety.

## Code Examples

### Build + Serve Workflow

```bash
# Convert checkpoint (FP8, 4-way tensor parallel)
python3 -m tensorrt_llm.commands.convert_checkpoint \
    --model_dir /models/llama-3.3-70b \
    --output_dir /checkpoints/llama-70b-fp8-tp4 \
    --dtype float16 --tp_size 4 --use_fp8

# Build TRT engine
trtllm-build \
    --checkpoint_dir /checkpoints/llama-70b-fp8-tp4 \
    --output_dir /engines/llama-70b-fp8-tp4 \
    --max_batch_size 64 --max_input_len 4096 --max_seq_len 8192 \
    --gemm_plugin auto --use_paged_context_fmha enable

# Serve (OpenAI-compatible API)
python3 -m tensorrt_llm.commands.serve \
    --engine_dir /engines/llama-70b-fp8-tp4 \
    --tokenizer_dir /models/llama-3.3-70b \
    --host 0.0.0.0 --port 8000
```

## MoE Support

TensorRT-LLM provides best-in-class MoE support via **Expert Parallelism (EP)**: each GPU holds complete experts (rather than slicing every expert across GPUs). Communication is AllToAll (proportional to active experts) instead of AllReduce (proportional to total experts).

- **Wide EP**: For models with hundreds of experts (e.g., DeepSeek V3 with 256 experts)
- **Fused MoE kernels**: Routing + expert computation fused, with AutoTuner
- **FP8/NVFP4 MoE**: Quantized expert computation

## Triton Inference Server Integration

For production deployment, TensorRT-LLM integrates with NVIDIA Triton:

```
Client (HTTP/gRPC) → Triton Server → TRT-LLM Backend
                      (scheduling)    (engine + KV cache + batching)
```

Provides health checks, metrics, model versioning, multi-model serving, and streaming support.

## Limitations

1. **Long build times**: 30min to hours for large models. Must rebuild for parameter changes.
2. **GPU-architecture bound**: H100 engines do not run on A100. Multiple engine builds for mixed fleets.
3. **NVIDIA-only**: No AMD, Intel, or Apple Silicon support.
4. **Model support lag**: New architectures arrive weeks/months after vLLM/SGLang.
5. **High complexity**: Many dependency versions to align. Debugging C++ runtime is difficult. Docker recommended.
6. **Low flexibility**: Custom architectures require reimplementation in TRT-LLM Python API. Modifying inference logic is much harder than in vLLM/SGLang.
7. **Less community-driven**: Development depends on NVIDIA's internal team.
8. **Diminishing returns**: For many workloads, vLLM/SGLang performance is close enough (within 10-20%) that TRT-LLM's complexity may not be justified.

## References

- NVIDIA, "TensorRT-LLM," 2023-2026. [GitHub](https://github.com/NVIDIA/TensorRT-LLM)
- NVIDIA, "TensorRT-LLM Documentation." [Docs](https://nvidia.github.io/TensorRT-LLM/)
- NVIDIA, "Triton Inference Server." [GitHub](https://github.com/triton-inference-server/server)
- NVIDIA, "FP8 Quantization," 2024.
- NVIDIA, "NVFP4 on Blackwell," GTC 2025.

## Related Pages

- [[vllm]] -- Open-source alternative, broader hardware support
- [[sglang]] -- Open-source alternative, better structured output
- [[quantization]] -- Quantization methods (GPTQ, AWQ, FP8)
- [[continuous-batching]] -- Theoretical basis for in-flight batching
- [[paged-attention]] -- Paged KV cache principles
- [[model-parallelism]] -- Tensor, pipeline, expert parallelism
- [[speculative-decoding]] -- EAGLE-3, MTP, N-gram speculation
- [[kv-cache-optimization]] -- KV cache management
- [[prefill-decode-disaggregation]] -- TRT-LLM's disaggregated serving
