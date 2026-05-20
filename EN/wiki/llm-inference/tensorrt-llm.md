---
title: "TensorRT-LLM: NVIDIA's Optimized Inference Engine"
category: llm-inference
tags: [tensorrt-llm, nvidia, inference-optimization, compilation, quantization, inflight-batching]
created: 2026-04-13
updated: 2026-05-20
status: mature
---

# TensorRT-LLM: NVIDIA's Optimized Inference Engine

> [!abstract]+ TL;DR
> NVIDIA's high-performance library purpose-built for LLM inference: a Python model-definition API plus a heavily optimized C++/CUDA runtime, delivering **peak raw kernel performance** on NVIDIA GPUs through graph-level optimizations, operator fusion, and custom CUDA kernels. Core capabilities: native **FP8/NVFP4** quantization, In-Flight Batching, paged KV cache, TP/PP/EP parallelism, and **EAGLE-3 / MTP / N-gram** speculative decoding. The trade-off vs. [[vllm|vLLM]]/[[sglang|SGLang]]: highest single-GPU performance, but NVIDIA-only and high customization cost.

> [!info] Links
> [GitHub](https://github.com/NVIDIA/TensorRT-LLM) · [Docs](https://nvidia.github.io/TensorRT-LLM/)

## Overview

TensorRT-LLM is NVIDIA's high-performance library purpose-built for LLM inference. It combines a Python model-definition API with a heavily optimized C++/CUDA runtime to deliver **peak raw kernel performance** on NVIDIA GPUs.

Positioning:

```
                    Inference engine ecosystem

  Ease of use ◀────────────────────────────────────▶ Peak performance
  
  Hugging Face          vLLM / SGLang         TensorRT-LLM
  Transformers          (open, flexible)      (NVIDIA closed kernels)
  │                     │                     │
  ● Plug-and-play       ● Production-grade    ● Peak performance
  ● Most flexible       ● PagedAttention      ● Graph optimization + compile
  ● Average perf        ● Community-driven    ● Steep learning curve
```

**Key features**:
- Graph optimization and operator fusion (eliminate redundant ops)
- Custom high-performance CUDA kernels (Flash Attention variants, ...)
- Native FP8/NVFP4 quantization support
- In-flight batching (continuous batching)
- Paged KV cache
- Multi-GPU / multi-node tensor, pipeline, and expert parallelism
- Speculative decoding (EAGLE-3, MTP, N-gram)

[GitHub](https://github.com/NVIDIA/TensorRT-LLM) | [Docs](https://nvidia.github.io/TensorRT-LLM/)


## Architecture

TensorRT-LLM operates in two phases: **build** and **runtime**:

```
┌─────────────────────────────────────────────────────────────────┐
│                    TensorRT-LLM architecture                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │ Model weights│     │ Model defn   │     │ Quant config │    │
│  │  (HF format) │     │  (Python)    │     │  (FP8/INT4)  │    │
│  └──────┬───────┘     └──────┬───────┘     └──────┬───────┘    │
│         │                    │                    │             │
│         ▼                    ▼                    ▼             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                     Build phase                          │   │
│  │                                                         │   │
│  │  1. Model conversion: HF → TRT-LLM internal repr        │   │
│  │  2. Graph optimization: op fusion, const folding, memory │   │
│  │  3. Quantization: weight/activation quant (if configured)│   │
│  │  4. Kernel selection: pick optimal CUDA kernel per op   │   │
│  │  5. Compile: emit TensorRT Engine file                  │   │
│  │                                                         │   │
│  │  ⚠ Build time: minutes to hours (depends on model size) │   │
│  │  ⚠ Engine is tied to GPU SKU (H100 ≠ A100)              │   │
│  └────────────────────┬────────────────────────────────────┘   │
│                       │                                        │
│                       ▼                                        │
│              ┌────────────────┐                                 │
│              │  TRT Engine    │  (.engine file)                  │
│              │  (optimized   │  serialized compute graph        │
│              │   compute     │  + selected kernels              │
│              │   graph)       │                                  │
│              └────────┬───────┘                                 │
│                       │                                        │
│                       ▼                                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                     Runtime phase                        │   │
│  │                                                         │   │
│  │  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │   │
│  │  │ In-Flight│  │ KV Cache │  │ Attention│  │ Sampling│  │   │
│  │  │ Batcher │→│ Manager  │→│ Kernels  │→│ + Output│  │   │
│  │  └─────────┘  └──────────┘  └──────────┘  └─────────┘ │   │
│  │                                                         │   │
│  │  C++/CUDA runtime, efficient memory mgmt, dynamic batch │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Build phase deep dive

The build phase is what sets TensorRT-LLM apart from other engines: it compiles the model from a "description" into an "executable optimized compute graph".

**Graph optimization** includes:

```
Before:                          After (operator fusion):
  ┌─────────┐                     ┌──────────────────┐
  │ LayerNorm│                     │                  │
  └────┬─────┘                     │  FusedAttention  │
       ▼                           │  (LN + QKV +     │
  ┌─────────┐                     │   Attention +     │
  │ Q proj  │                     │   Output proj)   │
  └────┬─────┘                     │                  │
       ▼                           └────────┬─────────┘
  ┌─────────┐                              │
  │ K proj  │                    A single kernel does it all
  └────┬─────┘                    → fewer kernel launches
       ▼                          → fewer memory reads/writes
  ┌─────────┐                     → better register utilization
  │ V proj  │
  └────┬─────┘
       ▼
  ┌─────────────┐
  │  Attention  │
  └────┬────────┘
       ▼
  ┌─────────────┐
  │ Output proj │
  └─────────────┘
```

### Build configuration example

```python
# Build an engine using the trtllm-build CLI
# Step 1: convert checkpoint
"""
python convert_checkpoint.py \
    --model_dir /models/Llama-3.3-70B-Instruct \
    --output_dir /checkpoints/llama-70b-fp8 \
    --dtype float16 \
    --tp_size 4 \
    --quant_ckpt_path /calibration/fp8_scales.json
"""

# Step 2: build the TRT engine
"""
trtllm-build \
    --checkpoint_dir /checkpoints/llama-70b-fp8 \
    --output_dir /engines/llama-70b-fp8-tp4 \
    --gemm_plugin float16 \
    --max_batch_size 64 \
    --max_input_len 4096 \
    --max_seq_len 8192 \
    --max_num_tokens 8192 \
    --workers 4 \
    --use_paged_context_fmha enable \
    --multiple_profiles enable
"""
```


## Core optimizations

### Graph optimization and kernel fusion

TensorRT-LLM performs heavy graph-level optimizations at build time:

1. **Operator fusion**: merge adjacent ops into a single CUDA kernel
   - QKV projection fusion: 3 matmuls → 1
   - Attention + softmax + projection fusion
   - LayerNorm + bias fusion
   - GeLU/SiLU activation fused with matmul

2. **Constant folding**: precompute expressions that do not depend on inputs

3. **Memory optimization**:
   - Tensor reuse: immediately release memory of tensors no longer needed
   - Memory layout optimization: pick the GPU-optimal data layout

4. **Kernel auto-tuning (AutoTuner)**:
   - For each op, try multiple kernel implementations and pick the fastest on the target GPU

### In-Flight Batching

Equivalent to [[continuous-batching]] but with NVIDIA's own implementation:

```
Traditional static batching:
  Req A: [████████████──────────]  waits after finishing
  Req B: [██████████████████████]  longest request
  Req C: [██████──────────────── ]  waits after finishing
  ──────────────────────────────→  time
  All wait for the longest, low GPU utilization

In-Flight Batching:
  Req A: [████████████]
  Req B: [██████████████████████]
  Req C: [██████]
  Req D:        [████████████████]   ← joins right after A finishes
  Req E:              [██████████]   ← joins right after C finishes
  ──────────────────────────────→  time
  Dynamic management, GPU stays loaded
```

### KV cache management

TensorRT-LLM uses a paged KV cache (similar to [[paged-attention]]):

- **Block allocation**: KV cache is allocated in fixed-size blocks to avoid fragmentation
- **KV Cache Connector API**: supports cross-request KV reuse (prefix caching) and cross-node KV transfer (for [[prefill-decode-disaggregation]])
- **KV cache quantization**: supports FP8/INT8 KV cache, halving memory

### FP8/INT4 quantization support

TensorRT-LLM offers the most comprehensive quantization support:

| Method | Bits | Hardware | Note |
|----------|------|------|------|
| FP8 | 8-bit | Hopper+ | Weights + activations, nearly lossless |
| NVFP4 | 4-bit | Blackwell | Two-level scaling, native support |
| Block-scale FP8 | 8-bit | Hopper+ | For DeepSeek V3 etc. |
| INT4 AWQ | 4-bit | All | Weight-only quantization |
| INT8 SmoothQuant | 8-bit | All | W8A8 |
| INT4 GPTQ | 4-bit | All | Weight-only quantization |

### Custom CUDA kernels

- **Flash Attention variants**: multiple versions tuned for different sequence lengths and head counts
- **Fused MoE kernel**: efficient mixture-of-experts routing + compute
- **Paged Context FMHA**: Flash Attention supporting paged KV cache
- **FP8 GEMM**: native FP8 matmul kernels


## Comparison with vLLM/SGLang

| Dimension | TensorRT-LLM | vLLM | SGLang |
|------|-------------|------|--------|
| **Developer** | NVIDIA | UC Berkeley + community | LMSYS + community |
| **Open source** | Partial (kernels closed) | Fully open | Fully open |
| **Hardware** | NVIDIA only | NVIDIA, AMD, TPU, CPU | NVIDIA, AMD, TPU |
| **Setup complexity** | High (engine build required) | Low (pip install) | Low (pip install) |
| **Raw kernel performance** | Highest | High | High |
| **Quantization** | Most comprehensive (incl. NVFP4) | Comprehensive | Comprehensive |
| **Model support** | Lags | Broadest | Broad |
| **Prefill throughput** | Highest | High | High |
| **Decode latency** | Lowest | Low | Low |
| **Continuous batching** | In-Flight Batching | Continuous Batching | Continuous Batching |
| **Speculative decoding** | EAGLE-3, MTP, N-gram | EAGLE, Medusa, Draft | EAGLE-3 |
| **Structured output** | Basic | Supported | Best (RadixAttention) |
| **MoE support** | Best (Expert Parallel) | Supported | Supported |
| **Production deployment** | Triton Server integration | Standalone or Ray | Standalone |
| **Community activity** | Moderate | Highest | High |
| **Customization difficulty** | High | Low | Low |

### Selection guide

```
Pick TensorRT-LLM if:
  ✓ Pure NVIDIA environment
  ✓ Chasing the final 10-20% of performance
  ✓ Dedicated MLOps team
  ✓ Fixed model (rarely changed)
  ✓ Need NVFP4 or Block-scale FP8

Pick vLLM if:
  ✓ Multi-hardware support needed
  ✓ Fast prototyping and iteration
  ✓ Community support and docs
  ✓ Wide model coverage

Pick SGLang if:
  ✓ Structured output is critical
  ✓ Complex LLM programs
  ✓ EAGLE-3 integration
  ✓ RadixAttention prefix cache
```

### Performance comparison reference

| Scenario | TensorRT-LLM | vLLM | Gap |
|------|-------------|------|------|
| Llama-70B, FP8, single-request latency | ~35ms/tok | ~42ms/tok | TRT 17% faster |
| Llama-70B, high-concurrency throughput | ~3200 tok/s | ~2800 tok/s | TRT 14% faster |
| Mixtral 8x7B, EP8 | ~4100 tok/s | ~3200 tok/s | TRT 28% faster |

Note: numbers are approximate; the actual gap depends on configuration and workload. As vLLM/SGLang iterate rapidly, the gap keeps shrinking.


## Code examples

### End-to-end build + serve flow

```bash
# ======== Environment prep ========
# Recommended: official NVIDIA Docker image
docker pull nvcr.io/nvidia/tritonserver:24.12-trtllm-python-py3
# or
pip install tensorrt-llm

# ======== Step 1: download model ========
huggingface-cli download meta-llama/Llama-3.3-70B-Instruct \
    --local-dir /models/llama-3.3-70b

# ======== Step 2: convert + quantize (FP8) ========
python3 -m tensorrt_llm.commands.convert_checkpoint \
    --model_dir /models/llama-3.3-70b \
    --output_dir /checkpoints/llama-70b-fp8-tp4 \
    --dtype float16 \
    --tp_size 4 \
    --use_fp8

# ======== Step 3: build TRT engine ========
trtllm-build \
    --checkpoint_dir /checkpoints/llama-70b-fp8-tp4 \
    --output_dir /engines/llama-70b-fp8-tp4 \
    --max_batch_size 64 \
    --max_input_len 4096 \
    --max_seq_len 8192 \
    --gemm_plugin auto \
    --use_paged_context_fmha enable \
    --workers 4

# ======== Step 4: start inference service ========
# Option A: use the built-in HTTP server
python3 -m tensorrt_llm.commands.serve \
    --engine_dir /engines/llama-70b-fp8-tp4 \
    --tokenizer_dir /models/llama-3.3-70b \
    --host 0.0.0.0 \
    --port 8000

# Option B: use Triton Inference Server (recommended for production)
# see Triton integration section below
```

### Python API

```python
import tensorrt_llm
from tensorrt_llm.runtime import ModelRunner

# Load the compiled engine
runner = ModelRunner.from_dir(
    engine_dir="/engines/llama-70b-fp8-tp4",
    rank=tensorrt_llm.mpi_rank(),
)

# Generate
prompts = [
    "Explain quantum computing in simple terms:",
    "Write a Python function to sort a list:",
]

outputs = runner.generate(
    prompts,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    end_id=tokenizer.eos_token_id,
    pad_id=tokenizer.pad_token_id,
)

for i, output in enumerate(outputs):
    print(f"Prompt: {prompts[i]}")
    print(f"Output: {output}")
```

### OpenAI-compatible API

```bash
# TensorRT-LLM supports the OpenAI-compatible API format
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.3-70b",
    "prompt": "San Francisco is a",
    "max_tokens": 100,
    "temperature": 0.7
  }'

# Chat completions
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.3-70b",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 100
  }'
```


## MoE support

TensorRT-LLM provides industry-best support for Mixture-of-Experts models.

### Expert parallelism

```
Tensor Parallelism (TP):           Expert Parallelism (EP):
each GPU holds part of every layer  each GPU holds a subset of experts

  GPU 0: [Expert1_part1, E2_p1, ...]   GPU 0: [Expert 1, 2]
  GPU 1: [Expert1_part2, E2_p2, ...]   GPU 1: [Expert 3, 4]
  GPU 2: [Expert1_part3, E2_p3, ...]   GPU 2: [Expert 5, 6]
  GPU 3: [Expert1_part4, E2_p4, ...]   GPU 3: [Expert 7, 8]

  Needs AllReduce (heavy comm)         Needs AllToAll (route on demand)
  
EP is more efficient for MoE: comm scales with active experts, not total experts
```

### Wide Expert Parallelism

For very large MoE models (e.g. DeepSeek V3 with 256 experts), Wide EP distributes experts across more GPUs:

```python
# Configuring Mixtral 8x7B with Expert Parallelism
"""
python convert_checkpoint.py \
    --model_dir /models/Mixtral-8x7B-Instruct \
    --output_dir /checkpoints/mixtral-ep8 \
    --dtype float16 \
    --tp_size 1 \
    --ep_size 8 \
    --moe_tp_size 1
"""

# DeepSeek V3 (256 experts) config example
"""
python convert_checkpoint.py \
    --model_dir /models/DeepSeek-V3 \
    --output_dir /checkpoints/deepseek-v3-ep16 \
    --dtype float16 \
    --tp_size 1 \
    --ep_size 16 \
    --pp_size 2
"""
```

### Fused MoE kernels

TensorRT-LLM implements efficient fused MoE kernels:

- **Routing + expert compute fusion**: fewer kernel launches and intermediate tensors
- **AutoTuner**: automatically picks optimal block sizes and thread configurations
- **FP8/NVFP4 MoE**: quantized expert computation
- **Load balancing**: optimize token-to-expert assignment, reduce inter-GPU imbalance


## Triton Inference Server integration

In production, TensorRT-LLM is typically deployed through NVIDIA Triton Inference Server.

### Architecture

```
┌────────────────────────────────────────────────────┐
│                Client (HTTP/gRPC)                   │
└───────────────────────┬────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────┐
│              Triton Inference Server                │
│                                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────┐ │
│  │  HTTP/gRPC   │  │  Scheduler   │  │  Model   │ │
│  │  frontend    │→│  (Dynamic    │→│  repo    │ │
│  │              │  │   Batching)  │  │          │ │
│  └──────────────┘  └──────┬───────┘  └──────────┘ │
│                           │                        │
│                           ▼                        │
│  ┌─────────────────────────────────────────────┐   │
│  │          TensorRT-LLM Backend               │   │
│  │                                             │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐   │   │
│  │  │ In-Flight│ │ KV Cache │ │ TRT      │   │   │
│  │  │ Batcher  │ │ Manager  │ │ Engine   │   │   │
│  │  └──────────┘ └──────────┘ └──────────┘   │   │
│  │                                             │   │
│  │  GPU 0    GPU 1    GPU 2    GPU 3          │   │
│  └─────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────┘
```

### Deployment configuration

```bash
# Model repository structure
# model_repository/
# └── llama-70b/
#     ├── config.pbtxt
#     └── 1/
#         └── (engine file link or copy)

# config.pbtxt example
cat > model_repository/llama-70b/config.pbtxt << 'EOF'
name: "llama-70b"
backend: "tensorrtllm"
max_batch_size: 64

model_transaction_policy {
  decoupled: True
}

input [
  {
    name: "text_input"
    data_type: TYPE_STRING
    dims: [ 1 ]
  },
  {
    name: "max_tokens"
    data_type: TYPE_INT32
    dims: [ 1 ]
  },
  {
    name: "temperature"
    data_type: TYPE_FP32
    dims: [ 1 ]
  }
]

output [
  {
    name: "text_output"
    data_type: TYPE_STRING
    dims: [ -1 ]
  }
]

parameters: {
  key: "engine_dir"
  value: {
    string_value: "/engines/llama-70b-fp8-tp4"
  }
}

parameters: {
  key: "tokenizer_dir"
  value: {
    string_value: "/models/llama-3.3-70b"
  }
}

parameters: {
  key: "max_tokens_in_paged_kv_cache"
  value: {
    string_value: "131072"
  }
}

parameters: {
  key: "batch_scheduler_policy"
  value: {
    string_value: "inflight_fused_batching"
  }
}

parameters: {
  key: "kv_cache_free_gpu_mem_fraction"
  value: {
    string_value: "0.85"
  }
}
EOF

# Launch Triton Server
docker run --gpus all -it --rm \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v /engines:/engines \
  -v /models:/models \
  -v $(pwd)/model_repository:/model_repository \
  nvcr.io/nvidia/tritonserver:24.12-trtllm-python-py3 \
  tritonserver --model-repository=/model_repository
```

### Streaming inference client

```python
import tritonclient.grpc as grpcclient
import numpy as np

# Connect to Triton
client = grpcclient.InferenceServerClient(url="localhost:8001")

# Build request
inputs = [
    grpcclient.InferInput("text_input", [1], "BYTES"),
    grpcclient.InferInput("max_tokens", [1], "INT32"),
    grpcclient.InferInput("temperature", [1], "FP32"),
]

prompt = "Explain the theory of relativity:"
inputs[0].set_data_from_numpy(
    np.array([prompt.encode()], dtype=object)
)
inputs[1].set_data_from_numpy(np.array([256], dtype=np.int32))
inputs[2].set_data_from_numpy(np.array([0.7], dtype=np.float32))

# Streaming inference
outputs = [grpcclient.InferRequestedOutput("text_output")]

def callback(result, error):
    if error:
        print(f"Error: {error}")
    else:
        output = result.as_numpy("text_output")
        print(output[0].decode(), end="", flush=True)

client.start_stream(callback=callback)
client.async_stream_infer(
    model_name="llama-70b",
    inputs=inputs,
    outputs=outputs,
)
# Wait for completion
client.stop_stream()
```


## Limitations

1. **Long build time**: engine builds for large models (70B+) can take 30 minutes to several hours. Every model change or parameter tweak (e.g. max_batch_size, max_seq_len) requires a rebuild.

2. **GPU-SKU binding**: a built engine is tied to a specific GPU architecture. An engine built on H100 will not run on A100. Heterogeneous fleets need multiple engines.

3. **NVIDIA only**: no support for AMD, Intel, Apple Silicon, or any non-NVIDIA hardware.

4. **Model support lag**: support for new model architectures (e.g. fresh community releases) typically lags vLLM/SGLang by weeks to months. NVIDIA's team or community contributors have to add support.

5. **High setup complexity**:
   - Many dependencies (CUDA, cuDNN, TensorRT, NCCL — versions must match)
   - Complex build pipeline
   - Hard to debug (C++ runtime error messages are unfriendly)
   - Docker images are recommended

6. **Less flexibility**:
   - Custom model architectures require re-defining in TRT-LLM's Python API
   - Modifying inference logic (e.g. custom sampling) is much harder than in vLLM/SGLang
   - Closed-source kernels cannot be debugged or modified

7. **Insufficient community traction**: compared to vLLM's vibrant open-source community, TRT-LLM's development depends more on NVIDIA internal teams. Bug fixes and feature responses can be slower.

8. **Over-specification risk**: for many scenarios, vLLM/SGLang are already fast enough (10-20% gap), and TRT-LLM's complexity may not be worth it.


## References

- NVIDIA, "TensorRT-LLM: A TensorRT Toolbox for Optimized Large Language Model Inference," 2023-2026. [GitHub](https://github.com/NVIDIA/TensorRT-LLM)
- NVIDIA, "TensorRT-LLM Documentation," [docs](https://nvidia.github.io/TensorRT-LLM/)
- NVIDIA, "Triton Inference Server," [GitHub](https://github.com/triton-inference-server/server)
- NVIDIA, "FP8 Quantization: The Power of the Exponent," 2024. [Blog](https://developer.nvidia.com/blog/nvidia-tensorrt-llm-supercharges-large-language-model-inference-on-nvidia-h100-gpus/)
- NVIDIA, "NVFP4 on Blackwell: Native 4-bit Inference," GTC 2025.
- NVIDIA, "TensorRT-LLM Best Practices Guide," 2024.


## Related pages

- [[vllm]] -- Open-source alternative, broader hardware support, easier to use
- [[sglang]] -- Open-source alternative, better structured-output support
- [[quantization]] -- Quantization method details (GPTQ, AWQ, FP8, ...)
- [[continuous-batching]] -- Theoretical foundation of In-Flight Batching
- [[paged-attention]] -- Principle of paged KV cache
- [[model-parallelism]] -- Tensor parallelism and pipeline parallelism strategies
- [[speculative-decoding]] -- Speculative decoding theory (EAGLE-3, ...)
- [[kv-cache-optimization]] -- KV cache management and optimization
- [[prefill-decode-disaggregation]] -- TRT-LLM's disaggregated serving
