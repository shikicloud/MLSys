---
title: "TensorRT-LLM：NVIDIA 优化推理引擎"
category: llm-inference
tags: [tensorrt-llm, nvidia, 推理优化, 编译, 量化, inflight-batching]
created: 2026-04-13
updated: 2026-05-07
status: mature
---

# TensorRT-LLM：NVIDIA 优化推理引擎

> [!abstract]+ TL;DR
> NVIDIA 专为 LLM 推理构建的高性能库：Python 模型定义 API + 高度优化的 C++/CUDA 运行时，通过图级优化、算子融合和自定义 CUDA 内核在 NVIDIA GPU 上实现**最高原始内核性能**。核心能力：原生 **FP8/NVFP4** 量化、In-Flight Batching、分页 KV 缓存、TP/PP/EP 并行、**EAGLE-3 / MTP / N-gram** 投机解码。与 [[vllm|vLLM]]/[[sglang|SGLang]] 的权衡：单 GPU 性能最高，但仅限 NVIDIA、定制成本高。

> [!info] 相关链接
> [GitHub](https://github.com/NVIDIA/TensorRT-LLM) · [文档](https://nvidia.github.io/TensorRT-LLM/)

## 概述

TensorRT-LLM 是 NVIDIA 专为大语言模型推理构建的高性能库。它将 LLM 的模型定义（Python API）与高度优化的 C++/CUDA 运行时结合，在 NVIDIA GPU 上实现 **最高原始内核性能**。

核心定位：

```
                    推理引擎生态定位

  易用性 ◀────────────────────────────────────▶ 极致性能
  
  Hugging Face          vLLM / SGLang         TensorRT-LLM
  Transformers          (开源, 灵活)           (NVIDIA 闭源内核)
  │                     │                     │
  ● 开箱即用             ● 生产级              ● 最高性能
  ● 最灵活              ● PagedAttention       ● 图优化 + 编译
  ● 性能一般             ● 社区驱动             ● 学习曲线陡峭
```

**关键特性**：
- 图优化与算子融合（消除冗余操作）
- 自定义高性能 CUDA 内核（Flash Attention 变体等）
- 原生 FP8/NVFP4 量化支持
- In-flight batching（连续批处理）
- 分页 KV 缓存
- 多 GPU/多节点张量并行、流水线并行、专家并行
- 投机解码（EAGLE-3、MTP、N-gram）

[GitHub](https://github.com/NVIDIA/TensorRT-LLM) | [文档](https://nvidia.github.io/TensorRT-LLM/)


## 架构

TensorRT-LLM 的工作流程分为 **构建阶段（Build）** 和 **运行阶段（Runtime）**：

```
┌─────────────────────────────────────────────────────────────────┐
│                    TensorRT-LLM 架构                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │  模型权重     │     │  模型定义     │     │  量化配置     │    │
│  │  (HF 格式)   │     │  (Python)    │     │  (FP8/INT4)  │    │
│  └──────┬───────┘     └──────┬───────┘     └──────┬───────┘    │
│         │                    │                    │             │
│         ▼                    ▼                    ▼             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  构建阶段 (Build Phase)                   │   │
│  │                                                         │   │
│  │  1. 模型转换: HF → TRT-LLM 内部表示                      │   │
│  │  2. 图优化: 算子融合, 常量折叠, 内存优化                    │   │
│  │  3. 量化: 权重/激活量化 (如果配置)                         │   │
│  │  4. 内核选择: 为每个算子选择最优 CUDA 内核                  │   │
│  │  5. 编译: 生成 TensorRT Engine 文件                      │   │
│  │                                                         │   │
│  │  ⚠ 构建耗时: 几分钟到几小时 (取决于模型大小)               │   │
│  │  ⚠ 构建结果与 GPU 型号绑定 (H100 ≠ A100)                │   │
│  └────────────────────┬────────────────────────────────────┘   │
│                       │                                        │
│                       ▼                                        │
│              ┌────────────────┐                                 │
│              │  TRT Engine    │  (.engine 文件)                  │
│              │  (优化后的      │  序列化的计算图                   │
│              │   计算图)       │  + 选定的内核                    │
│              └────────┬───────┘                                 │
│                       │                                        │
│                       ▼                                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  运行阶段 (Runtime Phase)                  │   │
│  │                                                         │   │
│  │  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │   │
│  │  │ In-Flight│  │ KV Cache │  │ Attention│  │ Sampling│  │   │
│  │  │ Batcher │→│ Manager  │→│ Kernels  │→│ + Output│  │   │
│  │  └─────────┘  └──────────┘  └──────────┘  └─────────┘ │   │
│  │                                                         │   │
│  │  C++/CUDA 运行时, 高效内存管理, 动态批处理                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 构建阶段详解

构建阶段是 TensorRT-LLM 区别于其他引擎的核心：它将模型从"描述"编译为"可执行的优化计算图"。

**图优化**包括：

```
优化前:                          优化后 (算子融合):
  ┌─────────┐                     ┌──────────────────┐
  │ LayerNorm│                     │                  │
  └────┬─────┘                     │  FusedAttention  │
       ▼                           │  (LN + QKV +     │
  ┌─────────┐                     │   Attention +     │
  │ Q proj  │                     │   Output proj)   │
  └────┬─────┘                     │                  │
       ▼                           └────────┬─────────┘
  ┌─────────┐                              │
  │ K proj  │                    单个内核完成所有操作
  └────┬─────┘                    → 减少内核启动开销
       ▼                          → 减少显存读写
  ┌─────────┐                     → 更好的寄存器利用
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

### 构建配置示例

```python
# 使用 trtllm-build CLI 构建引擎
# 步骤 1: 转换 checkpoint
"""
python convert_checkpoint.py \
    --model_dir /models/Llama-3.3-70B-Instruct \
    --output_dir /checkpoints/llama-70b-fp8 \
    --dtype float16 \
    --tp_size 4 \
    --quant_ckpt_path /calibration/fp8_scales.json
"""

# 步骤 2: 构建 TRT Engine
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


## 核心优化

### 图优化与内核融合

TensorRT-LLM 在构建阶段执行大量图级优化：

1. **算子融合**：将多个相邻算子合并为单个 CUDA 内核
   - QKV 投影融合：3 个矩阵乘法 → 1 个
   - Attention + Softmax + 投影融合
   - LayerNorm + 偏置融合
   - GeLU/SiLU 激活与矩阵乘法融合

2. **常量折叠**：预计算不依赖输入的表达式

3. **内存优化**：
   - 张量重用：不再需要的张量的内存立即释放
   - 内存布局优化：选择对 GPU 最优的数据排列方式

4. **内核自动调优（AutoTuner）**：
   - 对每个算子，尝试多种内核实现，选择在目标 GPU 上最快的

### In-Flight Batching

等价于 [[continuous-batching]]，但使用 NVIDIA 自己的实现：

```
传统静态 batch:
  请求 A: [████████████──────────]  完成后等待
  请求 B: [██████████████████████]  最长请求
  请求 C: [██████──────────────── ]  完成后等待
  ──────────────────────────────→  时间
  所有请求等最长的完成, GPU 利用率低

In-Flight Batching:
  请求 A: [████████████]
  请求 B: [██████████████████████]
  请求 C: [██████]
  请求 D:        [████████████████]   ← A 完成后立即加入
  请求 E:              [██████████]   ← C 完成后立即加入
  ──────────────────────────────→  时间
  动态管理, GPU 持续满载
```

### KV 缓存管理

TensorRT-LLM 使用分页 KV 缓存（类似 [[paged-attention]]）：

- **块分配**：KV 缓存按固定大小的块分配，避免碎片化
- **KV Cache Connector API**：支持 KV 缓存的跨请求复用（前缀缓存）和跨节点传输（用于 [[prefill-decode-disaggregation]]）
- **KV 缓存量化**：支持 FP8/INT8 KV 缓存，内存减半

### FP8/INT4 量化支持

TensorRT-LLM 提供最全面的量化支持：

| 量化方法 | 位宽 | 硬件 | 说明 |
|----------|------|------|------|
| FP8 | 8-bit | Hopper+ | 权重 + 激活，近乎无损 |
| NVFP4 | 4-bit | Blackwell | 两级缩放，原生支持 |
| Block-scale FP8 | 8-bit | Hopper+ | 用于 DeepSeek V3 等 |
| INT4 AWQ | 4-bit | 所有 | 仅权重量化 |
| INT8 SmoothQuant | 8-bit | 所有 | W8A8 |
| INT4 GPTQ | 4-bit | 所有 | 仅权重量化 |

### 自定义 CUDA 内核

- **Flash Attention 变体**：针对不同序列长度和 head 数量优化的多个版本
- **Fused MoE 内核**：高效的混合专家路由 + 计算
- **Paged Context FMHA**：支持分页 KV 缓存的 Flash Attention
- **FP8 GEMM**：原生 FP8 矩阵乘法内核


## 与 vLLM/SGLang 对比

| 维度 | TensorRT-LLM | vLLM | SGLang |
|------|-------------|------|--------|
| **开发者** | NVIDIA | UC Berkeley + 社区 | LMSYS + 社区 |
| **开源** | 部分（内核闭源） | 完全开源 | 完全开源 |
| **硬件支持** | 仅 NVIDIA | NVIDIA, AMD, TPU, CPU | NVIDIA, AMD, TPU |
| **设置复杂度** | 高（需编译引擎） | 低（pip install） | 低（pip install） |
| **原始内核性能** | 最高 | 高 | 高 |
| **量化支持** | 最全面（含 NVFP4） | 全面 | 全面 |
| **模型支持** | 有滞后 | 最广 | 广 |
| **Prefill 吞吐** | 最高 | 高 | 高 |
| **Decode 延迟** | 最低 | 低 | 低 |
| **连续批处理** | In-Flight Batching | Continuous Batching | Continuous Batching |
| **投机解码** | EAGLE-3, MTP, N-gram | EAGLE, Medusa, Draft | EAGLE-3 |
| **结构化输出** | 基础支持 | 支持 | 最佳（RadixAttention） |
| **MoE 支持** | 最优（Expert Parallel） | 支持 | 支持 |
| **生产部署** | Triton Server 集成 | 独立或 Ray | 独立 |
| **社区活跃度** | 中 | 最高 | 高 |
| **自定义难度** | 高 | 低 | 低 |

### 选择建议

```
选 TensorRT-LLM:
  ✓ 纯 NVIDIA 环境
  ✓ 追求极致性能（最后 10-20% 的优化）
  ✓ 有专门的 MLOps 团队
  ✓ 模型固定（不频繁切换）
  ✓ 需要 NVFP4 或 Block-scale FP8

选 vLLM:
  ✓ 多硬件支持需求
  ✓ 快速原型和迭代
  ✓ 社区支持和文档
  ✓ 模型支持广度

选 SGLang:
  ✓ 结构化输出重要
  ✓ 复杂的 LLM 程序
  ✓ EAGLE-3 集成
  ✓ RadixAttention 前缀缓存
```

### 性能对比参考

| 场景 | TensorRT-LLM | vLLM | 差距 |
|------|-------------|------|------|
| Llama-70B, FP8, 单请求延迟 | ~35ms/tok | ~42ms/tok | TRT 快 17% |
| Llama-70B, 高并发吞吐 | ~3200 tok/s | ~2800 tok/s | TRT 快 14% |
| Mixtral 8x7B, EP8 | ~4100 tok/s | ~3200 tok/s | TRT 快 28% |

注：性能数据为近似值，实际差距取决于具体配置和工作负载。随着 vLLM/SGLang 的快速迭代，差距持续缩小。


## 代码示例

### 完整的构建 + 服务流程

```bash
# ======== 环境准备 ========
# 推荐使用 NVIDIA 官方 Docker 镜像
docker pull nvcr.io/nvidia/tritonserver:24.12-trtllm-python-py3
# 或
pip install tensorrt-llm

# ======== 步骤 1: 下载模型 ========
huggingface-cli download meta-llama/Llama-3.3-70B-Instruct \
    --local-dir /models/llama-3.3-70b

# ======== 步骤 2: 转换 + 量化 (FP8) ========
python3 -m tensorrt_llm.commands.convert_checkpoint \
    --model_dir /models/llama-3.3-70b \
    --output_dir /checkpoints/llama-70b-fp8-tp4 \
    --dtype float16 \
    --tp_size 4 \
    --use_fp8

# ======== 步骤 3: 构建 TRT Engine ========
trtllm-build \
    --checkpoint_dir /checkpoints/llama-70b-fp8-tp4 \
    --output_dir /engines/llama-70b-fp8-tp4 \
    --max_batch_size 64 \
    --max_input_len 4096 \
    --max_seq_len 8192 \
    --gemm_plugin auto \
    --use_paged_context_fmha enable \
    --workers 4

# ======== 步骤 4: 启动推理服务 ========
# 方式 A: 使用内置 HTTP 服务器
python3 -m tensorrt_llm.commands.serve \
    --engine_dir /engines/llama-70b-fp8-tp4 \
    --tokenizer_dir /models/llama-3.3-70b \
    --host 0.0.0.0 \
    --port 8000

# 方式 B: 使用 Triton Inference Server (生产推荐)
# 见下方 Triton 集成章节
```

### Python API 使用

```python
import tensorrt_llm
from tensorrt_llm.runtime import ModelRunner

# 加载编译好的引擎
runner = ModelRunner.from_dir(
    engine_dir="/engines/llama-70b-fp8-tp4",
    rank=tensorrt_llm.mpi_rank(),
)

# 生成
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

### OpenAI 兼容 API

```bash
# TensorRT-LLM 支持 OpenAI 兼容的 API 格式
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


## MoE 支持

TensorRT-LLM 对混合专家（Mixture of Experts）模型提供了业界最优的支持。

### 专家并行（Expert Parallelism）

```
Tensor Parallelism (TP):           Expert Parallelism (EP):
每个 GPU 持有每层的一部分            每个 GPU 持有部分专家

  GPU 0: [Expert1_part1, E2_p1, ...]   GPU 0: [Expert 1, 2]
  GPU 1: [Expert1_part2, E2_p2, ...]   GPU 1: [Expert 3, 4]
  GPU 2: [Expert1_part3, E2_p3, ...]   GPU 2: [Expert 5, 6]
  GPU 3: [Expert1_part4, E2_p4, ...]   GPU 3: [Expert 7, 8]

  需要 AllReduce (高通信量)           需要 AllToAll (按需路由)
  
EP 对 MoE 更高效: 通信量与激活的专家数成正比, 而非总专家数
```

### Wide Expert Parallelism

对于超大 MoE 模型（如 DeepSeek V3 的 256 专家），Wide EP 将专家分布到更多 GPU：

```python
# 配置 Mixtral 8x7B with Expert Parallelism
"""
python convert_checkpoint.py \
    --model_dir /models/Mixtral-8x7B-Instruct \
    --output_dir /checkpoints/mixtral-ep8 \
    --dtype float16 \
    --tp_size 1 \
    --ep_size 8 \
    --moe_tp_size 1
"""

# DeepSeek V3 (256 experts) 配置示例
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

### Fused MoE 内核

TensorRT-LLM 实现了高效的 MoE 融合内核：

- **路由 + 专家计算融合**：减少内核启动和中间张量
- **AutoTuner**：自动选择最优的分块大小和线程配置
- **FP8/NVFP4 MoE**：量化的专家计算
- **负载均衡**：优化 token 到专家的分配，减少 GPU 间不均衡


## Triton Inference Server 集成

在生产环境中，TensorRT-LLM 通常通过 NVIDIA Triton Inference Server 部署。

### 架构

```
┌────────────────────────────────────────────────────┐
│                 客户端 (HTTP/gRPC)                   │
└───────────────────────┬────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────┐
│              Triton Inference Server                │
│                                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────┐ │
│  │  HTTP/gRPC   │  │  调度器      │  │  模型     │ │
│  │  前端        │→│  (Dynamic    │→│  仓库     │ │
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

### 配置部署

```bash
# 模型仓库结构
# model_repository/
# └── llama-70b/
#     ├── config.pbtxt
#     └── 1/
#         └── (引擎文件链接或复制)

# config.pbtxt 示例
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

# 启动 Triton Server
docker run --gpus all -it --rm \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v /engines:/engines \
  -v /models:/models \
  -v $(pwd)/model_repository:/model_repository \
  nvcr.io/nvidia/tritonserver:24.12-trtllm-python-py3 \
  tritonserver --model-repository=/model_repository
```

### 流式推理客户端

```python
import tritonclient.grpc as grpcclient
import numpy as np

# 连接 Triton
client = grpcclient.InferenceServerClient(url="localhost:8001")

# 构建请求
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

# 流式推理
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
# 等待完成
client.stop_stream()
```


## 不足

1. **构建时间长**：大模型（70B+）的引擎构建可能需要 30 分钟到几小时。每次更换模型、修改参数（如 max_batch_size、max_seq_len）都需要重新构建。

2. **GPU 型号绑定**：构建的引擎与特定 GPU 架构绑定。H100 上构建的引擎不能在 A100 上运行。部署多种 GPU 时需要维护多份引擎。

3. **仅支持 NVIDIA**：完全不支持 AMD、Intel、Apple Silicon 或任何非 NVIDIA 硬件。

4. **模型支持滞后**：新模型架构（如社区新发布的模型）的支持通常晚于 vLLM/SGLang 数周到数月。需要 NVIDIA 团队或社区贡献者添加支持。

5. **设置复杂度高**：
   - 安装依赖多（CUDA、cuDNN、TensorRT、NCCL 等版本需匹配）
   - 编译流程复杂
   - 调试困难（C++ 运行时错误信息不友好）
   - 推荐使用 Docker 镜像

6. **灵活性较低**：
   - 自定义模型架构需要用 TRT-LLM 的 Python API 重新定义
   - 修改推理逻辑（如自定义采样策略）比 vLLM/SGLang 困难得多
   - 闭源内核无法调试或修改

7. **社区驱动不足**：相比 vLLM 的活跃开源社区，TRT-LLM 的发展更依赖 NVIDIA 内部团队。Bug 修复和功能响应可能较慢。

8. **过度规格化风险**：对于许多场景，vLLM/SGLang 的性能已经足够（差距在 10-20%），而 TRT-LLM 的复杂度可能不值得。


## 参考文献

- NVIDIA, "TensorRT-LLM: A TensorRT Toolbox for Optimized Large Language Model Inference," 2023-2026. [GitHub](https://github.com/NVIDIA/TensorRT-LLM)
- NVIDIA, "TensorRT-LLM Documentation," [docs](https://nvidia.github.io/TensorRT-LLM/)
- NVIDIA, "Triton Inference Server," [GitHub](https://github.com/triton-inference-server/server)
- NVIDIA, "FP8 Quantization: The Power of the Exponent," 2024. [Blog](https://developer.nvidia.com/blog/nvidia-tensorrt-llm-supercharges-large-language-model-inference-on-nvidia-h100-gpus/)
- NVIDIA, "NVFP4 on Blackwell: Native 4-bit Inference," GTC 2025.
- NVIDIA, "TensorRT-LLM Best Practices Guide," 2024.


## 相关页面

- [[vllm]] -- 开源替代，更广硬件支持，更易用
- [[sglang]] -- 开源替代，更好的结构化输出支持
- [[quantization]] -- 量化方法详情（GPTQ、AWQ、FP8 等）
- [[continuous-batching]] -- In-Flight Batching 的理论基础
- [[paged-attention]] -- 分页 KV 缓存的原理
- [[model-parallelism]] -- 张量并行、流水线并行策略
- [[speculative-decoding]] -- 投机解码原理（EAGLE-3 等）
- [[kv-cache-optimization]] -- KV 缓存管理与优化
- [[prefill-decode-disaggregation]] -- TRT-LLM 的 disaggregated serving
