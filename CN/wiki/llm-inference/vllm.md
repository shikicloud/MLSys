---
title: "vLLM：高吞吐量 LLM 服务引擎"
category: llm-inference
tags: [vllm, 服务, paged-attention, 推理引擎, v1架构, continuous-batching]
created: 2026-04-13
updated: 2026-04-14
status: mature
---

# vLLM：高吞吐量 LLM 服务引擎

> [!abstract]+ TL;DR
> vLLM 是开源、高吞吐量、内存高效的 LLM 推理与服务引擎，由 UC Berkeley Sky Computing Lab 于 2023 年推出。核心创新是 [[paged-attention|PagedAttention]] —— 借鉴操作系统虚拟内存分页思想管理 KV 缓存，把碎片化降到接近零。截至 2026 年 4 月（v0.19.0）已是最广泛部署的服务框架之一（50k+ GitHub Star），硬件支持最广（NVIDIA / AMD / TPU / Gaudi / Trainium），V1 重写带来**最高 1.7× 吞吐量**提升。

> [!info] 相关链接
> [GitHub](https://github.com/vllm-project/vllm) · [官方文档](https://docs.vllm.ai/) · [vLLM Blog](https://vllm.ai/blog)

**核心竞争力**：

- **极致的内存效率**：通过 PagedAttention 把 KV 缓存浪费降到接近零
- **高吞吐量**：[[continuous-batching|连续批处理]] + 前缀缓存 + 分块预填充
- **最广硬件支持**：NVIDIA（Ampere/Hopper/Blackwell）、AMD、Intel Gaudi、Google TPU、AWS Trainium
- **OpenAI 兼容 API**：可直接替换 OpenAI 后端
- **丰富模型生态**：Transformer、MoE、多模态、多 LoRA
- **活跃社区**：两周一版本，最新 v0.19.0（2026 年 4 月）

---

## 架构演进

### V0 架构（2023-2024）

V0 的核心问题是 **Scheduler 与 Worker 0 共享进程**，导致非对称架构、CPU 任务阻塞 GPU、全量状态传输、Prefill/Decode 调度路径分离、扩展性受限。

### V1 架构（v0.8.0 起默认，2025 年 1 月）

V1 是对引擎的从零重写（ground-up rewrite），目标是解决 V0 的所有架构问题。V1 相比 V0 吞吐量提升高达 **1.7 倍**，同时代码更简洁、更易扩展。

**V0 → V1 核心变化总结**：

| 维度 | V0 | V1 |
|------|----|----|
| Worker 架构 | 非对称（W0 特殊） | 对称（所有 Worker 相同） |
| 调度器 | Prefill/Decode 分离 | 统一调度 `{req_id: n_tokens}` |
| 状态传输 | 全量传输 | 增量差异传输（diffs） |
| 批处理 | 每步重建输入 | 持久化批处理（缓存+差异） |
| CPU 任务 | 与调度器同进程 | 隔离到独立进程 |
| 进程间通信 | 直接调用 | ZeroMQ IPC |
| CUDA Graph | 传统全图捕获 | 分段 CUDA Graph（Piecewise） |

---

## V1 架构详解

### 整体架构图

```
┌── API Server 进程 ──────────────────────────────────────┐
│ HTTP → OpenAI API → Tokenization → AsyncLLM             │
│                        │ ZMQ IPC ▲                      │
│  ┌─ EngineCore 进程 ───▼─────────┴──────────────────┐   │
│  │  Input Queue → Scheduler{req_id:n} → KV Cache Mgr│   │
│  │                    → MultiProcExecutor            │   │
│  └────────────────────────┬──────────────────────────┘   │
│ Detokenization ◀── Output Queue ◀── Results              │
└───────────────────────────┼──────────────────────────────┘
       共享内存              │ rpc_broadcast_mq
    ┌───────────┬───────────┼───────────┐
    ▼           ▼           ▼           ▼
 Worker 0   Worker 1    Worker 2    Worker N
 GPU 0      GPU 1       GPU 2       GPU N
 ModelRunner + KV Cache(PagedAtt) + 本地状态缓存
    └─────── NCCL AllReduce ────────────┘
```

### EngineCore 隔离

EngineCore 在独立进程中运行，通过 ZeroMQ 与 API Server 进程通信。这种设计的关键优势：

1. **绕过 GIL**：Python 的全局解释器锁（GIL）不再成为瓶颈。CPU 密集型任务（tokenization、多模态数据处理、detokenization、响应流式传输）在 API Server 进程中执行，与 GPU 执行完全并行
2. **调度不阻塞**：EngineCore 运行一个忙循环（busy loop），持续执行调度和模型前向传播，GPU 永远不会因为 CPU 任务而空闲
3. **清晰的关注点分离**：API 层处理 HTTP、认证、格式转换；EngineCore 专注于调度和模型执行

```python
# EngineCore 核心循环（简化）
while True:
    new_reqs = input_queue.get_nowait()     # 1. 拉取新请求
    scheduler.add_requests(new_reqs)
    schedule = scheduler.schedule()          # 2. {req_id: num_tokens}
    output = executor.execute_model(schedule)# 3. GPU 前向传播
    output_queue.put(output)                 # 4. 结果入队
```

### 统一调度器（Unified Scheduler）

V1 调度器的核心创新是将所有调度决策统一为一个简单的字典：

```
{request_id: num_tokens}
```

这个抽象统一了以下所有场景：

| 场景 | 调度表示 | 说明 |
|------|----------|------|
| 常规 Prefill | `{req_1: 512}` | 一次性处理所有 prompt token |
| 分块 Prefill | `{req_1: 256}` | Prompt 太长，分块处理 |
| 常规 Decode | `{req_2: 1}` | 逐 token 自回归生成 |
| 投机解码 | `{req_3: 5}` | Draft 模型提议 5 个 token |
| 前缀缓存命中 | `{req_4: 128}` | 跳过已缓存的前缀，只处理剩余部分 |

调度流程：(1) 优先为所有 running 请求分配 decode token → (2) 计算剩余预算 → (3) 从 waiting 队列取 prefill 请求（检查前缀缓存、必要时分块、分配 KV blocks）→ (4) 输出 `{req_id: num_tokens}`。

### 持久化批处理（Persistent Batch）

V0 中每步都需要重建完整的输入张量，浪费 CPU 时间。V1 引入持久化批处理：

```
步骤 t:    batch = [req_1, req_2, req_3, req_4]
                      ↓ req_2 完成，req_5 到达
步骤 t+1:  batch = [req_1, -----, req_3, req_4, req_5]
                           只需更新 diff!

具体操作：
  - 删除 req_2 的位置（标记为可用）
  - 将 req_5 的 token 放入空位
  - 其他位置的缓存张量保持不变
```

使用 NumPy 操作替代 Python 原生操作来高效应用差异（diffs），显著降低每步的 CPU 开销。

### 对称 Worker 架构

V1 中所有 Worker 完全相同（V0 中 Worker 0 有特殊职责）。每个 Worker 本地缓存请求状态，仅接收增量更新（新请求/完成/抢占），通过共享内存 `rpc_broadcast_mq` 接收指令，`worker_response_mq` 返回结果。

### 请求生命周期

```
客户端 POST /v1/chat/completions
  → API Server (HTTP 验证) → AsyncLLM (Tokenization)
  → [ZMQ IPC] → EngineCore Input Queue
  → Scheduler ({req_id: num_tokens} + KV block 分配)
  → MultiProcExecutor → Workers (前向传播 + Sampling)
  → [ZMQ IPC] → AsyncLLM (Detokenization)
  → API Server → SSE Streaming 响应
```

---

## 核心技术栈

### PagedAttention 集成

vLLM 的基石是 [[paged-attention|PagedAttention]]（Kwon et al., SOSP 2023），借鉴了操作系统虚拟内存分页机制：

```
传统分配：每请求预分配最大长度 → 平均浪费 ~40%
PagedAttention：物理块池 [B0][B1][B2]... + 页表映射
  Req1: B0→B3→B5 | Req2: B1→B4 | Req3: B2→B6→B7
  → 仅最后 block 有碎片 (<4% 浪费)
```

关键参数：
- `block_size`：每个物理块存储的 token 数量（默认 16）
- `gpu_memory_utilization`：GPU 内存用于 KV cache 的比例（默认 0.9）
- KV cache 内存 = 总 GPU 内存 × `gpu_memory_utilization` - 模型权重 - 激活值

### 前缀缓存（Automatic Prefix Caching）

vLLM 的前缀缓存在 V1 中默认开启，采用基于哈希的块级缓存机制。核心思想：多个请求如果共享相同的前缀（如系统提示词），可以复用已计算的 KV cache。

**哈希计算**：

```python
# 每个 KV block 的哈希由以下元素组成
block_hash = hash(
    parent_block_hash,     # 父块的哈希（链式依赖）
    tuple(block_tokens),   # 该块中的 token 序列
    extra_hashes           # LoRA ID / 多模态输入哈希 / cache salt
)
```

**BlockPool 数据结构**：

```
BlockPool
├── blocks[]: KVCacheBlock 对象数组
│     每个 block 包含:
│     - block_id: 物理块 ID
│     - ref_count: 引用计数（当前使用该块的请求数）
│     - last_access_time: 最后访问时间
│     - block_hash: 内容哈希
│
├── free_block_queue: 双向链表（LRU 顺序）
│     空闲块按最近使用时间排列
│
└── cached_block_map: {hash → KVCacheBlock}
      全局哈希表，用于快速查找已缓存的块
```

**LRU 驱逐策略**：

```
驱逐优先级（从高到低）：
1. ref_count == 0（没有请求在使用）
2. 在 ref_count == 0 的块中，优先驱逐最久未使用的（LRU）
3. 如果 last_access_time 相同，优先驱逐在最长前缀末尾的块
   （即：先驱逐后面的块，保留前面的块）
```

**可配置哈希算法**：

| 算法 | 序列化方式 | 特点 |
|------|-----------|------|
| `sha256`（默认） | Python pickle | 安全、通用 |
| `sha256_cbor` | cbor2 | 可复现哈希 |
| `xxhash` | pickle + xxHash 128-bit | 更快的非加密哈希 |

**性能特征**：
- 0% 命中率时：额外开销 < 1%（几乎免费）
- 高命中率时：乘法级别的吞吐量提升（跳过已缓存前缀的计算）
- 典型场景（多轮对话、共享系统提示词）：命中率 60-90%

### 连续批处理（Continuous Batching）

vLLM 实现了 [[continuous-batching|连续批处理]]（也称为 iteration-level scheduling），核心机制：

静态批处理中，完成的请求留下空槽直到整个批次结束。连续批处理立即用新请求填充空槽（如 Req2 完成后立刻调度 Req4），确保 GPU 始终满载。

### 分块预填充（Chunked Prefill）

分块预填充是 V1 中默认启用的优化，将长 prompt 拆分为多个小块，与 decode 请求交错执行：

```
无 Chunked Prefill：Step1: Prefill_A(4096) | Step2: Decode_B,C,D  ← 长prefill阻塞decode
有 Chunked Prefill：Step1: Prefill_A_chunk1(512)+Decode_B,C | Step2: chunk2+Decode_B,C | ...
→ Prefill/Decode 交错，decode 延迟稳定
```

**核心原理**：Prefill 是计算密集型（compute-bound），Decode 是内存密集型（memory-bound）。两者混合批处理可以同时利用 GPU 的计算和内存带宽资源。调度器先分配所有 decode 请求（各 1 token），剩余 `max_num_batched_tokens` 预算分给 prefill 分块。

### CUDA Graph 集成

V1 采用 **分段 CUDA Graph**（Piecewise）：在 Attention 操作处分割计算图，非 Attention 部分捕获为 CUDA Graph，Attention 以 Eager 模式执行（FlashAttention 3）。模式：`FULL_AND_PIECEWISE`（默认，最优性能，最多内存）、`PIECEWISE`、`NONE`。

---

## 并行策略

vLLM 支持多种并行策略，可灵活组合以适应不同模型规模和硬件配置。详细的并行策略原理见 [[parallelism-strategies-deep-dive]]。

### 支持的并行方式

| 策略 | 原理 | 通信 | 效果 |
|------|------|------|------|
| **TP** | 每层切分到多 GPU | AllReduce | 降低单请求延迟 |
| **PP** | 按层分配到不同 GPU | 流水线 | 支持跨节点、非 2^n GPU |
| **DP** | 独立副本处理不同请求 | 无 | 提升并发吞吐量 |
| **EP** | Expert 分布到不同 GPU（仅 MoE） | AllToAll/AllReduce | 必须与 TP 或 DP 组合 |

### DP Attention + EP（面向 MoE 的核心策略）

DP Attention 是专门为 MoE 模型设计的数据并行变体，与传统数据并行有本质区别：

```
传统 DP：每 GPU 一个完整副本（KV Cache 完整复制）
DP Attention + EP：单逻辑副本，Attention 独立 + KV Cache 按请求分区
┌──────────┐  ┌──────────┐  ┌──────────┐
│  GPU 0    │  │  GPU 1    │  │  GPU 2    │
│ Attn(独立)│  │ Attn(独立)│  │ Attn(独立)│
│ KV(分区)  │  │ KV(分区)  │  │ KV(分区)  │
│ Expert0-2 │  │ Expert3-5 │  │ Expert6-8 │
└─────┬─────┘  └─────┬─────┘  └─────┬─────┘
      └──── AllToAll (Expert 通信) ──┘
```

**为什么 MoE 模型需要 DP Attention？**

DeepSeek-V3/R1 使用 Multi-Head Latent Attention（MLA），只有单个 KV head。在传统 TP 中，KV cache 无法按 head 维度切分，必须在所有 TP rank 上完整复制。DP Attention 通过将 KV cache 按请求分区来解决此问题。

**Expert 分布方式**：

| 方式 | 说明 | 通信 |
|------|------|------|
| Sharded Expert（无 EP flag） | 所有 Expert 都在每个 GPU 上，但权重被切分 | AllReduce |
| Split Expert（`--enable-expert-parallel`） | 每个 GPU 持有不同 Expert 的完整权重 | AllToAll（DP>1） |

公式：每 GPU Expert 数 = 总 Expert 数 / (TP_SIZE x DP_SIZE)

### 配置示例

**基本 Tensor Parallel（单节点 4 GPU）**：

```bash
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4
```

**Pipeline + Tensor Parallel（跨 2 节点，每节点 4 GPU）**：

```bash
vllm serve meta-llama/Llama-3.1-405B-Instruct \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 2
```

**DeepSeek-R1 低并发场景（TP+EP，8 GPU）**：

```bash
# 适用于 ≤128 并发请求
# 比 DP 变体吞吐量高 52%，TTFT 低 80%
vllm serve deepseek-ai/DeepSeek-R1 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel
```

**DeepSeek-R1 高并发场景（DP+EP，8 GPU）**：

```bash
# 适用于 ≥512 并发请求
# 高并发下吞吐量高 47%
vllm serve deepseek-ai/DeepSeek-R1 \
    --tensor-parallel-size 1 \
    --data-parallel-size 8 \
    --enable-expert-parallel
```

### 并行策略选择指南

- **Dense 模型**：单 GPU 能装 → 单 GPU；否则 → TP+PP
- **MoE 模型（Expert 激活 >3%）**：低并发(<=128) → TP+EP；高并发(>=512) → DP+EP
- **MoE 模型（Expert 激活 <1%）**：不用 EP（AllToAll 开销超过收益）
- **MLA 模型（DeepSeek）**：必须 DP+EP（KV cache 无法按 head 切分）

---

## 投机解码支持

vLLM 支持多种 [[speculative-decoding|投机解码]] 方法，通过"先猜后验"加速自回归生成。详细原理见 [[speculative-decoding]]。

### 支持的方法

| 方法 | 描述 | 加速比 | 适用场景 |
|------|------|--------|----------|
| Draft Model | 小模型生成候选 token，大模型验证 | 1.5-2.5x | 有对应小模型时 |
| EAGLE-1/3 | 基于特征外推的轻量级 draft head | 2-3x | 推荐，准确率高 |
| Medusa | 多头并行预测 | 1.5-2x | 无需额外模型 |
| N-gram | 基于输入 n-gram 匹配预测 | 1.2-1.5x | 翻译/总结等重复性任务 |
| MLP Speculator | 轻量级 MLP 预测头 | 1.5-2x | 低开销场景 |

### 配置示例

```bash
# Draft Model 投机解码
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model meta-llama/Llama-3.1-8B-Instruct \
    --num-speculative-tokens 5 \
    --speculative-draft-tensor-parallel-size 1

# EAGLE-3 投机解码
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model yuhuili/EAGLE3-LLaMA3.1-Instruct-70B \
    --speculative-method eagle \
    --num-speculative-tokens 5

# N-gram 投机解码（无需额外模型）
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model [ngram] \
    --num-speculative-tokens 5 \
    --ngram-prompt-lookup-max 4
```

**性能参考**（EAGLE-3 on Llama 3.1 70B, H100）：
- Draft 接受率：约 70-85%
- 平均接受长度：3.2-4.1 tokens
- 端到端加速比：2.0-2.8x（取决于任务）

V1 中投机解码的调度表示与其他方式完全统一：`{req_id: num_speculative_tokens}`。调度器无需区分投机解码和普通 decode，这是 V1 统一调度器的优势之一。

---

## 量化支持

vLLM 通过 [[quantization|量化]] 技术减少模型内存占用并加速推理。详细的量化原理见 [[quantization]]。

### 支持的量化格式

| 量化方法 | 精度 | 内核支持 | 推荐场景 |
|----------|------|----------|----------|
| FP8 (E4M3) | W8A8 | vLLM 原生 | **推荐**：精度损失最小，性能提升显著 |
| FP4 (NV) | W4A4 | Blackwell SM120+ | NVIDIA Blackwell 专属 |
| AWQ | W4A16 | Marlin / GEMM | 仅权重量化中最佳推理性能 |
| GPTQ | W4A16 | Marlin / Machete / Exllama | Ampere+ 优化 |
| SmoothQuant | W8A8 | 原生 | 权重+激活联合量化 |
| INT4 | W4A16 | Compute Cap > 8.0 | 极致内存节省 |
| GGUF | 混合精度 | llama.cpp 兼容 | 兼容 llama.cpp 生态 |
| bitsandbytes | W4/W8 | HuggingFace 兼容 | 快速实验 |

### 量化使用示例

```bash
# 使用 FP8 量化模型
vllm serve neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8 \
    --tensor-parallel-size 4

# 使用 AWQ 量化模型
vllm serve TheBloke/Llama-2-70B-Chat-AWQ \
    --quantization awq \
    --tensor-parallel-size 4

# KV Cache 量化（FP8）
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --kv-cache-dtype fp8
```

### 量化对性能的影响

以 Llama 3.1 70B 在 H100 上为例：

| 配置 | 所需 GPU 数 | 吞吐量 (tok/s) | 相对 BF16 |
|------|-----------|----------------|-----------|
| BF16（基线） | 4x H100 | ~4,800 | 1.0x |
| FP8 | 2x H100 | ~5,200 | 1.08x |
| AWQ W4 | 2x H100 | ~4,600 | 0.96x |
| GPTQ W4 | 2x H100 | ~4,500 | 0.94x |

FP8 是目前推荐的量化方案：精度损失极小（<0.5% 在多数基准测试上），同时减少一半内存并小幅提升吞吐量。

---

## 代码示例

### 基本离线推理

```python
from vllm import LLM, SamplingParams

# 初始化模型
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    dtype="bfloat16",
    gpu_memory_utilization=0.9,      # GPU 内存利用率
    max_model_len=8192,              # 最大上下文长度
    enable_prefix_caching=True,      # 启用前缀缓存（V1 默认开启）
)

# 配置采样参数
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    max_tokens=512,
    repetition_penalty=1.1,
    stop=["<|eot_id|>"],
)

# 构造 chat 格式
messages_list = [
    [
        {"role": "system", "content": "你是一个有用的 AI 助手。"},
        {"role": "user", "content": "解释 Transformer 中的注意力机制。"},
    ],
    [
        {"role": "system", "content": "你是一个有用的 AI 助手。"},
        {"role": "user", "content": "什么是 KV Cache？"},
    ],
]

# Chat 推理（自动应用 chat template）
outputs = llm.chat(messages_list, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt[:50]}...")
    print(f"Output: {generated_text}\n")
```

### OpenAI 兼容 API 服务

**启动服务器**：

```bash
# 基本启动
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --api-key my-secret-key

# 生产环境配置
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.92 \
    --enable-prefix-caching \
    --host 0.0.0.0 \
    --port 8000
```

**客户端调用**：

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="my-secret-key",
)

# 流式输出
stream = client.chat.completions.create(
    model="meta-llama/Llama-3.1-70B-Instruct",
    messages=[
        {"role": "system", "content": "你是一个有用的 AI 助手。"},
        {"role": "user", "content": "用 Python 实现快速排序"},
    ],
    temperature=0.7,
    max_tokens=1024,
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### 多 GPU 服务配置

```bash
# 4 GPU Tensor Parallel
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --host 0.0.0.0 --port 8000

# 8 GPU TP + PP（跨节点）
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

### 常用采样参数

```python
from vllm import SamplingParams

greedy   = SamplingParams(temperature=0, max_tokens=256)                              # 确定性
creative = SamplingParams(temperature=1.2, top_p=0.95, top_k=100, max_tokens=2048)    # 创意
code_gen = SamplingParams(temperature=0.2, top_p=0.9, max_tokens=4096,
                          stop=["```\n"], repetition_penalty=1.05)                     # 代码
beam     = SamplingParams(use_beam_search=True, best_of=4, temperature=0)              # Beam Search
```

---

## 性能分析

### 基准测试数据

以下基准测试基于 NVIDIA H100 SXM5 80GB 进行，使用 Llama 系列模型：

**单 GPU 吞吐量（Llama 3.1 8B, BF16）**：

| 引擎版本 | 吞吐量 (tok/s) | 相对提升 |
|----------|----------------|----------|
| vLLM V0 | ~7,500 | 基线 |
| vLLM V1 | ~12,500 | **1.67x** |

**多 GPU 吞吐量（Llama 3.3 70B, FP8, H100）**：

| 并发请求数 | vLLM (tok/s) | SGLang (tok/s) | TensorRT-LLM (tok/s) |
|-----------|-------------|----------------|----------------------|
| 1 | 120 | 125 | 130 |
| 10 | 650 | 680 | 710 |
| 50 | 1,850 | 1,920 | 2,100 |
| 100 | 2,400 | 2,460 | 2,780 |

> 测试条件：512 avg input tokens, 256 avg output tokens, 200 prompts

**延迟指标（Llama 3.3 70B, FP8, 单 H100）**：

| 并发 | vLLM TTFT p50/p95 (ms) | TensorRT-LLM TTFT p50/p95 (ms) |
|------|------------------------|----------------------------------|
| 1 | 45 / 68 | 38 / 55 |
| 10 | 120 / 195 | 105 / 170 |
| 50 | 380 / 720 | 340 / 620 |
| 100 | 740 / 1,450 | 680 / 1,280 |

**MoE 大规模部署（DeepSeek-R1, 671B）**：

| 部署方案 | 硬件 | 吞吐量 |
|----------|------|--------|
| Wide-EP (H200 集群) | 多节点 H200 | **2,200 tok/s/GPU** |
| Wide-EP (GB200) Prefill | 4 prefill + 1 decode | 26,200 prefill TPGS |
| Wide-EP (GB200) Decode | 4 prefill + 1 decode | 10,100 decode TPGS |

### VRAM 使用

| 状态 | vLLM | TensorRT-LLM | SGLang |
|------|------|-------------|--------|
| 模型加载（空闲） | 71 GB | 74 GB | 72 GB |
| 50 并发峰值 | 76 GB | 77 GB | 75 GB |
| 100 并发峰值 | 78 GB | 79 GB | 78 GB |

### 冷启动时间

| 引擎 | 冷启动时间 | 说明 |
|------|-----------|------|
| vLLM | ~62 秒 | 加载模型权重 + CUDA Graph 捕获 |
| SGLang | ~58 秒 | 与 vLLM 接近 |
| TensorRT-LLM | ~28 分钟 | 一次性引擎编译（后续启动快） |

---

## vLLM vs SGLang vs TensorRT-LLM

详细对比三大主流推理引擎：

| 维度 | vLLM | [[sglang\|SGLang]] | [[tensorrt-llm\|TensorRT-LLM]] |
|------|------|--------|---------------|
| **开发者** | UC Berkeley + 社区 | UC Berkeley | NVIDIA |
| **开源协议** | Apache 2.0 | Apache 2.0 | Apache 2.0 |
| **核心优势** | 广泛兼容、生态丰富 | 前缀重计算优化、结构化输出 | 极致性能 |
| **前缀缓存** | Hash-based LRU（块级） | RadixAttention（token 级基数树） | 有限支持 |
| **前缀密集场景** | 良好 | **快 29%** | 一般 |
| **硬件支持** | **最广**（NVIDIA/AMD/TPU/Gaudi/Trainium） | 主要 NVIDIA + AMD | 仅 NVIDIA |
| **量化支持** | FP4/FP8/AWQ/GPTQ/INT4/GGUF | FP8/AWQ/GPTQ | FP8/INT4/INT8 (TensorRT) |
| **投机解码** | EAGLE-3, Draft Model, Medusa, N-gram | EAGLE, Draft Model | Draft Model |
| **结构化输出** | XGrammar | **SGLang Grammar（更优）** | 有限 |
| **多模态** | 全面支持 | 支持 | 支持 |
| **LoRA** | 多 LoRA 批处理 | 支持 | 有限 |
| **部署复杂度** | 低（pip install） | 低（pip install） | 高（需编译引擎） |
| **冷启动** | ~60s | ~60s | ~28min（首次编译） |
| **社区规模** | ~50k stars | ~20k stars（快速增长） | ~10k stars |
| **生产成熟度** | **最成熟** | 快速追赶 | 成熟（NVIDIA 生态） |
| **最适用场景** | 多样硬件、通用部署、批处理 | 多轮对话、Agent、结构化输出 | 固定模型、极致性能 |

**选型建议**：

```
需要支持 AMD/TPU/Gaudi？ → vLLM
多轮对话、共享前缀？     → SGLang（RadixAttention 优势明显）
固定模型、追求极致吞吐？  → TensorRT-LLM
通用场景、快速上线？      → vLLM（生态最丰富、文档最全）
MoE 大规模部署？         → vLLM（DP Attention + EP 最成熟）
```

---

## 部署实践

### Docker 部署

```bash
# 基本 Docker 启动
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model meta-llama/Llama-3.1-8B-Instruct

# 生产环境 Docker Compose
```

```yaml
# docker-compose.yml (关键配置)
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
      start_period: 120s    # 模型加载需要时间
    ipc: host               # NCCL 共享内存
```

### Kubernetes 部署

推荐使用 [vLLM Production Stack](https://github.com/vllm-project/production-stack) Helm Chart 部署。关键配置要点：

1. **共享内存**：必须挂载 `emptyDir{medium: Memory}` 到 `/dev/shm`（NCCL 通信需要）
2. **启动探针**：`failureThreshold × periodSeconds > 120s`（模型加载耗时 30-120s）
3. **模型持久化**：PVC 存储权重，避免 Pod 重启重新下载；多副本需要 ReadWriteMany
4. **GPU 资源**：`nvidia.com/gpu: N` 限制，配合 `--tensor-parallel-size N`
5. **健康检查**：`/health` 端点用于 startup/readiness/liveness probe

### 生产监控指标

| 指标 | 说明 | 告警阈值建议 |
|------|------|-------------|
| `vllm:num_requests_running` | 当前运行请求数 | 接近 `max_num_seqs` |
| `vllm:num_requests_waiting` | 等待队列长度 | > 100 持续 5min |
| `vllm:gpu_cache_usage_perc` | KV Cache 使用率 | > 95% |
| `vllm:avg_prompt_throughput_toks_per_s` | Prefill 吞吐量 | 低于基线 20% |
| `vllm:avg_generation_throughput_toks_per_s` | Decode 吞吐量 | 低于基线 20% |
| `vllm:e2e_request_latency_seconds` | 端到端延迟 | p99 > SLA |
| `vllm:time_to_first_token_seconds` | TTFT | p95 > 目标 |
| `vllm:prefix_cache_hit_rate` | 前缀缓存命中率 | 监控趋势 |

---

## 不足与局限

### 已知限制

| 限制 | 详细说明 |
|------|----------|
| **CPU/边缘部署** | vLLM 专为 GPU 设计，不适合 CPU-only 或边缘场景（此时应使用 llama.cpp / Ollama） |
| **低并发场景** | 单用户或极低并发时，vLLM 的调度开销可能不如轻量级方案 |
| **前缀缓存粒度** | 块级哈希（block-level）不如 SGLang 的 token-level RadixAttention 灵活 |
| **CUDA Graph 内存** | 默认的 `FULL_AND_PIECEWISE` 模式消耗额外 GPU 内存，小 GPU 可能需要降级 |
| **PCIe 拓扑** | 在没有 NVLink 的 PCIe 机器上可能出现 peer access 错误 |
| **超长序列** | 极长序列（>128K tokens）可能导致 KV cache 压力，需要精细调优 |
| **引擎编译延迟** | 首次启动时 CUDA Graph 捕获和 torch.compile 会增加冷启动时间 |

### 不适用场景

- **单用户桌面应用**：Ollama / llama.cpp 更合适
- **嵌入式/移动端**：需要 MLC-LLM、llama.cpp 等轻量级方案
- **需要最低延迟的固定模型**：TensorRT-LLM 编译后的引擎延迟更低
- **极度内存受限（<16GB VRAM）**：考虑 GGUF + llama.cpp

### 安全注意事项

- vLLM Completions API 曾发现不安全反序列化漏洞，生产环境务必启用 API Key 认证
- 不应直接暴露在公网，建议通过反向代理（Nginx/Envoy）+ API Gateway 部署
- 对于敏感数据场景，需要额外的安全审计

---

## 发展路线

### 近期发布

| 版本 | 日期 | 关键特性 |
|------|------|----------|
| v0.19.0 | 2026-04 | Gemma 4 MoE 支持、零气泡异步调度 + 投机解码、Vision ViT CUDA Graph |
| v0.18.x | 2026-03 | Model Runner V2、分段 CUDA Graph for PP |
| v0.15.1 | 2026-02 | NVIDIA Blackwell SM120 + GB200 支持、Wide-EP 成熟化 |
| v0.9.0 | 2025-Q2 | DP Attention + EP for MoE（首次引入）|
| v0.8.5 | 2025-Q1 | EAGLE-1/3 集成 |
| v0.8.0 | 2025-01 | V1 架构默认启用 |

### Q1 2026 路线图重点

- **EngineCore 优化**：Scheduler、KV Cache Manager 数据结构效率提升
- **PyTorch 编译集成**：自定义编译 + 融合 pass、vLLM IR 用于内核注册
- **硬件支持**：GB300 nightly wheels、所有前沿模型 Day-0 精度验证
- **两周发布节奏**：Q1 计划 6 个版本
- **vLLM-Omni**：TTS、Diffusion、World Models、VLA 模型支持

### 长期方向

- [[prefill-decode-disaggregation|预填充-解码分离]]（Mooncake Transfer Engine 集成）
- KV Cache offloading（CPU/SSD 卸载）
- vLLM Semantic Router（智能路由层，v0.1 Iris 已发布）
- 更深度的 [[speculative-decoding|投机解码]] 优化（零气泡 overlap）
- 多模态推理能力增强（Omni 模型）

---

## 参考文献

### 核心论文

- Kwon et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention" (SOSP 2023) — [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)

### 官方资源

- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [vLLM 官方文档](https://docs.vllm.ai/)
- [vLLM Blog](https://vllm.ai/blog)
- [vLLM V1 Alpha Release Blog](https://vllm.ai/blog/v1-alpha-release)
- [vLLM Large Scale Serving Blog](https://vllm.ai/blog/large-scale-serving)
- [vLLM Production Stack](https://github.com/vllm-project/production-stack)

### 社区资源

- [Life of an Inference Request (vLLM V1) — Ubicloud](https://www.ubicloud.com/blog/life-of-an-inference-request-vllm-v1)
- [The vLLM MoE Playbook — AMD ROCm Blog](https://rocm.blogs.amd.com/software-tools-optimization/vllm-moe-guide/README.html)
- [vLLM vs TensorRT-LLM vs SGLang Benchmarks — Spheron](https://www.spheron.network/blog/vllm-vs-tensorrt-llm-vs-sglang-benchmarks/)
- [Speculators v0.3.0 — vLLM Blog](https://vllm.ai/blog/speculators-v030)
- [EAGLE-3 with vLLM — Red Hat Developer](https://developers.redhat.com/articles/2025/07/01/fly-eagle3-fly-faster-inference-vllm-speculative-decoding)

---

## 相关页面

- [[paged-attention]] — 核心内存管理算法
- [[continuous-batching]] — 连续批处理调度策略
- [[speculative-decoding]] — 投机解码加速技术
- [[kv-cache-optimization]] — KV Cache 优化技术
- [[quantization]] — 模型量化方法
- [[model-parallelism]] — 模型并行基础
- [[parallelism-strategies-deep-dive]] — 并行策略深入分析
- [[prefill-decode-disaggregation]] — 预填充-解码分离架构
- [[sglang]] — 替代推理引擎（RadixAttention）
- [[tensorrt-llm]] — NVIDIA TensorRT-LLM 推理引擎
