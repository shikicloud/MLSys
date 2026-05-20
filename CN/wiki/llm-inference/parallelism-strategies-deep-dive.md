---
title: "LLM 并行策略完全指南：DP / TP / PP / SP / CP / EP / EDP / ETP"
category: llm-inference
tags: [张量并行, 数据并行, 专家并行, 流水线并行, 序列并行, 上下文并行, moe, 多gpu, 分布式推理, 分布式训练]
created: 2026-04-14
updated: 2026-05-13
status: mature
---

# LLM 并行策略完全指南：DP / TP / PP / SP / CP / EP / EDP / ETP

> [!abstract]+ TL;DR
> 系统覆盖大模型多 GPU 扩展的八大并行策略：**DP**（数据）、**TP**（张量/层内权重）、**PP**（流水线/层间）、**SP**（非 TP 区域的序列）、**CP**（注意力的序列）、**EP**（专家/MoE）、**EDP**（专家数据）、**ETP**（专家张量），加上 **ZeRO/FSDP**（分片数据）与 **DP Attention**（推理 KV 分区）。讲清每种策略切分什么、核心通信原语，以及如何组合 3D / 4D / 5D 并行训练与推理。生产级案例：DeepSeek-V3 部署。

## 1. 概述

大模型的规模早已超越单 GPU 的内存和算力极限。并行策略决定了"如何把一个巨大的模型 + 数据切分到多卡上运行"。下表是本文讨论的所有并行策略的速览：

| 缩写 | 别名 | 全称 | 切分对象 | 作用范围 | 核心通信原语 |
|------|------|------|---------|---------|------------|
| **DP** | — | Data Parallelism | 数据 batch | Dense 层 | AllReduce |
| **ZeRO/FSDP** | — | Sharded Data Parallelism | 数据 + 优化器/梯度/参数 | Dense 层 | AllGather + ReduceScatter |
| **TP** | — | Tensor Parallelism | 层内权重矩阵 | Dense 层 | AllReduce |
| **SP** | — | Sequence Parallelism | 序列维度（非TP区域） | Dense 层 | AllGather + ReduceScatter |
| **PP** | — | Pipeline Parallelism | 连续层组 | 所有层 | 点对点 Send/Recv |
| **CP** | — | Context Parallelism | 序列维度（注意力） | Attention 层 | Ring P2P / AllToAll |
| **EP** | — | Expert Parallelism | MoE 专家（整个专家网络） | MoE 层 | AllToAll（token 路由） |
| **EDP** | **DEP** | Expert Data Parallelism | MoE 层的数据 batch | MoE 层 | AllReduce（专家梯度同步） |
| **ETP** | **TEP** | Expert Tensor Parallelism | 单个专家的权重矩阵 | MoE 层 | AllGather + ReduceScatter |
| **DP Attention** | — | Data-Parallel Attention | KV Cache 分区 | 推理 Attention | AllGather |

**核心直觉**：DP 和 TP 是处理 dense 模型的经典手段；EP / EDP / ETP 是它们在 MoE 架构下的"对应物"。PP 把模型按层分段，SP 和 CP 都在序列维度上做文章但作用域不同，ZeRO/FSDP 优化了 DP 的内存效率，DP Attention 则是推理专用的 KV Cache 分区方案。

```
                          ┌──────────────────────────────────┐
                          │          全部 GPU 资源             │
                          └────────────────┬─────────────────┘
                                           │
                    ┌──────────────────────┴─────────────────────────┐
                    │                                                │
             Dense 层 (Attention + MLP)                      MoE 层 (Router + Experts)
                    │                                                │
         ┌────┬────┼────┬────┐                        ┌─────────────┼─────────────┐
         │    │    │    │    │                         │             │             │
        DP   TP   SP   PP   CP                       EP           EDP           ETP
      (复制  (切分  (序列  (按层  (序列              (切分专家     (复制专家      (切分单个
      模型,  权重,  维度  分段,   维度               到不同GPU)    处理不同数据)   专家的权重)
      分batch) 每层  切分  跨节点) 切分
              通信) 非TP区)      注意力)

             ZeRO/FSDP                               DP Attention
          (切片优化器/                              (推理专用,
           梯度/参数)                               KV Cache分区)
```

> **为什么要分 Dense 层和 MoE 层？** 因为 2025 年 NVIDIA 提出的 **MoE Parallel Folding** 框架已经将两者的并行维度完全解耦：
> - Dense 层：`N_total = TP × SP × CP × DP × PP`（SP 通常 = TP）
> - MoE 层：`N_total = ETP × EP × EDP × PP`
>
> 两者只需要 PP（流水线并行）保持一致，其余维度独立配置。

---

## 2. DP — 数据并行 (Data Parallelism)

### 2.1 原理

数据并行是最简单也最经典的并行策略：**每张 GPU 持有完整的模型副本，各自处理不同的数据子集**。

```
                    ┌──────────────────────┐
                    │     全局 Batch        │
                    │  [x₁, x₂, x₃, x₄]   │
                    └──────────┬───────────┘
                               │ 拆分
                    ┌──────────┴───────────┐
                    │                      │
              GPU 0: [x₁, x₂]       GPU 1: [x₃, x₄]
              ┌──────────┐           ┌──────────┐
              │ 完整模型W  │           │ 完整模型W  │
              │  (副本0)  │           │  (副本1)  │
              └─────┬────┘           └─────┬────┘
                    │ 前向+反向              │ 前向+反向
                    │                      │
                    ▼                      ▼
              grad₀ = ∂L/∂W          grad₁ = ∂L/∂W
                    │                      │
                    └───────┬──────────────┘
                            │ AllReduce
                            ▼
                    avg_grad = (grad₀ + grad₁) / 2
                            │
                    ┌───────┴───────┐
                    │               │
               GPU 0: W -= lr·avg  GPU 1: W -= lr·avg
```

### 2.2 通信分析

| 阶段 | 通信原语 | 通信量（每 GPU） |
|------|---------|----------------|
| 梯度同步 | AllReduce | `2 × P × sizeof(dtype)` （P = 参数量） |
| ZeRO-1 (优化器状态切分) | ReduceScatter + AllGather | 相同总量，但峰值内存更低 |
| ZeRO-3 / FSDP | 每层 AllGather(前向) + ReduceScatter(反向) | 相同总量，可 pipeline |

Ring AllReduce 的通信量公式：每 GPU 传输 `2 × P × (N-1)/N` 字节，N 很大时趋近 `2P`。

### 2.3 代码示例：PyTorch DDP

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 初始化进程组
dist.init_process_group("nccl")
local_rank = dist.get_rank()
torch.cuda.set_device(local_rank)

# 每张 GPU 持有完整模型
model = MyModel().cuda(local_rank)
model = DDP(model, device_ids=[local_rank])

# 不同 GPU 拿到不同 batch（通过 DistributedSampler）
sampler = torch.utils.data.distributed.DistributedSampler(dataset)
loader = DataLoader(dataset, sampler=sampler)

for batch in loader:
    loss = model(batch)
    loss.backward()        # DDP 在反向传播时自动 AllReduce 梯度
    optimizer.step()
```

### 2.4 使用场景

- 模型能放进单张 GPU → **首选 DP**（最简单，效率最高）
- 需要扩大吞吐量（batch size）
- 梯度同步频率低（一次完整前向+反向后才同步一次）

### 2.5 不足

| 不足 | 说明 |
|------|------|
| **内存冗余** | 每张卡都存完整模型 → 内存利用率低 |
| **通信量与模型大小成正比** | 模型越大，梯度同步越贵 |
| **无法处理超大模型** | 单卡放不下 → 必须结合 TP/PP |

> **ZeRO / FSDP** 通过将优化器状态、梯度、参数在 DP 组内切片来解决内存冗余问题，详见 §3。

---

## 3. ZeRO / FSDP — 分片数据并行 (Sharded Data Parallelism)

### 3.1 问题背景：DP 的内存冗余

标准 DP 中，每张 GPU 都存储完整的模型参数、梯度和优化器状态。以 Adam 为例，训练一个 P 参数的 FP16 模型，每张 GPU 的内存开销为：

```
每 GPU 内存（标准 DP, FP16 Mixed Precision + Adam）:
  参数 (FP16):          2P bytes
  梯度 (FP16):          2P bytes
  优化器状态 (FP32):
    - 参数主副本 (FP32): 4P bytes
    - 一阶矩 m (FP32):  4P bytes
    - 二阶矩 v (FP32):  4P bytes
  ────────────────────────────
  总计:                  16P bytes

  例: P = 7B → 16 × 7B = 112 GB / GPU    ← 超过单张 80GB H100!
```

ZeRO (Zero Redundancy Optimizer, Rajbhandari et al., 2020) 的核心思想：**DP 组内 N 张 GPU 没必要每张都存完整的 16P，可以把不同部分分片存储**。

### 3.2 ZeRO 三个阶段

```
标准 DP (每张 GPU):                ZeRO Stage 1:
┌──────────────────────┐          ┌──────────────────────┐
│  参数 W       2P     │          │  参数 W       2P     │
│  梯度 G       2P     │          │  梯度 G       2P     │
│  优化器 OS   12P     │          │  优化器 OS   12P/N   │ ← 切分!
│  ─────────────────── │          │  ─────────────────── │
│  总计        16P     │          │  总计     4P + 12P/N │
└──────────────────────┘          └──────────────────────┘

ZeRO Stage 2:                     ZeRO Stage 3 (= FSDP):
┌──────────────────────┐          ┌──────────────────────┐
│  参数 W       2P     │          │  参数 W       2P/N   │ ← 切分!
│  梯度 G       2P/N   │ ← 切分! │  梯度 G       2P/N   │ ← 切分!
│  优化器 OS   12P/N   │ ← 切分! │  优化器 OS   12P/N   │ ← 切分!
│  ─────────────────── │          │  ─────────────────── │
│  总计     2P + 14P/N │          │  总计        16P/N   │
└──────────────────────┘          └──────────────────────┘
```

### 3.3 各阶段的内存公式

| 阶段 | 每 GPU 内存 | N=8 时 (7B 模型) | 切分了什么 |
|------|-----------|-----------------|----------|
| 标准 DP | `16P` | 112 GB | 无 |
| ZeRO-1 | `4P + 12P/N` | 38.5 GB | 优化器状态 |
| ZeRO-2 | `2P + 14P/N` | 26.25 GB | 优化器状态 + 梯度 |
| ZeRO-3 / FSDP | `16P/N` | 14 GB | 优化器状态 + 梯度 + 参数 |

### 3.4 通信模式对比

```
标准 DP:
  反向传播后 → AllReduce(梯度)
  通信量: 2P bytes/GPU    通信次数: 1 次/step

ZeRO-1:
  反向传播后 → ReduceScatter(梯度) → 各自更新本地优化器 → AllGather(更新后参数)
  通信量: 2P bytes/GPU    通信次数: 2 次/step (但可 pipeline)

ZeRO-2:
  反向传播中 → ReduceScatter(梯度, 逐层) → 各自更新 → AllGather(参数)
  通信量: 2P bytes/GPU    通信次数: 与 ZeRO-1 相同
  优势: 梯度可以在 ReduceScatter 后立即释放

ZeRO-3 / FSDP:
  前向: AllGather(参数, 逐层) → 计算 → 释放非本地参数
  反向: AllGather(参数, 逐层) → 计算梯度 → ReduceScatter(梯度)
  通信量: 3P bytes/GPU    ← 比标准 DP 多 50%!
  通信次数: 每层 2 次 AllGather + 1 次 ReduceScatter
```

| 策略 | 通信量/GPU | 通信次数 | 内存 | 适用场景 |
|------|-----------|---------|------|---------|
| 标准 DP | `2P` | 1/step | 16P | 模型小，GPU 内存充足 |
| ZeRO-1 | `2P` | 2/step | 4P + 12P/N | 优化器状态是瓶颈 |
| ZeRO-2 | `2P` | 2/step | 2P + 14P/N | 梯度也很大 |
| ZeRO-3/FSDP | `3P` | 每层多次 | 16P/N | 模型极大，内存极紧 |

### 3.5 FSDP：PyTorch 的 ZeRO-3 实现

PyTorch 的 **Fully Sharded Data Parallelism (FSDP)** 本质上是 ZeRO-3 的原生实现：

```python
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

# ZeRO-3 等效: 参数+梯度+优化器全部切分
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # = ZeRO-3
    device_id=local_rank,
)

# ZeRO-2 等效: 只切分梯度+优化器，参数保留完整
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,  # = ZeRO-2
    device_id=local_rank,
)

# 训练循环与 DDP 几乎相同
for batch in loader:
    loss = model(batch)
    loss.backward()
    optimizer.step()
```

FSDP2 (PyTorch 2.4+) 进一步改进了 API 和性能，支持更细粒度的 per-parameter sharding。

### 3.6 什么时候用哪个阶段

```
选择 ZeRO 阶段的决策树:

  模型 + Adam 能放进单卡 (16P < GPU_mem)?
    ├── 是 → 标准 DP (最快, 通信最少)
    └── 否 → ZeRO-1 能放下 (4P + 12P/N < GPU_mem)?
              ├── 是 → ZeRO-1 (通信量不变, 推荐首选)
              └── 否 → ZeRO-2 能放下?
                        ├── 是 → ZeRO-2 (通信量不变, 梯度也切)
                        └── 否 → ZeRO-3 / FSDP
                                  (通信量 +50%, 但内存最省)
                                  如果还不够 → 结合 TP/PP
```

### 3.7 不足

| 不足 | 说明 |
|------|------|
| **ZeRO-3 通信量增加 50%** | 前向也需要 AllGather 参数，相比标准 DP 多一半通信 |
| **逐层通信增加延迟** | ZeRO-3 每层都要 AllGather → 计算 → ReduceScatter |
| **推理不适用** | ZeRO 是训练专用（推理不需要优化器/梯度） |
| **与 TP/EP 组合时需注意通信组** | 不同并行维度的通信组不能冲突 |

---

## 4. TP — 张量并行 (Tensor Parallelism)

### 4.1 原理

张量并行（Megatron-LM, 2019）在**每一层内部**切分权重矩阵，把单个矩阵乘法分散到多张 GPU 上。核心有两种切法：

#### 列并行 (Column Parallel)

将权重矩阵 `A [D, F]` 按列切分为 `A₁ [D, F/N], A₂ [D, F/N], ...`，输入 `X` 复制到每张卡：

```
          输入 X [B, D]  (所有 GPU 持有相同副本)
               │
    ┌──────────┼──────────┐
    │          │          │
  GPU 0      GPU 1      GPU 2
  A₁[D,F/3]  A₂[D,F/3]  A₃[D,F/3]      ← 权重按列切分
    │          │          │
    ▼          ▼          ▼
  Y₁=X·A₁   Y₂=X·A₂   Y₃=X·A₃          ← 各自计算，无需通信
  [B, F/3]   [B, F/3]   [B, F/3]

  → GeLU(Y₁)  GeLU(Y₂)  GeLU(Y₃)        ← 激活函数可独立施加！
```

**关键**：GeLU 等逐元素激活函数可以在切分后独立施加，因为 `GeLU([Y₁, Y₂]) = [GeLU(Y₁), GeLU(Y₂)]`。

#### 行并行 (Row Parallel)

将权重矩阵 `B [F, D]` 按行切分为 `B₁ [F/N, D], B₂ [F/N, D], ...`：

```
  GeLU(Y₁)   GeLU(Y₂)   GeLU(Y₃)        ← 来自列并行的输出
  [B, F/3]   [B, F/3]   [B, F/3]
    │          │          │
  GPU 0      GPU 1      GPU 2
  B₁[F/3,D]  B₂[F/3,D]  B₃[F/3,D]      ← 权重按行切分
    │          │          │
    ▼          ▼          ▼
  Z₁=Y₁·B₁  Z₂=Y₂·B₂  Z₃=Y₃·B₃        ← 各自计算部分结果
  [B, D]     [B, D]     [B, D]
    │          │          │
    └──────────┼──────────┘
               │ AllReduce (求和)
               ▼
         Z = Z₁ + Z₂ + Z₃               ← 最终结果
             [B, D]
```

#### MLP 块的完整流程

```
           ┌─── f ───┐
           │ (identity fwd, AllReduce bwd)
           │
    X ─────┤
           │    Column Parallel          Row Parallel
           │    (gate_proj + up_proj)    (down_proj)
           │         │                       │
           │      GeLU/SiLU              AllReduce ─── g ───→ 输出
           │         │                       │        (AllReduce fwd,
           │         └───────────────────────┘         identity bwd)
           └──────────────────────────────────────────────────────────

    ★ 每个 MLP 块：前向 1 次 AllReduce，反向 1 次 AllReduce
```

#### Attention 块

Q, K, V 投影按**注意力头**切分（列并行），每张 GPU 处理一部分头。输出投影用行并行：

```
    GPU 0: heads [0,1]     GPU 1: heads [2,3]     GPU 2: heads [4,5]
        │                      │                      │
        ▼                      ▼                      ▼
    Attention_0            Attention_1            Attention_2
        │                      │                      │
        └──────────────────────┼──────────────────────┘
                               │ AllReduce (output projection)
                               ▼
                          合并后的输出
```

**每个 Transformer 层总计**：4 次 AllReduce（MLP 前向1 + 反向1，Attention 前向1 + 反向1）。

### 4.2 通信分析

| 指标 | 公式 |
|------|------|
| 每次 AllReduce 通信量 | `2 × B × S × D × sizeof(dtype)` |
| 每 Transformer 层通信量 (前向) | `2 × 2BSD = 4BSD` 字节 |
| 每 Transformer 层通信量 (总) | `4 × 2BSD = 8BSD` 字节 |
| TP 效率（经验值） | TP=2: 85-95%, TP=4: 70-85%, TP=8: 56-75% |

**计算-通信比** (JAX Scaling Book)：
```
T_comms = (4 × B × D) / W_ici        ← 通信时间
T_compute = (8 × B × D × F) / (N × C) ← 计算时间

当 B/N > C/W_ici 时，计算密集型（理想状态）
H100 NVLink: C ≈ 990 TFLOPS, W ≈ 900 GB/s → 阈值 ≈ 1100 tokens/GPU
```

### 4.3 代码示例：列并行 + 行并行

```python
import torch
import torch.distributed as dist

class ColumnParallelLinear(torch.nn.Module):
    """列并行线性层：权重按列（输出维度）切分"""
    def __init__(self, in_features, out_features, tp_group):
        super().__init__()
        self.tp_group = tp_group
        self.tp_size = dist.get_world_size(tp_group)
        self.tp_rank = dist.get_rank(tp_group)
        # 每张卡只持有 1/N 的输出维度
        assert out_features % self.tp_size == 0
        self.local_out = out_features // self.tp_size
        self.weight = torch.nn.Parameter(
            torch.randn(self.local_out, in_features)  # [F/N, D]
        )

    def forward(self, x):
        # x: [B, S, D] — 所有 GPU 持有相同输入
        # 各自计算 Y_i = X @ W_i^T → [B, S, F/N]
        return torch.nn.functional.linear(x, self.weight)
        # 无需通信！GeLU 可直接在输出上施加


class RowParallelLinear(torch.nn.Module):
    """行并行线性层：权重按行（输入维度）切分"""
    def __init__(self, in_features, out_features, tp_group):
        super().__init__()
        self.tp_group = tp_group
        self.tp_size = dist.get_world_size(tp_group)
        assert in_features % self.tp_size == 0
        self.local_in = in_features // self.tp_size
        self.weight = torch.nn.Parameter(
            torch.randn(out_features, self.local_in)  # [D, F/N]
        )

    def forward(self, x):
        # x: [B, S, F/N] — 每张卡持有输入的一部分
        local_out = torch.nn.functional.linear(x, self.weight)  # [B, S, D]
        # AllReduce 求和 → Z = Z₁ + Z₂ + ... + Z_N
        dist.all_reduce(local_out, group=self.tp_group)
        return local_out


class TPTransformerMLP(torch.nn.Module):
    """张量并行的 MLP 块"""
    def __init__(self, hidden_dim, ffn_dim, tp_group):
        super().__init__()
        self.gate_proj = ColumnParallelLinear(hidden_dim, ffn_dim, tp_group)
        self.down_proj = RowParallelLinear(ffn_dim, hidden_dim, tp_group)

    def forward(self, x):
        # Column Parallel → 无通信
        h = self.gate_proj(x)
        h = torch.nn.functional.silu(h)       # 激活函数可独立施加
        # Row Parallel → AllReduce
        return self.down_proj(h)
```

### 4.4 使用场景

- 模型单层参数超出单 GPU 内存
- **低延迟推理**（单请求延迟最小化）
- 总是保持在 **NVLink 域内**（通常 1 个节点 = 8 GPU）

### 4.5 不足

| 不足 | 说明 |
|------|------|
| **每层都要通信** | AllReduce 是同步阻塞的，每层 2 次 |
| **效率随 TP 度下降** | TP=8 时效率可能只有 56-75% |
| **必须 NVLink** | PCIe 上 TP=4 时通信可能占推理时间 40-50% |
| **扩展性有限** | 通常 TP ≤ 8（一个 NVLink 域） |

---

## 5. SP — 序列并行 (Sequence Parallelism)

### 5.1 动机：TP 的激活内存盲区

TP 切分了权重矩阵，使得 TP 区域内（Attention、MLP 的矩阵乘法）的激活内存按 `1/TP` 缩小。然而，Transformer 层中的 **LayerNorm** 和 **Dropout** 操作不在 TP 的切分范围内——这些操作需要完整的隐藏维度输入，因此每张 GPU 都持有完整的激活张量。

```
一个 Transformer 层的激活内存分布 (无 SP):

    LayerNorm₁   →  Attention (TP区域)  →  Dropout₁  →  LayerNorm₂  →  MLP (TP区域)  →  Dropout₂
    ┌──────┐       ┌──────────────┐       ┌──────┐     ┌──────┐       ┌──────────┐     ┌──────┐
    │ B×S×D │       │  B×S×D/TP    │       │ B×S×D │     │ B×S×D │       │ B×S×D/TP  │     │ B×S×D │
    │ (完整) │       │  (切分)      │       │ (完整) │     │ (完整) │       │  (切分)    │     │ (完整) │
    └──────┘       └──────────────┘       └──────┘     └──────┘       └──────────┘     └──────┘
       ★ 未切分的部分占激活内存 > 50%！
```

SP (Megatron-LM, Korthikanti et al., 2022) 的解决方案：**在非 TP 区域沿序列维度切分**。

### 5.2 工作原理

SP 的核心思想非常优雅：利用 TP 已有的 AllReduce 通信，将其分解为 AllGather + ReduceScatter，在两种区域之间无缝转换。

```
    ┌─────────────────────────────────────────────────────────────────┐
    │                    Transformer 层 (TP + SP)                     │
    │                                                                 │
    │   [SP 区域]           [TP 区域]           [SP 区域]              │
    │   LayerNorm           Attention/MLP        Dropout               │
    │   每 GPU: [B,S/TP,D]  每 GPU: [B,S,D/TP]  每 GPU: [B,S/TP,D]   │
    │   沿 seq 切分          沿 hidden 切分       沿 seq 切分           │
    │        │                     │                    │              │
    │        └── AllGather(seq) ──→┘                    │              │
    │              (聚合序列维度)                         │              │
    │                              └── ReduceScatter ──→┘              │
    │                                   (规约+分发序列维度)             │
    │                                                                 │
    │   ★ AllReduce = AllGather + ReduceScatter                       │
    │   ★ 总通信量不变！只是把一次 AllReduce 拆成两步                   │
    │   ★ 但 LayerNorm/Dropout 的激活内存从 B×S×D 降到 B×(S/TP)×D     │
    └─────────────────────────────────────────────────────────────────┘
```

具体流程（以一个 Transformer 层为例）：

```
    GPU0 持有 seq 的 [0 : S/TP] 部分
    GPU1 持有 seq 的 [S/TP : 2S/TP] 部分
    ...

    Step 1: LayerNorm — 每 GPU 独立对自己的 S/TP tokens 做 LayerNorm
            输入: [B, S/TP, D]  输出: [B, S/TP, D]

    Step 2: AllGather(seq维度) — 收集完整序列
            输入: [B, S/TP, D]  输出: [B, S, D]
            ★ 现在每 GPU 有完整序列，可进入 TP 区域

    Step 3: Attention / MLP (TP 区域) — 沿 hidden 维度切分计算
            每 GPU: [B, S, D/TP]

    Step 4: ReduceScatter(seq维度) — 规约+重新按序列分发
            输入: [B, S, D] (每 GPU 的部分结果)
            输出: [B, S/TP, D] (规约后的完整结果，按序列切分)
            ★ 回到 SP 区域

    Step 5: Dropout — 每 GPU 独立对自己的 S/TP tokens 做 Dropout
            输入: [B, S/TP, D]  输出: [B, S/TP, D]
```

### 5.3 内存节省

```
激活内存对比 (每 GPU):

              无 SP (仅 TP)                     有 SP (TP + SP)
    LayerNorm:  B × S × D                   B × S/TP × D       ← 减少 TP 倍!
    Attention:  B × S × D/TP                B × S × D/TP       (不变)
    Dropout:    B × S × D                   B × S/TP × D       ← 减少 TP 倍!
    MLP:        B × S × D/TP                B × S × D/TP       (不变)

    总体: 非 TP 区域的激活内存减少为 1/TP
    对于 TP=8, 这意味着总激活内存可减少约 40-60%
```

### 5.4 为什么 TP+EP 时必须启用 SP

当 TP 和 EP 同时使用时，Megatron-LM 强制要求 `--sequence-parallel`。原因：

1. Dense 层（Attention）使用 TP，MoE 层使用 EP
2. TP 组和 EP 组的 GPU 划分通常不同
3. SP 的 AllGather/ReduceScatter 为两种层之间的数据格式转换提供了自然的桥梁
4. 没有 SP，TP 的 AllReduce 无法正确地与 EP 的 AllToAll 衔接

### 5.5 代码示例

```python
import torch
import torch.distributed as dist

class SequenceParallelLayerNorm(torch.nn.Module):
    """序列并行的 LayerNorm：每 GPU 只处理 S/TP 个 token"""
    def __init__(self, hidden_dim, tp_group):
        super().__init__()
        self.norm = torch.nn.LayerNorm(hidden_dim)
        self.tp_group = tp_group

    def forward(self, x):
        # x: [B, S/TP, D] — 序列已按 TP 组切分
        return self.norm(x)  # LayerNorm 沿 hidden_dim 操作，不需要完整序列


def allgather_seq(x, tp_group):
    """AllGather: [B, S/TP, D] → [B, S, D]"""
    tp_size = dist.get_world_size(tp_group)
    gathered = [torch.empty_like(x) for _ in range(tp_size)]
    dist.all_gather(gathered, x, group=tp_group)
    return torch.cat(gathered, dim=1)  # concat 沿 seq 维度


def reducescatter_seq(x, tp_group):
    """ReduceScatter: [B, S, D] → [B, S/TP, D] (规约后按 seq 切分)"""
    tp_size = dist.get_world_size(tp_group)
    chunks = list(x.chunk(tp_size, dim=1))
    output = torch.empty_like(chunks[0])
    dist.reduce_scatter(output, chunks, group=tp_group)
    return output


class SPTransformerBlock(torch.nn.Module):
    """带序列并行的 Transformer 块"""
    def __init__(self, hidden_dim, ffn_dim, n_heads, tp_group):
        super().__init__()
        self.ln1 = SequenceParallelLayerNorm(hidden_dim, tp_group)
        self.attn = TPAttention(hidden_dim, n_heads, tp_group)  # TP 区域
        self.ln2 = SequenceParallelLayerNorm(hidden_dim, tp_group)
        self.mlp = TPTransformerMLP(hidden_dim, ffn_dim, tp_group)  # TP 区域
        self.tp_group = tp_group

    def forward(self, x):
        # x: [B, S/TP, D] — SP 区域
        residual = x

        # SP → TP 过渡
        h = self.ln1(x)                              # [B, S/TP, D] — SP
        h = allgather_seq(h, self.tp_group)           # [B, S, D] — 完整序列
        h = self.attn(h)                              # [B, S, D/TP] — TP 区域
        h = reducescatter_seq(h, self.tp_group)       # [B, S/TP, D] — 回到 SP
        h = torch.nn.functional.dropout(h, p=0.1)    # SP 区域
        x = residual + h

        # 同样的流程用于 MLP
        residual = x
        h = self.ln2(x)
        h = allgather_seq(h, self.tp_group)
        h = self.mlp(h)
        h = reducescatter_seq(h, self.tp_group)
        h = torch.nn.functional.dropout(h, p=0.1)
        x = residual + h

        return x  # [B, S/TP, D]
```

### 5.6 SP vs CP —— 撞名引发的混乱

"Sequence parallelism"这个词至少被三篇不同论文重用过，引发的混乱让人原地转圈。**本 wiki 里的 SP 特指** Megatron-LM v2（Korthikanti et al., 2022）的定义：在一个 TP 组 *内部* 仅切 LayerNorm/Dropout 的激活。**本 wiki 里的 CP** 是更宽泛的"在独立 GPU 维度上切整个 attention 计算"。

论文和代码里你会撞见的三种"序列并行"：

| 实际切了什么 | 大家叫它 | 本 wiki 称呼 |
|---|---|---|
| TP 组内 LayerNorm / Dropout 激活（Megatron v2） | "Sequence parallelism" | **SP** |
| 整个 attention 跨独立 GPU 维度切，AllToAll 实现（DeepSpeed Ulysses） | "Sequence parallelism" | **CP**（Ulysses 变体） |
| 整个 attention 跨独立 GPU 维度切，Ring P2P 实现（Ring Attention） | "Sequence parallelism" 或 "Ring Attention" | **CP**（Ring 变体） |

两个问题就能区分任何论文：

1. **切的是什么？** 只切 LayerNorm/Dropout → SP。整个 attention → CP。
2. **是 TP 组内还是独立 GPU 维度？** TP 组内 → SP（大小永远等于 TP）。独立维度 → CP（大小与 TP 无关）。

完整对比：

| 维度 | SP (Megatron v2) | CP |
|------|------------------|----|
| 切什么 | LayerNorm + Dropout 激活（非 TP 区域） | 整个 attention（QKV + softmax + 输出投影） |
| 与 TP 关系 | 永远在 TP 组内；SP 大小 = TP 大小 | 独立 GPU 维度；CP 大小与 TP 无关 |
| 通信原语 | 进 TP 区域 AllGather + 出 TP 区域 ReduceScatter | Ring P2P（Ring Attention）或 AllToAll（Ulysses） |
| 存在意义 | 降低 TP 区域的激活显存 | 支持单卡 KV cache 装不下的超长序列 |
| 典型大小 | SP = TP = 8（取决于 TP 设置） | CP = 2 / 4 / 8 / ... / 64+ |
| 是否必须 | Megatron-LM 在 TP+EP 同时用时强制 SP | 序列超出单卡容量时才用 |
| 跟谁通信 | 与 TP 区域的 AllReduce（被拆成 AG + RS） | 与持有相邻序列段的其它 CP rank |
| 总通信量 | 与朴素 TP 相同（AllReduce 被拆解，没消除） | 在 TP/DP 之上额外增加 |

DeepSpeed Ulysses 论文标题 *"Sequence Parallelism for Long Sequence Training"* 是最痛的撞名 —— 它说的"sequence parallelism"恰好是其他所有框架现在叫 CP 的东西。**读任何论文时若看到"sequence parallelism"，看维度上切了什么，别看名字**。改了 attention 数学就是 CP；只挪了 LayerNorm/Dropout 的激活就是 SP。

### 5.7 不足

| 不足 | 说明 |
|------|------|
| **总通信量不变** | 只是把 AllReduce 拆成 AG + RS，总字节数相同 |
| **依赖 TP** | SP 是 TP 的附属，没有 TP 就没有 SP |
| **实现复杂度** | 需要在 TP 和 SP 区域之间正确转换张量形状 |

---

## 6. PP — 流水线并行 (Pipeline Parallelism)

### 6.1 基本概念

流水线并行将模型的 **连续层组** 分配到不同的 GPU 上。每个 GPU 只负责一部分层的前向和反向计算，通过**点对点通信 (Send/Recv)** 在 stage 之间传递激活值和梯度。

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   GPU 0     │    │   GPU 1     │    │   GPU 2     │    │   GPU 3     │
│  Stage 0    │───▶│  Stage 1    │───▶│  Stage 2    │───▶│  Stage 3    │
│ Layer 0-7   │    │ Layer 8-15  │    │ Layer 16-23 │    │ Layer 24-31 │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                 Send/Recv          Send/Recv          Send/Recv
            (只传激活向量)       (只传激活向量)       (只传激活向量)
```

**PP 的核心优势**：
- **通信量小**：只需在 stage 之间传递一次激活值（形状为 `[B, S, H]`），而非像 TP 那样每层都做 AllReduce
- **对带宽需求低**：点对点通信即可，不需要 NVLink，PCIe 甚至跨节点网络均可
- **内存效率**：每个 GPU 只持有 `1/PP` 的模型参数

**PP 的核心问题**：**流水线气泡 (Pipeline Bubble)**——当一个 stage 在等待上游数据或下游梯度时，处于空闲状态。

### 6.2 Naive PP 与气泡问题

最朴素的流水线并行：一个 micro-batch 依次经过所有 stage，任一时刻只有一个 GPU 在工作。

```
GPU 0: │██F██│                        │██B██│
GPU 1: │     │██F██│            │██B██│
GPU 2: │     │     │██F██│██B██│
GPU 3: │     │     │     │█F+B█│        空白 = 气泡 (idle)
```

**利用率** = `1 / (2p)` 。p=4 时仅 12.5%——完全不可接受。

### 6.3 GPipe：微批次流水线

**GPipe** (Huang et al., 2019) 的核心思想：将一个 mini-batch 切分为 **m 个 micro-batch**，让多个 micro-batch 在流水线中依次流过，从而让多个 GPU 同时工作。

**调度方式**：先完成所有 micro-batch 的前向（全部 F），再依次执行所有 micro-batch 的反向（全部 B）。中间有一个 **pipeline flush**（排空）。

```
GPipe Schedule (p=4 stages, m=4 micro-batches)

时间 ──────────────────────────────────────────────────────────────────────▶

GPU 0: │F₁│F₂│F₃│F₄│         │B₄│B₃│B₂│B₁│
GPU 1: │  │F₁│F₂│F₃│F₄│      │B₄│B₃│B₂│B₁│
GPU 2: │  │  │F₁│F₂│F₃│F₄│   │B₄│B₃│B₂│B₁│
GPU 3: │  │  │  │F₁│F₂│F₃│F₄│B₄│B₃│B₂│B₁│
                               ▲
                          pipeline flush
                          (同步屏障)

    █ = 计算    空白 = 气泡 (bubble)
```

**气泡率公式**：

```
                   p - 1
Bubble Rate = ─────────────
                m + p - 1
```

- p = pipeline stage 数
- m = micro-batch 数

| p (stages) | m (micro-batches) | 气泡率 |
|-----------|-------------------|-------|
| 4 | 4 | 42.9% |
| 4 | 8 | 27.3% |
| 4 | 16 | 15.8% |
| 4 | 32 | 8.6% |
| 8 | 32 | 17.9% |
| 16 | 64 | 19.0% |

**经验法则**：`m >= 4 × p` 时，气泡率约 `< 20%`。但增大 m 意味着每个 micro-batch 更小，可能影响计算效率，且需要存储更多激活值（GPipe 通过激活重计算来解决）。

### 6.4 1F1B 调度（PipeDream）

**1F1B (One Forward One Backward)** 由 PipeDream (Narayanan et al., 2019) 提出，核心思想：**不再等所有前向做完再做反向，而是尽早交替执行前向和反向**。

**三个阶段**：
1. **Warmup 阶段**：各 stage 依次启动前向，填满流水线
2. **Steady State（稳态）**：每个 GPU 严格交替执行 1 次 F 和 1 次 B
3. **Cooldown 阶段**：清空流水线中剩余的反向

```
1F1B Schedule (p=4, m=8)

时间 ──────────────────────────────────────────────────────────────────────▶

GPU 0: │F₁│F₂│F₃│F₄│B₁│F₅│B₂│F₆│B₃│F₇│B₄│F₈│B₅│B₆│B₇│B₈│
GPU 1: │  │F₁│F₂│F₃│B₁│F₄│B₂│F₅│B₃│F₆│B₄│F₇│B₅│F₈│B₆│B₇│B₈│
GPU 2: │  │  │F₁│F₂│B₁│F₃│B₂│F₄│B₃│F₅│B₄│F₆│B₅│F₇│B₆│F₈│B₇│B₈│
GPU 3: │  │  │  │F₁│B₁│F₂│B₂│F₃│B₃│F₄│B₄│F₅│B₅│F₆│B₆│F₇│B₇│F₈│B₈│

        ◄─warmup─▶◄────────── steady state ──────────▶◄─cooldown─▶
```

**1F1B 的优势**：
- **气泡率与 GPipe 相同**：`(p-1) / (m+p-1)`
- **峰值内存更低**：不需要同时存储所有 m 个 micro-batch 的激活。稳态时每个 GPU 只保存 p 个 micro-batch 的激活（而 GPipe 需要 m 个）
- **更早释放内存**：反向执行后立即释放对应的激活

**内存对比**：

| 调度 | 峰值激活存储量 |
|------|-------------|
| GPipe | m 个 micro-batch 的激活 |
| 1F1B | p 个 micro-batch 的激活 |

当 m >> p 时，1F1B 的内存优势非常明显。

### 6.5 Interleaved 1F1B（虚拟 Stage）

**Interleaved 1F1B** (Narayanan et al., 2021, Megatron-LM v2) 在 1F1B 基础上引入 **虚拟流水线 (virtual pipeline)**：每个 GPU 不再只负责连续的一段层，而是负责 **多个不连续的 chunk (model chunk)**。

例如，4 个 GPU、virtual_pipeline_size=2 时：

```
物理分配:
  GPU 0: Layer 0-3  + Layer 16-19    (chunk 0 + chunk 4)
  GPU 1: Layer 4-7  + Layer 20-23    (chunk 1 + chunk 5)
  GPU 2: Layer 8-11 + Layer 24-27    (chunk 2 + chunk 6)
  GPU 3: Layer 12-15 + Layer 28-31   (chunk 3 + chunk 7)

逻辑流水线 (8 个虚拟 stage):
  VS0 → VS1 → VS2 → VS3 → VS4 → VS5 → VS6 → VS7
  GPU0   GPU1   GPU2   GPU3   GPU0   GPU1   GPU2   GPU3
```

**气泡率**：

```
                         p - 1
Bubble Rate = ──────────────────────
               m × v + p - 1

v = virtual_pipeline_model_parallel_size (每 GPU 的 chunk 数)
```

当 v=2 时，气泡率约为原来的一半。

**代价**：
- 通信量增加 v 倍（每个虚拟 stage 都需要额外的 Send/Recv）
- 实现复杂度上升

### 6.6 Zero Bubble PP (Qi et al., ICLR 2024)

**核心洞察**：将反向传播拆分为两个独立的部分：
- **B** (backward_input)：计算输入的梯度 ∂L/∂x（需要下游梯度，时间敏感）
- **W** (backward_weight)：计算权重的梯度 ∂L/∂W（只依赖本地激活，可延迟执行）

传统 1F1B 中 B 和 W 绑定在一起执行，Zero Bubble 将它们解耦，用 W 来**填充气泡**。

#### ZB-H1：基于 1F1B 的改进

ZB-H1 大体遵循 1F1B 调度，但将 W 推迟执行，用 W 填充 1F1B 尾部的气泡。**气泡率约为 1F1B 的 1/3**。

```
ZB-H1 示意: F...F│B│F│B│F│...│B│...│B│W│W│W│W│W│W│...
                                       ▲ W 填充原本的气泡区域
```

#### ZB-H2：接近零气泡

允许更大内存（更多 in-flight micro-batch），W 可以进一步填满所有气泡，**理论上实现零气泡**。

#### ZB-V：V 形虚拟 Stage

每 GPU 分配 2 个 chunk，依赖关系呈 V 形：`VS0→VS1→VS2→VS3→VS7→VS6→VS5→VS4`。在 `T_F ≈ T_B ≈ T_W` 时实现零气泡。

**各调度策略气泡率总结**：

| 调度策略 | 气泡时间 | 条件 |
|---------|---------|------|
| **GPipe** | `(p-1) × (T_F + T_B)` | — |
| **1F1B** | `(p-1) × (T_F + T_B)` | 峰值内存更低 |
| **Interleaved 1F1B** | `(p-1) × (T_F + T_B) / v` | v = virtual stages |
| **ZB-H1** | `(p-1) × T_W` | ≈ 1F1B 的 1/3 |
| **ZB-H2** | 接近 0 | 需要更多内存 |
| **ZB-V** | 0 (当 T_F = T_B = T_W) | 2 virtual stages |

### 6.7 DualPipe (DeepSeek-V3)

**DualPipe** 是 DeepSeek-V3/R1 训练中使用的双向流水线并行算法，专门设计用于 **重叠计算与通信**，特别是应对 MoE 模型中大量的跨节点 AllToAll 通信。

#### 核心思想

1. **将每个 chunk 分解为 4 个组件**：
   - **ATTN**：注意力计算（纯计算）
   - **DISPATCH**：AllToAll 分发，将 token 路由到专家（通信）
   - **MLP**：专家/FFN 计算（纯计算）
   - **COMBINE**：AllToAll 合并，汇聚专家输出（通信）

2. **双向调度**：从流水线的**两端**同时送入 micro-batch，正向和反向流交错执行。

3. **计算-通信重叠**：一个 micro-batch 的通信（DISPATCH/COMBINE）与另一个 micro-batch 的计算（ATTN/MLP）重叠执行。

```
DualPipe 双向调度示意 (p=4 stages)

                    正向流 (micro-batch 从 stage 0 → stage 3)
                    ──────────────────────────────────────▶
GPU 0: │F→│F→│F→│...│B→│B→│...│                        │W│W│W│...
GPU 1: │  │F→│F→│...│   │B→│...│   │F←│...│B←│...│     │W│W│W│...
GPU 2: │  │  │F→│...│   │   │...│F←│F←│...│B←│B←│...│  │W│W│W│...
GPU 3: │                        │F←│F←│F←│...│B←│B←│B←│...│W│W│W│...
                    ◀──────────────────────────────────────
                    反向流 (micro-batch 从 stage 3 → stage 0)

    F→ = 正向流的前向    F← = 反向流的前向
    B→ = 正向流的反向    B← = 反向流的反向
    W  = 权重梯度计算
```

#### 计算-通信重叠细节

```
一对 forward + backward chunk 的 4 个时间步:

  Step 1: ATTN(fwd) [计算] + COMBINE(bwd) [通信]   ← 重叠
  Step 2: DISPATCH(fwd) [通信] + MLP(bwd) [计算]    ← 重叠
  Step 3: MLP(fwd) [计算] + DISPATCH(bwd) [通信]    ← 重叠
  Step 4: COMBINE(fwd) [通信] + ATTN(bwd) [计算]    ← 重叠

  每个时间步都同时有计算和通信 → full overlap
```

#### DualPipe 气泡与资源

**气泡时间**：

```
DualPipe Bubble = (p/2 - 1) × T_{F&B}

其中 T_{F&B} = 一对重叠的 forward+backward chunk 的执行时间
```

对比 1F1B 的 `(p-1) × (T_F + T_B)`，DualPipe 在 p 较大时气泡减少约 **50%+**（考虑到 T_{F&B} < T_F + T_B 因为重叠）。

**内存代价**：需要 **2x 参数内存**（因为双向流各自需要一份参数），激活内存为 p+1 个 micro-batch。

### 6.8 通信模式分析

```
通信拓扑: Stage 0 ──send──▶ Stage 1 ──send──▶ Stage 2 ──send──▶ Stage 3
                  ◀──recv──         ◀──recv──         ◀──recv──
前向: 左→右传递激活    反向: 右→左传递梯度
通信量/次 = B × S × H × sizeof(dtype)   例: 1×4096×8192×2B = 64 MB
```

**PP vs TP 通信对比**：

| 特性 | PP | TP |
|------|----|----|
| 通信原语 | Send/Recv (点对点) | AllReduce / AllGather |
| 每层通信次数 | 0（只在 stage 边界） | 2 次 AllReduce (前向) |
| 对带宽要求 | 低（PCIe 即可） | 高（需要 NVLink） |
| 延迟影响 | 增加 pipeline latency | 增加每层延迟 |

### 6.9 PP 的设计选择

#### Stage 数量

Stage 数量 p 决定了**内存节省** (`1/p`) 与**气泡开销** (`(p-1)/(m+p-1)`) 的权衡。

**选择原则**：
- **p 尽可能小**：满足内存需求即可，不要过度切分
- **m >> p**：确保 micro-batch 数远大于 stage 数
- **典型配置**：p = 2, 4, 8。p > 16 时气泡开销通常不可接受（除非使用 Zero Bubble/DualPipe）

#### 负载均衡

不同层的计算量可能不同（例如 MoE 层 vs Dense 层、第一层有 embedding、最后一层有 LM head）。**不均衡的 stage 划分会放大气泡**——最慢的 stage 决定整体速度。

**策略**：
- **Profiling**：实测每层的计算时间，按计算量均匀划分
- **Megatron-LM `--pipeline-model-parallel-layout`**：支持灵活的层分配
- **DeepSeek-V3**：对于 61 层 decoder + 1 层 MTP，使用 PP=16, VPP=2，自定义 layout

#### Micro-batch 大小选择

micro-batch 大小 b 影响：
1. **计算效率**：b 太小 → GPU 利用率低（kernel launch overhead 显著）
2. **气泡率**：m = B/b，m 越大气泡越小，但 b 越小
3. **激活内存**：每个 in-flight micro-batch 都需要存储激活

**实践建议**：
- 训练：b 选择使单个 stage 的 GPU 计算效率最高的值，然后调整 B 使 m >= 4p
- 推理：b 通常等于当前 batch 中的请求数，PP 的优化空间有限

#### PP 用于训练 vs 推理

| 维度 | 训练 | 推理 |
|------|------|------|
| **主要目标** | 减少气泡、提高训练吞吐 | 降低延迟、提高服务吞吐 |
| **micro-batch 数** | 可自由调节 (m >> p) | 受限于当前并发请求数 |
| **气泡问题** | 通过高 m 缓解 | 低并发时气泡严重 |
| **反向传播** | 有（B+W 都需要） | 无（只有前向） |
| **调度复杂度** | 高（1F1B, ZB, DualPipe） | 低（简单的前向流水线） |
| **内存** | 参数 + 优化器 + 激活 | 参数 + KV cache |

### 6.10 PP 在推理中的应用

#### 何时选择 PP 而非 TP

```
决策树:

模型超出单 GPU?
  ├── 否 → 单 GPU 推理 (可能 + 量化)
  └── 是 → 有 NVLink?
          ├── 是 → 低延迟需求?
          │       ├── 是 → TP (NVLink domain 内)
          │       └── 否 → TP + PP 或纯 PP (高吞吐)
          └── 否 (仅 PCIe) → PP 或 DP
```

**PP 推理的适用场景**：
1. **PCIe 系统**：没有 NVLink 时，TP 的通信开销过大（40-50%），PP 只需要 P2P
2. **跨节点部署**：多节点间只有网络连接，PP 的点对点通信更适合
3. **高并发吞吐场景**：大量请求可以填满流水线，掩盖气泡
4. **成本优化**：PP 不要求昂贵的 NVLink 硬件

#### 推理中的 Pipeline Bubble

推理只有前向计算，没有反向传播。但单个请求仍然需要依次经过所有 stage，导致**sequential latency**。

```
推理中的 PP (单请求):

GPU 0: │████ F ████│                                    │
GPU 1: │            │████ F ████│                        │
GPU 2: │            │            │████ F ████│            │
GPU 3: │            │            │            │████ F ████│

延迟 = p × T_stage  (串行，无法并行)
单请求延迟比 TP 差
```

**高并发时的 PP 推理**：

```
多请求流水线 (PP=4, 多个请求):

GPU 0: │F_R1│F_R2│F_R3│F_R4│F_R5│F_R6│...
GPU 1: │    │F_R1│F_R2│F_R3│F_R4│F_R5│...
GPU 2: │    │    │F_R1│F_R2│F_R3│F_R4│...
GPU 3: │    │    │    │F_R1│F_R2│F_R3│...

稳态时: 每 T_stage 时间完成一个请求 → 吞吐 = 1/T_stage
```

### 6.11 代码示例

#### PyTorch Pipeline Parallelism

```python
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe, Schedule1F1B

# 每个 rank 只实例化自己负责的层
if rank == 0:
    stage_module = nn.Sequential(model.embed, *model.layers[:8])
elif rank == 1:
    stage_module = nn.Sequential(*model.layers[8:16])
elif rank == 2:
    stage_module = nn.Sequential(*model.layers[16:24])
else:
    stage_module = nn.Sequential(*model.layers[24:], model.head)

stage = PipelineStage(stage_module, stage_index=rank,
                      num_stages=world_size, device=f"cuda:{rank}")

schedule = ScheduleGPipe(stage, n_microbatches=8)  # 或 Schedule1F1B
if rank == 0:
    schedule.step(input_batch)
else:
    output = schedule.step()
```

#### Megatron-LM PP 配置

```bash
# 基础 PP: TP=4, PP=8
--pipeline-model-parallel-size 8 --tensor-model-parallel-size 4

# Virtual Pipeline (Interleaved 1F1B): 每 GPU 2 个 model chunk
--pipeline-model-parallel-size 8 --virtual-pipeline-model-parallel-size 2

# DeepSeek-V3 自定义 layout (PP=16, VPP=2)
--pipeline-model-parallel-size 16 --virtual-pipeline-model-parallel-size 2 \
--pipeline-model-parallel-layout "Et*3|(tt|)*29,m|L"
```

#### 推荐 PP Stage 数量

```python
def recommend_pp_stages(model_params_gb, gpu_memory_gb, kv_cache_gb, overhead=1.3):
    """推荐 PP stage 数量 — 满足内存约束的最小 2 的幂"""
    available = gpu_memory_gb - kv_cache_gb
    min_gpus = math.ceil(model_params_gb * overhead / available)
    pp = 1
    while pp < min_gpus:
        pp *= 2
    return pp

# 例: Llama-3-70B (140GB FP16) on 80GB H100, KV≈10GB → pp=4
```

### 6.12 不足

| 不足 | 说明 |
|------|------|
| **Pipeline Bubble** | 无论多先进的调度策略，都很难完全消除气泡 |
| **负载不均衡** | 不同 stage 的计算量差异会放大气泡 |
| **单请求延迟差** | PP 增加了串行的 stage 间延迟 |
| **调度复杂度** | 高级策略（ZB、DualPipe）的实现和调试难度大 |
| **内存碎片化** | 每个 stage 只有部分层，可能导致 GPU 利用率不均 |

---

## 7. CP — 上下文并行 (Context Parallelism)

### 7.1 为什么需要 CP

随着 LLM 的上下文窗口从 4K 扩展到 128K、1M 甚至 10M tokens，单张 GPU 的内存已经无法容纳注意力计算所需的 KV 缓存和中间激活：

```
KV Cache 内存 (单请求):
  = 2 × n_layers × n_kv_heads × d_head × seq_len × sizeof(dtype)

  例: Llama-3-70B, seq_len=1M, FP16
  = 2 × 80 × 8 × 128 × 1,000,000 × 2 bytes
  = 327.68 GB  ← 远超单 GPU 内存

注意力计算复杂度: O(S²) — 序列长度翻倍，计算量翻 4 倍
```

**CP 的核心思想**：将**序列维度**切分到多个 GPU 上，每个 GPU 只处理序列的一部分，通过通信协作完成完整的注意力计算。

### 7.2 Ring Attention

**Ring Attention** (Liu et al., ICLR 2024) 是 CP 最经典的实现方式，灵感来自 Ring AllReduce 的环形通信模式。

#### 工作原理

1. 将序列均匀切分为 N 段，分配到 N 个 GPU
2. 每个 GPU 持有自己的 Q（Query）段，以及当前轮次的 KV 段
3. 在 **ring 拓扑** 中，KV 块沿环传递，每 GPU 每轮计算一次部分注意力
4. N 轮后，每个 GPU 完成了对所有 KV 的注意力计算

```
Ring Attention (4 GPUs, 序列切分为 4 段)

初始状态:
  GPU 0: Q₀, KV₀    GPU 1: Q₁, KV₁    GPU 2: Q₂, KV₂    GPU 3: Q₃, KV₃

Round 1: 本地计算 Attn(Qᵢ, KVᵢ)
  GPU 0: Attn(Q₀,KV₀)  GPU 1: Attn(Q₁,KV₁)  GPU 2: Attn(Q₂,KV₂)  GPU 3: Attn(Q₃,KV₃)
  同时: KV₀→GPU1, KV₁→GPU2, KV₂→GPU3, KV₃→GPU0  (ring 传递)

Round 2: 接收邻居的 KV，计算 Attn(Qᵢ, KVᵢ₋₁)
  GPU 0: Attn(Q₀,KV₃)  GPU 1: Attn(Q₁,KV₀)  GPU 2: Attn(Q₂,KV₁)  GPU 3: Attn(Q₃,KV₂)
  同时: KV 继续沿 ring 传递

Round 3: ...
Round 4: 最后一轮

                 ┌─────────────────────────────────────┐
                 │          Ring Topology               │
                 │                                      │
                 │    GPU 0 ───KV──▶ GPU 1              │
                 │      ▲                │              │
                 │      │KV            KV│              │
                 │      │                ▼              │
                 │    GPU 3 ◀──KV─── GPU 2              │
                 │                                      │
                 └─────────────────────────────────────┘
```

#### 通信-计算重叠

Ring Attention 的关键优化：**当第 i 轮的 KV 正在传输时，GPU 同时在用第 i-1 轮收到的 KV 做注意力计算**。

```
GPU 0 的时间线:

Round 1:  │ Compute: Attn(Q₀, KV₀) │ Send: KV₀ → GPU1  │
Round 2:  │ Compute: Attn(Q₀, KV₃) │ Send: KV₃ → GPU1  │  ← 计算和通信重叠!
Round 3:  │ Compute: Attn(Q₀, KV₂) │ Send: KV₂ → GPU1  │
Round 4:  │ Compute: Attn(Q₀, KV₁) │                    │

只要: T_compute >= T_communicate, 通信可以被完全隐藏
条件: 每段序列足够长 (S/N 足够大)，使计算时间 > 传输时间
```

**Ring Attention 的特点**：
- 内存：每 GPU 的序列内存 ∝ S/N（线性扩展）
- 计算：总 FLOPs 不变，与标准注意力一致
- 通信：每轮传递 KV 块，共 N-1 轮，但与计算重叠
- 上下文长度可随 GPU 数量线性扩展

**实际成果**：
- Meta：1M tokens 在单台 H100 主机上 <1 分钟；10M tokens 在 32 台主机上 <1 分钟
- RingX (SC'24)：在 Frontier 超算 4096 GPU 上训练 Llama3 8B 1M 序列，达到 38% MFU

### 7.3 Ulysses (DeepSpeed)

**DeepSpeed-Ulysses** (Jacobs et al., 2023) 采用与 Ring Attention 不同的方法：基于 **AllToAll** 通信实现序列并行。

#### 工作原理

1. 序列在 GPU 间沿序列维度切分（每 GPU 持有 S/N 个 token）
2. 在注意力计算前，通过 **AllToAll** 重新排列：
   - 输入：每 GPU 持有所有 head、部分序列
   - 输出：每 GPU 持有部分 head、完整序列
3. 每个 GPU 对自己的 head 子集执行**完整序列的标准注意力**
4. 注意力计算后，再通过 **AllToAll** 恢复原始分布

```
Ulysses AllToAll 模式 (4 GPUs, 8 heads)

Step 1: 初始分布 — 序列切分
  GPU 0: [seq 0:S/4, heads 0-7]     ← 所有 head, 部分序列
  GPU 1: [seq S/4:S/2, heads 0-7]
  GPU 2: [seq S/2:3S/4, heads 0-7]
  GPU 3: [seq 3S/4:S, heads 0-7]

Step 2: AllToAll — 转换为 head 切分
  GPU 0: [seq 0:S, heads 0-1]       ← 部分 head, 完整序列
  GPU 1: [seq 0:S, heads 2-3]
  GPU 2: [seq 0:S, heads 4-5]
  GPU 3: [seq 0:S, heads 6-7]

Step 3: 标准注意力 (每 GPU 独立计算自己的 head)
  GPU 0: Attn(Q₀₋₁, K₀₋₁, V₀₋₁)   ← 用 FlashAttention, 完整序列
  GPU 1: Attn(Q₂₋₃, K₂₋₃, V₂₋₃)
  ...

Step 4: AllToAll — 恢复序列切分
  GPU 0: [seq 0:S/4, heads 0-7]
  ...
```

#### Ring Attention vs Ulysses 对比

| 特性 | Ring Attention | Ulysses |
|------|---------------|---------|
| 通信原语 | P2P Send/Recv (ring) | AllToAll |
| 通信轮数 | N-1 轮 | 2 次 AllToAll |
| 可否与计算重叠 | 是（主要优势） | 否（AllToAll 是阻塞的） |
| 对注意力实现的侵入性 | 高（需修改注意力 kernel） | 低（只在注意力前后加通信） |
| 适用带宽条件 | 低带宽也可工作 | 需要较高带宽 |
| GPU 数限制 | 无（只需 P2P） | 受限于 head 数 (N ≤ n_heads) |
| 与 FlashAttention 兼容 | 需要适配 | 直接兼容 |

**Hybrid CP**：实践中可以结合两者——在节点内使用 Ulysses（带宽高），跨节点使用 Ring Attention（可隐藏延迟）。PyTorch 的 Context Parallel 实现就支持这种混合模式。详见 [[#7.6 现在常用的策略 —— 混合 CP|§7.6 混合 CP]]。

### 7.4 Megatron CP（Megatron-LM 的实现）

Megatron 的 CP 本质是 **Ring Attention 加上三项生产级优化**，是 NVIDIA / Megatron-LM 系（NeMo、NeMo-Megatron-Core、NVIDIA 风格的 Llama 训练栈）长上下文训练的事实标准。

**优化 1 —— Causal mask 的 zigzag 负载均衡**

Plain Ring + causal mask 下 GPU 0 几乎不算东西、GPU $N{-}1$ 几乎全算。Megatron 把序列切成 $2N$ 块（不是 $N$ 块），再把"靠前"和"靠后"的块配对发给同一张 GPU：

```
Plain ring (N=4 GPU):
  GPU0: [block 0]        ← causal 下几乎不算
  GPU1: [block 1]
  GPU2: [block 2]
  GPU3: [block 3]        ← causal 下几乎全算

Zigzag (N=4 GPU, 切 8 块):
  GPU0: [block 0, block 7]    ← 一前一后，负载均衡
  GPU1: [block 1, block 6]
  GPU2: [block 2, block 5]
  GPU3: [block 3, block 4]
```

每张 GPU 同时持有一个"靠前"块和一个"靠后"块，causal 下每张卡的计算量大致相当。相对 plain Ring + causal，有效吞吐近乎翻倍。

**优化 2 —— Ring P2P 嵌入 FlashAttention kernel 内部**

不是在框架层调度 Ring P2P（把 FlashAttention 当黑盒），而是直接改 FlashAttention 的 tile 循环，在 tile 之间发 `send` / `recv`。这才是真正能做到计算 / 通信重叠的实现 —— 框架层版本在 iteration 之间常会卡住。

**优化 3 —— 与 TP / PP / DP / SP / EP 一等公民式集成**

CP 作为另一个并行维度通过 `--context-parallel-size <N>` 暴露。框架自动构造正确的 NCCL 通信组、管理 KV cache 跨 CP 分片、处理 CP 边界的 mask、与其他 4 个维度自动组合。这些 plumbing 你一行不用写。

这就是为什么 DeepSeek-V3、Llama-3-405B 长上下文版、NVIDIA Nemotron 长上下文变体都用 Megatron-CP 而不是自己造轮子。

### 7.5 CP vs TP 处理长序列

| 维度 | CP | TP |
|------|----|----|
| 切分什么 | 序列 (每 GPU 持有 S/N tokens) | 权重 (每 GPU 持有 W/N) |
| KV cache | 切分（每 GPU 只存 S/N 的 KV） | 复制（每 GPU 存完整 KV）|
| 通信时机 | 仅在注意力层 | 每个 transformer 层 |
| 适用场景 | 超长序列 (>128K) | 常规序列 |
| 扩展限制 | 可扩展到极大 N | N ≤ NVLink domain (通常 8) |

**关键洞察**：对于长序列，TP 的 KV cache 复制是巨大的内存浪费。如果 8 卡 TP，每张卡都存完整的 1M token KV cache，而 CP=8 时每张卡只存 125K token 的 KV cache。

### 7.6 现在常用的策略 —— 混合 CP

2024–2026 的生产部署里没人单独用某一种 CP 实现，混合策略才是规范：

```
节点内 (≤ 8 GPU):  Ulysses
   → NVLink AllToAll 极快；head 数通常够。

节点间:             Ring / Megatron-CP
   → IB / RoCE 上的 P2P 扩展性好；计算/通信重叠把延迟藏掉。

组合:  hybrid Ulysses (节点内) × Ring (节点间)
   → Ulysses=8 × Ring=4 = 4 节点 32 路 CP。
   → 1M-token 训练、256 GPU 量级部署的常见配置。
```

这个混合模式的具体实现：

- **Tencent USP** (Unified Sequence Parallel) —— 显式提出混合模式的论文。
- **Megatron-LM CP** 的 `--context-parallel-size` 把两种变体都封进去，根据通信拓扑自动选。
- **PyTorch 原生 Context Parallel**（较新 PyTorch 版本）把 Ring + Ulysses 暴露成可组合的 backend。
- **SGLang 长上下文模式** + **xDiT**（视频 diffusion）都用混合模式。
- **FlashAttention-3** 自带匹配这种混合的分布式原语。

**训练 vs 推理**：CP 历史上是 *训练* 技术 —— 激活和梯度内存随 $S$ 暴涨，最先压垮单卡。在 *推理* 里，KV cache 压缩（[[saw-int4|SAW-INT4]]、MLA、KV 裁剪）和 [[prefill-decode-disaggregation|PD 分离]] 通常是更早的选择，不一定先考虑 CP。**Meta** 是唯一公开演示了推理 CP 的前沿实验室：["Context Parallelism for Scalable Million-Token Inference" (arXiv:2411.01783)](https://arxiv.org/abs/2411.01783) 展示 Llama 3 405B 在 128 张 H100 上 77 秒完成 1M token prefill，用 **pass-KV / pass-Q 混合** ring attention。引擎层面今天的支持：**vLLM** 出了 *Decode Context Parallel (DCP)* —— 一个更窄的"仅 KV 去重"变体，通过 `--decode-context-parallel-size` 开启（PR [#24864](https://github.com/vllm-project/vllm/pull/24864), 2025-10）—— 但完整 prefill CP 还在开发中（RFC [#22693](https://github.com/vllm-project/vllm/issues/22693)、[#26133](https://github.com/vllm-project/vllm/issues/26133) 都被 close 了）。**TensorRT-LLM** 把 `context_parallel_size` 作为一等公民 API 暴露（[文档](https://nvidia.github.io/TensorRT-LLM/features/parallel-strategy.html)）。**SGLang** 对百万 token 服务明确选了流水线并行而不是 CP（[blog](https://www.lmsys.org/blog/2026-01-15-chunked-pipeline/), 2026-01）。**其它前沿实验室（Google Gemini、Anthropic Claude、OpenAI GPT-4）没公开过自己的长上下文 serving 是否用 CP** —— 任何"它们用 CP"的说法都是未证实猜测。

### 7.7 不足

| 不足 | 说明 |
|------|------|
| **通信开销** | 序列太短时，通信时间无法被计算隐藏 |
| **因果注意力的三角性** | causal attention 中，早期 token 的 Q 不需要访问后面的 KV，导致计算不均衡 |
| **GPU 数限制 (Ulysses)** | GPU 数不能超过注意力头数 |
| **实现复杂度** | 需要修改注意力实现（Ring Attention），或插入 AllToAll 通信（Ulysses） |
| **短序列无收益** | 当 S/N 过小时，通信成本 > 计算收益 |

---

## 8. EP — 专家并行 (Expert Parallelism)

### 8.1 先回顾 MoE 架构

MoE (Mixture of Experts) 将 Transformer 的 FFN 层替换为多个"专家"网络 + 一个路由器：

```
    输入 token x
         │
         ▼
    ┌─────────┐
    │  Router  │ ← 门控网络，为每个 token 选择 top-K 个专家
    └────┬────┘
         │ 路由决策: token → Expert IDs + 权重
         │
    ┌────┼────┬────┬────┬────┬────┐
    │    │    │    │    │    │    │
   E₀   E₁   E₂   E₃   E₄   E₅  E₆  E₇   ← 256个专家(如 DeepSeek-V3)
    │    │                   │
    ▼    ▼                   ▼
  (被选中的专家各自处理 token)
    │    │                   │
    └────┴───────────────────┘
              │ 加权求和
              ▼
         输出 = Σ gᵢ · Expertᵢ(x)
```

**关键特性**：每个 token 只激活少量专家（如 DeepSeek-V3 激活 8/256），实现了"总参数量巨大但每 token 计算量可控"。

### 8.2 EP 原理

EP 将**完整的专家网络**分配到不同 GPU 上。如果有 E 个专家、EP_size 张 GPU，则每张 GPU 持有 `E / EP_size` 个专家。

```
    假设：8个专家, EP_size=4

    GPU 0: [E₀, E₁]    GPU 1: [E₂, E₃]    GPU 2: [E₄, E₅]    GPU 3: [E₆, E₇]

    ┌─────────────────────────────────────────────────────────────┐
    │ Step 1: Router 在每张 GPU 上独立计算路由决策                    │
    │         token_0 → E₂, E₅   token_1 → E₀, E₇ ...           │
    │                                                             │
    │ Step 2: AllToAll DISPATCH — 把 token 发送到持有目标专家的 GPU    │
    │                                                             │
    │   GPU 0 ──token_0──→ GPU 1 (for E₂)                        │
    │   GPU 0 ──token_0──→ GPU 2 (for E₅)                        │
    │   GPU 1 ──token_1──→ GPU 0 (for E₀)                        │
    │   GPU 1 ──token_1──→ GPU 3 (for E₇)                        │
    │                                                             │
    │ Step 3: 每张 GPU 用本地专家处理收到的 token                    │
    │                                                             │
    │ Step 4: AllToAll COMBINE — 把结果发回原始 GPU                  │
    │                                                             │
    │   GPU 1 ──result──→ GPU 0 (token_0 的 E₂ 结果)              │
    │   GPU 2 ──result──→ GPU 0 (token_0 的 E₅ 结果)              │
    │                                                             │
    │ Step 5: 加权求和                                             │
    └─────────────────────────────────────────────────────────────┘
```

### 8.3 通信分析

每个 MoE 层有 **2 次 AllToAll**（dispatch + combine）：

| 指标 | 公式 |
|------|------|
| 每次 AllToAll 通信量（每 GPU） | `tokens × top_k × hidden_dim × dtype_bytes × (EP-1) / EP` |
| 每 MoE 层总通信量 | `2 × tokens × top_k × H × dtype × (EP-1) / EP` |

**AllToAll vs AllReduce 的关键区别**：AllToAll 的数据流是**动态、不规则**的——具体有多少数据在哪两张卡之间流动取决于 router 的决策，每个 batch 都不同。"热门"专家会导致不对称的通信量。

### 8.4 代码示例：EP 的 AllToAll 路由

```python
import torch
import torch.distributed as dist

def expert_parallel_forward(
    hidden_states,    # [num_tokens, hidden_dim] — 本 GPU 的 token
    router_logits,    # [num_tokens, num_experts] — 路由分数
    local_experts,    # nn.ModuleList — 本 GPU 持有的专家
    ep_group,         # 通信组
    num_experts,      # 总专家数
    top_k=2,          # 每 token 选几个专家
):
    ep_size = dist.get_world_size(ep_group)
    ep_rank = dist.get_rank(ep_group)
    experts_per_gpu = num_experts // ep_size

    # ---- Step 1: 路由决策 ----
    scores = torch.softmax(router_logits, dim=-1)
    topk_weights, topk_indices = torch.topk(scores, top_k, dim=-1)
    # topk_indices: [num_tokens, top_k] — 每个 token 选中的专家 ID

    # ---- Step 2: 按目标 GPU 分组 token ----
    # 确定每个 token-expert pair 应发往哪张 GPU
    target_gpu = topk_indices // experts_per_gpu   # [num_tokens, top_k]

    # 构造 AllToAll 的发送计数
    send_counts = torch.zeros(ep_size, dtype=torch.long)
    for gpu_id in range(ep_size):
        send_counts[gpu_id] = (target_gpu == gpu_id).sum()

    # 交换计数信息 — 让每张卡知道要从其他卡接收多少 token
    recv_counts = torch.zeros_like(send_counts)
    dist.all_to_all_single(recv_counts, send_counts, group=ep_group)

    # ---- Step 3: AllToAll DISPATCH ----
    # 按目标 GPU 排列 token，然后执行 AllToAll
    # (简化示意，实际实现用 permutation indices)
    sorted_tokens = permute_tokens_by_target(hidden_states, topk_indices, target_gpu)
    received_tokens = all_to_all(sorted_tokens, send_counts, recv_counts, ep_group)

    # ---- Step 4: 本地专家计算 ----
    expert_outputs = torch.zeros_like(received_tokens)
    for i, expert in enumerate(local_experts):
        local_expert_id = ep_rank * experts_per_gpu + i
        mask = (received_expert_ids == local_expert_id)
        if mask.any():
            expert_outputs[mask] = expert(received_tokens[mask])

    # ---- Step 5: AllToAll COMBINE — 结果发回原始 GPU ----
    returned_outputs = all_to_all(expert_outputs, recv_counts, send_counts, ep_group)

    # ---- Step 6: 加权求和 ----
    final_output = weighted_sum(returned_outputs, topk_weights)
    return final_output
```

### 8.5 负载均衡策略

EP 最大的挑战是**负载不均衡**——如果某些专家被选中的频率远高于其他专家：

```
    理想情况（均衡）:              实际情况（不均衡）:
    E₀: ████  (25%)              E₀: ████████████ (60%)  ← 热门专家!
    E₁: ████  (25%)              E₁: ██ (10%)
    E₂: ████  (25%)              E₂: ██ (10%)
    E₃: ████  (25%)              E₃: ████ (20%)

    → 每 GPU 工作量相同             → GPU 0 成为瓶颈, 其他 GPU 空等
```

主流解决方案：

| 方法 | 说明 | 代表 |
|------|------|------|
| **辅助损失 (Auxiliary Loss)** | 在训练 loss 中加入均衡正则项 | GShard, Switch Transformer |
| **无辅助损失的偏置调整** | 动态调整路由偏置而非加 loss | DeepSeek-V3 |
| **容量因子 (Capacity Factor)** | 限制每个专家接收的最大 token 数，溢出丢弃 | Switch Transformer |
| **节点限制路由** | 限制 token 最多发往 M 个节点 | DeepSeek-V3 (M=4) |
| **冗余专家** | 热门专家在多张 GPU 上复制 | DeepSeek-V3 推理 |

### 8.6 使用场景

- MoE 模型的专家总参数超出单 GPU 内存
- 稀疏激活模型（如 DeepSeek-V3: 671B 总参数，每 token 仅激活 37B）
- 当单个专家足够小可以放进一张 GPU 时，**EP 优于 ETP**

### 8.7 不足

| 不足 | 说明 |
|------|------|
| **AllToAll 是同步阻塞点** | 所有 GPU 必须完成 token 交换才能继续 |
| **动态路由导致负载不均** | 热门专家的 GPU 成为瓶颈 |
| **跨节点 EP 通信昂贵** | 需要高带宽互联（InfiniBand） |
| **通信量随 EP 度线性增长** | EP 越大，AllToAll 涉及的 GPU 越多 |

---

## 9. EDP/DEP — 专家数据并行 (Expert Data Parallelism)

### 9.1 原理

EDP（也常写作 DEP）是在 MoE 层应用的数据并行：**多张 GPU 持有相同的专家副本，各自处理不同的 token 子集**。

这是一个容易被忽视但极其重要的概念。理解 EDP 的关键是看清它和 EP 的**正交关系**：

- **EP 维度**：在同一组 GPU 内，把不同专家**分散**到不同卡（AllToAll 路由）
- **EDP 维度**：把整个 EP 组**复制**多份，每份处理不同的数据（AllReduce 梯度同步）

```
    假设：8个专家, EP=4, EDP=2 → 共需 4×2=8 张 GPU

    ┌──────────── EDP 副本 0 ─────────────┐  ┌──────────── EDP 副本 1 ─────────────┐
    │                                      │  │                                      │
    │  GPU 0    GPU 1    GPU 2    GPU 3    │  │  GPU 4    GPU 5    GPU 6    GPU 7    │
    │  [E₀,E₁] [E₂,E₃] [E₄,E₅] [E₆,E₇] │  │  [E₀,E₁] [E₂,E₃] [E₄,E₅] [E₆,E₇] │
    │     │        │        │        │     │  │     │        │        │        │     │
    │     └────────┴────────┴────────┘     │  │     └────────┴────────┴────────┘     │
    │           AllToAll (EP 路由)          │  │           AllToAll (EP 路由)          │
    │                                      │  │                                      │
    │         处理 Batch A 的 token         │  │         处理 Batch B 的 token         │
    └──────────────────────────────────────┘  └──────────────────────────────────────┘
           │                                           │
           └─────────────── AllReduce ─────────────────┘
              专家梯度同步: GPU0↔GPU4, GPU1↔GPU5, GPU2↔GPU6, GPU3↔GPU7

    ★ EP 组内: AllToAll (token 路由到正确的专家)
    ★ EDP 组间: AllReduce (相同专家的梯度在不同副本间同步)
```

### 9.2 EDP 的通信组拓扑

理解 EDP 最重要的是理解**通信组是如何划分的**。以 16 GPU、EP=4、EDP=4 为例：

```
    GPU ID:  0   1   2   3  |  4   5   6   7  |  8   9  10  11  | 12  13  14  15
    专家:   E0  E1  E2  E3  | E0  E1  E2  E3  | E0  E1  E2  E3  | E0  E1  E2  E3
             │               │               │               │
             └── EP Group 0 ─┘               └── EP Group 2 ─┘
                              └── EP Group 1 ─┘               └── EP Group 3 ─┘

    EP  组 (AllToAll): {0,1,2,3}, {4,5,6,7}, {8,9,10,11}, {12,13,14,15}
    EDP 组 (AllReduce): {0,4,8,12}, {1,5,9,13}, {2,6,10,14}, {3,7,11,15}
                         ↑ 同一个专家(E₀)的 4 个副本      ↑ E₃ 的 4 个副本
```

**关键观察**：EP 组和 EDP 组是**正交**的——EP 组是行方向（同一行内分散不同专家），EDP 组是列方向（同一列内复制相同专家）。

### 9.3 与标准 DP 的区别

| 维度 | 标准 DP | EDP |
|------|---------|-----|
| 复制对象 | 整个模型 | 仅 MoE 专家部分 |
| 同步内容 | 所有参数梯度 | 仅专家参数梯度 |
| 作用范围 | Dense 层 (Attention + MLP) | MoE 层 |
| 可独立于 TP | 是 | 是（MoE Parallel Folding 后） |
| 通信发生时机 | 反向传播结束后 | 反向传播结束后（仅专家参数） |
| 与谁正交 | TP, PP | EP, ETP, PP |

**历史约束**：在 MoE Parallel Folding (NVIDIA, 2025) 之前，Megatron-LM 要求 `EP ≤ DP`，即 EP 只能是 DP 组的子集。这严重限制了 EP 的规模——如果 DP=8，那 EP 最多也只能是 8。Parallel Folding 移除了这一约束，允许 Dense 层和 MoE 层使用完全独立的并行配置。

### 9.4 MoE Parallel Folding 公式

```
Dense 层:  N_total = TP × SP × CP × DP × PP   (SP 通常 = TP)
MoE  层:  N_total = ETP × EP × EDP × PP

约束: PP 必须一致，其余维度完全独立

示例 (128 GPU):
  Dense:  TP=2, CP=2, PP=8 → DP  = 128/(2×2×8) = 4
  MoE:    ETP=1, EP=8, PP=8 → EDP = 128/(1×8×8) = 2

  → Attention 用 TP=2 切分注意力头, CP=2 切分长序列, DP=4 复制 4 份
  → MoE 用 EP=8 把专家分到 8 卡, 复制 2 份 (EDP=2)
  → 同样 128 张 GPU，但两种层的并行方式完全不同！
```

### 9.5 推理中的 "DP Attention + EP" 模式

EDP 在推理中的一个重要应用是 **DP Attention**（vLLM 的核心架构之一）。这种模式下：

```
    8 GPU, DeepSeek-R1, DP=8 + EP=8

    Attention 层 (DP=8):
    ┌────────────────────────────────────────────────────────────┐
    │  GPU 0      GPU 1      GPU 2      ...      GPU 7          │
    │  完整Attn   完整Attn   完整Attn            完整Attn        │
    │  KV分区0    KV分区1    KV分区2             KV分区7         │
    │  处理请求   处理请求   处理请求             处理请求         │
    │  {0,8,16}   {1,9,17}   {2,10,18}           {7,15,23}      │
    │                                                            │
    │  ★ 每张卡独立处理不同请求                                   │
    │  ★ KV Cache 按请求分区, 不重复! (vs TP 模式下 KV 被复制)    │
    │  ★ 需要 AllGather 汇集 KV 用于注意力计算                    │
    └────────────────────────────────────────────────────────────┘

    MoE 层 (EP=8):
    ┌────────────────────────────────────────────────────────────┐
    │  GPU 0      GPU 1      GPU 2      ...      GPU 7          │
    │  Expert     Expert     Expert              Expert          │
    │  {0-31}     {32-63}    {64-95}             {224-255}       │
    │                                                            │
    │  ★ AllToAll: 把所有 GPU 的 token 路由到持有目标专家的 GPU    │
    │  ★ 所有 GPU 的 token 混合在一起参与路由                     │
    └────────────────────────────────────────────────────────────┘

    为什么这比 TP+EP 好?（高并发时）
    ├── KV Cache 不重复 → 可以服务更多并发请求
    ├── Attention 无需 AllReduce → 减少通信
    └── 代价: 需要 AllGather 收集 KV，但在高并发时值得
```

### 9.6 通信量分析

| 阶段 | 通信原语 | 通信量（每 GPU） |
|------|---------|----------------|
| EP 组内 token 路由 (前向) | AllToAll | `tokens × top_k × H × dtype × (EP-1)/EP` |
| EP 组内结果返回 (前向) | AllToAll | 同上 |
| EDP 组间梯度同步 (训练) | AllReduce | `2 × expert_params_per_gpu × dtype` |
| DP Attention KV 汇集 (推理) | AllGather | `batch × seq × kv_dim × dtype` |

**EDP 的 AllReduce 通信量远小于全模型 DP**：因为只需要同步专家参数的梯度，而非全部参数。以 DeepSeek-V3 为例，专家参数约占总参数的 ~95%（636B/671B），但每张 GPU 只持有 `636B / EP` 的专家参数。

### 9.7 代码示例：EDP 通信组的创建

```python
import torch.distributed as dist

def create_edp_groups(world_size, ep_size, pp_size, etp_size=1):
    """
    创建 Expert Data Parallelism 通信组

    world_size = ETP × EP × EDP × PP
    EDP = world_size / (ETP × EP × PP)
    """
    edp_size = world_size // (etp_size * ep_size * pp_size)
    print(f"EDP size: {edp_size} (每组专家复制 {edp_size} 份)")

    ep_groups = []   # AllToAll 通信组
    edp_groups = []  # AllReduce 通信组

    for pp_rank in range(pp_size):
        base = pp_rank * (etp_size * ep_size * edp_size)

        # EP 组: 同一 EDP 副本内, 持有不同专家的 GPU
        for edp_rank in range(edp_size):
            for etp_rank in range(etp_size):
                ranks = []
                for ep_rank in range(ep_size):
                    r = base + edp_rank * (ep_size * etp_size) + ep_rank * etp_size + etp_rank
                    ranks.append(r)
                group = dist.new_group(ranks)
                ep_groups.append((ranks, group))

        # EDP 组: 持有相同专家的 GPU (跨不同副本)
        for ep_rank in range(ep_size):
            for etp_rank in range(etp_size):
                ranks = []
                for edp_rank in range(edp_size):
                    r = base + edp_rank * (ep_size * etp_size) + ep_rank * etp_size + etp_rank
                    ranks.append(r)
                group = dist.new_group(ranks)
                edp_groups.append((ranks, group))

    return ep_groups, edp_groups


# 示例: 128 GPU, EP=8, EDP=2, PP=8, ETP=1
ep_groups, edp_groups = create_edp_groups(
    world_size=128, ep_size=8, pp_size=8, etp_size=1
)
# EP 组 (AllToAll): 每组 8 GPU, 共 16 组 (2 EDP副本 × 8 PP阶段)
# EDP 组 (AllReduce): 每组 2 GPU, 共 64 组 (8 EP位置 × 8 PP阶段)
```

### 9.8 EDP 与 DP 的关系总结

```
    传统框架 (Parallel Folding 之前):
    ┌──────────────────────────────────┐
    │    DP 组 (比如 DP=8)              │
    │    ┌───────────────────────┐     │
    │    │ EP 子组 (EP ≤ DP)     │     │
    │    │ EP=4, 剩下 DP/EP=2   │     │
    │    │ 就是 EDP=2            │     │
    │    └───────────────────────┘     │
    │    ★ EP 被限制为 DP 的子集       │
    └──────────────────────────────────┘

    MoE Parallel Folding (2025):
    ┌─────────────────┐    ┌─────────────────┐
    │ Dense 层         │    │ MoE 层           │
    │ TP × SP × CP ×  │    │ ETP × EP × EDP × │
    │ DP × PP          │    │ PP                │
    │ (完全独立配置)    │    │ (完全独立配置)    │
    └─────────────────┘    └─────────────────┘
    ★ EP 可以任意大, 不受 DP 限制
    ★ 例如 Dense 用 TP=4,DP=4; MoE 用 EP=64,EDP=1
```

### 9.9 使用场景与不足

**使用场景**：

| 场景 | 说明 |
|------|------|
| **EP 度 < 总 GPU 数** | EDP 利用剩余 GPU 提升训练吞吐量 |
| **高并发推理 (DP Attention)** | 每张卡独立持有 KV Cache 分区，通过 EDP 扩大总并发 |
| **降低 EP 通信压力** | EDP 越大 → 每个 EP 组越小 → AllToAll 通信范围缩小 |

**不足**：

| 不足 | 说明 |
|------|------|
| **内存冗余** | 每个 EDP 副本都持有完整的专家参数副本 |
| **梯度同步开销** | EDP 组间需要 AllReduce 同步专家梯度 |
| **不减少单卡专家参数量** | 要减少每卡参数量需要增大 EP 或 ETP |

---

## 10. ETP/TEP — 专家张量并行 (Expert Tensor Parallelism)

### 10.1 原理

ETP（也常写作 TEP）在**单个专家内部**施加张量并行——本质上就是把第4节介绍的 TP 技术（列并行 + 行并行）应用到每个专家的 FFN 权重上。

**核心区别**：TP 切分的是 Attention 和 Dense MLP 的权重；ETP 切分的是 MoE 层中每个专家的权重。

```
    EP: 不同 GPU 持有不同的完整专家
    ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
    │ GPU 0    │  │ GPU 1    │  │ GPU 2    │  │ GPU 3    │
    │ Expert 0 │  │ Expert 1 │  │ Expert 2 │  │ Expert 3 │
    │ (完整)   │  │ (完整)   │  │ (完整)   │  │ (完整)   │
    └──────────┘  └──────────┘  └──────────┘  └──────────┘

    ETP: 同一个专家的权重切分到多张 GPU
    ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
    │ GPU 0    │  │ GPU 1    │  │ GPU 2    │  │ GPU 3    │
    │ Expert 0 │  │ Expert 0 │  │ Expert 1 │  │ Expert 1 │
    │ (左半)   │  │ (右半)   │  │ (左半)   │  │ (右半)   │
    └──────────┘  └──────────┘  └──────────┘  └──────────┘
    ←── ETP=2 ──→              ←── ETP=2 ──→
    ←──────────── EP=2 ──────────────────────→
```

### 10.2 ETP 内部的切分方式

ETP 对每个专家的 FFN 做的切分和 TP 完全一样——列并行 + 行并行：

```
    专家 E₀ 的 FFN (gate_proj + up_proj + down_proj):

    不用 ETP (ETP=1):
    ┌─────────────────────────────────────────────┐
    │  GPU 0 持有 E₀ 的全部权重:                    │
    │                                              │
    │  gate_proj: [hidden_dim, ffn_dim]  ← 完整    │
    │  up_proj:   [hidden_dim, ffn_dim]  ← 完整    │
    │  down_proj: [ffn_dim, hidden_dim]  ← 完整    │
    └─────────────────────────────────────────────┘

    用 ETP=2:
    ┌──────────────────────┐  ┌──────────────────────┐
    │  GPU 0 持有 E₀ 左半:  │  │  GPU 1 持有 E₀ 右半:  │
    │                      │  │                      │
    │  gate_proj:          │  │  gate_proj:          │
    │  [hidden, ffn/2]     │  │  [hidden, ffn/2]     │
    │  (列并行, 前半列)     │  │  (列并行, 后半列)     │
    │                      │  │                      │
    │  up_proj:            │  │  up_proj:            │
    │  [hidden, ffn/2]     │  │  [hidden, ffn/2]     │
    │                      │  │                      │
    │  down_proj:          │  │  down_proj:          │
    │  [ffn/2, hidden]     │  │  [ffn/2, hidden]     │
    │  (行并行, 前半行)     │  │  (行并行, 后半行)     │
    └──────────────────────┘  └──────────────────────┘

    ★ gate_proj, up_proj 用列并行 → SiLU 激活可独立施加
    ★ down_proj 用行并行 → 需要 AllReduce/ReduceScatter 求和
```

### 10.3 完整通信 Pipeline

ETP 的通信叠加在 EP 的 AllToAll 之上，形成一个 6 步的 pipeline：

```
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    一个 MoE 层的完整前向传播                         │
    │                                                                     │
    │  Step 1: Permutation                                                │
    │  ├── Router 为每个 token 选择 top-K 专家                            │
    │  └── 按目标专家 ID 对 token 排序                                    │
    │           │                                                         │
    │  Step 2: AllToAll-V (EP 维度)                                       │
    │  ├── 把 token 发送到持有目标专家的 GPU                               │
    │  └── 通信量: tokens × top_k × H × (EP-1)/EP                        │
    │           │                                                         │
    │  Step 3: AllGather-V (ETP 维度)                      ← ETP 独有!    │
    │  ├── ETP 组内每张卡收集完整的输入 token                              │
    │  └── 通信量: received_tokens × H × (ETP-1)/ETP                     │
    │           │                                                         │
    │  Step 4: Expert Compute (本地)                                      │
    │  ├── 列并行: partial_h = SiLU(x @ gate_shard) * (x @ up_shard)     │
    │  └── 行并行: partial_out = partial_h @ down_shard                   │
    │           │                                                         │
    │  Step 5: ReduceScatter-V (ETP 维度)                  ← ETP 独有!    │
    │  ├── 聚合行并行的部分结果并分发                                      │
    │  └── 通信量: 同 Step 3                                              │
    │           │                                                         │
    │  Step 6: AllToAll-V (EP 维度)                                       │
    │  ├── 把计算结果发回原始 GPU                                          │
    │  └── 通信量: 同 Step 2                                              │
    │           │                                                         │
    │  Step 7: Un-permutation + 加权求和                                   │
    │                                                                     │
    │  总通信: 2×AllToAll(EP) + AllGather(ETP) + ReduceScatter(ETP)       │
    │  对比纯EP: 2×AllToAll(EP)                                           │
    │  ★ ETP 额外增加了 2 次集合通信!                                      │
    └─────────────────────────────────────────────────────────────────────┘
```

### 10.4 通信量定量对比

假设：每个 MoE 层有 T 个 token（来自一个 micro-batch），隐藏维度 H，EP 度为 E，ETP 度为 P：

| 通信操作 | 通信量（每 GPU） | 来源 |
|---------|----------------|------|
| AllToAll dispatch (EP) | `T × top_k × H × (E-1)/E` | token 路由 |
| AllToAll combine (EP) | 同上 | 结果返回 |
| AllGather (ETP) | `T_recv × H × (P-1)/P` | 收集完整输入 |
| ReduceScatter (ETP) | `T_recv × H × (P-1)/P` | 聚合部分输出 |

其中 `T_recv = T × top_k / E` 是每个 EP 位置收到的 token 数。

**额外开销比例**：
```
Overhead = (P-1) / (P × (E-1))

当 E=8, P=2: Overhead = 1/(2×7) ≈ 7%    ← 看起来不多
当 E=4, P=4: Overhead = 3/(4×3) = 25%   ← 显著!
当 E=2, P=8: Overhead = 7/(8×1) = 87.5% ← 几乎翻倍!
```

**结论**：EP 越小、ETP 越大，额外开销越大。这就是为什么建议 "最大化 EP，最小化 ETP"。

### 10.5 代码示例：ETP 组的 AllGather + ReduceScatter

```python
import torch
import torch.distributed as dist

def expert_tp_forward(
    dispatched_tokens,   # [T_recv, H] 经 AllToAll 路由后到达本 ETP 组的 token
    gate_weight_shard,   # [H, ffn_dim/ETP] 列并行切片
    up_weight_shard,     # [H, ffn_dim/ETP] 列并行切片
    down_weight_shard,   # [ffn_dim/ETP, H] 行并行切片
    etp_group,           # ETP 通信组
):
    etp_size = dist.get_world_size(etp_group)

    # ---- Step 1: AllGather (ETP) ----
    # 让 ETP 组内每张卡都拿到完整的输入 token
    gathered = [torch.empty_like(dispatched_tokens) for _ in range(etp_size)]
    dist.all_gather(gathered, dispatched_tokens, group=etp_group)
    full_input = torch.cat(gathered, dim=0)  # [T_recv * ETP, H]

    # ---- Step 2: 列并行 FFN 计算 ----
    gate_out = torch.nn.functional.linear(full_input, gate_weight_shard.T)
    up_out   = torch.nn.functional.linear(full_input, up_weight_shard.T)
    hidden   = torch.nn.functional.silu(gate_out) * up_out

    # ---- Step 3: 行并行 down_proj ----
    partial_output = torch.nn.functional.linear(hidden, down_weight_shard.T)

    # ---- Step 4: ReduceScatter (ETP) ----
    chunks = list(partial_output.chunk(etp_size, dim=0))
    output = torch.empty_like(dispatched_tokens)
    dist.reduce_scatter(output, chunks, group=etp_group)

    return output
```

### 10.6 配置与实测

```bash
# Megatron-LM 训练配置
python pretrain_gpt.py \
    --num-experts 8 \
    --expert-model-parallel-size 4 \    # EP=4
    --expert-tensor-parallel-size 2 \   # ETP=2
    --tensor-model-parallel-size 4 \    # TP=4 (for attention)
    --pipeline-model-parallel-size 2 \  # PP=2
    --sequence-parallel                 # TP+EP 时必须启用

# TensorRT-LLM 推理配置
python convert_checkpoint.py \
    --tp_size 4 \
    --moe_tp_size 2 \    # ETP=2
    --moe_ep_size 2      # EP=2
```

**Megatron-LM 实测结果（MFU = Model FLOPS Utilization）**：

| 模型 | GPU数 | 配置 | MFU |
|------|------|------|-----|
| Mixtral 8x7B | 64 | EP=8, ETP=1, TP=2, PP=4 | **49.3%** |
| Mixtral 8x7B | 64 | EP=4, ETP=2, TP=2, PP=4 | 45.1% |
| Qwen2-57B-A14B | 64 | EP=4, ETP=1, TP=2, PP=4 | **39.0%** |
| Qwen2-57B-A14B | 64 | EP=2, ETP=2, TP=2, PP=4 | 35.7% |

### 10.7 经验法则 — 通常应设 ETP=1

```
    ★ 核心原则: 最大化 EP, 最小化 ETP

    细粒度 MoE (256 experts, small FFN dim):
    └── ETP = 1, 尽量增大 EP
        └── 因为: 每个专家太小, TP 切分后计算量不足以摊平通信

    粗粒度 MoE (8 experts, large FFN dim):
    └── 先尝试 ETP = 1
        └── 如果 OOM → 尝试 ETP = 2
            └── 如果还 OOM → ETP = 4 (最后手段)

    推理场景:
    └── 几乎总是 ETP = 1
        └── 因为: 不需要存梯度/优化器, 单卡内存足以放下专家
            └── 如果确实放不下 → 考虑量化而非 ETP

    如果 ETP > 1 不可避免:
    ├── 保持 ETP 组在 NVLink 域内 (AllGather/RS 需要高带宽)
    ├── EP × ETP ≤ 8 (一个节点)
    └── 确保 num_experts % (EP × ETP) == 0
```

---

## 11. DP Attention — 数据并行注意力 (Data-Parallel Attention for MoE Inference)

### 11.1 背景

在 MoE 模型推理中，传统的 TP+EP 组合存在一个关键问题：**TP 导致 KV Cache 在每张 GPU 上被完全复制**。

```
    TP+EP 模式 (8 GPU, DeepSeek-R1):
    Attention (TP=8): 每张 GPU 存完整 KV Cache → 8 份副本!
    MoE (EP=8):       每张 GPU 存不同专家 → 无冗余

    问题: KV Cache 内存 ∝ 并发请求数 × 序列长度
          8 份 KV Cache 副本 = 8× 内存浪费
          → 严重限制并发请求数
```

**DP Attention** (vLLM 2025) 的核心思想：将 Attention 层改为数据并行模式——**每张 GPU 只持有部分请求的 KV Cache，而非全部请求的完整副本**。

### 11.2 工作原理

```
DP Attention + EP 架构 (8 GPU):

┌─────────────────── Attention 层 (DP=8) ──────────────────┐
│                                                           │
│  GPU 0         GPU 1         GPU 2        ...  GPU 7      │
│  ┌──────┐     ┌──────┐     ┌──────┐          ┌──────┐    │
│  │Attn  │     │Attn  │     │Attn  │          │Attn  │    │
│  │模型  │     │模型  │     │模型  │          │模型  │    │
│  │(完整) │     │(完整) │     │(完整) │          │(完整) │    │
│  ├──────┤     ├──────┤     ├──────┤          ├──────┤    │
│  │KV for│     │KV for│     │KV for│          │KV for│    │
│  │Req   │     │Req   │     │Req   │          │Req   │    │
│  │0,8,16│     │1,9,17│     │2,10  │          │7,15  │    │
│  └──────┘     └──────┘     └──────┘          └──────┘    │
│                                                           │
│  ★ 每张 GPU 持有完整的 Attention 模型参数                  │
│  ★ 但只存 1/8 请求的 KV Cache → 不重复!                    │
│  ★ 独立计算，Prefill 时无需通信                            │
│  ★ Decode 时可能需要 AllGather 跨 GPU 的 KV (取决于实现)   │
└───────────────────────────────────────────────────────────┘
                            │
                    AllToAll (token 路由)
                            │
┌─────────────────── MoE 层 (EP=8) ────────────────────────┐
│                                                           │
│  GPU 0         GPU 1         GPU 2        ...  GPU 7      │
│  Expert        Expert        Expert            Expert     │
│  {0-31}        {32-63}       {64-95}           {224-255}  │
│                                                           │
│  ★ 所有 GPU 的 token 混合在一起，通过 AllToAll 路由         │
└───────────────────────────────────────────────────────────┘
                            │
                    AllToAll (结果返回)
                            │
                    Attention 层 (下一层)
```

### 11.3 与传统 DP 的关键区别

| 维度 | 传统 DP (训练) | DP Attention (推理) |
|------|--------------|-------------------|
| 模型参数 | 完全复制 | 完全复制（Attention 部分） |
| KV Cache | N/A (训练无 KV Cache) | **分区**，每 GPU 只持有 1/N 请求的 KV |
| 梯度同步 | AllReduce | 无（推理不需要） |
| 请求分配 | 数据 batch 均匀分 | 请求按 round-robin 分配到不同 GPU |
| 与 EP 配合 | 独立 | 紧密配合：DP Attention → AllToAll → EP → AllToAll → DP Attention |

### 11.4 为什么对 MoE 推理至关重要

```
内存对比 (DeepSeek-R1, 8 GPU, 1000 并发请求, 平均 seq_len=4096):

TP=8 + EP=8:
  KV Cache / GPU = 1000 × 4096 × 2 × n_layers × kv_dim × 2B
                 = 1000 份完整的 KV (每 GPU 都存!)
  ★ 内存瓶颈: KV Cache 远大于模型参数

DP=8 + EP=8 (DP Attention):
  KV Cache / GPU = 125 × 4096 × 2 × n_layers × kv_dim × 2B
                 = 只存 125 份 KV (1000/8)
  ★ KV Cache 减少 8x → 可以服务 8x 的并发请求!
```

这就是为什么在高并发 MoE 推理中，DP Attention + EP 优于 TP + EP。

### 11.5 通信模式

```
DP Attention + EP 的通信流:

  Attention 层:
    Prefill: 无通信 (每 GPU 独立处理各自的请求)
    Decode:  可能需要 AllGather KV (取决于请求是否需要跨 GPU 的 KV)
             通信量: batch × seq × kv_dim × dtype

  Attention → MoE 过渡:
    AllToAll (dispatch): 把所有 GPU 的 token 汇集并路由到专家
    通信量: total_tokens × top_k × hidden_dim × dtype × (EP-1)/EP

  MoE 层:
    本地专家计算 (无通信)

  MoE → Attention 过渡:
    AllToAll (combine): 把结果发回各 GPU
    通信量: 同 dispatch

总通信: 主要来自 EP 的 AllToAll，而非 Attention
```

### 11.6 vLLM 配置与性能

```bash
# DP Attention + EP 模式 (vLLM)
vllm serve deepseek-ai/DeepSeek-R1 \
    --tensor-parallel-size 1 \
    --data-parallel-size 8 \
    --enable-expert-parallel
    # DP=8 for attention, EP=8 for MoE

# 对比: TP + EP 模式
vllm serve deepseek-ai/DeepSeek-R1 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel
    # TP=8 for attention, EP=8 for MoE
```

**性能对比**：

```
    吞吐量 (tokens/s)      DP Attention + EP vs TP + EP
    │
    │                                          DP+EP ●
    │                                       ●
    │                                    ●
    │                              ●
    │                        ●                      ← DP+EP 在高并发时
    │                  TP+EP ●─ ─ ─ ─●─ ─ ─ ─●       领先 ~47%
    │              ●
    │         ●                                     ← TP+EP 在低并发时
    │    ●                                            领先 ~52%
    │●
    └────────────────────────────────────────────→
    1    16   64   128  256  512  1024  2048  4096
                      并发请求数

    交叉点: ~256-512 并发
```

### 11.7 适用场景与限制

**适用场景**：
- MoE 模型的高并发推理（>256 并发请求）
- 内存受限场景（KV Cache 是瓶颈）
- DeepSeek-V3/R1 等大规模 MoE 模型

**限制**：
- 低并发时延迟不如 TP+EP
- Attention 参数被完全复制（但通常 Attention 参数远小于 Expert 参数）
- 需要框架支持（vLLM 2025 已原生支持）

---

## 12. 混合并行策略：如何组合

### 12.1 总 GPU 资源的分解公式

```
                  ┌──────────────────────────────────────────┐
                  │          N_total 张 GPU                   │
                  │                                          │
                  │  Dense 层: N = TP × SP × CP × DP × PP    │
                  │           (SP 通常 = TP)                  │
                  │  MoE  层: N = ETP × EP × EDP × PP        │
                  │                                          │
                  │  约束: PP 必须在两者间保持一致              │
                  └──────────────────────────────────────────┘
```

### 12.2 通信模式汇总

```
    ┌──────────────────────────────────────────────────────────────────┐
    │                     一个 Transformer Block                       │
    │                                                                  │
    │  ┌─────────────────┐     ┌──────────────────────────────────┐   │
    │  │   Attention 层   │     │           MoE 层                  │   │
    │  │                 │     │                                    │   │
    │  │  TP: AllReduce  │     │  EP:  AllToAll (dispatch)         │   │
    │  │  SP: AG + RS    │     │       Expert Compute              │   │
    │  │  CP: Ring/A2A   │     │       AllToAll (combine)          │   │
    │  │  DP: (无, 推理) │     │  ETP: AllGather + ReduceScatter   │   │
    │  │      AllReduce  │     │  EDP: AllReduce (训练梯度)         │   │
    │  │      (训练梯度) │     │                                    │   │
    │  │                 │     │                                    │   │
    │  │  通信频率: 每层  │     │  通信频率: 每 MoE 层               │   │
    │  │  带宽需求: 极高  │     │  带宽需求: 高 (AllToAll 量大)      │   │
    │  └─────────────────┘     └──────────────────────────────────┘   │
    │                                                                  │
    │  PP: 只在 stage 边界有点对点通信 (最低频)                         │
    └──────────────────────────────────────────────────────────────────┘
```

### 12.3 典型组合模式

#### 模式 A: TP + EP（低并发推理）

```
    8 GPU (1 node, NVLink)

    Attention: TP=8 (切分到 8 卡)
    MoE:       EP=8 (8 个专家组, 每卡 num_experts/8 个专家)

    特点: 单请求延迟最低
    通信: Attention 用 AllReduce, MoE 用 AllToAll
    适用: 延迟敏感场景, 并发 < 256
```

```bash
# vLLM 配置
vllm serve deepseek-ai/DeepSeek-R1 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel
```

#### 模式 B: DP Attention + EP（高并发推理）

```
    8 GPU (1 node, NVLink)

    Attention: DP=8 (每卡独立处理不同请求, KV Cache 分区)
    MoE:       EP=8 (专家分布到 8 卡)

    特点: 吞吐量最高, KV Cache 不重复
    通信: Attention 用 AllGather (KV), MoE 用 AllToAll
    适用: 高并发, > 512 并发请求
```

```bash
# vLLM 配置
vllm serve deepseek-ai/DeepSeek-R1 \
    --tensor-parallel-size 1 \
    --data-parallel-size 8 \
    --enable-expert-parallel
```

#### 模式 C: TP 节点内 + PP 节点间

最常见的混合并行配置：在 NVLink 域内使用 TP，跨节点使用 PP。

```
节点 0 (8 GPU, NVLink)              节点 1 (8 GPU, NVLink)
┌────────────────────────────┐    ┌────────────────────────────┐
│ GPU0  GPU1  GPU2  GPU3     │    │ GPU0  GPU1  GPU2  GPU3     │
│ ◄── TP=4, Stage 0 ──▶     │    │ ◄── TP=4, Stage 2 ──▶     │
│                            │    │                            │
│ GPU4  GPU5  GPU6  GPU7     │    │ GPU4  GPU5  GPU6  GPU7     │
│ ◄── TP=4, Stage 1 ──▶     │    │ ◄── TP=4, Stage 3 ──▶     │
└────────────────────────────┘    └────────────────────────────┘
              │ PP (跨节点, InfiniBand)  │
              └──────────────────────────┘

配置: TP=4, PP=4, 总 GPU = 4×4 = 16
```

#### 模式 D: PP + EP（大规模 MoE 训练）

DeepSeek-V3 训练配置 (2048 H800)：

```
DeepSeek-V3 Training:
  PP = 16 (跨节点流水线, DualPipe 调度)
  EP = 64 (专家分布到 64 GPU, 跨 8 个节点)
  DP = 2  (ZeRO-1 数据并行)
  TP = 1  (无张量并行! MLA 的 KV 维度小，TP 开销 > 收益)

  使用 DualPipe 实现 PP，重叠 AllToAll 通信
  Node-limited routing (M=4) 减少约 50% 跨节点流量
```

#### 模式 E: 4D/5D 并行

完整的多维并行：TP × PP × CP × DP (+EP for MoE)。

```
4D 并行示例: 128 GPUs

    ┌──────────────── PP=4 (跨节点) ──────────────────┐
    │                                                  │
    │  ┌─── TP=4 (节点内 NVLink) ───┐  × CP=2  × DP=4 │
    │  │ GPU0 GPU1 GPU2 GPU3       │                  │
    │  └────────────────────────────┘                  │
    │                                                  │
    └──────────────────────────────────────────────────┘

总 GPU = TP × PP × CP × DP = 4 × 4 × 2 × 4 = 128
```

### 12.4 实际配置参考

| 模型 | 总 GPU | TP | PP | CP | DP | EP | 备注 |
|------|--------|----|----|----|----|----|----|
| Llama-3-70B 训练 | 256 | 8 | 4 | 1 | 8 | — | 经典配置 |
| Llama-3-405B 训练 | 16384 | 8 | 16 | 1 | 128 | — | Meta 规模 |
| DeepSeek-V3 训练 | 2048 | 1 | 16 | 1 | 2 | 64 | DualPipe, MoE |
| DeepSeek-V3 推理 (prefill) | 32 | 4 | 1 | 1 | 8 | 32 | 冗余专家 |
| 1M 长序列训练 | 64 | 4 | 2 | 8 | 1 | — | CP 为主 |

### 12.5 vLLM 的 TP+EP vs DP+EP 性能对比

```
    吞吐量 (tokens/s)
    │
    │                                          DP=8+EP ●
    │                                       ●
    │                                    ●
    │                              ●
    │                        ●
    │                  TP=8+EP ●─ ─ ─ ─●─ ─ ─ ─●─ ─ ─ ─ ─●
    │              ●
    │         ●
    │    ●
    │●
    └────────────────────────────────────────────────────→
    1    16   64   128  256  512  1024  2048  4096
                      并发请求数

    ★ 交叉点在 ~256-512 并发
    ★ 低并发: TP+EP 延迟更低 (52% throughput advantage at concurrency=64)
    ★ 高并发: DP+EP 吞吐更高 (47% throughput advantage at concurrency=1024)
```

### 12.6 混合使用的注意事项

| 注意事项 | 说明 |
|---------|------|
| **TP + EP 时必须启用序列并行** | Megatron-LM 的硬性要求：`--sequence-parallel` |
| **EP × ETP 应在 NVLink 域内** | ETP 的 AllGather/ReduceScatter 需要高带宽 |
| **PP 在两种层间必须一致** | Dense 层和 MoE 层共享同一 PP 划分 |
| **num_experts % EP == 0** | 专家数必须被 EP 度整除 |
| **AllToAll 与 AllReduce 无法直接融合** | 不同通信模式在网络上可能竞争带宽 |
| **专家激活密度影响 EP 收益** | 激活密度 > 3% 时用 EP；< 1% 时 AllToAll 开销可能大于收益 |

> **专家激活密度** = `experts_per_token / total_routed_experts × 100%`
> - DeepSeek-V3: 8/256 = 3.1% → 适合 EP
> - Llama-4-Maverick: 1/128 ≈ 0.8% → EP 收益可疑

### 12.7 TP × CP 组合详解

最容易让人想不通的组合："TP 切 head、CP 切 sequence，这两个怎么叠？"答案是 **它们是 GPU 2D 网格里的正交轴**，单次 forward pass 里两个通信组永远不会同时通信。

**例子：8 张 GPU，TP=2 × CP=4。**

```
         CP rank 0    CP rank 1    CP rank 2    CP rank 3
        ┌──────────┬──────────┬──────────┬──────────┐
TP rank 0│  GPU 0   │  GPU 1   │  GPU 2   │  GPU 3   │
        ├──────────┼──────────┼──────────┼──────────┤
TP rank 1│  GPU 4   │  GPU 5   │  GPU 6   │  GPU 7   │
        └──────────┴──────────┴──────────┴──────────┘
```

Attention 入口张量形状是 `(B, S, H, D)`。两层切分：

- **TP 切 $H$**：网格每一行（同 TP rank）拥有 $H/2$ 个 head
- **CP 切 $S$**：网格每一列（同 CP rank）拥有 $S/4$ 个 token

每张 GPU 持有形状 `(B, S/4, H/2, D)` 的子张量。

**两个独立的通信组：**

```
TP 通信组（垂直，2 个成员）：       用于 TP AllReduce
  {GPU 0, GPU 4}, {GPU 1, GPU 5},
  {GPU 2, GPU 6}, {GPU 3, GPU 7}

CP 通信组（水平，4 个成员）：       用于 Ring P2P 或 AllToAll
  {GPU 0, GPU 1, GPU 2, GPU 3}     (TP rank 0 那一行)
  {GPU 4, GPU 5, GPU 6, GPU 7}     (TP rank 1 那一行)
```

**一层 transformer 的 attention 部分 —— 执行流：**

```
1. Input LayerNorm + Dropout（SP）：
     按 sequence 维度切（与 CP 协调，不冲突）。

2. QKV projection（TP 区域）：
     每张 GPU 拿 input (B, S/4, D) 乘自己持有的 W_qkv 切片（column-parallel）
     → (B, S/4, H/2 × D × 3)。无通信。

3. Attention 计算（CP 区域）：
     每张 GPU 拿自己的 Q, K, V（形状 (B, S/4, H/2, D)）
     在自己的 CP 组（4 GPU）内跑 Ring Attention 或 Ulysses。
     输出：(B, S/4, H/2, D)。
     → 这一阶段所有 CP 通信只在 row 内进行。TP 不参与；H/2 head 子集
       是各 TP rank 的私有数据。

4. Output projection（TP 区域）：
     row-parallel TP matmul → 在 2 卡 TP 组上 AllReduce。
     → 这一阶段所有 TP 通信只在 column 内进行。CP 不参与；
       sequence 切片不动。

5. Residual + LayerNorm + FFN（TP + SP）→ 下一层。
```

**关键观察：每张 GPU 同时是两个组的成员，但同一时刻只有一个组在通信。** CP attention 阶段 GPU 0 只跟 {GPU 1, 2, 3} 通信，跟 GPU 4 没关系。TP output-projection 阶段 GPU 0 只跟 GPU 4 通信，跟 GPU 1/2/3 没关系。两个组在同一次 forward 的不同轴上工作。

**数学上为什么成立**：multi-head attention 在 head 维度上 *天然并行* —— 不同 head $h$ 在 attention 内部 *完全不交互*。所以 TP 切 head 不需要在 attention 内部通信。Token 之间则 *全部需要交互*（每个 query 看全 KV），所以 CP 切 sequence 需要在 attention 内部通信。**两个轴的"通信发生位置"恰好不重叠**，这正是它们可以自由组合的根本原因。

**推广**：生产 5 维并行（TP × PP × DP × CP × EP）下每张 GPU 是 5 维网格里的一个坐标；每个并行维度对应一个独立的 NCCL 通信组；每层 forward 根据当前 operator 决定走哪个轴的通信组。不同层 / 不同阶段用不同的组；从来不需要两个组同时通信。

---

## 13. 选择指南：决策流程

```
                            开始
                              │
                    ┌─────────▼──────────┐
                    │ 模型能放进 1 张 GPU？ │
                    └─────────┬──────────┘
                         是 ╱   ╲ 否
                          ╱       ╲
                    ┌────▼───┐  ┌──▼─────────────────┐
                    │ 用 DP  │  │ 有 NVLink 互联?      │
                    │ 最简单 │  └──┬─────────────────┘
                    │(或ZeRO │  是╱    ╲否
                    │节省内存)│  ╱        ╲
                    └────────┘
                    ┌──────────▼───┐  ┌───▼──────────┐
                    │ TP ≤ 8       │  │ PP (跨节点)    │
                    │ (NVLink域内) │  │ + DP (跨节点)  │
                    │ + SP (启用)  │  └──────────────┘
                    └──────┬───────┘
                           │
                    ┌──────▼───────────────────┐
                    │ 还需要更多 GPU?             │
                    └──────┬───────────────────┘
                      是 ╱     ╲ 否
                       ╱         ╲
              ┌───────▼──────┐  ┌─▼────────────┐
              │ 加 PP 跨节点  │  │ 加 DP 扩吞吐  │
              │ + CP 长序列?  │  │ 或 ZeRO 节省  │
              └───────┬──────┘  │ 内存          │
                      │         └──────────────┘
              ┌───────▼──────────────────────┐
              │ 是 MoE 模型?                   │
              └───────┬──────────────────────┘
                 是 ╱     ╲ 否
                  ╱         ╲
         ┌──────▼──────┐  ┌─▼────────────┐
         │ 加 EP        │  │ 剩余GPU用DP  │
         │ 分布专家      │  │ 扩大batch     │
         └───────┬──────┘  └──────────────┘
                 │
         ┌───────▼──────────────────────┐
         │ 单个专家放得进 1 张 GPU?       │
         └───────┬──────────────────────┘
            是 ╱     ╲ 否
             ╱         ╲
     ┌──────▼──────┐ ┌──▼──────────┐
     │ ETP=1       │ │ ETP=2 或 4  │
     │ 最大化 EP   │ │ (万不得已)   │
     └──────┬──────┘ └─────────────┘
            │
     ┌──────▼──────────────────────┐
     │ 推理还是训练?                 │
     └──────┬──────────────────────┘
       推理 ╱     ╲ 训练
          ╱         ╲
  ┌──────▼──────┐ ┌──▼──────────────┐
  │ 高并发?      │ │ 剩余GPU用EDP    │
  │ 是→DP Attn  │ │ + ZeRO-1/FSDP  │
  │    +EP      │ │ 同步专家梯度     │
  │ 否→TP+EP    │ └─────────────────┘
  └─────────────┘
```

### 13.1 硬件需求

| 策略 | 最低互联要求 | 推荐互联 |
|------|------------|---------|
| TP / SP | NVLink (600+ GB/s) | NVLink 4.0 (900 GB/s) |
| EP (节点内) | NVLink | NVLink |
| EP (跨节点) | InfiniBand (200+ Gbps) | IB NDR (400 Gbps) |
| PP | 任意 (点对点，低频) | 100 Gbps 以太网即可 |
| CP (Ring) | 中等 (可隐藏通信) | 200+ Gbps |
| CP (Ulysses) | 高 (阻塞通信) | NVLink |
| DP / EDP | 任意 (梯度同步，低频) | 随模型大小而定 |
| ZeRO-3 / FSDP | 中-高 (逐层通信) | IB 200+ Gbps |

### 13.2 关键经验法则

1. **TP 永远保持在 NVLink 域内** — 跨节点 TP 几乎不可行
2. **启用 SP 配合 TP** — 减少激活内存，TP+EP 时是强制要求
3. **最大化 EP，最小化 ETP** — EP 比 ETP 通信效率高得多
4. **PP 用于跨节点扩展** — 低频点对点通信，带宽需求最低
5. **CP 用于超长序列** — 序列 >128K 时考虑
6. **DP/EDP 用于提升吞吐量** — 模型放得下后再加 DP
7. **ZeRO-1 优先于 ZeRO-3** — 通信量不变但内存大幅改善
8. **推理中根据并发选 TP+EP 或 DP Attention+EP** — 交叉点约在 256-512 并发

---

## 14. 实战案例：DeepSeek-V3

### 14.1 模型概况

- 总参数：671B，每 token 激活 37B
- 61 层 Transformer，每层 256 个路由专家 + 1 个共享专家
- 每 token 激活 8 个专家
- 使用 MLA (Multi-head Latent Attention)，KV 压缩到 512 维

### 14.2 训练配置 (2048 H800 GPU)

```
    256 节点 × 8 GPU/节点 = 2048 GPU
    节点内: NVLink (160 GB/s)
    节点间: InfiniBand (50 GB/s)

    ┌─────────────────────────────────────────────────────┐
    │ PP = 16    (16 个流水线 stage, DualPipe 调度)         │
    │ EP = 64    (跨 8 个节点, 每张卡 256/64 = 4 个专家)   │
    │ DP = ZeRO-1 (优化器状态切分)                         │
    │ TP = 1     (不用张量并行!)                            │
    └─────────────────────────────────────────────────────┘

    为什么 TP=1?
    ├── MLA 的 KV 维度只有 512, TP 切分 KV Cache 收益微小
    ├── TP 每层 2 次 AllReduce, 跨 8 节点代价太高
    └── EP+PP+DP 已经足够, 且可以通过 DualPipe 隐藏通信
```

### 14.3 PP=16 的 DualPipe 调度

DeepSeek-V3 使用 PP=16 的双向流水线，这是已知最大规模的 DualPipe 应用：

```
    PP=16: 16 个 stage, 每 stage 约 4 层

    DualPipe 气泡: (PP/2 - 1) × T_{F&B} = 7 × T_{F&B}
    对比 1F1B:     (PP - 1) × (T_F + T_B) = 15 × (T_F + T_B)

    由于 DualPipe 将计算与通信完全重叠:
    T_{F&B} ≈ max(T_compute, T_comm) < T_F + T_B

    实际效果:
    ├── 气泡从 ~15 个时间单位降至 ~7 个 (>50% 减少)
    ├── AllToAll 通信被完全隐藏
    └── 代价: 2x 参数内存 (双向流各存一份)

    PP=16 与 EP=64 的交互:
    ├── 每个 PP stage 包含 ~4 层, 每层有 AllToAll (EP)
    ├── DualPipe 将这些 AllToAll 与其他 stage 的计算重叠
    ├── 自定义 layout: "Et*3|(tt|)*29,m|L"
    │   E = embedding, t = transformer layer, m = MTP, L = LM head
    └── VPP=2: 每 GPU 负责 2 个不连续的 chunk
```

### 14.4 节点限制路由

```
    EP=64 横跨 8 个节点

    不做限制:
    token → 可能发往全部 8 个节点 → 大量跨节点通信

    节点限制 M=4:
    token → 最多发往 4 个节点 → 跨节点通信减少约 50%
    8 个专家分散在 ≤4 个节点 → 平均每节点约 2 个专家

    通信量上界 = M × IB_bandwidth = 4 × 50 GB/s = 200 GB/s
```

### 14.5 自定义通信内核

```
    H800 GPU: ~132 个 SM
    ├── 112 个 SM: 用于计算 (Attention, MLP, ...)
    └── 20 个 SM: 专门用于通信
        ├── 10 个通信 channel
        │   ├── IB 发送/接收
        │   ├── IB → NVLink 转发
        │   └── NVLink 传输
        │
        └── 足以饱和:
            ├── IB: 50 GB/s ✓
            └── NVLink: 160 GB/s ✓
```

### 14.6 推理配置

| 配置项 | Prefill (32 GPU, 4节点) | Decode (320 GPU, 40节点) |
|-------|------------------------|-------------------------|
| Attention | TP=4 + SP, DP=8 | TP=4 + SP, DP=80 |
| MoE | EP=32 | EP=320 (每卡~1个专家) |
| 冗余专家 | 32个 (每卡1个热门专家副本) | — |
| 每卡专家数 | 256/32 = 8 + 1冗余 | 256/320 ≈ 1 |

```
    Prefill 阶段 (32 GPU):

    ┌─── Node 0 ──────────────────────────────────────┐
    │ GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7  │
    │ TP=4 group   TP=4 group                         │
    │ ├G0─G1─G2─G3┤├G4─G5─G6─G7┤                     │
    └─────────────────────────────────────────────────┘
    ┌─── Node 1 ──┐ ┌─── Node 2 ──┐ ┌─── Node 3 ──┐
    │ 同上        │ │ 同上        │ │ 同上        │
    └─────────────┘ └─────────────┘ └─────────────┘

    所有 32 GPU 组成一个 EP=32 的 MoE 专家组
    DP=8: 有 8 组 TP=4 的 Attention 副本
```

### 14.7 LMSYS 的 96 H100 部署

```
    12 节点 × 8 GPU = 96 H100

    Prefill: EP=32, 4 节点
    Decode:  EP=72, 9 节点
    DP Attention: KV Cache 分区

    性能:
    ├── 52.3k input tokens/sec/node
    ├── 22.3k output tokens/sec/node
    └── 成本: ~$0.20/1M output tokens
```

---

## 15. 全面对比

| 维度 | DP | ZeRO/FSDP | TP | SP | PP | CP | EP | EDP | ETP | DP Attn |
|------|----|-----------|----|----|----|----|----|-----|-----|---------|
| **切分什么** | 数据 batch | 数据+优化器/梯度/参数 | 权重矩阵 | 序列(非TP区) | 连续层组 | 序列(注意力) | 完整专家 | MoE 数据 batch | 专家权重矩阵 | KV Cache 分区 |
| **作用层** | Dense | Dense | Dense | Dense | 所有 | Attention | MoE | MoE | MoE | 推理 Attn |
| **通信原语** | AllReduce | AG + RS | AllReduce | AG + RS | Send/Recv | Ring/A2A | AllToAll | AllReduce | AG + RS | AllGather |
| **通信频率** | 1次/step | 每层多次 | 每层2-4次 | 每层2次 | stage边界 | 注意力层 | 每MoE层2次 | 1次/step | 每MoE层2次 | 每注意力层 |
| **带宽需求** | 低 | 中-高 | 极高(NVLink) | 极高(=TP) | 低(PCIe) | 中-高 | 高(IB) | 低-中 | 极高(NVLink) | 中 |
| **内存效率** | 低(全复制) | 高(16P/N) | 高(1/N) | 更高(激活/N) | 高(1/PP) | 高(KV/N) | 高(1/EP) | 低(复制) | 高(1/ETP) | 高(KV/N) |
| **扩展性** | 极好 | 极好 | ≤8 | =TP | 好 | 好 | 好 | 极好 | ≤8 | 好 |
| **适用阶段** | 训练+推理 | 训练 | 训练+推理 | 训练+推理 | 训练+推理 | 训练+推理 | 训练+推理 | 训练+推理 | 训练+推理 | 推理 |
| **推荐度** | ★★★★★ | ★★★★ | ★★★★ | ★★★★ | ★★★ | ★★★ | ★★★★★ | ★★★★ | ★★(尽量=1) | ★★★★ |

---

## 16. PP 各调度策略性能对比

假设 p=8 stages, m=32 micro-batches, T_F = T_B/2 = T_W:

| 调度策略 | 气泡率 | 归一化吞吐 |
|---------|-------|-----------|
| **GPipe** | 17.9% | 1.00x |
| **1F1B** | 17.9% | 1.00x (内存更优) |
| **Interleaved 1F1B (v=2)** | 10.4% | ~1.09x |
| **ZB-H1** | ~6% | ~1.14x |
| **ZB-H2** | ~1% | ~1.20x |
| **DualPipe** | ~4% + 全重叠通信 | ~1.25x (MoE 场景) |

**PP vs TP 选择快速对照表**：

| 考虑因素 | 选 TP | 选 PP |
|---------|-------|-------|
| 互连带宽 | NVLink (900 GB/s) | PCIe (64 GB/s) 或网络 |
| 优先指标 | 低延迟 (TTFT, TPOT) | 高吞吐 (tokens/s) |
| 并发量 | 低 (<200 请求) | 高 (>500 请求) |
| GPU 数量 | ≤8 (单节点) | >8 (多节点) |
| 模型类型 | Dense | MoE (PP + EP) |

---

## 参考文献

- [Megatron-LM: Training Multi-Billion Parameter Language Models (Shoeybi et al., 2019)](https://arxiv.org/abs/1909.08053)
- [GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism (Huang et al., 2019)](https://arxiv.org/abs/1811.06965)
- [PipeDream: Generalized Pipeline Parallelism (Narayanan et al., 2019)](https://arxiv.org/abs/1806.03377)
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models (Rajbhandari et al., 2020)](https://arxiv.org/abs/1910.02054)
- [Megatron-LM v2: Reducing Activation Recomputation (Korthikanti et al., 2022)](https://arxiv.org/abs/2205.05198) — Sequence Parallelism
- [Efficient Large-Scale Language Model Training on GPU Clusters (Narayanan et al., 2021)](https://arxiv.org/abs/2104.04473) — Interleaved 1F1B
- [Zero Bubble Pipeline Parallelism (Qi et al., ICLR 2024)](https://arxiv.org/abs/2401.10241)
- [DeepSeek-V3 Technical Report (2024)](https://arxiv.org/abs/2412.19437) — DualPipe
- [Ring Attention with Blockwise Transformers (Liu et al., ICLR 2024)](https://arxiv.org/abs/2310.01889)
- [DeepSpeed Ulysses: Extreme Long Sequence Transformer (Jacobs et al., 2023)](https://arxiv.org/abs/2309.14509)
- [A Unified Sequence Parallelism Approach (2024)](https://arxiv.org/abs/2405.07719) — Hybrid Ring+Ulysses
- [MoE Parallel Folding (NVIDIA, 2025)](https://arxiv.org/abs/2504.14960)
- [DeepSpeed-TED: Efficient MoE via Tensor-Expert-Data Parallelism (2023)](https://arxiv.org/abs/2303.06318)
- [Megatron Core MoE Documentation](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html)
- [vLLM MoE Playbook (AMD/ROCm)](https://rocm.blogs.amd.com/software-tools-optimization/vllm-moe-guide/README.html)
- [LMSYS: Large-Scale EP Deployment](https://www.lmsys.org/blog/2025-05-05-large-scale-ep/)
- [JAX Scaling Book - Training Chapter](https://jax-ml.github.io/scaling-book/training/)
- [NVIDIA Hybrid-EP Blog](https://developer.nvidia.com/blog/optimizing-communication-for-mixture-of-experts-training-with-hybrid-expert-parallel/)
- [Meta: Scaling LLM Inference (2025)](https://engineering.fb.com/2025/10/17/ai-research/scaling-llm-inference-innovations-tensor-parallelism-context-parallelism-expert-parallelism/)
- [TensorRT-LLM Expert Parallelism](https://nvidia.github.io/TensorRT-LLM/advanced/expert-parallelism.html)
- [PyTorch FSDP Documentation](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [PyTorch Pipeline Parallelism Documentation](https://docs.pytorch.org/docs/stable/distributed.pipelining.html)

---

## 相关页面

- [[vllm]] — vLLM 推理框架，实现 DP Attention+EP / TP+EP
- [[prefill-decode-disaggregation]] — Prefill-Decode 分离部署
- [[distributed-training]] — 训练侧并行（相关但关注点不同）
- [[quantization]] — 通过量化减少对并行的需求
