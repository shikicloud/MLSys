---
title: "分布式训练：数据/模型/流水线并行"
category: ml-infra
tags: [分布式训练, 数据并行, 张量并行, 流水线并行, zero, fsdp, 混合精度, 梯度检查点, 容错]
created: 2026-04-13
updated: 2026-04-14
status: mature
---

# 分布式训练：数据/模型/流水线并行

> [!abstract]+ TL;DR
> 现代 LLM（Llama 3.1 405B、DeepSeek-V3）需要数千 GPU 和数万亿 token 才能训练。分布式训练通过**数据并行**（复制模型、切分数据）、**模型并行**（张量 + 流水线 + 专家）和**显存优化**（ZeRO、FSDP、混合精度、激活检查点）让大模型在可获得的硬件上训练。本页聚焦训练专有问题；姊妹页 [[parallelism-strategies-deep-dive]] 详述与推理共用的切分机制。

---

## 1. 概述

### 1.1 为什么需要分布式训练

现代大语言模型的参数规模已经远远超越单 GPU 的承载能力：

| 模型 | 参数量 | 训练 GPU | 训练 Token | 训练时间 |
|------|--------|----------|-----------|---------|
| GPT-3 (2020) | 175B | ~1,000 V100 | 300B | ~34 天 |
| LLaMA 2 70B (2023) | 70B | 2,048 A100 | 2T | ~25 天 |
| LLaMA 3.1 405B (2024) | 405B | 16,384 H100 | 15.6T | ~54 天 |
| DeepSeek-V3 (2024) | 671B (37B active) | 2,048 H800 | 14.8T | ~55 天 |
| GPT-4 (推测, 2023) | ~1.8T MoE | ~25,000 A100 | ~13T | ~90 天 |

**核心挑战**：

1. **显存瓶颈** — 一个 70B FP16 模型仅权重就需要 ~140 GB，单张 H100 只有 80 GB
2. **计算瓶颈** — 1T token × 70B 参数 ≈ 4.2×10²² FLOPs，单 H100 (990 TFLOPS BF16) 需要约 490 天
3. **通信瓶颈** — 多卡/多节点间的数据传输成为新的性能瓶颈

### 1.2 训练显存构成

理解显存消耗是分布式训练设计的基础。对于一个参数量为 $\Phi$ 的模型，使用 AdamW + 混合精度训练，每张 GPU 上的显存需求为：

```
训练显存 = 模型参数 + 梯度 + 优化器状态 + 激活值

模型参数（混合精度）:
  - FP16/BF16 参数: 2Φ bytes
  - FP32 主权重 (master weights): 4Φ bytes

梯度: 2Φ bytes (FP16/BF16)

优化器状态 (AdamW):
  - FP32 一阶动量 (m): 4Φ bytes
  - FP32 二阶动量 (v): 4Φ bytes

总计 (不含激活): 2Φ + 4Φ + 2Φ + 4Φ + 4Φ = 16Φ bytes
```

**示例**：70B 参数模型 → 16 × 70B × 1 byte = **1,120 GB**（不含激活值），远超单 GPU 显存。

激活值的显存占用与序列长度、batch size、模型隐藏维度相关：

```
激活值（每层每样本，Transformer）≈ s × b × h × (34 + 5·a·s/h)
  - s: 序列长度
  - b: micro-batch size
  - h: 隐藏维度
  - a: 注意力头数
```

---

## 2. 数据并行 (DP) 训练细节

> 关于 DP 的基本原理与通信模式，参见 [[parallelism-strategies-deep-dive#2. DP — 数据并行]]。本节聚焦训练中的梯度同步实现。

### 2.1 DistributedDataParallel (DDP) 工作流程

PyTorch DDP 是最基础的数据并行实现：

```
   GPU 0              GPU 1              GPU 2              GPU 3
┌──────────┐      ┌──────────┐      ┌──────────┐      ┌──────────┐
│ Model    │      │ Model    │      │ Model    │      │ Model    │
│ (完整副本)│      │ (完整副本)│      │ (完整副本)│      │ (完整副本)│
├──────────┤      ├──────────┤      ├──────────┤      ├──────────┤
│ Data     │      │ Data     │      │ Data     │      │ Data     │
│ Shard 0  │      │ Shard 1  │      │ Shard 2  │      │ Shard 3  │
├──────────┤      ├──────────┤      ├──────────┤      ├──────────┤
│ Forward  │      │ Forward  │      │ Forward  │      │ Forward  │
│ Backward │      │ Backward │      │ Backward │      │ Backward │
├──────────┤      ├──────────┤      ├──────────┤      ├──────────┤
│ Grad_0   │      │ Grad_1   │      │ Grad_2   │      │ Grad_3   │
└────┬─────┘      └────┬─────┘      └────┬─────┘      └────┬─────┘
     │                 │                 │                 │
     └────────────┬────┴────────────┬────┘                 │
                  │   AllReduce     │                       │
                  │  (求和再除 N)   ├───────────────────────┘
                  │                 │
     ┌────────────┴────┬────────────┴────┬─────────────────┐
     ▼                 ▼                 ▼                 ▼
   Avg Grad         Avg Grad         Avg Grad         Avg Grad
   (相同)            (相同)            (相同)            (相同)
     │                 │                 │                 │
   Update            Update           Update           Update
   Params            Params           Params           Params
```

**关键特性**：
- 每个 GPU 持有完整模型副本
- 数据在 worker 间分片，每个 worker 处理不同的 micro-batch
- 反向传播后通过 AllReduce 同步梯度
- 所有 worker 更新后的参数完全一致

### 2.2 Ring AllReduce 算法

Ring AllReduce 是 DDP 梯度同步的核心算法，将 N 个 GPU 组成逻辑环：

```
步骤 1: ReduceScatter — 每个 GPU 把梯度分成 N 份，
        通过 N-1 步的"传递 + 累加"使得每个 GPU 持有 1/N 梯度的全局和

GPU 0 ──→ GPU 1 ──→ GPU 2 ──→ GPU 3
  ↑                                │
  └────────────────────────────────┘

初始状态（4 GPU，梯度分为 4 chunk）:
GPU 0: [A0, B0, C0, D0]
GPU 1: [A1, B1, C1, D1]
GPU 2: [A2, B2, C2, D2]
GPU 3: [A3, B3, C3, D3]

Round 1: 每个 GPU 向右发送一个 chunk，接收并累加
GPU 0: [A0,      B0,      C0,      D0+D3  ]
GPU 1: [A1+A0,   B1,      C1,      D1     ]
GPU 2: [A2,      B2+B1,   C2,      D2     ]
GPU 3: [A3,      B3,      C3+C2,   D3     ]

Round 2:
GPU 0: [A0,      B0,      C0+C3+C2, D0+D3  ]
GPU 1: [A1+A0,   B1,      C1,       D1+D0+D3]
GPU 2: [A2+A1+A0,B2+B1,   C2,       D2      ]
GPU 3: [A3,      B3+B2+B1,C3+C2,    D3      ]

Round 3 (ReduceScatter 完成):
GPU 0: [A0,       B0+B3+B2+B1, C0+C3+C2, D0+D3  ]
GPU 1: [A1+A0,    B1,          C1+C0+C3+C2, D1+D0+D3]
GPU 2: [A2+A1+A0, B2+B1,       C2,       D2+D1+D0+D3]
GPU 3: [A3+A2+A1+A0, B3+B2+B1, C3+C2,    D3      ]
         ↑ 全局和       ↑ 全局和     ↑ 全局和   ↑ 全局和
         (在 GPU 3)      (在 GPU 0)  (在 GPU 1)  (在 GPU 2)

步骤 2: AllGather — 再 N-1 步把各段全局和广播给所有 GPU

最终: 每个 GPU 都有完整的全局梯度和
```

**通信复杂度分析**：

| 算法 | 通信量 (每 GPU) | 步骤数 | 特点 |
|------|----------------|--------|------|
| Naive AllReduce | $(N-1) \cdot D$ | $N-1$ | 每个 worker 向 root 发送 |
| Ring AllReduce | $2 \cdot \frac{N-1}{N} \cdot D$ | $2(N-1)$ | 与 GPU 数量几乎无关 |
| Tree AllReduce | $2 \cdot D \cdot \log N$ | $2\log N$ | 步骤少但带宽利用率低 |

其中 $D$ 是梯度数据量，$N$ 是 GPU 数量。Ring AllReduce 的优势在于**通信量与 GPU 数量基本无关**（$N$ 很大时 $(N-1)/N \approx 1$），因此可以很好地扩展到大规模集群。

### 2.3 梯度累积 (Gradient Accumulation)

当 GPU 显存不足以容纳理想的 micro-batch size 时，可通过梯度累积来获得等效的大 batch：

```python
# 梯度累积示例
accumulation_steps = 4  # 等效 batch = micro_batch × accumulation_steps × world_size
optimizer.zero_grad()

for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps  # 注意: 需要对 loss 做缩放
    loss.backward()  # 梯度累积到 .grad 中

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()   # 只在累积完成时更新
        optimizer.zero_grad()
```

**有效 batch size 计算**：
```
effective_batch_size = micro_batch_size × accumulation_steps × world_size (DP degree)
```

**注意事项**：
- Loss 需要除以 `accumulation_steps`（否则梯度尺度不对）
- DDP 默认每次 `backward()` 都触发 AllReduce，累积时应该只在最后一步同步
- PyTorch 中使用 `model.no_sync()` 上下文管理器跳过中间步骤的同步：

```python
for i, batch in enumerate(dataloader):
    context = model.no_sync() if (i + 1) % accum_steps != 0 else nullcontext()
    with context:
        loss = model(batch) / accum_steps
        loss.backward()
    if (i + 1) % accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 2.4 同步训练 vs 异步训练

| 特性 | 同步训练 (BSP) | 异步训练 (ASP) |
|------|---------------|---------------|
| 梯度一致性 | 所有 worker 使用相同梯度 | 使用 stale 梯度 |
| 同步屏障 | 有 (AllReduce) | 无 |
| 收敛性 | 等同于单卡（理想情况） | 可能发散或收敛变慢 |
| 效率 | 受最慢 worker 制约 | 无掉队者 (straggler) 问题 |
| 适用场景 | **主流**（几乎所有 LLM 训练） | 大规模推荐/广告系统 |

**现状**：大模型训练几乎全部使用**同步训练**。异步训练的 stale gradient 问题会导致收敛不稳定，且在同构 GPU 集群上 straggler 问题不严重。异步训练主要用于工业推荐系统的 embedding 更新。

---

## 3. ZeRO 优化

ZeRO（Zero Redundancy Optimizer）由 DeepSpeed 团队提出，核心思想是**消除数据并行中的内存冗余**。标准 DDP 中每个 GPU 持有完整的模型参数、梯度和优化器状态——这是巨大的浪费。

### 3.1 三个阶段

```
                  标准 DDP（N=4 GPU 情况）
  ┌──────────────────────────────────────────────────────┐
  │  GPU 0          GPU 1          GPU 2          GPU 3  │
  │ ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐│
  │ │Params  │    │Params  │    │Params  │    │Params  ││
  │ │(全量)  │    │(全量)  │    │(全量)  │    │(全量)  ││
  │ │ 2Φ     │    │ 2Φ     │    │ 2Φ     │    │ 2Φ     ││
  │ ├────────┤    ├────────┤    ├────────┤    ├────────┤│
  │ │Grads   │    │Grads   │    │Grads   │    │Grads   ││
  │ │(全量)  │    │(全量)  │    │(全量)  │    │(全量)  ││
  │ │ 2Φ     │    │ 2Φ     │    │ 2Φ     │    │ 2Φ     ││
  │ ├────────┤    ├────────┤    ├────────┤    ├────────┤│
  │ │Opt State│   │Opt State│   │Opt State│   │Opt State│
  │ │(全量)  │    │(全量)  │    │(全量)  │    │(全量)  ││
  │ │ 12Φ    │    │ 12Φ    │    │ 12Φ    │    │ 12Φ    ││
  │ └────────┘    └────────┘    └────────┘    └────────┘│
  │ 每 GPU: 16Φ   每 GPU: 16Φ   每 GPU: 16Φ  每 GPU: 16Φ│
  │                  总冗余: 4 × 16Φ = 64Φ               │
  └──────────────────────────────────────────────────────┘

  ZeRO-1: 仅分区优化器状态
  ┌──────────────────────────────────────────────────────┐
  │  GPU 0          GPU 1          GPU 2          GPU 3  │
  │ ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐│
  │ │Params  │    │Params  │    │Params  │    │Params  ││
  │ │(全量)  │    │(全量)  │    │(全量)  │    │(全量)  ││
  │ │ 2Φ     │    │ 2Φ     │    │ 2Φ     │    │ 2Φ     ││
  │ ├────────┤    ├────────┤    ├────────┤    ├────────┤│
  │ │Grads   │    │Grads   │    │Grads   │    │Grads   ││
  │ │(全量)  │    │(全量)  │    │(全量)  │    │(全量)  ││
  │ │ 2Φ     │    │ 2Φ     │    │ 2Φ     │    │ 2Φ     ││
  │ ├────────┤    ├────────┤    ├────────┤    ├────────┤│
  │ │Opt 1/4 │    │Opt 1/4 │    │Opt 1/4 │    │Opt 1/4 ││
  │ │ 3Φ     │    │ 3Φ     │    │ 3Φ     │    │ 3Φ     ││
  │ └────────┘    └────────┘    └────────┘    └────────┘│
  │ 每 GPU: 7Φ   (vs DDP 的 16Φ → 节省 56%)              │
  └──────────────────────────────────────────────────────┘

  ZeRO-2: 分区优化器状态 + 梯度
  ┌──────────────────────────────────────────────────────┐
  │  GPU 0          GPU 1          GPU 2          GPU 3  │
  │ ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐│
  │ │Params  │    │Params  │    │Params  │    │Params  ││
  │ │(全量)  │    │(全量)  │    │(全量)  │    │(全量)  ││
  │ │ 2Φ     │    │ 2Φ     │    │ 2Φ     │    │ 2Φ     ││
  │ ├────────┤    ├────────┤    ├────────┤    ├────────┤│
  │ │Grad 1/4│    │Grad 1/4│    │Grad 1/4│    │Grad 1/4││
  │ │ 0.5Φ   │    │ 0.5Φ   │    │ 0.5Φ   │    │ 0.5Φ   ││
  │ ├────────┤    ├────────┤    ├────────┤    ├────────┤│
  │ │Opt 1/4 │    │Opt 1/4 │    │Opt 1/4 │    │Opt 1/4 ││
  │ │ 3Φ     │    │ 3Φ     │    │ 3Φ     │    │ 3Φ     ││
  │ └────────┘    └────────┘    └────────┘    └────────┘│
  │ 每 GPU: 5.5Φ  (vs DDP 的 16Φ → 节省 66%)             │
  └──────────────────────────────────────────────────────┘

  ZeRO-3: 分区全部（参数 + 梯度 + 优化器状态）= FSDP
  ┌──────────────────────────────────────────────────────┐
  │  GPU 0          GPU 1          GPU 2          GPU 3  │
  │ ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐│
  │ │Par 1/4 │    │Par 1/4 │    │Par 1/4 │    │Par 1/4 ││
  │ │ 0.5Φ   │    │ 0.5Φ   │    │ 0.5Φ   │    │ 0.5Φ   ││
  │ ├────────┤    ├────────┤    ├────────┤    ├────────┤│
  │ │Grad 1/4│    │Grad 1/4│    │Grad 1/4│    │Grad 1/4││
  │ │ 0.5Φ   │    │ 0.5Φ   │    │ 0.5Φ   │    │ 0.5Φ   ││
  │ ├────────┤    ├────────┤    ├────────┤    ├────────┤│
  │ │Opt 1/4 │    │Opt 1/4 │    │Opt 1/4 │    │Opt 1/4 ││
  │ │ 3Φ     │    │ 3Φ     │    │ 3Φ     │    │ 3Φ     ││
  │ └────────┘    └────────┘    └────────┘    └────────┘│
  │ 每 GPU: 4Φ    (vs DDP 的 16Φ → 节省 75%)             │
  └──────────────────────────────────────────────────────┘
```

### 3.2 各阶段显存分析

设模型参数量为 $\Phi$，DP 并行度为 $N_d$，使用 AdamW + 混合精度（FP16 参数 + FP32 主权重/优化器状态）：

| 组件 | DDP | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|------|-----|--------|--------|--------|
| FP16 参数 | $2\Phi$ | $2\Phi$ | $2\Phi$ | $2\Phi / N_d$ |
| FP16 梯度 | $2\Phi$ | $2\Phi$ | $2\Phi / N_d$ | $2\Phi / N_d$ |
| FP32 主权重 | $4\Phi$ | $4\Phi / N_d$ | $4\Phi / N_d$ | $4\Phi / N_d$ |
| FP32 一阶动量 | $4\Phi$ | $4\Phi / N_d$ | $4\Phi / N_d$ | $4\Phi / N_d$ |
| FP32 二阶动量 | $4\Phi$ | $4\Phi / N_d$ | $4\Phi / N_d$ | $4\Phi / N_d$ |
| **总计** | **$16\Phi$** | **$4\Phi + 12\Phi/N_d$** | **$2\Phi + 14\Phi/N_d$** | **$16\Phi/N_d$** |

**数值示例**（70B 参数，$N_d = 64$）：

| 阶段 | 每 GPU 显存 (不含激活) | 节省比例 |
|------|----------------------|---------|
| DDP | 1,120 GB | — |
| ZeRO-1 | 280 + 13.1 = **293 GB** | 73.8% |
| ZeRO-2 | 140 + 15.3 = **155 GB** | 86.2% |
| ZeRO-3 | **17.5 GB** | 98.4% |

### 3.3 各阶段通信开销

| 阶段 | 前向传播 | 反向传播 | 参数更新后 | 总通信量 |
|------|---------|---------|-----------|---------|
| DDP | 无 | AllReduce (梯度) | 无 | $2\Phi$ |
| ZeRO-1 | 无 | ReduceScatter (梯度) | AllGather (更新后参数) | $2\Phi$ |
| ZeRO-2 | 无 | ReduceScatter (梯度) | AllGather (更新后参数) | $2\Phi$ |
| ZeRO-3 | AllGather (参数) | AllGather + ReduceScatter | 无 | $3\Phi$ |

**关键结论**：
- ZeRO-1/2 的通信量与 DDP 相同（$2\Phi$），但显存大幅减少
- ZeRO-3 需要额外的前向 AllGather，通信量增加 50%（$3\Phi$），但显存节省最大
- 实际中 ZeRO-3 / FSDP 可以通过 prefetch 和 overlap 来隐藏额外的通信开销

### 3.4 ZeRO++ 增强

ZeRO++（2023）在 ZeRO-3 基础上进一步优化跨节点通信：

1. **量化权重通信 (qwZ)** — AllGather 时使用 INT8 量化，跨节点通信量减半
2. **分层分区 (hpZ)** — 权重在节点内保留完整副本，只在节点间分片
3. **量化梯度通信 (qgZ)** — ReduceScatter 时使用 INT4 量化

效果：跨节点通信量减少 **4×**，模型训练速度提升 2.16× 至 2.49×。

---

## 4. 混合精度训练

### 4.1 为什么使用低精度

| 数据类型 | 位宽 | 指数位 | 尾数位 | 动态范围 | 精度 | 用途 |
|---------|------|--------|--------|---------|------|------|
| FP32 | 32 | 8 | 23 | ~1e±38 | 高 | 主权重、优化器状态 |
| FP16 | 16 | 5 | 10 | ~6.5e4 | 中 | 前向/反向计算 |
| BF16 | 16 | 8 | 7 | ~1e±38 | 较低 | 前向/反向计算 (推荐) |
| FP8 (E4M3) | 8 | 4 | 3 | ~448 | 低 | 前向 (实验性) |
| FP8 (E5M2) | 8 | 5 | 2 | ~57344 | 很低 | 反向 (实验性) |

**收益**：
- 显存减半（FP16/BF16 vs FP32）
- 计算速度 2-3× 提升（利用 Tensor Core）
- 通信带宽减半

### 4.2 混合精度训练流程

```
┌─────────────────────────────────────────────────────┐
│               混合精度训练循环                         │
│                                                     │
│  ┌───────────────┐                                  │
│  │ FP32 主权重    │ ←───── 优化器在 FP32 下更新       │
│  │ (Master Copy) │                                  │
│  └───────┬───────┘                                  │
│          │ Cast to BF16/FP16                         │
│          ▼                                          │
│  ┌───────────────┐                                  │
│  │ BF16 工作副本  │ ──→ 前向传播 (BF16 Tensor Core)  │
│  └───────────────┘                                  │
│          │                                          │
│          ▼                                          │
│  ┌───────────────┐                                  │
│  │ BF16 Loss     │                                  │
│  └───────┬───────┘                                  │
│          │ (FP16 时需要 Loss Scaling)                 │
│          ▼                                          │
│  ┌───────────────┐                                  │
│  │ BF16 梯度      │ ←── 反向传播 (BF16 Tensor Core)  │
│  └───────┬───────┘                                  │
│          │ AllReduce / ReduceScatter                 │
│          ▼                                          │
│  ┌───────────────┐                                  │
│  │ FP32 梯度      │ ←── Cast to FP32                 │
│  └───────┬───────┘                                  │
│          │                                          │
│          ▼                                          │
│  ┌───────────────┐                                  │
│  │ FP32 优化器    │ ──→ Adam(m, v, master_weights)   │
│  │ 状态更新       │                                  │
│  └───────────────┘                                  │
└─────────────────────────────────────────────────────┘
```

### 4.3 FP16 的 Loss Scaling

FP16 的动态范围较小（最小正数 ~6×10⁻⁸），小梯度容易下溢为零。Loss Scaling 通过放大 loss 来保持小梯度的精度：

```python
# 动态 Loss Scaling 伪代码
scaler = torch.amp.GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        output = model(batch)
        loss = criterion(output)

    # 1. 放大 loss → 放大梯度 (防止下溢)
    scaler.scale(loss).backward()

    # 2. Unscale 梯度 → 检查 inf/nan
    scaler.unscale_(optimizer)

    # 3. 梯度裁剪 (在 unscale 之后)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # 4. 更新参数 (如果梯度中有 inf/nan 则跳过)
    scaler.step(optimizer)

    # 5. 动态调整 scale factor
    scaler.update()
```

**动态 Scaling 策略**：
- 初始 scale = 2¹⁶ 或更大
- 连续 N 步无 overflow → scale × 2（增大）
- 出现 overflow → scale / 2（减小），跳过本步更新

### 4.4 BF16 的优势

BF16 是 LLM 训练的主流选择（A100/H100 开始原生支持）：

- **与 FP32 相同的动态范围**（8 位指数）→ 不需要 loss scaling
- 简化训练流程（不需要 GradScaler）
- 代价是精度略低于 FP16（7 位 vs 10 位尾数），但实践中对 LLM 影响很小

```python
# BF16 混合精度（更简单，无需 loss scaling）
with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
    output = model(batch)
    loss = criterion(output)

loss.backward()
optimizer.step()  # 直接 step，无需 scaler
```

### 4.5 FP8 训练（前沿）

FP8 训练利用 H100/H200 的 FP8 Tensor Core，理论吞吐量是 BF16 的 2×：

- **前向传播**使用 E4M3（更高精度，范围 ±448）
- **反向传播**使用 E5M2（更大范围，范围 ±57344）
- 需要 per-tensor 或 per-block 的动态缩放
- TorchTitan + Float8 + FSDP2 已实现 **50% 吞吐量提升**

**当前状态**（2026 年初）：
- Megatron-LM 和 TorchTitan 支持 FP8 前向 + BF16 反向
- 全 FP8 训练（前向 + 反向）仍在实验阶段
- DeepSeek-V3 使用了 FP8 训练，是首个大规模成功案例

---

## 5. 3D 并行

> 关于 TP、PP、EP 的切分原理和通信模式，参见 [[parallelism-strategies-deep-dive]]。
> 关于 PP 的调度策略（GPipe、1F1B、Interleaved 1F1B、Zero Bubble），参见 [[model-parallelism]]。
> 本节聚焦如何将多个并行维度**组合**起来训练超大模型。

### 5.1 为什么需要组合并行

单一并行策略各有局限：

| 策略 | 显存效率 | 计算效率 | 通信需求 | 扩展瓶颈 |
|------|---------|---------|---------|---------|
| DP | 低（全副本） | 高 | AllReduce (梯度) | 模型必须装入单 GPU |
| TP | 高（分片权重） | 高 | AllReduce (每层 2×) | 需要高带宽 NVLink |
| PP | 高（层分组） | 中 (气泡) | 点对点 | 气泡开销 |
| ZeRO-3/FSDP | 高（全分片） | 中 | AllGather + ReduceScatter | 大规模通信量 |

**3D 并行** = TP × PP × DP，取各策略之长：

```
                    集群拓扑映射
  ┌─────────────────────────────────────────────────┐
  │                  全部 GPU                        │
  │                                                 │
  │  DP replica 0          DP replica 1             │
  │  ┌──────────────────┐  ┌──────────────────┐     │
  │  │ PP stage 0       │  │ PP stage 0       │     │
  │  │ ┌──┬──┬──┬──┐   │  │ ┌──┬──┬──┬──┐   │     │
  │  │ │G0│G1│G2│G3│←TP│  │ │G8│G9│GA│GB│←TP│     │
  │  │ └──┴──┴──┴──┘   │  │ └──┴──┴──┴──┘   │     │
  │  │ PP stage 1       │  │ PP stage 1       │     │
  │  │ ┌──┬──┬──┬──┐   │  │ ┌──┬──┬──┬──┐   │     │
  │  │ │G4│G5│G6│G7│←TP│  │ │GC│GD│GE│GF│←TP│     │
  │  │ └──┴──┴──┴──┘   │  │ └──┴──┴──┴──┘   │     │
  │  └──────────────────┘  └──────────────────┘     │
  │                                                 │
  │  TP = 4 (intra-node NVLink)                     │
  │  PP = 2 (inter-node or intra-node)              │
  │  DP = 2 (cross-replica AllReduce)               │
  │  Total = 4 × 2 × 2 = 16 GPU                    │
  └─────────────────────────────────────────────────┘
```

### 5.2 硬件拓扑映射原则

**核心原则**：通信密集的并行维度 → 高带宽链路

| 并行维度 | 通信模式 | 频率 | 推荐拓扑 |
|---------|---------|------|---------|
| **TP** | AllReduce (每层 2×) | 极高 | **节点内 NVLink** (900 GB/s) |
| **PP** | 点对点 (层间激活) | 中等 | 节点内或跨节点 |
| **DP** | AllReduce (每 step 1×) | 低 | **跨节点 InfiniBand** |
| **EP** | AllToAll (每 MoE 层) | 中高 | 视负载选择节点内/跨节点 |
| **CP/SP** | AllToAll 或 Ring | 中等 | 节点内 NVLink |

**典型配置**（8 GPU/node）：
```
TP = 8 (一个 8-GPU 节点的全部 GPU 通过 NVLink 连接)
PP = N_nodes / DP_replicas (跨节点流水线)
DP = Total_GPUs / (TP × PP) (剩余维度给数据并行)
```

### 5.3 大模型配置示例

**LLaMA 3.1 405B**:
```
参数量: 405B (dense)
GPU: 16,384 × H100 80GB
TP = 8  (8 GPU per node via NVLink)
PP = 16 (16 个流水线阶段)
DP = 128 (128 个数据并行副本)
总计: 8 × 16 × 128 = 16,384 GPU
训练数据: 15.6T token
```

**DeepSeek-V3**:
```
参数量: 671B (MoE, 37B active per token)
GPU: 2,048 × H800 80GB
TP = 1  (不使用 TP! MoE 模型激活参数少)
PP = 16
DP = 128 (with ZeRO-1)
EP = 64 (专家分布在 64 个 GPU 上)
特殊: 使用 FP8 训练 + DualPipe 流水线调度
```

> 注意 DeepSeek-V3 不使用 TP 是因为 MoE 架构每个 token 只激活 37B 参数，单 GPU 可以容纳，且避免了 TP 的通信开销。

### 5.4 4D / 5D 并行

- **4D 并行** = TP + PP + DP + **CP/SP**（序列/上下文并行）
- **5D 并行** = TP + PP + DP + CP + **EP**（MoE 专用）

2025 年 NVIDIA 提出的 **MoE Parallel Folding** 框架将 Dense 层和 MoE 层的并行完全解耦：
```
Dense 层: N_total = TP × CP × DP × PP
MoE 层:   N_total = ETP × EP × EDP × PP
```

详见 [[parallelism-strategies-deep-dive#MoE Parallel Folding]]。

---

## 6. 梯度检查点 (Activation Checkpointing)

### 6.1 基本原理

标准反向传播需要保存所有中间激活值，显存随层数线性增长。梯度检查点 (也称激活重计算) 的核心思想是**用计算换显存**：

```
标准反向传播:
  前向: 保存所有层的激活值 a1, a2, ..., aL
  反向: 直接使用保存的激活值计算梯度
  显存: O(L) — 所有层的激活值

梯度检查点:
  前向: 只保存部分"检查点"层的激活值
  反向: 从最近的检查点重新前向计算所需的激活值
  显存: O(√L) — 仅检查点层的激活值
  代价: ~33% 额外前向计算
```

```
标准方式 (保存所有激活):
Layer:  1    2    3    4    5    6    7    8
Save:  [a1] [a2] [a3] [a4] [a5] [a6] [a7] [a8]
                            ↑ 显存占用大

梯度检查点 (每 2 层检查一次):
Layer:  1    2    3    4    5    6    7    8
Save:  [a1]  ×   [a3]  ×   [a5]  ×   [a7]  ×
              ↑         ↑         ↑         ↑
          从 a1 重算  从 a3 重算  从 a5 重算  从 a7 重算
```

### 6.2 全量 vs 选择性检查点

**全量检查点 (Full Checkpointing)**：
- 每个 Transformer 层只保存输入，层内所有激活重算
- 显存节省最大（激活从 $O(L \cdot s \cdot b \cdot h)$ 降到 $O(L \cdot s \cdot b \cdot h_{input})$）
- 计算开销约 33%

**选择性检查点 (Selective Checkpointing)**：
- 只对显存占用大的操作做检查点（如 self-attention 的 QKV 中间矩阵）
- 保留计算代价小的激活（如 LayerNorm 输出）
- Megatron-LM 的策略：保留非注意力激活，只重计算 attention

```python
# 选择性检查点示例 (Megatron-LM 风格)
def transformer_layer(x):
    # 注意力部分 — 做检查点 (QK^T 矩阵占显存大)
    attn_out = checkpoint(self_attention, x)

    # FFN 部分 — 不做检查点 (相对便宜)
    ffn_out = feed_forward(attn_out + x)

    return ffn_out + attn_out + x
```

### 6.3 显存节省分析

对于一个 L 层 Transformer，每层激活值约 $A$ bytes：

| 策略 | 激活值显存 | 额外计算 | 适用场景 |
|------|-----------|---------|---------|
| 无检查点 | $L \cdot A$ | 0 | 显存充足 |
| 全量检查点 | $L \cdot A_{input}$ ≈ $L \cdot A / k$ | ~33% | 显存紧张 |
| 选择性 | $L \cdot A_{selected}$ | ~10-15% | 折中方案 |
| √L 检查点 | $\sqrt{L} \cdot A$ | ~33% | 理论最优 |

**实际效果**（70B 模型，seq=4096，micro-batch=1）：
- 无检查点：~30 GB 激活值
- 全量检查点：~3 GB 激活值
- 选择性检查点：~8 GB 激活值

---

## 7. 通信优化

### 7.1 DDP 中的 Bucketing

PyTorch DDP 不会为每个参数单独触发 AllReduce，而是将参数梯度打包成 **bucket**：

```
参数梯度:  p1.grad  p2.grad  p3.grad  p4.grad  p5.grad  p6.grad
           └───── bucket 0 ─────┘  └───── bucket 1 ─────┘
                     │                        │
                 AllReduce                AllReduce
```

**好处**：
- 减少通信次数（大消息比小消息吞吐高）
- 默认 bucket size = 25 MB
- Bucket 从最后一层开始填充（因为梯度从后往前计算），实现通信与计算重叠

### 7.2 通信与计算重叠

```
时间线 →

无重叠:
[=== Forward ===][=== Backward ===][== AllReduce ==][Update]
                                                     ↑ 全部完成才能开始

有重叠 (DDP bucketing):
[=== Forward ===][=== Backward ==================]
                 [bucket N AR][bucket N-1 AR]...  [Update]
                  ↑ 最后一层梯度就绪就开始通信

FSDP 前向 AllGather 重叠:
[AG layer1][Fwd layer1][AG layer2][Fwd layer2]...
 ↑ Prefetch 下一层参数

FSDP 反向重叠:
[AG+Bwd layer L][RS layer L][AG+Bwd layer L-1][RS layer L-1]...
 ↑ AllGather 当前层 + Backward
                  ↑ ReduceScatter 梯度
```

### 7.3 梯度压缩

减少通信量的技术：

| 技术 | 压缩比 | 精度影响 | 应用场景 |
|------|--------|---------|---------|
| FP16 梯度通信 | 2× | 几乎无 | 通用 |
| INT8 量化 (ZeRO++) | 4× | 很小 | 跨节点 |
| Top-K 稀疏化 | 10-100× | 需要 error feedback | 研究阶段 |
| 幂次量化 | 8× | 小 | 研究阶段 |

### 7.4 NCCL 优化

NCCL (NVIDIA Collective Communications Library) 是 GPU 集群通信的标准库：

**关键配置**：
```bash
# 环境变量调优
NCCL_IB_DISABLE=0                 # 启用 InfiniBand
NCCL_IB_GID_INDEX=3              # IB GID 索引
NCCL_NET_GDR_LEVEL=5             # GPU Direct RDMA 级别
NCCL_SOCKET_IFNAME=eth0          # 网络接口
NCCL_P2P_LEVEL=NVL               # P2P 传输级别
NCCL_ALGO=Ring                   # 选择算法 (Ring/Tree)
NCCL_PROTO=Simple                # 协议 (Simple/LL/LL128)
NCCL_MIN_NCHANNELS=4             # 最小通道数
NCCL_CROSS_NIC=1                 # 跨 NIC 通信
```

**调优技巧**：
- 小消息用 Tree AllReduce（延迟低），大消息用 Ring AllReduce（带宽高）
- 开启 GPU Direct RDMA (GDR) 避免 CPU 中转
- 使用多 rail InfiniBand 提高跨节点带宽
- SHARP (Scalable Hierarchical Aggregation and Reduction Protocol) 利用交换机做 in-network reduction

---

## 8. 容错

### 8.1 为什么容错至关重要

大规模训练面临的故障概率：

```
单 GPU 年故障率 ≈ 3-5%
16,384 GPU 集群:
  - 平均每天 ~1.5-2.5 次 GPU 故障
  - LLaMA 3.1 405B 训练约 54 天
  - 预期故障次数 ≈ 80-135 次

不做容错 → 训练无法完成
```

### 8.2 检查点策略

**同步检查点**（传统方式）：
```
[Training step 1000] → [暂停训练] → [保存全量检查点] → [继续训练]
                                      ↑ 全部 GPU 同步写入存储
                                      耗时: 数十分钟 (大模型)
```

**异步检查点**（现代方式）：
```
[Training step 1000] → [后台异步保存] → [继续训练 step 1001]
                        ↑ 不阻塞训练
                        GPU → Host pinned memory (快) → Disk/S3 (慢，后台)
```

**检查点频率权衡**：
| 频率 | 优点 | 缺点 |
|------|------|------|
| 高 (每 100 步) | 丢失工作少 | I/O 开销大，存储成本高 |
| 低 (每 1000 步) | 开销小 | 故障后丢失更多训练进度 |
| 自适应 | 平衡 | 实现复杂 |

### 8.3 分布式检查点

Megatron-LM 和 PyTorch 的分布式检查点方案：

```python
# PyTorch Distributed Checkpoint (DCP)
import torch.distributed.checkpoint as dcp

# 保存 — 每个 rank 只保存自己的分片
dcp.save(
    state_dict={"model": model.state_dict(), "optimizer": optimizer.state_dict()},
    storage_writer=dcp.FileSystemWriter("/checkpoint/step_1000"),
)

# 加载 — 支持 resharding (不同并行度恢复)
dcp.load(
    state_dict={"model": model.state_dict(), "optimizer": optimizer.state_dict()},
    storage_reader=dcp.FileSystemReader("/checkpoint/step_1000"),
)
```

**分布式检查点的优势**：
- 每个 rank 并行写入自己的分片 → 大幅加速
- 支持 resharding：保存时 TP=8，加载时可以 TP=4
- 增量检查点：只保存变化的部分

### 8.4 弹性训练 (Elastic Training)

弹性训练允许训练过程动态调整 worker 数量（故障后减少，新节点加入后增加）：

```
正常训练: 128 GPU
  ↓ GPU 17 故障
检测故障 (~秒级)
  ↓
剩余 127 GPU 重新组织
  ↓
从最近检查点恢复 (减少 1 个 DP replica)
  ↓
继续训练: 127 GPU
  ↓ 替换节点上线
增加到 128 GPU
  ↓
从检查点恢复 (增加 1 个 DP replica)
```

**PyTorch Elastic (torchrun)**：
```bash
# 弹性启动: 最少 120 GPU，最多 128 GPU
torchrun \
    --nnodes=15:16 \        # 弹性节点数范围
    --nproc_per_node=8 \
    --rdzv_backend=c10d \   # 集合点后端
    --rdzv_endpoint=master:29400 \
    --max_restarts=3 \      # 最大重启次数
    train.py
```

### 8.5 故障检测与恢复

**常见故障类型**：

| 故障类型 | 检测方式 | 恢复方式 |
|---------|---------|---------|
| GPU 硬件故障 | NCCL timeout / ECC error | 重启节点 + 恢复检查点 |
| GPU 显存溢出 (OOM) | CUDA OOM exception | 减小 batch / 检查内存泄漏 |
| 网络故障 | NCCL timeout | 重新建立通信组 |
| 静默数据损坏 (SDC) | Loss 异常检测 | 回滚到更早的检查点 |
| 进程崩溃 | Heartbeat timeout | torchrun 自动重启 |

**静默数据损坏 (SDC)** 是最危险的故障类型——GPU 计算结果错误但不报错。大规模集群中 SDC 概率不可忽略：
- LLaMA 3 团队报告在训练中遇到多次 SDC
- 检测方法：定期检查 loss 是否异常尖峰、对比不同 replica 的结果
- 缓解：使用 ECC 显存、定期 validation loss 检查

---

## 9. 代码示例

### 9.1 PyTorch DDP 基础设置

```python
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)

    # 模型
    model = MyLargeModel().to(rank)
    model = DDP(model, device_ids=[rank])

    # 数据
    dataset = MyDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # 确保每个 epoch shuffle 不同
        for batch in dataloader:
            batch = batch.to(rank)
            loss = model(batch)
            loss.backward()       # DDP 自动触发 AllReduce
            optimizer.step()
            optimizer.zero_grad()

    cleanup()

# 启动
# torchrun --nproc_per_node=8 train.py
```

### 9.2 FSDP 配置

```python
import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import LlamaDecoderLayer

# 混合精度策略
mp_policy = MixedPrecision(
    param_dtype=torch.bfloat16,     # 计算使用 BF16
    reduce_dtype=torch.bfloat16,    # AllReduce 使用 BF16
    buffer_dtype=torch.bfloat16,
)

# 自动包装策略: 每个 Transformer 层是一个 FSDP 单元
auto_wrap_policy = transformer_auto_wrap_policy(
    transformer_layer_cls={LlamaDecoderLayer}
)

# FSDP 包装
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,   # ZeRO-3
    # ShardingStrategy.SHARD_GRAD_OP → ZeRO-2
    # ShardingStrategy.NO_SHARD → DDP
    mixed_precision=mp_policy,
    auto_wrap_policy=auto_wrap_policy,
    cpu_offload=CPUOffload(offload_params=False),     # 可选 CPU offload
    device_id=torch.cuda.current_device(),
    limit_all_gathers=True,  # 限制同时进行的 AllGather 数量
)
```

### 9.3 混合精度 + 梯度累积 完整示例

```python
import torch
from torch.amp import autocast, GradScaler
from contextlib import nullcontext

# 配置
use_bf16 = True  # H100/A100 推荐 BF16
dtype = torch.bfloat16 if use_bf16 else torch.float16
accum_steps = 4
max_grad_norm = 1.0

# BF16 不需要 GradScaler
scaler = None if use_bf16 else GradScaler()

optimizer.zero_grad()
for step, batch in enumerate(dataloader):
    # 梯度累积: 中间步骤不同步
    is_accumulating = (step + 1) % accum_steps != 0
    sync_context = model.no_sync() if is_accumulating else nullcontext()

    with sync_context:
        with autocast(device_type='cuda', dtype=dtype):
            loss = model(batch) / accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

    if not is_accumulating:
        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        optimizer.zero_grad()
```

### 9.4 激活检查点配置

```python
from torch.utils.checkpoint import checkpoint

class TransformerLayerWithCheckpoint(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        # 使用 checkpoint 包装, 前向时不保存中间激活
        return checkpoint(self.layer, x, use_reentrant=False)

# FSDP + 激活检查点
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    apply_activation_checkpointing,
    CheckpointImpl,
)

# 对每个 Transformer 层应用检查点
apply_activation_checkpointing(
    model,
    checkpoint_wrapper_fn=lambda m: checkpoint_wrapper(
        m, checkpoint_impl=CheckpointImpl.NO_REENTRANT
    ),
    check_fn=lambda m: isinstance(m, LlamaDecoderLayer),
)
```

---

## 10. 大规模训练案例

### 10.1 LLaMA 3.1 405B (Meta, 2024)

| 配置项 | 值 |
|--------|-----|
| 参数量 | 405B (dense) |
| GPU | 16,384 × H100 80GB |
| 互联 | 400 Gbps RoCE (节点间), NVLink (节点内) |
| 并行策略 | TP=8, PP=16, DP=128 |
| 序列长度 | 8K → 128K (分阶段) |
| 训练数据 | 15.6T token |
| 精度 | BF16 混合精度 |
| 容错 | 平均 ~12 小时一次故障，自动检测 + 检查点恢复 |
| 关键技术 | 选择性激活检查点, 异步检查点, NCCL 调优 |
| 训练效率 | ~38-43% MFU (Model FLOPs Utilization) |

**经验教训**：
- 在 16K GPU 规模，网络故障比 GPU 故障更频繁
- 静默数据损坏 (SDC) 需要额外监控
- 分阶段训练（先短序列后长序列）提高效率

### 10.2 DeepSeek-V3 (DeepSeek, 2024)

| 配置项 | 值 |
|--------|-----|
| 参数量 | 671B (MoE, 256 experts, top-8, 37B active) |
| GPU | 2,048 × H800 80GB |
| 互联 | IB (节点间), NVLink (节点内) |
| 并行策略 | TP=1, PP=16, DP=128, EP=64 |
| 训练数据 | 14.8T token |
| 精度 | **FP8 混合精度** (首创大规模使用) |
| 训练成本 | ~$5.5M (vs LLaMA 3.1 405B 的 ~$100M+) |
| 关键技术 | DualPipe, FP8 训练, Multi-Token Prediction, 无辅助损失负载均衡 |

**DualPipe 流水线调度**：DeepSeek-V3 提出的新 PP 调度，双向同时推进前向和反向传播，显著减少气泡时间。

**为什么成本低 10-20×**：
1. MoE 架构 → 每 token 只计算 37B 参数
2. FP8 训练 → 计算吞吐量翻倍
3. 无 TP → 节省大量节点内通信
4. 高效的辅助损失设计 → 专家利用率高

### 10.3 训练规模 vs 效率趋势

```
MFU (Model FLOPs Utilization) 趋势:
  单节点小模型:     50-60%
  数百 GPU:         40-50%
  数千 GPU:         35-45%
  16K+ GPU:         30-43%

GPU 利用率下降原因:
  1. 通信占比增加
  2. PP 气泡
  3. 故障恢复的停机时间
  4. 负载不均衡 (MoE)
  5. 检查点 I/O
```

---

## 11. 参考文献

- Rajbhandari et al., *"ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"* (SC'20)
- Rajbhandari et al., *"ZeRO-Infinity: Breaking the GPU Memory Wall"* (SC'21)
- Wang et al., *"ZeRO++: Extremely Efficient Collective Communication"* (2023)
- Micikevicius et al., *"Mixed Precision Training"* (ICLR 2018)
- Narayanan et al., *"Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM"* (SC'21)
- Zhao et al., *"PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel"* (VLDB 2023)
- Liang et al., *"TorchTitan: One-stop PyTorch Native Solution for Production Ready LLM Pre-training"* (ICLR 2025)
- Chen et al., *"Training Deep Nets with Sublinear Memory Cost"* (arXiv 2016) — 激活检查点
- Meta, *"Llama 3.1 Model Card and Technical Report"* (2024)
- DeepSeek, *"DeepSeek-V3 Technical Report"* (2024)

---

## 12. 相关页面

- [[parallelism-strategies-deep-dive]] — DP / TP / EP / EDP / ETP 切分原理（训练+推理通用）
- [[model-parallelism]] — 推理中的并行策略与 PP 调度
- [[training-frameworks]] — Megatron-LM、DeepSpeed、FSDP 框架实现
- [[gpu-cluster-management]] — GPU 集群管理与硬件拓扑
- [[checkpointing]] — 检查点保存与恢复的详细讨论
- [[rl-training-frameworks]] — 强化学习训练的分布式策略
