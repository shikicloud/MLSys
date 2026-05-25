---
title: "预填充-解码分离架构"
category: llm-inference
tags: [prefill-decode, 分离部署, splitwise, distserve, mooncake, kv传输]
created: 2026-04-13
updated: 2026-05-13
status: mature
---

# 预填充-解码分离架构

> [!abstract]+ TL;DR
> LLM 推理两阶段计算特性差异巨大：**预填充**（计算密集型，处理完整 prompt）vs. **解码**（内存带宽密集型，逐 token 生成）。共置在同一 GPU 池时，长预填充会阻塞并发解码，导致 TPOT 膨胀 **2–30 倍**。预填充-解码分离把两阶段物理分到不同 GPU 池，独立优化和扩展。2025–2026 成为生产默认架构 —— **NVIDIA Dynamo、llm-d、[[vllm|vLLM]]、[[sglang|SGLang]]** 原生支持。关键系统：**DistServe**（OSDI 2024）、**Splitwise**（ISCA 2024）、**Mooncake**（FAST 2025）。

```
分离架构的核心思想：

  共置架构（传统）：           分离架构：
  ┌─────────────────┐         ┌──────────┐    ┌──────────┐
  │  GPU Pool        │         │ Prefill  │    │ Decode   │
  │  ┌────┐ ┌────┐  │         │ Pool     │    │ Pool     │
  │  │P+D │ │P+D │  │         │ ┌────┐   │    │ ┌────┐   │
  │  └────┘ └────┘  │         │ │ P  │   │    │ │ D  │   │
  │  ┌────┐ ┌────┐  │    →    │ └────┘   │    │ └────┘   │
  │  │P+D │ │P+D │  │         │ ┌────┐   │ KV │ ┌────┐   │
  │  └────┘ └────┘  │         │ │ P  │──────►│ D  │   │
  │  互相干扰!       │         │ └────┘   │    │ └────┘   │
  └─────────────────┘         │ 高算力   │    │ 大内存   │
                              └──────────┘    └──────────┘
```

到 2025-2026 年，分离架构已经成为大规模 LLM 服务的**默认部署方式**，被 NVIDIA Dynamo、llm-d、[[vllm|vLLM]]、[[sglang|SGLang]] 等主流框架原生支持。

---

## Prefill vs Decode 特性对比

### 计算特性差异

预填充和解码阶段有着截然不同的计算特性：

```
预填充阶段（Prefill）：
  - 输入: 完整的 prompt (数百到数万 tokens)
  - 操作: 一次性处理所有 token 的注意力计算
  - 特点: 大矩阵乘法，高算术强度
  - 瓶颈: GPU 计算能力 (FLOPS)
  - 类比: "阅读整本书" —— 计算量大但只做一次

解码阶段（Decode）：
  - 输入: 单个 token (上一步生成的)
  - 操作: 与所有历史 KV 缓存做注意力计算
  - 特点: 小矩阵乘法，低算术强度，大量内存读取
  - 瓶颈: GPU 内存带宽 (GB/s)
  - 类比: "逐字写作" —— 每步计算少但需要反复读取记忆
```

### 定量对比

```
Prefill vs Decode 定量对比（Llama 3.1 70B, A100 80GB）：

┌────────────────────┬──────────────────┬──────────────────┐
│ 指标                │ Prefill          │ Decode           │
├────────────────────┼──────────────────┼──────────────────┤
│ 输入 token 数      │ N (数百~数万)     │ 1                │
│ 计算量 (FLOPS)     │ ~2 × P × N       │ ~2 × P           │
│ 算术强度           │ 高 (~100+ ops/B) │ 低 (~1 ops/B)    │
│ (Arithmetic        │                  │                  │
│  Intensity)        │                  │                  │
│ GPU 计算利用率     │ 60-80%           │ 1-5%             │
│ 内存带宽利用率     │ 20-40%           │ 80-95%           │
│ 最佳批次大小       │ 1-4              │ 64-512           │
│ 延迟指标           │ TTFT             │ TPOT / TBT       │
│ 可并行性           │ 高 (TP 有效)     │ 低 (DP 更有效)   │
│ 耗时占比 (典型)    │ 10-30%           │ 70-90%           │
└────────────────────┴──────────────────┴──────────────────┘

P = 模型参数数量, N = 输入 token 数
```

### Roofline 模型分析

```
Roofline 模型：Prefill vs Decode

性能
(TFLOPS)  │
          │              ╱ ← 计算上限 (A100: 312 TFLOPS FP16)
    312 ──│─ ─ ─ ─ ─ ─╱─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
          │          ╱
          │        ╱        ★ Prefill (算术强度 ~100)
    200 ──│      ╱          → 接近计算上限
          │    ╱
          │  ╱
     50 ──│╱
          │  ▲ Decode (算术强度 ~1)
          │  → 远低于计算上限
          │  → 受内存带宽限制
          │
          └──────────────────────────────────────
          1    10    100    1000
                算术强度 (FLOPS/Byte)
                    ↑            ↑
                  Decode       Prefill
              (内存带宽受限)  (计算受限)
```

核心洞察：**同一个 GPU 无法同时为两种截然不同的工作负载提供最优配置**。这是分离架构的根本动机。

---

## 为什么要分离

### 问题 1：预填充干扰解码延迟

这是分离架构的首要动机。当 prefill 和 decode 共置时：

```
头部阻塞 (Head-of-Line Blocking) 问题：

场景：3 个请求正在解码，1 个新请求到达需要预填充

共置架构下的执行：
时间步:    1    2    3    4    5    6    7    8    9   10
         ┌────────────────────────────┐
新请求P:  │      预填充 (长 prompt)     │                    ← 占用 GPU
         └────────────────────────────┘
解码 A:   ■    ·    ·    ·    ·    ·    ·    ■    ■    ■    ← TPOT 膨胀!
解码 B:   ■    ·    ·    ·    ·    ·    ·    ■    ■    ■    ← TPOT 膨胀!
解码 C:   ■    ·    ·    ·    ·    ·    ·    ■    ■    ■    ← TPOT 膨胀!

■ = 正常解码步    · = 被预填充阻塞，无法解码

正常 TPOT:  ~30ms
被阻塞 TPOT: ~210ms (7x 膨胀)
```

分块预填充可以缓解但不能消除：

```
分块预填充（部分缓解）：
时间步:    1    2    3    4    5    6    7    8    9   10
         ┌──┐      ┌──┐      ┌──┐      ┌──┐
新请求P:  │C1│      │C2│      │C3│      │C4│
         └──┘      └──┘      └──┘      └──┘
解码 A:        ■         ■         ■         ■    ■    ■
解码 B:        ■         ■         ■         ■    ■    ■
解码 C:        ■         ■         ■         ■    ■    ■

缓解了，但仍然有干扰：
- 预填充块占用计算时间 → 解码频率降低
- 小块大小 → TTFT 增加
- 权衡仍然存在
```

### 问题 2：不同的最优批次大小

```
最优批次大小差异：

Prefill:
  - 单个请求就能利用大部分 GPU 计算
  - 最优批次大小: 1-4
  - 更大批次 → 超出 GPU 显存或计算能力

Decode:
  - 单个请求只能利用 ~1% GPU 计算
  - 最优批次大小: 64-512
  - 需要大批次才能摊薄模型权重读取开销

共置时的矛盾：
  如果按 prefill 优化 (小批次) → decode 利用率极低
  如果按 decode 优化 (大批次)  → prefill 可能 OOM
  两者的最优配置互相矛盾！
```

### 问题 3：不同的并行策略

```
最优并行策略差异：

Prefill（计算密集）→ 张量并行 (TP) 更有效：
  - 大矩阵运算可以高效切分到多 GPU
  - 通信开销被大计算量摊薄
  - 典型配置: TP=4 或 TP=8

Decode（内存密集）→ 数据并行 (DP) 更有效：
  - 每个请求计算量小，TP 的通信开销相对过大
  - 不如用更多独立实例处理更大批次
  - 典型配置: DP=N, TP=1 或 TP=2

分离架构允许两个池独立选择并行策略：

  ┌─────────────────┐        ┌─────────────────┐
  │   Prefill Pool   │        │   Decode Pool    │
  │                  │        │                  │
  │  TP=4, 少量实例  │  ──►   │  DP=8, 多实例    │
  │  高算力利用      │  KV    │  高带宽利用      │
  │                  │        │                  │
  └─────────────────┘        └─────────────────┘
```

### 问题 4：独立扩缩容

```
负载变化下的扩缩容：

场景 1: 大量短 prompt，长输出 (如对话)
  → Prefill 负载低，Decode 负载高
  → 需要更多 Decode 实例

场景 2: 大量长 prompt，短输出 (如摘要)
  → Prefill 负载高，Decode 负载低
  → 需要更多 Prefill 实例

共置架构: 只能整体扩缩容，无法针对性优化
分离架构: 两个池独立扩缩容，资源利用率最优
```

---

## 分离架构设计

### 通用架构

```
预填充-解码分离通用架构：

  客户端请求
      │
      ▼
┌──────────────┐
│  路由/调度器   │  ← 全局请求路由
│  (Router)     │
└──────┬───────┘
       │
  ┌────┴────┐
  ▼         ▼
┌────────┐  ┌────────────┐
│Prefill │  │  Decode     │
│Pool    │  │  Pool       │
│        │  │             │
│ GPU 0  │  │  GPU A      │
│ GPU 1  │  │  GPU B      │
│ GPU 2  │  │  GPU C      │
│ ...    │  │  GPU D      │
│        │  │  ...        │
└───┬────┘  └─────┬──────┘
    │              ▲
    │   KV Cache   │
    └──────────────┘
      传输 (Transfer)
```

请求处理流程：

```
分离架构的请求处理流程：

1. 客户端发送请求 (prompt + 采样参数)
      │
      ▼
2. 路由器接收请求，发送到 Prefill Pool
      │
      ▼
3. Prefill 实例处理 prompt
   - 执行完整的前向传播
   - 生成 KV 缓存 + 第一个输出 token
      │
      ▼
4. KV 缓存传输到 Decode Pool
   - 通过 RDMA/NVLink/PCIe
   - 这是关键瓶颈！
      │
      ▼
5. Decode 实例继续自回归生成
   - 接收 KV 缓存
   - 逐 token 生成直到 EOS
      │
      ▼
6. 结果返回给客户端
   - 可以流式返回
```

### Splitwise (Microsoft, ISCA 2024)

Splitwise 是第一个系统性提出 PD 分离的工作。

核心思想：
- 利用**异构硬件**：计算优化的 GPU 做 prefill，内存优化的 GPU 做 decode
- 同一台机器内的 GPU 可以通过 NVLink 高速传输 KV 缓存
- 引入 **mixed** 模式：当一种类型 GPU 空闲时，可以临时执行另一种任务

```
Splitwise 架构：

┌────────────────────────────────────────────┐
│                同一台机器                    │
│                                            │
│  ┌──────────────┐    NVLink    ┌──────────────┐
│  │ GPU 0 (H100) │ ◄─────────► │ GPU 1 (H100) │
│  │  Prefill     │   900 GB/s  │  Decode       │
│  │  (高算力)    │             │  (高带宽)     │
│  └──────────────┘             └──────────────┘
│                                            │
└────────────────────────────────────────────┘

性能：
  - 吞吐量提升 1.4x
  - 成本降低 20%
  - 关键洞察：机器内 NVLink 传输几乎免费
```

局限性：
- 仅考虑机器内分离（NVLink），未考虑跨机器场景
- 异构硬件假设在实际部署中不一定成立

### DistServe (OSDI 2024)

DistServe 将分离扩展到集群级别，支持跨机器的 PD 分离。

核心创新：
- **拉取式调度（Pull-Based Scheduling）**：Decode 实例主动从 Prefill 实例拉取准备好的 KV 缓存
- **Goodput 优化**：在满足 SLO 约束下最大化有效吞吐量
- 支持 prefill 和 decode 使用不同的并行策略

```
DistServe 架构：

  ┌─────────────────────────────────────────┐
  │            全局调度器                     │
  │  (Goodput 优化 + SLO 感知)              │
  └────────────┬──────────────┬─────────────┘
               │              │
         ┌─────┴─────┐  ┌────┴──────┐
         │ Prefill    │  │ Decode    │
         │ Instance 0 │  │ Instance 0│ ← 拉取 KV
         │ (TP=4)     │  │ (TP=1)   │
         ├────────────┤  ├──────────┤
         │ Prefill    │  │ Decode    │
         │ Instance 1 │  │ Instance 1│ ← 拉取 KV
         │ (TP=4)     │  │ (TP=1)   │
         └────────────┘  ├──────────┤
                         │ Decode    │
                         │ Instance 2│
                         │ (TP=1)   │
                         ├──────────┤
                         │ ...      │
                         └──────────┘

  Prefill: 少量高 TP 实例
  Decode:  大量低 TP 实例
```

性能：
- 相比共置系统，在相同 SLO 下 goodput 提升 **7.4x**
- 在更严格 SLO 下提升 **12.6x**
- 验证了跨机器分离的可行性和优势

### Mooncake (Moonshot AI, FAST 2025 最佳论文)

Mooncake 是月之暗面（Moonshot AI）的生产系统，服务于 Kimi 产品，日处理超过 **1000 亿 token**。

```
Mooncake 架构：

                    ┌─────────────────────┐
                    │    Conductor         │
                    │  (全局协调器)         │
                    └──────────┬──────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
         ▼                     ▼                     ▼
  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
  │  Prefill Node │    │  KV Cache    │    │  Decode Node  │
  │              │    │  Pool        │    │              │
  │  GPU Cluster │    │              │    │  GPU Cluster │
  │  (高算力)    │    │  CPU DRAM    │    │  (高带宽)    │
  │              │───►│  SSD         │───►│              │
  │  TP=4, EP=32│    │  (池化存储)   │    │  TP=4, DP=N  │
  └──────────────┘    └──────────────┘    └──────────────┘
                            ↑
                    Transfer Engine
                    (RDMA + 零拷贝)
```

Mooncake 的关键创新：

1. **以 KV 缓存为中心**：将 KV 缓存视为独立的存储资源，而非 GPU 的附属品
2. **池化存储**：CPU DRAM + SSD 组成的分布式 KV 缓存池
3. **Transfer Engine**：基于 RDMA 的高性能传输引擎，支持零拷贝
4. **预测性调度**：基于请求特征预测输出长度，优化资源分配

性能：
- 容量提升 **59-498%**（取决于负载类型）
- Transfer Engine 已集成到 [[vllm|vLLM]] V1
- 在生产环境中验证了大规模分离架构的可行性

### TetriInfer

TetriInfer 进一步优化了分离架构的调度，利用类似俄罗斯方块（Tetris）的思想将 prefill 和 decode 请求紧密拼接以最大化 GPU 利用率。

```
TetriInfer 的"拼接"调度：

传统调度:                  TetriInfer 调度:
┌──────────┐               ┌──────────┐
│ Prefill  │               │PP│DD│PP│DD│  ← 紧密拼接
│   大块   │               │DD│PP│DD│PP│  ← 消除空闲
│          │               │PP│DD│PP│DD│
├──────────┤               └──────────┘
│ Decode   │
│  小块    │               更高 GPU 利用率
│  (空闲)  │
└──────────┘
```

---

## KV 缓存传输

### 核心瓶颈

KV 缓存传输是 PD 分离架构中**最关键的挑战**。预填充完成后，生成的 KV 缓存必须从 Prefill GPU 传输到 Decode GPU，这个传输时间直接影响 TTFT。

```
KV 缓存大小估算：

KV 缓存大小 = 2 × num_layers × hidden_dim × num_kv_heads × seq_len × dtype_size

示例（Llama 3.1 70B, BF16）：
  - num_layers = 80
  - hidden_dim = 8192
  - num_kv_heads = 8 (GQA)
  - dtype_size = 2 bytes (BF16)
  
  每 token KV 大小 = 2 × 80 × 8192 × 8 × 2 / 8192 = 2.5 MB/token
                                          (除以 GQA group)
  
  ┌─────────────┬────────────────┬─────────────────┐
  │ 序列长度     │ KV 缓存大小     │ 传输时间 (PCIe) │
  ├─────────────┼────────────────┼─────────────────┤
  │ 1K tokens   │ ~0.32 GB       │ ~10ms           │
  │ 4K tokens   │ ~1.25 GB       │ ~40ms           │
  │ 32K tokens  │ ~10 GB         │ ~320ms          │
  │ 128K tokens │ ~40 GB         │ ~1.28s          │
  └─────────────┴────────────────┴─────────────────┘
  
  注：PCIe Gen4 x16 ≈ 32 GB/s, 实际传输约 25-28 GB/s
```

### 传输方式对比

```
KV 缓存传输方式对比：

┌────────────────┬────────────┬──────────────┬──────────────────┐
│ 传输方式        │ 带宽        │ 延迟 (32K)   │ 适用场景          │
├────────────────┼────────────┼──────────────┼──────────────────┤
│ NVLink (同机)  │ 900 GB/s   │ ~6ms         │ 同机器分离        │
│ PCIe Gen4      │ 32 GB/s    │ ~32ms        │ 同机器/跨机器     │
│ PCIe Gen5      │ 64 GB/s    │ ~16ms        │ 新一代硬件        │
│ RDMA (IB)      │ 100-400    │ ~10-40ms     │ 跨机器分离        │
│                │ Gbps       │              │                  │
│ TCP/IP         │ 10-100     │ ~100-1000ms  │ 不推荐            │
│                │ Gbps       │              │                  │
└────────────────┴────────────┴──────────────┴──────────────────┘
```

### Mooncake Transfer Engine

Mooncake 的 Transfer Engine 是目前最成熟的 KV 缓存传输解决方案：

```
Mooncake Transfer Engine 架构：

  Prefill GPU                      Decode GPU
  ┌──────────┐                     ┌──────────┐
  │ KV Cache │                     │ KV Cache │
  │ (GPU Mem)│                     │ (GPU Mem)│
  └────┬─────┘                     └────▲─────┘
       │  GPUDirect                     │  GPUDirect
       │  RDMA                          │  RDMA
       ▼                                │
  ┌──────────┐    RDMA fabric     ┌──────────┐
  │   NIC    │ ◄───────────────► │   NIC    │
  └──────────┘    零拷贝传输      └──────────┘

特点：
  - GPUDirect RDMA: GPU 内存直接通过网卡传输，不经过 CPU
  - 零拷贝: 无需 GPU→CPU→网卡 的多次复制
  - 流水线化: 边生成边传输，重叠计算和通信
  - 已集成到 vLLM V1
```

### KV 缓存压缩

为了减少传输量，可以对 KV 缓存进行压缩：

```
KV 缓存压缩策略：

┌──────────────────┬────────────┬──────────┬────────────────┐
│ 压缩方法          │ 压缩比      │ 精度损失  │ 说明            │
├──────────────────┼────────────┼──────────┼────────────────┤
│ FP16 → INT8      │ 2x         │ 极小      │ 量化 KV 缓存    │
│ FP16 → INT4      │ 4x         │ 小        │ 更激进的量化     │
│ Token 剪枝       │ 1.5-3x     │ 可控      │ 丢弃不重要 token │
│ 稀疏注意力       │ 2-10x      │ 可控      │ 只保留关键 token │
│ 低秩近似         │ 2-4x       │ 中等      │ SVD 压缩         │
└──────────────────┴────────────┴──────────┴────────────────┘
```

### 延迟分析：何时传输优于重算

一个关键决策点是：**何时应该传输 KV 缓存，何时应该在 Decode 端重新计算？**

```
传输 vs 重算的决策：

传输时间 = KV_size / bandwidth
重算时间 = prefill_time(seq_len)

                 传输更优    │    重算更优
                            │
  传输时间  ──────────────X─────────────
                            │
  重算时间  ────X───────────────────────
                │           │
                │           │
            短序列       长序列+低带宽

决策规则：
  if 传输时间 < 重算时间:
      选择传输 (大多数情况)
  else:
      选择重算 (短序列 + 高延迟网络)

典型阈值（Llama 70B, RDMA 100Gbps）：
  序列长度 < ~256 tokens → 重算可能更快
  序列长度 > ~256 tokens → 传输更快
```

### 流水线化传输

高级优化：将 KV 缓存传输与计算重叠。

```
流水线化传输（Layer-by-Layer Streaming）：

无流水线：
  Prefill: [████ 计算所有层 ████]
  传输:                           [████ 传输所有 KV ████]
  Decode:                                                [████ 开始解码 ████]

流水线化：
  Prefill: [Layer0][Layer1][Layer2][Layer3]...
  传输:         [KV0 ][KV1 ][KV2 ][KV3 ]...
  Decode:                              [开始解码] ← 更早开始!

  节省时间: 传输时间被计算时间"掩盖"
```

---

## DeepSeek-V3 的分离部署

DeepSeek-V3/R1 是分离架构在大规模生产系统中的标志性案例，展示了如何为 MoE 模型优化 PD 分离配置。

### 部署配置

```
DeepSeek-V3 分离部署：

┌──────────────────────────────────────────────────────┐
│                  Prefill Pool                         │
│                                                      │
│  32 GPUs (4 nodes × 8 H800)                         │
│  TP = 4, EP = 32                                     │
│  专家并行覆盖所有 32 GPU → 每个专家一个 GPU           │
│  高算力利用，大批次                                   │
│                                                      │
│  特点:                                               │
│  - MoE 的 all-to-all 通信在 prefill 时可以被         │
│    大计算量掩盖                                       │
│  - 高 TP 度确保单请求 TTFT 低                         │
└──────────────────────────────┬───────────────────────┘
                               │
                          KV 缓存传输
                         (RDMA 网络)
                               │
┌──────────────────────────────▼───────────────────────┐
│                  Decode Pool                          │
│                                                      │
│  320 GPUs (40 nodes × 8 H800)                        │
│  TP = 4, EP = 320                                    │
│  10x Prefill Pool 的规模                              │
│                                                      │
│  特点:                                               │
│  - 大量 GPU 提供足够的内存带宽和 KV 缓存空间          │
│  - 更多专家并行 → 每个 GPU 只需处理少量专家           │
│  - 每个 GPU 承载更多并发请求                          │
│  - MoE all-to-all 通信在 decode 时开销更低            │
│    (每步只有 1 个 token)                              │
└──────────────────────────────────────────────────────┘
```

### 为什么 Decode 比 Prefill 需要更多 GPU

```
Decode 需要更多 GPU 的原因：

1. 时间占比：Decode 阶段耗时占 70-90%
   - 100 token 输出 = 100 个 decode 步骤
   - 每步只生成 1 个 token，但需要读取全部模型权重

2. 内存需求：每个并发请求都需要 KV 缓存空间
   - 1000 个并发 decode 请求 × 10GB/请求 = 10TB KV 缓存
   - 需要分布在大量 GPU 上

3. 带宽需求：Decode 受内存带宽限制
   - 更多 GPU = 更多总带宽
   - A100 80GB: 2TB/s → 10 块 = 20TB/s

典型 Prefill:Decode GPU 比例：
  - 短输出场景 (对话): 1:3 到 1:5
  - 长输出场景 (代码): 1:8 到 1:10
  - DeepSeek-V3:       1:10 (32:320)
```

### 与并行策略的关系

DeepSeek-V3 的分离配置与其 [[parallelism-strategies-deep-dive|并行策略]] 紧密相关：

```
DeepSeek-V3 的并行配置细节：

Prefill Pool (32 GPUs):
  ├── TP = 4 (4 GPU 一组处理同一请求的同一层)
  ├── EP = 32 (每个 GPU 承载不同的 MoE 专家)
  └── 等效于 32/4 = 8 个 TP 组
      每个 TP 组处理 32/8 = 4 个专家

Decode Pool (320 GPUs):
  ├── TP = 4 (保持与 prefill 相同，简化 KV 传输)
  ├── EP = 320 (更大的专家并行度)
  └── 等效于 320/4 = 80 个 TP 组
      每个 TP 组处理更少的专家 → 更快

关键设计考量：
  - TP 保持一致 (=4): KV 缓存的 shape 相同，传输简化
  - EP 不同: 各自优化专家分布
  - Prefill 的 EP=32: 专家利用率高，计算密集
  - Decode 的 EP=320: 每 GPU 专家少，内存开销低
```

---

## 代码示例

### vLLM PD 分离配置

```python
"""vLLM 预填充-解码分离配置示例"""

# === Prefill 实例配置 ===
# 文件: prefill_config.yaml

prefill_config = {
    "model": "deepseek-ai/DeepSeek-V3",
    "tensor_parallel_size": 4,
    
    # 分离模式配置
    "served_model_name": "deepseek-v3",
    "kv_transfer_config": {
        "kv_connector": "MooncakeConnector",  # 使用 Mooncake Transfer Engine
        "kv_role": "kv_producer",             # Prefill = KV 生产者
        "kv_rank": 0,
        "kv_parallel_size": 2,                # 2 个 prefill 实例
    },
    
    # Prefill 优化参数
    "max_num_seqs": 32,                # 较小批次（计算密集）
    "max_num_batched_tokens": 8192,    # 大 token 预算（处理长 prompt）
    "gpu_memory_utilization": 0.85,
}

# === Decode 实例配置 ===
# 文件: decode_config.yaml

decode_config = {
    "model": "deepseek-ai/DeepSeek-V3",
    "tensor_parallel_size": 4,
    
    # 分离模式配置
    "kv_transfer_config": {
        "kv_connector": "MooncakeConnector",
        "kv_role": "kv_consumer",             # Decode = KV 消费者
        "kv_rank": 0,
        "kv_parallel_size": 10,               # 10 个 decode 实例
    },
    
    # Decode 优化参数
    "max_num_seqs": 512,               # 大批次（内存带宽密集）
    "max_num_batched_tokens": 2048,    # 较小 token 预算
    "gpu_memory_utilization": 0.92,    # 更多内存给 KV 缓存
}
```

```bash
# 启动 Prefill 实例
vllm serve deepseek-ai/DeepSeek-V3 \
    --tensor-parallel-size 4 \
    --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer","kv_rank":0}' \
    --max-num-seqs 32 \
    --port 8100

# 启动 Decode 实例
vllm serve deepseek-ai/DeepSeek-V3 \
    --tensor-parallel-size 4 \
    --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer","kv_rank":0}' \
    --max-num-seqs 512 \
    --port 8200
```

### 简单的分离架构伪代码

```python
"""预填充-解码分离架构的简化实现"""

from dataclasses import dataclass
from typing import Optional
import asyncio
from collections import deque


@dataclass
class InferenceRequest:
    request_id: str
    prompt_tokens: list[int]
    max_output_tokens: int
    priority: int = 0


@dataclass
class KVCacheHandle:
    """KV 缓存的引用句柄"""
    request_id: str
    prefill_node_id: str
    kv_cache_address: int       # 远程内存地址
    num_layers: int
    seq_len: int
    size_bytes: int


class PrefillInstance:
    """预填充实例 —— 负责处理 prompt 并生成 KV 缓存"""
    
    def __init__(self, model, gpu_id: int, transfer_engine):
        self.model = model
        self.gpu_id = gpu_id
        self.transfer_engine = transfer_engine
        self.node_id = f"prefill_{gpu_id}"
    
    async def process_prefill(
        self, request: InferenceRequest
    ) -> tuple[int, KVCacheHandle]:
        """
        处理预填充请求。
        返回: (第一个输出 token, KV 缓存句柄)
        """
        # 1. 在 GPU 上执行预填充前向传播
        kv_cache, first_token = self.model.prefill(
            request.prompt_tokens
        )
        
        # 2. 注册 KV 缓存到传输引擎（使其可被远程访问）
        kv_handle = self.transfer_engine.register_kv(
            request_id=request.request_id,
            kv_cache=kv_cache,
            node_id=self.node_id,
        )
        
        return first_token, kv_handle


class DecodeInstance:
    """解码实例 —— 负责自回归 token 生成"""
    
    def __init__(self, model, gpu_id: int, transfer_engine):
        self.model = model
        self.gpu_id = gpu_id
        self.transfer_engine = transfer_engine
        self.active_requests: dict[str, "DecodeState"] = {}
    
    async def accept_request(
        self,
        request: InferenceRequest,
        kv_handle: KVCacheHandle,
        first_token: int,
    ):
        """接收来自 prefill 实例的请求和 KV 缓存"""
        # 1. 通过 RDMA 拉取 KV 缓存
        local_kv = await self.transfer_engine.fetch_kv(kv_handle)
        
        # 2. 初始化解码状态
        self.active_requests[request.request_id] = DecodeState(
            request=request,
            kv_cache=local_kv,
            output_tokens=[first_token],
        )
    
    async def decode_step(self) -> list[str]:
        """执行一步解码（所有活跃请求）"""
        if not self.active_requests:
            return []
        
        # 批量解码
        batch_inputs = {
            req_id: state.output_tokens[-1]
            for req_id, state in self.active_requests.items()
        }
        
        new_tokens = self.model.decode_batch(batch_inputs)
        
        # 更新状态，检查完成
        finished = []
        for req_id, token in new_tokens.items():
            state = self.active_requests[req_id]
            state.output_tokens.append(token)
            
            EOS = 2
            if (token == EOS 
                    or len(state.output_tokens) >= 
                       state.request.max_output_tokens):
                finished.append(req_id)
        
        # 清理已完成的请求
        for req_id in finished:
            del self.active_requests[req_id]
        
        return finished


class DisaggregatedRouter:
    """全局路由器 —— 协调 prefill 和 decode 实例"""
    
    def __init__(
        self,
        prefill_instances: list[PrefillInstance],
        decode_instances: list[DecodeInstance],
    ):
        self.prefill_pool = prefill_instances
        self.decode_pool = decode_instances
        self.request_queue: deque[InferenceRequest] = deque()
    
    def select_prefill_instance(self) -> PrefillInstance:
        """选择负载最低的 prefill 实例"""
        return min(self.prefill_pool, 
                   key=lambda p: p.current_load())
    
    def select_decode_instance(self) -> DecodeInstance:
        """选择负载最低的 decode 实例"""
        return min(self.decode_pool,
                   key=lambda d: len(d.active_requests))
    
    async def handle_request(self, request: InferenceRequest):
        """处理一个完整的推理请求"""
        # 1. 选择 prefill 实例并执行预填充
        prefill_inst = self.select_prefill_instance()
        first_token, kv_handle = await prefill_inst.process_prefill(
            request
        )
        
        # 2. 选择 decode 实例
        decode_inst = self.select_decode_instance()
        
        # 3. 传输 KV 缓存并开始解码
        await decode_inst.accept_request(
            request, kv_handle, first_token
        )
        
        # 4. 通知 prefill 实例释放 KV 缓存
        prefill_inst.transfer_engine.release_kv(kv_handle)


@dataclass
class DecodeState:
    request: InferenceRequest
    kv_cache: object
    output_tokens: list[int]
```

---

## 性能分析

### 延迟改善

```
分离架构的延迟改善（Llama 70B，A100 集群）：

共置架构 (baseline):
  TTFT (P50):  150ms    TTFT (P99):  800ms
  TPOT (P50):   35ms    TPOT (P99):  250ms    ← prefill 干扰导致

分离架构:
  TTFT (P50):  120ms    TTFT (P99):  400ms    ← 改善 (prefill 独立优化)
  TPOT (P50):   28ms    TPOT (P99):   45ms    ← 大幅改善! 无干扰

                 TPOT P99 改善: 250ms → 45ms (5.6x)
```

### 吞吐量增益

```
吞吐量对比 (满足 SLO 约束下)：

SLO 约束: TTFT < 500ms, TPOT < 100ms

┌──────────────────┬──────────────┬────────────────┐
│ 系统              │ 有效吞吐量    │ vs 共置提升     │
├──────────────────┼──────────────┼────────────────┤
│ 共置 (vLLM)      │ 1.0x         │ baseline       │
│ 分块预填充       │ 1.8x         │ 1.8x           │
│ DistServe        │ 7.4x         │ 7.4x           │
│ Mooncake         │ 5.0-6.0x     │ 5.0-6.0x       │
│ Splitwise        │ 1.4x         │ 1.4x           │
└──────────────────┴──────────────┴────────────────┘

注: DistServe 的高提升来自其在严格 SLO 下的优势。
    Mooncake 的数据来自生产环境，包含了传输开销。
```

### 成本分析

```
成本分析（月度，1000 QPS，Llama 70B）：

共置架构:
  GPU: 64 × A100 (统一配置)
  总计: 64 GPU
  月成本: ~$128,000 (按 $2/GPU-hour)

分离架构:
  Prefill: 8 × A100 (TP=4, 2组)
  Decode:  48 × A100 (TP=2, 24组)
  总计: 56 GPU
  月成本: ~$112,000

节省: ~12.5%

更大收益来自：
  1. 可以为 Decode 使用更便宜的 GPU（内存优化型）
  2. 独立扩缩容避免资源浪费
  3. 更高的有效吞吐量意味着处理相同请求需要更少资源
```

---

## 不足与挑战

### KV 缓存传输开销

最核心的挑战是 KV 缓存传输带来的额外延迟：

```
传输开销分析：

场景：Llama 70B, 不同序列长度

┌──────────────┬──────────┬────────────┬──────────────────────┐
│ 序列长度      │ KV 大小   │ 传输时间    │ 是否值得分离？          │
│              │          │ (RDMA)     │                      │
├──────────────┼──────────┼────────────┼──────────────────────┤
│ 256 tokens   │ 80 MB    │ ~1ms       │ 边界情况               │
│ 1K tokens    │ 320 MB   │ ~3ms       │ ✓ 值得               │
│ 4K tokens    │ 1.25 GB  │ ~10ms      │ ✓ 明显值得            │
│ 32K tokens   │ 10 GB    │ ~80ms      │ ✓ 非常值得            │
│ 128K tokens  │ 40 GB    │ ~320ms     │ ✓ 但传输时间显著       │
└──────────────┴──────────┴────────────┴──────────────────────┘

规律：序列越长，分离的收益越大（prefill 干扰越严重）
     但传输时间也越长（需要高带宽网络）
```

### 系统复杂性

```
分离架构引入的额外复杂性：

1. 网络依赖
   - 需要高速网络（RDMA/InfiniBand）
   - 网络故障影响整个系统
   - 网络拥塞导致 TTFT 波动

2. 状态管理
   - KV 缓存的生命周期跨越两个实例
   - 需要分布式垃圾回收
   - 请求取消需要通知两侧

3. 容错
   - Prefill 实例崩溃：丢失正在处理的请求
   - Decode 实例崩溃：丢失正在生成的请求
   - 传输中断：需要重传或重算

4. 调试困难
   - 问题可能出在 prefill、传输、decode 任一环节
   - 需要端到端的追踪和监控

5. 部署运维
   - 两组不同配置的实例
   - 扩缩容策略更复杂
   - 需要监控两个池的利用率
```

### 何时不值得分离

```
不适合分离的场景：

1. 小模型（参数 < 7B）
   - Prefill 和 decode 都很快
   - 传输开销相对于计算时间过大
   - 共置 + 分块预填充就够用

2. 短序列（< 256 tokens）
   - KV 缓存小，传输快
   - 但 prefill 干扰也小
   - 分离的收益不明显

3. 低流量场景
   - 没有并发请求之间的干扰
   - 分离架构增加固定开销
   - 不如投资于单请求优化

4. 缺乏高速网络
   - 仅有 TCP/IP 时传输开销太大
   - 至少需要 RDMA 或 NVLink
   
5. 交互式/多轮场景中的短 turn
   - 每轮的 prompt 增量很小
   - KV 缓存传输频繁但数据量小
   - 分块预填充可能更合适
```

---

## 与 chunked prefill 的组合

一个很自然的疑问："prefill 都跑到独立节点上了，为什么还需要 [[continuous-batching#分块预填充-chunked-prefill|chunked prefill]]？" 答案是 PD 分离和 chunked prefill 在**两个不同粒度上**解决**两个不同的干扰问题**：

- **PD 分离**消除 *prefill ↔ decode* 在**节点**层面的干扰。
- **Chunked prefill** 消除 *prefill ↔ prefill*（和 *prefill ↔ 在飞 decode*）在 **iteration** 层面的干扰 —— 既在 prefill 池内、也在 decode 池的"扩展 prefill"路径上。

三个分离部署里 chunked prefill 依然承重的具体场景：

**1. Prefill 池内部的 prefill ↔ prefill 干扰**：两条长请求同时到同一个 prefill 节点，仍然要排队：

```
没有 chunked prefill 的 prefill 节点：
  [req A 16K prefill][req B 16K prefill][...]
  req A TTFT = 2.3 s
  req B TTFT = 2.3 s + 2.3 s = 4.6 s    ← B 排 A 后面

启用 chunked prefill 的 prefill 节点：
  [chunk_A1 + chunk_B1][chunk_A2 + chunk_B2]...
  req A TTFT ≈ 2.5 s    ← B 的 chunk 蹭计算，A 略多一点
  req B TTFT ≈ 2.5 s    ← 与 A 几乎并行推进，不再排队等
```

第二条请求的 TTFT 从"等 4.6 秒"变成"和 A 几乎同时拿到首 token"。

**2. Decode 节点上的"扩展 prefill"**：decode 节点严格意义上并不只跑 decode：

- **多轮对话**：新一轮用户输入到达，新 token 必须先 prefill 进已有 KV cache 才能继续 decode。
- **工具调用返回**：返回的工具结果作为新 token 拼回去，要 prefill 一段。
- **投机解码回滚**：被拒的投机序列要回退并重 prefill 一小段。

这些"扩展 prefill"典型 50–2000 token —— 跟首轮 prompt 比短，但仍然长到能卡住节点上正在跑的 decode。Decode 池开 chunked prefill 把这种小段也切开混入。

**3. 池内的流量整形**：PD 分离只解决"角色分离"。每个角色内部仍然需要平滑负载、控制尾延迟、防止个别长请求毒化整批。Chunked prefill 是 prefill 池的负载平滑工具；decode 池上小 chunk 用来驯服上面那种扩展 prefill。

口诀：

```
PD 分离      =  prefill 池 ↔ decode 池        别混
Chunked      =  prefill 池内部 / decode 池的    混着混，但每次混一小段
                "扩展 prefill"
```

两层是不同尺度上的同一种"避免阻塞"思想。正交叠加，不是替代关系。

## 前沿方向

### 注意力-FFN 分离

下一代分离的前沿是将 Transformer 内部的**注意力层**和 **FFN 层**分离到不同硬件 —— 完整介绍见独立页面 [[af-disaggregation]]。简要动机：

```
注意力-FFN 分离：

传统 Transformer 层：
  ┌──────────────────────────┐
  │  Attention  →  FFN       │  ← 两者绑定在同一 GPU
  └──────────────────────────┘

分离后：
  ┌──────────────┐    ┌──────────────┐
  │  Attention   │    │  FFN         │
  │  (内存密集)  │    │  (计算密集)  │
  │  专用硬件    │ →  │  专用硬件    │
  └──────────────┘    └──────────────┘

对 MoE 模型特别自然（DP attention + EP MoE 就是结构性 AF 分离）。
```

### 跨数据中心 PD（Prefill-as-a-Service）

Hybrid attention 模型（Kimi Linear、MiMo-V2-Flash、Qwen3.5-397B、Ring-2.5-1T）把 KVCache 相比 dense GQA 砍了约 13× —— 每实例 KV 吞吐从约 60 Gbps 降到约 5 Gbps，于是 PD 的可部署网络边界从 RDMA 级 fabric **推到跨 DC 的普通以太网**。**PrfaaS**（Moonshot/清华，[arXiv:2604.15039](https://arxiv.org/abs/2604.15039)）基于 Mooncake 做了这件事：选择性把长上下文 prefill（$l > t$）外放到 compute-dense PrfaaS 集群（H200 / Rubin CPX），KVCache 通过以太网流到本地 decode 集群（H20 / LPU）。三件事：长度阈值路由、混合 prefix cache 池、双时间尺度调度。1T hybrid 模型案例研究：vs 同构 PD 吞吐 +54 %、P90 TTFT −64 %。完整论文精读见 [[prfaas]]。

### 全局 KV 缓存管理

```
全局 KV 缓存管理（Mooncake 方向）：

不再将 KV 缓存视为 GPU 的本地资源，
而是作为集群级别的共享存储：

┌────────────────────────────────────┐
│       全局 KV 缓存池                │
│   (分布在 CPU DRAM + SSD 上)       │
│                                    │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ │
│  │KV-A │ │KV-B │ │KV-C │ │KV-D │ │
│  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ │
│     │       │       │       │     │
└─────┼───────┼───────┼───────┼─────┘
      │       │       │       │
  ┌───▼───┐ ┌─▼─┐ ┌──▼──┐ ┌─▼───┐
  │GPU-P0 │ │D0 │ │D1   │ │D2   │
  └───────┘ └───┘ └─────┘ └─────┘

优势：
  - 多个请求可以共享相同前缀的 KV 缓存
  - Decode 实例可以迁移（KV 不在本地）
  - 支持长上下文缓存复用
```

---

## 参考文献

- **Splitwise**: Patel et al., "Splitwise: Efficient Generative LLM Inference Using Phase Splitting", ISCA 2024. [arXiv:2311.18677](https://arxiv.org/abs/2311.18677)
  - 首次系统性提出 PD 分离架构，异构硬件优化

- **DistServe**: Zhong et al., "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving", OSDI 2024. [arXiv:2401.09670](https://arxiv.org/abs/2401.09670)
  - 拉取式调度，集群级分离，Goodput 优化

- **Mooncake**: Qin et al., "Mooncake: A KVCache-Centric Disaggregated Architecture for LLM Serving", FAST 2025 (Best Paper). [arXiv:2407.00079](https://arxiv.org/abs/2407.00079)
  - 以 KV 缓存为中心的生产系统，Transfer Engine

- **TetriInfer**: Xiao et al., "TetriInfer: Distributed LLM Inference via Tetris-like Scheduling", 2024. [arXiv:2401.11181](https://arxiv.org/abs/2401.11181)
  - 俄罗斯方块式调度优化

- **Sarathi-Serve**: Agrawal et al., "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve", OSDI 2024. [arXiv:2403.02310](https://arxiv.org/abs/2403.02310)
  - 分块预填充，PD 分离的替代/互补方案

---

## 相关页面

- [[continuous-batching]] — 连续批处理调度，分离架构的基础
- [[vllm]] — 支持分离部署的主流推理引擎
- [[sglang]] — 另一个支持分离的推理引擎
- [[paged-attention]] — KV 缓存内存管理
- [[kv-cache-optimization]] — KV 缓存优化技术（压缩、量化等）
- [[model-parallelism]] — 并行策略，与分离架构配合使用
- [[parallelism-strategies-deep-dive]] — DeepSeek-V3 的详细并行配置
