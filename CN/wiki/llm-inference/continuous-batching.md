---
title: "连续批处理：动态请求调度"
category: llm-inference
tags: [continuous-batching, 调度, iteration-level, 动态批处理, 吞吐量]
created: 2026-04-13
updated: 2026-05-13
status: mature
---

# 连续批处理：动态请求调度

> [!abstract]+ TL;DR
> 批处理通过摊薄权重加载成本提升 GPU 利用率，但 LLM 输出长度差异巨大（几 token 到数千 token），**静态批处理**会被最先完成的请求拖累。**连续批处理**（迭代级调度）在每个解码步骤动态调整批次组成 —— 旧请求完成时立刻插入新请求 —— 消除护航效应。由 **Orca（OSDI 2022）** 提出，现已成为 [[vllm|vLLM]]、[[sglang|SGLang]]、[[tensorrt-llm|TensorRT-LLM]] 的核心调度机制。生产部署相比静态批处理通常获得 **2–5 倍吞吐**。

```
核心思想：不再等待整个批次完成，而是在每个 token 生成步骤
         检查并替换已完成的请求，实现"即来即走"。
```

---

## 静态批处理的问题

### 基本原理

静态批处理（Static Batching）是最朴素的方法：收集一批请求，同时开始处理，等待**所有请求都完成**后才能接收下一批。

```
静态批处理示意图：

时间 ──────────────────────────────────────────────►

请求 A: |████████████████|                          (生成 16 tokens)
请求 B: |████████████████████████████████████████|   (生成 40 tokens)
请求 C: |████████|                                   (生成 8 tokens)
请求 D: |████████████████████████|                   (生成 24 tokens)
         ↑                       ↑                ↑
       批次开始              C,A 已完成        B 完成，批次结束
                          但仍在等待 B        才能接收新批次

         |◄──────────── 整个批次的延迟 ──────────────►|
```

### 问题分析

**护航效应（Convoy Effect）**：整个批次的延迟由最长的请求决定。短请求在完成后仍然占用 GPU 槽位，只是不做任何有意义的计算。

```
GPU 利用率分析（静态批处理）：

时间步:  1  2  3  4  5  6  7  8  9 10 11 12 ... 40
请求 A:  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ... □   ← 第16步完成
请求 B:  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ... ■   ← 第40步完成
请求 C:  ■  ■  ■  ■  ■  ■  ■  ■  □  □  □  □  ... □   ← 第8步完成
请求 D:  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ... □   ← 第24步完成

■ = 有效计算    □ = GPU 空闲（浪费）

有效计算量: 16 + 40 + 8 + 24 = 88 token-steps
总计算量:   4 × 40 = 160 token-steps
GPU 利用率: 88 / 160 = 55%
浪费:       45%
```

### 量化分析

假设批次中有 $N$ 个请求，各请求输出长度为 $L_1, L_2, \ldots, L_N$，最长输出长度为 $L_{\max} = \max(L_i)$。

- **静态批处理 GPU 利用率**：

$$
\text{Utilization}_{\text{static}} = \frac{\sum_{i=1}^{N} L_i}{N \times L_{\max}}
$$

当输出长度差异大时（例如对话场景中短回复 vs 长回复），利用率可能低至 **20-30%**。

- **排队延迟**：新请求必须等待当前批次完全结束才能开始处理，导致尾部延迟（tail latency）极高。

### 动态批处理（过渡方案）

有些系统采用"动态批处理"（Dynamic Batching）：在一个时间窗口内收集到达的请求，然后组成一个批次。但这只是优化了**批次形成**的过程，一旦批次开始执行，仍然面临相同的护航效应问题。

```
三种批处理策略对比：

┌─────────────┬───────────────────┬──────────────────┬───────────────────┐
│   策略       │ 批次形成           │ 批次执行          │ GPU 利用率         │
├─────────────┼───────────────────┼──────────────────┼───────────────────┤
│ 静态批处理   │ 预设固定大小       │ 等待全部完成       │ 低 (20-55%)       │
│ 动态批处理   │ 时间窗口内收集     │ 等待全部完成       │ 中 (40-65%)       │
│ 连续批处理   │ 每步检查           │ 即来即走          │ 高 (85-98%)       │
└─────────────┴───────────────────┴──────────────────┴───────────────────┘
```

---

## 连续批处理 (Continuous / Iteration-Level Batching)

### Orca 论文的贡献 (Yu et al., OSDI 2022)

Orca 是第一篇系统性提出**迭代级调度**的论文，发表于 OSDI 2022。其核心观察是：

> LLM 自回归解码的每一步（即生成每一个 token）都是一个独立的调度点。
> 不必等待整个批次完成，可以在**每个迭代步骤**决定哪些请求参与计算。

Orca 提出了两个关键机制：

1. **迭代级调度（Iteration-Level Scheduling）**：在每个 token 生成步骤重新评估批次组成
2. **选择性批处理（Selective Batching）**：只对兼容的操作进行批处理（例如，将 prefill 和 decode 分别批处理）

在 GPT-3 175B 上的实验中，Orca 相比 NVIDIA FasterTransformer 实现了 **36.9 倍吞吐量提升**。

### 工作原理

连续批处理的核心逻辑：

```
连续批处理工作流程：

每个解码迭代：
  ┌─────────────────────────────────────┐
  │ 1. 执行一步前向传播（所有活跃请求）   │
  │ 2. 检查哪些请求生成了 <EOS>          │
  │ 3. 移除已完成的请求，返回结果给客户端 │
  │ 4. 从等待队列中取新请求填入空槽位     │
  │ 5. 为新请求执行预填充               │
  │ 6. 回到步骤 1                       │
  └─────────────────────────────────────┘
```

### 时间线示意图

```
连续批处理时间线（最大并发 = 4 个槽位）：

时间步:   1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
槽位 0:  [A  A  A  A  A  A  A  A][E  E  E  E  E  E  E][H  H  H  H  H...
槽位 1:  [B  B  B  B  B  B  B  B  B  B  B  B][F  F  F  F  F  F  F  F...
槽位 2:  [C  C  C  C][D  D  D  D  D  D  D  D  D  D][G  G  G  G  G  G...
槽位 3:  [·  ·  ·  ·  ·  ·  ·  ·  ·  ·][·  ·  ·  ·  ·  ·  ·  ·  ·  ·

■ 字母 = 活跃请求    · = 空闲槽位
[ ] = 请求生命周期

关键事件：
  步骤 4:  C 完成 → 槽位 2 释放 → D 立即填入
  步骤 8:  A 完成 → 槽位 0 释放 → E 立即填入
  步骤 10: D 完成 → 但等待队列暂无请求 → 槽位 2 空闲
  步骤 12: B 完成 → 槽位 1 释放 → F 填入
  步骤 14: 新请求 G 到达 → 填入槽位 2
  步骤 15: E 完成 → 槽位 0 释放 → H 填入
```

### 利用率对比

```
同样的请求序列，对比两种策略：

静态批处理（批次大小=4）：
  批次1: [A(8), B(12), C(4), D(10)] → 等待12步 → 利用率 = 34/48 = 71%
  批次2: [E(7), F(8), G(6), H(5)]   → 等待8步  → 利用率 = 26/32 = 81%
  总延迟: 20 步，新请求排队等待

连续批处理（最大并发=4）：
  所有请求在完成后立即释放槽位
  利用率接近: (8+12+4+10+7+8+6+5) / (4 × 20) ≈ 75-95%
  关键优势: 延迟大幅降低，新请求无需等待整个批次完成
```

### 为什么能实现近 100% GPU 利用率

连续批处理之所以能大幅提升 GPU 利用率，原因在于：

1. **消除护航效应**：短请求完成后立即释放资源
2. **填充空闲槽位**：新请求可以随时进入
3. **流水线重叠**：新请求的 prefill 可以与其他请求的 decode 同时进行
4. **自适应负载**：批次大小根据实际负载动态调整

在高流量场景下（请求持续到达），GPU 几乎不会出现空闲槽位。

---

## 分块预填充 (Chunked Prefill)

### 问题：长预填充阻塞解码

连续批处理解决了解码阶段的护航效应，但引入了一个新问题：**预填充阻塞**。

当一个新请求进入系统时，需要先处理其完整的 prompt（预填充阶段）。如果 prompt 很长（例如 32K tokens），这个预填充操作会：

1. **独占 GPU 计算资源**：长 prompt 的注意力计算是 $O(n^2)$ 的
2. **阻塞解码请求**：正在解码的请求必须等待预填充完成才能继续生成 token
3. **TPOT 膨胀**：解码请求的 Time Per Output Token 可能增加 2-30 倍

```
预填充阻塞问题：

                    时间 ──────────────────────────────►
正在解码的请求:     ■ ■ ■ |████████████████████| ■ ■ ■ ■ ■
                          ↑                    ↑
                    新请求到达           预填充完成
                    (32K prompt)        解码才能继续

                    |◄── 这段时间内 ──►|
                    |  所有解码请求     |
                    |  的 TPOT 膨胀    |
```

### 解决方案：分块预填充

**分块预填充**（Chunked Prefill）的核心思想是：将长 prompt 的预填充拆分成多个小块（chunks），与解码请求**交替执行**。

```
分块预填充的基本原理：

原始预填充（32K tokens，一次完成）：
  [████████████████████████████████████████████████]
   ↑ 一个巨大的 prefill 操作，耗时很长

分块预填充（每块 512 tokens）：
  [████][████][████][████] ... [████][████]
    ↑      ↑      ↑      ↑
    每个块之间可以插入解码步骤
```

### 工作流程

```
分块预填充交错执行：

时间步:    1        2        3        4        5        6
        ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
预填充:  │Chunk1│ │      │ │Chunk2│ │      │ │Chunk3│ │      │
        │512tok│ │      │ │512tok│ │      │ │512tok│ │      │
        └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘
        ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
解码:    │Decode│ │Decode│ │Decode│ │Decode│ │Decode│ │Decode│
        │batch │ │batch │ │batch │ │batch │ │batch │ │batch │
        └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘

效果：解码请求在每个时间步都能获得计算资源
     预填充被分摊到多个步骤，不再阻塞
```

### Sarathi-Serve 方法

Sarathi-Serve（Agrawal et al., 2024）提出了一种更精细的分块预填充方案：

1. **统一调度**：将预填充块和解码 token 打包到同一个计算批次中
2. **token 预算**：每个迭代步骤有一个固定的 token 预算（例如 2048），预填充块和解码 token 共享这个预算
3. **流水线友好**：块大小可以调整以适应流水线并行

```
Sarathi-Serve 的 token 预算模型：

每步 token 预算 = 2048

步骤 1:  [Prefill Chunk: 512 tokens] + [Decode: 200 tokens] = 712  ✓
步骤 2:  [Prefill Chunk: 512 tokens] + [Decode: 201 tokens] = 713  ✓
步骤 3:  [Decode only: 202 tokens]                          = 202  ✓
步骤 4:  [New Prefill: 1024 tokens]  + [Decode: 203 tokens] = 1227 ✓

token 预算确保每步计算量可控，
避免个别大请求导致延迟波动。
```

### TTFT vs TBT 权衡

分块预填充引入了一个重要的权衡：

```
分块预填充的 TTFT vs TBT 权衡：

┌──────────────┬────────────────────┬────────────────────┐
│ 块大小        │ TTFT (首 token 延迟) │ TBT (token 间延迟)  │
├──────────────┼────────────────────┼────────────────────┤
│ 非常大        │ ✓ 低（快速完成预填充）│ ✗ 高（阻塞解码）    │
│ (如整个prompt)│                    │                    │
├──────────────┼────────────────────┼────────────────────┤
│ 中等          │ ○ 中等             │ ○ 中等              │
│ (如 512-2048) │                    │                    │
├──────────────┼────────────────────┼────────────────────┤
│ 非常小        │ ✗ 高（预填充分散）  │ ✓ 低（解码顺畅）    │
│ (如 64-128)   │                    │                    │
└──────────────┴────────────────────┴────────────────────┘

TTFT = Time To First Token（用户看到第一个输出的延迟）
TBT  = Time Between Tokens（token 之间的间隔，影响流式体验）
```

- **大块**：预填充更快完成（TTFT 低），但每个块执行时间长，阻塞解码（TBT 高）
- **小块**：解码更流畅（TBT 低），但预填充需要更多步骤才能完成（TTFT 高）
- **最佳平衡**：通常在 512-2048 tokens 之间，具体取决于模型大小和 GPU 算力

vLLM V1 默认采用分块预填充，块大小可通过 `max_num_batched_tokens` 参数配置。

### 为什么 prefill 会"挡住" decode

"prefill blocks decode"里"blocking"这个词承重很大。机制上原因就两条：

**(1) Prefill 在长上下文下是真的慢。** Forward 的 FLOPs $\approx 2 \cdot N_{\text{params}} \cdot N_{\text{tokens}}$。Llama-70B 跑 16K prompt：

$$
2 \times 70 \times 10^9 \times 16384 \approx 2.3 \times 10^{15} \text{ FLOPs} = 2.3 \text{ PFLOPs}
$$

一张 H100 在 ~989 TFLOPs/s FP16 下要 **~2.3 秒**的纯计算时间，外加 attention 的 $O(S^2)$ 项（score matrix 大小 $16{\text{K}} \times 16{\text{K}} \times \text{num\_heads} \times \text{head\_dim}$）、kernel launch、内存读写。模型小 / prompt 短按比例缩，但"大模型长 prompt 的 prefill 是秒级"这个数量级是常识。

**(2) 一次 forward pass 是不可分的调度单元。** 你塞进这次 forward 的东西 —— prefill token、decode token 或两者 —— 是一连串绑定的 CUDA kernel，没有中断点。调度器只能在 iteration *之间* 切换，不能 iteration 内部切。

合在一起：如果你朴素地把 16K prefill 塞进 iteration $k$，所有在飞的 decode 请求都要等 2.3 秒才轮到 iteration $k{+}1$。它们这一 iteration 的 TBT 从 ~30 ms 跳到 2300 ms —— 流式输出那一边就是肉眼可见的卡顿：

```
iter k:    forward([prefill 16K of req X])                ← 2.3 s
iter k+1:  forward([decode 1 token × 64 requests])        ← 30 ms each
```

Chunked prefill 的活就是**保证任何单个 iteration 都短到不让人卡**。每次 iteration 把一小段 prefill chunk 和一批 decode token 一起打包，既推进长 prefill 又给在飞 decode 出 token：

```
iter k:    forward([prefill 512 of X] + [decode 64 requests])   ← ~50 ms
iter k+1:  forward([prefill 512 of X] + [decode 64 requests])   ← ~50 ms
...
```

这也解释了为什么调度粒度重要：iteration 时长越小越均匀，尾延迟越受控。

### Chunk size 的数学

TTFT/TBT 折中有个清楚的闭式。设：

- $T$ = prefill 总 token 数（如 16384）
- $c$ = chunk size（每 iteration 的 token 数）
- $a$ = forward pass 的每 token 增量代价（每 token 的算力 + 内存带宽消耗）
- $b$ = 每 iteration 的固定开销（kernel launch、调度器、内存操作 —— 通常几百微秒）

每 iter 时长与 prefill 所需的 iter 数：

$$
t_{\text{iter}} = a \cdot c + b, \qquad N_{\text{iter}} = T / c
$$

两个指标：

$$
\text{TBT（别的 decode 感受到的）} = a \cdot c + b
$$

$$
\text{TTFT（本请求拿到首 token 的延迟）} = N_{\text{iter}} \cdot t_{\text{iter}} = \frac{T}{c}\,(a \cdot c + b) = a \cdot T + \frac{b \cdot T}{c}
$$

两个推论：

- **TBT 随 $c$ 线性增长。** Chunk 越大 → iteration 越长 → 别的 decode 等得越久。
- **TTFT 有两项。** $a \cdot T$ 是常数（prefill 工作量本来就那么多）。$b \cdot T / c$ 是**固定开销税**：每 iteration 都付一份 $b$，一共要 $T/c$ 次。Chunk 越小这一项越大。

所以小 chunk 对 TTFT 是 *更差*，不是更好 —— 反直觉直到你看见公式。最优 $c^*$ 由固定开销 $b$ 与每 token 代价 $a$ 的比值决定。生产推理引擎上的甜点通常是 **512–2048 token**：

- 重 $b$（kernel launch 多、Python 调度器慢）→ $c^*$ 偏大。
- 轻 $b$（CUDA graphs、融合调度）→ $c^*$ 可以偏小。
- 每 iteration 装更多 decode 请求 → TBT 对 $c$ 更敏感 → $c$ 往小推。

vLLM 里的 `max_num_batched_tokens` 就是这个 $c$（严格说是 prefill+decode 合并预算）。4096 是常见生产默认。

### FlashAttention 在 chunked prefill 里扮演什么角色

很自然的追问：[[paged-attention|FlashAttention]] 不是把 $O(S^2)$ 的 attention matrix 干掉了吗？那是不是就可以放心用大 chunk 了？

**短答**：FA *因为* 抬高内存上限而让大 chunk 成为可能，但**不改变 TTFT/TBT 折中本身**。

详细：

- **FA 没减少 attention 的 FLOPs。** Attention 计算量本来就是 $O(S^2 \cdot D)$，与实现无关。FA 改的是**内存峰值**：不再 materialize 完整 $S \times S$ score matrix，而是分块流式算，只保留 $O(S)$ 的活跃内存。**FLOPs 不变；内存 $O(S^2) \to O(S)$**。
- **没有 FA 时 chunk size 被显存上限卡住。** 8K chunk 的 attention matrix 是 $8192^2 \times \text{num\_heads} \times 2 \text{ B}$ ≈ 几十 GB —— 直接 OOM。Pre-FA 时代你被迫小 chunk 只是为了让 attention 跑得起来。
- **有了 FA，chunk size 只被你对 TBT 的偏好限制。** 显存不再是瓶颈，前一节的折中（每 iteration 时长 $a \cdot c + b$）才是真正约束。
- **FA-2 / FA-3 顺便提供了 chunked prefill 需要的 kernel**：即 "新 Q chunk 对前面已 cache 的 KV 算 cross-attention" —— varlen Q + paged KV —— 自 FA-2 起就是标配路径。没有这个 kernel，chunked prefill 实现起来非常别扭。

正确口径：

> **FA 让 chunked prefill kernel 层面可行、并让你能根据正确的理由选 chunk size** —— 不是给你随便放大 chunk 的免费午餐。

实践中 512–2048 还是甜点，但 FA 是你**有这个区间**而不是被显存限制到 256 的根本原因。

### Chunked Prefill 不是什么

名字里有"切"和"块"，最容易被误以为是某种并行技术。三个值得钉死的混淆，按误导程度递增：

**它不是把一条序列切到多张 GPU 上。** Chunked prefill 把整条请求保留在同一张 GPU（或同一个 TP 组）里。切的是这条请求的 *工作量*，分到这张 GPU 的多个调度 iteration 上。那条 32K-token 的 prefill 还是住在单卡显存里，只是不在一次 forward pass 里算完。

**它不是并行技术。** [[parallelism-strategies-deep-dive|并行]]（TP、PP、DP、CP、EP）决定"哪张 GPU 算哪部分" —— **空间**切分。Chunked prefill 决定"哪个 iteration 算哪些 token" —— **时间**切分。两者正交可叠加：1M-token 请求可以同时 CP=8 跨 GPU 切 *并* 在每张 GPU 自己的片段上做 chunked prefill。

**它不是解决"prefill 挡 decode"的唯一办法。** 正交方案是 [[prefill-decode-disaggregation|PD 分离]] —— 把 prefill worker 和 decode worker 放到 *不同的物理节点* 上，根本不共享 forward pass。Chunked prefill 说 *混着也能跑得好*；PD 分离说 *干脆别混*。两者取舍：

| 维度 | Chunked prefill | PD 分离 |
|------|-----------------|---------|
| Prefill 和 decode 在哪 | 同 GPU 不同 iteration | 不同节点 |
| 主要代价 | TTFT 变高（prefill 拆成更多 pass） | KV cache 跨节点传输 |
| 适合场景 | 中小规模部署、流量混杂 | 大规模部署、流量画像清晰 |
| 显存压力 | 单池子，互相挤 | 双池子，按角色独立 |
| 吞吐扩展性 | 一边卡住影响另一边 | 两边独立扩缩 |

生产系统经常两者叠加 —— prefill 专用节点组 *内部* 用 chunked prefill 平滑负载，prefill / decode 节点组 *之间* 用 PD 分离消除跨角色干扰。

---

## 调度策略

### FCFS（先来先服务）

最基本的调度策略，按请求到达顺序处理：

```python
class FCFSScheduler:
    """先来先服务调度器"""
    
    def __init__(self, max_batch_size: int, max_num_tokens: int):
        self.max_batch_size = max_batch_size
        self.max_num_tokens = max_num_tokens
        self.waiting_queue: list[Request] = []     # 等待预填充的请求
        self.running_batch: list[Request] = []     # 正在解码的请求
    
    def schedule(self) -> ScheduleOutput:
        """每个迭代步骤调用一次"""
        # 1. 移除已完成的请求
        self.running_batch = [
            req for req in self.running_batch 
            if not req.is_finished()
        ]
        
        # 2. 计算当前 token 预算
        num_decode_tokens = len(self.running_batch)  # 每个解码请求1个token
        remaining_budget = self.max_num_tokens - num_decode_tokens
        remaining_slots = self.max_batch_size - len(self.running_batch)
        
        # 3. 按 FCFS 顺序填入新请求
        new_prefills = []
        while self.waiting_queue and remaining_slots > 0:
            request = self.waiting_queue[0]
            prefill_tokens = request.get_prompt_length()
            
            if prefill_tokens <= remaining_budget:
                self.waiting_queue.pop(0)
                new_prefills.append(request)
                self.running_batch.append(request)
                remaining_budget -= prefill_tokens
                remaining_slots -= 1
            else:
                break  # 预算不足，等待下一步
        
        return ScheduleOutput(
            decode_requests=self.running_batch,
            prefill_requests=new_prefills
        )
```

### 抢占和优先级调度

当 GPU 内存不足时，需要**抢占**（preemption）一些正在运行的请求：

```
抢占策略对比：

┌──────────────┬──────────────────────────┬──────────────────────────┐
│ 策略          │ Swap（交换到 CPU 内存）   │ Recompute（丢弃重算）     │
├──────────────┼──────────────────────────┼──────────────────────────┤
│ 操作          │ 将 KV 缓存复制到 CPU    │ 丢弃 KV 缓存             │
│ 恢复          │ 从 CPU 复制回 GPU       │ 重新执行预填充            │
│ 内存          │ 需要 CPU 内存           │ 不需要额外内存            │
│ 适用          │ 长序列（重算代价大）     │ 短序列（重算代价小）       │
│ 延迟          │ 受 PCIe 带宽限制        │ 受 GPU 计算速度限制       │
└──────────────┴──────────────────────────┴──────────────────────────┘
```

优先级调度允许高优先级请求抢占低优先级请求的资源：

```python
class PriorityScheduler:
    """基于优先级的调度器"""
    
    def __init__(self, max_batch_size: int, max_num_tokens: int):
        self.max_batch_size = max_batch_size
        self.max_num_tokens = max_num_tokens
        self.waiting_queue: list[Request] = []
        self.running_batch: list[Request] = []
    
    def add_request(self, request: Request):
        """按优先级插入等待队列（优先级高的在前）"""
        import bisect
        bisect.insort(self.waiting_queue, request, 
                      key=lambda r: -r.priority)
    
    def schedule(self) -> ScheduleOutput:
        # 1. 移除已完成的请求
        self.running_batch = [
            req for req in self.running_batch 
            if not req.is_finished()
        ]
        
        # 2. 检查是否需要抢占（高优先级请求等待中但资源不足）
        preempted = []
        while (self.waiting_queue 
               and len(self.running_batch) >= self.max_batch_size):
            # 最高优先级等待请求 vs 最低优先级运行请求
            waiting_top = self.waiting_queue[0]
            running_lowest = min(self.running_batch, 
                                key=lambda r: r.priority)
            
            if waiting_top.priority > running_lowest.priority:
                # 抢占低优先级请求
                self.running_batch.remove(running_lowest)
                preempted.append(running_lowest)
            else:
                break
        
        # 3. 填入新请求（与 FCFS 类似）
        # ... (省略，逻辑同上)
        
        return ScheduleOutput(
            decode_requests=self.running_batch,
            prefill_requests=new_prefills,
            preempted_requests=preempted
        )
```

### vLLM 的调度实现

vLLM V1 采用**统一调度器**，其核心设计：

```
vLLM V1 统一调度器：

输入: 当前运行请求 + 等待队列 + 内存状态

输出: {request_id: num_tokens} 字典
      ↓
  这个简单的映射统一了以下所有场景：
  - 普通解码:     {req_1: 1, req_2: 1, req_3: 1}
  - 分块预填充:   {req_1: 1, req_2: 1, new_req: 512}
  - 投机解码:     {req_1: 5, req_2: 5}  (每步验证多个 token)
  - 前缀缓存命中: {new_req: 100}  (只需处理未缓存的部分)
```

vLLM 调度器的关键配置参数：

```python
# vLLM 服务配置示例
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    
    # === 调度相关参数 ===
    max_num_seqs=256,              # 最大并发序列数（批次大小上限）
    max_num_batched_tokens=2048,   # 每步最大 token 预算
                                   # （控制分块预填充的块大小）
    
    # === 内存相关参数 ===
    gpu_memory_utilization=0.90,   # GPU 内存使用率上限
    swap_space=4,                  # CPU swap 空间 (GB)
    
    # === 抢占策略 ===
    preemption_mode="recompute",   # "recompute" 或 "swap"
    
    # === 前缀缓存 ===
    enable_prefix_caching=True,    # V1 默认开启
)
```

### SGLang 的调度实现

[[sglang|SGLang]] 采用了不同的调度优化：

1. **RadixAttention**：基于基数树（Radix Tree）的前缀缓存，支持 token 级粒度的缓存复用
2. **持续批处理**：类似 vLLM，但在多轮对话等前缀密集场景下性能更优
3. **零开销调度**：调度决策在 Python 端完成，不阻塞 GPU 计算

```
SGLang vs vLLM 调度对比：

┌─────────────┬──────────────────┬──────────────────┐
│ 特性         │ vLLM V1          │ SGLang           │
├─────────────┼──────────────────┼──────────────────┤
│ 前缀缓存     │ 哈希 LRU         │ RadixAttention   │
│ 缓存粒度     │ 块级 (16 token)  │ Token 级         │
│ 调度表示     │ {id: num_tokens} │ Tree-based       │
│ 分块预填充   │ ✓                │ ✓                │
│ 多轮对话优化 │ 好               │ 更好 (+29%)      │
└─────────────┴──────────────────┴──────────────────┘
```

### SLA 管理与请求优先级

在生产环境中，不同类型的请求有不同的 SLA（Service Level Agreement）要求：

```
SLA 驱动的调度示例：

┌──────────────┬───────────────┬──────────────┬───────────────┐
│ 请求类型      │ TTFT SLA      │ TBT SLA      │ 优先级         │
├──────────────┼───────────────┼──────────────┼───────────────┤
│ 实时对话      │ < 200ms       │ < 50ms       │ 高             │
│ 流式生成      │ < 500ms       │ < 100ms      │ 中             │
│ 批量处理      │ < 5s          │ 无要求        │ 低             │
│ 后台任务      │ 无要求         │ 无要求        │ 最低           │
└──────────────┴───────────────┴──────────────┴───────────────┘

调度器根据 SLA 动态调整优先级：
- 接近 SLA 截止时间的请求优先级自动提升
- 已超出 SLA 的请求可能被降级（减少资源浪费）
```

---

## 内存管理与调度的交互

### PagedAttention 如何支撑连续批处理

连续批处理的实现强依赖于灵活的内存管理。[[paged-attention|PagedAttention]] 通过以下方式使连续批处理成为可能：

```
PagedAttention 与连续批处理的协作：

物理内存（GPU HBM）：
┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
│ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │ 9 │10 │11 │  物理块
└───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘

时间步 T（请求 A, B, C 在运行）：
  请求 A: 逻辑块 [0,1,2] → 物理块 [0,3,7]
  请求 B: 逻辑块 [0,1]   → 物理块 [1,5]
  请求 C: 逻辑块 [0,1,2,3] → 物理块 [2,4,8,9]
  空闲块: [6, 10, 11]

时间步 T+1（请求 C 完成，请求 D 加入）：
  请求 A: 逻辑块 [0,1,2,3] → 物理块 [0,3,7,2]  ← 复用了 C 的块 2
  请求 B: 逻辑块 [0,1,2]   → 物理块 [1,5,4]     ← 复用了 C 的块 4
  请求 D: 逻辑块 [0,1]     → 物理块 [8,9]        ← 复用了 C 的块 8,9
  空闲块: [6, 10, 11]

关键：不需要连续内存，不需要预分配，
     新请求可以即时获得内存块。
```

### 块级内存分配

调度器在每个迭代步骤中与内存管理器交互：

```python
class SchedulerWithMemory:
    """调度器与内存管理器的交互"""
    
    def __init__(self, block_manager, max_batch_size, max_num_tokens):
        self.block_manager = block_manager
        self.max_batch_size = max_batch_size
        self.max_num_tokens = max_num_tokens
        self.waiting_queue = []
        self.running_batch = []
    
    def schedule(self) -> ScheduleOutput:
        # 1. 释放已完成请求的内存块
        finished = [r for r in self.running_batch if r.is_finished()]
        for req in finished:
            self.block_manager.free(req.request_id)
        self.running_batch = [
            r for r in self.running_batch if not r.is_finished()
        ]
        
        # 2. 为正在运行的请求分配新块（如果当前块已满）
        for req in self.running_batch:
            if req.needs_new_block():
                if self.block_manager.has_free_blocks():
                    self.block_manager.allocate(req.request_id, num_blocks=1)
                else:
                    # 内存不足，需要抢占
                    self._preempt_lowest_priority()
        
        # 3. 尝试调度新请求
        new_prefills = []
        while self.waiting_queue:
            request = self.waiting_queue[0]
            # 计算新请求需要的块数
            needed_blocks = self._compute_needed_blocks(request)
            
            if (self.block_manager.get_free_blocks() >= needed_blocks
                    and len(self.running_batch) < self.max_batch_size):
                self.waiting_queue.pop(0)
                self.block_manager.allocate(
                    request.request_id, num_blocks=needed_blocks)
                self.running_batch.append(request)
                new_prefills.append(request)
            else:
                break
        
        return ScheduleOutput(
            decode_requests=self.running_batch,
            prefill_requests=new_prefills,
        )
    
    def _preempt_lowest_priority(self):
        """抢占最低优先级的请求释放内存"""
        victim = min(self.running_batch, key=lambda r: r.priority)
        if self.preemption_mode == "swap":
            # 将 KV 缓存交换到 CPU 内存
            self.block_manager.swap_out(victim.request_id)
        else:
            # 丢弃 KV 缓存，稍后重新计算
            self.block_manager.free(victim.request_id)
            victim.mark_for_recompute()
        self.running_batch.remove(victim)
        self.waiting_queue.insert(0, victim)  # 放回队首
```

### 抢占策略：Swap vs Recompute

```
Swap vs Recompute 决策树：

                    需要抢占？
                       │
                   ┌───┴───┐
                   ▼       ▼
              序列长度 > 阈值?
              │              │
           是 │              │ 否
              ▼              ▼
           Swap           Recompute
    (保存到 CPU 内存)    (丢弃，稍后重算)

  考虑因素：
  ┌──────────────────┬──────────────────────┐
  │ 选择 Swap         │ 选择 Recompute       │
  ├──────────────────┼──────────────────────┤
  │ 序列已经很长      │ 序列较短              │
  │ PCIe 带宽充足    │ PCIe 带宽是瓶颈       │
  │ CPU 内存充足      │ CPU 内存有限          │
  │ 重算代价大        │ 重算代价小            │
  │ 长上下文窗口场景  │ 短对话场景            │
  └──────────────────┴──────────────────────┘
```

vLLM 的默认策略：当 GPU 内存不足时，优先尝试 swap。如果 CPU 内存也不足，则回退到 recompute。可以通过 `preemption_mode` 参数指定。

---

## 代码示例

### 完整的连续批处理调度器伪代码

```python
"""
完整的连续批处理调度器实现（简化版）。
展示核心调度逻辑，省略了实际的模型前向传播。
"""

from dataclasses import dataclass, field
from enum import Enum
from collections import deque
from typing import Optional
import time


class RequestState(Enum):
    WAITING = "waiting"          # 等待预填充
    RUNNING_PREFILL = "prefill"  # 正在预填充
    RUNNING_DECODE = "decode"    # 正在解码
    FINISHED = "finished"        # 已完成


@dataclass
class Request:
    request_id: str
    prompt_tokens: list[int]
    max_output_tokens: int
    arrival_time: float
    priority: int = 0
    
    # 运行时状态
    state: RequestState = RequestState.WAITING
    output_tokens: list[int] = field(default_factory=list)
    prefill_progress: int = 0    # 已处理的 prompt token 数
    
    def is_prefill_complete(self) -> bool:
        return self.prefill_progress >= len(self.prompt_tokens)
    
    def is_finished(self) -> bool:
        """检查是否生成了 EOS 或达到最大长度"""
        if not self.output_tokens:
            return False
        EOS_TOKEN = 2
        return (self.output_tokens[-1] == EOS_TOKEN 
                or len(self.output_tokens) >= self.max_output_tokens)
    
    def get_remaining_prefill(self) -> int:
        return len(self.prompt_tokens) - self.prefill_progress


@dataclass
class ScheduleOutput:
    """调度器每步的输出"""
    scheduled_requests: dict[str, int]  # {request_id: num_tokens}
    preempted: list[str]                # 被抢占的 request_id
    finished: list[str]                 # 已完成的 request_id


class ContinuousBatchingScheduler:
    """连续批处理调度器（支持分块预填充）"""
    
    def __init__(
        self,
        max_batch_size: int = 256,
        max_num_tokens: int = 2048,
        chunk_size: int = 512,
    ):
        self.max_batch_size = max_batch_size
        self.max_num_tokens = max_num_tokens
        self.chunk_size = chunk_size
        
        self.waiting: deque[Request] = deque()
        self.running: dict[str, Request] = {}
    
    def add_request(self, request: Request):
        """添加新请求到等待队列"""
        request.state = RequestState.WAITING
        self.waiting.append(request)
    
    def schedule(self) -> ScheduleOutput:
        """核心调度逻辑 —— 每个迭代步骤调用一次"""
        scheduled: dict[str, int] = {}
        finished_ids: list[str] = []
        
        # ---- 阶段 1: 移除已完成的请求 ----
        for req_id in list(self.running.keys()):
            req = self.running[req_id]
            if req.is_finished():
                req.state = RequestState.FINISHED
                finished_ids.append(req_id)
                del self.running[req_id]
        
        # ---- 阶段 2: 为正在运行的解码请求分配 token ----
        token_budget = self.max_num_tokens
        
        for req_id, req in self.running.items():
            if req.state == RequestState.RUNNING_DECODE:
                scheduled[req_id] = 1   # 解码：每步 1 个 token
                token_budget -= 1
            elif req.state == RequestState.RUNNING_PREFILL:
                # 继续分块预填充
                remaining = req.get_remaining_prefill()
                chunk = min(remaining, self.chunk_size, token_budget)
                if chunk > 0:
                    scheduled[req_id] = chunk
                    token_budget -= chunk
                    req.prefill_progress += chunk
                    if req.is_prefill_complete():
                        req.state = RequestState.RUNNING_DECODE
        
        # ---- 阶段 3: 从等待队列调度新请求 ----
        while (self.waiting 
               and len(self.running) < self.max_batch_size 
               and token_budget > 0):
            
            req = self.waiting[0]
            prompt_len = len(req.prompt_tokens)
            
            # 计算第一个 chunk 的大小
            first_chunk = min(prompt_len, self.chunk_size, token_budget)
            
            if first_chunk <= 0:
                break
            
            # 从等待队列取出
            self.waiting.popleft()
            req.state = RequestState.RUNNING_PREFILL
            req.prefill_progress = first_chunk
            
            self.running[req.request_id] = req
            scheduled[req.request_id] = first_chunk
            token_budget -= first_chunk
            
            # 如果 prompt 很短，一步就完成预填充
            if req.is_prefill_complete():
                req.state = RequestState.RUNNING_DECODE
        
        return ScheduleOutput(
            scheduled_requests=scheduled,
            preempted=[],
            finished=finished_ids,
        )


# ---- 使用示例 ----
def main():
    scheduler = ContinuousBatchingScheduler(
        max_batch_size=4,
        max_num_tokens=2048,
        chunk_size=512,
    )
    
    # 模拟添加请求
    requests = [
        Request("req_0", prompt_tokens=list(range(100)),
                max_output_tokens=50, arrival_time=time.time()),
        Request("req_1", prompt_tokens=list(range(2000)),
                max_output_tokens=100, arrival_time=time.time()),
        Request("req_2", prompt_tokens=list(range(50)),
                max_output_tokens=20, arrival_time=time.time()),
    ]
    
    for req in requests:
        scheduler.add_request(req)
    
    # 模拟调度循环
    for step in range(10):
        output = scheduler.schedule()
        print(f"Step {step}: {output.scheduled_requests}")
        
        # 模拟 token 生成（实际中这里是模型前向传播）
        for req_id in output.scheduled_requests:
            if req_id in scheduler.running:
                req = scheduler.running[req_id]
                if req.state == RequestState.RUNNING_DECODE:
                    req.output_tokens.append(42)  # 模拟生成 token


if __name__ == "__main__":
    main()
```

### vLLM 服务配置示例

```python
"""vLLM 服务器配置 —— 调度相关参数调优"""

# 方式 1: 通过 Python API
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,
    
    # 调度参数
    max_num_seqs=256,                # 最大并发请求数
    max_num_batched_tokens=4096,     # 每步 token 预算
    
    # 内存参数
    gpu_memory_utilization=0.90,     # 90% GPU 内存用于 KV 缓存
    swap_space=8,                    # 8GB CPU swap 空间
    
    # 前缀缓存
    enable_prefix_caching=True,      # 自动缓存公共前缀
)
```

```bash
# 方式 2: 通过命令行启动 vLLM 服务器
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --max-num-seqs 256 \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.90 \
    --swap-space 8 \
    --enable-prefix-caching \
    --preemption-mode recompute
```

### 批处理参数调优指南

```
参数调优决策树：

                    你的场景是什么？
                         │
            ┌────────────┼────────────┐
            ▼            ▼            ▼
        实时对话      流式生成       离线批量
            │            │            │
            ▼            ▼            ▼
    max_num_seqs:   max_num_seqs:   max_num_seqs:
    32-64           128-256         512-1024
            │            │            │
            ▼            ▼            ▼
    max_num_batched  max_num_batched  max_num_batched
    _tokens: 1024   _tokens: 2048   _tokens: 8192
            │            │            │
            ▼            ▼            ▼
    优化目标:        优化目标:        优化目标:
    低 TTFT + TBT   平衡 TTFT/TBT   最大吞吐量
```

关键调优原则：

| 参数 | 增大效果 | 减小效果 |
|------|---------|---------|
| `max_num_seqs` | 吞吐量提升，延迟可能增加 | 延迟降低，吞吐量下降 |
| `max_num_batched_tokens` | 预填充更快（TTFT 降低），但单步时间增加 | TBT 更稳定，但 TTFT 可能增加 |
| `gpu_memory_utilization` | 更多 KV 缓存空间 → 更大批次 | 更安全，减少 OOM 风险 |
| `swap_space` | 减少抢占丢弃 | 减少 CPU 内存占用 |

---

## 性能分析

### 吞吐量提升

连续批处理相比静态批处理的吞吐量提升取决于多个因素：

```
吞吐量提升因素分析：

┌─────────────────┬─────────────┬──────────────────────────────┐
│ 因素             │ 影响程度     │ 说明                          │
├─────────────────┼─────────────┼──────────────────────────────┤
│ 输出长度方差     │ 高          │ 方差越大，静态批处理浪费越多   │
│ 请求到达速率     │ 高          │ 高流量下空槽位更快被填满       │
│ 批次大小         │ 中          │ 大批次时护航效应更显著         │
│ 模型大小         │ 中          │ 大模型每步时间长，等待更浪费   │
│ prompt 长度方差  │ 中          │ 影响分块预填充的效果           │
└─────────────────┴─────────────┴──────────────────────────────┘
```

典型提升数据：

| 场景 | 静态批处理 | 连续批处理 | 提升 |
|------|-----------|-----------|------|
| 对话（短输出） | ~1000 tok/s | ~3000 tok/s | 3x |
| 代码生成（中输出） | ~800 tok/s | ~2500 tok/s | 3.1x |
| 摘要（长输出，低方差） | ~900 tok/s | ~1500 tok/s | 1.7x |
| 混合负载（高方差） | ~600 tok/s | ~2800 tok/s | 4.7x |

> 注：以上数据基于 Llama 2 13B，单 A100 80GB，仅供参考。实际提升与负载分布高度相关。

### 延迟分析

```
延迟对比（P50 / P99）：

静态批处理：
  TTFT:  200ms / 2000ms    ← P99 很高（排队等待）
  TBT:   30ms  / 300ms     ← P99 被大 prefill 阻塞

连续批处理（无分块预填充）：
  TTFT:  100ms / 500ms     ← 显著改善
  TBT:   30ms  / 200ms     ← 仍有 prefill 阻塞问题

连续批处理 + 分块预填充：
  TTFT:  150ms / 600ms     ← 略有增加（预填充被分块）
  TBT:   25ms  / 50ms      ← 大幅改善（不再被阻塞）
```

### 调度开销

连续批处理的调度在每个 token 生成步骤都需要执行，因此调度开销不可忽视：

- **Python 调度开销**：典型 0.1-1ms/步（vLLM V1 通过 EngineCore 分离优化）
- **批次动态变化**：需要重新组织输入张量，增加约 0.05-0.5ms
- **内存管理**：块分配/释放约 0.01-0.1ms

总调度开销通常占单步时间的 **1-5%**，在大模型（单步时间 > 30ms）上几乎可以忽略。

---

## 不足与权衡

### TTFT vs TBT 的根本权衡

连续批处理（特别是分块预填充）迫使我们在两个核心指标之间权衡：

1. **TTFT（Time To First Token）**：用户等待第一个输出 token 的时间
   - 受预填充速度影响
   - 分块预填充会增加 TTFT（预填充被分散到多步）
   
2. **TBT（Time Between Tokens）**：相邻 token 之间的间隔
   - 受解码干扰影响
   - 分块预填充会降低 TBT（解码不再被长预填充阻塞）

> 最终解决方案是 [[prefill-decode-disaggregation|预填充-解码分离]]：将两个阶段物理分离到不同 GPU 池，彻底消除干扰。

### 调度复杂性

连续批处理的调度器需要在每个 token 步骤做出决策，随着系统功能增加，调度复杂性也在增长：

```
调度器需要考虑的因素：

1. 内存约束:   当前可用 KV 缓存块数
2. 计算约束:   token 预算限制
3. 并发约束:   最大批次大小
4. 优先级:     SLA 要求和请求优先级
5. 前缀缓存:   缓存命中的请求应优先（减少计算）
6. 投机解码:   验证步骤的 token 数不固定
7. 抢占决策:   何时抢占、抢占谁、swap 还是 recompute
8. 公平性:     避免低优先级请求饿死

随着功能增加，这些因素的组合爆炸使调度器成为
推理引擎中最复杂的组件之一。
```

### 小批量场景的局限

当请求到达速率很低时（例如单用户场景），连续批处理的优势不明显：
- 批次中可能只有 1-2 个请求
- 动态调度的开销相对于单步计算时间不可忽略
- 此时更重要的是单请求优化（如 [[speculative-decoding|投机解码]]）

---

## 参考文献

- **Orca**: Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models", OSDI 2022. [Paper](https://www.usenix.org/conference/osdi22/presentation/yu)
  - 首次提出迭代级调度和选择性批处理
  
- **Sarathi-Serve**: Agrawal et al., "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve", OSDI 2024. [arXiv:2403.02310](https://arxiv.org/abs/2403.02310)
  - 分块预填充和混合批处理

- **vLLM**: Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention", SOSP 2023. [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
  - PagedAttention 使连续批处理的内存管理成为可能

- **FastServe**: Wu et al., "Fast Distributed Inference Serving for Large Language Models", 2023. [arXiv:2305.05920](https://arxiv.org/abs/2305.05920)
  - 抢占式调度和作业完成时间优化

---

## 相关页面

- [[vllm]] — 实现连续批处理的主流推理引擎
- [[sglang]] — 另一个高性能推理引擎，RadixAttention 优化前缀缓存
- [[paged-attention]] — 支撑连续批处理的内存管理机制
- [[prefill-decode-disaggregation]] — 通过物理分离彻底解决 prefill/decode 干扰
- [[kv-cache-optimization]] — KV 缓存优化技术
- [[speculative-decoding]] — 与连续批处理互补的单请求优化技术
