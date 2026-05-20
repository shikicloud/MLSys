---
title: "DeepSpeed Ulysses：用 head-sharding AllToAll 做序列并行"
category: llm-inference
tags: [deepspeed-ulysses, context-parallelism, sequence-parallelism, long-context, attention, alltoall, microsoft, paper-review]
created: 2026-05-19
updated: 2026-05-19
status: mature
paper: arXiv:2309.14509
code: https://github.com/deepspeedai/DeepSpeed
---

# DeepSpeed Ulysses：用 head-sharding AllToAll 做序列并行

> [!info] 论文元信息
> - **论文**：[arXiv:2309.14509](https://arxiv.org/abs/2309.14509) —— *DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models*（Sam Ade Jacobs, Masahiro Tanaka, Chengming Zhang, Minjia Zhang, Shuaiwen Leon Song, Samyam Rajbhandari, Yuxiong He；Microsoft）
> - **发布**：2023-09-25 (v1)、2023-10-04 (v2 current)
> - **源码**：[deepspeedai/DeepSpeed](https://github.com/deepspeedai/DeepSpeed) —— 主文件 `deepspeed/sequence/layer.py`（`DistributedAttention` 类）
> - **博客**：[DeepSpeed Ulysses README](https://github.com/microsoft/DeepSpeed/blob/master/blogs/deepspeed-ulysses/README.md)
> - **教程**：[deepspeed.ai/tutorials/ds-sequence](https://www.deepspeed.ai/tutorials/ds-sequence/)
> - **后续**：[Ulysses-Offload (FPDT)](https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/ulysses-offload/README.md) —— 2024-12，32× A100 上 4M token
> - **配套页面**：[[ring-attention]] —— P2P 环路线的替代方案

> [!abstract]+ TL;DR
> DeepSpeed Ulysses 通过每 attention 层做两次 **AllToAll 转置** 来解决长上下文：attention 前数据布局从序列切分的 `[N/P, d]` 翻转成 head 切分的 `[N, d/P]` —— 每张 GPU 临时持有 *完整序列* 但只持 $1/P$ 的 head，**本地跑标准 FlashAttention**，然后 AllToAll 再翻回序列切分。结果：每层每 link 通信量是 $4Nh/P$ —— **$N$ 和 $P$ 同比扩时常数**，是 SP 家族里最干净的扩展性。但有硬上限：$P \leq \text{num\_heads}$（GQA 模型下 $P \leq \text{num\_kv\_heads}$，可能小到 8）。Microsoft 报告：**256× A100、GPT 1.2B–30B、~175 TFLOPs/GPU 持续（54% 硬件峰值）、1M token 序列**。生产用户：**HuggingFace TRL/Accelerate（2025 起）**、**Tencent xDiT**（diffusion video，通过 [USP](https://arxiv.org/abs/2405.07719)）、**verl** RL 框架、**Microsoft 自家 Megatron-DeepSpeed**。**对比** [[ring-attention|Ring Attention]]：Ulysses 受 head 数限制但每卡通信恒定、causal mask 天然均衡；Ring 扩展无限但需要高带宽 fabric。生产答案是 **混合（USP）** —— 节点内 Ulysses × 节点间 Ring。

---

## 背景：为什么序列并行是独立问题

LLM 训练有四个正交并行轴（[[parallelism-strategies-deep-dive|TP / PP / DP / EP]]），每个切模型的不同维度。**没一个切 activation 的序列维度**。过 128K token 后这就是瓶颈：

| 轴 | 切什么 | 长 $N$ 下的 activation 内存 |
| -- | ------ | --------------------------- |
| DP | Batch | 每 rank：完整 $N$ × 完整 hidden |
| TP | 权重 / head | KV cache 在 TP rank 间 **复制** |
| PP | 层 | 每个 pipeline stage 持有完整 $N$ |
| EP | 专家 | 只在 MoE；不切序列 |

需要的是 *序列并行* 轴：把 $N$ 个 token 切到 $P$ 个 rank。问题：attention 要求每个 query 看到每个 key，naive 切法把数学破坏掉。

2023 年出现两个对手答案：

1. **[[ring-attention|Ring Attention]]**（Liu/Zaharia/Abbeel, 2023-10）—— Q 留在每张卡，K/V 沿环旋转，FlashAttention streaming softmax 跨轮次累计。
2. **DeepSpeed Ulysses**（Microsoft, 2023-09）—— *重塑数据布局*，让 attention 能在每张卡的本地切片上跑。两次 AllToAll 在序列切分和 head 切分之间切换。

两者都产出数学上完全相同的 attention 输出。差异在 **哪个维度何时被切**、**哪个通信原语动数据**、**扩展上限**。

论文对环路线的论点：

> "Existing systems [Megatron-SP, ColAI-SP] incur communication volume O(M), making their effective throughput poor as the sequence length and parallelism degree increase. **DeepSpeed Ulysses incurs O(M/P) communication volume**, allowing it to scale to longer sequences and larger parallelism degrees without communication bottleneck."

Table 1 原文：

| 方法 | 通信复杂度 | Activation 内存 | 参数内存 | Attention 机制无关 |
| ---- | ---------- | -------------- | -------- | ------------------ |
| ColAI-SP (Ring) | $O(M)$ | ✓ | ✗ | ✗ |
| Megatron-SP | $O(M)$ | ✓ | ✗ | ✗ |
| **DS-Ulysses** | $\mathbf{O(M/P)}$ | ✓ | ✓ | ✓ |

---

## 核心思想：AllToAll 在序列切分和 head 切分之间翻转

> [!quote] 一句话总结贡献
> Attention 前用 AllToAll 把激活从 **序列切分** `[N/P, d]` 转成 **head 切分** `[N, d/P]`（每张 GPU 持完整序列但只持 head 子集，本地跑标准 FlashAttention），attention 后再 AllToAll 翻回去。

三个支撑次级声明：

- **通信量随 $N$ 亚线性**。每次 AllToAll 每 link 搬 $Nh/P$。每层 4 次（QKV 进、output 出）。总：$4Nh/P$ —— $N$ 和 $P$ 同比扩时 **恒定**。Ring 每旋转量也恒定，但 Ring 要 $P{-}1$ 轮；Ulysses 永远 2（或算 QKV 4）次。
- **Attention kernel 不动**。第一次 AllToAll 后每张 GPU 持 *完整序列* 上 $1/P$ 的 head —— 标准 FlashAttention 直接跑。无自定义 streaming softmax、无每步 ring 编排。用户已有的任何 attention 实现（FlashAttention 2/3、SDPA、Triton sparse）原样工作。
- **Causal mask 天然均衡**。每张 GPU 在自己的 head 子集上处理完整序列，每 rank 的 causal 三角形相同 —— 不需要 zigzag 调度、不需要 Striped Attention 后续。

去掉任何一个：失去 AllToAll 的 $1/P$ 扩展性就退化回 ring 风格 $P{-}1$ 通信轮；失去 head-shard 的本地 attention 就需要定制 kernel；失去 causal 均衡就需要额外调度。

---

## 实现细节

### 算法

论文 Figure 2 描述 forward：

```
                       序列切分                       head 切分
                       [N/P, d]                       [N, d/P]
                       每 GPU                         每 GPU

   输入 X     ─────►   X_local                                       ─┐
                       │                                              │
                       ▼                                              │
                       Q_local, K_local, V_local                      │
                       （每个 [N/P, d]）                              │
                       │                                              │
   AllToAll #1  ───────┘  scatter heads, gather sequence              │
                       ▼                                              │
                       Q, K, V （每个 [N, hc/P, hs]）  ──► attention │
                       │                                  （本地）    │
                       │                                              │
                       Context  [N, hc/P, hs] = [N, d/P]              │
                       │                                              │
   AllToAll #2  ───────┘  scatter sequence, gather heads              │
                       ▼                                              │
                       Context_local [N/P, d]                         │
                       │                                              │
   Output proj W_O ───►Output  [N/P, d]                              ─┘
```

论文符号：$N$ = 序列长，$b$ = micro-batch，$d$ = hidden，$h_c$ = head 数，$h_s = d/h_c$ = head size，$P$ = SP degree。

**数学上为什么成立**。AllToAll #1 后每张 GPU 持完整序列的 $Q, K, V$，但只持自己分到的 head。Multi-head attention 在 head 维度上 **embarrassingly parallel** —— 不同 head 在 attention 内部从不交互。所以持 $h_c / P$ 个 head 的 GPU 用任何标准 FlashAttention kernel 都能 *精确* 算出那些 head 的 attention 输出。AllToAll #2 后输出又回到序列切分，准备进 output projection 和下游 FFN。

### 通信量（论文标题数字）

§3.2 原文：

> "On modern clusters with intra-node NVSwitch interconnect and inter-node fat tree IB topology, the communication volume transmitted per link for an all-to-all for aggregate message of size $M$ over $P$ GPUs is $M/P$. For a transformer model with hidden size $h$, sequence length of $N$, and parallelism degree of $P$, DS-Sequence performs all-to-all for the QKV projections with an aggregate message size of $3Nh$ before the attention computation, and another all-to-all for output context projection with a size $Nh$ for each transformer layer. Therefore, DeepSpeed sequence parallelism incurs an aggregate communication volume per link of $\mathbf{4Nh/P}$ (or with the complexity of $O(N/P)$). Note that this communication volume is constant when both $N$ and $P$ are increased proportionally."

对比：Megatron-SP 每 link 量 $4Nh$（不带 $/P$ 扩展），$P\times$ 更大。

### 硬上限：$P \leq \text{num\_heads}$

AllToAll #1 后每张 GPU 要持 *整数个 head* —— $h_c / P$ 必须是正整数。所以 $P \leq h_c$。

**GQA 模型**（大多数现代开源权重）受限于 `num_kv_heads`，远小于 `num_q_heads`：

| 模型 | num_q_heads | num_kv_heads | 最大 Ulysses $P$ |
| ---- | ----------- | ------------ | ---------------- |
| Llama-3-8B | 32 | 8 | **8** |
| Llama-3-70B | 64 | 8 | **8** |
| Mistral 7B (GQA) | 32 | 8 | **8** |
| MQA 模型（单 KV head） | $h_c$ | **1** | **1 —— 等于死了** |

这就是为什么今天 Ulysses 主要 **节点内** 用（8× H100/A100 NVLink 完美适配），节点间通过 [USP](https://arxiv.org/abs/2405.07719) 或 [LoongTrain](https://arxiv.org/abs/2406.18485) 跟 Ring 结合。

### 实现：`DistributedAttention`

`deepspeed/sequence/layer.py`：

```python
class DistributedAttention(torch.nn.Module):
    def __init__(self, local_attention, sequence_process_group,
                 scatter_idx: int = 2, gather_idx: int = 0,
                 sp_stream=None):
        self.local_attn = local_attention      # FlashAttention / SDPA / Triton
        self.spg = sequence_process_group
        self.scatter_idx = scatter_idx          # head 维
        self.gather_idx  = gather_idx           # sequence 维

    def forward(self, query, key, value, batch_dim_idx, ...):
        # AllToAll #1: scatter heads, gather sequence  →  [b, N, hc/P, hs]
        q = _SeqAllToAll.apply(self.spg, query, self.scatter_idx,
                                self.gather_idx, ...)
        k = _SeqAllToAll.apply(self.spg, key,   self.scatter_idx,
                                self.gather_idx, ...)
        v = _SeqAllToAll.apply(self.spg, value, self.scatter_idx,
                                self.gather_idx, ...)

        context = self.local_attn(q, k, v, *args, **kwargs)
        # ↑ head 切分布局上的标准 FlashAttention / SDPA 调用

        # AllToAll #2: scatter sequence, gather heads  →  [b, N/P, hc, hs]
        return _SeqAllToAll.apply(self.spg, context, self.gather_idx,
                                   self.scatter_idx, ...)
```

`_SeqAllToAll` 是 `torch.autograd.Function`。**Backward 对称** —— 同样的 op，swap `scatter_idx` 和 `gather_idx`。梯度 AllToAll 免费。

`sp_stream` / `sp_overlap_comm` 在独立 CUDA stream 上做 backward 重叠（发表之后才加）。**Forward AllToAll 仍然阻塞** —— 见下小节。

### 为什么 forward 没有计算-通信重叠

两次 AllToAll 都是同步集合 op。第一次 AllToAll 的输出直接喂下一个 op（attention 本身），关键路径上没有先前可独立的计算来重叠。论文不声明重叠。

这在跨节点 IB（~25 GB/s vs 节点内 NVLink ~600 GB/s）上是真实代价 —— Ulysses 每层都付全部 AllToAll 延迟。[[ring-attention|Ring]] 的 P2P 反过来天然跟 attention 计算重叠。这是 Ulysses vs Ring 讨论里的主要扩展性弱点。

USP / LoongTrain 实际上 *修复了* 这个 —— 把 QKV AllToAll 切成 per-head 流水线阶段，但这是算法层重构，不在 stock Ulysses recipe 里。

### Causal mask：天然均衡

AllToAll #1 后每 rank 持完整序列的 head 子集，每 rank 的 causal 三角形相同 —— 不需要 [[ring-attention#causal-mask-负载不均衡被掩盖的弱点|Striped/zigzag 修复]]。这是 Ulysses 相对 Ring 最干净的一点。

### FlashAttention 集成

§3.4：*"DeepSpeed Ulysses works with efficient attention implementations such as FlashAttention v2 (Dao 2023)."* 用户代码传任何 local attention 模块：

```python
from deepspeed.sequence.layer import DistributedAttention
from flash_attn import flash_attn_func

local_attn = lambda q, k, v: flash_attn_func(q, k, v, causal=True)
dist_attn = DistributedAttention(local_attn,
                                  sequence_process_group=spg)
```

然后 `dist_attn(q, k, v)` 跑 AllToAll + FlashAttention。FlashAttention v3（Hopper 优化版）同样适用。

### 内存

§3.3：Ulysses 减 **activation 内存**，不减 model-state 内存。它跟 **ZeRO-3** 集成 —— 把 model state（权重 + 梯度 + 优化器状态）切到合并的 $\text{DP} \times \text{SP}$ 组上，每 rank 持 $1/(\text{DP} \cdot \text{SP})$。

KV cache 在层内存在两种布局：
- **Attention 外**：序列切分 `[N/P, d]` per rank —— 总 $Nd/P$ bytes
- **Attention 内**：head 切分 `[N, d/P]` per rank —— 总 $Nd/P$ bytes

都是完整的 $1/P$，只是不同维度。

---

## 实验

### 硬件

**最多 256 张 A100 (40 GB)**，NVSwitch 节点内 + IB fat-tree 节点间。模型：GPT 1.2B、7B、30B；dense 和 blocked-sparse attention 变体。

### 标志数字（论文原文）

| 模型 | 硬件 | 序列 | 吞吐 | vs Megatron-SP |
| ---- | ---- | ---- | ---- | -------------- |
| GPT 1.2B（Fig 3，strong scaling） | 8–64 A100 | 8K → 1M | ~100 TFLOPs/GPU 持续 | —（Megatron 早期 OOM） |
| GPT 7B dense（Fig 4） | 32 A100 | 8K | 159 TFLOPs | 106 TFLOPs |
| GPT 7B dense | 32 A100 | 64K | 175 TFLOPs | OOM |
| GPT 7B dense | 32 A100 | 256K | runs | OOM |
| GPT 30B dense（Fig 5） | 64 A100 | 8K | 165 TFLOPs | 45 TFLOPs |
| GPT 30B dense | 64 A100 | 256K | 134 TFLOPs | OOM (≥128K) |
| GPT 7B sparse（Fig 6） | 32 A100 | 8K → 256K | 132 → 68 TFLOPs | —（Megatron 256K OOM） |
| GPT 30B sparse（Fig 7） | 64 A100 | 256K | 73 TFLOPs | OOM (≥128K) |

Weak scaling（Table 3）：64K @ 64 GPU：161.4 TFLOPs；128K @ 128 GPU：157.4 TFLOPs；**256K @ 256 GPU：147.4 TFLOPs**。"$N/P$ 恒定时通信恒定"的声明实证成立。

**摘要标题**："比现有 SOTA 基线快 2.5×，序列长 4×"、"持续吞吐超过 175 TFLOPs/GPU（54% 硬件峰值）"。

**收敛性（Fig 8）**：1.3B GPT、32K、8 A100、$SP=4$ —— Megatron-SP、Ulysses + ZeRO-1、ZeRO-2、ZeRO-3 的 loss 曲线完全相同。数学上等价。

### 生产部署（已验证）

| 用户 | 怎么用 | 来源 |
| ---- | ------ | ---- |
| **Microsoft Megatron-DeepSpeed** | 论文里点名的主要集成目标 | [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed) —— `--ds-sequence-parallel-size N` flag |
| **HuggingFace TRL / Accelerate（2025 SFT）** | 官方 Ulysses SP 后端；`sp_backend="deepspeed"`、`sp_size=N` | [huggingface.co/blog/ulysses-sp](https://huggingface.co/blog/ulysses-sp) |
| **Ulysses-Offload (FPDT, 2024-12)** | DeepSpeed 后续。**Llama-70B 32× A100 上 4M token**、**8B 模型 4× A100-40GB 上 2M token**、"2.7B 到 80B 区间 55% MFU" | [Ulysses-Offload blog](https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/ulysses-offload/README.md) |
| **Tencent xDiT**（diffusion video） | USP (Ulysses + Ring) 用于 HunyuanVideo、CogVideoX、Wan2.1/2.2、Mochi-1。8 GPU 上 Ulysses-2 × Ring-2 × CFG-2 → **vs 单 GPU 6.12× 加速** | [xdit-project/xDiT](https://github.com/xdit-project/xDiT)、[arXiv:2411.01738](https://arxiv.org/abs/2411.01738) |
| **verl** RL post-training | Ulysses for 长上下文 rollout | [PyTorch 论坛](https://discuss.pytorch.org/t/support-for-ulysses-ring-distributed-attention-for-long-context-training-32k-for-32b-dense-models/223106) |
| **ms-swift (Alibaba ModelScope)** | Ulysses + Ring zigzag | 社区框架 |

### 与 [[ring-attention|Ring Attention]] 对比

| 性质 | DeepSpeed Ulysses | [[ring-attention\|Ring Attention]] |
| ---- | ----------------- | ---------------- |
| GPU 间动什么 | QKV 用 AllToAll 重排（每层 2 次） | KV 块沿环旋转（$N{-}1$ 轮） |
| 每层每 link 通信量 | $4Nh/P$ —— $O(N/P)$（$P$ 同比扩时 $N$ 亚线性） | 每旋转 $\sim 2cd$ × $(N{-}1)$ 轮 |
| 计算 / 通信重叠 | ✗ 阻塞 | ✓ $c \geq F/B$ 时完全隐藏 |
| GPU 数硬上限 | $P \leq \text{num\_heads}$（GQA 下更严） | 无 |
| 跨节点扩展 | ✗ AllToAll 在 IB 上掉性能 | ✓ P2P 带宽友好 |
| Causal mask 均衡 | ✓ 天然均衡 | ✗ 需要 Striped / zigzag 修复 |
| Attention kernel 改动 | 无 —— 用标准 FlashAttention | 有 —— 跟 FA streaming softmax 融合 |
| 最适合 fabric | 节点内 NVLink / NVSwitch | NVLink 或节点间 IB |

---

## 优势与限制

最强两点：(1) **$O(N/P)$ 通信量** 是 SP 家族里最干净的扩展行为 —— $N$ 和 $P$ 同比扩时恒定；(2) **Attention kernel 不动** —— 标准 FlashAttention 直接用，不要 fused 分布式 kernel，causal mask 不需调度技巧天然均衡。

诚实承认的限制：

- **$P \leq \text{num\_heads}$ 是硬上限**。GQA 模型（Llama-3、Mistral）8 个 KV head 的话最大 Ulysses degree 是 8 —— 单节点 NVLink 内合适但出不了节点。MQA（单 KV head）模型根本用不了 Ulysses。
- **没有 forward 计算-通信重叠**。AllToAll 阻塞。跨节点 IB（~25 GB/s vs NVLink ~600 GB/s）上每层都付 AllToAll 延迟。这是相对 [[ring-attention|Ring]] 的主要弱点。
- **跟 Megatron TP 和 PP 不兼容**（stock DeepSpeed，见教程），把用户锁在 ZeRO-3。
- **Backward 重叠是事后加的**。当前源码的 `sp_overlap_comm` / `sp_stream` 路径在独立 CUDA stream 上把 backward AllToAll 跟下一层计算重叠，但这是改良，不在原算法里。
- **通信量声明是 per-link 不是 per-rank**。"$4Nh/P$" 标题数字是每 link 搬的量，不是总通信工作。考虑 AllToAll 里 rank 的参与度后总 comm 更高。

> [!warning] Ulysses 何时赢、Ring 何时赢
> 风格化 2026 决策规则（综合自 USP paper 的 Table 3-4 和 Tencent xDiT benchmark）：
>
> - **节点内、head 数 ≥ GPU 数、NVLink/NVSwitch**：**Ulysses 赢**。NVLink AllToAll 飞快；通信量恒定；无调度复杂度。
> - **跨节点 IB、head 数限制扩展**：**Ring 赢**。P2P 重叠；无 head 数上限。
> - **大规模两者都有（多数生产）**：**混合 —— USP / LoongTrain**。内层 Ulysses degree = 节点大小；外层 Ring degree = 节点数。两全。

---

## 这意味着什么

两条值得跟踪的预测：

1. **Ulysses 作为节点内原语会留下**。MoE 模型变大，每节点 GPU 数稳定在 8（NVLink 域），每节点 Ulysses-degree-8 会继续做默认内层 SP。Tencent USP、LoongTrain、PyTorch 原生 CP 都收敛到这个模式。
2. **GQA / MQA / MLA 趋势会让 head 数上限更严**。DeepSeek-V3 的 MLA 把 KV 压成极小的低秩表示；Llama-4 据报道继续 GQA 趋势 head 更少。纯 Ulysses 会越来越弱 vs Ring。出路要么跟 Ring 结合（USP）、要么在 *完整* head-count 层级（不在 KV head 数）操作（要分布式广播复制 KV —— 加通信）。

这 *不是*：万能长上下文解药（head 数上限挡了前沿规模纯 Ulysses），也不是推理原语（Ulysses 是训练侧，decode 是另一回事），也不是 FlashAttention 替代品（Ulysses 包 FlashAttention，不替代）。

---

## 源码与复现

### 已发布

| Artifact | 状态 |
| -------- | ---- |
| `DistributedAttention` 类 | ✓ Apache-2.0 —— `deepspeed/sequence/layer.py` |
| 测试 | ✓ `tests/unit/sequence_parallelism/test_ulysses.py` |
| Megatron-DeepSpeed 集成 | ✓ `--ds-sequence-parallel-size N` flag |
| Ulysses-Offload (FPDT) | ✓ DeepSpeed v0.13+ |

### 最小复现（裸版）

```python
import torch
from deepspeed.sequence.layer import DistributedAttention
from flash_attn import flash_attn_func

# 在大小为 P 的 torch.distributed 进程组里（P 必须整除 num_kv_heads）：
spg = ...  # sequence_process_group

local_attn = lambda q, k, v: flash_attn_func(q, k, v, causal=True)
dist_attn = DistributedAttention(local_attn, sequence_process_group=spg)

# 输入序列切分：每 rank (batch, N/P, num_heads, head_dim)
output = dist_attn(q, k, v, batch_dim_idx=0)
# 输出同样序列切分
```

### 生产集成 recipe

**Megatron-DeepSpeed（CLI flag）**：

```bash
deepspeed --num_gpus=8 train.py \
  --ds-sequence-parallel-size 8 \
  --deepspeed_config ds_config.json \
  --use-flash-attn-triton \
  ...
```

需要 DeepSpeed v0.10.2+。跟 Megatron TP 和 PP 不兼容（见[教程](https://www.deepspeed.ai/tutorials/ds-sequence/)）。

**HuggingFace TRL / Accelerate (2025+)**：

```python
from trl import SFTConfig, SFTTrainer

config = SFTConfig(
    output_dir="./ulysses_sft",
    sp_backend="deepspeed",
    sp_size=8,
    ...
)
trainer = SFTTrainer(model=model, args=config, train_dataset=ds)
trainer.train()
```

见 [huggingface.co/blog/ulysses-sp](https://huggingface.co/blog/ulysses-sp)。

**Tencent USP (Ulysses + Ring 混合)**：

```python
# 伪代码 —— 见 github.com/feifeibear/long-context-attention
mesh = init_2d_mesh(ulysses_size=8, ring_size=4)  # 共 32 GPU
output = usp_attn(q, k, v, mesh=mesh, causal=True)
```

每节点内 8 路 Ulysses × 节点间 4 路 Ring = 32 GPU 总 SP。

### 值得先读的文件

| 文件 | 角色 |
| ---- | ---- |
| `deepspeed/sequence/layer.py` | `DistributedAttention` 类、`_SeqAllToAll` autograd Function —— 整套 100 行实现 |
| `tests/unit/sequence_parallelism/test_ulysses.py` | Round-trip 测试，断言 AllToAll(AllToAll(x)) = x |
| `blogs/deepspeed-ulysses/README.md` | 官方 walkthrough |
| `blogs/ulysses-offload/README.md` | FPDT 扩展，4M token 训练 |

---

## 相关阅读

- [[ring-attention]] —— P2P 环路线的替代方案。本页姐妹。
- [[parallelism-strategies-deep-dive#7. CP — 上下文并行 (Context Parallelism)]] —— Ulysses 在并行图景里的位置；与 Ring、Megatron CP 对比表。
- [[paged-attention]] —— FlashAttention；Ulysses AllToAll 后原样用它作本地 attention kernel。
- [[kv-cache-optimization]] —— KV cache 压缩与长上下文训练 + 推理交叉。
- [[parallelism-strategies-deep-dive#11. DP Attention — 数据并行注意力 (Data-Parallel Attention for MoE Inference)]] —— attention 在规模下的另一种并行方案；MoE 推理 vs 训练。
- [[long-context-serving]] —— 生产长上下文推理；Ulysses 是训练侧，serving 是另一回事。

## 参考文献

- **DeepSpeed Ulysses 论文**：Jacobs et al., Microsoft. [arXiv:2309.14509](https://arxiv.org/abs/2309.14509) · [ar5iv HTML](https://ar5iv.labs.arxiv.org/html/2309.14509)
- **官方博客**：[DeepSpeed Ulysses README](https://github.com/microsoft/DeepSpeed/blob/master/blogs/deepspeed-ulysses/README.md)
- **教程**：[deepspeed.ai/tutorials/ds-sequence](https://www.deepspeed.ai/tutorials/ds-sequence/)
- **源码**：[`deepspeed/sequence/layer.py`](https://github.com/deepspeedai/DeepSpeed/blob/master/deepspeed/sequence/layer.py) · [test](https://github.com/deepspeedai/DeepSpeed/blob/master/tests/unit/sequence_parallelism/test_ulysses.py)
- **Ulysses-Offload (FPDT)**：[blog](https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/ulysses-offload/README.md) · [tutorial](https://www.deepspeed.ai/tutorials/ulysses-offload/)
- **USP (Ulysses × Ring 混合)**：[arXiv:2405.07719](https://arxiv.org/abs/2405.07719) · [feifeibear/long-context-attention](https://github.com/feifeibear/long-context-attention)
- **LoongTrain (2D-Attention 混合)**：[arXiv:2406.18485](https://arxiv.org/abs/2406.18485)
- **xDiT (Ulysses + Ring for diffusion video)**：[xdit-project/xDiT](https://github.com/xdit-project/xDiT) · [arXiv:2411.01738](https://arxiv.org/abs/2411.01738)
- **HuggingFace Ulysses-SP 博客**：[huggingface.co/blog/ulysses-sp](https://huggingface.co/blog/ulysses-sp)
- **PyTorch Context Parallel** (相关，Ring 为主)：[docs.pytorch.org/.../context_parallel](https://docs.pytorch.org/tutorials/unstable/context_parallel.html)
- **[[ring-attention|Ring Attention]]** (替代)：[arXiv:2310.01889](https://arxiv.org/abs/2310.01889)
- **Insujang CP 综述**：[insujang.github.io/.../introducing-context-parallelism](https://insujang.github.io/2024-09-20/introducing-context-parallelism/)
- **FlashAttention** (本地 kernel)：[Tri Dao FA blog](https://tridao.me/blog/2024/flash3/)
