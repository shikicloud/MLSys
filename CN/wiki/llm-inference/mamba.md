---
title: "Mamba：基于选择性状态空间的线性时间序列建模"
category: llm-inference
tags: [mamba, ssm, state-space-model, selective-ssm, parallel-scan, hardware-aware, linear-attention, hybrid-attention, paper-review]
created: 2026-06-02
updated: 2026-06-02
status: mature
paper: arXiv:2312.00752
code: https://github.com/state-spaces/mamba
---

# Mamba：基于选择性状态空间的线性时间序列建模

> [!info] 论文元数据
> - **论文**：[arXiv:2312.00752](https://arxiv.org/abs/2312.00752) —— *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*，2023 年 12 月，NeurIPS 2024
> - **代码**：[state-spaces/mamba](https://github.com/state-spaces/mamba) —— Apache-2.0，Albert Gu 的参考实现含 CUDA kernel
> - **作者**：Albert Gu（CMU）、Tri Dao（Princeton，也是 FlashAttention 作者）
> - **谱系**：S4（Gu et al., ICLR 2022）→ S5（Smith et al., 2023）→ **S6 / Mamba**（本文）

> [!important] 为什么这个 page 在 2026 年重要
> 2026 年所有 hybrid-attention 生产 LLM —— [[#后续工作：hybrid attention 时代|Qwen3-Next、Qwen3.5、Qwen3.6、Nemotron-H、Kimi Linear、MiMo-V2-Flash、DeepSeek-V3.2 NSA]] —— 的线性 attention 层都是 Mamba 的后裔。理解 Selective SSM 机制、chunkwise 并行扫描、以及 Mamba 特定的弱点（associative recall、ICL）是理解 **为什么业界走向 *hybrid* 而非全 Mamba** 的前置条件。

---

## 摘要（2 分钟读完）

**它是什么。** Mamba 是一个序列模型，把递归模型的 **O(N) 推理复杂度**、Transformer 的**并行训练**、结构化状态空间模型 (SSM) 的**长程建模能力**三者结合。核心技术贡献是 *Selective SSM*（又称 S6）—— 一个 input-dependent 的状态空间转移机制，打破了之前 SSM 的 LTI（线性时不变）假设，让模型能做内容感知的过滤和选择性记忆。

**核心 idea。** **让 SSM 参数 input-dependent**，然后设计 hardware-aware 并行扫描算法让产生的非 LTI 系统仍能 scale 训练。三个部件支撑这个 idea：

1. **Selective SSM（S6）**：矩阵 $\bar{A}_t, \bar{B}_t, C_t$ 变成当前输入 $x_t$ 的函数，通过 linear projection 计算。这给 SSM "决定记什么" 的能力 —— 正是让 S4 在文本任务如 Selective Copy 上失败的缺失能力。
2. **Hardware-aware 并行扫描**：因为 $\bar{A}_t$ 每个 token 不同，S4 用的卷积形式技巧（FFT）失效。改用 Blelloch 并行扫描 + kernel fusion（discretization + scan + projection 融成一个 CUDA kernel），整个 scan 期间 SSM state 保持在 SRAM。这是让 selective SSM 训练可行的关键。
3. **Mamba block**：单一 block 类型替代 attention 和 FFN —— gated MLP 但用 SSM 代替 attention 作为核心。架构*同构* —— 每个 block 都是 Mamba block，没有 transformer 交错。

去掉选择性就退回 S4（不能 copy / filter）；去掉并行扫描训练时间爆炸；去掉同构 block 就退回 ad-hoc SSM-transformer 杂交。

**头号实验结果。** 3B 参数 + 300B token 在 The Pile 上，Mamba 在语言建模 perplexity 上匹敌 Transformer++（带所有 trick 的现代 transformer baseline：RoPE、GLU、RMSNorm 等）。在 2K 序列长度上 decoding throughput **比 Transformer++ 快 5×**。序列长度 scaling **线性**而 attention 是二次。在长上下文音频建模和 DNA 序列 benchmark 上 SOTA。

**为什么重要。**

- **第一个可信的大规模 attention 替代品。** Mamba 之前的 SSM（S4、S5、H3、Hyena）都是研究级，在语言上不能匹敌 transformer。Mamba 是第一个在真实语言 benchmark 上 demonstrably 匹敌 Transformer++ at 3B 规模的。
- **重启了 linear-attention 研究 program** —— 同时暴露了它特定的失败模式。Mamba 暴露的 associative-recall 弱点催生了 2 年的后续工作（Mamba-2、Gated DeltaNet、Lightning Attention、KDA）和主导 2026 的 hybrid-attention 生产架构。
- **让 hardware-aware kernel 协同设计成为标准。** Selective scan kernel 跟算法本身一样是贡献。后续 linear-attention 变种（Gated DeltaNet 的 WY 递归、KDA 的 delta-rule kernel）都遵循同一模式：并行扫描 + SRAM-resident state + kernel fusion。
- **推理 serving 的影响。** Mamba 的 per-sequence state 是 *固定大小*（不像 KV cache 跟长度相关）。对长上下文 serving，这彻底改变内存预算计算 —— 并强制 hybrid-attention KV manager 重新设计（见 [[paged-attention]] hybrid 扩展，如 vLLM PR #22688）。

---

# 详细内容（深入阅读从这里开始）

## 背景：SSM 谱系和缺失的能力

深度学习里的状态空间模型经过三代。

### 经典 SSM（连续时间）

连续时间线性 SSM 描述系统为：

$$
h'(t) = A h(t) + B x(t), \quad y(t) = C h(t)
$$

为了在深度学习中使用，用步长 $\Delta$ 离散化：

$$
h_t = \bar{A} h_{t-1} + \bar{B} x_t, \quad y_t = C h_t
$$

其中 $\bar{A} = \exp(\Delta A)$，$\bar{B} = (\Delta A)^{-1}(\exp(\Delta A) - I) \Delta B$。**关键是 $\bar{A}, \bar{B}, C, \Delta$ 全部跟数据无关** —— 这是 LTI（Linear Time-Invariant）假设。

### S4（Gu et al., ICLR 2022）

S4 通过以下方式让 SSM 能 scale 训练：

1. **结构化矩阵初始化**（HiPPO 理论）：选特定结构形式的 $A$（对角加低秩），在 SSM kernel 中捕捉长程依赖。
2. **卷积视角**：因为 LTI，整个输出可以写成单个卷积 $y = \bar{K} * x$，其中 $\bar{K}$ 是 SSM kernel。
3. **FFT 并行训练**：长度 $N$ 的卷积用 FFT 在 $O(N \log N)$ 算出 —— 并行友好。
4. **递归推理**：decode 时用递归形式 $h_t = \bar{A} h_{t-1} + \bar{B} x_t$ —— O(N) 而不是 O(N²)。

S4 在 Long Range Arena (LRA)、音频、时序 benchmark 上 SOTA。但在语言建模上**没**匹敌 transformer。

### S5、H3、Hyena —— 增量改进

- **S5**（Smith et al., 2023）：对角 SSM + 并行扫描递归（代替 FFT 卷积）。GPU 上更快。
- **H3**（Fu et al., 2023）：加了 "Hungry Hungry Hippos" gating，能做简单合成任务如 induction heads。更接近 transformer 性能但仍有 gap。
- **Hyena**（Poli et al., 2023）：带隐式参数化的长卷积。

这些全部保持 LTI。**它们在 Selective Copy 和 Induction Heads 任务上灾难性失败** —— 这些任务模型需要基于内容决定 *记住哪些* token。

> [!quote] 诊断性的 Selective Copy 任务
> 给一个长序列里散布着几个 "key" token 在噪音里，输出顺序输出 key token。**LTI SSM 无法做到这点**，因为转移矩阵 $\bar{A}$ 必须对噪音 token 和 key token 一样。Mamba 的选择性直接针对这个。

论文把 LTI 诊断为瓶颈：

> "LTI models have a fundamental limitation in handling content-based reasoning. ... The model's dynamics are constant over time, which means the model cannot select what information to focus on or what to ignore based on the content of the sequence."

## Selective SSM（S6）—— Mamba 的核心贡献

修复在概念上简单：**让 $\bar{B}, C, \Delta$ 成为输入 $x_t$ 的函数**。

### 选择性更新

对每个输入 token $x_t \in \mathbb{R}^D$：

$$
\begin{aligned}
B_t &= s_B(x_t) \quad \in \mathbb{R}^N \\
C_t &= s_C(x_t) \quad \in \mathbb{R}^N \\
\Delta_t &= \tau_\Delta(\text{Parameter} + s_\Delta(x_t)) \quad \in \mathbb{R}_+^D
\end{aligned}
$$

其中 $s_B, s_C, s_\Delta$ 是 linear projection（$\Delta_t$ 多一个跨 $D$ channel 的广播），$\tau_\Delta$ 是 softplus 保证 $\Delta_t > 0$。

$A$ 保持跟数据无关（保持结构化，通常用对角初始化 A-real-imag 形式）。离散化用 input-dependent $\Delta_t$：

$$
\bar{A}_t = \exp(\Delta_t A), \quad \bar{B}_t = \Delta_t B_t
$$

递归现在是非 LTI 的：

$$
h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t, \quad y_t = C_t h_t
$$

**直观解释**：$\Delta_t$ 控制输入多大程度 "leak 进 state" —— 小 $\Delta_t$ 表示 "忽略这个 token"（state 不变），大 $\Delta_t$ 表示 "重置 / 覆盖 state"。结合 input-dependent 的 $B_t, C_t$，模型能做内容感知过滤。

### 选择性买到什么

| 能力 | LTI SSM (S4, S5, H3) | Selective SSM (Mamba) |
| ---- | -------------------- | --------------------- |
| Selective Copy 任务 | ✗ 无法解决 | ✓ 100% 解决 |
| Induction Heads | ✗ 无法外推 | ✓ 外推到更长序列 |
| 语言建模（Pile）| 落后 Transformer++ | **匹敌 Transformer++** |
| 内容感知过滤 | ✗ | ✓ |

这三个 benchmark 让 Mamba 立住了"第一个在语言任务上能跟 transformer 竞争的 SSM"地位。

### 代价：没有卷积形式

LTI 让你能写 $y = \bar{K} * x$ 通过 FFT 算。Selective（$\bar{A}_t$ 每 token 不同）**打破这点**。你只剩串行递归，朴素实现在 $O(N)$ 但串行 —— GPU 上太糟。

## Hardware-aware 并行扫描 —— 让 selective SSM 训练可行

Selective SSM 的递归 $h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t$ 是 *结合扫描* 的形式：每步把前一 state 和当前输入通过结合操作合并。结合扫描通过 **Blelloch scan** 算法并行，深度 $O(\log N)$。

### 朴素结合形式

每个 token 的更新：

$$
(h_t, \cdot) = (\bar{A}_t, \bar{B}_t x_t) \oplus (h_{t-1}, \cdot)
$$

其中 $\oplus$ 是结合操作：

$$
(A_1, b_1) \oplus (A_2, b_2) = (A_2 A_1, A_2 b_1 + b_2)
$$

标准 Blelloch scan 在 $N$ token 上用 $O(N)$ work 和 $O(\log N)$ depth —— 原则上跨 N GPU 并行。

### Mamba 的实际实现：kernel-fusion + SRAM-resident state

Selective scan 作为独立 CUDA kernel 还有致命性能问题：**中间 state materialize 到 HBM 占主导运行时间**。隐藏 state $h_t$ 形状 $(B, D, N)$，$D \approx 2048$，$N$（state dim）$\approx 16$。长度 2K 序列 materialize 所有 $h_t$ 用 $2K \times B \times D \times N = $ 每层数百 MB —— HBM 带宽成为瓶颈。

Mamba 的 selective scan kernel 做三件事：

1. **融合 discretization + scan + 输出投影** 进一个 CUDA kernel。输入进去，输出出来，不 materialize 中间。
2. **整个 scan 期间 SSM state 保持在 SRAM**。State 小（$D \times N = 32K$ float per token batch）能 fit SRAM。
3. **在 backward 重新计算 scan** 而不是存所有中间 state。这跟 FlashAttention 在 attention 上用的 trick 一样 —— backward 用计算换内存。

论文测量结果：

| 实现 | seqlen 2K 吞吐 | 内存 |
| ---- | ------------: | ---: |
| 朴素 PyTorch（materialize state）| 1× baseline | 100% |
| Selective scan kernel | **20-40× 更快** | <10% |

**没有 fused kernel，selective SSM 比 transformer 慢**，尽管渐近优势。Kernel 让算法实用。

> [!note]- 超长序列的 chunkwise scan
>
> 对单次 kernel launch 处理不下的超长序列（如 1M+ token），Mamba 用 *chunkwise* scan：把序列切成 chunk（如 256），每个 chunk 内部算自己的 scan，chunk 之间通过 state passing 链接。这是后来 hybrid 模型（Qwen3-Next、Gated DeltaNet）继承和进一步优化的。

## Mamba block —— 单一同构 block 替代 attention 和 FFN

Mamba 架构异乎寻常地干净：**每层都是 Mamba block**，没有 transformer 交错，没有单独的 FFN 层。

Mamba block（论文 Fig. 3）对输入 $x \in \mathbb{R}^{B \times L \times D}$ 做：

```
x_in  ──→ Linear (D → 2·E·D)  ──┬──→ Conv1d (kernel=4) ──→ SiLU ──→ SSM (Selective) ──┐
                                  │                                                       │
                                  └────────────────────────── (作为残差?) ────────→ ⊗ gate
                                                                                          │
                                                                                          ▼
x_in  ──→ Linear (D → E·D) ────────────→ SiLU ──────────────────────────────────────────⊗
                                                                                          │
                                                                                          ▼
                                                                            Linear (E·D → D) ──→ x_out
```

其中：
- $E$ 是 "expansion factor"，通常 2。
- SSM 是上面 §3 的 Selective SSM。
- SSM 前的 Conv1d 作为 local mixer（覆盖小窗口 pattern）。
- Gating 是 GLU 风格（SiLU(Wx) ⊗ SSM_path）—— 借自 gated MLP 设计。

**没有 attention。没有单独 FFN。block 内部没有 layer norm**（LN 在 block 之间应用）。完整 Mamba 模型就是 $\text{LN} \to \text{MambaBlock} \to \text{LN} \to \text{MambaBlock} \to \cdots$。

这种同构性是有意的：强制所有序列建模能力进 SSM，让模型 scale 不需要 attention/FFN 比例、head 数等架构选择。

## 头号实验证据

### The Pile 上的语言建模

3B 参数 + 300B token，Mamba 匹敌 Transformer++（强 baseline 带 RoPE、GLU、RMSNorm 等）。模型规模 scaling 曲线（论文 Fig. 5）显示 Mamba 在 125M 到 1.4B 每个规模都稍微 *领先* Transformer++，gap 在 3B 处缩小（统计噪声可能）。

### 推理时吞吐

| 模型 | batch 32, seqlen 2K decoding 吞吐（tokens/sec）|
| ---- | -------------------------------------------: |
| Transformer++ (3B) | 1× baseline |
| **Mamba (3B)** | **~5× 更快** |

5× 来自两个来源：
1. **没有 KV cache** —— Mamba 的 state 固定大小，不随序列长度增长。
2. **O(N) per-token compute** —— 序列长度线性，而 transformer 每 token O(N) attention compute（生成总共 O(N²)）。

### 长程任务

- **音频**：YouTubeMix、SC09 SOTA。Mamba 的线性时间 scaling 让它能训长音频序列（16K+ 采样），二次 attention 负担不起。
- **DNA**：Genomic-Benchmarks SOTA。长 DNA 序列（10K+ base pair）是线性模型的自然领域。
- **合成 Selective Copy**：100%（对比 S4 的 ~0%，transformer 也能通过 attention 解决）。

### 序列长度 scaling

Mamba 的 compute 跟序列长度**线性** scale，attention 是二次。实证上 16K 上下文 gap 很大；1K 上下文 transformer 在绝对速度上仍赢，因为更好的硬件 fit。

> [!success] 真正令人惊讶的结果
> Mamba 之前 SSM 在语言建模上以显著 margin 输给 transformer。Mamba 在 3B 上匹敌强 transformer baseline。**选择性是缺失的那块** —— 不是更大 SSM、不是更好离散化、不是更深栈。光是选择性就关闭了 gap。

## 优势与局限

**优势。**

- **真正的 attention 架构替代品** —— 第一个在语言建模上 at scale 匹敌 transformer 的 SSM。
- **线性时间推理** —— 长上下文的杀手特性。
- **Hardware-aware kernel 设计** —— Selective scan kernel 本身是贡献，不只是工程。
- **架构干净** —— 同构 block，不需要调 attention/FFN 比例。
- **不需要上下文窗口工程就能 scale 长序列**（没有 RoPE 扩展、没有 positional 插值）。

**局限**（一些不在论文里，被后续工作识别）。

> [!warning] Associative recall 和 in-context learning 弱
> Mamba 发布 6 个月内，业界发现 Mamba **在 associative recall 上失败**（Phonebook 风格任务 —— 给一列 (key, value) pair，按 key 检索 value）。这是架构根本性的：SSM state 是 *固定大小*，无法无损存任意 key→value 映射。Attention 有无界历史 lookup；Mamba 有有限内存瓶颈。

- **In-context learning 比 attention 弱。** Mamba 能做 *一些* ICL 但在更难的 ICL benchmark 上落后 transformer（特别是 in-context 示例长的时候）。
- **论文级别 scaling 没超过 3B。** 论文只验证到 3B 参数。Mamba 的收益是否在 70B+ 还成立这篇论文单独不知道。
- **Selective scan kernel 是 CUDA 特定的。** Triton port 后来才有；ROCm 和 TPU port 在 2026 年仍不完整。
- **$\Delta_t$ 参数化脆弱。** 初始化敏感性是真的；尝试复现的组报告 at scale 训练不稳定。
- **没讨论推理时 *prefix sharing*** —— 固定大小 state 意味着你**不能像共享 KV cache 那样**在两个共享 prefix 的请求间共享 SSM state，这复杂化 RadixAttention 等 serving 优化。

## 这意味着什么 —— hybrid attention 时代

Mamba 同时是正面结果和负面诊断。它证明了 selective SSM 能在标准语言建模上匹敌 transformer，但也*精确*暴露了它们做不到什么（associative recall、难 ICL、检索）。

2024-2026 业界响应几乎一致是 **hybrid 架构**：大多数层用 linear-attention（Mamba 后裔）做 compute 效率，但插入少量 full-attention 层（通常每 4-8 层 1 层）恢复检索 / ICL 能力。

### 后续工作：hybrid attention 时代

| 工作 | 年份 | Linear 变种 | Hybrid 比例 | 备注 |
| ---- | ---- | ---------- | ----------- | ---- |
| **Mamba-2 / SSD**（Gu+Dao, 2024 年 5 月，arXiv:2405.21060）| 2024 | SSD (Structured State-Space Duality) | 全 Mamba | 把 SSM 和 attention 统一为 structured matrix；比 Mamba-1 快 2-8× |
| **Jamba**（AI21, 2024 年 3 月）| 2024 | Mamba | 1:7 attention:Mamba | 第一个生产 hybrid；12B active / 52B total |
| **Zamba**（Zyphra, 2024）| 2024 | Mamba | 单一 global attention block | 7B；同规模 SOTA |
| **Samba**（Microsoft, 2024）| 2024 | Mamba + Sliding-Window Attention | 交错 | 开源 3.8B |
| **Nemotron-H**（NVIDIA, 2024）| 2024 | Mamba-2 + attention | hybrid 层 3:1 比例 | NVIDIA 首个生产 hybrid |
| **Gated DeltaNet**（Yang et al., ICLR 2025, arXiv:2412.06464）| 2025 | Gated DeltaNet（delta rule）| base 全 Mamba | 驱动 Qwen3-Next、Qwen3.5、Qwen3.6 |
| **Lightning Attention**（Qin et al., 2024）| 2024 | Lightning 类 linear | 全 linear | MiniMax 用 |
| **KDA (Kimi Delta Attention)** | 2025 | Delta rule 变种 | Hybrid | Kimi Linear、K2 家族用 |
| **Mamba-3**（2026 提到）| 2026 | 下一代 Mamba | 强调 hybrid | 活跃研究 |

### 2027 年的三个预测

1. **全 Mamba 生产模型死了。** Associative recall 弱点根本性强到没人把旗舰 LLM 当纯 SSM 出。Hybrid 3:1 或 7:1 仍是标准模式。
2. **Mamba state 成为 serving 系统下一个前线。** KV cache 管理已经成熟；hybrid-aware 内存管理器（vLLM 的 heheda12345 PR 系列、SGLang 的 hybrid 扩展）仍在成熟。预期跨架构 KV/state 传输协议（hybrid 的 PD 解耦，见 [[prfaas]]）的新 paper。
3. **Selective scan kernel 成为标准库原语。** 像 FlashAttention 的 `flash_attn_func`，预期 FLA / NVlabs 的 `selective_scan_fn` 成为每个框架调用的标准 API。

## 源代码与复现

```bash
# 安装（要求 CUDA 11.6+ 和 PyTorch 1.12+）
pip install mamba-ssm
pip install causal-conv1d  # 必需依赖
```

**最小 Mamba block 用法**（来自参考 repo）：

```python
import torch
from mamba_ssm import Mamba

batch, length, dim = 2, 1024, 16
x = torch.randn(batch, length, dim).cuda()
model = Mamba(
    d_model=dim,        # 模型维度
    d_state=16,         # SSM state size（paper 里的 N）
    d_conv=4,           # Conv1d kernel size
    expand=2,           # E expansion factor
).cuda()
y = model(x)
assert y.shape == (batch, length, dim)
```

**作者发布的预训练 checkpoint**：

| Checkpoint | 参数 | 训练 token | URL |
| ---------- | --: | --------: | --- |
| `mamba-130m` | 130M | 50B | https://huggingface.co/state-spaces/mamba-130m |
| `mamba-370m` | 370M | 50B | https://huggingface.co/state-spaces/mamba-370m |
| `mamba-790m` | 790M | 50B | https://huggingface.co/state-spaces/mamba-790m |
| `mamba-1.4b` | 1.4B | 300B | https://huggingface.co/state-spaces/mamba-1.4b |
| `mamba-2.8b` | 2.8B | 300B | https://huggingface.co/state-spaces/mamba-2.8b |

| 文件路径 | 角色 |
| ------- | ---- |
| `mamba_ssm/modules/mamba_simple.py` | `Mamba` block —— 顶层用户接口 |
| `mamba_ssm/ops/selective_scan_interface.py` | 包装 CUDA kernel 的 selective scan op |
| `csrc/selective_scan/selective_scan_fwd_kernel.cuh` | Forward CUDA kernel（hardware-aware 实现）|
| `csrc/selective_scan/selective_scan_bwd_kernel.cuh` | Backward CUDA kernel（recompute-in-backward 技巧）|
| `mamba_ssm/models/mixer_seq_simple.py` | 完整 Mamba LM stack |

**值得知道的生产级实现**：

- **[fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention)** —— Mamba、Mamba-2、GLA、Gated DeltaNet 及 ~20 个其它 linear-attention 变种的 Triton kernel。**vLLM 和 SGLang 里 hybrid 模型的事实 kernel 库**。
- **[NVlabs/GatedDeltaNet](https://github.com/NVlabs/GatedDeltaNet)** —— NVIDIA 的 GDN（Qwen3-Next / Qwen3.5 的 linear-attention 变种）参考实现。
- **vLLM hybrid KV manager** —— heheda12345 的 PR #22688 把 attention KV cache 和 Mamba state 在一个 allocator 下统一；serving Qwen3-Next、Nemotron-H 必需。

## 相关阅读

- [[paged-attention]] —— PagedAttention 的 hybrid 扩展是 Mamba 的 serving 侧回应：如何在一个内存 allocator 内管理 *attention KV block* 和 *Mamba state*。
- [[kv-cache-optimization]] —— Mamba 完全绕过 KV cache（固定大小 state）。对长上下文 serving，这彻底改变内存数学。
- [[prfaas]] —— Cross-DC PD 解耦显式由 hybrid-attention 模型 KV 吞吐远低于 dense GQA（论文称 13× 更低）motivate —— 这是 Mamba 线性架构直接的下游后果。
- [[ring-attention]] —— 长上下文的序列并行 attention；hybrid 通过减少 attention 层数部分 obsolete 的替代路径。
- [[sglang]]、[[vllm]] —— ship hybrid 模型支持的推理引擎；SGLang 通过 FLA Triton，vLLM 通过 hybrid KV manager。
- [[speculative-decoding]] —— 投机解码跟 Mamba 固定大小 state 有非平凡交互：拒绝需要 state checkpoint，比 KV 截断更贵。见 *Component-Aware SSD*（arXiv:2605.01106）的 18× α gap between parallel 和 sequential hybrid。
- [[quantization]] —— Mamba state 量化是 underexplored 区域；当前实现 KV cache 即使 FP8 或 NVFP4 时 state 仍保 BF16。
- [[continuous-batching]] —— Hybrid 模型 batching 需要 Mamba-state-aware 调度；不是所有调度器都干净处理。

## 参考文献

- Albert Gu, Tri Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* arXiv:2312.00752, 2023 年 12 月。NeurIPS 2024. https://arxiv.org/abs/2312.00752
- state-spaces/mamba. https://github.com/state-spaces/mamba
- Albert Gu. *Efficiently Modeling Long Sequences with Structured State Spaces*（S4）。ICLR 2022.
- Dao, Gu. *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality*（Mamba-2）。arXiv:2405.21060, 2024 年 5 月。
- Yang et al. *Gated Delta Networks: Improving Mamba2 with Delta Rule.* arXiv:2412.06464, ICLR 2025.
- Lieber et al. *Jamba: A Hybrid Transformer-Mamba Language Model.* AI21, 2024 年 3 月。
- Glorioso et al. *Zamba: A Compact 7B SSM Hybrid Model.* Zyphra, 2024.
- Ren et al. *Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling.* Microsoft, 2024.
- fla-org/flash-linear-attention. https://github.com/fla-org/flash-linear-attention
