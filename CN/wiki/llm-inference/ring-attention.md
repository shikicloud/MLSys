---
title: "Ring Attention：跨设备切分序列维度并完美隐藏通信"
category: llm-inference
tags: [ring-attention, context-parallelism, long-context, attention, flash-attention, blockwise-transformer, striped-attention, paper-review]
created: 2026-05-19
updated: 2026-05-21
status: mature
paper: arXiv:2310.01889
code: https://github.com/lhao499/RingAttention
---

# Ring Attention：跨设备切分序列维度并完美隐藏通信

> [!info] 论文元信息
> - **论文**：[arXiv:2310.01889](https://arxiv.org/abs/2310.01889) —— *Ring Attention with Blockwise Transformers for Near-Infinite Context*（Hao Liu, Matei Zaharia, Pieter Abbeel；UC Berkeley）
> - **会议**：**ICLR 2024** (poster)。早期工作坊版本在 NeurIPS 2023 FMDM。*不是* ICML 2024 paper —— 常见误记。
> - **源码（JAX/TPU 权威版）**：[lhao499/RingAttention](https://github.com/lhao499/RingAttention) —— Apache-2.0
> - **PyTorch 移植**：[zhuzilin/ring-flash-attention](https://github.com/zhuzilin/ring-flash-attention)（最常用）、[lucidrains/ring-attention-pytorch](https://github.com/lucidrains/ring-attention-pytorch)
> - **前置论文**：[Blockwise Parallel Transformer (Liu & Abbeel, 2023)](https://arxiv.org/abs/2305.19370) —— 单卡 blockwise attention + FFN；Ring Attention 是它的分布式版本
> - **配套后续**：[Striped Attention](https://arxiv.org/abs/2311.09431)（同一组人；修复 causal mask 负载不均衡，1.45–1.65× 提速）
> - **配套页面**：[[deepspeed-ulysses]] —— AllToAll 路线的替代方案

---

## 摘要（2 分钟读完这一节就够）

**它是什么**。Ring Attention 把 attention 的 **序列维度** 切到 $N$ 张 GPU 排成的 **环** 上。每张 GPU 保留一份固定的本地 Q 块；K/V 块每步沿环旋转一跳。$N{-}1$ 轮旋转之后每个 Q 都跟每个 K/V 配对过 —— 得到 **与单卡 attention 完全相同的输出**,而每卡 activation 内存 **与总序列长度 $S$ 无关**。

**核心思想**。Q 不动、KV 旋转、softmax 流式。三个支柱:

1. **Q 不动 + KV 旋转**。每张卡的 $S/N$ 个 query 固定不动；KV 块每步通过 P2P 向左传一跳（JAX 用 `lax.ppermute`，PyTorch 用 `dist.batch_isend_irecv`）。
2. **FlashAttention 风格的 online softmax**：每个 Q 行在旋转步间维护 `(running_max, sum, output)` 三元组,任何地方都不实例化完整 $S \times S$ attention matrix。
3. **$c \geq F/B$ 时计算完全藏住通信**（块大小 $\geq$ host-FLOPs / 互联带宽）。A100 + NVLink 上是几百 token —— 轻松满足。

少任一支柱:某处会重新实例化完整 attention matrix、被通信拖死、或者得出错误结果。

**头条结果**。7B 模型在 **32× A100 上训到 4M token context**、**TPU v4-1024 上 8M token**,与单卡 attention 比无质量损失。撑起这篇论文的是内存对比图:

![Vanilla / Memory-Efficient Attn / + Memory-Efficient FFN / Ring Attention 跨模型规模可达到的最大 context size（论文 Fig. 4）](CN/wiki/llm-inference/ring-attention-figs/max-context-vs-baselines.png)

固定显存预算下,Ring Attention 把可训练 context 比仅用 Memory-Efficient Attention 又延长 **2–3 个数量级** —— 因为每卡内存与 $S$ 无关。

**为什么这重要**。

- **生产 CP 已经全面收敛到 Ring**。Megatron-Core Context Parallelism、Meta Llama 3 训练、Tencent USP、PyTorch 2.7+ 原生 CP —— 都是 Ring Attention 加工程加固（zigzag 调度、pass-Q 变体、与 [[deepspeed-ulysses|Ulysses]] 的混合）。裸 2023 算法不能直接用;2026 生产栈才是。
- **无精度妥协**。输出与单卡 attention 比特对齐。区别于近似长上下文路线（线性 attention、sliding window、检索）。
- **下一个瓶颈是互联,不是算法**。$c^* = F/B$ 是设计点 —— NVLink 级别几乎是必要条件,PCIe 不可行。
- **对比 [[deepspeed-ulysses|DeepSpeed Ulysses]]**。Ring 可以扩到任意 $N$,但需要高带宽;Ulysses 卡在 $N \leq \text{num\_heads}$,但每卡通信量恒定。混合（USP）是生产答案。

---

# 深度部分（往下展开细节）

上面摘要是 executive 层。下面是给愿意细读算法和代码的人准备的。

## 背景:为什么要切序列维度（而不是 head 或层）

Transformer 在长上下文下有三个结构性事实:

1. **KV cache 超出单卡显存**。Llama-3-70B 在 $S = 1\text{M}$、FP16、80 层、8 个 KV head、$d_{\text{head}}=128$:$2 \times 80 \times 8 \times 128 \times 10^6 \times 2 \approx 328$ GB / 请求 —— 远超 H100 的 HBM。
2. **Attention FLOPs 是 $O(S^2)$**。序列翻倍计算量翻 4 倍。任何内存技巧改不了。
3. **现有并行轴帮不上**:
   - [[parallelism-strategies-deep-dive#4. TP — 张量并行 (Tensor Parallelism)|TP]] 切权重,不切激活 —— KV cache 在 TP rank 间是复制的。
   - [[parallelism-strategies-deep-dive#6. PP — 流水线并行 (Pipeline Parallelism)|PP]] 切层 —— 每层还是持有全序列。
   - [[parallelism-strategies-deep-dive#2. DP — 数据并行 (Data Parallelism)|DP]] 切 batch —— 每张卡还是持有一条完整序列。

[[paged-attention|FlashAttention]] 解决了单卡内存（流式 tile 化 attention）,但 $N = 1$ 时还是 $O(S/N)$ 每卡内存。自然下一步:把 **序列维度本身** 切到多卡上,每张卡只持有 $S/N$ 个 token。

但 attention 要求每个 query 看到每个 key。naively,每张卡上的 Q 要访问分布在所有 $N$ 卡上的全部 $S$ 个 key —— 全局 gather,又贵又不省内存。Ring Attention 的贡献就是 **找对了 KV 移动的调度方式**,让中间过程从不在任何一张卡上实例化完整 attention matrix。

前置工作 Blockwise Parallel Transformer（BPT,[arXiv:2305.19370](https://arxiv.org/abs/2305.19370),同组 2023-05）解决了单卡内存:blockwise streaming attention + blockwise streaming FFN。Ring Attention 拿同一个 kernel,**把 Q 切到多卡,KV 沿环旋转** —— 把单卡 $32\times$ 长上下文技术变成了 $32\times \times \text{设备数}$ 长上下文的分布式技术。

## 核心思想:Q 不动、KV 旋转、softmax 流式

> [!quote] 一句话总结贡献
> $N$ 张 GPU 每张固定保留自己那段 Q 块,每步从左邻居那里接收一段新的 K/V 块;$N{-}1$ 步之后每个 Q 都看过所有 K/V,FlashAttention 风格的 online softmax 维护流式统计量,每卡 activation 内存 $O(b \cdot c \cdot h)$ 与总序列长度 *无关*。

论文自己的示意图把整个编舞收进一张图:

![Ring Attention:(a) 两卡视角的 KV 块在 blockwise transformer 层之间向左移位;(b) 单卡视角的 Q-outer / KV-inner 循环,Q 不动,K/V 周期性流入（论文 Fig. 1–2）](CN/wiki/llm-inference/ring-attention-figs/ring-rotation-blockwise.png)

三个支撑次级声明:

- **每卡内存与 $S$ 无关**。论文 Table 1:每个 host 的 attention activation 是 $6 \cdot b \cdot c \cdot h$ —— 只依赖块大小 $c$ 和隐藏 $h$,**不依赖** 总 $S$。序列长度随设备数 $N$ 线性增长但每卡压力不变。
- **通信在干净条件下免费**。每步计算 $4 d c^2 / F$（块 attention FLOPs / host FLOPs/s）;每步通信 $4 c d / B$（一个 KV 块 / 互联字节/s）。重叠要求 $c \geq F/B$。A100 + NVLink 上 $F/B \approx \text{几百 token}$ —— 容易满足。
- **数学上严格等价**。不近似:Ring Attention 输出与单卡 attention 比特对齐。诀窍在 online softmax —— 把跨旋转步的部分结果以数值稳定方式合并。

去掉任何一个:失去 Q 不动就在某处重新实例化完整 attention matrix;失去重叠条件通信就主导时间;失去 online softmax 部分结果就合并不正确。

## 实现细节

### 算法

论文 Algorithm 1 原文:

```
要求:输入序列 x,host 数 N_h
初始化:
  把输入序列切成 N_h 块（每个 host 一块）
  每个 host 在本地块上计算 query、key、value

对每个 transformer 层 do
  for count = 1 到 N_h − 1 do
    每个 host 并发 do:
      用本地 query, key, value 块增量算 memory-efficient attention
      把 key、value 块发给下一个 host
      从前一个 host 接收 key、value 块
```

每个 host 的状态:一份固定的本地 Q 块 $Q_i$,加一份 K/V 块 —— 第 $k$ 步那个 K/V 块是环上比它早 $k$ 个位置出发的块。

### Online softmax 机制（方法的核心）

每个 Q 行在旋转步间维护三个流式统计量:

| 状态 | 形状 | 初始化 |
| ---- | ---- | ------ |
| `numerator` | `(B, q_len, H, d_head)` | 零 |
| `denominator` | `(B, H, q_len)` | 零 |
| `prev_max_score` | `(B, H, q_len)` | `-inf` |

每来一段 KV chunk（取自 `ringattention/ringattention_jax.py` 132–143 行）:

```python
attn_weights = einsum('bqhd,bkhd->bhqk', q_chunk, k_chunk) / scale
attn_weights += bias_chunk
max_score_chunk = maximum(prev_max_score_chunk,
                          attn_weights.max(axis=-1))
exp_weights = exp(attn_weights - max_score_chunk[..., None])
exp_values = einsum('bhqk,bkhd->bqhd', exp_weights, value_chunk)
correction = exp(prev_max_score_chunk - max_score_chunk)
numerator_chunk   = numerator_chunk   * correction + exp_values
denominator_chunk = denominator_chunk * correction + exp_weights.sum(-1)
```

最终输出 per Q row:`numerator / denominator`。

这就是 **FlashAttention / online-softmax recurrence**,只是 `k_chunk, v_chunk` 来自分布式旋转 buffer（JAX 里的 `lax.ppermute`,PyTorch 里的 `dist.batch_isend_irecv`）,不是 HBM。

### 计算 / 通信重叠条件

论文 §3 给出形式化条件:

$$
\underbrace{\frac{4 d c^2}{F}}_{\text{每步计算}} \;\geq\; \underbrace{\frac{4 c d}{B}}_{\text{每步 KV 传输}} \quad\Longrightarrow\quad c \;\geq\; \frac{F}{B}
$$

$c$ = 块大小（token）,$d$ = 隐藏维,$F$ = host FLOPs/s,$B$ = 主机间字节/s。满足时环旋转在本地 attention 计算完之前完成 —— 通信完全藏在计算之后。

数值实例:

| 硬件 | $F$ (FP16) | $B$ | $c^* = F/B$ |
| ---- | ---------: | ---: | ----------: |
| A100 + NVLink 机内 | ~312 TFLOPs/s | ~600 GB/s | 几百 token |
| TPU v4 + ICI | 类似 | 类似 | 类似 |
| H100 + IB 机间 | ~989 TFLOPs/s | ~25 GB/s | 几千 token |
| 仅 PCIe 节点 | ~H100 | ~25 GB/s | 几万 —— 一般不可行 |

硬件故事:Ring Attention 是 **高带宽互联** 技术。NVLink 级别是实际可用的前提。

每次旋转、每卡:$2 \cdot c \cdot d \cdot \text{dtype\_bytes}$（一个 K 块 + 一个 V 块）—— **与总 $S$ 无关**。$c = 4096, d = 8192$、FP16:128 MiB / 旋转;600 GB/s NVLink ~210 µs / 旋转。本地 $c \times c$ 块 attention 计算在毫秒级 —— 通信藏住了。

### 内存分析

论文 Table 1,Ring Attention 每 host attention activation 成本:

$$
\boxed{\,6 \cdot b \cdot c \cdot h \text{ bytes / host}\,}
$$

—— **与总 $S$ 无关**。分解:1 个当前 Q 块、2 个当前 K/V、2 个收到的 K/V、1 个输出。

| 方法 | 每 host attention activation |
| ---- | ---------------------------- |
| 朴素 attention | $O(b \cdot S^2)$ |
| FlashAttention（单卡） | $2 \cdot b \cdot S \cdot h$ |
| **Ring Attention** | $\mathbf{6 \cdot b \cdot c \cdot h}$ |

KV cache 平均切片:每 host $O(S/N)$。

### 辅助机制（可跳读）

> [!note]- Causal mask 负载不均衡与 Striped / zigzag 修复 —— 如果你要部署就展开
>
> 原论文 **没有** 讨论这一点 —— 算法是在一般 attention 下呈现的。但 causal mask 下连续 token 范围分块时:
>
> - GPU 0（最早 token）每一步都有差不多一半的 attention 是空的 —— 它的大部分 key 在未来。
> - GPU $N{-}1$（最晚 token）完全利用。
>
> 实现里跳过所有 $(q, k)$ 都被 mask 的环步,导致 **关键路径不均衡**,最坏 rank 决定整体。
>
> **Striped Attention**（[arXiv:2311.09431](https://arxiv.org/abs/2311.09431),同组 2023-11）用 **条带排列** 修复:token $t$ → GPU $(t \bmod N)$。每张卡持有早晚 token 的混合,causal 工作量均衡。报告增益:**A100 256K 1.45× 吞吐**、**16× TPU v4 786K 1.65×**。
>
> `zhuzilin/ring-flash-attention` 和 Megatron-Core CP 用一种 **zigzag** 方案做同样的均衡。2026 生产级 Ring Attention 总是配一种这样的修复。RoPE offset 在 striped/zigzag layout 下需要适配。

> [!note]- 推理 / 解码的别扭与 pass-Q 变体 —— 如果你服务长上下文就展开
>
> 自回归生成时持久 KV cache **不能** 自由旋转 —— cache 在自己 home rank 上,要在那里被 append。
>
> Meta 的 [arXiv:2411.01783](https://arxiv.org/abs/2411.01783) 提出 **pass-Q** 变体:解码时让 *query* token 绕环旋转,而不是 KV。这反转了通信模式,适合 KV 又大又钉死（解码）而不是 KV 与 Q 平衡（训练）的场景。报告数据:Llama 3 405B 在 128× H100 上 77 秒做 1M token prefill,93% 并行效率。

## 头条证据

**规模数据**。

| 模型 | 硬件 | 达到的 context | vs baseline |
| ---- | ---- | -------------- | ----------- |
| 7B | 32× A100 | **4M token** | ~32× 更长 |
| 7B | TPU v4-1024（1024 chip） | **8M token** | ~512× 更长 |
| （摘要声明） | 任意 | 最长 = 设备数 × 单卡 | — |

这些都是 *训练* 上下文长度,数学上与单卡 attention 等价 —— 没有近似。

**算力效率**。论文对比实际 MFU 与期望 MFU（通信完全藏住时的上界）:

![不同训练配置（7B / 13B / 30B / 65B）下的 MFU(论文 Fig. 5):Memory-Efficient Attn+FFN 单卡上限 vs Ring Attention 期望 vs 实际](CN/wiki/llm-inference/ring-attention-figs/mfu-expected-vs-actual.png)

所有测试配置下,Ring 实际 MFU 都在期望上限 ~1 个百分点之内 —— $c \geq F/B$ 重叠条件在 NVLink/ICI fabric 上确实成立的实证。

> [!success] 内存对比图才是这个技术买你数量级
> 前面那张最大 context 对比图给的是固定显存预算下 2–3 个数量级的可训练长度。这是真正的头条数字;上面的 MFU 图只是确认拿这个收益不需要付出算力税。

**关键消融:去掉 KV 旋转的重叠**。没有重叠时（块更小,违反 $c \geq F/B$）,吞吐按通信占比塌缩 —— 生产规模下一般 40–60%。上面的 MFU 图就是重叠在实际块大小下确实工作的证据。

> [!example]- 加速热力图与下游演示（展开）
>
> **加速热力图**。在模型规模 × context 长度共同增长下相对单卡 vanilla attention 的加速:
>
> ![相对单卡 vanilla baseline 的加速,横轴 context 长度(8K–100M),纵轴模型规模(7B–1TB)(论文 Fig. 6)](CN/wiki/llm-inference/ring-attention-figs/speedup-heatmap.png)
>
> 右下角的深蓝带是 Ring Attention 给你 $10^2$–$10^3 \times$ 加速的区间 —— 既因为 vanilla 在那里 OOM,也因为 Ring 的每卡计算保持有界。左上角小加速是 model-bound 而非 memory-bound 的区间。
>
> **下游生产示范:Large World Model (LWM)**。[Liu, Yan, Zaharia, Abbeel, *World Model on Million-Length Video And Language with Blockwise RingAttention*](https://arxiv.org/abs/2402.08268)（2024-02）。7B 模型从 4K 渐进训到 **1M token context**。技术最有说服力的端到端演示。
>
> **生产部署**。
>
> | 部署 | 变体 | 来源 |
> | ---- | ---- | ---- |
> | **Megatron-Core CP**（NVIDIA） | Ring + cuDNN FlashAttention + zigzag 调度;4D 并行（TP × CP × PP × DP） | [docs.nvidia.com](https://docs.nvidia.com/megatron-core/developer-guide/0.16.0/user-guide/features/context_parallel.html) |
> | **Meta Llama 3 训练** | "pass-KV"（经典 Ring）训练 + "pass-Q" 变体用于持久 KV cache 解码。Llama 3 405B 在 128× H100 上 77 秒做 1M token prefill,93% 并行效率,63% MFU | [arXiv:2411.01783](https://arxiv.org/abs/2411.01783) |
> | **Tencent USP** | Ring × [[deepspeed-ulysses\|Ulysses]] 混合 2D mesh;47% MFU 训 Llama-3-8B 在 2×8×A800 上 208K | [arXiv:2405.07719](https://arxiv.org/abs/2405.07719) |
> | **PyTorch 原生 CP** | Ring 为主,All-to-all 传输为次（PyTorch 2.7+） | [docs.pytorch.org](https://docs.pytorch.org/tutorials/unstable/context_parallel.html) |
>
> **与 [[deepspeed-ulysses|DeepSpeed Ulysses]] 对比**。
>
> | 性质 | Ring Attention | DeepSpeed Ulysses |
> | ---- | -------------- | ----------------- |
> | GPU 间动的是什么 | KV 块沿环旋转 | QKV 用 AllToAll 重排 |
> | 通信原语 | $N{-}1$ 轮 P2P send/recv | 每 attention 层 4× AllToAll |
> | 计算/通信重叠 | ✓ $c \geq F/B$ 时完全藏 | ✗ 阻塞 —— 无重叠 |
> | GPU 数硬上限 | 无 | $\leq$ num_heads（GQA 下更严） |
> | 跨节点扩展 | ✓ P2P 带宽友好 | ✗ AllToAll 在 IB 上掉性能 |
> | Causal mask 均衡 | ✗ 需要 Striped/zigzag 修复 | ✓ 天然均衡 |
> | Attention kernel 改动 | 是 —— 跟 FA streaming softmax 融合 | 无 —— 用标准 FlashAttention |

## 优势与限制

最强两点:(1) **跟单卡 attention 数学等价** —— 不近似、不折损质量;(2) **内存扩展性根本不同** —— 每卡 activation 内存与 $S$ 无关,序列长度随设备数线性扩展。

诚实承认的限制:

- **二次 FLOPs 没变**。总计算仍是 $O(S^2 \cdot d)$。Ring 是内存和调度的胜利,不是算法复杂度的胜利。10M 训练仍是算力预算限制。
- **Causal mask 不均衡** 是真的,原论文不解决。生产用 Striped 或 zigzag。
- **低带宽下吃通信**。重叠条件 $c \geq F/B$ 在 PCIe 或以太网上很难,块大小爆炸伤吞吐。NVLink 级别（或 TPU ICI）几乎是必要条件。
- **推理 / 解码很别扭**。自回归生成时持久 KV cache 不能随意旋转;Meta 的 pass-Q 变体就是为此设计的。
- **很多小矩阵乘伤算术强度**。相对 Ulysses（每卡上的 full-head attention）,Ring 每步的块更小,利用率更低。
- **作为构建块本身不新**。Online softmax 是 FlashAttention。Blockwise streaming 是 BPT。新的是 *分布式调度* —— 重要,但不是算法层面新。

> [!warning] "近无限上下文" —— marketing 还是实在?
> **内存** 上实在:每卡 activation 内存 $O(S/N)$,所以 $S$ 可以随 $N$ 扩。
> **实践** 上 marketing:总成本仍是 $O(S^2)$ FLOPs,10M token 训练仍然是算力预算约束,不是内存。LWM 1M 上下文 7B 是最有说服力的演示。任何"Gemini 1.5 的 10M context 用 Ring Attention"的说法都是 **未证实猜测** —— Google 没公开过。

## 这意味着什么

两条值得跟踪的预测:

1. **Ring Attention 已经赢了生产 CP 之战 —— 以不同的名字**。Megatron-Core CP、Llama 3 训练、Tencent USP、PyTorch 原生 CP —— 都是 Ring Attention 加工程修复（zigzag、混合、pass-Q）。裸 2023 算法不能直接用;2026 生产栈就是算法 + 三年工程。
2. **下一个轴是互联拓扑,不是算法**。Ring 在 NVLink 上工作;IB 上挣扎;PCIe 上失败。HBM 带宽继续涨而跨节点 fabric 涨得慢的话,$c^* = F/B$ 块大小成本会上升。未来工作会是让 Ring 容忍慢 fabric（pipelining、传输时 KV 压缩）,而不是重新发明调度。

这 *不是*:$O(S^2)$ FLOPs 的修复（任何方法都不是）,也不是裸形态下的推理 / 解码原语（Meta 的 pass-Q 是补丁）,也不是万能长上下文解药（Ulysses + Ring + KV 压缩各有角色）。

## 源码与复现

### 权威实现（JAX/TPU）

[lhao499/RingAttention](https://github.com/lhao499/RingAttention) —— Apache-2.0,参考实现。

| 文件 | 角色 |
| ---- | ---- |
| `ringattention/ringattention_jax.py` | 核心 GPU/TPU forward kernel（前述 ~20 行核心代码） |
| `ringattention/ringattention_jax_inference.py` | 推理变体 |
| `ringattention/ringattention_pallas_tpu.py` | Pallas (TPU) kernel |

环旋转原语（JAX）:

```python
k, v = map(lambda x: lax.ppermute(
    x, axis_name,
    perm=[(i, (i + 1) % axis_size) for i in range(axis_size)]
), (k, v))
```

`lax.ppermute` 是 SPMD 感知的 P2P send/recv 沿设备 mesh 轴。

### PyTorch 移植

| 仓库 | 备注 |
| ---- | ---- |
| **[zhuzilin/ring-flash-attention](https://github.com/zhuzilin/ring-flash-attention)** | 最常用 PyTorch 实现。把 Tri Dao 的 FlashAttention 包成每步的 inner kernel。变体:`ring_flash_attn_func`、`zigzag_ring_flash_attn_func`、`stripe_flash_attn_func`,以及 `_varlen` / `_qkvpacked` 版本。H800 上 zigzag fwd+bwd 达单卡 FlashAttention 吞吐的 ~90%。需 NVLink。 |
| [lucidrains/ring-attention-pytorch](https://github.com/lucidrains/ring-attention-pytorch) | Phil Wang 的教学性 PyTorch 移植。 |
| [gpu-mode/ring-attention](https://github.com/gpu-mode/ring-attention) | GPU MODE 社区脚手架（Lecture 13）。 |

### 生产级集成:Megatron-Core CP

NVIDIA 产品化的 Ring Attention。用 `--context-parallel-size <N>` flag 暴露。组合成 4D 并行 $\text{TP} \times \text{CP} \times \text{PP} \times \text{DP}$。用 cuDNN FlashAttention 作 inner kernel,zigzag 调度处理 causal 均衡。[文档](https://docs.nvidia.com/megatron-core/developer-guide/0.16.0/user-guide/features/context_parallel.html)。

### 最小复现（PyTorch,zhuzilin/ring-flash-attention）

```python
import torch
from ring_flash_attn import zigzag_ring_flash_attn_func

# 在大小为 N 的 torch.distributed 进程组里:
# q, k, v 形状 (batch, seq_len/N, num_heads, head_dim) —— 序列已分片
out = zigzag_ring_flash_attn_func(q, k, v, causal=True)
# out 与 q 同形
```

库内部处理环旋转（NCCL `batch_isend_irecv`）、每步 FlashAttention 调用、online softmax 累计、causal 均衡的 zigzag 分块。RoPE offset 在 striped/zigzag layout 下需要适配（见 README）。

## 相关阅读

- [[deepspeed-ulysses]] —— AllToAll 路线的替代方案。本页姐妹。
- [[parallelism-strategies-deep-dive#7. CP — 上下文并行 (Context Parallelism)]] —— Ring Attention 在并行图景里的位置;与 Ulysses、Megatron CP 的对比表。
- [[paged-attention]] —— FlashAttention 的单卡 streaming kernel;Ring Attention 每步的 inner loop。
- [[kv-cache-optimization]] —— KV cache 压缩与长上下文交叉的地方。
- [[long-context-serving]] —— 生产长上下文推理;Ring 是训练侧,[[saw-int4|SAW-INT4]] / 量化是 serving 侧对偶。
- [[das-spec-rl]] —— rollout 阶段的投机解码;与 Ring 正交但推理层互补。

## 参考文献

- **Ring Attention 论文**:Liu, Zaharia, Abbeel. *Ring Attention with Blockwise Transformers for Near-Infinite Context*. ICLR 2024. [arXiv:2310.01889](https://arxiv.org/abs/2310.01889) · [OpenReview](https://openreview.net/forum?id=WsRHpHH4s0) · [ICLR proceedings PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/1119587863e78451f080da2a768c4935-Paper-Conference.pdf)
- **Blockwise Parallel Transformer**（前置）:Liu, Abbeel. [arXiv:2305.19370](https://arxiv.org/abs/2305.19370)
- **Striped Attention**（causal mask 修复）:Brandon, Nrusimha, Qian, Ankner, Jin, Song, Liu, Ragan-Kelley. [arXiv:2311.09431](https://arxiv.org/abs/2311.09431)
- **Large World Model**（Ring Attention 1M context 应用）:Liu, Yan, Zaharia, Abbeel. [arXiv:2402.08268](https://arxiv.org/abs/2402.08268)
- **BurstAttention**（双 buffer ring 变体）:[arXiv:2403.09347](https://arxiv.org/pdf/2403.09347)
- **USP (Ring × Ulysses 混合)**:[arXiv:2405.07719](https://arxiv.org/abs/2405.07719) · [feifeibear/long-context-attention](https://github.com/feifeibear/long-context-attention)
- **Meta Context Parallelism (pass-KV / pass-Q)**:Yang et al. [arXiv:2411.01783](https://arxiv.org/abs/2411.01783)
- **TokenRing**（双向环）:[arXiv:2412.20501](https://arxiv.org/abs/2412.20501)
- **Megatron-Core CP docs**:[docs.nvidia.com/megatron-core/.../context_parallel.html](https://docs.nvidia.com/megatron-core/developer-guide/0.16.0/user-guide/features/context_parallel.html)
- **解读文章**:[Coconut Mode walkthrough](https://coconut-mode.com/posts/ring-attention/) · [GPU MODE Lecture 13](https://christianjmills.com/posts/cuda-mode-notes/lecture-013/) · [Insujang CP overview](https://insujang.github.io/2024-09-20/introducing-context-parallelism/)
- **DeepSpeed Ulysses**（替代方案）:[arXiv:2309.14509](https://arxiv.org/abs/2309.14509) —— 见 [[deepspeed-ulysses]]
- **FlashAttention-3**（inner kernel）:[arXiv:2407.08608](https://arxiv.org/abs/2407.08608) · [Tri Dao 博客](https://tridao.me/blog/2024/flash3/)
