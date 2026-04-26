---
title: "投机解码：草稿-验证加速"
category: llm-inference
tags: [投机解码, 草稿模型, eagle, medusa, 无损加速]
created: 2026-04-13
updated: 2026-04-14
status: mature
---

# 投机解码：草稿-验证加速

## 概述

自回归语言模型生成文本时，每一步只能产出一个 token，而每步都需要从 GPU 显存加载整个模型权重。对于大模型（如 70B 参数），单 token 生成已严重受限于 **内存带宽**（memory-bandwidth-bound），而非计算瓶颈。这意味着 GPU 的大量算力在解码阶段被浪费了。

**投机解码（Speculative Decoding）** 是一种在不改变输出分布的前提下加速自回归生成的方法。其核心思想来自一个简单观察：

> 验证 K 个 token 的正确性（一次前向传播）与生成 1 个 token 的成本几乎相同。

因此，如果能用一个轻量级的"草稿模型"快速猜测未来 K 个 token，再用完整的目标模型在一次前向传播中验证这些猜测，就能在单步中产出多个 token。由于验证过程使用了严格的拒绝采样，**输出分布与目标模型数学上完全一致** -- 这是无损加速。

```
传统自回归解码（每步 1 token）：
  步骤 1: [The] → 目标模型 → [cat]
  步骤 2: [The cat] → 目标模型 → [sat]
  步骤 3: [The cat sat] → 目标模型 → [on]
  步骤 4: [The cat sat on] → 目标模型 → [the]
  步骤 5: [The cat sat on the] → 目标模型 → [mat]
  总计: 5 次前向传播 → 5 个 token

投机解码（每步可能多个 token）：
  草稿模型快速猜: [cat, sat, on, the, mat]
  目标模型一次验证: 全部接受!
  总计: 1 次草稿 + 1 次验证 → 5 个 token
```

投机解码最早由 Leviathan et al. (2023) 和 Chen et al. (2023) 独立提出，现已成为生产推理系统的标配优化。


## 核心原理

### 草稿-验证范式

投机解码的工作流程分为三个阶段：

```
┌─────────────────────────────────────────────────────────────┐
│                     投机解码流程                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  阶段 1: 草稿生成                                            │
│  ┌───────────┐                                              │
│  │ 草稿模型   │──→ t1, t2, t3, ..., tK  (K 个候选 token)    │
│  │ (轻量级)   │     (快速，低质量)                            │
│  └───────────┘                                              │
│        │                                                    │
│        ▼                                                    │
│  阶段 2: 并行验证                                            │
│  ┌───────────┐                                              │
│  │ 目标模型   │──→ 对 [prefix, t1, t2, ..., tK] 做          │
│  │ (完整模型) │     单次前向传播，得到每个位置的概率分布         │
│  └───────────┘                                              │
│        │                                                    │
│        ▼                                                    │
│  阶段 3: 接受/拒绝                                           │
│  ┌───────────────────────────────────────┐                  │
│  │ 逐个比较:                              │                  │
│  │   t1: P_target(t1) / P_draft(t1) ≥ r? │                  │
│  │     → 接受, 继续检查 t2                 │                  │
│  │   t2: P_target(t2) / P_draft(t2) ≥ r? │                  │
│  │     → 接受, 继续检查 t3                 │                  │
│  │   t3: P_target(t3) / P_draft(t3) < r? │                  │
│  │     → 拒绝! 从修正分布中重新采样 t3'    │                  │
│  │   输出: t1, t2, t3' (本步产出 3 token)  │                  │
│  └───────────────────────────────────────┘                  │
│                                                             │
│  循环: 用新序列继续下一轮草稿-验证                             │
└─────────────────────────────────────────────────────────────┘
```

### 为什么验证 K 个 token 的成本约等于生成 1 个？

这与 LLM 推理的计算特性有关：

- **Prefill 阶段**（处理输入 prompt）是 **计算受限**的：batch 矩阵乘法可以充分利用 GPU 算力
- **Decode 阶段**（逐 token 生成）是 **内存带宽受限**的：每次只做一个 token 的矩阵-向量乘法，GPU 大部分时间在等待数据从显存加载

验证 K 个候选 token 本质上类似于一次小规模 prefill：可以将 K 个 token 组成一个小 batch 并行处理。当 K 较小（如 5-10）时，计算量的增加几乎不影响总耗时，因为瓶颈在于加载模型权重，而权重只需加载一次。

### 无损保证的数学基础

投机解码的关键性质是 **无损性**：最终输出的 token 分布与直接使用目标模型生成完全一致。这通过 **修正拒绝采样（Modified Rejection Sampling）** 实现。


## 验证算法

### 标准拒绝采样回顾

经典拒绝采样用于从目标分布 p(x) 采样，当 p(x) 难以直接采样但可以计算时，借助一个容易采样的提议分布 q(x)：

1. 从 q(x) 采样 x
2. 计算接受概率 α = p(x) / (M * q(x))，其中 M = max_x p(x)/q(x)
3. 以概率 α 接受 x，否则拒绝并重试

### 投机解码的修正拒绝采样

投机解码使用了一种改进版本。设：
- `p(x)` = 目标模型在当前位置的概率分布
- `q(x)` = 草稿模型在当前位置的概率分布

对于草稿 token `x ~ q`：

```
接受概率 = min(1, p(x) / q(x))
```

**关键步骤**：

1. 从草稿模型采样 token x，概率为 q(x)
2. 计算比率 r = p(x) / q(x)
3. 若 r >= 1（目标模型比草稿模型更可能生成 x），**确定接受**
4. 若 r < 1，以概率 r 接受，以概率 1-r 拒绝
5. 若拒绝，从 **修正分布** 重新采样：

```
p'(x) = max(0, p(x) - q(x)) / Σ_x max(0, p(x) - q(x))
```

**为什么这保证了无损性？**

实际被接受的 token 的分布为：

```
P(accept x) = q(x) * min(1, p(x)/q(x))
             = min(q(x), p(x))
```

拒绝后从修正分布 `p'(x) ∝ max(0, p(x) - q(x))` 采样。两部分合起来：

```
P(output x) = min(q(x), p(x)) + [1 - Σ_x min(q(x), p(x))] * p'(x)
            = min(q(x), p(x)) + max(0, p(x) - q(x))
            = p(x)
```

这就证明了输出分布恒等于目标模型分布 p(x)。

### 多 Token 验证流程

对于 K 个草稿 token (t1, t2, ..., tK) 的验证：

```python
def speculative_verify(draft_tokens, draft_probs, target_probs):
    """
    draft_tokens: [t1, t2, ..., tK] 草稿 token 序列
    draft_probs:  [q1, q2, ..., qK] 草稿模型在各位置的分布
    target_probs: [p1, p2, ..., pK, pK+1] 目标模型在各位置的分布
    """
    accepted = []
    for i in range(K):
        ti = draft_tokens[i]
        r = random.uniform(0, 1)
        
        if r < min(1, target_probs[i][ti] / draft_probs[i][ti]):
            # 接受 token ti
            accepted.append(ti)
        else:
            # 拒绝 token ti, 从修正分布重新采样
            residual = np.maximum(0, target_probs[i] - draft_probs[i])
            residual /= residual.sum()
            new_token = np.random.choice(vocab, p=residual)
            accepted.append(new_token)
            return accepted  # 后续 token 全部丢弃
    
    # 所有 K 个 token 都被接受! 额外从 pK+1 采样一个
    bonus_token = np.random.choice(vocab, p=target_probs[K])
    accepted.append(bonus_token)
    return accepted  # 产出 K+1 个 token
```

### 接受率分析

**接受率 alpha** 定义为草稿 token 被接受的概率。它决定了投机解码的加速效果：

```
alpha = E_x~q [min(1, p(x)/q(x))]
      = Σ_x min(p(x), q(x))
      = 1 - (1/2) * Σ_x |p(x) - q(x)|    （与 TV 距离的关系）
```

即接受率 = 1 - TV_distance(p, q)。草稿模型与目标模型的分布越接近，接受率越高。

**每步期望产出 token 数**（设草稿长度为 gamma）：

```
E[tokens_per_step] = (1 - alpha^(gamma+1)) / (1 - alpha)
```

| alpha (接受率) | gamma=5 | 期望 token/步 |
|---------------|---------|--------------|
| 0.5 | 5 | ~1.97 |
| 0.6 | 5 | ~2.37 |
| 0.7 | 5 | ~2.94 |
| 0.8 | 5 | ~3.78 |
| 0.9 | 5 | ~5.22 |
| 0.9 | 8 | ~6.13 |

可以看出，接受率从 0.7 提升到 0.9，每步产出几乎翻倍。这就是为什么草稿质量至关重要。


## 草稿模型选择

不同的草稿策略在质量、速度、内存开销之间做出不同权衡。

### 独立草稿模型

最直观的方案：使用一个同架构但更小的模型作为草稿模型。

- **示例**：用 Llama-3-8B 作为 Llama-3-70B 的草稿模型
- **优点**：实现简单，草稿质量较高
- **缺点**：需要额外显存加载草稿模型；需要两个模型共享词表

### 自投机解码（Self-Speculative / LayerSkip）

用目标模型自身的一部分作为草稿模型，通过跳过部分层来加速草稿生成。

- **原理**：跳过中间层（如 32 层模型只用前 8 层 + 最后 4 层作草稿）
- **优点**：无需额外模型，零显存开销
- **缺点**：接受率通常较低（0.5-0.7），因为浅层模型质量有限
- **代表工作**：LayerSkip (Meta, 2024)

### EAGLE 系列（当前 SOTA）

EAGLE（Extrapolation Algorithm for Greater Language-model Efficiency）是目前最高效的投机解码方法。

#### EAGLE (ICML 2024)

核心发现：在 **特征层面**（倒数第二层隐藏状态）进行自回归预测比在 token 层面更容易。

```
传统草稿模型:   token → embedding → ... → logits → token → ...
EAGLE:         feature(t) → lightweight_head → feature(t+1) → ...
```

- 训练一个轻量级网络（通常 1-2 层 Transformer），输入当前 token 的特征，预测下一个 token 的特征
- 特征空间的自回归性比 token 空间更强（更可预测）
- 草稿开销极低，接受率高

#### EAGLE-2 (EMNLP 2024)

在 EAGLE 基础上引入 **动态草稿树**：

- 根据每个候选 token 的置信度动态调整树结构
- 高置信度的分支展开更多，低置信度的提前剪枝
- 自适应地分配"验证预算"

#### EAGLE-3 (NeurIPS 2025)

最新版本，三大改进：

1. **多层特征融合**：不仅用倒数第二层，还融合低层和中间层特征
2. **训练时测试（Training-time Test）**：在训练中模拟推理时的自回归误差累积
3. **更优的树结构**

性能：**3.0-6.5 倍加速**，比 EAGLE-2 提升 20-40%。

论文：[arXiv:2503.01840](https://arxiv.org/abs/2503.01840)

### Medusa (ICML 2024)

```
                       ┌──→ Head 1 → 预测位置 +1 的 token
                       │
目标模型最后一层 ──────┼──→ Head 2 → 预测位置 +2 的 token
  隐藏状态              │
                       ├──→ Head 3 → 预测位置 +3 的 token
                       │
                       └──→ Head K → 预测位置 +K 的 token
```

- 在目标模型顶部添加 K 个独立的前馈预测头
- 每个头直接从当前隐藏状态预测未来第 k 个 token（非自回归）
- 使用 **树注意力（Tree Attention）** 高效验证多个候选组合
- 加速：2.2-3.6 倍
- **优点**：只需训练几个小 head，参数量极少
- **缺点**：非自回归预测质量不如 EAGLE 的特征级自回归

论文：[arXiv:2401.10774](https://arxiv.org/abs/2401.10774)

### Lookahead Decoding

- 利用 Jacobi 迭代的思想，并行猜测多个位置的 token
- 每次前向传播同时更新多个位置
- 无需训练额外模型
- 实际加速有限（1.5-2x），但实现简单

### Prompt Lookup / N-gram 匹配

- 在已有的上下文（prompt）中查找 n-gram 匹配
- 如果当前生成的后缀在 prompt 中出现过，直接复用后续 token 作为草稿
- **零开销**：不需要任何额外模型或计算
- 非常适合代码补全、摘要等输出与输入高度重叠的场景
- 对创造性生成几乎无效

### 方法对比

| 方法 | 接受率 | 加速 | 额外显存 | 需要训练 | 适用场景 |
|------|--------|------|----------|----------|----------|
| 独立草稿模型 | 0.6-0.8 | 1.5-2.5x | 高 | 否 | 通用 |
| LayerSkip | 0.5-0.7 | 1.3-1.8x | 零 | 否 | 显存受限 |
| EAGLE | 0.7-0.85 | 2.0-3.5x | 低 | 是 | 通用（最推荐） |
| EAGLE-2 | 0.75-0.9 | 2.5-4.5x | 低 | 是 | 通用 |
| EAGLE-3 | 0.8-0.95 | 3.0-6.5x | 低 | 是 | 通用（SOTA） |
| Medusa | 0.6-0.8 | 2.2-3.6x | 极低 | 是 | 显存紧张 |
| Lookahead | N/A | 1.5-2.0x | 零 | 否 | 快速部署 |
| N-gram | 变化大 | 1.0-3.0x | 零 | 否 | 代码/摘要 |


## Token 树验证

### 为什么用树而不用链？

线性草稿（链式）每次只生成一条候选序列。如果第 2 个 token 被拒绝，后面所有 token 都浪费了。

**树结构推测** 允许在每个位置探索多个候选，形成一棵候选树：

```
                        [The]
                       /  |  \
                    [cat] [dog] [big]
                    / \      |
               [sat] [is]  [ran]
                |     |      |
              [on]  [very] [fast]
```

这样即使某个分支被拒绝，其他分支仍可能被接受。

### 树验证的高效实现

关键问题：如何在一次前向传播中验证整棵树？

答案是使用 **因果注意力掩码（Causal Attention Mask）** 的变体：

```
标准因果掩码（链式）：        树形注意力掩码：
  t1 t2 t3 t4                  t1 t2 t3 t4 t5 t6 t7
  1  0  0  0  t1               1  0  0  0  0  0  0  t1 (root)
  1  1  0  0  t2               1  1  0  0  0  0  0  t2 (child of t1)
  1  1  1  0  t3               1  0  1  0  0  0  0  t3 (child of t1)
  1  1  1  1  t4               1  1  0  1  0  0  0  t4 (child of t2)
                                1  1  0  0  1  0  0  t5 (child of t2)
                                1  0  1  0  0  1  0  t6 (child of t3)
                                1  0  1  0  0  0  1  t7 (child of t3)
```

每个节点只能看到从根到自身路径上的 token，这通过自定义注意力掩码实现。

### 树的构建策略

不同方法使用不同的树构建策略：

1. **静态树**：预定义固定的树结构（如 top-k 扩展）
2. **动态树（EAGLE-2/3）**：根据置信度分数动态决定哪些分支展开
3. **Medusa 树**：使用笛卡尔积构建候选组合，用 tree attention 验证

**动态树的优势**：

```
高置信度序列 → 展开更深（更多 token）
             "The capital of France is" → [Paris] → [.] → 深度 3

低置信度序列 → 提前停止
             "The meaning of life is" → [a/the/to/...] → 宽度优先，深度 1
```


## 代码示例

### 基础投机解码伪代码

```python
import torch
import numpy as np

def speculative_decode(
    target_model,
    draft_model,
    input_ids,
    max_tokens=100,
    gamma=5,          # 草稿长度
    temperature=1.0,
):
    """基础投机解码实现"""
    generated = list(input_ids)
    
    while len(generated) - len(input_ids) < max_tokens:
        prefix = torch.tensor([generated])
        
        # ---- 阶段 1: 草稿生成 ----
        draft_tokens = []
        draft_probs = []
        draft_input = prefix.clone()
        
        for _ in range(gamma):
            with torch.no_grad():
                logits = draft_model(draft_input).logits[:, -1, :]
                probs = torch.softmax(logits / temperature, dim=-1)
                token = torch.multinomial(probs, 1)
                draft_tokens.append(token.item())
                draft_probs.append(probs[0].cpu().numpy())
                draft_input = torch.cat([draft_input, token], dim=-1)
        
        # ---- 阶段 2: 目标模型验证 ----
        # 将 prefix + 所有草稿 token 一次送入目标模型
        verify_input = torch.tensor([generated + draft_tokens])
        with torch.no_grad():
            target_logits = target_model(verify_input).logits
        
        # 提取目标模型在各位置的概率分布
        # 位置: len(generated)-1 到 len(generated)+gamma-1
        start_pos = len(generated) - 1
        target_probs = []
        for i in range(gamma + 1):
            p = torch.softmax(
                target_logits[:, start_pos + i, :] / temperature, dim=-1
            )
            target_probs.append(p[0].cpu().numpy())
        
        # ---- 阶段 3: 接受/拒绝 ----
        n_accepted = 0
        for i in range(gamma):
            ti = draft_tokens[i]
            p_target = target_probs[i][ti]
            p_draft = draft_probs[i][ti]
            
            r = np.random.uniform()
            if r < min(1.0, p_target / p_draft):
                generated.append(ti)
                n_accepted += 1
            else:
                # 从修正分布采样
                residual = np.maximum(0, target_probs[i] - draft_probs[i])
                residual /= residual.sum()
                new_token = np.random.choice(len(residual), p=residual)
                generated.append(new_token)
                break
        else:
            # 所有 gamma 个 token 都接受, 额外采样一个
            bonus = np.random.choice(
                len(target_probs[gamma]), p=target_probs[gamma]
            )
            generated.append(bonus)
    
    return generated
```

### vLLM 投机解码配置

```python
from vllm import LLM, SamplingParams

# ---- 方式 1: 独立草稿模型 ----
llm = LLM(
    model="meta-llama/Llama-3.3-70B-Instruct",
    speculative_model="meta-llama/Llama-3.2-1B-Instruct",
    num_speculative_tokens=5,
    tensor_parallel_size=4,
    # 草稿模型也可以设置 tensor parallel
    speculative_model_tensor_parallel_size=1,
)

# ---- 方式 2: EAGLE 草稿 ----
llm = LLM(
    model="meta-llama/Llama-3.3-70B-Instruct",
    speculative_model="yuhuili/EAGLE3-LLaMA3.3-70B-Instruct",
    speculative_method="eagle",
    num_speculative_tokens=5,
    tensor_parallel_size=4,
)

# ---- 方式 3: N-gram 匹配（无需额外模型）----
llm = LLM(
    model="meta-llama/Llama-3.3-70B-Instruct",
    speculative_model="[ngram]",
    num_speculative_tokens=5,
    ngram_prompt_lookup_max=4,
    ngram_prompt_lookup_min=2,
)

# 正常使用
params = SamplingParams(temperature=0.7, max_tokens=512)
outputs = llm.generate(["Explain quantum computing"], params)
```

### 接受率监控

```python
# vLLM 内置指标
# Prometheus endpoint: /metrics
# 关键指标:
#   vllm:spec_decode_draft_acceptance_rate  - 草稿 token 接受率
#   vllm:spec_decode_efficiency             - 每步平均产出 token 数
#   vllm:num_spec_tokens                    - 每步草稿 token 数

# SGLang 监控
# 启动参数加 --enable-metrics
# 指标:
#   sglang:spec_accept_length_mean
#   sglang:spec_accept_length_histogram

# 实践建议:
# - 接受率 < 0.5: 考虑更换草稿模型或减少 gamma
# - 接受率 0.5-0.7: 正常范围, 可尝试优化
# - 接受率 > 0.8: 可以增加 gamma 获取更大加速
# - 监控不同 prompt 类型的接受率变化
```


## 性能分析

### 加速公式

投机解码的实际加速取决于多个因素：

```
                    E[accepted_tokens_per_step]
Speedup ≈ ──────────────────────────────────────────
           1 + (draft_cost / target_verify_cost)
```

其中：
- `E[accepted_tokens_per_step]` = (1 - alpha^(gamma+1)) / (1 - alpha)
- `draft_cost` = 草稿模型生成 gamma 个 token 的时间
- `target_verify_cost` = 目标模型验证一次的时间

**理想情况**（草稿成本忽略不计）：加速 ≈ E[accepted_tokens]

**实际情况**：草稿模型通常占目标模型验证时间的 5-20%

### 不同场景的基准测试

| 配置 | 模型 | 草稿方法 | 并发 | 加速 |
|------|------|----------|------|------|
| 4xA100 | Llama-3.3-70B | EAGLE-3 | 1 | 3.2x |
| 4xA100 | Llama-3.3-70B | EAGLE-3 | 4 | 2.5x |
| 4xA100 | Llama-3.3-70B | EAGLE-3 | 16 | 1.4x |
| 4xA100 | Llama-3.3-70B | 独立 8B | 1 | 2.1x |
| 1xA100 | Llama-3-8B | EAGLE-2 | 1 | 2.8x |
| 1xA100 | Llama-3-8B | Medusa | 1 | 2.3x |
| 1xH100 | Llama-3.3-70B | EAGLE-3 + FP8 | 1 | 3.8x |

### 什么时候投机解码有帮助 vs 无帮助？

**有帮助的场景**：

| 条件 | 原因 |
|------|------|
| 低并发（batch <=10） | 解码阶段内存带宽受限最严重 |
| 大模型（>=13B） | 模型越大，前向传播越慢，节省越多 |
| 可预测输出（代码/格式化/翻译） | 接受率高 |
| 延迟敏感应用 | 直接降低单请求延迟 |

**无帮助/反而变慢**：

| 条件 | 原因 |
|------|------|
| 高并发（batch 32+） | decode 变成 compute-bound，验证成本不再"免费" |
| Prefill 占主导 | 投机解码只加速 decode 阶段 |
| 极短生成（<50 token） | 草稿开销占比过大 |
| 创意写作/开放问答 | 接受率低，草稿频繁被拒绝 |
| 草稿模型不匹配 | 接受率过低导致负优化 |

### 与批处理的交互

投机解码与 [[continuous-batching]] 的结合需要注意：

- 不同请求的草稿可能接受不同数量的 token，导致序列长度不齐
- 验证步骤中，batch 内每个请求的有效长度不同
- 实际系统（如 [[vllm]]、[[sglang]]）使用 padding 或分桶来处理这种不规则性
- 高并发时，这种不规则性的开销会削弱投机解码的收益


## 不足与局限

1. **草稿质量决定一切**：接受率低于 0.5 时几乎没有加速，甚至可能变慢。草稿模型需要与目标模型在目标任务上分布接近。

2. **内存开销**：独立草稿模型方案需要同时在 GPU 上加载两个模型。对于 70B 目标模型 + 8B 草稿模型，额外需要约 16GB 显存（FP16）。EAGLE/Medusa 头的开销较小（<1GB）。

3. **高并发场景收益递减**：当 batch size 增大，decode 阶段从 memory-bound 转向 compute-bound，验证 K 个 token 不再"免费"。这是投机解码在生产高吞吐场景中的主要限制。

4. **对 prefill 无帮助**：投机解码只加速 decode 阶段。对于长输入 prompt 的处理（prefill），需要其他优化如 [[prefill-decode-disaggregation]]。

5. **实现复杂性**：树注意力掩码、KV 缓存管理、与连续批处理的集成都增加了系统复杂度。

6. **温度依赖**：高温度（更随机）采样时接受率下降，因为草稿模型更难预测目标分布。贪婪解码（temperature=0）时接受率最高。

7. **域外草稿**：如果草稿模型没有在目标任务的数据上充分训练（或者目标模型经过了特定微调），接受率会很低。需要为特定模型训练专用的 EAGLE/Medusa 头。


## 参考文献

- Leviathan et al., "Fast Inference from Transformers via Speculative Decoding," ICML 2023. [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)
- Chen et al., "Accelerating Large Language Model Decoding with Speculative Sampling," 2023. [arXiv:2302.01318](https://arxiv.org/abs/2302.01318)
- Li et al., "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty," ICML 2024. [arXiv:2401.15077](https://arxiv.org/abs/2401.15077)
- Li et al., "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees," EMNLP 2024. [arXiv:2406.16858](https://arxiv.org/abs/2406.16858)
- Li et al., "EAGLE-3: Scaling Up Speculative Decoding with Feature Fusion and Training-time Test," NeurIPS 2025. [arXiv:2503.01840](https://arxiv.org/abs/2503.01840)
- Cai et al., "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads," ICML 2024. [arXiv:2401.10774](https://arxiv.org/abs/2401.10774)
- Fu et al., "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding," 2024. [arXiv:2402.02057](https://arxiv.org/abs/2402.02057)
- Elhoushi et al., "LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding," 2024. [arXiv:2404.16710](https://arxiv.org/abs/2404.16710)


## 相关页面

- [[vllm]] -- 支持多种投机解码策略
- [[sglang]] -- EAGLE-3 深度集成
- [[continuous-batching]] -- 与投机解码的调度交互
- [[kv-cache-optimization]] -- 草稿模型的 KV 缓存管理
- [[quantization]] -- 量化可与投机解码结合使用
- [[prefill-decode-disaggregation]] -- 投机解码只加速 decode，prefill 需要其他优化
- [[tensorrt-llm]] -- 支持 EAGLE-3 和 N-gram 投机解码
