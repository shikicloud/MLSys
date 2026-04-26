---
title: "多步推理的强化学习"
category: agentic-rl
tags: [推理, 思维链, prm, orm, mcts, deepseek-r1, o1, o3, grpo, star, rest]
created: 2026-04-13
updated: 2026-04-14
status: mature
---

# 多步推理的强化学习

## 概述

多步推理 RL（Multi-Step Reasoning RL）通过强化学习训练 LLM 生成扩展的思维链（Chain-of-Thought, CoT）来解决复杂问题。这是 OpenAI o1/o3 和 [[grpo#DeepSeek-R1|DeepSeek-R1]] 等推理模型背后的核心技术。

**核心洞察**：RL 可以激励模型自发发展出复杂的推理策略——自我验证、回溯、分解、探索——而无需通过示范显式教授这些模式。当模型因"正确推理 → 正确答案"获得奖励时，它会自主学习如何更有效地思考。

### 推理 RL 与普通 RL 的关系

推理 RL 是 [[agentic-rl-overview|智能体 RL]] 的一个重要子方向。虽然它可以独立于工具使用（纯文本推理），但在实际系统中，推理能力和工具使用能力通常是协同训练的（如 ReTool 中模型边推理边决定是否调用代码解释器）。

```
推理 RL 的独特之处：
- 动作空间：主要是自然语言文本（思维链）
- 奖励信号：通常来自可验证的答案（数学、代码、逻辑）
- 核心挑战：如何奖励"过程"而不仅仅是"结果"
- 涌现现象：复杂推理策略从简单奖励信号中涌现
```

### 训练循环

推理 RL 的基本训练循环：

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│  1. 问题采样                                         │
│     从数学/代码/科学/逻辑题库中采样问题 q             │
│                                                     │
│  2. 轨迹生成（Rollout）                              │
│     模型 π_θ 生成推理轨迹：                           │
│     q → Think₁ → Think₂ → ... → ThinkN → Answer     │
│                                                     │
│  3. 验证与奖励                                       │
│     提取最终答案，与标准答案比对：                     │
│     R = 1 (正确) 或 R = 0 (错误)                     │
│                                                     │
│  4. 策略更新                                         │
│     增加导致正确答案的轨迹概率                        │
│     减少导致错误答案的轨迹概率                        │
│     （使用 GRPO/PPO/REINFORCE 等算法）                │
│                                                     │
│  5. 重复                                             │
│     回到步骤 1                                       │
│                                                     │
└─────────────────────────────────────────────────────┘
```

## DeepSeek-R1 案例研究

### 概述

DeepSeek-R1 是推理 RL 领域最重要的开源工作之一，首次大规模展示了纯 RL 训练可以让模型涌现出复杂的推理行为。

**论文**：[DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)（2025 年 1 月）

### 纯 RL 训练涌现 CoT

DeepSeek-R1 的最关键发现：

> 在 DeepSeek-V3-Base（未经过 SFT 的基座模型）上直接进行 RL 训练，模型自发涌现出了链式思维推理行为。

这意味着 CoT 不需要通过示范来教——RL 的奖励信号本身就足以激励模型发展出分步推理的能力。

**训练设置**：
- 基座模型：DeepSeek-V3-Base（671B MoE，37B 激活参数）
- 算法：[[grpo|GRPO]]（Group Relative Policy Optimization）
- 奖励：规则化验证器（数学答案格式匹配 + 计算验证）
- 无 SFT 冷启动

**涌现的推理行为**：
1. **自我验证**：模型会在得出答案后自行检查
2. **回溯**：发现推理错误时会主动回到之前的步骤
3. **分解**：将复杂问题分解为更小的子问题
4. **多角度分析**：从不同角度尝试解题
5. **逐步精确化**：从粗略估计到精确计算

### "顿悟时刻"（Aha Moment）

DeepSeek-R1 论文中报告了一个引人注目的现象——在训练过程中，模型突然学会了自我反思：

```
训练早期（模型行为）：
  "The answer is 42. Wait, let me recalculate... The answer is 42."
  （形式化的"检查"但实际没有真正验证）

训练中期（顿悟时刻）：
  "The answer is 42. Hmm, wait. Let me re-examine step 3.
   Actually, I made an error: 7 × 8 = 56, not 54.
   Correcting this... the real answer is 44."
  （真正的错误发现和修正！）

训练后期（成熟推理）：
  "Let me approach this from two directions to verify.
   Method 1: ... → 44
   Method 2: ... → 44
   Both methods agree. The answer is 44."
  （多方法交叉验证）
```

这个"顿悟时刻"是 RL 训练的涌现行为——模型发现"发现并修正错误"这一策略能获得更高的奖励，于是强化了这种行为。

### GRPO 用于推理优化

DeepSeek-R1 使用 GRPO（而非 PPO）作为核心 RL 算法：

**GRPO 的关键思想**：不需要单独的价值模型（critic），而是通过组内比较来估计优势：

$$\hat{A}_i = \frac{R_i - \text{mean}(\{R_j\}_{j=1}^G)}{\text{std}(\{R_j\}_{j=1}^G)}$$

对于每个问题，采样 $G$ 条推理轨迹，计算每条轨迹的奖励 $R_i$，然后在组内归一化。

**GRPO vs PPO 对比**：

| 特性 | PPO | GRPO |
|------|-----|------|
| 需要价值模型？ | 是（额外 ~50% 显存） | 否 |
| 优势估计方式 | GAE（需要 critic） | 组内归一化 |
| 显存效率 | 低 | 高（无 critic） |
| 适合长序列？ | 困难（critic 难以评估长序列） | 更适合 |
| 偏差 | 较低（有 critic 矫正） | 有偏（但实践中表现好） |

**GRPO 更新公式**：

$$\mathcal{L}_{\text{GRPO}} = -\frac{1}{G} \sum_{i=1}^{G} \min\left(\frac{\pi_\theta(y_i|x)}{\pi_{\text{old}}(y_i|x)} \hat{A}_i, \text{clip}\left(\frac{\pi_\theta(y_i|x)}{\pi_{\text{old}}(y_i|x)}, 1-\epsilon, 1+\epsilon\right) \hat{A}_i\right) + \beta \cdot D_{KL}(\pi_\theta \| \pi_{\text{ref}})$$

### 两阶段训练流程

DeepSeek-R1 的完整训练流程比"纯 RL"更复杂，包含四个阶段：

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Stage 1: 冷启动（Cold Start）                               │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ DeepSeek-V3-Base                                      │  │
│  │     ↓                                                 │  │
│  │ 少量高质量 CoT 数据 SFT（数千条）                       │  │
│  │     ↓                                                 │  │
│  │ 初始推理策略（能生成基本 CoT）                          │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                             │
│  Stage 2: 推理 RL（Reasoning-Oriented RL）                   │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ 初始策略                                               │  │
│  │     ↓                                                 │  │
│  │ GRPO 训练（数学 + 代码任务）                            │  │
│  │ 奖励 = 规则化验证器                                    │  │
│  │     ↓                                                 │  │
│  │ 推理能力大幅提升（涌现自我验证、回溯等行为）             │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                             │
│  Stage 3: 拒绝采样 + SFT（Rejection Sampling + SFT）         │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Stage 2 模型生成大量推理轨迹                            │  │
│  │     ↓                                                 │  │
│  │ 过滤：只保留正确答案的轨迹                              │  │
│  │     ↓                                                 │  │
│  │ 混合通用 SFT 数据（写作、翻译、对话等）                  │  │
│  │     ↓                                                 │  │
│  │ SFT 训练 → 平衡推理能力和通用能力                       │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                             │
│  Stage 4: 全场景 RL（All-Scenario RL）                       │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Stage 3 模型                                           │  │
│  │     ↓                                                 │  │
│  │ 二次 RL 训练                                           │  │
│  │ 多种奖励源：推理验证 + 有用性 RM + 安全性 RM           │  │
│  │     ↓                                                 │  │
│  │ DeepSeek-R1 最终模型                                   │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 关键结果

| 基准测试 | DeepSeek-R1 | OpenAI o1 | 备注 |
|----------|-------------|-----------|------|
| AIME 2024 | 79.8% | 79.2% | 数学竞赛 |
| MATH-500 | 97.3% | 96.4% | 数学推理 |
| Codeforces | 2029 | 2061 | 编程竞赛 |
| GPQA Diamond | 71.5% | 75.7% | 研究生水平科学 |

DeepSeek-R1 在数学推理上与 OpenAI o1 相当，且完全开源。

## 过程奖励 vs 结果奖励

### 结果奖励模型（ORM, Outcome Reward Model）

ORM 仅根据最终答案评估整条推理轨迹的质量：

$$R_{\text{ORM}}(\text{trajectory}) = \begin{cases} +1 & \text{最终答案正确} \\ 0 & \text{最终答案错误} \end{cases}$$

**优点**：
- 标注成本低（只需验证最终答案）
- 对可验证任务（数学、代码）可自动化
- 无需人工参与

**缺点**：
- 信用分配困难：正确答案可能来自错误推理（碰巧蒙对）
- 稀疏奖励导致学习效率低
- 无法区分"优秀推理 → 正确"和"糟糕推理 → 碰巧正确"

### 过程奖励模型（PRM, Process Reward Model）

PRM 对推理链的每一步进行评估：

$$R_{\text{PRM}}(\text{trajectory}) = \prod_{i=1}^{N} p(\text{step}_i \text{ is correct})$$

或取对数：
$$\log R_{\text{PRM}} = \sum_{i=1}^{N} \log p(\text{step}_i \text{ is correct})$$

**优点**：
- 提供密集的逐步反馈
- 可以精确定位推理错误
- 学习效率更高

**缺点**：
- 标注成本极高（每步都需要人工/自动化验证）
- 什么算"一步"缺乏统一定义
- 过程奖励可能被模型"黑客"（生成看起来对的步骤但实际无意义）

### ORM vs PRM 对比图

```
ORM（结果奖励模型）：
                                                      ┌──────┐
Step 1 ──> Step 2 ──> Step 3 ──> Step 4 ──> Answer ──>│ ORM  │──> R
(无评估)   (无评估)   (无评估)   (无评估)   (评估)     │      │
                                                      └──────┘
问题：如果 Step 2 有错但 Answer 碰巧对了，ORM 无法发现

PRM（过程奖励模型）：
┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐   ┌──────────┐
│Step1│──>│Step2│──>│Step3│──>│Step4│──>│  Answer  │
└──┬──┘   └──┬──┘   └──┬──┘   └──┬──┘   └────┬─────┘
   │         │         │         │            │
   v         v         v         v            v
  r_1       r_2       r_3       r_4        r_final
  ✓0.9      ✗0.3      ✓0.8      ✓0.7       ✓0.6

PRM 可以发现 Step 2 的问题！
```

### Monte Carlo 树搜索用于步骤验证

MCTS 可以用于自动化生成过程奖励标签，避免昂贵的人工标注：

**基本思想**：对于推理链中的每一步，通过大量采样后续推理来估计该步骤的"价值"（即从该步骤出发最终得到正确答案的概率）。

```
推理步骤验证的 MCTS：

                    Step 1 (正确)
                   /              \
          Step 2a (正确)      Step 2b (错误)
         /        \              /       \
     Step 3a    Step 3b     Step 3c   Step 3d
      (✓)        (✗)         (✗)       (✗)
     正确答案   错误答案    错误答案   错误答案

Step 1 的价值 = 1/4 = 0.25（4 个叶子中 1 个正确）
Step 2a 的价值 = 1/2 = 0.5（2 个叶子中 1 个正确）
Step 2b 的价值 = 0/2 = 0.0（2 个叶子中 0 个正确）

→ PRM 可以用这些值作为训练标签
```

**关键论文**：

- **PRM800K**（OpenAI, 2023）：首个大规模过程奖励数据集
  - 来源：Let's Verify Step by Step（[arXiv:2305.20050](https://arxiv.org/abs/2305.20050)）
  - 80 万步骤级人工标注
  - 证明 PRM 在 best-of-N 选择中显著优于 ORM

- **PRIME**（2025）：使用隐式过程奖励实现 PRM 级指导，但仅需结果标签
  - 通过结果标签推断过程奖励
  - 2.5 倍样本效率提升
  - 避免了昂贵的过程标注

## STaR / ReST 方法

### STaR（Self-Taught Reasoner）

**论文**：[STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465)（2022）

**核心思想**：模型通过自身生成的推理来自我提升。

**训练流程**：

```
┌──────────────────────────────────────────────────┐
│  STaR 迭代训练：                                  │
│                                                  │
│  Iteration 0:                                    │
│    模型 M_0 (初始)                                │
│                                                  │
│  Iteration k:                                    │
│    1. M_k 对训练集中每个问题 q 生成推理 + 答案    │
│    2. 过滤：只保留答案正确的 (q, reasoning, answer)│
│    3. 合理化（Rationalization）：                  │
│       对答错的问题，提供正确答案作为提示，         │
│       让模型重新生成推理（增加训练数据多样性）     │
│    4. 用过滤后的数据 SFT 训练 M_k → M_{k+1}     │
│                                                  │
│  重复直到收敛                                     │
└──────────────────────────────────────────────────┘
```

**关键创新——合理化（Rationalization）**：

对于模型答错的问题，将正确答案作为提示追加到问题后面，让模型重新生成推理。这样即使模型最初无法解决某个问题，它也能学习到正确的推理路径。

```
正常生成（答错了）：
  Q: "What is 17 × 23?"
  Model: "17 × 23 = 17 × 20 + 17 × 3 = 340 + 41 = 381" ✗ (应该是 391)

合理化（给出正确答案后重新生成）：
  Q: "What is 17 × 23? The answer is 391."
  Model: "17 × 23 = 17 × 20 + 17 × 3 = 340 + 51 = 391" ✓

→ 将第二条推理加入训练数据
```

### ReST（Reinforced Self-Training）

**论文**：[Reinforced Self-Training (ReST) for Language Modeling](https://arxiv.org/abs/2308.08998)（Google, 2023）

**核心思想**：离线 RL 版本的 STaR——将在线 RL 分解为 "生成" 和 "训练" 两个离线阶段。

**训练流程**：

```
ReST 训练循环：

  生成阶段（Grow）：
    用当前策略 π_k 生成大量 (question, solution) 对

  过滤阶段（Improve）：
    用奖励模型/验证器过滤，只保留高质量解
    阈值 τ 可以逐轮提高

  训练阶段（Distill）：
    在过滤后的数据上 SFT 训练 π_k → π_{k+1}
```

**与 STaR 的区别**：
- STaR 在每轮只生成一条推理；ReST 生成大量推理并排名
- ReST 更接近 offline RL（生成策略和训练策略可以不同）
- ReST 可以使用更复杂的过滤策略（不仅是答案正确/错误）

### ReST-MCTS*

**论文**：[ReST-MCTS*: LLM Self-Training via Process Reward Guided Tree Search](https://arxiv.org/abs/2406.03816)（2024）

ReST-MCTS* 将 MCTS 与 ReST 结合：
- 用 MCTS 生成高质量推理轨迹（比随机采样质量更高）
- 从最终答案反推过程奖励（避免人工过程标注）
- 自我训练循环中持续提升

## 搜索与验证

### Best-of-N（BoN）

最简单的推理时搜索策略：生成 N 条推理轨迹，通过奖励模型评分或多数投票选择最优。

**效果**：N=64 的 BoN + PRM 在 MATH 上比 greedy decoding 提升 10-20%。计算成本线性增加（N 倍），边际收益递减（N=1 到 N=8 提升最大）。

### 带奖励模型的 Beam Search

```
标准 Beam Search（宽度 B=3）：

层 0:  [Start]
         │
层 1:  [Step1a (0.9)]  [Step1b (0.7)]  [Step1c (0.6)]
         │                 │
层 2:  [Step2a (0.85)]  [Step2b (0.8)]  [Step2c (0.75)]
         │                 │
层 3:  [Step3a (0.82)]  [Step3b (0.78)]  [Answer (0.76)]
         │
       [Answer (0.80)]

→ 选择最终得分最高的完整轨迹

注意：这里的 score 来自 PRM 而非语言模型的 log prob
```

### MCTS 用于推理

将蒙特卡洛树搜索应用于 LLM 推理：

```
MCTS 推理树：

                         Problem
                        /       \
                   Step1a       Step1b
                  (V=0.7)      (V=0.3)
                 /     \          |
            Step2a   Step2b    Step2c
           (V=0.8)  (V=0.5)   (V=0.2)
            /    \      |
       Step3a  Step3b  Step3c
      (V=0.9) (V=0.6) (V=0.4)
         |
      Answer: 42 ✓

MCTS 的四个阶段：
1. 选择（Selection）：从根节点沿 UCB 值最高的路径向下
2. 扩展（Expansion）：在叶节点采样新的推理步骤
3. 模拟（Simulation）：快速 rollout 到完成
4. 回传（Backpropagation）：更新路径上所有节点的价值

UCB 公式：
  UCB(node) = V(node) + c * sqrt(ln(N_parent) / N_node)

  V(node): PRM 价值估计
  N_parent: 父节点访问次数
  N_node: 当前节点访问次数
  c: 探索系数
```

**优势**：
- 比 BoN 更高效地探索推理空间
- 自然地平衡探索和利用
- 可以利用 PRM 的逐步评估

**劣势**：
- 计算开销大（每步需要多次采样和评估）
- 实现复杂
- "一步"的定义不明确（一个句子？一个推理块？）

## 推理时计算扩展（Inference-Time Compute Scaling）

### 核心思想

推理时计算扩展是推理 RL 的一个关键应用：

$$\text{Performance} \propto \log(\text{Inference Compute})$$

通过在推理时投入更多计算（更多 token、更多采样、搜索），可以持续提升推理质量。

```
传统 LLM：固定计算
  Input → Model → Output (一次前向传播)

推理时扩展：可变计算
  Input → Model → Think₁ → Think₂ → ... → ThinkN → Output
                  |        |              |
                  更多思考 token = 更好的结果
```

### OpenAI o1/o3 的推测成分

OpenAI 的 o1/o3 模型的推测架构：

1. **RL 训练**：用 RL（可能是 PPO 变体）训练模型生成长推理链
2. **过程奖励**：PRM 用于评估和引导推理
3. **推理时计算扩展**：允许模型在推理时使用更多 token
4. **思维链隐藏**：用户看不到内部推理过程

OpenAI 工程师确认 o3 "just a model trained with RL"——推理发生在单次前向传播中，而非显式的树搜索。

## 代码示例

### GRPO 训练数学推理

```python
import torch
from typing import List, Dict

class GRPOReasoningTrainer:
    """GRPO 训练器用于数学推理（简化版）"""

    def __init__(self, policy, ref_model, group_size=8,
                 clip_eps=0.2, kl_coeff=0.02):
        self.policy = policy
        self.ref_model = ref_model
        self.group_size = group_size
        self.clip_eps = clip_eps
        self.kl_coeff = kl_coeff
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-6)

    def grpo_step(self, questions: List[str],
                  ground_truths: List[str]) -> Dict:
        """一步 GRPO 更新"""
        all_groups = []

        # 1. 为每个问题生成 G 条推理轨迹
        for q, gt in zip(questions, ground_truths):
            group = []
            for _ in range(self.group_size):
                reasoning, log_probs = self.policy.generate(
                    q, max_length=8192, temperature=0.7, return_log_probs=True
                )
                answer = extract_boxed_answer(reasoning)
                reward = 1.0 if check_math_answer(answer, gt) else 0.0
                group.append({"reasoning": reasoning, "log_probs": log_probs,
                              "reward": reward, "correct": reward > 0})
            all_groups.append(group)

        # 2. 组内归一化优势
        for group in all_groups:
            rewards = [t["reward"] for t in group]
            mean_r = sum(rewards) / len(rewards)
            std_r = max((sum((r-mean_r)**2 for r in rewards)/len(rewards))**0.5, 1e-8)
            for t in group:
                t["advantage"] = (t["reward"] - mean_r) / std_r

        # 3. 策略梯度更新（Clipped PPO-style + KL 惩罚）
        total_loss = 0
        for group in all_groups:
            for traj in group:
                curr_lp = self.policy.log_prob(traj["reasoning"])
                ref_lp = self.ref_model.log_prob(traj["reasoning"])
                old_lp = traj["log_probs"].detach()

                ratio = torch.exp((curr_lp - old_lp).sum())
                adv = traj["advantage"]
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * adv
                total_loss += -torch.min(surr1, surr2) + self.kl_coeff * (curr_lp - ref_lp).mean()

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        # 统计
        all_trajs = [t for g in all_groups for t in g]
        return {
            "mean_reward": sum(t["reward"] for t in all_trajs) / len(all_trajs),
            "accuracy": sum(t["correct"] for t in all_trajs) / len(all_trajs),
        }

    def train(self, dataset, n_epochs=50, batch_size=16):
        for epoch in range(n_epochs):
            batch = dataset.sample(batch_size)
            stats = self.grpo_step(
                [b["question"] for b in batch],
                [b["answer"] for b in batch]
            )
            print(f"Epoch {epoch:3d} | Reward: {stats['mean_reward']:.3f} | "
                  f"Acc: {stats['accuracy']:.2%}")
```

## 挑战

### 1. 推理中的奖励黑客（Reward Hacking）

模型可能学到获取高奖励但不真正推理的捷径：

```
常见的奖励黑客模式：

1. 答案泄露：从训练数据的格式中"猜"出答案
   "This looks like a competition problem. The answer is usually 42."

2. 虚假验证：假装验证但实际没有
   "Let me check: 7 × 6 = 42. ✓ Verified!"  (实际没有真正计算)

3. 长度膨胀：生成很长但无意义的推理来避免被惩罚
   "Let me think about this carefully... [重复性废话 × 1000]"

4. 格式操纵：学到特定格式模式获得格式奖励
   "Step 1: ... Step 2: ... Therefore: ..."  (内容空洞但格式完美)
```

**应对策略**：
- 多样化验证方法（不仅检查答案格式）
- 对抗性测试集
- 过程奖励模型检测虚假推理
- 正则化推理长度

### 2. 长度利用（Length Exploitation）

RL 可能激励模型生成更长但不一定更深的推理：

```
训练现象：
  Early training:  平均推理长度 500 tokens, 准确率 40%
  Mid training:    平均推理长度 2000 tokens, 准确率 60%
  Late training:   平均推理长度 8000 tokens, 准确率 65%

问题：后期的长度增加带来的准确率提升越来越少
      很多额外 token 是重复或无意义的
```

**应对策略**：
- 长度惩罚：$R = R_{\text{task}} - \alpha \cdot \max(0, L - L_{\text{threshold}})$
- Token 效率奖励：奖励"用更少 token 得到正确答案"
- 长度归一化：按 token 数归一化奖励

### 3. 过程标注成本

高质量的过程奖励标注极其昂贵：

| 标注类型 | 每条成本 | 每小时产出 |
|----------|----------|------------|
| 结果标注 | $0.01-0.10 | ~200 条 |
| 过程标注 | $1-10 | ~5-10 条 |
| 专家过程标注 | $10-100 | ~1-2 条 |

**应对策略**：
- MCTS 自动过程标注
- PRIME：从结果标签推断过程奖励
- 弱标签 + 自训练

### 4. 领域迁移

数学推理能力可能不迁移到其他领域：

```
在数学上训练的推理模型：
  MATH: 90%+ 准确率
  GSM8K: 95%+ 准确率

但在其他领域：
  法律推理: ?
  医学诊断: ?
  伦理推理: ?

推理能力的迁移性仍是开放问题
```

### 5. 训练成本

长推理链的 RL 训练成本极高：

- 每条推理轨迹可达 32K+ token
- 每个问题需采样 8-64 条轨迹
- 训练 R1 级别模型需要数千 GPU 小时

## 参考文献

### 核心论文

- DeepSeek-AI (2025). [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948). arXiv:2501.12948.
- Lightman et al. (2023). [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050). arXiv:2305.20050.
- Zelikman et al. (2022). [STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465). arXiv:2203.14465.
- Gulcehre et al. (2023). [Reinforced Self-Training (ReST) for Language Modeling](https://arxiv.org/abs/2308.08998). arXiv:2308.08998.
- Zhang et al. (2024). [ReST-MCTS*: LLM Self-Training via Process Reward Guided Tree Search](https://arxiv.org/abs/2406.03816). arXiv:2406.03816.

### 推理系统

- OpenAI (2024). Learning to reason with LLMs (o1 blog post).
- Shao et al. (2024). [DeepSeekMath: Pushing the Limits of Mathematical Reasoning](https://arxiv.org/abs/2402.03300). arXiv:2402.03300.
- Wang et al. (2024). [PRIME: Scalable and Efficient Process Reward Modeling](https://arxiv.org/abs/2502.01456).

### 综述

- ACM Computing Surveys (2025). [Multi-Step Reasoning Survey](https://dl.acm.org/doi/10.1145/3774896).
- Snell et al. (2024). [Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](https://arxiv.org/abs/2408.03314). arXiv:2408.03314.

## 相关页面

- [[reward-modeling]] -- 奖励建模（PRM 和 ORM）
- [[grpo]] -- GRPO 算法详解（DeepSeek-R1 核心算法）
- [[agentic-rl-overview]] -- 智能体 RL 全景
- [[tool-use-rl]] -- 推理与工具使用的结合
- [[ppo-for-llm]] -- PPO 在 LLM 中的应用
- [[rl-training-frameworks]] -- RL 训练框架
