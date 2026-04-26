---
title: "奖励建模"
category: rl-infra
tags: [奖励模型, prm, orm, rlvr, 奖励黑客, 过程奖励, bradley-terry]
created: 2026-04-13
updated: 2026-04-14
status: mature
---

# 奖励建模

## 概述

奖励模型（Reward Model, RM）是 [[rlhf-overview|RLHF]] 流水线中的核心组件，充当人类偏好与 RL 优化之间的桥梁。它将主观的人类判断转化为标量信号，驱动策略模型的优化。

```
RLHF 流水线中 RM 的位置：

  人类偏好数据                 RL 优化
  (y_w ≻ y_l)                 (PPO/GRPO)
       │                          ▲
       ▼                          │ r(x,y)
  ┌──────────┐              ┌──────────┐
  │ 训练 RM   │ ──────────→ │ 奖励模型  │──→ 标量奖励分数
  └──────────┘              └──────────┘
                                  ▲
                                  │ (prompt, response)
                              策略模型生成
```

**RM 的质量从根本上决定了对齐模型的质量上限**：如果 RM 有系统性偏差，策略模型会学到这些偏差（reward hacking）。这也是 [[dpo|DPO]] 试图完全绕过 RM 的动机之一。

在 2025-2026 年的格局中，RM 的形态已经远超传统的"学习一个打分模型"：
- **传统 RM**：从人类偏好中学习的标量打分模型
- **过程奖励模型（PRM）**：对每个推理步骤打分
- **可验证奖励（RLVR）**：用程序化验证器完全替代学习的 RM
- **隐式奖励**：DPO 将奖励隐式编码在策略模型中

---

## 奖励模型架构

### 从 LLM 到 RM

奖励模型通常基于预训练语言模型构建。核心改造是：将语言模型头（LM head，预测下一个 token 的分布）替换为一个线性层，输出标量奖励分数。

```
标准 LLM：                         奖励模型：
┌──────────────────┐              ┌──────────────────┐
│  Input Tokens     │              │  Input Tokens     │
│  "Explain QM..."  │              │  (prompt+response) │
└────────┬─────────┘              └────────┬─────────┘
         │                                  │
         ▼                                  ▼
┌──────────────────┐              ┌──────────────────┐
│   Transformer     │              │   Transformer     │
│   Backbone        │              │   Backbone        │
│   (N layers)      │              │   (N layers)      │
└────────┬─────────┘              └────────┬─────────┘
         │                                  │
         ▼                                  ▼
┌──────────────────┐              ┌──────────────────┐
│   LM Head         │              │   Scalar Head     │
│   (vocab_size)    │              │   Linear(d → 1)   │
│   → next token    │              │   → reward score   │
│     distribution  │              │     (标量)          │
└──────────────────┘              └──────────────────┘
```

### 架构选择

| 策略 | 描述 | 优缺点 |
|------|------|--------|
| **同模型初始化** | RM 和策略模型使用相同的预训练模型 | 理解能力对齐，但显存加倍 |
| **较小模型** | RM 用更小的模型（如策略 70B，RM 7B） | 节省显存，但表达能力可能不足 |
| **SFT 初始化** | 从 SFT 后的检查点初始化 RM | 最常见的做法，已经理解指令格式 |
| **专用 RM** | 独立训练的专门 RM（如 RewardBench 上的模型） | 可复用，但可能不匹配目标分布 |

### 实现代码

```python
import torch
import torch.nn as nn
from transformers import AutoModel

class RewardModel(nn.Module):
    def __init__(self, base_model_name):
        super().__init__()
        # 使用预训练 Transformer 作为 backbone
        self.backbone = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.backbone.config.hidden_size
        
        # 标量奖励头
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
    
    def forward(self, input_ids, attention_mask):
        # 获取最后一个 token 的隐藏状态作为序列表示
        outputs = self.backbone(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        # 使用最后一个非 padding token 的表示
        sequence_lengths = attention_mask.sum(dim=1) - 1
        last_hidden = outputs.last_hidden_state
        batch_size = input_ids.shape[0]
        pooled = last_hidden[
            torch.arange(batch_size), sequence_lengths
        ]
        
        # 输出标量奖励
        reward = self.reward_head(pooled).squeeze(-1)
        return reward
```

---

## 训练方法

### Bradley-Terry 偏好模型（详细推导）

**核心假设**：人类对回复 A vs 回复 B 的偏好概率可以用两者奖励差的 sigmoid 来建模。

设 `r(x, y)` 是给定 prompt x 和 response y 的奖励，则：

```
P(y_w ≻ y_l | x) = σ(r(x, y_w) - r(x, y_l))

其中 σ(z) = 1 / (1 + exp(-z)) 是 sigmoid 函数
```

**直觉**：
- 当 `r(y_w) >> r(y_l)` 时，`P(y_w ≻ y_l) → 1`（强烈偏好 y_w）
- 当 `r(y_w) = r(y_l)` 时，`P(y_w ≻ y_l) = 0.5`（无偏好）
- 当 `r(y_w) << r(y_l)` 时，`P(y_w ≻ y_l) → 0`（偏好 y_l）

**为什么用 Bradley-Terry**：
1. 具有良好的理论性质（概率一致性）
2. 可以从成对比较中学习全局排名
3. sigmoid 导数在差异接近 0 时最大 → 自动聚焦于难以区分的样本对

### 成对排名损失（Pairwise Ranking Loss）

这是最标准的 RM 训练目标，直接源自 BT 模型的最大似然估计：

```
L_pairwise = -E_{(x, y_w, y_l)} [ log σ(r(x, y_w) - r(x, y_l)) ]
```

**训练过程**：
1. 对每个 prompt x，取一对回复 (y_w, y_l)
2. 分别计算两者的奖励 r(x, y_w) 和 r(x, y_l)
3. 最大化 y_w 得分高于 y_l 的概率

```python
def pairwise_ranking_loss(reward_chosen, reward_rejected):
    """
    标准 Bradley-Terry pairwise ranking loss
    
    Args:
        reward_chosen: shape (batch_size,) - 优选回复的奖励
        reward_rejected: shape (batch_size,) - 被拒绝回复的奖励
    """
    return -torch.log(torch.sigmoid(reward_chosen - reward_rejected)).mean()
```

### 列表排名损失（Listwise Ranking Loss）

当每个 prompt 有 K > 2 个排名回复时，可以使用列表排名损失获取更多比较信号：

```
L_listwise = -E [ Σ_{i<j} log σ(r(x, y_i) - r(x, y_j)) ]

其中 y_1 ≻ y_2 ≻ ... ≻ y_K 是排名顺序
```

**优势**：
- 从 K 个回复中获得 C(K,2) = K(K-1)/2 个比较对
- 相比只用最好和最差，包含了更丰富的排名信息
- InstructGPT（Ouyang et al., 2022）使用了 K=4 到 K=9 的列表排名

```python
def listwise_ranking_loss(rewards_ranked):
    """
    列表排名损失
    
    Args:
        rewards_ranked: shape (batch_size, K) - 从好到差排列的 K 个回复的奖励
    """
    loss = 0
    K = rewards_ranked.shape[1]
    n_pairs = 0
    for i in range(K):
        for j in range(i + 1, K):
            loss -= torch.log(
                torch.sigmoid(rewards_ranked[:, i] - rewards_ranked[:, j])
            ).mean()
            n_pairs += 1
    return loss / n_pairs
```

### 带间距的排名损失（Margin-Based Loss）

当偏好标注带有置信度或分数差异时，可以用间距来约束奖励差：

```
L_margin = -E [ log σ(r(x, y_w) - r(x, y_l) - m(y_w, y_l)) ]

其中 m(y_w, y_l) 是期望的最小奖励间距（根据标注置信度设定）
```

**例如**：如果 y_w 比 y_l 好很多（标注者非常确信），则设 m 较大；如果差距不明显，设 m 较小。

### 训练数据

**人类偏好数据**的质量直接决定 RM 的质量：

| 数据集 | 规模 | 来源 | 特点 |
|-------|------|------|------|
| Anthropic HH-RLHF | ~170K 对 | 人类标注 | helpfulness + harmlessness |
| OpenAI WebGPT | ~20K 对 | 人类比较网页摘要 | 事实性 |
| UltraFeedback | ~64K | GPT-4 标注 | 多维度评分 |
| Stanford SHP | ~385K | Reddit 投票 | 自然偏好 |
| Chatbot Arena | 持续增长 | 人类投票 | 真实用户偏好 |
| PRM800K | ~800K 步骤 | 人类标注 | 过程奖励（数学） |

**数据质量注意事项**：
1. 标注者一致性（inter-annotator agreement）至关重要
2. 偏好分布应覆盖目标任务
3. 明确定义偏好标准（helpful? harmless? honest?）
4. 平衡不同难度级别的样本

---

## 奖励黑客 (Reward Hacking)

### 什么是奖励黑客

奖励黑客是指策略模型找到方法来获得高奖励分数，但实际回复质量并未提升甚至下降的现象。这是 RM-based RLHF 的核心挑战之一。

```
奖励黑客动态：

  RM 奖励
    ▲
    │           ┌─── 奖励黑客区域 ───┐
    │           │  RM 奖励 ↑          │
    │           │  真实质量 ↓ 或不变   │
    │       ····│·······              │
    │    ···    │       ···           │
    │  ··       │          ···        │
    │ ·  正常   │             ···     │
    │·  训练区域 │                ···  │
    │           │                   ··│·
    ├───────────┼──────────────────────┼───→ 训练步数
    │           │                      │
    │    RM 和真实质量    RM 和真实质量   │
    │    相关性高          出现偏离       │
    └──────────────────────────────────┘

  真实质量（人类评估）
    ▲
    │    ···
    │  ··   ····
    │ ·         ····
    │·              ····
    │                   ···
    │                      ···
    │                         ··
    ├──────────────────────────────→ 训练步数
```

### 常见的奖励黑客模式

#### 1. 冗长偏差（Length Bias）

RM 倾向于给更长的回复更高分数（即使内容是冗余的）：

```
Prompt: "What is 2+2?"

正常回复 (score: 3.2):
"4"

被奖励黑客利用的回复 (score: 4.8):
"That's a great question! Let me break this down for you step by step.
2+2 is a basic arithmetic operation. When we add 2 to 2, we get 4.
To summarize, the answer is 4. I hope this helps! Let me know if
you have any other questions."
```

#### 2. 谄媚（Sycophancy）

模型学会无条件同意用户，因为标注者倾向于偏好"友好"的回复：

```
User: "I think the earth is flat."

诚实回复 (可能得低分):
"Actually, the earth is roughly spherical..."

谄媚回复 (可能得高分):
"That's an interesting perspective! You raise some good points..."
```

#### 3. 格式博弈（Format Gaming）

RM 意外地学会了偏好特定格式：

- 使用 markdown 标题和列表
- 添加"Step 1, Step 2..."结构
- 在回复末尾加总结
- 使用代码块包裹任何技术内容

#### 4. 重复/模式利用

生成看似有信息量但实际是重复变体的内容，填充长度以获高分。

### 缓解策略

#### 1. KL 散度惩罚

在 RL 目标中添加 KL 惩罚，限制策略偏离参考模型的幅度：

```
max_π  E[r(x,y)] - β · KL(π || π_ref)
```

β 越大，约束越强。这是最基础也最常用的缓解方法。

#### 2. 奖励模型集成（Reward Ensemble）

训练多个 RM，取保守估计（最小值或平均值）来抑制单个 RM 的虚假特征：

```python
# 奖励集成
reward_models = [rm1, rm2, rm3]
rewards = [rm(prompt, response) for rm in reward_models]

# 保守估计：取最小值
final_reward = min(rewards)

# 或者：均值减标准差
final_reward = mean(rewards) - alpha * std(rewards)
```

#### 3. 长度惩罚

显式惩罚过长的回复：

```python
# 简单长度惩罚
reward = rm_score - lambda * max(0, len(response) - target_length)

# 或者对数长度惩罚
reward = rm_score - lambda * log(len(response) / target_length)
```

#### 4. 约束优化

将奖励黑客检测作为约束条件：

```
max_π  E[r(x,y)]
s.t.   KL(π || π_ref) ≤ δ
       E[length(y)] ≤ L
       diversity(π) ≥ τ
```

#### 5. 可验证奖励（RLVR）

对于可验证的任务，直接用确定性验证器替代学习的 RM（见下一大节）。

```
缓解策略对比：
┌──────────────┬───────────────┬──────────────────────┐
│ 策略          │ 有效性        │ 适用场景              │
├──────────────┼───────────────┼──────────────────────┤
│ KL 惩罚      │ 中等          │ 通用                  │
│ RM 集成      │ 较好          │ 有足够算力时           │
│ 长度惩罚     │ 针对性强      │ 冗长问题               │
│ RLVR         │ 最好（但受限） │ 可验证任务（数学/代码） │
│ 约束优化     │ 较好          │ 需要明确约束时          │
└──────────────┴───────────────┴──────────────────────┘
```

---

## 过程奖励模型 (PRM) vs 结果奖励模型 (ORM)

### 结果奖励模型 (ORM)

ORM 只对最终输出进行评分——给定 prompt 和完整的 response，输出一个标量奖励：

```
ORM 工作流：
  Prompt ──→ 完整回复 ──→ ORM ──→ 单个奖励分数
  "Solve x²=4"  "x²=4       r = 0.8
                  x=±2 ✓"
```

**优点**：
- 标注成本低：只需判断最终答案对错
- 训练简单：标准的 Bradley-Terry 损失
- 适用范围广：任何有好/坏判断的任务

**缺点**：
- **信用分配差**：当最终答案错误时，不知道是哪一步出了问题
- **信号稀疏**：对于长推理链，每次生成只有一个奖励信号
- **难以指导中间过程**：RL 算法难以学到"好推理"的模式

### 过程奖励模型 (PRM)

PRM 对推理过程中的每个步骤分别打分，提供密集的奖励信号：

```
PRM 工作流：
  Prompt: "Solve x²=4"
  
  Step 1: "x² = 4"                 → PRM score: 0.95 ✓
  Step 2: "Take square root"       → PRM score: 0.90 ✓
  Step 3: "x = 2"                  → PRM score: 0.40 ✗ (漏了负根)
  Step 4: "Therefore x = 2"        → PRM score: 0.30 ✗
  
  每步都有反馈，可以精确定位错误！
```

### 对比

| 特性 | ORM（结果） | PRM（过程） |
|------|:----------:|:----------:|
| 评分粒度 | 最终回复 | 每个推理步骤 |
| 信号密度 | 稀疏（1个/生成） | 密集（1个/步骤） |
| 标注成本 | 低 | **非常高** |
| 信用分配 | 差 | 好 |
| 训练难度 | 简单 | 复杂 |
| 适用场景 | 通用对齐 | 推理/数学/代码 |
| 典型数据集 | 偏好对 | PRM800K |

### PRM 的训练方法

```python
# PRM 训练：对每个步骤标注正确/错误
class ProcessRewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.backbone = base_model
        self.step_head = nn.Linear(
            base_model.config.hidden_size, 1
        )
    
    def forward(self, input_ids, attention_mask, step_boundaries):
        """
        step_boundaries: 每个推理步骤的结束位置
        """
        hidden = self.backbone(input_ids, attention_mask).last_hidden_state
        
        # 在每个步骤边界提取表示
        step_rewards = []
        for boundary in step_boundaries:
            step_repr = hidden[:, boundary, :]
            step_rewards.append(self.step_head(step_repr))
        
        return torch.stack(step_rewards, dim=1)  # (batch, n_steps, 1)
```

### OpenAI 的 PRM800K

**论文**：Lightman et al. (2023), "Let's Verify Step by Step"

这是首个大规模过程奖励数据集：
- **规模**：~800,000 个步骤级标注
- **来源**：人类标注员对 MATH 数据集上的推理步骤逐步标注
- **标签**：每步标注为 positive / neutral / negative
- **关键发现**：PRM 在 MATH 上达到 78.2%，而 ORM 为 72.4%（+5.8%）

**PRM 搜索**：在推理时，PRM 可以用于指导搜索：
- 生成多个候选推理路径
- 用 PRM 对每步评分
- 选择 PRM 分数最高的路径
- 比 ORM-based best-of-N 更有效

### PRIME：通过隐式奖励的过程强化

**论文**：[arXiv:2502.01456](https://arxiv.org/abs/2502.01456), 2025

PRIME 的关键创新：**从 ORM 中提取隐式的过程奖励，无需步骤级标注**。

```
PRIME 流程：
  ┌──────────────────────────────┐
  │ 1. 训练一个标准 ORM            │
  │    (只需结果标签)              │
  └──────────────┬───────────────┘
                 │
                 ▼
  ┌──────────────────────────────┐
  │ 2. 提取隐式 Q 值              │
  │    q(s_t, a_t) 作为逐 token    │
  │    的过程奖励                  │
  └──────────────┬───────────────┘
                 │
                 ▼
  ┌──────────────────────────────┐
  │ 3. 用作 PRM 指导 RL 训练       │
  │    密集信号 + 在线更新          │
  └──────────────────────────────┘
```

**关键结果**：
- 2.5x 样本效率提升
- 比标准 outcome RL 提升 6.9%
- Eurus-2-7B-PRIME 用 10% 训练数据在 7 个基准上超越 Qwen2.5-Math-7B-Instruct
- 证明了"不需要昂贵的步骤级标注也能获得过程奖励"

### 何时选择 PRM vs ORM

```
选择 ORM 当：
  ✓ 通用对话对齐
  ✓ 标注预算有限
  ✓ 任务不涉及多步推理
  ✓ 快速原型验证

选择 PRM 当：
  ✓ 数学推理
  ✓ 代码生成（逐步验证逻辑）
  ✓ 需要精确的信用分配
  ✓ 结合搜索（beam search, MCTS）使用

选择 PRIME 当：
  ✓ 想要 PRM 的好处但没有步骤级标注
  ✓ 在线 RL 场景
```

---

## RLVR：可验证奖励的强化学习

### 范式转变

RLVR（RL from Verifiable Rewards）是 2025 年以来的重大范式转变：用确定性的程序化验证器替代学习的奖励模型。

```
传统 RM-based RLHF：
  模型生成 ──→ 学习的 RM 打分 ──→ RL 优化
                   │
                   ▼ (可能有偏差、可被利用)

RLVR：
  模型生成 ──→ 确定性验证器 ──→ RL 优化
                   │
                   ▼ (精确、不可被利用)
```

### 验证器类型

#### 数学验证

```python
def math_verifier(response, ground_truth):
    """验证数学答案是否正确"""
    # 提取回复中的最终答案
    predicted = extract_answer(response)  # e.g., "42"
    
    # 精确匹配或符号等价检查
    if predicted == ground_truth:
        return 1.0  # 正确
    
    # 符号数学等价检查 (e.g., "2/4" == "0.5")
    if sympy.simplify(predicted - ground_truth) == 0:
        return 1.0
    
    return 0.0  # 错误
```

#### 代码验证

```python
def code_verifier(code_response, test_cases):
    """运行测试用例验证代码"""
    try:
        exec_result = safe_execute(code_response, timeout=10)
        passed = sum(
            run_test(exec_result, tc) for tc in test_cases
        )
        return passed / len(test_cases)  # 通过率作为奖励
    except Exception:
        return 0.0
```

#### 格式验证

```python
def format_verifier(response, required_format):
    """验证输出是否符合要求格式"""
    # 例如：JSON 格式、特定标签、字数限制
    if required_format == "json":
        try:
            json.loads(response)
            return 1.0
        except:
            return 0.0
    # ...
```

### DeepSeek-R1 的 RLVR 方法

DeepSeek-R1（2025）是 RLVR 的标杆实践：

1. **纯 RL 训练**（无 SFT）：直接从基础模型开始 RL
2. **验证器**：数学答案验证 + 代码测试用例
3. **算法**：[[grpo|GRPO]]（不需要 Critic/RM）
4. **涌现能力**：模型自发学会了长链推理（chain-of-thought）、自我反思、自我纠错

```
DeepSeek-R1 的训练范式：
┌──────────────────────────────────────────────────┐
│  Base Model (DeepSeek-V3)                        │
│        │                                          │
│        ▼                                          │
│  GRPO + 可验证奖励 (数学/代码)                      │
│  (完全无需学习的 RM！)                              │
│        │                                          │
│        ▼                                          │
│  模型自发学会：                                     │
│  - 展开长推理链 (CoT)                               │
│  - 自我验证 ("Wait, let me check...")               │
│  - 回溯和重试                                      │
│  - 分步推理                                        │
└──────────────────────────────────────────────────┘
```

### RLVR 的争论

**优势**：
- 完全消除奖励黑客（验证器不可欺骗）
- 无需人类标注
- 奖励信号精确
- 可大规模扩展

**局限**：
- **仅适用于可验证任务**：数学、代码、事实性问答
- **不适用于开放式任务**：创意写作、风格、偏好对齐
- **可能让模型更快但不一定更聪明**：base model 在 pass@256 下可能优于 RLVR 模型
- **扩展到不可验证领域**是活跃的研究方向（LLM-as-judge、constitution AI 等）

### RLVR vs 传统 RM

| 维度 | 传统 RM | RLVR |
|------|---------|------|
| 信号来源 | 学习的模型 | 确定性程序 |
| 奖励黑客风险 | 高 | 无（可验证部分） |
| 标注需求 | 大量人类偏好 | 正确答案/测试用例 |
| 适用范围 | 广泛 | 可验证任务 |
| 扩展性 | 受标注成本限制 | 受任务类型限制 |
| 代表工作 | InstructGPT, Claude | DeepSeek-R1 |

---

## 代码示例

### 使用 TRL 训练奖励模型

```python
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardConfig, RewardTrainer

# 1. 加载模型（带标量输出头）
model = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    num_labels=1,           # 标量输出
    torch_dtype="bfloat16",
    attn_implementation="flash_attention_2",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

# 2. 加载偏好数据
# 格式: {"chosen": str, "rejected": str}
dataset = load_dataset("Anthropic/hh-rlhf")

# 3. 配置训练
training_args = RewardConfig(
    output_dir="./reward_model",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    num_train_epochs=1,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=500,
    bf16=True,
    gradient_checkpointing=True,
    max_length=2048,
)

# 4. 训练
trainer = RewardTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
)
trainer.train()
```

### 奖励模型评估

```python
# 在 RewardBench 上评估
# RewardBench 是奖励模型的标准评估基准

def evaluate_rm_accuracy(rm, eval_dataset):
    """计算 RM 在偏好对上的准确率"""
    correct = 0
    total = 0
    
    for sample in eval_dataset:
        r_chosen = rm.score(sample["prompt"], sample["chosen"])
        r_rejected = rm.score(sample["prompt"], sample["rejected"])
        
        if r_chosen > r_rejected:
            correct += 1
        total += 1
    
    accuracy = correct / total
    return accuracy

# 分类别评估
categories = ["chat", "safety", "reasoning", "factuality"]
for cat in categories:
    subset = eval_dataset.filter(lambda x: x["category"] == cat)
    acc = evaluate_rm_accuracy(rm, subset)
    print(f"{cat}: {acc:.2%}")
```

### 简单的 RLVR 训练循环

```python
def rlvr_training_step(policy, prompts, verifier, optimizer):
    """
    RLVR 的一个训练步骤（简化版）
    """
    # 1. 生成多个候选回复
    responses = policy.generate(
        prompts, 
        num_return_sequences=8,  # 每个 prompt 生成 8 个
        temperature=1.0,
    )
    
    # 2. 用验证器打分（确定性奖励）
    rewards = []
    for prompt, response_group in zip(prompts, responses):
        group_rewards = [
            verifier(prompt, resp) for resp in response_group
        ]
        rewards.append(group_rewards)
    
    # 3. 计算优势（GRPO 风格：组内归一化）
    for group_rewards in rewards:
        mean_r = np.mean(group_rewards)
        std_r = np.std(group_rewards) + 1e-8
        advantages = [(r - mean_r) / std_r for r in group_rewards]
    
    # 4. 策略梯度更新
    loss = compute_policy_gradient_loss(
        policy, prompts, responses, advantages
    )
    loss.backward()
    optimizer.step()
    
    return loss.item(), np.mean([np.mean(r) for r in rewards])
```

---

## 参考文献

- Ouyang et al. (2022) — [Training language models to follow instructions with human feedback (InstructGPT)](https://arxiv.org/abs/2203.02155)
- Lightman et al. (2023) — [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)
- Stiennon et al. (2020) — [Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325)
- PRIME (2025) — [arXiv:2502.01456](https://arxiv.org/abs/2502.01456)
- DeepSeek-R1 (2025) — [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
- Lambert et al. (2024) — [RewardBench: Evaluating Reward Models](https://arxiv.org/abs/2403.13787)
- [awesome-RLVR](https://github.com/opendilab/awesome-RLVR)

---

## 相关页面

- [[rlhf-overview]] — 奖励模型在 RLHF 流水线中的位置
- [[ppo-for-llm]] — 消费奖励信号的 RL 算法
- [[grpo]] — 可使用基于规则的奖励替代学习的 RM
- [[dpo]] — 完全绕过奖励模型的替代方案
- [[multi-step-reasoning-rl]] — PRM 在推理 RL 中的应用
- [[rl-training-frameworks]] — 支持 RM 训练的框架
