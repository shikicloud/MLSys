---
title: "RLHF：基于人类反馈的强化学习"
category: rl-infra
tags: [rlhf, 对齐, 强化学习, 人类反馈, instructgpt, 奖励模型, sft, bradley-terry]
created: 2026-04-13
updated: 2026-04-14
status: mature
---

# RLHF：基于人类反馈的强化学习

## 概述

RLHF（Reinforcement Learning from Human Feedback）是当前将大语言模型（LLM）与人类意图对齐的核心技术范式。其基本思想是：

1. **语言任务的奖励无法程序化定义** — 对于"有帮助"、"真实"、"无害"这样的属性，不存在简单的数学公式。
2. **人类可以比较** — 虽然人类很难给回复打绝对分数，但可以可靠地判断"A 比 B 好"。
3. **用比较数据训练奖励模型** — 将人类的比较偏好蒸馏为可微分的奖励函数。
4. **用 RL 优化奖励** — 在奖励模型的引导下，通过强化学习微调语言模型。

RLHF 产出的模型更有帮助（helpful）、更真实（truthful）、更安全（harmless），是 ChatGPT、Claude、Gemini 等现代对话模型背后的关键技术。

### 为什么不能只用监督学习？

监督微调（SFT）需要"标准答案"，但对于开放式对话任务：
- 同一个问题可以有无数种合理回答
- 人类标注员很难从零写出"完美"回答
- SFT 只能模仿训练数据的分布，无法超越标注质量

RLHF 的突破在于：**模型可以探索训练数据之外的回答空间，并通过奖励信号持续改进**。InstructGPT 论文（Ouyang et al., 2022）的标志性结果是 1.3B 参数的 RLHF 模型被人类评为优于 175B 的 GPT-3。

---

## 历史时间线

| 年份 | 里程碑 | 关键贡献 |
|------|--------|----------|
| **2017** | Christiano et al. — "Deep RL from Human Preferences" | 奠基性论文。在人类对轨迹片段的比较上训练奖励模型。智能体仅用约 900 bits 反馈就学会了后空翻。首次系统化"偏好学习 → 奖励模型 → RL 优化"的框架。 |
| **2019** | Ziegler et al. (OpenAI) — "Fine-Tuning Language Models from Human Preferences" | 首次将 RLHF 应用于语言模型，在摘要生成和情感续写任务上微调 GPT-2。引入 KL 散度惩罚防止策略崩溃。 |
| **2020** | Stiennon et al. — "Learning to Summarize with Human Feedback" | 将 RLHF 扩展到 1.3B 参数的摘要生成模型。RLHF 摘要生成器超越了当时的 SOTA。证明 RLHF 可以产生人类偏好超越纯 SFT 的效果。 |
| **2022.01** | Anthropic — "Training a Helpful and Harmless Assistant from Human Feedback" | 系统研究了 RLHF 在 helpful 和 harmless 两个维度上的效果。引入了 "HH" 数据集。 |
| **2022.03** | Ouyang et al. (OpenAI) — **InstructGPT** | 里程碑论文。1.3B 参数的 InstructGPT 在人类评估中被认为优于 175B GPT-3。正式确立了 SFT → RM → PPO 的三阶段流水线。部署为 OpenAI API 默认模型。 |
| **2022.11** | OpenAI — **ChatGPT** 发布 | 基于 RLHF 训练，引爆全球对 LLM 的关注。RLHF 从学术技术变为工业标准。 |
| **2023** | Anthropic — **Constitutional AI (CAI)** | 用 AI 反馈替代部分人类反馈（RLAIF），降低标注成本，提高可扩展性。 |
| **2023** | Rafailov et al. — **DPO** | 证明可以不训练显式奖励模型，直接从偏好数据优化策略，简化了 RLHF 流水线。 |
| **2024** | DeepSeek — **GRPO** | 去掉 Critic 模型，通过组内比较估计优势，大幅降低内存和计算开销。 |
| **2025** | DeepSeek — **DeepSeek-R1** | GRPO 大规模应用于推理模型训练，展现了涌现的思维链推理能力。发表于 Nature。 |
| **2025** | ACM Computing Surveys — RLHF 综述 | 全面梳理 RLHF 技术发展脉络。 |
| **2025** | Lambert — **RLHF Book** | 首本 RLHF 教科书，系统化整理了理论与实践。 |

---

## 三阶段流水线详解

RLHF 的标准流水线由三个阶段组成。以下 ASCII 图展示了完整的数据流：

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RLHF 三阶段流水线                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  阶段 1: SFT                                                        │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │  预训练模型    │───>│   SFT 数据    │───>│  SFT 模型     │          │
│  │  (Base LLM)  │    │ (prompt,resp)│    │  π_SFT       │          │
│  └──────────────┘    └──────────────┘    └──────┬───────┘          │
│                                                  │                  │
│  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─│─ ─ ─ ─ ─ ─ ─   │
│                                                  │                  │
│  阶段 2: RM 训练                                  │                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────┴───────┐          │
│  │  SFT 模型     │───>│ 人类偏好数据   │───>│   奖励模型    │          │
│  │  (初始化 RM) │    │ (x, y_w, y_l)│    │   r_φ(x,y)  │          │
│  └──────────────┘    └──────────────┘    └──────┬───────┘          │
│                                                  │                  │
│  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─│─ ─ ─ ─ ─ ─ ─   │
│                                                  │                  │
│  阶段 3: RL 优化 (PPO)                             │                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────┴───────┐          │
│  │  SFT 模型     │    │  当前策略生成   │<──>│   奖励模型    │          │
│  │  (参考 π_ref)│    │  回复并更新    │    │   打分       │          │
│  └──────┬───────┘    └──────────────┘    └──────────────┘          │
│         │                    │                                      │
│         └─── KL 惩罚 ───────>│                                      │
│                              ▼                                      │
│                     ┌──────────────┐                                │
│                     │  对齐后的模型   │                                │
│                     │  π_θ (RLHF)  │                                │
│                     └──────────────┘                                │
└─────────────────────────────────────────────────────────────────────┘
```

### 阶段一 — 监督微调 (SFT)

**目标**：将预训练模型从"续写文本"转变为"遵循指令"。

**数据要求**：
- 数据格式：(prompt, desired_response) 对
- 典型数据量：数千到数万条高质量标注
- InstructGPT 使用了约 13,000 条人类标注数据
- 数据质量远比数量重要 — LIMA 论文（Zhou et al., 2023）用仅 1,000 条精选数据就获得了优秀效果

**训练细节**：
- 标准的语言模型交叉熵损失，只在 response 部分计算 loss（prompt 部分被 mask）
- 学习率通常较低（1e-5 ~ 5e-6），防止遗忘预训练知识
- 训练 1-3 个 epoch，避免过拟合
- 可以使用数据混合策略：部分对话数据 + 部分预训练数据

**常见陷阱**：
- **过拟合**：SFT 数据量小，容易过拟合 → 模型输出过于固定化
- **灾难性遗忘**：SFT 可能损害预训练中学到的世界知识
- **数据质量不一致**：标注员风格差异大，导致模型行为不一致
- **格式偏好 vs. 内容质量**：SFT 容易学到表面格式而非深层能力

```python
# SFT 训练伪代码
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

model = AutoModelForCausalLM.from_pretrained("base_model")
training_args = TrainingArguments(
    learning_rate=2e-5,
    num_train_epochs=2,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
)

# 注意：只在 response tokens 上计算 loss
# prompt tokens 的 labels 设为 -100（忽略）
trainer = Trainer(model=model, args=training_args, train_dataset=sft_dataset)
trainer.train()
```

### 阶段二 — 奖励模型 (RM) 训练

**目标**：将人类的比较偏好蒸馏为可微分的标量奖励函数。

#### 数据收集

1. 对同一个 prompt，用 SFT 模型生成 K 个不同回复（通常 K=4~9）
2. 人类标注员对这些回复进行排序（或两两比较）
3. 从排序中提取成对偏好：(prompt, y_w, y_l)，其中 y_w 优于 y_l

InstructGPT 收集了约 33,000 组比较数据（每组 4~9 个回复的排序）。

#### Bradley-Terry 偏好模型

奖励模型的训练基于 **Bradley-Terry 模型**，这是一个经典的配对比较概率模型。其核心思想来源于 Elo 评分系统（国际象棋）：

给定提示词 x，两个回复 y_w（偏好）和 y_l（非偏好），奖励模型 r_φ 将回复映射为标量分数。Bradley-Terry 模型假设人类偏好 y_w > y_l 的概率为：

```
P(y_w > y_l | x) = σ(r_φ(x, y_w) - r_φ(x, y_l))
```

其中 σ 是 sigmoid 函数：σ(z) = 1/(1+e^(-z))。

**直觉理解**：这就像国际象棋的 Elo 评分 — 两个选手（回复）的胜率取决于它们评分之差。评分差越大，强者获胜概率越接近 1。

#### RM 损失函数推导

给定偏好数据集 D = {(x_i, y_w^i, y_l^i)}，我们最大化似然函数：

```
max_φ  Π P(y_w^i > y_l^i | x_i)
```

取对数并取负号得到损失函数：

```
L_RM(φ) = -E_{(x, y_w, y_l) ~ D} [log σ(r_φ(x, y_w) - r_φ(x, y_l))]
```

这就是 RM 训练的标准交叉熵损失。

**重要性质**：
- RM 只学习**相对**分数（差值有意义，绝对值无意义）
- 因此 RM 的输出可以有任意偏移（常通过 normalization 处理）
- 可以在排序数据上提取 C(K,2) 个比较对，提高数据利用效率

#### RM 模型架构

```
┌─────────────────────────────────┐
│  输入: [prompt] + [response]     │
│           ▼                     │
│  Transformer (从 SFT 模型初始化) │
│           ▼                     │
│  最后一个 token 的隐藏状态        │
│           ▼                     │
│  线性层 (hidden_dim → 1)        │
│           ▼                     │
│  标量奖励分数 r(x, y)            │
└─────────────────────────────────┘
```

通常从 SFT 模型初始化（去掉 LM head，加上标量输出头），因为 SFT 模型已经"理解"了指令和回复的语义。

#### RM 训练技巧

- **数据清洗**：去掉标注者分歧大的样本（低 agreement），它们是噪声
- **同时利用排序中的多对比较**：从 K 个回复的排序中提取所有 C(K,2) 个对，每对按排序距离加权
- **奖励模型校准**：定期检查 RM 分数的分布，确保其区分度
- **防止过拟合**：RM 通常比策略模型小（或相同大小），使用 dropout 和早停
- **评估指标**：偏好预测准确率（通常 65-75% 就能支撑 RLHF；InstructGPT 报告约 72%）

### 阶段三 — RL 优化（PPO）

**目标**：最大化奖励模型给出的分数，同时不偏离 SFT 模型太远。

#### RL 目标函数

```
max_θ  E_{x~D, y~π_θ(·|x)} [r_φ(x, y)] - β · KL(π_θ || π_ref)
```

其中：
- π_θ — 正在训练的策略（LLM）
- π_ref — 参考策略（冻结的 SFT 模型）
- r_φ(x, y) — 奖励模型的评分
- β — KL 惩罚系数（关键超参数）
- KL(π_θ || π_ref) — 当前策略与参考策略之间的 KL 散度

在实践中，KL 惩罚被整合进奖励：

```
r_total(x, y) = r_φ(x, y) - β · Σ_t log[π_θ(y_t|x,y_{<t}) / π_ref(y_t|x,y_{<t})]
```

这是一个 **逐 token 的 KL 惩罚**，在序列的每个位置都施加约束。

#### KL 惩罚的作用

KL 惩罚是 RLHF 中至关重要的正则化机制：

| KL 系数 β | 效果 |
|-----------|------|
| β 过小 | 策略过度优化 RM → 奖励黑客（exploitation of RM weaknesses） |
| β 过大 | 策略过于保守 → 与 SFT 几乎无差别，RL 无法发挥作用 |
| β 适中 | 平衡探索与稳定，获得最佳人类偏好 |

InstructGPT 使用 β 的自适应调节：目标 KL 值为 6 nats，β 在训练过程中动态调整。

#### GAE 优势估计

在 PPO 更新中，需要估计每个 token 位置的优势函数 A_t。使用 **广义优势估计（GAE）**：

```
δ_t = r_t + γ · V(s_{t+1}) - V(s_t)     (TD 残差)

A_t^GAE(γ,λ) = Σ_{l=0}^{T-t} (γλ)^l · δ_{t+l}
```

其中：
- V(s_t) — Critic 模型在位置 t 的价值估计
- γ — 折扣因子（LLM RLHF 中通常设为 1.0）
- λ — GAE 参数（控制偏差-方差权衡，通常 0.95）

关于 PPO 和 GAE 的详细推导，见 [[ppo-for-llm]]。

#### PPO 训练循环

每次训练迭代包含以下步骤：

```
对于每个训练批次:
  1. 从提示词集合中采样一批 prompts
  2. 用当前策略 π_θ 生成回复 y ~ π_θ(·|x)
  3. 用奖励模型评分: r = r_φ(x, y)
  4. 计算 KL 惩罚: kl_t = log[π_θ(y_t|...) / π_ref(y_t|...)]
  5. 计算调整后的奖励: r_total = r - β · Σ kl_t
  6. 用 Critic 估计每个 token 的价值 V(s_t)
  7. 用 GAE 计算优势函数 A_t
  8. 执行 K 轮 PPO 更新（通常 K=1~4）:
     - 计算概率比 r_t(θ) = π_θ(y_t|...) / π_old(y_t|...)
     - 裁剪替代目标: L = min(r_t·A_t, clip(r_t, 1-ε, 1+ε)·A_t)
     - 更新 Actor 和 Critic
```

---

## 数学公式推导

### Bradley-Terry 偏好模型

**假设**：每个回复 y 有一个潜在的"质量分数" r(x, y)。人类在比较时，选择 y_w 的概率遵循 logistic 模型：

```
P(y_w ≻ y_l | x) = exp(r(x, y_w)) / [exp(r(x, y_w)) + exp(r(x, y_l))]
                  = 1 / [1 + exp(-(r(x, y_w) - r(x, y_l)))]
                  = σ(r(x, y_w) - r(x, y_l))
```

这与 Elo 评分系统数学上完全等价（Elo 使用 base-10 对数，Bradley-Terry 使用自然对数）。

### 奖励模型损失函数

```
L_RM(φ) = -E_{(x, y_w, y_l)} [log σ(r_φ(x, y_w) - r_φ(x, y_l))]
```

**梯度**：

```
∂L/∂φ = -E [(1 - σ(r_φ(y_w) - r_φ(y_l))) · (∂r_φ(y_w)/∂φ - ∂r_φ(y_l)/∂φ)]
```

直觉：当模型对偏好判断不自信时（σ 接近 0.5），梯度更大，推动模型拉大好回复与差回复的分数差距。

### RL 优化目标

```
max_θ  J(θ) = E_{x~D, y~π_θ} [r_φ(x,y)] - β · E_{x~D} [KL(π_θ(·|x) || π_ref(·|x))]
```

展开 KL 散度：

```
KL(π_θ || π_ref) = E_{y~π_θ} [log π_θ(y|x) - log π_ref(y|x)]
                 = Σ_t E [log π_θ(y_t|x,y_{<t}) - log π_ref(y_t|x,y_{<t})]
```

因此总奖励可以写成逐 token 的形式：

```
r_total = r_φ(x,y) - β · Σ_t [log π_θ(y_t|x,y_{<t}) - log π_ref(y_t|x,y_{<t})]
```

只有序列最后一个 token 位置获得 RM 奖励，中间 token 只有 KL 惩罚项。

### GAE 优势估计

```
δ_t = r_t + γ · V(s_{t+1}) - V(s_t)

A_t^GAE = Σ_{l=0}^{∞} (γλ)^l · δ_{t+l}
        = δ_t + γλ · δ_{t+1} + (γλ)^2 · δ_{t+2} + ...
```

当 λ=0 时：A_t = δ_t = r_t + γV(s_{t+1}) - V(s_t)（高偏差，低方差）
当 λ=1 时：A_t = Σ γ^l r_{t+l} - V(s_t)（低偏差，高方差 — Monte Carlo）

实践中 λ=0.95 是常见选择。

---

## RLHF 的变体与演进

### Online RLHF vs. Offline RLHF

| 维度 | Online RLHF | Offline RLHF |
|------|-------------|--------------|
| 数据来源 | 当前策略实时生成 | 预先收集的固定数据集 |
| 代表算法 | PPO, GRPO | DPO, IPO, KTO |
| 奖励模型 | 在线使用，可能随策略更新 | 不需要显式 RM |
| 优势 | 探索能力强，避免分布外问题 | 实现简单，训练稳定 |
| 劣势 | 计算开销大，实现复杂 | 受限于离线数据分布 |

**趋势**：在线方法在复杂任务（推理、代码、智能体）上表现更优；离线方法在简单对齐任务上性价比更高。

### RLAIF — AI 反馈替代人类反馈

**Constitutional AI (Anthropic, 2023)**：
- 用 AI 模型替代人类标注员进行偏好判断
- AI 根据预设的"宪法原则"（如"回答应该无害"）评判回复质量
- 大幅降低标注成本，提高可扩展性
- 核心发现：AI 反馈在很多场景下与人类反馈效果相当

**RLAIF 工作流**：
```
原始回复 → AI 根据原则修订 → 修订后回复 vs. 原始回复 → AI 判断偏好 → 训练 RM → PPO
```

### RLVR — 可验证奖励的强化学习

对于数学和代码等可验证任务，可以完全绕过人类反馈和学习的奖励模型：

```
奖励 = { 1.0  如果答案正确（通过验证器验证）
        { 0.0  如果答案错误
```

**代表性工作**：
- **DeepSeek-R1**：用答案正确性作为奖励，训练出强大的推理能力（见 [[grpo]]）
- **数学任务**：答案与标准答案比较
- **代码任务**：通过单元测试验证
- 参见 [[reward-modeling#RLVR|RLVR 详解]]

RLVR 的优势：奖励信号完全准确（无噪声），避免了奖励模型的偏差。
RLVR 的局限：只适用于有明确正确答案的任务。

### 迭代 RLHF (Iterative RLHF)

标准 RLHF 是"一次性"流程：收集数据 → 训练 RM → RL → 完成。
迭代 RLHF 将此过程循环进行：

```
SFT 模型 → RLHF 轮次 1 → 新策略
    ↓                       ↓
用新策略生成回复 → 收集新的人类反馈 → 更新 RM → RLHF 轮次 2 → ...
```

优势：
- 奖励模型在当前策略的分布上训练，避免分布偏移
- 策略持续改进
- 更接近真实的在线学习范式

### Best-of-N 采样 (Rejection Sampling)

最简单的"RLHF"方法 — 不需要 RL：

```
1. 对同一个 prompt，用策略生成 N 个回复
2. 用奖励模型对每个回复打分
3. 选择分数最高的回复
```

**性质**：
- 推理时计算量增加 N 倍，但不需要 RL 训练
- Best-of-N 的有效 KL 惩罚约为 log(N) - (N-1)/N
- 常用作 RLHF 的基线方法
- DeepSeek-R1 训练中也使用了 Best-of-N 进行数据筛选

---

## 代码示例

### 使用 TRL 库的 RLHF 训练循环

```python
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer

# === 阶段准备：加载模型 ===
model = AutoModelForCausalLMWithValueHead.from_pretrained("sft_model_path")
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained("sft_model_path")
tokenizer = AutoTokenizer.from_pretrained("sft_model_path")
reward_model = load_reward_model("rm_model_path")  # 自定义加载

# === PPO 配置 ===
ppo_config = PPOConfig(
    model_name="sft_model",
    learning_rate=1.41e-5,
    batch_size=64,
    mini_batch_size=16,
    ppo_epochs=4,              # 每批数据的 PPO 更新轮数
    kl_penalty="kl",           # KL 惩罚类型
    init_kl_coef=0.2,          # 初始 KL 系数 β
    target_kl=6.0,             # 目标 KL 值（自适应调节 β）
    cliprange=0.2,             # PPO 裁剪参数 ε
    cliprange_value=0.2,       # 价值函数裁剪参数
    gamma=1.0,                 # 折扣因子
    lam=0.95,                  # GAE λ 参数
)

ppo_trainer = PPOTrainer(ppo_config, model, ref_model, tokenizer)

# === 训练循环 ===
for epoch in range(num_epochs):
    for batch in dataloader:
        # 1. 生成回复
        query_tensors = [tokenizer.encode(q, return_tensors="pt") for q in batch["query"]]
        response_tensors = ppo_trainer.generate(query_tensors, max_new_tokens=256)
        
        # 2. 计算奖励
        texts = [tokenizer.decode(r.squeeze()) for r in response_tensors]
        rewards = [reward_model.score(q, r) for q, r in zip(batch["query"], texts)]
        rewards = [torch.tensor(r) for r in rewards]
        
        # 3. PPO 更新（内部自动计算 KL、GAE、裁剪目标）
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        
        # 4. 日志
        print(f"mean_reward: {stats['ppo/mean_scores']:.3f}, "
              f"kl: {stats['ppo/mean_non_score_reward']:.3f}")
```

### 奖励模型训练代码

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class RewardModel(nn.Module):
    """基于 Bradley-Terry 模型的奖励模型"""
    
    def __init__(self, base_model_name):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(base_model_name)
        self.reward_head = nn.Linear(self.backbone.config.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # 取最后一个 token 的隐藏状态
        last_hidden = outputs.last_hidden_state
        # 找到每个序列的最后一个非 padding token
        seq_lengths = attention_mask.sum(dim=1) - 1
        last_token_hidden = last_hidden[range(len(seq_lengths)), seq_lengths]
        reward = self.reward_head(last_token_hidden).squeeze(-1)
        return reward

def compute_rm_loss(reward_model, chosen_ids, chosen_mask, rejected_ids, rejected_mask):
    """Bradley-Terry 损失函数"""
    r_chosen = reward_model(chosen_ids, chosen_mask)      # (batch,)
    r_rejected = reward_model(rejected_ids, rejected_mask) # (batch,)
    
    # L = -E[log σ(r_chosen - r_rejected)]
    loss = -torch.log(torch.sigmoid(r_chosen - r_rejected)).mean()
    
    # 准确率：r_chosen > r_rejected 的比例
    accuracy = (r_chosen > r_rejected).float().mean()
    
    return loss, accuracy

# 训练循环
optimizer = torch.optim.AdamW(reward_model.parameters(), lr=1e-5, weight_decay=0.01)

for epoch in range(num_epochs):
    for batch in preference_dataloader:
        loss, acc = compute_rm_loss(
            reward_model,
            batch["chosen_ids"], batch["chosen_mask"],
            batch["rejected_ids"], batch["rejected_mask"]
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reward_model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        print(f"RM Loss: {loss.item():.4f}, Accuracy: {acc.item():.3f}")
```

---

## 挑战与开放问题

### 1. 人类标注的瓶颈

- **成本高昂**：高质量标注需要领域专家，数学/代码标注尤其昂贵
- **标注者间分歧**：不同标注者对同一对比较的判断经常不一致（agreement 通常只有 70-80%）
- **系统性偏差**：标注者倾向于偏好更长、更正式、更符合"助手风格"的回复，而非内容质量
- **不可扩展性**：人类标注速度有限，成为训练速度的瓶颈

### 2. 奖励模型的局限

- **[[reward-modeling|奖励模型]]质量决定了最终策略的性能上限** — RM 的错误会被 RL 放大
- **分布外泛化差**：RM 在训练分布内准确，但对新颖的回复风格可能给出不可靠的分数
- **奖励模型大小困境**：太小 → 表达能力不足；太大 → 计算开销高
- **多维度奖励**：单一标量难以捕捉 helpfulness、harmlessness、honesty 等多个维度

### 3. [[reward-modeling#奖励黑客|奖励黑客]]

策略学会利用奖励模型的弱点获得高分，而非真正改善回复质量：
- 过于冗长（RM 可能偏好长回复）
- 使用 RM 偏好的特定短语或格式
- "表面讨好"而非真正有帮助
- 随着 RL 训练进行，这种倾向会被不断放大

**缓解策略**：KL 惩罚、奖励模型集成（多个 RM 投票）、定期更新 RM

### 4. 训练稳定性

- KL 系数 β 的调节极为困难 — 过低 → 奖励黑客，过高 → 策略几乎不更新
- PPO 的超参数（学习率、裁剪范围、批次大小等）需要仔细调节
- 训练过程中奖励可能突然崩溃或饱和
- "Alignment tax"：RLHF 可能在某些维度上降低模型能力

### 5. 对齐税与能力-安全权衡

- RLHF 可能让模型在某些任务（如创意写作、角色扮演）上变得"过度对齐"，过于保守
- 安全性与有用性之间存在根本张力
- 过度训练可能导致"sycophancy"（阿谀奉承）— 模型倾向于同意用户而非给出真实回答

### 6. 可扩展监督问题 (Scalable Oversight)

- 当模型能力超越人类评估者时，如何保证 RLHF 仍然有效？
- 超人类 AI 的回答可能超出人类判断能力
- 这是长期 AI 安全的核心问题之一
- 可能的方向：AI 辅助评估、辩论（debate）、递归奖励建模

---

## 参考文献

- Christiano et al. (2017) — [Deep RL from Human Preferences](https://arxiv.org/abs/1706.03741)
- Ziegler et al. (2019) — [Fine-Tuning Language Models from Human Preferences](https://arxiv.org/abs/1909.08593)
- Stiennon et al. (2020) — [Learning to Summarize with Human Feedback](https://arxiv.org/abs/2009.01325)
- Bai et al. (2022) — [Training a Helpful and Harmless Assistant from Human Feedback](https://arxiv.org/abs/2204.05862)
- Ouyang et al. (2022) — [InstructGPT](https://arxiv.org/abs/2203.02155)
- Bai et al. (2022) — [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)
- Rafailov et al. (2023) — [DPO: Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- Zhou et al. (2023) — [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206)
- Lambert (2025) — [RLHF Book](https://rlhfbook.com/book.pdf) ([arXiv:2504.12501](https://arxiv.org/abs/2504.12501))
- ACM Computing Surveys (2025) — [RLHF Deciphered](https://dl.acm.org/doi/10.1145/3743127)

---

## 相关页面

- [[ppo-for-llm]] — RLHF 中使用的核心 RL 算法，详细推导 PPO 目标函数
- [[grpo]] — 无需 Critic 模型的 PPO 替代方案，DeepSeek 提出
- [[dpo]] — 直接偏好优化，完全绕过奖励建模
- [[reward-modeling]] — 奖励模型的工作原理与训练方法详解
- [[rl-training-frameworks]] — RLHF 训练框架（OpenRLHF, veRL, TRL 等）
- [[multi-step-reasoning-rl]] — RLHF/GRPO 在推理模型中的应用
