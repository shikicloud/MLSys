---
title: "DPO：直接偏好优化"
category: rl-infra
tags: [dpo, 偏好优化, 对齐, 离线rl, simpo, kto, ipo, orpo]
created: 2026-04-13
updated: 2026-04-14
status: mature
---

# DPO：直接偏好优化

## 概述

DPO（Direct Preference Optimization，Rafailov et al., 2023）是 LLM 对齐领域的一项里程碑式工作。它的核心洞察在于：**完全跳过奖励模型的训练**，直接在人类偏好数据上用监督学习目标优化语言模型。

传统 [[rlhf-overview|RLHF]] 流水线需要三个阶段：
1. SFT（监督微调）
2. 训练奖励模型（Reward Model）
3. 用 [[ppo-for-llm|PPO]] 等 RL 算法优化策略

DPO 将第 2 步和第 3 步合并为一步：直接在偏好对 (prompt, chosen, rejected) 上训练。整个训练过程只需 **2 个模型**（策略模型 + 冻结的参考模型），而 PPO 需要 4 个模型（actor, critic, reference, reward model）。

```
传统 RLHF 流水线：
┌──────────────────────────────────────────────────────────────────┐
│  人类偏好数据 ──→ 训练奖励模型 ──→ PPO 优化策略                    │
│  (prompt, y_w, y_l)    (RM)         (Actor + Critic + Ref + RM)  │
│                                     4 个模型，训练不稳定            │
└──────────────────────────────────────────────────────────────────┘

DPO 流水线：
┌──────────────────────────────────────────────────────────────────┐
│  人类偏好数据 ──→ 直接优化策略                                      │
│  (prompt, y_w, y_l)   (Policy + Ref)                             │
│                        2 个模型，稳定的监督学习                      │
└──────────────────────────────────────────────────────────────────┘
```

这一简化带来了显著的工程优势：更少的 GPU 显存占用、无需调 RL 超参、训练稳定性大幅提升。DPO 论文一经发布便迅速成为最流行的对齐方法之一。

---

## 从 RLHF 到 DPO 的推导

DPO 的数学推导是理解其本质的关键。下面我们从 RLHF 的目标函数出发，逐步推导出 DPO 的损失函数。

### 第一步：RLHF 的优化目标

RLHF 的核心目标是最大化奖励的同时，防止策略偏离参考模型太远。数学上表示为：

```
max_π  E_{x~D, y~π(·|x)} [ r(x, y) ] - β · KL( π(·|x) || π_ref(·|x) )
```

其中：
- `r(x, y)` 是（学习到的）奖励模型对回复 y 的评分
- `π(·|x)` 是我们要优化的策略
- `π_ref(·|x)` 是参考策略（通常是 SFT 后的模型）
- `β` 是控制 KL 惩罚强度的温度参数
- `KL` 散度防止策略退化（reward hacking）

### 第二步：求解最优策略的封闭形式

将 KL 散度展开：

```
KL(π || π_ref) = E_{y~π} [ log π(y|x) - log π_ref(y|x) ]
```

代入优化目标并展开：

```
max_π  E_{y~π} [ r(x,y) - β · log π(y|x) + β · log π_ref(y|x) ]
     = max_π  E_{y~π} [ r(x,y) + β · log π_ref(y|x) - β · log π(y|x) ]
```

这是一个标准的最大熵 RL 问题。对 π(y|x) 求变分最优解，令导数为零：

```
r(x,y) + β · log π_ref(y|x) - β · log π(y|x) - β = 0
```

（其中最后的 `-β` 来自归一化约束的拉格朗日乘子）

整理得到最优策略的封闭形式：

```
π*(y|x) = (1/Z(x)) · π_ref(y|x) · exp( r(x,y) / β )
```

其中 `Z(x)` 是配分函数（归一化常数）：

```
Z(x) = Σ_y π_ref(y|x) · exp( r(x,y) / β )
```

**直觉理解**：最优策略是参考策略按照奖励的指数进行加权。奖励高的回复概率被放大，奖励低的被缩小。

### 第三步：用策略表示奖励

这是 DPO 最关键的一步。将上面的最优策略公式取对数并重排：

```
log π*(y|x) = log π_ref(y|x) + r(x,y)/β - log Z(x)
```

解出 `r(x,y)`：

```
r(x, y) = β · log( π*(y|x) / π_ref(y|x) ) + β · log Z(x)
```

**核心洞察**：奖励函数可以完全用策略和参考策略的对数概率比来表示！配分函数 `Z(x)` 只依赖于 prompt x，不依赖于 response y。

### 第四步：代入 Bradley-Terry 偏好模型

Bradley-Terry 模型描述了人类偏好概率：

```
P(y_w ≻ y_l | x) = σ( r(x, y_w) - r(x, y_l) )
```

其中 `σ` 是 sigmoid 函数，`y_w` 是优选（chosen）响应，`y_l` 是被拒绝（rejected）响应。

将第三步的奖励表达式代入：

```
r(x, y_w) - r(x, y_l) 
= β · log(π(y_w|x) / π_ref(y_w|x)) + β·log Z(x) 
  - β · log(π(y_l|x) / π_ref(y_l|x)) - β·log Z(x)
```

**关键：`β · log Z(x)` 相互抵消！**

```
r(x, y_w) - r(x, y_l) = β · [ log(π(y_w|x)/π_ref(y_w|x)) - log(π(y_l|x)/π_ref(y_l|x)) ]
```

这意味着我们不需要计算不可处理的配分函数 Z(x)。

### 第五步：最终 DPO 损失函数

将上式代入 Bradley-Terry 模型的负对数似然（即最大化偏好数据的似然），得到 DPO 的训练目标：

```
L_DPO(π_θ; π_ref) = -E_{(x, y_w, y_l) ~ D} [ 
    log σ( β · ( log(π_θ(y_w|x) / π_ref(y_w|x)) 
                - log(π_θ(y_l|x) / π_ref(y_l|x)) ) ) 
]
```

更紧凑的写法，定义隐式奖励差：

```
Δ = β · log(π_θ(y_w|x)/π_ref(y_w|x)) - β · log(π_θ(y_l|x)/π_ref(y_l|x))

L_DPO = -E [ log σ(Δ) ]
```

这就是一个标准的**二元交叉熵损失**，目标是让模型对优选响应的隐式奖励（相对于参考模型的对数概率比）高于被拒绝响应。

### 推导总结

```
RLHF 目标 ──→ 最优策略封闭解 ──→ 用策略表示奖励 ──→ 代入 BT 模型
   │                │                    │                  │
   │           π* ∝ π_ref·exp(r/β)    r = β·log(π/π_ref)   Z(x) 抵消
   │                │                 + β·log Z(x)          │
   └──→ 不需要训练 RM ←──── 不需要计算 Z(x) ←──── DPO 损失函数
```

---

## DPO 的直觉理解

### 梯度分析

DPO 损失函数的梯度告诉我们它在做什么：

```
∇_θ L_DPO ∝ -β · E [ σ(-Δ) · (
    ∇_θ log π_θ(y_w|x) - ∇_θ log π_θ(y_l|x)
)]
```

其中 `σ(-Δ)` 是一个加权项：
- 当模型已经"答对了"（Δ 大，即对优选响应已分配更高概率），`σ(-Δ)` 接近 0，梯度很小 → **不再更新**
- 当模型"答错了"（Δ 小或为负），`σ(-Δ)` 接近 1，梯度很大 → **大幅更新**

这意味着 DPO 同时做两件事：
1. **提升**优选响应 `y_w` 的概率（`+∇_θ log π_θ(y_w|x)` 方向）
2. **降低**被拒绝响应 `y_l` 的概率（`-∇_θ log π_θ(y_l|x)` 方向）
3. 两者的力度由当前"错误程度"动态调节

### 参考模型的角色

```
DPO 训练流程：
                        ┌─────────────┐
  偏好数据              │  参考模型     │ (冻结)
  (x, y_w, y_l) ──────→│  π_ref       │──→ log π_ref(y_w|x), log π_ref(y_l|x)
       │                └─────────────┘              │
       │                ┌─────────────┐              │
       └───────────────→│  策略模型     │              │
                        │  π_θ        │──→ log π_θ(y_w|x), log π_θ(y_l|x)
                        └──────┬──────┘              │
                               │                      │
                               ▼                      ▼
                        ┌──────────────────────────────┐
                        │  计算隐式奖励差 Δ             │
                        │  Δ = β·[log(π_θ/π_ref)(y_w)  │
                        │      - log(π_θ/π_ref)(y_l)]  │
                        └──────────────┬───────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │  L = -log σ(Δ)               │
                        │  反向传播，更新 π_θ            │
                        └──────────────────────────────┘
```

参考模型的作用类似于"锚点"：
- 防止策略模型发生灾难性遗忘
- 限制模型偏离原始分布的幅度
- 没有参考模型的约束，优化可能导致模型退化（输出变得极端/重复）

### 与 RLHF 的直觉对比

```
RLHF：先学一个"评委"(RM)，再让模型在"评委"指导下练习(PPO)
        → 评委可能有偏见，练习过程可能不稳定

DPO： 直接从"比赛结果"(偏好数据)中学习，跳过评委
        → 更直接，但只能从历史比赛中学习(离线)
```

---

## 实现细节

### 数据格式

DPO 的训练数据是三元组：`(prompt, chosen, rejected)`

```python
# 典型的 DPO 数据集格式
{
    "prompt": "解释什么是量子纠缠",
    "chosen": "量子纠缠是一种量子力学现象，当两个粒子...(高质量回复)",
    "rejected": "量子纠缠就是两个东西连在一起...(低质量回复)"
}
```

**数据来源**：
1. **人类标注**：人类对同一 prompt 的多个回复进行排序
2. **AI 反馈**：用强模型（GPT-4, Claude）对回复打分
3. **自我对弈**：模型生成多个回复，用规则/验证器挑选
4. **现有数据集**：UltraFeedback、Anthropic-HH、Stanford Human Preferences

### 参考模型管理

```python
# 方式 1：冻结副本（最常见）
ref_model = AutoModelForCausalLM.from_pretrained("sft_model")
ref_model.eval()
for param in ref_model.parameters():
    param.requires_grad = False

# 方式 2：定期更新参考模型（Online DPO）
# 每 N 步将策略模型的权重复制给参考模型
if step % update_interval == 0:
    ref_model.load_state_dict(policy_model.state_dict())

# 方式 3：无参考模型（SimPO, ORPO）
# 直接用策略模型自身的平均 log 概率作为奖励
```

**显存优化**：参考模型不需要梯度，可以用半精度加载甚至量化：

```python
ref_model = AutoModelForCausalLM.from_pretrained(
    "sft_model",
    torch_dtype=torch.bfloat16,
    load_in_4bit=True,  # 量化以节省显存
)
```

### β 参数的影响

`β` 是 DPO 中最重要的超参数，控制 KL 惩罚的强度：

| β 值 | 行为 | 适用场景 |
|------|------|---------|
| β → 0 | 完全忽略参考模型，激进优化 | 几乎不用，容易退化 |
| β = 0.05-0.1 | 较强的偏好学习 | 偏好信号清晰的数据 |
| β = 0.1-0.5 | 常用范围 | 大多数场景 |
| β = 0.5-1.0 | 保守更新，紧贴参考 | 噪声较多的偏好数据 |
| β → ∞ | 完全不更新 | 无意义 |

**调参策略**：
- 从 `β = 0.1` 开始
- 如果训练不稳定或输出退化，增大 β
- 如果模型变化太小（对齐效果弱），减小 β
- 监控 chosen 和 rejected 的 log 概率差：应该在训练中逐渐增大

### 标签平滑（Label Smoothing）

人类标注的偏好对并非总是正确。标签平滑可以提高对噪声标签的鲁棒性：

```python
# 标准 DPO 损失
loss = -log(σ(Δ))

# 带标签平滑的 DPO 损失 (ε ∈ [0, 0.5])
loss = -(1-ε) · log(σ(Δ)) - ε · log(σ(-Δ))
```

TRL 的 DPOTrainer 支持 `label_smoothing` 参数：

```python
training_args = DPOConfig(
    label_smoothing=0.1,  # 10% 的标签平滑
    ...
)
```

### 训练中的关键指标

```
需要监控的指标：
┌──────────────────────────────────────────────────────────────┐
│  1. loss（应该下降）                                          │
│  2. rewards/chosen（chosen 的隐式奖励，应该上升）               │
│  3. rewards/rejected（rejected 的隐式奖励，应该下降）           │
│  4. rewards/margins（chosen - rejected，应该增大）             │
│  5. rewards/accuracies（模型判断正确的比例，应该上升）           │
│  6. logps/chosen（chosen 的 log 概率，不应该大幅下降！）         │
│  7. logps/rejected（rejected 的 log 概率，应该下降）            │
└──────────────────────────────────────────────────────────────┘
```

**特别注意**：如果 `logps/chosen` 大幅下降，说明模型在降低优选响应的概率——这就是"优选响应退化"问题。

---

## DPO 的变体

DPO 催生了大量变体，每种都针对不同的局限性进行改进。

### IPO (Identity Preference Optimization)

**论文**：Azar et al. (Google DeepMind), 2023

**动机**：DPO 基于 Bradley-Terry 偏好模型的假设过强——它假设人类偏好可以精确地用奖励差的 sigmoid 来建模。当偏好数据有噪声时，DPO 可能过拟合。

**方法**：不再假设 Bradley-Terry 模型，而是直接对偏好概率施加正则化约束：

```
L_IPO = E [ (log(π_θ(y_w|x)/π_ref(y_w|x)) - log(π_θ(y_l|x)/π_ref(y_l|x)) - 1/(2β))² ]
```

**优势**：
- 对噪声标签更鲁棒
- 可以安全地训练到收敛（DPO 过度训练会退化）
- 不依赖 Bradley-Terry 假设

### KTO (Kahneman-Tversky Optimization)

**论文**：Ethayarajh et al. (Stanford), 2024

**动机**：收集成对偏好数据（同一 prompt 的 chosen 和 rejected）成本很高。能否只用二元反馈（好/坏）来训练？

**核心思想**：借鉴行为经济学中的前景理论（Prospect Theory）：
- 人类对损失比收益更敏感（损失厌恶）
- 效用函数不是线性的

```
KTO 损失：
L_KTO = E_w [ w(x,y) · (1 - σ(β · log(π_θ(y|x)/π_ref(y|x)) - z_ref)) ]  (好回复)
      + E_l [ w(x,y) · (1 - σ(z_ref - β · log(π_θ(y|x)/π_ref(y|x)))) ]  (差回复)
```

其中 `z_ref` 是参考点（来自参考策略的基线估计），`w(x,y)` 是损失厌恶系数。

**优势**：
- 只需**未配对**的二元反馈，无需成对数据
- 数据效率更高，可以利用更多低成本数据
- 在数据稀缺的场景中特别有用

### ORPO (Odds Ratio Preference Optimization)

**论文**：Hong et al. (KAIST), 2024

**动机**：DPO 仍需要分两步——先 SFT 再 DPO。能否合并为一步？

**方法**：将 SFT 损失和偏好优化合并，用优势比（odds ratio）来衡量偏好：

```
L_ORPO = L_SFT + λ · L_OR

L_OR = -E [ log σ( log(odds(y_w|x) / odds(y_l|x)) ) ]
odds(y|x) = π_θ(y|x) / (1 - π_θ(y|x))
```

**优势**：
- **无需参考模型**，进一步减少显存
- **SFT + 偏好学习一步完成**
- 简化训练流程

### SimPO (Simple Preference Optimization)

**论文**：Meng et al. (Princeton), NeurIPS 2024

**动机**：DPO 的隐式奖励（`log(π/π_ref)`）在推理时不可用，因为推理时不使用参考模型。SimPO 认为应该用推理时实际使用的度量（sequence log probability）作为奖励。

**方法**：用**平均 log 概率**作为隐式奖励，无需参考模型：

```
r_SimPO(x, y) = (1/|y|) · log π_θ(y|x)

L_SimPO = -E [ log σ( β · (r_SimPO(x, y_w) - r_SimPO(x, y_l)) - γ ) ]
```

其中 `γ > 0` 是一个目标奖励间距（target reward margin）。

**优势**：
- 无参考模型（更少显存）
- 长度归一化，避免偏向长回复
- AlpacaEval 2 上比 DPO +6.4 LC win-rate
- Arena-Hard 上比 DPO +7.5

### Online DPO / Iterative DPO

**动机**：标准 DPO 是离线的，只能从固定的偏好数据集学习。这导致分布不匹配：训练数据来自 SFT 模型，但训练中策略模型的分布在变化。

**方法**：
1. 用当前策略模型生成新的回复
2. 用奖励模型（或人类、AI）对新回复标注偏好
3. 在新数据上进行 DPO 训练
4. 重复

```
Online DPO 循环：
┌─────────────────────────────────────────────────┐
│  π_θ 生成新回复 ──→ RM/人类标注偏好 ──→ DPO 训练  │
│      ▲                                    │      │
│      └────────────────────────────────────┘      │
│                    迭代重复                        │
└─────────────────────────────────────────────────┘
```

**注意**：Online DPO 引入了奖励模型，部分失去了 DPO "无需 RM" 的优势，但解决了分布不匹配问题。

### 变体对比表

| 变体 | 需要成对数据？ | 需要参考模型？ | 需要 SFT 阶段？ | 核心创新 |
|------|:------------:|:------------:|:--------------:|---------|
| **DPO** | 是 | 是 | 是 | 用策略表示奖励 |
| **IPO** | 是 | 是 | 是 | 正则化替代 BT 模型 |
| **KTO** | **否** (二元) | 是 | 是 | 前景理论 |
| **ORPO** | 是 | **否** | **否** (合并) | 优势比 + SFT 合一 |
| **SimPO** | 是 | **否** | 是 | 平均 log prob + 间距 |
| **Online DPO** | 在线生成 | 是 | 是 | 迭代生成新数据 |
| **BPO** (ICLR 2026) | 是 | 是 | 是 | 修复 chosen 退化 |

---

## DPO vs RLHF/PPO

### 优势

| 维度 | DPO | PPO |
|------|-----|-----|
| 模型数量 | 2 (策略 + 参考) | 4 (actor + critic + 参考 + RM) |
| GPU 显存 | 低 | 高（约 2-3x） |
| 训练稳定性 | 高（监督学习） | 低（RL 训练不稳定） |
| 超参敏感度 | 低（主要是 β） | 高（学习率、clip range、GAE 等） |
| 实现复杂度 | 简单 | 复杂 |
| 训练速度 | 快 | 慢（generation + training loop） |

### 劣势

| 维度 | DPO | PPO |
|------|-----|-----|
| 数据来源 | 离线（固定数据集） | 在线（生成新数据） |
| 探索能力 | 无 | 有 |
| 分布匹配 | 容易不匹配 | 天然匹配（on-policy） |
| 多轮/智能体 | 不自然扩展 | 天然支持 |
| 奖励信号 | 隐式 | 显式、灵活 |
| 推理任务 | 表现较弱 | 更强（可探索） |

### 选择指南

```
选择 DPO 当：
  ✓ 有高质量的成对偏好数据
  ✓ 单轮对话对齐（helpfulness, harmlessness）
  ✓ 计算资源有限
  ✓ 需要快速迭代
  ✓ 团队缺乏 RL 工程经验

选择 PPO/GRPO 当：
  ✓ 推理任务（数学、代码）
  ✓ 智能体/工具使用场景
  ✓ 多轮交互
  ✓ 需要在线探索
  ✓ 有可验证的奖励（RLVR）
  ✓ 追求最佳性能
```

**ICML 2024 关键发现**：PPO 在正确调优后可以匹配或超越 DPO，但工程开销巨大。

**2025-2026 趋势**：
- 简单对齐任务 → DPO 及其变体（SimPO, KTO）
- 推理和智能体任务 → [[grpo|GRPO]]、[[ppo-for-llm|PPO]]、REINFORCE++
- DeepSeek-R1 选择了 [[grpo|GRPO]] 而非 DPO 进行推理训练

---

## 代码示例

### 使用 TRL DPOTrainer

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

# 1. 加载模型
model_name = "meta-llama/Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="bfloat16",
    attn_implementation="flash_attention_2",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 2. 加载偏好数据
# 数据集格式: {"prompt": str, "chosen": str, "rejected": str}
dataset = load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned")

# 3. 配置 DPO 训练
training_args = DPOConfig(
    output_dir="./dpo_output",
    beta=0.1,                    # KL 惩罚强度
    learning_rate=5e-7,          # DPO 通常用较小的学习率
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=1,          # DPO 通常只训练 1 个 epoch
    warmup_ratio=0.1,
    logging_steps=10,
    bf16=True,
    gradient_checkpointing=True,
    label_smoothing=0.0,         # 可选：对噪声数据设为 0.1
    max_length=2048,
    max_prompt_length=1024,
)

# 4. 初始化 DPOTrainer（自动创建参考模型）
trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    processing_class=tokenizer,
    # ref_model=None 时 TRL 自动创建冻结副本
)

# 5. 训练
trainer.train()

# 6. 保存
trainer.save_model("./dpo_final")
```

### 数据准备示例

```python
def prepare_dpo_data(raw_dataset):
    """
    将原始标注数据转换为 DPO 格式
    假设原始数据包含 prompt 和多个带评分的回复
    """
    dpo_data = []
    for item in raw_dataset:
        prompt = item["prompt"]
        responses = item["responses"]
        scores = item["scores"]
        
        # 按分数排序
        sorted_pairs = sorted(zip(responses, scores), 
                              key=lambda x: x[1], reverse=True)
        
        # 取最高和最低分作为 chosen/rejected
        chosen = sorted_pairs[0][0]
        rejected = sorted_pairs[-1][0]
        
        # 确保分数差异足够大（过滤噪声对）
        if sorted_pairs[0][1] - sorted_pairs[-1][1] > 0.5:
            dpo_data.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            })
    
    return dpo_data
```

### β 调参策略

```python
# β 网格搜索示例
import wandb

betas = [0.05, 0.1, 0.2, 0.5]
for beta in betas:
    wandb.init(project="dpo-beta-search", name=f"beta_{beta}")
    
    training_args = DPOConfig(
        beta=beta,
        output_dir=f"./dpo_beta_{beta}",
        num_train_epochs=1,
        # ... 其他参数
    )
    
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
    )
    trainer.train()
    
    # 评估：检查 win-rate 和 logps/chosen 是否退化
    metrics = trainer.evaluate()
    wandb.log(metrics)
    wandb.finish()
```

### TRL CLI 快速启动 (v1.0+)

```bash
# 一行命令启动 DPO 训练
trl dpo \
  --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
  --dataset_name argilla/ultrafeedback-binarized-preferences-cleaned \
  --beta 0.1 \
  --learning_rate 5e-7 \
  --output_dir ./dpo_output \
  --bf16 \
  --gradient_checkpointing
```

---

## 不足与争论

### "DPO 本质上是离线 RL"

DPO 虽然看起来像监督学习，但在数学上等价于一种特定的离线 RL 算法。这意味着它继承了离线 RL 的所有问题：

1. **分布不匹配**：偏好数据来自 SFT 模型或人类，但策略模型在训练中不断变化。如果策略偏离数据生成分布太远，学到的偏好信号就不准确了。

2. **缺乏探索**：DPO 永远不会探索数据集之外的行为空间。对于需要创造性推理的任务（数学、代码），这是致命缺陷。

### 优选响应退化问题

实证发现，DPO 在降低 rejected 响应概率的同时，也可能降低 chosen 响应的概率——即使损失函数理论上应该提升 chosen 的概率。

```
理想情况：                    实际可能发生的：
  chosen ↑                     chosen ↓ (轻微)
  rejected ↓                   rejected ↓↓ (大幅)
  margin 增大 ✓                margin 仍然增大 ✓ (但 chosen 变差了)
```

这是因为 DPO 优化的是 chosen 和 rejected 之间的**相对差异**，而不是 chosen 的**绝对质量**。BPO（ICLR 2026）专门针对此问题提出了修复方案。

### 在线 vs 离线的争论

| 立场 | 论据 |
|------|------|
| DPO 够用 | 简单、高效、在 chat alignment 上表现好 |
| 需要在线 RL | 离线数据覆盖不够、推理需要探索、DeepSeek-R1 的成功证明了在线 RL |

### 为什么 DeepSeek 选择 GRPO 而非 DPO

DeepSeek-R1（2025）是推理模型的标杆之作，它选择了 [[grpo|GRPO]] 而非 DPO，原因包括：

1. **在线探索**：推理需要模型尝试不同的思路，DPO 的离线数据无法提供
2. **可验证奖励**：数学和代码可以用确定性验证器（[[reward-modeling#RLVR|RLVR]]），无需学习奖励模型
3. **自举**：模型需要从自身生成的好/坏推理中持续学习
4. **长链推理**：DPO 对长序列的偏好学习效果不佳

### 当前共识（2025-2026）

```
DPO 及变体的定位：
  ✓ 通用对话对齐（helpfulness, harmlessness, style）
  ✓ 快速原型验证
  ✓ 资源有限时的首选

在线 RL (PPO/GRPO) 的定位：
  ✓ 推理（数学、代码、逻辑）
  ✓ 智能体和工具使用
  ✓ 追求 SOTA 性能
  ✓ 自我提升/自我对弈

混合方法越来越常见：
  先 DPO 做基础对齐 → 再用 GRPO/PPO 做推理/智能体强化
```

---

## 参考文献

- Rafailov et al. (2023) — [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
- Azar et al. (2023) — [A General Theoretical Paradigm to Understand Learning from Human Feedback (IPO)](https://arxiv.org/abs/2310.12036)
- Ethayarajh et al. (2024) — [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306)
- Hong et al. (2024) — [ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691)
- Meng et al. (2024) — [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734)
- Xu et al. (2024) — [Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study](https://arxiv.org/abs/2404.10719) (ICML 2024)
- DPO 变体综合综述 — [arXiv:2410.15595](https://arxiv.org/html/2410.15595v3)

---

## 相关页面

- [[ppo-for-llm]] — 在线 RL 替代方案，DPO 试图简化的方法
- [[grpo]] — 无 Critic 的在线 RL，DeepSeek-R1 的核心算法
- [[reward-modeling]] — DPO 所绕过的环节
- [[rlhf-overview]] — DPO 所简化的完整 RLHF 流水线
- [[rl-training-frameworks]] — 支持 DPO 训练的框架（TRL, OpenRLHF, veRL）
- [[multi-step-reasoning-rl]] — DPO 在推理场景中的局限性
