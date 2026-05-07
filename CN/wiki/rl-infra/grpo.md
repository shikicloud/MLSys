---
title: "GRPO：组相对策略优化"
category: rl-infra
tags: [grpo, deepseek, 强化学习, 无critic, deepseek-r1, deepseek-math, 策略优化, 推理]
created: 2026-04-13
updated: 2026-04-14
status: mature
---

# GRPO：组相对策略优化

> [!abstract]+ TL;DR
> GRPO（DeepSeek, 2024）是**无 Critic 的策略优化算法**：不再训练价值函数，而是对同一 prompt 采样一组回复，**用组内奖励统计作为基线**。把 [[ppo-for-llm|PPO]] 的 4 模型架构（actor + critic + ref + reward）简化为 2–3 模型，内存和计算量减少 **~50 %**，推理和对齐任务上达到相当或更优的性能。因 **DeepSeek-R1** 大放异彩，迅速成为 2024–2025 大规模 RL 训练的主流，尤其是推理模型训练。

---

## 动机：为什么要去掉 Critic？

### PPO 的内存瓶颈

[[ppo-for-llm|PPO-based RLHF]] 需要四个模型同时在内存中：

```
PPO 四模型:                        GRPO 简化:
┌──────────────┐                  ┌──────────────┐
│ Actor (可训练) │                  │ Actor (可训练) │
├──────────────┤                  ├──────────────┤
│ Critic (可训练)│  ← GRPO 去掉    │              │
├──────────────┤                  ├──────────────┤
│ 参考模型 (冻结) │                  │ 参考模型 (冻结) │
├──────────────┤                  ├──────────────┤
│ 奖励模型 (冻结) │                  │ 奖励模型 (冻结) │
└──────────────┘                  └──────────────┘
内存: ~4× 模型大小                   内存: ~3× 模型大小
                                   (+ rollout 缓存)
```

对于 70B 模型，去掉 Critic 直接节省约 140GB（fp16）的 GPU 内存。

### Critic 训练的不稳定性

Critic 模型在 LLM RLHF 中存在固有问题：

1. **奖励稀疏**：RM 只在序列末尾给出奖励，Critic 需要在所有中间位置预测期望回报，这是一个困难的信用分配问题
2. **序列长度变化大**：不同回复的长度差异可以从几十 token 到数千 token，Critic 在不同长度上的泛化能力有限
3. **训练目标不一致**：Critic 的训练目标（预测回报）与 Actor 的目标（生成好回复）不同步
4. **初始化问题**：Critic 从 SFT 模型初始化，但其最终任务与语言建模无关

### GRPO 的关键洞察

> **"我们不需要学习一个基线（Critic），可以直接从采样中估计基线。"**

对于每个 prompt，如果我们采样足够多的回复：
- 这些回复的平均奖励就是期望回报的无偏估计
- 用 (r_i - mean) / std 归一化后得到的就是优势的近似
- 这在统计上等价于使用一个"完美的" prompt-specific 基线

---

## 算法详解

### 算法流程

```
输入: 策略 π_θ, 参考策略 π_ref, 提示词集合 D, 组大小 G

对于每个训练迭代:
  1. 从 D 中采样一批提示词 {x_1, ..., x_B}
  
  2. 对每个提示词 x_i:
     a. 从当前策略采样 G 个回复: {y_i^1, ..., y_i^G} ~ π_θ(·|x_i)
     b. 计算每个回复的奖励: r_i^j = R(x_i, y_i^j)   (j = 1,...,G)
  
  3. 计算组相对优势:
     μ_i = mean({r_i^1, ..., r_i^G})
     σ_i = std({r_i^1, ..., r_i^G})
     A_i^j = (r_i^j - μ_i) / σ_i              (组内归一化)
  
  4. 计算 GRPO 目标并更新策略:
     L_GRPO = E_i,j [ min(ρ_i^j · A_i^j, clip(ρ_i^j, 1-ε, 1+ε) · A_i^j) 
                       - β · KL(π_θ || π_ref) ]
     
     其中 ρ_i^j = π_θ(y_i^j|x_i) / π_old(y_i^j|x_i)
  
  5. 更新 θ 以最大化 L_GRPO
```

### 核心公式推导

#### 步骤 1：组内采样与评分

对于提示词 x，从当前策略 π_θ 采样 G 个独立回复：

```
y_1, y_2, ..., y_G  ~  π_θ(·|x)     (i.i.d. 采样)
r_j = R(x, y_j)                       (奖励评分)
```

#### 步骤 2：组相对优势计算

```
μ = (1/G) Σ_{j=1}^{G} r_j            (组内平均奖励)
σ = sqrt[(1/G) Σ_{j=1}^{G} (r_j - μ)^2]   (组内标准差)

A_j = (r_j - μ) / σ                   (归一化优势)
```

**为什么这是合理的？**

在 PPO 中，优势函数 A(s,a) = Q(s,a) - V(s)。对于序列级奖励：
- Q(x, y_j) ≈ r_j（回复 y_j 的奖励就是 Q 值的估计）
- V(x) ≈ μ = E_{y~π_θ}[r(x,y)]（组平均奖励是期望回报的无偏估计）
- 因此 A_j ≈ r_j - μ（与 PPO 的优势含义一致）

标准差归一化 (/ σ) 的作用是：
- 使优势在不同 prompt 之间可比较（有些 prompt 的奖励方差大，有些小）
- 防止高方差 prompt 主导梯度
- 类似于优势白化（advantage whitening）

#### 步骤 3：裁剪策略梯度

GRPO 复用了 PPO 的裁剪替代目标：

```
L_GRPO(θ) = (1/B) Σ_{i=1}^{B} (1/G) Σ_{j=1}^{G} 
            min(ρ_{ij}(θ) · A_{ij}, clip(ρ_{ij}(θ), 1-ε, 1+ε) · A_{ij})
```

其中概率比 ρ 的计算：

```
ρ_{ij}(θ) = π_θ(y_i^j | x_i) / π_old(y_i^j | x_i)
           = exp[Σ_t log π_θ(y_{i,t}^j | x_i, y_{i,<t}^j) 
                 - Σ_t log π_old(y_{i,t}^j | x_i, y_{i,<t}^j)]
```

**注意**：这里的 ρ 是序列级的概率比（所有 token 的概率比之积），而 PPO 中的 ρ 是 token 级的。

#### 步骤 4：KL 正则化

GRPO 使用 KL 散度防止策略偏离参考策略太远：

```
KL_j = Σ_t [log π_θ(y_{j,t}|x, y_{j,<t}) - log π_ref(y_{j,t}|x, y_{j,<t})]
```

**DeepSeek-R1 的 KL 实现**：使用了近似 KL 散度（非精确 KL）：

```
KL_approx = (π_ref / π_θ) - log(π_ref / π_θ) - 1
```

这个形式在 π_θ 偏离 π_ref 时惩罚更对称。

#### 完整的 GRPO 优化目标

```
max_θ  L_GRPO(θ) = E_{x~D} [ (1/G) Σ_{j=1}^{G} 
    min(ρ_j · A_j, clip(ρ_j, 1-ε, 1+ε) · A_j) - β · KL_j ]
```

### GRPO 流水线 ASCII 图

```
┌─────────────────────────────────────────────────────────────────────┐
│                         GRPO 训练流水线                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  对于每个提示词 x:                                                    │
│                                                                     │
│  ┌──────────┐   采样 G 个回复   ┌─────────────────────────────┐     │
│  │ 当前策略   │ ──────────────> │  y_1, y_2, ..., y_G          │     │
│  │ π_θ      │                  │  (G=16~64)                   │     │
│  └──────────┘                  └──────────┬──────────────────┘     │
│                                           │                         │
│                                           ▼                         │
│                                ┌─────────────────────┐             │
│                                │   奖励评分            │             │
│                                │  r_1, r_2, ..., r_G  │             │
│                                │  (RM 或 规则验证器)    │             │
│                                └──────────┬──────────┘             │
│                                           │                         │
│                                           ▼                         │
│                                ┌─────────────────────┐             │
│                                │   组内归一化          │             │
│                                │  μ = mean(r)         │             │
│                                │  σ = std(r)          │             │
│                                │  A_j = (r_j-μ)/σ    │             │
│                                └──────────┬──────────┘             │
│                                           │                         │
│                                           ▼                         │
│  ┌──────────┐  KL 惩罚   ┌────────────────────────────────┐       │
│  │ 参考策略   │ ────────> │  裁剪策略梯度更新                 │       │
│  │ π_ref    │            │  L = min(ρ·A, clip(ρ)·A) - β·KL│       │
│  │ (冻结)    │            │  更新 θ                         │       │
│  └──────────┘            └────────────────────────────────┘       │
│                                                                     │
│  无需 Critic! 组均值替代了学习的价值函数。                               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 与 PPO 的对比

### 目标函数并排对比

```
PPO:
  L_PPO = E_t [min(ρ_t · A_t^GAE, clip(ρ_t, 1-ε, 1+ε) · A_t^GAE)]
  
  其中 A_t^GAE 由 Critic V(s_t) 和 GAE 计算（token 级）

GRPO:
  L_GRPO = E_{i,j} [min(ρ_j · A_j^group, clip(ρ_j, 1-ε, 1+ε) · A_j^group)]
  
  其中 A_j^group = (r_j - mean(r)) / std(r)（序列级）
```

### 系统性对比

| 维度 | PPO | GRPO |
|------|-----|------|
| **模型数量** | 4（Actor, Critic, Ref, RM） | 3（Actor, Ref, RM）或 2（RLVR 无 RM） |
| **优势估计** | Token 级 GAE（由 Critic 支持） | 序列级组归一化 |
| **基线来源** | 学习的 Critic V(s) | 组内平均奖励 μ |
| **内存占用(7B)** | ~150GB | ~110GB |
| **裁剪维度** | Token 级概率比 | 序列级概率比 |
| **采样开销** | 每 prompt 1 个回复 | 每 prompt G 个回复（G=16~64） |
| **优势精度** | 高（token 级信用分配） | 中等（只有序列级信号） |
| **训练稳定性** | 中等（Critic 可能不稳定） | 高（无 Critic 训练） |
| **超参数** | 多（lr_actor, lr_critic, ε, β, γ, λ, K） | 较少（lr, ε, β, G） |
| **实现复杂度** | 高 | 中等 |

### 什么时候 GRPO 优于 PPO？

- **大模型**（>30B 参数）：内存节省至关重要
- **推理任务**：奖励信号清晰（对/错），不需要 token 级信用分配
- **代码/数学**：可验证奖励，GRPO 的组内比较天然适配
- **快速迭代**：更少的超参数，更容易调优

### 什么时候 PPO 可能更优？

- **细粒度控制**：需要 token 级别的信用分配（如风格控制、安全约束）
- **小样本**：每个 prompt 只能生成少量回复（组大小受限）
- **非常长的序列**：序列级优势可能信息损失大
- **已有成熟的 PPO 基础设施**：如果团队已经有调优好的 PPO 流水线

---

## GRPO 在 DeepSeek 中的应用

### DeepSeek-R1-Zero：纯 RL 的惊人发现

DeepSeek-R1-Zero 是一个里程碑式的实验，直接在 base 模型上（无 SFT）使用 GRPO + 可验证奖励训练。

**训练设置**：
- 基座模型：DeepSeek-V3-Base
- 奖励信号：仅答案正确性（数学题对/错）+ 格式奖励（要求 `<think>...</think>` 标签）
- 无人类标注、无 SFT、无奖励模型
- 使用 GRPO 优化

**涌现现象（Emergent Behaviors）**：

```
训练进程:
早期 ──────────────────────────────── 中期 ──────────────────────── 后期
  ↓                                    ↓                           ↓
直接给答案                            开始分步骤                     复杂推理链
（通常错误）                           简单验证                      自我反思
                                                                   "等等，让我重新检查..."
                                                                   多策略尝试
                                                                   "aha moment"
```

**"Aha Moment"**：训练过程中，模型自发地学会了：
1. **思维链推理**（Chain-of-Thought）— 无需示范
2. **自我反思**（Self-reflection）— "Wait, let me reconsider..."
3. **自我验证**（Self-verification）— 检查自己的答案
4. **策略切换**（Strategy switching）— 一种方法不行就换另一种

这些能力完全是从纯 RL 中涌现的，没有任何人类示范。

**R1-Zero 的局限**：
- 可读性差（混合语言、格式混乱）
- 有时出现无限循环推理
- 仅在可验证任务上有效

### DeepSeek-R1：完整流水线

DeepSeek-R1 在 R1-Zero 的基础上，使用了更完整的训练流水线：

```
阶段 1: 冷启动 SFT
  - 收集少量高质量 CoT 示范数据（人类编写 + R1-Zero 筛选）
  - 在 base 模型上做 SFT，教会模型基本的推理格式
  
阶段 2: 大规模 GRPO
  - 在可验证任务（数学、代码、逻辑）上用规则奖励
  - 在开放式任务上用学习的奖励模型
  - 大规模 GRPO 训练

阶段 3: 拒绝采样 + SFT
  - 用 GRPO 训练后的模型生成大量推理数据
  - 过滤高质量样本（正确答案 + 良好格式）
  - 在这些数据上做 SFT，提高输出质量和可读性

阶段 4: 第二轮 GRPO
  - 继续 RL 训练，进一步优化
```

### 训练参数（来自 DeepSeek-R1 论文）

| 参数 | 值 | 说明 |
|------|-----|------|
| 学习率 | 3e-6 → 1e-6（余弦衰减） | 较低，防止大模型不稳定 |
| KL 系数 β | 0.001 | 非常小，允许充分探索 |
| 裁剪比率 ε | 0.2 (后期用 10) | 后期放大允许更大更新 |
| 采样温度 | 1.0 | 保证多样性 |
| 组大小 G | 64 | 每个 prompt 采样 64 个回复 |
| 每步提示词数 | 16 | |
| 最大生成长度 | 32,768 tokens | 支持长推理链 |
| 每批次策略更新 | 1（单轮） | 避免过拟合 |

**裁剪比率的特殊处理**：DeepSeek-R1 在后期使用了很大的裁剪比率（ε=10），这与标准 PPO（ε=0.2）差异巨大。这表明在训练后期，更大的更新幅度是有益的。

### 奖励信号设计

| 任务类型 | 奖励来源 | 奖励值 |
|---------|---------|-------|
| 数学 | 答案匹配 | r=1（正确）, r=0（错误） |
| 代码 | 单元测试通过 | r=1（全部通过）, r=0（失败） |
| 格式 | 正则匹配 `<think>...</think><answer>...</answer>` | r=+0.5（正确格式）, r=-0.5（错误格式） |
| 开放式 | 学习的奖励模型 | 连续分数 |

---

## 代码示例

### GRPO 训练循环伪代码

```python
import torch
import torch.nn.functional as F

class GRPOTrainer:
    """GRPO 训练器（简化版）"""
    
    def __init__(self, policy, ref_policy, reward_fn,
                 group_size=16, lr=1e-5, clip_eps=0.2, 
                 kl_coef=0.01):
        self.policy = policy           # 可训练
        self.ref_policy = ref_policy   # 冻结
        self.reward_fn = reward_fn     # RM 或 规则验证器
        
        self.G = group_size
        self.clip_eps = clip_eps
        self.kl_coef = kl_coef
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    
    def train_step(self, prompts):
        """一个 GRPO 训练步骤"""
        batch_size = len(prompts)
        
        # === 步骤 1: 采样 G 个回复 ===
        all_responses = []
        all_old_log_probs = []
        
        with torch.no_grad():
            for prompt in prompts:
                # 对每个 prompt 采样 G 个回复
                responses, log_probs = self.policy.generate(
                    prompt, 
                    num_samples=self.G,
                    temperature=1.0,
                    return_log_probs=True,
                )
                all_responses.append(responses)       # (G, seq_len)
                all_old_log_probs.append(log_probs)   # (G,) 序列级
        
        # === 步骤 2: 计算奖励 ===
        all_rewards = []
        for prompt, responses in zip(prompts, all_responses):
            rewards = self.reward_fn(prompt, responses)  # (G,)
            all_rewards.append(rewards)
        
        # === 步骤 3: 组内归一化 → 优势 ===
        all_advantages = []
        for rewards in all_rewards:
            mu = rewards.mean()
            sigma = rewards.std() + 1e-8
            advantages = (rewards - mu) / sigma   # (G,)
            all_advantages.append(advantages)
        
        # === 步骤 4: PPO 风格的裁剪更新 ===
        self.optimizer.zero_grad()
        total_loss = 0
        
        for i in range(batch_size):
            prompt = prompts[i]
            responses = all_responses[i]      # (G, seq_len)
            old_lp = all_old_log_probs[i]     # (G,)
            advantages = all_advantages[i]     # (G,)
            
            # 重新计算当前策略的 log_probs
            new_lp = self.policy.log_probs(prompt, responses)  # (G,)
            
            # 概率比
            ratio = torch.exp(new_lp - old_lp)  # (G,)
            
            # 裁剪目标
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-self.clip_eps, 
                                1+self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # KL 惩罚
            ref_lp = self.ref_policy.log_probs(prompt, responses)
            kl = (new_lp - ref_lp).mean()
            
            total_loss += policy_loss + self.kl_coef * kl
        
        total_loss /= batch_size
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
        
        # 统计
        mean_reward = torch.cat(all_rewards).mean().item()
        return {"loss": total_loss.item(), "mean_reward": mean_reward}
```

### 使用 TRL GRPOTrainer

```python
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

# === 模型加载 ===
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
)

# === 定义奖励函数 ===
def reward_function(completions, **kwargs):
    """
    基于规则的奖励函数（数学任务）
    
    Args:
        completions: 模型生成的回复列表
    Returns:
        rewards: 奖励值列表
    """
    rewards = []
    for completion in completions:
        # 检查格式
        has_think = "<think>" in completion and "</think>" in completion
        # 提取答案
        answer = extract_answer(completion)
        # 验证正确性
        correct = verify_answer(answer, ground_truth)
        
        reward = 0.0
        if has_think:
            reward += 0.5    # 格式奖励
        if correct:
            reward += 1.0    # 正确性奖励
        rewards.append(reward)
    return rewards

# === GRPO 配置 ===
config = GRPOConfig(
    output_dir="grpo_output",
    
    # --- 核心 GRPO 参数 ---
    num_generations=16,            # 组大小 G
    
    # --- PPO 风格参数 ---
    learning_rate=5e-6,
    cliprange=0.2,
    
    # --- KL 惩罚 ---
    beta=0.01,                     # KL 系数
    
    # --- 批次参数 ---
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    
    # --- 生成参数 ---
    max_completion_length=2048,
    temperature=1.0,
    
    # --- 其他 ---
    logging_steps=10,
    save_steps=100,
    bf16=True,
)

# === 创建训练器并训练 ===
trainer = GRPOTrainer(
    model=model,
    config=config,
    tokenizer=tokenizer,
    train_dataset=math_dataset,
    reward_funcs=reward_function,
)

trainer.train()
```

### 用于推理任务的配置

```python
# === 针对推理任务的 GRPO 配置 ===
reasoning_config = GRPOConfig(
    # 更大的组大小（推理任务方差大）
    num_generations=64,
    
    # 更长的生成长度（支持长推理链）
    max_completion_length=8192,
    
    # 更低的 KL 惩罚（允许更多探索）
    beta=0.001,
    
    # 更大的裁剪范围（推理任务需要更大的策略变化）
    cliprange=0.2,   # 可以在后期增大
    
    # 温度 1.0 保证多样性
    temperature=1.0,
    
    # 使用 bf16 节省内存
    bf16=True,
    
    # 梯度检查点（长序列必须）
    gradient_checkpointing=True,
)
```

---

## GRPO 的变体与改进

### DAPO（Decoupled Clip and Dynamic Sampling Policy Optimization）

**来源**：字节跳动 Seed + 清华 AIR ([arXiv:2503.14476](https://arxiv.org/abs/2503.14476))

DAPO 在 GRPO 的基础上提出了四个关键改进：

#### 1. Clip-Higher（非对称裁剪）

标准 PPO/GRPO 使用对称裁剪 [1-ε_low, 1+ε_high]，其中 ε_low = ε_high。
DAPO 使用非对称裁剪：

```
ε_low = 0.2（正常值）
ε_high = 0.28（更大的上限）
```

**直觉**：对称裁剪会导致"熵崩溃" — 策略很快变得确定性，失去探索能力。增大上限裁剪范围允许策略更容易增加低概率动作的概率，促进多样性。

#### 2. Dynamic Sampling（动态采样）

当一个组内所有回复都正确（或都错误）时，优势全为零，这个组对训练没有贡献。DAPO 动态过滤这些"无信息"的组：

```python
# 只保留有信息量的组
useful_groups = [g for g in groups if 0 < sum(g.rewards) < len(g.rewards)]
```

这大幅提高了训练效率。

#### 3. Token-Level Policy Gradient Loss（Token 级策略梯度损失）

标准 GRPO 使用序列级概率比：

```
ρ_seq = Π_t π_θ(y_t|...) / π_old(y_t|...)
```

DAPO 改为 token 级损失并做序列长度归一化：

```
L = (1/T) Σ_t min(ρ_t · A, clip(ρ_t, 1-ε, 1+ε) · A)
```

这对长思维链至关重要 — 否则长序列的梯度会远大于短序列。

#### 4. Overlong Reward Shaping（超长奖励整形）

当生成超过最大长度被截断时，直接给 r=0 会引入噪声（可能已经接近正确答案）。DAPO 使用渐进式惩罚：

```
r_overlong = max(min_reward, r_original - penalty * (len - max_len))
```

**DAPO 性能**：在 AIME 2024 达到 50 分（DeepSeek-R1-Zero 为 47 分），使用 50% 更少的训练步骤。使用 [[rl-training-frameworks#veRL|veRL]] 训练。

### Dr. GRPO（Variance-Reduced GRPO）

**问题**：标准 GRPO 的组归一化引入了偏差。当组大小 G 有限时：

```
E[A_j] = E[(r_j - μ) / σ] ≠ 0     (偏差非零)
```

这是因为分母中的 σ 是随机变量。

**Dr. GRPO 的修正**：
- 去掉标准差归一化：A_j = r_j - μ（只减去均值）
- 在 loss 计算中加入方差修正项
- 理论上无偏，实践中训练更稳定

### RLOO（REINFORCE Leave-One-Out）

另一种与 GRPO 相关的方法，使用 leave-one-out 基线：

```
对于第 j 个样本：
  baseline_j = (1/(G-1)) Σ_{k≠j} r_k     (排除自身的平均值)
  A_j = r_j - baseline_j
```

这比 GRPO 的组均值基线方差更低（因为基线与被评估的样本独立）。

### REINFORCE++ 与 GRPO 的关系

| 特性 | GRPO | REINFORCE++ |
|------|------|-------------|
| 基线 | 组内均值 | 运行均值（exponential moving average） |
| 采样 | 每 prompt G 个 | 每 prompt 1 个 |
| 归一化 | 组内 std 归一化 | 全局方差归一化 |
| 计算开销 | 高（G 倍 rollout） | 低（单次 rollout） |
| 适用场景 | 大规模推理 RL | 一般 RLHF |

---

## 性能分析

### 基准对比

| 模型/方法 | AIME 2024 | MATH-500 | Codeforces | 训练成本 |
|-----------|-----------|----------|------------|---------|
| DeepSeek-R1-Zero (GRPO, 纯 RL) | 71.0% | 95.9% | 1444 Elo | 基准 |
| DeepSeek-R1 (GRPO, 完整流水线) | 79.8% | 97.3% | 2029 Elo | ~2× |
| OpenAI o1 | 79.2% | 96.4% | 2061 Elo | 未公开 |
| DAPO (GRPO 改进) | ~50/90 * | 类似 | - | ~0.5× R1-Zero |

*AIME 分数制度不同

### GRPO vs. PPO 性能

根据多个独立研究（Yu et al. 2025, Hu et al. 2025）：

- 在推理任务（数学、代码）上：GRPO ≈ PPO（甚至略优）
- 在对齐任务（helpfulness、harmlessness）上：GRPO ≈ PPO（需要足够大的组）
- 训练速度：GRPO 快 30-50%（无 Critic 开销）
- 调优难度：GRPO 显著更容易（更少超参数）

### 组大小对性能的影响

```
       性能
  ↑    ___________________
  |   /
  |  /
  | /     性能随组大小增大而提升
  |/      但边际收益递减
  |
  +──────────────────────> 组大小 G
  1    8    16   32   64   128

  推荐:
  - 简单任务: G=8~16
  - 推理任务: G=32~64
  - 大于 64 通常边际收益很小
```

---

## 不足与局限

### 1. 组大小敏感性

- **组大小 G 太小**（如 G=2~4）：组统计量噪声大，优势估计不可靠，归一化偏差显著
- **组大小 G 太大**（如 G>64）：每个 prompt 需要大量 rollout，计算成本高
- 最优 G 取决于任务复杂度和奖励分布 — 需要经验调优

### 2. 奖励模型依赖

- 对于非可验证任务，GRPO 仍然依赖学习的奖励模型
- 奖励模型的偏差会被组内比较放大（如果所有 G 个回复都有同样的偏差方向）
- 组内归一化无法修正系统性的 RM 偏差

### 3. 序列级优势的信息损失

- GRPO 只有序列级的优势信号，不像 PPO 有 token 级的信用分配
- 对于"前半段好但后半段差"的回复，无法给出精确的反馈
- 长序列中，哪些 token 贡献了奖励是模糊的

### 4. 采样效率

- 每个 prompt 需要采样 G 个回复，推理计算量是 PPO 的 G 倍
- 对于长序列（如 32K tokens），rollout 生成是训练的主要瓶颈
- 需要高效的推理引擎（如 vLLM）来加速采样

### 5. 组内多样性要求

- GRPO 依赖于组内回复的质量差异来产生有意义的优势信号
- 如果策略已经很强（大部分回复都正确），组内差异消失，训练信号变弱
- 这是 DAPO 提出 Dynamic Sampling 的动机

### 6. 理论保证有限

- GRPO 的组归一化引入了偏差（有限样本统计量的偏差）
- 收敛性证明比 PPO 弱
- 与 PPO 的最优性差距的理论上界尚不明确

---

## 参考文献

- Shao et al. (2024) — [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)（GRPO 原始论文）
- DeepSeek-AI (2025) — [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
- DeepSeek-R1 Nature 2025 — [doi:10.1038/s41586-025-09422-z](https://www.nature.com/articles/s41586-025-09422-z)
- Yu et al. (2025) — [Revisiting GRPO: On-Policy and Off-Policy](https://arxiv.org/html/2505.22257v1)
- DAPO (2025) — [arXiv:2503.14476](https://arxiv.org/abs/2503.14476)
- Hu et al. (2025) — [REINFORCE++](https://arxiv.org/abs/2501.03262)
- Ahmadian et al. (2024) — [Back to Basics: Revisiting REINFORCE-Style Optimization](https://arxiv.org/abs/2402.14740)
- Liu et al. (2025) — [Understanding R1-Zero-Like Training](https://arxiv.org/abs/2503.20783)

---

## 相关页面

- [[ppo-for-llm]] — 基于 Critic 的 PPO 方法，GRPO 的前身
- [[rlhf-overview]] — 完整的 RLHF 三阶段流水线
- [[dpo]] — 直接偏好优化，另一种简化 RLHF 的方法
- [[reward-modeling]] — 奖励信号的来源：学习的 RM 与规则验证器
- [[rl-training-frameworks]] — 支持 GRPO 的训练框架（veRL, OpenRLHF, TRL）
- [[multi-step-reasoning-rl]] — GRPO 在推理模型中的大规模应用
