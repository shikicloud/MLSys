---
title: "PPO 用于 LLM 对齐"
category: rl-infra
tags: [ppo, 强化学习, 对齐, rlhf, 策略优化, 近端策略优化, gae, critic]
created: 2026-04-13
updated: 2026-04-14
status: mature
---

# PPO 用于 LLM 对齐

## 概述

近端策略优化（Proximal Policy Optimization, PPO），由 Schulman et al. (2017) 提出，是使 [[rlhf-overview|RLHF]] 在大语言模型上切实可行的核心 RL 算法。它通过裁剪替代目标（clipped surrogate objective）提供稳定的策略更新，防止破坏性的大幅参数变动。PPO 驱动了 InstructGPT、ChatGPT 和早期的 Claude 模型。

**为什么 PPO 而不是其他 RL 算法？**

- 语言模型是巨大的策略网络（数十亿参数），一次不稳定的更新就可能永久损坏模型
- PPO 通过裁剪机制自然地限制了每步更新的幅度
- 相比 TRPO，PPO 实现简单得多（不需要二阶优化），且计算效率更高
- PPO 在 Atari/MuJoCo 等标准 RL 基准上已经被充分验证

**PPO 在 LLM 对齐中的地位变迁**：PPO 曾是 RLHF 的唯一选择（2019-2023），但随着 DPO（2023）和 [[grpo]]（2024）的出现，PPO 不再是唯一方案。然而，在需要在线探索的复杂任务（推理、代码生成、工具使用）中，PPO/GRPO 仍然是不可替代的。

---

## PPO 算法回顾

### 策略梯度基础

RL 的目标是最大化期望累积奖励：

```
J(θ) = E_{τ~π_θ} [Σ_t γ^t · r_t]
```

策略梯度定理（Policy Gradient Theorem）告诉我们：

```
∇_θ J(θ) = E_{τ~π_θ} [Σ_t ∇_θ log π_θ(a_t|s_t) · A_t]
```

其中 A_t 是优势函数（advantage function）。这就是 REINFORCE 算法的基础。

**问题**：REINFORCE 的梯度方差极大，需要大量样本才能稳定训练。

### 从 TRPO 到 PPO 的演进

**TRPO (Trust Region Policy Optimization, Schulman et al. 2015)**

TRPO 的核心思想：限制每次更新的策略变化幅度，确保在"信任域"内更新。

```
max_θ  E [r_t(θ) · A_t]
s.t.   E [KL(π_old || π_θ)] ≤ δ
```

其中 r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t) 是概率比（importance sampling ratio）。

TRPO 需要计算 Fisher 信息矩阵的逆，对于数十亿参数的 LLM 完全不现实。

**PPO — TRPO 的实用化近似**

PPO 用裁剪机制替代硬约束，将受约束的优化问题转化为无约束问题：

```
L^CLIP(θ) = E_t [min(r_t(θ) · A_t, clip(r_t(θ), 1-ε, 1+ε) · A_t)]
```

这个设计天才般地简洁：
- 当 A_t > 0（好动作）时：r_t 不能超过 1+ε，防止过度增加好动作的概率
- 当 A_t < 0（坏动作）时：r_t 不能低于 1-ε，防止过度降低坏动作的概率
- 效果等价于在信任域内优化，但计算成本极低

### 裁剪替代目标详解

```
L^CLIP(θ) = E_t [min(r_t(θ) · A_t, clip(r_t(θ), 1-ε, 1+ε) · A_t)]
```

其中：
- `r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t)` — 新旧策略的概率比
- `A_t` — 优势函数估计（通常通过 GAE 计算）
- `ε` — 裁剪参数（通常 0.1~0.2）

**裁剪的几何直觉**：

```
                    A_t > 0 (好动作)
目标函数 L          ___________
    ↑              /
    |             /
    |            /
    |───────────/──────────────── r_t(θ)
    |          1-ε    1    1+ε
    |
    |  A_t < 0 (坏动作)
    |───────────────\──────────── r_t(θ)
    |                \___________
    |          1-ε    1    1+ε

裁剪范围 [1-ε, 1+ε] 之外的梯度为零
→ 阻止策略变化过大
```

当 A_t > 0：目标被裁剪在 r_t = 1+ε 处 → 即使增大概率有利，也只允许增大到一定程度。
当 A_t < 0：目标被裁剪在 r_t = 1-ε 处 → 即使减小概率有利，也只允许减小到一定程度。

### 价值函数损失

Critic 模型预测每个状态的期望回报 V(s_t)，其损失函数：

```
L^VF(θ) = E_t [(V_θ(s_t) - V_t^target)^2]
```

其中 V_t^target = A_t^GAE + V_old(s_t)（GAE 估计的回报）。

实践中也常对价值函数损失进行裁剪：

```
L^VF_clipped = E_t [max(
    (V_θ(s_t) - V_t^target)^2,
    (clip(V_θ(s_t), V_old(s_t)-ε_v, V_old(s_t)+ε_v) - V_t^target)^2
)]
```

### 熵奖励 (Entropy Bonus)

为鼓励探索、防止策略过早收敛，PPO 在损失中加入熵奖励：

```
S(π_θ) = -E_t [Σ_a π_θ(a|s_t) log π_θ(a|s_t)]
```

### PPO 完整损失函数

```
L^PPO(θ) = -L^CLIP(θ) + c_1 · L^VF(θ) - c_2 · S(π_θ)
```

其中：
- `c_1` — 价值函数损失系数（通常 0.5~1.0）
- `c_2` — 熵奖励系数（通常 0.01~0.05）
- 负号是因为 L^CLIP 是要最大化的，而总损失要最小化

**注意**：在 LLM RLHF 中，Actor 和 Critic 通常是分开的模型（不共享参数），因此 c_1 和 c_2 的设置与标准 PPO 有所不同。LLM 场景下熵奖励通常不使用或使用很小的值，因为 KL 惩罚已经起到了类似的防崩溃作用。

---

## PPO 在 LLM 中的特殊适配

### LLM 作为策略

在 RLHF 的 PPO 框架中，LLM 被形式化为一个 RL 策略：

| RL 概念 | LLM 对应 |
|---------|---------|
| 状态 s_t | prompt + 已生成的前缀 (x, y_{<t}) |
| 动作 a_t | 下一个 token y_t |
| 策略 π(a\|s) | LLM 的条件概率 π_θ(y_t \| x, y_{<t}) |
| 轨迹 τ | 完整的 (prompt, response) 对 |
| 奖励 | RM 评分（通常只在序列末尾给出） |

### 生成作为序列决策

LLM 的文本生成可以被视为一个 **有限时间步的马尔可夫决策过程（MDP）**：

```
                    Token 1    Token 2    Token 3         Token T
状态:  [prompt] ──> s_1 ────> s_2 ────> s_3 ──...──> s_T
动作:              a_1=y_1   a_2=y_2   a_3=y_3       a_T=y_T
奖励:              -β·kl_1   -β·kl_2   -β·kl_3       r_RM - β·kl_T
```

每一步：
- 状态 = prompt + 已生成的所有 token
- 动作 = 从词表 V 中选择下一个 token
- 中间奖励 = 仅有 KL 惩罚项（-β·kl_t）
- 终端奖励 = RM 评分 + KL 惩罚

### Token 级 vs. 序列级奖励

**序列级奖励**（RM 评分）：只在最后一个 token 给出，所有中间 token 的 RM 奖励为 0。

**Token 级 KL 惩罚**：在每个 token 位置都有：

```
kl_t = log π_θ(y_t|x, y_{<t}) - log π_ref(y_t|x, y_{<t})
```

**总奖励的分配**：

```
r_t = { -β · kl_t                    若 t < T（中间 token）
      { r_RM(x, y) - β · kl_t        若 t = T（最后一个 token）
```

这种奖励稀疏性（大部分奖励集中在序列末尾）是 LLM PPO 的一个核心挑战，需要 GAE 来有效传播信用。

### KL 惩罚的整合

```
r_total(x, y) = r_RM(x, y) - β · Σ_{t=1}^{T} kl_t
```

KL 惩罚的三种实现方式：
1. **逐 token KL**（最常用）：如上所示，在每个 token 位置施加惩罚
2. **序列级 KL**：在序列末尾一次性加上总 KL
3. **自适应 KL**（InstructGPT 方式）：动态调整 β 使实际 KL 接近目标值

```python
# 自适应 KL 系数调节 (InstructGPT)
target_kl = 6.0  # 目标 KL 值

if actual_kl > 1.5 * target_kl:
    beta *= 1.5   # KL 太大，增大惩罚
elif actual_kl < target_kl / 1.5:
    beta /= 1.5   # KL 太小，减小惩罚（允许更多探索）
```

---

## PPO-RLHF 训练循环

```
┌─────────────────────────────────────────────────────────────────┐
│                   PPO-RLHF 训练循环                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────┐  采样提示词   ┌──────────┐  生成回复              │
│  │ 提示词库  │────────────>│  Actor    │───────────┐            │
│  │ (Dataset)│             │  π_θ     │           │            │
│  └─────────┘             └──────────┘           ▼            │
│                                           ┌──────────┐        │
│  ┌─────────────┐                          │ 回复 y   │        │
│  │ 参考模型      │───── KL 惩罚 ──────────>│          │        │
│  │ π_ref (冻结) │                          └────┬─────┘        │
│  └─────────────┘                               │              │
│                                                 ▼              │
│                           ┌──────────┐    ┌──────────┐        │
│                           │ Critic    │    │ 奖励模型  │        │
│                           │ V_φ      │    │ r_ψ(x,y)│        │
│                           └─────┬────┘    └─────┬────┘        │
│                                 │               │              │
│                                 ▼               ▼              │
│                           ┌──────────────────────────┐        │
│                           │ 计算 GAE 优势函数          │        │
│                           │ A_t = Σ(γλ)^l · δ_{t+l} │        │
│                           └────────────┬─────────────┘        │
│                                        │                      │
│                                        ▼                      │
│                           ┌──────────────────────────┐        │
│                           │ PPO 裁剪更新              │        │
│                           │ L = min(r·A, clip(r)·A) │        │
│                           │ 更新 Actor + Critic       │        │
│                           └──────────────────────────┘        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 四模型架构

PPO-based RLHF 需要四个模型同时存在于内存中：

```
┌─────────────────────────────────────────────────────────────────┐
│                      四模型架构                                    │
├──────────────────┬──────────────────────────────────────────────┤
│                  │                                              │
│  可训练模型        │  冻结模型                                     │
│                  │                                              │
│  ┌────────────┐  │  ┌────────────┐                             │
│  │   Actor     │  │  │ 参考模型    │                             │
│  │  (策略 π_θ) │  │  │ (π_ref)   │                             │
│  │  生成回复    │  │  │ 计算 KL   │                             │
│  │  [可训练]   │  │  │ [冻结]     │                             │
│  │  ~14GB(7B) │  │  │ ~14GB(7B) │                             │
│  └────────────┘  │  └────────────┘                             │
│                  │                                              │
│  ┌────────────┐  │  ┌────────────┐                             │
│  │  Critic     │  │  │  奖励模型   │                             │
│  │  (价值 V_φ) │  │  │  (r_ψ)    │                             │
│  │  价值估计    │  │  │  评分      │                             │
│  │  [可训练]   │  │  │  [冻结]     │                             │
│  │  ~14GB(7B) │  │  │  ~14GB(7B) │                             │
│  └────────────┘  │  └────────────┘                             │
│                  │                                              │
├──────────────────┴──────────────────────────────────────────────┤
│  7B 模型总内存需求:                                               │
│  模型参数: 4 × 14GB = 56GB (fp16)                                │
│  优化器状态 (Actor+Critic): ~56GB (Adam: 2× 参数大小)             │
│  激活值/梯度: ~20-40GB (取决于序列长度和批次大小)                    │
│  总计: ~130-150GB → 需要多张 A100 80GB                           │
└─────────────────────────────────────────────────────────────────┘
```

### 各模型详解

| 模型 | 初始化来源 | 是否可训练 | 功能 | 输出 |
|------|----------|-----------|------|------|
| **Actor (策略)** | SFT 模型 | 是 | 生成回复 token | π_θ(y_t \| x, y_{<t}) |
| **Critic (价值函数)** | SFT 模型或 RM | 是 | 估计每个位置的期望回报 | V_φ(s_t) ∈ R |
| **参考模型** | SFT 模型 (冻结) | 否 | 提供 KL 惩罚的基准 | π_ref(y_t \| x, y_{<t}) |
| **奖励模型** | RM 训练得到 (冻结) | 否 | 对完整回复评分 | r_ψ(x, y) ∈ R |

### 内存优化策略

| 策略 | 描述 | 节省 |
|------|------|------|
| **参数共享** | Actor 和 Critic 共享 Transformer backbone | ~14GB |
| **LoRA/QLoRA** | 只训练低秩适配器参数 | 60-90% |
| **模型并行** | 将大模型分布到多张 GPU | 支持更大模型 |
| **卸载到 CPU** | 将不常用模型（参考、RM）放在 CPU 上 | GPU 内存 |
| **量化** | 冻结模型使用 int8/int4 | 50-75% |
| **去掉 Critic** | 使用 [[grpo]] 代替 PPO | ~25% |

---

## GAE（广义优势估计）

### 问题背景

优势函数 A(s_t, a_t) = Q(s_t, a_t) - V(s_t) 衡量"在状态 s_t 采取动作 a_t 比平均好多少"。但直接估计 A 有两种极端方法：

| 方法 | 公式 | 偏差 | 方差 |
|------|------|------|------|
| **TD(0)** | A_t = r_t + γV(s_{t+1}) - V(s_t) | 高（依赖 V 的准确性） | 低 |
| **Monte Carlo** | A_t = Σ_{l=0}^{T-t} γ^l r_{t+l} - V(s_t) | 低 | 高（受全轨迹随机性影响） |

**GAE 在两者之间插值**，通过参数 λ 控制偏差-方差权衡。

### GAE 公式推导

**第一步**：定义 TD 残差

```
δ_t = r_t + γ · V(s_{t+1}) - V(s_t)
```

**第二步**：定义 n-step 优势估计

```
A_t^(1) = δ_t                                          (1-step, 高偏差低方差)
A_t^(2) = δ_t + γ·δ_{t+1}                              (2-step)
A_t^(3) = δ_t + γ·δ_{t+1} + γ^2·δ_{t+2}               (3-step)
...
A_t^(∞) = Σ_{l=0}^{∞} γ^l · δ_{t+l}                    (∞-step, 低偏差高方差)
```

**第三步**：GAE 是所有 n-step 估计的指数加权平均

```
A_t^GAE(γ,λ) = (1-λ)(A_t^(1) + λ·A_t^(2) + λ^2·A_t^(3) + ...)
             = Σ_{l=0}^{∞} (γλ)^l · δ_{t+l}
```

**简洁的递推形式**（实现中常用）：

```
A_T^GAE = δ_T
A_t^GAE = δ_t + γλ · A_{t+1}^GAE     (从后往前递推)
```

### λ 参数的影响

```
λ = 0:   A_t = δ_t = r_t + γV(s_{t+1}) - V(s_t)    ← TD(0)，高偏差
λ = 1:   A_t = Σ γ^l r_{t+l} - V(s_t)               ← Monte Carlo，高方差
λ = 0.95: 实践中最常用的折中值
```

### Token 级 GAE 在 LLM 中的实现

```python
def compute_gae(rewards, values, gamma=1.0, lam=0.95):
    """
    逐 token 计算 GAE 优势估计
    
    Args:
        rewards: shape (batch, seq_len) — 每个 token 位置的奖励
                 (中间 token 只有 -β·kl_t，最后 token 有 r_RM - β·kl_T)
        values:  shape (batch, seq_len) — Critic 在每个位置的价值估计
        gamma:   折扣因子 (LLM 中通常 = 1.0)
        lam:     GAE λ 参数 (通常 0.95)
    
    Returns:
        advantages: shape (batch, seq_len) — GAE 优势估计
        returns:    shape (batch, seq_len) — 目标回报 (用于 Critic 损失)
    """
    batch_size, seq_len = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = 0
    
    for t in reversed(range(seq_len)):
        if t == seq_len - 1:
            next_value = 0  # 序列结束后价值为 0
        else:
            next_value = values[:, t + 1]
        
        # TD 残差
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        
        # GAE 递推
        advantages[:, t] = delta + gamma * lam * last_gae
        last_gae = advantages[:, t]
    
    returns = advantages + values  # V_target = A + V
    return advantages, returns
```

---

## 实现细节与技巧

### 奖励归一化

PPO 对奖励的尺度很敏感。常见的归一化策略：

```python
# 方法 1: 运行均值方差归一化（推荐）
reward_mean = running_mean(rewards)
reward_std = running_std(rewards)
normalized_reward = (reward - reward_mean) / (reward_std + 1e-8)

# 方法 2: 裁剪到固定范围
reward = torch.clamp(reward, -10.0, 10.0)
```

**注意**：InstructGPT 使用了奖励的运行均值方差归一化，并将 KL 惩罚排除在归一化之外。

### 优势归一化

在每个 mini-batch 内对优势进行归一化：

```python
# 每个 mini-batch 内归一化优势
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

这不改变最优策略，但能稳定训练。

### 梯度裁剪

```python
# 全局梯度范数裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 学习率调度

```python
# 常见策略：余弦退火 + 预热
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=total_steps,
)

# InstructGPT: 恒定学习率，无预热
# 也有用线性衰减的
```

### Mini-batch 训练

PPO 的一个关键设计：同一批数据可以用于多轮更新（PPO epochs）：

```python
for ppo_epoch in range(K):  # K 通常 = 1~4
    # 将数据打乱并分成 mini-batches
    indices = torch.randperm(batch_size)
    for mb_start in range(0, batch_size, mini_batch_size):
        mb_indices = indices[mb_start:mb_start + mini_batch_size]
        
        # 计算新的 log_probs 和 values
        new_log_probs = actor(states[mb_indices])
        new_values = critic(states[mb_indices])
        
        # PPO 裁剪更新
        ratio = torch.exp(new_log_probs - old_log_probs[mb_indices])
        surr1 = ratio * advantages[mb_indices]
        surr2 = torch.clamp(ratio, 1-eps, 1+eps) * advantages[mb_indices]
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # Critic 更新
        critic_loss = F.mse_loss(new_values, returns[mb_indices])
        
        # 反向传播并更新
        (actor_loss + 0.5 * critic_loss).backward()
        optimizer.step()
```

**LLM 场景中的注意事项**：
- PPO epochs K 不宜过大（通常 1-2），因为 LLM 的策略空间巨大，多轮更新容易导致过拟合
- Mini-batch 大小受限于 GPU 内存（需要重新前向传播 Actor + Critic）

### 常见陷阱与调试技巧

| 现象 | 可能原因 | 解决方案 |
|------|---------|---------|
| 奖励快速上升然后崩溃 | 奖励黑客（reward hacking） | 增大 KL 惩罚 β；检查 RM 质量 |
| 奖励完全不上升 | 学习率太低/KL 惩罚太大 | 减小 β；增大学习率；检查梯度 |
| KL 散度爆炸 | 策略更新过大 | 减小学习率；减小 ε；增大 β |
| 生成质量下降但奖励上升 | RM 被利用 | 使用 RM 集成；增大 KL 惩罚 |
| Critic loss 不下降 | Critic 模型容量不足/学习率不当 | 增大 Critic 容量；分开调节 Critic 学习率 |
| 训练极不稳定 | 批次大小太小 | 增大 batch size；使用梯度累积 |
| 响应长度不断增加 | RM 偏好长回复 | 加入长度惩罚；修正 RM 训练数据 |

---

## 代码示例

### PPO 训练步骤完整伪代码

```python
import torch
import torch.nn.functional as F

class PPOTrainerForLLM:
    """LLM RLHF 的 PPO 训练器（简化版）"""
    
    def __init__(self, actor, critic, ref_model, reward_model,
                 lr=1e-5, clip_eps=0.2, kl_coef=0.1, 
                 gamma=1.0, lam=0.95, ppo_epochs=2):
        self.actor = actor
        self.critic = critic
        self.ref_model = ref_model      # 冻结
        self.reward_model = reward_model # 冻结
        
        self.clip_eps = clip_eps
        self.kl_coef = kl_coef
        self.gamma = gamma
        self.lam = lam
        self.ppo_epochs = ppo_epochs
        
        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr)
    
    @torch.no_grad()
    def generate_and_score(self, prompts):
        """步骤 1-2: 生成回复并计算奖励"""
        # 1. Actor 生成回复
        responses, log_probs = self.actor.generate(prompts, return_log_probs=True)
        
        # 2. 参考模型计算 log_probs（用于 KL）
        ref_log_probs = self.ref_model.log_probs(prompts, responses)
        
        # 3. 奖励模型评分（序列级）
        rm_scores = self.reward_model.score(prompts, responses)
        
        # 4. 逐 token 计算 KL 惩罚
        kl_per_token = log_probs - ref_log_probs  # shape: (batch, seq_len)
        
        # 5. 构建逐 token 的奖励
        rewards = -self.kl_coef * kl_per_token
        rewards[:, -1] += rm_scores  # RM 奖励只加到最后一个 token
        
        # 6. Critic 估计价值
        values = self.critic(prompts, responses)
        
        # 7. GAE 计算优势
        advantages, returns = compute_gae(
            rewards, values, self.gamma, self.lam
        )
        
        return {
            "responses": responses,
            "old_log_probs": log_probs,
            "advantages": advantages,
            "returns": returns,
            "rm_scores": rm_scores,
        }
    
    def ppo_update(self, prompts, data):
        """步骤 3: PPO 裁剪更新"""
        for epoch in range(self.ppo_epochs):
            # 重新计算 log_probs 和 values（因为参数已更新）
            new_log_probs = self.actor.log_probs(prompts, data["responses"])
            new_values = self.critic(prompts, data["responses"])
            
            # --- Actor 更新 ---
            ratio = torch.exp(new_log_probs - data["old_log_probs"])
            advantages = data["advantages"]
            # 优势归一化
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()
            
            # --- Critic 更新 ---
            critic_loss = F.mse_loss(new_values, data["returns"])
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.critic_optimizer.step()
        
        return {"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item()}
```

### OpenRLHF / TRL PPO 配置

```python
# === TRL PPOConfig 典型配置 ===
from trl import PPOConfig

config = PPOConfig(
    # --- 基础参数 ---
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    learning_rate=1.41e-5,
    
    # --- PPO 特有参数 ---
    ppo_epochs=4,              # 每批数据的 PPO 更新轮数
    cliprange=0.2,             # 策略裁剪参数 ε
    cliprange_value=0.2,       # 价值函数裁剪参数
    
    # --- KL 惩罚 ---
    init_kl_coef=0.2,          # 初始 KL 系数
    target_kl=6.0,             # 目标 KL 值
    kl_penalty="kl",           # KL 惩罚类型: "kl", "abs", "mse"
    
    # --- GAE 参数 ---
    gamma=1.0,                 # 折扣因子
    lam=0.95,                  # GAE λ
    
    # --- 批次参数 ---
    batch_size=64,             # 每步采样的提示词数
    mini_batch_size=16,        # PPO 更新的 mini-batch 大小
    
    # --- 生成参数 ---
    max_new_tokens=256,
    temperature=1.0,
    top_k=0,
    top_p=1.0,
    
    # --- 奖励处理 ---
    whiten_rewards=True,       # 奖励白化（归一化）
)
```

```yaml
# === OpenRLHF PPO 配置示例 (YAML) ===
model:
  actor: "sft_model_path"
  critic: "sft_model_path"  # 或 "rm_model_path"
  reward: "rm_model_path"
  ref: "sft_model_path"

training:
  actor_lr: 9.65e-6
  critic_lr: 5e-6
  kl_coef: 0.02
  clip_range: 0.2
  ppo_epochs: 1
  gamma: 1.0
  gae_lambda: 0.95
  batch_size: 128
  micro_batch_size: 8
  max_seq_len: 2048
  max_new_tokens: 512

# 分布式配置
distributed:
  strategy: "colossalai"  # 或 "deepspeed", "megatron"
  num_gpus: 8
```

---

## PPO vs. 替代方案

| 维度 | PPO | REINFORCE | [[grpo]] | [[dpo]] |
|------|-----|-----------|------|-----|
| **需要 Critic** | 是 | 否 | 否 | 否 |
| **需要奖励模型** | 是（在线使用） | 是（在线使用） | 是（在线使用） | 否（隐式） |
| **内存占用** | 4 个模型 | 2 个模型 | 2 个模型 + rollouts | 2 个模型 |
| **训练稳定性** | 中等 | 低（高方差） | 高 | 非常高 |
| **性能上限** | 高（调优后） | 中等 | 高 | 良好（受限于离线数据） |
| **实现复杂度** | 高 | 低 | 中等 | 低 |
| **超参数数量** | 多（ε, β, γ, λ, lr...） | 少 | 中（组大小, ε, β） | 少（β） |
| **在线探索** | 是 | 是 | 是 | 否 |
| **样本效率** | 高（多轮 PPO epochs） | 低（单次使用） | 中等 | N/A（离线） |
| **适用场景** | 复杂任务、成熟团队 | 简单验证 | 大规模推理 RL | 简单对齐 |

**关键发现**（Xu et al., ICML 2024）：PPO 在正确调优后可以匹配或超越 DPO — 但工程开销巨大。这一发现挑战了"DPO 严格优于 PPO"的简单叙事。

**REINFORCE++ (Hu et al., 2025)**：REINFORCE 的改进版本，加入了 token 级 KL 惩罚、奖励归一化、优势白化等 PPO 的工程技巧，但不需要 Critic。在多个基准上与 PPO 和 GRPO 持平。

---

## 不足与局限

### 1. 四模型内存压力

对于 70B 级别模型，4 个模型的参数就需要约 560GB（fp16），加上优化器状态和激活值，总计需要 1TB+ 的 GPU 内存。这使得 PPO 在大模型上的训练需要大量 GPU 集群。

### 2. 训练不稳定性

PPO for LLM 的训练不稳定是业界公认的难题：
- Actor 和 Critic 需要协调更新，但它们的学习动态不同
- 奖励信号稀疏（只有序列末尾有 RM 分数），信用分配困难
- Critic 在训练初期估计不准，导致优势估计有偏，进而影响 Actor 更新
- "死循环"风险：Critic 估计差 → 优势不准 → Actor 更新方向错误 → 生成质量下降 → Critic 更难学...

### 3. 超参数敏感性

PPO for LLM 有大量需要调节的超参数：

| 超参数 | 典型范围 | 影响 |
|--------|---------|------|
| 学习率 (Actor) | 5e-7 ~ 5e-5 | 太大 → 不稳定，太小 → 不收敛 |
| 学习率 (Critic) | 1e-6 ~ 1e-4 | 通常比 Actor 大 2-10 倍 |
| KL 系数 β | 0.01 ~ 0.5 | 太大 → 过于保守，太小 → 奖励黑客 |
| 裁剪参数 ε | 0.1 ~ 0.3 | 太大 → 不稳定，太小 → 更新太慢 |
| GAE λ | 0.9 ~ 1.0 | 影响偏差-方差权衡 |
| PPO epochs K | 1 ~ 4 | 太大 → 过拟合当前批次 |
| Batch size | 32 ~ 512 | 太小 → 梯度噪声大 |

**经验法则**："让 PPO 在 LLM 上良好工作需要大量未文档化的工程专业知识"（Zheng et al., 2023）。

### 4. Critic 模型的开销与局限

- Critic 增加了 ~25% 的总内存和计算开销
- Critic 的训练目标（预测未来回报）与 Actor 的训练目标（生成好回复）不完全一致
- 在序列长度变化大的场景下，Critic 的泛化能力有限
- 这正是 [[grpo]] 去掉 Critic 的动机

---

## 参考文献

- Schulman et al. (2015) — [Trust Region Policy Optimization (TRPO)](https://arxiv.org/abs/1502.05477)
- Schulman et al. (2017) — [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- Schulman et al. (2015) — [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- Ouyang et al. (2022) — [InstructGPT](https://arxiv.org/abs/2203.02155)
- Zheng et al. (2023) — [Secrets of RLHF in Large Language Models](https://arxiv.org/abs/2307.04964)
- Xu et al. (2024) — [Is DPO Superior to PPO for LLM Alignment?](https://arxiv.org/abs/2404.10719) (ICML 2024)
- Wu et al. (2023) — [P3O: Pairwise Proximal Policy Optimization](https://arxiv.org/abs/2310.00212)
- Hu et al. (2025) — [REINFORCE++: A Simple and Efficient Approach for Aligning Large Language Models](https://arxiv.org/abs/2501.03262)
- Ahmadian et al. (2024) — [Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs](https://arxiv.org/abs/2402.14740)

---

## 相关页面

- [[rlhf-overview]] — 完整的 RLHF 三阶段流水线
- [[grpo]] — 无 Critic 的 PPO 替代方案，由 DeepSeek 提出
- [[dpo]] — 直接偏好优化，完全离线的对齐方法
- [[reward-modeling]] — 奖励模型的训练与评估
- [[rl-training-frameworks]] — 实现 PPO/GRPO 的训练框架（OpenRLHF, veRL, TRL）
- [[multi-step-reasoning-rl]] — PPO/GRPO 在推理模型中的应用
