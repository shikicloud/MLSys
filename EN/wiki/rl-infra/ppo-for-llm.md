---
title: "PPO for LLM Alignment"
category: rl-infra
tags: [ppo, reinforcement-learning, alignment, rlhf, policy-optimization, gae, critic]
created: 2026-04-13
updated: 2026-05-20
status: mature
---

# PPO for LLM Alignment

> [!abstract]+ TL;DR
> Proximal Policy Optimization (PPO, Schulman et al., 2017) is the RL algorithm that made [[rlhf-overview|RLHF]] practical on LLMs. Its **clipped surrogate objective** prevents destructive large policy updates — on policy networks with billions of parameters, a single bad update can permanently damage model capabilities. PPO powered InstructGPT, ChatGPT, and early Claude. It is gradually being replaced by [[grpo|GRPO]] and DPO on memory efficiency, but remains irreplaceable for complex tasks requiring online exploration (reasoning, code generation, tool use).

## Overview

Proximal Policy Optimization (PPO), introduced by Schulman et al. (2017), is the core RL algorithm that made [[rlhf-overview|RLHF]] practical on large language models. It provides stable policy updates via a clipped surrogate objective, preventing destructive large parameter changes. PPO powered InstructGPT, ChatGPT, and early Claude models.

**Why PPO and not other RL algorithms?**

- Language models are huge policy networks (billions of parameters); a single unstable update can permanently damage the model
- PPO naturally bounds the magnitude of each update through clipping
- Compared to TRPO, PPO is much simpler to implement (no second-order optimization) and far more compute-efficient
- PPO has been thoroughly validated on standard RL benchmarks like Atari/MuJoCo

**Position of PPO in LLM alignment**: PPO was the only choice for RLHF (2019-2023), but with the arrival of DPO (2023) and [[grpo]] (2024), it is no longer the only option. However, for complex tasks requiring online exploration (reasoning, code generation, tool use), PPO/GRPO are still irreplaceable.

---

## PPO Algorithm Recap

### Policy Gradient Basics

The RL objective is to maximize expected cumulative reward:

```
J(θ) = E_{τ~π_θ} [Σ_t γ^t · r_t]
```

The Policy Gradient Theorem tells us:

```
∇_θ J(θ) = E_{τ~π_θ} [Σ_t ∇_θ log π_θ(a_t|s_t) · A_t]
```

where A_t is the advantage function. This is the basis of REINFORCE.

**Problem**: REINFORCE has very high gradient variance and requires many samples for stable training.

### From TRPO to PPO

**TRPO (Trust Region Policy Optimization, Schulman et al. 2015)**

TRPO's core idea: constrain the policy change at each update to lie within a "trust region".

```
max_θ  E [r_t(θ) · A_t]
s.t.   E [KL(π_old || π_θ)] ≤ δ
```

where r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t) is the importance-sampling probability ratio.

TRPO requires inverting a Fisher information matrix — infeasible for LLMs with billions of parameters.

**PPO — A Practical Approximation to TRPO**

PPO replaces the hard constraint with a clipping mechanism, turning a constrained optimization into an unconstrained one:

```
L^CLIP(θ) = E_t [min(r_t(θ) · A_t, clip(r_t(θ), 1-ε, 1+ε) · A_t)]
```

The design is brilliantly simple:
- When A_t > 0 (good action): r_t cannot exceed 1+ε, preventing over-increasing the probability of good actions
- When A_t < 0 (bad action): r_t cannot drop below 1-ε, preventing over-decreasing the probability of bad actions
- The effect is equivalent to optimizing within a trust region, but at very low compute cost

### Clipped Surrogate Objective in Detail

```
L^CLIP(θ) = E_t [min(r_t(θ) · A_t, clip(r_t(θ), 1-ε, 1+ε) · A_t)]
```

where:
- `r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t)` — probability ratio between new and old policy
- `A_t` — advantage estimate (typically from GAE)
- `ε` — clip parameter (typically 0.1~0.2)

**Geometric intuition of clipping**:

```
                    A_t > 0 (good action)
Objective L         ___________
    ↑              /
    |             /
    |            /
    |───────────/──────────────── r_t(θ)
    |          1-ε    1    1+ε
    |
    |  A_t < 0 (bad action)
    |───────────────\──────────── r_t(θ)
    |                \___________
    |          1-ε    1    1+ε

Gradient is zero outside the clip range [1-ε, 1+ε]
→ Prevents large policy changes
```

When A_t > 0: the objective is clipped at r_t = 1+ε → even if increasing probability is beneficial, only allow up to a point.
When A_t < 0: the objective is clipped at r_t = 1-ε → even if decreasing probability is beneficial, only allow down to a point.

### Value Function Loss

The critic predicts the expected return V(s_t) at each state; its loss is:

```
L^VF(θ) = E_t [(V_θ(s_t) - V_t^target)^2]
```

where V_t^target = A_t^GAE + V_old(s_t) (the GAE-estimated return).

In practice the value loss is often clipped too:

```
L^VF_clipped = E_t [max(
    (V_θ(s_t) - V_t^target)^2,
    (clip(V_θ(s_t), V_old(s_t)-ε_v, V_old(s_t)+ε_v) - V_t^target)^2
)]
```

### Entropy Bonus

To encourage exploration and prevent premature convergence, PPO adds an entropy bonus to the loss:

```
S(π_θ) = -E_t [Σ_a π_θ(a|s_t) log π_θ(a|s_t)]
```

### Full PPO Loss

```
L^PPO(θ) = -L^CLIP(θ) + c_1 · L^VF(θ) - c_2 · S(π_θ)
```

where:
- `c_1` — value loss coefficient (typically 0.5~1.0)
- `c_2` — entropy bonus coefficient (typically 0.01~0.05)
- The negative sign on L^CLIP is because we maximize it but minimize the total loss

**Note**: in LLM RLHF, Actor and Critic are usually separate models (no shared parameters), so c_1 and c_2 are set differently from standard PPO. In the LLM setting the entropy bonus is typically not used or set very small, because the KL penalty already plays a similar anti-collapse role.

---

## Adapting PPO to LLMs

### LLM as a Policy

In the PPO framework for RLHF, the LLM is formalized as an RL policy:

| RL concept | LLM analog |
|---------|---------|
| State s_t | prompt + already-generated prefix (x, y_{<t}) |
| Action a_t | next token y_t |
| Policy π(a\|s) | LLM conditional probability π_θ(y_t \| x, y_{<t}) |
| Trajectory τ | full (prompt, response) pair |
| Reward | RM score (typically only at the end of the sequence) |

### Generation as Sequential Decision-Making

LLM text generation can be viewed as a **finite-horizon Markov Decision Process (MDP)**:

```
                    Token 1    Token 2    Token 3         Token T
State:  [prompt] ──> s_1 ────> s_2 ────> s_3 ──...──> s_T
Action:             a_1=y_1   a_2=y_2   a_3=y_3       a_T=y_T
Reward:             -β·kl_1   -β·kl_2   -β·kl_3       r_RM - β·kl_T
```

At each step:
- State = prompt + all generated tokens so far
- Action = pick the next token from vocabulary V
- Intermediate reward = only the KL penalty term (-β·kl_t)
- Terminal reward = RM score + KL penalty

### Token-Level vs Sequence-Level Reward

**Sequence-level reward** (RM score): given only at the last token; RM reward is 0 at all intermediate tokens.

**Token-level KL penalty**: present at every token position:

```
kl_t = log π_θ(y_t|x, y_{<t}) - log π_ref(y_t|x, y_{<t})
```

**Total reward decomposition**:

```
r_t = { -β · kl_t                    if t < T (intermediate tokens)
      { r_RM(x, y) - β · kl_t        if t = T (last token)
```

This reward sparsity (most reward concentrated at the end) is a core challenge for LLM PPO, and GAE is needed for effective credit assignment.

### Integrating the KL Penalty

```
r_total(x, y) = r_RM(x, y) - β · Σ_{t=1}^{T} kl_t
```

Three ways to implement KL penalty:
1. **Per-token KL** (most common): penalty applied at every token position as shown above
2. **Sequence-level KL**: add total KL once at sequence end
3. **Adaptive KL** (InstructGPT-style): dynamically tune β to keep actual KL near a target

```python
# Adaptive KL coefficient (InstructGPT)
target_kl = 6.0  # target KL

if actual_kl > 1.5 * target_kl:
    beta *= 1.5   # KL too large, increase penalty
elif actual_kl < target_kl / 1.5:
    beta /= 1.5   # KL too small, reduce penalty (allow more exploration)
```

---

## The PPO-RLHF Training Loop

```
┌─────────────────────────────────────────────────────────────────┐
│                   PPO-RLHF training loop                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────┐  sample prompts ┌──────────┐  generate response   │
│  │ Prompt  │────────────────>│  Actor   │───────────┐          │
│  │ dataset │                 │  π_θ     │           │          │
│  └─────────┘                 └──────────┘           ▼          │
│                                              ┌──────────┐      │
│  ┌─────────────┐                              │ Response │      │
│  │ Reference   │───── KL penalty ────────────>│   y      │      │
│  │ π_ref(froz) │                              └────┬─────┘      │
│  └─────────────┘                                  │            │
│                                                    ▼            │
│                           ┌──────────┐    ┌──────────┐          │
│                           │ Critic   │    │ Reward   │          │
│                           │ V_φ      │    │ r_ψ(x,y)│          │
│                           └─────┬────┘    └─────┬────┘          │
│                                 │               │               │
│                                 ▼               ▼               │
│                           ┌──────────────────────────┐          │
│                           │ Compute GAE advantage    │          │
│                           │ A_t = Σ(γλ)^l · δ_{t+l} │          │
│                           └────────────┬─────────────┘          │
│                                        │                        │
│                                        ▼                        │
│                           ┌──────────────────────────┐          │
│                           │ PPO clipped update       │          │
│                           │ L = min(r·A, clip(r)·A) │          │
│                           │ Update Actor + Critic    │          │
│                           └──────────────────────────┘          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Four-Model Architecture

PPO-based RLHF requires four models simultaneously in memory:

```
┌─────────────────────────────────────────────────────────────────┐
│                      Four-model architecture                      │
├──────────────────┬──────────────────────────────────────────────┤
│                  │                                              │
│  Trainable       │  Frozen                                      │
│                  │                                              │
│  ┌────────────┐  │  ┌────────────┐                             │
│  │  Actor      │  │  │ Reference  │                             │
│  │ (policy π_θ)│  │  │ (π_ref)   │                             │
│  │  generates  │  │  │ computes   │                             │
│  │ [trainable] │  │  │ KL [frozen]│                             │
│  │ ~14GB(7B)   │  │  │ ~14GB(7B) │                             │
│  └────────────┘  │  └────────────┘                             │
│                  │                                              │
│  ┌────────────┐  │  ┌────────────┐                             │
│  │  Critic     │  │  │ Reward     │                             │
│  │ (value V_φ) │  │  │  (r_ψ)    │                             │
│  │ value est.  │  │  │  scoring  │                             │
│  │ [trainable] │  │  │ [frozen]  │                             │
│  │ ~14GB(7B)   │  │  │ ~14GB(7B) │                             │
│  └────────────┘  │  └────────────┘                             │
│                  │                                              │
├──────────────────┴──────────────────────────────────────────────┤
│  Total memory for a 7B model:                                    │
│  Model parameters: 4 × 14GB = 56GB (fp16)                        │
│  Optimizer state (Actor+Critic): ~56GB (Adam: 2× param size)    │
│  Activations/grads: ~20-40GB (depends on seq len and batch)     │
│  Total: ~130-150GB → multiple A100 80GB GPUs                    │
└─────────────────────────────────────────────────────────────────┘
```

### The Four Models in Detail

| Model | Init from | Trainable | Function | Output |
|------|----------|-----------|------|------|
| **Actor (policy)** | SFT model | Yes | Generate response tokens | π_θ(y_t \| x, y_{<t}) |
| **Critic (value function)** | SFT model or RM | Yes | Estimate expected return per position | V_φ(s_t) ∈ R |
| **Reference model** | SFT model (frozen) | No | Provide KL penalty baseline | π_ref(y_t \| x, y_{<t}) |
| **Reward model** | From RM training (frozen) | No | Score complete responses | r_ψ(x, y) ∈ R |

### Memory Optimizations

| Strategy | Description | Savings |
|------|------|------|
| **Parameter sharing** | Actor and Critic share Transformer backbone | ~14GB |
| **LoRA/QLoRA** | Train only low-rank adapter parameters | 60-90% |
| **Model parallelism** | Distribute large models across GPUs | Enables bigger models |
| **CPU offload** | Move infrequently used models (ref, RM) to CPU | GPU memory |
| **Quantization** | Frozen models in int8/int4 | 50-75% |
| **Drop the critic** | Use [[grpo]] instead of PPO | ~25% |

---

## GAE (Generalized Advantage Estimation)

### Background

The advantage A(s_t, a_t) = Q(s_t, a_t) - V(s_t) measures "how much better is action a_t in state s_t than average". Two extreme estimators of A:

| Method | Formula | Bias | Variance |
|------|------|------|------|
| **TD(0)** | A_t = r_t + γV(s_{t+1}) - V(s_t) | High (depends on V accuracy) | Low |
| **Monte Carlo** | A_t = Σ_{l=0}^{T-t} γ^l r_{t+l} - V(s_t) | Low | High (full-trajectory randomness) |

**GAE interpolates between them** via a parameter λ that controls the bias-variance trade-off.

### Deriving GAE

**Step 1**: Define the TD residual

```
δ_t = r_t + γ · V(s_{t+1}) - V(s_t)
```

**Step 2**: Define n-step advantage estimators

```
A_t^(1) = δ_t                                          (1-step, high bias, low var)
A_t^(2) = δ_t + γ·δ_{t+1}                              (2-step)
A_t^(3) = δ_t + γ·δ_{t+1} + γ^2·δ_{t+2}               (3-step)
...
A_t^(∞) = Σ_{l=0}^{∞} γ^l · δ_{t+l}                    (∞-step, low bias, high var)
```

**Step 3**: GAE is an exponentially-weighted average of all n-step estimators

```
A_t^GAE(γ,λ) = (1-λ)(A_t^(1) + λ·A_t^(2) + λ^2·A_t^(3) + ...)
             = Σ_{l=0}^{∞} (γλ)^l · δ_{t+l}
```

**Compact recursion** (common in implementations):

```
A_T^GAE = δ_T
A_t^GAE = δ_t + γλ · A_{t+1}^GAE     (backward recursion)
```

### Effect of λ

```
λ = 0:   A_t = δ_t = r_t + γV(s_{t+1}) - V(s_t)    ← TD(0), high bias
λ = 1:   A_t = Σ γ^l r_{t+l} - V(s_t)               ← Monte Carlo, high variance
λ = 0.95: the most common practical compromise
```

### Token-Level GAE for LLMs

```python
def compute_gae(rewards, values, gamma=1.0, lam=0.95):
    """
    Per-token GAE advantage estimation
    
    Args:
        rewards: shape (batch, seq_len) — reward at each token position
                 (intermediate tokens have only -β·kl_t; last token has r_RM - β·kl_T)
        values:  shape (batch, seq_len) — critic value at each position
        gamma:   discount factor (typically = 1.0 for LLM)
        lam:     GAE λ parameter (typically 0.95)
    
    Returns:
        advantages: shape (batch, seq_len) — GAE advantage estimates
        returns:    shape (batch, seq_len) — target returns (for critic loss)
    """
    batch_size, seq_len = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = 0
    
    for t in reversed(range(seq_len)):
        if t == seq_len - 1:
            next_value = 0  # value is 0 after end of sequence
        else:
            next_value = values[:, t + 1]
        
        # TD residual
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        
        # GAE recursion
        advantages[:, t] = delta + gamma * lam * last_gae
        last_gae = advantages[:, t]
    
    returns = advantages + values  # V_target = A + V
    return advantages, returns
```

---

## Implementation Details and Tricks

### Reward Normalization

PPO is sensitive to reward scale. Common normalization strategies:

```python
# Method 1: running mean-variance normalization (recommended)
reward_mean = running_mean(rewards)
reward_std = running_std(rewards)
normalized_reward = (reward - reward_mean) / (reward_std + 1e-8)

# Method 2: clip to a fixed range
reward = torch.clamp(reward, -10.0, 10.0)
```

**Note**: InstructGPT used running mean-variance normalization on rewards and excluded the KL penalty from normalization.

### Advantage Normalization

Normalize advantages within each mini-batch:

```python
# Normalize advantages within mini-batch
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

This does not change the optimal policy but stabilizes training.

### Gradient Clipping

```python
# Global gradient norm clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Learning Rate Schedule

```python
# Common: cosine decay + warmup
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=total_steps,
)

# InstructGPT: constant LR, no warmup
# Linear decay also seen
```

### Mini-batch Training

A key PPO design: the same batch of data can be used for multiple update epochs (PPO epochs):

```python
for ppo_epoch in range(K):  # K typically 1~4
    # Shuffle data and split into mini-batches
    indices = torch.randperm(batch_size)
    for mb_start in range(0, batch_size, mini_batch_size):
        mb_indices = indices[mb_start:mb_start + mini_batch_size]
        
        # Recompute log_probs and values
        new_log_probs = actor(states[mb_indices])
        new_values = critic(states[mb_indices])
        
        # PPO clipped update
        ratio = torch.exp(new_log_probs - old_log_probs[mb_indices])
        surr1 = ratio * advantages[mb_indices]
        surr2 = torch.clamp(ratio, 1-eps, 1+eps) * advantages[mb_indices]
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # Critic update
        critic_loss = F.mse_loss(new_values, returns[mb_indices])
        
        # Backprop and update
        (actor_loss + 0.5 * critic_loss).backward()
        optimizer.step()
```

**LLM-specific caveats**:
- PPO epochs K should not be too large (typically 1-2) because the LLM policy space is huge and multiple epochs easily overfit
- Mini-batch size is limited by GPU memory (re-forward Actor + Critic required)

### Common Pitfalls and Debugging Tips

| Symptom | Possible cause | Fix |
|------|---------|---------|
| Reward rises fast then collapses | Reward hacking | Increase KL penalty β; check RM quality |
| Reward does not rise at all | LR too low / KL penalty too large | Reduce β; raise LR; check gradients |
| KL divergence explodes | Policy update too large | Reduce LR; reduce ε; increase β |
| Generation quality drops while reward rises | RM being exploited | Use RM ensemble; increase KL penalty |
| Critic loss does not decrease | Critic capacity insufficient / bad LR | Increase critic capacity; tune critic LR separately |
| Highly unstable training | Batch too small | Increase batch size; use gradient accumulation |
| Response length keeps growing | RM prefers long answers | Add length penalty; fix RM training data |

---

## Code Examples

### Full PPO Training-Step Pseudocode

```python
import torch
import torch.nn.functional as F

class PPOTrainerForLLM:
    """PPO trainer for LLM RLHF (simplified)"""
    
    def __init__(self, actor, critic, ref_model, reward_model,
                 lr=1e-5, clip_eps=0.2, kl_coef=0.1, 
                 gamma=1.0, lam=0.95, ppo_epochs=2):
        self.actor = actor
        self.critic = critic
        self.ref_model = ref_model      # frozen
        self.reward_model = reward_model # frozen
        
        self.clip_eps = clip_eps
        self.kl_coef = kl_coef
        self.gamma = gamma
        self.lam = lam
        self.ppo_epochs = ppo_epochs
        
        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr)
    
    @torch.no_grad()
    def generate_and_score(self, prompts):
        """Steps 1-2: generate responses and compute rewards"""
        # 1. Actor generates responses
        responses, log_probs = self.actor.generate(prompts, return_log_probs=True)
        
        # 2. Reference model log_probs (for KL)
        ref_log_probs = self.ref_model.log_probs(prompts, responses)
        
        # 3. RM scoring (sequence-level)
        rm_scores = self.reward_model.score(prompts, responses)
        
        # 4. Per-token KL penalty
        kl_per_token = log_probs - ref_log_probs  # shape: (batch, seq_len)
        
        # 5. Build per-token rewards
        rewards = -self.kl_coef * kl_per_token
        rewards[:, -1] += rm_scores  # RM reward applied only at last token
        
        # 6. Critic value estimates
        values = self.critic(prompts, responses)
        
        # 7. GAE advantages
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
        """Step 3: PPO clipped update"""
        for epoch in range(self.ppo_epochs):
            # Recompute log_probs and values (params have changed)
            new_log_probs = self.actor.log_probs(prompts, data["responses"])
            new_values = self.critic(prompts, data["responses"])
            
            # --- Actor update ---
            ratio = torch.exp(new_log_probs - data["old_log_probs"])
            advantages = data["advantages"]
            # Advantage normalization
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()
            
            # --- Critic update ---
            critic_loss = F.mse_loss(new_values, data["returns"])
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.critic_optimizer.step()
        
        return {"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item()}
```

### OpenRLHF / TRL PPO Configuration

```python
# === Typical TRL PPOConfig ===
from trl import PPOConfig

config = PPOConfig(
    # --- Basic ---
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    learning_rate=1.41e-5,
    
    # --- PPO-specific ---
    ppo_epochs=4,              # PPO update epochs per batch
    cliprange=0.2,             # policy clip param ε
    cliprange_value=0.2,       # value clip param
    
    # --- KL penalty ---
    init_kl_coef=0.2,          # initial KL coefficient
    target_kl=6.0,             # target KL
    kl_penalty="kl",           # KL penalty type: "kl", "abs", "mse"
    
    # --- GAE ---
    gamma=1.0,                 # discount factor
    lam=0.95,                  # GAE λ
    
    # --- Batch ---
    batch_size=64,             # prompts per step
    mini_batch_size=16,        # PPO update mini-batch size
    
    # --- Generation ---
    max_new_tokens=256,
    temperature=1.0,
    top_k=0,
    top_p=1.0,
    
    # --- Reward ---
    whiten_rewards=True,       # reward whitening (normalization)
)
```

```yaml
# === Example OpenRLHF PPO config (YAML) ===
model:
  actor: "sft_model_path"
  critic: "sft_model_path"  # or "rm_model_path"
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

# Distributed config
distributed:
  strategy: "colossalai"  # or "deepspeed", "megatron"
  num_gpus: 8
```

---

## PPO vs Alternatives

| Aspect | PPO | REINFORCE | [[grpo]] | [[dpo]] |
|------|-----|-----------|------|-----|
| **Needs critic** | Yes | No | No | No |
| **Needs reward model** | Yes (online) | Yes (online) | Yes (online) | No (implicit) |
| **Memory footprint** | 4 models | 2 models | 2 models + rollouts | 2 models |
| **Training stability** | Medium | Low (high variance) | High | Very high |
| **Performance ceiling** | High (when tuned) | Medium | High | Good (limited by offline data) |
| **Implementation complexity** | High | Low | Medium | Low |
| **Number of hyperparameters** | Many (ε, β, γ, λ, lr...) | Few | Medium (group size, ε, β) | Few (β) |
| **Online exploration** | Yes | Yes | Yes | No |
| **Sample efficiency** | High (multiple PPO epochs) | Low (single use) | Medium | N/A (offline) |
| **Use case** | Complex tasks, mature teams | Simple validation | Large-scale reasoning RL | Simple alignment |

**Key finding** (Xu et al., ICML 2024): a properly tuned PPO can match or surpass DPO — but the engineering cost is enormous. This finding challenges the simple narrative that "DPO is strictly better than PPO".

**REINFORCE++ (Hu et al., 2025)**: an improved REINFORCE that adds token-level KL penalty, reward normalization, advantage whitening, and other PPO engineering tricks, but without a critic. On several benchmarks it matches PPO and GRPO.

---

## Limitations

### 1. Four-Model Memory Pressure

For 70B-scale models, the four models' parameters alone take ~560GB (fp16); add optimizer state and activations and the total exceeds 1TB of GPU memory. This means PPO at scale requires large GPU clusters.

### 2. Training Instability

PPO-for-LLM instability is a notorious industry problem:
- Actor and Critic must update in a coordinated way, but their learning dynamics differ
- Reward signal is sparse (only the end of the sequence carries RM score), making credit assignment hard
- The Critic is inaccurate early in training, biasing advantage estimates and Actor updates
- "Death spiral" risk: bad Critic → bad advantages → wrong Actor updates → worse generations → Critic finds it harder to learn...

### 3. Hyperparameter Sensitivity

PPO-for-LLM has many tunable hyperparameters:

| Hyperparameter | Typical range | Effect |
|--------|---------|------|
| LR (Actor) | 5e-7 ~ 5e-5 | Too big → unstable; too small → no convergence |
| LR (Critic) | 1e-6 ~ 1e-4 | Often 2-10× Actor's |
| KL coef β | 0.01 ~ 0.5 | Too big → too conservative; too small → reward hacking |
| Clip ε | 0.1 ~ 0.3 | Too big → unstable; too small → updates too slow |
| GAE λ | 0.9 ~ 1.0 | Bias-variance trade-off |
| PPO epochs K | 1 ~ 4 | Too big → overfits current batch |
| Batch size | 32 ~ 512 | Too small → noisy gradients |

**Rule of thumb**: "Getting PPO to work well on LLMs requires a large amount of undocumented engineering expertise" (Zheng et al., 2023).

### 4. Overhead and Limitations of the Critic

- The Critic adds ~25% to total memory and compute
- The Critic's training objective (predict future return) is misaligned with the Actor's (generate good response)
- For responses with large length variation, Critic generalization is limited
- This is precisely the motivation for dropping the critic in [[grpo]]

---

## References

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

## Related Pages

- [[rlhf-overview]] — the complete RLHF three-stage pipeline
- [[grpo]] — critic-free PPO alternative from DeepSeek
- [[dpo]] — Direct Preference Optimization, fully offline alignment
- [[reward-modeling]] — reward model training and evaluation
- [[rl-training-frameworks]] — training frameworks that implement PPO/GRPO (OpenRLHF, veRL, TRL)
- [[multi-step-reasoning-rl]] — PPO/GRPO applied to reasoning models
