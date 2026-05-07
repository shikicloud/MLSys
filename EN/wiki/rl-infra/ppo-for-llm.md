---
title: "PPO for LLM Alignment"
category: rl-infra
tags: [ppo, reinforcement-learning, alignment, rlhf, policy-optimization, gae, critic]
created: 2026-04-13
updated: 2026-04-14
status: mature
---

# PPO for LLM Alignment

> [!abstract]+ TL;DR
> Proximal Policy Optimization (Schulman et al., 2017) is the RL algorithm that made [[rlhf-overview|RLHF]] practical for LLMs. Its **clipped surrogate objective** prevents destructively large policy updates — critical when the policy is a billion-parameter LM where a single bad update can permanently damage capabilities. Powered InstructGPT, ChatGPT, and early Claude. Increasingly supplemented by [[grpo|GRPO]] and DPO for memory efficiency, but PPO remains essential for online RL tasks requiring exploration (reasoning, code generation, tool use).

---

## PPO Algorithm Review

### Policy Gradient Foundation

RL objective: maximize expected cumulative reward `J(θ) = E[Σ γ^t r_t]`. The policy gradient theorem gives:

```
∇J(θ) = E[Σ_t ∇ log π_θ(a_t|s_t) · A_t]
```

**Problem**: REINFORCE has extremely high variance.

### TRPO to PPO Evolution

**TRPO** constrains policy updates via a KL trust region -- requires computing the Fisher information matrix inverse, infeasible for billion-parameter LLMs.

**PPO** replaces the hard constraint with a clipped objective:

```
L^CLIP(θ) = E[min(r_t(θ) · A_t, clip(r_t(θ), 1-ε, 1+ε) · A_t)]
```

Where `r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t)` is the probability ratio and `ε` ~ 0.1-0.2.

**Clipping intuition**: when A_t > 0 (good action), r_t is capped at 1+ε; when A_t < 0 (bad action), r_t is floored at 1-ε. This bounds how much the policy can change per update.

### Full PPO Loss

```
L^PPO = -L^CLIP + c_1 · L^VF - c_2 · S(π_θ)
```

- `L^VF = E[(V_θ(s_t) - V_target)^2]` -- value function loss
- `S(π_θ)` -- entropy bonus for exploration
- In LLM RLHF, actor and critic are separate models; entropy bonus is usually minimal since KL penalty serves a similar purpose.

---

## PPO Adaptations for LLMs

### LLM as Policy

| RL Concept | LLM Mapping |
|------------|-------------|
| State s_t | prompt + generated prefix (x, y_{<t}) |
| Action a_t | next token y_t |
| Policy π(a\|s) | π_θ(y_t \| x, y_{<t}) |
| Trajectory τ | complete (prompt, response) pair |
| Reward | RM score (typically only at sequence end) |

### Token-Level vs. Sequence-Level Rewards

```
Token 1    Token 2    ...    Token T
Reward:   -β·kl_1   -β·kl_2  ...   r_RM - β·kl_T
```

RM reward is given only at the last token; intermediate tokens receive only KL penalty. This sparsity makes credit assignment challenging -- motivating GAE.

### KL Penalty Integration

```
r_total = r_RM(x,y) - β · Σ_t kl_t
where kl_t = log π_θ(y_t|...) - log π_ref(y_t|...)
```

InstructGPT uses adaptive β: increase β when actual KL > 1.5 * target, decrease when < target / 1.5.

---

## Four-Model Architecture

```
┌─────────────────────────────────────────────────────┐
│  Trainable             │  Frozen                     │
│  ┌──────────────┐      │  ┌──────────────┐          │
│  │ Actor (π_θ)   │      │  │ Ref Model     │          │
│  │ generates     │      │  │ (π_ref, KL)  │          │
│  │ ~14GB (7B)   │      │  │ ~14GB (7B)   │          │
│  ├──────────────┤      │  ├──────────────┤          │
│  │ Critic (V_φ)  │      │  │ Reward Model  │          │
│  │ value est.   │      │  │ (r_ψ, scores)│          │
│  │ ~14GB (7B)   │      │  │ ~14GB (7B)   │          │
│  └──────────────┘      │  └──────────────┘          │
│                         │                             │
│  Total for 7B: ~56GB params + ~56GB optimizer         │
│  + 20-40GB activations = ~130-150GB                   │
└─────────────────────────────────────────────────────┘
```

Memory optimization strategies: LoRA/QLoRA, model parallelism, CPU offloading, quantizing frozen models, or eliminating the critic via [[grpo]].

---

## GAE (Generalized Advantage Estimation)

GAE interpolates between high-bias TD(0) and high-variance Monte Carlo via parameter λ:

```
δ_t = r_t + γ · V(s_{t+1}) - V(s_t)       (TD residual)
A_t^GAE = Σ_{l=0}^{∞} (γλ)^l · δ_{t+l}    (exponentially-weighted sum)
```

Recursive form (used in implementation):
```
A_T = δ_T
A_t = δ_t + γλ · A_{t+1}
```

In LLM RLHF: γ = 1.0, λ = 0.95 are standard. The GAE propagates the sparse terminal RM reward back through all token positions.

---

## Implementation Details and Tips

| Technique | Details |
|-----------|---------|
| **Reward normalization** | Running mean/std; exclude KL penalty from normalization |
| **Advantage normalization** | Per-mini-batch: `(A - mean) / (std + 1e-8)` |
| **Gradient clipping** | Global norm clipping to 1.0 |
| **Learning rate** | Actor: 5e-7 to 5e-5; Critic often 2-10x larger |
| **PPO epochs** | 1-4 per batch (1-2 for LLMs to avoid overfitting) |
| **Adaptive KL** | Target KL ~6 nats; β dynamically adjusted |

### Common Pitfalls

| Symptom | Cause | Fix |
|---------|-------|-----|
| Reward spikes then crashes | Reward hacking | Increase β; check RM quality |
| Reward never increases | LR too low / β too high | Decrease β; increase LR |
| KL divergence explodes | Updates too large | Decrease LR, ε; increase β |
| Increasing length, decreasing quality | RM prefers verbosity | Add length penalty; fix RM data |

---

## Code Examples

### PPO Training Step Pseudocode

```python
class PPOTrainerForLLM:
    def __init__(self, actor, critic, ref_model, reward_model,
                 clip_eps=0.2, kl_coef=0.1, gamma=1.0, lam=0.95):
        self.actor, self.critic = actor, critic
        self.ref_model, self.reward_model = ref_model, reward_model
        # ... optimizers ...

    @torch.no_grad()
    def generate_and_score(self, prompts):
        responses, log_probs = self.actor.generate(prompts, return_log_probs=True)
        ref_log_probs = self.ref_model.log_probs(prompts, responses)
        rm_scores = self.reward_model.score(prompts, responses)
        kl_per_token = log_probs - ref_log_probs
        rewards = -self.kl_coef * kl_per_token
        rewards[:, -1] += rm_scores
        values = self.critic(prompts, responses)
        advantages, returns = compute_gae(rewards, values, self.gamma, self.lam)
        return {"old_log_probs": log_probs, "advantages": advantages, "returns": returns}

    def ppo_update(self, prompts, data):
        for epoch in range(self.ppo_epochs):
            new_lp = self.actor.log_probs(prompts, data["responses"])
            ratio = torch.exp(new_lp - data["old_log_probs"])
            adv = (data["advantages"] - data["advantages"].mean()) / (data["advantages"].std() + 1e-8)
            actor_loss = -torch.min(ratio * adv, torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * adv).mean()
            critic_loss = F.mse_loss(self.critic(prompts, data["responses"]), data["returns"])
            # backprop and step ...
```

---

## PPO vs. Alternatives

| Dimension | PPO | REINFORCE | [[grpo]] | [[dpo]] |
|-----------|-----|-----------|------|-----|
| Critic needed | Yes | No | No | No |
| Memory | 4 models | 2 models | 2-3 models | 2 models |
| Stability | Medium | Low | High | Very High |
| Performance ceiling | High | Medium | High | Good (offline) |
| Online exploration | Yes | Yes | Yes | No |
| Hyperparameter count | Many | Few | Moderate | Few |
| Best for | Complex tasks, mature teams | Simple verification | Large-scale reasoning RL | Simple alignment |

**Key finding** (Xu et al., ICML 2024): PPO, when properly tuned, matches or exceeds DPO -- but engineering overhead is substantial.

---

## Limitations

1. **Four-model memory pressure** -- 70B model requires ~1TB+ GPU memory; needs multi-node clusters
2. **Training instability** -- Actor-Critic coordination is fragile; sparse rewards make credit assignment hard; Critic errors cascade into Actor
3. **Hyperparameter sensitivity** -- LR, β, ε, λ, K, batch size all interact; "tribal knowledge" required (Zheng et al., 2023)
4. **Critic overhead** -- adds ~25% compute/memory; its training objective misaligns with the Actor's; this motivates [[grpo]]

---

## References

- Schulman et al. (2015) -- [TRPO](https://arxiv.org/abs/1502.05477)
- Schulman et al. (2015) -- [GAE](https://arxiv.org/abs/1506.02438)
- Schulman et al. (2017) -- [PPO](https://arxiv.org/abs/1707.06347)
- Ouyang et al. (2022) -- [InstructGPT](https://arxiv.org/abs/2203.02155)
- Zheng et al. (2023) -- [Secrets of RLHF in LLMs](https://arxiv.org/abs/2307.04964)
- Xu et al. (2024) -- [Is DPO Superior to PPO?](https://arxiv.org/abs/2404.10719) (ICML 2024)
- Hu et al. (2025) -- [REINFORCE++](https://arxiv.org/abs/2501.03262)
- Ahmadian et al. (2024) -- [Back to Basics: REINFORCE for LLMs](https://arxiv.org/abs/2402.14740)

## Related Pages

- [[rlhf-overview]] -- The full RLHF three-stage pipeline
- [[grpo]] -- Critic-free PPO alternative by DeepSeek
- [[dpo]] -- Direct Preference Optimization (offline)
- [[reward-modeling]] -- Reward model training and evaluation
- [[rl-training-frameworks]] -- Frameworks implementing PPO/GRPO (OpenRLHF, veRL, TRL)
