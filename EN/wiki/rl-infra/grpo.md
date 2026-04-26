---
title: "GRPO: Group Relative Policy Optimization"
category: rl-infra
tags: [grpo, deepseek, reinforcement-learning, critic-free, deepseek-r1, deepseek-math, policy-optimization, reasoning]
created: 2026-04-13
updated: 2026-04-14
status: mature
---

# GRPO: Group Relative Policy Optimization

## Overview

GRPO (Group Relative Policy Optimization), introduced by DeepSeek in the DeepSeekMath paper (Shao et al., 2024) and made famous by DeepSeek-R1, is a **critic-free policy optimization algorithm**. Instead of training a value function (critic) to estimate advantages, GRPO samples a group of responses per prompt and uses within-group reward statistics as the baseline.

**Core innovation**: replace PPO's learned critic with group-normalized rewards, reducing the four-model architecture to two-three models and cutting memory/compute by ~50% while matching or exceeding PPO performance on reasoning and alignment tasks.

GRPO rapidly became the dominant algorithm for large-scale RL training in 2024-2025, especially for reasoning model training.

---

## Motivation: Why Eliminate the Critic?

### PPO's Memory Bottleneck

[[ppo-for-llm|PPO-based RLHF]] requires four models in memory. GRPO removes the critic:

```
PPO: Actor + Critic + Ref + RM  =  4 models (~150GB for 7B)
GRPO: Actor + Ref + RM          =  3 models (~110GB for 7B)
```

For 70B models, dropping the critic saves ~140GB (fp16).

### Critic Training Issues

1. **Sparse rewards**: RM scores only at sequence end; critic must predict returns at all intermediate positions
2. **Length variability**: responses range from tens to thousands of tokens; critic generalizes poorly
3. **Objective mismatch**: critic's prediction task diverges from actor's generation task
4. **GRPO's insight**: "We don't need to *learn* a baseline -- we can *estimate* it from samples." The group mean reward is an unbiased estimator of expected return.

---

## Algorithm Details

### Algorithm Flow

```
For each training iteration:
  1. Sample prompts {x_1, ..., x_B}
  2. For each x_i: sample G responses from π_θ, compute rewards {r_1,...,r_G}
  3. Group-relative advantage: A_j = (r_j - mean(r)) / std(r)
  4. Clipped policy gradient with KL penalty:
     L = E[min(ρ·A, clip(ρ, 1-ε, 1+ε)·A) - β·KL(π_θ||π_ref)]
  5. Update θ
```

### Core Formulas

**Group-relative advantage**:
```
μ = (1/G) Σ r_j,    σ = std({r_1,...,r_G})
A_j = (r_j - μ) / σ
```

This is sound because: Q(x, y_j) ~ r_j and V(x) ~ μ = E[r(x,y)], so A_j ~ r_j - μ, consistent with PPO's advantage. Division by σ normalizes across prompts with different reward variances.

**GRPO objective** (reuses PPO's clipped surrogate):
```
L_GRPO = (1/BG) Σ_{i,j} min(ρ_ij · A_ij, clip(ρ_ij, 1-ε, 1+ε) · A_ij) - β · KL_j
```

where `ρ_ij = π_θ(y_i^j|x_i) / π_old(y_i^j|x_i)` is the sequence-level probability ratio.

### Pipeline Diagram

```
┌──────────────────────────────────────────────────────────┐
│                  GRPO Training Pipeline                   │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  π_θ ──> sample G responses ──> reward scoring           │
│                                      │                   │
│                              group normalization         │
│                              A_j = (r_j - μ) / σ        │
│                                      │                   │
│  π_ref ──> KL penalty ──> clipped policy gradient update │
│                                                          │
│  No critic needed. Group mean replaces learned V(s).     │
└──────────────────────────────────────────────────────────┘
```

---

## Comparison with PPO

| Dimension | PPO | GRPO |
|-----------|-----|------|
| Models in memory | 4 (Actor, Critic, Ref, RM) | 3 (Actor, Ref, RM) or 2 (RLVR) |
| Advantage estimation | Token-level GAE via Critic | Sequence-level group normalization |
| Baseline source | Learned V(s) | Group mean μ |
| Samples per prompt | 1 | G (typically 16-64) |
| Advantage granularity | Token-level credit assignment | Sequence-level only |
| Training stability | Medium (Critic can be unstable) | High (no Critic to train) |
| Hyperparameters | Many (lr_actor, lr_critic, ε, β, γ, λ, K) | Fewer (lr, ε, β, G) |

**GRPO > PPO when**: large models (>30B), reasoning tasks with clear rewards, need fast iteration.
**PPO may be better when**: fine-grained token-level control needed, small group sizes forced, mature PPO infrastructure exists.

---

## GRPO in DeepSeek

### DeepSeek-R1-Zero: Pure RL Discovery

Trained directly on base model (no SFT) with GRPO + answer-correctness reward only. **Emergent behaviors**:

- Chain-of-thought reasoning (no demonstrations)
- Self-reflection: "Wait, let me reconsider..."
- Self-verification and strategy switching
- "Aha moments" during training

These capabilities emerged purely from RL, without any human demonstrations.

### DeepSeek-R1: Full Pipeline

```
Stage 1: Cold-start SFT (small amount of CoT demonstrations)
Stage 2: Large-scale GRPO (rule-based rewards for math/code, learned RM for open-ended)
Stage 3: Rejection sampling + SFT (filter high-quality outputs, retrain for readability)
Stage 4: Second round of GRPO
```

Key hyperparameters: G=64, β=0.001, lr=3e-6, max_length=32K, ε=0.2 (later 10).

---

## Code Examples

### GRPO Training Loop

```python
class GRPOTrainer:
    def __init__(self, policy, ref_policy, reward_fn, group_size=16, clip_eps=0.2, kl_coef=0.01):
        self.policy, self.ref_policy, self.reward_fn = policy, ref_policy, reward_fn
        self.G, self.clip_eps, self.kl_coef = group_size, clip_eps, kl_coef

    def train_step(self, prompts):
        # 1. Sample G responses per prompt
        all_responses, all_old_lp = [], []
        with torch.no_grad():
            for p in prompts:
                resp, lp = self.policy.generate(p, num_samples=self.G, return_log_probs=True)
                all_responses.append(resp); all_old_lp.append(lp)

        # 2. Compute rewards and group-relative advantages
        all_advantages = []
        for p, resp in zip(prompts, all_responses):
            rewards = self.reward_fn(p, resp)
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            all_advantages.append(advantages)

        # 3. Clipped policy gradient update
        loss = 0
        for i, p in enumerate(prompts):
            new_lp = self.policy.log_probs(p, all_responses[i])
            ratio = torch.exp(new_lp - all_old_lp[i])
            surr1 = ratio * all_advantages[i]
            surr2 = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * all_advantages[i]
            ref_lp = self.ref_policy.log_probs(p, all_responses[i])
            kl = (new_lp - ref_lp).mean()
            loss += -torch.min(surr1, surr2).mean() + self.kl_coef * kl
        loss /= len(prompts)
        loss.backward(); self.optimizer.step()
```

### TRL GRPOTrainer Usage

```python
from trl import GRPOConfig, GRPOTrainer

config = GRPOConfig(
    num_generations=16, learning_rate=5e-6, cliprange=0.2, beta=0.01,
    per_device_train_batch_size=4, max_completion_length=2048, temperature=1.0, bf16=True,
)
trainer = GRPOTrainer(model=model, config=config, tokenizer=tokenizer,
                      train_dataset=dataset, reward_funcs=reward_fn)
trainer.train()
```

---

## GRPO Variants

### DAPO (ByteDance Seed + Tsinghua AIR)

Four improvements over GRPO ([arXiv:2503.14476](https://arxiv.org/abs/2503.14476)):

1. **Clip-Higher**: asymmetric clipping (ε_high > ε_low) to prevent entropy collapse
2. **Dynamic Sampling**: filter groups where all responses are correct/incorrect (zero gradient)
3. **Token-Level Policy Gradient Loss**: normalize by sequence length; critical for long CoT
4. **Overlong Reward Shaping**: gradual penalty for truncated outputs

Result: 50 on AIME 2024 (vs. R1-Zero's 47) with 50% fewer training steps. Trained with [[rl-training-frameworks#veRL|veRL]].

### Dr. GRPO (Variance-Reduced)

Removes std normalization (A_j = r_j - μ only) to eliminate finite-sample bias from dividing by the random variable σ.

### RLOO (REINFORCE Leave-One-Out)

Uses leave-one-out baseline: `baseline_j = mean(r_{k≠j})`. Lower variance than GRPO's group mean since baseline is independent of the evaluated sample.

---

## Performance Analysis

| Model/Method | AIME 2024 | MATH-500 | Training Cost |
|-------------|-----------|----------|---------------|
| DeepSeek-R1-Zero (GRPO, pure RL) | 71.0% | 95.9% | Baseline |
| DeepSeek-R1 (GRPO, full pipeline) | 79.8% | 97.3% | ~2x |
| OpenAI o1 | 79.2% | 96.4% | Undisclosed |

Across independent studies: GRPO matches PPO on reasoning and alignment tasks while training 30-50% faster.

---

## Limitations

1. **Group size sensitivity** -- G too small: noisy statistics, biased normalization; G too large: expensive rollouts
2. **Reward model dependency** -- for non-verifiable tasks, still relies on learned RM; group normalization cannot correct systematic RM bias
3. **Sequence-level advantage only** -- no token-level credit assignment; ambiguous which tokens contributed to reward
4. **Sampling cost** -- G rollouts per prompt; bottleneck for long sequences (32K tokens); needs efficient inference (vLLM)
5. **Diversity requirement** -- if policy is already strong, group variance vanishes, training signal weakens (motivating DAPO's Dynamic Sampling)
6. **Limited theory** -- convergence guarantees weaker than PPO; optimality gap bounds unclear

---

## References

- Shao et al. (2024) -- [DeepSeekMath](https://arxiv.org/abs/2402.03300) (original GRPO)
- DeepSeek-AI (2025) -- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
- DeepSeek-R1 in Nature 2025 -- [doi:10.1038/s41586-025-09422-z](https://www.nature.com/articles/s41586-025-09422-z)
- Yu et al. (2025) -- [Revisiting GRPO](https://arxiv.org/html/2505.22257v1)
- DAPO (2025) -- [arXiv:2503.14476](https://arxiv.org/abs/2503.14476)
- Hu et al. (2025) -- [REINFORCE++](https://arxiv.org/abs/2501.03262)
- Liu et al. (2025) -- [Understanding R1-Zero-Like Training](https://arxiv.org/abs/2503.20783)

## Related Pages

- [[ppo-for-llm]] -- The critic-based alternative
- [[rlhf-overview]] -- The overall alignment pipeline
- [[dpo]] -- Direct Preference Optimization (offline alternative)
- [[reward-modeling]] -- Reward signal sources: learned RM and rule-based verifiers
- [[rl-training-frameworks]] -- Frameworks supporting GRPO (veRL, OpenRLHF, TRL)
- [[multi-step-reasoning-rl]] -- GRPO for reasoning models
