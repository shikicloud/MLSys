---
title: "GRPO: Group Relative Policy Optimization"
category: rl-infra
tags: [grpo, deepseek, reinforcement-learning, critic-free, deepseek-r1, deepseek-math, policy-optimization, reasoning]
created: 2026-04-13
updated: 2026-05-20
status: mature
---

# GRPO: Group Relative Policy Optimization

> [!abstract]+ TL;DR
> GRPO (DeepSeek, 2024) is a **critic-free policy optimization algorithm**: instead of training a value function, it samples a group of responses for the same prompt and **uses intra-group reward statistics as the baseline**. It collapses [[ppo-for-llm|PPO]]'s 4-model architecture (actor + critic + ref + reward) down to 2–3 models, cutting memory and compute by **~50%**, while matching or beating PPO on reasoning and alignment. Catalyzed by **DeepSeek-R1**, it quickly became the mainstream choice for large-scale RL training in 2024–2025, especially for reasoning models.

---

## Motivation: Why Remove the Critic?

### PPO's Memory Bottleneck

[[ppo-for-llm|PPO-based RLHF]] requires four models in memory at once:

```
PPO 4 models:                       GRPO simplified:
┌──────────────┐                  ┌──────────────┐
│ Actor (train)│                  │ Actor (train)│
├──────────────┤                  ├──────────────┤
│ Critic (train)│ ← GRPO removes  │              │
├──────────────┤                  ├──────────────┤
│ Reference     │                  │ Reference     │
│ (frozen)      │                  │ (frozen)      │
├──────────────┤                  ├──────────────┤
│ Reward (frozen)│                 │ Reward (frozen)│
└──────────────┘                  └──────────────┘
Memory: ~4× model size              Memory: ~3× model size
                                    (+ rollout cache)
```

For a 70B model, dropping the critic directly saves ~140GB (fp16) of GPU memory.

### Critic Training Instability

The critic model has inherent problems in LLM RLHF:

1. **Sparse reward**: the RM only gives a reward at sequence end; the critic has to predict expected return at every intermediate position — a difficult credit-assignment problem
2. **Large variation in sequence length**: response lengths can range from tens to thousands of tokens; the critic generalizes poorly across length
3. **Objective mismatch**: the critic's objective (predict return) is not synchronous with the actor's (generate good responses)
4. **Initialization issue**: the critic is initialized from the SFT model, but its final task is unrelated to language modeling

### GRPO's Key Insight

> **"We don't need to learn a baseline (critic); we can estimate it directly from samples."**

For each prompt, if we sample enough responses:
- The mean reward of those responses is an unbiased estimator of expected return
- (r_i - mean) / std after normalization approximates the advantage
- Statistically this is equivalent to using a "perfect" prompt-specific baseline

---

## Algorithm in Detail

### Algorithm Flow

```
Input: policy π_θ, reference policy π_ref, prompt set D, group size G

For each training iteration:
  1. Sample a batch of prompts {x_1, ..., x_B} from D
  
  2. For each prompt x_i:
     a. Sample G responses from current policy: {y_i^1, ..., y_i^G} ~ π_θ(·|x_i)
     b. Compute reward for each response: r_i^j = R(x_i, y_i^j)  (j = 1,...,G)
  
  3. Compute group-relative advantage:
     μ_i = mean({r_i^1, ..., r_i^G})
     σ_i = std({r_i^1, ..., r_i^G})
     A_i^j = (r_i^j - μ_i) / σ_i              (intra-group normalization)
  
  4. Compute GRPO objective and update policy:
     L_GRPO = E_i,j [ min(ρ_i^j · A_i^j, clip(ρ_i^j, 1-ε, 1+ε) · A_i^j) 
                       - β · KL(π_θ || π_ref) ]
     
     where ρ_i^j = π_θ(y_i^j|x_i) / π_old(y_i^j|x_i)
  
  5. Update θ to maximize L_GRPO
```

### Core Formula Derivation

#### Step 1: Group Sampling and Scoring

For prompt x, sample G independent responses from the current policy π_θ:

```
y_1, y_2, ..., y_G  ~  π_θ(·|x)     (i.i.d. samples)
r_j = R(x, y_j)                       (reward scoring)
```

#### Step 2: Group-Relative Advantage

```
μ = (1/G) Σ_{j=1}^{G} r_j            (intra-group mean reward)
σ = sqrt[(1/G) Σ_{j=1}^{G} (r_j - μ)^2]   (intra-group std)

A_j = (r_j - μ) / σ                   (normalized advantage)
```

**Why is this sensible?**

In PPO, advantage A(s,a) = Q(s,a) - V(s). For sequence-level reward:
- Q(x, y_j) ≈ r_j (the reward for response y_j estimates Q)
- V(x) ≈ μ = E_{y~π_θ}[r(x,y)] (group mean is an unbiased estimate of expected return)
- Hence A_j ≈ r_j - μ (consistent with the PPO advantage meaning)

Std normalization (/ σ) serves to:
- Make advantages comparable across prompts (some prompts have high-variance reward, others low)
- Prevent high-variance prompts from dominating the gradient
- Acts like advantage whitening

#### Step 3: Clipped Policy Gradient

GRPO reuses PPO's clipped surrogate objective:

```
L_GRPO(θ) = (1/B) Σ_{i=1}^{B} (1/G) Σ_{j=1}^{G} 
            min(ρ_{ij}(θ) · A_{ij}, clip(ρ_{ij}(θ), 1-ε, 1+ε) · A_{ij})
```

where the probability ratio ρ is:

```
ρ_{ij}(θ) = π_θ(y_i^j | x_i) / π_old(y_i^j | x_i)
           = exp[Σ_t log π_θ(y_{i,t}^j | x_i, y_{i,<t}^j) 
                 - Σ_t log π_old(y_{i,t}^j | x_i, y_{i,<t}^j)]
```

**Note**: this ρ is a sequence-level probability ratio (product of token-level ratios), whereas PPO's ρ is token-level.

#### Step 4: KL Regularization

GRPO uses KL divergence to keep the policy from drifting too far from the reference:

```
KL_j = Σ_t [log π_θ(y_{j,t}|x, y_{j,<t}) - log π_ref(y_{j,t}|x, y_{j,<t})]
```

**DeepSeek-R1's KL implementation**: uses an approximate KL (not exact KL):

```
KL_approx = (π_ref / π_θ) - log(π_ref / π_θ) - 1
```

This form penalizes more symmetrically when π_θ drifts from π_ref.

#### Full GRPO Objective

```
max_θ  L_GRPO(θ) = E_{x~D} [ (1/G) Σ_{j=1}^{G} 
    min(ρ_j · A_j, clip(ρ_j, 1-ε, 1+ε) · A_j) - β · KL_j ]
```

### GRPO Pipeline (ASCII)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         GRPO training pipeline                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  For each prompt x:                                                 │
│                                                                     │
│  ┌──────────┐  sample G responses ┌─────────────────────────────┐  │
│  │ Current  │ ──────────────────> │  y_1, y_2, ..., y_G         │  │
│  │ policy π_θ│                    │  (G=16~64)                  │  │
│  └──────────┘                     └──────────┬──────────────────┘  │
│                                              │                      │
│                                              ▼                      │
│                                   ┌─────────────────────┐          │
│                                   │   Reward scoring    │          │
│                                   │  r_1, r_2, ..., r_G │          │
│                                   │  (RM or rule verifier)│       │
│                                   └──────────┬──────────┘          │
│                                              │                      │
│                                              ▼                      │
│                                   ┌─────────────────────┐          │
│                                   │ Intra-group normalize│         │
│                                   │  μ = mean(r)         │          │
│                                   │  σ = std(r)          │          │
│                                   │  A_j = (r_j-μ)/σ    │          │
│                                   └──────────┬──────────┘          │
│                                              │                      │
│                                              ▼                      │
│  ┌──────────┐  KL penalty   ┌────────────────────────────────┐    │
│  │ Reference│ ────────────> │  Clipped policy gradient update │    │
│  │ π_ref    │               │  L = min(ρ·A, clip(ρ)·A) - β·KL│    │
│  │ (frozen) │               │  Update θ                       │    │
│  └──────────┘               └────────────────────────────────┘    │
│                                                                     │
│  No critic! Group mean replaces the learned value function.         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Comparison with PPO

### Side-by-Side Objectives

```
PPO:
  L_PPO = E_t [min(ρ_t · A_t^GAE, clip(ρ_t, 1-ε, 1+ε) · A_t^GAE)]
  
  where A_t^GAE is computed from Critic V(s_t) and GAE (token-level)

GRPO:
  L_GRPO = E_{i,j} [min(ρ_j · A_j^group, clip(ρ_j, 1-ε, 1+ε) · A_j^group)]
  
  where A_j^group = (r_j - mean(r)) / std(r) (sequence-level)
```

### Systematic Comparison

| Aspect | PPO | GRPO |
|------|-----|------|
| **Number of models** | 4 (Actor, Critic, Ref, RM) | 3 (Actor, Ref, RM) or 2 (RLVR, no RM) |
| **Advantage estimation** | Token-level GAE (critic-backed) | Sequence-level group normalization |
| **Baseline source** | Learned critic V(s) | Intra-group mean reward μ |
| **Memory (7B)** | ~150GB | ~110GB |
| **Clipping granularity** | Token-level probability ratio | Sequence-level probability ratio |
| **Sampling overhead** | 1 response per prompt | G responses per prompt (G=16~64) |
| **Advantage precision** | High (token-level credit assignment) | Medium (sequence-level signal only) |
| **Training stability** | Medium (critic may be unstable) | High (no critic training) |
| **Hyperparameters** | Many (lr_actor, lr_critic, ε, β, γ, λ, K) | Fewer (lr, ε, β, G) |
| **Implementation complexity** | High | Medium |

### When GRPO Wins

- **Large models** (>30B params): memory savings are critical
- **Reasoning tasks**: reward signal is clear (right/wrong), no token-level credit assignment needed
- **Code/math**: verifiable reward, group comparison is a natural fit
- **Fast iteration**: fewer hyperparameters, easier to tune

### When PPO May Win

- **Fine-grained control**: need token-level credit assignment (e.g., style control, safety constraints)
- **Few samples**: only a few responses can be generated per prompt (group size limited)
- **Very long sequences**: sequence-level advantage may lose too much information
- **Mature PPO infrastructure**: if the team already has a tuned PPO pipeline

---

## GRPO Applied at DeepSeek

### DeepSeek-R1-Zero: A Surprising Pure-RL Result

DeepSeek-R1-Zero is a landmark experiment: training directly on a base model (no SFT) using GRPO + verifiable reward.

**Training setup**:
- Base model: DeepSeek-V3-Base
- Reward signal: only answer correctness (math right/wrong) + format reward (require `<think>...</think>` tags)
- No human annotation, no SFT, no reward model
- Optimized via GRPO

**Emergent behaviors**:

```
Training progress:
Early ───────────────────────────────── Mid ─────────────────────────── Late
  ↓                                      ↓                              ↓
Direct answer                         Step-by-step                 Complex reasoning chains
(often wrong)                         simple verify                Self-reflection
                                                                   "Wait, let me reconsider..."
                                                                   Multi-strategy attempts
                                                                   "aha moment"
```

**"Aha moment"**: during training, the model spontaneously learns:
1. **Chain-of-Thought reasoning** — without demonstration
2. **Self-reflection** — "Wait, let me reconsider..."
3. **Self-verification** — checks its own answers
4. **Strategy switching** — if one approach fails, try another

These abilities emerge purely from RL, with no human demonstrations.

**Limitations of R1-Zero**:
- Poor readability (mixed languages, messy formatting)
- Occasionally produces infinite reasoning loops
- Only effective on verifiable tasks

### DeepSeek-R1: The Full Pipeline

DeepSeek-R1 builds on R1-Zero with a more complete training pipeline:

```
Stage 1: Cold-start SFT
  - Collect small high-quality CoT demos (human-written + R1-Zero filtered)
  - SFT on base model to teach basic reasoning format
  
Stage 2: Large-scale GRPO
  - Rule-based rewards on verifiable tasks (math, code, logic)
  - Learned reward model on open-ended tasks
  - Large-scale GRPO training

Stage 3: Rejection sampling + SFT
  - Generate large amounts of reasoning data with the GRPO-trained model
  - Filter high-quality samples (correct answers + good format)
  - SFT on this data to improve output quality and readability

Stage 4: Second GRPO round
  - Continue RL training for further refinement
```

### Training Parameters (from the DeepSeek-R1 paper)

| Parameter | Value | Notes |
|------|-----|------|
| Learning rate | 3e-6 → 1e-6 (cosine decay) | Low; prevents instability at large scale |
| KL coef β | 0.001 | Very small; allows full exploration |
| Clip ratio ε | 0.2 (later raised to 10) | Larger updates allowed late in training |
| Sampling temperature | 1.0 | Ensures diversity |
| Group size G | 64 | 64 responses per prompt |
| Prompts per step | 16 | |
| Max generation length | 32,768 tokens | Supports long reasoning chains |
| Policy updates per batch | 1 (single epoch) | Avoid overfitting |

**Special handling of clip ratio**: DeepSeek-R1 used a very large clip ratio (ε=10) late in training — drastically different from standard PPO (ε=0.2). This suggests larger updates are beneficial in late training.

### Reward Signal Design

| Task type | Reward source | Reward value |
|---------|---------|-------|
| Math | Answer matching | r=1 (correct), r=0 (wrong) |
| Code | Unit tests pass | r=1 (all pass), r=0 (fail) |
| Format | Regex match `<think>...</think><answer>...</answer>` | r=+0.5 (correct format), r=-0.5 (wrong format) |
| Open-ended | Learned reward model | Continuous score |

---

## Code Examples

### GRPO Training Loop Pseudocode

```python
import torch
import torch.nn.functional as F

class GRPOTrainer:
    """GRPO trainer (simplified)"""
    
    def __init__(self, policy, ref_policy, reward_fn,
                 group_size=16, lr=1e-5, clip_eps=0.2, 
                 kl_coef=0.01):
        self.policy = policy           # trainable
        self.ref_policy = ref_policy   # frozen
        self.reward_fn = reward_fn     # RM or rule verifier
        
        self.G = group_size
        self.clip_eps = clip_eps
        self.kl_coef = kl_coef
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    
    def train_step(self, prompts):
        """One GRPO training step"""
        batch_size = len(prompts)
        
        # === Step 1: Sample G responses ===
        all_responses = []
        all_old_log_probs = []
        
        with torch.no_grad():
            for prompt in prompts:
                # G responses per prompt
                responses, log_probs = self.policy.generate(
                    prompt, 
                    num_samples=self.G,
                    temperature=1.0,
                    return_log_probs=True,
                )
                all_responses.append(responses)       # (G, seq_len)
                all_old_log_probs.append(log_probs)   # (G,) sequence-level
        
        # === Step 2: Compute rewards ===
        all_rewards = []
        for prompt, responses in zip(prompts, all_responses):
            rewards = self.reward_fn(prompt, responses)  # (G,)
            all_rewards.append(rewards)
        
        # === Step 3: Intra-group normalization → advantages ===
        all_advantages = []
        for rewards in all_rewards:
            mu = rewards.mean()
            sigma = rewards.std() + 1e-8
            advantages = (rewards - mu) / sigma   # (G,)
            all_advantages.append(advantages)
        
        # === Step 4: PPO-style clipped update ===
        self.optimizer.zero_grad()
        total_loss = 0
        
        for i in range(batch_size):
            prompt = prompts[i]
            responses = all_responses[i]      # (G, seq_len)
            old_lp = all_old_log_probs[i]     # (G,)
            advantages = all_advantages[i]     # (G,)
            
            # Recompute log_probs under current policy
            new_lp = self.policy.log_probs(prompt, responses)  # (G,)
            
            # Probability ratio
            ratio = torch.exp(new_lp - old_lp)  # (G,)
            
            # Clipped surrogate
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-self.clip_eps, 
                                1+self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # KL penalty
            ref_lp = self.ref_policy.log_probs(prompt, responses)
            kl = (new_lp - ref_lp).mean()
            
            total_loss += policy_loss + self.kl_coef * kl
        
        total_loss /= batch_size
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
        
        # Stats
        mean_reward = torch.cat(all_rewards).mean().item()
        return {"loss": total_loss.item(), "mean_reward": mean_reward}
```

### Using TRL GRPOTrainer

```python
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

# === Load model ===
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
)

# === Reward function ===
def reward_function(completions, **kwargs):
    """
    Rule-based reward function (math task)
    
    Args:
        completions: list of model-generated responses
    Returns:
        rewards: list of reward values
    """
    rewards = []
    for completion in completions:
        # Check format
        has_think = "<think>" in completion and "</think>" in completion
        # Extract answer
        answer = extract_answer(completion)
        # Verify correctness
        correct = verify_answer(answer, ground_truth)
        
        reward = 0.0
        if has_think:
            reward += 0.5    # format reward
        if correct:
            reward += 1.0    # correctness reward
        rewards.append(reward)
    return rewards

# === GRPO config ===
config = GRPOConfig(
    output_dir="grpo_output",
    
    # --- Core GRPO params ---
    num_generations=16,            # group size G
    
    # --- PPO-style params ---
    learning_rate=5e-6,
    cliprange=0.2,
    
    # --- KL penalty ---
    beta=0.01,                     # KL coefficient
    
    # --- Batch ---
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    
    # --- Generation ---
    max_completion_length=2048,
    temperature=1.0,
    
    # --- Misc ---
    logging_steps=10,
    save_steps=100,
    bf16=True,
)

# === Create trainer and train ===
trainer = GRPOTrainer(
    model=model,
    config=config,
    tokenizer=tokenizer,
    train_dataset=math_dataset,
    reward_funcs=reward_function,
)

trainer.train()
```

### Configuration for Reasoning Tasks

```python
# === GRPO config tuned for reasoning ===
reasoning_config = GRPOConfig(
    # Larger group size (reasoning tasks are high-variance)
    num_generations=64,
    
    # Longer generation (supports long reasoning chains)
    max_completion_length=8192,
    
    # Lower KL penalty (allow more exploration)
    beta=0.001,
    
    # Wider clip range (reasoning needs larger policy changes)
    cliprange=0.2,   # can be increased late in training
    
    # Temperature 1.0 ensures diversity
    temperature=1.0,
    
    # bf16 for memory
    bf16=True,
    
    # Gradient checkpointing (mandatory for long sequences)
    gradient_checkpointing=True,
)
```

---

## GRPO Variants and Improvements

### DAPO (Decoupled Clip and Dynamic Sampling Policy Optimization)

**Source**: ByteDance Seed + Tsinghua AIR ([arXiv:2503.14476](https://arxiv.org/abs/2503.14476))

DAPO proposes four key improvements on top of GRPO:

#### 1. Clip-Higher (Asymmetric Clipping)

Standard PPO/GRPO uses symmetric clipping [1-ε_low, 1+ε_high] with ε_low = ε_high.
DAPO uses asymmetric clipping:

```
ε_low = 0.2 (normal)
ε_high = 0.28 (larger upper bound)
```

**Intuition**: symmetric clipping leads to "entropy collapse" — the policy quickly becomes deterministic and loses exploration. A larger upper-bound clip lets the policy more easily raise the probability of low-probability actions, promoting diversity.

#### 2. Dynamic Sampling

When all responses in a group are correct (or all wrong), advantages are all zero and that group contributes nothing to training. DAPO dynamically filters these "uninformative" groups:

```python
# Keep only informative groups
useful_groups = [g for g in groups if 0 < sum(g.rewards) < len(g.rewards)]
```

This dramatically improves training efficiency.

#### 3. Token-Level Policy Gradient Loss

Standard GRPO uses a sequence-level probability ratio:

```
ρ_seq = Π_t π_θ(y_t|...) / π_old(y_t|...)
```

DAPO switches to a token-level loss with sequence-length normalization:

```
L = (1/T) Σ_t min(ρ_t · A, clip(ρ_t, 1-ε, 1+ε) · A)
```

This is critical for long chains-of-thought — otherwise long sequences would produce far larger gradients than short ones.

#### 4. Overlong Reward Shaping

When generation exceeds max length and gets truncated, assigning r=0 introduces noise (the answer may have been close to correct). DAPO uses graduated penalty:

```
r_overlong = max(min_reward, r_original - penalty * (len - max_len))
```

**DAPO performance**: 50 on AIME 2024 (DeepSeek-R1-Zero scored 47) using 50% fewer training steps. Trained on [[rl-training-frameworks#veRL|veRL]].

### Dr. GRPO (Variance-Reduced GRPO)

**Problem**: standard GRPO's group normalization introduces bias. For finite group size G:

```
E[A_j] = E[(r_j - μ) / σ] ≠ 0     (non-zero bias)
```

This is because σ in the denominator is a random variable.

**Dr. GRPO's fix**:
- Drop std normalization: A_j = r_j - μ (subtract mean only)
- Add a variance correction term in the loss
- Theoretically unbiased, more stable in practice

### RLOO (REINFORCE Leave-One-Out)

A method closely related to GRPO that uses a leave-one-out baseline:

```
For the j-th sample:
  baseline_j = (1/(G-1)) Σ_{k≠j} r_k     (mean excluding itself)
  A_j = r_j - baseline_j
```

This has lower variance than GRPO's group-mean baseline (because the baseline is independent of the sample being evaluated).

### REINFORCE++ vs GRPO

| Feature | GRPO | REINFORCE++ |
|------|------|-------------|
| Baseline | Intra-group mean | Running mean (exponential moving average) |
| Sampling | G per prompt | 1 per prompt |
| Normalization | Intra-group std | Global variance |
| Compute overhead | High (G× rollouts) | Low (single rollout) |
| Use case | Large-scale reasoning RL | General RLHF |

---

## Performance Analysis

### Benchmark Comparison

| Model/Method | AIME 2024 | MATH-500 | Codeforces | Training cost |
|-----------|-----------|----------|------------|---------|
| DeepSeek-R1-Zero (GRPO, pure RL) | 71.0% | 95.9% | 1444 Elo | baseline |
| DeepSeek-R1 (GRPO, full pipeline) | 79.8% | 97.3% | 2029 Elo | ~2× |
| OpenAI o1 | 79.2% | 96.4% | 2061 Elo | undisclosed |
| DAPO (GRPO improvement) | ~50/90 * | similar | - | ~0.5× R1-Zero |

*AIME scoring scheme differs

### GRPO vs PPO Performance

Per multiple independent studies (Yu et al. 2025, Hu et al. 2025):

- On reasoning tasks (math, code): GRPO ≈ PPO (even slightly better)
- On alignment tasks (helpfulness, harmlessness): GRPO ≈ PPO (with sufficient group size)
- Training speed: GRPO is 30-50% faster (no critic overhead)
- Tuning difficulty: GRPO is notably easier (fewer hyperparameters)

### Effect of Group Size

```
       Performance
  ↑    ___________________
  |   /
  |  /
  | /     Performance rises with group size
  |/      but with diminishing returns
  |
  +──────────────────────> group size G
  1    8    16   32   64   128

  Recommendations:
  - Easy tasks: G=8~16
  - Reasoning tasks: G=32~64
  - Beyond 64, marginal gains are small
```

---

## Limitations

### 1. Group-Size Sensitivity

- **G too small** (e.g., G=2~4): group statistics are noisy, advantage estimates unreliable, normalization bias significant
- **G too large** (e.g., G>64): each prompt needs many rollouts, compute cost is high
- Optimal G depends on task complexity and reward distribution — requires empirical tuning

### 2. Reward Model Dependence

- For non-verifiable tasks, GRPO still relies on a learned reward model
- RM biases can be amplified by intra-group comparison (if all G responses share the same bias direction)
- Intra-group normalization cannot correct systematic RM biases

### 3. Information Loss from Sequence-Level Advantage

- GRPO only has sequence-level advantage signal, lacking PPO's token-level credit assignment
- For responses with "good first half, bad second half", it cannot give precise feedback
- In long sequences, which tokens contributed to the reward is unclear

### 4. Sampling Efficiency

- Each prompt requires G samples, so inference compute is G× that of PPO
- For long sequences (e.g., 32K tokens), rollout generation is the main training bottleneck
- Requires an efficient inference engine (e.g., vLLM) to accelerate sampling

### 5. Intra-Group Diversity Requirement

- GRPO depends on quality variation among responses in a group to produce meaningful advantage signal
- If the policy is already strong (most responses are correct), intra-group variation vanishes and training signal weakens
- This motivates DAPO's Dynamic Sampling

### 6. Limited Theoretical Guarantees

- Group normalization in GRPO introduces bias (finite-sample statistical bias)
- Convergence proof is weaker than PPO's
- Theoretical upper bound on the optimality gap vs PPO is unclear

---

## References

- Shao et al. (2024) — [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) (original GRPO paper)
- DeepSeek-AI (2025) — [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
- DeepSeek-R1 Nature 2025 — [doi:10.1038/s41586-025-09422-z](https://www.nature.com/articles/s41586-025-09422-z)
- Yu et al. (2025) — [Revisiting GRPO: On-Policy and Off-Policy](https://arxiv.org/html/2505.22257v1)
- DAPO (2025) — [arXiv:2503.14476](https://arxiv.org/abs/2503.14476)
- Hu et al. (2025) — [REINFORCE++](https://arxiv.org/abs/2501.03262)
- Ahmadian et al. (2024) — [Back to Basics: Revisiting REINFORCE-Style Optimization](https://arxiv.org/abs/2402.14740)
- Liu et al. (2025) — [Understanding R1-Zero-Like Training](https://arxiv.org/abs/2503.20783)

---

## Related Pages

- [[ppo-for-llm]] — critic-based PPO, the predecessor of GRPO
- [[rlhf-overview]] — the complete RLHF three-stage pipeline
- [[dpo]] — Direct Preference Optimization, another simplification of RLHF
- [[reward-modeling]] — sources of reward signal: learned RM vs rule verifiers
- [[rl-training-frameworks]] — training frameworks supporting GRPO (veRL, OpenRLHF, TRL)
- [[multi-step-reasoning-rl]] — large-scale GRPO applied to reasoning models
