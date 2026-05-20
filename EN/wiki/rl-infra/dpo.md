---
title: "DPO: Direct Preference Optimization"
category: rl-infra
tags: [dpo, preference-optimization, alignment, offline-rl, simpo, kto, ipo, orpo]
created: 2026-04-13
updated: 2026-05-20
status: mature
---

# DPO: Direct Preference Optimization

> [!abstract]+ TL;DR
> DPO (Rafailov et al., 2023) **skips reward model training entirely**: instead of SFT → train RM → [[ppo-for-llm|PPO]], it directly optimizes the LM on preference pairs with a supervised learning objective. Only **2 models** needed (policy + frozen reference), compared to PPO's 4 (actor + critic + ref + reward) — less memory, no RL instability, simpler implementation. Variants: **IPO** (regularized), **KTO** (unpaired binary feedback), **ORPO** (reference-free), **SimPO** (reference-free, +6.4 on AlpacaEval 2 over DPO). ICML 2024 finding: well-tuned PPO can match or exceed DPO accuracy, but DPO is far simpler to implement.

## Overview

DPO (Direct Preference Optimization, Rafailov et al., 2023) is a landmark work in LLM alignment. Its key insight: **skip reward model training entirely** and directly optimize the language model on human preference data with a supervised learning objective.

The traditional [[rlhf-overview|RLHF]] pipeline requires three stages:
1. SFT (supervised fine-tuning)
2. Train a reward model (RM)
3. Optimize the policy with [[ppo-for-llm|PPO]] or another RL algorithm

DPO merges steps 2 and 3 into one: train directly on preference triples (prompt, chosen, rejected). The whole training process needs only **2 models** (policy + frozen reference), whereas PPO needs 4 (actor, critic, reference, reward model).

```
Traditional RLHF pipeline:
┌──────────────────────────────────────────────────────────────────┐
│  Human preference data ──→ Train RM ──→ PPO policy optimization   │
│  (prompt, y_w, y_l)         (RM)        (Actor + Critic + Ref + RM)│
│                                          4 models, unstable training│
└──────────────────────────────────────────────────────────────────┘

DPO pipeline:
┌──────────────────────────────────────────────────────────────────┐
│  Human preference data ──→ Directly optimize policy                │
│  (prompt, y_w, y_l)         (Policy + Ref)                        │
│                              2 models, stable supervised learning  │
└──────────────────────────────────────────────────────────────────┘
```

This simplification yields significant engineering wins: lower GPU memory footprint, no RL hyperparameters to tune, and far better training stability. The DPO paper quickly became one of the most popular alignment methods after its release.

---

## Derivation: From RLHF to DPO

The mathematical derivation of DPO is key to understanding its essence. We start from the RLHF objective and step-by-step derive the DPO loss.

### Step 1: The RLHF objective

The core RLHF objective is to maximize reward while preventing the policy from drifting too far from the reference. Mathematically:

```
max_π  E_{x~D, y~π(·|x)} [ r(x, y) ] - β · KL( π(·|x) || π_ref(·|x) )
```

Where:
- `r(x, y)` is the (learned) reward model score for response y
- `π(·|x)` is the policy we are optimizing
- `π_ref(·|x)` is the reference policy (typically the SFT model)
- `β` is the temperature controlling KL penalty strength
- `KL` divergence prevents policy degeneration (reward hacking)

### Step 2: Closed-form optimal policy

Expanding the KL divergence:

```
KL(π || π_ref) = E_{y~π} [ log π(y|x) - log π_ref(y|x) ]
```

Substituting into the objective and expanding:

```
max_π  E_{y~π} [ r(x,y) - β · log π(y|x) + β · log π_ref(y|x) ]
     = max_π  E_{y~π} [ r(x,y) + β · log π_ref(y|x) - β · log π(y|x) ]
```

This is a standard max-entropy RL problem. Taking the variational optimum over π(y|x), setting derivative to zero:

```
r(x,y) + β · log π_ref(y|x) - β · log π(y|x) - β = 0
```

(The trailing `-β` comes from the Lagrange multiplier for the normalization constraint.)

Rearranging gives the closed-form optimal policy:

```
π*(y|x) = (1/Z(x)) · π_ref(y|x) · exp( r(x,y) / β )
```

Where `Z(x)` is the partition function (normalization constant):

```
Z(x) = Σ_y π_ref(y|x) · exp( r(x,y) / β )
```

**Intuition**: The optimal policy is the reference policy reweighted exponentially by reward. High-reward responses get amplified, low-reward ones suppressed.

### Step 3: Express reward via the policy

This is the crucial step in DPO. Take the log of the optimal policy formula and rearrange:

```
log π*(y|x) = log π_ref(y|x) + r(x,y)/β - log Z(x)
```

Solve for `r(x,y)`:

```
r(x, y) = β · log( π*(y|x) / π_ref(y|x) ) + β · log Z(x)
```

**Key insight**: The reward function can be expressed entirely in terms of the log-probability ratio of policy to reference. The partition function `Z(x)` depends only on prompt x, not on response y.

### Step 4: Substitute into the Bradley-Terry preference model

The Bradley-Terry model describes human preference probabilities:

```
P(y_w ≻ y_l | x) = σ( r(x, y_w) - r(x, y_l) )
```

Where `σ` is the sigmoid, `y_w` is the chosen (preferred) response, `y_l` is the rejected one.

Substituting the reward expression from Step 3:

```
r(x, y_w) - r(x, y_l)
= β · log(π(y_w|x) / π_ref(y_w|x)) + β·log Z(x)
  - β · log(π(y_l|x) / π_ref(y_l|x)) - β·log Z(x)
```

**Crucially: `β · log Z(x)` cancels out!**

```
r(x, y_w) - r(x, y_l) = β · [ log(π(y_w|x)/π_ref(y_w|x)) - log(π(y_l|x)/π_ref(y_l|x)) ]
```

This means we never need to compute the intractable partition function Z(x).

### Step 5: The final DPO loss

Substituting into the negative log-likelihood of the Bradley-Terry model (i.e., maximize the likelihood of the preference data), we get DPO's training objective:

```
L_DPO(π_θ; π_ref) = -E_{(x, y_w, y_l) ~ D} [
    log σ( β · ( log(π_θ(y_w|x) / π_ref(y_w|x))
                - log(π_θ(y_l|x) / π_ref(y_l|x)) ) )
]
```

More compactly, define the implicit reward margin:

```
Δ = β · log(π_θ(y_w|x)/π_ref(y_w|x)) - β · log(π_θ(y_l|x)/π_ref(y_l|x))

L_DPO = -E [ log σ(Δ) ]
```

This is a standard **binary cross-entropy loss**: push the model's implicit reward (log-prob ratio to reference) on the chosen response above that of the rejected response.

### Derivation summary

```
RLHF objective ──→ Closed-form optimum ──→ Reward via policy ──→ Plug into BT model
   │                  │                       │                    │
   │             π* ∝ π_ref·exp(r/β)        r = β·log(π/π_ref)   Z(x) cancels
   │                  │                    + β·log Z(x)             │
   └──→ No RM training ←──── No Z(x) computation ←──── DPO loss
```

---

## Intuition

### Gradient analysis

The gradient of the DPO loss tells us what it actually does:

```
∇_θ L_DPO ∝ -β · E [ σ(-Δ) · (
    ∇_θ log π_θ(y_w|x) - ∇_θ log π_θ(y_l|x)
)]
```

Where `σ(-Δ)` is a weighting term:
- When the model is already "right" (Δ large, i.e., higher prob on chosen), `σ(-Δ) ≈ 0`, gradient small → **little update**
- When the model is "wrong" (Δ small or negative), `σ(-Δ) ≈ 1`, gradient large → **big update**

So DPO simultaneously does two things:
1. **Increase** the probability of chosen `y_w` (direction `+∇_θ log π_θ(y_w|x)`)
2. **Decrease** the probability of rejected `y_l` (direction `-∇_θ log π_θ(y_l|x)`)
3. Strength dynamically modulated by current "wrongness"

### Role of the reference model

```
DPO training flow:
                        ┌─────────────┐
  Preference data       │ Reference   │ (frozen)
  (x, y_w, y_l) ──────→│  π_ref       │──→ log π_ref(y_w|x), log π_ref(y_l|x)
       │                └─────────────┘              │
       │                ┌─────────────┐              │
       └───────────────→│   Policy    │              │
                        │   π_θ       │──→ log π_θ(y_w|x), log π_θ(y_l|x)
                        └──────┬──────┘              │
                               │                      │
                               ▼                      ▼
                        ┌──────────────────────────────┐
                        │  Compute implicit margin Δ   │
                        │  Δ = β·[log(π_θ/π_ref)(y_w)  │
                        │      - log(π_θ/π_ref)(y_l)]  │
                        └──────────────┬───────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │  L = -log σ(Δ)               │
                        │  Backprop, update π_θ        │
                        └──────────────────────────────┘
```

The reference model acts as an "anchor":
- Prevents catastrophic forgetting in the policy
- Limits how far the model drifts from its original distribution
- Without it, optimization can lead to degeneration (extreme/repetitive outputs)

### Intuitive comparison with RLHF

```
RLHF: Learn a "judge" (RM) first, then practice under the judge (PPO)
       → Judge may be biased, practice may be unstable

DPO:  Learn directly from "match results" (preference data), skip the judge
       → More direct, but only learns from past matches (offline)
```

---

## Implementation Details

### Data format

DPO training data is a triple: `(prompt, chosen, rejected)`

```python
# Typical DPO dataset format
{
    "prompt": "Explain what quantum entanglement is",
    "chosen": "Quantum entanglement is a quantum mechanical phenomenon where two particles... (high-quality response)",
    "rejected": "Quantum entanglement is just two things being connected... (low-quality response)"
}
```

**Data sources**:
1. **Human annotation**: humans rank multiple responses to the same prompt
2. **AI feedback**: a strong model (GPT-4, Claude) scores responses
3. **Self-play**: model generates multiple responses, picked by rules/verifiers
4. **Existing datasets**: UltraFeedback, Anthropic-HH, Stanford Human Preferences

### Reference model management

```python
# Option 1: Frozen copy (most common)
ref_model = AutoModelForCausalLM.from_pretrained("sft_model")
ref_model.eval()
for param in ref_model.parameters():
    param.requires_grad = False

# Option 2: Periodically update reference (Online DPO)
# Every N steps, copy policy weights to reference
if step % update_interval == 0:
    ref_model.load_state_dict(policy_model.state_dict())

# Option 3: Reference-free (SimPO, ORPO)
# Use the policy's own average log-prob directly as reward
```

**Memory optimization**: the reference model needs no gradients, so it can be loaded in half precision or even quantized:

```python
ref_model = AutoModelForCausalLM.from_pretrained(
    "sft_model",
    torch_dtype=torch.bfloat16,
    load_in_4bit=True,  # quantize to save memory
)
```

### Impact of β

`β` is the most important DPO hyperparameter; it controls KL penalty strength:

| β value | Behavior | Use case |
|---------|----------|----------|
| β → 0 | Ignore reference, aggressive update | Almost never used, prone to degeneration |
| β = 0.05-0.1 | Strong preference learning | Clean preference signals |
| β = 0.1-0.5 | Common range | Most scenarios |
| β = 0.5-1.0 | Conservative, stay close to ref | Noisy preference data |
| β → ∞ | No update | Useless |

**Tuning strategy**:
- Start with `β = 0.1`
- If training is unstable or outputs degenerate, increase β
- If the model barely changes (weak alignment), decrease β
- Monitor the log-prob gap between chosen and rejected: it should grow during training

### Label smoothing

Human-labeled preference pairs are not always correct. Label smoothing increases robustness to noise:

```python
# Standard DPO loss
loss = -log(σ(Δ))

# DPO loss with label smoothing (ε ∈ [0, 0.5])
loss = -(1-ε) · log(σ(Δ)) - ε · log(σ(-Δ))
```

TRL's DPOTrainer supports `label_smoothing`:

```python
training_args = DPOConfig(
    label_smoothing=0.1,  # 10% label smoothing
    ...
)
```

### Key metrics during training

```
Metrics to monitor:
┌──────────────────────────────────────────────────────────────┐
│  1. loss (should decrease)                                    │
│  2. rewards/chosen (implicit reward of chosen, should rise)   │
│  3. rewards/rejected (implicit reward of rejected, should fall)│
│  4. rewards/margins (chosen - rejected, should grow)          │
│  5. rewards/accuracies (fraction correctly ranked, should rise)│
│  6. logps/chosen (log-prob of chosen, must NOT fall sharply!) │
│  7. logps/rejected (log-prob of rejected, should fall)         │
└──────────────────────────────────────────────────────────────┘
```

**Watch closely**: if `logps/chosen` drops sharply, the model is degrading the probability of preferred responses — the "chosen response degradation" problem.

---

## DPO Variants

DPO has spawned a host of variants, each addressing different limitations.

### IPO (Identity Preference Optimization)

**Paper**: Azar et al. (Google DeepMind), 2023

**Motivation**: DPO's assumption of the Bradley-Terry preference model is too strong — it assumes human preferences can be precisely modeled by a sigmoid of reward differences. With noisy preferences, DPO can overfit.

**Method**: Drop the Bradley-Terry assumption and directly regularize the preference probability:

```
L_IPO = E [ (log(π_θ(y_w|x)/π_ref(y_w|x)) - log(π_θ(y_l|x)/π_ref(y_l|x)) - 1/(2β))² ]
```

**Advantages**:
- More robust to label noise
- Can be safely trained to convergence (DPO degenerates with over-training)
- No Bradley-Terry assumption

### KTO (Kahneman-Tversky Optimization)

**Paper**: Ethayarajh et al. (Stanford), 2024

**Motivation**: Collecting paired preferences (chosen/rejected for the same prompt) is expensive. Can we train with just binary feedback (good/bad)?

**Core idea**: Borrow Prospect Theory from behavioral economics:
- Humans are more sensitive to losses than gains (loss aversion)
- The utility function is not linear

```
KTO loss:
L_KTO = E_w [ w(x,y) · (1 - σ(β · log(π_θ(y|x)/π_ref(y|x)) - z_ref)) ]  (good)
      + E_l [ w(x,y) · (1 - σ(z_ref - β · log(π_θ(y|x)/π_ref(y|x)))) ]  (bad)
```

Where `z_ref` is a reference point (baseline estimate from the reference policy) and `w(x,y)` is a loss-aversion coefficient.

**Advantages**:
- Only needs **unpaired** binary feedback, no pairs required
- Higher data efficiency, can leverage more low-cost data
- Particularly useful in data-scarce settings

### ORPO (Odds Ratio Preference Optimization)

**Paper**: Hong et al. (KAIST), 2024

**Motivation**: DPO still requires two stages — SFT then DPO. Can we merge them?

**Method**: Combine SFT loss and preference optimization, using odds ratio for preferences:

```
L_ORPO = L_SFT + λ · L_OR

L_OR = -E [ log σ( log(odds(y_w|x) / odds(y_l|x)) ) ]
odds(y|x) = π_θ(y|x) / (1 - π_θ(y|x))
```

**Advantages**:
- **No reference model**, further reducing memory
- **SFT + preference learning in one stage**
- Simplified training pipeline

### SimPO (Simple Preference Optimization)

**Paper**: Meng et al. (Princeton), NeurIPS 2024

**Motivation**: DPO's implicit reward (`log(π/π_ref)`) is unavailable at inference time because the reference model isn't used then. SimPO argues we should use the metric actually used at inference (sequence log probability) as the reward.

**Method**: Use **average log probability** as the implicit reward, no reference model needed:

```
r_SimPO(x, y) = (1/|y|) · log π_θ(y|x)

L_SimPO = -E [ log σ( β · (r_SimPO(x, y_w) - r_SimPO(x, y_l)) - γ ) ]
```

Where `γ > 0` is a target reward margin.

**Advantages**:
- Reference-free (less memory)
- Length-normalized, avoids bias toward long responses
- +6.4 LC win-rate on AlpacaEval 2 vs DPO
- +7.5 on Arena-Hard vs DPO

### Online DPO / Iterative DPO

**Motivation**: Standard DPO is offline — it can only learn from a fixed preference dataset. This causes distribution mismatch: training data comes from the SFT model, but the policy's distribution shifts during training.

**Method**:
1. Generate new responses with the current policy
2. Label preferences on the new responses using an RM (or human, AI)
3. Run DPO on the new data
4. Repeat

```
Online DPO loop:
┌─────────────────────────────────────────────────┐
│  π_θ generates new responses ──→ RM/human labels ──→ DPO  │
│      ▲                                       │     │
│      └──────────────────────────────────────┘     │
│                    Iterate                          │
└─────────────────────────────────────────────────┘
```

**Note**: Online DPO reintroduces a reward model, partly losing DPO's "no-RM" advantage, but it fixes the distribution mismatch.

### Variant comparison

| Variant | Paired data? | Reference model? | SFT stage? | Key innovation |
|---------|:-----------:|:---------------:|:-----------:|----------------|
| **DPO** | Yes | Yes | Yes | Reward expressed via policy |
| **IPO** | Yes | Yes | Yes | Regularization replaces BT |
| **KTO** | **No** (binary) | Yes | Yes | Prospect Theory |
| **ORPO** | Yes | **No** | **No** (merged) | Odds ratio + SFT in one |
| **SimPO** | Yes | **No** | Yes | Avg log prob + margin |
| **Online DPO** | Online generation | Yes | Yes | Iteratively generate new data |
| **BPO** (ICLR 2026) | Yes | Yes | Yes | Fix chosen degradation |

---

## DPO vs RLHF/PPO

### Advantages

| Dimension | DPO | PPO |
|-----------|-----|-----|
| # of models | 2 (policy + reference) | 4 (actor + critic + reference + RM) |
| GPU memory | Low | High (~2-3x) |
| Training stability | High (supervised) | Low (RL is unstable) |
| Hyperparameter sensitivity | Low (mainly β) | High (lr, clip range, GAE, ...) |
| Implementation complexity | Simple | Complex |
| Training speed | Fast | Slow (generation + training loop) |

### Disadvantages

| Dimension | DPO | PPO |
|-----------|-----|-----|
| Data source | Offline (fixed dataset) | Online (generates new data) |
| Exploration | None | Yes |
| Distribution matching | Easy to mismatch | Natural match (on-policy) |
| Multi-turn / agentic | Doesn't extend naturally | Natural support |
| Reward signal | Implicit | Explicit, flexible |
| Reasoning tasks | Weaker | Stronger (can explore) |

### Selection guide

```
Choose DPO when:
  ✓ You have high-quality paired preference data
  ✓ Single-turn alignment (helpfulness, harmlessness)
  ✓ Compute-constrained
  ✓ Need fast iteration
  ✓ Team lacks RL engineering experience

Choose PPO/GRPO when:
  ✓ Reasoning tasks (math, code)
  ✓ Agentic / tool-use scenarios
  ✓ Multi-turn interaction
  ✓ Need online exploration
  ✓ Have verifiable rewards (RLVR)
  ✓ Chasing peak performance
```

**ICML 2024 key finding**: PPO, properly tuned, can match or exceed DPO — but engineering overhead is huge.

**2025-2026 trends**:
- Simple alignment tasks → DPO and variants (SimPO, KTO)
- Reasoning and agentic tasks → [[grpo|GRPO]], [[ppo-for-llm|PPO]], REINFORCE++
- DeepSeek-R1 chose [[grpo|GRPO]] over DPO for reasoning training

---

## Code Example

### Using TRL DPOTrainer

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

# 1. Load model
model_name = "meta-llama/Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="bfloat16",
    attn_implementation="flash_attention_2",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 2. Load preference data
# Format: {"prompt": str, "chosen": str, "rejected": str}
dataset = load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned")

# 3. Configure DPO training
training_args = DPOConfig(
    output_dir="./dpo_output",
    beta=0.1,                    # KL penalty strength
    learning_rate=5e-7,          # DPO typically uses a small LR
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=1,          # DPO usually trains 1 epoch
    warmup_ratio=0.1,
    logging_steps=10,
    bf16=True,
    gradient_checkpointing=True,
    label_smoothing=0.0,         # Optional: 0.1 for noisy data
    max_length=2048,
    max_prompt_length=1024,
)

# 4. Initialize DPOTrainer (auto-creates reference model)
trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    processing_class=tokenizer,
    # When ref_model=None, TRL auto-creates a frozen copy
)

# 5. Train
trainer.train()

# 6. Save
trainer.save_model("./dpo_final")
```

### Data preparation example

```python
def prepare_dpo_data(raw_dataset):
    """
    Convert raw annotation data to DPO format.
    Assumes raw data contains a prompt and several scored responses.
    """
    dpo_data = []
    for item in raw_dataset:
        prompt = item["prompt"]
        responses = item["responses"]
        scores = item["scores"]

        # Sort by score
        sorted_pairs = sorted(zip(responses, scores),
                              key=lambda x: x[1], reverse=True)

        # Use highest and lowest as chosen/rejected
        chosen = sorted_pairs[0][0]
        rejected = sorted_pairs[-1][0]

        # Ensure score gap is large enough (filter noisy pairs)
        if sorted_pairs[0][1] - sorted_pairs[-1][1] > 0.5:
            dpo_data.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            })

    return dpo_data
```

### β tuning strategy

```python
# β grid search example
import wandb

betas = [0.05, 0.1, 0.2, 0.5]
for beta in betas:
    wandb.init(project="dpo-beta-search", name=f"beta_{beta}")

    training_args = DPOConfig(
        beta=beta,
        output_dir=f"./dpo_beta_{beta}",
        num_train_epochs=1,
        # ... other args
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
    )
    trainer.train()

    # Eval: check win-rate and whether logps/chosen degenerates
    metrics = trainer.evaluate()
    wandb.log(metrics)
    wandb.finish()
```

### TRL CLI quickstart (v1.0+)

```bash
# One-line DPO training
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

## Limitations and Debates

### "DPO is essentially offline RL"

Although DPO looks like supervised learning, it is mathematically equivalent to a specific offline RL algorithm. This means it inherits all the problems of offline RL:

1. **Distribution mismatch**: Preference data comes from the SFT model or humans, but the policy shifts during training. If the policy drifts too far from the data-generating distribution, the preference signals become inaccurate.

2. **No exploration**: DPO never explores behaviors outside the dataset. For tasks requiring creative reasoning (math, code), this is fatal.

### Chosen response degradation

Empirically, DPO can lower the probability of the rejected response and *also* lower the probability of the chosen response — even though the loss should in theory push chosen probability up.

```
Ideal case:                  What can actually happen:
  chosen ↑                    chosen ↓ (slight)
  rejected ↓                  rejected ↓↓ (sharp)
  margin grows ✓              margin still grows ✓ (but chosen got worse)
```

This is because DPO optimizes the **relative gap** between chosen and rejected, not the **absolute quality** of chosen. BPO (ICLR 2026) specifically addresses this issue.

### Online vs offline debate

| Position | Argument |
|----------|----------|
| DPO is enough | Simple, efficient, works well on chat alignment |
| Need online RL | Offline data has limited coverage, reasoning needs exploration, DeepSeek-R1 success proves online RL |

### Why DeepSeek chose GRPO over DPO

DeepSeek-R1 (2025), the flagship reasoning model, chose [[grpo|GRPO]] instead of DPO, for reasons including:

1. **Online exploration**: reasoning requires trying different ideas, DPO's offline data can't provide that
2. **Verifiable rewards**: math and code can be checked deterministically ([[reward-modeling#RLVR|RLVR]]), no RM training needed
3. **Bootstrapping**: the model needs to continuously learn from its own good/bad reasoning
4. **Long-chain reasoning**: DPO struggles with preference learning over long sequences

### Current consensus (2025-2026)

```
Role of DPO and variants:
  ✓ General chat alignment (helpfulness, harmlessness, style)
  ✓ Rapid prototype validation
  ✓ First choice when resources are constrained

Role of online RL (PPO/GRPO):
  ✓ Reasoning (math, code, logic)
  ✓ Agentic and tool-use tasks
  ✓ SOTA performance
  ✓ Self-improvement / self-play

Hybrid pipelines are increasingly common:
  DPO for baseline alignment → GRPO/PPO for reasoning/agent training
```

---

## References

- Rafailov et al. (2023) — [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
- Azar et al. (2023) — [A General Theoretical Paradigm to Understand Learning from Human Feedback (IPO)](https://arxiv.org/abs/2310.12036)
- Ethayarajh et al. (2024) — [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306)
- Hong et al. (2024) — [ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691)
- Meng et al. (2024) — [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734)
- Xu et al. (2024) — [Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study](https://arxiv.org/abs/2404.10719) (ICML 2024)
- Comprehensive DPO variants survey — [arXiv:2410.15595](https://arxiv.org/html/2410.15595v3)

---

## Related Pages

- [[ppo-for-llm]] — the online RL alternative DPO tries to simplify
- [[grpo]] — critic-free online RL, the core algorithm of DeepSeek-R1
- [[reward-modeling]] — the step DPO bypasses
- [[rlhf-overview]] — the full RLHF pipeline that DPO simplifies
- [[rl-training-frameworks]] — frameworks supporting DPO training (TRL, OpenRLHF, veRL)
- [[multi-step-reasoning-rl]] — limitations of DPO in reasoning scenarios
