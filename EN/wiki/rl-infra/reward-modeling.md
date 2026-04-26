---
title: "Reward Modeling"
category: rl-infra
tags: [reward-model, prm, orm, rlvr, reward-hacking, process-reward, bradley-terry]
created: 2026-04-13
updated: 2026-04-14
status: mature
---

# Reward Modeling

## Overview

Reward models (RMs) are the bridge between human preferences and RL optimization in the [[rlhf-overview|RLHF]] pipeline. They transform subjective judgments into scalar signals that drive policy optimization. **RM quality fundamentally caps aligned model quality** — systematic RM biases are learned by the policy (reward hacking).

The RM landscape in 2025-2026 extends well beyond "learn a scoring model":
- **Traditional RM**: Scalar model trained on human preferences
- **Process Reward Models (PRM)**: Score each reasoning step
- **Verifiable Rewards (RLVR)**: Deterministic programmatic verifiers replace learned RMs
- **Implicit Rewards**: [[dpo|DPO]] encodes rewards implicitly in the policy

---

## Reward Model Architecture

The core transformation: replace the LM head (next-token distribution) with a linear layer outputting a scalar reward score.

```
LLM:                          Reward Model:
Input Tokens                  Input: (prompt + response)
    ↓                              ↓
Transformer Backbone          Transformer Backbone
    ↓                              ↓
LM Head (vocab_size)          Scalar Head: Linear(d → 1)
  → next token dist.           → reward score (scalar)
```

**Architecture choices**:
| Strategy | Description | Trade-off |
|----------|-------------|-----------|
| Same-model init | RM = same pretrained model as policy | Good alignment, 2x memory |
| Smaller model | Policy 70B, RM 7B | Saves memory, less expressive |
| SFT init | Initialize RM from SFT checkpoint | Most common approach |

---

## Training Methods

### Bradley-Terry Preference Model

Human preference probability modeled as sigmoid of reward difference:

```
P(y_w ≻ y_l | x) = σ(r(x, y_w) - r(x, y_l))
```

**Training loss** (negative log-likelihood):

```
L_RM = -E [ log σ(r(x, y_w) - r(x, y_l)) ]
```

### Listwise Ranking Loss

When K > 2 ranked responses are available per prompt, extract C(K,2) comparison pairs:

```
L_listwise = -E [ Σ_{i<j} log σ(r(x, y_i) - r(x, y_j)) ]
```

InstructGPT used K=4 to K=9 responses per prompt.

### Margin-Based Loss

When annotations include confidence scores, enforce a minimum reward gap:

```
L_margin = -E [ log σ(r(x, y_w) - r(x, y_l) - m(y_w, y_l)) ]
```

---

## Reward Hacking

When the policy exploits spurious RM features to inflate scores without genuine improvement.

```
RM Score vs True Quality over Training:

  RM Score ↑↑↑ (keeps rising)
  True Quality ↑→↓ (rises then falls — overfitting to RM artifacts)
```

**Common patterns**: Verbosity bias (longer = higher score), sycophancy (agreeing with user), format gaming (exploiting markdown/structure preferences).

**Mitigation strategies**:

| Strategy | Effectiveness | Applicability |
|----------|:------------:|:-------------:|
| KL penalty | Medium | Universal |
| RM ensemble (conservative estimate) | Good | When compute allows |
| Length penalty | Targeted | Verbosity problem |
| Verifiable rewards (RLVR) | Best (but limited) | Math/code/factual |
| Constrained optimization | Good | When constraints are clear |

---

## Process Reward Models (PRM) vs Outcome Reward Models (ORM)

| Feature | ORM (Outcome) | PRM (Process) |
|---------|:-------------:|:-------------:|
| Granularity | Final response | Each reasoning step |
| Signal density | Sparse (1 per generation) | Dense (1 per step) |
| Labeling cost | Low | Very high |
| Credit assignment | Poor | Good |
| Best for | General alignment | Reasoning, math, code |

**OpenAI "Let's Verify Step by Step" (2023)**: PRM achieved 78.2% on MATH vs ORM's 72.4%.

**PRM at inference**: Guide search by scoring candidate reasoning paths per-step. More effective than ORM-based best-of-N.

### PRIME: Process Reinforcement through Implicit Rewards

Major 2025 advance ([arXiv:2502.01456](https://arxiv.org/abs/2502.01456)): Extract implicit per-token Q-values from an ORM to use as process reward — **no step-level annotation needed**.
- 2.5x sample efficiency, +6.9% over standard outcome RL
- Eurus-2-7B-PRIME surpasses Qwen2.5-Math-7B-Instruct on 7 benchmarks using 10% training data

---

## RLVR: RL from Verifiable Rewards

A major paradigm shift (2025-present): deterministic programmatic verifiers replace learned RMs.

```
Traditional RM:  Model generates → Learned RM scores → RL (hackable)
RLVR:            Model generates → Verifier checks  → RL (unhackable)
```

**Verifier types**:
- **Math**: Check if final answer matches ground truth (exact or symbolic equivalence)
- **Code**: Run test suites, check pass rate
- **Format**: Validate JSON, check constraints

**DeepSeek-R1's approach**: Pure RL from base model using [[grpo|GRPO]] + math/code verifiers. No learned RM at all. The model emergently learned chain-of-thought, self-verification, and backtracking.

**Debate**: RLVR models may be faster (better pass@1) but not necessarily smarter (base models can outperform at pass@256). Extending RLVR to non-verifiable domains remains an active research area.

---

## Code Example

### Training a Reward Model with TRL

```python
from trl import RewardConfig, RewardTrainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    num_labels=1,
    torch_dtype="bfloat16",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

training_args = RewardConfig(
    output_dir="./reward_model",
    learning_rate=1e-5,
    num_train_epochs=1,
    bf16=True,
    gradient_checkpointing=True,
)

trainer = RewardTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],  # {chosen, rejected}
    processing_class=tokenizer,
)
trainer.train()
```

### Evaluation

Evaluate on RewardBench across categories: chat, safety, reasoning, factuality. Key metric: accuracy on held-out preference pairs.

---

## References

- Ouyang et al. (2022) — [InstructGPT](https://arxiv.org/abs/2203.02155)
- Lightman et al. (2023) — [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)
- PRIME (2025) — [arXiv:2502.01456](https://arxiv.org/abs/2502.01456)
- DeepSeek-R1 (2025) — [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
- Lambert et al. (2024) — [RewardBench](https://arxiv.org/abs/2403.13787)
- [awesome-RLVR](https://github.com/opendilab/awesome-RLVR)

## Related Pages

- [[rlhf-overview]] — Where reward models fit in the pipeline
- [[ppo-for-llm]] — RL algorithm consuming reward signals
- [[grpo]] — Can use rule-based rewards instead of learned RMs
- [[dpo]] — Bypasses reward models entirely
- [[multi-step-reasoning-rl]] — PRMs guide reasoning
- [[rl-training-frameworks]] — Frameworks supporting RM training
