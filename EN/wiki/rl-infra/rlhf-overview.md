---
title: "RLHF: Reinforcement Learning from Human Feedback"
category: rl-infra
tags: [rlhf, alignment, reinforcement-learning, human-feedback, instructgpt, reward-model, sft, bradley-terry]
created: 2026-04-13
updated: 2026-04-14
status: mature
---

# RLHF: Reinforcement Learning from Human Feedback

> [!abstract]+ TL;DR
> RLHF is the dominant paradigm for aligning LLMs with human intent. Core insight: rewards for "helpful / truthful / harmless" can't be programmed, but **humans can reliably compare A vs. B**, so train a reward model from preference comparisons and fine-tune the LLM with RL to maximize that reward. Landmark result: InstructGPT (Ouyang et al., 2022) — a **1.3B RLHF model preferred over 175B GPT-3**. Three-stage pipeline: SFT → reward modeling (Bradley-Terry) → PPO/GRPO with KL penalty. Modern variants: [[grpo|GRPO]] (critic-free), [[dpo|DPO]] (no RL loop), RLAIF, RLVR.

### Why Not Just Supervised Learning?

SFT requires "gold answers" but open-ended dialogue has infinitely many valid responses. RLHF's breakthrough: **the model can explore beyond the training distribution and continuously improve via reward signals**.

---

## Historical Timeline

| Year | Milestone | Key Contribution |
|------|-----------|-----------------|
| **2017** | Christiano et al. -- "Deep RL from Human Preferences" | Foundational paper. Agent learned a backflip from ~900 bits of feedback. |
| **2019** | Ziegler et al. (OpenAI) | First RLHF on language models (GPT-2 summarization). Introduced KL penalty. |
| **2020** | Stiennon et al. -- "Learning to Summarize" | Scaled RLHF to 1.3B. RLHF summarizer outperformed SOTA. |
| **2022** | Ouyang et al. -- **InstructGPT** | Established SFT -> RM -> PPO pipeline. 1.3B beat 175B GPT-3. |
| **2022** | OpenAI -- **ChatGPT** | RLHF-trained model ignited global LLM interest. |
| **2023** | Anthropic -- **Constitutional AI** | AI feedback (RLAIF) replacing human annotators. |
| **2023** | Rafailov et al. -- **DPO** | No explicit reward model needed; direct preference optimization. |
| **2024** | DeepSeek -- **GRPO** | Critic-free PPO alternative; ~50% memory savings. |
| **2025** | DeepSeek -- **R1** | GRPO at scale for reasoning; emergent CoT. Published in Nature. |

---

## Three-Stage Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│                   RLHF Three-Stage Pipeline                  │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Stage 1: SFT                                                │
│  Pretrained LLM ──> SFT data (prompt, response) ──> π_SFT   │
│                                                   │          │
│  Stage 2: RM Training                             │          │
│  π_SFT (init RM) ──> Human preferences ──> r_φ(x,y)         │
│                                              │               │
│  Stage 3: RL Optimization (PPO)              │               │
│  π_SFT (ref) ──> KL penalty ──┐              │               │
│  π_θ generates ──> RM scores ──┘──> PPO ──> π_θ (aligned)   │
└──────────────────────────────────────────────────────────────┘
```

### Stage 1 -- Supervised Fine-Tuning (SFT)

Transforms the base model from "text completion" to "instruction following". Key points:
- Data: (prompt, desired_response) pairs; quality matters far more than quantity (LIMA showed 1K examples suffice)
- Loss: cross-entropy on response tokens only (prompt tokens masked)
- Typical: 1-3 epochs at low learning rate (1e-5 to 5e-6)
- Pitfalls: overfitting, catastrophic forgetting, format bias over content quality

### Stage 2 -- Reward Model Training

**Bradley-Terry preference model**: given prompt x and two responses y_w (preferred) and y_l (rejected):

```
P(y_w > y_l | x) = σ(r_φ(x, y_w) - r_φ(x, y_l))
```

**RM loss**:
```
L_RM(φ) = -E[log σ(r_φ(x, y_w) - r_φ(x, y_l))]
```

Architecture: SFT model with LM head replaced by a scalar output head. InstructGPT used ~33K comparison groups and achieved ~72% preference prediction accuracy.

### Stage 3 -- RL Optimization (PPO)

**Objective**:
```
max_θ  E_{x~D, y~π_θ}[r_φ(x,y)] - β · KL(π_θ || π_ref)
```

**Per-token reward decomposition**:
```
r_t = -β · kl_t                     (intermediate tokens)
r_T = r_RM(x,y) - β · kl_T          (last token)
```

Each iteration: sample responses -> RM scores -> GAE advantages -> PPO clipped update. See [[ppo-for-llm]] for full PPO derivation.

---

## Mathematical Formulations

**Bradley-Terry model** (equivalent to Elo ratings):
```
P(y_w ≻ y_l | x) = σ(r(x, y_w) - r(x, y_l))
```

**RM loss**:
```
L_RM = -E[log σ(r_φ(y_w) - r_φ(y_l))]
```

**RL objective** with KL-expanded reward:
```
r_total = r_φ(x,y) - β · Σ_t [log π_θ(y_t|x,y_{<t}) - log π_ref(y_t|x,y_{<t})]
```

**GAE advantage estimation**:
```
δ_t = r_t + γ · V(s_{t+1}) - V(s_t)
A_t^GAE = Σ_{l=0}^{∞} (γλ)^l · δ_{t+l}
```

---

## RLHF Variants and Evolution

### Online vs. Offline RLHF

| Dimension | Online (PPO, GRPO) | Offline (DPO, IPO, KTO) |
|-----------|--------------------|-------------------------|
| Data source | Current policy generates | Fixed pre-collected dataset |
| Exploration | Strong | None |
| Stability | Medium | High |
| Best for | Complex tasks (reasoning, code) | Simple alignment |

### RLAIF -- AI Feedback

**Constitutional AI** (Anthropic, 2023): AI judges responses against predefined principles, drastically reducing annotation cost while matching human feedback quality.

### RLVR -- Verifiable Rewards

For math/code tasks, bypass human feedback entirely: reward = 1 if answer is correct, 0 otherwise. Used in DeepSeek-R1. See [[reward-modeling#RLVR]].

### Iterative RLHF

Cycle the pipeline: RLHF round 1 -> generate with new policy -> collect new preferences -> update RM -> RLHF round 2. Avoids distribution shift in the reward model.

### Best-of-N Sampling

Simplest "RLHF" -- no RL needed: generate N responses, pick the highest-scoring one. Effective KL ~ log(N). Often used as a baseline.

---

## Code Examples

### TRL RLHF Training Loop

```python
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead

model = AutoModelForCausalLMWithValueHead.from_pretrained("sft_model")
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained("sft_model")

config = PPOConfig(
    learning_rate=1.41e-5, batch_size=64, mini_batch_size=16,
    ppo_epochs=4, init_kl_coef=0.2, target_kl=6.0,
    cliprange=0.2, gamma=1.0, lam=0.95,
)
trainer = PPOTrainer(config, model, ref_model, tokenizer)

for batch in dataloader:
    responses = trainer.generate(batch["query_tensors"])
    rewards = [reward_model.score(q, r) for q, r in zip(batch["queries"], responses)]
    stats = trainer.step(batch["query_tensors"], responses, rewards)
```

### Reward Model Training

```python
class RewardModel(nn.Module):
    def __init__(self, base_model_name):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(base_model_name)
        self.reward_head = nn.Linear(self.backbone.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        seq_lengths = attention_mask.sum(dim=1) - 1
        return self.reward_head(hidden[range(len(seq_lengths)), seq_lengths]).squeeze(-1)

# Bradley-Terry loss
r_chosen = rm(chosen_ids, chosen_mask)
r_rejected = rm(rejected_ids, rejected_mask)
loss = -torch.log(torch.sigmoid(r_chosen - r_rejected)).mean()
```

---

## Challenges and Open Problems

1. **Annotation bottleneck** -- expensive, slow, noisy (inter-annotator agreement ~70-80%), systematic bias toward longer/formal responses
2. **Reward model limitations** -- RM quality caps policy; poor OOD generalization; single scalar can't capture multi-dimensional quality
3. **[[reward-modeling#Reward Hacking|Reward hacking]]** -- policy exploits RM weaknesses (verbosity, sycophancy); mitigated by KL penalty and RM ensembles
4. **Training instability** -- KL coefficient tuning is delicate; PPO hyperparameters require significant expertise
5. **Alignment tax** -- RLHF may over-constrain the model, reducing creativity; safety-helpfulness tension
6. **Scalable oversight** -- when models surpass human evaluators, how do we ensure RLHF still works?

---

## References

- Christiano et al. (2017) -- [Deep RL from Human Preferences](https://arxiv.org/abs/1706.03741)
- Ziegler et al. (2019) -- [Fine-Tuning LMs from Human Preferences](https://arxiv.org/abs/1909.08593)
- Stiennon et al. (2020) -- [Learning to Summarize with Human Feedback](https://arxiv.org/abs/2009.01325)
- Ouyang et al. (2022) -- [InstructGPT](https://arxiv.org/abs/2203.02155)
- Bai et al. (2022) -- [Constitutional AI](https://arxiv.org/abs/2212.08073)
- Rafailov et al. (2023) -- [DPO](https://arxiv.org/abs/2305.18290)
- Lambert (2025) -- [RLHF Book](https://rlhfbook.com/book.pdf) ([arXiv:2504.12501](https://arxiv.org/abs/2504.12501))
- ACM Computing Surveys (2025) -- [RLHF Deciphered](https://dl.acm.org/doi/10.1145/3743127)

## Related Pages

- [[ppo-for-llm]] -- The RL algorithm behind RLHF
- [[grpo]] -- Critic-free alternative to PPO
- [[dpo]] -- Bypasses reward modeling entirely
- [[reward-modeling]] -- How reward models work
- [[rl-training-frameworks]] -- Frameworks for RLHF training
