---
title: "RLHF: Reinforcement Learning from Human Feedback"
category: rl-infra
tags: [rlhf, alignment, reinforcement-learning, human-feedback, instructgpt, reward-model, sft, bradley-terry]
created: 2026-04-13
updated: 2026-05-20
status: mature
---

# RLHF: Reinforcement Learning from Human Feedback

> [!abstract]+ TL;DR
> RLHF is the dominant paradigm for aligning LLMs with human intent. The key insight: "helpful/truthful/harmless" cannot be programmatically defined, but **humans can reliably compare A vs B**, so train a reward model on comparison data and use RL to fine-tune the LLM to maximize reward. Milestone: InstructGPT (Ouyang et al., 2022) — **the 1.3B RLHF model was rated by humans as better than the 175B GPT-3**. Three-stage pipeline: SFT → reward modeling (Bradley-Terry) → PPO/GRPO + KL penalty. Modern variants: [[grpo|GRPO]] (no critic), [[dpo|DPO]] (no RL loop), RLAIF, RLVR.

### Why Not Just Supervised Learning?

Supervised fine-tuning (SFT) needs "ground-truth answers", but for open-ended dialog tasks:
- The same question can have countless reasonable answers
- Human annotators struggle to write "perfect" responses from scratch
- SFT only mimics the distribution of training data and cannot exceed annotation quality

The breakthrough of RLHF: **the model can explore the answer space beyond the training data and continuously improve via reward signal**. The signature result from the InstructGPT paper (Ouyang et al., 2022) is that the 1.3B-parameter RLHF model was rated by humans as better than the 175B GPT-3.

---

## Historical Timeline

| Year | Milestone | Key contribution |
|------|--------|----------|
| **2017** | Christiano et al. — "Deep RL from Human Preferences" | Foundational paper. Reward model trained on human comparisons of trajectory snippets. The agent learned a backflip from only ~900 bits of feedback. First systematic framing of "preference learning → reward model → RL optimization". |
| **2019** | Ziegler et al. (OpenAI) — "Fine-Tuning Language Models from Human Preferences" | First application of RLHF to language models; fine-tuned GPT-2 on summarization and sentiment continuation. Introduced KL penalty to prevent policy collapse. |
| **2020** | Stiennon et al. — "Learning to Summarize with Human Feedback" | Scaled RLHF to a 1.3B-parameter summarization model. The RLHF summarizer surpassed the contemporary SOTA. Proved RLHF can yield human-preferred outputs beyond pure SFT. |
| **2022.01** | Anthropic — "Training a Helpful and Harmless Assistant from Human Feedback" | Systematic study of RLHF along helpfulness and harmlessness. Introduced the "HH" dataset. |
| **2022.03** | Ouyang et al. (OpenAI) — **InstructGPT** | Milestone paper. The 1.3B InstructGPT was judged by humans as better than the 175B GPT-3. Formally established the SFT → RM → PPO three-stage pipeline. Deployed as the default OpenAI API model. |
| **2022.11** | OpenAI — **ChatGPT** released | Trained with RLHF, ignited global attention on LLMs. RLHF moved from academic technique to industry standard. |
| **2023** | Anthropic — **Constitutional AI (CAI)** | Replaced part of human feedback with AI feedback (RLAIF), cutting annotation cost and improving scalability. |
| **2023** | Rafailov et al. — **DPO** | Showed you can optimize the policy directly from preference data without training an explicit reward model — simplified the RLHF pipeline. |
| **2024** | DeepSeek — **GRPO** | Removed the critic model; uses intra-group comparison to estimate advantage, dramatically reducing memory and compute. |
| **2025** | DeepSeek — **DeepSeek-R1** | GRPO at large scale for reasoning model training, exhibiting emergent chain-of-thought reasoning. Published in Nature. |
| **2025** | ACM Computing Surveys — RLHF survey | Comprehensive review of RLHF technical evolution. |
| **2025** | Lambert — **RLHF Book** | First RLHF textbook, systematic treatment of theory and practice. |

---

## The Three-Stage Pipeline

The standard RLHF pipeline consists of three stages. The ASCII diagram below shows the full data flow:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RLHF three-stage pipeline                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Stage 1: SFT                                                       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │ Pretrained    │───>│  SFT data    │───>│   SFT model  │          │
│  │ (Base LLM)   │    │ (prompt,resp)│    │  π_SFT       │          │
│  └──────────────┘    └──────────────┘    └──────┬───────┘          │
│                                                  │                  │
│  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─│─ ─ ─ ─ ─ ─ ─   │
│                                                  │                  │
│  Stage 2: RM training                            │                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────┴───────┐          │
│  │  SFT model   │───>│ Human pref   │───>│  Reward model │          │
│  │  (init RM)   │    │ (x, y_w, y_l)│    │   r_φ(x,y)  │          │
│  └──────────────┘    └──────────────┘    └──────┬───────┘          │
│                                                  │                  │
│  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─│─ ─ ─ ─ ─ ─ ─   │
│                                                  │                  │
│  Stage 3: RL optimization (PPO)                  │                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────┴───────┐          │
│  │  SFT model   │    │ Current      │<──>│ Reward model  │          │
│  │  (ref π_ref) │    │ policy gen   │    │   scoring     │          │
│  └──────┬───────┘    │ and update   │    └──────────────┘          │
│         │            └──────────────┘                              │
│         │                    │                                      │
│         └─── KL penalty ────>│                                      │
│                              ▼                                      │
│                     ┌──────────────┐                                │
│                     │ Aligned model │                                │
│                     │  π_θ (RLHF)  │                                │
│                     └──────────────┘                                │
└─────────────────────────────────────────────────────────────────────┘
```

### Stage 1 — Supervised Fine-Tuning (SFT)

**Goal**: turn the pretrained model from "text continuation" to "instruction following".

**Data requirements**:
- Format: (prompt, desired_response) pairs
- Typical size: thousands to tens of thousands of high-quality annotations
- InstructGPT used ~13,000 human-annotated examples
- Quality matters far more than quantity — LIMA (Zhou et al., 2023) achieved great results with only 1,000 curated examples

**Training details**:
- Standard LM cross-entropy loss, loss computed only on response tokens (prompt tokens masked)
- Learning rate typically low (1e-5 ~ 5e-6) to prevent forgetting pretraining knowledge
- Train 1-3 epochs, avoid overfitting
- Optionally mix data: part conversation data + part pretraining data

**Common pitfalls**:
- **Overfitting**: SFT data is small; easy to overfit → model outputs become overly fixed
- **Catastrophic forgetting**: SFT may damage world knowledge learned during pretraining
- **Inconsistent annotation quality**: large stylistic variation across annotators leads to inconsistent model behavior
- **Format preference vs content quality**: SFT easily learns surface formatting instead of deep capability

```python
# SFT training pseudocode
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

model = AutoModelForCausalLM.from_pretrained("base_model")
training_args = TrainingArguments(
    learning_rate=2e-5,
    num_train_epochs=2,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
)

# Note: loss is computed only on response tokens
# prompt tokens have labels set to -100 (ignored)
trainer = Trainer(model=model, args=training_args, train_dataset=sft_dataset)
trainer.train()
```

### Stage 2 — Reward Model (RM) Training

**Goal**: distill human comparison preferences into a differentiable scalar reward function.

#### Data Collection

1. For the same prompt, sample K different responses from the SFT model (typically K=4~9)
2. Human annotators rank these responses (or perform pairwise comparison)
3. Extract pairwise preferences from rankings: (prompt, y_w, y_l) where y_w is preferred over y_l

InstructGPT collected ~33,000 comparison sets (each a ranking of 4~9 responses).

#### Bradley-Terry Preference Model

RM training is based on the **Bradley-Terry model**, a classical pairwise comparison probability model. Its core idea comes from the Elo rating system (chess):

Given prompt x and two responses y_w (preferred) and y_l (not preferred), the reward model r_φ maps a response to a scalar score. Bradley-Terry assumes the probability that humans prefer y_w over y_l is:

```
P(y_w > y_l | x) = σ(r_φ(x, y_w) - r_φ(x, y_l))
```

where σ is the sigmoid: σ(z) = 1/(1+e^(-z)).

**Intuition**: this is just like Elo ratings in chess — the win rate of two players (responses) depends on their rating difference. Larger gap → stronger player's win probability approaches 1.

#### RM Loss Derivation

Given a preference dataset D = {(x_i, y_w^i, y_l^i)}, maximize the likelihood:

```
max_φ  Π P(y_w^i > y_l^i | x_i)
```

Taking the log and negating gives the loss:

```
L_RM(φ) = -E_{(x, y_w, y_l) ~ D} [log σ(r_φ(x, y_w) - r_φ(x, y_l))]
```

This is the standard cross-entropy loss for RM training.

**Important properties**:
- RM learns only **relative** scores (differences matter, absolute values do not)
- Hence the RM output can have arbitrary offset (often handled via normalization)
- From ranked data you can extract C(K,2) comparison pairs, improving data utilization

#### RM Architecture

```
┌─────────────────────────────────┐
│  Input: [prompt] + [response]    │
│           ▼                     │
│  Transformer (init from SFT)    │
│           ▼                     │
│  Hidden state of last token      │
│           ▼                     │
│  Linear (hidden_dim → 1)        │
│           ▼                     │
│  Scalar reward r(x, y)          │
└─────────────────────────────────┘
```

Typically initialized from the SFT model (remove LM head, add scalar output head), since the SFT model already "understands" instructions and responses.

#### RM Training Tricks

- **Data cleaning**: drop samples with high annotator disagreement (low agreement) — they are noise
- **Use multiple pairs from a single ranking**: extract all C(K,2) pairs, weighted by rank distance
- **RM calibration**: regularly inspect the RM score distribution to ensure discriminative power
- **Anti-overfitting**: RMs are usually as small as or smaller than the policy; use dropout and early stopping
- **Evaluation metric**: preference prediction accuracy (typically 65-75% is enough to support RLHF; InstructGPT reports ~72%)

### Stage 3 — RL Optimization (PPO)

**Goal**: maximize the score from the reward model while not drifting too far from the SFT model.

#### RL Objective

```
max_θ  E_{x~D, y~π_θ(·|x)} [r_φ(x, y)] - β · KL(π_θ || π_ref)
```

where:
- π_θ — the policy being trained (LLM)
- π_ref — reference policy (frozen SFT model)
- r_φ(x, y) — reward model score
- β — KL penalty coefficient (key hyperparameter)
- KL(π_θ || π_ref) — KL divergence between the current policy and reference

In practice, the KL penalty is folded into the reward:

```
r_total(x, y) = r_φ(x, y) - β · Σ_t log[π_θ(y_t|x,y_{<t}) / π_ref(y_t|x,y_{<t})]
```

This is a **per-token KL penalty** that applies a constraint at every position of the sequence.

#### The Role of the KL Penalty

The KL penalty is a crucial regularizer in RLHF:

| KL coef β | Effect |
|-----------|------|
| β too small | Over-optimization of RM → reward hacking (exploitation of RM weaknesses) |
| β too large | Policy too conservative → barely different from SFT, RL has no effect |
| β moderate | Balances exploration and stability, yields best human preference |

InstructGPT used adaptive β tuning: target KL = 6 nats, β adjusted dynamically during training.

#### GAE Advantage Estimation

PPO updates need to estimate the advantage A_t at each token position. We use **Generalized Advantage Estimation (GAE)**:

```
δ_t = r_t + γ · V(s_{t+1}) - V(s_t)     (TD residual)

A_t^GAE(γ,λ) = Σ_{l=0}^{T-t} (γλ)^l · δ_{t+l}
```

where:
- V(s_t) — value estimate from the critic at position t
- γ — discount factor (often 1.0 for LLM RLHF)
- λ — GAE parameter (controls bias-variance trade-off; typically 0.95)

For details on PPO and GAE, see [[ppo-for-llm]].

#### The PPO Training Loop

Each training iteration consists of:

```
For each training batch:
  1. Sample a batch of prompts from the prompt set
  2. Generate responses with the current policy y ~ π_θ(·|x)
  3. Score with the reward model: r = r_φ(x, y)
  4. Compute KL penalty: kl_t = log[π_θ(y_t|...) / π_ref(y_t|...)]
  5. Compute adjusted reward: r_total = r - β · Σ kl_t
  6. Estimate per-token value V(s_t) with the critic
  7. Compute advantage A_t via GAE
  8. Run K PPO updates (typically K=1~4):
     - Compute probability ratio r_t(θ) = π_θ(y_t|...) / π_old(y_t|...)
     - Clipped surrogate: L = min(r_t·A_t, clip(r_t, 1-ε, 1+ε)·A_t)
     - Update Actor and Critic
```

---

## Mathematical Derivations

### Bradley-Terry Preference Model

**Assumption**: each response y has a latent "quality score" r(x, y). When humans compare, the probability of choosing y_w follows a logistic model:

```
P(y_w ≻ y_l | x) = exp(r(x, y_w)) / [exp(r(x, y_w)) + exp(r(x, y_l))]
                  = 1 / [1 + exp(-(r(x, y_w) - r(x, y_l)))]
                  = σ(r(x, y_w) - r(x, y_l))
```

Mathematically this is fully equivalent to the Elo rating system (Elo uses log base 10, Bradley-Terry uses natural log).

### Reward Model Loss

```
L_RM(φ) = -E_{(x, y_w, y_l)} [log σ(r_φ(x, y_w) - r_φ(x, y_l))]
```

**Gradient**:

```
∂L/∂φ = -E [(1 - σ(r_φ(y_w) - r_φ(y_l))) · (∂r_φ(y_w)/∂φ - ∂r_φ(y_l)/∂φ)]
```

Intuition: when the model is uncertain about the preference (σ near 0.5), the gradient is larger, pushing the model to widen the gap between the good and bad responses.

### RL Optimization Objective

```
max_θ  J(θ) = E_{x~D, y~π_θ} [r_φ(x,y)] - β · E_{x~D} [KL(π_θ(·|x) || π_ref(·|x))]
```

Expanding the KL divergence:

```
KL(π_θ || π_ref) = E_{y~π_θ} [log π_θ(y|x) - log π_ref(y|x)]
                 = Σ_t E [log π_θ(y_t|x,y_{<t}) - log π_ref(y_t|x,y_{<t})]
```

Hence the total reward can be written per-token:

```
r_total = r_φ(x,y) - β · Σ_t [log π_θ(y_t|x,y_{<t}) - log π_ref(y_t|x,y_{<t})]
```

Only the last token in the sequence gets the RM reward; intermediate tokens get only the KL penalty term.

### GAE Advantage Estimation

```
δ_t = r_t + γ · V(s_{t+1}) - V(s_t)

A_t^GAE = Σ_{l=0}^{∞} (γλ)^l · δ_{t+l}
        = δ_t + γλ · δ_{t+1} + (γλ)^2 · δ_{t+2} + ...
```

When λ=0: A_t = δ_t = r_t + γV(s_{t+1}) - V(s_t) (high bias, low variance)
When λ=1: A_t = Σ γ^l r_{t+l} - V(s_t) (low bias, high variance — Monte Carlo)

In practice λ=0.95 is a common choice.

---

## RLHF Variants and Evolution

### Online RLHF vs Offline RLHF

| Aspect | Online RLHF | Offline RLHF |
|------|-------------|--------------|
| Data source | Generated live by current policy | Pre-collected fixed dataset |
| Representative algorithms | PPO, GRPO | DPO, IPO, KTO |
| Reward model | Used online, may update with policy | No explicit RM needed |
| Pros | Strong exploration, avoids OOD issues | Simple, stable training |
| Cons | High compute, complex implementation | Limited by offline data distribution |

**Trend**: online methods perform better on complex tasks (reasoning, code, agents); offline methods are more cost-effective for simple alignment.

### RLAIF — AI Feedback Replacing Human Feedback

**Constitutional AI (Anthropic, 2023)**:
- Replace human annotators with an AI model for preference judgments
- The AI judges response quality against preset "constitutional principles" (e.g., "answers should be harmless")
- Dramatically reduces annotation cost and improves scalability
- Key finding: AI feedback matches human feedback in many settings

**RLAIF workflow**:
```
Original response → AI revises by principles → Revised vs original → AI judges preference → train RM → PPO
```

### RLVR — Reinforcement Learning with Verifiable Rewards

For math and code, you can entirely bypass human feedback and learned reward models:

```
reward = { 1.0  if answer correct (verified)
         { 0.0  if answer incorrect
```

**Representative work**:
- **DeepSeek-R1**: uses answer correctness as reward, trains strong reasoning capability (see [[grpo]])
- **Math tasks**: compare against ground-truth answer
- **Code tasks**: verify via unit tests
- See [[reward-modeling#RLVR|RLVR details]]

RLVR advantage: reward signal is exact (no noise), avoids reward-model bias.
RLVR limitation: only applicable to tasks with clearly correct answers.

### Iterative RLHF

Standard RLHF is a one-shot pipeline: collect data → train RM → RL → done.
Iterative RLHF loops this process:

```
SFT model → RLHF round 1 → new policy
    ↓                       ↓
Generate with new policy → collect new human feedback → update RM → RLHF round 2 → ...
```

Advantages:
- The RM is trained on the current policy distribution, avoiding distribution shift
- Policy continues to improve
- Closer to true online learning

### Best-of-N Sampling (Rejection Sampling)

The simplest "RLHF" method — no RL needed:

```
1. For each prompt, generate N responses with the policy
2. Score each with the reward model
3. Pick the highest-scoring response
```

**Properties**:
- Inference compute increases by N×, but no RL training needed
- Best-of-N has effective KL penalty ≈ log(N) - (N-1)/N
- Often used as an RLHF baseline
- DeepSeek-R1 also used Best-of-N for data filtering

---

## Code Examples

### RLHF Training Loop with the TRL Library

```python
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer

# === Stage setup: load models ===
model = AutoModelForCausalLMWithValueHead.from_pretrained("sft_model_path")
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained("sft_model_path")
tokenizer = AutoTokenizer.from_pretrained("sft_model_path")
reward_model = load_reward_model("rm_model_path")  # custom loader

# === PPO config ===
ppo_config = PPOConfig(
    model_name="sft_model",
    learning_rate=1.41e-5,
    batch_size=64,
    mini_batch_size=16,
    ppo_epochs=4,              # PPO update epochs per batch
    kl_penalty="kl",           # KL penalty type
    init_kl_coef=0.2,          # initial KL coef β
    target_kl=6.0,             # target KL (for adaptive β)
    cliprange=0.2,             # PPO clip param ε
    cliprange_value=0.2,       # value clipping
    gamma=1.0,                 # discount factor
    lam=0.95,                  # GAE λ
)

ppo_trainer = PPOTrainer(ppo_config, model, ref_model, tokenizer)

# === Training loop ===
for epoch in range(num_epochs):
    for batch in dataloader:
        # 1. Generate responses
        query_tensors = [tokenizer.encode(q, return_tensors="pt") for q in batch["query"]]
        response_tensors = ppo_trainer.generate(query_tensors, max_new_tokens=256)
        
        # 2. Compute rewards
        texts = [tokenizer.decode(r.squeeze()) for r in response_tensors]
        rewards = [reward_model.score(q, r) for q, r in zip(batch["query"], texts)]
        rewards = [torch.tensor(r) for r in rewards]
        
        # 3. PPO update (KL, GAE, clip done internally)
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        
        # 4. Log
        print(f"mean_reward: {stats['ppo/mean_scores']:.3f}, "
              f"kl: {stats['ppo/mean_non_score_reward']:.3f}")
```

### Reward Model Training Code

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class RewardModel(nn.Module):
    """Reward model based on the Bradley-Terry preference model"""
    
    def __init__(self, base_model_name):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(base_model_name)
        self.reward_head = nn.Linear(self.backbone.config.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Take the hidden state of the last token
        last_hidden = outputs.last_hidden_state
        # Find the last non-padding token of each sequence
        seq_lengths = attention_mask.sum(dim=1) - 1
        last_token_hidden = last_hidden[range(len(seq_lengths)), seq_lengths]
        reward = self.reward_head(last_token_hidden).squeeze(-1)
        return reward

def compute_rm_loss(reward_model, chosen_ids, chosen_mask, rejected_ids, rejected_mask):
    """Bradley-Terry loss"""
    r_chosen = reward_model(chosen_ids, chosen_mask)      # (batch,)
    r_rejected = reward_model(rejected_ids, rejected_mask) # (batch,)
    
    # L = -E[log σ(r_chosen - r_rejected)]
    loss = -torch.log(torch.sigmoid(r_chosen - r_rejected)).mean()
    
    # Accuracy: fraction where r_chosen > r_rejected
    accuracy = (r_chosen > r_rejected).float().mean()
    
    return loss, accuracy

# Training loop
optimizer = torch.optim.AdamW(reward_model.parameters(), lr=1e-5, weight_decay=0.01)

for epoch in range(num_epochs):
    for batch in preference_dataloader:
        loss, acc = compute_rm_loss(
            reward_model,
            batch["chosen_ids"], batch["chosen_mask"],
            batch["rejected_ids"], batch["rejected_mask"]
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reward_model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        print(f"RM Loss: {loss.item():.4f}, Accuracy: {acc.item():.3f}")
```

---

## Challenges and Open Questions

### 1. The Human-Annotation Bottleneck

- **High cost**: high-quality annotation requires domain experts; math/code annotation is especially expensive
- **Annotator disagreement**: different annotators frequently disagree on the same pair (agreement often only 70-80%)
- **Systematic biases**: annotators prefer longer, more formal, more "assistant-styled" responses over content quality
- **Unscalable**: human annotation rate is limited and becomes a training bottleneck

### 2. Reward Model Limitations

- **[[reward-modeling|Reward model]] quality caps final policy performance** — RM errors get amplified by RL
- **Poor OOD generalization**: RM is accurate in-distribution but may give unreliable scores for novel response styles
- **RM size dilemma**: too small → insufficient capacity; too large → expensive compute
- **Multi-dimensional reward**: a single scalar struggles to capture helpfulness, harmlessness, honesty, and other axes

### 3. [[reward-modeling#Reward Hacking|Reward Hacking]]

The policy learns to exploit RM weaknesses for high scores rather than truly improving response quality:
- Excessive verbosity (RM may prefer long responses)
- Use of specific phrases or formats the RM prefers
- "Surface-level pleasing" rather than truly helpful
- This tendency amplifies over RL training

**Mitigations**: KL penalty, RM ensembles (multi-RM voting), periodic RM updates

### 4. Training Stability

- Tuning KL coef β is extremely tricky — too small → reward hacking; too large → policy barely updates
- PPO hyperparameters (lr, clip range, batch size, etc.) require careful tuning
- Rewards can suddenly collapse or saturate during training
- "Alignment tax": RLHF can degrade certain model capabilities

### 5. Alignment Tax and Capability-Safety Trade-off

- RLHF can make models "over-aligned" on tasks like creative writing or role-play — too conservative
- A fundamental tension exists between safety and usefulness
- Over-training can cause "sycophancy" — the model tends to agree with the user rather than give a truthful answer

### 6. Scalable Oversight

- When model capability exceeds human evaluators, how can RLHF remain effective?
- Superhuman AI's answers may exceed human judgment ability
- This is one of the core long-term AI-safety problems
- Possible directions: AI-assisted evaluation, debate, recursive reward modeling

---

## References

- Christiano et al. (2017) — [Deep RL from Human Preferences](https://arxiv.org/abs/1706.03741)
- Ziegler et al. (2019) — [Fine-Tuning Language Models from Human Preferences](https://arxiv.org/abs/1909.08593)
- Stiennon et al. (2020) — [Learning to Summarize with Human Feedback](https://arxiv.org/abs/2009.01325)
- Bai et al. (2022) — [Training a Helpful and Harmless Assistant from Human Feedback](https://arxiv.org/abs/2204.05862)
- Ouyang et al. (2022) — [InstructGPT](https://arxiv.org/abs/2203.02155)
- Bai et al. (2022) — [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)
- Rafailov et al. (2023) — [DPO: Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- Zhou et al. (2023) — [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206)
- Lambert (2025) — [RLHF Book](https://rlhfbook.com/book.pdf) ([arXiv:2504.12501](https://arxiv.org/abs/2504.12501))
- ACM Computing Surveys (2025) — [RLHF Deciphered](https://dl.acm.org/doi/10.1145/3743127)

---

## Related Pages

- [[ppo-for-llm]] — core RL algorithm used in RLHF; detailed derivation of the PPO objective
- [[grpo]] — critic-free PPO alternative proposed by DeepSeek
- [[dpo]] — Direct Preference Optimization, fully bypasses reward modeling
- [[reward-modeling]] — reward model mechanics and training details
- [[rl-training-frameworks]] — RLHF training frameworks (OpenRLHF, veRL, TRL, etc.)
- [[multi-step-reasoning-rl]] — RLHF/GRPO applied to reasoning models
