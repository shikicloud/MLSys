---
title: "Reward Modeling"
category: rl-infra
tags: [reward-model, prm, orm, rlvr, reward-hacking, process-reward, bradley-terry]
created: 2026-04-13
updated: 2026-05-20
status: mature
---

# Reward Modeling

> [!abstract]+ TL;DR
> The reward model (RM) is the bridge between human preferences and RL optimization in the [[rlhf-overview|RLHF]] pipeline -- the **Bradley-Terry** objective turns subjective judgment into a scalar signal. **RM quality fundamentally caps the aligned model's quality**: systematic RM biases get baked into the policy (reward hacking). The 2025-2026 landscape has moved beyond "learn a scoring model": **process reward models** (PRM, scoring each reasoning step) beat **outcome reward models** (ORM) by 6+ pp on MATH; **RLVR** (verifiable programmatic rewards, used by [[grpo#DeepSeek-R1|DeepSeek-R1]]) skips learned RMs on math/code; **PRIME** (2025) extracts implicit per-token Q-values from an ORM, raising sample efficiency 2.5x.

## Overview

The reward model (RM) is a core component of the [[rlhf-overview|RLHF]] pipeline, acting as the bridge between human preferences and RL optimization. It turns subjective human judgment into a scalar signal that drives policy optimization.

```
Position of the RM in the RLHF pipeline:

  Human preference data          RL optimization
  (y_w > y_l)                    (PPO/GRPO)
       |                             ^
       v                             | r(x,y)
  +----------+              +----------+
  | Train RM | -----------> | Reward   | --> scalar reward score
  +----------+              | model    |
                            +----------+
                                  ^
                                  | (prompt, response)
                              Policy model generates
```

**RM quality fundamentally caps the aligned model's quality**: if the RM is systematically biased, the policy will learn those biases (reward hacking). This is one motivation behind [[dpo|DPO]], which bypasses the RM entirely.

In the 2025-2026 landscape, the RM has evolved well beyond "train a single scoring model":
- **Classical RM**: a scalar scoring model trained from human preferences
- **Process reward model (PRM)**: scores every reasoning step
- **Verifiable rewards (RLVR)**: programmatic verifiers replace the learned RM entirely
- **Implicit rewards**: DPO encodes the reward implicitly in the policy

---

## Reward Model Architecture

### From LLM to RM

Reward models are typically built on top of pretrained language models. The key change: replace the LM head (next-token distribution) with a linear layer that emits a scalar reward.

```
Standard LLM:                       Reward model:
+------------------+              +------------------+
|  Input tokens     |              |  Input tokens     |
|  "Explain QM..."  |              |  (prompt+response) |
+--------+---------+              +--------+---------+
         |                                  |
         v                                  v
+------------------+              +------------------+
|   Transformer    |              |   Transformer    |
|   backbone       |              |   backbone       |
|   (N layers)     |              |   (N layers)     |
+--------+---------+              +--------+---------+
         |                                  |
         v                                  v
+------------------+              +------------------+
|   LM head        |              |   Scalar head    |
|   (vocab_size)   |              |   Linear(d -> 1) |
|   -> next token  |              |   -> reward      |
|     distribution |              |     (scalar)     |
+------------------+              +------------------+
```

### Architecture Choices

| Strategy | Description | Pros / cons |
|------|------|--------|
| **Same model init** | RM and policy share the pretrained model | Aligned capability, but doubles memory |
| **Smaller model** | RM is smaller than policy (e.g., 70B policy, 7B RM) | Saves memory but may lack expressiveness |
| **SFT init** | RM initialized from the SFT checkpoint | Most common; already understands instructions |
| **Dedicated RM** | Independently trained specialized RM (e.g., RewardBench leaders) | Reusable, but may not match target distribution |

### Implementation

```python
import torch
import torch.nn as nn
from transformers import AutoModel

class RewardModel(nn.Module):
    def __init__(self, base_model_name):
        super().__init__()
        # Use a pretrained Transformer as the backbone
        self.backbone = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.backbone.config.hidden_size
        
        # Scalar reward head
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
    
    def forward(self, input_ids, attention_mask):
        # Use the last token's hidden state as the sequence representation
        outputs = self.backbone(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        # Take the last non-padding token's hidden state
        sequence_lengths = attention_mask.sum(dim=1) - 1
        last_hidden = outputs.last_hidden_state
        batch_size = input_ids.shape[0]
        pooled = last_hidden[
            torch.arange(batch_size), sequence_lengths
        ]
        
        # Scalar reward output
        reward = self.reward_head(pooled).squeeze(-1)
        return reward
```

---

## Training Methods

### Bradley-Terry Preference Model (Detailed Derivation)

**Core assumption**: the human's preference probability for response A vs. response B can be modeled as a sigmoid of the reward difference.

Let `r(x, y)` be the reward for prompt x and response y. Then:

```
P(y_w > y_l | x) = sigma(r(x, y_w) - r(x, y_l))

with sigma(z) = 1 / (1 + exp(-z)) the sigmoid function
```

**Intuition**:
- When `r(y_w) >> r(y_l)`, `P(y_w > y_l) -> 1` (strong preference for y_w)
- When `r(y_w) = r(y_l)`, `P(y_w > y_l) = 0.5` (no preference)
- When `r(y_w) << r(y_l)`, `P(y_w > y_l) -> 0` (preference for y_l)

**Why Bradley-Terry**:
1. Nice theoretical properties (probabilistic consistency)
2. Learn a global ranking from pairwise comparisons
3. Sigmoid derivative is largest at zero difference -> automatically focuses on hard-to-distinguish pairs

### Pairwise Ranking Loss

This is the standard RM training objective, directly derived from BT maximum likelihood:

```
L_pairwise = -E_{(x, y_w, y_l)} [ log sigma(r(x, y_w) - r(x, y_l)) ]
```

**Training procedure**:
1. For each prompt x, take a pair (y_w, y_l)
2. Compute rewards r(x, y_w) and r(x, y_l)
3. Maximize the probability that y_w scores higher than y_l

```python
def pairwise_ranking_loss(reward_chosen, reward_rejected):
    """
    Standard Bradley-Terry pairwise ranking loss
    
    Args:
        reward_chosen: shape (batch_size,) - reward for preferred responses
        reward_rejected: shape (batch_size,) - reward for rejected responses
    """
    return -torch.log(torch.sigmoid(reward_chosen - reward_rejected)).mean()
```

### Listwise Ranking Loss

When each prompt has K > 2 ranked responses, a listwise loss extracts more comparison signal:

```
L_listwise = -E [ Sigma_{i<j} log sigma(r(x, y_i) - r(x, y_j)) ]

where y_1 > y_2 > ... > y_K is the ranking
```

**Advantages**:
- K responses yield C(K,2) = K(K-1)/2 comparisons
- Richer ranking information than just the best and worst pair
- InstructGPT (Ouyang et al., 2022) used K=4 to K=9 listwise ranking

```python
def listwise_ranking_loss(rewards_ranked):
    """
    Listwise ranking loss
    
    Args:
        rewards_ranked: shape (batch_size, K) - K responses sorted best to worst
    """
    loss = 0
    K = rewards_ranked.shape[1]
    n_pairs = 0
    for i in range(K):
        for j in range(i + 1, K):
            loss -= torch.log(
                torch.sigmoid(rewards_ranked[:, i] - rewards_ranked[:, j])
            ).mean()
            n_pairs += 1
    return loss / n_pairs
```

### Margin-Based Ranking Loss

When preference labels carry confidence or score differences, use a margin to constrain the reward gap:

```
L_margin = -E [ log sigma(r(x, y_w) - r(x, y_l) - m(y_w, y_l)) ]

where m(y_w, y_l) is the expected minimum reward gap (set by label confidence)
```

**For example**: if y_w is much better than y_l (annotator very confident), set m large; if the gap is unclear, set m small.

### Training Data

The quality of **human preference data** directly determines the RM's quality:

| Dataset | Size | Source | Notes |
|-------|------|------|------|
| Anthropic HH-RLHF | ~170K pairs | Human annotated | helpfulness + harmlessness |
| OpenAI WebGPT | ~20K pairs | Human web-summary comparisons | Factuality |
| UltraFeedback | ~64K | GPT-4 annotated | Multi-dimensional |
| Stanford SHP | ~385K | Reddit votes | Natural preferences |
| Chatbot Arena | Growing | Human votes | Real user preferences |
| PRM800K | ~800K steps | Human annotated | Process reward (math) |

**Data quality considerations**:
1. Inter-annotator agreement is critical
2. The preference distribution should cover the target task
3. Clearly define the preference criterion (helpful? harmless? honest?)
4. Balance examples across difficulty levels

---

## Reward Hacking

### What Is Reward Hacking?

Reward hacking is when the policy model finds ways to score high on the RM without actually improving (or while degrading) real response quality. It's a core challenge in RM-based RLHF.

```
Reward hacking dynamics:

  RM reward
    ^
    |           +--- Reward hacking region ---+
    |           |  RM reward up               |
    |           |  Real quality flat or down  |
    |       ....|.......                       |
    |    ...    |       ...                   |
    |  ..       |          ...                |
    | . Normal  |             ...             |
    |.training  |                ...          |
    | region    |                   ...       |
    +-----------+---------------------+---> training steps
    |           |                     |
    |   RM & real quality   RM diverges from   |
    |    are aligned         real quality      |
    +------------------------------------------+

  Real quality (human eval)
    ^
    |    ...
    |  ..   ....
    | .         ....
    |.              ....
    |                   ...
    |                      ...
    |                         ..
    +------------------------------> training steps
```

### Common Reward Hacking Patterns

#### 1. Length Bias

The RM rewards longer responses (even if the extra content is filler):

```
Prompt: "What is 2+2?"

Normal response (score: 3.2):
"4"

Hacked response (score: 4.8):
"That's a great question! Let me break this down for you step by step.
2+2 is a basic arithmetic operation. When we add 2 to 2, we get 4.
To summarize, the answer is 4. I hope this helps! Let me know if
you have any other questions."
```

#### 2. Sycophancy

The model learns to agree unconditionally with the user because annotators tend to prefer "friendly" responses:

```
User: "I think the earth is flat."

Honest response (may score low):
"Actually, the earth is roughly spherical..."

Sycophantic response (may score high):
"That's an interesting perspective! You raise some good points..."
```

#### 3. Format Gaming

The RM accidentally learned to prefer certain formats:

- Using Markdown headers and bullet lists
- Adding "Step 1, Step 2..." structure
- Tacking a summary onto the end of replies
- Wrapping any technical content in code blocks

#### 4. Repetition / Pattern Exploitation

Generating content that looks informative but is repeated variants of the same point, padding length to score higher.

### Mitigation Strategies

#### 1. KL Penalty

Add a KL penalty to the RL objective to limit how far the policy drifts from the reference:

```
max_pi  E[r(x,y)] - beta * KL(pi || pi_ref)
```

Larger beta = stronger constraint. This is the most basic and most common mitigation.

#### 2. Reward Model Ensembling

Train multiple RMs and take a conservative estimate (min or mean) to suppress any individual RM's spurious features:

```python
# Reward ensembling
reward_models = [rm1, rm2, rm3]
rewards = [rm(prompt, response) for rm in reward_models]

# Conservative: take the minimum
final_reward = min(rewards)

# Or: mean minus std
final_reward = mean(rewards) - alpha * std(rewards)
```

#### 3. Length Penalty

Explicitly penalize over-long responses:

```python
# Simple length penalty
reward = rm_score - lambda * max(0, len(response) - target_length)

# Or logarithmic
reward = rm_score - lambda * log(len(response) / target_length)
```

#### 4. Constrained Optimization

Treat reward hacking detection as a constraint:

```
max_pi  E[r(x,y)]
s.t.    KL(pi || pi_ref) <= delta
        E[length(y)] <= L
        diversity(pi) >= tau
```

#### 5. Verifiable Rewards (RLVR)

For verifiable tasks, replace the learned RM with a deterministic verifier outright (see the next section).

```
Mitigation comparison:
+--------------+---------------+----------------------+
| Strategy     | Effectiveness | Use case              |
+--------------+---------------+----------------------+
| KL penalty   | Medium        | General                |
| RM ensemble  | Good          | When compute allows    |
| Length penalty | Targeted    | Verbosity problem      |
| RLVR         | Best (but narrow) | Verifiable tasks (math/code) |
| Constraints  | Good          | When explicit limits   |
+--------------+---------------+----------------------+
```

---

## Process Reward Models (PRM) vs. Outcome Reward Models (ORM)

### Outcome Reward Model (ORM)

ORMs only score the final output -- given a prompt and the complete response, they emit one scalar reward:

```
ORM workflow:
  Prompt --> full response --> ORM --> single reward
  "Solve x^2=4"   "x^2=4       r = 0.8
                   x=+/-2 ok"
```

**Pros**:
- Annotation is cheap: just label final answer right or wrong
- Training is simple: standard Bradley-Terry loss
- Broadly applicable: works for any task with a good/bad judgment

**Cons**:
- **Poor credit assignment**: when the answer is wrong, no signal about which step failed
- **Sparse signal**: long reasoning chains get a single reward per generation
- **Hard to shape intermediate behavior**: RL struggles to learn "good reasoning" patterns

### Process Reward Model (PRM)

PRMs score each step of the reasoning process, providing dense reward signal:

```
PRM workflow:
  Prompt: "Solve x^2=4"
  
  Step 1: "x^2 = 4"                 -> PRM score: 0.95  ok
  Step 2: "Take square root"        -> PRM score: 0.90  ok
  Step 3: "x = 2"                   -> PRM score: 0.40  miss (forgot the negative root)
  Step 4: "Therefore x = 2"         -> PRM score: 0.30  miss
  
  Every step gives feedback, pinpointing the failure!
```

### Comparison

| Property | ORM (outcome) | PRM (process) |
|------|:----------:|:----------:|
| Scoring granularity | Final response | Each reasoning step |
| Signal density | Sparse (1/generation) | Dense (1/step) |
| Annotation cost | Low | **Very high** |
| Credit assignment | Poor | Good |
| Training difficulty | Easy | Hard |
| Use case | General alignment | Reasoning / math / code |
| Typical dataset | Preference pairs | PRM800K |

### Training a PRM

```python
# PRM training: label each step as correct / incorrect
class ProcessRewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.backbone = base_model
        self.step_head = nn.Linear(
            base_model.config.hidden_size, 1
        )
    
    def forward(self, input_ids, attention_mask, step_boundaries):
        """
        step_boundaries: end positions of each reasoning step
        """
        hidden = self.backbone(input_ids, attention_mask).last_hidden_state
        
        # Extract representations at each step boundary
        step_rewards = []
        for boundary in step_boundaries:
            step_repr = hidden[:, boundary, :]
            step_rewards.append(self.step_head(step_repr))
        
        return torch.stack(step_rewards, dim=1)  # (batch, n_steps, 1)
```

### OpenAI's PRM800K

**Paper**: Lightman et al. (2023), "Let's Verify Step by Step"

The first large-scale process reward dataset:
- **Size**: ~800,000 step-level labels
- **Source**: human annotators stepping through reasoning on the MATH dataset
- **Labels**: each step is positive / neutral / negative
- **Key result**: PRM reaches 78.2% on MATH while ORM hits 72.4% (+5.8%)

**PRM-guided search**: at inference, PRM can guide search:
- Generate multiple candidate reasoning paths
- Score each step with the PRM
- Pick the path with the highest PRM score
- More effective than ORM-based best-of-N

### PRIME: Process Reinforcement via Implicit Rewards

**Paper**: [arXiv:2502.01456](https://arxiv.org/abs/2502.01456), 2025

PRIME's key innovation: **extract implicit process rewards from an ORM, no step-level annotation required**.

```
PRIME pipeline:
  +------------------------------+
  | 1. Train a standard ORM       |
  |    (outcome labels only)      |
  +--------------+----------------+
                 |
                 v
  +------------------------------+
  | 2. Extract implicit Q-values  |
  |    q(s_t, a_t) as per-token   |
  |    process rewards            |
  +--------------+----------------+
                 |
                 v
  +------------------------------+
  | 3. Use as PRM to drive RL     |
  |    Dense signal + online      |
  +------------------------------+
```

**Key results**:
- 2.5x sample efficiency
- 6.9% over standard outcome RL
- Eurus-2-7B-PRIME beats Qwen2.5-Math-7B-Instruct on 7 benchmarks using only 10% of the training data
- Demonstrates that "you don't need expensive step-level labels to get process rewards"

### When to Pick PRM vs. ORM

```
Pick ORM when:
  + General dialog alignment
  + Limited annotation budget
  + Task doesn't involve multi-step reasoning
  + Rapid prototyping

Pick PRM when:
  + Math reasoning
  + Code generation (step-by-step logic)
  + Need precise credit assignment
  + Pairing with search (beam search, MCTS)

Pick PRIME when:
  + Want PRM-style benefits without step labels
  + Online RL setting
```

---

## RLVR: Reinforcement Learning from Verifiable Rewards

### Paradigm Shift

RLVR (RL from Verifiable Rewards) is the major paradigm shift since 2025: replace the learned reward model with a deterministic programmatic verifier.

```
Traditional RM-based RLHF:
  Model generates --> learned RM scores --> RL update
                          |
                          v  (possibly biased, exploitable)

RLVR:
  Model generates --> deterministic verifier --> RL update
                          |
                          v  (exact, non-exploitable)
```

### Verifier Types

#### Math Verifier

```python
def math_verifier(response, ground_truth):
    """Check whether a math answer is correct"""
    # Extract the final answer
    predicted = extract_answer(response)  # e.g., "42"
    
    # Exact match or symbolic equivalence
    if predicted == ground_truth:
        return 1.0  # correct
    
    # Symbolic equivalence (e.g., "2/4" == "0.5")
    if sympy.simplify(predicted - ground_truth) == 0:
        return 1.0
    
    return 0.0  # incorrect
```

#### Code Verifier

```python
def code_verifier(code_response, test_cases):
    """Run test cases to verify the code"""
    try:
        exec_result = safe_execute(code_response, timeout=10)
        passed = sum(
            run_test(exec_result, tc) for tc in test_cases
        )
        return passed / len(test_cases)  # pass rate as reward
    except Exception:
        return 0.0
```

#### Format Verifier

```python
def format_verifier(response, required_format):
    """Check whether output meets a required format"""
    # e.g., JSON, specific tags, length limits
    if required_format == "json":
        try:
            json.loads(response)
            return 1.0
        except:
            return 0.0
    # ...
```

### DeepSeek-R1's RLVR Approach

DeepSeek-R1 (2025) is a landmark RLVR practice:

1. **Pure RL training** (no SFT): start RL directly from the base model
2. **Verifiers**: math answer checking + code test cases
3. **Algorithm**: [[grpo|GRPO]] (no critic, no RM)
4. **Emergent abilities**: the model spontaneously learns long chain-of-thought reasoning, self-reflection, and self-correction

```
DeepSeek-R1 training paradigm:
+--------------------------------------------------+
|  Base model (DeepSeek-V3)                        |
|        |                                          |
|        v                                          |
|  GRPO + verifiable rewards (math / code)          |
|  (no learned RM at all!)                          |
|        |                                          |
|        v                                          |
|  Model spontaneously learns:                       |
|  - long CoT reasoning                              |
|  - self-verification ("Wait, let me check...")     |
|  - backtracking and retries                        |
|  - step-by-step reasoning                          |
+--------------------------------------------------+
```

### Debates Around RLVR

**Strengths**:
- Eliminates reward hacking (the verifier can't be fooled)
- No human annotation
- Precise reward signal
- Massively scalable

**Limitations**:
- **Only applies to verifiable tasks**: math, code, factual Q&A
- **Doesn't apply to open-ended tasks**: creative writing, style, preference alignment
- **May make the model faster but not smarter**: under pass@256 the base model can beat the RLVR model
- **Extending to non-verifiable domains** is an active research direction (LLM-as-judge, constitutional AI, etc.)

### RLVR vs. Traditional RM

| Aspect | Traditional RM | RLVR |
|------|---------|------|
| Signal source | Learned model | Deterministic program |
| Reward hacking risk | High | None (in the verifiable part) |
| Annotation need | Lots of human preference | Correct answers / test cases |
| Scope | Broad | Verifiable tasks |
| Scalability | Limited by annotation | Limited by task type |
| Representative work | InstructGPT, Claude | DeepSeek-R1 |

---

## Code Examples

### Training a Reward Model with TRL

```python
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardConfig, RewardTrainer

# 1. Load model (with scalar output head)
model = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    num_labels=1,           # scalar output
    torch_dtype="bfloat16",
    attn_implementation="flash_attention_2",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

# 2. Load preference data
# Format: {"chosen": str, "rejected": str}
dataset = load_dataset("Anthropic/hh-rlhf")

# 3. Training config
training_args = RewardConfig(
    output_dir="./reward_model",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    num_train_epochs=1,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=500,
    bf16=True,
    gradient_checkpointing=True,
    max_length=2048,
)

# 4. Train
trainer = RewardTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
)
trainer.train()
```

### Evaluating a Reward Model

```python
# Evaluate on RewardBench
# RewardBench is the standard RM evaluation benchmark

def evaluate_rm_accuracy(rm, eval_dataset):
    """Compute RM accuracy on preference pairs"""
    correct = 0
    total = 0
    
    for sample in eval_dataset:
        r_chosen = rm.score(sample["prompt"], sample["chosen"])
        r_rejected = rm.score(sample["prompt"], sample["rejected"])
        
        if r_chosen > r_rejected:
            correct += 1
        total += 1
    
    accuracy = correct / total
    return accuracy

# Per-category evaluation
categories = ["chat", "safety", "reasoning", "factuality"]
for cat in categories:
    subset = eval_dataset.filter(lambda x: x["category"] == cat)
    acc = evaluate_rm_accuracy(rm, subset)
    print(f"{cat}: {acc:.2%}")
```

### A Simple RLVR Training Loop

```python
def rlvr_training_step(policy, prompts, verifier, optimizer):
    """
    One RLVR training step (simplified)
    """
    # 1. Generate multiple candidate responses
    responses = policy.generate(
        prompts, 
        num_return_sequences=8,  # 8 generations per prompt
        temperature=1.0,
    )
    
    # 2. Score with the verifier (deterministic reward)
    rewards = []
    for prompt, response_group in zip(prompts, responses):
        group_rewards = [
            verifier(prompt, resp) for resp in response_group
        ]
        rewards.append(group_rewards)
    
    # 3. Compute advantages (GRPO-style: group-normalized)
    for group_rewards in rewards:
        mean_r = np.mean(group_rewards)
        std_r = np.std(group_rewards) + 1e-8
        advantages = [(r - mean_r) / std_r for r in group_rewards]
    
    # 4. Policy-gradient update
    loss = compute_policy_gradient_loss(
        policy, prompts, responses, advantages
    )
    loss.backward()
    optimizer.step()
    
    return loss.item(), np.mean([np.mean(r) for r in rewards])
```

---

## References

- Ouyang et al. (2022) -- [Training language models to follow instructions with human feedback (InstructGPT)](https://arxiv.org/abs/2203.02155)
- Lightman et al. (2023) -- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)
- Stiennon et al. (2020) -- [Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325)
- PRIME (2025) -- [arXiv:2502.01456](https://arxiv.org/abs/2502.01456)
- DeepSeek-R1 (2025) -- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
- Lambert et al. (2024) -- [RewardBench: Evaluating Reward Models](https://arxiv.org/abs/2403.13787)
- [awesome-RLVR](https://github.com/opendilab/awesome-RLVR)

---

## Related Pages

- [[rlhf-overview]] -- where the reward model sits in the RLHF pipeline
- [[ppo-for-llm]] -- the RL algorithm that consumes the reward signal
- [[grpo]] -- can use rule-based rewards in place of a learned RM
- [[dpo]] -- alternative that bypasses the reward model entirely
- [[multi-step-reasoning-rl]] -- using PRMs in reasoning RL
- [[rl-training-frameworks]] -- frameworks that support RM training
