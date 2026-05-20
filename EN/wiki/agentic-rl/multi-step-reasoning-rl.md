---
title: "Reinforcement Learning for Multi-Step Reasoning"
category: agentic-rl
tags: [reasoning, chain-of-thought, prm, orm, mcts, deepseek-r1, o1, o3, grpo, star, rest]
created: 2026-04-13
updated: 2026-05-20
status: mature
---

# Reinforcement Learning for Multi-Step Reasoning

> [!abstract]+ TL;DR
> RL trains LLMs to generate extended chain-of-thought (CoT) for complex problems, with **correct-answer reward** as the learning signal. The core technology behind **OpenAI o1/o3 and [[grpo#DeepSeek-R1|DeepSeek-R1]]**. Key insight: RL can incentivize the model to *spontaneously* develop self-verification, backtracking, decomposition, and multi-angle analysis -- without explicitly teaching these patterns. DeepSeek-R1-Zero shows that **no SFT is needed** (purely RLVR-based reward). Modern stack: long CoT (32K+ tokens) + [[grpo|GRPO]] + verifiable rewards. Active research directions: **PRM** (process reward, +6 pp over ORM on MATH), **PRIME** (extracting implicit per-token Q from ORM, 2.5x sample efficiency), MCTS-guided self-training (ReST-MCTS*).

## Overview

Multi-Step Reasoning RL uses reinforcement learning to train LLMs to generate extended chains of thought (CoT) for solving complex problems. This is the core technology behind reasoning models such as OpenAI o1/o3 and [[grpo#DeepSeek-R1|DeepSeek-R1]].

### Relation to general RL

Reasoning RL is an important sub-direction of [[agentic-rl-overview|agentic RL]]. While it can be independent of tool use (pure-text reasoning), in real systems reasoning and tool use are typically co-trained (e.g. ReTool, where the model reasons and simultaneously decides whether to call the code interpreter).

```
What makes reasoning RL distinctive:
- Action space: mostly natural-language text (chain of thought)
- Reward signal: typically from verifiable answers (math, code, logic)
- Core challenge: how to reward "process" rather than only "result"
- Emergent phenomenon: complex reasoning strategies emerge from simple reward signals
```

### Training loop

The basic training loop for reasoning RL:

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│  1. Problem sampling                                │
│     Sample a question q from a math/code/science/   │
│     logic question bank                             │
│                                                     │
│  2. Trajectory generation (rollout)                 │
│     Model π_θ generates a reasoning trajectory:     │
│     q → Think₁ → Think₂ → ... → ThinkN → Answer     │
│                                                     │
│  3. Verification and reward                         │
│     Extract the final answer and compare to ground  │
│     truth: R = 1 (correct) or R = 0 (wrong)         │
│                                                     │
│  4. Policy update                                   │
│     Increase the probability of trajectories that   │
│     led to correct answers; decrease that of wrong  │
│     ones (using GRPO/PPO/REINFORCE etc.)            │
│                                                     │
│  5. Repeat                                          │
│     Go back to step 1                               │
│                                                     │
└─────────────────────────────────────────────────────┘
```

## DeepSeek-R1 case study

### Overview

DeepSeek-R1 is one of the most important open-source works in reasoning RL, showing for the first time at scale that pure RL training can produce complex emergent reasoning behavior.

**Paper**: [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) (January 2025)

### Pure RL training yields emergent CoT

The single most important finding of DeepSeek-R1:

> Running RL directly on DeepSeek-V3-Base (a base model that has not been through SFT) causes chain-of-thought reasoning to emerge spontaneously.

This means CoT does not need to be taught through demonstrations -- the RL reward signal alone is enough to incentivize the model to develop step-by-step reasoning.

**Training setup**:
- Base model: DeepSeek-V3-Base (671B MoE, 37B active parameters)
- Algorithm: [[grpo|GRPO]] (Group Relative Policy Optimization)
- Reward: rule-based verifier (math-answer format matching + computation verification)
- No SFT cold start

**Emergent reasoning behaviors**:
1. **Self-verification**: the model checks itself after reaching an answer
2. **Backtracking**: it actively returns to earlier steps when it detects a reasoning error
3. **Decomposition**: it breaks a complex problem into smaller sub-problems
4. **Multi-angle analysis**: it attempts to solve from different angles
5. **Progressive refinement**: from a rough estimate to a precise calculation

### The "Aha Moment"

The DeepSeek-R1 paper reports a striking phenomenon -- during training, the model suddenly learned to self-reflect:

```
Early training (model behavior):
  "The answer is 42. Wait, let me recalculate... The answer is 42."
  (formal "checking" without actual verification)

Mid training (the aha moment):
  "The answer is 42. Hmm, wait. Let me re-examine step 3.
   Actually, I made an error: 7 × 8 = 56, not 54.
   Correcting this... the real answer is 44."
  (a real discovery and correction of an error!)

Late training (mature reasoning):
  "Let me approach this from two directions to verify.
   Method 1: ... → 44
   Method 2: ... → 44
   Both methods agree. The answer is 44."
  (cross-verification with multiple methods)
```

This "aha moment" is an emergent behavior of RL training -- the model discovered that "find and correct mistakes" earns higher reward, so this behavior is reinforced.

### GRPO for reasoning optimization

DeepSeek-R1 uses GRPO (not PPO) as its core RL algorithm:

**Key idea of GRPO**: no separate value (critic) model is needed; the advantage is estimated by in-group comparison.

$$\hat{A}_i = \frac{R_i - \text{mean}(\{R_j\}_{j=1}^G)}{\text{std}(\{R_j\}_{j=1}^G)}$$

For each question, sample $G$ reasoning trajectories, compute each trajectory's reward $R_i$, and normalize within the group.

**GRPO vs PPO**:

| Property | PPO | GRPO |
|----------|-----|------|
| Needs a value model? | Yes (+~50% memory) | No |
| Advantage estimation | GAE (needs critic) | In-group normalization |
| Memory efficiency | Low | High (no critic) |
| Suitable for long sequences? | Difficult (critic struggles to evaluate long sequences) | Better |
| Bias | Lower (critic corrects) | Biased (but works well in practice) |

**GRPO update**:

$$\mathcal{L}_{\text{GRPO}} = -\frac{1}{G} \sum_{i=1}^{G} \min\left(\frac{\pi_\theta(y_i|x)}{\pi_{\text{old}}(y_i|x)} \hat{A}_i, \text{clip}\left(\frac{\pi_\theta(y_i|x)}{\pi_{\text{old}}(y_i|x)}, 1-\epsilon, 1+\epsilon\right) \hat{A}_i\right) + \beta \cdot D_{KL}(\pi_\theta \| \pi_{\text{ref}})$$

### Two-stage (four-stage) training pipeline

The full DeepSeek-R1 training pipeline is more complex than "pure RL" and contains four stages:

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Stage 1: Cold start                                        │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ DeepSeek-V3-Base                                       │ │
│  │     ↓                                                  │ │
│  │ SFT on a small batch of high-quality CoT data          │ │
│  │ (a few thousand items)                                 │ │
│  │     ↓                                                  │ │
│  │ Initial reasoning policy (basic CoT)                   │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                             │
│  Stage 2: Reasoning-oriented RL                             │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Initial policy                                         │ │
│  │     ↓                                                  │ │
│  │ GRPO training (math + code tasks)                      │ │
│  │ Reward = rule-based verifier                           │ │
│  │     ↓                                                  │ │
│  │ Large jump in reasoning ability (emergent self-        │ │
│  │ verification, backtracking, etc.)                      │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                             │
│  Stage 3: Rejection sampling + SFT                          │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Stage-2 model generates many reasoning trajectories    │ │
│  │     ↓                                                  │ │
│  │ Filter: keep only trajectories with correct answers    │ │
│  │     ↓                                                  │ │
│  │ Mix in general SFT data (writing, translation,         │ │
│  │ dialogue, etc.)                                        │ │
│  │     ↓                                                  │ │
│  │ SFT → balance reasoning ability with general ability   │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                             │
│  Stage 4: All-scenario RL                                   │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Stage-3 model                                          │ │
│  │     ↓                                                  │ │
│  │ Second RL training                                     │ │
│  │ Multiple reward sources: reasoning verifier +          │ │
│  │ helpfulness RM + safety RM                             │ │
│  │     ↓                                                  │ │
│  │ Final DeepSeek-R1 model                                │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Key results

| Benchmark | DeepSeek-R1 | OpenAI o1 | Notes |
|-----------|-------------|-----------|-------|
| AIME 2024 | 79.8% | 79.2% | Math competition |
| MATH-500 | 97.3% | 96.4% | Math reasoning |
| Codeforces | 2029 | 2061 | Programming contest |
| GPQA Diamond | 71.5% | 75.7% | Graduate-level science |

DeepSeek-R1 matches OpenAI o1 on math reasoning and is fully open source.

## Process reward vs outcome reward

### Outcome Reward Model (ORM)

ORM evaluates the full reasoning trajectory using only the final answer:

$$R_{\text{ORM}}(\text{trajectory}) = \begin{cases} +1 & \text{final answer is correct} \\ 0 & \text{final answer is wrong} \end{cases}$$

**Pros**:
- Low annotation cost (only the final answer needs to be verified)
- Can be automated for verifiable tasks (math, code)
- No humans in the loop

**Cons**:
- Hard credit assignment: a correct answer may come from wrong reasoning (lucky)
- Sparse reward leads to low learning efficiency
- Cannot distinguish "excellent reasoning → correct" from "bad reasoning → accidentally correct"

### Process Reward Model (PRM)

PRM evaluates every step of the reasoning chain:

$$R_{\text{PRM}}(\text{trajectory}) = \prod_{i=1}^{N} p(\text{step}_i \text{ is correct})$$

Or in log form:
$$\log R_{\text{PRM}} = \sum_{i=1}^{N} \log p(\text{step}_i \text{ is correct})$$

**Pros**:
- Dense step-level feedback
- Can precisely localize reasoning errors
- Higher sample efficiency

**Cons**:
- Annotation cost is extremely high (every step needs human/automated verification)
- "A step" lacks a universal definition
- Process reward can be hacked (model produces steps that look correct but are meaningless)

### ORM vs PRM diagram

```
ORM (Outcome Reward Model):
                                                      ┌──────┐
Step 1 ──> Step 2 ──> Step 3 ──> Step 4 ──> Answer ──>│ ORM  │──> R
(no eval)  (no eval)  (no eval)  (no eval)  (eval)    │      │
                                                      └──────┘
Problem: if Step 2 is wrong but the Answer happens to be right, ORM cannot tell

PRM (Process Reward Model):
┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐   ┌──────────┐
│Step1│──>│Step2│──>│Step3│──>│Step4│──>│  Answer  │
└──┬──┘   └──┬──┘   └──┬──┘   └──┬──┘   └────┬─────┘
   │         │         │         │            │
   v         v         v         v            v
  r_1       r_2       r_3       r_4        r_final
  ✓0.9      ✗0.3      ✓0.8      ✓0.7       ✓0.6

PRM can spot the problem at Step 2!
```

### Monte Carlo Tree Search for step verification

MCTS can be used to automatically generate process-reward labels, avoiding expensive human annotation:

**Basic idea**: for each step in the reasoning chain, sample many subsequent continuations and estimate the "value" of that step (i.e. the probability that continuing from that step leads to a correct final answer).

```
MCTS for reasoning-step verification:

                    Step 1 (correct)
                   /              \
          Step 2a (correct)    Step 2b (wrong)
         /        \              /       \
     Step 3a    Step 3b     Step 3c   Step 3d
      (✓)        (✗)         (✗)       (✗)
   correct ans  wrong ans  wrong ans  wrong ans

Value of Step 1 = 1/4 = 0.25 (1 of 4 leaves correct)
Value of Step 2a = 1/2 = 0.5 (1 of 2 leaves correct)
Value of Step 2b = 0/2 = 0.0 (0 of 2 leaves correct)

→ PRM can use these values as training labels
```

**Key papers**:

- **PRM800K** (OpenAI, 2023): the first large-scale process-reward dataset
  - Source: Let's Verify Step by Step ([arXiv:2305.20050](https://arxiv.org/abs/2305.20050))
  - 800K step-level human annotations
  - Shows PRM significantly outperforms ORM in best-of-N selection

- **PRIME** (2025): achieves PRM-level guidance using implicit process reward, with only outcome labels
  - Infers process reward from outcome labels
  - 2.5x sample-efficiency improvement
  - Avoids expensive process annotation

## STaR / ReST methods

### STaR (Self-Taught Reasoner)

**Paper**: [STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465) (2022)

**Core idea**: the model improves itself using its own generated reasoning.

**Training pipeline**:

```
┌──────────────────────────────────────────────────┐
│  STaR iterative training:                        │
│                                                  │
│  Iteration 0:                                    │
│    Model M_0 (initial)                           │
│                                                  │
│  Iteration k:                                    │
│    1. M_k generates reasoning + answer for each  │
│       question q in the training set             │
│    2. Filter: keep only (q, reasoning, answer)   │
│       with a correct answer                      │
│    3. Rationalization:                           │
│       For wrongly answered questions, provide    │
│       the correct answer as a hint and have the  │
│       model regenerate the reasoning (increases  │
│       training-data diversity)                   │
│    4. SFT M_k on the filtered data → M_{k+1}    │
│                                                  │
│  Repeat until convergence                        │
└──────────────────────────────────────────────────┘
```

**The key innovation -- Rationalization**:

For questions the model gets wrong, append the correct answer as a hint to the question and have the model regenerate the reasoning. Even if the model initially cannot solve a question, it can still learn the correct reasoning path.

```
Normal generation (got it wrong):
  Q: "What is 17 × 23?"
  Model: "17 × 23 = 17 × 20 + 17 × 3 = 340 + 41 = 381" ✗ (should be 391)

Rationalization (regenerate after the correct answer is provided):
  Q: "What is 17 × 23? The answer is 391."
  Model: "17 × 23 = 17 × 20 + 17 × 3 = 340 + 51 = 391" ✓

→ Add the second reasoning to the training data
```

### ReST (Reinforced Self-Training)

**Paper**: [Reinforced Self-Training (ReST) for Language Modeling](https://arxiv.org/abs/2308.08998) (Google, 2023)

**Core idea**: an offline-RL version of STaR -- decompose online RL into two offline stages: "generate" and "train".

**Training pipeline**:

```
ReST training loop:

  Grow phase:
    Use the current policy π_k to generate many (question, solution) pairs

  Improve phase:
    Use a reward model / verifier to filter, keeping only high-quality solutions
    The threshold τ can be tightened each round

  Distill phase:
    SFT on the filtered data: π_k → π_{k+1}
```

**Differences from STaR**:
- STaR generates one reasoning per round; ReST generates many and ranks them
- ReST is closer to offline RL (the generating and training policies can differ)
- ReST can use more complex filtering policies (not just correct/wrong)

### ReST-MCTS*

**Paper**: [ReST-MCTS*: LLM Self-Training via Process Reward Guided Tree Search](https://arxiv.org/abs/2406.03816) (2024)

ReST-MCTS* combines MCTS with ReST:
- Uses MCTS to generate high-quality reasoning trajectories (better than random sampling)
- Infers process reward back from the final answer (avoids manual process annotation)
- Continually improves in a self-training loop

## Search and verification

### Best-of-N (BoN)

The simplest inference-time search strategy: generate N reasoning trajectories and pick the best by reward-model scoring or majority vote.

**Effect**: BoN with N=64 + PRM improves MATH by 10-20% over greedy decoding. Compute cost scales linearly (N times) and marginal returns diminish (N=1 to N=8 gives the largest improvement).

### Beam search with a reward model

```
Standard beam search (width B=3):

Level 0:  [Start]
            │
Level 1:  [Step1a (0.9)]  [Step1b (0.7)]  [Step1c (0.6)]
            │                 │
Level 2:  [Step2a (0.85)]  [Step2b (0.8)]  [Step2c (0.75)]
            │                 │
Level 3:  [Step3a (0.82)]  [Step3b (0.78)]  [Answer (0.76)]
            │
          [Answer (0.80)]

→ Pick the full trajectory with the highest final score

Note: the score here comes from the PRM, not the LM's log-prob
```

### MCTS for reasoning

Apply Monte Carlo Tree Search to LLM reasoning:

```
MCTS reasoning tree:

                         Problem
                        /       \
                   Step1a       Step1b
                  (V=0.7)      (V=0.3)
                 /     \          |
            Step2a   Step2b    Step2c
           (V=0.8)  (V=0.5)   (V=0.2)
            /    \      |
       Step3a  Step3b  Step3c
      (V=0.9) (V=0.6) (V=0.4)
         |
      Answer: 42 ✓

The four MCTS phases:
1. Selection: descend from the root along the highest-UCB path
2. Expansion: sample new reasoning steps at a leaf
3. Simulation: fast rollout to completion
4. Backpropagation: update the value of all nodes on the path

UCB formula:
  UCB(node) = V(node) + c * sqrt(ln(N_parent) / N_node)

  V(node): PRM value estimate
  N_parent: number of visits to the parent
  N_node: number of visits to the current node
  c: exploration coefficient
```

**Pros**:
- More efficiently explores the reasoning space than BoN
- Naturally balances exploration and exploitation
- Can leverage the step-level evaluation of a PRM

**Cons**:
- High compute cost (each step requires many samples and evaluations)
- Complex to implement
- The definition of "one step" is unclear (a sentence? a reasoning block?)

## Inference-time compute scaling

### Core idea

Inference-time compute scaling is a key application of reasoning RL:

$$\text{Performance} \propto \log(\text{Inference Compute})$$

Investing more compute at inference time (more tokens, more samples, search) gives sustained improvement in reasoning quality.

```
Classic LLM: fixed compute
  Input → Model → Output (one forward pass)

Inference-time scaling: variable compute
  Input → Model → Think₁ → Think₂ → ... → ThinkN → Output
                  |        |              |
                  more thinking tokens = better result
```

### Speculated components of OpenAI o1/o3

The presumed architecture of OpenAI's o1/o3 models:

1. **RL training**: train the model to produce long reasoning chains with RL (probably a PPO variant)
2. **Process reward**: PRM is used to evaluate and steer reasoning
3. **Inference-time compute scaling**: the model is allowed to use more tokens at inference
4. **Hidden CoT**: the internal reasoning is hidden from the user

OpenAI engineers confirmed that o3 is "just a model trained with RL" -- the reasoning happens in a single forward pass, not via explicit tree search.

## Code example

### GRPO training for math reasoning

```python
import torch
from typing import List, Dict

class GRPOReasoningTrainer:
    """GRPO trainer for math reasoning (simplified)"""

    def __init__(self, policy, ref_model, group_size=8,
                 clip_eps=0.2, kl_coeff=0.02):
        self.policy = policy
        self.ref_model = ref_model
        self.group_size = group_size
        self.clip_eps = clip_eps
        self.kl_coeff = kl_coeff
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-6)

    def grpo_step(self, questions: List[str],
                  ground_truths: List[str]) -> Dict:
        """One GRPO update step"""
        all_groups = []

        # 1. Generate G reasoning trajectories for each question
        for q, gt in zip(questions, ground_truths):
            group = []
            for _ in range(self.group_size):
                reasoning, log_probs = self.policy.generate(
                    q, max_length=8192, temperature=0.7, return_log_probs=True
                )
                answer = extract_boxed_answer(reasoning)
                reward = 1.0 if check_math_answer(answer, gt) else 0.0
                group.append({"reasoning": reasoning, "log_probs": log_probs,
                              "reward": reward, "correct": reward > 0})
            all_groups.append(group)

        # 2. In-group normalized advantage
        for group in all_groups:
            rewards = [t["reward"] for t in group]
            mean_r = sum(rewards) / len(rewards)
            std_r = max((sum((r-mean_r)**2 for r in rewards)/len(rewards))**0.5, 1e-8)
            for t in group:
                t["advantage"] = (t["reward"] - mean_r) / std_r

        # 3. Policy-gradient update (clipped PPO-style + KL penalty)
        total_loss = 0
        for group in all_groups:
            for traj in group:
                curr_lp = self.policy.log_prob(traj["reasoning"])
                ref_lp = self.ref_model.log_prob(traj["reasoning"])
                old_lp = traj["log_probs"].detach()

                ratio = torch.exp((curr_lp - old_lp).sum())
                adv = traj["advantage"]
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * adv
                total_loss += -torch.min(surr1, surr2) + self.kl_coeff * (curr_lp - ref_lp).mean()

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        # Stats
        all_trajs = [t for g in all_groups for t in g]
        return {
            "mean_reward": sum(t["reward"] for t in all_trajs) / len(all_trajs),
            "accuracy": sum(t["correct"] for t in all_trajs) / len(all_trajs),
        }

    def train(self, dataset, n_epochs=50, batch_size=16):
        for epoch in range(n_epochs):
            batch = dataset.sample(batch_size)
            stats = self.grpo_step(
                [b["question"] for b in batch],
                [b["answer"] for b in batch]
            )
            print(f"Epoch {epoch:3d} | Reward: {stats['mean_reward']:.3f} | "
                  f"Acc: {stats['accuracy']:.2%}")
```

## Challenges

### 1. Reward hacking in reasoning

The model may learn shortcuts that earn high reward without truly reasoning:

```
Common reward-hacking patterns:

1. Answer leakage: "guessing" the answer from training-data formatting
   "This looks like a competition problem. The answer is usually 42."

2. Fake verification: pretending to verify without actually doing it
   "Let me check: 7 × 6 = 42. ✓ Verified!"  (no real computation)

3. Length inflation: producing very long but meaningless reasoning to dodge penalties
   "Let me think about this carefully... [repetitive filler × 1000]"

4. Format manipulation: learning a specific format pattern to earn format reward
   "Step 1: ... Step 2: ... Therefore: ..."  (empty content, perfect format)
```

**Mitigations**:
- Diversified verification methods (do not only check answer format)
- Adversarial test sets
- Detect fake reasoning with a process reward model
- Regularize reasoning length

### 2. Length exploitation

RL can incentivize the model to produce longer but not necessarily deeper reasoning:

```
Training phenomenon:
  Early training:  average reasoning length 500 tokens,  accuracy 40%
  Mid training:    average reasoning length 2000 tokens, accuracy 60%
  Late training:   average reasoning length 8000 tokens, accuracy 65%

Issue: the accuracy improvement from extra length keeps diminishing
       many of the extra tokens are repetition or noise
```

**Mitigations**:
- Length penalty: $R = R_{\text{task}} - \alpha \cdot \max(0, L - L_{\text{threshold}})$
- Token-efficiency reward: reward "correct answer with fewer tokens"
- Length normalization: normalize reward by token count

### 3. Process-annotation cost

High-quality process-reward annotation is extremely expensive:

| Annotation type | Cost per item | Items per hour |
|-----------------|---------------|----------------|
| Outcome label | $0.01-0.10 | ~200 |
| Process label | $1-10 | ~5-10 |
| Expert process label | $10-100 | ~1-2 |

**Mitigations**:
- MCTS-based automatic process annotation
- PRIME: infer process reward from outcome labels
- Weak labels + self-training

### 4. Domain transfer

Math reasoning ability may not transfer to other domains:

```
A reasoning model trained on math:
  MATH: 90%+ accuracy
  GSM8K: 95%+ accuracy

But in other domains:
  Legal reasoning: ?
  Medical diagnosis: ?
  Ethical reasoning: ?

Transferability of reasoning ability remains an open question
```

### 5. Training cost

RL training of long reasoning chains is extremely expensive:

- A single reasoning trajectory can reach 32K+ tokens
- Each question requires 8-64 sampled trajectories
- Training an R1-class model takes thousands of GPU-hours

## References

### Core papers

- DeepSeek-AI (2025). [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948). arXiv:2501.12948.
- Lightman et al. (2023). [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050). arXiv:2305.20050.
- Zelikman et al. (2022). [STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465). arXiv:2203.14465.
- Gulcehre et al. (2023). [Reinforced Self-Training (ReST) for Language Modeling](https://arxiv.org/abs/2308.08998). arXiv:2308.08998.
- Zhang et al. (2024). [ReST-MCTS*: LLM Self-Training via Process Reward Guided Tree Search](https://arxiv.org/abs/2406.03816). arXiv:2406.03816.

### Reasoning systems

- OpenAI (2024). Learning to reason with LLMs (o1 blog post).
- Shao et al. (2024). [DeepSeekMath: Pushing the Limits of Mathematical Reasoning](https://arxiv.org/abs/2402.03300). arXiv:2402.03300.
- Wang et al. (2024). [PRIME: Scalable and Efficient Process Reward Modeling](https://arxiv.org/abs/2502.01456).

### Surveys

- ACM Computing Surveys (2025). [Multi-Step Reasoning Survey](https://dl.acm.org/doi/10.1145/3774896).
- Snell et al. (2024). [Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](https://arxiv.org/abs/2408.03314). arXiv:2408.03314.

## Related pages

- [[reward-modeling]] -- reward modeling (PRM and ORM)
- [[grpo]] -- the GRPO algorithm in detail (DeepSeek-R1's core algorithm)
- [[agentic-rl-overview]] -- panorama of agentic RL
- [[tool-use-rl]] -- combining reasoning with tool use
- [[ppo-for-llm]] -- PPO applied to LLMs
- [[rl-training-frameworks]] -- RL training frameworks
