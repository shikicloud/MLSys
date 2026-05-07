---
title: "RL for Multi-Step Reasoning"
category: agentic-rl
tags: [reasoning, chain-of-thought, prm, orm, mcts, deepseek-r1, o1, o3, grpo, star, rest]
created: 2026-04-13
updated: 2026-05-07
status: mature
---

# RL for Multi-Step Reasoning

> [!abstract]+ TL;DR
> RL trains LLMs to generate extended chains of thought (CoT) for solving complex problems, with **correct-answer reward** as the learning signal. Core technique behind **OpenAI o1/o3 and [[grpo#DeepSeek-R1|DeepSeek-R1]]**. Key insight: RL can incentivize models to *spontaneously* develop self-verification, backtracking, decomposition, multi-angle analysis — without explicitly teaching these patterns. DeepSeek-R1-Zero showed this works **with no SFT** when rewards are RLVR-based. Modern stack: long-CoT (32K+ tokens) + [[grpo|GRPO]] + verifiable rewards. Active research: **PRM** (process reward, +6 pp on MATH vs ORM), **PRIME** (implicit per-token Q from ORMs, 2.5× sample efficiency), MCTS-guided self-training (ReST-MCTS*).

### Training Loop

1. Sample problem (math, code, science, logic)
2. Model generates long CoT reasoning trace
3. Extract final answer and verify (via [[reward-modeling#RLVR|RLVR]] or reward model)
4. Update policy: increase probability of traces leading to correct answers

## DeepSeek-R1 Case Study

[arXiv:2501.12948](https://arxiv.org/abs/2501.12948) (Jan 2025). The most important open-source work in reasoning RL.

### Pure RL Produces Emergent CoT

Pure RL training on DeepSeek-V3-Base (no SFT) spontaneously produces chain-of-thought reasoning. The reward signal alone (answer correctness) is sufficient to incentivize step-by-step reasoning.

### The "Aha Moment"

During training, the model suddenly transitions from superficial "checking" to genuine self-correction:
- **Early**: "The answer is 42. Let me check... yes, 42." (fake verification)
- **Mid (aha)**: "The answer is 42. Wait -- step 3 has 7x8=54, but it should be 56. Correcting... the answer is 44." (real error detection)
- **Late**: "Let me verify with two methods. Method 1: 44. Method 2: 44. Both agree." (cross-validation)

### GRPO for Reasoning

DeepSeek-R1 uses [[grpo|GRPO]] instead of PPO. Key advantage: no value model needed (saves ~50% memory), which is critical for long reasoning traces.

$$\hat{A}_i = \frac{R_i - \text{mean}(\{R_j\}_{j=1}^G)}{\text{std}(\{R_j\}_{j=1}^G)}$$

For each problem, sample $G$ trajectories, compute rewards, normalize within the group.

### Four-Stage Training Pipeline

```
Stage 1: Cold Start -- Small amount of high-quality CoT SFT
Stage 2: Reasoning RL -- GRPO on math/code tasks (emergent reasoning)
Stage 3: Rejection Sampling + SFT -- Filter correct traces, mix with general SFT data
Stage 4: All-Scenario RL -- Second RL round with reasoning + helpfulness + safety rewards
```

Results: 79.8% on AIME 2024 (comparable to OpenAI o1), 97.3% on MATH-500.

## Process Reward vs Outcome Reward

### ORM (Outcome Reward Model)

Evaluates only the final answer: $R = \mathbb{1}[\text{correct}]$. Cheap to label but suffers from credit assignment issues -- a correct answer may result from flawed reasoning (lucky guess).

### PRM (Process Reward Model)

Evaluates each reasoning step: $R = \sum_i \log p(\text{step}_i \text{ correct})$. Provides dense per-step feedback, enabling precise error localization. But annotation is 10-100x more expensive.

```
ORM: Step1 -> Step2 -> Step3 -> Answer -> [ORM] -> R
     (no per-step evaluation)

PRM: Step1 -> Step2 -> Step3 -> Answer
      |        |        |        |
     r_1=0.9  r_2=0.3  r_3=0.8  r_4=0.6
     PRM detects Step 2 is problematic!
```

### MCTS for Step Verification

Monte Carlo Tree Search can auto-generate process reward labels by estimating each step's "value" (probability of reaching a correct answer from that step) through sampling.

Key papers:
- **PRM800K** (OpenAI, 2023): 800K human step-level labels. PRM significantly outperforms ORM for best-of-N selection.
- **PRIME** (2025): Achieves PRM-level guidance using only outcome labels. 2.5x sample efficiency.
- **ReST-MCTS*** ([arXiv:2406.03816](https://arxiv.org/abs/2406.03816)): Self-training via process-reward-guided tree search.

## STaR / ReST Methods

### STaR (Self-Taught Reasoner)

[arXiv:2203.14465](https://arxiv.org/abs/2203.14465) (2022). Iterative self-improvement: model generates reasoning, keeps correct ones, **rationalizes** incorrect ones (re-generates reasoning given the correct answer as hint), then SFT on the filtered set.

### ReST (Reinforced Self-Training)

[arXiv:2308.08998](https://arxiv.org/abs/2308.08998) (Google, 2023). Offline RL variant: (1) Generate many solutions, (2) Filter by reward model, (3) SFT on filtered data. More scalable than online RL; threshold can increase each iteration.

## Search and Verification

### Best-of-N (BoN)

Generate N reasoning traces, select best by reward model or majority vote. N=64 + PRM gives 10-20% improvement on MATH over greedy decoding. Linear compute cost, diminishing returns past N=8.

### Beam Search with Reward Model

Beam search where each beam's score comes from PRM rather than log-probability. Explores diverse reasoning paths while pruning unpromising ones.

### MCTS for Reasoning

Each node = partial reasoning state. MCTS explores via sampling continuations, using PRM for value estimates:

$$\text{UCB}(node) = V(node) + c \sqrt{\frac{\ln N_{parent}}{N_{node}}}$$

Balances exploration and exploitation. More efficient than BoN for complex problems but significantly more expensive to implement.

### Inference-Time Compute Scaling

$\text{Performance} \propto \log(\text{Inference Compute})$. More thinking tokens = better answers. OpenAI engineers confirmed o3 is "just a model trained with RL" -- reasoning happens in a single forward pass, not explicit tree search.

## Code Example: GRPO for Math Reasoning

```python
class GRPOReasoningTrainer:
    def __init__(self, policy, ref_model, group_size=8,
                 clip_eps=0.2, kl_coeff=0.02):
        self.policy = policy
        self.ref_model = ref_model
        self.group_size = group_size
        self.clip_eps = clip_eps
        self.kl_coeff = kl_coeff

    def grpo_step(self, questions, ground_truths):
        all_groups = []
        for q, gt in zip(questions, ground_truths):
            group = []
            for _ in range(self.group_size):
                traj = self.generate_reasoning(q)
                traj.reward = 1.0 if check_math_answer(traj.answer, gt) else 0.0
                group.append(traj)
            # Group-relative advantage normalization
            rewards = [t.reward for t in group]
            mean_r, std_r = mean(rewards), max(std(rewards), 1e-8)
            for t in group:
                t.advantage = (t.reward - mean_r) / std_r
            all_groups.append(group)

        # Policy gradient with clipping + KL penalty
        loss = 0
        for group in all_groups:
            for traj in group:
                ratio = exp(policy.log_prob(traj) - traj.old_log_prob)
                clipped = clamp(ratio, 1-self.clip_eps, 1+self.clip_eps)
                loss -= min(ratio * traj.advantage, clipped * traj.advantage)
                loss += self.kl_coeff * kl(policy, ref_model, traj)
        loss.backward()
        self.optimizer.step()
```

## Challenges

1. **Reward hacking**: Models find shortcuts (answer leakage from format, fake verification, length inflation without depth). Counter with diverse verification and adversarial testing.
2. **Length exploitation**: RL incentivizes longer traces but marginal accuracy gains diminish. Use length penalties: $R = R_{\text{task}} - \alpha \cdot \max(0, L - L_{\text{threshold}})$.
3. **Process annotation cost**: Step-level labels are 10-100x more expensive than outcome labels. MCTS-based auto-labeling and PRIME's implicit process rewards help.
4. **Domain transfer**: Math reasoning may not transfer to legal/medical/ethical reasoning. Open question.
5. **Training cost**: 32K+ token traces, 8-64 samples per problem, thousands of GPU hours for R1-class models.

## References

- DeepSeek-AI (2025). [DeepSeek-R1](https://arxiv.org/abs/2501.12948). arXiv:2501.12948.
- Lightman et al. (2023). [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050). arXiv:2305.20050.
- Zelikman et al. (2022). [STaR](https://arxiv.org/abs/2203.14465). arXiv:2203.14465.
- Gulcehre et al. (2023). [ReST](https://arxiv.org/abs/2308.08998). arXiv:2308.08998.
- Zhang et al. (2024). [ReST-MCTS*](https://arxiv.org/abs/2406.03816). arXiv:2406.03816.
- Snell et al. (2024). [Scaling LLM Test-Time Compute](https://arxiv.org/abs/2408.03314). arXiv:2408.03314.
- ACM Computing Surveys (2025). [Multi-Step Reasoning Survey](https://dl.acm.org/doi/10.1145/3774896).

## Related Pages

- [[reward-modeling]] -- PRMs and ORMs for reasoning
- [[grpo]] -- GRPO algorithm (DeepSeek-R1's core algorithm)
- [[agentic-rl-overview]] -- Broader agentic RL landscape
- [[tool-use-rl]] -- Combining reasoning with tool use
- [[ppo-for-llm]] -- PPO for LLM training
- [[rl-training-frameworks]] -- RL training frameworks
