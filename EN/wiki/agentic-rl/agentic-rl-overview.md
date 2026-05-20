---
title: "Agentic RL Overview"
category: agentic-rl
tags: [agentic-rl, multi-turn-rl, agent-training, agent-r1, agentrl, rlhf, grpo, ppo]
created: 2026-04-13
updated: 2026-05-20
status: mature
---

# Agentic RL Overview

> [!abstract]+ TL;DR
> Agentic RL trains AI agents to interact with external environments/tools — optimizing **action sequences, tool calls, and task completion** over multi-turn interactions rather than only single-turn text quality as in traditional [[rlhf-overview|RLHF]]. Treat the LLM as a policy in an MDP: state = dialogue history + environment state, action = text/tool call, reward = task completion signal. **Paradigm shift**: from "training the model to say things" to "training the model to do things". Powers DeepSeek-R1-style reasoning, WebGPT-style search, ReTool-style code-interpreter agents.

## Overview

Agentic RL is a class of reinforcement learning methods specifically for training AI agents to interact with external environments and tools. Unlike traditional [[rlhf-overview|RLHF]] (which mostly optimizes single-turn text generation quality), Agentic RL focuses on optimizing action sequences, tool calls, and task completion across multi-turn interactions.

**Core definition**: Agentic RL frames the LLM as a policy operating in a Markov Decision Process (MDP):
- **State**: dialogue history + environment state (file system, web pages, databases, etc.)
- **Action**: text generation, including natural language responses and structured tool calls
- **Transition**: environment executes the action and returns an observation
- **Reward**: task completion signal, efficiency metric, safety constraint

This is a fundamental paradigm shift from "training the model to say things" to "training the model to do things". In 2025-2026, with the success of DeepSeek-R1, WebGPT, ReTool, etc., Agentic RL has become a core technology path for building next-generation AI agents.

### Why Agentic RL?

Traditional SFT (supervised fine-tuning) trains agents by imitating expert trajectories, but has fundamental limits:

1. **Distribution shift**: SFT models degrade sharply on out-of-distribution states
2. **Insufficient exploration**: imitation can't discover better policies not shown by the expert
3. **Error accumulation**: in multi-step decisions, small deviations compound exponentially
4. **Weak generalization**: SFT agents struggle to adapt to new environment configurations

RL solves these via trial-and-error: agents learn to recover from mistakes, explore new strategies, and optimize behavior over long horizons.

## Differences from Traditional RLHF

### Paradigm comparison

| Dimension | Traditional RLHF | Agentic RL |
|-----------|------------------|------------|
| **Optimization target** | Text quality (helpful, safe, honest) | Action sequences and task completion |
| **Interaction turns** | Single-turn (prompt → response) | Multi-turn (tens to hundreds) |
| **State space** | Text only (prompt + response) | Text + environment state |
| **Action space** | Natural language text | Text + tool calls + code execution |
| **Reward signal** | Human preference (dense, subjective) | Task completion (sparse, objectively verifiable) |
| **Reward source** | Reward model (RM) | Environment execution feedback (RLEF) |
| **Time horizon** | Single step | Long (episodes up to hundreds of steps) |
| **Credit assignment** | Simple (whole response) | Complex (which action caused success?) |
| **Environment dep.** | None | Needs sandboxed execution environment |
| **Safety constraints** | Content safety | Content safety + behavior safety |

### Flow comparison (ASCII)

```
Traditional RLHF flow:
┌──────────────────────────────────────────────────┐
│                                                  │
│  User Prompt ──> LLM ──> Response ──> Reward Model ──> Score
│       │                                    │           │
│       │                                    │           │
│       └────────────── Policy Update <──────┘───────────┘
│                                                  │
│  Single-turn interaction, text quality optimization│
└──────────────────────────────────────────────────┘

Agentic RL flow:
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  Task ──> LLM Agent ──┬──> Text Response                     │
│    ^                  │                                      │
│    │                  ├──> Tool Call ──> Environment ──> Obs  │
│    │                  │                      │                │
│    │                  ├──> Code Exec ──> Sandbox ──> Result   │
│    │                  │                      │                │
│    │                  └──> ... (multi-turn loop) ...           │
│    │                                         │                │
│    │         Task Completion? ──> Reward ─────┘                │
│    │              │                                           │
│    └──── Policy Update <──────────────────────────────────────┘
│                                                              │
│  Multi-turn interaction, environment execution, sparse rewards│
└──────────────────────────────────────────────────────────────┘
```

### Key differences in depth

**1. Reward delay and sparsity**

In RLHF, every response gets an immediate reward score. In Agentic RL, reward usually only appears at the end of the task:

```
RLHF:    Prompt → Response → Reward ✓ (immediate)
Agentic: Task → Action1 → Obs1 → Action2 → Obs2 → ... → ActionN → Reward ✓ (delayed)
```

This makes credit assignment the core challenge: when a 50-step task finally succeeds, the model must learn which steps were critical.

**2. Mixed action space**

RLHF's action space is pure text tokens. Agentic RL's is mixed:

```
Action space = {
    natural language text,        # "Let me analyze this problem..."
    tool call (name, args),       # search("quantum computing")
    code execution (code),        # python: result = solve(equation)
    special actions               # <submit>, <give_up>, <ask_user>
}
```

**3. Irreversibility of environment interaction**

In RLHF, generated text can be discarded and regenerated. In Agentic RL, some actions are irreversible (deleting files, sending emails, executing trades), introducing the need for safety constraints.

## Core Paradigm

### LLM as Policy

In the Agentic RL framework, the LLM is formalized as a policy function:

$$\pi_\theta(a_t | s_t) = \text{LLM}_\theta(\text{action} | \text{history}_t, \text{env\_state}_t)$$

Where:
- $s_t = (h_t, e_t)$: state is the dialogue history $h_t$ plus environment state $e_t$
- $a_t$: action at timestep $t$ (text or tool call)
- $\theta$: LLM parameters

**Key properties**:
- Actions are autoregressively generated token sequences
- A single "action" may contain multiple tokens (e.g., a complete function call)
- Observations are injected into the context as part of the next state

### Episode structure

A typical Agentic RL episode:

```
Episode = {
    (s_0, a_0, r_0, s_1),   # initial state, first action
    (s_1, a_1, r_1, s_2),   # observe tool output, second action
    ...
    (s_T, a_T, R_T, done)   # final action, task-level reward
}

Where:
  s_t = concat(system_prompt, task, history_{0:t-1}, env_obs_t)
  a_t = LLM_θ(s_t)  # may be text or tool call
  r_t = 0 (intermediate steps) or R(task_result) (final step)
```

### Reward design

Reward design is the most critical and difficult part of Agentic RL:

**1. Outcome reward**
$$R_{\text{outcome}} = \begin{cases} +1 & \text{task completed successfully} \\ 0 & \text{task failed} \\ -1 & \text{harmful outcome} \end{cases}$$

**2. Efficiency reward**
$$R_{\text{efficiency}} = -\alpha \cdot \text{num\_steps} - \beta \cdot \text{num\_tool\_calls}$$

Encourages completing tasks in fewer steps and tool calls.

**3. Process reward**
$$R_{\text{process}} = \sum_{t=0}^{T} r_t^{\text{progress}}$$

Intermediate reward per step (e.g., correctly chose a tool, generated valid arguments).

**4. Safety penalty**
$$R_{\text{safety}} = -\gamma \cdot \mathbb{1}[\text{unsafe\_action}]$$

Penalty for dangerous actions (e.g., attempts to delete system files, access unauthorized APIs).

**5. Composite reward**
$$R_{\text{total}} = R_{\text{outcome}} + \lambda_1 R_{\text{efficiency}} + \lambda_2 R_{\text{process}} + \lambda_3 R_{\text{safety}}$$

### Exploration challenges

The exploration problem in Agentic RL is more severe than in traditional RL:

1. **Huge action space**: the LLM's action space is the combinatorial space of all possible token sequences, essentially infinite
2. **Sparse rewards**: positive reward only on task completion, many exploration trajectories yield zero
3. **Long horizon**: must make coherent decision sequences over tens to hundreds of steps
4. **Combinatorial explosion**: the tool × argument × execution-order combination space grows exponentially

**Mitigations**:
- **Curriculum learning**: start with easy tasks and gradually increase difficulty
- **SFT warm start**: initial training on expert trajectories to narrow exploration
- **Dense reward shaping**: add intermediate rewards to guide exploration
- **Experience replay**: store successful trajectories and learn from them repeatedly
- **Hierarchical exploration**: learn sub-skills first, then compose them

## Key Research Directions

### 1. Tool-use RL

[[tool-use-rl|Tool-use RL]] studies how to use RL to train LLMs to learn when and how to call external tools.

**Core questions**:
- When to call a tool vs. when to reason purely?
- How to pick the right tool and arguments?
- How to extract useful info from tool outputs?

**Representative work**: ReTool, Toolformer, RLEF

### 2. Multi-step reasoning RL

[[multi-step-reasoning-rl|Multi-step reasoning RL]] uses RL to train LLMs to generate extended chains of thought for complex problems.

**Core questions**:
- How to incentivize self-verification, backtracking, and advanced reasoning?
- Tradeoff between process and outcome rewards?
- How to prevent reasoning length inflation without depth?

**Representative work**: DeepSeek-R1, OpenAI o1/o3, STaR

### 3. Environment design

[[environment-design|Environment design]] determines what an agent can learn and how well.

**Core questions**:
- How to design high-fidelity, massively parallelizable training environments?
- How to balance environment diversity and realism?
- How to design effective curricula from simple to complex?

**Representative work**: SWE-bench, WebArena, OpenReward

### 4. Agent paradigms

Different task domains gave rise to different agent paradigms:

| Paradigm | Description | Representative systems |
|----------|-------------|------------------------|
| **WebAgent** | Navigate web pages, click, fill forms | WebGPT, WebArena, MindAct |
| **CodeAgent** | Write, debug, execute code | SWE-Agent, OpenHands, Devin |
| **SearchAgent** | Search, retrieve, synthesize | Perplexity, SearchGPT |
| **ToolAgent** | Call various APIs and tools | Gorilla, ToolLLM, API-Bank |
| **OSAgent** | Operate desktop GUI and CLI | OSWorld, CogAgent |
| **MultiAgent** | Multi-agent collaboration | MARTI, AutoGen, CrewAI |

## Representative Work

### DeepSeek-R1: RL-trained emergent reasoning

**Paper**: [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL](https://arxiv.org/abs/2501.12948) (January 2025)

**Key findings**:
- Pure RL (no SFT) can elicit complex reasoning behaviors
- Uses [[grpo|GRPO]], trained on DeepSeek-V3-Base
- Observed "Aha Moment": the model spontaneously learns self-reflection and error correction
- Two-stage training: (1) RL warm start, (2) RL + SFT mixed training + second RL

**Training pipeline**:
```
Stage 1: Cold Start
  DeepSeek-V3-Base → SFT on small set of high-quality CoT data → initial policy

Stage 2: Reasoning RL
  initial policy → GRPO training (math/code tasks) → emergent reasoning

Stage 3: Rejection Sampling + SFT
  RL model generates reasoning trajectories → filter → train with mixed general SFT data

Stage 4: All-scenario RL
  mixed model → second-stage RL (reasoning + alignment) → final model
```

**Result**: 79.8% pass@1 on AIME 2024, comparable to OpenAI o1.

### WebGPT: web browsing via RL

**Paper**: [WebGPT: Browser-Assisted Question-Answering with Human Feedback](https://arxiv.org/abs/2112.09332) (OpenAI, 2021)

**Core design**:
- Give GPT-3 a set of browsing actions: search, click links, scroll, cite
- Train the browsing policy via RL from Human Feedback
- Model learns to search relevant info, combine multiple sources, generate cited answers

**Training pipeline**:
1. Collect human web-browsing demonstrations → behavior cloning (BC)
2. Train reward model (based on human preference comparisons)
3. Optimize browsing policy with PPO

**Significance**: WebGPT was among the earliest successful "LLM + environment interaction + RL" systems.

### Toolformer: self-supervised tool learning

**Paper**: [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761) (Meta, 2023)

**Core method**:
- Doesn't use RL — uses self-supervision to teach the model to insert tool calls into text
- Model self-annotates positions where inserting a tool call would lower perplexity
- Supports calculator, search engine, translator, calendar tools

**Relation to RL**:
- Toolformer is SFT-based. Subsequent work (EMNLP 2025) showed the same capability can be learned from scratch via pure RL
- ReTool extends these ideas in the RL framework with stronger performance

### ReAct: Reasoning + Acting

**Paper**: [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) (2022)

**Core idea**:
- Alternate Thought (reasoning) and Action (acting) steps
- Reasoning helps the model plan, track state, handle exceptions
- Acting helps the model fetch external info, execute operations

**Example format**:
```
Thought 1: I need to search for Apple's latest market cap
Action 1: search("Apple Inc market cap 2026")
Observation 1: Apple's market cap is $4.2 trillion as of April 2026...
Thought 2: Found the info, let me extract the key number
Action 2: extract_number("$4.2 trillion")
Observation 2: 4200000000000
Thought 3: Now I can answer the user's question
Action 3: finish("Apple's market cap is approximately $4.2 trillion.")
```

**Significance**: ReAct defined the basic interaction pattern of modern LLM agents; nearly all later work uses a similar Thought-Action-Observation loop.

### RLEF: RL from Execution Feedback

**Core idea**: Use the actual results of environment execution as the reward signal, rather than human preference or a learned reward model.

**Advantages**:
- Reward is objective and verifiable (code passes tests, API returns correct result)
- No expensive human labels
- Avoids reward model bias and overfitting

**Use cases**:
```
Code execution RLEF:  code → execute → tests pass/fail → reward
SQL RLEF:             SQL → execute → result correct/wrong → reward
API RLEF:             API call → execute → return value valid/invalid → reward
Web RLEF:             browse action → execute → reached target page or not → reward
```

## Major Frameworks (2025-2026)

### AgentGym-RL (ICLR 2026 Oral)

A unified framework for training long-horizon decision-making agents, supporting web navigation, deep search, multi-step reasoning, etc. Proposes a standardized agent environment interface and training pipeline.

[arXiv:2509.08755](https://arxiv.org/abs/2509.08755)

### AgentRL

Multi-turn, multi-task agentic RL framework:
- Fully asynchronous generation-training pipeline
- Cross-policy sampling: different tasks can use different sampling policies
- Task advantage normalization: handles inconsistent reward scales across tasks

[arXiv:2510.04206](https://arxiv.org/abs/2510.04206)

### Agent-R1

Modular framework extending single-turn RL to multi-turn agentic tasks:
- Supports PPO, GRPO, REINFORCE++ and other algorithms
- Multi-tool coordination
- Flexible environment interface

[arXiv:2511.14460](https://arxiv.org/abs/2511.14460)

### MARTI (ICLR 2026, Tsinghua)

Multi-Agent Reinforced Training and Inference:
- Asynchronous tool use
- Multi-agent RL workflow
- Inter-agent cooperation and competition

### ProRL Agent (NVIDIA, March 2026)

"Rollout-as-a-Service" architecture:
- Decouples agent rollout orchestration from the training loop
- Supports massively parallel rollouts
- Flexible environment backends

## Technical Challenges

### 1. Credit assignment

Over long horizons, determining which action mattered most is extremely hard:

```
Episode: a_1, a_2, a_3, ..., a_50 → Reward = +1

Question: which a_i are "good"? Which are "irrelevant"? Which are "harmful but salvaged later"?
```

**Existing methods**:
- Turn-level reward: an auxiliary reward model scores each turn
- Process reward model (PRM): train a model to evaluate each intermediate step
- TD learning: use value functions for step-by-step credit assignment
- GAE (generalized advantage estimation): bias-variance balanced advantage estimation

### 2. Sparse rewards

Most real tasks only give reward at the end:

```
Coding task: 50 interaction steps → all tests pass? → +1 or 0
Web navigation: 20 browsing steps → reached target page? → +1 or 0
```

**Mitigations**:
- Reward shaping: add heuristic intermediate rewards
- Curiosity-driven exploration
- Hierarchical RL: decompose into sub-goals
- Dense process reward models

### 3. Safety constraints

Agents acting in real environments introduce serious safety risks:

- **Irreversible actions**: delete files, send messages, execute trades
- **Information leakage**: access sensitive data, expose system info
- **Resource abuse**: infinite loops, excessive API calls, resource hogging
- **Adversarial attacks**: malicious environments trying to trick the agent

**Safety mechanisms**:
- Sandboxed execution environments
- Action allowlist/denylist
- Safety critic model
- Human-in-the-loop approval

### 4. Environment fidelity

The gap between training and real environments (sim-to-real gap) severely affects policy transfer:

- Simulators may lack real-world noise and uncertainty
- API behavior changes over time
- User behavior is hard to model precisely

### 5. Scalability

Training agents requires large amounts of environment interaction:

- Each rollout needs real tool execution (running code, calling APIs, web requests)
- Tool execution latency is much higher than pure text generation
- Need thousands to tens of thousands of parallel environment instances

### 6. Evaluation difficulty

Evaluating agent capability is much harder than evaluating text generation:

- Need end-to-end task completion evaluation
- The same task may have multiple valid paths
- How to score partial completion?
- Multi-dimensional evaluation across efficiency, safety, etc.

## Code Example

### Simple agentic RL training loop (pseudocode)

```python
import torch
from typing import List, Dict, Tuple

class AgenticRLTrainer:
    """Agentic RL trainer: train an LLM to complete tasks in an environment"""

    def __init__(self, policy_model, ref_model, env, reward_fn,
                 lr=1e-6, kl_coeff=0.01, clip_eps=0.2):
        self.policy = policy_model        # current policy (LLM)
        self.ref_model = ref_model        # reference model (for KL constraint)
        self.env = env                     # interaction environment
        self.reward_fn = reward_fn         # reward function
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.kl_coeff = kl_coeff
        self.clip_eps = clip_eps

    def collect_rollouts(self, tasks: List[str], n_samples: int = 4
                        ) -> List[Dict]:
        """Collect multiple trajectories for each task"""
        rollouts = []
        for task in tasks:
            for _ in range(n_samples):
                trajectory = self._run_episode(task)
                rollouts.append(trajectory)
        return rollouts

    def _run_episode(self, task: str) -> Dict:
        """Run one complete interaction episode"""
        obs = self.env.reset(task)
        history = [{"role": "system", "content": "You are a helpful agent."},
                   {"role": "user", "content": task}]
        trajectory = {"task": task, "turns": [], "total_reward": 0.0}

        for step in range(self.max_steps):
            # 1. Policy generates action
            action, log_prob = self.policy.generate(history)

            # 2. Check if it's a tool call
            if self._is_tool_call(action):
                tool_name, tool_args = self._parse_tool_call(action)
                obs, done, info = self.env.step(tool_name, tool_args)
                history.append({"role": "assistant", "content": action})
                history.append({"role": "tool", "content": obs})
            else:
                # Plain text reply
                obs, done, info = action, True, {}
                history.append({"role": "assistant", "content": action})

            # 3. Record trajectory
            trajectory["turns"].append({
                "action": action,
                "log_prob": log_prob,
                "observation": obs,
            })

            if done:
                break

        # 4. Compute episode reward
        trajectory["total_reward"] = self.reward_fn(task, trajectory)
        return trajectory

    def compute_advantages(self, rollouts: List[Dict]) -> List[Dict]:
        """Compute advantages per trajectory (GRPO style: group normalization)"""
        # Group by task
        task_groups = {}
        for r in rollouts:
            task_groups.setdefault(r["task"], []).append(r)

        for task, group in task_groups.items():
            rewards = [r["total_reward"] for r in group]
            mean_r = sum(rewards) / len(rewards)
            std_r = (sum((r - mean_r)**2 for r in rewards) / len(rewards))**0.5
            std_r = max(std_r, 1e-8)

            for r in group:
                # Group-normalized advantage
                r["advantage"] = (r["total_reward"] - mean_r) / std_r

        return rollouts

    def update_policy(self, rollouts: List[Dict]):
        """PPO/GRPO-style policy update"""
        rollouts = self.compute_advantages(rollouts)

        total_loss = 0
        for rollout in rollouts:
            advantage = rollout["advantage"]

            for turn in rollout["turns"]:
                # Current policy log prob
                curr_log_prob = self.policy.log_prob(turn["action"])
                old_log_prob = turn["log_prob"]

                # Importance sampling ratio
                ratio = torch.exp(curr_log_prob - old_log_prob)

                # PPO clip
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio,
                                    1 - self.clip_eps,
                                    1 + self.clip_eps) * advantage
                policy_loss = -torch.min(surr1, surr2)

                # KL penalty
                ref_log_prob = self.ref_model.log_prob(turn["action"])
                kl_penalty = curr_log_prob - ref_log_prob

                total_loss += policy_loss + self.kl_coeff * kl_penalty

        # Gradient update
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

    def train(self, task_dataset, n_epochs=100, batch_size=32):
        """Main training loop"""
        for epoch in range(n_epochs):
            # 1. Sample tasks
            tasks = task_dataset.sample(batch_size)

            # 2. Collect rollouts
            rollouts = self.collect_rollouts(tasks)

            # 3. Policy update
            self.update_policy(rollouts)

            # 4. Evaluation
            if epoch % 10 == 0:
                eval_score = self.evaluate(task_dataset.eval_set)
                print(f"Epoch {epoch}: eval_score = {eval_score:.3f}")
```

### Reward function example

```python
def agentic_reward(task: str, trajectory: Dict) -> float:
    """Composite reward function"""
    # 1. Task completion reward
    task_reward = 1.0 if trajectory["task_completed"] else 0.0

    # 2. Efficiency reward (step penalty)
    num_steps = len(trajectory["turns"])
    efficiency_reward = -0.01 * num_steps

    # 3. Tool use correctness
    tool_calls = [t for t in trajectory["turns"] if t.get("is_tool_call")]
    valid_calls = sum(1 for t in tool_calls if t.get("tool_success"))
    tool_reward = valid_calls / max(len(tool_calls), 1)

    # 4. Safety penalty
    safety_violations = sum(1 for t in trajectory["turns"]
                           if t.get("safety_violation"))
    safety_penalty = -1.0 * safety_violations

    # Composite
    return (task_reward
            + 0.1 * efficiency_reward
            + 0.2 * tool_reward
            + safety_penalty)
```

## References

### Core papers

- DeepSeek-AI (2025). [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948). arXiv:2501.12948.
- Nakano et al. (2021). [WebGPT: Browser-Assisted Question-Answering with Human Feedback](https://arxiv.org/abs/2112.09332). arXiv:2112.09332.
- Schick et al. (2023). [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761). arXiv:2302.04761.
- Yao et al. (2022). [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629). arXiv:2210.03629.

### Frameworks and systems

- Xi et al. (2025). [AgentGym: Evolving Large Language Model-based Agents across Diverse Environments](https://arxiv.org/abs/2509.08755). arXiv:2509.08755. (ICLR 2026 Oral)
- AgentRL Team (2025). [AgentRL: Training Language Model Agents with Reinforcement Learning](https://arxiv.org/abs/2510.04206). arXiv:2510.04206.
- Agent-R1 Team (2025). [Agent-R1: Training Powerful LLM Agents with End-to-End Reinforcement Learning](https://arxiv.org/abs/2511.14460). arXiv:2511.14460.

### Surveys and guides

- NeurIPS 2025. A Practitioner's Guide to Multi-turn Agentic RL.
- HuggingFace (2026). [When LLMs Grow Hands and Feet: Agentic RL Systems](https://huggingface.co/blog/AmberLJC/agentic-rl-systems).

## Related Pages

- [[tool-use-rl]] -- RL for tool use and API calls
- [[multi-step-reasoning-rl]] -- RL for multi-step reasoning
- [[environment-design]] -- environment design for agentic RL
- [[rl-training-frameworks]] -- training infrastructure (veRL, OpenRLHF, etc.)
- [[ai-agent-overview]] -- AI agent architecture
- [[rlhf-overview]] -- RLHF overview (for comparison)
- [[grpo]] -- GRPO algorithm (used by DeepSeek-R1)
- [[ppo-for-llm]] -- PPO applied to LLM training
- [[reward-modeling]] -- reward modeling (ORM/PRM)
