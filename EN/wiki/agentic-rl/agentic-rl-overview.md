---
title: "Agentic RL Overview"
category: agentic-rl
tags: [agentic-rl, multi-turn-rl, agent-training, agent-r1, agentrl, rlhf, grpo, ppo]
created: 2026-04-13
updated: 2026-04-14
status: mature
---

# Agentic RL Overview

## Overview

Agentic RL is a class of reinforcement learning methods specifically designed to train AI agents that interact with external environments and tools. Unlike traditional [[rlhf-overview|RLHF]] (which optimizes single-turn text quality), Agentic RL focuses on optimizing action sequences, tool use, and task completion over multi-turn interactions.

**Core formulation**: The LLM agent is a policy in an MDP where:
- **State**: conversation history + environment state (file system, web pages, databases, etc.)
- **Action**: text generation including natural language and structured tool calls
- **Transition**: environment executes actions and returns observations
- **Reward**: task completion signals, efficiency metrics, safety constraints

This represents a fundamental paradigm shift from "training models to say things" to "training models to do things."

### Why Agentic RL over SFT?

Supervised fine-tuning on expert trajectories has fundamental limitations for agent training:
1. **Distribution shift**: SFT agents degrade sharply in out-of-distribution states
2. **No exploration**: Imitation learning cannot discover strategies experts didn't demonstrate
3. **Error compounding**: Small per-step errors compound exponentially over multi-step tasks
4. **Poor generalization**: SFT agents struggle to adapt to new environment configurations

RL addresses these via trial-and-error learning -- agents learn to recover from mistakes, explore novel strategies, and optimize behavior over long horizons.

## Distinction from Traditional RLHF

| Dimension | Traditional RLHF | Agentic RL |
|-----------|-------------------|------------|
| **Objective** | Text quality (helpful, safe, honest) | Action sequences and task completion |
| **Turns** | Single (prompt -> response) | Multi-turn (tens to hundreds) |
| **Action space** | Natural language text | Text + tool calls + code execution |
| **Reward** | Human preferences (dense, subjective) | Task completion (sparse, verifiable) |
| **Reward source** | Reward model (RM) | Execution feedback (RLEF) |
| **Credit assignment** | Simple (entire response) | Complex (which action caused success?) |
| **Environment** | None | Requires sandboxed execution |

```
Traditional RLHF:
  User Prompt --> LLM --> Response --> Reward Model --> Score
       |                                   |            |
       └────────── Policy Update <─────────┘────────────┘

Agentic RL:
  Task --> LLM Agent --+--> Text Response
    ^                  +--> Tool Call --> Environment --> Observation
    |                  +--> Code Exec --> Sandbox --> Result
    |                  └--> ... (multi-turn loop) ...
    |                                        |
    |         Task Complete? --> Reward ──────┘
    └──── Policy Update <───────────────────────────────────
```

## Core Paradigm

### LLM as Policy

$$\pi_\theta(a_t | s_t) = \text{LLM}_\theta(\text{action} | \text{history}_t, \text{env\_state}_t)$$

Actions are autoregressively generated token sequences. A single "action" may span many tokens (e.g., a complete function call). Observations are injected into context as part of the next state.

### Reward Design

Agentic RL typically uses a composite reward:

$$R_{\text{total}} = R_{\text{outcome}} + \lambda_1 R_{\text{efficiency}} + \lambda_2 R_{\text{process}} + \lambda_3 R_{\text{safety}}$$

- **Outcome reward**: +1 for task success, 0 for failure
- **Efficiency reward**: penalty proportional to steps/tool calls used
- **Process reward**: intermediate credit for sub-goal completion
- **Safety penalty**: negative reward for dangerous actions

### Exploration Challenges

The exploration problem in Agentic RL is far harder than in traditional RL:
- **Vast action space**: all possible token sequences (effectively infinite)
- **Sparse rewards**: only task completion yields positive reward
- **Long horizons**: coherent decisions over dozens to hundreds of steps
- **Combinatorial explosion**: tool x argument x ordering combinations

Mitigation strategies: curriculum learning, SFT warmup, dense reward shaping, experience replay, hierarchical exploration.

## Key Research Directions

1. **[[tool-use-rl|Tool Use RL]]**: Learning when/how to call external tools via RL
2. **[[multi-step-reasoning-rl|Multi-Step Reasoning RL]]**: Chain-of-thought improvement via RL
3. **[[environment-design|Environment Design]]**: Building effective training environments
4. **Agent paradigms**: WebAgent, CodeAgent, SearchAgent, OSAgent, MultiAgent

## Representative Works

### DeepSeek-R1

[arXiv:2501.12948](https://arxiv.org/abs/2501.12948) (Jan 2025). Pure RL training on DeepSeek-V3-Base produces emergent CoT reasoning. Uses [[grpo|GRPO]]. Achieves 79.8% on AIME 2024 (comparable to OpenAI o1). Reports "aha moment" -- model spontaneously learns self-verification and backtracking.

### WebGPT

[arXiv:2112.09332](https://arxiv.org/abs/2112.09332) (OpenAI, 2021). GPT-3 with web browsing actions (search, click, scroll, quote). Trained via behavioral cloning + RLHF. One of the earliest LLM + environment + RL successes.

### Toolformer

[arXiv:2302.04761](https://arxiv.org/abs/2302.04761) (Meta, 2023). Self-supervised tool learning via perplexity-based annotation. Not RL, but established the foundation; later work (EMNLP 2025) showed the same capability can be learned purely via RL.

### ReAct

[arXiv:2210.03629](https://arxiv.org/abs/2210.03629) (2022). Alternating Thought-Action-Observation format. Defined the standard interaction pattern for modern LLM agents.

### RLEF

Reinforcement Learning from Execution Feedback. Uses actual tool execution results as reward signals (code tests pass/fail, API returns valid/invalid). Zero annotation cost, fully objective, naturally adapts to API changes.

## Major Frameworks (2025-2026)

- **AgentGym-RL** (ICLR 2026 Oral): Unified framework for long-horizon agent decisions. [arXiv:2509.08755](https://arxiv.org/abs/2509.08755)
- **AgentRL**: Fully async generation-training pipeline with cross-policy sampling and task advantage normalization. [arXiv:2510.04206](https://arxiv.org/abs/2510.04206)
- **Agent-R1**: Modular framework extending single-turn RL to multi-turn tasks. Supports PPO, GRPO, REINFORCE++. [arXiv:2511.14460](https://arxiv.org/abs/2511.14460)
- **MARTI** (ICLR 2026, Tsinghua): Multi-Agent Reinforced Training and Inference.
- **ProRL Agent** (NVIDIA, March 2026): "Rollout-as-a-Service" -- decouples rollout orchestration from training.

## Technical Challenges

1. **Credit assignment**: In a 50-step episode, which actions were critical? Approaches: turn-level rewards, PRMs, TD learning, GAE.
2. **Sparse rewards**: Most real tasks only reward at the end. Solutions: reward shaping, curiosity-driven exploration, hierarchical RL.
3. **Safety constraints**: Irreversible actions (file deletion, email sending), information leakage, resource abuse. Requires sandboxing, action whitelists, safety critics, human-in-the-loop.
4. **Environment fidelity**: Sim-to-real gap between training and deployment environments.
5. **Scalability**: Each rollout requires actual tool execution (slow); needs thousands of parallel environment instances.
6. **Evaluation**: End-to-end task assessment with multiple valid paths; multi-dimensional scoring.

## Code Example

```python
class AgenticRLTrainer:
    """Simplified agentic RL training loop"""

    def __init__(self, policy, ref_model, env, reward_fn):
        self.policy = policy
        self.ref_model = ref_model
        self.env = env
        self.reward_fn = reward_fn

    def collect_rollout(self, task):
        obs = self.env.reset(task)
        history = [{"role": "user", "content": task}]
        trajectory = {"task": task, "turns": []}

        for step in range(self.max_steps):
            action, log_prob = self.policy.generate(history)
            if self._is_tool_call(action):
                tool, args = self._parse_tool_call(action)
                obs, done, info = self.env.step(tool, args)
                history.append({"role": "assistant", "content": action})
                history.append({"role": "tool", "content": obs})
            else:
                done = True
                history.append({"role": "assistant", "content": action})
            trajectory["turns"].append({"action": action, "log_prob": log_prob})
            if done:
                break

        trajectory["reward"] = self.reward_fn(task, trajectory)
        return trajectory

    def grpo_update(self, tasks, n_samples=8):
        groups = {}
        for task in tasks:
            groups[task] = [self.collect_rollout(task) for _ in range(n_samples)]

        for task, rollouts in groups.items():
            rewards = [r["reward"] for r in rollouts]
            mean_r, std_r = mean(rewards), max(std(rewards), 1e-8)
            for r in rollouts:
                r["advantage"] = (r["reward"] - mean_r) / std_r

        # Policy gradient update with clipping + KL penalty
        self._update_weights(groups)
```

## References

- DeepSeek-AI (2025). [DeepSeek-R1](https://arxiv.org/abs/2501.12948). arXiv:2501.12948.
- Nakano et al. (2021). [WebGPT](https://arxiv.org/abs/2112.09332). arXiv:2112.09332.
- Schick et al. (2023). [Toolformer](https://arxiv.org/abs/2302.04761). arXiv:2302.04761.
- Yao et al. (2022). [ReAct](https://arxiv.org/abs/2210.03629). arXiv:2210.03629.
- NeurIPS 2025. A Practitioner's Guide to Multi-turn Agentic RL.
- HuggingFace (2026). [When LLMs Grow Hands and Feet](https://huggingface.co/blog/AmberLJC/agentic-rl-systems).

## Related Pages

- [[tool-use-rl]] -- RL for tool use and API calling
- [[multi-step-reasoning-rl]] -- RL for multi-step reasoning
- [[environment-design]] -- Environment design for agent training
- [[rl-training-frameworks]] -- Training infrastructure (veRL, OpenRLHF, etc.)
- [[ai-agent-overview]] -- AI agent architectures
- [[rlhf-overview]] -- RLHF overview (comparison)
- [[grpo]] -- GRPO algorithm (used by DeepSeek-R1)
- [[ppo-for-llm]] -- PPO for LLM training
- [[reward-modeling]] -- Reward modeling (ORM/PRM)
