---
title: "RL for Tool Use and API Calling"
category: agentic-rl
tags: [tool-use, rl, retool, code-interpreter, api-calling, toolformer, gorilla, function-calling]
created: 2026-04-13
updated: 2026-05-07
status: mature
---

# RL for Tool Use and API Calling

> [!abstract]+ TL;DR
> A core direction within [[agentic-rl-overview|Agentic RL]] — training LLMs to decide **when** to invoke tools, **which** tool, **how** to format arguments, and **how to interpret** results, all via RL with execution feedback rather than expert SFT. Landmark system: **ReTool** (2025) — RL-trained 32B reaches **72.5% AIME**, surpassing OpenAI o1-preview by 27.9 pp; trained via [[rl-training-frameworks#veRL|veRL]] + PPO with two-stage cold-start SFT then tool-enhanced RL. Emergent behavior: code self-correction (the "aha moment" of adaptive tool use). Pure RL from scratch (no SFT) shown viable in EMNLP 2025 Findings.

## Formal MDP for Tool Use

**State**: $s_t = (\text{task}, h_{1:t-1}, \text{tool\_results}_{1:t-1})$

**Action space** (hybrid):
$$a_t \in \mathcal{A}_{\text{text}} \cup \mathcal{A}_{\text{tool}} \cup \mathcal{A}_{\text{special}}$$

- $\mathcal{A}_{\text{text}}$: generate natural language
- $\mathcal{A}_{\text{tool}} = \{(\text{tool\_name}, \text{args})\}$: call a tool
- $\mathcal{A}_{\text{special}} = \{\text{submit}, \text{give\_up}\}$: terminal actions

**Transition**: for text actions, state appends the text; for tool actions, environment executes the tool and appends the result.

```
Tool-Use MDP Flow:
  State s_t --> LLM Policy --> Decision: Text or Tool?
                                  |          |          |
                                  v          v          v
                              [Text]  [Tool Call]  [Submit]
                                         |
                                    Environment
                                    (Execute)
                                         |
                                    Observation
                                         |
                              s_{t+1} = s_t + action + obs
                                         |
                                  Done? No -> next turn
                                        Yes -> compute R
```

## Training Methods

### Toolformer: Self-Supervised Tool Annotation

[arXiv:2302.04761](https://arxiv.org/abs/2302.04761) (Meta, 2023). The model decides where to insert tool calls by comparing perplexity with and without the tool result. SFT-based (not RL), but foundational. Later work (EMNLP 2025) showed the same capability can be learned purely through RL.

### RLEF: Execution Feedback as Reward

Uses actual tool execution results as RL reward signals. Binary (pass/fail), continuous (partial correctness), or multi-dimensional (correctness + efficiency + safety). Zero annotation cost, fully objective.

### Process Reward for Tool Selection

Intermediate rewards for each tool-use decision:
- Correct tool chosen: +0.3
- Valid argument syntax: +0.1
- Unnecessary tool use: -0.1
- Should have used tool but didn't: -0.2

### ReTool (2025): Landmark System

[arXiv:2504.11536](https://arxiv.org/abs/2504.11536). Frames "call Python interpreter vs. keep thinking" as an explicit RL decision.

**Two-stage training**:
1. Cold-start SFT on tool-use demonstrations
2. Tool-enhanced RL with real-time code execution (veRL + PPO)

**Key results**: ReTool-32B achieves 67% on AIME (400 steps) vs. text-only RL 40% (1080 steps). Final: 72.5%, surpassing o1-preview by 27.9%.

**Emergent behavior**: Code self-correction -- model runs code, discovers bug, reflects, fixes, and re-executes. This "aha moment" was never explicitly demonstrated in training data.

## Reward Design

| Reward Component | Formula | Purpose |
|-----------------|---------|---------|
| Task completion | $R = \mathbb{1}[\text{correct}]$ | Primary signal |
| Tool efficiency | $R = -\alpha N_{\text{calls}} - \beta N_{\text{steps}}$ | Fewer calls = better |
| Tool correctness | $R = \text{valid\_calls} / \text{total\_calls}$ | Penalize format errors |
| Safety | $R = -\gamma \cdot \mathbb{1}[\text{unsafe}]$ | Prevent dangerous actions |

**Pitfalls**: Reward hacking (finding shortcuts), over-penalizing tool use (model avoids tools entirely), format overfitting (perfect syntax, meaningless content), sparse reward deadlock (nothing to learn from).

## Representative Systems

- **WebGPT** (OpenAI, 2021): GPT-3 with web browsing actions, trained via RLHF
- **Gorilla** (UC Berkeley, NeurIPS 2024): Correct API call generation from documentation
- **ToolLLM**: 16,000+ real-world APIs with DFSDT reasoning strategy
- **API-Bank**: 314 tool APIs benchmark across three evaluation levels

## Code Example

```python
class ToolUseRLTrainer:
    def collect_rollout(self, task, max_turns=20):
        obs = self.env.reset(task)
        messages = [{"role": "system", "content": TOOL_PROMPT},
                    {"role": "user", "content": obs}]
        trajectory = {"task": task, "turns": []}

        for _ in range(max_turns):
            action, log_probs = self.policy.generate(messages)
            obs, done, info = self.env.step(action)
            trajectory["turns"].append({
                "action": action, "log_probs": log_probs, "info": info
            })
            messages.append({"role": "assistant", "content": action})
            if obs:
                messages.append({"role": "tool", "content": obs})
            if done:
                break
        return trajectory

    def compute_reward(self, task, trajectory):
        task_reward = 1.0 if check_answer(trajectory) else 0.0
        tool_turns = [t for t in trajectory["turns"]
                      if is_tool_call(t["action"])]
        tool_quality = (sum(t["info"].get("tool_success", 0)
                       for t in tool_turns) / max(len(tool_turns), 1))
        efficiency = max(0, 1.0 - 0.02 * len(trajectory["turns"]))
        return 1.0 * task_reward + 0.2 * tool_quality + 0.1 * efficiency
```

## Challenges

1. **Hallucinated tool calls**: Non-existent tools, wrong argument types, fabricated outputs. Mitigate with constrained decoding and negative reward for invalid calls.
2. **Action space explosion**: Hundreds of API endpoints with diverse argument schemas. Mitigate with hierarchical action spaces and retrieval-augmented tool selection.
3. **Credit assignment**: In multi-step tool chains, which call was responsible for success/failure? Dense process rewards and turn-level critics help.
4. **Execution latency**: Tool calls (especially web search: 1-10s) bottleneck rollout collection. Mitigate with async parallel rollouts and result caching.
5. **Safety**: Code injection, resource exhaustion, data leakage. Requires sandboxing with network disabled, memory limits, and forbidden pattern detection.

## References

- Schick et al. (2023). [Toolformer](https://arxiv.org/abs/2302.04761). arXiv:2302.04761.
- Feng et al. (2025). [ReTool](https://arxiv.org/abs/2504.11536). arXiv:2504.11536.
- Nakano et al. (2021). [WebGPT](https://arxiv.org/abs/2112.09332). arXiv:2112.09332.
- Patil et al. (2024). [Gorilla](https://arxiv.org/abs/2305.15334). NeurIPS 2024.
- Qin et al. (2023). [ToolLLM](https://arxiv.org/abs/2307.16789). arXiv:2307.16789.
- Li et al. (2023). [API-Bank](https://arxiv.org/abs/2304.08244). arXiv:2304.08244.

## Related Pages

- [[agentic-rl-overview]] -- Agentic RL landscape
- [[environment-design]] -- Sandbox and execution environment design
- [[tool-use]] -- Tool use from the agent architecture perspective
- [[multi-step-reasoning-rl]] -- Combining reasoning with tool use
- [[rl-training-frameworks]] -- RL training frameworks (veRL, etc.)
- [[grpo]] -- GRPO algorithm
- [[ppo-for-llm]] -- PPO algorithm
