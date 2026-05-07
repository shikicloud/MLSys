---
title: "Environment Design for Agentic RL"
category: agentic-rl
tags: [environment, sandbox, openreward, ares, agent-training, simulation, swe-bench, webarena, curriculum]
created: 2026-04-13
updated: 2026-05-07
status: mature
---

# Environment Design for Agentic RL

> [!abstract]+ TL;DR
> Environment design determines what an agentic-RL agent can learn, how well, and whether it generalizes — must balance **high fidelity** (close to real scenarios), **high efficiency** (massively parallelizable), **safety** (isolate dangerous actions), and **diversity** (many tasks/scenarios). **Brain vs. body** architecture decouples the LLM policy from sandbox execution. 2025–2026 brought infrastructure platforms: **OpenReward** (330+ environments via API, ORS standard extending [[mcp-protocol|MCP]] for RL), **ARES** (Martian, RL-first for coding), **Daytona** (sub-90ms sandbox creation). Environment scaling law mirrors model scaling law — bigger models need more diverse training environments.

### Brain vs. Body Architecture

Modern agentic RL decouples the system:
- **Brain** (LLM policy): runs on GPU clusters, generates actions, updates weights
- **Body** (execution environment): runs on separate infrastructure, executes actions in sandboxes, returns observations

Communication via gRPC/REST/WebSocket. Container-based isolation (Kubernetes). Sub-100ms sandbox creation (Daytona achieves sub-90ms).

## Environment Types

### Text-Based Environments

**Code execution**: Agent generates code -> sandbox executes -> returns output/errors. Deterministic, verifiable, safe. Latency: 100ms-30s.

**Web browsing**: Agent issues browser actions (search, click, scroll) -> browser executes -> returns page content. Stateful, non-deterministic. Latency: 1-10s.

**Bash/CLI**: Agent issues shell commands -> execution -> returns stdout/stderr. Powerful but dangerous; requires strict permission controls.

### Sandboxed Execution Environments

| Isolation Level | Technology | Security | Performance | Creation Time |
|----------------|------------|----------|-------------|---------------|
| Process | Process sandbox | Low | Highest | <10ms |
| Container | Docker/containerd | Medium | High | 50-100ms |
| VM | Firecracker/gVisor | High | Medium | 100-500ms |
| Full isolation | Physical machine | Highest | Highest | Minutes |

### Simulated Environments

**LLM-simulated**: LLMs generate tool responses. Scalable and cheap but may lack realism.

**Hybrid**: Real execution for safe/cheap actions (code, math), LLM simulation for risky/expensive ones (APIs, web), human review for high-stakes actions.

### Real-World Environments

Direct interaction with real systems. Requires layered safety: input filtering -> permission control -> operation auditing -> human oversight -> emergency stop.

## Key Design Principles

### 1. Fidelity

How close to the real world. Trade-off: low-fidelity = fast/cheap but sim-to-real gap; high-fidelity = slow/expensive but directly applicable.

**Practical strategy**: Start with low-fidelity for exploration, switch to high-fidelity for fine-tuning, evaluate in real environments.

### 2. Safety

Layered protection: static rule checking (blocked commands) -> dynamic risk assessment (safety model) -> rate limiting -> execution monitoring with rollback.

### 3. Scalability

Large-scale RL training needs thousands of parallel environment instances. Key: sub-100ms sandbox creation, async rollout workers, environment pools with pre-warming.

### 4. Observability

What the agent can "see" determines what it can learn. Options: full state (highest info, context window limit), summary state (compressed), incremental (only changes), multimodal (text + screenshots + structured data).

## Representative Environments

| Environment | Type | Tasks | Action Space | Reward | Fidelity |
|------------|------|-------|--------------|--------|----------|
| SWE-bench | Code | 2,294 | Code edits + commands | Test pass/fail | High |
| WebArena | Web | 812 | Browser actions | Function match | High |
| InterCode | Code | 3,898 | Code execution | Output match | High |
| MINT | Multi-tool | 586 | Mixed | Task completion | Med-High |
| OSWorld | OS | 369 | GUI + CLI | State match | Highest |

**SWE-bench** (Princeton/CMU, 2024): Real GitHub issues. Agent must produce a code patch that passes the project's test suite without breaking existing tests.

**WebArena** (CMU, 2024): Self-hosted realistic websites (e-commerce, forums, GitLab, maps). Agent navigates and completes tasks like "find blue sneakers under $50 and checkout with code SAVE10."

**InterCode** (Princeton, 2024): Interactive code execution (Python, SQL, Bash). Multi-turn: write code, observe output, debug, retry.

**OSWorld** (2024): Full Ubuntu desktop. Tasks span file management, application use, system configuration. Actions include keyboard, mouse, and terminal commands.

## Environment Synthesis (2025-2026)

Manual environment creation is expensive and doesn't scale. Automated synthesis is key.

**Agent World Model (AWM)** (Snowflake, 2026): Fully synthetic pipeline generating 1,000 executable SQL-backed tool-use environments. LLM generates environment specs, tool implementations, test data, and tasks.

**ScaleEnv** (Feb 2026): Constructs fully interactive environments from scratch. Zero RL training on Qwen-3 showed substantial OOD gains, proving synthetic environments can produce generalizable agents.

**LLM-in-Sandbox** (Jan 2026): Investigates whether a code sandbox alone can elicit general agentic intelligence. Strong models achieve up to 15.5% gains without additional training -- the code sandbox is the most universal "meta-tool."

## Reward Signal Design

**Task-specific rewards**:
- Code: fraction of tests passed (0.0 to 1.0)
- Web: weighted combination of URL match, content similarity, form state match
- SQL: Jaccard similarity between predicted and ground truth result sets

**Intermediate rewards**: Per-turn credit for sub-goal progress. Must be carefully designed to avoid reward hacking (e.g., "reward per tool call" leads to gratuitous tool use).

**Safety penalties**: Negative rewards for restricted resource access (-1.0), destructive actions (-2.0), information leakage (-1.5), excessive resource usage (-0.5).

## Infrastructure Platforms

### OpenReward (General Reasoning, 2026)

330+ environments, 4.5M+ unique tasks. Built on **Open Reward Standard (ORS)**: extends Anthropic's [[mcp-protocol|MCP]] with RL primitives (episodes, reward signals, curriculum management). ORS is to RL environments what MCP is to tool integration.

### ARES (Martian)

Open-source, RL-first framework for coding agents. Gym-like interface, massively parallel async rollouts, tens of thousands of verifiable coding tasks.

### Daytona

Purpose-built sandbox infrastructure. Sub-90ms creation, complete isolation, Kubernetes-native, multi-language support.

## Code Example: Gym-Like Environment

```python
from abc import ABC, abstractmethod
from typing import Tuple, Dict

class AgentEnvironment(ABC):
    """Base class for LLM agent RL environments"""

    def __init__(self, config):
        self.config = config
        self.turn_count = 0
        self.done = False

    @abstractmethod
    def reset(self, task: str) -> str:
        """Reset environment, return initial observation"""
        pass

    @abstractmethod
    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        """Execute action -> (observation, reward, done, info)"""
        pass

    @abstractmethod
    def compute_reward(self) -> float:
        """Compute episode-level reward"""
        pass


class CodeExecutionEnv(AgentEnvironment):
    def reset(self, task: str) -> str:
        self.sandbox = create_sandbox(memory="512m", network=False)
        self.sandbox.execute(f"git clone {task['repo']} /workspace")
        return f"Issue: {task['issue']}\nRepo cloned to /workspace."

    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        self.turn_count += 1
        if "<submit>" in action:
            self.done = True
            return "Submitted.", self.compute_reward(), True, {}
        result = self.sandbox.execute(action, timeout=30)
        obs = f"stdout: {result['stdout']}\nstderr: {result['stderr']}"
        return obs, 0.0, False, {}

    def compute_reward(self) -> float:
        result = self.sandbox.execute("pytest", timeout=120)
        passed, total = parse_test_results(result["stdout"])
        return passed / max(total, 1)
```

## Challenges

1. **Diversity-fidelity tradeoff**: Synthetic environments scale but may miss real-world complexity. Best practice: train on synthetic, fine-tune on real.
2. **Reward engineering**: Rewards must capture genuine progress without enabling hacking. Requires iterative design and adversarial testing.
3. **Environment leakage**: Agents overfit to specific environment implementations (e.g., "the 3rd search result is always best"). Counter with randomization and diverse implementations.
4. **Curriculum design**: Sequencing from simple to complex. Adaptive: promote when pass rate >80%, demote when <20%.
5. **Real execution cost**: Per-1M rollouts -- Python execution ~$1K, web search ~$10K, API calls $1K-$100K. Must balance training efficiency with budget.

## References

### Environments and Benchmarks
- Jimenez et al. (2024). [SWE-bench](https://arxiv.org/abs/2310.06770). arXiv:2310.06770.
- Zhou et al. (2024). [WebArena](https://arxiv.org/abs/2307.13854). arXiv:2307.13854.
- Yang et al. (2024). [InterCode](https://arxiv.org/abs/2306.14898). arXiv:2306.14898.
- Xie et al. (2024). [OSWorld](https://arxiv.org/abs/2404.07972). arXiv:2404.07972.

### Environment Synthesis
- AWM Team (2026). Agent World Model. Snowflake.
- ScaleEnv Team (2026). [ScaleEnv](https://arxiv.org/abs/2602.06820). arXiv:2602.06820.
- LLM-in-Sandbox Team (2026). [Code Sandbox for Agentic Intelligence](https://arxiv.org/html/2601.16206). arXiv:2601.16206.

### Infrastructure
- [OpenReward](https://openreward.ai/). General Reasoning (2026).
- [HuggingFace: When LLMs Grow Hands and Feet](https://huggingface.co/blog/AmberLJC/agentic-rl-systems).
- [Taxonomy of RL Environments](https://leehanchung.github.io/blogs/2026/03/21/rl-environments-for-llm-agents/).

## Related Pages

- [[agentic-rl-overview]] -- Agentic RL landscape
- [[tool-use-rl]] -- Tool-using agents that need environments
- [[multi-step-reasoning-rl]] -- Reasoning RL (code/math environments)
- [[rl-training-frameworks]] -- Training infrastructure (veRL, OpenRLHF, etc.)
- [[mcp-protocol]] -- MCP protocol (ORS foundation)
- [[ai-agent-overview]] -- AI agent architectures
