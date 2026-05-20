---
title: "Environment Design for Agentic RL"
category: agentic-rl
tags: [environment, sandbox, openreward, ares, agent-training, simulation, swe-bench, webarena, curriculum]
created: 2026-04-13
updated: 2026-05-20
status: mature
---

# Environment Design for Agentic RL

> [!abstract]+ TL;DR
> Environment design determines what agentic RL can learn, how well, and whether it generalizes — must balance **high fidelity** (close to real), **high efficiency** (massively parallelizable), **safety** (isolate dangerous operations), and **diversity** (many tasks and scenarios). **Brain vs. body** architecture decouples LLM policy from sandbox execution. 2025-2026 brings infrastructure platforms: **OpenReward** (330+ environment APIs, ORS extends [[mcp-protocol|MCP]] for RL), **ARES** (Martian, RL-first coding agent), **Daytona** (sub-90ms sandbox creation). Environment scaling law mirrors model scaling law — larger models need more diverse training environments.

## Overview

Environment design is the cornerstone of [[agentic-rl-overview|agentic RL]] — environments determine what the agent learns, how well, and whether it generalizes to the real world. A good training environment must simultaneously satisfy: **high fidelity** (close to real scenarios), **high efficiency** (massively parallelizable), **safety** (isolate dangerous operations), and **diversity** (covering many tasks and scenarios).

### Architecture: Brain vs Body

Modern agentic RL systems decouple into two parts:

```
┌──────────────────────────────────────────────────────────┐
│                  Agentic RL Architecture                  │
│                                                          │
│  ┌─────────────────┐          ┌─────────────────────┐    │
│  │     Brain       │          │       Body          │    │
│  │   LLM policy    │  action  │  Execution env      │    │
│  │                 │ ──────> │                     │    │
│  │  GPU cluster    │          │  Separate infra     │    │
│  │  generate action│ <────── │  Sandboxed execution│    │
│  │  policy update  │  obs     │  Returns result     │    │
│  │                 │          │                     │    │
│  │  ┌───────────┐ │          │  ┌───────────────┐  │    │
│  │  │ Policy π_θ│ │          │  │ Tool Execution│  │    │
│  │  │ Value V_φ │ │          │  │ Code Sandbox  │  │    │
│  │  │ Ref Model │ │          │  │ Web Browser   │  │    │
│  │  └───────────┘ │          │  │ File System   │  │    │
│  │                 │          │  │ Database      │  │    │
│  └─────────────────┘          │  └───────────────┘  │    │
│                               └─────────────────────┘    │
│                                                          │
│  Communication: gRPC / REST / WebSocket                   │
│  Isolation: Container (Docker/K8s) / VM / process sandbox│
│                                                          │
└──────────────────────────────────────────────────────────┘
```

**Key metrics**:
- Sandbox creation time: <100ms (Daytona achieves <90ms)
- Each rollout gets an independently isolated environment
- Containerized deployment (Kubernetes-orchestrated)
- Auto-scaling

## Environment Types

### 1. Text-based environments

The simplest and most common agent environment; the agent interacts via text.

| Subtype | Interaction | Characteristics |
|---------|-------------|-----------------|
| **Code execution** | Agent generates code -> sandbox executes -> returns output | Deterministic, verifiable, latency 100ms-30s |
| **Web browsing** | Agent issues browse action -> browser executes -> returns page | Stateful, non-deterministic, latency 1-10s |
| **CLI/Bash** | Agent issues shell command -> execution -> returns stdout/stderr | Powerful but dangerous, strict permission control needed |

### 2. Sandboxed execution environments

Isolated environments running in containers or VMs:

**Isolation levels**:

| Isolation level | Technology | Safety | Performance | Creation time |
|-----------------|------------|--------|-------------|----------------|
| Process | Process sandbox | Low | Highest | <10ms |
| Container | Docker/containerd | Medium | High | 50-100ms |
| VM | Firecracker/gVisor | High | Medium | 100-500ms |
| Full isolation | Bare metal | Highest | Highest | Minutes |

### 3. Simulated environments

Use LLMs or rule engines to simulate real-environment behavior:

- **LLM simulation**: LLM generates tool responses. Scalable but possibly unrealistic.
- **Hybrid simulation**: safe/cheap actions executed for real, dangerous/expensive ones simulated by LLM.

### 4. Real-world environments (with safety constraints)

Agent interacts directly with real systems; needs multi-layer safety: input filtering (block dangerous operations) -> permission control (least-privilege) -> operation auditing (log + anomaly detection + rollback) -> human oversight (confirmation for high-risk operations).

## Key Design Principles

### 1. Fidelity

How close the environment is to the real world. The core tradeoff: low-fidelity environments are fast and cheap but have a sim-to-real gap; high-fidelity ones are realistic but expensive.

**Practical strategy**: initial exploration on low-fidelity environments (fast trajectory generation) -> fine-tuning on high-fidelity environments (better policy quality) -> final evaluation in real environments.

### 2. Safety

Prevent the agent from executing harmful operations. Safety should be layered:
1. **Static rule checks**: block known dangerous patterns (`rm -rf /`, `mkfs`, fork bomb, etc.)
2. **Dynamic risk assessment**: a safety model scores action risk; high-risk actions are blocked
3. **Rate limiting**: prevent API abuse and resource exhaustion
4. **Execution monitoring**: real-time anomaly detection, supports rollback

### 3. Scalability

Large-scale RL training needs thousands to tens of thousands of parallel environment instances. Key metrics: sandbox creation <100ms, destruction <50ms, concurrency 10,000+.

Typical architecture: N Rollout Workers, each managing an Environment Pool (~1000 sandboxes), maximizing GPU utilization via async rollouts.

### 4. Observability

What the agent can "see" determines what it can learn. Observation space design options:

| Observation type | Content | Information | Challenge |
|------------------|---------|-------------|-----------|
| Full state | All environment info | Highest | Context window limit |
| Summary state | Key info summary | Medium | Information loss |
| Incremental | Only latest changes | Low | Requires model memory |
| Multi-modal | Text + screenshot + structured data | High | Processing complexity |

When environment state is too large, compression is needed (e.g., 100KB HTML -> 5KB key elements -> 2KB structured JSON).

## Representative Environments

### SWE-bench: software engineering tasks

**Source**: Princeton / CMU (2024). 2,294 real GitHub Issues; the agent must generate code patches that make the test suite pass. Subsets: SWE-bench Lite (300), SWE-bench Verified (500). Each task requires a full repo clone, dependency installation, and an isolated file system. Reward = test pass rate.

### WebArena: web interaction

**Source**: CMU (2024). Self-hosted real websites (e-commerce, forums, GitLab, maps). 812 tasks; action space includes click/type/scroll/goto/submit. Evaluation: functional correctness + URL match + content match.

### InterCode: code execution

**Source**: Princeton (2024). 3,898 interactive code execution tasks (Python/SQL/Bash). Supports multi-turn interaction: write code -> execute -> observe -> modify -> retry. Real execution, automatic scoring.

### MINT: multi-turn interaction

**Source**: (2024). 586 multi-turn tool-use tasks (Python + Web + Bash + knowledge bases). Evaluates tool selection accuracy, interaction efficiency, error recovery, and task completion rate.

### OSWorld: operating system tasks

**Source**: (2024). 369 tasks operating on a full Ubuntu desktop (GUI + CLI). Covers file management, application operation, system configuration, multi-app coordination. Action space includes keyboard, mouse, and terminal commands.

### Environment comparison

| Environment | Type | Tasks | Action space | Reward type | Fidelity |
|-------------|------|-------|--------------|-------------|----------|
| SWE-bench | Code | 2,294 | Code edit + commands | Tests pass | High |
| WebArena | Web | 812 | Browser actions | Functional match | High |
| InterCode | Code | 3,898 | Code execution | Output match | High |
| MINT | Multi-tool | 586 | Mixed | Task completion | Med-high |
| OSWorld | OS | 369 | GUI + CLI | State match | Highest |

## Environment Synthesis (2025-2026)

Hand-building environments is expensive and hard to scale. Automated environment synthesis is the key to overcoming this bottleneck.

### Agent World Model (AWM)

**Source**: Snowflake (2026)

```
AWM synthesis pipeline:

Step 1: Environment spec generation
  LLM → generates environment description (tools, data model, task definitions)

Step 2: Tool implementation
  LLM → generates tool code per spec (Python functions + SQL backend)

Step 3: Data population
  LLM → generates realistic test data

Step 4: Task generation
  LLM → generates environment-based tasks + ground-truth answers

Step 5: Verification
  Auto-run ground-truth solutions to verify environment usability

Result: 1,000 executable SQL-backend tool-use environments
```

### ScaleEnv

**Source**: (February 2026)

ScaleEnv builds fully interactive environments from scratch:

```
Innovations:
  - Doesn't rely on existing platforms/websites
  - Environment spec and implementation fully LLM-generated
  - Supports Zero RL training (no SFT warm start needed)

Results:
  - Zero RL training on Qwen-3
  - Significant out-of-distribution (OOD) performance gains
  - Proves synthetic environments can produce generalizable capabilities
```

### LLM-in-Sandbox

**Source**: (January 2026)

Investigates a core question: can a code sandbox alone elicit general agentic intelligence?

```
Experimental design:
  - Give the LLM a Python sandbox (no other tools)
  - Observe whether the model can spontaneously learn to:
    1. Simulate other tools via code (write code to implement search)
    2. Reason via code verification (write code to verify math proofs)
    3. Manage information via code (write code to organize/retrieve data)

Results:
  - Strong models (GPT-4 level) gain 15.5% with no extra training
  - Code sandbox is the most general "meta-tool"
```

## Reward Signal Design

### Task-specific rewards

| Environment type | Reward computation | Example |
|------------------|---------------------|---------|
| **Code** | Test pass rate = passed / total | 8/10 tests pass -> R = 0.8 |
| **Web** | Weighted: 0.4 URL match + 0.3 content sim + 0.3 form state | reach target page + correct form -> R = 0.9 |
| **SQL** | Result set Jaccard similarity | 80% rows match -> R = 0.8 |

### Intermediate rewards

To overcome reward sparsity, intermediate rewards can be given per step:

```
Intermediate reward design for a code debugging task:

Turn 1: read error message → +0.1 (info gathering)
Turn 2: locate error line → +0.2 (problem localization)
Turn 3: analyze cause → +0.1 (problem analysis)
Turn 4: write fix code → +0.0 (not yet verified)
Turn 5: tests pass → +0.6 (task completed)

Total reward: 1.0
```

**Risks of reward shaping**:

Poorly designed intermediate rewards can cause reward hacking:

```
Bad intermediate reward: +0.1 per tool call
  → Model keeps calling tools to accumulate reward, even when unnecessary

Good intermediate reward: +0.1 for successfully solving sub-problems with a tool
  → Model learns purposeful tool use
```

### Safety penalties

Negative rewards for dangerous behaviors: unauthorized access (-1.0), destructive ops (-2.0), information leakage (-1.5), resource abuse (-0.5).

## Infrastructure Platforms

### OpenReward (General Reasoning, 2026)

```
OpenReward platform architecture:

┌─────────────────────────────────────────────┐
│              OpenReward Platform              │
│                                              │
│  ┌──────────────────────────────────────┐    │
│  │        Open Reward Standard (ORS)     │    │
│  │  Based on MCP + RL primitive extensions│   │
│  │  - Episode management                  │   │
│  │  - Reward signal interface             │   │
│  │  - Curriculum management               │   │
│  └──────────────────────────────────────┘    │
│                     │                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  Code     │  │  Web      │  │ Database │   │
│  │  envs 130+│  │  envs 80+ │  │ envs 50+ │   │
│  └──────────┘  └──────────┘  └──────────┘   │
│       ⋮              ⋮             ⋮          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  API     │  │  OS       │  │ Multi-   │   │
│  │  envs 40+│  │  envs 20+ │  │ tool 10+ │   │
│  └──────────┘  └──────────┘  └──────────┘   │
│                                              │
│  Total: 330+ environments, 4.5M+ unique tasks │
│  Auto-scaling sandbox compute                 │
│                                              │
└─────────────────────────────────────────────┘
```

**ORS and MCP**:
- [[mcp-protocol|MCP]] (Model Context Protocol): Anthropic's tool integration standard
- ORS extends MCP with RL-specific primitives:
  - `episode.start()` / `episode.end()`: episode management
  - `reward.signal(value)`: reward signal
  - `curriculum.next()`: curriculum management
  - `checkpoint.save()` / `checkpoint.restore()`: state management

### ARES (Martian)

Open-source, RL-first coding agent framework:
- Gym-like interface
- Large-scale parallel async rollout
- Tens of thousands of verifiable coding tasks

### Daytona

Sandbox infrastructure built for safe AI code execution:
- Sub-90ms sandbox creation
- Full isolation (independent env per execution)
- Kubernetes-native
- Multi-language support (Python, JavaScript, Go, etc.)

## Code Example

### Simple Gym-like environment interface

```python
from abc import ABC, abstractmethod
from typing import Dict, Tuple, List
from dataclasses import dataclass, field

@dataclass
class EnvConfig:
    """Environment configuration"""
    max_turns: int = 50
    timeout_per_action: int = 30
    sandbox_memory: str = "512m"
    sandbox_network: bool = False
    tools: List[str] = field(default_factory=lambda: ["python", "bash"])


class AgentEnvironment(ABC):
    """LLM agent RL environment base class (Gym-like interface)"""

    def __init__(self, config: EnvConfig):
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
    """Code execution environment (SWE-bench style)"""

    def reset(self, task: str) -> str:
        self.turn_count = 0
        self.done = False
        self.task_info = json.loads(task)
        self.sandbox = create_sandbox(
            memory=self.config.sandbox_memory,
            network=self.config.sandbox_network,
        )
        self.sandbox.execute(f"git clone {self.task_info['repo_url']} /workspace")
        return f"Issue: {self.task_info['issue_description']}\n" \
               f"Repo cloned to /workspace. Fix and run tests."

    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        self.turn_count += 1
        if "<submit>" in action or self.turn_count >= self.config.max_turns:
            self.done = True
            return "Submitted.", self.compute_reward(), True, {}

        result = self.sandbox.execute(action, timeout=self.config.timeout_per_action)
        obs = f"stdout: {result['stdout']}\nstderr: {result['stderr']}"
        return obs, 0.0, False, {"turn": self.turn_count}

    def compute_reward(self) -> float:
        result = self.sandbox.execute("cd /workspace && pytest", timeout=120)
        if result["exit_code"] == 0:
            return 1.0
        passed, total = self._parse_test_results(result["stdout"])
        return passed / max(total, 1)


# Environment factory + training integration
def create_environment(env_type: str, config: EnvConfig) -> AgentEnvironment:
    envs = {"code": CodeExecutionEnv, "web": WebBrowsingEnv}
    return envs[env_type](config)

async def rl_training_with_envs(policy, env_configs, dataset):
    for epoch in range(100):
        rollouts = []
        for task in dataset.sample(batch_size=32):
            env = create_environment(task["env_type"], env_configs[task["env_type"]])
            try:
                obs = env.reset(json.dumps(task))
                trajectory = {"task": task, "turns": []}
                while not env.done:
                    action = await policy.generate_async(obs)
                    obs, reward, done, info = env.step(action)
                    trajectory["turns"].append({"action": action, "obs": obs})
                trajectory["final_reward"] = env.compute_reward()
                rollouts.append(trajectory)
            finally:
                env.close()
        policy.update(rollouts)
```

## Challenges

### 1. Diversity-fidelity tradeoff

```
             Synthetic env             Real env
     ┌────────────────────┐      ┌────────────────────┐
     │ Pros:               │      │ Pros:               │
     │ - Mass produced     │      │ - Fully realistic    │
     │ - Cheap             │      │ - No sim-to-real gap │
     │ - Safe              │      │                     │
     │                     │      │ Cons:               │
     │ Cons:               │      │ - Expensive         │
     │ - May be unrealistic│      │ - Hard to parallelize│
     │ - sim-to-real gap   │      │ - Safety risks       │
     │ - Overfit risk      │      │ - Hard to scale      │
     └────────────────────┘      └────────────────────┘

Best practice: start training on synthetic envs, gradually introduce real envs for fine-tuning
```

### 2. Reward engineering

Designing rewards that accurately reflect task goals without being exploited is an ongoing challenge:

- Too simple a reward leads to reward hacking
- Too complex a reward introduces human bias
- Requires continuous iteration and adversarial testing

### 3. Environment leakage

Agents may overfit to environment-specific implementation details:

```
Overfit example:
  Model learns "in WebArena, the 3rd search result is always most relevant"
  → This heuristic doesn't hold for real search engines

Countermeasures:
  - Environment diversity (multiple different implementations)
  - Randomization (random environment parameter variations)
  - Out-of-distribution evaluation (test on unseen environments)
```

### 4. Curriculum design

How to order environments from simple to complex:

```
Curriculum learning example (code tasks):

Level 1: single-file bug fix (1-line change)
Level 2: single-file feature add (10-20 lines)
Level 3: multi-file bug fix (need to understand code structure)
Level 4: cross-module feature (need to understand system architecture)
Level 5: complete GitHub Issue (may need 100+ lines of changes)

Adaptive curriculum:
  if pass_rate at current level > 80%:
      advance to next level
  elif pass_rate at current level < 20%:
      step back to previous level
  else:
      stay at current level
```

### 5. Real execution cost

Cost of running real tools at RL scale:

| Operation | Per call | Per 1M rollouts |
|-----------|----------|-----------------|
| Python execution | ~$0.001 | ~$1,000 |
| Web search | ~$0.01 | ~$10,000 |
| API call | ~$0.001-0.10 | $1,000 - $100,000 |
| Database query | ~$0.001 | ~$1,000 |
| GPU code execution | ~$0.01-1.00 | $10,000 - $1,000,000 |

Tradeoff between training efficiency and cost is required.

## References

### Environments and benchmarks

- Jimenez et al. (2024). [SWE-bench: Can Language Models Resolve Real-World GitHub Issues?](https://arxiv.org/abs/2310.06770). arXiv:2310.06770.
- Zhou et al. (2024). [WebArena: A Realistic Web Environment for Building Autonomous Agents](https://arxiv.org/abs/2307.13854). arXiv:2307.13854.
- Yang et al. (2024). [InterCode: Standardizing and Benchmarking Interactive Coding with Execution Feedback](https://arxiv.org/abs/2306.14898). arXiv:2306.14898.
- Xie et al. (2024). [OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments](https://arxiv.org/abs/2404.07972). arXiv:2404.07972.

### Environment synthesis

- AWM Team (2026). Agent World Model: Fully Synthetic Pipeline for Tool-Use Environments. Snowflake.
- ScaleEnv Team (2026). [ScaleEnv: Constructing Fully Interactive Environments from Scratch](https://arxiv.org/abs/2602.06820). arXiv:2602.06820.
- LLM-in-Sandbox Team (2026). [Can a Code Sandbox Alone Elicit General Agentic Intelligence?](https://arxiv.org/html/2601.16206). arXiv:2601.16206.

### Infrastructure

- [OpenReward](https://openreward.ai/) — General Reasoning (2026).
- [HuggingFace Blog: When LLMs Grow Hands and Feet](https://huggingface.co/blog/AmberLJC/agentic-rl-systems).
- [Taxonomy of RL Environments for LLM Agents](https://leehanchung.github.io/blogs/2026/03/21/rl-environments-for-llm-agents/).

## Related Pages

- [[agentic-rl-overview]] -- agentic RL landscape
- [[tool-use-rl]] -- tool-use RL that requires environment support
- [[multi-step-reasoning-rl]] -- reasoning RL (code/math environments)
- [[rl-training-frameworks]] -- training infrastructure (veRL, OpenRLHF, etc.)
- [[mcp-protocol]] -- MCP protocol (the basis of ORS)
- [[ai-agent-overview]] -- AI agent architecture
