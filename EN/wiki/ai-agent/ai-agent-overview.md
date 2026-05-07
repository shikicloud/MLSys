---
title: "AI Agent Architectures and Patterns"
category: ai-agent
tags: [agent, react, plan-and-execute, tree-of-thought, reflexion, cognitive-architecture, multi-agent, frameworks, coding-agents, memory, evaluation]
created: 2026-04-13
updated: 2026-05-07
status: mature
---

# AI Agent Architectures and Patterns

> [!abstract]+ TL;DR
> AI agents are LLM-powered systems operating autonomously in a **perceive-reason-act loop**. Unlike chatbots, agents decompose goals into subtasks, invoke tools, process observations, and iteratively refine. Core architectures: **ReAct** (ICLR 2023, interleaved reasoning + actions), **Plan-and-Execute** (decoupled planner/executor for cost efficiency), **Tree of Thoughts** (NeurIPS 2023, multiple reasoning paths with backtracking), **LATS** (ICML 2024, MCTS + LLM value functions), **Reflexion** (verbal self-reflection without weight updates, +10–20 pp on coding). 2025–2026 trend: reasoning-model integration (o1/o3, [[grpo#DeepSeek-R1|DeepSeek-R1]]) — inference-time compute scaling will claim 75 % of total AI compute by 2030.

```
┌────────────────────────────────┐
│         Agent Loop             │
│  ┌─────────┐  ┌─────────┐    │
│  │Perceive │─>│ Reason  │    │
│  └─────────┘  └────┬────┘    │
│       ^             │         │
│       │             v         │
│  ┌─────────┐  ┌─────────┐    │
│  │ Observe │<─│  Act    │    │
│  └─────────┘  └─────────┘    │
└────────────────────────────────┘
```

As of early 2026, inference-time compute scaling is the dominant trend. Analysts project inference will claim 75% of total AI compute by 2030, and Gartner predicts 40% of enterprise apps will embed agents by end of 2026.

---

## Core Architectures

### ReAct (Reason + Act) -- ICLR 2023

Interleaves reasoning traces with action execution: `Thought -> Action -> Observation -> Thought -> ...`

```python
def react_loop(llm, tools, task, max_steps=10):
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    history.append({"role": "user", "content": task})
    for step in range(max_steps):
        response = llm.generate(history, tools=tools)
        if response.is_final_answer:
            return response.content
        observation = tools[response.tool_call.name].execute(
            **response.tool_call.arguments
        )
        history.append({"role": "assistant", "content": response.content})
        history.append({"role": "tool", "content": observation})
    return "Max steps reached"
```

Extensions: PreAct (prediction before reasoning), RP-ReAct (planner-executor split), ReSpAct (user clarification).

### Plan-and-Execute

Decouples planning from execution. A planner (strong model) decomposes tasks; executors (potentially cheaper models) carry them out. Faster, more cost-effective, supports replanning.

### Tree of Thoughts (ToT) -- NeurIPS 2023

Explores multiple reasoning paths simultaneously with BFS/DFS search and backtracking. Extension: TouT (uncertainty quantification).

### LATS -- ICML 2024

Language Agent Tree Search -- integrates MCTS with LLM value functions. Most capable but most expensive; suited for high-stakes problems like competitive programming.

### Reflexion -- NeurIPS 2023

Verbal self-reflection without weight updates. Stores natural-language reflections as episodic memory. +10-20 pp on coding benchmarks.

### CoALA

Cognitive Architectures for Language Agents. Maps agents to classical cognitive architectures (ACT-R, Soar) with modular memory components: working, episodic, semantic, procedural.

---

## Agent Patterns

**Single Agent**: One LLM instance with tools. Best for well-defined tasks, latency-sensitive scenarios.

**Multi-Agent Topologies**:

```
Orchestrator/Delegate:   [Orchestrator] -> [Research] / [Code] / [Test]
Debate/Adversarial:      [Proposer] <-> [Critic] <-> [Summarizer]
Pipeline:                [Analyze] -> [Design] -> [Implement] -> [Test]
Democratic:              [A] + [B] + [C] -> [Vote] -> Decision
```

Google's **A2A protocol** (Apr 2025) standardizes agent-to-agent communication, complementing [[mcp-protocol|MCP]] (agent-to-tool).

---

## Framework Comparison (2025-2026)

| Framework | Developer | Strength | Best For |
|-----------|-----------|----------|----------|
| **LangGraph** | LangChain | Graph-based workflows, state machines | Complex controllable flows |
| **CrewAI** | CrewAI | Role-driven multi-agent, task delegation | Multi-role collaboration |
| **AutoGen** | Microsoft | Multi-agent conversation, code execution | Research/prototyping |
| **OpenAI Agents SDK** | OpenAI | Native tool calling, Handoff mechanism | OpenAI ecosystem |
| **Anthropic Agents SDK** | Anthropic | Claude-native, MCP integration | Claude ecosystem |
| **smolagents** | HuggingFace | Lightweight, code agents | Research/small projects |

---

## Coding Agents

Coding agents are the most successful agent category in 2025-2026, operating directly on codebases.

| Product | Developer | Mode | Key Feature |
|---------|-----------|------|-------------|
| **Claude Code** | Anthropic | CLI/IDE | Terminal + file system access |
| **Cursor** | Cursor Inc. | AI-native IDE | Built-in agent mode |
| **Devin** | Cognition | Web/autonomous | First "AI software engineer" |
| **GitHub Copilot** | GitHub/MS | IDE plugin | Largest user base, Agent mode (2025) |
| **Windsurf** | Codeium | AI IDE | Cascade multi-step agent |
| **Codex CLI** | OpenAI | CLI | Open-source coding agent |

---

## Agent Memory Systems

```
┌─────────────────────────────────────────┐
│         Agent Memory Taxonomy            │
│                                          │
│  Short-Term / Working Memory             │
│    Current conversation context          │
│    Lifetime: single session              │
│                                          │
│  Long-Term Memory                        │
│    Vector DB, user preferences, summaries│
│    Lifetime: persistent across sessions  │
│                                          │
│  Episodic Memory                         │
│    Past task experiences & reflections   │
│    Similar to Reflexion's learning log   │
│                                          │
│  Procedural Memory                       │
│    Tool usage patterns, SOPs, best pracs │
└─────────────────────────────────────────┘
```

```python
class AgentMemory:
    def __init__(self):
        self.long_term = chromadb.Collection("long_term")
        self.episodic = chromadb.Collection("episodic")
        self.working_memory = []

    def store_experience(self, task, result, success):
        self.episodic.add(
            documents=[f"Task: {task}\nResult: {result}\nSuccess: {success}"],
            metadatas=[{"timestamp": now(), "success": success}],
            ids=[f"exp_{timestamp()}"]
        )

    def recall_similar(self, query, n=3):
        return self.episodic.query(query_texts=[query], n_results=n)
```

---

## Evaluation Benchmarks

| Benchmark | Evaluates | Best (2026 Q1) | Notes |
|-----------|-----------|----------------|-------|
| **SWE-bench Verified** | Real GitHub issue fixing | ~65% | 500 human-verified Python repo issues |
| **WebArena** | Web browsing & interaction | ~42% | Tasks on real websites |
| **GAIA** | General AI assistant | ~75% (L1) | Multi-step reasoning + tools |
| **HumanEval** | Code generation | ~95%+ | Function-level completion |
| **TAU-bench** | Real tool use | ~50% | Simulated customer service |

---

## Limitations and Challenges

**Reliability decay**: Even 95% single-step success yields only ~36% over 20 steps (`0.95^20`).

```
Single-step  5 steps  10 steps  20 steps
  99%        95.1%    90.4%     81.8%
  95%        77.4%    59.9%     35.8%
  90%        59.0%    34.9%     12.2%
```

Key challenges:
- **Error propagation** in multi-step chains
- **"Underthinking"** in reasoning models -- reaching correct intermediate steps then deviating
- **Cost-accuracy tradeoffs** for search-based agents (ToT, LATS)
- **Hallucination** in tool call parameters and observation interpretation
- **Context window pressure** from long agent sessions
- **Security** risks from arbitrary code/API execution

---

## References

- Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models," ICLR 2023
- Yao et al., "Tree of Thoughts: Deliberate Problem Solving with LLMs," NeurIPS 2023
- Shinn et al., "Reflexion: Language Agents with Verbal Reinforcement Learning," NeurIPS 2023
- Zhou et al., "Language Agent Tree Search," ICML 2024
- Sumers et al., "Cognitive Architectures for Language Agents," 2023
- SWE-bench, https://swe-bench.github.io

---

## Related Pages

- [[tool-use]] -- How agents interact with tools
- [[mcp-protocol]] -- Model Context Protocol for tool integration
- [[multi-agent-systems]] -- Multiple agents collaborating
- [[agent-frameworks]] -- Implementation frameworks
- [[agent-memory]] -- Memory architectures
- [[agentic-rl-overview]] -- Training agents via RL
- [[compound-ai-systems]] -- Compound AI systems
- [[agent-serving-challenges]] -- Serving challenges for agents
- [[multi-turn-optimization]] -- Multi-turn optimization
