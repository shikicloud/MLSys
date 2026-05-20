---
title: "AI Agent Architectures and Patterns"
category: ai-agent
tags: [agent, react, plan-and-execute, tree-of-thoughts, reflexion, cognitive-architecture, multi-agent, frameworks, coding-agent, memory, evaluation]
created: 2026-04-13
updated: 2026-05-20
status: mature
---

# AI Agent Architectures and Patterns

> [!abstract]+ TL;DR
> An AI agent is a system that uses an LLM as its core reasoning engine and runs autonomously in a **perceive-reason-act loop**. Unlike chatbots, agents can decompose goals, call tools, process observations, and iteratively self-correct. Core architectures: **ReAct** (ICLR 2023, interleaved reasoning and acting), **Plan-and-Execute** (planner/executor decoupled, lower cost), **Tree of Thoughts** (NeurIPS 2023, multi-path + backtracking), **LATS** (ICML 2024, MCTS + LLM value function), **Reflexion** (verbal self-reflection with no weight update, +10-20 pp on coding benchmarks). 2025-2026 trend: integration of reasoning models (o1/o3, [[grpo#DeepSeek-R1|DeepSeek-R1]]) -- inference-time compute scaling will account for 75% of total AI compute by 2030.

## Overview

An AI agent is a system that uses an LLM as its core reasoning engine and runs autonomously in a perceive-reason-act loop. Unlike a single-turn chatbot, an agent can:

- **Plan autonomously**: decompose complex goals into sub-task sequences
- **Use tools**: call external APIs, execute code, read/write files
- **Perceive the environment**: process observations from tool-execution results, user feedback, and system state
- **Iteratively self-correct**: adjust strategy based on feedback and recover from errors

The agent's core loop can be expressed as:

```
┌─────────────────────────────────────────┐
│              Agent Loop                  │
│                                          │
│   ┌──────────┐    ┌──────────┐          │
│   │ Perceive │───>│  Reason  │          │
│   │ (observe)│    │ (reason) │          │
│   └──────────┘    └────┬─────┘          │
│        ^               │                 │
│        │               v                 │
│   ┌──────────┐    ┌──────────┐          │
│   │ Observe  │<───│   Act    │          │
│   │ (result) │    │ (execute)│          │
│   └──────────┘    └──────────┘          │
│                                          │
└─────────────────────────────────────────┘
```

As of early 2026, inference-time compute scaling has become a mainstream trend, with analysts predicting that inference will account for 75% of total AI compute by 2030. Gartner projects that by the end of 2026, 40% of enterprise applications will embed agents.

---

## Core architectures

### ReAct (Reason + Act)

**Source**: Yao et al., ICLR 2023

ReAct is the most basic and the most widely used agent architecture. Its core idea is to alternate between reasoning (Thought) and acting (Action):

```
Thought 1: I need to look up the user's order status
Action 1:  search_orders(user_id="12345")
Observation 1: Found order #ORD-789, status: Shipped
Thought 2: The order has shipped; I need to fetch the tracking info
Action 2:  get_tracking(order_id="ORD-789")
Observation 2: Tracking number SF12345, expected to arrive tomorrow
Thought 3: I can now reply to the user
Action 3:  respond("Your order has shipped, tracking number SF12345, expected to arrive tomorrow")
```

**Key advantages**:
- The reasoning process is transparent and traceable
- Naturally combines reasoning with tool calling
- Simple to implement, fits most scenarios

**Extended variants**:
| Variant | Core improvement | Use case |
|---------|------------------|----------|
| PreAct | Predict the result before reasoning | Reduce unnecessary tool calls |
| RP-ReAct | Separate planner and executor | Complex multi-step tasks |
| ReSpAct | Add user-clarification steps | Scenarios that need user interaction |

**Python sketch**:

```python
def react_loop(llm, tools, task, max_steps=10):
    """Basic ReAct loop"""
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    history.append({"role": "user", "content": task})

    for step in range(max_steps):
        # Reasoning + action decision
        response = llm.generate(history, tools=tools)

        if response.is_final_answer:
            return response.content

        # Execute the tool call
        tool_name = response.tool_call.name
        tool_args = response.tool_call.arguments
        observation = tools[tool_name].execute(**tool_args)

        # Append the observation to history
        history.append({"role": "assistant", "content": response.content})
        history.append({"role": "tool", "content": observation})

    return "Max steps reached"
```

### Plan-and-Execute

**Core idea**: decouple planning and execution into two independent stages.

```
┌─────────────────────────────────────────────────┐
│                Plan-and-Execute                 │
│                                                 │
│  ┌─────────┐     ┌──────────────────────────┐   │
│  │ Planner │────>│ Step 1: query database   │   │
│  │         │     │ Step 2: analyze data     │   │
│  │ GPT-4o  │     │ Step 3: generate report  │   │
│  └─────────┘     │ Step 4: send email       │   │
│       ^          └──────────┬───────────────┘   │
│       │                     │                   │
│       │  replan if needed   v                   │
│       │          ┌──────────────────────────┐   │
│       └──────────│  Executor                │   │
│                  │  can use a cheaper model │   │
│                  └──────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

**Advantages**:
- Planning and execution can use different models (strong model for planning, cheap model for execution)
- Lower overall cost and faster
- On plan failure, can replan without having to redo completed steps

**Use cases**: complex multi-step tasks such as data-analysis pipelines, automated testing, document generation.

### Tree of Thoughts (ToT)

**Source**: Yao et al., NeurIPS 2023

```
              [Problem]
             /   |   \
           /     |     \
        [Path A] [Path B] [Path C]
        /  \      |      /  \
      [A1] [A2] [B1]  [C1] [C2]
       X    |     X     |    X
            v           v
          [A2 expand] [C1 expand]
            |           |
            v           v
          [Sol 1]     [Sol 2]
                        ^
                     optimal
```

**Core mechanism**:
- Explore multiple reasoning paths in parallel
- Score each path
- Support backtracking and pruning of low-quality paths
- BFS or DFS search strategy

**Extension**: TouT adds uncertainty quantification.

### LATS (Language Agent Tree Search)

**Source**: ICML 2024

Combines Monte Carlo Tree Search (MCTS) with an LLM value function:

```
                   [Initial state]
                   /    |    \
                [s1]  [s2]  [s3]     ← Expansion
                 |     |     |
               [v=0.8][v=0.3][v=0.6] ← LLM evaluation
                 |                    
            UCB picks the best        ← Selection
               / | \
           [s1a][s1b][s1c]            ← Continue expanding
             |    |    |
           [0.9][0.4][0.7]            ← Evaluate again
             |
           [Final solution]            ← Backpropagation
```

**Properties**: most capable but most expensive; suitable for high-quality solution scenarios (e.g. competitive programming).

### Reflexion

**Source**: Shinn et al., NeurIPS 2023

```
┌──────────────────────────────────────────┐
│              Reflexion loop              │
│                                          │
│   [Task] ──> [Attempt] ──> [Execute/Test]│
│                                ^    │    │
│                                │    v    │
│           [Store in memory] <── [Reflect on failure] │
│           (natural language)    (if failed)          │
│                                          │
│   Example memory entries:                │
│   "Last time I forgot to handle n=0"     │
│   "I should validate input first"        │
│                                          │
└──────────────────────────────────────────┘
```

**Core innovation**:
- No weight updates needed
- Learns through natural-language reflection
- +10-20 percentage points on coding benchmarks

### CoALA (Cognitive Architectures for Language Agents)

Maps LLM agents to classical cognitive architectures (ACT-R, Soar) and proposes a modular memory-component framework:

```
┌─────────────────────────────────────────┐
│              CoALA architecture          │
│                                          │
│  ┌──────────┐  ┌──────────┐  ┌────────┐ │
│  │ Working  │  │ Episodic │  │Semantic│ │
│  │ Memory   │  │ Memory   │  │ Memory │ │
│  └────┬─────┘  └────┬─────┘  └───┬────┘ │
│       │              │            │      │
│       v              v            v      │
│  ┌──────────────────────────────────┐    │
│  │       Decision process (LLM)     │    │
│  └──────────────┬───────────────────┘    │
│                 │                         │
│       ┌─────────┴─────────┐              │
│       v                   v              │
│  ┌──────────┐       ┌──────────┐         │
│  │ Internal │       │ External │         │
│  │ action   │       │ action   │         │
│  │(reason/  │       │ (tool    │         │
│  │ retrieve)│       │  call)   │         │
│  └──────────┘       └──────────┘         │
└─────────────────────────────────────────┘
```

---

## Core agent components

### 1. Perception

The agent receives input from many sources:

| Source type | Examples |
|-------------|----------|
| User input | Natural-language instructions, code, images |
| Tool return values | API responses, database query results, file contents |
| Environment state | System logs, error messages, test results |
| Other agents | Message passing in multi-agent systems |

### 2. Reasoning

The LLM serves as the core reasoning engine, responsible for:
- **Task understanding**: parse user intent
- **Plan decomposition**: break a complex task into sub-steps
- **Tool selection**: decide which tool to use and what arguments to pass
- **Result interpretation**: understand what the tool returned
- **Error diagnosis**: identify problems and decide on a fix

Reasoning models (e.g. o1/o3, DeepSeek-R1) lift complex-reasoning ability through inference-time compute scaling.

### 3. Action

Action types the agent can execute:

```python
# Action-type definitions
class ActionType:
    TOOL_CALL = "tool_call"       # call an external tool
    CODE_EXEC = "code_execution"  # execute code
    RESPOND = "respond"           # reply to the user
    DELEGATE = "delegate"         # delegate to a sub-agent
    WAIT = "wait"                 # wait for an external event
    TERMINATE = "terminate"       # end the task
```

### 4. Memory

See the [[#Agent memory system]] section below.

---

## Agent patterns

### Single-agent pattern

A single LLM instance completes all tasks alone. Suitable for:
- Well-defined tasks (code fixing, QA)
- Latency-sensitive scenarios
- Cost-constrained deployments

```
User ──> [single agent + toolset] ──> result
```

### Multi-agent pattern

Several specialized agents collaborate on a complex task.

**Common topologies**:

```
Pattern 1: master/delegate
┌───────────────┐
│ Orchestrator  │──> [research agent]
│ agent         │──> [coding agent]
│               │──> [testing agent]
└───────────────┘

Pattern 2: debate/adversarial
[Proposer] <──> [Reviewer] <──> [Summarizer]

Pattern 3: pipeline
[Analyze] ──> [Design] ──> [Implement] ──> [Test] ──> [Deploy]

Pattern 4: democratic vote
[Agent A] ──┐
[Agent B] ──┼──> [vote/consensus] ──> final decision
[Agent C] ──┘
```

**Google A2A protocol**: defines a standard protocol for inter-agent communication (released April 2025), complementary to [[mcp-protocol|MCP]] -- MCP handles agent-to-tool communication, A2A handles agent-to-agent communication.

---

## Framework comparison

### Detailed comparison of mainstream frameworks (2025-2026)

| Framework | Developer | Core feature | Best for | Language |
|-----------|-----------|--------------|----------|----------|
| **LangGraph** | LangChain | Graph-based workflow definition, state machine | Complex, controllable flows | Python/JS |
| **CrewAI** | CrewAI | Role-driven multi-agent, task delegation | Multi-role collaboration | Python |
| **AutoGen** | Microsoft | Multi-agent dialogue, code execution | Research / prototype | Python |
| **OpenAI Agents SDK** | OpenAI | Native tool calling, Handoff mechanism | OpenAI ecosystem | Python |
| **Anthropic Agents SDK** | Anthropic | Claude-native, MCP integration | Claude ecosystem | Python |
| **Bee Agent** | IBM | Open-source, TypeScript-native | Enterprise TS projects | TypeScript |
| **smolagents** | HuggingFace | Lightweight, code agents | Research / small projects | Python |

### LangGraph example

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated

class AgentState(TypedDict):
    messages: list
    plan: list[str]
    current_step: int

def planner(state: AgentState) -> AgentState:
    """Planning stage: decompose the task"""
    plan = llm.invoke(
        f"Decompose the following task into steps: {state['messages'][-1]}"
    )
    return {"plan": plan.steps, "current_step": 0}

def executor(state: AgentState) -> AgentState:
    """Execution stage: complete the current step"""
    step = state["plan"][state["current_step"]]
    result = llm.invoke(f"Execute: {step}", tools=available_tools)
    return {
        "messages": state["messages"] + [result],
        "current_step": state["current_step"] + 1
    }

def should_continue(state: AgentState) -> str:
    if state["current_step"] >= len(state["plan"]):
        return END
    return "executor"

# Build the graph
graph = StateGraph(AgentState)
graph.add_node("planner", planner)
graph.add_node("executor", executor)
graph.add_edge("planner", "executor")
graph.add_conditional_edges("executor", should_continue)
graph.set_entry_point("planner")

agent = graph.compile()
```

### OpenAI Agents SDK example

```python
from openai import agents

# Define a tool
@agents.tool
def search_web(query: str) -> str:
    """Search the web for information"""
    return web_search(query)

# Define an agent
research_agent = agents.Agent(
    name="Research Agent",
    instructions="You are a research assistant skilled at finding and summarizing information.",
    tools=[search_web],
    model="gpt-4o"
)

# Define a Handoff (delegation)
triage_agent = agents.Agent(
    name="Triage Agent",
    instructions="Based on the user's request, decide which specialist agent to delegate to.",
    handoffs=[research_agent, coding_agent, writing_agent]
)

# Run
result = agents.run(triage_agent, "Help me research the latest advances in Transformer architectures")
```

---

## Coding agents

Coding agents are the most successful category of agent applications in 2025-2026, performing tasks directly on a code base.

### Comparison of major products

| Product | Developer | Features | Mode |
|---------|-----------|----------|------|
| **Claude Code** | Anthropic | CLI tool, directly operates terminal and filesystem | Terminal / IDE |
| **Cursor** | Cursor Inc. | AI-native IDE with built-in agent mode | IDE |
| **Devin** | Cognition | First "AI software engineer", fully autonomous | Web / autonomous |
| **GitHub Copilot** | GitHub / MS | Largest user base, Agent mode released 2025 | IDE plugin |
| **Windsurf** | Codeium | AI IDE, Cascade multi-step agent | IDE |
| **Codex CLI** | OpenAI | Open-source CLI coding agent | Terminal |
| **Augment Code** | Augment | Enterprise-grade coding agent | IDE plugin |

### Claude Code workflow example

```bash
# Typical Claude Code workflow
$ claude "Fix the JWT-expiry issue in the auth module"

# Claude Code will:
# 1. Search the code base for relevant files
# 2. Read the code to understand the logic
# 3. Identify the bug (e.g. missing expiry check)
# 4. Modify the code
# 5. Run tests to verify the fix
# 6. Summarize the changes
```

### Core capabilities of a coding agent

```
┌────────────────────────────────────────────────┐
│           Core coding-agent capabilities        │
│                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │ Code     │  │ Code     │  │ Code     │     │
│  │ reading &│  │ generate │  │ edit &   │     │
│  │ understd │  │ & comple │  │ refactor │     │
│  └──────────┘  └──────────┘  └──────────┘     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │ Test     │  │ Bug      │  │ Terminal │     │
│  │ write &  │  │ diagnose │  │ ops &    │     │
│  │ run      │  │ & fix    │  │ commands │     │
│  └──────────┘  └──────────┘  └──────────┘     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │ Code     │  │ Git ops  │  │ Doc      │     │
│  │ search & │  │ & PR     │  │ generate │     │
│  │ navigate │  │          │  │ & comment│     │
│  └──────────┘  └──────────┘  └──────────┘     │
└────────────────────────────────────────────────┘
```

---

## Agent memory system

An agent's memory is the key to "continual learning" and "context retention".

### Memory types

```
┌─────────────────────────────────────────────────┐
│              Agent memory system                 │
│                                                  │
│  ┌───────────────────────────────────────────┐  │
│  │  Short-Term / Working Memory              │  │
│  │  - Current dialogue history               │  │
│  │  - Information inside the context window  │  │
│  │  - Lifetime: a single session             │  │
│  └───────────────────────────────────────────┘  │
│                                                  │
│  ┌───────────────────────────────────────────┐  │
│  │  Long-Term Memory                         │  │
│  │  - Knowledge stored in a vector DB        │  │
│  │  - User preferences and history summary   │  │
│  │  - Lifetime: persisted across sessions    │  │
│  └───────────────────────────────────────────┘  │
│                                                  │
│  ┌───────────────────────────────────────────┐  │
│  │  Episodic Memory                          │  │
│  │  - Past task-execution experience         │  │
│  │  - Success/failure cases and reflections  │  │
│  │  - Reflexion-like learning record         │  │
│  └───────────────────────────────────────────┘  │
│                                                  │
│  ┌───────────────────────────────────────────┐  │
│  │  Procedural Memory                        │  │
│  │  - How to use a tool                      │  │
│  │  - Standard Operating Procedures (SOP)    │  │
│  │  - Coding conventions and best practices  │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

### Implementation

```python
import chromadb
from datetime import datetime

class AgentMemory:
    """Example agent memory-system implementation"""

    def __init__(self):
        self.client = chromadb.Client()
        # Long-term memory - vector DB
        self.long_term = self.client.create_collection("long_term")
        # Episodic memory - task-execution experience
        self.episodic = self.client.create_collection("episodic")
        # Short-term memory - current session
        self.working_memory = []

    def store_experience(self, task: str, result: str, success: bool):
        """Store a task-execution experience into episodic memory"""
        self.episodic.add(
            documents=[f"Task: {task}\nResult: {result}\nSuccess: {success}"],
            metadatas=[{
                "timestamp": datetime.now().isoformat(),
                "success": success
            }],
            ids=[f"exp_{datetime.now().timestamp()}"]
        )

    def recall_similar(self, query: str, n_results: int = 3) -> list:
        """Retrieve past experience relevant to the current task"""
        results = self.episodic.query(
            query_texts=[query],
            n_results=n_results
        )
        return results["documents"]

    def add_to_working(self, message: dict):
        """Add a message to working memory"""
        self.working_memory.append(message)
        # Sliding-window management to avoid exceeding context length
        if len(self.working_memory) > 50:
            self._summarize_and_compress()

    def _summarize_and_compress(self):
        """Compress working memory: summarize older dialogue"""
        old_messages = self.working_memory[:30]
        summary = llm.summarize(old_messages)
        self.working_memory = [
            {"role": "system", "content": f"Summary of earlier dialogue: {summary}"}
        ] + self.working_memory[30:]
```

---

## Evaluation benchmarks

### Major benchmarks

| Benchmark | Evaluates | Best score (2026 Q1) | Notes |
|-----------|-----------|----------------------|-------|
| **SWE-bench Verified** | Fixing real GitHub issues | ~65% (Claude 3.5 + scaffold) | 500 human-verified Python-repo issues |
| **SWE-bench Full** | Full version, 2294 problems | ~50% | Larger and more diverse |
| **WebArena** | Web browsing and operation | ~42% | Completing tasks on real websites |
| **GAIA** | General AI-assistant ability | ~75% (Level 1) | Requires multi-step reasoning + tool use |
| **HumanEval** | Code generation | ~95%+ | Function-level code completion |
| **MATH** | Math problem solving | ~95%+ | Competition-level math |
| **TAU-bench** | Real tool use | ~50% (airline domain) | Simulates real customer-service scenarios |
| **ToolBench** | Tool-use ability | ~70% | 16,000+ real APIs |

### SWE-bench progress trend

```
SWE-bench Verified solve rate (%)
│
65%│                                          ●  2026 Q1 (best)
60%│                                     ●
55%│                                ●
50%│                           ●
45%│                      ●
40%│                 ●
35%│            ●
30%│       ●
25%│  ●
20%│●
   └──────────────────────────────────────────
    2024    2024     2025    2025    2025   2026
    Q2      Q4       Q1      Q2      Q4     Q1
```

---

## State of the art (2025-2026)

### Key trends

1. **Inference-time compute scaling**: reasoning models (o1/o3, DeepSeek-R1, Claude 3.5 Extended Thinking) allocate variable test-time compute, with significant gains on complex tasks

2. **Dual-paradigm fusion**: combining symbolic/classical methods with neural/generative methods
   - Classical planning algorithms + LLM reasoning
   - Formal verification + neural generation

3. **Coding agents lead**: coding is the most successful agent-application domain
   - Auto-verifiable (run tests)
   - Clear success criteria
   - Abundant training data

4. **Multi-modal agents**: combining vision, code, and web browsing into one capability

5. **Long-horizon autonomy**: autonomous task execution from the minute scale to the hour scale

### The "underthinking" problem

Reasoning models exhibit an "overthinking" problem:
- Drift away after reaching the right direction in an intermediate step
- Over-explore low-quality reasoning paths
- Waste reasoning time on simple problems

---

## Limitations and challenges

### Core limitations

1. **Error propagation**: in a multi-step chain, early errors cascade
2. **Cost-accuracy trade-off**: search-style agents (ToT, LATS) perform well but cost is huge
3. **Reliability**: even at 95% per-step success, overall success after 20 steps is only ~36%
4. **Hallucination risk**: wrong tool-call arguments, misreading observations
5. **Safety**: arbitrary code and API calls bring security risks
6. **Evaluation difficulty**: end-to-end evaluation is expensive and not reproducible
7. **Context-window limit**: long tasks accumulate large context, degrading performance

### Reliability calculation

```
Multi-step agent reliability:

per-step success rate  5 steps  10 steps  20 steps  50 steps
        99%             95.1%    90.4%    81.8%    60.5%
        95%             77.4%    59.9%    35.8%     7.7%
        90%             59.0%    34.9%    12.2%     0.5%
        80%             32.8%    10.7%     1.2%     0.0%

Formula: P(all succeed) = p^n
Conclusion: every step of the agent has to be highly reliable
```

---

## References

- Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models," ICLR 2023
- Yao et al., "Tree of Thoughts: Deliberate Problem Solving with Large Language Models," NeurIPS 2023
- Shinn et al., "Reflexion: Language Agents with Verbal Reinforcement Learning," NeurIPS 2023
- Zhou et al., "Language Agent Tree Search," ICML 2024
- Sumers et al., "Cognitive Architectures for Language Agents," 2023
- Xi et al., "The Rise and Potential of Large Language Model Based Agents: A Survey," 2023
- Wang et al., "A Survey on Large Language Model based Autonomous Agents," 2023
- SWE-bench, https://swe-bench.github.io
- Zhou et al., "WebArena: A Realistic Web Environment for Building Autonomous Agents," 2023

---

## Related pages

- [[tool-use]] -- how agents interact with tools
- [[mcp-protocol]] -- Model Context Protocol standardizing tool integration
- [[multi-agent-systems]] -- multi-agent collaboration
- [[agent-frameworks]] -- implementation frameworks
- [[agent-memory]] -- memory architectures
- [[agentic-rl-overview]] -- training agents via RL
- [[compound-ai-systems]] -- compound AI systems
- [[agent-serving-challenges]] -- challenges of serving agents
- [[multi-turn-optimization]] -- multi-turn dialogue optimization
