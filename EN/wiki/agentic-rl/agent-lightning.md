---
title: "Agent Lightning: Train ANY AI Agents with Reinforcement Learning"
category: agentic-rl
tags: [agent-lightning, microsoft, training-agent-disaggregation, lightningrl, opentelemetry, langchain, autogen, openai-agents-sdk, paper-review]
created: 2026-06-02
updated: 2026-06-02
status: mature
paper: arXiv:2508.03680
code: https://github.com/microsoft/agent-lightning
---

# Agent Lightning: Train ANY AI Agents with Reinforcement Learning

> [!info] Paper metadata
> - **Paper**: [arXiv:2508.03680](https://arxiv.org/abs/2508.03680) — *Agent Lightning: Train ANY AI Agents with Reinforcement Learning*, 2025-08-05
> - **Code**: [microsoft/agent-lightning](https://github.com/microsoft/agent-lightning) — Apache-2.0, v0.3.0 (Dec 2025), 17.3k stars
> - **Authors**: Xufang Luo, Yuge Zhang, Zhiyuan He, Zilong Wang, Siyun Zhao, Dongsheng Li, Luna K. Qiu, Yuqing Yang
> - **Affiliation**: Microsoft Research
> - **Contact**: agent-lightning@microsoft.com

> [!important] Position in the agentic-RL stack
> Agent Lightning was the **first publicly-documented** framework (Aug 2025) to fully decouple agent execution from RL training. It predates [[prorl-agent|ProRL Agent]] (Mar 2026) by 7 months and [[polar|Polar]] (May 2026) by 9 months. The architectural pattern it pioneered — *agent runs on a client; LLM lives on a server speaking OpenAI-compatible API; trajectories ship back as transitions* — became the template the NVIDIA papers extend. Anyone surveying agentic-RL infra should read this first.

---

## Summary (read this if you have 2 minutes)

**What it is.** Agent Lightning is Microsoft Research's framework for RL-training any LLM-based agent with **almost zero code modification** to the agent itself. It works with agents built on LangChain, OpenAI Agents SDK, AutoGen, or from-scratch Python — the framework wraps your agent in a "Lightning Client" that captures every LLM call via OpenTelemetry, ships the resulting `(input, output, reward)` transitions to a "Lightning Server" colocated with the RL trainer (default: veRL), and updates the policy LLM whose endpoint your agent calls.

**The one idea.** **Cast agent execution as an MDP where one action = one LLM call's token sequence**, then collect those actions plus their context as a flat list of transitions. Three sub-pieces hold this up:

1. **Unified data interface**: every agent's trajectory reduces to `{(input^x,k_i, output^x,k_i, reward^x,k_i)}`, regardless of whether the agent is LangChain / AutoGen / OpenAI SDK / custom. Tool calls and other non-LLM components update state but aren't training transitions.
2. **LightningRL hierarchical credit assignment**: episode-level return → per-action value (currently identical = final return; future-work supports learned value head) → per-token advantage. Bypasses the sequence-concatenation-with-masking approach used by every other multi-turn RL framework.
3. **Training-Agent Disaggregation (TA Disaggregation)**: Lightning Server (controls RL, exposes OpenAI-like API for the trained model) ↔ Lightning Client (runs agent, captures OpenTelemetry traces, ships transitions back). Each task gets its own server-side endpoint URL.

Remove the MDP abstraction and you're back to per-framework integration hacks; remove LightningRL and you're back to context-bloat masking; remove TA Disaggregation and you're back to coupling that breaks scalability.

**Headline result.** Llama-3.2-3B-Instruct base + GRPO across three agent frameworks and three tasks, all showing stable RL improvement curves (paper Figs. 5-7):

| Task | Framework | Dataset | Reward curve |
| ---- | --------- | ------- | -----------: |
| Text-to-SQL | LangChain | Spider | 0.0 → 0.65 (train) / 0.15 → 0.55 (test) |
| Open-domain QA (RAG) | OpenAI Agents SDK | MuSiQue | 0.05 → 0.40 (train) / 0.0 → 0.22 (test) |
| Math QA (tool-use) | AutoGen | Calc-X | 0.05 → 0.85 (train) / 0.05 → 0.78 (test) |

The point isn't the absolute numbers (3B is small) — it's that **the same framework + same trainer optimizes agents written in three different frameworks without per-framework engineering**.

**Why it matters.**

- **First demonstrated trainer-agnostic + agent-agnostic decoupling.** The Lightning Server only talks to the trainer; the Lightning Client only talks to the agent. Either side can be swapped without touching the other. This is the architectural pattern [[polar|Polar]] later took further with the URL-swap trick.
- **OpenTelemetry is its differentiating asset.** Production teams that already instrument agents with OpenTelemetry (Langfuse, Phoenix, AgentOps) get RL training "for free" — the trace pipeline already exists.
- **Multi-agent selective optimization works.** The text-to-SQL experiment uses 3 agents (writer, checker, re-writer) but trains only 2; LightningRL handles credit assignment per agent's transitions naturally, no masking needed.
- **2027 prediction.** The MDP-with-LLM-action abstraction will be the canonical formal model for agentic RL papers — already adopted (in different framings) by ProRL Agent and Polar. Agent Lightning's specific contribution (OpenTelemetry-native + framework-pluralism + selective-optimization) survives because it complements rather than competes with the harness-proxy approach.

---

# Depth (drill-down starts here)

## Background: why agentic-RL infrastructure was broken before Agent Lightning

In mid-2025, every multi-turn agent-RL framework had the same shape: **trainer + agent in one process**, sequences concatenated, masked, fed to PPO/GRPO. The paper enumerates the failure modes explicitly (§5.2):

| Failure mode | What goes wrong |
| ------------ | --------------- |
| **Context-length explosion** | Concatenating turns produces sequences far exceeding the LLM's context window, especially with MCP tool exchanges and multi-agent communication |
| **Position-encoding disruption** | RoPE assumes token continuity; masking-based methods inject discontinuities that break the assumption |
| **Framework-specific masking** | Each combination of agent framework × trainer needs bespoke masking; it doesn't generalize |
| **Tight coupling** | Agent code lives inside the trainer process; swapping trainers requires re-implementing the agent loop |
| **Ray expertise required** | Most RL frameworks (veRL, ROLL, OpenRLHF) assume the user can integrate with Ray, an unrealistic bar for application developers |

Prior frameworks Agent Lightning positions against:

| Framework | Coupling pattern | Multi-turn handling |
| --------- | ---------------- | ------------------- |
| **OpenRLHF** ([Hu et al., 2024](https://arxiv.org/abs/2405.11143)) | Embeds agent loop in trainer | Sequence concat + masking |
| **TRL** ([von Werra et al., 2020](https://github.com/huggingface/trl)) | Same | Same |
| **veRL** ([Sheng et al., 2024](https://arxiv.org/abs/2409.19256)) | Same | Same |
| **AReaL** ([Fu et al., 2025](https://arxiv.org/abs/2505.24298)) | Same | Same |
| **ROLL** ([Wang et al., 2025a](https://arxiv.org/abs/2506.06122)) | Same | Same |
| **Search-R1** ([Jin et al., 2025](https://arxiv.org/abs/2503.09516)) | Application-specific | Retrieved-token loss masking ([[search-r1]]) |
| **rLLM** ([Tan et al., 2025](https://pretty-radio-b75.notion.site/rLLM-A-Framework-for-Post-Training-Language-Agents-21b81902c146819db63cd98a54ba5f31)) | Application-specific | Per-task masking |
| **DeepSWE** ([Luo et al., 2025](https://pretty-radio-b75.notion.site/DeepSWE-Training-a-Fully-Open-sourced-State-of-the-Art-Coding-Agent-by-Scaling-RL-22281902c1468193aabbe9a8c59bbe33)) | SWE-specific | SWE-specific |
| **Agent Lightning (this work)** | **Decoupled via TA Disaggregation** | **Per-transition; no masking** |

The paper's verbatim framing of the contribution (§1):

> "Agent Lightning is the first framework to achieve a full decoupling between agents and RL training. This decoupling enables Agent Lightning to be seamlessly applied to ANY AI agent regardless of implementation approach, with almost ZERO code modifications."

## Three components in detail

The framework has three load-bearing pieces. Everything else is plumbing.

### Component 1 — Unified data interface + MDP formulation

The first move is mathematical: cast agent execution as a POMDP. The full tuple is $\mathcal{M} = (\mathcal{S}, \mathcal{O}, \mathcal{A}, \mathcal{T}, \mathcal{R})$:

- **$\mathcal{S}$** — state space; one state = snapshot of all *semantic variables* in the agent's execution context
- **$\mathcal{O}$** — observation space; what the policy LLM actually sees as input (only the semantic variables visible to it)
- **$\mathcal{A}$** — action space; **one action = the entire token sequence generated by one LLM invocation** (not one token)
- **$\mathcal{T}(s'|s,a)$** — transition dynamics, usually unknown
- **$\mathcal{R}(s,a)$** — scalar reward function

The state at timestep $t$ for the $k$-th execution of task $x$ contains $V$ semantic variables:
$$\text{state}_t(x,k) = \{\text{variable}_i^{x,k,t}\}_{i=1}^V$$

Each component invocation (LLM call OR tool call) is a `call`:
$$\text{call}_i^{x,k} = (\text{meta}_i^{x,k}, \text{input}_i^{x,k}, \text{output}_i^{x,k}), \quad \text{output}_i^{x,k} = C_i(\text{input}_i^{x,k})$$

where $C_i \in \mathcal{M} \cup \mathcal{T}$ is either an LLM or a tool. Critically, **input and output are themselves semantic variables at specific timesteps**:
$$\text{input}_i^{x,k} \in \text{state}_{t_1}(x,k), \quad \text{output}_i^{x,k} \in \text{state}_{t_2}(x,k)$$

For RL data extraction, only the LLM calls become transitions (Eq. 6):
$$\text{execution}^{RL}(x,k) = \{(\text{input}_i^{x,k}, \text{output}_i^{x,k}, r_i^{x,k})\}_{i=1}^T$$

Tool calls modify the state but don't generate gradients — they're causally important but not trainable.

**Worked example: RAG agent.** Paper Fig. 2 traces a 4-step RAG agent through state updates:

![Unified data interface: 4-step RAG agent with state evolution and trajectory extraction, paper Fig. 2](EN/wiki/agentic-rl/agent-lightning-figs/unified-data-interface-rag.png)


```
state_0: {UserInput=U, Query=NA, Passages=NA, Answer=NA}
state_1: Q = LLM(U)             → {U, Q, NA, NA}
state_2: P = Search(Q)          → {U, Q, P,  NA}
state_3: A = LLM(U, P)          → {U, Q, P,  A }
```

The extracted trajectory contains **only the two LLM calls**, with rewards on the final answer:

```
trajectory = [
  (input=U,    output=Q, reward=NA),
  (input=(U,P), output=A, reward=R)
]
```

The Search tool's contribution to state is preserved (P becomes part of the second LLM call's input), but the search call itself doesn't appear as a training sample. This is the cleanest formulation in the literature for "what's trainable vs what's environment" in agentic RL.

> [!quote] The architectural payoff of this abstraction
> "The unified data interface captures all state changes, including those caused by non-LLM components, enabling flexible and highly customizable optimization methods. ... It accommodates arbitrary and complex agent interaction logic **without requiring explicit parsing of the entire execution DAG**, thereby greatly simplifying RL-based optimization for diverse agent workflows."

The phrase *"without requiring explicit parsing of the entire execution DAG"* is the load-bearing claim: the framework doesn't need to know about your agent's control flow, dependencies, or sub-agent structure. It just collects LLM calls and their inputs.

### Component 2 — LightningRL hierarchical credit assignment

Standard token-level RL loss (Eq. 8):
$$\mathcal{L}(\theta) = -\mathbb{E}_{x \sim \mathcal{X}, \text{output} \sim \pi_\theta(\cdot|x)} \left[ \sum_{j=1}^N \log \pi_\theta(y_j | x, y_{<j}) \cdot A_j \right]$$

The challenge in multi-turn agentic RL: each *action* is a multi-token LLM response, but rewards typically arrive only at the end of the trajectory. LightningRL is the credit-assignment glue that connects them.

**Two-step credit assignment** (§3.3.2):

1. **Step 1 — Episode return → per-action values.** A credit assignment module distributes episode-level return $R$ across each action $a_t$ within the episode.

   *Current implementation*: identity assignment — every action gets the same value, equal to the final return.
   
   *Acknowledged limitation*: this is a placeholder. The paper says future work will introduce a learned high-level value function for per-action advantage estimation.

2. **Step 2 — Per-action value → per-token token-level loss.** Within each action (= one LLM response), apply the standard PPO / GRPO / REINFORCE++ token-level loss with the action's assigned value as advantage. This is **single-turn RL machinery, unmodified**.

**Why this matters: it sidesteps the masking problem entirely.** Figure 3 in the paper contrasts three approaches:

![LightningRL vs Single-call GRPO and Previous multi-turn GRPO, paper Fig. 3](EN/wiki/agentic-rl/agent-lightning-figs/lightningrl-vs-grpo.png)


- **(a) Single-call GRPO**: same-task responses grouped for advantage estimation — works for single-turn but not multi-turn.
- **(b) Previous multi-turn GRPO**: full trajectory concatenated, non-LLM tokens masked. Breaks RoPE, complicates kernels, framework-specific.
- **(c) LightningRL**: trajectory **decomposed into independent transitions**; same-task transitions grouped for advantage. No masking. Native to single-turn RL infrastructure.

The paper's verbatim claim (§3.3.2):

> "Third, compared to the masking strategy, LightningRL offers a more robust and scalable implementation. Masking-based approaches not only require tight coupling between training and agent execution logic, but also disrupt the continuity of tokens in LLMs, which is assumed in the widely used position encoding approaches, such as Rotary Positional Embeddings (RoPE)."

**Multi-agent extension.** Because each transition is independent and tagged with its source agent, you can **selectively optimize a subset of agents** in a multi-agent system simply by only including their transitions in the gradient update. The text-to-SQL experiment does exactly this: 3 agents (SQL writer, checker, re-writer), 2 trained, 1 frozen — no masking required.

### Component 3 — Training-Agent Disaggregation (TA Disaggregation)

The system architecture has two processes that communicate over HTTP (Fig. 4 in the paper):

![Training-Agent Disaggregation architecture, paper Fig. 4](EN/wiki/agentic-rl/agent-lightning-figs/training-agent-disaggregation.png)

**Lightning Server** (colocated with RL framework on GPU servers):
- Orchestrates the RL training loop
- Manages tasks, batches, communication
- **Exposes an OpenAI-like API for the updated model** (Pydantic-typed)
- Inventory-style task dispatch: maintains a record of available tasks; assigns to clients as they become ready
- **Each task gets a unique OpenAI-like endpoint URL**, passed to the client alongside the task

**Lightning Client** (no GPU required; runs anywhere):
- Two sub-components:
  - Communication module: data upload/download to/from server
  - Runtime: executes the agent and collects traces
- Connects to server via the unique per-task API endpoint
- Reports rewarded traces back

**The event-driven control flow** (paper Appendix B sequence diagram):

```
1. Client uploads dataset to Server
2. Server starts RL framework, loads initial model
3. for each batch in dataset:
     for each task in batch:
       Server → Client: (task, task-specific API URL)
       Client launches agent execution
       for each LLM call in agent run:
         Agent → API URL → Server (forwards to vLLM/SGLang) → Agent
       Client collects trace + reward → Server
     Server → RL Framework: batch of trajectories
     RL Framework updates weights → Server publishes new model
```

The mutual independence is the key property:

> [!quote] Mutual independence between trainer and agent (paper §3.4.1)
> "This design renders the training framework (e.g., VeRL) **agent-agnostic**; its sole concern is the optimization of the LLM and management of hardware resources. Conversely, the agent, operating on the client side, is **trainer-agnostic**, allowing it to function independently of the training framework's implementation details."

**Data Parallelism for agent execution** is a two-level strategy:

- **Intra-node parallelism**: one Client process spawns multiple worker subprocesses on a single machine, each running an agent instance.
- **Inter-node parallelism**: multiple Client processes on different machines, each with their own workers.

This lets rollout throughput scale to whatever the agent execution + tool I/O can handle, independent of the training cluster size.

### Supporting machinery (skim or skip)

> [!note]- Automatic Intermediate Rewarding (AIR) — solving sparse reward in agents
>
> Long-horizon agent tasks suffer from sparse rewards (only the final task succeeds or fails). AIR is Agent Lightning's mechanism for **converting system monitoring signals into intermediate rewards** without human annotation.
>
> Concrete signals available:
> - Tool call return status (success / fail / timeout)
> - Code execution exit codes
> - Compilation errors
> - Network response codes
>
> The client maps these signals to per-transition reward components, additive to the final outcome reward. Developers customize the mapping per task type.
>
> The paper acknowledges this is a "training-side workaround for the sparse reward problem" — the cleaner solution (verifier-based PRMs) is positioned as future work.

> [!note]- Data capture without code modification — two paths
>
> Two instrumentation approaches the Client provides:
>
> 1. **OpenTelemetry + AgentOps** (preferred). The Client uses OpenTelemetry's tracing capabilities to automatically instrument agent code, capturing execution traces, LLM calls, and environment interactions. **Compatible with the production observability stack** (Langfuse, Phoenix, Datadog APM, AWS X-Ray, etc.) — any team already using OpenTelemetry for agent observability gets RL data collection for free.
> 2. **Basic tracing in the OpenAI-like API endpoint** (fallback). For agents that can't be instrumented with OpenTelemetry, the Client provides a basic capture mechanism embedded in the API endpoint itself. Lightweight; works without external dependencies.
>
> The OpenTelemetry path is the more interesting design choice — it positions Agent Lightning to integrate with the existing agent observability ecosystem, which has matured separately from RL training infrastructure.

> [!note]- Error handling and robustness
>
> The Client implements comprehensive error handling for RL-specific failure modes (more frequent than inference-only failures because of exploration):
>
> - Agent code crashes — detected, logged, task retried or reassigned to another worker
> - Network interruptions — tool calls with timeouts and retries
> - Long-hanging tool calls — wall-clock budget per task
> - Invalid outputs — logged for debugging, transition discarded
>
> Failed tasks don't disrupt overall training; the design assumes failures are common and engineers around them by isolating per-task state.

> [!note]- Environment and Reward Services for scalability
>
> Environments and reward functions vary in cost. The Client supports two deployment modes:
>
> 1. **In-worker** — light environments (e.g., calculator, small SQL DB) run inside the same worker as the agent.
> 2. **Pooled service** — heavy environments (mobile phone emulators, complex reward computation) are hosted as shared services that all workers connect to.
>
> Pooled services amortize initialization cost (e.g., loading a 10GB reward model once vs. per-worker) and reduce memory pressure. The paper notes this could be extended to "more sophisticated serverless architectures" but currently uses a simple pooling design.

## Headline evidence

**Setup.** Llama-3.2-3B-Instruct base model. RL algorithm not explicitly stated as PPO vs GRPO, but the paper cites both and discusses GRPO advantage normalization in the LightningRL framework. Three tasks, three different agent frameworks:

| Task | Framework | Dataset | Tool used | Total agents | Tuned agents |
| ---- | --------- | ------- | --------- | -----------: | -----------: |
| Text-to-SQL | LangChain | Spider | SQL executor | 3 | **2** |
| Open-domain QA | OpenAI Agents SDK | MuSiQue | Wikipedia retriever | 1 | 1 |
| Math QA | AutoGen | Calc-X | Calculator | 1 | 1 |

**The main results** (extracted from Figs. 5-7):

| Task | Train reward (start → end) | Test reward (start → end) |
| ---- | -------------------------: | ------------------------: |
| **Text-to-SQL (Spider)** | 0.0 → 0.65 | 0.15 → 0.55 |
| **MuSiQue (RAG)** | 0.05 → 0.40 | 0.0 → 0.22 |
| **Calc-X (Math QA)** | 0.05 → 0.85 | 0.05 → 0.78 |

> [!success] What the experiments actually show
> The numbers themselves are modest (3B model, learning-friendly tasks). The contribution is structural: **the same RL pipeline trained three different agents on three different frameworks without per-framework engineering**. That's the headline.

**The critical "ablation" is hidden in the multi-agent setup.** The text-to-SQL pipeline has 3 agents but only optimizes 2 — the third is held constant. LightningRL handles this by simply *not including the third agent's transitions in the gradient update*. With a masking-based approach, this would require careful mask construction for every multi-agent topology; here, it falls out for free.

> [!example]- All experimental results — full details
>
> **Text-to-SQL on Spider** (§4.1)
>
> The agent has 3 sub-agents implemented in LangChain: SQL writer, SQL checker, re-writer. The SQL writer generates a query; if execution fails, the checker decides whether to rewrite or directly answer; the re-writer either revises the query or answers from retrieved info. **Only the writer and re-writer are tuned**; the checker uses the same LLM but is held frozen. Reward = whether the final answer is correct.
>
> Llama-3.2-3B-Instruct starts at near-zero reward (failures dominated by syntax errors), climbs steadily to ~0.65 train / ~0.55 test over 400 steps. The test curve plateaus around step 200 (early convergence).
>
> **MuSiQue (RAG) on OpenAI Agents SDK** (§4.2)
>
> Multi-hop QA over Wikipedia (21M documents, BGE embeddings, cosine retrieval). The agent generates a search query, decides whether to refine or answer based on retrieved passages. Reward = $0.9 \cdot R_{\text{correctness}} + 0.1 \cdot R_{\text{format}}$ where format reward requires `<think>...</think>`, `<query>...</query>`, `<answer>...</answer>` structure.
>
> Test reward climbs from ~0 to ~0.22 over 200 steps. MuSiQue is genuinely hard (multi-hop, large retrieval corpus, free-form queries) and a 3B model is undersized; the absolute number isn't the point.
>
> **Calc-X (Math QA) on AutoGen** (§4.3)
>
> Math problems requiring calculator invocations. Single LLM agent that decides when to call the calculator and integrates results. Reward = answer correctness.
>
> The strongest result: train ~0.85, test ~0.78 over 450 steps. Tool-use tasks with deterministic verification are the easiest RL setting, and the curve reflects this.

## Strengths and limitations

**Strengths.**

- The **MDP formulation is the cleanest in the literature.** Casting one LLM call as one action, with semantic-variable state, makes the math work out for free with standard single-turn RL methods.
- **OpenTelemetry-native** is a real differentiator — production agent teams get to reuse their observability stack.
- **Trainer-agnostic + agent-agnostic** is a property, not just an aspiration; the architecture genuinely enforces it through the OpenAI-API boundary.
- **Multi-agent selective optimization** falls out of the abstraction without bespoke masking.

**Limitations.**

> [!warning] The credit assignment story is incomplete
> The paper is honest about this (§3.3.2): the current LightningRL implementation assigns the same value to every action in an episode, equal to the final return. For long-horizon agents where intermediate actions vary in importance, this conflates exploration with execution and **invites credit-misattribution-style reward hacking** — the same failure mode [[polar|Polar]]'s `per_request` mode hits. A learned value head is named as future work.

- **Tested only at 3B.** Llama-3.2-3B-Instruct is small. Scaling to 7B / 14B / 70B isn't shown; behavior under model-size scaling is unknown.
- **Tasks chosen are RL-friendly.** Spider / MuSiQue / Calc-X all have automated correctness checks. No SWE-Bench, no GAIA, no OSWorld. The "ANY AI agent" claim is technically supported but at the easy end of the difficulty spectrum.
- **"Almost zero code modification" requires wrapping the agent function in `Client`.** Real overhead is small (~5 lines, see Listing 2 below) but it's nonzero. Agents whose code can't be modified at all (e.g., closed-source binaries like Claude Code) **can't be trained** — that's the gap [[polar|Polar]] later filled with the URL-swap trick.
- **No head-to-head comparisons** with other agent-RL frameworks. The paper positions vs. SkyRL / VeRL-Tool / rLLM / GEM in the related-work section but doesn't run them on the same tasks.
- **Multi-LLM multi-agent setting only briefly discussed** (§3.2.2 mentions MARL extension); experiments are single-LLM-multi-role.

## What this means

Agent Lightning is the **paper that named the problem correctly**. Before it, every multi-turn-RL framework was a special case of "rebuild the agent inside the trainer"; afterwards, the disaggregated client-server pattern became the obvious target. ProRL Agent (March 2026) took the same architectural principle and made the rollout side a polished production service. Polar (May 2026) pushed the same idea further by replacing the Python-API integration boundary with an HTTP-API proxy boundary, removing the last code-modification requirement.

Three predictions for 2027:

1. **Agent Lightning's MDP-with-LLM-call-as-action becomes the canonical formalism.** Already, both ProRL Agent and Polar use essentially this abstraction (with different surface presentation). The math is too clean to lose.
2. **OpenTelemetry integration becomes table-stakes.** Production agent teams won't accept an RL framework that doesn't speak the observability protocol they already use. Polar/ProRL Agent will likely add OpenTelemetry support; veRL / NeMo-RL will need adapters.
3. **The LightningRL credit-assignment placeholder becomes the bottleneck.** Identity assignment works for short trajectories but breaks on long-horizon SWE-Bench-style tasks. Whoever ships a robust per-action value head with token-level GAE on top of LightningRL's transition format publishes the next obvious paper.

What Agent Lightning does *not* solve:

- **Closed-source / black-box agent training**: requires Python-API wrapping. Polar's URL-swap is the answer.
- **Strict token-faithfulness across re-tokenization**: trace capture uses text-level wire format, so the trained LLM's tokenizer must match what the server-side inference engine produces. ProRL Agent's token-in/token-out wire protocol addresses this; Polar generalizes it.
- **Production-scale empirical validation**: 3B + Calc-X is research-scale. Polar's SWE-Bench-Verified numbers on Qwen3.5-4B/8B/14B are the production-scale demonstration the field needed.

## Source code & reproduction

```bash
# Install
pip install agentlightning

# Or nightly from Test PyPI
pip install --index-url https://test.pypi.org/simple/ agentlightning
```

**Minimal end-to-end example** (paper Appendix A, Listing 2):

```python
# train.py
from agent import agent_function
from environments.game_server import GameServer
from agent_lightning import Client, Resource, Task

game_server = GameServer()

# Task corresponds to data in `.jsonl`; Resource contains LLM endpoint
def agent_run(resource: Resource, task: Task):
    game_url = game_server.create_game(task.game_seed)
    answer = agent_function(resource.model_api, game_url)
    reward = game_url.score(answer, task.ground_truth)
    return reward

client = Client(os.environ["AgentLightningServerUrl"])
client.upload_data("data/train.jsonl", test_file=None)
client.train(agent_run, nworkers=-1)
```

The original agent code (`agent.py`) is unmodified; the new file (`train.py`) wraps `agent_function` in the Client interface.

**Folder structure for an Agent Lightning project** (Listing 1):

```
agent/
├── data/train.jsonl              # Task dataset
├── environments/game_server.py   # Tool / env servers
└── agent.py                      # Original agent (unchanged)
```

| File path | Role |
| --------- | ---- |
| `agentlightning/client.py` | `Client` class — the user-facing wrapper |
| `agentlightning/server.py` | `Server` class — orchestrates RL framework |
| `agentlightning/runtime.py` | Agent execution + trace capture |
| `agentlightning/algorithms/` | LightningRL implementation + GRPO/PPO adapters |
| `dashboard/` | Monitoring UI (TypeScript, real-time trace inspection) |
| `examples/` | Reference integrations: LangChain, AutoGen, OpenAI SDK |

**External integrations validated by the community**:

- **Youtu-Agent** — verified up to 128-GPU RL training via a modified branch
- **DeepWerewolf** — multi-agent Werewolf game using AgentScope + Agent Lightning
- **vLLM blog (Oct 2025)** — *"No More Retokenization Drift: Returning Token IDs via the OpenAI Compatible API Matters in Agent RL"* discusses Agent Lightning's integration with vLLM, addressing the tokenizer-mismatch failure mode

## Related reading

- [[prorl-agent]] — ProRL Agent: NVIDIA's March-2026 framework that builds on Agent Lightning's disaggregation principle, adding a polished production-grade rollout service with rootless Singularity sandbox and Slurm-native deployment. Same architectural family.
- [[polar]] — Polar: The May-2026 successor to ProRL Agent that takes Agent Lightning's "almost zero code modification" claim to **"truly zero code modification"** via LLM-API proxy + URL swap. Trains closed-source agents (Codex, Claude Code) that Agent Lightning cannot reach.
- [[search-r1]] — Search-R1: the canonical earlier-2025 agentic-RL paper. Application-specific (tool-use for retrieval), uses sequence-concatenation + retrieved-token loss masking — the exact pattern Agent Lightning argues against.
- [[nemo-gym]] — NVIDIA's RL environment catalog; complements rather than competes with Agent Lightning (env layer vs. rollout-driver layer).
- [[agentic-rl-foundations]] — Onboarding hub for agentic RL; positions Agent Lightning in the broader landscape.
- [[on-policy-distillation]] — Alternative paradigm: dense teacher KL replaces RL entirely. Mutually exclusive direction.
- [[grpo]], [[ppo-for-llm]] — The single-turn RL algorithms LightningRL's per-action loss plugs into.
- [[rl-training-frameworks]] — Context: where Agent Lightning sits among veRL / OpenRLHF / TRL / ROLL.

## References

- Xufang Luo et al. *Agent Lightning: Train ANY AI Agents with Reinforcement Learning.* arXiv:2508.03680, August 2025. https://arxiv.org/abs/2508.03680
- microsoft/agent-lightning. https://github.com/microsoft/agent-lightning
- Documentation. https://microsoft.github.io/agent-lightning/
- OpenTelemetry. https://opentelemetry.io/
- AgentOps. https://www.agentops.ai/
- vLLM blog: *No More Retokenization Drift.* https://vllm.ai/blog/2025-10-22-agent-lightning-vllm (October 22, 2025)
