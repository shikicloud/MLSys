---
title: "ThunderAgent: A Simple, Fast and Program-Aware Agentic Inference System"
category: llm-serving-for-agents
tags: [thunderagent, agent-serving, program-abstraction, kv-cache, scheduling, vllm, sglang, openhands, swe-bench, icml-2026, paper-review]
created: 2026-06-15
updated: 2026-06-15
status: mature
paper: arXiv:2602.13692
code: https://github.com/ThunderAgent-org/ThunderAgent
---

# ThunderAgent: A Simple, Fast and Program-Aware Agentic Inference System

> [!info] Paper metadata
> - **Paper**: [arXiv:2602.13692](https://arxiv.org/abs/2602.13692) — *ThunderAgent: A Simple, Fast and Program-Aware Agentic Inference System*, v1 2026-02-14, v2 2026-03-10
> - **Venue**: **ICML 2026 Spotlight (top 2.2%)**
> - **Code**: [ThunderAgent-org/ThunderAgent](https://github.com/ThunderAgent-org/ThunderAgent) (MIT license, vLLM + SGLang backends, OpenAI-compatible API)
> - **Authors**: Hao Kang\*¹ (Georgia Tech), Ziyang Li\*² (Individual Researcher), Xinyu Yang\*³ (CMU), Weili Xu\*⁴ (UIUC), Yinfang Chen⁴ (UIUC), Junxiong Wang⁵ (Together AI), Beidi Chen³ (CMU), Tushar Krishna¹ (Georgia Tech), Chenfeng Xu⁵ (Together AI), Simran Arora⁵ (Together AI). \*Equal contribution. Correspondence to `hkang342@gatech.edu`.
> - **Affiliations**: Georgia Tech, CMU, UIUC, Together AI

> [!important] Where ThunderAgent fits in the agent-serving stack
> ThunderAgent is **simultaneously a competitor and a foundation** in the multi-turn agent serving landscape:
>
> - **Vs. [[continuum|Continuum]]**: contemporary competitor solving the same KV-cache-management problem with a fundamentally different approach. Continuum predicts tool durations and pins KV via TTL; ThunderAgent abstracts the workflow as an **agentic program**, monitors thrashing periodically, and pauses/restores programs adaptively. ThunderAgent **beats Continuum on all 6 benchmarks** (SWEAgent, OpenHands, ToolOrchestra, ScienceAgent across GLM-4.6 + Qwen3 models), often by 2-3× margin where Continuum's TTL predictions fail on stochastic tool calls.
> - **Vs. [[mori|MORI]]**: **ThunderAgent is the system layer MORI runs on top of** — MORI's ~3,300 lines of Python sit atop ThunderAgent + 500 lines on SGLang HiCache. MORI's two-tier idleness scheduling is orthogonal to ThunderAgent's program abstraction.
> - **Vs. [[infercept|InferCept]]**: extends InferCept's "interception is a first-class event" thesis to the *workflow* level. Where InferCept managed KV per-request with the Discard/Preserve/Swap taxonomy, ThunderAgent abstracts the entire multi-turn workflow as a program $P = \langle ID, c, \mathcal{T}, \mathcal{L}, \tau, s \rangle$ and schedules at program granularity.
>
> If you read one paper for understanding the current state of agent serving systems, this is one of the top three (alongside Continuum and MORI).

---

## Summary (read this if you have 2 minutes)

**What it is.** ThunderAgent (ICML 2026 Spotlight) is a **program-aware agentic inference and rollout library** that sits between agent clients and inference backends (vLLM, SGLang) and brokers KV cache + tool resources at the *workflow* granularity rather than per-request. It exposes an OpenAI-compatible API with a single extra `program_id` parameter, and the rest of the system tracks each multi-turn workflow as a first-class "agentic program" through its lifecycle.

**The one idea.** **Treat agentic workflows as LLM Programs.** A program $P = \langle ID, c, \mathcal{T}, \mathcal{L}, \tau, s \rangle$ unifies four things existing systems track separately:

1. **$c$** = current context length (KV cache footprint)
2. **$\mathcal{T}$** = set of tool environments (Docker sandboxes, network ports) the workflow depends on
3. **$\mathcal{L}$** = backend GPU node placement
4. **$\tau, s$** = execution phase ($\tau \in \{R, A\}$ = Reasoning / Acting) and scheduling status ($s \in \{$Active, Paused, Terminated$\}$)

Three sub-pieces make this abstraction load-bearing:

1. **Program-aware scheduler** — periodic thrashing monitor at fixed $\Delta t = 5$ s detects when `C_total < ∑ c_p` and triggers **Pause** (evict Acting programs with shortest context first, score $S_{pause}(P) = 1/c_P + \mathbb{1}(\tau = A)$) or **Restore** (admit Reasoning programs with shortest context first, score $S_{restore}(P) = 1/c_P + \mathbb{1}(\tau = R)$). The shortest-first eviction is **provably optimal** (Lemma 4.1: recompute cost is quadratic in context length $c_i^2$).
2. **Global program-aware waiting queue** across data-parallel backends — replaces per-node KV-aware routing (vLLM/SGLang behavior that causes 51% memory imbalance over 90 min) with workflow-level load balancing. Paused KV is node-agnostic so any backend with capacity can restore.
3. **Program-aware tool resource management** — **hook-based garbage collection** triggers immediate sandbox/port teardown on `Terminated` status (keeps disk usage near-constant vs. linear growth in baselines), and **asynchronous environment preparation** initializes Docker containers / package installs in parallel with the LLM phase of a high-priority Restoring program (hides 29-47 s init latency).

Remove the program abstraction and the scheduler can't distinguish "this request is mid-workflow, returning soon" from "this is a brand-new request" — back to vLLM/SGLang behavior. Remove the global queue and you get 51% cross-node imbalance. Remove hook-based GC and disk fills up after ~250 workflows. Remove async env prep and end-to-end latency takes the full 47 s setup cost on the critical path.

**Headline result.** Six benchmarks on 8×H100 (GLM-4.6 355B MoE, Qwen3-235B, Qwen3-8B):

| Workload | vs. vLLM | vs. Continuum |
| -------- | -------: | ------------: |
| SWEAgent / GLM-4.6 | **2.65×** | 1.52× |
| OpenHands(code) / GLM-4.6 | **3.58×** | 1.08× |
| ToolOrchestra(HLE) / Qwen3-8B | **1.48×** | 0.65× ← Continuum *loses* to vLLM here |
| SWEAgent / Qwen3-235B | **3.02×** | 1.44× |
| OpenHands(code) / Qwen3-235B | **2.43×** | 1.22× |
| ScienceAgent / GLM-4.6 | **1.24×** | 1.06× |

Plus:
- **RL rollout: 1.79–3.92× improvement** over prior SOTA on distributed GPU nodes.
- **Disk memory: up to 4.2× savings** (near-constant vs. linear growth).
- **KV cache hit rate**: ThunderAgent maintains ~100% on predictable-tool workloads (a, b, d, e); Continuum drops to ~60% under high concurrency.

**Why it matters.**

- **First system to treat the entire multi-turn workflow as a scheduling unit.** vLLM/SGLang/InferCept schedule at the request granularity; Continuum partially aggregates via TTL but still pins to one node. ThunderAgent's $P$ tuple is the cleanest "program-as-scheduling-primitive" formulation in the literature.
- **The closed-form quadratic-cost result (Lemma 4.1) settles the eviction question.** Recompute cost $\propto c_i^2$ means **always evict the shortest-context program**. This is a small but durable theorem that should appear in every agent-serving scheduler going forward.
- **The system that everything else builds on.** MORI's two-tier idleness scheduler is built directly on ThunderAgent. Subsequent agent-serving systems will either build on ThunderAgent's program abstraction or invent an equivalent.
- **Beats Continuum decisively on workloads with stochastic tool execution.** Continuum's TTL prediction model fails when tool latencies are heavy-tailed (ToolOrchestra panel c — Continuum is 0.65× vs vLLM, i.e. *slower*). ThunderAgent's time-decay weighting $f(t) = 2^{-t}$ on Acting programs handles both deterministic and stochastic regimes.
- **2027 prediction.** "Program ID" becomes a first-class field in vLLM's `Request` struct. Per-program scheduling APIs become standard. The Continuum-vs-ThunderAgent debate (predictive TTL vs reactive thrashing monitor) becomes a textbook trade-off discussion.

---

# Depth (drill-down starts here)

## Background: three failure modes of "request-aware" systems

Today's stack — vLLM/SGLang for inference, Kubernetes/Docker for tools — schedules at request granularity. Each LLM call is independent; each tool invocation is independent. The paper's Figures 1 and 2 quantify three failures of this architecture.

![Performance comparison of ThunderAgent against prior agent inference systems as parallel workflow number increases. (a) Throughput degradation. (b) KV cache thrashing (E2E latency and KV hit rate). (c) Speedup across SWE-Agent, OpenHands, ToolOrchestra. (paper Fig. 1)](EN/wiki/llm-serving-for-agents/thunderagent-figs/fig1-motivation.png)

### Failure 1: KV cache thrashing — **7.14× E2E latency blow-up**

Request-aware engines evict KV cache during tool execution intervals to make room for newly arrived requests. When the tool call completes, the entire history must be **re-prefilled** (because each turn appends to the previous full context). The paper measures **7.14× E2E latency increase** at high concurrency from this thrashing alone — and visually in Figure 1b, vLLM's KV cache hit rate collapses from ~80% to ~30% as parallel workflow count goes from 24 → 96, while E2E latency triples.

This is the same problem [[continuum|Continuum]] addresses with TTL pinning and [[infercept|InferCept]] addresses with the Preserve action — but those solutions assume you can *predict* the tool duration (Continuum) or that the interception is per-request (InferCept). Real agentic workflows like ToolOrchestra interleave many tools with unpredictable durations (web APIs from milliseconds to minutes).

### Failure 2: Cross-node memory imbalance — **51% peak imbalance**

vLLM's KV-aware router and SGLang's prefix-aware router both **greedily route requests with the same prefix to the same DP node** to maximize cache reuse. Reasonable for chatbot workloads with diverse prompts; pathological for agent workloads where **every workflow starts with the same system prompt + tool definitions**.

The paper's measurement (Figure 2a): on 90-minute OpenHands RL rollout with two 8×H100 nodes, memory imbalance between Node 0 and Node 1 exceeds 20% for **37+ minutes** and peaks at **51%**. Half the GPU memory is idle while the other half thrashes.

### Failure 3: Tool lifecycle obliviousness — **47 s env prep, linear disk growth**

Tool orchestrators (Kubernetes, Docker) don't know when the *agentic workflow* is done — only when individual `docker run` calls return. Result:

- **Disk usage grows linearly** with processed workflows (Figure 2b): unused Docker images and stopped sandboxes from finished workflows accumulate forever. After ~250 workflows, disk usage exceeds the 2 TB capacity.
- **Environment preparation time grows from 29.9 s → 47.2 s** as parallel workflow count goes 24 → 96 (Figure 2c). When this is on the critical path of every workflow start, throughput craters.

## The agentic program abstraction

Section 4.1 defines an **agentic program** $P$ as the tuple:

$$P = \langle ID, c, \mathcal{T}, \mathcal{L}, \tau, s \rangle \quad (1)$$

| Symbol | Meaning |
| ------ | ------- |
| $ID$ | unique global identifier; clients pass this as `program_id` in the OpenAI-compatible API |
| $c$ | number of tokens in current context (= KV cache footprint) |
| $\mathcal{T}$ | set of tool environments (Docker sandboxes, network ports, database connections) |
| $\mathcal{L}$ | backend (GPU node) placement for spatial locality |
| $\tau$ | execution phase: **R**easoning (LLM decoding) or **A**cting (waiting on tool) |
| $s$ | scheduling status: Active, Paused, Terminated |

Two **primitive operations** transition program state:

- **Restore$(P)$**: $\langle ID, c, \mathcal{T}, \emptyset, \tau, \text{Paused}\rangle \to \langle ID, c, \mathcal{T}, \mathcal{L}', \tau, \text{Active}\rangle$ — admit into backend $\mathcal{L}'$ with available capacity.
- **Pause$(P)$**: $\langle ID, c, \mathcal{T}, \mathcal{L}, \tau, \text{Active}\rangle \to \langle ID, c, \mathcal{T}, \emptyset, \tau, \text{Paused}\rangle$ — release KV cache; tool environments stay alive on disk so resumption is fast.

This abstraction is the conceptual hinge. The cost model (next section) becomes a tractable optimization once you have a program-level state vector to optimize over.

## The cost model

Section 4.2 uses the **Space-Time Product (STP)** as the canonical cost metric:

$$\text{Cost}_x = \int_0^{t_x} M_x(t) \, dt \quad (2)$$

Since KV memory $M_x(t)$ is directly the token count, the cost reduces to **integral of token count over time**. Total agentic inference cost decomposes into five terms:

$$\text{Cost}_{\text{total}} \approx \text{Cost}_{\text{decode}} + \text{Cost}_{\text{prefill}} + \text{Cost}_{\text{recompute}} + \text{Cost}_{\text{unused}} + \text{Cost}_{\text{caching}} \quad (3)$$

| Term | Productive? | Sources |
| ---- | :---------: | ------- |
| $\text{Cost}_{\text{decode}}$ | ✓ | Active token generation |
| $\text{Cost}_{\text{prefill}}$ | ✓ | First-turn or new-context prefill |
| $\text{Cost}_{\text{recompute}}$ | ✗ | KV thrashing → re-prefill on resume |
| $\text{Cost}_{\text{unused}}$ | ✗ | Memory imbalance across DP nodes |
| $\text{Cost}_{\text{caching}}$ | ✗ | KV pinned during long tool calls |

The scheduler's objective: **minimize $\text{Cost}_{\text{recompute}} + \text{Cost}_{\text{unused}} + \text{Cost}_{\text{caching}}$**. Each of ThunderAgent's three components attacks one term.

> [!important] Lemma 4.1 (Quadratic Recomputation Cost)
> For a program $P_i$ with context length $c_i$:
>
> $$\text{Cost}_{\text{recompute}} = \int_0^{t_{\text{recompute}}} c_i(t) \, dt \propto c_i^2 \quad (8)$$
>
> The intuition: recompute time scales linearly with $c$, and during that time the memory held is also $c$ → integral is $c^2$.
>
> **Consequence**: when forced to evict programs to release $\Delta C$ memory, the optimal subset $S$ minimizes $\sum_{i \in S} c_i^2$ subject to $\sum_{i \in S} c_i \geq \Delta C$. **Greedy answer: evict the shortest-context programs first.** This is the formal justification for ThunderAgent's shortest-first eviction policy.

## Three components in detail

![ThunderAgent architecture overview: global waiting queue feeds N inference backends; periodic thrashing monitor triggers Pause/Restore across backends with program-aware metadata. (paper Fig. 3)](EN/wiki/llm-serving-for-agents/thunderagent-figs/fig3-architecture.png)

### Component 1 — Program-aware scheduler with periodic thrashing monitor (§4.3.1)

Unlike Continuum's admission-time TTL prediction (one-shot at workflow arrival), ThunderAgent runs a **periodic monitor every $\Delta t = 5$ s** that evaluates the thrashing condition on each backend $\mathcal{L}$:

$$C_{\text{total}} < \sum_{p \in \mathcal{L}} c_p \quad (6)$$

When the sum of context lengths exceeds the backend's KV capacity, thrashing is imminent. With a hysteresis window controlled by high/low watermarks $\lambda_{\max}, \lambda_{\min}$ (default both = 1.0), the scheduler:

- **Pauses** Acting programs with shortest context first (score $S_{\text{pause}}(P) = 1/c_P + \mathbb{1}(\tau = A)$) until $\sum c_p < \lambda_{\max} \cdot C_{\text{total}}$.
- **Restores** Reasoning programs with shortest context first (score $S_{\text{restore}}(P) = 1/c_P + \mathbb{1}(\tau = R)$) when $\sum c_p < \lambda_{\min} \cdot C_{\text{total}}$.

The indicator functions encode a critical priority: **pause Acting before Reasoning** (Acting programs aren't producing tokens anyway, so pausing them costs nothing but caching), and **restore Reasoning before Acting** (Reasoning programs immediately produce decoded tokens, maximizing throughput).

A subtle but important refinement: instead of binary pause/keep, the scheduler applies a **time-decay function** $f(t) = 2^{-t}$ to acting programs' token weight in the thrashing check:

$$C_{\text{total}} < \sum_{p \in \mathcal{L}, \tau = R} c_p + \sum_{q \in \mathcal{L}, \tau = A} c_q \times f(t_q) \quad (7)$$

The longer a program has been Acting (long-running tool call), the lower its effective weight, so the scheduler progressively prefers to evict it. The paper proves (Appendix E.1) that **exponential decay is optimal when tool execution times satisfy the memoryless property** — i.e., the remaining time is independent of elapsed duration. This is the formal counterpart to Continuum's TTL expiry, but reactive rather than predictive.

### Component 2 — Global program-aware waiting queue (§4.3.2)

Cross-node memory imbalance (Failure 2 above) arises because paused programs sit in *per-node* waiting queues, so a node with idle memory can't accept work from an overloaded node's queue. ThunderAgent's fix: **one global queue across all DP replicas**.

The insight that makes this safe: **once a program is paused, its KV cache is gone — so it's node-agnostic on restore**. Routing a paused program to a different node doesn't lose any cache benefit. This is the opposite of vLLM's KV-aware router (which sticks programs to nodes for cache locality) but is correct because the *cache locality is already destroyed by pausing*.

Formal bound: $\text{Cost}_{\text{unused}} < c_{\min} \cdot \Delta t$ per node, where $c_{\min}$ is the minimum token length among paused programs. So memory imbalance is bounded by the smallest paused program's footprint × monitoring interval — small.

### Component 3 — Tool resource management (§4.4)

Two mechanisms working in tandem:

**Hook-based garbage collection.** Lifecycle hooks couple tool resource teardown to program scheduling status. When $s = \text{Terminated}$, the collector immediately tears down sandboxes, network sockets, and compute slots. Figure 2b shows the result: active disk usage stays near-constant (~0.5 TB) regardless of how many workflows have been processed, while the request-aware baseline grows linearly past the 2 TB capacity at ~300 workflows.

**Asynchronous environment preparation.** The slow path of tool environment setup (29.9-47.2 s for Docker image pulls and dependency installs) is overlapped with LLM reasoning. The scheduler monitors the global waiting queue; when a high-priority program (high $S_{\text{restore}}$) approaches the restoration threshold, the system **asynchronously starts spinning up its tool environment before GPU memory is even allocated**. By the time the LLM is ready to decode, the environment is ready too.

This is essentially **prefetching** applied to agent infrastructure — orthogonal to the LLM prefetching that systems like [[speccache|SpecCache]] do for action observations.

## Headline evidence

![ThunderAgent vs vLLM and Continuum across 6 workloads × 3 models × 2-4 datasets. ThunderAgent outperforms by 1.24-3.58× on serving throughput; Continuum *loses* to vLLM on stochastic tool workloads (c, f). (paper Fig. 4)](EN/wiki/llm-serving-for-agents/thunderagent-figs/fig4-serving-results.png)

Six panels, all 8×H100 nodes (except ToolOrchestra on RTX 5090):

| Panel | Workload | Model | vLLM (baseline) | Continuum | ThunderAgent | TA vs Continuum |
| ----- | -------- | ----- | --------------- | --------- | ------------ | --------------- |
| (a) | SWEAgent | GLM-4.6 | 1× | 1.52× | **2.65×** | 1.74× faster |
| (b) | OpenHands(code) | GLM-4.6 | 1× | 1.08× | **3.58×** | 3.31× faster |
| (c) | ToolOrchestra (HLE) | Qwen3-8B | 1× | **0.65×** ← Continuum loses to vLLM | **1.48×** | 2.28× faster |
| (d) | SWEAgent | Qwen3-235B | 1× | 1.44× | **3.02×** | 2.10× faster |
| (e) | OpenHands(code) | Qwen3-235B | 1× | 1.22× | **2.43×** | 1.99× faster |
| (f) | ScienceAgent | GLM-4.6 | 1× | 1.06× | **1.24×** | 1.17× faster |

**Panel (c) is the critical evidence.** ToolOrchestra on HLE (Humanity's Last Exam) has *highly stochastic* tool durations — web API latency varies orders of magnitude. **Continuum's TTL predictions break under this stochasticity** and the system over-pins KV, leading to worse performance than vanilla vLLM. ThunderAgent's reactive thrashing monitor handles both deterministic and stochastic tool patterns.

**RL rollout.** Same workflows on 2× 8×H100 nodes for RL data collection: **1.79–3.92× improvement** over the prior SOTA combination of vLLM + SGLang Gateway (the leading distributed RL rollout setup at the time of publication).

> [!example]- KV cache hit rate breakdown (paper Fig. 5)
> ThunderAgent maintains **~100% KV hit rate** on workflows with predictable tool call times (a, b, d, e) up to 192 parallel workflows. On stochastic workloads (c, f) ThunderAgent dynamically trades hit rate for less idle caching — still beats vLLM and Continuum. Continuum's hit rate drops from >90% to ~60% under high concurrency because over-pinned KV starves new admissions.

## Limitations and open questions

**Limitations the paper acknowledges (or doesn't).**

- **Single-cluster only.** $\mathcal{L}$ assumes all backends are reachable from one scheduler. Geo-distributed deployments (per [[prfaas|PrFaaS]]) need a hierarchy on top.
- **No SLO/priority discussion.** All programs are treated equally; there's no notion of "this program is the user-facing one, prioritize it." Real production needs this.
- **The time-decay constant** $f(t) = 2^{-t}$ is hard-coded; the paper proves optimality under memoryless tool times but doesn't measure sensitivity for non-memoryless distributions.
- **Δt = 5 s monitoring interval** is a fixed config — no adaptive mechanism for bursty arrival patterns.
- **Hook-based GC assumes well-behaved tools** that respect SIGTERM/cleanup hooks. Tools that fork detached processes or leak resources outside the sandbox aren't covered.
- **No fairness across programs.** A program with a quickly-growing context can starve smaller programs (shortest-first eviction is greedy in the wrong direction for fairness).
- **No comparison vs InferCept.** The lineage from InferCept's Discard/Preserve/Swap is not explicitly engaged with — InferCept appears only as background.
- **Workload coverage**: SWE-Agent (coding), OpenHands (coding + science), ToolOrchestra (routing), ScienceAgent (science). No GUI agents, no multi-modal agents, no human-in-the-loop chatbot benchmarks.

## Strengths

- **The program abstraction is the cleanest one in the literature.** $P = \langle ID, c, \mathcal{T}, \mathcal{L}, \tau, s \rangle$ is durable — it'll appear in textbooks.
- **The quadratic-cost lemma settles eviction.** No more "should we evict by LRU or by size?" debates — the answer is by *size squared*, and the greedy result is *shortest-first*.
- **Wins on all 6 benchmarks, including the stochastic ones.** Continuum's panel (c) and (f) losses are damning for the predictive-TTL approach; ThunderAgent's reactive approach handles both.
- **OpenAI-compatible API with one extra parameter.** Integration is trivial — clients add `program_id` to their requests, nothing else changes.
- **Open source under MIT with vLLM + SGLang backends.** No vendor lock-in.
- **System layer for MORI.** Anything MORI does works on top of ThunderAgent's primitives, so future two-tier / multi-tier offloading research builds on this stack.

## Implementation notes

- **Architecture**: scheduler process sits in front of one or more vLLM/SGLang backends. Communicates via HTTP (OpenAI-compatible) plus internal RPC for Pause/Restore.
- **Periodic monitor**: thread sweeps backend states every $\Delta t = 5$ s.
- **Backends**: vLLM and SGLang supported out of the box. Tool orchestrator backend is configurable (Kubernetes, Docker Compose, custom).
- **Code size**: small (a scheduler + minor wrapper) — MORI adds ~3,300 lines on top, which gives a sense of ThunderAgent's own LoC.
- **Hyperparameters**: $\Delta t = 5$ s, $f(t) = 2^{-t}$, $\lambda_{\max} = \lambda_{\min} = 1.0$ (no hysteresis by default).
- **Models tested**: GLM-4.6 (355B MoE, FP8, TP=8), Qwen3-235B (FP8, TP=8), Qwen3-8B (FP16).
- **Hardware tested**: 8×H100 nodes (large models), single RTX 5090 (ToolOrchestra small-model setup).

## What this means

**For agent serving research.** ThunderAgent is the **system-layer foundation** that future agent-serving research will build on. The lineage looks like:

- **Generation 1 (2023-2024)**: vLLM (PagedAttention) → SGLang (RadixAttention) — request-level scheduling, agent-oblivious.
- **Generation 2 (2024)**: [[infercept|InferCept]] (Discard/Preserve/Swap per request) — first agent-aware system but still request-granular.
- **Generation 3a (Nov 2025)**: [[continuum|Continuum]] (predictive TTL per program) — program-aware but predictive.
- **Generation 3b (Feb 2026)**: **ThunderAgent (reactive thrashing monitor per program)** — program-aware and reactive. **Beats 3a on stochastic workloads.**
- **Generation 4 (May 2026)**: [[mori|MORI]] (two-tier GPU+CPU on top of ThunderAgent) — extends 3b with hierarchical memory.

The Continuum-vs-ThunderAgent split is a **predictive vs reactive** trade-off. Predictive (TTL) wins when tool durations are well-modeled and stable; reactive (thrashing monitor) wins when they're stochastic and adversarial. Production systems probably want both, gated by a workload classifier.

**For production deployment.** ThunderAgent is the most production-ready of the agent-serving research systems today: OpenAI-compatible API, MIT license, vLLM + SGLang support, asynchronous environment management built in. The natural deployment path is:

1. Drop ThunderAgent in front of an existing vLLM/SGLang cluster.
2. Modify client code to pass `program_id` (one line per request).
3. Configure tool orchestrator backend (Kubernetes/Docker).
4. Observe the 1.5-3.6× throughput improvement.

For multi-tenant deployments needing fairness or SLOs, custom scheduling extensions are needed (the paper doesn't address this).

**2027 prediction.** The "agentic program" data structure will appear in vLLM upstream as a first-class concept. The Continuum-vs-ThunderAgent debate will become the canonical "predictive vs reactive" trade-off discussion in agent serving textbooks.

## Related reading

- [[continuum]] — Continuum: contemporary competitor; predictive TTL approach vs ThunderAgent's reactive monitor.
- [[mori]] — MORI: two-tier GPU+CPU offloader built **on top of ThunderAgent** (~3,300 lines Python).
- [[infercept]] — InferCept: per-request KV management with Discard/Preserve/Swap; ThunderAgent extends to per-program.
- [[speccache]] — orthogonal speculative cache for *extra-LLM environment observations*; composes with ThunderAgent.
- [[agent-serving-challenges]] — broader context for what makes agent serving different.
- [[multi-turn-optimization]] — multi-turn KV reuse, prefix caching, session management.
- [[agentic-ai-workload-characteristics]] — the workload measurement that motivates all of these scheduler designs.

## Source code

[github.com/ThunderAgent-org/ThunderAgent](https://github.com/ThunderAgent-org/ThunderAgent) — MIT license, vLLM + SGLang backends, OpenAI-compatible API.

```bash
git clone https://github.com/ThunderAgent-org/ThunderAgent.git
cd ThunderAgent
pip install -e .
```

Client integration is a one-line change — add `program_id` to your OpenAI-compatible request and ThunderAgent does the rest.

## Paper citation

See [[thunderagent-citation|sources/papers/thunderagent/citation.md]] for the full citation, BibTeX, and headline numbers.
