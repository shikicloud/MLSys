---
title: "Agentic AI Workload Characteristics: End-to-End Tracing of ReAct Agents"
category: llm-serving-for-agents
tags: [agentic-ai, workload-characterization, react, opentelemetry, vllm, claude-code, qwen, gemma, swe-bench, terminal-bench, gaia, paper-review]
created: 2026-06-03
updated: 2026-06-03
status: mature
paper: arXiv:2605.26297
---

# Agentic AI Workload Characteristics

> [!info] Paper metadata
> - **Paper**: [arXiv:2605.26297v1](https://arxiv.org/abs/2605.26297) — *Agentic AI Workload Characteristics*, 2026-05-25
> - **Code**: Not released
> - **Authors**: Yichao Yuan (UIUC), Ankita Nayak (Gimlet Labs), Souvik Kundu (Intel), Nishil Talati (UIUC)
> - **Contact**: `{yichaoy2, nishil}@illinois.edu`
> - **Implementation**: vLLM v0.20.0, Claude Code agent framework, Harbor evaluation environment, OpenTelemetry tracing via Jaeger

> [!important] What this paper is and isn't
> This is **the** end-to-end agentic workload characterization paper for 2026 — single-agent tracing on real ReAct-style agents (Claude Code) running real benchmarks (SWE-Bench Pro, GAIA, Terminal-Bench), with OpenTelemetry-grade per-turn / per-tool / per-token instrumentation. It's the **measurement reference** that papers like [[continuum|Continuum]], [[cpu-centric-agentic-ai|CPU-Centric Perspective]], and [[rose|ROSE]] either build on or contrast with. **Not** an optimization paper — proposes no system, only measures. The contribution is the numbers, methodology, and ten or so consolidated insights that recalibrate how to think about agent serving.

---

## Summary (read this if you have 2 minutes)

**What it is.** A measurement study of ReAct-style LLM agents running real benchmarks, instrumented from request entry through tool execution back to next-turn LLM input. The authors run **Claude Code** orchestrating two dense LLMs (**Qwen3.6-27B** + **Gemma4-31B**, both Thinking and Instant variants) on **five benchmarks** (ADE-Bench, DABStep, GAIA, SWE-Bench Pro, Terminal-Bench 2.0) on a 2×H100 NVL + Intel Xeon platform. They collect ~100 task traces per (model, benchmark, mode) combination. The output is a structured catalog of *what production agentic serving actually looks like* at the granularity of turns, tokens, prefill/decode phases, cache reuse, tool call types, and failure modes.

**The one idea.** **Agentic LLM serving is not conventional long-prompt serving — it is repeated model re-entry over a growing cached context, with tool execution as a small but workload-dependent latency tail.** Four sub-pieces of this thesis:

1. **Decode dominates by 91–98.6%** of LLM time, not prefill (1.4–9.0%). Prior assumption that "long context → expensive prefill" misframes agentic workloads.
2. **Context cache reuse is 84.6–99.5%** empirically — most input tokens at each turn are reused from prior cache. Loss of KV state turns the decode-dominated workload into expensive recomputation.
3. **Tool execution is 2–29% of E2E time** (LLM is the other 71–98%), but the spread is large and workload-dependent: GAIA reaches 28.7% (heavy WebFetch), DABStep stays at 2–4% (LLM-bound), SWE-Bench Pro and Terminal Bench sit at 17.6–17.7%.
4. **Reasoning reduces pathological trajectories**, not just per-turn cost — Gemma Thinking on ADE-Bench averages 18 turns vs Gemma Instant's 108.8 turns (with 786-turn worst case), because thinking prevents the agent from getting trapped in edit-failure loops.

Drop any one of these observations and you misdesign the serving system: optimizing prefill is misguided (1.4–9% of time), losing KV state is catastrophic, optimizing only LLM ignores 20–30% of time, and ignoring reasoning's effect on workload shape underestimates its serving value.

**Headline numbers.** From the figures and tables, the most-quoted are:

| Metric | Value | Note |
| ------ | ----- | ---- |
| LLM time as % of E2E | **71–98%** | Range across (workload, model variant) |
| Tool time as % of E2E | **2–29%** | Workload-dependent |
| GAIA tool time | **28.7%** (Gemma Thinking) | WebFetch / WebSearch heavy |
| SWE-Bench Pro tool time | **17.7%** | Bash + Read + Edit dominate |
| Terminal-Bench tool time | **17.6%** | Bash + script execution |
| DABStep tool time | **2–4%** | Almost pure LLM compute |
| Decode % of LLM time | **91–98.6%** | Decode dominates over prefill |
| Empirical KV cache hit ratio | **84.6–99.5%** | Most input is reused across turns |
| Append-to-output ratio (median) | **<1.5×** | Small new input per turn vs response |
| Mean turns: Gemma Thinking on ADE | **18.0** | Reasoning compactifies trajectories |
| Mean turns: Gemma Instant on ADE | **108.8** (worst 786) | Edit-failure pathology |

**Why it matters.**

- **The "tool=70%" or "tool=2%" debates are both wrong as universals.** Workload+hardware determine the ratio; the actual range for production coding agents on dense models is **17–30%**, with [[cpu-centric-agentic-ai|CPU-Centric Perspective]]'s 88% being for explicitly compute-heavy tools (200GB RAG, RDKit) not in the typical agent set.
- **Optimizing prefill is misguided** — 91–98.6% of LLM time is decode, so kernel/quantization optimization on decode delivers far more than chunked-prefill tweaks for agents.
- **Reasoning models have a workload-level benefit** beyond per-turn accuracy: they reduce turn count, accumulated context, and failure-cascade pathologies. Serving system design must account for this.
- **2027 prediction.** This paper becomes the standard citation for "agent workload characterization" in any agentic serving paper, the way "ShareGPT trace" was the standard for chatbot characterization in 2023-2024. Expect new agent serving systems to report their numbers in this paper's per-turn / per-cache / per-tool decomposition.

---

# Depth (drill-down starts here)

## Background: why agentic workloads are not chatbot workloads

The paper opens (Section 1) by enumerating the structural differences that make existing serving wisdom misapplied:

> "Agentic AI shifts LLM serving from isolated prompt-generation requests to stateful, multi-turn executions that repeatedly invoke the model, call tools, and grow context over time."

Three concrete failure modes of treating agent serving as chatbot serving:

1. **"Long prompt" is the wrong abstraction.** Chatbot serving sees a single long prompt per request; agentic serving sees many short *increments* over an accumulating cached context. Optimizing for long-prompt prefill misses this.
2. **Turn count and context length are different metrics.** Some workloads (ADE Instant) have 100+ turns with modest context per turn; others (SWE-Bench Pro) have moderate turn count but huge contexts. Both stress different parts of the serving stack.
3. **Tool execution is heterogeneous and long-tailed.** Tool latency varies from milliseconds (`grep`) to minutes (`Agent` delegation, slow API calls). A single uniform "tool latency" estimate doesn't capture this.

The paper's formal model of an agent (Section 2.3):

For agent $a$ at step $i$, let $H_{a,i}$ be the accumulated context (trajectory) and $C_{a,i} = |H_{a,i}|$ the context length. The LLM produces a three-part response:
$$z_{a,i} = (\theta_{a,i}, m_{a,i}, u_{a,i})$$
where $\theta_{a,i}$ = thinking tokens, $m_{a,i}$ = message tokens, $u_{a,i}$ = tool-call tokens. The environment then appends tool results $o_{a,i}$ and the context evolves:
$$H_{a,i+1} = H_{a,i} \| \Phi(\theta_{a,i}, m_{a,i}, u_{a,i}) \| o_{a,i}$$

where $\Phi(\cdot)$ is chat-template formatting and $\|$ is concatenation. **An agent turn is one $(\text{LLM call}, \text{tool result})$ pair**. Subagents (launched via `Agent` tool) are treated as part of the parent's tool call.

## Methodology in detail

### Hardware and software (Section 3.1)

| Component | Configuration |
| --------- | ------------- |
| GPU | 2× NVIDIA H100 NVL connected by 12 NVLinks |
| CPU | Intel Xeon Platinum 8592+ |
| Inference engine | vLLM v0.20.0, TP=2 |
| Tracing | vLLM OpenTelemetry support → Jaeger |
| Agent framework | Claude Code |
| Evaluation environment | Harbor (single-agent execution, objective + tools per task) |

The choice of vLLM 0.20.0 + TP=2 is deliberate: this configuration is representative of production deployments for 27-31B dense models.

### The three-component characterization infrastructure (Figure 2)

```
Harbor → Dockerized Claude Code agents
     ↓
LLM requests
     ↓
Forwarding proxy (records request metadata, modifies request to ensure OTEL trace)
     ↓
vLLM server
     ↓
Jaeger collects OpenTelemetry traces (prefill/decode/queue timings)

Per-agent wrapper assigns unique API key → groups traces per agent execution
```

The forwarding proxy is the most important design choice: it sits between the agent and vLLM, intercepts every LLM call, and ensures vLLM emits per-request OpenTelemetry traces with prefill/decode timing. This is the only way to get the per-turn / per-phase breakdown the paper reports.

### Workloads and configurations (Section 3.1)

**Models** (both Thinking and Instant variants tested):
- Qwen3.6-27B
- Gemma4-31B

**Benchmarks** (100 tasks sampled from each):
- ADE-Bench — data analytics
- DABStep — data agent tasks
- GAIA — general assistant (multi-step + tool use)
- SWE-Bench Pro — software engineering
- Terminal-Bench 2.0 — terminal-based software engineering + system interaction

This is **8 cells per workload** (Gemma-T/I × Qwen-T/I × 100 tasks), so ~4000 task traces total.

## Three components in detail

### Component 1 — Agent Execution Characterization (Section 4)

Four sub-questions: how many turns, how much context, what's the output composition, do failed/successful runs differ?

#### Turn counts (Figure 3)

| Workload | Gemma-T mean ± std | Gemma-I mean ± std | Qwen-T mean ± std | Qwen-I mean ± std | Worst turn (any) |
| -------- | -----------------: | -----------------: | -----------------: | -----------------: | ---------------: |
| ADE | 18 ± 15 | **109 ± 179** | 19 ± 32 | 17 ± 17 | **786** (Gemma-I) |
| DABStep | 16 ± 6 | **640** | 17 ± 8 | 16 ± 11 | 640 (Gemma-I) |
| GAIA | 22 ± 33 | 617 (P75 ~33) | 31 ± 26 | 17 ± 43 | 617 |
| SWE-Bench Pro | 31 ± 20 | 24 ± 23 | **41 ± 19** | 26 ± 19 | 226 |
| Terminal Bench | 74 ± 26 | 26 ± 62 | 25 ± 23 | 25 ± 43 | 197 |

Two takeaways:
- **Long tails are model-dependent, not just workload-dependent**. Gemma Instant on ADE reaches 786 turns; Gemma Thinking on ADE stays at 18. The cause (per paper §4.1): Gemma Instant on ADE makes 2,757 failing `Edit` calls (95.4% failure rate), getting trapped in retry loops.
- **"The benefit of reasoning is strongest for Gemma"** — Thinking reduces mean turns by 6× on ADE (18 vs 109). For Qwen, Thinking is only modestly better (sometimes worse — 41 vs 26 on SWE-Bench Pro).

#### Context length (Figure 4)

| Workload | Median context (K tokens) | Max context (K tokens) |
| -------- | ------------------------: | ---------------------: |
| ADE | 37–43 | up to 171 (Gemma-I) |
| DABStep | 44–48 | up to 126 |
| GAIA | 38–54 | up to 174 (Qwen-I) |
| **SWE-Bench Pro** | **69–80** | **146–166** (largest) |
| Terminal Bench | 45–66 | up to 167 |

> [!important] Context growth ≠ turn count
> SWE-Bench Pro has the largest contexts (mean 69–80K) but only moderate turn counts (mean 26–41). Conversely, ADE has the longest turn-count tail (786 turns) but moderate context (37–43K mean). **These are independent failure modes** for serving system design — high turn count stresses scheduling, large context stresses KV cache memory.

#### Output token composition (Figure 5)

The breakdown of generated tokens (thinking / message / tool-call):

| Model variant | Thinking % | Tool-call % |
| ------------- | ---------: | ----------: |
| **Gemma Thinking** | 45.8–67.6% | 18–48% |
| **Gemma Instant** | ~0% | **87.8–98.2%** |
| **Qwen Thinking** | 29.0–40.7% | 48.2–63.7% |
| **Qwen Instant** | ~0% | **70.4–81.6%** |

> [!quote] What this overturns
> "ReAct-style agents are not simply producing long natural-language responses; much of their generated output is structured action generation that drives interaction with the environment."

Most of an agent's decoding is **structured tool-call tokens** (JSON, XML, function-call format), not prose. This has direct implications for constrained-generation infrastructure ([[multi-turn-optimization|structured output]]) and for what kind of tokens to optimize.

#### Successful vs failed (Figure 6)

> "**Failed agents usually accumulate larger contexts than successful agents**, although the strength of this effect varies by workload and model."

GAIA failures reach 34.6K–64.7K tokens vs successful 25–47K tokens. Terminal Bench: failed 76–79K vs successful 44–46K. SWE-Bench Pro is context-heavy for both success and failure (Gemma Thinking 82.2K success vs 72.3K failure — counterexample where successful runs are *longer* because legitimate SWE work needs sustained context).

**Failure mechanism (Table 2)**: failed Bash/Edit/Read actions append error messages and stack traces, which become part of next turn's input context. Failed runs thus amplify load in two dimensions: more turns AND larger context.

### Component 2 — Runtime Performance Characterization (Section 5)

The three measurements that recalibrate serving system design.

#### Section 5.1 — LLM versus tool call time (Figure 7)

This is the most-quoted result.

| Workload | LLM % | Tool % |
| -------- | ----: | -----: |
| DABStep | **96–98%** | 2–4% |
| ADE | 86–92% | 8–14% |
| SWE-Bench Pro | 82.3% | **17.7%** |
| Terminal Bench | 82.4% | **17.6%** |
| **GAIA (Gemma Thinking)** | 71.3% | **28.7%** (highest) |
| GAIA (Qwen Thinking) | 75.1% | 24.9% |

> [!quote] The headline framing
> "Across all benchmarks and models, execution is primarily LLM-dominated, with LLM inference accounting for 71–98% of total runtime. ... However, tool execution is not negligible. Tools account for 2–29% of total runtime, with the largest fractions on GAIA, where tool time reaches 28.7% for Gemma Thinking and 24.9% for Qwen Thinking. This aligns with Table 1 where GAIA includes expensive WebFetch, Agent, and TaskOutput calls."

The paper acknowledges in §5.1's final paragraph that **concurrency would shift these numbers**: "While the degree of concurrency (i.e., number of agents running concurrently) may change the LLM- vs. -tool time, our results are in line with other works." This is the load-bearing caveat — these measurements are for *single-agent execution*. Multi-agent batched serving stretches LLM call queue time and changes the ratio.

#### Section 5.2 — Context cache effectiveness (Figure 8)

| Workload | Empirical cache hit ratio | Theoretical cache hit ratio | Mean append-to-output ratio |
| -------- | -------------------------: | --------------------------: | --------------------------: |
| ADE | 84.6–99.5% | 87.9–99.3% | 1.5–4.5× (median <1.5) |
| DABStep | similar | similar | 1.0–1.4× |
| GAIA | similar | similar | 1.0–1.3× |
| SWE-Bench Pro | similar | similar | 0.4–1.4× |
| Terminal Bench | similar | similar | 0.5–2.3× |

**Two consequences**:

1. **Agent serving is not "repeated long-prompt prefill"** — each turn only appends ~1–2× the response length in new input tokens. The rest of the context is reused from cache.
2. **Losing KV state is catastrophic** — without effective caching, a 80K-token agent step at turn 30 would re-prefill 80K tokens, turning a decode-dominated workload into a prefill-dominated one. This is exactly why systems like [[continuum|Continuum]] (TTL-based KV pinning), [[multi-turn-optimization|SGLang RadixAttention]] (prefix cache), and LMCache (CPU/disk offload) are load-bearing.

> [!quote] Reframing agent serving
> "Agent execution should be modeled as a sequence of repeated model re-entry over a growing cached context, but as repeated decode over a growing cached context, rather than repeated full-context prefilling."

#### Section 5.3 — Prefill versus decode time (Figure 9)

| Workload | Prefill % of LLM time | Decode % of LLM time |
| -------- | ---------------------: | -------------------: |
| All workloads | **1.4–9.0%** | **91.0–98.6%** |
| Largest prefill share | Gemma Thinking on SWE-Bench Pro: **9.0%** | — |
| Smallest prefill share | DABStep: **~1.4%** | 98.6% |

This is the single most impactful number for serving optimization priority: **decode is 91–98.6% of LLM time, so kernel/quantization/memory-bandwidth optimization on decode delivers far more leverage than prefill optimization**. PagedAttention, FlashAttention-2/3 decode kernels, KV cache compression — all of these target decode. Chunked prefill (Sarathi-Serve) optimizes the 1.4–9% slice.

### Component 3 — Tool Call Characterization (Section 6)

A finer-grained look at which tool types dominate and how their patterns shift over the agent's lifetime.

#### Tool breakdown by type (Figure 10)

Tool composition varies more by workload than by reasoning mode:

| Workload | Top tools |
| -------- | --------- |
| ADE | Read 29.8–41.7%, Bash 23.2–30.0% (Gemma-I: Edit 51.2%) |
| DABStep | Bash 50.5–87.8% (heavy) — coding tasks rely on command line |
| GAIA | WebFetch 42.3% (Gemma-T), WebSearch 29.7% (Gemma-I) — web-facing tools dominate |
| SWE-Bench Pro | Edit + Bash + Read all substantial, with workload-specific mix |
| Terminal Bench | Bash heavy (53.2–79.5% on Qwen) — terminal commands |

#### Bash command breakdown (Figure 11)

Bash is not a monolithic tool — what commands the agent issues differs sharply:

| Workload | Top Bash commands |
| -------- | ----------------- |
| ADE | `dbt` (24.7–50.9%) — database transformations |
| DABStep | `python3` (37.2–38.5% on Qwen), or `dbt`/`python3` on Gemma |
| GAIA | `curl` (24.2–25.2% for Qwen) — web-facing |
| SWE-Bench Pro / Terminal | Large "Other" fraction (50–65%) — diverse: `pytest`, `git`, `find`, `make`, `gcc`, `npm`, `pip`, etc. |

#### Top time-consuming tools (Table 1)

The expensive tools differ from frequently-called tools:

| Benchmark | Top time-consuming tool (Qwen-T) | Mean duration |
| --------- | --------------------------------: | ------------: |
| ADE | Bash (n=611) | 1.27s |
| **GAIA** | **Agent** (delegation, n=11) | **916s** |
| **SWE-Bench Pro** | **Agent** (delegation, n=75) | **59.15s** |

The **`Agent` tool** (which spawns a subagent for delegated work) is by far the most expensive — multi-minute on GAIA. Failed agents are concentrated in error-producing tools.

#### Top failing tools (Table 2)

The tools that fail most are those modifying state:

| Workload | Top failing tool, fail rate |
| -------- | --------------------------- |
| ADE | Gemma-I **Edit fail 95.4%** (n=2757) |
| ADE | Gemma-T Bash fail 28.4% (n=208) |
| GAIA | Gemma-T Read fail 48.9% |
| SWE-Bench Pro | Gemma-I Edit fail 77.8% (n=955), Bash fail 39.8% |

**Failures concentrate in state-modifying tools** (Edit, Bash), not in retrieval tools (Read, WebFetch — fail at <10%). This is the failure cascade source: Edit fails → error appended to context → next-turn LLM tries to fix → fails again → context grows + turn count grows.

#### Tool intention shifts over time (Figure 12)

A robust temporal pattern: agents start exploring, end executing.

```
Quartile 1 of agent progress:
  Read/Explore tools (Read, Glob, Grep, WebFetch, WebSearch): 60–80% of tool calls
  Execute/Write tools (Bash, Edit, Write, TodoWrite): 20–40%

Quartile 4 of agent progress:
  Read/Explore tools: 0–20%
  Execute/Write tools: 80–100%
```

> [!quote] The temporal characterization insight
> "Agents typically begin in an inspection-heavy mode and then transition toward action-heavy behavior. This transition is strongest on DABStep, SWE-bench Pro, and Terminal Bench."

This has direct implications for serving optimization:
- Early turns benefit from fast read/list operations (file system, search)
- Late turns benefit from fast write/exec operations (process spawn, file write)
- Caching strategies should account for the temporal shift (early-turn reads are likely re-read; late-turn writes are unique)

## Headline evidence (consolidation)

The 11 consolidated insights the paper produces:

1. **LLM time dominates** at 71–98% of E2E; tool time 2–29%.
2. **Decode dominates** LLM time at 91–98.6%; prefill 1.4–9.0%.
3. **Context cache hit** empirically 84.6–99.5%.
4. **Append-to-output ratio** median <1.5×, mean 1.5–7.3×.
5. **Reasoning compactifies** Gemma's trajectories on ADE 6× (18 vs 109 mean turns).
6. **Failed runs amplify load** in both turn count and context length.
7. **Tool output composition is structured-token-heavy** — 70–98% of Instant outputs are tool-call tokens.
8. **Tool call types are workload-specific** — DABStep dbt/python3, GAIA curl/WebFetch, SWE/Terminal Edit+Bash+Read.
9. **Tool intention shifts** from Read/Explore early to Execute/Write late.
10. **`Agent` (delegation) is the most expensive tool** — multi-minute on GAIA.
11. **State-modifying tools fail more** — Edit/Bash 28–95% failure rates vs Read/WebFetch <10%.

## Strengths and limitations

**Strengths.**

- **First production-grade end-to-end agentic workload characterization** with proper OpenTelemetry instrumentation through real ReAct frameworks (Claude Code), not synthetic agents.
- **Multi-dimensional measurement** — turns, contexts, output composition, prefill/decode, cache reuse, tool types, command types, success/failure, temporal patterns. Nothing missing.
- **Reasoning models tested** — not just Instant. Most prior characterization predates reasoning models.
- **Standard benchmarks** (SWE-Bench Pro, GAIA, Terminal-Bench 2.0) make results directly comparable to other papers.
- **Replicable methodology** — vLLM 0.20.0 + Claude Code + OpenTelemetry is a reproducible stack.

**Limitations.**

> [!warning] Single-agent execution only
> The paper explicitly notes (§5.1 last paragraph): "While the degree of concurrency (i.e., number of agents running concurrently) may change the LLM- vs. -tool time, our results are in line with other works." Production multi-agent serving (where multiple agents share GPU) has **different** LLM%/tool% ratios because LLM calls queue. The paper's numbers are a **per-agent lower bound** on tool% — concurrent serving inflates it.

- **Only 2 model families (Qwen3.6, Gemma4)** at one size range (27-31B). Behavior at 7B or 70B+ may differ.
- **Only Claude Code agent framework.** Other harnesses (Codex, Cursor, Aider, OpenHands) have different turn structures, prompt templates, and tool sets.
- **vLLM only** — SGLang's RadixAttention may yield different cache hit ratios and prefill/decode splits.
- **No serving-system optimization proposed** — measurement only. The reader has to translate insights into design choices themselves.
- **100 tasks per cell is small** — long-tail behavior (e.g., 786-turn ADE outlier) is sampled but may not be representative of full distribution.
- **TP=2 fixed** — doesn't measure how TP=4 or TP=8 changes decode dominance (likely it stays decode-bound but ratios shift).
- **No code release** as of June 2026.

> [!bug] The 28.7% GAIA number is for **Gemma Thinking**, not Qwen
> Several papers (including some of my earlier wiki entries) cite "GAIA tool time 28.7%" without noting the model. The paper specifically reports **28.7% for Gemma Thinking, 24.9% for Qwen Thinking**. For Instant variants the numbers are different again. When citing, name the model.

## What this means

This paper does what every serving-systems researcher should want from a workload paper: **provides numbers, not vibes**. The 11 insights above are concrete enough to be cited in design documents and rigorous enough to settle field disputes (notably the "tool dominates" vs "LLM dominates" question — see the per-workload breakdown).

Three implications for serving system design:

1. **Optimize decode, not prefill.** The 91–98.6% finding is decisive. Spec decoding, decode-side quantization (e.g., [[saw-int4|SAW-INT4]]), KV cache compression — these are 5–10× more impactful than chunked prefill optimization for agent workloads.
2. **Make KV cache retention load-bearing.** The 84.6–99.5% cache hit ratio means losing state is catastrophic. [[continuum|Continuum]]'s TTL-based pinning is the right primitive; [[multi-turn-optimization|RadixAttention]] in SGLang and prefix caching in vLLM are both valuable but address the cache-hit problem, not the cache-retention problem.
3. **Tool latency optimization has bounded payoff** — at most you save 17–30% of E2E time on coding workloads, more on web workloads (GAIA). Speculative tool execution ([[#Related reading|PASTE, Speculative Tool Calls, Conveyor]]) is worthwhile but won't close more than this gap.

Three predictions for 2027:

1. **This paper becomes the standard citation** the way ShareGPT was for chatbot serving. Any agentic serving paper that doesn't compare against these numbers will be uncited.
2. **Multi-agent concurrent measurement becomes the next paper.** The single-agent caveat is the obvious follow-up direction. Expect a sister paper (or extension) within 6–12 months.
3. **Per-harness characterization expands.** Comparing Claude Code vs Codex vs Cursor vs Aider on the same benchmarks would expose how much of these numbers are harness-specific vs LLM-specific.

This paper does **not** cover:

- **Multi-tenant serving with concurrency.** Per-agent measurements only; production multi-agent scheduling not addressed.
- **GPU pool sizing or cost-benefit analysis.** The numbers inform design; they don't pick a configuration.
- **Long-running agents (24+ hour).** Sampling is to 100 tasks per cell; persistent agents not measured.
- **Tool-execution-side optimization.** Tool latency measured but not optimized; that's [[cpu-centric-agentic-ai|CPU-Centric Perspective]]'s remit.

## Source code & reproduction

```bash
# Code not yet released as of June 2026.
# Reproduction stack:
pip install vllm==0.20.0
# Plus Claude Code (proprietary), Harbor evaluation framework
# OpenTelemetry vLLM support + Jaeger collector
```

**Reproduction protocol** (Section 3.1):

| Component | Configuration |
| --------- | ------------- |
| Inference engine | vLLM v0.20.0, TP=2, OpenTelemetry support enabled |
| GPU | 2× NVIDIA H100 NVL with 12 NVLinks |
| CPU | Intel Xeon Platinum 8592+ |
| Agent framework | Claude Code |
| Evaluation environment | Harbor |
| Trace collector | Jaeger |
| Models | Qwen3.6-27B Thinking + Instant; Gemma4-31B Thinking + Instant |
| Benchmarks | ADE-Bench, DABStep (100-sample), GAIA, SWE-Bench Pro (100-sample), Terminal-Bench 2.0 |
| Tasks per cell | 100 (sampled from each benchmark) |
| Total cells | 8 per workload × 5 workloads = 40 |
| Total traces | ~4000 |

**Key infrastructure components** (described in §3.1):

- **Forwarding proxy** — intercepts every LLM request, modifies headers to ensure vLLM emits OpenTelemetry traces, attaches per-agent API keys for trace correlation
- **Per-agent wrapper** — launches Claude Code agents in Dockerized containers, assigns unique API keys
- **Trace aggregator** — combines Jaeger OTEL traces + Harbor execution results + agent request logs into characterization-ready data

## Related reading

- [[continuum]] — Continuum directly cites this paper's numbers (84.6-99.5% cache hit ratio specifically), and Continuum's TTL-based KV pinning is the system response to this paper's "losing state is catastrophic" insight.
- [[cpu-centric-agentic-ai]] — Complementary measurement from the CPU side; this paper's 17-30% tool numbers on standard coding workloads are the lower-bound counterpart to CPU-Centric's 88% on artificially-CPU-heavy tools (ENNS over 200GB, RDKit).
- [[agent-serving-challenges]] — Higher-level survey of agent serving differences; this paper provides the rigorous data.
- [[multi-turn-optimization]] — Multi-turn KV reuse strategies; this paper's 84.6-99.5% cache hit ratio is the empirical justification for those strategies.
- [[prefill-decode-disaggregation]] — PD disagg motivation revisited: this paper shows decode dominates 91-98.6% in agentic workloads, so PD-disagg's "balance prefill and decode" framing applies differently — agents are nearly all-decode.
- [[saw-int4]] — INT4 KV quantization; especially relevant given this paper shows decode (and thus KV cache) dominates LLM time.
- [[paged-attention]] — Underlying KV cache primitive; this paper validates that paged attention's amortization across long contexts is exactly what agentic workloads need.
- [[vllm]] — The inference engine used (vLLM v0.20.0).
- [[continuous-batching]] — Standard agent serving uses continuous batching; this paper shows decode dominance, which is exactly what continuous batching optimizes.
- [[rose]] — ROSE's measurement that rollout > 70% of training wall-clock is the agentic-RL counterpart to this paper's per-task measurement; both motivate cache-state preservation.
- [[ai-agent-overview]] — Higher-level ReAct paradigm description.

## References

- Yichao Yuan, Ankita Nayak, Souvik Kundu, Nishil Talati. *Agentic AI Workload Characteristics.* arXiv:2605.26297, May 2026. https://arxiv.org/abs/2605.26297
- ReAct: Yao et al. (cited as [37] in the paper).
- Claude Code: Anthropic ([2]).
- Codex: OpenAI ([4]).
- Harbor evaluation framework: ([10]).
- Open benchmarks: ADE-Bench [31], DABStep [7], GAIA [21], SWE-Bench Pro [6], Terminal-Bench [20].
- vLLM: ([15]).
- OpenTelemetry: ([24]).
- Jaeger: ([11]).
- Concurrent ReAct workload analysis: KAIROS ([38]) — cited for the multi-agent concurrent perspective.
