---
title: "CPU-Centric Perspective on Agentic AI: Characterization + COMB + MAS"
category: llm-serving-for-agents
tags: [cpu-centric, agentic-ai, characterization, comb, mas, micro-batching, scheduling, cpu-gpu-coordination, vllm, paper-review]
created: 2026-06-02
updated: 2026-06-02
status: mature
paper: arXiv:2511.00739
---

# Towards Understanding, Analyzing, and Optimizing Agentic AI Execution: A CPU-Centric Perspective

> [!info] Paper metadata
> - **Paper**: [arXiv:2511.00739](https://arxiv.org/abs/2511.00739) — *Towards Understanding, Analyzing, and Optimizing Agentic AI Execution: A CPU-Centric Perspective*, 2025-11-01 (v3: 2026-04-16)
> - **Code**: Not yet released
> - **Authors**: Ritik Raj (Georgia Tech), Souvik Kundu (Intel), Ishita Vohra (Georgia Tech), Hong Wang (Intel), Tushar Krishna (Georgia Tech)
> - **Affiliations**: Georgia Institute of Technology + Intel
> - **Implementation**: vLLM v0.14.0, PyTorch 2.8.0; closed-loop and open-loop arrival systems

> [!important] What this paper is *not*
> Despite a title that suggests pure characterization, the load-bearing contributions are **two scheduling algorithms** (COMB + MAS) plus a tracer-grade measurement of where CPU bottlenecks actually live in agentic serving. The characterization data (tool time up to 88% of E2E latency on tool-dominated workloads) is the *motivation*; the two algorithms are the deliverable. Read with both layers in mind.

---

## Summary (read this if you have 2 minutes)

**What it is.** A characterization-and-optimization study of agentic AI serving from the CPU's perspective. The authors trace five representative agentic workloads (Toolformer, SWE-Agent, RAG/Haystack, ChemCrow, Web-Augmented Agent / LangChain) across two asymmetric CPU-GPU systems (high-CPU+low-GPU vs low-CPU+high-GPU), identify CPU-side bottlenecks that GPU-centric serving systems ignore, and then propose two scheduling optimizations — **COMB** for homogeneous batches and **MAS** for heterogeneous mixed workloads.

**The one idea.** **CPU and GPU are largely *disjoint phases* in agentic pipelines, and naive scheduling under-utilizes the idle one.** Three sub-pieces:

1. **Compile-time taxonomy + runtime characterization** — agentic workloads vary along three orthogonal axes (LLM-vs-Host orchestrator, Static-vs-Dynamic path, Single-vs-Multi-step). The paper picks 5 representative workloads spanning the corners and measures where time actually goes. Headline measurement: **tool processing can consume up to 88% of end-to-end latency on tool-dominated workloads** (e.g., ENNS retrieval in RAG on Sys 2, conformer generation in ChemCrow Heavy).
2. **COMB (CPU-Aware Overlapped Micro-Batching)** — partition a large batch into capped micro-batches of size $B_{cap} \approx 1\text{-}2 \times \#\text{CPUs}$, then *overlap* the CPU stage of micro-batch $i+1$ with the GPU stage of micro-batch $i$. This avoids CPU over-subscription (which kills CPU efficiency) while sustaining GPU utilization.
3. **MAS (Mixed Agentic Scheduling)** — for heterogeneous workloads where some requests are CPU-heavy and others GPU-heavy, admit them through separate queue caps ($E_{cap,CPU}$, $E_{cap,GPU}$) with a shared elastic queue ($E_{cap,shared}$). Prevents the dominant request type from starving the minority.

Remove COMB and CPU oversubscription destroys throughput at large batches; remove MAS and minority request types suffer head-of-line blocking; remove the characterization and you can't size $B_{cap}$ or $E_{cap}$ correctly for your hardware.

**Headline results.**

| Setting | Optimization | Metric | Improvement |
| ------- | ------------ | ------ | ----------- |
| Standalone batch (BS=128) homogeneous | COMB | P50 latency | **1.7× lower** |
| Open-loop homogeneous, $\lambda=13$ req/s | COMB ($N_{cap}=64$) | Service latency P50/P90 | **2.9× / 3.9× lower** |
| Open-loop homogeneous, $\lambda=13$ req/s | COMB | Total latency P50/P90 | **1.6× / 1.8× lower** |
| Open-loop homogeneous | COMB ($N_{cap}=64$) | Throughput | **1.7× higher** |
| Heterogeneous open-loop, $p_{LLM}=0.25$, Sys 1 | MAS | GPU-heavy P50/P90 | **2.37× / 2.49× lower** |
| Heterogeneous open-loop, $p_{LLM}=0.50$, Sys 1 | MAS | Total P50/P90 (avg) | **1.62× / 1.30× lower** |

**Why it matters.**

- **First end-to-end measurement reconciling the "tool dominates" vs "LLM dominates" debate** — answer is workload- and hardware-dependent, not single-number. Tool time ranges from ~10% (Toolformer on Sys 1, GPU bottleneck) to 88% (RAG ENNS on Sys 2, CPU bottleneck) depending on which side has better hardware.
- **HP CPU + LP GPU can match HP GPU + LP CPU** for tool-dominated workloads — has real cost-deployment implications (don't buy H200 if your bottleneck is retrieval; buy more Xeon cores).
- **Identifies that CPU parallelism efficiency is fundamentally lower than GPU** — empirically observed via STREAM-style saturation at >80% with just 4 processes per socket; over-subscription beyond cores triggers OS scheduler contention.
- **2027 prediction.** COMB-style CPU-GPU overlap micro-batching becomes standard in vLLM/SGLang for agent workloads. MAS-style request-type-aware admission becomes the default scheduling discipline. Both will be incorporated as `--enable-cpu-overlap` and `--enable-mixed-type-admission` flags within 12 months.

---

# Depth (drill-down starts here)

## Background: the CPU bottleneck no one was measuring

Production LLM serving systems (vLLM, SGLang, TensorRT-LLM) optimize aggressively for the GPU side — paged attention, continuous batching, prefill-decode disaggregation, FP8 KV cache, etc. They treat tool execution and Python orchestration as "external" and out of scope. For chatbot serving this is correct: tool calls are rare.

**For agents, this assumption breaks.** The paper enumerates the failure modes (Section 1):

> "While prior approaches on AI efficiency aggressively focused on GPU kernels and KV-cache management, they become ineffective for the CPU-centric tool execution of the agentic AI workloads. A recent work shows that ENNS accounts for more than 75% of the end-to-end (E2E) latency on a 200 GB document corpus for a Retrieval Augmented Generation (RAG) workload with a Llama-3-70B."

The paper's contribution is to make this concrete across a representative workload suite and turn the measurements into scheduling algorithms.

### Compile-time characterization (Section 2)

The authors propose a **three-axis taxonomy** for agentic systems:

| Axis | Values | Meaning |
| ---- | ------ | ------- |
| **Orchestrator** | LLM-orchestrated / Host-orchestrated | Who decides the next tool — LLM (ReAct/AutoGPT style) or Python code (LangChain/Haystack style) |
| **Path** | Static / Dynamic | Workflow fixed at design time (Haystack) or chosen at runtime by LLM (Reflexion, LATS) |
| **Repetitiveness** | Single-step / Multi-step | One forward pass through the agent (RAG QA) or iterative loop (WebArena, Balrog) |

This taxonomy is *a priori* (platform-agnostic) and used to pick five workloads spanning the corners.

### The five representative workloads (Table 2)

| Workload | Orchestrator | Path | Flow | Tools | Application |
| -------- | ------------ | ---- | ---- | ----- | ----------- |
| **Toolformer** | LLM | Dynamic | Single-step | Calculator API, Calendar | MLQA, Math |
| **SWE-Agent** | LLM | Static | Multi-step | Bash File I/O, Python | SDE, Data analysis |
| **RAG (Haystack)** | Python (Host) | Static | Single-step | Web search, Document retrieval | RAG QA |
| **ChemCrow** | LLM | Dynamic | Multi-step | Conformer Gen, Reaction tools | Chemistry research |
| **Web-Augmented Agent (LangChain)** | Python (Host) | Static | Single-step | Web search, Summarizer | Web QA, DevOps |

Models used are small LMs (≤32B params), justified as appropriate for production agentic deployments where most agent steps don't need flagship-LLM intelligence.

### Hardware tested (Table 1) — the asymmetry is the point

The authors deliberately pick **asymmetric CPU-GPU systems** to isolate where the bottleneck actually lives:

| Component | Sys 1 (HP CPU + LP GPU) | Sys 2 (LP CPU + HP GPU) |
| --------- | ----------------------- | ----------------------- |
| **CPU** | 64-core Intel Granite Rapids | 72-core Nvidia Grace |
| CPU Memory | 512 GB DDR5 | 480 GB LPDDR5 |
| **GPU** | Nvidia RTX-Pro 6000 Blackwell | Nvidia H200 |
| GPU Memory | 96 GB GDDR7 | 96 GB HBM3e |

By comparing the same workload across both systems, you see when the bottleneck shifts CPU↔GPU.

## Three components in detail

### Component 1 — Runtime characterization (Section 3)

The empirical heart of the paper. Figure 2 plots end-to-end latency for all 5 workloads × 2 systems × multiple benchmark inputs, broken into LLM inference (GPU) vs tool processing (CPU). Six headline observations:

**(a) RAG / Haystack** — ENNS retrieval dominates: **83%, 81%, 82% of total latency on Sys 1** for NQ, HotpotQA, TriviaQA respectively; up to **89% on Sys 2** (LP CPU). The HP GPU on Sys 2 makes LLM faster but CPU becomes the constraint.

**(b) Toolformer** — LLM-dominated on Sys 1 (**~88% LLM inference**), drops to **77% on Sys 2** (HP GPU). The WolframAlpha API call has negligible CPU cost.

**(c) Web-Augmented Agent (LangChain)** — LexRank summarization takes **55% (freshQA) and 48% (QASC) on Sys 1**; URL fetch variance dominates.

**(d) SWE-Agent** — Bash + Python execution takes **38% (APPS), 25% (BigCodeBench) on Sys 2**, but up to **65% of E2E latency on Sys 2** in some configurations. The well-optimized LLM on Sys 2 forces the bottleneck to CPU.

**(e) ChemCrow** — Conformer generation (RDKit) dominates at **85% (Heavy) and 88% (Sys 1, Heavy)**; for Medium molecules LLM dominates (58% on Sys 1, 53% on Sys 2).

> [!important] The "tool dominates" / "LLM dominates" question has no single answer
> Whether tools or LLM dominate depends on (1) which side has better hardware, (2) the specific tool's compute profile, and (3) input characteristics (small molecules → LLM-bound, heavy molecules → tool-bound for ChemCrow). The paper's contribution is to quantify this *empirically* across a representative spread, rather than picking one workload and over-generalizing.

**Key Takeaway 1 (from paper):** "Tool processing on CPUs can take significant chunk of E2E latency, motivating a CPU-centric optimization strategy."

**Key Takeaway 2:** "HP CPU system can shift the bottleneck from GPU to CPU when tool execution latency is comparable to LLM inference latency, making them more CPU-bounded systems with LP GPU, motivating system-aware optimization strategies."

**Key Takeaway 3 (Section 3.3):** "CPU-parallelization strategies fundamentally exhibit lower efficiency compared to GPU. In agentic AI workloads, they prematurely saturate the throughput, subsequently bottle-necking the system and degrading the utilization of costly GPU resources."

### Component 2 — COMB: CPU-Aware Overlapped Micro-Batching (Section 4.1)

The first algorithmic contribution. Key observation: **CPU efficiency saturates around $1\text{-}2 \times \#\text{cores}$ active processes**; beyond that, OS contention dominates. Meanwhile, GPU efficiency at small batches is *also* limited (low utilization), but rises much more steeply.

**COMB's two design moves:**

1. **Cap the micro-batch size** at $B_{cap} \approx 1\text{-}2 \times \#\text{CPUs}$ (derived from CPU parallelization efficiency saturation point). For large incoming batches $B_{max} > B_{cap}$, split into ceiling($B_{max} / B_{cap}$) micro-batches.
2. **Overlap consecutive micro-batches** — after an overlap interval $s$, while micro-batch $i$ executes its GPU stage (LLM), micro-batch $i+1$ executes its CPU stage (tool). This is *pipelined* execution across stages, not within them.

Visualized (paper Figure 4):

```
Multi-processing baseline (BS=128):
  CPU 0-128: [████████████ tool ████████████]  ←  64 cores oversubscribed 2×
  GPU:                                            [██████████ LLM ██████████]

COMB ($B_{cap}=64$, $s$ = 16s):
  CPU 0-31:  [██████████ μ1 tool ██████████]
  CPU 31-63:                                  [██████████ μ2 tool ██████████]
  CPU 64-95:                                  [██████████ μ3 tool ██████████]
  CPU 96-128:                                                        [██ μ4 tool ██]
  GPU:                              [██ μ1 LLM ██][██ μ2 LLM ██][██ μ3 LLM ██][μ4]
                                    ↑
                                    starts after s=16s overlap with μ1 CPU
```

### Concretely, what this saves

The cost of CPU oversubscription is real. From Table 3:

| Workload | Sys 1 r(64) | Sys 1 r(128) | Sys 2 r(64) | Sys 2 r(128) |
| -------- | ----------: | -----------: | ----------: | -----------: |
| Web-Augmented Agent (LangChain) | 1.94 | 1.00 | 1.76 | 1.05 |
| SWE-Agent | 1.43 | 1.18 | 1.45 | 1.15 |

Where $r(BS) = T(BS = 2^n) / T(BS = 2^{n-1})$ is the throughput gain ratio. Values $\approx 1$ (e.g., LangChain at BS=128, both Sys) mean throughput **doesn't improve** when doubling batch — confirming CPU saturation. COMB's $B_{cap}$ pick is essentially "the largest batch before $r(BS) \approx 1$."

**Effectiveness inversely proportional to $r(BS)$**: when $r(BS) \approx 1$ (saturated), COMB delivers ~2× P50 latency reduction; when $1 < r(BS) < 1.5$ (partial saturation), micro-batching alone is less effective but overlapping compensates.

> [!example]- COMB open-loop measurement (Section 5.2)
>
> Open-loop arrival with Poisson rate $\lambda$. Baseline uses $N_{max}=256$ (max GPU utilization). COMB tested at $N_{cap}=64$ (knee of throughput curve).
>
> At $\lambda = 13$ req/s, baseline CPU utilization $\rho_{CPU} = 3.09$ (over-subscribed 3×), CPU contention dominates and queuing delay spikes. COMB at $N_{cap}=64$ holds $\rho_{CPU}$ in range 0.89–1.13, keeping CPU near optimal load.
>
> Results:
> - Service latency P50/P90: **2.9× / 3.9× lower** vs baseline
> - Total latency P50/P90: **1.6× / 1.8× lower**
> - Throughput: **1.7× higher**
>
> Lower $N_{cap}$ values (48) increase service time (under-utilize); higher values (82, 96) reduce $N_{cap}$ benefits.

### Component 3 — MAS: Mixed Agentic Scheduling (Section 4.2)

The second algorithmic contribution, for heterogeneous workloads where requests have different (CPU vs GPU) bottlenecks.

**The problem MAS targets**: standard FCFS scheduling (vLLM, SGLang) admits requests one queue. When a chatbot server gets a mix of pure-LLM chat requests (GPU-heavy) and tool-using agent requests (CPU-heavy), the dominant request type monopolizes admission. The minority type sits in queue while head-of-line blocked.

**MAS's two policies:**

1. **Request-type-aware concurrent admission**: separate execution queue caps $E_{cap,CPU}$ and $E_{cap,GPU}$ for CPU-heavy and GPU-heavy requests respectively. Within each cap, the request type cannot exceed its allocation; bursts of one type don't starve the other.
2. **Shared reserved queue $E_{cap,shared}$** for elastic absorption. Requests that exceed their type-cap can spill into this shared queue, providing elasticity without compromising the minority's protection.

The caps are derived from COMB analysis: $E_{cap,CPU} = N_{cap}$ from COMB on CPU-heavy workloads; $E_{cap,GPU}$ is the remaining concurrency budget allocated for GPU-heavy.

**Empirical evaluation** (Sys 2, request-type probability $p_{LLM} \in \{0.25, 0.50, 0.75\}$):

| $p_{LLM}$ | Workload type | P50 improvement | P90 improvement |
| --------: | ------------- | --------------: | --------------: |
| 0.25 | GPU-heavy minority | **1.82× lower** | **1.78× lower** |
| 0.25 (Sys 1) | GPU-heavy minority | **2.37× lower** | **2.49× lower** |
| 0.50 | GPU-heavy | 1.39× lower | 1.18× lower |
| 0.50 | CPU-heavy | 1.1× lower | 1.1× lower |
| 0.50 (Sys 1, avg) | Total | **1.62× lower** | **1.30× lower** |
| 0.75 | CPU-heavy minority | 2.09× lower | 2.15× lower |

The minority request type benefits most — MAS protects against head-of-line starvation.

> [!note]- Why MAS matters: the chat+agent workload mix
>
> Production chatbot services increasingly host both pure-conversational requests (no tools) and agentic requests (with tool calls). Without MAS, the dominant type (whichever has more arrivals) dominates admission, and the minority type sees inflated P50/P90 latency despite using a *different resource bottleneck*.
>
> MAS's contribution is to recognize that agentic requests are fundamentally a *different scheduling class* than chat requests — they need their own admission discipline. This generalizes beyond LLM serving: any heterogeneous-workload system where requests compete on different resources benefits from the same principle.

## Headline evidence

### COMB standalone batch (Section 5.1)

Standalone batch processing BS=128, comparing baseline (MP all 128), micro-batching ($B_{cap}=64$), and COMB ($B_{cap}=64$, varying $s$):

| Workload | System | Baseline (s) | Micro-batching (s) | COMB best (s) |
| -------- | ------ | -----------: | -----------------: | ------------: |
| SWE-Agent | Sys 1 | 61.4 | 71.4 (worse) | **53.3** |
| SWE-Agent | Sys 2 | 38.5 | 55.9 (worse) | **45.1** |
| Web-Augmented Agent | Sys 1 | 20.3 | 21.4 (similar) | **21.1** |
| Web-Augmented Agent | Sys 2 | 14.2 | 16.5 (worse) | **15.6** |

Note: micro-batching alone *can hurt* (SWE-Agent on Sys 2: 55.9s vs baseline 38.5s) when CPU isn't saturated; COMB recovers via overlap. SWE-Agent's $r(BS) \approx 1.15$ at $BS=128$ means micro-batching alone is inefficient.

### Open-loop COMB and MAS (Section 5.2 + 5.3)

Already summarized in the headline table above. Key takeaways:

- COMB's improvement scales with CPU load: at $\lambda \approx 11\text{-}14$ req/s, throughput improves 1.7×; below this, fewer benefits.
- MAS protects minority request types but doesn't hurt majority — Pareto-improving for heterogeneous mixes.

### Ablation on CPU-constrained platform (Section 5.4)

A third system with **16-core Intel Emerald Rapids + same RTX-6000 Pro Blackwell GPU**. CPU is 1/4 the capacity of Sys 1 (64-core) but GPU is identical. Tests whether COMB/MAS still work when CPU is the tight bottleneck:

| Configuration | Web-Augmented Agent first-batch time |
| ------------- | -----------------------------------: |
| Baseline | 51.5 s |
| Micro-batching | 26.4 s (1.95× faster) |
| COMB ($s=16$s) | **40.1 s** (worse than micro-batching) |

On 16-core CPU, $r(BS) \approx 1$ (already saturated), so micro-batching alone captures the gain; COMB's overlap *reduces* total time but at a tail-latency cost (~1.05× P90). This is the bounded regime — paper notes COMB effectiveness is "inversely proportional to gain ratio."

> [!example]- Energy profiling — secondary contribution
>
> Section 5.4.2 (not shown in detail above) measures CPU and GPU energy across workloads. Finding: agentic AI has substantial CPU dynamic energy overhead beyond what GPU-centric studies report. Direct quantitative number: total CPU energy ranges from 20% (LLM-dominated) to 70%+ (tool-dominated) of E2E system energy. Implication for sustainable serving deployments: CPU power optimization matters more than current frameworks acknowledge.

## Strengths and limitations

**Strengths.**

- **First end-to-end measurement** across a representative agentic workload suite on asymmetric hardware. The compile-time taxonomy is a useful conceptual tool independent of the specific paper.
- **Real algorithmic contributions** (COMB + MAS), not just characterization. Both are simple enough to implement, principled enough to defend.
- **Quantifies CPU parallelism efficiency** — the empirical observation that CPU saturates at $1\text{-}2 \times \#\text{cores}$ is reusable beyond agentic serving.
- **Distinguishes service latency from total latency**: total = service + wait. Many prior papers report only service latency; MAS's wait-latency reduction is a separable contribution.

**Limitations.**

> [!warning] Tool selection is biased toward "embarrassingly parallel CPU work"
> The five workloads chosen (Toolformer, SWE-Agent, RAG, ChemCrow, Web-Augmented Agent) all have well-defined CPU bottlenecks (retrieval, RDKit, summarization, Bash). Modern agentic workloads with **network-dominated tools** (WebFetch, vector DB across DC, slow third-party APIs) have *different* bottleneck structure — neither CPU nor GPU bound, just *I/O wait*. COMB doesn't apply to I/O-wait-dominated tools.

- **Small models (≤32B)**. The CPU/GPU ratio shifts dramatically at flagship scale (300B+). Whether COMB's $B_{cap}$ scales to large models isn't measured.
- **Closed-loop and open-loop only**, no real production traffic replay. Synthetic Poisson arrivals are simpler than the on/off bursty patterns the CPU-Centric paper's own MAS uses (two-state ON/OFF model).
- **MAS evaluation uses synthetic $p_{LLM}$ mix** with two-state ON/OFF arrival. Real chat+agent workloads have correlated bursts (user sessions vs agent campaigns) that this doesn't capture.
- **No comparison to [[continuum|Continuum]]**, which tackles the same problem (multi-turn agent scheduling) from the KV-cache TTL angle. COMB + Continuum are likely composable but the paper doesn't address this.
- **No code release** as of June 2026.

> [!bug] CPU efficiency factor estimation is empirical
> The $B_{cap} \approx 1\text{-}2 \times \#\text{CPUs}$ heuristic comes from offline profiling on the workloads tested. For a new workload or hardware, you must re-profile. The paper doesn't provide an online estimator.

## What this means

This paper does two useful things at once: **diagnoses** where CPU bottlenecks actually live in agentic serving (with workload + hardware sensitivity quantified), and **prescribes** two scheduling algorithms that close the gap. The diagnosis is the more durable contribution; once everyone agrees on the failure modes, alternative scheduling algorithms will follow.

Three predictions for 2027:

1. **COMB-style CPU-GPU overlap micro-batching enters vLLM/SGLang as a first-class scheduling mode**, probably as `--enable-cpu-tool-overlap` with auto-detected $B_{cap}$.
2. **MAS-style request-type-aware admission becomes the default scheduler** for any deployment hosting both chatbot and agent traffic. The "single FCFS queue" pattern from chat-only deployments doesn't survive heterogeneous workloads.
3. **The "tool dominates 70-90%" claim becomes nuanced.** It's true on specific workloads with specific hardware (RAG on Sys 2, ChemCrow Heavy). It's false on others (Toolformer on Sys 1 = LLM-bound). Future characterization papers will adopt this paper's per-workload-per-hardware reporting style instead of single-number summaries.

This paper does **not** solve:

- **Memory bandwidth bottlenecks** — focus is CPU compute, not memory bandwidth. Different optimization story.
- **Multi-turn KV-cache retention** — orthogonal to and composable with [[continuum|Continuum]]'s TTL-based pinning.
- **Distributed serving** — single-node only; multi-node coordination of COMB is unexplored.
- **Cost-aware scheduling** — energy is measured but not used as a scheduling input.

## Source code & reproduction

```bash
# Code not yet released as of June 2026.
# Implementation built atop:
pip install vllm==0.14.0 torch==2.8.0
```

**Reproduction protocol** (from Section 3.1):

| Component | Configuration |
| --------- | ------------- |
| vLLM | 0.14.0 |
| PyTorch | 2.8.0 |
| Sys 1 | 64-core Intel Granite Rapids + Nvidia RTX-Pro 6000 Blackwell, 512 GB DDR5 |
| Sys 2 | 72-core Nvidia Grace + Nvidia H200, 480 GB LPDDR5 |
| Models | GPT-OSS-20B, GPT-J-6B (vLLM), Qwen2.5-Coder-32B (vLLM) |
| Tools | ENNS retrieval, WolframAlpha API, URL fetch, LexRank summarization, Bash+Python, RDKit conformer gen |
| Workloads | Toolformer, SWE-Agent, RAG (Haystack), ChemCrow, Web-Augmented Agent (LangChain) |
| Each run | 5 trials for statistical variance |

## Related reading

- [[continuum]] — Continuum: complementary KV-cache TTL system for multi-turn agent scheduling. COMB and Continuum are orthogonal (one addresses CPU oversubscription, the other addresses KV cache eviction); composing them would attack both bottlenecks simultaneously.
- [[agent-serving-challenges]] — Broader survey of agent serving challenges; this paper's characterization data is the most rigorous data point referenced there.
- [[multi-turn-optimization]] — Multi-turn KV reuse; orthogonal to CPU-side optimizations here.
- [[vllm]] — Base inference engine used in measurement (vLLM 0.14.0).
- [[continuous-batching]] — Standard batching policy this paper extends with COMB.
- [[paged-attention]] — KV cache primitive; paper notes "PagedAttention reduce memory fragmentation but they do not eliminate the underlying capacity and bandwidth limits of GPU memory."
- [[prefill-decode-disaggregation]] — Another batching/scheduling axis; PD-disagg and COMB are composable.

## References

- Ritik Raj, Souvik Kundu, Ishita Vohra, Hong Wang, Tushar Krishna. *Towards Understanding, Analyzing, and Optimizing Agentic AI Execution: A CPU-Centric Perspective.* arXiv:2511.00739, November 2025 (v3 April 2026). https://arxiv.org/abs/2511.00739
- ENNS retrieval bottleneck reference: Wang et al., RAG-200GB study (cited as [55] in the paper).
- WebArena CPU-latency analysis: Yao et al. (cited as [73] in the paper) — "partial tool execution can cut request latency by up to 38.8%, highlighting tool execution as a major source of E2E latency."
- LMStream micro-batching for stream processing (cited as [38]).
- Ayo stage-local micro-batching (cited as [63]) — predecessor of cross-stage overlap.
