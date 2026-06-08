---
title: "InferCept: Efficient Intercept Support for Augmented LLM Inference"
category: llm-serving-for-agents
tags: [infercept, kv-cache, interception, augmented-llm, vllm, paged-attention, icml-2024, agent-serving, paper-review]
created: 2026-06-08
updated: 2026-06-08
status: mature
paper: arXiv:2402.01869
code: https://github.com/WukLab/InferCept
---

# InferCept: Efficient Intercept Support for Augmented Large Language Model Inference

> [!info] Paper metadata
> - **Paper**: [arXiv:2402.01869](https://arxiv.org/abs/2402.01869) — *InferCept: Efficient Intercept Support for Augmented Large Language Model Inference*, ICML 2024 (PMLR 235); v1 2024-02-02, v2 2024-05-30
> - **Code**: [WukLab/InferCept](https://github.com/WukLab/InferCept) — released by UCSD WukLab (vLLM fork)
> - **Authors**: Reyna Abhyankar\* (UCSD), Zijian He\* (UCSD), Vikranth Srivatsa (UCSD), Hao Zhang (UCSD), Yiying Zhang (UCSD)
> - **Affiliation**: UC San Diego (all authors)
> - **Implementation**: vLLM fork; leverages PagedAttention for KV memory management

> [!important] Where InferCept fits in the agent-serving lineage
> InferCept is **the first LLM inference framework designed for augmented LLMs with interceptions** — the predecessor that both [[continuum|Continuum]] and [[mori|MORI]] explicitly cite as baseline. It established the core thesis that **a tool call / human response is an *interception*, not an end-of-request**, and that the KV cache during that interception is *temporarily unused context* worth managing as a first-class resource. The two failure modes that Continuum and MORI later target — per-turn queueing delay, and unbounded retention under long-tailed tool latencies — both inherit from gaps in InferCept's reload-cost-only decision model.

---

## Summary (read this if you have 2 minutes)

**What it is.** InferCept (ICML 2024, UCSD WukLab) is a vLLM-fork LLM inference framework purpose-built for **augmented LLM workloads** — LLMs that are *intercepted* by external entities (tool calls, human chat turns, image-generation calls, TTS, virtual environments). It frames the unsolved problem as "what to do with the KV cache during an interception" and proposes **min-waste interception**: dynamically pick among `Discard` / `Preserve` / `Swap` per request based on which option minimizes GPU memory waste.

**The one idea.** **Quantify GPU memory waste of each KV-cache option with a closed-form equation, then pick the option that minimizes it per-request, per-interception.** Three sub-pieces make it work:

1. **Waste equations** — closed-form formulas for `WasteDiscard`, `WastePreserve`, `WasteSwap`, and `WasteChunkD` (chunked recompute) parameterized by interception duration, context length, batch size, and the per-token KV memory $M$.
2. **Engineering improvements that shrink each option's waste term** — *swap pipelining + chunking* (overlap CPU↔GPU PCIe with forward kernels at layer granularity) eliminates 96% of `Swap` waste; *recomputation chunking* (split a recompute into S-sized chunks where S is the GPU saturation point) cuts `Discard` waste by ~half; *budgeted swap-out* prevents the PCIe link from being saturated.
3. **Min-waste scheduling** — at every iteration, sort intercepted requests by descending `min(WastePreserve, WasteChunkD)`, then assign each to swap (if within swap budget), preserve, or discard. A `t_now − t_call` dynamic estimator handles the highly-variable interception durations (chatbot human responses, image gen) and achieves 93% of oracle performance.

Remove the waste equations and InferCept reduces to ad-hoc heuristics; remove swap pipelining and `Swap`'s 26% wastage destroys throughput; remove the min-waste scheduler and the system devolves to vLLM-style "treat interception = end of request."

**Headline result.** Mixed-workload evaluation across **GPT-J-6B, Vicuna-13B, Llama-3-70B** on **A100 GPUs**, with six augmentation types (math, QA, VE, chatbot, image, TTS):

| Setting | Baseline | InferCept | Improvement |
| ------- | -------- | --------- | ----------- |
| **6B / 1 GPU** | vLLM (Discard) | **1.6× higher request arrival rate** at same normalized latency | 1.9×–5.7× lower normalized latency |
| **13B / 1 GPU** | vLLM | 1.25× higher request rates | 3.1× higher load with smallest TTFT |
| **13B / 2 GPUs (TP)** | vLLM | 1.8× higher request arrival rates | **1.6×–10× lower normalized latency** |
| **70B / 4 GPUs (TP)** | vLLM | 2× higher request arrival rate at same RPS | **1.3×–12× lower normalized latency** |
| Throughput vs SOTA | vLLM | **1.6×–2× higher throughput**; 2× more completed reqs/sec | The abstract's headline number |

GPU memory waste: vanilla vLLM ~25%, full InferCept **0.69%** — eliminates >60% of recompute waste and 96% of swap waste.

**Why it matters.**

- **First-mover on the augmented-LLM problem.** Before InferCept, every serving system (vLLM, Orca, DeepSpeed-Inference, TensorRT-LLM, FastServe) treated interception as termination — discard context, re-initiate when the tool returns. InferCept named the problem ("min-waste interception") and the three actions (`Discard` / `Preserve` / `Swap`) that subsequent work — Continuum, MORI, Autellix, Pie — all reuse as vocabulary.
- **The closed-form waste model is the durable contribution.** Even after [[continuum|Continuum]] adds the per-turn queueing-delay term that InferCept's model omits, the four `Waste*` equations remain the canonical framing. They appear (with extensions) in MORI's two-tier idleness derivation and in Autellix's program-aware scheduling.
- **Recomputation chunking is genuinely novel.** Splitting a recompute into S-sized chunks (S = GPU query-token saturation point) so that recompute fills the unused GPU cores during decode is a clean co-optimization that Sarathi-Serve's chunked prefill later generalized.
- **Open limitations seeded the next two years of work.** (1) The model uses *reload cost* alone — no per-turn queueing penalty — which Continuum fixes with TTL. (2) No bound on retention time — a hung tool call can pin KV indefinitely — which Continuum's TTL expiry fixes. (3) Single-tier GPU only — no CPU/disk idleness tier — which MORI addresses with the two-tier offloader.

---

# Depth (drill-down starts here)

## Background: what makes augmented LLM inference different

An *augmented LLM* is an LLM that calls external entities mid-generation — a calculator, a knowledge-base QA tool, a virtual environment (ALFWorld), a chatbot human, an image-generation model (Stable Diffusion), a TTS model (Bark). Each of these creates an **interception**: the LLM decoding pauses, waits for an external response, and then must resume. The paper's §2 measures interception properties across six augmentations (Table 1 of the paper, paraphrased here):

| Augmentation | Int. time (s) — mean, var | Num interceptions / req | Context length (tokens) |
| ------------ | -------------------------- | ----------------------- | ----------------------- |
| **Math (calculator, GSM8K)** | 9e-5, 6e-5 | 3.75, 1.3 | 1422, 738 |
| **QA (Multihop, Wikipedia)** | 0.69, 0.17 | 2.52, 1.73 | 1846, 428 |
| **Virtual Env (ALFWorld)** | 0.09, 0.014 | 28.18, 15.2 | 2185, 115 |
| **Chatbot (ShareGPT)** | 28.6, 15.6 | 4.45, 1.96 | 753, 703 |
| **Image (ChatGPT + SD)** | 20.03, 7.8 | 6.91, 3.93 | 1247, 792 |
| **TTS (ChatGPT + Bark)** | 17.24, 7.6 | 6.91, 3.93 | 1251, 792 |

Three structural observations from this table drive the entire design:

1. **Bimodal interception times.** Short-running automated augmentations (Math sub-ms, QA sub-second, VE ~90 ms) vs. long-running human-or-model augmentations (Chatbot 28.6 s, Image 20 s, TTS 17 s). No single policy is universal.
2. **Large variance for the long-running ones.** Chatbot variance is 15.6 s vs. a 28.6 s mean — the *next* interception of the same type could take 13 s or 44 s. Static profiling can't predict this.
3. **Context lengths are big.** Even VE has 2185 tokens; Image/TTS 1247–1251. Storing this in GPU during a 28 s human-response window is enormous waste; tossing it and recomputing is a massive prefill bill. There is no "small KV" simplification available.

Existing inference systems (vLLM, Orca, DeepSpeed-Inference, FastServe, TensorRT-LLM) all treat the interception as the **end** of a request and discard the KV. When the interception finishes, a new request is enqueued and re-prefilled from scratch. The paper measures this `Discard` strategy and reports that **recomputation accounts for 37–40% of total model forwarding time** under mixed workloads — i.e., almost half the GPU's flops are spent re-doing prefill of context that existed seconds ago.

The naive fix `Preserve` (keep KV pinned during the interception) wastes GPU memory for the entire interception window. The next naive fix `Swap` (move KV to CPU during interception, swap back when done) avoids GPU memory waste but blocks foreground compute on PCIe bandwidth and incurs many small kernel launches (a single intercepted context is scattered across many PagedAttention blocks).

InferCept's contribution is to make **the choice between these three actions principled**, per-request, per-interception.

## The four waste equations

The waste model quantifies "GPU memory × time" lost during an interception. Let $C_i^j$ = number of context tokens of request $i$ at interception $j$; $M$ = per-token KV memory; $T_{fwd}(C)$ = forward-pass latency for a context of size $C$; $T_{INT}^j$ = duration of interception $j$; $T_{swap}(C)$ = swap latency for $C$ tokens; $C_{other}$ = sum of context lengths of all other concurrently running requests; $C_{batch}$ = total batch context.

**Discard waste** — recomputation cost + stall on other requests:

$$\text{WasteDiscard}_i^j = T_{fwd}(C_i^j) \times C_i^j \times M + T_{fwd}(C_i^j) \times C_{other} \times M \quad (1)$$

The two terms: (a) memory occupied while recomputing the discarded request, (b) memory other-running requests must hold while waiting for the all-at-once recompute to finish (this is the *iteration-time blowup* term).

**Preserve waste** — GPU memory held idle during the entire interception:

$$\text{WastePreserve}_i^j = T_{INT}^j \times C_i \times M \quad (2)$$

Linear in interception duration. Cheap for short interceptions (Math, QA) but catastrophic for long ones (Chatbot, Image, TTS).

**Swap waste** — both swap directions block compute:

$$\text{WasteSwap}_i^j = 2 \times T_{swap}(C_i^j) \times C_{batch} \times M \quad (3)$$

Factor of 2 = swap-out + swap-in. The cost is paid in *other* requests' idled memory while the swap kernels hold the GPU.

**Chunked-discard waste** — split the recompute over $n$ iterations of chunk size $C_i^j / n$:

$$\text{WasteChunkD}_i^j = \frac{T_{fwd}(C_i^j) \times C_i^j \times M}{2} + n \times T_{fwd}\!\left(\frac{C_i^j}{n}\right) \times C_{other} \times M \quad (4)$$

The first term is *half* of the all-at-once recompute cost (because chunked recompute interleaves with decode and uses otherwise-idle GPU cores); the right term replaces a single huge $T_{fwd}(C_i^j)$ with $n$ smaller $T_{fwd}(C_i^j/n)$ — usually a net win when $n \times T_{fwd}(C/n) \leq T_{fwd}(C)$, which holds whenever the GPU's query-token saturation point $S$ is reached.

The min-waste scheduling rule (eq. 5):

$$\text{Waste}_i^j = \min\!\left(\text{WastePreserve}_i^j, \text{WasteChunkD}_i^j\right)$$

Swap is treated as a *budgeted* third option — applied to the requests with the *highest* `min(Preserve, ChunkD)` waste, up to a swap budget that respects PCIe bandwidth and free-CPU-memory constraints.

> [!warning] What the waste model omits — and why Continuum / MORI exist
> Eq. 1's $C_{other}$ term captures the *concurrent* iteration-time blowup but **not the queueing delay** that a *returning* request pays when it's released back into a queue full of newly-admitted work. If a request was preserved or swapped, when the interception ends it must wait for the existing batch to make room — and that wait time *isn't* counted in Eqs. 2 or 3. [[continuum|Continuum]]'s cost-benefit model adds an `OutOfOrderCost` term precisely to fix this.
>
> Eq. 2 also has no upper bound on $T_{INT}$. If a `pytest` tool call hangs for 60 s, `WastePreserve` grows linearly — the model correctly says "switch to discard" past a threshold — but the *already-pinned* KV from the moment the tool started has no eviction mechanism. [[continuum|Continuum]]'s TTL expiry is the bound that InferCept's static decision lacks.

## Three components in detail

### Component 1 — Swap pipelining and chunking (§4.1)

Naive `Swap` (the strawman) launches CUDA memcpy kernels synchronously when the interception happens, blocking foreground forward passes for the entire $T_{swap}(C)$ window. The paper measures this and reports `Swap` *itself* wastes 26% of GPU resources and >25% of total workload time is spent waiting for swap kernels.

**Swap pipelining** treats each model layer's swap as a separate pipeline stage. When the swap kernel for layer $i+2$ is launched, the swap for layer $i+1$ is moving data, and layer $i$'s context has already been freed and is participating in normal forward computation. This is essentially **layer-level pipelining of memcpy and matmul** — equivalent to ZeRO-Offload's overlap but on a per-token-block granularity.

**Swap chunking** further subdivides a single swap-out or swap-in across multiple model-forwarding iterations. The paper computes a *swap limit* $N_i = T_{fwd}^{-1}(B_i)$ where $B_i$ is the batch size and $T_{fwd}^{-1}$ inverts the per-iteration latency — i.e., how many tokens can be swapped "for free" hidden behind one iteration's forward latency.

**Swap-in / swap-out budgeting** is a constrained optimization. At each iteration, choose how many tokens to swap out and in such that:

1. Total in + out ≤ $N_i$ (don't exceed the swap-hideable budget)
2. Swap-out memory ≤ free CPU + swap-in (CPU memory capacity)
3. Swap-in + new-token memory ≤ swap-out + free GPU (GPU memory capacity)

This is a small LP solved at scheduling time. The end result: 96% of `Swap`-equation waste is eliminated.

### Component 2 — Recomputation chunking (§4.2)

Decoding requests need GPU cores for *one query token* per request; recomputation needs cores for *the entire context length* of one request. These are at opposite ends of the GPU's compute/memory utilization spectrum. The key insight: **a batch of decoding requests usually cannot fill all GPU cores before running out of GPU memory** — there's spare compute capacity that recomputation can fill without violating the memory budget.

The paper defines the **GPU saturation point** $S$ as the query-token count beyond which iteration time grows monotonically. $S$ is obtained from offline profiling per model architecture. The chunk size for recomputation is then $S - \text{running\_group\_size}$ — exactly the spare query-token capacity per iteration.

Naive `Discard` recomputes a context in one giant iteration, paying $T_{fwd}(C_i^j)$ once but blocking all other requests for that one iteration. Chunked recomputation pays $n \times T_{fwd}(C_i^j / n)$ across $n$ iterations but *each iteration is shorter*, so other-request blocking is minimized — Eq. 4 quantifies the net win.

This is conceptually adjacent to Sarathi's *chunked prefill* (also 2023) but with two differences: (a) Sarathi chunks new-request prefills to balance per-iteration latency; InferCept chunks *recomputations* of intercepted requests; (b) InferCept's chunk size is dynamically determined by the running batch's spare capacity, not a static config.

### Component 3 — Inter-request action decision and scheduling (§4.3)

At each scheduling step, InferCept maintains **three queues**:

1. **Running queue** — requests currently decoding.
2. **Swap queue** — requests that have been resumed but were previously swapped out during their interception. Sorted by **original arrival time** (FCFS preserved across the interception, not bumped to the back).
3. **Waiting queue** — discarded-resumed + brand-new + previously-running-but-evicted requests. Same FCFS-by-original-arrival ordering.

The two-phase decision per iteration:

**Phase 1 — Intercepted-request scheduling.** For each newly-intercepted request, compute `min(WastePreserve, WasteChunkD)` (Eq. 5). Sort all intercepted requests in **descending** order of this waste, then assign them to swap (one by one) until the swap-out budget for this iteration is exhausted. Whatever's left over either preserves or discards based on which Eq. 5 component was smaller.

**Phase 2 — Resume/admit scheduling.** Pull from waiting + swap queues in FCFS order, admit new tokens until the GPU saturation point $S$ is hit. The swap queue is maintained *separately* from the waiting queue because swap-in memory is *additional* to GPU resources and shouldn't compete with new admissions.

The architecture diagram from the paper (Figure 1) shows this end-to-end:

![InferCept and alternative approaches: the four timelines (Today's Discard, Discard, Swap, vs. InferCept's MinWasteDiscard + chunked recompute + MinWasteSwap + Preserve) (paper Fig. 1)](EN/wiki/llm-serving-for-agents/infercept-figs/fig1-architecture.png)

The top three rows are the strawman timelines (recompute-blocked, swap-blocked); the bottom three rows are InferCept's chunked-recomputed Discard, pipelined Swap, and clean Preserve, with the central `Executor + Offline Profiler` block making per-request action decisions based on the waste equations.

### Component 4 — Interception duration estimation (§4.4)

WastePreserve (Eq. 2) needs $T_{INT}$, but for chatbot / image / TTS interceptions the duration is highly variable (variance ≈ mean for chatbot). Offline profiling won't help.

**Dynamic estimator**: $\hat{T}_{INT} = t_{now} - t_{call}$ — the longer an interception has been running, the larger the estimated remaining duration. This is a simple form of **memoryless / exponential-tail prior** and works because long-tailed interception distributions are roughly heavy-tailed: the longer it's been waiting, the longer it's likely to wait still.

The paper reports this estimator achieves **93% of the performance of an oracle** that knows exact interception durations — i.e., 7% performance left on the table for sophistication that probably isn't worth it.

## Headline evidence

**End-to-end performance (Figure 2, mixed workload):**

![End-to-end performance on mixed workload across 6B/13B/70B models: normalized latency (top), throughput (middle), TTFT (bottom). InferCept (green) sustains higher load with lower latency and lower TTFT than vLLM (red), ImprovedDiscard, Preserve, Swap. (paper Fig. 2)](EN/wiki/llm-serving-for-agents/infercept-figs/fig2-e2e-results.png)

Three rows of plots: (top) normalized latency vs request rate, (middle) throughput, (bottom) time-to-first-token. Across all four configurations (6B, 13B, 13B-TP2, 70B-TP4), InferCept (green) maintains the right-most curve in normalized latency (sustains highest load) and lowest TTFT.

**Technique breakdown (Figure 3):**

![Technique breakdown: incrementally adding each InferCept component to vanilla vLLM. ImprovedDiscard alone -24.5% latency; +MinWasteDiscard +7.8%; +MinWasteSwap +12.7%; +HeuristicPreserves +46.1%; full InferCept +46.4% more, final 0.69% memory waste. (paper Fig. 3)](EN/wiki/llm-serving-for-agents/infercept-figs/fig3-technique-breakdown.png)

The cumulative improvement per technique on Vicuna-13B / mixed workload @ 2 RPS:

| Variant | Cumulative Δ vs vLLM | GPU memory waste |
| ------- | -------------------- | ---------------- |
| vLLM (Discard) | 0 % | ~25 % |
| ImprovedDiscard | −24.5 % latency | ~20 % |
| + MinWaste Discard (chunked recompute) | +7.8 % | ~21 % |
| + MinWaste Swap (budgeted, pipelined) | +12.7 % | ~10 % |
| + Heuristic Preserves | +46.1 % | ~1.5 % |
| + Min-waste schedule (full InferCept) | +46.4 % | **0.69 %** |

The two biggest jumps come from **adding Preserve as an option** (the first time the system can pin instead of recompute/swap, +46.1%) and from **switching the heuristic preserve to min-waste-based dynamic preserve** (+46.4%). The recomputation chunking and swap pipelining are smaller individual contributions (~10% each) but are what makes the discard/swap options non-terrible so that the min-waste choice has good options to choose from.

**Single-augmentation workloads:** QA-only is up to 2.3× faster than vLLM (most QA tool calls are short → preserve dominates); Chatbot-only is 1.9× faster (long human-response durations → swap dominates, with chunked recompute used selectively).

**Distributed setting amplifies the gain:** 13B on 2 GPUs (tensor parallel) → 1.6×–10× normalized latency improvement, larger than single-GPU 13B. The reason: more KV cache fits in aggregated GPU memory → more concurrent requests admitted → more interceptions → InferCept has more decisions to optimize.

**70B on 4 GPUs:** 2.4× higher load at same TTFT, 1.3×–12× lower normalized latency. The Llama3-70B's grouped-query attention (GQA) compresses KV — this *helps* Preserve and Swap (less data to hold/move) and *moderates* the relative gain of optimal scheduling, but the absolute gain remains large because the baseline waste was also enormous.

> [!example]- Chatbot single-augment breakdown (paper §5.2)
> For the single-augment Chatbot workload, the paper reports that chunking and pipelining (which target `Swap` and `Discard`) contribute **54% of total speedup** — far more than for the mixed workload — because Chatbot's long interceptions allow more requests to execute concurrently, triggering more swaps and recomputes that the chunking/pipelining make non-blocking.

## Strengths and limitations

**Strengths.**

- **First system to name and solve the augmented-LLM interception problem.** The vocabulary `Discard` / `Preserve` / `Swap` and the four waste equations are now canonical — every subsequent agent-serving paper (Continuum, MORI, Autellix, Pie) builds on or contrasts against this taxonomy.
- **Clean closed-form decision rule.** Min-waste selection is interpretable, debuggable, and the offline profiling needed to instantiate it (one $T_{fwd}$ curve, one $T_{swap}$ curve, one saturation point $S$) is one-time and lightweight.
- **Modular & orthogonal.** The paper explicitly notes that the techniques are designed to be integrable into DeepSpeed-Inference, Orca, TensorRT-LLM. Most of the engineering is at the scheduler layer, not the kernel layer (no custom CUDA needed beyond the swap pipelining wrappers).
- **Recomputation chunking is genuinely novel.** Sarathi-Serve's chunked prefill is the closest peer but operates on new-request prefills, not intercepted-request recomputes. InferCept's variant is dynamic (chunk size = saturation point − running group size) and stays within the GPU memory boundary.

**Limitations (what later work fixed).**

- **Reload cost only.** The waste model accounts for $T_{fwd} \times C_{other}$ (concurrent-iteration blowup) but not the *per-turn queueing penalty* a returning request pays when re-admitted into a busy queue. [[continuum|Continuum]]'s `OutOfOrderCost` term is the fix.
- **No retention bound.** A `pytest` that hangs for 60 s under Preserve burns memory the whole time; Eq. 2 says "switch to discard" but only on the *next* decision — there's no eviction of already-preserved KV. [[continuum|Continuum]]'s TTL expiry mechanism is the fix.
- **Single-tier (GPU) only.** Swap targets *CPU memory* as a binary "GPU vs not-GPU" hierarchy. No notion of GPU-resident-but-idle vs CPU-resident-but-warm vs disk. [[mori|MORI]]'s two-tier idleness metric is the fix.
- **Interception is binary.** A request is either fully in `Discard`, fully `Preserve`, or fully `Swap` — no support for partial pinning of recent vs. ancient KV blocks. Prefix caching (SGLang/Preble) and Pie's programmable cache are the orthogonal extensions.
- **Workload-specific tuning.** $S$ (saturation point), swap budgets, and the heuristic-preserve threshold are profiled per (model, hardware) tuple. Cluster operators with mixed model fleets pay this cost N times.
- **No discussion of multi-tenancy SLOs.** The paper optimizes mean throughput + mean latency. There's no per-tenant fairness, no priority, no SLO-aware decision rule. Continuum's program-level FCFS is the first agent-aware fairness mechanism in this lineage.

## Implementation notes

- **Base:** vLLM (Kwon et al., SOSP 2023). PagedAttention is reused as the KV memory manager — InferCept's scheduling layer sits above page allocation.
- **Four key components in code:** scheduling, waste calculation, chunked recompute + swap, augment-specific knobs (interception duration profiles, chunking budgets per augment type).
- **No custom CUDA kernels.** The swap pipelining uses standard CUDA streams + memcpy; the chunked recompute uses vLLM's existing prefill kernel with smaller chunk sizes.
- **Profiling overhead:** $T_{fwd}$ and $T_{swap}$ curves are profiled once per model/hardware; not re-profiled at runtime.
- **Repo:** [WukLab/InferCept](https://github.com/WukLab/InferCept). The vLLM version is from early 2024 — predates a lot of vLLM's later kernel improvements (FlashAttention-3, FP8, etc.) so contemporary baseline comparison would need re-running on a more recent vLLM.

## What this means

**For the agent-serving research thread.** InferCept established the canonical taxonomy that all later work uses. The natural successor work each picks one limitation:

- **[[continuum|Continuum]] (Nov 2025)** — adds per-turn queueing-delay term + TTL expiry bound. Same actions (preserve/discard/swap-equivalent CPU offload), enriched waste model and a hard retention bound.
- **[[mori|MORI]] (May 2026)** — adds the two-tier (GPU HBM + CPU DRAM) hierarchy with a continuous idleness metric, going beyond InferCept's binary GPU-vs-CPU and Continuum's single-tier TTL.
- **Autellix (2025)** — adds program-aware scheduling (multi-request agent programs) at the FCFS-unit level.
- **Pie (SOSP 2025)** — exposes the KV cache as a programmable resource so application code (not just the inference engine) can decide what to preserve.

**For production deployment.** InferCept itself is the simplest-to-deploy of the four — pure scheduler-layer changes, no new hardware tier, no programmable KV. For workloads dominated by short-running automated tools (Math/QA/VE), InferCept's heuristic preserves capture most of the win without the operational complexity Continuum/MORI add. For workloads dominated by long-running variable-latency interactions (Chatbot/Image/TTS), Continuum's TTL or MORI's idleness tiering becomes the better fit.

**2027 expectation.** The InferCept vocabulary (`Discard` / `Preserve` / `Swap` / `MinWaste`) becomes the standard request-state taxonomy in agent-serving APIs. vLLM's `RequestStatus` enum is likely to gain `INTERCEPTED_PRESERVE`, `INTERCEPTED_SWAPPED`, `INTERCEPTED_DISCARDED` states as first-class citizens.

## Related reading

- [[continuum]] — Continuum: TTL-based KV pinning + program-level FCFS; directly extends InferCept's waste model with per-turn queueing-delay term and retention bound.
- [[mori]] — MORI: two-tier (GPU HBM + CPU DRAM) memory offloader with continuous idleness metric; extends InferCept's binary swap decision into a continuous spectrum.
- [[multi-turn-optimization]] — broader context for cross-turn KV reuse, prefix caching, and session management.
- [[agent-serving-challenges]] — agent vs chatbot workload characterization; InferCept's six-augmentation taxonomy fits in here.

## Source code

[github.com/WukLab/InferCept](https://github.com/WukLab/InferCept) — vLLM fork, four key components (`scheduler`, `waste_calc`, `chunked_swap`, `augment_profiles`). Read order: start with `scheduler/min_waste.py` for the Eq. 5 decision logic, then `swap/pipeline.py` for the layer-pipelined memcpy, then `recompute/chunked.py` for the saturation-point-based chunking.

## Paper citation

See [[infercept-citation|sources/papers/infercept/citation.md]] for the full citation, BibTeX entry, and headline numbers.
