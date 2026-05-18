---
title: "Continuous Batching: Dynamic Request Scheduling"
category: llm-inference
tags: [continuous-batching, scheduling, iteration-level, dynamic-batching, throughput]
created: 2026-04-13
updated: 2026-05-13
status: mature
---

# Continuous Batching: Dynamic Request Scheduling

> [!abstract]+ TL;DR
> Batching amortizes the cost of loading model weights but LLM outputs vary wildly in length (few tokens to thousands), making **static batching** waste GPU on whichever request finished first. **Continuous batching** (iteration-level scheduling) dynamically adjusts batch composition at every decode step, inserting new requests as old ones finish — eliminating the convoy effect. Introduced by **Orca (OSDI 2022)**, now the core scheduling mechanism in [[vllm|vLLM]], [[sglang|SGLang]], [[tensorrt-llm|TensorRT-LLM]]. Production deployments see **2–5× throughput** vs. static batching.

## The Problem with Static Batching

Static batching collects a fixed batch, processes all requests together, and waits for **every** request to finish before accepting the next batch.

```
Static Batching (batch of 4):

Time  ──────────────────────────────────────────►

Req A: |████████████████|                          (16 tokens)
Req B: |████████████████████████████████████████|   (40 tokens)
Req C: |████████|                                   (8 tokens)
Req D: |████████████████████████|                   (24 tokens)
        ↑                                       ↑
     Batch start                        Batch ends when B finishes
                                        (C,A,D sit idle)

Utilization: (16+40+8+24) / (4×40) = 88/160 = 55%
Wasted: 45%
```

The **convoy effect**: the entire batch's latency is dictated by the longest request. Shorter requests occupy GPU slots without doing useful work.

| Approach | Batch Formation | Execution | GPU Utilization |
|----------|----------------|-----------|-----------------|
| **Static** | Fixed size | Wait for all | Low (20-55%) |
| **Dynamic** | Time-window collection | Wait for all | Medium (40-65%) |
| **Continuous** | Per-iteration | Requests leave immediately | High (85-98%) |

## How Continuous Batching Works

Introduced by the **Orca** paper (Yu et al., OSDI 2022), which proposed **iteration-level scheduling** and **selective batching**. Orca achieved **36.9x throughput** over NVIDIA FasterTransformer on GPT-3 175B.

Core loop -- executed every decode step:

1. Run one forward pass for all active requests
2. Check which requests generated `<EOS>`
3. Remove completed requests, return results to clients
4. Pull new requests from the waiting queue into freed slots
5. Run prefill for new requests
6. Repeat

```
Continuous Batching Timeline (max concurrency = 4):

Step:    1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
Slot 0: [A  A  A  A  A  A  A  A][E  E  E  E  E  E  E]
Slot 1: [B  B  B  B  B  B  B  B  B  B  B  B][F  F  F...
Slot 2: [C  C  C  C][D  D  D  D  D  D  D  D  D  D][G...
Slot 3: [·  ·  ·  ·  ·  ·  ·  ·  ·  ·][·  ·  ·  ·  ·

Key events:
  Step 4:  C finishes → Slot 2 freed → D enters immediately
  Step 8:  A finishes → Slot 0 freed → E enters
  Step 12: B finishes → Slot 1 freed → F enters
```

Near-100% GPU utilization is achievable under high traffic because freed slots are immediately filled by queued requests.

## Chunked Prefill

Continuous batching solved the convoy effect for decode, but introduced a new problem: **prefill blocking**. A long prompt (e.g., 32K tokens) monopolizes GPU compute and blocks concurrent decode requests, inflating TPOT by 2-30x.

**Chunked prefill** (Sarathi-Serve, Agrawal et al. 2024) breaks long prefills into fixed-size chunks interleaved with decode steps:

```
Chunked Prefill Interleaving:

Step:     1        2        3        4        5
       ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
Pfill: │Chunk1│ │      │ │Chunk2│ │      │ │Chunk3│
       │512tok│ │      │ │512tok│ │      │ │512tok│
       └──────┘ └──────┘ └──────┘ └──────┘ └──────┘
       ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
Decode:│batch │ │batch │ │batch │ │batch │ │batch │
       └──────┘ └──────┘ └──────┘ └──────┘ └──────┘
```

**Token budget model**: each step has a fixed token budget (e.g., 2048). Prefill chunks and decode tokens share this budget, keeping per-step compute predictable.

**TTFT vs TBT trade-off**: Large chunks → lower TTFT (prefill finishes faster) but higher TBT (decode blocked). Small chunks → lower TBT but higher TTFT. Typical sweet spot: 512-2048 tokens.

### Why prefill blocks decode at all

The "blocking" word in *prefill blocks decode* is doing a lot of work. The mechanical reason it happens, in two facts:

**(1) Prefill is genuinely slow at long context.** Forward-pass FLOPs $\approx 2 \cdot N_{\text{params}} \cdot N_{\text{tokens}}$. Llama-70B on a 16 K prompt:

$$
2 \times 70 \times 10^9 \times 16384 \approx 2.3 \times 10^{15} \text{ FLOPs} = 2.3 \text{ PFLOPs}
$$

An H100 at ~989 TFLOPs/s FP16 needs about **2.3 seconds** of pure compute, plus attention's $O(S^2)$ contribution (score matrix is $16{\text{K}} \times 16{\text{K}} \times \text{num\_heads} \times \text{head\_dim}$), plus kernel-launch and memory overhead. Smaller models / shorter prompts scale down, but "seconds for big-model long-prompt prefill" is the order of magnitude.

**(2) A forward pass is one indivisible scheduling unit.** Whatever you packed into a forward pass — prefill tokens, decode tokens, or both — runs as a fused sequence of CUDA kernels with no preemption point. The scheduler can only switch *between* iterations, not inside one.

Combine the two: if you naively put a 16 K prefill in iteration $k$, every decode request in flight has to wait 2.3 s before iteration $k{+}1$ runs. Their TBT for that one iteration jumps from ~30 ms to 2300 ms — a visible "freeze" in streaming output:

```
iter k:    forward([prefill 16K of req X])                ← 2.3 s
iter k+1:  forward([decode 1 token × 64 requests])        ← 30 ms each
```

Chunked prefill's job is to make sure *no single iteration is long enough to freeze anyone*. By packing a small prefill chunk together with a batch of decode tokens per iteration, every iteration both advances the long prefill AND emits a token for the in-flight decoders:

```
iter k:    forward([prefill 512 of X] + [decode 64 requests])   ← ~50 ms
iter k+1:  forward([prefill 512 of X] + [decode 64 requests])   ← ~50 ms
...
```

This is also why scheduling granularity matters: the smaller and more uniform iteration time, the better the tail-latency story.

### The chunk-size math

The TTFT/TBT trade-off has a clean closed form. Set:

- $T$ = total prefill token count (e.g. 16384)
- $c$ = chunk size (tokens per iteration)
- $a$ = per-token incremental cost of a forward pass (compute + memory traffic per token)
- $b$ = per-iteration fixed overhead (kernel launches, scheduler, memory ops — typically a few hundred µs)

Per-iteration time and per-prefill iteration count:

$$
t_{\text{iter}} = a \cdot c + b, \qquad N_{\text{iter}} = T / c
$$

The two metrics:

$$
\text{TBT (other decodes)} = a \cdot c + b
$$

$$
\text{TTFT (this request)} = N_{\text{iter}} \cdot t_{\text{iter}} = \frac{T}{c}\,(a \cdot c + b) = a \cdot T + \frac{b \cdot T}{c}
$$

Two observations fall out:

- **TBT grows linearly in $c$.** Bigger chunk → longer iteration → other decodes wait longer.
- **TTFT has two terms.** $a \cdot T$ is constant (you can't avoid doing the prefill work). $b \cdot T / c$ is the *overhead tax*: every iteration pays $b$, and you need $T/c$ iterations. Small $c$ blows this term up.

So small chunks are *worse* for TTFT, not better — counter-intuitive until you see the math. The sweet spot $c^*$ depends on the ratio of fixed overhead $b$ to per-token cost $a$. For typical inference engines this lands at **512–2048 tokens**:

- Heavy `b` (lots of kernel launches, Python scheduler overhead) → larger $c^*$.
- Light `b` (CUDA graphs, fused scheduling) → smaller $c^*$ is acceptable.
- More decode requests packed per iteration → TBT becomes more sensitive to $c$ → push $c$ smaller.

vLLM's `max_num_batched_tokens` is the knob that sets $c$ (technically the *combined* prefill+decode budget per iteration). 4096 is a common production default.

### FlashAttention's role in chunked prefill

A natural follow-up: doesn't [[paged-attention|FlashAttention]] avoid the $O(S^2)$ attention matrix? Doesn't that let us use larger chunks?

**Short answer**: FlashAttention enables larger chunks by raising the *memory* ceiling, but does not change the TTFT/TBT trade-off itself.

The longer version:

- **FlashAttention does not reduce attention FLOPs.** Attention compute is $O(S^2 \cdot D)$ regardless of implementation. FA's trick is the *memory peak*: instead of materializing the full $S \times S$ score matrix, it streams attention in tiles and keeps only $O(S)$ memory live. **FLOPs unchanged; memory $O(S^2) \to O(S)$.**
- **Without FA, chunk size is bounded by memory.** An 8 K chunk's attention matrix is $8192^2 \times \text{num\_heads} \times 2 \text{ B} \approx$ tens of GB — instant OOM. Pre-FA, you were forced to small chunks just to keep attention alive.
- **With FA, chunk size is bounded only by your TBT preference.** Memory is no longer the gating factor; the trade-off in the previous subsection (compute time per iteration $a \cdot c + b$) is what limits you now.
- **FA-2 and FA-3 also ship the *kernel that chunked prefill needs***. Specifically, "new Q chunk attending to a previously-cached KV prefix" — varlen Q with paged KV — has been the standard FA path since FA-2. Without that kernel, implementing chunked prefill efficiently is awkward.

So the right framing:

> **FlashAttention is what makes chunked prefill *kernel-feasible* and lets you choose any chunk size you want for the right reasons** — not a free pass to make chunks arbitrarily large.

In practice 512–2048 stays the sweet spot, but FA is why you have that range at all instead of being forced into 256 by memory limits.

### What chunked prefill is NOT

The name suggests "splitting" and "chunks" — easy to confuse with parallelism techniques. Three mix-ups worth nailing down, in increasing order of how badly they mislead:

**It does not split one sequence across multiple GPUs.** Chunked prefill keeps the entire request on one GPU (or one TP group). What it splits is the *work* for that request across multiple scheduler iterations on the *same* GPU. The 32K-token prefill still lives in one device's memory; it just isn't computed in a single forward pass.

**It is not a parallelism technique.** [[parallelism-strategies-deep-dive|Parallelism]] (TP, PP, DP, CP, EP) decides *which GPU* computes *which part* — a **spatial** split. Chunked prefill decides *which iteration* computes *which token* — a **temporal** split. The two are orthogonal and compose: a 1M-token request can run with CP=8 across GPUs *and* chunked prefill on each GPU's local slice, simultaneously.

**It is not the only way to solve "prefill blocks decode."** The orthogonal alternative is [[prefill-decode-disaggregation|PD disaggregation]] — put prefill workers and decode workers on *different physical nodes* so they never share a forward pass. Chunked prefill says *mix them smartly*; PD disaggregation says *don't mix them at all*. The trade-offs:

| | Chunked prefill | PD disaggregation |
|---|---|---|
| Where prefill and decode run | Same GPU(s), different iterations | Different nodes |
| Primary cost | Higher TTFT (prefill split into more passes) | KV cache transfer between nodes |
| Best when | Small / medium deployments; mixed traffic | Large deployments; well-characterized traffic |
| Memory pressure | Single shared pool, contended | Two pools, dedicated per role |
| Throughput scaling | One stage's stall hurts the other | Stages scale independently |

Production systems often use both — chunked prefill *within* a prefill-dedicated node group to smooth its internal load, plus PD disaggregation *across* prefill/decode node groups to eliminate cross-role interference.

## Scheduling Strategies

**FCFS**: Process requests in arrival order. Simple, fair, but no SLA differentiation.

**Preemption**: When GPU memory is exhausted, evict low-priority running requests via:
- **Swap**: Copy KV cache to CPU memory (good for long sequences)
- **Recompute**: Discard KV cache, re-prefill later (good for short sequences)

**vLLM V1 Unified Scheduler**: Represents scheduling decisions as `{request_id: num_tokens}`, unifying chunked prefill, prefix caching, and [[speculative-decoding|speculative decoding]] under a single abstraction.

**SGLang**: Uses RadixAttention (radix-tree prefix cache) with token-level granularity, ~29% faster than vLLM on prefix-heavy workloads (multi-turn chat).

```python
# vLLM key scheduling parameters
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,
    max_num_seqs=256,              # Max concurrent sequences
    max_num_batched_tokens=4096,   # Per-step token budget
    gpu_memory_utilization=0.90,
    swap_space=8,                  # CPU swap space (GB)
    preemption_mode="recompute",   # "recompute" or "swap"
    enable_prefix_caching=True,
)
```

## Memory Management Interaction

[[paged-attention|PagedAttention]] is what makes continuous batching practical:

- KV cache split into fixed-size blocks (typically 16 tokens)
- Blocks allocated on-demand from a free pool -- no pre-allocation needed
- When a request finishes, its blocks are instantly returned to the pool
- New requests can allocate blocks immediately -- no contiguous memory required

This means the scheduler can freely add/remove requests every step without memory fragmentation.

```
Memory lifecycle during continuous batching:

Step T:   Req A → blocks [0,3,7]  |  Req B → blocks [1,5]
          Req C → blocks [2,4,8,9] | Free: [6,10,11]

Step T+1: C finishes → blocks [2,4,8,9] freed
          D enters   → allocates blocks [2,4] from free pool
          A grows    → allocates block [8] for new KV
```

## Performance

Typical throughput improvements over static batching:

| Workload | Static | Continuous | Improvement |
|----------|--------|-----------|-------------|
| Chat (short output) | ~1000 tok/s | ~3000 tok/s | 3x |
| Code gen (medium) | ~800 tok/s | ~2500 tok/s | 3.1x |
| Summarization (low variance) | ~900 tok/s | ~1500 tok/s | 1.7x |
| Mixed (high variance) | ~600 tok/s | ~2800 tok/s | 4.7x |

*(Llama 2 13B, single A100 80GB -- illustrative)*

Latency comparison:

| Metric | Static | Continuous | Continuous + Chunked |
|--------|--------|-----------|---------------------|
| TTFT P50 | 200ms | 100ms | 150ms |
| TTFT P99 | 2000ms | 500ms | 600ms |
| TBT P50 | 30ms | 30ms | 25ms |
| TBT P99 | 300ms | 200ms | **50ms** |

Scheduling overhead: ~1-5% of step time (0.1-1ms), negligible for large models.

## Limitations

1. **TTFT vs TBT trade-off**: Chunked prefill improves TBT at the cost of TTFT. The ultimate solution is [[prefill-decode-disaggregation|PD disaggregation]].
2. **Scheduling complexity**: The scheduler must balance memory constraints, compute budgets, priorities, prefix cache hits, speculative decoding, and preemption -- combinatorial complexity grows with features.
3. **Low-traffic scenarios**: With few concurrent requests, continuous batching offers little advantage. Single-request optimizations like [[speculative-decoding|speculative decoding]] matter more.

## References

- **Orca**: Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models", OSDI 2022. [Paper](https://www.usenix.org/conference/osdi22/presentation/yu)
- **Sarathi-Serve**: Agrawal et al., "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve", OSDI 2024. [arXiv:2403.02310](https://arxiv.org/abs/2403.02310)
- **vLLM**: Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention", SOSP 2023. [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
- **FastServe**: Wu et al., "Fast Distributed Inference Serving for Large Language Models", 2023. [arXiv:2305.05920](https://arxiv.org/abs/2305.05920)

## Related Pages

- [[vllm]] — Primary inference engine implementing continuous batching
- [[sglang]] — Alternative engine with RadixAttention prefix caching
- [[paged-attention]] — Memory management enabling continuous batching
- [[prefill-decode-disaggregation]] — Physical separation to eliminate prefill/decode interference
- [[kv-cache-optimization]] — KV cache optimization techniques
- [[speculative-decoding]] — Complementary single-request optimization
