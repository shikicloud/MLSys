---
title: "Continuous Batching: Dynamic Request Scheduling"
category: llm-inference
tags: [continuous-batching, scheduling, iteration-level, dynamic-batching, throughput]
created: 2026-04-13
updated: 2026-05-20
status: mature
---

# Continuous Batching: Dynamic Request Scheduling

> [!abstract]+ TL;DR
> Batching amortizes the cost of loading model weights to improve GPU utilization, but LLM output lengths vary wildly (a few tokens to thousands), so **static batching** gets dragged down by whichever request finishes first. **Continuous batching** (iteration-level scheduling) dynamically adjusts batch composition at every decode step — inserting new requests the moment old ones complete — eliminating the convoy effect. Introduced by **Orca (OSDI 2022)**, it is now the core scheduling mechanism in [[vllm|vLLM]], [[sglang|SGLang]], and [[tensorrt-llm|TensorRT-LLM]]. Production deployments typically see **2–5× throughput** vs. static batching.

```
Core idea: instead of waiting for the entire batch to finish, check at every
           token-generation step and replace completed requests on the fly.
```

---

## The Problem with Static Batching

### Basic Mechanics

Static batching is the naive approach: collect a batch of requests, start them together, and wait for **all of them** to finish before accepting the next batch.

```
Static batching diagram:

Time ──────────────────────────────────────────────►

Req A: |████████████████|                          (16 tokens generated)
Req B: |████████████████████████████████████████|   (40 tokens generated)
Req C: |████████|                                   (8 tokens generated)
Req D: |████████████████████████|                   (24 tokens generated)
        ↑                       ↑                ↑
     Batch start          C,A finish        B finishes,
                          but still wait    batch can accept
                          for B             new requests

        |◄──────────── End-to-end batch latency ──────►|
```

### Where the Waste Comes From

**Convoy effect**: the latency of the whole batch is dictated by the longest request. Short requests sit in GPU slots without doing useful work after they finish.

```
GPU utilization analysis (static batching):

Step:    1  2  3  4  5  6  7  8  9 10 11 12 ... 40
Req A:   ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ... □   ← done at step 16
Req B:   ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ... ■   ← done at step 40
Req C:   ■  ■  ■  ■  ■  ■  ■  ■  □  □  □  □  ... □   ← done at step 8
Req D:   ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ... □   ← done at step 24

■ = useful compute    □ = GPU idle (wasted)

Useful work:   16 + 40 + 8 + 24 = 88 token-steps
Total budget:  4 × 40 = 160 token-steps
Utilization:   88 / 160 = 55%
Waste:         45%
```

### Quantitative Analysis

For a batch of $N$ requests with output lengths $L_1, L_2, \ldots, L_N$ and maximum length $L_{\max} = \max(L_i)$:

- **Static batching GPU utilization**:

$$
\text{Utilization}_{\text{static}} = \frac{\sum_{i=1}^{N} L_i}{N \times L_{\max}}
$$

In high-variance workloads (e.g. chat with mixed short and long replies), utilization can drop to **20–30%**.

- **Queueing delay**: new requests must wait for the entire current batch to finish before starting, making tail latency catastrophic.

### Dynamic Batching (Transitional Approach)

Some systems use "dynamic batching": collect arrivals within a time window, then form a batch. This only optimizes **batch formation** — once execution starts, you still hit the convoy effect.

```
Three batching strategies compared:

┌─────────────┬───────────────────┬──────────────────┬───────────────────┐
│ Strategy     │ Batch formation   │ Batch execution  │ GPU utilization   │
├─────────────┼───────────────────┼──────────────────┼───────────────────┤
│ Static       │ Pre-fixed size    │ Wait for all     │ Low (20-55%)      │
│ Dynamic      │ Time-window       │ Wait for all     │ Medium (40-65%)   │
│ Continuous   │ Per-step          │ Leave immediately│ High (85-98%)     │
└─────────────┴───────────────────┴──────────────────┴───────────────────┘
```

---

## Continuous (Iteration-Level) Batching

### Orca's Contribution (Yu et al., OSDI 2022)

Orca was the first paper to systematically propose **iteration-level scheduling**, published at OSDI 2022. Its key observation:

> Every step of LLM autoregressive decoding (i.e. every generated token) is an independent scheduling point.
> There's no need to wait for the entire batch — you can decide which requests participate **at every iteration**.

Orca proposed two key mechanisms:

1. **Iteration-Level Scheduling**: re-evaluate batch composition at every token-generation step
2. **Selective Batching**: only batch compatible operations (e.g. batch prefill and decode separately)

On GPT-3 175B, Orca achieved **36.9× throughput** over NVIDIA FasterTransformer.

### How It Works

Core loop of continuous batching:

```
Continuous batching workflow:

Every decode iteration:
  ┌─────────────────────────────────────────────┐
  │ 1. Run one forward pass (all active requests)│
  │ 2. Check which requests produced <EOS>       │
  │ 3. Remove finished requests, return to client│
  │ 4. Pull new requests from queue into slots   │
  │ 5. Run prefill for the new ones              │
  │ 6. Back to step 1                            │
  └─────────────────────────────────────────────┘
```

### Timeline Diagram

```
Continuous batching timeline (max concurrency = 4 slots):

Step:    1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
Slot 0: [A  A  A  A  A  A  A  A][E  E  E  E  E  E  E][H  H  H  H  H...
Slot 1: [B  B  B  B  B  B  B  B  B  B  B  B][F  F  F  F  F  F  F  F...
Slot 2: [C  C  C  C][D  D  D  D  D  D  D  D  D  D][G  G  G  G  G  G...
Slot 3: [·  ·  ·  ·  ·  ·  ·  ·  ·  ·][·  ·  ·  ·  ·  ·  ·  ·  ·  ·

■ letter = active request    · = idle slot
[ ] = request lifetime

Key events:
  Step 4:  C finishes → Slot 2 freed → D enters immediately
  Step 8:  A finishes → Slot 0 freed → E enters immediately
  Step 10: D finishes → queue temporarily empty → Slot 2 idle
  Step 12: B finishes → Slot 1 freed → F enters
  Step 14: New request G arrives → enters Slot 2
  Step 15: E finishes → Slot 0 freed → H enters
```

### Utilization Comparison

```
Same request sequence under both strategies:

Static batching (batch size = 4):
  Batch 1: [A(8), B(12), C(4), D(10)] → wait 12 steps → util = 34/48 = 71%
  Batch 2: [E(7), F(8), G(6), H(5)]   → wait 8 steps  → util = 26/32 = 81%
  Total latency: 20 steps, new requests queued

Continuous batching (max concurrency = 4):
  Every request frees its slot the moment it finishes
  Utilization: (8+12+4+10+7+8+6+5) / (4 × 20) ≈ 75-95%
  Key win: latency drops sharply, new requests don't wait for batch end
```

### Why It Approaches 100% GPU Utilization

Continuous batching pushes GPU utilization high for four reasons:

1. **No convoy effect**: short requests free their resources immediately
2. **Slot refilling**: new requests can enter at any time
3. **Pipeline overlap**: new-request prefill can overlap with other requests' decode
4. **Adaptive load**: batch size dynamically tracks actual load

Under sustained high traffic, GPU slots are almost never idle.

---

## Chunked Prefill

### Problem: Long Prefill Blocks Decode

Continuous batching solves the convoy effect for decode but introduces a new problem: **prefill blocking**.

When a new request enters, its full prompt must first be processed (prefill phase). If the prompt is long (say 32K tokens), this prefill operation will:

1. **Monopolize GPU compute**: the attention compute is $O(n^2)$ in prompt length
2. **Block in-flight decode**: decoding requests must wait for prefill to finish before generating the next token
3. **Inflate TPOT**: time per output token can blow up by 2–30×

```
Prefill-blocking problem:

                   Time ──────────────────────────────►
In-flight decode:  ■ ■ ■ |████████████████████| ■ ■ ■ ■ ■
                         ↑                    ↑
                  New request arrives    Prefill done,
                  (32K prompt)           decode resumes

                  |◄── During this window ──►|
                  |  every decode request    |
                  |  sees inflated TPOT      |
```

### Solution: Chunked Prefill

**Chunked prefill** breaks the long prefill into smaller chunks and **interleaves** them with decode steps.

```
Chunked prefill, the basic idea:

Original prefill (32K tokens, one shot):
  [████████████████████████████████████████████████]
   ↑ one giant prefill op, takes a long time

Chunked prefill (512 tokens per chunk):
  [████][████][████][████] ... [████][████]
    ↑      ↑      ↑      ↑
    decode can be inserted between any two chunks
```

### How It Runs

```
Chunked prefill interleaved with decode:

Step:      1        2        3        4        5        6
        ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
Pfill:  │Chunk1│ │      │ │Chunk2│ │      │ │Chunk3│ │      │
        │512tok│ │      │ │512tok│ │      │ │512tok│ │      │
        └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘
        ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
Decode: │batch │ │batch │ │batch │ │batch │ │batch │ │batch │
        └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘

Effect: decode requests get compute time every step;
        prefill is spread across multiple steps and no longer blocks.
```

### The Sarathi-Serve Approach

Sarathi-Serve (Agrawal et al., 2024) proposed a more refined chunked-prefill scheme:

1. **Unified scheduling**: prefill chunks and decode tokens are packed into the same compute batch
2. **Token budget**: every iteration has a fixed token budget (e.g. 2048) that prefill chunks and decode tokens share
3. **Pipeline-friendly**: chunk size can be tuned to fit pipeline-parallel scheduling

```
Sarathi-Serve token-budget model:

Per-step token budget = 2048

Step 1:  [Prefill chunk: 512 tokens] + [Decode: 200 tokens] = 712  ✓
Step 2:  [Prefill chunk: 512 tokens] + [Decode: 201 tokens] = 713  ✓
Step 3:  [Decode only: 202 tokens]                          = 202  ✓
Step 4:  [New prefill: 1024 tokens]  + [Decode: 203 tokens] = 1227 ✓

A token budget keeps per-step compute bounded —
no single outlier request can spike latency.
```

### TTFT vs TBT Trade-off

Chunked prefill introduces a key trade-off:

```
Chunked prefill TTFT vs TBT trade-off:

┌──────────────┬─────────────────────┬─────────────────────┐
│ Chunk size    │ TTFT (first token)  │ TBT (between tokens)│
├──────────────┼─────────────────────┼─────────────────────┤
│ Very large    │ ✓ Low (fast prefill)│ ✗ High (blocks dec) │
│ (full prompt) │                     │                     │
├──────────────┼─────────────────────┼─────────────────────┤
│ Medium        │ ○ Medium            │ ○ Medium            │
│ (512–2048)    │                     │                     │
├──────────────┼─────────────────────┼─────────────────────┤
│ Very small    │ ✗ High (split-up)   │ ✓ Low (smooth dec)  │
│ (64–128)      │                     │                     │
└──────────────┴─────────────────────┴─────────────────────┘

TTFT = Time To First Token (how long until user sees first output)
TBT  = Time Between Tokens (per-token interval, streaming feel)
```

- **Large chunks**: prefill finishes faster (lower TTFT), but each chunk takes long to run and blocks decode (higher TBT)
- **Small chunks**: decode stays smooth (lower TBT), but prefill needs more steps to complete (higher TTFT)
- **Sweet spot**: typically 512–2048 tokens, depending on model size and GPU compute

vLLM V1 enables chunked prefill by default; the chunk size is controlled via `max_num_batched_tokens`.

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

| Dimension | Chunked prefill | PD disaggregation |
|-----------|-----------------|-------------------|
| Where prefill and decode run | Same GPU(s), different iterations | Different nodes |
| Primary cost | Higher TTFT (prefill split into more passes) | KV cache transfer between nodes |
| Best when | Small/medium deployments, mixed traffic | Large deployments, well-characterized traffic |
| Memory pressure | Single shared pool, contended | Two pools, dedicated per role |
| Throughput scaling | One stage's stall hurts the other | Stages scale independently |

Production systems often use both — chunked prefill *within* a prefill-dedicated node group to smooth its internal load, plus PD disaggregation *across* prefill/decode node groups to eliminate cross-role interference.

---

## Scheduling Strategies

### FCFS (First-Come-First-Served)

The most basic scheduling policy: process requests in arrival order.

```python
class FCFSScheduler:
    """First-come-first-served scheduler"""
    
    def __init__(self, max_batch_size: int, max_num_tokens: int):
        self.max_batch_size = max_batch_size
        self.max_num_tokens = max_num_tokens
        self.waiting_queue: list[Request] = []     # Awaiting prefill
        self.running_batch: list[Request] = []     # Currently decoding
    
    def schedule(self) -> ScheduleOutput:
        """Called once per iteration step"""
        # 1. Remove finished requests
        self.running_batch = [
            req for req in self.running_batch 
            if not req.is_finished()
        ]
        
        # 2. Compute the current token budget
        num_decode_tokens = len(self.running_batch)  # 1 token per decode req
        remaining_budget = self.max_num_tokens - num_decode_tokens
        remaining_slots = self.max_batch_size - len(self.running_batch)
        
        # 3. Fill in new requests in FCFS order
        new_prefills = []
        while self.waiting_queue and remaining_slots > 0:
            request = self.waiting_queue[0]
            prefill_tokens = request.get_prompt_length()
            
            if prefill_tokens <= remaining_budget:
                self.waiting_queue.pop(0)
                new_prefills.append(request)
                self.running_batch.append(request)
                remaining_budget -= prefill_tokens
                remaining_slots -= 1
            else:
                break  # Not enough budget — wait for next step
        
        return ScheduleOutput(
            decode_requests=self.running_batch,
            prefill_requests=new_prefills
        )
```

### Preemption and Priority Scheduling

When GPU memory runs out, the scheduler must **preempt** some running requests:

```
Preemption strategies:

┌──────────────┬──────────────────────────┬──────────────────────────┐
│ Strategy      │ Swap (to CPU memory)     │ Recompute (drop & redo)  │
├──────────────┼──────────────────────────┼──────────────────────────┤
│ Operation     │ Copy KV cache to CPU     │ Discard KV cache         │
│ Resume        │ Copy back from CPU       │ Re-execute prefill       │
│ Memory        │ Needs CPU memory         │ No extra memory          │
│ Best for      │ Long seqs (recompute     │ Short seqs (recompute    │
│               │ is expensive)            │ is cheap)                │
│ Latency       │ PCIe bandwidth-bound     │ GPU-compute-bound        │
└──────────────┴──────────────────────────┴──────────────────────────┘
```

Priority scheduling lets high-priority requests preempt lower-priority ones:

```python
class PriorityScheduler:
    """Priority-based scheduler"""
    
    def __init__(self, max_batch_size: int, max_num_tokens: int):
        self.max_batch_size = max_batch_size
        self.max_num_tokens = max_num_tokens
        self.waiting_queue: list[Request] = []
        self.running_batch: list[Request] = []
    
    def add_request(self, request: Request):
        """Insert into waiting queue by priority (highest first)"""
        import bisect
        bisect.insort(self.waiting_queue, request, 
                      key=lambda r: -r.priority)
    
    def schedule(self) -> ScheduleOutput:
        # 1. Remove finished requests
        self.running_batch = [
            req for req in self.running_batch 
            if not req.is_finished()
        ]
        
        # 2. Preempt if a higher-priority request is waiting
        preempted = []
        while (self.waiting_queue 
               and len(self.running_batch) >= self.max_batch_size):
            # Highest-priority waiting vs. lowest-priority running
            waiting_top = self.waiting_queue[0]
            running_lowest = min(self.running_batch, 
                                key=lambda r: r.priority)
            
            if waiting_top.priority > running_lowest.priority:
                # Preempt the lower-priority running request
                self.running_batch.remove(running_lowest)
                preempted.append(running_lowest)
            else:
                break
        
        # 3. Fill in new requests (same as FCFS)
        # ... (omitted, identical logic)
        
        return ScheduleOutput(
            decode_requests=self.running_batch,
            prefill_requests=new_prefills,
            preempted_requests=preempted
        )
```

### vLLM's Scheduling Implementation

vLLM V1 uses a **unified scheduler**. Its key design:

```
vLLM V1 unified scheduler:

Input: current running requests + waiting queue + memory state

Output: {request_id: num_tokens} dict
        ↓
  This simple mapping unifies all of the following:
  - Plain decode:     {req_1: 1, req_2: 1, req_3: 1}
  - Chunked prefill:  {req_1: 1, req_2: 1, new_req: 512}
  - Speculative dec.: {req_1: 5, req_2: 5}  (verify multiple tokens/step)
  - Prefix cache hit: {new_req: 100}  (only un-cached portion)
```

vLLM scheduler key configuration parameters:

```python
# vLLM serving config example
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    
    # === Scheduling parameters ===
    max_num_seqs=256,              # Max concurrent sequences (upper batch size)
    max_num_batched_tokens=2048,   # Per-step token budget
                                   # (controls chunked-prefill chunk size)
    
    # === Memory parameters ===
    gpu_memory_utilization=0.90,   # GPU memory utilization ceiling
    swap_space=4,                  # CPU swap space (GB)
    
    # === Preemption ===
    preemption_mode="recompute",   # "recompute" or "swap"
    
    # === Prefix cache ===
    enable_prefix_caching=True,    # On by default in V1
)
```

### SGLang's Scheduling Implementation

[[sglang|SGLang]] adopts different scheduling optimizations:

1. **RadixAttention**: a radix-tree-based prefix cache, supporting token-level cache reuse granularity
2. **Continuous batching**: similar to vLLM but optimized further for prefix-heavy workloads (multi-turn chat)
3. **Zero-overhead scheduling**: scheduling decisions happen on the Python side without blocking GPU compute

```
SGLang vs vLLM scheduling comparison:

┌─────────────┬──────────────────┬──────────────────┐
│ Feature      │ vLLM V1          │ SGLang           │
├─────────────┼──────────────────┼──────────────────┤
│ Prefix cache │ Hash LRU         │ RadixAttention   │
│ Cache gran.  │ Block (16 tok)   │ Token-level      │
│ Sched. repr. │ {id: num_tokens} │ Tree-based       │
│ Chunked pf.  │ ✓                │ ✓                │
│ Multi-turn   │ Good             │ Better (+29%)    │
└─────────────┴──────────────────┴──────────────────┘
```

### SLA Management and Request Priority

In production, different request types come with different SLA (Service Level Agreement) requirements:

```
SLA-driven scheduling example:

┌──────────────┬───────────────┬──────────────┬───────────────┐
│ Request type  │ TTFT SLA      │ TBT SLA      │ Priority      │
├──────────────┼───────────────┼──────────────┼───────────────┤
│ Realtime chat │ < 200ms       │ < 50ms       │ High          │
│ Streaming gen │ < 500ms       │ < 100ms      │ Medium        │
│ Batch jobs    │ < 5s          │ No req.      │ Low           │
│ Background    │ No req.       │ No req.      │ Lowest        │
└──────────────┴───────────────┴──────────────┴───────────────┘

Scheduler adjusts priority dynamically by SLA:
- Requests near their SLA deadline get boosted automatically
- Requests already past SLA may be demoted (don't waste resources)
```

---

## Memory Management ↔ Scheduling Interaction

### How PagedAttention Enables Continuous Batching

Continuous batching depends heavily on flexible memory management. [[paged-attention|PagedAttention]] enables continuous batching through:

```
PagedAttention and continuous batching cooperating:

Physical memory (GPU HBM):
┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
│ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │ 9 │10 │11 │  Physical blocks
└───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘

Step T (requests A, B, C running):
  Req A: logical [0,1,2]   → physical [0,3,7]
  Req B: logical [0,1]     → physical [1,5]
  Req C: logical [0,1,2,3] → physical [2,4,8,9]
  Free blocks: [6, 10, 11]

Step T+1 (C finished, D joins):
  Req A: logical [0,1,2,3] → physical [0,3,7,2]  ← reuses C's block 2
  Req B: logical [0,1,2]   → physical [1,5,4]     ← reuses C's block 4
  Req D: logical [0,1]     → physical [8,9]        ← reuses C's blocks 8,9
  Free blocks: [6, 10, 11]

Key point: no contiguous memory required, no pre-allocation,
           new requests get blocks instantly.
```

### Block-Level Memory Allocation

The scheduler interacts with the memory manager at every iteration:

```python
class SchedulerWithMemory:
    """Scheduler ↔ memory manager interaction"""
    
    def __init__(self, block_manager, max_batch_size, max_num_tokens):
        self.block_manager = block_manager
        self.max_batch_size = max_batch_size
        self.max_num_tokens = max_num_tokens
        self.waiting_queue = []
        self.running_batch = []
    
    def schedule(self) -> ScheduleOutput:
        # 1. Free blocks held by finished requests
        finished = [r for r in self.running_batch if r.is_finished()]
        for req in finished:
            self.block_manager.free(req.request_id)
        self.running_batch = [
            r for r in self.running_batch if not r.is_finished()
        ]
        
        # 2. Allocate new blocks for running requests (if their last is full)
        for req in self.running_batch:
            if req.needs_new_block():
                if self.block_manager.has_free_blocks():
                    self.block_manager.allocate(req.request_id, num_blocks=1)
                else:
                    # Out of memory — need to preempt
                    self._preempt_lowest_priority()
        
        # 3. Try to schedule new requests
        new_prefills = []
        while self.waiting_queue:
            request = self.waiting_queue[0]
            # Compute the new request's block requirement
            needed_blocks = self._compute_needed_blocks(request)
            
            if (self.block_manager.get_free_blocks() >= needed_blocks
                    and len(self.running_batch) < self.max_batch_size):
                self.waiting_queue.pop(0)
                self.block_manager.allocate(
                    request.request_id, num_blocks=needed_blocks)
                self.running_batch.append(request)
                new_prefills.append(request)
            else:
                break
        
        return ScheduleOutput(
            decode_requests=self.running_batch,
            prefill_requests=new_prefills,
        )
    
    def _preempt_lowest_priority(self):
        """Preempt the lowest-priority request to free memory"""
        victim = min(self.running_batch, key=lambda r: r.priority)
        if self.preemption_mode == "swap":
            # Swap KV cache to CPU memory
            self.block_manager.swap_out(victim.request_id)
        else:
            # Discard KV cache, mark for later recompute
            self.block_manager.free(victim.request_id)
            victim.mark_for_recompute()
        self.running_batch.remove(victim)
        self.waiting_queue.insert(0, victim)  # Push back to front of queue
```

### Preemption Policy: Swap vs Recompute

```
Swap vs recompute decision tree:

                  Need to preempt?
                       │
                   ┌───┴───┐
                   ▼       ▼
              seq length > threshold?
              │              │
           yes │              │ no
              ▼              ▼
            Swap           Recompute
    (save to CPU mem)    (drop, recompute later)

  Considerations:
  ┌──────────────────┬──────────────────────┐
  │ Choose Swap       │ Choose Recompute     │
  ├──────────────────┼──────────────────────┤
  │ Sequence is long  │ Sequence is short    │
  │ PCIe BW plenty    │ PCIe BW bottleneck   │
  │ CPU mem plenty    │ CPU mem limited      │
  │ Recompute costly  │ Recompute cheap      │
  │ Long-context work │ Short-chat scenarios │
  └──────────────────┴──────────────────────┘
```

vLLM's default: try swap first when GPU memory is tight; fall back to recompute if CPU memory is also insufficient. Configurable via `preemption_mode`.

---

## Code Examples

### Full continuous-batching scheduler pseudocode

```python
"""
Full continuous-batching scheduler implementation (simplified).
Shows the core scheduling logic; omits the actual model forward pass.
"""

from dataclasses import dataclass, field
from enum import Enum
from collections import deque
from typing import Optional
import time


class RequestState(Enum):
    WAITING = "waiting"          # Awaiting prefill
    RUNNING_PREFILL = "prefill"  # Prefilling
    RUNNING_DECODE = "decode"    # Decoding
    FINISHED = "finished"        # Done


@dataclass
class Request:
    request_id: str
    prompt_tokens: list[int]
    max_output_tokens: int
    arrival_time: float
    priority: int = 0
    
    # Runtime state
    state: RequestState = RequestState.WAITING
    output_tokens: list[int] = field(default_factory=list)
    prefill_progress: int = 0    # Number of prompt tokens processed
    
    def is_prefill_complete(self) -> bool:
        return self.prefill_progress >= len(self.prompt_tokens)
    
    def is_finished(self) -> bool:
        """Check whether EOS was generated or max length reached"""
        if not self.output_tokens:
            return False
        EOS_TOKEN = 2
        return (self.output_tokens[-1] == EOS_TOKEN 
                or len(self.output_tokens) >= self.max_output_tokens)
    
    def get_remaining_prefill(self) -> int:
        return len(self.prompt_tokens) - self.prefill_progress


@dataclass
class ScheduleOutput:
    """Scheduler output per step"""
    scheduled_requests: dict[str, int]  # {request_id: num_tokens}
    preempted: list[str]                # Preempted request_ids
    finished: list[str]                 # Finished request_ids


class ContinuousBatchingScheduler:
    """Continuous-batching scheduler (with chunked prefill support)"""
    
    def __init__(
        self,
        max_batch_size: int = 256,
        max_num_tokens: int = 2048,
        chunk_size: int = 512,
    ):
        self.max_batch_size = max_batch_size
        self.max_num_tokens = max_num_tokens
        self.chunk_size = chunk_size
        
        self.waiting: deque[Request] = deque()
        self.running: dict[str, Request] = {}
    
    def add_request(self, request: Request):
        """Add a new request to the waiting queue"""
        request.state = RequestState.WAITING
        self.waiting.append(request)
    
    def schedule(self) -> ScheduleOutput:
        """Core scheduling logic — called once per iteration step"""
        scheduled: dict[str, int] = {}
        finished_ids: list[str] = []
        
        # ---- Phase 1: remove finished requests ----
        for req_id in list(self.running.keys()):
            req = self.running[req_id]
            if req.is_finished():
                req.state = RequestState.FINISHED
                finished_ids.append(req_id)
                del self.running[req_id]
        
        # ---- Phase 2: assign tokens to running decode requests ----
        token_budget = self.max_num_tokens
        
        for req_id, req in self.running.items():
            if req.state == RequestState.RUNNING_DECODE:
                scheduled[req_id] = 1   # Decode: 1 token per step
                token_budget -= 1
            elif req.state == RequestState.RUNNING_PREFILL:
                # Continue chunked prefill
                remaining = req.get_remaining_prefill()
                chunk = min(remaining, self.chunk_size, token_budget)
                if chunk > 0:
                    scheduled[req_id] = chunk
                    token_budget -= chunk
                    req.prefill_progress += chunk
                    if req.is_prefill_complete():
                        req.state = RequestState.RUNNING_DECODE
        
        # ---- Phase 3: schedule new requests from the waiting queue ----
        while (self.waiting 
               and len(self.running) < self.max_batch_size 
               and token_budget > 0):
            
            req = self.waiting[0]
            prompt_len = len(req.prompt_tokens)
            
            # Compute first-chunk size
            first_chunk = min(prompt_len, self.chunk_size, token_budget)
            
            if first_chunk <= 0:
                break
            
            # Pop from waiting queue
            self.waiting.popleft()
            req.state = RequestState.RUNNING_PREFILL
            req.prefill_progress = first_chunk
            
            self.running[req.request_id] = req
            scheduled[req.request_id] = first_chunk
            token_budget -= first_chunk
            
            # If the prompt is short, prefill completes in one step
            if req.is_prefill_complete():
                req.state = RequestState.RUNNING_DECODE
        
        return ScheduleOutput(
            scheduled_requests=scheduled,
            preempted=[],
            finished=finished_ids,
        )


# ---- Usage example ----
def main():
    scheduler = ContinuousBatchingScheduler(
        max_batch_size=4,
        max_num_tokens=2048,
        chunk_size=512,
    )
    
    # Simulate adding requests
    requests = [
        Request("req_0", prompt_tokens=list(range(100)),
                max_output_tokens=50, arrival_time=time.time()),
        Request("req_1", prompt_tokens=list(range(2000)),
                max_output_tokens=100, arrival_time=time.time()),
        Request("req_2", prompt_tokens=list(range(50)),
                max_output_tokens=20, arrival_time=time.time()),
    ]
    
    for req in requests:
        scheduler.add_request(req)
    
    # Simulate the scheduling loop
    for step in range(10):
        output = scheduler.schedule()
        print(f"Step {step}: {output.scheduled_requests}")
        
        # Simulate token generation (in real code this is a model forward pass)
        for req_id in output.scheduled_requests:
            if req_id in scheduler.running:
                req = scheduler.running[req_id]
                if req.state == RequestState.RUNNING_DECODE:
                    req.output_tokens.append(42)  # Simulated token

if __name__ == "__main__":
    main()
```

### vLLM Serving Configuration Example

```python
"""vLLM server config — tuning scheduling parameters"""

# Option 1: Python API
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,
    
    # Scheduling
    max_num_seqs=256,                # Max concurrent requests
    max_num_batched_tokens=4096,     # Per-step token budget
    
    # Memory
    gpu_memory_utilization=0.90,     # 90% of GPU memory for KV cache
    swap_space=8,                    # 8 GB CPU swap space
    
    # Prefix caching
    enable_prefix_caching=True,      # Auto-cache common prefixes
)
```

```bash
# Option 2: launching the vLLM server from CLI
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --max-num-seqs 256 \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.90 \
    --swap-space 8 \
    --enable-prefix-caching \
    --preemption-mode recompute
```

### Batching Parameter Tuning Guide

```
Parameter-tuning decision tree:

                  What's your scenario?
                         │
            ┌────────────┼────────────┐
            ▼            ▼            ▼
       Realtime      Streaming      Offline
       chat          generation     batch
            │            │            │
            ▼            ▼            ▼
    max_num_seqs:   max_num_seqs:   max_num_seqs:
    32-64           128-256         512-1024
            │            │            │
            ▼            ▼            ▼
    max_num_batched  max_num_batched  max_num_batched
    _tokens: 1024   _tokens: 2048   _tokens: 8192
            │            │            │
            ▼            ▼            ▼
    Optimize for:    Optimize for:    Optimize for:
    low TTFT + TBT   balanced TTFT/TBT max throughput
```

Key tuning principles:

| Parameter | Increase effect | Decrease effect |
|-----------|----------------|-----------------|
| `max_num_seqs` | Throughput up, latency may rise | Latency down, throughput drops |
| `max_num_batched_tokens` | Faster prefill (lower TTFT), but per-step time up | More stable TBT, but TTFT may rise |
| `gpu_memory_utilization` | More KV cache → larger batches | Safer, lower OOM risk |
| `swap_space` | Fewer preemption drops | Smaller CPU footprint |

---

## Performance Analysis

### Throughput Improvement

The throughput gain of continuous batching over static batching depends on several factors:

```
Throughput-gain factor analysis:

┌─────────────────┬─────────────┬──────────────────────────────┐
│ Factor           │ Impact      │ Explanation                  │
├─────────────────┼─────────────┼──────────────────────────────┤
│ Output-length    │ High        │ Higher variance → more waste │
│ variance         │             │ in static batching           │
│ Arrival rate     │ High        │ Empty slots filled faster    │
│                  │             │ under high traffic           │
│ Batch size       │ Medium      │ Larger batches → worse       │
│                  │             │ convoy effect                │
│ Model size       │ Medium      │ Bigger model → longer step → │
│                  │             │ more waste from idle slots   │
│ Prompt-length    │ Medium      │ Affects chunked-prefill      │
│ variance         │             │ effectiveness                │
└─────────────────┴─────────────┴──────────────────────────────┘
```

Typical numbers:

| Scenario | Static | Continuous | Improvement |
|----------|--------|-----------|------------|
| Chat (short output) | ~1000 tok/s | ~3000 tok/s | 3× |
| Code gen (medium) | ~800 tok/s | ~2500 tok/s | 3.1× |
| Summarization (long output, low variance) | ~900 tok/s | ~1500 tok/s | 1.7× |
| Mixed workload (high variance) | ~600 tok/s | ~2800 tok/s | 4.7× |

> Note: numbers based on Llama 2 13B on a single A100 80GB. Illustrative; real-world gains are workload-dependent.

### Latency Analysis

```
Latency comparison (P50 / P99):

Static batching:
  TTFT:  200ms / 2000ms    ← P99 very high (queueing)
  TBT:   30ms  / 300ms     ← P99 blocked by long prefills

Continuous batching (no chunked prefill):
  TTFT:  100ms / 500ms     ← significantly improved
  TBT:   30ms  / 200ms     ← still subject to prefill blocking

Continuous batching + chunked prefill:
  TTFT:  150ms / 600ms     ← slight rise (prefill split up)
  TBT:   25ms  / 50ms      ← dramatically improved (no blocking)
```

### Scheduling Overhead

Continuous batching runs its scheduler at every token-generation step, so scheduling overhead matters:

- **Python scheduling overhead**: typically 0.1–1 ms/step (vLLM V1 mitigates via EngineCore separation)
- **Dynamic batch reshaping**: input-tensor rewrite costs roughly 0.05–0.5 ms
- **Memory management**: block alloc/free roughly 0.01–0.1 ms

Total scheduling overhead is usually **1–5%** of per-step time — negligible on large models (per-step time > 30 ms).

---

## Limitations and Trade-offs

### The Fundamental TTFT vs TBT Trade-off

Continuous batching (especially with chunked prefill) forces a trade-off between two core metrics:

1. **TTFT (Time To First Token)**: time the user waits for the first output token
   - Driven by prefill speed
   - Chunked prefill raises TTFT (prefill is spread out)
   
2. **TBT (Time Between Tokens)**: interval between consecutive tokens
   - Driven by decode interference
   - Chunked prefill lowers TBT (no more long-prefill blocking)

> The ultimate resolution is [[prefill-decode-disaggregation|PD disaggregation]]: physically split the two phases onto separate GPU pools to eliminate interference entirely.

### Scheduler Complexity

The continuous-batching scheduler must make decisions every token-step, and as features pile on, its complexity grows:

```
Things the scheduler must juggle:

1. Memory constraints:   currently available KV-cache blocks
2. Compute constraints:  per-step token budget
3. Concurrency limits:   max batch size
4. Priority:             SLA requirements and request priorities
5. Prefix cache:         cache-hit requests should be prioritized (less compute)
6. Speculative decoding: verify-step token counts vary
7. Preemption decisions: when, who, swap or recompute?
8. Fairness:             prevent low-priority starvation

As features stack, the combinatorial explosion makes the scheduler
one of the most complex components in an inference engine.
```

### Limitations at Small Scale

When arrival rate is low (e.g. single-user scenarios), continuous batching offers little advantage:

- Batches may contain only 1–2 requests
- Dynamic-scheduling overhead becomes a non-trivial fraction of per-step compute
- Single-request optimizations like [[speculative-decoding|speculative decoding]] matter more here

---

## References

- **Orca**: Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models", OSDI 2022. [Paper](https://www.usenix.org/conference/osdi22/presentation/yu)
  - First systematic proposal of iteration-level scheduling and selective batching

- **Sarathi-Serve**: Agrawal et al., "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve", OSDI 2024. [arXiv:2403.02310](https://arxiv.org/abs/2403.02310)
  - Chunked prefill and hybrid batching

- **vLLM**: Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention", SOSP 2023. [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
  - PagedAttention makes continuous batching's memory management practical

- **FastServe**: Wu et al., "Fast Distributed Inference Serving for Large Language Models", 2023. [arXiv:2305.05920](https://arxiv.org/abs/2305.05920)
  - Preemptive scheduling and job-completion-time optimization

---

## Related Pages

- [[vllm]] — Primary inference engine implementing continuous batching
- [[sglang]] — Alternative high-performance engine with RadixAttention for prefix caching
- [[paged-attention]] — Memory management that enables continuous batching
- [[prefill-decode-disaggregation]] — Physical separation to eliminate prefill/decode interference
- [[kv-cache-optimization]] — KV cache optimization techniques
- [[speculative-decoding]] — Complementary single-request optimization
