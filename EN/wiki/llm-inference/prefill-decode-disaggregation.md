---
title: "Prefill-Decode Disaggregation"
category: llm-inference
tags: [prefill-decode, disaggregation, splitwise, distserve, mooncake, kv-transfer]
created: 2026-04-13
updated: 2026-05-20
status: mature
---

# Prefill-Decode Disaggregation

> [!abstract]+ TL;DR
> LLM inference has two phases with fundamentally different compute profiles: **prefill** (compute-bound, processes full prompt) vs. **decode** (memory-bandwidth-bound, generates tokens one at a time). Colocated on the same GPU pool, long prefills block concurrent decode requests, inflating TPOT by **2–30×**. PD disaggregation physically separates the two phases onto different GPU pools, enabling independent optimization and scaling. By 2025–2026 it became the default production architecture — **NVIDIA Dynamo, llm-d, [[vllm|vLLM]], [[sglang|SGLang]]** all support it natively. Key systems: **DistServe** (OSDI 2024), **Splitwise** (ISCA 2024), **Mooncake** (FAST 2025).

```
Disaggregation, the core idea:

  Colocated (traditional):           Disaggregated:
  ┌─────────────────┐                ┌──────────┐    ┌──────────┐
  │  GPU Pool        │                │ Prefill  │    │ Decode   │
  │  ┌────┐ ┌────┐  │                │ Pool     │    │ Pool     │
  │  │P+D │ │P+D │  │                │ ┌────┐   │    │ ┌────┐   │
  │  └────┘ └────┘  │                │ │ P  │   │    │ │ D  │   │
  │  ┌────┐ ┌────┐  │       →        │ └────┘   │    │ └────┘   │
  │  │P+D │ │P+D │  │                │ ┌────┐   │ KV │ ┌────┐   │
  │  └────┘ └────┘  │                │ │ P  │──────►│ D  │   │
  │  interference!   │                │ └────┘   │    │ └────┘   │
  └─────────────────┘                │ High FLOP│    │ High BW  │
                                     └──────────┘    └──────────┘
```

By 2025–2026, disaggregated architecture has become the **default deployment pattern** for large-scale LLM serving, with native support from NVIDIA Dynamo, llm-d, [[vllm|vLLM]], [[sglang|SGLang]], and other major frameworks.

---

## Prefill vs Decode Profiles

### Compute Profile Differences

Prefill and decode have radically different compute profiles:

```
Prefill phase:
  - Input:   the entire prompt (hundreds to tens of thousands of tokens)
  - Op:      compute attention over all tokens in one shot
  - Profile: large matmul, high arithmetic intensity
  - Bottleneck: GPU compute (FLOPS)
  - Analogy: "reading a whole book" — heavy compute, done once

Decode phase:
  - Input:   a single token (the one just generated)
  - Op:      attention against all cached KV
  - Profile: small matmul, low arithmetic intensity, lots of memory reads
  - Bottleneck: GPU memory bandwidth (GB/s)
  - Analogy: "writing word by word" — little compute each step,
             but reread memory every time
```

### Quantitative Comparison

```
Prefill vs decode quantitative comparison (Llama 3.1 70B, A100 80GB):

┌────────────────────┬──────────────────┬──────────────────┐
│ Metric              │ Prefill          │ Decode           │
├────────────────────┼──────────────────┼──────────────────┤
│ Input token count   │ N (hundreds–10k+)│ 1                │
│ Compute (FLOPS)     │ ~2 × P × N       │ ~2 × P           │
│ Arithmetic          │ High (~100+      │ Low (~1 ops/B)   │
│ intensity           │  ops/B)          │                  │
│ GPU compute util.   │ 60-80%           │ 1-5%             │
│ Memory BW util.     │ 20-40%           │ 80-95%           │
│ Optimal batch size  │ 1-4              │ 64-512           │
│ Latency metric      │ TTFT             │ TPOT / TBT       │
│ Parallelism fit     │ High (TP works)  │ Low (DP wins)    │
│ Time share (typ.)   │ 10-30%           │ 70-90%           │
└────────────────────┴──────────────────┴──────────────────┘

P = number of model parameters, N = input token count
```

### Roofline-Model Analysis

```
Roofline model: prefill vs decode

Performance
(TFLOPS)  │
          │              ╱ ← Compute ceiling (A100: 312 TFLOPS FP16)
    312 ──│─ ─ ─ ─ ─ ─╱─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
          │          ╱
          │        ╱        ★ Prefill (arithmetic intensity ~100)
    200 ──│      ╱          → close to compute ceiling
          │    ╱
          │  ╱
     50 ──│╱
          │  ▲ Decode (arithmetic intensity ~1)
          │  → far below ceiling
          │  → memory-bandwidth bound
          │
          └──────────────────────────────────────
          1    10    100    1000
              Arithmetic intensity (FLOPS/Byte)
                  ↑                ↑
                Decode           Prefill
            (memory-bound)   (compute-bound)
```

Core insight: **a single GPU cannot be optimally configured for two such different workloads simultaneously**. This is the fundamental motivation for disaggregation.

---

## Why Disaggregate

### Problem 1: Prefill Interferes with Decode Latency

This is the primary motivation. When prefill and decode are colocated:

```
Head-of-line blocking problem:

Scenario: 3 decode requests running, 1 new request arrives needing prefill

Colocated execution:
Step:     1    2    3    4    5    6    7    8    9   10
        ┌────────────────────────────┐
New P:  │      prefill (long prompt)  │                    ← occupies GPU
        └────────────────────────────┘
Decode A: ■    ·    ·    ·    ·    ·    ·    ■    ■    ■    ← TPOT inflated!
Decode B: ■    ·    ·    ·    ·    ·    ·    ■    ■    ■    ← TPOT inflated!
Decode C: ■    ·    ·    ·    ·    ·    ·    ■    ■    ■    ← TPOT inflated!

■ = normal decode step    · = blocked by prefill

Normal TPOT:   ~30 ms
Blocked TPOT:  ~210 ms (7× inflation)
```

Chunked prefill mitigates but does not eliminate this:

```
Chunked prefill (partial mitigation):
Step:     1    2    3    4    5    6    7    8    9   10
        ┌──┐      ┌──┐      ┌──┐      ┌──┐
New P:  │C1│      │C2│      │C3│      │C4│
        └──┘      └──┘      └──┘      └──┘
Decode A:      ■         ■         ■         ■    ■    ■
Decode B:      ■         ■         ■         ■    ■    ■
Decode C:      ■         ■         ■         ■    ■    ■

Mitigated, but interference remains:
- Prefill chunks still consume compute time → fewer decode steps
- Small chunks → higher TTFT
- The trade-off persists
```

### Problem 2: Different Optimal Batch Sizes

```
Different optimal batch sizes:

Prefill:
  - A single request can already utilize most of the GPU compute
  - Optimal batch size: 1-4
  - Larger batches → exceed GPU memory or compute capacity

Decode:
  - A single request uses ~1% of GPU compute
  - Optimal batch size: 64-512
  - Large batch needed to amortize weight-loading cost

The colocation conflict:
  Optimize for prefill (small batch) → decode utilization tanks
  Optimize for decode (large batch) → prefill OOMs
  The optimal configs contradict each other!
```

### Problem 3: Different Parallelism Strategies

```
Different optimal parallelism strategies:

Prefill (compute-heavy) → tensor parallelism (TP) wins:
  - Large matmuls split cleanly across GPUs
  - Communication overhead amortized by heavy compute
  - Typical config: TP=4 or TP=8

Decode (memory-heavy) → data parallelism (DP) wins:
  - Each request has little compute, TP's communication is relatively too costly
  - Better to run more independent replicas with larger batches
  - Typical config: DP=N, TP=1 or TP=2

Disaggregation lets each pool pick its own strategy:

  ┌─────────────────┐        ┌─────────────────┐
  │   Prefill Pool   │        │   Decode Pool    │
  │                  │        │                  │
  │  TP=4, few inst.│  ──►   │  DP=8, many inst.│
  │  high compute   │  KV    │  high bandwidth  │
  │  util            │        │  util            │
  └─────────────────┘        └─────────────────┘
```

### Problem 4: Independent Scaling

```
Scaling under varying loads:

Scenario 1: many short prompts, long outputs (e.g. chat)
  → low prefill load, high decode load
  → need more decode instances

Scenario 2: many long prompts, short outputs (e.g. summarization)
  → high prefill load, low decode load
  → need more prefill instances

Colocated: scale only as a whole, no targeted optimization
Disaggregated: each pool scales independently, optimal resource usage
```

---

## Disaggregated Architecture Designs

### General Architecture

```
General PD-disaggregation architecture:

  Client request
       │
       ▼
┌──────────────┐
│  Router/Sched│  ← Global request routing
│  (Router)     │
└──────┬───────┘
       │
  ┌────┴────┐
  ▼         ▼
┌────────┐  ┌────────────┐
│Prefill │  │  Decode     │
│Pool    │  │  Pool       │
│        │  │             │
│ GPU 0  │  │  GPU A      │
│ GPU 1  │  │  GPU B      │
│ GPU 2  │  │  GPU C      │
│ ...    │  │  GPU D      │
│        │  │  ...        │
└───┬────┘  └─────┬──────┘
    │              ▲
    │   KV Cache   │
    └──────────────┘
       Transfer
```

Request lifecycle:

```
Disaggregated request flow:

1. Client sends request (prompt + sampling params)
      │
      ▼
2. Router receives request, sends to Prefill Pool
      │
      ▼
3. Prefill instance processes prompt
   - Runs full forward pass
   - Produces KV cache + the first output token
      │
      ▼
4. KV cache transferred to Decode Pool
   - Via RDMA / NVLink / PCIe
   - Critical bottleneck!
      │
      ▼
5. Decode instance continues autoregressive generation
   - Receives KV cache
   - Generates token by token until EOS
      │
      ▼
6. Result returned to client
   - Can be streamed
```

### Splitwise (Microsoft, ISCA 2024)

Splitwise is the first system to systematically propose PD disaggregation.

Core ideas:
- Use **heterogeneous hardware**: compute-optimized GPUs for prefill, memory-optimized GPUs for decode
- Same-machine GPUs can transfer KV cache via high-speed NVLink
- A **mixed** mode: when one GPU type is idle, it can temporarily take on the other type's work

```
Splitwise architecture:

┌────────────────────────────────────────────┐
│                Same machine                 │
│                                            │
│  ┌──────────────┐    NVLink    ┌──────────────┐
│  │ GPU 0 (H100) │ ◄─────────► │ GPU 1 (H100) │
│  │  Prefill     │   900 GB/s  │  Decode       │
│  │  (high FLOPS)│             │  (high BW)    │
│  └──────────────┘             └──────────────┘
│                                            │
└────────────────────────────────────────────┘

Results:
  - Throughput up 1.4×
  - Cost down 20%
  - Key insight: intra-machine NVLink transfer is essentially free
```

Limitations:
- Considers only intra-machine disaggregation (NVLink); cross-machine left out
- The heterogeneous-hardware assumption doesn't always hold in production

### DistServe (OSDI 2024)

DistServe extends disaggregation to the cluster level, supporting cross-machine PD disaggregation.

Core innovations:
- **Pull-based scheduling**: decode instances actively pull pre-built KV caches from prefill instances
- **Goodput optimization**: maximize effective throughput under SLO constraints
- Supports different parallelism strategies for prefill and decode

```
DistServe architecture:

  ┌─────────────────────────────────────────┐
  │            Global scheduler              │
  │  (Goodput optimization + SLO-aware)      │
  └────────────┬──────────────┬─────────────┘
               │              │
         ┌─────┴─────┐  ┌────┴──────┐
         │ Prefill    │  │ Decode    │
         │ Instance 0 │  │ Instance 0│ ← pulls KV
         │ (TP=4)     │  │ (TP=1)    │
         ├────────────┤  ├──────────┤
         │ Prefill    │  │ Decode    │
         │ Instance 1 │  │ Instance 1│ ← pulls KV
         │ (TP=4)     │  │ (TP=1)    │
         └────────────┘  ├──────────┤
                         │ Decode    │
                         │ Instance 2│
                         │ (TP=1)    │
                         ├──────────┤
                         │ ...      │
                         └──────────┘

  Prefill: few instances with high TP
  Decode:  many instances with low TP
```

Results:
- Vs. colocated systems under the same SLO, goodput up **7.4×**
- Under stricter SLOs, up **12.6×**
- Validates the feasibility and value of cross-machine disaggregation

### Mooncake (Moonshot AI, FAST 2025 Best Paper)

Mooncake is Moonshot AI's production system, serving Kimi and processing **100B+ tokens per day**.

```
Mooncake architecture:

                    ┌─────────────────────┐
                    │    Conductor         │
                    │  (Global coordinator)│
                    └──────────┬──────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
         ▼                     ▼                     ▼
  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
  │  Prefill Node │    │  KV Cache    │    │  Decode Node  │
  │              │    │  Pool        │    │              │
  │  GPU Cluster │    │              │    │  GPU Cluster │
  │  (high FLOPS)│    │  CPU DRAM    │    │  (high BW)   │
  │              │───►│  SSD         │───►│              │
  │  TP=4, EP=32│    │  (pooled)    │    │  TP=4, DP=N  │
  └──────────────┘    └──────────────┘    └──────────────┘
                            ↑
                    Transfer Engine
                    (RDMA + zero-copy)
```

Mooncake's key innovations:

1. **KVCache-centric**: treats KV cache as an independent storage resource rather than a GPU appendage
2. **Pooled storage**: a distributed KV cache pool built from CPU DRAM + SSD
3. **Transfer Engine**: an RDMA-based high-performance transfer engine with zero-copy
4. **Predictive scheduling**: predicts output length from request features and allocates resources accordingly

Results:
- Capacity up **59-498%** depending on workload
- The Transfer Engine has been integrated into [[vllm|vLLM]] V1
- Validates large-scale PD disaggregation in production

### TetriInfer

TetriInfer further optimizes disaggregation scheduling with a Tetris-like approach, tightly packing prefill and decode requests to maximize GPU utilization.

```
TetriInfer's "tight-packing" scheduling:

Traditional:               TetriInfer:
┌──────────┐               ┌──────────┐
│ Prefill  │               │PP│DD│PP│DD│  ← tightly packed
│  large   │               │DD│PP│DD│PP│  ← idle eliminated
│  block   │               │PP│DD│PP│DD│
├──────────┤               └──────────┘
│ Decode   │
│  small   │               Higher GPU utilization
│  block   │
│ (idle)   │
└──────────┘
```

---

## KV Cache Transfer

### The Core Bottleneck

KV cache transfer is the **most critical challenge** in PD disaggregation. After prefill completes, the produced KV cache must travel from the prefill GPU to the decode GPU, and this transfer time directly affects TTFT.

```
KV cache size estimate:

KV cache size = 2 × num_layers × hidden_dim × num_kv_heads × seq_len × dtype_size

Example (Llama 3.1 70B, BF16):
  - num_layers = 80
  - hidden_dim = 8192
  - num_kv_heads = 8 (GQA)
  - dtype_size = 2 bytes (BF16)
  
  Per-token KV size = 2 × 80 × 8192 × 8 × 2 / 8192 = 2.5 MB/token
                                            (divided by GQA group)
  
  ┌─────────────┬────────────────┬─────────────────┐
  │ Seq length   │ KV cache size  │ Transfer (PCIe) │
  ├─────────────┼────────────────┼─────────────────┤
  │ 1K tokens    │ ~0.32 GB       │ ~10 ms          │
  │ 4K tokens    │ ~1.25 GB       │ ~40 ms          │
  │ 32K tokens   │ ~10 GB         │ ~320 ms         │
  │ 128K tokens  │ ~40 GB         │ ~1.28 s         │
  └─────────────┴────────────────┴─────────────────┘
  
  Note: PCIe Gen4 x16 ≈ 32 GB/s, in practice ~25-28 GB/s
```

### Transfer Methods Compared

```
KV cache transfer methods compared:

┌────────────────┬────────────┬──────────────┬──────────────────┐
│ Method          │ Bandwidth  │ Latency (32K)│ Use case         │
├────────────────┼────────────┼──────────────┼──────────────────┤
│ NVLink (intra) │ 900 GB/s   │ ~6 ms        │ Same-machine split│
│ PCIe Gen4      │ 32 GB/s    │ ~32 ms       │ Intra/inter-node │
│ PCIe Gen5      │ 64 GB/s    │ ~16 ms       │ Newer hardware   │
│ RDMA (IB)      │ 100-400    │ ~10-40 ms    │ Cross-machine    │
│                │ Gbps       │              │                  │
│ TCP/IP         │ 10-100     │ ~100-1000 ms │ Not recommended  │
│                │ Gbps       │              │                  │
└────────────────┴────────────┴──────────────┴──────────────────┘
```

### Mooncake Transfer Engine

Mooncake's Transfer Engine is currently the most mature KV cache transfer solution:

```
Mooncake Transfer Engine architecture:

  Prefill GPU                      Decode GPU
  ┌──────────┐                     ┌──────────┐
  │ KV Cache │                     │ KV Cache │
  │ (GPU Mem)│                     │ (GPU Mem)│
  └────┬─────┘                     └────▲─────┘
       │  GPUDirect                     │  GPUDirect
       │  RDMA                          │  RDMA
       ▼                                │
  ┌──────────┐    RDMA fabric     ┌──────────┐
  │   NIC    │ ◄───────────────► │   NIC    │
  └──────────┘    zero-copy       └──────────┘

Features:
  - GPUDirect RDMA: GPU memory transferred directly via NIC, bypassing CPU
  - Zero-copy: no GPU → CPU → NIC multi-copy steps
  - Pipelined: transfer overlaps with generation
  - Integrated into vLLM V1
```

### KV Cache Compression

To reduce transfer volume, KV cache can be compressed:

```
KV cache compression strategies:

┌──────────────────┬────────────┬──────────┬────────────────┐
│ Method            │ Ratio       │ Loss     │ Notes          │
├──────────────────┼────────────┼──────────┼────────────────┤
│ FP16 → INT8      │ 2×          │ Minimal  │ KV quantization │
│ FP16 → INT4      │ 4×          │ Small    │ More aggressive │
│ Token pruning    │ 1.5-3×      │ Tunable  │ Drop unimportant│
│                  │             │          │ tokens          │
│ Sparse attention │ 2-10×       │ Tunable  │ Keep top tokens │
│ Low-rank approx. │ 2-4×        │ Moderate │ SVD compression │
└──────────────────┴────────────┴──────────┴────────────────┘
```

### Latency Analysis: When to Transfer vs Recompute

A key design decision: **when should we transfer the KV cache vs recompute it on the decode side?**

```
Transfer vs recompute decision:

Transfer time = KV_size / bandwidth
Recompute time = prefill_time(seq_len)

                 Transfer wins  │  Recompute wins
                                │
  Transfer time ──────────────X─────────────
                                │
  Recompute time ────X───────────────────────
                     │          │
                     │          │
                 Short seq   Long seq + low BW

Decision rule:
  if transfer_time < recompute_time:
      transfer (most cases)
  else:
      recompute (short seqs + high-latency network)

Typical threshold (Llama 70B, RDMA 100 Gbps):
  seq length < ~256 tokens → recompute may be faster
  seq length > ~256 tokens → transfer faster
```

### Pipelined Transfer

Advanced optimization: overlap KV cache transfer with compute.

```
Pipelined transfer (layer-by-layer streaming):

Without pipelining:
  Prefill:  [████ compute all layers ████]
  Transfer:                              [████ transfer all KV ████]
  Decode:                                                          [████ start ████]

With pipelining:
  Prefill:  [Layer0][Layer1][Layer2][Layer3]...
  Transfer:        [KV0  ][KV1  ][KV2  ][KV3  ]...
  Decode:                                  [start decoding] ← starts much earlier!

  Saved time: transfer is hidden behind compute
```

---

## DeepSeek-V3 Disaggregated Deployment

DeepSeek-V3/R1 is a landmark case of disaggregation in a large-scale production system, showing how to tune PD disaggregation for MoE models.

### Deployment Configuration

```
DeepSeek-V3 disaggregated deployment:

┌──────────────────────────────────────────────────────┐
│                  Prefill Pool                         │
│                                                      │
│  32 GPUs (4 nodes × 8 H800)                         │
│  TP = 4, EP = 32                                     │
│  Expert parallelism covers all 32 GPUs               │
│   → one expert per GPU                               │
│  High compute utilization, large batches             │
│                                                      │
│  Notes:                                              │
│  - MoE all-to-all communication during prefill       │
│    is hidden by the heavy compute                    │
│  - High TP keeps single-request TTFT low             │
└──────────────────────────────┬───────────────────────┘
                               │
                       KV cache transfer
                          (RDMA network)
                               │
┌──────────────────────────────▼───────────────────────┐
│                  Decode Pool                          │
│                                                      │
│  320 GPUs (40 nodes × 8 H800)                        │
│  TP = 4, EP = 320                                    │
│  10× the size of the Prefill Pool                    │
│                                                      │
│  Notes:                                              │
│  - Many GPUs supply enough memory BW and KV-cache    │
│    space                                             │
│  - More expert parallelism → each GPU handles fewer  │
│    experts                                           │
│  - Each GPU carries more concurrent requests         │
│  - MoE all-to-all communication overhead is lower    │
│    during decode (only 1 token per step)             │
└──────────────────────────────────────────────────────┘
```

### Why Decode Needs More GPUs than Prefill

```
Reasons decode needs more GPUs:

1. Time share: decode is 70-90% of total time
   - 100-token output = 100 decode steps
   - Each step only generates 1 token, but reads the full weights

2. Memory: each concurrent request needs KV cache space
   - 1000 concurrent decode reqs × 10 GB/req = 10 TB of KV cache
   - Must be distributed across many GPUs

3. Bandwidth: decode is memory-bandwidth bound
   - More GPUs = more total bandwidth
   - A100 80GB: 2 TB/s → 10 GPUs = 20 TB/s

Typical prefill:decode GPU ratios:
  - Short-output (chat):   1:3 to 1:5
  - Long-output (code):    1:8 to 1:10
  - DeepSeek-V3:           1:10 (32:320)
```

### Relationship with Parallelism Strategy

DeepSeek-V3's disaggregation config is tightly coupled with its [[parallelism-strategies-deep-dive|parallelism strategies]]:

```
DeepSeek-V3 parallelism details:

Prefill Pool (32 GPUs):
  ├── TP = 4   (every 4 GPUs handle one layer of one request)
  ├── EP = 32  (each GPU hosts a different MoE expert)
  └── Equivalent to 32/4 = 8 TP groups
      Each TP group handles 32/8 = 4 experts

Decode Pool (320 GPUs):
  ├── TP = 4   (matches prefill, simplifies KV transfer)
  ├── EP = 320 (much higher expert parallelism)
  └── Equivalent to 320/4 = 80 TP groups
      Each TP group handles fewer experts → faster

Key design choices:
  - Keep TP equal (=4): same KV cache shape, simpler transfer
  - Different EP: each pool optimizes expert distribution separately
  - Prefill EP=32: high expert utilization, compute-heavy
  - Decode EP=320: fewer experts per GPU, lower memory overhead
```

---

## Code Examples

### vLLM PD Disaggregation Configuration

```python
"""vLLM prefill-decode disaggregation configuration example"""

# === Prefill instance config ===
# File: prefill_config.yaml

prefill_config = {
    "model": "deepseek-ai/DeepSeek-V3",
    "tensor_parallel_size": 4,
    
    # Disaggregation mode
    "served_model_name": "deepseek-v3",
    "kv_transfer_config": {
        "kv_connector": "MooncakeConnector",  # Use Mooncake Transfer Engine
        "kv_role": "kv_producer",             # Prefill = KV producer
        "kv_rank": 0,
        "kv_parallel_size": 2,                # 2 prefill instances
    },
    
    # Prefill optimization
    "max_num_seqs": 32,                # Smaller batch (compute-heavy)
    "max_num_batched_tokens": 8192,    # Large token budget (long prompts)
    "gpu_memory_utilization": 0.85,
}

# === Decode instance config ===
# File: decode_config.yaml

decode_config = {
    "model": "deepseek-ai/DeepSeek-V3",
    "tensor_parallel_size": 4,
    
    # Disaggregation mode
    "kv_transfer_config": {
        "kv_connector": "MooncakeConnector",
        "kv_role": "kv_consumer",             # Decode = KV consumer
        "kv_rank": 0,
        "kv_parallel_size": 10,               # 10 decode instances
    },
    
    # Decode optimization
    "max_num_seqs": 512,               # Large batch (memory-BW-heavy)
    "max_num_batched_tokens": 2048,    # Smaller token budget
    "gpu_memory_utilization": 0.92,    # More memory for KV cache
}
```

```bash
# Launch the prefill instance
vllm serve deepseek-ai/DeepSeek-V3 \
    --tensor-parallel-size 4 \
    --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer","kv_rank":0}' \
    --max-num-seqs 32 \
    --port 8100

# Launch the decode instance
vllm serve deepseek-ai/DeepSeek-V3 \
    --tensor-parallel-size 4 \
    --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer","kv_rank":0}' \
    --max-num-seqs 512 \
    --port 8200
```

### Simplified Disaggregation Pseudocode

```python
"""Simplified prefill-decode disaggregation implementation"""

from dataclasses import dataclass
from typing import Optional
import asyncio
from collections import deque


@dataclass
class InferenceRequest:
    request_id: str
    prompt_tokens: list[int]
    max_output_tokens: int
    priority: int = 0


@dataclass
class KVCacheHandle:
    """Reference handle to a KV cache"""
    request_id: str
    prefill_node_id: str
    kv_cache_address: int       # Remote memory address
    num_layers: int
    seq_len: int
    size_bytes: int


class PrefillInstance:
    """Prefill instance — processes the prompt and produces KV cache"""
    
    def __init__(self, model, gpu_id: int, transfer_engine):
        self.model = model
        self.gpu_id = gpu_id
        self.transfer_engine = transfer_engine
        self.node_id = f"prefill_{gpu_id}"
    
    async def process_prefill(
        self, request: InferenceRequest
    ) -> tuple[int, KVCacheHandle]:
        """
        Process a prefill request.
        Returns: (first output token, KV cache handle)
        """
        # 1. Run the prefill forward pass on GPU
        kv_cache, first_token = self.model.prefill(
            request.prompt_tokens
        )
        
        # 2. Register the KV cache with the Transfer Engine (remote-accessible)
        kv_handle = self.transfer_engine.register_kv(
            request_id=request.request_id,
            kv_cache=kv_cache,
            node_id=self.node_id,
        )
        
        return first_token, kv_handle


class DecodeInstance:
    """Decode instance — runs autoregressive token generation"""
    
    def __init__(self, model, gpu_id: int, transfer_engine):
        self.model = model
        self.gpu_id = gpu_id
        self.transfer_engine = transfer_engine
        self.active_requests: dict[str, "DecodeState"] = {}
    
    async def accept_request(
        self,
        request: InferenceRequest,
        kv_handle: KVCacheHandle,
        first_token: int,
    ):
        """Accept request + KV cache from a prefill instance"""
        # 1. Pull KV cache via RDMA
        local_kv = await self.transfer_engine.fetch_kv(kv_handle)
        
        # 2. Initialize decode state
        self.active_requests[request.request_id] = DecodeState(
            request=request,
            kv_cache=local_kv,
            output_tokens=[first_token],
        )
    
    async def decode_step(self) -> list[str]:
        """Run one decode step over all active requests"""
        if not self.active_requests:
            return []
        
        # Batched decode
        batch_inputs = {
            req_id: state.output_tokens[-1]
            for req_id, state in self.active_requests.items()
        }
        
        new_tokens = self.model.decode_batch(batch_inputs)
        
        # Update state, check completion
        finished = []
        for req_id, token in new_tokens.items():
            state = self.active_requests[req_id]
            state.output_tokens.append(token)
            
            EOS = 2
            if (token == EOS 
                    or len(state.output_tokens) >= 
                       state.request.max_output_tokens):
                finished.append(req_id)
        
        # Clean up completed requests
        for req_id in finished:
            del self.active_requests[req_id]
        
        return finished


class DisaggregatedRouter:
    """Global router — coordinates prefill and decode instances"""
    
    def __init__(
        self,
        prefill_instances: list[PrefillInstance],
        decode_instances: list[DecodeInstance],
    ):
        self.prefill_pool = prefill_instances
        self.decode_pool = decode_instances
        self.request_queue: deque[InferenceRequest] = deque()
    
    def select_prefill_instance(self) -> PrefillInstance:
        """Pick the least-loaded prefill instance"""
        return min(self.prefill_pool, 
                   key=lambda p: p.current_load())
    
    def select_decode_instance(self) -> DecodeInstance:
        """Pick the least-loaded decode instance"""
        return min(self.decode_pool,
                   key=lambda d: len(d.active_requests))
    
    async def handle_request(self, request: InferenceRequest):
        """Handle a full inference request"""
        # 1. Pick a prefill instance and run prefill
        prefill_inst = self.select_prefill_instance()
        first_token, kv_handle = await prefill_inst.process_prefill(
            request
        )
        
        # 2. Pick a decode instance
        decode_inst = self.select_decode_instance()
        
        # 3. Transfer the KV cache and start decoding
        await decode_inst.accept_request(
            request, kv_handle, first_token
        )
        
        # 4. Tell prefill instance to release the KV cache
        prefill_inst.transfer_engine.release_kv(kv_handle)


@dataclass
class DecodeState:
    request: InferenceRequest
    kv_cache: object
    output_tokens: list[int]
```

---

## Performance Analysis

### Latency Improvement

```
Latency improvement under disaggregation (Llama 70B, A100 cluster):

Colocated (baseline):
  TTFT (P50):  150 ms    TTFT (P99):  800 ms
  TPOT (P50):   35 ms    TPOT (P99):  250 ms    ← prefill interference

Disaggregated:
  TTFT (P50):  120 ms    TTFT (P99):  400 ms    ← improved (prefill optimized)
  TPOT (P50):   28 ms    TPOT (P99):   45 ms    ← huge improvement, no interference

                 TPOT P99 improvement: 250 ms → 45 ms (5.6×)
```

### Throughput Gain

```
Throughput comparison (under SLO constraints):

SLO: TTFT < 500ms, TPOT < 100ms

┌──────────────────┬──────────────┬────────────────┐
│ System            │ Effective    │ vs colocated   │
│                   │ throughput   │ improvement    │
├──────────────────┼──────────────┼────────────────┤
│ Colocated (vLLM) │ 1.0×         │ baseline       │
│ Chunked prefill  │ 1.8×         │ 1.8×           │
│ DistServe        │ 7.4×         │ 7.4×           │
│ Mooncake         │ 5.0-6.0×     │ 5.0-6.0×       │
│ Splitwise        │ 1.4×         │ 1.4×           │
└──────────────────┴──────────────┴────────────────┘

Note: DistServe's high gains come from advantages under strict SLOs.
      Mooncake's numbers are from production and include transfer overhead.
```

### Cost Analysis

```
Cost analysis (monthly, 1000 QPS, Llama 70B):

Colocated:
  GPU: 64 × A100 (uniform config)
  Total: 64 GPUs
  Monthly: ~$128,000 (at $2/GPU-hour)

Disaggregated:
  Prefill: 8 × A100 (TP=4, 2 groups)
  Decode:  48 × A100 (TP=2, 24 groups)
  Total: 56 GPUs
  Monthly: ~$112,000

Savings: ~12.5%

Bigger gains come from:
  1. Cheaper memory-optimized GPUs can be used for decode
  2. Independent scaling avoids wasted resources
  3. Higher effective throughput → fewer GPUs needed
     for the same request volume
```

---

## Limitations and Challenges

### KV Cache Transfer Overhead

The most fundamental challenge is the extra latency from KV cache transfer:

```
Transfer overhead analysis:

Scenario: Llama 70B, varying sequence lengths

┌──────────────┬──────────┬────────────┬──────────────────────┐
│ Seq length    │ KV size  │ Transfer    │ Worth disaggregating?│
│              │          │ (RDMA)      │                      │
├──────────────┼──────────┼────────────┼──────────────────────┤
│ 256 tokens   │ 80 MB    │ ~1 ms       │ Borderline            │
│ 1K tokens    │ 320 MB   │ ~3 ms       │ ✓ yes                │
│ 4K tokens    │ 1.25 GB  │ ~10 ms      │ ✓ clearly yes        │
│ 32K tokens   │ 10 GB    │ ~80 ms      │ ✓ definitely         │
│ 128K tokens  │ 40 GB    │ ~320 ms     │ ✓ yes, but transfer  │
│              │          │             │   becomes significant │
└──────────────┴──────────┴────────────┴──────────────────────┘

Rule: longer sequences = greater benefit (prefill interference is worse)
      but also longer transfer (high-BW network required)
```

### System Complexity

```
Extra complexity introduced by disaggregation:

1. Network dependence
   - Needs high-speed network (RDMA/InfiniBand)
   - Network failure breaks the whole system
   - Network congestion → TTFT jitter

2. State management
   - KV cache lifetime spans two instances
   - Distributed garbage collection needed
   - Cancellation must notify both sides

3. Fault tolerance
   - Prefill crash: lose in-flight prefill requests
   - Decode crash: lose in-flight decode requests
   - Transfer break: retransmit or recompute

4. Debugging difficulty
   - Problems can be in prefill, transfer, or decode
   - End-to-end tracing and monitoring needed

5. Deployment ops
   - Two different instance configs
   - Scaling policy is more complex
   - Need to monitor both pools' utilization
```

### When Disaggregation Is Not Worth It

```
Scenarios where disaggregation doesn't pay off:

1. Small models (< 7B parameters)
   - Both prefill and decode are fast
   - Transfer overhead is large relative to compute
   - Colocated + chunked prefill is enough

2. Short sequences (< 256 tokens)
   - Small KV cache, fast to transfer
   - But prefill interference is also small
   - Benefit is unclear

3. Low-traffic scenarios
   - No interference between concurrent requests
   - Disaggregation adds fixed overhead
   - Better to invest in single-request optimizations

4. No high-speed network
   - TCP/IP alone makes transfer prohibitive
   - At minimum need RDMA or NVLink

5. Short turns in interactive/multi-turn scenarios
   - Per-turn prompt increment is small
   - KV transfer is frequent but small per piece
   - Chunked prefill may be more appropriate
```

---

## Composing with chunked prefill

A natural confusion: "if prefill is on its own node, why would I still need [[continuous-batching#Chunked Prefill|chunked prefill]] on top?" The answer is that PD disaggregation and chunked prefill operate at **two different granularities** and solve **two different interference problems**:

- **PD disaggregation** eliminates *prefill ↔ decode* interference at the **node** level.
- **Chunked prefill** smooths *prefill ↔ prefill* (and *prefill ↔ in-flight decode*) interference at the **iteration** level — both within the prefill pool and on the decode pool's "extension prefill" path.

Three concrete scenarios where chunked prefill is still load-bearing inside a disaggregated deployment:

**1. Prefill ↔ prefill interference on the prefill pool.** Two long requests arriving close together at the same prefill node still queue behind each other:

```
Without chunked prefill on the prefill node:
  [16K prefill of req A][16K prefill of req B][...]
  req A TTFT = 2.3 s
  req B TTFT = 2.3 s + 2.3 s = 4.6 s    ← B sits behind A

With chunked prefill on the prefill node:
  [chunk_A1 + chunk_B1][chunk_A2 + chunk_B2]...
  req A TTFT ≈ 2.5 s    ← small extra because chunk_B is co-resident
  req B TTFT ≈ 2.5 s    ← almost parallel progress, no queuing wait
```

The second request's TTFT goes from "wait 4.6 s" to "get first token at the same time as A."

**2. Extension prefill on the decode pool.** A decode node is not "decode-only" in the strict sense:

- **Multi-turn dialogue**: when a new user turn arrives, the new tokens must be prefilled into the existing KV cache before decoding resumes.
- **Tool-call returns**: the returned tool result is appended as new tokens that must be prefilled.
- **Speculative-decoding rollback**: a rejected speculation sequence requires re-prefilling a small stretch.

These "extension prefills" are typically 50–2000 tokens — short by initial-prompt standards but still long enough to block a node's in-flight decodes if not chunked. Chunked prefill on the decode pool smooths these out.

**3. Traffic shaping inside each pool.** PD disaggregation only solves *role separation*. Within each role, you still need to smooth load, control tail latency, and prevent occasional outlier requests from poisoning the batch. Chunked prefill is the load-smoothing knob for the prefill pool; small chunks on the decode pool tame the extension-prefill case above.

The mnemonic:

```
PD disaggregation  =  prefill pool ↔ decode pool      DON'T mix
Chunked prefill    =  inside prefill pool / inside    DO mix the
                      decode pool's extension prefill   chunks smartly
```

Same "avoid blocking" idea operating at different scales. They're orthogonal layers, not alternatives.

## Frontier Directions

### Attention-FFN Disaggregation

The next frontier in disaggregation is splitting **attention** and **FFN** layers within the Transformer onto different hardware — full coverage is on the dedicated page [[af-disaggregation]]. Brief motivation:

```
Attention-FFN disaggregation:

Traditional Transformer layer:
  ┌──────────────────────────┐
  │  Attention  →  FFN       │  ← both bound to the same GPU
  └──────────────────────────┘

Disaggregated:
  ┌──────────────┐    ┌──────────────┐
  │  Attention   │    │  FFN         │
  │  (memory-    │    │  (compute-   │
  │   heavy)     │    │   heavy)     │
  │  dedicated HW│ →  │  dedicated HW│
  └──────────────┘    └──────────────┘

Especially natural for MoE models — DP attention + EP MoE
is structurally an AF-disaggregation pattern.
```

### Global KV Cache Management

```
Global KV cache management (Mooncake direction):

Stop treating KV cache as a GPU-local resource,
treat it as cluster-level shared storage:

┌────────────────────────────────────┐
│       Global KV cache pool          │
│   (across CPU DRAM + SSD)          │
│                                    │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ │
│  │KV-A │ │KV-B │ │KV-C │ │KV-D │ │
│  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ │
│     │       │       │       │     │
└─────┼───────┼───────┼───────┼─────┘
      │       │       │       │
  ┌───▼───┐ ┌─▼─┐ ┌──▼──┐ ┌─▼───┐
  │GPU-P0 │ │D0 │ │D1   │ │D2   │
  └───────┘ └───┘ └─────┘ └─────┘

Benefits:
  - Multiple requests can share KV cache for common prefixes
  - Decode instances are migratable (KV isn't pinned locally)
  - Supports long-context cache reuse
```

---

## References

- **Splitwise**: Patel et al., "Splitwise: Efficient Generative LLM Inference Using Phase Splitting", ISCA 2024. [arXiv:2311.18677](https://arxiv.org/abs/2311.18677)
  - First systematic PD disaggregation, heterogeneous-hardware optimization

- **DistServe**: Zhong et al., "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving", OSDI 2024. [arXiv:2401.09670](https://arxiv.org/abs/2401.09670)
  - Pull-based scheduling, cluster-level disaggregation, goodput optimization

- **Mooncake**: Qin et al., "Mooncake: A KVCache-Centric Disaggregated Architecture for LLM Serving", FAST 2025 (Best Paper). [arXiv:2407.00079](https://arxiv.org/abs/2407.00079)
  - KV-cache-centric production system, Transfer Engine

- **TetriInfer**: Xiao et al., "TetriInfer: Distributed LLM Inference via Tetris-like Scheduling", 2024. [arXiv:2401.11181](https://arxiv.org/abs/2401.11181)
  - Tetris-style scheduling optimization

- **Sarathi-Serve**: Agrawal et al., "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve", OSDI 2024. [arXiv:2403.02310](https://arxiv.org/abs/2403.02310)
  - Chunked prefill, an alternative/complement to PD disaggregation

---

## Related Pages

- [[continuous-batching]] — Continuous batching, the scheduling foundation of disaggregation
- [[vllm]] — Mainstream inference engine supporting disaggregated serving
- [[sglang]] — Another inference engine supporting disaggregation
- [[paged-attention]] — KV cache memory management
- [[kv-cache-optimization]] — KV cache optimization techniques (compression, quantization, etc.)
- [[model-parallelism]] — Parallelism strategies that work alongside disaggregation
- [[parallelism-strategies-deep-dive]] — DeepSeek-V3 detailed parallelism configuration
