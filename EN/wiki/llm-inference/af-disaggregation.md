---
title: "Attention-FFN Disaggregation: Splitting Operators, Not Phases"
category: llm-inference
tags: [af-disaggregation, attention-ffn, megascale-infer, moe-serving, disaggregated-inference, hardware-heterogeneity, concept]
created: 2026-05-13
updated: 2026-05-21
status: mature
---

# Attention-FFN Disaggregation: Splitting Operators, Not Phases

> [!info] Page metadata
> - **Type**: technique umbrella (concept-synthesis page; not a single-paper review)
> - **Canonical paper**: [MegaScale-Infer (ByteDance, ICML 2025)](https://arxiv.org/abs/2504.02263) — the first explicit AF disaggregation system
> - **Adjacent / structurally-AF systems**: DeepSeek-V3 (logical AF via DP attention + EP MoE on uniform hardware), Mooncake (KV-storage-decoupled architecture)
> - **Companion page**: [[prefill-decode-disaggregation]] — the predecessor pattern (PD splits phases; AF splits operators)

---

## Summary (read this if you have 2 minutes)

**What it is.** **Attention-FFN (AF) disaggregation** runs *attention layers* and *FFN layers* of the same model on **different GPU pools**, passing the hidden state across the network between every layer. The motivation is a roofline asymmetry: decode attention sits at $1\text{–}2$ FLOPs/byte (memory-bandwidth-bound, far below H100's 295 FLOPs/byte ridge), while batched FFN at $B \geq 64$ easily exceeds $1000$ FLOPs/byte (compute-bound). Same chip, opposite bottlenecks — one resource is always wasted.

**The one idea.** Treat attention and FFN as two operators with opposite bottlenecks and put them on hardware pools sized for their *actual* resource demands. Three pieces hold the pattern up:

1. **Operator-level specialization** — attention nodes provisioned for HBM bandwidth; FFN/expert nodes provisioned for raw FLOPs. The asymmetry sharpens further under MoE sparsity (FFN becomes memory-bound at low batch but compute-bound at high batch, while attention stays memory-bound throughout).
2. **Per-layer activation transfer** — a hidden-state round trip between pools every layer, twice. On NVLink-class fabric the cost is ~5 % of forward time, which is recoverable by the savings.
3. **Ping-pong pipeline parallelism** — split the batch into micro-batches and shuttle them between attention and FFN pools so each pool stays busy while the other is computing (MegaScale-Infer's contribution).

Remove any one and AF collapses: lose specialization → no savings; lose a fat interconnect → transfer cost eats the gain; lose ping-pong → either pool stalls half the time.

**The canonical implementations.**

| System | What it does | AF-ness |
| ------ | ------------ | ------- |
| [MegaScale-Infer](https://arxiv.org/abs/2504.02263) (ByteDance, ICML 2025) | Two physical pools (Attention Node + Expert Node) connected via custom **M2N** RDMA library; ping-pong PP; heterogeneous hardware optional | Explicit |
| [DeepSeek-V3](https://arxiv.org/abs/2412.19437) production inference | DP attention + EP MoE on a single cluster; AllToAll across the operator boundary | Logical (uniform HW) |
| [Mooncake](https://arxiv.org/abs/2407.00079) (Moonshot AI, FAST 2025) | KV-cache-centric PD system that decouples KV storage from compute | Partial (KV-storage axis) |

MegaScale-Infer reports up to **1.90× per-GPU throughput** vs. state-of-the-art baselines on Mixtral 8×22B / DBRX / Scarlett / DeepSeek-V2 on Ampere and Hopper.

**Why it matters.**

- **MoE-serving cost will be dominated by AF-shaped deployments.** As MoE sparsity grows (DeepSeek-V3: 256 experts top-8; future models pushing toward 1024+), the attention/FFN compute imbalance grows until ignoring it costs more than the operational complexity of separating pools.
- **The "hardware tier" boundary will follow.** HBM-heavy (HBM4) and FLOPs-heavy (next-gen tensor cores) accelerators diverge. AF gives you the natural seam to put the right silicon under the right operator.
- **It composes with PD.** AF and [[prefill-decode-disaggregation|PD]] split along orthogonal axes (operators vs. phases) and stack into a 2×2 hardware matrix at flagship scale.

---

# Depth (drill-down starts here)

The summary above is the executive layer. Everything below is for the careful reader who wants the roofline math, the three system designs, and the cost calculus.

## Background: why AF disaggregation needed inventing

Standard inference packs attention and FFN onto the same GPUs because that's how Transformer layers are written: `attention → FFN → attention → FFN → ...`. But the two operators have very different hardware demands, and at scale that asymmetry costs real money on uniform hardware.

The roofline picture on H100 (HBM3 at 3.35 TB/s, FP16 at 989 TFLOPs/s) gives a ridge at **≈ 295 FLOPs/byte**. Operators below the ridge are memory-bandwidth-bound; above the ridge, compute-bound:

| Property | Attention (during decode) | FFN (large batched matmul) |
|----------|---------------------------|---------------------------|
| Arithmetic intensity | 1–2 FLOPs / byte | depends on batch; easily $\geq 1000$ FLOPs / byte at $B = 64$ |
| Bottleneck | HBM bandwidth (reading KV cache) | FLOPs (dense matmul) |
| Scaling with batch | Sublinear (per-request KV reads dominate) | Linear (more tokens = more matmul) |
| Scaling with sequence | $O(S)$ (read all of KV per step) | Constant per token |
| Best hardware fit | High-HBM, high-bandwidth GPUs (H100/H200, MI300X) | High-FLOPs accelerators; raw compute matters more than HBM |
| KV cache | Holds it | Doesn't touch it |
| Parallelism that helps | DP attention (partitioned KV cache) | TP / EP (partitioned weights) |

The same chip running two operators with opposite bottlenecks means **one of the two resources is always wasted**:

- When attention runs, the FP16 tensor cores are idle (waiting on HBM reads).
- When FFN runs, HBM bandwidth is slack (compute is the bottleneck).

MegaScale-Infer's Fig. 1 visualizes the asymmetry directly:

![GPU utilization of attention vs FFN across dense, MoE, and AF deployments (MegaScale-Infer Fig. 1)](EN/wiki/llm-inference/af-disaggregation-figs/gpu-utilization-comparison.png)

In a dense model (a), FFN saturates the GPU at modest batch size while attention stays low. Under MoE (b) the gap widens: each expert receives only a fraction of the batch, so FFN now needs *much* larger global batch to saturate, but attention compute stays the same. In MegaScale-Infer (c), separating the two pools lets each push to its own `max bs` independently — pulling the FFN curve left (higher utilization per unit cost) and lifting the attention curve via DP scaling.

The natural response is to specialize: build "attention nodes" sized for HBM bandwidth and "expert nodes" sized for compute, each at the right cost-per-resource for its actual bottleneck, and pass activations between them. This is what [[prefill-decode-disaggregation|PD disaggregation]] already does at the *phase* level — AF extends the same idea down to the *operator* level inside one forward pass.

| Compared with | Splits what | Cross-pool transfer cadence | Hardware-specialization motivation |
| ------------- | ----------- | --------------------------- | ---------------------------------- |
| **PD disaggregation** | phases (prefill vs decode) | per-request (KV handoff, once) | prefill compute-bound vs decode memory-bound |
| **AF disaggregation** | operators (attention vs FFN) | per-layer per-direction (60–80×/forward) | attention memory-bound vs FFN compute-bound |

## The key idea: specialize hardware to operator bottlenecks

> [!quote] The contribution in one sentence
> Treat attention and FFN as two operators with opposite bottlenecks, put them on hardware pools sized for their actual resource demands, and pay a per-layer activation round trip in exchange for not wasting half the resources on every GPU.

Three sub-claims hold the pattern up:

- **The compute / bandwidth asymmetry is real.** Decode attention sits at $\sim 1\text{–}2$ FLOPs/byte (deep below H100's 295 ridge); batched FFN sits well above. Uniform hardware always wastes one resource.
- **MoE makes the asymmetry sharper.** Attention is dense; MoE FFN is sparse. Each expert sees a small slice of the batch, so FFN needs a larger *global* batch to reach the roofline, while attention's intensity doesn't change. DP attention and EP MoE also have incompatible parallelism that already requires a cross-operator AllToAll — the structural seams for AF are already there.
- **The per-layer transfer cost is tolerable on NVLink-class fabric.** With $H = 8192$, $B = 64$, decode, the per-layer hidden-state round trip is ~1 MiB per direction. Times 80 layers and 2 directions = ~160 MiB per forward step. ~3.4 ms over 400 Gb/s NVLink — ~5 % overhead vs. a typical 50–80 ms forward, tolerable if specialization recovers more.

Remove any one: lose the asymmetry argument and AF doesn't save you anything; lose MoE and the gain shrinks for dense models; lose the fast interconnect and transfer overhead eats the savings.

## How it works

### Architecture in one picture

The MegaScale-Infer runtime instance architecture (paper Fig. 3) is the canonical AF physical layout:

![MegaScale-Infer runtime instance architecture (paper Fig. 3)](EN/wiki/llm-inference/af-disaggregation-figs/megascale-infer-architecture.png)

Two physical pools — an **Attention Node** (replicated $M$-fold across requests, holding attention parameters + KV cache, parallelism via TP within the node + DP across nodes) and an **Expert Node** (sharded across $N$ experts, holding only that expert's parameters, parallelism via TP within and EP across). The pools are bridged by two custom RDMA primitives — **M2N** (Attention → Expert dispatch) and **N2M** (Expert → Attention combine) — running over IB/Ethernet. The whole thing operates as a **Ping-Pong Pipeline**: layer $L$'s attention micro-batch runs in parallel with layer $L{-}1$'s FFN micro-batch on the opposite pool.

Or in ASCII, the per-layer dataflow vs. conventional in-place execution:

```
Conventional (per-layer in-place):

  GPU pool ──► [LayerNorm → Attention → Add → LayerNorm → FFN → Add] × 60 layers ──► output
              \____________________________________________________/
                            same hardware, same memory

AF disaggregation:

  Attention nodes (HBM-rich)                 FFN / Expert nodes (compute-rich)
  ┌──────────────────────────┐               ┌──────────────────────────┐
  │ KV cache pool            │               │ FFN / expert weights     │
  │ Attention compute        │   M2N         │ Expert compute           │
  │ DP across requests       │  ─────►       │ EP across experts        │
  │                          │               │                          │
  │ (Layer N attention out)  │   N2M         │ (Layer N FFN computes)   │
  │                          │  ◄─────       │ (returns Layer N output) │
  └──────────────────────────┘  activation   └──────────────────────────┘
       │                                                │
       └───── Repeated for each of the 60 layers ───────┘
                  (60× crossings per forward pass)
```

### Activation transfer cost

For batch $B$, sequence step $S_{\text{step}}$, hidden dim $H$, dtype 2 bytes, per layer one direction:

$$
\text{bytes per direction per layer} = B \cdot S_{\text{step}} \cdot H \cdot 2
$$

Concrete (Llama-70B-class, $H = 8192$, batch 64, decode step):

$$
64 \times 1 \times 8192 \times 2 = 1\,\text{MiB per direction per layer}
$$

Times 2 directions × 80 layers = **160 MiB per forward step**. Over 400 Gb/s NVLink-class fabric (~50 GB/s): ~3.4 ms. Over 200 Gb/s InfiniBand: ~6.8 ms. Doable on NVLink, marginal on IB, infeasible without a fat fabric.

> [!example] Per-step transfer math, decode
> | Quantity | Value |
> | -------- | ----: |
> | Per-layer one-direction cost on 400 Gb/s NVLink ($B = 64$, $H = 8192$) | $\approx 21$ μs |
> | Total transfer per forward step (× 2 × 80 layers) | $\approx 3.4$ ms |
> | Typical decode forward time | 50–80 ms |
> | Transfer overhead | ~5 % |

### Why MoE makes AF natural

For dense models, AF is interesting but a stretch — attention and FFN want slightly different hardware, but a single GPU serves both adequately. For MoE models the asymmetry sharpens until AF is almost structurally forced:

**1. Attention is dense; MoE FFN is sparse.** Attention runs on every token every layer. MoE FFN routes each token to top-$k$ experts (e.g. 8 of 256 for DeepSeek-V3). The two operators have fundamentally different compute and memory access patterns.

**2. The parallelism strategies are incompatible.**

- Attention wants **DP** (each GPU handles a partition of *requests* with its own KV cache slice) — see [[parallelism-strategies-deep-dive#11. DP Attention|DP Attention]].
- MoE FFN wants **EP** (each GPU holds a subset of *experts*; AllToAll dispatch routes tokens).

A single GPU pool trying to be both DP-for-attention and EP-for-FFN ends up doing AllToAll between operators within the same layer — which is *already* an AF-like structure, just on the same hardware.

**3. The attention → FFN transition is already a cross-operator AllToAll** (in EP-MoE deployments). Adding a physical node boundary at the same transition is cheap incremental complexity.

So the line between "DP-attention + EP-MoE on the same cluster" and "explicit AF disaggregation" is thin — the first is the second's structural shadow.

### Ping-pong pipeline parallelism (MegaScale-Infer)

Naïvely, if attention and FFN sit on different pools, one pool is always idle while the other computes. MegaScale-Infer fixes this by splitting the batch into **micro-batches** and shuttling them between pools in a ping-pong pattern:

- Micro-batch A: attention(L) → M2N → expert(L) → N2M → attention(L+1) → ...
- Micro-batch B: while A is on the expert pool, B is on the attention pool — and vice versa.

With two micro-batches, both pools stay busy continuously, hiding the M2N/N2M transfer behind compute. The per-pool parallelism (TP within, DP/EP across) is independent for each pool, so each can be sized to fully saturate its bottleneck without affecting the other.

The trade-off is latency: ping-pong fills the pipeline at the cost of waiting one micro-batch worth of time before the first output token. For decode at high concurrency, where throughput is the gating metric, this is the right deal.

### Composing AF with PD

The two compose into a 2×2 hardware matrix:

```
                    │  Attention pool       │  FFN / Expert pool
────────────────────┼───────────────────────┼─────────────────────
Prefill nodes       │  HBM-rich, prefill    │  Compute-rich,
                    │  attention            │  prefill FFN
                    │  (large batch matmul) │  (large batch matmul)
────────────────────┼───────────────────────┼─────────────────────
Decode nodes        │  HBM-rich, decode     │  Compute-rich,
                    │  attention (KV-heavy) │  decode FFN
                    │                       │  (smaller batch)
```

Four pools instead of two. Each pool sized exactly for its bottleneck. This is the configuration MegaScale-Infer-style systems are pushing toward for large-scale MoE serving. Operationally heavy, but justified for cluster scales $> 256$ GPUs.

## Concrete systems

The "Experiments" slot for a technique-umbrella page — known deployments and their AF-ness, ordered from most explicit to most adjacent.

### MegaScale-Infer (ByteDance, ICML 2025)

The explicit AF disaggregation system. [arXiv:2504.02263](https://arxiv.org/abs/2504.02263).

| Aspect | Design |
| ------ | ------ |
| Pool layout | Two physical pools — *Attention Node* (replicated $M$-fold, TP within) and *Expert Node* (sharded across $N$ experts, TP within, EP across), connected by RDMA |
| Activation transfer | Per-layer round trip via custom **M2N** (attention → expert dispatch) and **N2M** (expert → attention combine) libraries — eliminate GPU-to-CPU copies, group init overhead, and GPU sync that crippled MoE AllToAll |
| Pipeline | Ping-pong pipeline parallelism — split each batch into micro-batches and shuttle between pools so each is always busy |
| Parallelism within each pool | Attention: TP + DP across requests; Expert: TP + EP across experts |
| Hardware | Optional heterogeneity — attention pool can use HBM-rich tiers (H100/H200), expert pool can use compute-dense tiers; uniform also works |
| Reported gain | Up to **1.90× per-GPU throughput** vs. SOTA baselines on Mixtral 8×22B / DBRX / Scarlett / DeepSeek-V2; experiments on both Ampere and Hopper |

The M2N library is the deepest engineering contribution: native MoE token dispatch via `torch.distributed`'s AllToAll has tail-latency pathologies (Fig. 5 in the paper), so the team rewrote the primitive from scratch on top of GDRCopy with separate Sender/Receiver state machines (Figs. 6–7).

### DeepSeek-V3 inference (structurally AF, uniform hardware)

DeepSeek-V3 production inference uses **DP attention + EP MoE on the same cluster** (see [[parallelism-strategies-deep-dive#13. DeepSeek-V3 Case Study]]):

| Phase | TP | EP | DP | Notes |
| ----- | -- | -- | -- | ----- |
| Prefill | 4 + SP | 32 | 8 | 32 redundant experts |
| Decode | 4 + SP | 320 | 80 | — |

Attention and FFN layers communicate via AllToAll within the cluster. This is **logical AF disaggregation** — the parallelism boundaries follow the operator boundary — but the physical hardware is uniform NVL72 / H800 nodes for both. MegaScale-Infer takes the same logical structure and pushes it onto heterogeneous physical hardware while adding ping-pong PP to hide transfer cost.

### Mooncake (Moonshot AI, FAST 2025)

Primarily a [[prefill-decode-disaggregation|PD disaggregation]] system, but its KVCache-Centric architecture decouples KV storage from compute in a way adjacent to AF:

- Centralized KV cache pool (CPU DRAM + SSD), shared across many compute nodes.
- Compute nodes pull KV blocks from the pool as needed.
- Separates KV storage hardware sizing from attention compute hardware sizing — partial AF logic on the storage axis.

Not full AF, but on the same trajectory.

### When AF pays off

The economics depend on whether the activation-transfer cost is small relative to the specialization savings.

**Cost side.** Per-layer one-direction transfer time (decode, $S_{\text{step}} = 1$):

$$
t_{\text{transfer}} = \frac{B \cdot H \cdot 2}{\text{interconnect bandwidth}}
$$

With $H = 8192$, $B = 64$, 400 Gb/s NVLink (= 50 GB/s): $t_{\text{transfer}} \approx 21$ μs per layer per direction.

**Savings side.** Depend on:

- **MoE sparsity** — sparser experts → more compute concentrated in FFN per active expert → bigger win for FFN-specialized hardware.
- **Sequence length** — longer sequence → more KV cache → bigger win for attention-specialized HBM.
- **Batch size** — larger batch → FFN more compute-bound → bigger win.
- **Interconnect** — slower fabric eats more savings via transfer overhead.

> [!tip] Rough rule of thumb
> AF pays off when the cluster is **large enough to make hardware-tier specialization meaningful** ($\geq 100$ GPUs), the model is **MoE with high sparsity** (active fraction $\leq 5$%), and the interconnect between pools is **NVLink-class or 400+ Gbps RDMA**.

## Strengths and limitations

The two strongest points: (1) the **hardware-specialization opportunity is real and growing** — as HBM-heavy (HBM4) and FLOPs-heavy (next-gen tensor cores) accelerator tiers diverge, AF's underlying asymmetry sharpens; (2) **MoE's structural AllToAll** means the boundary between "DP-attention + EP-MoE on one cluster" and "explicit AF on two clusters" is small — much of the engineering is already done.

Where the work is honest about scope but the limits matter:

- **Interconnect dependency.** Every layer round-trips activations. Below NVLink-class bandwidth (or 200+ Gbps RDMA), transfer cost eats the specialization savings.
- **Ping-pong adds a latency floor.** Micro-batching means waiting one micro-batch worth of time before first output token; throughput-friendly, TTFT-unfriendly.
- **Heterogeneous hardware operational cost.** Running two different GPU tiers in production multiplies infra complexity (driver versions, NCCL topology, monitoring, failure handling).
- **Mostly MoE-specific value.** Dense models gain less — the attention/FFN asymmetry is smaller for them.
- **Tail-latency interaction.** Activation transfer adds a per-layer latency floor that the slowest pool dominates; if either pool stalls, the whole forward stalls.
- **Still early.** Outside of MegaScale-Infer, public AF deployments are rare. Many "AF-like" systems (DeepSeek-V3 inference, SGLang MoE) are really DP-attention + EP-MoE on uniform hardware — the *logical* structure but not the *physical* specialization.

> [!warning] PD vs AF in one table
> Easy to confuse because they share the "disaggregation" word but split along orthogonal axes.
>
> | Property | [[prefill-decode-disaggregation\|PD disaggregation]] | AF disaggregation |
> | -------- | ----------------------- | ----------------- |
> | What splits | **Phases** along time axis | **Operators** inside one forward pass |
> | Cross-pool transfer | KV cache, once per request | Hidden state, every layer × every step |
> | Transfer cadence | Per-request | Per-layer |
> | Without specialized hardware? | Still useful (traffic isolation) | Much less useful |
> | Maturity | Mainstream (Splitwise, DistServe, Mooncake) | Emerging (MegaScale-Infer) |
> | Composable with the other? | ✓ — see *Composing AF with PD* | ✓ |

## What this means

Two predictions worth tracking:

1. **MoE serving will pull AF into the mainstream.** As MoE models get sparser (DeepSeek-V3 at 256-experts-top-8, future models pushing toward 1024+), the attention/FFN compute imbalance grows until ignoring it costs more than the operational complexity of separating pools.
2. **The "hardware tier" boundary will move.** Right now H100 / H200 are treated as one tier. As HBM scaling (HBM4 is bandwidth-heavy) decouples from compute scaling (next-gen tensor cores are FLOPs-heavy), the attention/FFN split will follow the hardware split, and AF will become the obvious deployment shape for new flagship MoE models.

What this is *not*: a free 2× speedup for any deployment. It's a cost-efficiency argument at scale, conditional on the right cluster topology and the right model architecture. Small dense models on single-node deployments get nothing.

## Source code & reproduction

There is no widely-deployed open-source AF disaggregation framework as of mid-2026. Status of relevant systems:

| System | What you can run | What's closed |
| ------ | ---------------- | ------------- |
| MegaScale-Infer (ByteDance) | Paper only — [arXiv:2504.02263](https://arxiv.org/abs/2504.02263) | All code, M2N library |
| DeepSeek-V3 inference (logical AF) | Open weights; DP-attention + EP-MoE running via vLLM / SGLang on a single cluster | Production deployment topology |
| Mooncake | Open: [github.com/kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake) — KV-pool / Transfer Engine | Full PD + KV-pool integration into a hosted service |
| vLLM + SGLang DP-attention | DP attention exists; AF disaggregation does not (yet) | — |

To approximate AF on uniform hardware today: **deploy DP-attention + EP-MoE on a single cluster with vLLM or SGLang**. The parallelism boundaries follow the operator boundary, so you get the logical AF structure without the heterogeneous-hardware physical specialization. [[parallelism-strategies-deep-dive#13. DeepSeek-V3 Case Study|DeepSeek-V3's published topology]] is the canonical recipe.

To approximate true AF, you'd need to fork an inference engine and add (a) cross-pool RDMA transport for hidden states (the equivalent of M2N/N2M), (b) ping-pong micro-batch scheduler that overlaps activation transfer with compute, (c) per-pool parallelism config. Significant engineering — not a weekend project.

## Related reading

- [[prefill-decode-disaggregation]] — Disaggregation across phases (prefill vs decode); the predecessor pattern. Composes with AF into a 2×2 hardware matrix.
- [[prfaas]] — PD disaggregation extended to cross-datacenter scope via hybrid-attention KVCache reduction; the natural sibling-of-AF on the time axis when intra-DC PD won't compose with the available hardware geography.
- [[parallelism-strategies-deep-dive#11. DP Attention — Data-Parallel Attention for MoE Inference]] — DP attention is the parallelism shape that makes AF natural inside a single cluster.
- [[parallelism-strategies-deep-dive#13. DeepSeek-V3 Case Study]] — Production DP-attention + EP-MoE deployment; the logical AF structure on uniform hardware.
- [[kv-cache-optimization]] — KV cache compression reduces attention's memory-bandwidth cost and shifts the AF cost calculus.
- [[continuous-batching]] — Scheduler-layer load smoothing that runs *inside* each AF pool.
- [[vllm]] / [[sglang]] — The inference engines you'd fork to add AF support.

## References

- **MegaScale-Infer**: Zhu et al., *MegaScale-Infer: Serving Mixture-of-Experts at Scale with Disaggregated Expert Parallelism*, ICML 2025 / ByteDance. [arXiv:2504.02263](https://arxiv.org/abs/2504.02263) — the explicit AF disaggregation paper.
- **DeepSeek-V3 technical report**: production DP-attention + EP-MoE deployment that structurally resembles AF on uniform hardware. [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)
- **Mooncake**: Qin et al., *Mooncake: A KVCache-Centric Disaggregated Architecture for LLM Serving*, FAST 2025. [arXiv:2407.00079](https://arxiv.org/abs/2407.00079) — decouples KV storage from compute, adjacent to AF.
- **DistServe**: Zhong et al., OSDI 2024. [arXiv:2401.09670](https://arxiv.org/abs/2401.09670) — foundational PD-disaggregation system whose architectural logic AF extends.
- **Splitwise**: Patel et al., ISCA 2024. [arXiv:2311.18677](https://arxiv.org/abs/2311.18677) — original PD-disaggregation paper.
