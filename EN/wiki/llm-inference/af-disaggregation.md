---
title: "Attention-FFN Disaggregation: Splitting Operators, Not Phases"
category: llm-inference
tags: [af-disaggregation, attention-ffn, megascale-infer, moe-serving, disaggregated-inference, hardware-heterogeneity]
created: 2026-05-13
updated: 2026-05-19
status: mature
---

# Attention-FFN Disaggregation: Splitting Operators, Not Phases

> [!info] Page metadata
> - **Type**: technique umbrella (not a single-paper review)
> - **Canonical system**: [MegaScale-Infer (ByteDance, 2024)](https://arxiv.org/abs/2404.02015) — the first explicit AF disaggregation paper
> - **Adjacent / structurally-AF systems**: DeepSeek-V3 (DP attention + EP MoE on uniform hardware), Mooncake (KV-storage-decoupled architecture)
> - **Companion page**: [[prefill-decode-disaggregation]] — the predecessor pattern (PD splits phases; AF splits operators)

> [!abstract]+ TL;DR
> **Attention-FFN (AF) disaggregation** runs *attention layers* and *FFN layers* of the same model on **different GPU pools**, passing the hidden state across the network between every layer. Why: attention is memory-bandwidth-bound (arithmetic intensity ≈ 1–2 FLOPs/byte in decode), FFN is compute-bound (arithmetic intensity grows with batch, easily 1000+ FLOPs/byte at scale). Optimal hardware differs — attention wants HBM bandwidth, FFN wants raw FLOPs. The pattern is **structurally implicit in DP-attention + EP-MoE deployments** ([[parallelism-strategies-deep-dive#13. DeepSeek-V3 Case Study|DeepSeek-V3]], SGLang/vLLM MoE serving) and **explicitly realized in MegaScale-Infer** (ByteDance, 2024). Distinct from [[prefill-decode-disaggregation|PD disaggregation]]: PD splits *phases* (prefill vs decode) along the time axis; AF splits *operators* inside a single forward pass. The two compose into a 2×2 hardware matrix at flagship scale.

---

## Background: why AF disaggregation needed inventing

Standard inference packs attention and FFN onto the same GPUs because that's how Transformer layers are written: `attention → FFN → attention → FFN → ...`. But the two operators have very different hardware demands, and at scale that asymmetry costs real money on uniform hardware.

The roofline picture on H100 (HBM3 at 3.35 TB/s, FP16 at 989 TFLOPs/s) gives a ridge at **≈295 FLOPs/byte**. Operators below the ridge are memory-bandwidth-bound; above the ridge, compute-bound:

| Property | Attention (during decode) | FFN (large batched matmul) |
|----------|---------------------------|---------------------------|
| Arithmetic intensity | 1–2 FLOPs / byte | depends on batch; easily $\geq$ 1000 FLOPs / byte at $B = 64$ |
| Bottleneck | HBM bandwidth (reading KV cache) | FLOPs (dense matmul) |
| Scaling with batch | Sublinear (per-request KV reads dominate) | Linear (more tokens = more matmul) |
| Scaling with sequence | Linear in $S$ (read all of KV per step) | Constant per token |
| Best hardware fit | High-HBM, high-bandwidth GPUs (H100/H200, MI300X) | High-FLOPs accelerators; raw compute matters more than HBM |
| KV cache | Holds it | Doesn't touch it |
| Parallelism that helps | DP attention (partitioned KV cache) | TP / EP (partitioned weights) |

Same chip running two operators with opposite bottlenecks means **one of the two resources is always wasted**:

- When attention runs, the FP16 tensor cores are idle (waiting on HBM reads).
- When FFN runs, HBM bandwidth is slack (compute is the bottleneck).

The natural response: don't pay the cost of provisioning both bandwidth *and* compute on every GPU. Specialize. Build "attention nodes" sized for HBM bandwidth and "FFN nodes" sized for compute, each at the right cost-per-resource for its actual bottleneck, and pass activations between them. This is exactly what [[prefill-decode-disaggregation|PD disaggregation]] already does at the phase level — AF disaggregation extends it down to the operator level inside one forward pass.

| Compared with | Splits what | Cross-pool transfer cadence | Hardware-specialization motivation |
| ------------- | ----------- | --------------------------- | ---------------------------------- |
| **PD disaggregation** | phases (prefill vs decode) | per-request (KV handoff, once) | prefill compute-bound vs decode memory-bound |
| **AF disaggregation** | operators (attention vs FFN) | per-layer per-direction (60–80×/forward) | attention memory-bound vs FFN compute-bound |

---

## The key idea: specialize hardware to operator bottlenecks

> [!quote] The contribution in one sentence
> Treat attention and FFN as two operators with opposite bottlenecks and put them on hardware pools sized for their actual resource demands, paying a per-layer activation round-trip in exchange for not wasting half the resources on every GPU.

Three sub-claims hold the pattern up:

- **The compute / bandwidth asymmetry is real.** Decode attention sits at ~1–2 FLOPs/byte (deep below H100's 295 ridge); batched FFN sits well above. Uniform hardware always wastes one resource.
- **MoE makes the asymmetry sharper.** Attention is dense; MoE FFN is sparse. DP attention and EP MoE have incompatible parallelism that already requires a cross-operator AllToAll — the structural seams for AF are already there.
- **The per-layer transfer cost is tolerable on NVLink-class fabric.** With $H = 8192$, $B = 64$, decode, the per-layer hidden-state round trip is ~1 MiB per direction. Times 80 layers and 2 directions = ~160 MiB per forward step. ~3.4 ms over 400 Gb/s NVLink — ~5 % overhead vs typical 50–80 ms forward, tolerable if specialization recovers more.

Remove any one: lose the asymmetry argument and AF doesn't save you anything; lose MoE and the gain shrinks for dense models; lose the fast interconnect and transfer overhead eats the savings.

---

## How it works

### Architecture in one picture

```
Conventional (per-layer in-place):

  GPU pool ──► [LayerNorm → Attention → Add → LayerNorm → FFN → Add] × 60 layers ──► output
              \____________________________________________________/
                            same hardware, same memory

AF disaggregation:

  Attention nodes (HBM-rich)                 FFN nodes (compute-rich)
  ┌──────────────────────────┐               ┌──────────────────────────┐
  │ KV cache pool             │               │ FFN weights              │
  │ Attention compute         │  hidden       │ FFN compute              │
  │ DP across requests        │  state        │ TP / EP across experts   │
  │                           │  ─────►       │                          │
  │ (Layer N attention out)   │               │ (Layer N FFN computes)   │
  │                           │  ◄─────       │ (returns Layer N output) │
  └──────────────────────────┘   activation   └──────────────────────────┘
       │                                                │
       └───── Repeated for each of the 60 layers ───────┘
                  (60× crossings per forward pass)
```

Per layer, **the hidden state crosses the network twice** — attention output to FFN input, then FFN output back to the next layer's attention input.

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

| Per-layer one-direction cost on 400 Gb/s NVLink ($B = 64$, $H = 8192$) | $\approx 21$ μs |
| **Total transfer per forward step** (× 2 × 80 layers) | **≈ 3.4 ms** |
| Typical decode forward time | ~50–80 ms |
| Transfer overhead | ~5 % |

### Why MoE makes AF natural

For dense models, AF is interesting but a stretch — attention and FFN want slightly different hardware, but a single GPU serves both adequately. For MoE models the asymmetry sharpens until AF is almost structurally forced:

**1. Attention is dense; MoE FFN is sparse.** Attention runs on every token every layer. MoE FFN routes each token to top-$k$ experts (e.g. 8 of 256 for DeepSeek-V3). The two operators have fundamentally different compute and memory access patterns.

**2. The parallelism strategies are incompatible.**

- Attention wants **DP** (each GPU handles a partition of *requests* with its own KV cache slice) — see [[parallelism-strategies-deep-dive#11. DP Attention|DP Attention]].
- MoE FFN wants **EP** (each GPU holds a subset of *experts*; AllToAll dispatch routes tokens).

A single GPU pool trying to be both DP-for-attention and EP-for-FFN ends up doing AllToAll between operators within the same layer — which is *already* an AF-like structure, just on the same hardware.

**3. The attention → FFN transition is already a cross-operator AllToAll** (in EP-MoE deployments). Adding a physical node boundary at the same transition is cheap incremental complexity.

So the line between "DP-attention + EP-MoE on the same cluster" and "explicit AF disaggregation" is thin — the first is the second's structural shadow.

### Composing AF with PD

The two compose into a 2×2 hardware matrix:

```
                    │  Attention pool       │  FFN pool
────────────────────┼───────────────────────┼─────────────────────
Prefill nodes       │  HBM-rich, prefill    │  Compute-rich,
                    │  attention            │  prefill FFN
                    │  (large batch matmul) │  (large batch matmul)
────────────────────┼───────────────────────┼─────────────────────
Decode nodes        │  HBM-rich, decode     │  Compute-rich,
                    │  attention (KV-heavy) │  decode FFN
                    │                       │  (smaller batch)
```

Four pools instead of two. Each pool sized exactly for its bottleneck. This is the configuration MegaScale-Infer-style systems are pushing toward for large-scale MoE serving. Operationally heavy, but justified for cluster scales > 256 GPUs.

---

## Concrete systems

The "Experiments" slot for a technique umbrella — known deployments and their AF-ness, ordered from most explicit to most adjacent.

### MegaScale-Infer (ByteDance, 2024)

The explicit AF disaggregation system. [arXiv:2404.02015](https://arxiv.org/abs/2404.02015).

| Aspect | Design |
| ------ | ------ |
| Pool layout | Two physical GPU pools (attention + FFN) connected by RDMA |
| Activation transfer | Per-layer round trip between pools |
| Pipeline | PP between attention and FFN — layer $L$'s attention runs in parallel with layer $L{-}1$'s FFN on respective pools |
| Parallelism within each pool | Attention: DP (partitioned KV); FFN: EP (expert distribution) |
| Hardware | Optional heterogeneity — attention pool can use H100/H200 (HBM3e), FFN pool can use cheaper compute-dense tiers |
| Reported gain | 1.7×–2.5× throughput improvement over standard MoE inference at the same hardware cost |

### DeepSeek-V3 inference (structurally AF, uniform hardware)

DeepSeek-V3 production inference uses **DP attention + EP MoE on the same cluster** (see [[parallelism-strategies-deep-dive#13. DeepSeek-V3 Case Study]]):

| Phase | TP | EP | DP | Notes |
| ----- | -- | -- | -- | ----- |
| Prefill | 4 + SP | 32 | 8 | 32 redundant experts |
| Decode | 4 + SP | 320 | 80 | — |

Attention and FFN layers communicate via AllToAll within the cluster. This is **logical AF disaggregation** — the parallelism boundaries follow the operator boundary — but the physical hardware is uniform NVL72 / H800 nodes for both. MegaScale-Infer takes the same logical structure and pushes it onto heterogeneous physical hardware.

### Mooncake (Moonshot AI, FAST 2025)

Primarily a [[prefill-decode-disaggregation|PD disaggregation]] system, but its KVCache-Centric architecture decouples KV storage from compute in a way adjacent to AF:

- Centralized KV cache pool (CPU DRAM + SSD), shared across many compute nodes.
- Compute nodes pull KV blocks from the pool as needed.
- Separates KV storage hardware sizing from attention compute hardware sizing — partial AF logic.

Not full AF, but on the same trajectory.

### Activation-transfer cost — when AF pays off

The economics depend on whether the activation-transfer cost is small relative to the specialization savings.

**Cost side.** Per-layer one-direction transfer time (decode, $S_{\text{step}} = 1$):

$$
t_{\text{transfer}} = \frac{B \cdot H \cdot 2}{\text{interconnect bandwidth}}
$$

With $H = 8192$, $B = 64$, 400 Gb/s NVLink (= 50 GB/s): $t_{\text{transfer}} \approx 21$ μs per layer per direction.

**Savings side.** Depend on:

- **MoE sparsity** — sparser experts → more compute concentrated in FFN → bigger win for FFN-specialized hardware.
- **Sequence length** — longer sequence → more KV cache → bigger win for attention-specialized HBM.
- **Batch size** — larger batch → FFN more compute-bound → bigger win.
- **Interconnect** — slower fabric eats more savings via transfer overhead.

Rough rule of thumb (not load-bearing): AF pays off when the cluster is **large enough to make hardware-tier specialization meaningful** (≥ 100 GPUs), **MoE with high sparsity** (active fraction $\leq 5$%), and **NVLink-class or 400+ Gbps interconnect** between pools.

---

## Strengths and limitations

The two strongest points: (1) the **hardware specialization opportunity is real and growing** — as HBM-heavy (HBM4) and FLOPs-heavy (next-gen tensor cores) accelerator tiers diverge, AF's underlying asymmetry sharpens; (2) **MoE's structural AllToAll** means the boundary between "DP-attention + EP-MoE on one cluster" and "explicit AF on two clusters" is small — much of the engineering is already done.

Where the work is honest about scope but the limits matter:

- **Interconnect dependency.** Every layer round-trips activations. Below NVLink-class bandwidth (or 200+ Gbps RDMA), transfer cost eats the specialization savings.
- **Pipeline scheduling complexity.** To overlap activation transfer with compute, you need fine-grained PP between attention and FFN. Bubble management is non-trivial.
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
> | Composable with the other? | ✓ — see [[#Composing AF with PD]] | ✓ |

---

## What this means

Two predictions worth tracking:

1. **MoE serving will pull AF into the mainstream.** As MoE models get sparser (DeepSeek-V3 at 256-experts-top-8, future models pushing toward 1024+), the attention/FFN compute imbalance grows until ignoring it costs more than the operational complexity of separating pools.
2. **The "hardware tier" boundary will move.** Right now H100 / H200 are treated as one tier. As HBM scaling (HBM4 is bandwidth-heavy) decouples from compute scaling (next-gen tensor cores are FLOPs-heavy), the attention/FFN split will follow the hardware split, and AF will become the obvious deployment shape for new flagship MoE models.

What this is *not*: a free 2× speedup for any deployment. It's a cost-efficiency argument at scale, conditional on the right cluster topology and the right model architecture. Small dense models on single-node deployments get nothing.

---

## Source code & reproduction

There is no widely-deployed open-source AF disaggregation framework as of mid-2026. Status of relevant systems:

| System | What you can run | What's closed |
| ------ | ---------------- | ------------- |
| MegaScale-Infer (ByteDance) | Paper only — [arXiv:2404.02015](https://arxiv.org/abs/2404.02015) | All code |
| DeepSeek-V3 inference (logical AF) | Open weights; DP-attention + EP-MoE running via vLLM / SGLang on a single cluster | Production deployment topology |
| Mooncake | Open: [github.com/kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake) — KV-pool / Transfer Engine | Full PD + KV-pool integration into a hosted service |
| vLLM + SGLang DP-attention | DP attention exists; AF disaggregation does not (yet) | — |

To approximate AF on uniform hardware today: **deploy DP-attention + EP-MoE on a single cluster with vLLM or SGLang**. The parallelism boundaries follow the operator boundary, so you get the logical AF structure without the heterogeneous-hardware physical specialization. [[parallelism-strategies-deep-dive#13. DeepSeek-V3 Case Study|DeepSeek-V3's published topology]] is the canonical recipe.

To approximate true AF, you'd need to fork an inference engine and add (a) cross-pool RDMA transport for hidden states, (b) PP scheduler that overlaps activation transfer with compute, (c) per-pool parallelism config. Significant engineering — not a weekend project.

---

## Related reading

- [[prefill-decode-disaggregation]] — Disaggregation across phases (prefill vs decode); the predecessor pattern. Composes with AF into a 2×2 hardware matrix.
- [[parallelism-strategies-deep-dive#11. DP Attention — Data-Parallel Attention for MoE Inference]] — DP attention is the parallelism shape that makes AF natural inside a single cluster.
- [[parallelism-strategies-deep-dive#13. DeepSeek-V3 Case Study]] — Production DP-attention + EP-MoE deployment; the logical AF structure on uniform hardware.
- [[kv-cache-optimization]] — KV cache compression reduces attention's memory-bandwidth cost and shifts the AF cost calculus.
- [[continuous-batching]] — Scheduler-layer load smoothing that runs *inside* each AF pool.
- [[vllm]] / [[sglang]] — The inference engines you'd fork to add AF support.

## References

- **MegaScale-Infer**: ByteDance (2024). *Attention/FFN-Disaggregated MoE Inference*. [arXiv:2404.02015](https://arxiv.org/abs/2404.02015) — the explicit AF disaggregation paper.
- **DeepSeek-V3 technical report**: production DP-attention + EP-MoE deployment that structurally resembles AF on uniform hardware. [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)
- **Mooncake**: Qin et al., *Mooncake: A KVCache-Centric Disaggregated Architecture for LLM Serving*, FAST 2025. [arXiv:2407.00079](https://arxiv.org/abs/2407.00079) — decouples KV storage from compute, adjacent to AF.
- **DistServe**: Zhong et al., OSDI 2024. [arXiv:2401.09670](https://arxiv.org/abs/2401.09670) — foundational PD-disaggregation system whose architectural logic AF extends.
- **Splitwise**: Patel et al., ISCA 2024. [arXiv:2311.18677](https://arxiv.org/abs/2311.18677) — original PD-disaggregation paper.
