---
title: "Attention-FFN Disaggregation: Splitting Operators, Not Phases"
category: llm-inference
tags: [af-disaggregation, attention-ffn, megascale-infer, moe-serving, disaggregated-inference, hardware-heterogeneity]
created: 2026-05-13
updated: 2026-05-13
status: mature
---

# Attention-FFN Disaggregation: Splitting Operators, Not Phases

> [!abstract]+ TL;DR
> **Attention-FFN (AF) disaggregation** runs *attention layers* and *FFN layers* of the same model on **different GPU pools**, passing the hidden state across the network between every layer. Why: attention is memory-bandwidth-bound (arithmetic intensity ≈ 1–2 FLOPs/byte in decode), FFN is compute-bound (arithmetic intensity grows with batch, easily 1000+ FLOPs/byte). Optimal hardware differs — attention wants HBM bandwidth, FFN wants raw FLOPs. The pattern is **structurally implicit in DP-attention + EP-MoE deployments** (DeepSeek-V3 inference, SGLang/vLLM MoE serving) and **explicitly realized in MegaScale-Infer** (ByteDance, 2024). Distinct from [[prefill-decode-disaggregation|PD disaggregation]]: PD splits *phases*, AF splits *operators* inside a single forward pass.

---

## The asymmetry that makes AF disaggregation interesting

Standard inference packs attention and FFN onto the same GPUs because that's how Transformer layers are written. But the two operators have very different demands:

| Property | Attention (during decode) | FFN (large batched matmul) |
|----------|---------------------------|---------------------------|
| Bottleneck | HBM bandwidth (reading KV cache) | FLOPs (dense matmul) |
| Arithmetic intensity | 1–2 FLOPs / byte | depends on batch; easily $\geq$ 1000 FLOPs / byte |
| Scaling with batch | Sublinear (KV per request reads dominate) | Linear (more tokens = more matmul work) |
| Scaling with sequence | Linear in $S$ (read all of KV cache per step) | Constant per token |
| Best hardware fit | High-HBM, high-bandwidth GPUs (H100/H200, MI300X) | High-FLOPs accelerators; raw compute matters more than HBM |
| KV cache | Holds it | Doesn't touch it |
| Parallelism that helps | DP attention (KV partitioned across requests) | TP / EP (weights partitioned) |

The roofline picture for decode on H100 (HBM3 at 3.35 TB/s, FP16 at 989 TFLOPs/s) gives a ridge at **≈295 FLOPs/byte**. Below the ridge → memory-bandwidth-bound. Above → compute-bound. Attention sits at ~1–2 (deep below the ridge), FFN at batched workloads sits well above. Same chip running two operators with opposite bottlenecks means **one of the two resources is wasted at any time**:

- When attention runs, the FP16 tensor cores are idle (waiting on HBM reads).
- When FFN runs, HBM bandwidth is slack (compute is the bottleneck).

AF disaggregation says: instead of paying the cost of provisioning both bandwidth *and* compute on every GPU, **specialize**. Build "attention nodes" (HBM-rich) and "FFN nodes" (compute-rich), each sized for its actual bottleneck, and pass activations between them.

---

## The architecture in one picture

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

Per layer, **the hidden state crosses the network twice** (attention output → FFN input, then FFN output → next layer's attention input on the attention node). For batch $B$, sequence $S$, hidden dim $H$, dtype 2 bytes, per layer one direction:

$$
\text{bytes} = B \cdot S \cdot H \cdot 2
$$

Concrete: Llama-70B-class, $H = 8192$, batch 64, decode (1 token per step), per direction per layer:

$$
64 \times 1 \times 8192 \times 2 = 1\,\text{MiB}
$$

Times 2 (round trip) × 80 layers = **160 MiB per forward step**. Over 400 Gb/s NVLink-class fabric: ~3 ms; over 200 Gb/s InfiniBand: ~6 ms. Doable, but only with a fat interconnect — the design space is real but narrow.

---

## Why MoE makes AF disaggregation natural

For dense models, AF disaggregation is interesting but a stretch — attention and FFN want the *same* hardware to be slightly different, but a single GPU can serve both adequately. For MoE models, the asymmetry sharpens to the point where it's almost structurally forced:

**(1) Attention is dense; MoE FFN is sparse.** Attention runs on every token, every layer. MoE FFN routes each token to top-$k$ experts (e.g. 8 of 256 for DeepSeek-V3). The FFN compute pattern is fundamentally different — most experts are idle most of the time on any given GPU, but in aggregate the FFN compute dominates.

**(2) The parallelism strategies are incompatible.**
- Attention wants **DP** (each GPU handles a partition of *requests*, with its own KV cache slice) — see [[parallelism-strategies-deep-dive#11. DP Attention|DP Attention]].
- MoE FFN wants **EP** (each GPU holds a subset of *experts*, AllToAll dispatch routes tokens).
A single GPU pool that wants to be both DP-for-attention and EP-for-FFN ends up doing AllToAll between operators within the same layer — which is *already* an AF-like structure, just on the same hardware.

**(3) The transition between attention and FFN is already an AllToAll (in EP-MoE deployments).** MoE inference already has the activation-transfer step baked into every layer (for expert routing). Adding a *physical* node boundary at the same transition is cheap incremental complexity.

So in practice, **the line between "DP-attention + EP-MoE on the same cluster" and "explicit AF disaggregation" is thin** — the first is the second's structural shadow.

---

## Concrete systems

### MegaScale-Infer (ByteDance, 2024)

The explicit AF-disaggregation system. Reported design:

- **Two physical GPU pools** — attention pool and FFN pool — connected by RDMA.
- **Per-layer activation transfer** between pools (round trip per layer).
- **Pipeline parallelism between attention and FFN** to overlap transfer with compute. Layer $L$'s attention and layer $L{-}1$'s FFN run in parallel on their respective pools.
- **Different parallelism within each pool**: attention pool runs DP attention (each attention GPU partitions request KV caches), FFN pool runs EP for MoE experts.
- **Hardware heterogeneity (optional)**: attention pool can use H100/H200 (HBM3e bandwidth); FFN pool can use cheaper compute-dense tiers.

Reported claim: 1.7×–2.5× throughput improvement over standard MoE inference at the same hardware cost, by sizing each pool for its actual bottleneck.

### DeepSeek-V3 inference (structurally AF-shaped, not explicitly disaggregated)

DeepSeek-V3 production inference uses **DP attention + EP MoE on the same cluster** (see [[parallelism-strategies-deep-dive#13. DeepSeek-V3 Case Study|DeepSeek-V3 case study]]):

- Prefill: TP=4 + SP, EP=32, DP=8, 32 redundant experts.
- Decode: TP=4 + SP, EP=320, DP=80.

The attention and FFN layers communicate via AllToAll within the cluster. This is **logical AF disaggregation** — the parallelism boundaries follow the operator boundary — but the *physical* hardware is uniform. Same NVL72 / H800 nodes for both.

This is what the MegaScale-Infer architecture takes one step further: the same logical structure, but on **heterogeneous physical hardware**.

### Mooncake (Moonshot AI, FAST 2025)

Mooncake is primarily a PD-disaggregation system, but its KVCache-Centric architecture decouples KV storage from compute in a way adjacent to AF disaggregation:

- Centralized KV cache pool (CPU DRAM + SSD) shared across many compute nodes.
- Compute nodes pull KV blocks from the pool as needed.
- This separates KV storage hardware sizing from attention compute hardware sizing — partial AF logic.

Not full AF, but on the same trajectory.

---

## Activation transfer cost — when does AF pay off?

The economics depend on whether the activation-transfer cost is small relative to the savings from specialization.

**Cost side (per layer, one direction):**

$$
t_{\text{transfer}} = \frac{B \cdot S_{\text{step}} \cdot H \cdot 2}{\text{interconnect bandwidth}}
$$

For decode ($S_{\text{step}} = 1$), $H = 8192$, $B = 64$, NVLink (400 Gb/s = 50 GB/s):

$$
t_{\text{transfer}} = \frac{64 \times 8192 \times 2}{50 \times 10^9} \approx 21\,\mu\text{s per layer per direction}
$$

× 2 directions × 80 layers = **3.4 ms per forward step**. Compare to a typical 50–80 ms / forward decode → ~5% overhead, tolerable if specialization gain is bigger.

**Savings side:**

If attention nodes can be specced for HBM bandwidth (e.g. H200 with HBM3e at 4.8 TB/s, costlier per FLOP but cheaper per byte), and FFN nodes can use H100-class or even compute-dense accelerators (better per FLOP), the overall cost-per-throughput drops. The breakeven depends on:

- **MoE sparsity** — sparser experts → more compute concentrated in FFN → bigger win for FFN-specialized hardware.
- **Sequence length** — longer sequence → more KV cache → bigger win for attention-specialized HBM.
- **Batch size** — larger batch → FFN more compute-bound → bigger win.
- **Interconnect** — slower fabric eats more of the savings via transfer overhead.

Rule of thumb (rough, not load-bearing): AF pays off when the cluster is **large enough to make hardware-tier specialization meaningful** (≥ 100 GPUs), **MoE with high sparsity** (active fraction $\leq$ 5%), and **NVLink-class or 400+ Gbps interconnect** between pools.

---

## Comparison: PD vs AF disaggregation

| Property | [[prefill-decode-disaggregation\|PD disaggregation]] | AF disaggregation |
|----------|-----------------------|-------------------|
| What splits | **Phases** (prefill vs decode) of a request lifecycle | **Operators** (attention vs FFN) inside one forward pass |
| Cross-pool transfer | KV cache, once per request (prefill → decode handoff) | Hidden state, once per *layer per direction* (60–80×/forward) |
| Transfer volume | Large but rare (~MB per request, once) | Small but frequent (~MB per layer, every step) |
| Transfer cadence | Per-request | Per-layer |
| Hardware specialization motivation | Different compute profile per phase (prefill compute-bound, decode memory-bound) | Different compute profile per operator (attention memory-bound, FFN compute-bound) |
| Failure mode if you skip it | Long prefills inflate decode TBT | Bottlenecked hardware idle for half the layer |
| Works without specialized hardware? | Yes — useful purely for traffic isolation | Less useful without hardware specialization |
| Maturity in production | Mainstream (Splitwise, DistServe, Mooncake, DeepSeek-V3) | Emerging (MegaScale-Infer; structurally implicit in DP-attention + EP-MoE) |
| Composable with the other? | ✓ — see [[#Composing AF with PD]] | ✓ |

The key conceptual difference: **PD splits across the time axis** (prefill happens first, decode later), so the cross-pool transfer is a one-time KV handoff. **AF splits within a single forward pass** (every layer crosses), so the cross-pool transfer is per-layer ongoing traffic.

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

Four pools instead of two. Each pool sized exactly for its bottleneck. This is the configuration MegaScale-Infer-style systems are pushing toward for large-scale MoE serving. Operationally heavy (more pools to scale and monitor), but the cost-per-throughput numbers in published benchmarks justify it for cluster scales > 256 GPUs.

---

## Limitations

- **Interconnect dependency.** Every layer round-trips activations. Below NVLink-class bandwidth (or 200+ Gbps RDMA), transfer cost eats the specialization savings.
- **Pipeline scheduling complexity.** To overlap activation transfer with compute, you need fine-grained PP between attention and FFN. Bubble management is non-trivial.
- **Heterogeneous hardware operational cost.** Running two different GPU types in production multiplies infra complexity (driver versions, NCCL topology, monitoring, failure-handling).
- **Mostly MoE-specific value.** Dense models gain less — the attention/FFN asymmetry is real but smaller than for MoE.
- **Tail-latency interaction.** Activation transfer adds a per-layer latency floor that the slowest pool dominates; if either pool stalls, the whole forward stalls.
- **Still early.** Outside of MegaScale-Infer, public AF deployments are rare. Many "AF-like" systems (DeepSeek-V3 inference, SGLang MoE) are really DP-attention + EP-MoE on uniform hardware — the *logical* structure but not the *physical* specialization.

---

## What this means

AF disaggregation is the natural endgame of the "use specialized hardware for the right thing" thesis that started with PD disaggregation. Two predictions worth tracking:

1. **MoE serving will pull AF into the mainstream.** As MoE models get sparser (DeepSeek-V3 at 256 experts top-8, future models pushing toward 1024+), the attention/FFN compute imbalance grows until ignoring it costs more than the operational complexity of separating pools.
2. **The "hardware tier" boundary will move.** Right now everyone treats H100 / H200 as one tier of GPU. As HBM and compute scaling decouple (HBM4 is bandwidth-heavy; tensor-core advances are FLOPs-heavy), the attention/FFN split will follow the hardware split, and AF will become the obvious deployment shape for new models.

What this is *not*: a free 2× speedup for any deployment. It's a cost-efficiency argument at scale, conditional on the right cluster topology and the right model architecture. Small dense models on single-node deployments get nothing.

---

## Related reading

- [[prefill-decode-disaggregation]] — Disaggregation across phases (prefill vs decode); the precursor pattern. Composes with AF.
- [[parallelism-strategies-deep-dive#11. DP Attention — Data-Parallel Attention for MoE Inference]] — DP attention is the parallelism shape that makes AF natural inside a single cluster.
- [[parallelism-strategies-deep-dive#13. DeepSeek-V3 Case Study]] — Production DP-attention + EP-MoE deployment; the logical AF structure on uniform hardware.
- [[kv-cache-optimization]] — KV cache compression reduces attention's memory-bandwidth cost and shifts the AF cost calculus.
- [[continuous-batching]] — Scheduler-layer load smoothing that runs *inside* each AF pool.

## References

- **MegaScale-Infer**: ByteDance (2024). *Attention/FFN-Disaggregated MoE Inference*. The explicit AF-disaggregation paper.
- **DeepSeek-V3 Technical Report** (2024). Production DP-attention + EP-MoE deployment that structurally resembles AF disaggregation on uniform hardware.
- **Mooncake**: Qin et al., *Mooncake: A KVCache-Centric Disaggregated Architecture for LLM Serving*, FAST 2025. Decouples KV storage from compute — adjacent to AF.
- **DistServe** (Zhong et al., OSDI 2024) and **Splitwise** (Patel et al., ISCA 2024): foundational PD-disaggregation systems whose architectural logic AF extends.
