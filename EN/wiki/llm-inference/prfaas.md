---
title: "PrfaaS: KVCache of Next-Generation Models Could Go Cross-Datacenter"
category: llm-inference
tags: [prfaas, pd-disaggregation, cross-datacenter, kvcache, hybrid-attention, mooncake, moonshot, paper-review]
created: 2026-05-22
updated: 2026-05-22
status: mature
paper: arXiv:2604.15039
code: not-released
---

# PrfaaS: KVCache of Next-Generation Models Could Go Cross-Datacenter

> [!info] Paper metadata
> - **Paper**: [arXiv:2604.15039](https://arxiv.org/abs/2604.15039) — *Prefill-as-a-Service: KVCache of Next-Generation Models Could Go Cross-Datacenter*, v1 2026-04-16, v2 2026-04-22
> - **Authors**: Ruoyu Qin¹², Weiran He¹, Yaoyu Wang¹, Zheming Li¹, Xinran Xu¹, Yongwei Wu², Weimin Zheng², Mingxing Zhang²
> - **Affiliations**: ¹Moonshot AI, ²Tsinghua University
> - **Corresponding**: Mingxing Zhang (zhang_mingxing@mail.tsinghua.edu.cn) — same lead as [[prefill-decode-disaggregation|Mooncake]]
> - **Code**: not released; references "in-house vLLM" forks of the [[vllm]] hybrid KVCache manager

---

## Summary (read this if you have 2 minutes)

**What it is.** PrfaaS is a **cross-datacenter** PD-disaggregation architecture: long-context prefill runs on a compute-dense GPU cluster (e.g. H200, Rubin CPX), the resulting KVCache flows over **commodity Ethernet** to a separate local PD cluster on bandwidth-optimized GPUs (e.g. H20, LPU) where decode happens. Same Moonshot/Tsinghua team behind [[prefill-decode-disaggregation|Mooncake]], pushing PD disaggregation one network tier outward — from "tightly-coupled RDMA inside one DC" to "loosely-coupled Ethernet across DCs."

**The one idea.** Hybrid-attention models (Kimi Linear, MiMo-V2-Flash, Qwen3.5-397B, Ring-2.5-1T) cut KVCache size by ~13× vs dense GQA, which shifts the deployable network boundary of PD disaggregation from RDMA-class fabric to commodity Ethernet. But "smaller KVCache" alone isn't sufficient — production workloads are bursty, length distributions are skewed, prefix-cache locality is uneven, inter-DC bandwidth fluctuates. PrfaaS pairs the model-side KV reduction with three system-side pieces: (1) **length-threshold selective offloading** — only requests with uncached prefix $l > t$ go cross-DC, short requests stay local; (2) **hybrid prefix cache pool** — separate management for full-attention block-level KV and linear-attention request-level recurrent state, unified block pool, prefix-cache vs transfer-cache categories; (3) **dual-timescale scheduling** — short-term bandwidth+cache-aware routing reacts to congestion, long-term re-allocation adjusts the PD-cluster prefill/decode ratio.

**Headline result.** Case study on an internal 1T Kimi-Linear-style hybrid model (KDA:MLA 3:1), 32 H200 PrfaaS + 64 H20 local PD vs 96 H20 homogeneous baseline:

| Metric | Homogeneous PD (96 H20) | Naive hetero (no scheduling) | **PrfaaS-PD** |
| ------ | ----------------------: | ---------------------------: | ------------: |
| $\Lambda_{\max}$ (req/s) | 2.11 | 2.45 | **3.24** |
| Speedup | 1.00× | 1.16× | **1.54×** |
| Mean / P90 TTFT (s) | 4.44 / 9.73 | 1.74 / 3.51 | 2.22 / **3.51** |
| Cross-DC bandwidth used | — | — | **13 Gbps (13 % of 100 Gbps link)** |

**+54 % throughput, –64 % P90 TTFT** vs homogeneous; **~15 % throughput gain at equal cost** when sizing the heterogeneous deployment to match the H20-baseline GPU budget. The naive-heterogeneous comparison is the *load-bearing* one: dropping the scheduler costs ~25 % of the throughput gain — selective offloading + bandwidth-aware routing is doing real work, not just the hardware mix.

**Why it matters.**

- **Closes the "heterogeneous PD" gap.** Vendors are shipping phase-specialized chips ([NVIDIA Rubin CPX](https://www.nvidia.com/en-us/data-center/products/rubin-cpx/) for prefill, [Groq LPU](https://wow.groq.com/lpu-inference-engine/) and Taalas HC1 for decode). Until now the lack of RDMA-class fabric between unlike accelerators meant you couldn't deploy them together cleanly. PrfaaS makes commodity Ethernet good enough.
- **Mooncake → PrfaaS is a coherent thesis.** Treating KVCache as a first-class systems resource started in 2024 with [[prefill-decode-disaggregation|Mooncake]]; PrfaaS extends the same idea past the DC boundary. The cross-cluster KV manager is a natural sequel to Mooncake's global KV pool.
- **First public deployment of hybrid-attention-aware PD scheduling.** The hybrid prefix cache pool (linear-attention recurrent states + full-attention block KV in a unified manager) is what makes the new generation of hybrid models PD-serveable at all.
- **Open systems agenda for 2026-27.** Cross-DC KV transfer becomes a real workload; expect vLLM / SGLang / Dynamo to add cross-cluster KV connectors; expect inter-DC network design (10G→100G→400G Ethernet) to start optimizing for KV traffic patterns specifically.

---

# Depth (drill-down starts here)

## Background: the bandwidth wall in conventional PD

PD disaggregation (Splitwise, DistServe, [[prefill-decode-disaggregation|Mooncake]]) cleanly separates prefill (compute-bound, $O(S^2)$ FLOPs) from decode (memory-bandwidth-bound, $O(S)$ DRAM reads per step). That separation enables phase-specific optimization, but the export of KVCache from prefill to decode is now a *cross-node transport problem*. In intra-DC deployments, NVLink + RDMA absorb that traffic invisibly. Once you want to push prefill and decode across DCs, the KVCache transfer becomes the binding constraint.

The paper formalizes this as the **KV throughput** of a model instance:

$$
\Phi_{kv}(l) = \frac{S_{kv}(l)}{T_{\text{prefill}}(l)}
$$

For an $N$-GPU cluster running instances at parallelism $P$, the minimum egress bandwidth to avoid GPU idling is:

$$
B_{\text{out}} = \frac{N}{P} \cdot \frac{\mathbb{E}[S_{kv}]}{\mathbb{E}[T_{\text{prefill}}]} \approx \frac{N}{P} \cdot \Phi_{kv}(L_{\text{avg}})
$$

For a 512-GPU prefill cluster at $L_{\text{avg}}=32K$ on dense models (MiniMax-M2.5 with GQA), this means ~**3.8 Tbps** of egress bandwidth — utterly impractical across DCs. Qwen3-235B needs ~**2.1 Tbps**. Conventional PD is therefore locked inside a single RDMA-class island.

### How hybrid attention changes the picture

The shift is on the model side, not the systems side. Hybrid stacks interleave a small number of full-attention layers with a larger number of linear-complexity layers (linear attention, SWA, etc.). Only the full-attention layers produce sequence-length-growing KVCache; the linear layers maintain fixed-size recurrent state.

KV-throughput table (8×H200, SGLang v0.5.9, from paper Table 3):

| Seq len | **Kimi Linear** | **MiMo-V2-Flash** | **Qwen3.5-397B** | **Ring-2.5-1T** | MiniMax-M2.5 (dense) | Qwen3-235B (dense) |
| ------- | --------------: | ----------------: | ---------------: | --------------: | -------------------: | -----------------: |
| 1K | 1.19 Gbps | 0.82 | 4.13 | 7.27 | 4.94 | 4.12 |
| 8K | 2.29 | 2.85 | 6.28 | 4.47 | 32.87 | 22.42 |
| 32K | 3.87 | 4.66 | 8.25 | 2.59 | **59.93** | **33.35** |
| 128K | 4.88 | 4.71 | 7.47 | 1.46 | 47.82 | 21.50 |

At 32K tokens, **MiMo-V2-Flash 4.66 Gbps vs MiniMax-M2.5 59.93 Gbps — a 13× reduction**. Ring-2.5-1T at 128K drops to 1.46 Gbps. The paper notes Ring-2.5-1T gets ~4.5× from MLA over GQA × ~8× from the 7:1 hybrid ratio = ~**36× overall KV memory saving**.

Re-running the 512-GPU sizing: hybrid Ring-2.5-1T needs only **170 Gbps**; routing only requests with $l>128K$ to PrfaaS drops it below **100 Gbps**. Even a 10,000-GPU DC's aggregate KV egress sits at ~1.8 Tbps — comfortably inside modern inter-DC link capacity.

> [!note] The two-row attention typology that anchors the paper (Table 2)
> | Mechanism | Prefill latency | KV throughput |
> | --------- | --------------- | -------------- |
> | GQA | High | High |
> | MLA | High | Low |
> | Sparse attention | Low | High |
> | SWA | Low | Low |
> | Linear attention | Low | Low |
>
> Hybrid stacks pick "Low / Low" — what lets cross-DC PrfaaS work. Sparse attention is conspicuously the wrong tradeoff for this architecture (compute cheap, network still expensive).

### Why heterogeneous hardware needs this

Phase-specialized chips already exist:
- **Prefill side**: NVIDIA Rubin CPX (compute-dense), H200 (used in the case study)
- **Decode side**: Groq LPU, Taalas HC1, H20 (used in the case study) — extreme memory bandwidth

But high-performance interconnects are tightly coupled to chip form factors and deployment environments. Forcing unlike accelerators behind the same RDMA island requires bespoke engineering AND inherits a fixed prefill-to-decode hardware ratio — fatal in production where the mix shifts continuously. PrfaaS sidesteps this by accepting that prefill and decode can live in *different DCs entirely*.

## The PrfaaS-PD architecture

![PrfaaS-PD deployment topology (paper Fig. 3)](EN/wiki/llm-inference/prfaas-figs/prfaas-topology.png)

Three subsystems:

| Subsystem | Components |
| --------- | ---------- |
| **Compute** | PrfaaS clusters (homogeneous, compute-dense, prefill-only) + Local PD clusters (homogeneous, bandwidth-optimized, conventional PD) |
| **Network** | Intra-cluster RDMA (latency-sensitive collectives + PD KV transfer) + Inter-cluster Ethernet via VPC peering or dedicated lines (cross-DC KV transfer) |
| **Storage** | Per-cluster distributed hybrid prefix cache pool + Global KVCache manager that tracks metadata across all clusters |

A global scheduler sits above this and routes requests by length / cache locality / network state.

### Component 1 — Length-threshold routing

The core mechanism that makes selective offloading sound:

```
Request arrives with incremental (uncached) prefill length l:
  if l > t:  route to PrfaaS cluster        (long-context, compute-bound)
  if l ≤ t:  route to local PD-P            (short, memory/comm-bound)
```

The intuition: short prefills are typically memory-bound or communication-bound at the local-PD scale, so PrfaaS's high-arithmetic-throughput accelerators are wasted on them. Only long prefills truly exploit a compute-dense H200/Rubin-CPX cluster. PrfaaS itself functions as a **stateless KVCache producer** whose throughput equals $\min(\text{prefill compute}, \text{egress bandwidth})$.

For prefix-cached requests (common in agentic workloads), the global KVCache manager tracks where every cached entry lives, so only the *incremental* portion crosses clusters. This is what makes the bandwidth calculation work in practice — agentic traffic is mostly incremental prefills.

### Component 2 — Hybrid prefix cache pool

Hybrid models break conventional KV cache assumptions:

| Attention type | KV state shape | Reuse semantics |
| -------------- | -------------- | --------------- |
| Linear / SWA | **Request-level** recurrent state, **fixed size** | Exact-match only (length must match exactly) |
| Full attention (MLA, GQA) | **Block-level** KV, **grows with seq len** | Standard prefix matching, partial reuse OK |

PrfaaS builds on the [vLLM hybrid KVCache manager (PR #29427)](https://github.com/vllm-project/vllm/pull/29427) but adapts it for cross-cluster transport. The design:

- **Separate KVCache groups** for linear states and full-attention KV, but **aligned block sizes** so both can allocate/free from a **shared block pool**
- Cache blocks split into two categories:
  - **Prefix-cache blocks** — fully populated, reusable across requests, intra-cluster only, block-aligned
  - **Transfer-cache blocks** — produced at the tail of a prefill request for PD-disaggregated transfer, **any length**, **discarded after transfer**

When a request arrives, the global KVCache manager computes prefix-match info for every cluster, and the router picks the prefill cluster + cache-affine node within it. Cache rebalancing happens between clusters when inter-cluster bandwidth allows.

### Component 3 — Dual-timescale scheduling

This is where the system contribution lives. Just having selective offloading doesn't survive real traffic — congestion builds up, queues swell, P90 TTFT explodes. The scheduler operates on two timescales:

**Short-term: bandwidth- and cache-aware routing**. The scheduler monitors PrfaaS egress utilization and queue depth continuously. When utilization approaches the bandwidth-imposed ceiling $B_{\text{out}}/S_{kv}(l_{\text{long}})$, it raises the routing threshold $t$ to send fewer (and longer) requests cross-DC, reducing per-request bandwidth demand.

For prefix-cache-hit requests, routing depends on whether bandwidth or compute is binding:
- **Bandwidth-scarce regime** — evaluate caches per cluster independently: `if l_total - l_pd ≤ t → PD-P; else → PrfaaS`
- **Bandwidth-abundant regime** — consider best cache across all clusters: `l_prefix = max(l_prfaas, l_pd); if l_total - l_prefix ≤ t → PD-P; else → PrfaaS`. If the best-cache cluster differs from the compute cluster, do a cross-cluster cache transfer first.

**Long-term: traffic-driven allocation re-optimization**. Traffic mix and volume shift on hour/day scales. When $\Theta_{\text{prfaas}} + \Theta_{\text{pd-p}} \ll \Theta_{\text{pd-d}}$, prefill is the bottleneck; reverse means decode is. The scheduler periodically re-evaluates load balance and **converts nodes between prefill and decode roles within the PD cluster** ($N_p \leftrightarrow N_d$), re-deriving the optimal threshold $t$ afterward. PrfaaS-cluster GPUs stay prefill-only because they're compute-dense and decode would waste them.

### The throughput model

The case-study results all come from this analytical model (paper §3.4). Three stages, the slowest sets $\Lambda_{\max}$:

$$
\Theta_{\text{prfaas}} = \min\!\left(\frac{N_{\text{prfaas}}}{T_{\text{prefill}}(l_{\text{long}})}, \;\frac{B_{\text{out}}}{S_{kv}(l_{\text{long}})}\right)
$$

$$
\Theta_{\text{pd-p}} = \frac{N_p}{T_{\text{prefill}}(l_{\text{short}})}, \qquad \Theta_{\text{pd-d}} = \frac{N_d \cdot \text{BS}_{\max}}{T_{\text{decode}} \cdot L_{\text{out}}}
$$

$$
\Lambda_{\max} = \min\!\left(\frac{\Theta_{\text{prfaas}}}{p}, \;\frac{\Theta_{\text{pd-p}}}{1-p}, \;\Theta_{\text{pd-d}}\right)
$$

Where $p = \Pr(L > t)$ is the PrfaaS-routed fraction. The two optimization knobs are $t$ (routing threshold → determines $p$, $l_{\text{long}}$, $l_{\text{short}}$) and $N_p/N_d$ (PD-cluster prefill/decode split). Optimality conditions:

$$
\frac{\Theta_{\text{prfaas}}}{p} = \frac{\Theta_{\text{pd-p}}}{1-p} \quad\text{(threshold balance)}
$$

$$
\Theta_{\text{prfaas}} + \Theta_{\text{pd-p}} = \Theta_{\text{pd-d}} \quad\text{(producer-consumer balance)}
$$

These are two equations in two unknowns ($t$, $N_p/N_d$) — solved by 2D grid search over the profiling-derived $T_{\text{prefill}}(l)$ and $S_{kv}(l)$ curves.

> [!example] The intuition behind the threshold optimum
> Increasing $t$ restricts PrfaaS to longer requests. For those, $T_{\text{prefill}}(l)$ grows near-quadratically (still full-attention-dominated even in hybrid models) while $S_{kv}(l)$ grows linearly — so $\Phi_{kv}$ goes *down* with $l$. Larger $t$ means less bandwidth pressure per offloaded request, more headroom under $B_{\text{out}}$. Decreasing $t$ floods PrfaaS with shorter requests whose high $\Phi_{kv}$ triggers the bandwidth ceiling. The optimum sits where PrfaaS and PD-P saturate simultaneously.

### Network engineering for sustained throughput

Even with reduced bandwidth demand, bursty traffic can cause transient congestion on Ethernet. The paper's network-side design:

- **Layer-wise prefill pipelining** — overlap KV generation with transmission so each layer's KV starts streaming as soon as it's produced, not at the end of prefill
- **Multi-connection TCP transport** — multiple parallel connections to fully utilize the available Ethernet bandwidth and tolerate per-flow congestion
- **Congestion monitoring integrated with the scheduler** — detect loss/retransmission signals early, feed them back into the short-term router so it can throttle PrfaaS routing before queue buildup

Note: this is **TCP, not RDMA**. That's the systems contribution — letting commodity infrastructure carry KV traffic.

## Headline evidence — the 1T-hybrid case study

**Setup.** Internal 1T-parameter Kimi-Linear-style hybrid (KDA:MLA 3:1), 8 GPUs/instance.

| Cluster | GPUs | Role | Network |
| ------- | ---- | ---- | ------- |
| PrfaaS | 32× H200 | Long-context prefill ($l > t$) | Cross-DC: 100 Gbps Ethernet |
| Local PD | 64× H20 | Short prefill + all decode | Intra-cluster: 800 Gbps RDMA |
| **Baseline** | **96× H20** | **Homogeneous PD** | **Intra-cluster RDMA only** |

Model profile (8×H200, in-house vLLM, paper Table 5):

| Seq len | KV size | Prefill latency | $\Phi_{kv}$ |
| ------- | ------: | --------------: | ----------: |
| 1K | 190.8 MiB | 0.44 s | 3.61 Gbps |
| 8K | 308.9 MiB | 0.72 s | 3.59 Gbps |
| 32K | 701.3 MiB | 1.84 s | 3.19 Gbps |
| 128K | 2316.3 MiB | 7.40 s | 2.62 Gbps |

Workload: input length log-normal ($\mu=9.90$, $\sigma=1.00$, truncated $[128, 128K]$, mean ~27K tokens), output length 1024, SLO 40 tok/s decode.

**Optimization result.** Grid search yields:

| Parameter | Value |
| --------- | ----: |
| Routing threshold $t$ | **19.4K tokens** |
| PrfaaS instances $N_{\text{prfaas}}$ | 4 |
| PD-P / PD-D instances | 3 / 5 |
| PrfaaS-routed fraction $p$ | **49.6 %** |
| $\mathbb{E}[L \mid L > t]$ | ~44K tokens |

**Three-way comparison** (paper Table 6):

| Metric | Homogeneous PD | Naive heterogeneous | **PrfaaS-PD** |
| ------ | -------------: | ------------------: | ------------: |
| Threshold $t$ | — | — | **19.4K** |
| Layout ($N_{\text{prfaas}}/N_p/N_d$) | —/9/3 | 4/—/8 | **4/3/5** |
| Mean / P90 TTFT (s) | 4.44 / 9.73 | 1.74 / 3.51 | **2.22 / 3.51** |
| $\Theta_{\text{prfaas}}/\Theta_{\text{pd-p}}/\Theta_{\text{pd-d}}$ (req/s) | —/2.11/2.35 | 2.45/—/6.25 | **1.61/1.64/3.91** |
| $\Lambda_{\max}$ (req/s) | 2.11 | 2.45 | **3.24** |
| Speedup | 1.00× | 1.16× | **1.54×** |

> [!success] The 13 Gbps headline
> Cross-DC egress at the optimal operating point is **13 Gbps — only 13 % of the 100 Gbps Ethernet link**. Plenty of headroom for traffic bursts. For comparison, a dense MiniMax-M2.5 deployment at the same scale would need ~60 Gbps just for steady-state traffic and would blow through any commodity link under burst.

> [!important] Naive heterogeneous is the right comparison
> The 1.54× vs homogeneous tells you "use H200+H20 with PrfaaS scheduling". The 1.54×/1.16× = **1.33× ratio over naive heterogeneous** tells you *the scheduler itself contributes ~25 % of the throughput gain*. The architecture isn't load-bearing without selective offloading + length-threshold routing.

### Why TTFT drops 64 % at P90

Two effects compound:

1. **Long-request isolation.** In the homogeneous baseline, long and short requests share the same prefill pool. Long requests inflate queueing delays for *everyone* — head-of-line blocking on the prefill side. In PrfaaS-PD, long requests are diverted to a separate cluster, so short-request TTFT stays low.
2. **Faster prefill compute.** H200 is meaningfully faster than H20 for long-context prefill; even after eating cross-cluster transfer latency, the modeled long-request prefill finishes faster on PrfaaS than on the H20 baseline.

Mean TTFT drops 50 % (2.22 vs 4.44 s); P90 drops 64 % (3.51 vs 9.73 s). The P90 gain is bigger because the homogeneous baseline's tail was dominated by the long-request queuing pathology — exactly what PrfaaS eliminates.

## Strengths and limitations

Two genuine strengths: (1) **first systems paper to operationalize cross-DC PD disaggregation** with a coherent end-to-end design — not just a "we should do this" essay but with a working throughput model, a scheduler, a cache architecture, and case-study numbers; (2) **the model-systems co-design framing is honest** — the paper repeatedly says "reduced KVCache size alone is not sufficient" and earns that by showing the naive-heterogeneous gap. Too many "model X enables system Y" papers fall down on the latter half; this one doesn't.

Where I'd push back:

- **It's a case study, not a deployment.** All numbers come from feeding profiling data ($T_{\text{prefill}}(l)$, $S_{kv}(l)$, $T_{\text{decode}}$) into a steady-state analytical model. There's no live 1T-model production run with real burst patterns, no measured TTFT under failure modes, no cross-DC link wobble experiments. The 13 Gbps / 13 % headroom claim is *modeled* under a log-normal workload assumption.
- **No reported source code.** Acknowledged in the paper as "in-house vLLM"; the hybrid KV cache pool builds on vLLM PR #29427 but the cross-cluster glue isn't open-sourced. Hard to validate the layer-wise pipelining + multi-connection TCP claims without running it.
- **Output length fixed at 1024.** Long-output reasoning workloads (DeepSeek-R1, o3 / o4 style) often emit 8K-32K tokens. Decode-stage throughput scales as $N_d \cdot \text{BS}_{\max}/(T_{\text{decode}} \cdot L_{\text{out}})$, so longer outputs shift the producer-consumer balance toward needing *more* PD-D, which changes the optimal allocation. The case study doesn't sweep $L_{\text{out}}$.
- **Single workload distribution.** One log-normal $(\mu=9.90, \sigma=1.00)$. Real production traffic is multi-modal (chat + long-context + agentic + RAG), and the optimal $t$ depends on the joint distribution. The dual-timescale scheduler is meant to handle this, but no experiment shows it adapting to a distribution shift.
- **Network failure mode untreated.** Inter-DC Ethernet drops are *normal* (bandwidth fluctuates, links flap). What happens when a chunk of in-flight KVCache is lost? Is there retransmission, or does the request just fail? The "congestion monitoring integrated with scheduler" line implies degraded routing but doesn't define a per-request fallback.
- **Privacy / multi-tenancy not discussed.** KVCache contains derived information from the request prompt. Shipping it across DCs over commodity Ethernet — even encrypted — has compliance and information-flow implications (GDPR, residency requirements). The paper is silent. For a production deployment this is the first question regulators ask.
- **Doesn't help dense GQA models at all.** This is explicit but worth restating: PrfaaS only works because hybrid attention dropped $\Phi_{kv}$ by an order of magnitude. Dense GQA models (Llama-3, Qwen3-235B) still need ~30 Gbps per instance — across 64 instances that's 2 Tbps, well outside the design range. The whole architecture is contingent on the hybrid-attention industry trend.
- **The 1T model is internal and undisclosed.** Following Kimi Linear architecturally but not publicly released. Reproducibility outside Moonshot is limited to "go test it on Kimi Linear's open release" (which is smaller than 1T).
- **No comparison to [[af-disaggregation|AF disaggregation]].** Both PD and AF are operator-vs-phase splitting strategies; both deal with cross-network state transfer. The paper doesn't position PrfaaS against the AF-disaggregation work (MegaScale-Infer / DP-attention+EP-MoE) that ships in production today. A reader new to disaggregation would want to know "PD-vs-AF in cross-DC settings."

> [!warning] What the paper does NOT claim (despite the title)
> "Could Go Cross-Datacenter" is honest — the paper shows feasibility under realistic assumptions, not that anyone is yet running cross-DC PD in production. Moonshot's actual Kimi-2 serving stack uses intra-DC Mooncake; PrfaaS-PD is a forward-looking design for future Rubin-CPX-class hardware deployments. Treating this as "what production is doing today" overshoots.

## What this means

The big systems trend: **KVCache is increasingly the resource you design around**. Mooncake made KVCache a first-class systems object inside one cluster. PrfaaS extends it across cluster boundaries. The next steps are obvious:

1. **Cross-DC KV connector standardization.** Expect vLLM / SGLang / Dynamo to add cross-cluster KV transport adapters in 2026 H2 — pluggable TCP / RDMA-over-DCI / dedicated-line variants. The interfaces will resemble Mooncake's `Transfer Engine` extended for WAN. There will be a brief period of competing protocols (NIXL, Mooncake, KVCache-over-gRPC).
2. **KV-aware DC placement decisions.** Cloud providers will start advertising "Rubin CPX DCs" and "LPU DCs" with documented inter-DC bandwidth guarantees. Prefill-heavy workloads (long-document summarization, code understanding) will route differently from decode-heavy workloads (agentic chains, reasoning). PrfaaS sets the template for what those routing decisions look like.
3. **Hybrid attention as the cross-DC requirement.** Dense-GQA-only model families will be unable to span DCs. That's a meaningful competitive disadvantage for model designers — expect Llama-5 / Qwen-4 / etc to ship with some hybrid-attention variant. Linear attention + SWA become not just "long-context efficiency" but "deployment-architecture-enabling."
4. **Phase-specialized DCs as the natural unit.** Today a DC is "filled with GPUs". Tomorrow it might be "this DC is prefill-optimized — compute-dense H200/Rubin, modest networking, used by every customer's long-context jobs"; "that DC is decode-optimized — bandwidth-monster GPUs, larger memory pool, used by reasoning workloads." PrfaaS is the system architecture that lets a workload span both.

What this is *not* — and the paper is honest about this in §5: a fully-general solution to inter-DC LLM serving. It's targeted at the long-context prefill bottleneck for hybrid-attention models, which is one slice (admittedly an important one) of the broader serving problem.

## Source code & reproduction

No public release as of the v2 preprint. The paper builds on:

| Component | Reference |
| --------- | --------- |
| Hybrid KVCache manager (single-cluster) | [vLLM PR #29427](https://github.com/vllm-project/vllm/pull/29427) — the basis for the per-cluster hybrid prefix pool |
| Mooncake KVCache pool | [Mooncake repo](https://github.com/kvcache-ai/Mooncake) — the single-cluster precursor; the global KVCache manager generalizes Mooncake's metadata layer across clusters |
| SGLang for $\Phi_{kv}$ benchmarks (Table 3) | [SGLang v0.5.9](https://github.com/sgl-project/sglang) |

Pseudocode of the dual-timescale scheduler (extracted from §3.4.3):

```python
class PrfaaSScheduler:
    def __init__(self):
        self.t = initial_threshold       # from grid-search optimum
        self.Np, self.Nd = init_alloc    # PD cluster prefill/decode split

    def route(self, request):
        # Short-term: bandwidth + cache aware
        l_total = request.input_length
        l_pd    = prefix_cache_hit(request, cluster="local-pd")
        l_prfaas = prefix_cache_hit(request, cluster="prfaas")

        if egress_util() > BANDWIDTH_CEILING:
            # Bandwidth-scarce: caches evaluated per-cluster
            incremental = l_total - l_pd
            return "PD-P" if incremental <= self.t else "PrfaaS"
        else:
            # Bandwidth-abundant: cross-cluster cache transfer OK
            l_best = max(l_pd, l_prfaas)
            incremental = l_total - l_best
            target = "PD-P" if incremental <= self.t else "PrfaaS"
            if l_best == l_prfaas and target == "PD-P":
                transfer_cache_to_pd(request)
            return target

    def periodic_reoptimize(self):
        # Long-term: re-balance N_p / N_d, re-grid-search t
        profile = collect_recent_profile()
        if Θ_prfaas() + Θ_pd_p() < Θ_pd_d():    # prefill bottleneck
            self.Np += 1; self.Nd -= 1
        elif Θ_prfaas() + Θ_pd_p() > Θ_pd_d():  # decode bottleneck
            self.Np -= 1; self.Nd += 1
        self.t, _ = grid_search_optimum(profile, self.Np, self.Nd)
```

To reproduce the case-study numbers, you need: (1) a hybrid-attention model close to Kimi Linear or Ring-2.5-1T, (2) profiling data $T_{\text{prefill}}(l)$ and $S_{kv}(l)$ across $l \in \{1K, 8K, 32K, 128K\}$ on H200, (3) the analytical model in §3.4 plus the workload assumption. The paper provides enough numbers to recompute Table 6 by hand from Table 5.

## Related reading

- [[prefill-decode-disaggregation]] — The single-cluster PD foundation; PrfaaS extends this across DC boundaries. Mooncake is the direct precursor and was co-authored by the same Tsinghua lead (Mingxing Zhang).
- [[af-disaggregation]] — The sibling disaggregation strategy: PD splits *phases* (prefill vs decode) along time; AF splits *operators* (attention vs FFN) within each forward pass. PrfaaS pushes PD across DCs; AF lives inside one cluster but spreads work across heterogeneous chips per layer. Composable in principle.
- [[paged-attention]] — Paged KV cache management; vLLM's hybrid KVCache manager (used as PrfaaS's storage layer) inherits from PagedAttention.
- [[kv-cache-optimization]] — H2O / KIVI / KVQuant family for shrinking KV beyond architectural means; cited in PrfaaS §5 as complementary.
- [[vllm]] — The serving framework PrfaaS extends.
- [[sglang]] — Used for $\Phi_{kv}$ measurements in Table 3.
- [[long-context-serving]] — The workload class where PrfaaS provides the largest gains.
- [[parallelism-strategies-deep-dive#7. CP — Context Parallelism]] — Context parallelism is the *intra-cluster* answer to long-context prefill; PrfaaS is the *inter-cluster* answer. They can compose (CP inside the PrfaaS cluster, PrfaaS across DCs).

## References

- Paper: Qin et al., *Prefill-as-a-Service: KVCache of Next-Generation Models Could Go Cross-Datacenter*, 2026-04-16. [arXiv:2604.15039](https://arxiv.org/abs/2604.15039)
- **Mooncake** (the precursor): Qin et al., FAST 2025. [arXiv:2407.00079](https://arxiv.org/abs/2407.00079)
- **Splitwise**: Patel et al., MICRO 2024. [arXiv:2311.18677](https://arxiv.org/abs/2311.18677)
- **DistServe**: Zhong et al., OSDI 2024. [arXiv:2401.09670](https://arxiv.org/abs/2401.09670)
- **NVIDIA Rubin CPX** (the prefill-specialized hardware target): [nvidia.com/en-us/data-center/products/rubin-cpx](https://www.nvidia.com/en-us/data-center/products/rubin-cpx/)
- **Groq LPU** (the decode-specialized counterpart): [wow.groq.com/lpu-inference-engine](https://wow.groq.com/lpu-inference-engine/)
- **Kimi Linear** (the architecture family the 1T model follows): [arXiv:2511.06257](https://arxiv.org/abs/2511.06257)
- **vLLM hybrid KVCache manager**: [github.com/vllm-project/vllm/pull/29427](https://github.com/vllm-project/vllm/pull/29427)
- KV-compression cited as complementary: **H2O** ([arXiv:2306.14048](https://arxiv.org/abs/2306.14048)), **KIVI** ([arXiv:2402.02750](https://arxiv.org/abs/2402.02750)), **KVQuant** ([arXiv:2401.18079](https://arxiv.org/abs/2401.18079))
- KV-reuse cited as complementary: **CacheGen** ([arXiv:2310.07240](https://arxiv.org/abs/2310.07240)), **CacheBlend** ([arXiv:2405.16444](https://arxiv.org/abs/2405.16444))
