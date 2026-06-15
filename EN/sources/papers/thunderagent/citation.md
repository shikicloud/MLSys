---
title: "ThunderAgent — Citation"
type: paper-citation
created: 2026-06-15
---

# ThunderAgent: A Simple, Fast and Program-Aware Agentic Inference System

- **arXiv**: [2602.13692](https://arxiv.org/abs/2602.13692) (v1 2026-02-14; v2 2026-03-10)
- **Venue**: **ICML 2026 Spotlight (top 2.2%)**
- **Code**: [ThunderAgent-org/ThunderAgent](https://github.com/ThunderAgent-org/ThunderAgent) (MIT, vLLM + SGLang backends, OpenAI-compatible API). Original first-author repo: [HaoKang-Timmy/ThunderAgent](https://github.com/HaoKang-Timmy/ThunderAgent).
- **Authors**: Hao Kang\* (Georgia Tech), Ziyang Li\* (Individual Researcher), Xinyu Yang\* (CMU), Weili Xu\* (UIUC), Yinfang Chen (UIUC), Junxiong Wang (Together AI), Beidi Chen (CMU), Tushar Krishna (Georgia Tech), Chenfeng Xu (Together AI), Simran Arora (Together AI). \*Equal contribution.
- **Affiliations**: Georgia Tech, Carnegie Mellon University, UIUC, Together AI
- **Correspondence**: `hkang342@gatech.edu`
- **License**: MIT
- **Wiki page**: [[thunderagent]]

## Why this paper matters

**The first agent-serving system to abstract multi-turn workflows as first-class `agentic programs` and schedule at program granularity rather than per-request.** Contemporary competitor to [[continuum|Continuum]] (predictive TTL) with a reactive thrashing-monitor design; **beats Continuum on all 6 benchmarks** including the stochastic-tool ToolOrchestra workload where Continuum's TTL predictions degrade below vLLM. ThunderAgent is also the system layer that [[mori|MORI]]'s two-tier GPU+CPU idleness offloader builds on top of (~3,300 lines on ThunderAgent + 500 lines on SGLang HiCache).

## Key technical contributions

1. **Agentic program abstraction** — $P = \langle ID, c, \mathcal{T}, \mathcal{L}, \tau, s \rangle$ unifying KV cache footprint, tool environments, node placement, execution phase (Reasoning/Acting), and scheduling status (Active/Paused/Terminated).
2. **Periodic thrashing monitor** at $\Delta t = 5$ s evaluating $C_{\text{total}} < \sum_p c_p$ on each backend; Pause/Restore primitives prioritize by $1/c_P + \mathbb{1}(\tau)$ — *shortest-context-first* eviction, *Reasoning-first* admission.
3. **Time-decay function** $f(t) = 2^{-t}$ on Acting programs' token weight (proven optimal under memoryless tool durations); reactive counterpart to Continuum's predictive TTL.
4. **Lemma 4.1** — $\text{Cost}_{\text{recompute}} \propto c_i^2$ (quadratic in context length); justifies greedy shortest-first eviction.
5. **Global program-aware waiting queue** across DP nodes — replaces per-node KV-aware routing; paused KV is node-agnostic so any backend with capacity can restore; bounds $\text{Cost}_{\text{unused}} < c_{\min} \cdot \Delta t$.
6. **Hook-based garbage collection** for tool resources — lifecycle hooks teardown sandboxes / network ports on `Terminated` status; keeps active disk usage near-constant.
7. **Asynchronous environment preparation** — spin up Docker containers / dependency installs in parallel with LLM phase of high-priority restoring programs; hides 29.9-47.2 s setup latency.
8. **OpenAI-compatible API** with a single extra `program_id` parameter — trivial client integration.

## Headline numbers

- **Serving throughput**: **1.48–3.58× vs vLLM** across 6 benchmarks (SWEAgent, OpenHands, ToolOrchestra, ScienceAgent × GLM-4.6 / Qwen3-235B / Qwen3-8B).
- **vs Continuum**: 1.17–3.31× faster on all 6 benchmarks; **Continuum loses to vLLM (0.65×)** on ToolOrchestra-HLE stochastic-tool workload, ThunderAgent **wins** (1.48× vs vLLM).
- **RL rollout**: **1.79–3.92× improvement** over prior SOTA (vLLM + SGLang Gateway) on 2× 8×H100.
- **Disk memory**: up to **4.2× savings** — near-constant active usage vs linear-growth baselines that exceed 2 TB capacity by ~250 workflows.
- **KV cache hit rate**: ~100% on predictable-tool workloads (a,b,d,e) at 192 parallel workflows; Continuum drops from >90% to ~60% under high concurrency.
- **Memory imbalance**: baseline vLLM+SGLang has 51% peak imbalance across DP nodes over 90 min RL rollout; ThunderAgent bounds it by $c_{\min} \cdot \Delta t$.
- **KV cache thrashing impact in baselines**: 7.14× E2E latency blowup; tool environment preparation grows from 29.9 → 47.2 s as workflows scale 24 → 96.

## BibTeX

```bibtex
@inproceedings{kang2026thunderagent,
  title={ThunderAgent: A Simple, Fast and Program-Aware Agentic Inference System},
  author={Kang, Hao and Li, Ziyang and Yang, Xinyu and Xu, Weili and Chen, Yinfang and Wang, Junxiong and Chen, Beidi and Krishna, Tushar and Xu, Chenfeng and Arora, Simran},
  booktitle={Proceedings of the 43rd International Conference on Machine Learning},
  note={Spotlight},
  year={2026}
}
```
