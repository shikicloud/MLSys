---
title: "MORI — Citation"
type: paper-citation
created: 2026-06-08
---

# Idleness is Relative: Exploiting Tool-Call Idle Windows for Offloading in Agentic Systems with MORI

- **arXiv**: [2606.00866v1](https://arxiv.org/abs/2606.00866)
- **Code**: Not released as of June 2026
- **Authors**: Tian Xia¹, Hanchen Li¹, Zhifei Li², Xiaokun Chen³, Hao Kang⁴, Yifan Qiao¹, Yi Xu¹, Ion Stoica¹
- **Affiliations**: ¹UC Berkeley, ²Renmin University of China, ³Stanford University, ⁴Georgia Institute of Technology
- **Date**: 2026-05-30
- **Implementation**: ~3,300 lines Python on ThunderAgent + 500 lines on SGLang v0.5.10 HiCache
- **Wiki page**: [[mori]]

## Why this paper matters

**The direct successor to [[continuum|Continuum]]** by the same first author Hanchen Li. Where Continuum was single-tier (GPU HBM only) with TTL pinning, MORI is two-tier (GPU + CPU DRAM) with a continuous *idleness* metric that ranks all programs and dynamically partitions them across memory tiers. The conceptual progression: Continuum decides "pin or not"; MORI decides "which tier" along a relative-idleness spectrum. This is the new state of the art for agent serving memory management.

## Key technical contributions

1. **Idleness metric**: $\iota = T_{acting}^{(k)}/(T_{reasoning}^{(k)}+T_{acting}^{(k)})$ over k=5 inference/tool-call cycles; windowed average is both responsive to phase transitions and robust to outliers.
2. **Three-tier queue architecture**: GPU queue (HBM, busy programs), CPU queue (DRAM with cache affinity, idle programs), Waiting queue (KV discarded).
3. **Sticky rebalancing scheduling policy**: programs stay in tier until idleness mismatch + capacity boundary crossed; avoids PCIe churn from per-tick reshuffling.
4. **Typed offloading on inference engine**: scheduler propagates busy/idle/inactive labels; engine LRU uses type as priority sort key with reversed priorities between tiers.

## Headline numbers (Claude Code on SWE-bench Pro, 80 concurrent programs)

| Hardware | Model | TA+O baseline | MORI | Improvement |
| -------- | ----- | ------------: | ---: | ----------: |
| H200 80GB | Qwen-2.5 7B | 667 tokens/s | 853 tokens/s | +28% |
| B200 | Llama-3.1 70B | 124 tokens/s | **213 tokens/s** | **+71%** |
| H200 DP=3 | various | — | — | +54-79% throughput |
| TTFT (avg) | — | — | — | 18-43% lower |
| TTFT (peak) | — | — | — | up to 2.8× lower |
| GPU utilization | 59-76% | **99%+** | — | — |

## Tool-call duration variance from real Claude Code traces (n=16,886)

| Percentile | Duration |
| ---------: | -------: |
| P50 | 1,096 ms |
| P90 | 2,034 ms |
| P99 | 19,980 ms |
| P99.95 | 83,626 ms |

Three orders of magnitude — motivates continuous idleness over binary busy/idle.

## BibTeX

```bibtex
@article{xia2026mori,
  title={Idleness is Relative: Exploiting Tool-Call Idle Windows for Offloading in Agentic Systems with MORI},
  author={Xia, Tian and Li, Hanchen and Li, Zhifei and Chen, Xiaokun and Kang, Hao and Qiao, Yifan and Xu, Yi and Stoica, Ion},
  journal={arXiv preprint arXiv:2606.00866},
  year={2026}
}
```
