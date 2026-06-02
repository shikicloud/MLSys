---
title: "ROSE — Citation"
type: paper-citation
created: 2026-06-02
---

# ROSE: Rollout On Serving GPUs via Cooperative Elasticity for Agentic RL

- **arXiv**: [2605.06534](https://arxiv.org/abs/2605.06534)
- **Code**: Not yet released
- **Authors**: Wei Gao\*†, Yuheng Zhao\*†, Dilxat Muhtar‡, Dakai An†, Xuchun Shang‡, Tianyuan Wu†, Lunxi Cao†, Shaopan Xiong‡, Weixun Wang‡, Ju Huang†, Teng Ma‡, Siran Yang‡, Jiamang Wang‡, Lin Qu‡, Bo Zheng‡, Wei Wang† (\*equal contribution; †HKUST; ‡Alibaba Group)
- **Date**: 2026-05-07 (v1); 2026-05-20 (v2)
- **Implementation**: ~5k lines Python atop ROLL + vLLM 0.10.0 + Megatron-LM + Mooncake v0.3.8 + Ray
- **License**: CC-BY 4.0
- **Wiki page**: [[rose]]

## Why this paper matters

First systems paper to harvest **production serving GPU capacity** (typically idle at 18.9% compute / 14.3% memory utilization) for agentic RL rollouts via cooperative elasticity. Solves three previously-coupled problems: (1) preserving serving SLOs under co-located rollout bursts, (2) cross-cluster weight sync at 10-200 Gbps Ethernet, (3) elastic rollout dispatch with cache locality. Allocation overhead is **50-80× better** than spot/serverless RL alternatives.

## Key technical contributions

1. **SLO-Safe Co-Serving Executor** with VMM-based cross-model KVC sharing, preemptive memory sharing policy, and dual-SLO admission control (TTFT + TPOT slack).
2. **Cross-Cluster Weight Transfer Engine** with async Mooncake Store relay, shard-aware mapping for heterogeneous TP×PP, and sparsity-aware COO compression exploiting **>95% sparsity in RL weight deltas $\Delta W_t$** (a novel empirical observation).
3. **Elastic Rollout Scheduler** with turn-wise routing (not whole-trajectory) and cache-affinity placement.
4. **End-to-end throughput improvements**: 1.31-1.46× on GRPO, 1.42-3.31× on DAPO, 1.44-2.69× on AReaL — with zero P99 SLO violations and 0.3-0.4% allocation overhead.

## Headline numbers

| Comparison | Model | Throughput improvement |
| ---------- | ----- | ---------------------- |
| vs ROLL-GRPO (fixed) | Qwen3-8B FrozenLake | 1.31× (max 2.16×) |
| vs ROLL-GRPO | Qwen3-32B ALFWorld | 1.46× (max 1.76×) |
| vs ROLL-DAPO | Qwen3-8B | 1.42× (max 4.82×) |
| vs ROLL-DAPO | Qwen3-32B | 3.31× (max 4.39×) |
| vs AReaL (async) | Qwen3-8B/32B | 1.44× / 2.69× |
| vs RLBoost+ (spot) | Rollout time | 1.20-1.26× faster |
| Allocation overhead | ROSE vs λRL | 0.3-0.4% vs 15.1-26.1% |

## BibTeX

```bibtex
@article{gao2026rose,
  title={ROSE: Rollout On Serving GPUs via Cooperative Elasticity for Agentic RL},
  author={Gao, Wei and Zhao, Yuheng and Muhtar, Dilxat and An, Dakai and Shang, Xuchun and Wu, Tianyuan and Cao, Lunxi and Xiong, Shaopan and Wang, Weixun and Huang, Ju and Ma, Teng and Yang, Siran and Wang, Jiamang and Qu, Lin and Zheng, Bo and Wang, Wei},
  journal={arXiv preprint arXiv:2605.06534},
  year={2026}
}
```
