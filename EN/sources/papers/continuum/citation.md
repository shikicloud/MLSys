---
title: "Continuum — Citation"
type: paper-citation
created: 2026-06-02
---

# Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live

- **arXiv**: [2511.02230](https://arxiv.org/abs/2511.02230)
- **Code**: To be open-sourced upon publication (paper states: "we will open-source our traces, code, and the agent serving testbed")
- **Authors**: Hanchen Li\* (UC Berkeley), Runyuan He\* (UC Berkeley), Qiuyang Mang (UC Berkeley), Qizheng Zhang (Stanford), Huanzhi Mao (UC Berkeley), Xiaokun Chen (Tensormesh), Hangrui Zhou (Tsinghua), Alvin Cheung (UC Berkeley), Joseph Gonzalez (UC Berkeley), Ion Stoica (UC Berkeley)
- **Affiliations**: UC Berkeley (1), Stanford (2), Tensormesh (3), Tsinghua (4)
- **Date**: 2025-11-04 (v1); 2026-05-25 (v6)
- **Implementation**: vLLM 0.10.2 fork, ~1k lines Python
- **License**: To be determined on code release
- **Wiki page**: [[continuum]]

## Why this paper matters

First serving system to model **per-turn queueing delay** as a first-class cost (alongside reload cost) for multi-turn LLM agent KV cache management. Where prior work (InferCept) used reload cost alone, Continuum's cost-benefit utility model accounts for the cumulative queueing penalty that returning requests incur when waiting in queue behind newly-admitted requests — a cost that persists even under fast CPU offload.

## Key technical contributions

1. **TTL utility model** with `Cost(τ, r) = MemUsage(r)/M × τ` and `Benefit(r) = CacheMissCost + OutOfOrderCost`, with memoryfulness factor `η = -Corr(k, N-k)` controlling program-ordering value.
2. **Program-level FCFS scheduling** via 3-key priority tuple (preempted, TTL_status, program_arrival), where program (not request) is the FCFS unit.
3. **TTL expiry mechanism** prevents long-tailed tool calls (e.g., `pytest` timeouts) from indefinitely blocking GPU memory.
4. **Drop-in vLLM extension** (~1k Python lines) with ~1ms scheduler overhead, no custom CUDA kernels required.

## Headline numbers

- **Trace replay**: 1.12–3.66× delay reduction, 1.10–3.22× throughput improvement across SWE-Bench / BFCL / OpenHands × Llama-3.1-8B/70B / Gemma-3-12B × A100/H100/B200.
- **Distributed real SWE-agent**: up to **8.18× delay reduction** + higher pass-rate vs SGLang/NVIDIA Dynamo on 500 SWE-Bench-Verified tasks (Tensormesh H100 testbed).
- **OpenHands RL rollout**: 144.9 steps/min (vs vLLM 93.4, ThunderAgent 114.8) on GLM-4.5-fp8 / 8×H100.

## BibTeX

```bibtex
@article{li2025continuum,
  title={Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live},
  author={Li, Hanchen and He, Runyuan and Mang, Qiuyang and Zhang, Qizheng and Mao, Huanzhi and Chen, Xiaokun and Zhou, Hangrui and Cheung, Alvin and Gonzalez, Joseph and Stoica, Ion},
  journal={arXiv preprint arXiv:2511.02230},
  year={2025}
}
```
