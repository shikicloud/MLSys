---
title: "CPU-Centric Perspective on Agentic AI — Citation"
type: paper-citation
created: 2026-06-02
---

# Towards Understanding, Analyzing, and Optimizing Agentic AI Execution: A CPU-Centric Perspective

- **arXiv**: [2511.00739](https://arxiv.org/abs/2511.00739)
- **Code**: Not yet released
- **Authors**: Ritik Raj (Georgia Institute of Technology), Souvik Kundu (Intel), Ishita Vohra (Georgia Institute of Technology), Hong Wang (Intel), Tushar Krishna (Georgia Institute of Technology)
- **Date**: 2025-11-01 (v1); 2026-04-16 (v3)
- **Implementation**: vLLM v0.14.0, PyTorch 2.8.0
- **License**: CC-BY 4.0
- **Wiki page**: [[cpu-centric-agentic-ai]]

## Why this paper matters

First end-to-end empirical study reconciling the "tool dominates" vs "LLM dominates" debate in agentic AI serving. The answer is **workload- and hardware-dependent** — tool processing ranges from 12% (Toolformer on HP CPU + LP GPU) to 88% (RAG/ENNS over 200GB corpus on LP CPU + HP GPU). Identifies a quantitative law: CPU parallelism efficiency saturates at 1-2× number of cores, beyond which OS contention dominates. Translates these measurements into two scheduling algorithms (COMB + MAS).

## Key technical contributions

1. **Three-axis compile-time taxonomy** for agentic systems (Orchestrator × Path × Repetitiveness).
2. **Runtime characterization** of five representative agentic workloads (Toolformer, SWE-Agent, RAG/Haystack, ChemCrow, Web-Augmented Agent) on two asymmetric CPU-GPU systems.
3. **COMB (CPU-Aware Overlapped Micro-Batching)**: capped micro-batches with cross-stage overlap; 1.7× P50 latency reduction, 3.9× service latency reduction in open-loop.
4. **MAS (Mixed Agentic Scheduling)**: request-type-aware concurrent admission for heterogeneous chat+agent workloads; 2.37×/2.49× P50/P90 improvement for minority request type.
5. **Energy characterization**: CPU dynamic energy is 20-70% of system E2E energy, depending on workload.

## BibTeX

```bibtex
@article{raj2025cpu,
  title={Towards Understanding, Analyzing, and Optimizing Agentic AI Execution: A CPU-Centric Perspective},
  author={Raj, Ritik and Kundu, Souvik and Vohra, Ishita and Wang, Hong and Krishna, Tushar},
  journal={arXiv preprint arXiv:2511.00739},
  year={2025}
}
```
