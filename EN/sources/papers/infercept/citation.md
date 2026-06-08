---
title: "InferCept — Citation"
type: paper-citation
created: 2026-06-08
---

# InferCept: Efficient Intercept Support for Augmented Large Language Model Inference

- **arXiv**: [2402.01869](https://arxiv.org/abs/2402.01869) (v1 2024-02-02; v2 2024-05-30)
- **Venue**: ICML 2024, PMLR 235
- **Code**: [WukLab/InferCept](https://github.com/WukLab/InferCept) — vLLM fork released by UCSD WukLab
- **Authors**: Reyna Abhyankar\* (UCSD), Zijian He\* (UCSD), Vikranth Srivatsa (UCSD), Hao Zhang (UCSD), Yiying Zhang (UCSD)
- **Affiliation**: UC San Diego (all)
- **Implementation**: vLLM fork; reuses PagedAttention as the KV memory manager
- **License**: Apache-2.0 (per repo)
- **Wiki page**: [[infercept]]

## Why this paper matters

**The first LLM inference framework designed for augmented LLMs with interceptions.** Establishes the canonical taxonomy (`Discard` / `Preserve` / `Swap`) and four closed-form *waste equations* that quantify GPU-memory-time loss under each action. All subsequent agent-serving work — Continuum (TTL), MORI (two-tier idleness), Autellix, Pie — builds on this baseline. Cited as the direct predecessor / baseline by both [[continuum]] and [[mori]] for KV-cache pinning during agent tool calls.

## Key technical contributions

1. **Waste equations** — closed-form formulas $\text{WasteDiscard}$, $\text{WastePreserve}$, $\text{WasteSwap}$, $\text{WasteChunkD}$ parameterized by interception duration, context length, batch size, per-token KV memory.
2. **Swap pipelining + chunking** — overlap CPU↔GPU PCIe with forward kernels at model-layer granularity; budgeted swap-out under PCIe + free-CPU constraints; eliminates 96% of Swap waste.
3. **Recomputation chunking** — split a recompute into chunks of size $S - \text{running\_group\_size}$ where $S$ is the GPU query-token saturation point; the chunked recompute fills decode's spare GPU cores; halves Discard waste.
4. **Min-waste scheduling** — per-iteration, sort intercepted requests by descending $\min(\text{WastePreserve}, \text{WasteChunkD})$ then assign to swap (within budget) / preserve / discard.
5. **Dynamic interception-duration estimator** — $\hat{T}_{INT} = t_{now} - t_{call}$ achieves 93% of oracle performance on high-variance human/model-driven interceptions (chatbot, image, TTS).

## Headline numbers

- **Throughput**: 1.6×–2× higher than vLLM under mixed augmented workload; **2× more completed requests per second**.
- **Latency**: 1.9×–5.7× lower normalized latency on 6B; **1.6×–10× lower on 13B-TP2**; **1.3×–12× lower on 70B-TP4**.
- **GPU memory waste**: vLLM ~25% → full InferCept **0.69%**; eliminates >60% of recompute waste and 96% of swap waste.
- **Single-augment**: QA 2.3× faster; Chatbot 1.9× faster (chunking + pipelining contribute 54% of the Chatbot speedup).
- **Recomputation cost in baseline systems**: 37–40% of total model forwarding time under mixed workload — the gap InferCept closes.

## BibTeX

```bibtex
@inproceedings{abhyankar2024infercept,
  title={InferCept: Efficient Intercept Support for Augmented Large Language Model Inference},
  author={Abhyankar, Reyna and He, Zijian and Srivatsa, Vikranth and Zhang, Hao and Zhang, Yiying},
  booktitle={Proceedings of the 41st International Conference on Machine Learning},
  series={PMLR 235},
  year={2024},
  publisher={PMLR}
}
```
