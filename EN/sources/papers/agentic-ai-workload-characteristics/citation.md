---
title: "Agentic AI Workload Characteristics — Citation"
type: paper-citation
created: 2026-06-03
---

# Agentic AI Workload Characteristics

- **arXiv**: [2605.26297v1](https://arxiv.org/abs/2605.26297)
- **Code**: Not released
- **Authors**: Yichao Yuan (UIUC), Ankita Nayak (Gimlet Labs), Souvik Kundu (Intel), Nishil Talati (UIUC)
- **Contact**: `{yichaoy2, nishil}@illinois.edu`
- **Date**: 2026-05-25
- **License**: CC-BY 4.0
- **Wiki page**: [[agentic-ai-workload-characteristics]]

## Why this paper matters

The canonical end-to-end agent workload measurement paper for 2026. Provides per-turn, per-token, per-tool, prefill/decode-resolved instrumentation across five real benchmarks and four model variants (Qwen3.6-27B Thinking/Instant + Gemma4-31B Thinking/Instant) using production-grade tools (Claude Code, vLLM, OpenTelemetry, Jaeger). The empirical reference that settles "tool dominates" vs "LLM dominates" debates for real coding/QA agent workloads.

## Key findings

1. LLM time = 71–98% of E2E; tool = 2–29% — workload-dependent.
2. Decode = 91–98.6% of LLM time; prefill = 1.4–9.0%.
3. Empirical KV cache hit ratio = 84.6–99.5%.
4. Median append-to-output ratio < 1.5×.
5. Reasoning compactifies trajectories — Gemma Thinking on ADE averages 18 turns vs Gemma Instant's 108.8.
6. Failed agents accumulate larger contexts (up to 1.8× mean).
7. Tool-call tokens dominate Instant variant outputs (70–98%).
8. Tool composition is workload-specific.
9. Tool intention shifts from Read/Explore (early) to Execute/Write (late).
10. Agent delegation tool is the most expensive (multi-minute on GAIA).
11. State-modifying tools (Edit, Bash) fail at 28–95%.

## Benchmark numbers

| Benchmark | LLM % | Tool % |
| --------- | ----: | -----: |
| DABStep | 96–98% | 2–4% |
| ADE-Bench | 86–92% | 8–14% |
| SWE-Bench Pro | 82.3% | 17.7% |
| Terminal-Bench 2.0 | 82.4% | 17.6% |
| GAIA (Gemma Thinking) | 71.3% | 28.7% |

## BibTeX

```bibtex
@article{yuan2026agentic,
  title={Agentic AI Workload Characteristics},
  author={Yuan, Yichao and Nayak, Ankita and Kundu, Souvik and Talati, Nishil},
  journal={arXiv preprint arXiv:2605.26297},
  year={2026}
}
```
