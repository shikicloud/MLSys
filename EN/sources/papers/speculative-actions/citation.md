---
title: "Speculative Actions — Citation"
type: paper-citation
created: 2026-06-03
---

# Speculative Actions: A Lossless Framework for Faster Agentic Systems

- **arXiv**: [2510.04371](https://arxiv.org/abs/2510.04371)
- **Code**: [naimengye/speculative-action](https://github.com/naimengye/speculative-action) — released
- **Authors**: Naimeng Ye\*, Arnav Ahuja\*, Georgios Liargkovas\*, Yunan Lu\*, Kostis Kaffes, Tianyi Peng (\*equal contribution)
- **Affiliation**: Columbia University, New York
- **Contact**: `{ny2336, aa5790, gl2902, yl4021, kk3664, tp2845}@columbia.edu`
- **Date**: 2025-10-05 (v1); 2026-04-23 (v2)
- **Wiki page**: [[speculative-actions]]

## Why this paper matters

Generalizes the speculate-verify pattern from token level (Leviathan 2023) and plan level (Hua 2024) to **API-call level across the entire agentic environment** — LLM calls, tool/MCP server invocations, browser-use APIs, even human-as-API. First framework to formalize and prove lossless speedup for agent actions with explicit closed-form cost-latency tradeoff theorems.

## Key technical contributions

1. **Actor/Speculator decomposition** — slow authoritative Actor (SOTA LLM with high reasoning) + fast cheap Speculator (smaller model or same model with reduced prompt/reasoning).
2. **Algorithm 1 (k-way parallel next calls)** — k speculative branches simulated and pre-launched in parallel with Actor deliberation.
3. **Three losslessness primitives**: semantic guards, safety envelopes, repair paths.
4. **Closed-form theorems**: Theorem 1 (50% speedup upper bound), Theorem 3 (cost tradeoff), Theorem 4 (confidence-aware selective branching).
5. **Lossy extension** for OS-tuning-like settings with last-write-wins semantics.

## Headline numbers

| Environment | Accuracy | Speedup |
| ----------- | --------: | ------: |
| Chess (TextArena) at k=3 | 54.7% prediction | 19.5% time saved |
| HotpotQA at top-3 | 46.25% prediction | substantial |
| OS tuning P95 latency | — | 37.93ms vs 102.97ms untuned |

## BibTeX

```bibtex
@article{ye2025speculative,
  title={Speculative Actions: A Lossless Framework for Faster Agentic Systems},
  author={Ye, Naimeng and Ahuja, Arnav and Liargkovas, Georgios and Lu, Yunan and Kaffes, Kostis and Peng, Tianyi},
  journal={arXiv preprint arXiv:2510.04371},
  year={2025}
}
```
