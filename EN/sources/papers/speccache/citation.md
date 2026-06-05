---
title: "What Limits Agentic Systems Efficiency? (SpecCache) — Citation"
type: paper-citation
created: 2026-06-05
---

# What Limits Agentic Systems Efficiency?

- **arXiv**: [2510.16276v1](https://arxiv.org/abs/2510.16276)
- **Code**: Not released as of June 2026
- **Authors**: Song Bian\* (UW-Madison), Minghao Yan\* (UW-Madison), Anand Jayarajan (University of Toronto + NVIDIA), Gennady Pekhimenko (University of Toronto + NVIDIA), Shivaram Venkataraman (UW-Madison) (\*equal contribution)
- **Contact**: `songbian@cs.wisc.edu`, `myan@cs.wisc.edu`
- **Date**: 2025-10-18 (preprint 2025-10-21)
- **Wiki page**: [[speccache]]

## Why this paper matters

A two-in-one contribution: (1) the most thorough public LLM API latency variance study to date (5 days, 5 providers, 9 models, 3 regions); (2) **SpecCache**, the system that empirically validates the draft-model-for-actions hypothesis that [[speculative-actions|Speculative Actions]] theorized — achieving 9-54× cache hit rate improvement and 3.2× web env latency reduction on Reflexion-based web agents without trajectory degradation.

## Key technical contributions

1. **Multi-provider, multi-region LLM API characterization**: 69.21× variance for fixed-input requests; coefficient of variation 42.61-135.21% by geography; OpenAI priority processing partially mitigates.
2. **Web environment latency analysis**: 53.7% of E2E latency in Reflexion-based agents on WebWalkerQA; median crawl 6s with long tail.
3. **SpecCache framework**: action-observation LRU cache + model-based prefetching via async draft LLM + lossless trajectory preservation.
4. **Empirical evidence that draft model identity matters less than existence**: self-draft (87.3%) vs cross-draft (83.3%) on WebWalkerQA differ by only 4 percentage points.

## Headline numbers

| Metric | Random caching | SpecCache | Improvement |
| ------ | -------------: | --------: | ----------: |
| Hit rate WebWalkerQA (o4-mini target) | 8.9% | 83.3% | 9.4× |
| Hit rate Frames (o4-mini target) | 1.0% | 54.0% | 54× |
| Web env latency | — | — | up to 3.2× reduction |
| Agentic performance | baseline | preserved | — |
| LLM API variance (fixed input) | — | — | **69.21× observed** |

## BibTeX

```bibtex
@article{bian2025limits,
  title={What Limits Agentic Systems Efficiency?},
  author={Bian, Song and Yan, Minghao and Jayarajan, Anand and Pekhimenko, Gennady and Venkataraman, Shivaram},
  journal={arXiv preprint arXiv:2510.16276},
  year={2025}
}
```
