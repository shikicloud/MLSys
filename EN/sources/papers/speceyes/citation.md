---
title: "SpecEyes — Citation"
type: paper-citation
created: 2026-06-03
---

# SpecEyes: Accelerating Agentic Multimodal LLMs via Speculative Perception and Planning

- **arXiv**: [2603.23483v1](https://arxiv.org/abs/2603.23483)
- **Code**: [MAC-AutoML/SpecEyes](https://github.com/MAC-AutoML/SpecEyes) — released
- **Authors**: Haoyu Huang\* (Xiamen U), Jinfa Huang\* (Rochester), Zhongwei Wan (OSU), Xiawu Zheng (Xiamen U), Rongrong Ji (Xiamen U), Jiebo Luo (Rochester) — \*equal contribution
- **Contact**: jhuang90@ur.rochester.edu, huanghaoyu@stu.xmu.edu.cn
- **Date**: 2026-03-24
- **Implementation**: Single 1× NVIDIA A100 40GB, greedy decoding
- **Wiki page**: [[speceyes]]

## Why this paper matters

The multimodal-LLM counterpart to [[speculative-actions|Speculative Actions]] — first framework to lift speculation from token-level to agentic-level for vision tool-using MLLMs. Solves the *stateful bottleneck* of agentic MLLMs (OpenAI o3, Gemini Agentic Vision, DeepEyes, Thyme) by routing tool-free queries to a small non-agentic MLLM, gated by a novel calibration-free **answer separability score** that outperforms softmax confidence by 14× in discriminability.

## Key technical contributions

1. **Four-phase pipeline**: tool-use judgment → speculative prediction → cognitive gating → agentic fallback.
2. **Answer separability score** $S_{sep} = (\ell_{[1]} - \mu_K) / (\sigma_K + \epsilon)$ — scale-invariant, competition-aware confidence metric replacing softmax.
3. **Min-aggregation** of token-level separability with Proposition 1 worst-case guarantee.
4. **Heterogeneous parallel funnel** delivering throughput speedup $\approx 1/(1-\beta\alpha)$.
5. **Improves accuracy and speed simultaneously** on most benchmarks (counterexample to speculation-accuracy-tradeoff intuition).

## Headline numbers

| Benchmark | Baseline → SpecEyes (min) | Speedup |
| --------- | -------------------------: | ------: |
| DeepEyes avg | 81.39% → **84.26%** (+2.87) | **1.73×** |
| Thyme avg | 82.29% → **83.99%** (+1.70) | **1.42×** |
| POPE Adversarial | 78.43% → **85.13%** (+6.7) | **2.13×** |
| KDE Δ (S_sep^min vs softmax) | — | **14× larger** |

## BibTeX

```bibtex
@article{huang2026speceyes,
  title={SpecEyes: Accelerating Agentic Multimodal LLMs via Speculative Perception and Planning},
  author={Huang, Haoyu and Huang, Jinfa and Wan, Zhongwei and Zheng, Xiawu and Ji, Rongrong and Luo, Jiebo},
  journal={arXiv preprint arXiv:2603.23483},
  year={2026}
}
```
