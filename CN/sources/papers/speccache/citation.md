---
title: "What Limits Agentic Systems Efficiency? (SpecCache) —— 引用"
type: paper-citation
created: 2026-06-05
---

# What Limits Agentic Systems Efficiency?

- **arXiv**：[2510.16276v1](https://arxiv.org/abs/2510.16276)
- **代码**：截至 2026-06 未发布
- **作者**：Song Bian\*（UW-Madison）、Minghao Yan\*（UW-Madison）、Anand Jayarajan（多伦多大学 + NVIDIA）、Gennady Pekhimenko（多伦多大学 + NVIDIA）、Shivaram Venkataraman（UW-Madison）（\*等贡献）
- **联系**：`songbian@cs.wisc.edu`、`myan@cs.wisc.edu`
- **日期**：2025-10-18（preprint 2025-10-21）
- **Wiki 页面**：[[speccache]]

## 为什么这篇论文重要

两合一贡献：(1) 迄今最彻底的公开 LLM API 延迟方差研究（5 天、5 provider、9 模型、3 区域）；(2) **SpecCache**，实证验证 [[speculative-actions|Speculative Actions]] 理论化的 draft-model-for-actions 假设的系统 —— 在 Reflexion-based web agent 上取得 9-54× cache 命中率改善和 3.2× web env 延迟降低，不降级轨迹。

## 核心技术贡献

1. **多 provider、多区域 LLM API characterization**：fixed-input 请求 69.21× 方差；按地理变异系数 42.61-135.21%；OpenAI priority processing 部分 mitigate。
2. **Web 环境延迟分析**：Reflexion-based agent 在 WebWalkerQA 上 E2E 延迟 53.7%；中位爬取 6s 带长尾。
3. **SpecCache 框架**：action-observation LRU cache + 通过异步 draft LLM 的模型基础 prefetching + 无损轨迹保留。
4. **实证证据 draft 模型身份重要性低于存在**：WebWalkerQA 上 self-draft（87.3%）vs cross-draft（83.3%）只差 4 个百分点。

## 头号数字

| 指标 | 随机 caching | SpecCache | 改善 |
| --- | -----------: | --------: | ---: |
| WebWalkerQA 命中率（o4-mini target）| 8.9% | 83.3% | 9.4× |
| Frames 命中率（o4-mini target）| 1.0% | 54.0% | 54× |
| Web env 延迟 | — | — | 多达 3.2× 降低 |
| Agentic 性能 | baseline | 保留 | — |
| LLM API 方差（fixed input）| — | — | **观察 69.21×** |

## BibTeX

```bibtex
@article{bian2025limits,
  title={What Limits Agentic Systems Efficiency?},
  author={Bian, Song and Yan, Minghao and Jayarajan, Anand and Pekhimenko, Gennady and Venkataraman, Shivaram},
  journal={arXiv preprint arXiv:2510.16276},
  year={2025}
}
```
