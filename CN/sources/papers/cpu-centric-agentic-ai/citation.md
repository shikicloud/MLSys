---
title: "CPU-Centric Perspective on Agentic AI —— 引用"
type: paper-citation
created: 2026-06-02
---

# Towards Understanding, Analyzing, and Optimizing Agentic AI Execution: A CPU-Centric Perspective

- **arXiv**：[2511.00739](https://arxiv.org/abs/2511.00739)
- **代码**：未发布
- **作者**：Ritik Raj（Georgia Institute of Technology）、Souvik Kundu（Intel）、Ishita Vohra（Georgia Institute of Technology）、Hong Wang（Intel）、Tushar Krishna（Georgia Institute of Technology）
- **日期**：2025-11-01（v1）；2026-04-16（v3）
- **实现**：vLLM v0.14.0、PyTorch 2.8.0
- **许可证**：CC-BY 4.0
- **Wiki 页面**：[[cpu-centric-agentic-ai]]

## 为什么这篇论文重要

第一个端到端实证研究调和 agentic AI serving 中 "tool dominates" vs "LLM dominates" 辩论。答案是 **workload 和硬件依赖** —— tool 处理从 12%（Toolformer on HP CPU + LP GPU）到 88%（RAG/ENNS over 200GB 语料 on LP CPU + HP GPU）。识别定量规律：CPU 并行效率在 1-2× 核心数处饱和，超过 OS 争用主导。把这些测量转成两个调度算法（COMB + MAS）。

## 核心技术贡献

1. **三轴编译期分类法** 给 agentic 系统（Orchestrator × Path × Repetitiveness）。
2. **五个代表性 agentic workload 的运行时 characterization**（Toolformer、SWE-Agent、RAG/Haystack、ChemCrow、Web-Augmented Agent）在两个非对称 CPU-GPU 系统上。
3. **COMB（CPU-Aware Overlapped Micro-Batching）**：带跨阶段 overlap 的有界 micro-batch；P50 延迟低 1.7×，开环服务延迟低 3.9×。
4. **MAS（Mixed Agentic Scheduling）**：异质 chat+agent workload 的请求类型感知并发接纳；少数请求类型 P50/P90 提升 2.37×/2.49×。
5. **能耗 characterization**：CPU 动态能耗占系统 E2E 能耗 20-70%，取决于 workload。

## BibTeX

```bibtex
@article{raj2025cpu,
  title={Towards Understanding, Analyzing, and Optimizing Agentic AI Execution: A CPU-Centric Perspective},
  author={Raj, Ritik and Kundu, Souvik and Vohra, Ishita and Wang, Hong and Krishna, Tushar},
  journal={arXiv preprint arXiv:2511.00739},
  year={2025}
}
```
