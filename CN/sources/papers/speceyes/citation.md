---
title: "SpecEyes —— 引用"
type: paper-citation
created: 2026-06-03
---

# SpecEyes: Accelerating Agentic Multimodal LLMs via Speculative Perception and Planning

- **arXiv**：[2603.23483v1](https://arxiv.org/abs/2603.23483)
- **代码**：[MAC-AutoML/SpecEyes](https://github.com/MAC-AutoML/SpecEyes) —— 已发布
- **作者**：Haoyu Huang\*（厦大）、Jinfa Huang\*（罗切斯特）、Zhongwei Wan（OSU）、Xiawu Zheng（厦大）、Rongrong Ji（厦大）、Jiebo Luo（罗切斯特）—— \*等贡献
- **联系**：jhuang90@ur.rochester.edu、huanghaoyu@stu.xmu.edu.cn
- **日期**：2026-03-24
- **实现**：单 1× NVIDIA A100 40GB，greedy decoding
- **Wiki 页面**：[[speceyes]]

## 为什么这篇论文重要

[[speculative-actions|Speculative Actions]] 的多模态 LLM 对应 —— 第一个把 speculation 从 token 级提升到 agentic 级 for vision tool-using MLLM 的框架。通过把 tool-free 查询路由到小非 agentic MLLM 解决 agentic MLLM 的*有状态瓶颈*（OpenAI o3、Gemini Agentic Vision、DeepEyes、Thyme），由新颖无校准 **answer separability score** gate，比 softmax 信心 discriminability 高 14×。

## 核心技术贡献

1. **四阶段 pipeline**：tool-use 判断 → speculative 预测 → cognitive gating → agentic fallback。
2. **Answer separability score** $S_{sep} = (\ell_{[1]} - \mu_K) / (\sigma_K + \epsilon)$ —— scale 不变、竞争感知信心度量取代 softmax。
3. **Token 级 separability 的 min 聚合** with Proposition 1 最坏情况保证。
4. **异质并行 funnel** 给吞吐加速 $\approx 1/(1-\beta\alpha)$。
5. **同时改善准确率和速度** 在多数 benchmark 上（speculation-准确率-tradeoff 直觉的反例）。

## 头号数字

| Benchmark | Baseline → SpecEyes (min) | 加速 |
| --------- | -------------------------: | ---: |
| DeepEyes 平均 | 81.39% → **84.26%**（+2.87）| **1.73×** |
| Thyme 平均 | 82.29% → **83.99%**（+1.70）| **1.42×** |
| POPE Adversarial | 78.43% → **85.13%**（+6.7）| **2.13×** |
| KDE Δ（S_sep^min vs softmax）| — | **大 14×** |

## BibTeX

```bibtex
@article{huang2026speceyes,
  title={SpecEyes: Accelerating Agentic Multimodal LLMs via Speculative Perception and Planning},
  author={Huang, Haoyu and Huang, Jinfa and Wan, Zhongwei and Zheng, Xiawu and Ji, Rongrong and Luo, Jiebo},
  journal={arXiv preprint arXiv:2603.23483},
  year={2026}
}
```
