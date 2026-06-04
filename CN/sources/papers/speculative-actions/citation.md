---
title: "Speculative Actions —— 引用"
type: paper-citation
created: 2026-06-03
---

# Speculative Actions: A Lossless Framework for Faster Agentic Systems

- **arXiv**：[2510.04371](https://arxiv.org/abs/2510.04371)
- **代码**：[naimengye/speculative-action](https://github.com/naimengye/speculative-action) —— 已发布
- **作者**：Naimeng Ye\*、Arnav Ahuja\*、Georgios Liargkovas\*、Yunan Lu\*、Kostis Kaffes、Tianyi Peng（\*等贡献）
- **机构**：哥伦比亚大学，纽约
- **联系**：`{ny2336, aa5790, gl2902, yl4021, kk3664, tp2845}@columbia.edu`
- **日期**：2025-10-05（v1）；2026-04-23（v2）
- **Wiki 页面**：[[speculative-actions]]

## 为什么这篇论文重要

把 speculate-verify 模式从 token 级（Leviathan 2023）和 plan 级（Hua 2024）泛化到**跨整个 agentic 环境的 API-call 级** —— LLM 调用、tool/MCP 服务器调用、browser-use API、甚至 human-as-API。第一个为 agent 动作正式化并证明 lossless 加速的框架，带显式闭式 cost-latency tradeoff 定理。

## 核心技术贡献

1. **Actor/Speculator 分解** —— 慢权威 Actor（SOTA LLM with 高推理）+ 快便宜 Speculator（更小模型或同模型 with 缩减 prompt/reasoning）。
2. **Algorithm 1（k-way 并行 next calls）** —— k 个 speculative 分支跟 Actor deliberation 并行模拟和 pre-launch。
3. **三个 losslessness 原语**：semantic guard、safety envelope、repair path。
4. **闭式定理**：Theorem 1（50% 加速上界）、Theorem 3（成本 tradeoff）、Theorem 4（confidence-aware 选择性分支）。
5. **有损扩展** 给 OS-tuning 类设置 with last-write-wins 语义。

## 头号数字

| 环境 | 准确率 | 加速 |
| --- | -----: | ---: |
| Chess（TextArena）at k=3 | 54.7% 预测 | 节省 19.5% 时间 |
| HotpotQA at top-3 | 46.25% 预测 | 显著 |
| OS tuning P95 延迟 | — | 37.93ms vs 102.97ms 未调 |

## BibTeX

```bibtex
@article{ye2025speculative,
  title={Speculative Actions: A Lossless Framework for Faster Agentic Systems},
  author={Ye, Naimeng and Ahuja, Arnav and Liargkovas, Georgios and Lu, Yunan and Kaffes, Kostis and Peng, Tianyi},
  journal={arXiv preprint arXiv:2510.04371},
  year={2025}
}
```
