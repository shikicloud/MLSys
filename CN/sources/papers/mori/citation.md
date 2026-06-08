---
title: "MORI —— 引用"
type: paper-citation
created: 2026-06-08
---

# Idleness is Relative: Exploiting Tool-Call Idle Windows for Offloading in Agentic Systems with MORI

- **arXiv**：[2606.00866v1](https://arxiv.org/abs/2606.00866)
- **代码**：截至 2026-06 未发布
- **作者**：Tian Xia¹、Hanchen Li¹、Zhifei Li²、Xiaokun Chen³、Hao Kang⁴、Yifan Qiao¹、Yi Xu¹、Ion Stoica¹
- **机构**：¹UC Berkeley、²中国人民大学、³斯坦福大学、⁴Georgia Institute of Technology
- **日期**：2026-05-30
- **实现**：~3,300 行 Python 在 ThunderAgent + 500 行在 SGLang v0.5.10 HiCache 上
- **Wiki 页面**：[[mori]]

## 为什么这篇论文重要

**同第一作者 Hanchen Li 的 [[continuum|Continuum]] 直接后继**。Continuum 是单层（仅 GPU HBM）带 TTL pinning，MORI 是两层（GPU + CPU DRAM）带连续 *idleness* metric 排所有 program 跨内存层动态分区。概念进化：Continuum 决定"pin 还是不 pin"；MORI 沿相对空闲度谱决定"哪一层"。这是 agent serving 内存管理的新 state of the art。

## 核心技术贡献

1. **Idleness metric**：$\iota = T_{acting}^{(k)}/(T_{reasoning}^{(k)}+T_{acting}^{(k)})$ over k=5 inference/tool-call 周期；滑窗平均同时对阶段转换响应又对异常值鲁棒。
2. **三层队列架构**：GPU queue（HBM，busy program）、CPU queue（DRAM 带 cache affinity，idle program）、Waiting queue（KV 丢弃）。
3. **Sticky rebalancing 调度策略**：program 留在层直到 idleness 不匹配 + 容量边界跨；避免 per-tick reshuffling 的 PCIe churn。
4. **推理引擎上的 typed offloading**：调度器传 busy/idle/inactive label；引擎 LRU 用 type 作优先级 sort key，两层间优先级反过来。

## 头号数字（Claude Code on SWE-bench Pro，80 并发 program）

| 硬件 | 模型 | TA+O baseline | MORI | 改善 |
| ---- | --- | ------------: | ---: | ---: |
| H200 80GB | Qwen-2.5 7B | 667 tokens/s | 853 tokens/s | +28% |
| B200 | Llama-3.1 70B | 124 tokens/s | **213 tokens/s** | **+71%** |
| H200 DP=3 | 各种 | — | — | 吞吐 +54-79% |
| TTFT（平均）| — | — | — | 低 18-43% |
| TTFT（峰值）| — | — | — | 低多达 2.8× |
| GPU 利用率 | 59-76% | **99%+** | — | — |

## 真实 Claude Code trace 的 tool 调用 duration 方差（n=16,886）

| 百分位 | Duration |
| -----: | -------: |
| P50 | 1,096 ms |
| P90 | 2,034 ms |
| P99 | 19,980 ms |
| P99.95 | 83,626 ms |

三个数量级 —— motivate 超过二元 busy/idle 的连续 idleness。

## BibTeX

```bibtex
@article{xia2026mori,
  title={Idleness is Relative: Exploiting Tool-Call Idle Windows for Offloading in Agentic Systems with MORI},
  author={Xia, Tian and Li, Hanchen and Li, Zhifei and Chen, Xiaokun and Kang, Hao and Qiao, Yifan and Xu, Yi and Stoica, Ion},
  journal={arXiv preprint arXiv:2606.00866},
  year={2026}
}
```
