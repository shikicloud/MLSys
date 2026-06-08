---
title: "InferCept —— 引用"
type: paper-citation
created: 2026-06-08
---

# InferCept: Efficient Intercept Support for Augmented Large Language Model Inference

- **arXiv**：[2402.01869](https://arxiv.org/abs/2402.01869)（v1 2024-02-02；v2 2024-05-30）
- **会议**：ICML 2024, PMLR 235
- **代码**：[WukLab/InferCept](https://github.com/WukLab/InferCept) —— UCSD WukLab 团队发布的 vLLM fork
- **作者**：Reyna Abhyankar\*（UCSD）、Zijian He\*（UCSD）、Vikranth Srivatsa（UCSD）、Hao Zhang（UCSD）、Yiying Zhang（UCSD）
- **机构**：UC San Diego（全部）
- **实现**：vLLM fork；沿用 PagedAttention 作为 KV 内存管理器
- **许可证**：Apache-2.0（仓库标注）
- **Wiki 页面**：[[infercept]]

## 为什么这篇论文重要

**第一个面向"被中断的增强型 LLM"专门设计的推理框架。** 确立了 canonical 分类法（`Discard` / `Preserve` / `Swap`）和四条闭式 *浪费方程*，量化每种动作下 GPU 内存×时间 的损失。后续 agent-serving 工作 —— Continuum（TTL）、MORI（两层 idleness）、Autellix、Pie —— 都以它为基线。[[continuum]] 与 [[mori]] 都把它作为 agent 工具调用期间 KV 保留的直接前驱 / baseline 引用。

## 核心技术贡献

1. **浪费方程** —— 闭式公式 $\text{WasteDiscard}$、$\text{WastePreserve}$、$\text{WasteSwap}$、$\text{WasteChunkD}$，参数化于中断时长、上下文长度、batch 大小、每 token KV 内存。
2. **Swap 流水线 + 分块** —— 按 model layer 粒度把 CPU↔GPU PCIe 与 forward kernel 重叠；在 PCIe + 空闲 CPU 约束下做 swap-out 预算化；消除 96% 的 Swap 浪费。
3. **重算分块** —— 把一次 recompute 切成大小为 $S - \text{running\_group\_size}$ 的块（$S$ 是 GPU query-token 饱和点）；分块重算去填充 decode 留下的空闲 GPU 核；把 Discard 浪费砍半。
4. **Min-waste 调度** —— 每迭代，按 $\min(\text{WastePreserve}, \text{WasteChunkD})$ 降序对被中断 request 排序，再依次分给 swap（预算内）/ preserve / discard。
5. **动态中断时长估计器** —— $\hat{T}_{INT} = t_{now} - t_{call}$ 在高方差 human/model 驱动的中断（chatbot、image、TTS）上达到 oracle 性能的 93%。

## 标志数字

- **吞吐**：混合增强负载下比 vLLM 高 1.6×–2×；**完成请求数 / 秒翻倍**。
- **延迟**：6B 上 normalized latency 降低 1.9×–5.7×；**13B-TP2 上 1.6×–10×**；**70B-TP4 上 1.3×–12×**。
- **GPU 内存浪费**：vLLM 约 25% → 完整 InferCept **0.69%**；消除超过 60% 的重算浪费、96% 的 swap 浪费。
- **单 augment**：QA 快 2.3×；Chatbot 快 1.9×（分块 + 流水线贡献 Chatbot 加速的 54%）。
- **基线系统的重算开销**：混合负载下占总 forward 时间 37–40% —— 即 InferCept 关掉的那个缺口。

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
