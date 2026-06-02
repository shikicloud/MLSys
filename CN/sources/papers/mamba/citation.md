---
title: "Mamba —— 引用"
type: paper-citation
created: 2026-06-02
---

# Mamba: Linear-Time Sequence Modeling with Selective State Spaces

- **arXiv**：[2312.00752](https://arxiv.org/abs/2312.00752)
- **代码**：[state-spaces/mamba](https://github.com/state-spaces/mamba)（Apache-2.0）
- **作者**：Albert Gu（CMU）、Tri Dao（Princeton）
- **日期**：2023 年 12 月
- **会议**：NeurIPS 2024
- **许可证**：代码 Apache-2.0
- **Wiki 页面**：[[mamba]]

## 为什么这篇论文重要

Hybrid-attention 时代的基础架构页面。2026 年所有 hybrid-attention 生产 LLM —— Qwen3-Next、Qwen3.5、Qwen3.6、Nemotron-H、Kimi Linear、MiMo-V2-Flash、DeepSeek-V3.2 NSA —— 的线性 attention 层都是 Mamba 的后裔。理解 Selective SSM、chunkwise 并行扫描、以及 Mamba 特定的弱点（associative recall、ICL）是理解 **为什么业界走向 *hybrid* 而非全 Mamba** 的前置条件。

## 核心技术贡献

1. Selective SSM（S6）：input-dependent 状态空间转移打破 LTI，让内容感知过滤和选择性记忆成为可能。
2. 硬件感知并行扫描：Blelloch scan + kernel fusion + SRAM-resident state + recompute-in-backward。比朴素 PyTorch 快 20-40×。
3. Mamba block：单一同构 block 替换 attention 和 FFN。
4. 第一个在 3B 参数 + 300B token 语言建模上匹敌 Transformer++ 的 SSM。
5. seqlen 2K 上 decoding 吞吐 5× 更快；线性时间序列 scaling。

## 后继谱系（2024-2026）

- **Mamba-2 / SSD**（arXiv:2405.21060, 2024-05）—— Structured State-Space Duality 把 SSM 和 attention 统一为 structured matrix。
- **Gated DeltaNet**（arXiv:2412.06464, ICLR 2025）—— delta-rule 变种；驱动 Qwen3-Next、Qwen3.5、Qwen3.6。
- **Jamba**（AI21, 2024-03）—— 首个生产 hybrid（1:7 attention:Mamba）。
- **Lightning Attention**（Qin et al., 2024）—— MiniMax 使用。
- **KDA (Kimi Delta Attention)** —— Kimi Linear、K2 家族使用。
- **Mamba-3**（2026）—— 活跃研究。

## BibTeX

```bibtex
@article{gu2023mamba,
  title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023}
}
```
