---
title: "Agentic AI Workload Characteristics —— 引用"
type: paper-citation
created: 2026-06-03
---

# Agentic AI Workload Characteristics

- **arXiv**：[2605.26297v1](https://arxiv.org/abs/2605.26297)
- **代码**：未发布
- **作者**：Yichao Yuan（UIUC）、Ankita Nayak（Gimlet Labs）、Souvik Kundu（Intel）、Nishil Talati（UIUC）
- **联系**：`{yichaoy2, nishil}@illinois.edu`
- **日期**：2026-05-25
- **许可证**：CC-BY 4.0
- **Wiki 页面**：[[agentic-ai-workload-characteristics]]

## 为什么这篇论文重要

2026 年标杆端到端 agent workload 测量论文。在 5 个真实 benchmark 和 4 个模型变体（Qwen3.6-27B Thinking/Instant + Gemma4-31B Thinking/Instant）上提供 per-turn、per-token、per-tool、prefill/decode 分辨的 instrumentation，用生产级工具（Claude Code、vLLM、OpenTelemetry、Jaeger）。给真实 coding/QA agent workload 定 "tool 主导" vs "LLM 主导" 辩论的实证参考。

## 关键发现

1. LLM 时间 = E2E 71-98%；tool = 2-29% —— workload 相关。
2. Decode = LLM 时间 91-98.6%；prefill = 1.4-9.0%。
3. 实证 KV cache 命中率 = 84.6-99.5%。
4. Append-to-output 中位比 < 1.5×。
5. 推理紧凑化轨迹 —— Gemma Thinking on ADE 平均 18 turn vs Gemma Instant 的 108.8。
6. 失败 agent 积累更大上下文（多达 1.8× 平均）。
7. Tool-call token 主导 Instant 变体输出（70-98%）。
8. Tool 组成 workload 特定。
9. Tool 意图从早 Read/Explore shift 到晚 Execute/Write。
10. Agent 委托 tool 最贵（GAIA 上多分钟）。
11. 改 state tool（Edit、Bash）失败 28-95%。

## Benchmark 数字

| Benchmark | LLM % | Tool % |
| --------- | ----: | -----: |
| DABStep | 96-98% | 2-4% |
| ADE-Bench | 86-92% | 8-14% |
| SWE-Bench Pro | 82.3% | 17.7% |
| Terminal-Bench 2.0 | 82.4% | 17.6% |
| GAIA（Gemma Thinking）| 71.3% | 28.7% |

## BibTeX

```bibtex
@article{yuan2026agentic,
  title={Agentic AI Workload Characteristics},
  author={Yuan, Yichao and Nayak, Ankita and Kundu, Souvik and Talati, Nishil},
  journal={arXiv preprint arXiv:2605.26297},
  year={2026}
}
```
