---
title: "Continuum —— 引用"
type: paper-citation
created: 2026-06-02
---

# Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live

- **arXiv**：[2511.02230](https://arxiv.org/abs/2511.02230)
- **代码**：发表时开源（论文称："we will open-source our traces, code, and the agent serving testbed"）
- **作者**：Hanchen Li\*（UC Berkeley）、Runyuan He\*（UC Berkeley）、Qiuyang Mang（UC Berkeley）、Qizheng Zhang（Stanford）、Huanzhi Mao（UC Berkeley）、Xiaokun Chen（Tensormesh）、Hangrui Zhou（清华）、Alvin Cheung（UC Berkeley）、Joseph Gonzalez（UC Berkeley）、Ion Stoica（UC Berkeley）
- **机构**：UC Berkeley (1)、Stanford (2)、Tensormesh (3)、清华 (4)
- **日期**：2025-11-04（v1）；2026-05-25（v6）
- **实现**：vLLM 0.10.2 fork，~1k 行 Python
- **许可证**：代码发布时确定
- **Wiki 页面**：[[continuum]]

## 为什么这篇论文重要

**第一个把 per-turn queueing delay 当一等代价**（跟 reload cost 并列）建模的多轮 LLM agent KV cache 管理 serving 系统。之前工作（InferCept）只用 reload cost，Continuum 的 cost-benefit 效用模型考虑返回请求排队等新接纳请求的累积代价 —— 这种代价即使在快速 CPU offload 下仍然存在。

## 核心技术贡献

1. **TTL 效用模型**：`Cost(τ, r) = MemUsage(r)/M × τ` 和 `Benefit(r) = CacheMissCost + OutOfOrderCost`，memoryfulness factor `η = -Corr(k, N-k)` 控制 program 顺序的价值。
2. **Program 级 FCFS 调度**：通过 3-key 优先级 tuple（preempted、TTL_status、program_arrival），其中 program（不是请求）是 FCFS 单位。
3. **TTL 过期机制** 避免长尾 tool 调用（如 `pytest` 超时）无限阻塞 GPU 内存。
4. **vLLM 即插即用扩展**（~1k Python 行），调度器开销 ~1ms，无需自定义 CUDA kernel。

## 头条数字

- **Trace replay**：跨 SWE-Bench / BFCL / OpenHands × Llama-3.1-8B/70B / Gemma-3-12B × A100/H100/B200 延迟降低 1.12–3.66×、吞吐提升 1.10–3.22×。
- **分布式真实 SWE-agent**：500 个 SWE-Bench-Verified 任务（Tensormesh H100 testbed）vs SGLang/NVIDIA Dynamo **延迟降低多达 8.18×** + pass-rate 更高。
- **OpenHands RL rollout**：GLM-4.5-fp8 / 8×H100 上 144.9 步/分（vs vLLM 93.4、ThunderAgent 114.8）。

## BibTeX

```bibtex
@article{li2025continuum,
  title={Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live},
  author={Li, Hanchen and He, Runyuan and Mang, Qiuyang and Zhang, Qizheng and Mao, Huanzhi and Chen, Xiaokun and Zhou, Hangrui and Cheung, Alvin and Gonzalez, Joseph and Stoica, Ion},
  journal={arXiv preprint arXiv:2511.02230},
  year={2025}
}
```
