---
title: "Agent Lightning —— 引用"
type: paper-citation
created: 2026-06-02
---

# Agent Lightning: Train ANY AI Agents with Reinforcement Learning

- **arXiv**：[2508.03680](https://arxiv.org/abs/2508.03680)
- **代码**：[microsoft/agent-lightning](https://github.com/microsoft/agent-lightning)（Apache-2.0，v0.3.0 2025-12）
- **作者**：Xufang Luo, Yuge Zhang, Zhiyuan He, Zilong Wang, Siyun Zhao, Dongsheng Li, Luna K. Qiu, Yuqing Yang
- **机构**：Microsoft Research
- **日期**：2025-08-05
- **联系**：agent-lightning@microsoft.com
- **文档**：https://microsoft.github.io/agent-lightning/
- **许可证**：代码 Apache-2.0
- **Wiki 页面**：[[agent-lightning]]

## 为什么这篇论文重要

第一个公开发表的（2025-08）将 agent 执行与 RL 训练完全解耦的框架。比 ProRL Agent（2026-03）早 7 个月，比 Polar（2026-05）早 9 个月。它开创的架构模式 ——*agent 跑在 client；LLM 在 server 上通过 OpenAI 兼容 API 暴露；trajectory 以 transition 形式回传*—— 成为后续 NVIDIA 两篇论文延续的模板。

## 核心技术贡献

1. Agent 执行的 MDP 形式化，带语义变量 state；一个 action = 一次 LLM 调用的 token 序列。
2. LightningRL 分层 credit assignment 完全绕开 sequence 拼接 + masking。
3. Training-Agent Disaggregation（TA 解耦）架构：Lightning Server + Lightning Client 通过 OpenAI 兼容 API。
4. OpenTelemetry 原生 trace 捕获。
5. 多 agent 选择性优化（自然训练任意子集 agent）。

## BibTeX

```bibtex
@article{luo2025agent,
  title={Agent Lightning: Train ANY AI Agents with Reinforcement Learning},
  author={Luo, Xufang and Zhang, Yuge and He, Zhiyuan and Wang, Zilong and Zhao, Siyun and Li, Dongsheng and Qiu, Luna K. and Yang, Yuqing},
  journal={arXiv preprint arXiv:2508.03680},
  year={2025}
}
```
