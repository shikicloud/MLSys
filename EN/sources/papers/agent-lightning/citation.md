---
title: "Agent Lightning — Citation"
type: paper-citation
created: 2026-06-02
---

# Agent Lightning: Train ANY AI Agents with Reinforcement Learning

- **arXiv**: [2508.03680](https://arxiv.org/abs/2508.03680)
- **Code**: [microsoft/agent-lightning](https://github.com/microsoft/agent-lightning) (Apache-2.0, v0.3.0 Dec 2025)
- **Authors**: Xufang Luo, Yuge Zhang, Zhiyuan He, Zilong Wang, Siyun Zhao, Dongsheng Li, Luna K. Qiu, Yuqing Yang
- **Affiliation**: Microsoft Research
- **Date**: 2025-08-05
- **Contact**: agent-lightning@microsoft.com
- **Documentation**: https://microsoft.github.io/agent-lightning/
- **License**: Code Apache-2.0
- **Wiki page**: [[agent-lightning]]

## Why this paper matters

First publicly-documented framework (Aug 2025) to fully decouple agent execution from RL training. Predates ProRL Agent (Mar 2026) by 7 months and Polar (May 2026) by 9 months. The architectural pattern it pioneered — *agent runs on a client; LLM lives on a server speaking OpenAI-compatible API; trajectories ship back as transitions* — became the template the NVIDIA papers extend.

## Key technical contributions

1. MDP formulation of agent execution with semantic-variable state; one action = one LLM call's token sequence.
2. LightningRL hierarchical credit assignment that bypasses sequence-concatenation-with-masking entirely.
3. Training-Agent Disaggregation (TA Disaggregation) architecture: Lightning Server + Lightning Client over OpenAI-like API.
4. OpenTelemetry-native trace capture.
5. Multi-agent selective optimization (train any subset of agents naturally).

## BibTeX

```bibtex
@article{luo2025agent,
  title={Agent Lightning: Train ANY AI Agents with Reinforcement Learning},
  author={Luo, Xufang and Zhang, Yuge and He, Zhiyuan and Wang, Zilong and Zhao, Siyun and Li, Dongsheng and Qiu, Luna K. and Yang, Yuqing},
  journal={arXiv preprint arXiv:2508.03680},
  year={2025}
}
```
