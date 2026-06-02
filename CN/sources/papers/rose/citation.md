---
title: "ROSE —— 引用"
type: paper-citation
created: 2026-06-02
---

# ROSE: Rollout On Serving GPUs via Cooperative Elasticity for Agentic RL

- **arXiv**：[2605.06534](https://arxiv.org/abs/2605.06534)
- **代码**：未发布
- **作者**：Wei Gao\*†、Yuheng Zhao\*†、Dilxat Muhtar‡、Dakai An†、Xuchun Shang‡、Tianyuan Wu†、Lunxi Cao†、Shaopan Xiong‡、Weixun Wang‡、Ju Huang†、Teng Ma‡、Siran Yang‡、Jiamang Wang‡、Lin Qu‡、Bo Zheng‡、Wei Wang†（\*等贡献；†HKUST；‡阿里集团）
- **日期**：2026-05-07（v1）；2026-05-20（v2）
- **实现**：~5k 行 Python 在 ROLL + vLLM 0.10.0 + Megatron-LM + Mooncake v0.3.8 + Ray 之上
- **许可证**：CC-BY 4.0
- **Wiki 页面**：[[rose]]

## 为什么这篇论文重要

第一个 harvest **生产 serving GPU 容量**（通常空闲在 18.9% 计算 / 14.3% 内存利用率）给 agentic RL rollout 的系统论文，通过 cooperative elasticity。解决三个之前耦合的问题：(1) 保留 co-located rollout burst 下的 serving SLO，(2) 10-200 Gbps Ethernet 下的跨集群 weight sync，(3) 带 cache 局部性的弹性 rollout 分派。分配 overhead 比 spot/serverless RL 替代品好 **50-80×**。

## 核心技术贡献

1. **SLO-Safe Co-Serving Executor**，带 VMM 基础跨模型 KVC 共享、抢占式内存共享策略、dual-SLO 接纳控制（TTFT + TPOT slack）。
2. **Cross-Cluster Weight Transfer Engine**，带异步 Mooncake Store relay、异质 TP×PP 的 shard 感知映射、利用 **RL 权重 delta $\Delta W_t$ 中 >95% 稀疏性** 的稀疏感知 COO 压缩（新颖实证观察）。
3. **Elastic Rollout Scheduler**，带 turn 级路由（不是整 trajectory）和 cache-affinity placement。
4. **端到端吞吐提升**：GRPO 1.31-1.46×，DAPO 1.42-3.31×，AReaL 1.44-2.69× —— 零 P99 SLO 违反，分配 overhead 0.3-0.4%。

## 头号数字

| 对比 | 模型 | 吞吐提升 |
| --- | --- | -------- |
| vs ROLL-GRPO（固定）| Qwen3-8B FrozenLake | 1.31×（最多 2.16×）|
| vs ROLL-GRPO | Qwen3-32B ALFWorld | 1.46×（最多 1.76×）|
| vs ROLL-DAPO | Qwen3-8B | 1.42×（最多 4.82×）|
| vs ROLL-DAPO | Qwen3-32B | 3.31×（最多 4.39×）|
| vs AReaL（异步）| Qwen3-8B/32B | 1.44× / 2.69× |
| vs RLBoost+（spot）| Rollout 时间 | 1.20-1.26× 更快 |
| 分配 overhead | ROSE vs λRL | 0.3-0.4% vs 15.1-26.1% |

## BibTeX

```bibtex
@article{gao2026rose,
  title={ROSE: Rollout On Serving GPUs via Cooperative Elasticity for Agentic RL},
  author={Gao, Wei and Zhao, Yuheng and Muhtar, Dilxat and An, Dakai and Shang, Xuchun and Wu, Tianyuan and Cao, Lunxi and Xiong, Shaopan and Wang, Weixun and Huang, Ju and Ma, Teng and Yang, Siran and Wang, Jiamang and Qu, Lin and Zheng, Bo and Wang, Wei},
  journal={arXiv preprint arXiv:2605.06534},
  year={2026}
}
```
