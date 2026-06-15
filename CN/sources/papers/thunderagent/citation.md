---
title: "ThunderAgent —— 引用"
type: paper-citation
created: 2026-06-15
---

# ThunderAgent: A Simple, Fast and Program-Aware Agentic Inference System

- **arXiv**：[2602.13692](https://arxiv.org/abs/2602.13692)（v1 2026-02-14；v2 2026-03-10）
- **会议**：**ICML 2026 Spotlight（top 2.2%）**
- **代码**：[ThunderAgent-org/ThunderAgent](https://github.com/ThunderAgent-org/ThunderAgent)（MIT 协议，vLLM + SGLang 后端，OpenAI 兼容 API）。原始一作仓库：[HaoKang-Timmy/ThunderAgent](https://github.com/HaoKang-Timmy/ThunderAgent)。
- **作者**：Hao Kang\*（Georgia Tech）、Ziyang Li\*（Individual Researcher）、Xinyu Yang\*（CMU）、Weili Xu\*（UIUC）、Yinfang Chen（UIUC）、Junxiong Wang（Together AI）、Beidi Chen（CMU）、Tushar Krishna（Georgia Tech）、Chenfeng Xu（Together AI）、Simran Arora（Together AI）。\* 表共同一作。
- **机构**：Georgia Tech、Carnegie Mellon University、UIUC、Together AI
- **通讯**：`hkang342@gatech.edu`
- **许可证**：MIT
- **Wiki 页面**：[[thunderagent]]

## 为什么这篇论文重要

**第一个把 multi-turn workflow 抽象成一等公民 `agentic program`、按 program 粒度（而非 per-request）调度的 agent-serving 系统。** [[continuum|Continuum]]（预测式 TTL）的同期竞争者，用反应式 thrashing-monitor 设计；**在所有 6 个 benchmark 上击败 Continuum**，包括随机工具的 ToolOrchestra workload（Continuum 在那里降到 vLLM 以下）。ThunderAgent 也是 [[mori|MORI]] 两层 GPU+CPU idleness offloader 跑在其上的系统层（MORI 的 ~3,300 行实现在 ThunderAgent + 500 行实现在 SGLang HiCache 上）。

## 核心技术贡献

1. **Agentic program 抽象** —— $P = \langle ID, c, \mathcal{T}, \mathcal{L}, \tau, s \rangle$ 统一 KV cache 占用、工具环境、节点 placement、执行阶段（Reasoning/Acting）、调度状态（Active/Paused/Terminated）。
2. **周期性 thrashing 监视器**，每 $\Delta t = 5$ s 对每个 backend 评估 $C_{\text{total}} < \sum_p c_p$；Pause/Restore 原语按 $1/c_P + \mathbb{1}(\tau)$ 排序 —— *最短 context 先驱*、*Reasoning 先复*。
3. **时间衰减函数** $f(t) = 2^{-t}$ 作用于 Acting program 的 token 权重（在 memoryless 工具时长下证明最优）；Continuum 预测式 TTL 的反应式对偶。
4. **Lemma 4.1** —— $\text{Cost}_{\text{recompute}} \propto c_i^2$（context 长度的二次方）；为贪心 shortest-first 驱逐提供形式化依据。
5. **跨 DP 节点的全局 program-aware waiting queue** —— 替代 per-node KV-aware 路由；paused KV 节点无关，所以任何有容量的 backend 都能 restore；上界 $\text{Cost}_{\text{unused}} < c_{\min} \cdot \Delta t$。
6. **Hook-based 垃圾回收** 对工具资源 —— 生命周期 hook 在 `Terminated` 状态拆 sandbox / 网络端口；保持 active 磁盘用量近常数。
7. **异步环境准备** —— 给高优先级 restoring program 在 LLM 阶段时就并行启动 Docker 容器 / 装依赖；隐藏 29.9-47.2 s 设置延迟。
8. **OpenAI 兼容 API** 只多一个 `program_id` 参数 —— 客户端集成trivial。

## 标志数字

- **Serving 吞吐**：跨 6 个 benchmark（SWEAgent、OpenHands、ToolOrchestra、ScienceAgent × GLM-4.6 / Qwen3-235B / Qwen3-8B）**对 vLLM 1.48–3.58×**。
- **vs Continuum**：6 个 benchmark 全胜，1.17–3.31× 更快；在 ToolOrchestra-HLE 随机工具 workload 上 **Continuum 输给 vLLM（0.65×）**、ThunderAgent **赢**（1.48× vs vLLM）。
- **RL rollout**：在 2× 8×H100 上比 prior SOTA（vLLM + SGLang Gateway）**改善 1.79–3.92×**。
- **磁盘内存**：节省多达 **4.2×** —— active 用量近常数 vs baseline 线性增长在 ~250 个 workflow 时溢出 2 TB 容量。
- **KV cache 命中率**：可预测工具 workload（a,b,d,e）在 192 并发 workflow 下 ~100%；Continuum 在高并发下从 >90% 跌到 ~60%。
- **内存不平衡**：baseline vLLM+SGLang 90 分钟 RL rollout 跨 DP 节点峰值 51% 不平衡；ThunderAgent 上界 $c_{\min} \cdot \Delta t$。
- **Baseline 里 KV cache thrashing 的影响**：E2E latency 暴涨 7.14×；工具环境准备时间随 workflow 数 24 → 96 从 29.9 涨到 47.2 s。

## BibTeX

```bibtex
@inproceedings{kang2026thunderagent,
  title={ThunderAgent: A Simple, Fast and Program-Aware Agentic Inference System},
  author={Kang, Hao and Li, Ziyang and Yang, Xinyu and Xu, Weili and Chen, Yinfang and Wang, Junxiong and Chen, Beidi and Krishna, Tushar and Xu, Chenfeng and Arora, Simran},
  booktitle={Proceedings of the 43rd International Conference on Machine Learning},
  note={Spotlight},
  year={2026}
}
```
