---
title: "Aurora：基于 RL 的在线投机解码训练"
category: llm-inference
tags: [投机解码, 在线学习, 强化学习, sglang, 草稿模型, together-ai, 论文精读]
created: 2026-05-20
updated: 2026-05-20
status: growing
---

# Aurora：基于 RL 的在线投机解码训练

> [!abstract]+ TL;DR
> **Aurora**（Together AI，ICML 2026）把投机解码 draft model 的训练变成 **在线 RL 问题**，并让训练 *在线上服务过程中* 发生。一个 SGLang 推理服务器把每一次被接受 *和* 被拒绝的 token 提议都送进分布式 buffer；一个异步训练服务器从 buffer 里学习并把改进后的 draft 权重热替换回服务实例，不打断流量。两个直接结果：(1) **Day-0 上线** —— 完全未训练的 draft 也能立刻服务并在飞行中进步，去掉了几天的离线预训练瓶颈；(2) **持续自适应** —— draft 跟着生产流量的分布漂移走，分布切换后约 10K 请求内 acceptance length 就能恢复。结果：在 Qwen3-Coder-Next (FP8) 和 MiniMax M2.1 上从零开始就有 **1.21–1.45× 吞吐**，对静态 speculator 有 **1.25×** 提升。和 [[das-spec-rl|DAS]]（RL 训练 rollouts 场景下的等价问题）一起，是 2026 年 "speculator 从自己的流量中学习" 这条线的代表作。

> [!info]+ 论文元信息
> - **论文**：[arXiv:2602.06932](https://arxiv.org/abs/2602.06932) —— *Aurora: When RL Meets Adaptive Speculative Training*
> - **代码**：[github.com/togethercomputer/aurora](https://github.com/togethercomputer/aurora)
> - **模型**：Qwen3-Coder Spec、MiniMax M2.1/M2.5 Spec 在 Hugging Face
> - **作者**：Junxiong Wang*、Fengxiang Bie*、Jisen Li、Zhongzhu Zhou、…… Xiaoxia Wu、Chenfeng Xu（项目负责人）
> - **机构**：Together AI（主要）+ Stanford、CMU
> - **会议**：ICML 2026
> - **项目页**：https://aurora-spec-ai.github.io/

## 背景：投机解码的训练是 serving 瓶颈

[[speculative-decoding|投机解码]] 通过让小的 *draft model* 提议 K 个 token、大的 *target model* 在一次并行 forward 里验证，给出 2–3× 推理吞吐提升。增益的天花板是 **acceptance length (AL)** —— K 个候选里 target 接受了几个。AL 强烈依赖 draft 与 target 输出 **分布是否对齐**。

生产中要持续维持这个对齐很难：

1. **离线训出来的 draft 会老化**。在通用语料上预训练的 draft 在生产 prompt（代码、agent 轨迹、长 RAG）上会偏离 target。EAGLE-3 这类方法要花几天 GPU 时间预训一个 draft，target 换型号还得重训。
2. **部署后分布会漂**。这周流量 40 % 是代码，下周变 70 % agent 轨迹，旧 draft 的 AL 直接崩。
3. **没有反馈回路**。serving 栈手上有解决问题需要的数据 —— verifier 清清楚楚知道哪些 token 被接受、哪些被拒 —— 但这个信号被扔掉了。

[[das-spec-rl|DAS]] 在 *RL 训练* 场景（rollouts 就是负载）下部分解决了这个问题，把 draft 在 rollout policy 分布上在线训练。Aurora 把这个思路推广到 **任意生产推理流量**。

## 核心思想：speculator 作为策略，进异步 RL loop

Aurora 把 draft model 看作 RL 里的 **策略 π**：

- **动作**：发出 K-token 续写
- **环境**：target model + verifier
- **奖励信号**：哪些 token 被接受（`paccept`），哪些被拒（`pdiscard`）

这个信号 verifier 在每一次请求里 *本来就在生成* —— Aurora 只是不再扔掉它。被接受的和被拒的 branch 都流进共享的分布式 buffer，异步训练服务器消费 buffer 并通过 GPU-aware RPC 把更新后的 draft 权重推回推理服务器。没有请求被训练阻塞，没有训练步被服务阻塞。

这个表述带来一个比之前 online-distillation 工作更强的论断：**在线 *从零开始* 训练能超过离线预训练 speculator**。可以根本不要那个几天的预训练阶段 —— draft 冷启动，让生产流量来训它。

## 工作机制

### 架构：SGLang 服务器 + 异步训练服务器

```
┌─────────────────────────────────────┐    ┌─────────────────────────────┐
│  SGLang 推理服务器                  │    │  异步训练服务器             │
│                                     │    │                             │
│  target + draft π                   │    │  收集 accept / reject       │
│  → verifier → 发 token              │    │  → 计算 loss                │
│  → log (Q, accepted, rejected)      ├───►│  → 更新 π                   │
│                                     │    │  → push 新权重              │
│  ◄── 热替换 draft 权重 ─────────────┤    │                             │
└─────────────────────────────────────┘    └─────────────────────────────┘
```

两边都在持续跑。推理服务器永远不停下来等训练；训练服务器永远不会阻塞在某次 forward。权重热替换通过 GPU-aware RPC 实现 —— draft model 参数 buffer 在线上请求飞行中被原地更新。

### 双项损失

对每一次投机步，verifier 把 K 个候选 token 切成 *被接受的前缀* 和 *被拒的后缀*。Aurora 同时从两边训练：

- **Accept 损失** —— 在被接受 token 上做交叉熵。标准的模仿学习目标：把 `pdraft` 拉向 target 选的那一个。

- **Reject 损失** —— KL 项，**把** `pdraft` **从被拒分支推开**，方法是作者称为 **Discard Sampling** 的采样方式。直觉：如果 target 拒了一个 branch，draft 下次就应该在那里放更少的概率，哪怕 target 自己没有明示替代方案。

组合目标：

$$
\mathcal{L} = \mathbb{E}_{x\sim p_{\text{accept}}}\!\left[ \mathrm{KL}(p_{\text{target}} \,\|\, p_{\text{draft}}) \right] + \lambda_{\text{discard}}\, \mathbb{E}_{x\sim p_{\text{discard}}}\!\left[ \mathrm{KL}(p_{\text{target}} \,\|\, p_{\text{draft}}) \right]
$$

Reject 项是非平凡的贡献。朴素的 online distillation 只从被接受 token（正样本）学习。Aurora 把被拒 branch 当成一等公民的学习信号。

### Tree Attention：把投机分支批量化

投机解码产出的是 token 续写的 *一棵树*，不是一条线。一次请求可能提议多个 branch，verifier 在每一层都拒一些。朴素地，对这棵树算 loss 要做多次 forward/backward —— 每个 branch 一次。

Aurora 的 **Tree Attention** 用自定义 attention mask 让 *单次* batched forward/backward 同时覆盖所有被接受 *和* 被拒的 branch。这是 kernel 层的关键工程 —— 没有它，训练吞吐跟不上服务吞吐，训练服务器会越落越远。

## 实验

### Day-0：冷启动 draft

| Target 模型 | 相对无投机吞吐 | batch=8 时 AL |
| ----------- | -------------- | ------------- |
| Qwen3-Coder-Next (FP8) | **1.21×** | 3.0 |
| MiniMax M2.1 | **1.45×** | 2.8 |

这些数字是 *从未训练过* 的 draft，只看了生产流量。没有任何离线预训练。重点：AL 在几小时（不是几天）服务后就收敛到可用值。

### 分布漂移后的自适应

在 Qwen3 / Llama3 上混合代码 + agent 流量：

- 比 *静态* speculator（不更新的那种）有 **1.25×** 提速。
- 注入一个分布切换后，AL 下降然后在 **约 10K 请求** 内恢复到切换前水平。

这是更强的论断：Aurora 比 prior art 的优势 *不仅仅* 是初始训练方案不同，而是 draft *会持续* 改进、*会从漂移中恢复*，而静态训练的对手 AL 一旦下降就回不来。

### 评估里缺的东西

- **没和 [[das-spec-rl|DAS]] 比** RL-rollout 工作负载，明显有重合却没对比。
- **代码偏多的 benchmark 组合**。评估倾向代码工作负载（Qwen3-Coder、MiniMax 也是代码强项）。长上下文摘要、数学推理、多模态都没跑。
- **持续分布漂移下的吞吐数据没有**。"10K 请求恢复" 是注入一次切换的结果，工作负载不断震荡会怎样？
- **生产规模 batch size 扫描没有**。报告 batch=8 时 AL=3，batch=32 或 batch=64（更常见的生产 batch size）下 AL 是多少？

## 优势与局限

**优势：**

- **去掉离线预训练阶段**。这是真实成本 —— 一个新 target 模型上 draft 预训练几天 GPU，是新模型上线总推理基础设施成本里不小的一块。Aurora 直接砍掉这条预算线。
- **分布漂移下自愈**。生产流量结构会变。静态训练的 draft 会衰减，Aurora 在持续进步。
- **用 verifier 本来就在产生的信号**。不需要额外 forward，不需要额外标签。每个已有的投机解码部署里，reject 信号都在地上 —— 没人捡。

**局限：**

- **训练-服务耦合**。飞行中热替换权重在运维上有风险。一次坏的梯度更新可能拖垮 AL，而 draft 没法干净回滚 —— 除非把整个权重 buffer 一起回滚。论文没讨论安全机制（梯度裁剪、AL 回归回滚、A/B 分流训练）。
- **GPU-aware RPC 依赖**。Aurora 的权重热替换需要推理和训练服务器共享 GPU fabric。对那些训练在独立 cluster 跑的部署，要补这条 RPC 不简单。
- **收敛依赖流量量**。Day-0 行得通是 *因为* 流量足够多能驱动学习。QPS 低的部署里，draft 可能学不够快，反而打不过通用离线预训练的 baseline。
- **隐式假设：target 不变**。如果 target 自己换了（新发布、微调变体），draft 得从头学。系统能处理 *prompt* 的分布漂移，处理不了 *target* 的换型。

## 启示

Aurora 是 2026 年第二篇（前一篇是 [[das-spec-rl|DAS]]）说同一件结构性事情的论文：**投机解码应该是个闭环系统，不是开环系统**。Draft 是个 *学出来的策略*，应该从生产中学，而不是离线烘焙好的死东西。

后面两条轨迹大概率：

1. **推理引擎会吸收这套 loop**。SGLang 已经是 Aurora 的底座；预计 vLLM 和 TRT-LLM 在 6-12 个月内会出对标的 online-spec 特性。"训练服务器作为 sidecar" 是最小阻力路径。
2. **"Draft 预训练" 这个市场会萎缩**。一些公司（Together 自己，但也包括卖按 target 预训 draft 服务的小供应商）把离线预训练 draft 作为服务卖。Aurora 的论断 —— *从零在线训能打过预训练* —— 如果在代码之外的工作负载也泛化得动，整条业务线被釜底抽薪。

最被低估的是 **Tree Attention** 这个 kernel。它是让其它部分变可行的脏活基础设施。预计这个 kernel 本身（脱离 RL 框架）会被搬到其它投机解码栈里，不管它们要不要 Aurora 的 RL loop。

## 相关阅读

- [[speculative-decoding|投机解码]] —— 更广义的技术，AL / 分布匹配的基础概念
- [[das-spec-rl|DAS]] —— **RL 训练 rollouts** 场景下的等价系统（不是生产推理）。Aurora 和 DAS 工程实现大概率会收敛，尽管目标负载不同
- [[sglang|SGLang]] —— Aurora 的推理后端；"训练服务器作为 sidecar" 模式靠 SGLang 的连续 batching 原语

## 参考文献

- Aurora 论文，arXiv:2602.06932 —— [论文](https://arxiv.org/abs/2602.06932)、[项目页](https://aurora-spec-ai.github.io/)、[代码](https://github.com/togethercomputer/aurora)
- **Online Speculative Decoding** (arXiv:2310.07177, 2023) —— 只做知识蒸馏的前身；Aurora 推广到 RL
- **EAGLE-3** (arXiv:2503.01840, 2025) —— 强离线预训练 speculator 的 baseline，Aurora 对标的对象
- [SGLang at NeurIPS 2024](https://arxiv.org/abs/2312.07104) —— Aurora 扩展的服务基础设施
