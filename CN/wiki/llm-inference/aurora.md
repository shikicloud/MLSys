---
title: "Aurora：基于 RL 的在线投机解码训练"
category: llm-inference
tags: [投机解码, 在线学习, 强化学习, sglang, 草稿模型, together-ai, 论文精读]
created: 2026-05-20
updated: 2026-05-21
status: growing
paper: arXiv:2602.06932
code: https://github.com/togethercomputer/aurora
---

# Aurora：基于 RL 的在线投机解码训练

> [!info] 论文元信息
> - **论文**：[arXiv:2602.06932](https://arxiv.org/abs/2602.06932) —— *Aurora: When RL Meets Adaptive Speculative Training*
> - **代码**：[github.com/togethercomputer/aurora](https://github.com/togethercomputer/aurora)
> - **模型**：Qwen3-Coder Spec、MiniMax M2.1/M2.5 Spec 在 Hugging Face
> - **作者**：Junxiong Wang*、Fengxiang Bie*、Jisen Li、Zhongzhu Zhou、Yinghui Liu、Yubo Wang、Avner May、Sri Yamamatra、Tri Dao、Percy Liang、Ce Zhang、Ben Athiwaratkun、Shuaiwen Leon Song、Chenfeng Xu、Xiaoxia Wu
> - **机构**：Together AI（主要）+ Stanford、CMU
> - **会议**：ICML 2026
> - **项目页**：https://aurora-spec-ai.github.io/

---

## 摘要（2 分钟读完这一节就够）

**它是什么**。Aurora（Together AI，ICML 2026）是一个统一的训练-服务框架，把投机解码 draft model 的训练变成 **在线 RL 问题**，并让训练 *在线上服务过程中* 发生。一个 SGLang 推理服务器把每一次被接受 *和* 被拒绝的 token 提议都送进分布式 buffer；一个异步训练服务器从 buffer 里学习，并把改进后的 draft 权重热替换回服务实例，不打断流量。

**核心思想**。把 draft model 看成 **策略**，用 verifier 的 accept/reject 信号在线 RL 训练 —— *这个信号 verifier 本来就在每次请求里免费生成*。三个支柱：

1. **异步 RL loop** —— Inference Server 和 Training Server 共享 data buffer + 热替换 RPC，时间上完全解耦。
2. **双项损失 + Discard Sampling** —— 接受 token 把 `pdraft` 拉向 `ptarget`；拒绝 token *把它推开*（一个 KL 项，大多数在线蒸馏方法都跳过了它）。
3. **Tree Attention** —— 自定义 attention mask，让单次 batched forward/backward 同时覆盖投机树里所有被接受 *和* 被拒的分支。

少任一支柱：draft 要么跟不上服务、要么只学正样本、要么训练吞吐跟不上。

**头条结果**。两个直接后果：

| 场景 | 相对无投机吞吐 | Acc. Length (BS=8) |
| ---- | -------------- | ------------------ |
| **Day-0** 冷启动，Qwen3-Coder-Next (FP8) | **1.21×** | 3.0 |
| **Day-0** 冷启动，MiniMax M2.1 | **1.45×** | 2.8 |
| 域漂下相对 **静态** speculator | **1.25×**（静态对比） | ~10K 请求恢复 |

Day-0 数字来自 **未训练** 的 draft，只看了生产流量 —— 没有任何离线预训练。域漂之后 1.25× 超过静态训练对手的论断最强：Aurora 在持续进步，静态 draft 在持续衰减。

**为什么这重要**。

- **去掉 draft 预训练**。每个新 target 模型上几天的离线 draft 预训练是真实成本 —— Aurora 砍掉这条预算线。
- **域漂下自愈**。生产流量结构会变；静态 draft 会衰减；Aurora 在复利。
- **用 verifier 本来就在产生的信号**。每个现有投机解码部署都在扔掉 reject 信号 —— Aurora 把它捡起来。
- **12 个月预测**。vLLM 和 TRT-LLM 上对标的 online-spec 特性；"draft 预训练即服务" 这个市场被挤压。

---

# 深度部分（往下展开细节）

上面摘要是 executive 层。下面是给愿意细读架构和代码的人准备的。

## 背景：投机解码的训练是 serving 瓶颈

[[speculative-decoding|投机解码]] 通过让小的 *draft model* 提议 K 个 token、大的 *target model* 在一次并行 forward 里验证，给出 2–3× 推理吞吐提升。增益的天花板是 **acceptance length (AL)** —— K 个候选里 target 接受了几个。AL 强烈依赖 draft 与 target 输出 **分布是否对齐**。

生产中要持续维持这个对齐很难：

1. **离线训出来的 draft 会老化**。在通用语料上预训练的 draft 在生产 prompt（代码、agent 轨迹、长 RAG）上会偏离 target。EAGLE-3 这类方法要花几天 GPU 时间预训一个 draft，target 换型号还得重训。
2. **部署后分布会漂**。这周流量 40 % 是代码，下周变 70 % agent 轨迹，旧 draft 的 AL 直接崩。
3. **没有反馈回路**。serving 栈手上有解决问题需要的数据 —— verifier 清清楚楚知道哪些 token 被接受、哪些被拒 —— 但这个信号被扔掉了。

[[das-spec-rl|DAS]] 在 *RL 训练* 场景（rollouts 就是负载）下部分解决了这个问题，把 draft 在 rollout policy 分布上在线训练。Aurora 把这个思路推广到 **任意生产推理流量**。

| 系统 | Draft 训练 | 适应漂移 | 工作负载 |
| ---- | ---------- | -------- | -------- |
| EAGLE-3 / 普通 SD | 离线、部署后静态 | 不 | 通用推理 |
| Online Speculative Decoding (2023) | 在线 KD，只学接受 | 部分 | 通用推理 |
| [[das-spec-rl\|DAS]] | 在线 RL | 是 | RL 训练 rollouts |
| **Aurora** | **在线 RL，接受 + 拒绝** | **是** | **生产推理** |

## 三大组件详解

Aurora 框架由两个互相耦合的 server 通过共享 data buffer 连接；loss 有两项；kernel 让树形训练可行。论文图展示了高层放置。

![Aurora 统一训练-服务框架（论文 Fig. 1）](CN/wiki/llm-inference/aurora-figs/system-architecture.png)

Inference Server 持有固定 target（verifier）和热替换 draft（speculator），跑投机解码循环，把 *所有* accept/reject 分支流到分布式 Data Buffer。Training Server 异步拉 batch，跑双项损失，通过 off-policy 更新把新 speculator 推回 —— 整个过程从不暂停 serving。

### 组件 1 —— 异步 RL loop + 热替换权重

两个 server 持续跑。推理端永远不停下来等训练；训练端永远不阻塞在服务的某次 forward。权重热替换通过 GPU-aware RPC 实现 —— draft 的参数 buffer 在线上请求飞行中被原地更新。

最小架构示意：

```
┌─────────────────────────────────────┐    ┌─────────────────────────────┐
│  SGLang 推理服务器                  │    │  异步训练服务器             │
│                                     │    │                             │
│  target (verifier) + draft π        │    │  pull (accept, reject)      │
│  → verify → 发 token                │    │  → 双项损失                 │
│  → log (Q, accepted, rejected)      ├───►│  → 更新 π                   │
│                                     │    │  → push 新 draft            │
│  ◄── 热替换 draft 权重 ─────────────┤    │                             │
└─────────────────────────────────────┘    └─────────────────────────────┘
```

**Day-0 上线**。因为训练 *在线上服务过程中* 发生，draft 可以 *未初始化* 启动并在飞行中进步。论文给出具体数字：冷启动 speculator 在几小时（不是几天）服务后 AL 就收敛到可用值。

**热替换是真实工程负担**。Aurora 用 `torch.distributed.rpc` 走 TensorPipe 做 GPU 直传，加可扩展 CUDA 显存段避免碎片。训练服务器持有线程安全的传输数据缓存，backward 跑在之前传输的（稳定的）micro-batch 上而不是 in-flight 那个。

### 组件 2 —— 双项损失：在 reject token 上做 Discard Sampling

对每一次投机步，verifier 把 K 个候选 token 切成 *被接受的前缀* 和 *被拒的后缀*。Aurora 同时从两边训练：

- **Accept 项** —— 在被接受 token 上做 KL `p_target || p_draft`。标准模仿：把 `p_draft` 拉向 target 选的那一个。
- **Reject 项** —— 在 *被拒* 分支上做 KL，方法是作者称为 **Discard Sampling** 的采样方式。直觉：如果 target 拒了一个 branch，draft 下次就应该在那里放更少的概率，哪怕 target 自己没有明示替代方案。

组合目标：

$$
\mathcal{L} = \mathbb{E}_{x \sim p_{\text{accept}}} \left[\, \mathrm{KL}(p_{\text{target}} \,\|\, p_{\text{draft}}) \,\right] + \lambda_{\text{discard}} \, \mathbb{E}_{x \sim p_{\text{discard}}} \left[\, \mathrm{KL}(p_{\text{target}} \,\|\, p_{\text{draft}}) \,\right]
$$

朴素的 online distillation 只从被接受 token（正样本）学习。Aurora 把被拒分支当成一等公民的学习信号 —— [[#头条证据]] 里的消融显示在域漂场景下，对 Static 的优势就来自这一项。

### 组件 3 —— Tree Attention：把投机分支批量化

投机解码产出的是 token 续写的 *一棵树*，不是一条线。一次请求可能提议多个 branch，verifier 在每一层都拒一些。朴素地，对这棵树算 loss 要做多次 forward/backward —— 每个 branch 一次。

Aurora 的 **Tree Attention** 用自定义 attention mask 让 *单次* batched forward/backward 同时覆盖所有被接受 *和* 被拒的 branch。图里展示了构造：完整 token 序列是接受前缀 + 被拒兄弟的并集，attention mask 是块结构的 —— 每个 branch 只 attend 它自己的祖先。

![Tree Attention mask：单次 forward 覆盖接受和拒绝分支（论文 Fig. 2）](CN/wiki/llm-inference/aurora-figs/tree-attention.png)

这是 kernel 层的关键工程 —— 没有它，训练吞吐跟不上服务吞吐，训练服务器会越落越远。预计这个 kernel（脱离 RL 框架）会被搬到其它投机解码栈里，不管它们要不要 Aurora 的 RL loop。

### 辅助机制（可跳读）

> [!note]- 异步同步策略 —— 想调 push 频率就展开
> Aurora 把 policy 刷新间隔（多久把新 speculator 热替换到 Inference Server）开成可调参数。论文 Figure 5 扫了这个 trade-off：
>
> - **Aggressive**（每 48 请求）：post-shift 适应更高（新分布上 draft 追得更快），但同步开销大 —— 净吞吐受损。
> - **Lazy**（每 1600 请求）：开销最低，但损失部分适应收益。
> - **Moderate**（约每 80 请求）：强 Pareto 点 —— 保住大部分适应能力同时给出最佳总体吞吐。
>
> 系统默认 moderate，并通过 config 暴露这个 knob。

> [!note]- Loss 变体扫描 —— 关心 RKL/FKL/NTP 就展开
> 论文 Section 5 消融了几种训练目标：Frozen Draft（Static Baseline，不在线更新）、Aurora (FKL) 用 forward KL、Aurora (RKL) 在接受 token 上用 reverse KL、Aurora (RKL + NTP) 加一个接受 token 上的辅助 next-token-prediction loss、Aurora (w discard) 在被拒分支上做 Discard Sampling。完整组合 —— **RKL + tree/discard + NTP** —— 在 Figure 6 里一致胜出，[[#头条证据]] 里的消融用的就是这些标签。

## 头条证据

**配置**。三种配置扫模型尺寸和服务条件：

1. **Day-0 冷启动**：Qwen3-Coder-Next (FP8) 和 MiniMax M2.1（230B Transformer-MoE）。
2. **Mixed streams** —— Day-0 在交替域（数学、代码、金融、指令）上适应。
3. **Ordered streams** —— Day-0 在尖锐域切换下适应，压测恢复能力。

算法：RKL + tree/discard + NTP。Lookahead K=5（Qwen3-8B）、K=10（Llama3.1-8B）。

**头条数字**。Aurora 的 Day-0 未训练 speculator 达到：

| Target 模型 | 相对无投机吞吐 | BS=8 时 AL |
| ----------- | -------------- | ---------- |
| Qwen3-Coder-Next (FP8) | **1.21×** | 3.0 |
| MiniMax M2.1 | **1.45×** | 2.8 |

论文原话："在 trained 模型上面起初会跌，但训练一段时间后能取得更好结果" —— 即 Day-0 在几小时内追上 *预训练* 的 baseline。

**域漂下的适应**。Ordered streams 上 Aurora vs Static：

![Ordered streams：未训练 speculator 的 Day-0 适应 vs Static + No-Speculator（论文 Fig. 4）](CN/wiki/llm-inference/aurora-figs/ordered-streams.png)

读法：x 轴每个 step 约 1K 服务请求。Aurora (Trained，蓝) 全程压在 Static (绿) 上方，每次尖锐切换（约 10K/20K/30K 处的悬崖）后约 10K 请求内恢复，并且 *在预训练之上* 继续改进而不是停滞。

> [!success] 漂移恢复数字
> 在 ordered streams 上，Aurora 相对静态 speculator 给出约 **1.25× 提速**，并在强制分布切换后约 **10K 请求** 内恢复到切换前的 AL。静态 draft 不能恢复 —— AL 一旦下降就回不来。

**关键消融：在 reject token 上做 Discard Sampling**。移除拒绝项（Aurora 只 RKL）会留 headroom 在桌上；加上 Discard Sampling（Aurora RKL + tree/discard + NTP）把这个 gap 关掉。Figure 6 在 Qwen3-8B-Instruct 上展示这一点：

![Lookahead 足够大时 Discard Sampling 关掉 gap（论文 Fig. 6）](CN/wiki/llm-inference/aurora-figs/discard-tokens-ablation.png)

论文表述：discard tokens 何时有用受 lookahead 闸门控制。Lookahead 小（5）时，预训练 speculator 已经接近其原生 AL 上限 —— discard tokens 没有发挥空间。Lookahead 10 时 headroom 打开，被拒分支的信号才有价值。这是个微妙的 scope 论断，部署前值得内化。

**扩展到 230B target**。MiniMax M2.1（BS=4 / BS=8）：

![MiniMax M2.1 上的可扩展性 —— Aurora (Scratch) vs No-Speculator（论文 Fig. 9）](CN/wiki/llm-inference/aurora-figs/scalability-minimax.png)

> [!example]- 全部实验结果（展开）
> **端到端吞吐数字**，扫不同 batch size（论文 Table 1）：
>
> | 模型 | 配置 | Speedup | Acc. Length |
> | ---- | ---- | ------- | ----------- |
> | MiniMax M2.1 | BS=4，H100 GPUs w/ TP=4 | 1.45×（Scratch） | 2.8 |
> | Qwen3-Coder-Next (FP8) | BS=4 | 1.21×（Scratch） | 3.0 |
> | Llama3.1-8B | K=10, lookahead | discard-sample 收益最大 | ~3.8 |
> | Qwen3-8B-Instruct | K=5 | K=10 时差距收紧 | ~3.0 |
>
> **Batch size 敏感性**（论文 Figure 8）：在 BS=4 和 BS=12 测试。Aurora 相对 static 的 AL 收益在不同 batch size 下保持，但 speedup 倍率在更大 batch 下收缩 —— 因为 target 已经摊销/效率更好，投机开销在 pipeline 里占比变大。AL 改进；净 speedup 不那么显著。
>
> **Discard tokens 上的 top-k 策略**（论文 Figure 7）：在全部 discard token vs top-k（k=0、10、50）上训练，差距 *只是* 微小。Top-k 省内存又不掉性能；不是一个脆弱超参。
>
> **异步策略刷新调度**（论文 Figure 5）：aggressive（48 req）提升适应但因同步开销切吞吐；moderate（80 req）是 Pareto 最优；lazy（1600 req）丢失适应收益。默认 = moderate。

### 评估里缺的东西

- **没和 [[das-spec-rl|DAS]] 正面比** RL-rollout 工作负载，明显有重合却没对比。
- **代码偏多的 benchmark 组合**。评估倾向代码工作负载（Qwen3-Coder、MiniMax 也是代码强项）。长上下文摘要、数学推理、多模态都没跑。
- **持续震荡下的吞吐数据没有**。"约 10K 请求恢复" 是注入一次切换的结果，工作负载不断震荡会怎样？
- **生产规模 batch size 扫描有限**。报告了 BS=4 和 BS=12；BS=32 / BS=64（更常见的生产 batch size）没测。

## 优势与局限

最突出的强项是三个组件 —— 异步 loop、双项损失、Tree Attention —— 每个都解决了 prior systems 里真实的失败模式，每个都有可量化证据落地。

论文说服力不足的地方：

- **训练-服务耦合在运维上有风险**。飞行中热替换权重：一次坏的梯度更新可能拖垮 AL，而 draft 没法干净回滚 —— 除非把整个权重 buffer 一起回滚。论文没讨论安全机制（梯度裁剪、AL 回归回滚、A/B 分流训练）。
- **GPU-aware RPC 依赖**。热替换需要 Inference 和 Training Server 共享 GPU fabric。训练在独立 cluster 跑的部署要补这条 RPC 不简单。
- **收敛依赖流量量**。Day-0 行得通是 *因为* 流量足够多能驱动学习。低 QPS 部署里 draft 可能学不够快，打不过通用离线预训练的 baseline。
- **隐式假设：target 不变**。如果 target 自己换了（新发布、微调变体），draft 得从头学。系统能处理 *prompt* 的分布漂移，处理不了 *target* 的换型。

> [!warning] Discard tokens 只在 lookahead 够大时有用
> Figure 6 显示 lookahead 5 时 discard tokens 给出 *零* 收益 —— 因为预训练 speculator 已经接近其 AL 上限。胜利只在 lookahead 10（有 headroom）时出现。这是个有用的 scope 警告；选错 K 的部署会看到 Aurora 最新颖的部分（Discard Sampling）一点都没买到。

## 启示

Aurora 是 2026 年第二篇（前一篇是 [[das-spec-rl|DAS]]）说同一件结构性事情的论文：**投机解码应该是个闭环系统，不是开环系统**。Draft 是个 *学出来的策略*，应该从生产中学，而不是离线烘焙好的死东西。

后面两条轨迹大概率：

1. **推理引擎会吸收这套 loop**。SGLang 已经是 Aurora 的底座；预计 vLLM 和 TRT-LLM 在 6–12 个月内会出对标的 online-spec 特性。"训练服务器作为 sidecar" 是最小阻力路径。
2. **"Draft 预训练即服务" 这个市场会萎缩**。Together 自己以及一些小供应商把按 target 预训的 draft 作为服务卖。Aurora 的论断 —— *从零在线训能打过预训练* —— 如果在代码之外的工作负载也泛化得动，整条业务线被釜底抽薪。

最被低估的是 **Tree Attention** 这个 kernel。它是让其它部分变可行的脏活基础设施。预计这个 kernel 本身（脱离 RL 框架）会被搬到其它投机解码栈里，不管它们要不要 Aurora 的 RL loop。

## 源码与指针

```bash
git clone https://github.com/togethercomputer/aurora
# 仓库包含：
#   - SGLang fork，带 Inference Server 热替换 hook
#   - Training Server，带 Tree Attention kernel + Discard Sampling
#   - Qwen3-Coder-Next 和 MiniMax M2.1 的 example launch config
```

预训练好的 speculator 在 Hugging Face 上发布于 `togethercomputer/Tougyuan/qwen3_8b_eagle3` 及类似 slug（MiniMax 变体）。

值得先读的文件（路径示意 —— 名字可能变）：

| 路径 | 作用 |
| ---- | ---- |
| `aurora/inference/sglang_patch.py` | 进 SGLang draft 模型的热替换 hook |
| `aurora/training/server.py` | 异步训练循环，batch accept/reject |
| `aurora/training/loss.py` | 双项损失 + Discard Sampling |
| `aurora/kernels/tree_attention.py` | 投机树的自定义 attention mask |
| `aurora/rpc/torch_distributed.py` | 基于 TensorPipe 的热替换 RPC |

## 相关阅读

- [[speculative-decoding|投机解码]] —— 更广义的技术，AL / 分布匹配的基础概念。
- [[das-spec-rl|DAS]] —— **RL 训练 rollouts** 场景下的等价系统（不是生产推理）。Aurora 和 DAS 工程实现大概率会收敛，尽管目标负载不同。
- [[sglang|SGLang]] —— Aurora 的推理后端；"训练服务器作为 sidecar" 模式靠 SGLang 的连续 batching 原语。
- [[kv-cache-optimization|KV cache 优化]] —— 正交的 serving 侧吞吐轴；投机解码 *乘上* 一个倍率，KV cache 优化让这个倍率在长上下文下保持得住。

## 参考文献

- Aurora 论文，arXiv:2602.06932 —— [论文](https://arxiv.org/abs/2602.06932)、[项目页](https://aurora-spec-ai.github.io/)、[代码](https://github.com/togethercomputer/aurora)
- **Online Speculative Decoding** (arXiv:2310.07177, 2023) —— 只做知识蒸馏的前身；Aurora 推广到 RL。
- **EAGLE-3** (arXiv:2503.01840, 2025) —— 强离线预训练 speculator 的 baseline，Aurora 对标的对象。
- [SGLang at NeurIPS 2024](https://arxiv.org/abs/2312.07104) —— Aurora 扩展的服务基础设施。
