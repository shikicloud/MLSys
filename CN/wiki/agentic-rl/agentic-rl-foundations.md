---
title: "Agentic RL 入门 Foundations —— 导航 hub"
category: agentic-rl
tags: [agentic-rl, hub, foundations, onboarding, family-overview]
created: 2026-05-26
updated: 2026-05-26
status: mature
---

# Agentic RL Foundations —— 入门导航 hub

> [!abstract]+ 这一页是什么
> Agentic-RL 入门的导航 hub。把代表性论文和基础设施按对新手最有用的顺序串起来，附 4 阶段阅读路径、每篇重要论文的迷你摘要、各 wiki 页面的指针。
>
> **新人**入门 agentic RL 想要一条策展过的路径，用这一页。深度细节请跟着 wiki 链接进 —— 这里提到的每篇论文都有（或将有）自己的页面。

> [!info] 2026 年 5 月的 agentic RL 状态快照
> - **算法侧**：PPO 和 GRPO 仍是主力 RL 算法；没有根本上新的算法取代它们。变体（DAPO、RLOO、Dr.GRPO、KDRL）在边缘调优
> - **架构侧**：领域围绕 "rollout-as-a-service" + "harness-as-blackbox" pattern 收敛。[[polar|Polar]]（NVIDIA，2026-05）是当前 SOTA
> - **前沿任务**：SWE-Bench（code）、WebArena（browser）、OSWorld（OS）、GAIA（通用 agent）。多工具组合、长 horizon planning、computer use
> - **前沿方法**：纯 outcome-reward RL 在中小规模仍工作；process rewards (PRMs) 和 LLM-as-judge 是 frontier 规模的开放问题

## 什么是 agentic RL

**Agentic RL** 是 LLM 的 RL 微调，但策略不只是生成文本，而是**跟有状态环境多轮交互**。环境可以是：

- 搜索引擎（[[search-r1]]）
- 代码执行沙箱（SWE agent）
- 浏览器（BrowserGym / Mind2Web）
- 操作系统（OSWorld）
- 多工具组合（计算器 + 搜索 + 代码）

跟普通 RLHF 的标准成分对比：

| 方面 | 普通 RLHF | Agentic RL |
| ---- | -------- | ---------- |
| Rollout | 单次 LLM forward | 多轮（LLM → 工具调用 → observation → LLM → ...） |
| 环境 | 无（LLM 在 prompt 条件下生成） | 真实、常常有状态（sandbox、DB、browser） |
| Trajectory 组成 | 纯模型生成 token | LLM token + 环境注入 token（交错） |
| Reward | RM scalar 或规则化 | 通常稀疏的 outcome reward |
| Loss masking | 只 pad | **必须屏蔽环境注入 token** |
| 基础设施 | Trainer + rollout (vLLM) | Trainer + rollout + 环境 server + sandbox |

"Agentic" 这个修饰强调**梯度必须在 LLM/环境边界上正确流动**，这正是这个领域有趣（也比 RLHF 难）的地方。

## 三个核心挑战

每篇 agentic-RL 论文都在跟这三个的子集搏斗。能识别它们才能流畅读论文：

1. **异构 trajectory 组成**。Rollout 含 LLM token（策略采样）和环境 token（工具/检索/sandbox 注入）。PPO loss 朴素地应用到两者上，模型会学着模仿环境输出，训练崩。解法是 **retrieved-token loss masking**（及其泛化）
2. **长 horizon 上的稀疏 outcome reward**。一条 10 轮 8K token 的轨迹只有末尾 1 bit reward。PPO 的 value function（或 GRPO 的 group baseline）要做所有 credit assignment。在长 horizon 上会崩
3. **环境基础设施跟训练规模化方式不同**。Rollout 是 I/O-bound（容器启停、网络调用、工具延迟），训练是 GPU-bound。在一个进程里耦合浪费算力且限制规模。解法是**面向服务架构**（rollout-as-a-service，ProRL Agent / Polar）

## 推荐阅读路径

4 周课程，给已经懂 LLM 和一点 PPO、但对 agentic RL 新手的人。

### Phase 1 —— RL 基础（1 周）

已经懂 PPO/GRPO 可跳过。否则按顺序读：

1. **[[ppo-for-llm]]** —— PPO 怎么适配到 LLM token 级优化。关注：4-model 架构（actor/critic/RM/ref）、GAE、clipped objective、KL penalty
2. **[[grpo]]** —— DeepSeek 让其出名的 "无 critic PPO" 变体。关注：group-mean baseline、为什么没 value function、GRPO 何时赢何时输
3. **[[on-policy-distillation]]**（Preliminaries 一节）—— 友好地解释：KL 散度、forward vs reverse KL、on-policy vs off-policy、credit assignment、value head。**§前置概念一节是 wiki 里这些概念最好的 20 分钟入门**

Phase 1 结束：能解释 PPO loss、GAE、GRPO、KL penalty，以及 PPO 风格（KL-in-reward）跟 GRPO 风格（KL-as-loss）KL 处理的区别。

### Phase 2 —— DeepSeek-R1 谱系（3-5 天）

仔细读 DeepSeek-R1 / R1-Zero（[arXiv:2501.12948](https://arxiv.org/abs/2501.12948)）。关键收获：

- 纯 RL（无 SFT 数据）能引出复杂推理能力
- Outcome-only reward（答对 = 1，否则 = 0）在数学 / code 任务上够
- "Aha moment" / 自反思涌现是真的
- GRPO + outcome reward 是最简可行 recipe

这是 agentic RL 的**概念基础**。Search-R1 就是 R1-Zero 从纯推理扩展到工具调用。

### Phase 3 —— 入门论文：Search-R1（1 周）

agentic-RL 标准入门论文。

1. **读 [[search-r1]]**（论文精读）。理解：多轮 rollout 协议、retrieved-token loss masking、outcome-only EM reward、PPO-vs-GRPO 反直觉结果、涌现图（Fig 2c/d）
2. **读 [[search-r1-codebase-walkthrough]]**（代码 walkthrough）。理解：`generation.py` 怎么编排多轮 rollout、`info_mask` 怎么变成 `loss_mask`、veRL 怎么把全部串起来
3. **跑参考代码** —— clone repo、建 Wikipedia FAISS index、训 Qwen2.5-3B 200-500 步。观察 response length 和 search call count 曲线涌现
4. **跑 `state_masking=false` 消融** —— 亲身验证 retrieved-token masking 的必要性

Phase 3 结束：端到端理解 **agentic-RL 训练循环 pattern** 并有实操经验。

### Phase 4 —— 生产基础设施（1 周）

把 Phase 3 的 pattern 扩展到真实部署的基础设施。

1. **[[prorl-agent]]** —— 首个 "rollout-as-a-service" agentic-RL 系统。NVIDIA，2026-03。关注：HTTP `POST /process` 契约、INIT/RUN/EVAL pipeline、rootless HPC 沙箱
2. **[[polar]]** —— ProRL Agent 的续作（同 repo、同团队，2026-05）。关注：LLM-API proxy 范式（harness 当黑盒）、token-faithful prefix merging、为什么泛化到 Codex / Claude Code / Qwen Code / Pi harness
3. **[[nemo-gym]]** —— NVIDIA 的环境 catalog（84 benchmark、19 agent harness）。关注：三 server 架构（resources / model / agent）、Apptainer sandbox、怎么跟 trainer 连接
4. **三者关系**：读 [[prorl-agent#ProRL Agent vs NeMo Gym —— 同族、不同层|ProRL-vs-NeMo-Gym 节]] 理解每层填什么，读 [[polar#这怎么改变了 ProRL Agent vs NeMo Gym 的图|Polar 的对应节]] 看 gap 怎么被桥接

Phase 4 结束：理解 agentic RL 在生产规模下怎么工作 —— 层次边界、谁提供什么服务、要扩展应该改哪

### Phase 5 —— 前沿（开放）

有了基础后，跟随你感兴趣的方向：

**多工具 / 通用 agent** —— 扩展到单工具检索之外：
- ToolRL —— Search-R1 pattern 的多工具扩展
- ReTool —— 同谱系，代码工具
- Agent Lightning（Microsoft）—— 基于 tracing 的 agent RL
- rLLM —— 跨框架 agent RL

**Browser / OS agent** —— 更难的环境：
- WebGPT-RL —— RL 训练的浏览器 agent
- OSWorld —— 操作系统 agent
- BrowserGym / Mind2Web

**Reward 设计** —— 超越 outcome-only：
- Process Reward Models (PRM) 用于数学
- LLM-as-judge 用于开放任务
- KDRL —— 组合 RL 与 on-policy distillation

**推理时 agentic** —— 桥接训练和服务：
- [[das-spec-rl]] —— RL rollout 的投机解码
- [[aurora]] —— 在线 spec-decoding 训练，框架为 RL

**长 horizon / 开放结尾** —— 超越 benchmark：
- DeepResearcher —— 多源 web 研究
- Computer-use agent (CUA-RL)
- 长上下文 agent（100K+ token 轨迹）

## 标志参考 —— 短摘要

### 基础论文（先读这些）

- **DeepSeek-R1 / R1-Zero**（DeepSeek，2025-01，[arXiv:2501.12948](https://arxiv.org/abs/2501.12948)）—— 纯 outcome-only RL 引出复杂推理。Agentic RL 的概念祖先。还没专门 wiki 页（在 [[grpo]] 上下文中覆盖）
- **[[search-r1]]**（UIUC + UMass + Google，COLM 2025，arXiv:2503.09516）—— R1-Zero 扩展到 tool use。**标准入门论文**。多轮 rollout、retrieved-token loss masking、outcome-only EM reward
- **[[grpo]]** —— DeepSeek 的 PPO-without-critic。2025-26 agentic RL 里最常用的算法
- **[[ppo-for-llm]]** —— 万事开始的基础算法

### 基础设施论文（基础之后再读）

- **[[prorl-agent]]**（NVIDIA，2026-03，arXiv:2603.18815）—— 首个 "rollout-as-a-service" agentic-RL 框架。面向服务设计、rootless HPC sandbox、token-in/token-out wire 协议。2026-05 被 Polar 取代
- **[[polar]]**（NVIDIA，2026-05，arXiv:2605.24220）—— ProRL Agent 续作。LLM-API proxy 范式；训练*任何未修改 harness*（Codex、Claude Code、Qwen Code、Pi）。注册为 NeMo Gym 环境
- **[[nemo-gym]]**（NVIDIA，2026）—— 环境 catalog 框架。84 benchmark、19 agent harness、三 server 架构
- **[[rl-training-frameworks]]** —— veRL、OpenRLHF、TRL —— 底层 RL 框架图景

### 邻近 / 支持论文

- **[[on-policy-distillation]]** —— agentic RL 的 non-RL 表兄。有 teacher 存在时，OPD 提供稠密 per-token 信号，没有 credit assignment 负担
- **[[das-spec-rl]]** —— RL rollout 的 Distribution-Aware Speculative decoding。训练时 rollout 加速 1.5-2×
- **[[aurora]]** —— 在线投机解码 draft 训练，被框架成线上服务流量上的 agentic RL
- **[[ring-attention]]** / **[[deepspeed-ulysses]]** —— 长上下文 attention 并行，跟长 horizon agentic rollout 有关

### Wiki 里的算法

- **[[grpo]]** —— Group Relative Policy Optimization
- **[[ppo-for-llm]]** —— Proximal Policy Optimization for LLMs
- **[[dpo]]** —— Direct Preference Optimization（跟 agentic RL 不太相关但有用对比）
- **[[on-policy-distillation]]** —— OPD 家族
- **[[reward-modeling]]** —— Reward model 怎么建以及哪里失败

## 常见困惑（FAQ）

> [!question] 问：Search-R1 还是 SOTA 吗？
>
> 不是 —— 但仍然是最好的**入门**。2026-05，agentic-RL 基础设施 SOTA 是 [[polar|Polar]]；训出来的模型 SOTA 取决于任务（通用 agent 是 DeepSeek-V4，竞赛数学/code 是 Nemotron-Cascade 2）。Search-R1 的角色是干净地教基础。

> [!question] 问：agentic RL 用 PPO 还是 GRPO？
>
> 看任务：
> - **短 horizon、单答案（数学、code）**：GRPO 常赢（DeepSeek-R1、GRPO 论文）。轨迹短时 group-mean baseline 没问题
> - **多轮、长 horizon（Search-R1、agentic 任务）**：PPO 常更稳（Search-R1 Table 3）。Value function 在长轨迹 credit assignment 上有帮助
> - **拿不准**：PPO 更保守；GRPO 迭代更快。在你的具体场景里都试一下

> [!question] 问：我需要 NVIDIA 研究集群那么多 GPU 吗？
>
> 不用。Search-R1 Qwen2.5-3B 在单 8×H100 / 8×A100 节点上约 2 天能跑完。最小有趣的 agentic-RL run 是 **单 8×A100 / 8×H100 + 256GB RAM + 2TB SSD**

> [!question] 问：什么时候用 OPD 代替 RL？
>
> 有更强 teacher 模型时，想把它的行为压缩到更小的 student 里。[[on-policy-distillation|OPD]] 比 RL 样本效率高很多（10× 少 compute）但上限是 teacher。RL 上限不封顶但贵。**生产里 hybrid (KDRL、dGRPO) 在变成默认**

> [!question] 问："Retrieved-token loss masking" 到底在做什么？
>
> 它告诉优化器"这些 token 是环境注入的，不是策略生成的 —— 别在它们上算梯度，也别在它们上算 KL"。没这个，PPO loss 在环境 token 上训模型模仿检索内容（错误行为），环境 token 上的 KL 跟无意义的 reference 分布对比。数学见 [[search-r1#Retrieved-token loss masking — the load-bearing trick|Search-R1 节]]，实现见 [[search-r1-codebase-walkthrough#4.5 ★ THE 关键函数 _info_masked_concatenate_with_padding|代码 walkthrough]]

> [!question] 问：agentic RL 跟 RLHF 有什么不同？
>
> RLHF 是**单轮**（一个 prompt、一个 response），用**学习式 reward model**。Agentic RL 是**多轮**（LLM ↔ 环境循环），用**规则化或环境派生的 reward**。基础设施重叠很大（同样的 PPO、actor-critic 切分、KL-to-reference），但 rollout 阶段根本不同

> [!question] 问：只能看一篇论文，看哪一篇？
>
> **[[search-r1]]**。它一次性教会你协议、数学、消融、涌现 pattern、代码架构。其它（Polar、ToolRL、ReSearch 等）都是它主题的变体

## 开放研究方向

2026 年中正在 actively 做的：

1. **长 horizon agentic RL**（10+ 轮、32K+ token）—— token budget、KV cache 管理、极端 horizon 下 reward 稀疏性。开放：horizon ~20 之后怎么拿到有用梯度？
2. **多工具组合** —— Search-R1 单工具扩到 3-10 工具（搜索 + 计算器 + 代码 + 浏览器 + ...）。开放：模型怎么在对的步选对的工具？
3. **Process reward models (PRMs)** —— 用步级监督替换 outcome-only。开放：怎么不要无限标注数据训出可靠 PRM？
4. **Self-improvement loop** —— 模型生成 rollout，用自己判断它们，在自己的判断上训。开放：怎么避免漂移 / reward hacking？
5. **Computer-use agent** —— 视觉 + 文本 agent 操作桌面应用（OSWorld、Anthropic Computer Use）。开放：怎么把视觉-语言模型桥接到 RL 训练栈
6. **成本感知 rollout** —— 生产搜索有 $$/延迟。开放：怎么训出工具调用节俭的模型
7. **大规模 trainer-rollout 解耦** —— [[polar]] 是当前前沿。开放：怎么扩展到 100B+ frontier 模型
8. **多轮场景的 off-policy 修正** —— 环境状态变、模型版本更新，怎么复用旧 rollout？开放：重要性加权多轮 RL
9. **Hybrid OPD + RL** —— KDRL、dGRPO 早期显示增益。开放：什么是最干净的方程

## 相关阅读（更广图景）

- [[agentic-rl-overview]] —— Agentic RL 更高层综述（老一些的页面，更概念性）
- [[tool-use-rl]] —— 工具使用 RL 专门
- [[multi-step-reasoning-rl]] —— 长 horizon 推理
- [[environment-design]] —— 怎么设计好的 RL 环境
- [[rl-training-frameworks]] —— veRL / OpenRLHF / TRL / NeMo-RL
- [[rlhf-overview]] —— RLHF 基础，前作

## 快速查询：哪页讲什么

| 你想学 | 读 |
| ------ | -- |
| **整个领域概览** | 本页 |
| **Search-R1 在做什么、为什么** | [[search-r1]] |
| **Search-R1 代码逐行** | [[search-r1-codebase-walkthrough]] |
| **PPO 在 LLM 上怎么工作** | [[ppo-for-llm]] |
| **为什么 GRPO 替代 PPO** | [[grpo]] |
| **KL 散度、on-policy、credit assignment** | [[on-policy-distillation]] 前置概念节 |
| **生产 rollout 架构** | [[polar]]（当前 SOTA）或 [[prorl-agent]]（前作） |
| **有哪些环境** | [[nemo-gym]] |
| **选哪个框架** | [[rl-training-frameworks]] |
| **rollout 加速** | [[das-spec-rl]] |
| **完全避免 RL（有 teacher 的话）** | [[on-policy-distillation]] |

---

这个 hub 缺什么？wiki 是开放的 —— 看到没链接或还没成页的，提出来。
