---
title: "MOPD：多 Domain On-Policy 蒸馏作为 Cascade-RL 的稳定器"
category: rl-infra
tags: [mopd, on-policy-distillation, nemotron-cascade-2, cascade-rl, multi-teacher, post-training, moe, paper-review]
created: 2026-05-19
updated: 2026-05-19
status: mature
paper: arXiv:2603.19220
---

# MOPD：多 Domain On-Policy 蒸馏作为 Cascade-RL 的稳定器

> [!info] 论文元信息
> - **Paper**：[arXiv:2603.19220](https://arxiv.org/abs/2603.19220) —— *Nemotron-Cascade 2: Post-Training LLMs with Cascade RL and Multi-Domain On-Policy Distillation*（Yang et al., NVIDIA；v1 2026-03-19，v2 2026-03-22；通讯作者 Wei Ping）
> - **项目页**：[research.nvidia.com/labs/nemotron/nemotron-cascade-2](https://research.nvidia.com/labs/nemotron/nemotron-cascade-2/)
> - **模型**：[Nemotron-Cascade-2-30B-A3B](https://huggingface.co/nvidia/Nemotron-Cascade-2-30B-A3B)（30B 总 / 3B 激活 MoE，1M 上下文，NVIDIA Open Model License）
> - **SFT 数据**：[nvidia/Nemotron-Cascade-2-SFT-Data](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-SFT-Data)
> - **RL 数据**：[nvidia/Nemotron-Cascade-2-RL-data](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-RL-data)
> - **参考框架**：[NVIDIA-NeMo/RL](https://github.com/NVIDIA-NeMo/RL)（`nemo_rl/algorithms/distillation.py`）；MOPD 的多教师路由到 2026-05 还 **不是 NeMo-RL 一等公民**
> - **前置工作**：[[on-policy-distillation|GKD (Agarwal 2024)]]、[Thinking Machines OPD 博客 (2025-10)](https://thinkingmachines.ai/blog/on-policy-distillation/)、Xiaomi MiMo-V2-Flash MOPD（[arXiv:2601.02780](https://arxiv.org/abs/2601.02780)；见 [撞名小节](#与-xiaomi-mimo-v2-flash-的撞名)）

> [!abstract]+ TL;DR
> **MOPD** 是 NVIDIA 7-stage Cascade RL 管线里的一个独立稳定阶段，用 [[on-policy-distillation|on-policy 蒸馏]] 从 **三个同家族 teacher** *恢复* 早期专项 RL 阶段引入的能力回归。Per-prompt 路由从三个 teacher（math SFT checkpoint / RLHF 侧分支 / multi-domain RL best checkpoint）选一个，计算 **采样 token 反向 KL "advantage"** $a_t = \log\pi_T(y_t) - \log\pi_\theta(y_t)$ 作为 REINFORCE 式更新的 token 权重，配 truncated importance weighting 处理 train-vs-inference policy gap。**三个 teacher 全部是 cascade 副产物** —— 不要额外训练、不要外部模型、不要 logit 缓存。报告战绩：AIME 2025 92.4（金牌级）、IMO 2025 35/42（金）、IOI 2025 439.28/600（金）、ICPC WF 2025 10/12（金）—— 而且 **MOPD 用 52 步达到 ArenaHard v2 85.5，RLHF 用 160 步只到 80.7**，约 3× 每步效率优势是这篇论文的主要实证论据。**MOPD 作为技术不算新** —— 是 GKD-反向-KL-OPD 加 per-prompt 教师路由，两个月前 Xiaomi MiMo-V2-Flash 在相同缩写下发布过（NVIDIA 把 "Multi-Teacher" 重新框成 "Multi-Domain"）。贡献是 **recipe**：选哪些 teacher、放在哪个阶段、什么 LR schedule，在 30B-A3B MoE 学生上由 IMO/IOI/ICPC 金牌验证。

---

## 背景：为什么需要发明 MOPD

Cascade RL 把专项 RL 阶段按顺序堆起来，每个阶段调一项能力（instruction following、数学推理、RLHF 帮助性、长上下文、代码、SWE-agent）。每个阶段内部 trainer 把模型在 *该* domain 上做得更好。跨阶段，**模型在之前训练过的 domain 上漂移**。Nemotron-Cascade 2 论文直接承认：某些 RL 阶段（特别是 code RL）"降低模型熵、缩短推理 trace，伤害数学能力"（[Labonne 解读](https://maximelabonne.substack.com/p/nemotron-cascade-2-on-policy-distillation)）。

朴素修法 —— 重跑回归 domain 的 RL —— 贵又不稳（你只会再漂一次）。两个结构性问题：

| 问题 | 跨阶段为什么显现 | 朴素修法代价 |
| ---- | --------------- | ----------- |
| **能力漂移** | 每阶段 reward 信号 domain-specific；优化一个 domain 把 policy 推离其它 | 重训回归 domain 的 RL —— 贵，可能引入另一种漂移 |
| **熵塌缩** | Code/agentic RL 倾向缩短推理 trace；这伤害数学 | 提高 exploration 系数 —— 可能毁掉你刚付钱拿到的收益 |
| **IF-RL 后失对齐** | IF-RL 改进指令遵循但伤害 human-preference | 之后跑 RLHF —— 但跑多少、放哪里？ |

MOPD 重新框架：别重训。用一个 *已经有* 该能力的 teacher **蒸馏** 回来。Teacher 不需要外部也不需要更大 —— 可以是 *cascade 内部* 某个在该 domain 漂之前就强的 checkpoint。于是你用稠密 per-token credit assignment 信号（像 [[on-policy-distillation|OPD]]）做能力恢复，每步代价只有 RL 一小部分。

跟现有 OPD recipe 对比：

| 方法 | Teacher 数 | Teacher 来源 | Per-prompt 路由 | KL 形式 | 管线角色 |
| ---- | --------- | ----------- | -------------- | ------- | -------- |
| [[on-policy-distillation\|TML OPD]] (2025-10) | 1 | 外部（Qwen3-32B） | 不适用 | 反向 KL，采样 token | RL 替代品 |
| Xiaomi MOPD（[MiMo-V2-Flash](https://arxiv.org/abs/2601.02780), 2026-01） | 多 | 独立训练的 specialist | 是 | 反向 KL，采样 token | Post-training merge |
| [[deepseek-v4-opd\|DeepSeek-V4 OPD]] (2026-04) | 10+ | 独立训练的 specialist | 加权和（每 prompt 多 teacher） | **全词表** KL | 替换整段 mixed-RL 阶段 |
| **NVIDIA MOPD**（本页，2026-03） | 3 | **同 cascade 的免费副产物** | 是（每 prompt 一个 teacher） | 反向 KL，采样 token | Cascade 内单一稳定阶段 |

关键差异：MOPD 的 teacher 免费。不要额外训练、不要外部模型、不要 FP4 QAT 或隐藏状态缓存基础设施。同 tokenizer、同 vocab、同 base 模型 —— 所以蒸馏可以用现有 NeMo-RL OPD 原语加一个 per-batch teacher 切换搞定。

---

## 核心思想：把 prompt 路由到对应 domain 的 teacher，用 OPD 当稳定器

> [!quote] 一句话总结贡献
> 在 Multi-domain RL 和 RLHF 之间插入一个短的 on-policy 蒸馏阶段，按 prompt 路由到三个 cascade 内部 teacher 之一（math SFT / RLHF 侧分支 / multi-domain best checkpoint），来恢复早期专项 RL 漂掉的能力 —— 免费，~50 步。

三个支撑次级声明：

- **Cascade 内部 teacher 够用**。你不需要更大的外部模型。Math SFT checkpoint 在专项 RL 动它之前数学已经强；RLHF 侧分支已经有好的 human-preference 对齐；multi-domain RL best checkpoint 有指令遵循的收益。每一个都是 *上一阶段在自己 domain 上的最佳*，正好是你要恢复的。
- **每 prompt 一个 teacher 够了**。跟 [[deepseek-v4-opd|DeepSeek-V4]] 加权和不同，MOPD 每个训练样本按 domain tag 选一个 teacher。信号更便宜、没有 teacher 冲突、dispatcher 更简单。
- **采样 token 反向 KL + 重要性裁剪够用**。30B-A3B 规模上不需要全词表 KL（V4 风格）；per-token 采样信号配 $r_t \in [0.5, 2.0]$ 截断稳定地处理 train/inference policy 偏差。

去掉任何一个：失去内部 teacher MOPD 就退化成"又一个外部 teacher OPD"；失去 per-prompt 路由单一 teacher 把所有 domain 拉向自己的风格；失去重要性裁剪异步 on-policy 训练会发散。

---

## 实现细节

### MOPD 在 cascade 里的位置

7 阶段 Nemotron-Cascade 2 管线（论文 Figure 2）：

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │                                                                       │
   │  SFT → IF-RL → Multi-domain RL → MOPD → RLHF → Long-context RL       │
   │                                                                       │
   │       → Code RL → SWE RL → Nemotron-Cascade 2                         │
   │                                                                       │
   └──────────────────────────────────────────────────────────────────────┘

   MOPD 是 Multi-domain RL 和 RLHF 之间的 **单一稳定阶段**。
   它 **不是** 与每个 RL 阶段交错的 per-round 循环。
   Cascade 后续阶段不会再回到 MOPD。
```

放置位置是关键 recipe 选择。MOPD 跑的时候：

- **SFT** 已经产出强数学推理者（变成 math teacher）。
- **IF-RL** 已经建立指令遵循能力但可能伤害 human alignment。
- **Multi-domain RL** 已经把 MCQA / agentic tool calling / structured output 整合（变成 multi-domain teacher）。
- **从 SFT 出发的并行 RLHF 侧分支** 已经训过 human preference（变成 RLHF teacher）。

MOPD 然后用三个 teacher 在进入主 cascade RLHF 之前稳定，cascade 后续 Long-context / Code / SWE RL 不间断地继续。

### Loss 函数（论文 Eq. 2–4）

设 $\pi^{\text{inf}}$ 为推理引擎里用于 rollout 的学生，$\pi^{\text{train}}$ 为正在优化的学生，$\pi^{\text{domain}_i}$ 为该 prompt domain 路由到的 teacher，$s_t = (x, y_{<t})$。

**Token 级蒸馏 advantage** —— 只在采样到的 token 上的反向 KL：

$$
a_t^{\text{MOPD}} = \log \pi^{\text{domain}_i}(y_t \mid s_t) - \log \pi^{\text{train}}(y_t \mid s_t) \tag{Eq. 2}
$$

**截断重要性权重** 处理 train-vs-inference policy 不匹配：

$$
r_t = \frac{\pi^{\text{train}}(y_t \mid s_t)}{\pi^{\text{inf}}(y_t \mid s_t)}, \qquad w_t = \text{sg}[r_t] \cdot \mathbf{1}\bigl[\epsilon_{\text{low}} \leq r_t \leq \epsilon_{\text{high}}\bigr] \tag{Eq. 3}
$$

$\epsilon_{\text{low}} = 0.5$、$\epsilon_{\text{high}} = 2.0$。越界 token 权重置零（不是梯度裁剪，是丢掉）。

**代理目标：**

$$
\boxed{\,\mathcal{L}_{\text{MOPD}} = -\,\mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi^{\text{inf}}(\cdot \mid x)}\!\left[\frac{1}{|\mathcal{V}(y)|}\sum_{t \in \mathcal{V}(y)} w_t \cdot \text{sg}[a_t^{\text{MOPD}}] \cdot \log \pi^{\text{train}}(y_t \mid s_t)\right]\,} \tag{Eq. 4}
$$

两个关键实现细节论文明说：

- **采样 token KL，不是全词表**。*"The log-probability difference is computed only on the student-sampled token rather than over the full vocabulary"*（论文 p.13）。这是相对 [[deepseek-v4-opd|DeepSeek-V4]] 全词表 KL 的刻意选择。
- **$a_t$ 和 $r_t$ 上都 stop-gradient**。唯一梯度路径是穿过 $\log\pi^{\text{train}}(y_t \mid s_t)$ —— "advantage" 是与 teacher 的 log-prob 差距的 token 加权 REINFORCE 式更新。

> [!quote] 跟 [[on-policy-distillation|GKD]] 的关系
> Eq. 4 是 GKD-反向-KL-OPD 配两个工程修改：(a) 截断重要性权重处理异步 on-policy gap（$\pi^{\text{inf}}$ 可能滞后 $\pi^{\text{train}}$），(b) 显式 stop-gradient 让实现保持 REINFORCE-shaped 而不是直接 KL-shaped。Reward $\log(\pi_T/\pi_\theta)$ 跟 MiniLLM/GKD/TML 一样。

### Teacher 选择 —— 三个 teacher，全部免费

论文（p.13）列举的三个 teacher：

| Teacher | 来源 | 擅长 | 成本 |
| ------- | ---- | ---- | ---- |
| **Math teacher** | 原 SFT checkpoint | SFT 数据精心整理过的强数学推理 | 0（已存在） |
| **RLHF teacher** | 从 SFT init 跑 RLHF 侧分支（25 步 RLHF，GenRM = Qwen3-235B-A22B-Thinking-2507） | Human-preference 对齐 | 25 步 RLHF（便宜） |
| **Multi-domain teacher** | IF-RL + Multi-domain RL 之后的 best checkpoint | 指令遵循、MCQA、agentic tool calling、structured output | 0（cascade 本来就要训） |

选取标准：*"the strongest validation checkpoint for each benchmark category"*（论文 p.12）。三个 teacher 跟 student 共享 **同 tokenizer、同 vocab、同 base 模型** —— 这正是 MOPD 不需要 DeepSeek-V4 风格 FP4 QAT、隐藏状态缓存、跨 vocab logit 投影的原因。只要把 rollout server 指向 per-prompt 正确的 checkpoint。

### Per-prompt 路由 —— 每个训练样本一个 teacher

Prompt 按其 domain 来源打 `teacher_id`：

```
训练池构成（论文 §4.4 大致）：

  math prompts        ─── AceReason-Math      ─► 路由到 math teacher (SFT)
  IF / multi-domain   ─── 来自 IF-RL /         ─► 路由到 multi-domain teacher
  prompts                  Multi-domain RL 池
  helpfulness         ─── 来自 RLHF 训练池      ─► 路由到 RLHF teacher
  prompts                 （HelpSteer3 等）
```

**没有跨 teacher 的 logit 混合**。Rollout 里每个 token 都由恰好一个 teacher 的 log-prob 监督。这是相对 DeepSeek-V4 $\sum_i w_i D_{\text{KL}}(\pi_\theta \| \pi_{E_i})$ 加权和的架构选择。

### 超参

**散文**（p.13）跟 **附录 Table 8** 有个值得知道的不一致：

| 设置 | 散文 §4.4 | Table 8（附录 B） |
| ---- | --------- | ----------------- |
| 学习率 | 2×10⁻⁶，前 30 步从 2×10⁻⁷ 线性 warmup | 3×10⁻⁶ |
| 步数 | "Typically converges within 40–50 steps" | 52 |
| 每 prompt rollout 数 | 4 | 4 |
| 每次更新 prompt 数（batch） | 128 | 128 |
| 有效 batch（response 数） | 512 | — |
| 最大 response 长度 | — | 98K |
| 重要性边界 | $\epsilon_{\text{low}} = 0.5$、$\epsilon_{\text{high}} = 2.0$ | — |
| Temperature / top-p | — | 1.0 / 1.0 |
| Overlong filtering | — | False |
| KL 形式 | 反向 KL，采样 token（非全词表） | — |

复现时以散文为主；LR 不一致要标记给做复现的人。论文没说哪个是 canonical。

### 为什么这在适度成本下有效

三个 MOPD 相对 RLHF 或重跑专项 RL 便宜的结构性原因：

- **Teacher 免费**。Math teacher = SFT checkpoint（不要额外训练）。Multi-domain teacher = 你 Multi-domain RL 期间无论如何都会保的 best checkpoint。RLHF teacher = 25 步侧分支 RLHF（比主 RLHF 阶段小）。
- **稠密 per-token 信号**。不像 RLHF 的 per-trajectory 标量 reward，MOPD 给每个 token 打分。~52 步稠密信号能恢复 160 步 RLHF 稀疏 reward 才能恢复的东西。
- **基础设施零变更**。同 tokenizer、同 vocab、同 base。NeMo-RL OPD 原语只要 per-batch teacher 切换就变成 MOPD —— 不要 DeepSeek-V4 风格的隐藏状态缓存、FP4 QAT、TileLang kernel。

---

## 实验

### 标志数字（Table 1–2）

Nemotron-Cascade-2-30B-A3B vs 同激活参数基线：

| Benchmark | Nemotron-Cascade-2-30B-A3B | Qwen3.5-35B-A3B (2026-02) | Nemotron-3-Super-120B-A12B (2026-03) |
| --------- | -------------------------- | -------------------------- | ------------------------------------ |
| **IMO 2025** | **35/42（金）** | — | — |
| IMO AnswerBench | **79.3** | 74.8 | 77.2 |
| IMO ProofBench | **72.9** | — | — |
| **AIME 2025** | **92.4**（98.6 TIR） | 91.9 | 90.2 |
| AIME 2026 | 90.9（95.0 TIR） | **91.1** | 89.8 |
| HMMT Feb25 | **94.6** | 89.0 | 93.7 |
| **IOI 2025** | **439.28/600（金）** | 348.6 | — |
| **ICPC WF 2025** | **10/12（金）** | — | — |
| LiveCodeBench v6 | **87.2**（88.4 TIR） | 74.6 | 78.7 |
| LCB Pro 25Q2 Med | **27.6**（36.8 TIR） | 17.8 | 23.2 |
| MMLU-Pro | 79.8 | **85.3** | 83.7 |
| GPQA-Diamond | 76.1 | **84.2** | 79.2 |
| ArenaHard v2 (Avg) | **83.5** | 65.4 | — |
| IFBench (prompt) | **82.9** | 70.2 | 72.6 |
| SWE Verified (OpenHands) | 50.2 | **69.2** | 60.5 |
| Terminal Bench 2.0 | 21.1 | **40.5** | 31.0 |
| 𝜏²-Bench | 58.9 | **81.2** | 61.2 |

论文 Footnote 1（p.4）：Nemotron-Cascade 2 是 *"the second open-weight LLM, after DeepSeek-V3.2-Speciale-671B-A37B, to achieve gold-medal performance in both the IMO and IOI"* —— 在 **3B 激活参数** 上，对比 DeepSeek 的 37B 激活。

### MOPD 特定结果：相对 RLHF 的步效率

MOPD 在论文里的主要论据（Table 3、散文 p.13）：

| Cascade 阶段 | 步数 | ArenaHard v2（hard / overall） | AIME 25 |
| ------------ | ---- | ------------------------------ | ------- |
| Multi-domain RL 输出 | — | — | 91.0 |
| **MOPD**（52 步） | **52** | **85.5 / 71.0** | **92.0** |
| RLHF（160 步） | 160 | 80.7 / 71.2 | — |

MOPD 用 ~3× 更少的步数达到比 RLHF 更高的 ArenaHard 分数。AIME 25 上 +1.0 绝对提升（91.0 → 92.0）适中，但 30 步达到，GRPO 用 25 步达到 91.0 —— 数学上 MOPD 跟 GRPO 大致 compute-matched，但 human-preference benchmark 上 MOPD 在步效率上大幅超 RLHF。

> [!important] 论文 *没有* 报告的
> 没有 MOPD 特定的 GPU-hour 或 wall-clock 对比。没有 leave-MOPD-out 消融在整个 cascade 上隔离 MOPD 每个 benchmark 上的贡献。没有 teacher 数消融（1 或 2 个够吗？4–6 个更好？）。步效率声明是 **per-step 不是 per-second** —— 这重要，因为每个 MOPD 步还要 teacher forward pass。

### MOPD 帮不到的地方

Nemotron-Cascade 2 输的 benchmark 说明问题：

| Benchmark | Cascade 2 | Qwen3.5-35B-A3B | Δ |
| --------- | --------- | --------------- | -- |
| MMLU-Pro | 79.8 | 85.3 | **−5.5** |
| GPQA-Diamond | 76.1 | 84.2 | **−8.1** |
| SWE Verified (OpenHands) | 50.2 | 69.2 | **−19.0** |
| Terminal Bench 2.0 | 21.1 | 40.5 | **−19.4** |
| 𝜏²-Bench | 58.9 | 81.2 | **−22.3** |

论文承认（p.5）Cascade 2 *"underperforms Qwen3.5-35B-A3B primarily on knowledge-intensive and agentic tasks."* 结构性原因回到 MOPD：**teacher 池里没有 GPQA teacher、没有 agentic-tool-use teacher**。MOPD 只能恢复 cascade 内部 *存在* teacher 的能力。知识 gap 和 agentic 任务 gap 需要外部 teacher 或者更长的 RL —— 两者都不在 recipe 里。

---

## 优势与限制

最强两点：(1) **cascade 内部 teacher** 的洞察是真正的 recipe 贡献 —— 一短次侧分支 RLHF 的代价拿到三个高质量 teacher，是 MOPD 效率背后的真实故事，不是 loss 函数；(2) **生产验证** 明确 —— 3B 激活参数拿 IMO/IOI/ICPC 金牌是 2026 早期最强的开放权重 reasoning 模型证据点。

诚实承认的限制：

- **不是算法层面新**。Loss 是 GKD 反向 KL OPD（[[on-policy-distillation|Agarwal 2024]]）加截断重要性权重（异步 on-policy RL 里标准做法 —— DAPO、PPO-clip）。Per-prompt teacher 路由两个月前 Xiaomi MiMo-V2-Flash 在相同缩写下发表过（见 [撞名小节](#与-xiaomi-mimo-v2-flash-的撞名)）。
- **没有 teacher 数消融**。1 或 2 个 teacher 行吗？加 GPQA teacher 能修知识任务 gap 吗？论文没说。
- **没有跨整个 cascade 的 leave-MOPD-out 消融**。Table 3 / Figure 3 只比 MOPD vs GRPO 在 AIME25 和 ArenaHard v2 上。无法隔离 MOPD 对 IMO/IOI/ICPC 金牌的贡献。
- **超参不一致**。散文说 LR 2e-6 + warmup；Table 8 说 3e-6。说明 schedule 被调过多于报告的程度。
- **采样 token KL 意味着低置信度 token 上信号稀疏**。全词表（V4 风格）能捕到更多但论文没在这规模上对比。
- **没分析 teacher 冲突**。Domain 边界上（如同时考数学和指令遵循的 prompt），只选一个 teacher —— 可能选错。
- **能力上限 = teacher 上限**。MOPD 没法扩展 teacher 没有的能力。GPQA / SWE Verified gap 是证据。
- **多 teacher 路由不在 NeMo-RL**。参考框架只实现单 teacher OPD。复现 MOPD 要 fork trainer 接受 teacher dict 并 per-batch 查询正确 teacher —— 非平凡。

> [!warning] 与 Xiaomi MiMo-V2-Flash 的撞名
> 缩写 **MOPD 小米先用过**：MiMo-V2-Flash（[arXiv:2601.02780](https://arxiv.org/abs/2601.02780), 2026-01-06）里它代表 **Multi-Teacher On-Policy Distillation**。Xiaomi MiMo 推特：*"Beyond arch innovation, MiMo-V2-Flash is cooked via a NEW post-training paradigm Multi-Teacher On-Policy Distillation (MOPD)"*（[来源](https://x.com/XiaomiMiMo/status/2000930865757741342)）。NVIDIA Nemotron-Cascade 2（2026-03）把缩写重新框成 **Multi-Domain On-Policy Distillation**，算法本质一样（按 prompt 路由的 token 级反向 KL 蒸馏）。Nemotron-Cascade 2 paper 引了 Xiao et al. 2026 当前置工作但没承认缩写重叠。读 2026 年任何 MOPD 引用时 **要看是 Xiaomi 的"Multi-Teacher"还是 NVIDIA 的"Multi-Domain"** —— 同想法、同字母、不同框架。

---

## 这意味着什么

两条值得跟踪的预测：

1. **OPD-作为-cascade-稳定器会变成标准阶段**。论据有实证支撑：52 步 MOPD 配 cascade 内部 teacher 替换 160 步 RLHF 做跨 domain 整合。便宜到不能不用。预期 Qwen、Mistral、开源实验室会在专项 RL 和最后阶段对齐之间加类似阶段。
2. **"免费内部 teacher"洞察会扩散**。DeepSeek-V4 风格 —— 训 10+ specialist 配全词表 KL 加权合并 —— 贵且基础设施重。NVIDIA 的 "teacher 就是你管线里早些时候的 checkpoint" 洞察显著更便宜。对多数没有 DeepSeek-V4 基础设施的从业者，MOPD recipe 是现实采纳路径。

这 *不是*：cascade-RL 漂移的万能解药。MOPD 只能恢复 cascade 内部 *存在* teacher 的能力。能力 *扩展* 仍然需要 RL with verifiable reward 或外部 teacher。

---

## 源码与复现

### 已发布的（开放）

| Artifact | 状态 |
| -------- | ---- |
| 模型权重（[Nemotron-Cascade-2-30B-A3B](https://huggingface.co/nvidia/Nemotron-Cascade-2-30B-A3B)） | ✓ 开放，NVIDIA Open Model License |
| SFT 数据（[nvidia/Nemotron-Cascade-2-SFT-Data](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-SFT-Data)） | ✓ 开放 |
| RL 数据（[nvidia/Nemotron-Cascade-2-RL-data](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-RL-data)） | ✓ 开放 |
| Tech report（[arXiv:2603.19220](https://arxiv.org/abs/2603.19220)） | ✓ 开放 |
| 端到端 MOPD 脚本 / 配置 | ✗ 未发布 |
| NeMo-RL 多教师 OPD trainer | ✗ 还不是原语（仅单 teacher） |

### NeMo-RL OPD 原语（最接近的一等公民参考）

[NVIDIA-NeMo/RL](https://github.com/NVIDIA-NeMo/RL)，文件 `nemo_rl/algorithms/distillation.py`（1,072 行）。[Discussion #1445](https://github.com/NVIDIA-NeMo/RL/discussions/1445)。[文档](https://docs.nvidia.com/nemo/rl/0.5.0/about/algorithms/on-policy-distillation.html)。

有的：
- 单 teacher / 单 student —— `MasterConfig.teacher: PolicyConfig`（line 117）。
- KL 选项：`forward` / `reverse` / `mixed`。
- Top-k 限制 KL（如 `topk_logits_k=64`）—— **注意这本身已经是相对 paper Eq. 2 的偏离**，paper 用的是仅采样 token log-prob，不是 top-k logits。
- 后端：DTensor + vLLM。Megatron 生成/训练暂不支持。

缺的（对 MOPD 具体而言）：
- 多 teacher 路由（per-batch / per-sample teacher 切换）。
- 截断重要性权重（Eq. 3）。
- 精确的采样 token KL（不要 top-k 扩展）。

在 NeMo-RL 上复现 MOPD 要扩展 `DistillationLossFn` 接受 teacher dict，per-sample 查询 `batch["teacher_id"]`，加上 importance-clip mask。

### 最接近的现成替代 —— veRL 多 teacher OPD

[veRL OPD 文档](https://verl.readthedocs.io/en/latest/algo/opd.html) 和 [异步 on-policy distill](https://verl.readthedocs.io/en/latest/advance/async-on-policy-distill.html)。veRL 已经支持通过 `data_source` 多教师路由：

```yaml
distillation:
  enabled: true
  teacher_models:
    math_teacher:
      model_path: /path/to/sft_ckpt
    rlhf_teacher:
      model_path: /path/to/rlhf_sidebranch_ckpt
    multi_domain_teacher:
      model_path: /path/to/multi_domain_rl_best_ckpt
  teacher_key: data_source     # 把每个 prompt 路由到匹配 teacher
  distillation_loss:
    loss_mode: k1              # K1 = log(π_S/π_T) —— 接近 Eq. 2
  use_policy_gradient: true
  use_task_rewards: false      # MOPD 风格：不用 outcome reward，只用 OPD 信号
```

这 **比 NeMo-RL 更接近 MOPD 论文** —— 设 `teacher_key="data_source"`、给每个 prompt 打 domain tag、框架处理 per-batch teacher 切换。比 fork NeMo-RL 侵入小。

### 最小复现 recipe（NeMo-RL fork 路径）

```python
# 概念草图 —— 完整实现需要扩展 NeMo-RL 的 DistillationLossFn 支持
# per-sample teacher 查询。

# 1. SFT → 保存为 math_teacher_ckpt
# 2. 跑 IF-RL（180 步，batch 128 × 16 rollouts，LR 3e-6）
# 3. 跑 Multi-domain RL（70 步）→ 保存 best 为 multi_domain_teacher_ckpt
# 4. 侧分支：从 SFT init 跑 RLHF（25 步，GenRM=Qwen3-235B，KL 0.03）
#    → 保存为 rlhf_teacher_ckpt
# 5. MOPD 阶段：
#    - 给 prompt 打 tag：math 从 AceReason-Math → math_teacher；
#                       multi-domain 从 RL 池 → multi_domain_teacher；
#                       helpfulness 从 HelpSteer3 → rlhf_teacher
#    - Loss：采样 token 反向 KL advantage，a_t 和 r_t 上 stop-gradient，
#            重要性裁剪 [0.5, 2.0]
#    - LR 2e-6 配 30 步从 2e-7 线性 warmup；约 52 步总
#    - Batch 128 prompts × 4 rollouts
# 6. 继续 cascade：RLHF（25 步）→ Long-context RL（30 步）
#    → Code RL（22 步）→ SWE RL（40–50 步）
```

### 底层 OPD 原语的其它开源实现

| 项目 | 路径 | 与 MOPD 距离 |
| ---- | ---- | ----------- |
| [HF TRL `GKDTrainer`](https://github.com/huggingface/trl/blob/main/trl/trainer/gkd_trainer.py) | 单 teacher GKD 配 $\lambda$ + $\beta$ 旋钮 | 最远 —— 要自定义数据路由 + 多 teacher 包装 |
| [veRL `algo/opd`](https://verl.readthedocs.io/en/latest/algo/opd.html) | 多教师路由 via `data_source` | **最接近的现成方案** |
| [NVIDIA NeMo-RL OPD](https://github.com/NVIDIA-NeMo/RL) | 单 teacher 配 top-k KL | 精神上最接近；需要多 teacher 扩展 |
| [Tinker cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) | 单 + 多教师 recipe | 多 teacher 但 token 级 KL，无 per-prompt 路由 |

---

## 相关阅读

- [[on-policy-distillation]] —— OPD 总伞页（GKD 原 paper、数学、变体、OPD-vs-RL 争论）。MOPD 在那页的变体小节里有提。
- [[deepseek-v4-opd]] —— 另一个 2026 旗舰多教师 OPD 部署。架构对比见 [本页背景对比表](#背景为什么需要发明-mopd)。
- [[grpo]] —— Cascade 2 专项 RL 阶段（IF-RL、Multi-domain RL、Code RL、SWE RL）用的 RL 算法。
- [[ppo-for-llm]] —— Eq. 3 重要性裁剪背后的 trust-region 直觉。
- [[rlhf-overview]] —— MOPD 在步效率上替代的部分阶段。
- [[rl-training-frameworks]] —— NeMo-RL 是 MOPD 会实现的地方；veRL 有最接近的现成多教师原语。
- [[nemo-gym]] —— 容纳任何 Cascade RL / MOPD 管线的 rollout 一侧。
- [[das-spec-rl]] —— rollout 阶段的投机解码加速；推理层互补。
- [[prorl-agent]] —— 与 SWE RL 阶段相邻的 rollout-即-服务基础设施。

## 参考文献

- **Nemotron-Cascade 2 tech report**：Yang et al., NVIDIA。[arXiv:2603.19220](https://arxiv.org/abs/2603.19220) · [项目页](https://research.nvidia.com/labs/nemotron/nemotron-cascade-2/) · [PDF](https://research.nvidia.com/labs/nemotron/files/Nemotron-Cascade-2.pdf)
- **模型与数据**：[Nemotron-Cascade-2-30B-A3B](https://huggingface.co/nvidia/Nemotron-Cascade-2-30B-A3B) · [SFT 数据](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-SFT-Data) · [RL 数据](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-RL-data)
- **Nemotron-Cascade 1**（前置）：Wang et al. [arXiv:2512.13607](https://arxiv.org/abs/2512.13607)
- **Xiaomi MiMo-V2-Flash**（*另一个* MOPD）：[arXiv:2601.02780](https://arxiv.org/abs/2601.02780) · [GitHub](https://github.com/XiaomiMiMo/MiMo-V2-Flash) · [Xiaomi MiMo 推文](https://x.com/XiaomiMiMo/status/2000930865757741342)
- **GKD**（底层 OPD 技术）：Agarwal et al., ICLR 2024。[arXiv:2306.13649](https://arxiv.org/abs/2306.13649)
- **Thinking Machines Lab OPD 博客**（Cascade 2 引为 ref [42]）：[thinkingmachines.ai/blog/on-policy-distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)
- **DeepSeek-V4 OPD**（架构对比）：[HF PDF](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf)
- **OPD Survey 2026**：[arXiv:2604.00626](https://arxiv.org/abs/2604.00626) —— 把 Nemotron-Cascade 2 / DeepSeek-V4 / MiMo-V2-Flash / GLM-5 / KAT-Coder-V2 / ORBIT / Uni-OPD 归为 "industrial multi-teacher/multi-domain OPD" 簇。
- **独立解读**：[Maxime Labonne, *Nemotron Cascade 2: On-policy distillation is back!*](https://maximelabonne.substack.com/p/nemotron-cascade-2-on-policy-distillation) · [VentureBeat](https://venturebeat.com/orchestration/nvidias-nemotron-cascade-2-wins-math-and-coding-gold-medals-with-3b-active) · [MarkTechPost](https://www.marktechpost.com/2026/03/20/nvidia-releases-nemotron-cascade-2-an-open-30b-moe-with-3b-active-parameters-delivering-better-reasoning-and-strong-agentic-capabilities/) · [Ritvik Rastogi 讲解](https://ritvik19.medium.com/papers-explained-552-nemotron-cascade-2-1ac869c28c8c)
- **NeMo-RL 实现**：[GitHub](https://github.com/NVIDIA-NeMo/RL) · [Discussion #1445](https://github.com/NVIDIA-NeMo/RL/discussions/1445) · [OPD 文档](https://docs.nvidia.com/nemo/rl/0.5.0/about/algorithms/on-policy-distillation.html)
- **veRL 多教师 OPD**：[docs](https://verl.readthedocs.io/en/latest/algo/opd.html) · [异步 OPD docs](https://verl.readthedocs.io/en/latest/advance/async-on-policy-distill.html)
- **HF TRL GKDTrainer**：[trl/trainer/gkd_trainer.py](https://github.com/huggingface/trl/blob/main/trl/trainer/gkd_trainer.py)
