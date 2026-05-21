---
title: "On-Policy Distillation (OPD)：用稠密的教师信号替代 RL"
category: rl-infra
tags: [on-policy-distillation, opd, gkd, minillm, distillation, rl-post-training, reverse-kl, family-overview]
created: 2026-05-19
updated: 2026-05-22
status: mature
paper: arXiv:2306.13649
code: https://github.com/huggingface/trl/blob/main/trl/trainer/gkd_trainer.py
---

# On-Policy Distillation (OPD)：用稠密的教师信号替代 RL

> [!info] 谱系元信息
> - **起源 paper (GKD)**：[arXiv:2306.13649](https://arxiv.org/abs/2306.13649) —— *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes*（Agarwal, Vieillard, Zhou, Stańczyk, Ramos, Geist, Bachem；DeepMind；ICLR 2024）。论文标题就是字面意义上的 "on-policy distillation of language models" —— 这 **就是** OPD 原始论文。
> - **配套 paper**：[arXiv:2306.08543](https://arxiv.org/abs/2306.08543) —— *MiniLLM: Knowledge Distillation of Large Language Models*（Gu, Dong, Wei, Huang；Microsoft / Tsinghua；NeurIPS 2024；v3 改名为 *MiniLLM: On-Policy Distillation of Large Language Models*）。提供 OPD = 策略梯度的推导。
> - **2025 重新框架**：[Thinking Machines Lab 博客](https://thinkingmachines.ai/blog/on-policy-distillation/)（Kevin Lu, 2025-10-27）。**不是新论文** —— 是 Qwen3 规模上的重新包装，让 "OPD" 这个标签和"RL 替代品"叙事流行起来。
> - **参考实现**：[HF TRL `GKDTrainer`](https://github.com/huggingface/trl/blob/main/trl/trainer/gkd_trainer.py)；veRL `algo/opd`、NVIDIA NeMo-RL、TML `tinker-cookbook` 也有。

---

## 摘要（2 分钟读完这一节就够）

**它是什么**。On-Policy Distillation (OPD) 是一族 post-training 技术：student 自己采 rollout，**冻结**的 teacher 通过每 token 反向 KL 给每个生成 token 打分。谱系是 **GKD (2023, ICLR 2024) → MiniLLM (2023) → Thinking Machines Lab 重新框架 (2025-10) → 2025-26 涌现的 10+ 个变体**。"OPD" 这个标签是 2025 年之后的市场叫法；算法本身就是 GKD 在 $(\lambda, \beta) = (1.0, \text{反向 KL})$ 上的特例。

**核心思想**。把 RL 的稀疏标量 reward 换成 teacher 的每 token log-probability，但仍然从 *当前 student* 采轨迹，让梯度方向对齐部署分布。三个支柱：

1. **反向 KL 是 mode-seeking** —— student 把概率质量集中到 teacher 高概率 token，而不是覆盖 teacher 的尾部。
2. **On-policy 轨迹** ($y \sim \pi_\theta$) —— 梯度在部署 student 实际会经过的状态上算，消除 SFT 的 compounding-error 病。
3. **不需要 reward / value model** —— 每 token teacher log-prob *就是* 稠密 reward 信号，整套 RL critic 基础设施塌缩。

数学上的标志洞察是 **策略梯度对偶**：每 token 反向 KL 就是 REINFORCE，每 token reward 等于 teacher log-ratio $\log(\pi_T/\pi_\theta)$，而 KL 本身又同时充当 trust-region 正则项。**OPD = 去掉稀疏 outcome reward 和 value head 的 GRPO**。去掉任一支柱：失去 mode-seeking 就浪费在覆盖 teacher 尾部（正向 KL）；失去 on-policy 就重引入 SFT 的分布偏移问题；带回 reward model 就重新发明了 RLHF。

**头条结果**。TML 2025-10 在 Qwen3-8B-Base 上的复现：

| 方法 | AIME'24 | 计算量（GPU-h，约） |
| ---- | ------: | ------------------: |
| 仅 SFT (400 K prompt) | 60 % | — |
| Qwen3 RL recipe | 67.6 % | ~17,920 |
| **On-Policy Distillation** | **74.4 %** | **~1,800** |

~10× 计算降低 + AIME 反而更高。10× 这个方向被 Qwen3 自己的 tech report 印证（OPD 阶段 1/10 GPU-h）；TML 进一步声称自蒸馏 50–100× 这一端到 2026-05 没有独立复现。

**为什么这重要**。

- **生产已验证**。到 2026 年中三个旗舰 recipe：NVIDIA Nemotron-Cascade 2（MOPD 与 cascade RL 交错）、Alibaba Qwen3 小模型、DeepSeek-V4（多教师全词表 OPD 完全替换 mixed-RL 阶段）。
- **塌缩 RL 基础设施**。有强 teacher 时，OPD 一招干掉 reward model、value head 和 credit-assignment 问题。
- **不是 RL 杀手**。OPD 是 imitation learning —— 被 teacher 能力封顶。前沿能力扩展（没 teacher）RL 仍有不可替代角色。
- **2026–27 预测**。多教师全词表 OPD 变成有 teacher 时的默认 post-training；OPD+RL 混合（KDRL、dGRPO）替代纯 RL 用于其它场景。

---

# 深度部分（往下展开细节）

上面摘要是 executive 层。下面是给愿意细读谱系、数学推导、变体分类和生产 recipe 细节的人准备的。

## 背景：为什么需要发明 on-policy 蒸馏

LLM post-training 把强 teacher 或可验证 reward 的能力迁移到小学生 / specialist 学生上，现有两条路，结构上各有缺陷：

| 路径 | 轨迹分布 | reward 密度 | 失败模式 |
| ---- | -------- | ----------- | -------- |
| **SFT / 离线 KD** | $y \sim \mathcal{D}$（teacher 出的固定语料） | 稠密（per-token 软标签） | Student 推理时走到 $\mathcal{D}$ 之外；误差 compound；学了风格没学行为 |
| **RL（[[grpo|GRPO]]、[[ppo-for-llm|PPO]]、DPO）** | $y \sim \pi_\theta$（student 现采） | **稀疏** —— 每 episode O(1) bit | 16K-token rollout 大多数 token 没 credit assignment；用一个标量反推哪些 token 起作用很贵 |

第一条 reward 密度对了但轨迹分布错了；第二条轨迹分布对了但 reward 密度错了。GKD 的贡献是这"看见就显然"的第三条路：**on-policy** 轨迹（$y \sim \pi_\theta$）配 **token 级稠密** 监督（每 token 反向 KL 到 teacher）。

五个竞争方法在同一组维度上对比：

| 方法 | On-policy ($y \sim \pi_\theta$)？ | 每 token 信号？ | 要 reward model？ | 要 value head？ |
| ---- | --------------------------------- | -------------- | ---------------- | -------------- |
| SFT + 软标签 | ✗ | ✓ | ✗ | ✗ |
| RLHF (PPO) | ✓ | ✗ | ✓ | ✓ |
| GRPO | ✓ | ✗ | ✓ | ✗ |
| DPO | ✗（preference 数据） | ✗ | ✗ | ✗ |
| **OPD / GKD** | **✓** | **✓** | **✗** | **✗** |

OPD 是唯一一行前两列都打钩、后两列都不打钩的。

## 前置概念：KL、on-policy、credit assignment、value head

后面所有数学依赖的四个概念。如果已经熟练可以跳过 —— 后续不再解释。

### KL 散度

**Kullback-Leibler 散度**衡量同一定义域上两个概率分布 $P, Q$ 的差异：

$$
\mathrm{KL}(P \,\|\, Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)} = \mathbb{E}_{x \sim P}\!\left[\log \frac{P(x)}{Q(x)}\right]
$$

性质：

- $\mathrm{KL}(P\|Q) \ge 0$，当且仅当 $P = Q$ 时取等
- **不对称**：$\mathrm{KL}(P\|Q) \ne \mathrm{KL}(Q\|P)$。这一非对称性是 "forward" vs "reverse" 之分的根源
- 信息论解读：用为 $Q$ 优化的编码去编码服从 $P$ 的样本，平均多花的比特数

期望是对 $P$ 取的 —— 积分只在"第一个参数"有质量的地方才有贡献。这就是两种方向行为差异的根本。

### Forward vs Reverse KL —— mode-covering vs mode-seeking

对同一对 $(P_{\text{target}}, P_{\text{model}})$ 的两种方向：

| 方向 | 公式 | 期望对谁取 | 行为 |
| ---- | ---- | --------- | ---- |
| **Forward KL** | $\mathrm{KL}(P_{\text{target}} \| P_{\text{model}})$ | $x \sim P_{\text{target}}$ | **Mode-covering**：target 有质量的地方 model 必须放质量，否则 $\log(P_{\text{target}}/P_{\text{model}})$ 爆炸。覆盖尾部 |
| **Reverse KL** | $\mathrm{KL}(P_{\text{model}} \| P_{\text{target}})$ | $x \sim P_{\text{model}}$ | **Mode-seeking**：model 不能在 target 没有质量的地方放质量，但可以忽略 target 的尾部（model 不在那里采样）。集中在最高概率 mode |

具体 LM 例子。假设老师下一 token 分布是 `the:0.35, a:0.25, this:0.15, [47 个尾部 token 合计]:0.25`。一个容量有限的学生：

- **Forward KL** 强制学生在所有 50 个 token 上分配非零概率，包括那 47 个尾部 —— 否则在老师采到的每个尾部 token 上都要付 $-\log 0 = +\infty$。容量浪费在尾部建模上
- **Reverse KL** 允许学生几乎全部质量放在 `{the, a, this}` —— 期望是对学生自己分布取的，学生不采的地方根本不算损失。容量集中到老师的 mode 上

对 LLM 蒸馏，**我们要 mode-seeking**，原因有二：(1) 尾部大多是噪声 / 罕见事件，学生不需要建模；(2) 推理时学生每步只生成一个 token，不是输出完整分布 —— 把质量集中在"老师最偏好的几个 token"上正是我们要的行为。

[GKD paper](https://arxiv.org/abs/2306.13649) 在生成任务上实验证实：reverse KL > forward KL > MLE，尤其当学生容量明显小于老师时差距更大。

### "每 token 反向 KL" 究竟是什么

在 Transformer LM 中，每个生成位置 $t$ 上学生和老师**各自对整个词表 $V$**（通常 $|V| \approx$ 10 万-20 万）输出一个完整的概率分布：

- 老师：$\pi_T(\cdot \mid y_{<t}, x)$ —— $|V|$ 维概率向量
- 学生：$\pi_\theta(\cdot \mid y_{<t}, x)$ —— $|V|$ 维概率向量

**位置 $t$ 上的每 token 反向 KL**：

$$
\mathrm{KL}\!\big(\pi_\theta(\cdot|y_{<t},x) \,\|\, \pi_T(\cdot|y_{<t},x)\big) = \sum_{v \in V} \pi_\theta(v|y_{<t},x) \log \frac{\pi_\theta(v|y_{<t},x)}{\pi_T(v|y_{<t},x)}
$$

"每 token" 的意思是：**每个位置 $t$ 有自己独立的一个 KL**，在该位置学生和老师的词表分布之间计算。OPD 损失把这些位置 KL 全部加起来。期望对学生分布取（所以叫 reverse）。

### "on-policy" 是什么意思

"On-policy" 意思是 **训练用的轨迹来自当前学生** $\pi_\theta$，**不是**来自固定数据集，也**不是**来自老师。

跟 off-policy SFT 的对比很尖锐：

| | Off-policy SFT / KD | On-policy OPD |
| --- | ------------------- | -------------- |
| 轨迹来源 | 固定语料（老师 rollout、人工 demo） | **当前学生 rollout 新轨迹** |
| 训练时的状态分布 | "老师空间"——老师会访问的状态 | "学生空间"——学生推理时实际访问的状态 |
| 失败模式 | **协变量偏移 / 错误累积**：学生训练时见到的状态推理时永远见不到。推理时一旦犯错就进入陌生地带，错误级联放大 | 没有这个问题——梯度直接在部署模型实际遇到的状态上计算 |

这正是 **DAGGER（Ross, Gordon, Bagnell, AISTATS 2011）** 的洞察：模仿学习者要在 **自己的错误** 上被纠正，而不是在专家的干净轨迹上。OPD 就是 LLM token 序列上的 DAGGER。

实现上 on-policy 意味着每个训练 step 做：

1. 学生 rollout 新一条轨迹：$y \sim \pi_\theta(\cdot \mid x)$ —— 通常长度到 `max_new_tokens`
2. 对 $y$ 中每个位置 $t$，在相同前缀 $(x, y_{<t})$ 上**跑老师 forward**，得到 $\pi_T(\cdot \mid y_{<t}, x)$
3. 计算学生和老师分布的每 token 反向 KL
4. 反传给学生

老师 forward 是开销大头（每条训练 rollout 一次老师 forward）。工程技巧 —— top-k KL、hidden-state caching、FP4 量化老师 —— 主要都是为了让这一步可负担。

### Credit assignment、sparse reward 和 value head

这三个概念解释了 **RL 付出的开销**，也正是 OPD 省下来的部分。

**Credit assignment（信用分配）** 是要弄清楚一条轨迹中哪些动作（token）该为最终结果负责（或受表扬）。LLM RL 典型场景：

- 学生 rollout 一条 500 token 的数学解答
- 验证器在最后返回 **1 个 scalar**：答案对了 `reward = 1`，错了 `reward = 0`
- 要更新每个 token 的 logprob，你得知道"这个 token 对成功有没有贡献" —— 但你只有 500 token 末尾的 1 bit 信息

奖励是 **sparse**（大部分 token 是 0）、**delayed**（信号在末尾才出现）、**coarse-grained**（序列级而非 token 级）。各算法各有 baseline 的解法：

| 算法 | 怎么做 credit assignment |
| ---- | ----------------------- |
| **原始 REINFORCE** | 整条轨迹每个 token 用同一个末尾 scalar 当权重。方差超高、训练不稳 |
| **PPO** | 训一个 **value head** $V_\phi(s_t)$ 预测从状态 $s_t$ 出发的期望回报。每 token 算 advantage $A_t = (r_t + \gamma V_\phi(s_{t+1})) - V_\phi(s_t)$，用 advantage 当 per-token 权重 |
| **GRPO** | 每个 prompt 采 $N$ 条 rollout，用组均值当 baseline：$A_t = R_i - \bar R$。不用 value head，但 rollout 成本 $\times N$ |

**Value head** 是一个小 MLP —— 通常就是策略模型最后 hidden state 之上接一个 Linear($H \to 1$) —— 预测一个 scalar："从这个状态往后的期望总回报"。7B PPO 配置下额外 ~7K 参数（参数量可忽略），但 value 分支的 forward/backward 翻倍，且要调 value loss。实现见 TRL 的 `AutoModelForCausalLMWithValueHead`（[`trl/models/modeling_value_head.py`](https://github.com/huggingface/trl/blob/main/trl/models/modeling_value_head.py)）。

**OPD 为什么能绕掉这一整套**。老师在每个位置都提供一个 *完整分布* —— 这就是 token 级稠密信号。没有 sparse reward、没有 credit assignment、不需要 value head。"奖励" $\log(\pi_T(y_t)/\pi_\theta(y_t))$ 自带 per-token 信息量、方差天然低。这个 RL critic 基础设施的整体坍缩，才是 OPD vs RL 的真正计算节省 —— 比任何算法新意都重要。

## 谱系：GKD → MiniLLM → TML 重框架 → 变体

技术家族的时间发展，附标志文献：

| 日期 | 工作 | 贡献 |
| ---- | ---- | ---- |
| 2010 | [DAGGER](https://arxiv.org/abs/1011.0686)（Ross, Gordon, Bagnell） | 前 LLM 时代的祖宗：on-policy imitation learning + expert correction。OPD 是 DAGGER 用到 LLM token 序列上的版本。 |
| 2023-06 | [**GKD**](https://arxiv.org/abs/2306.13649)（DeepMind, Agarwal et al.） | OPD 算法本身。用 $(\lambda, \beta)$ 旋钮泛化 KD；纯 OPD 是 $(\lambda{=}1, \text{反向 KL})$。ICLR 2024。 |
| 2023-06 | [**MiniLLM**](https://arxiv.org/abs/2306.08543)（Microsoft / Tsinghua, Gu et al.） | 独立的同期推导；显式证明 OPD = REINFORCE 配 teacher log-ratio reward。NeurIPS 2024。v3 改名 "On-Policy Distillation of Large Language Models"。 |
| 2024 | HF TRL `GKDTrainer` 落地 | 标志性开源实现；2025 多数工作都基于它。 |
| 2025-05 | [**Qwen3 tech report**](https://arxiv.org/abs/2505.09388) | 首个旗舰生产部署：离线 → on-policy 蒸馏替代 0.6B–14B + 30B-A3B MoE 小模型完整 RL 管线的 stage 3-4。报告 1/10 GPU-h 成本。 |
| 2025-06 | [**KDRL**](https://arxiv.org/abs/2506.02208)（Xu, Zhu et al.） | 首个干净的 OPD+RL 混合：把 GRPO 的 KL-to-old-policy 换成 KL-to-teacher；同一梯度步联合规则 reward + OPD。 |
| 2025-10 | [**TML 博客**](https://thinkingmachines.ai/blog/on-policy-distillation/)（Kevin Lu） | 把 GKD 重新框架成 "OPD" 和 *RL 替代品*。标志数字：Qwen3-8B-Base 74.4 % AIME'24 @ ~1,800 GPU-h vs Qwen3 RL recipe 67.6 % @ ~17,920 GPU-h。 |
| 2025-11 | [**Black-Box OPD / GAD**](https://arxiv.org/abs/2511.10643)（Ye, Dong et al.） | 只能拿到输出文本（看不到 logits）时的 OPD —— OpenAI / Anthropic teacher 用得上。对抗判别器。 |
| 2026-03 | [**NVIDIA Nemotron-Cascade 2**](https://research.nvidia.com/labs/nemotron/nemotron-cascade-2/) | 引入 MOPD —— 7-stage Cascade RL 内的单一 OPD 稳定阶段，3 个 cascade 内部 teacher。IMO/IOI/ICPC 2025 金牌在 3B 激活参数上。 |
| 2026-04 | [**DeepSeek-V4**](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) | 首个 *完全替换* mixed-RL 阶段的旗舰：多教师全词表 OPD。1.6T/49B MoE。 |
| 2026-04 | [**Rethinking OPD**](https://arxiv.org/abs/2604.13016)（清华） | 把成功 OPD 刻画为 97-99 % 共享高概率 token 集上的对齐；识别跨家族负迁移。 |

"OPD 作为概念" 是 TML 2025-10 博客让它流行的；算法本身从 2023 起被叫过 GKD、on-policy KD、RKL-KD 等多个名字。

## 数学核心

### OPD loss

GKD 的总形式，$\lambda \in [0, 1]$ 控制 on-policy 比例，$D$ 是按 $\beta$ 参数化的 generalized JSD：

$$
\mathcal{L}_{\text{GKD}}(\theta) = (1{-}\lambda)\,\mathbb{E}_{(x,y)\sim \mathcal{D}}\!\left[D(\pi_T\|\pi_\theta)(y\mid x)\right] + \lambda\,\mathbb{E}_{x,\,y\sim \pi_\theta}\!\left[D(\pi_T\|\pi_\theta)(y\mid x)\right]
$$

纯 OPD = GKD 在 $(\lambda, \text{方向}) = (1.0, \text{反向 KL})$ 下的特例：

$$
\mathcal{L}_{\text{OPD}}(\theta) = \mathbb{E}_{x,\,y\sim\pi_\theta(\cdot\mid x)}\!\left[\sum_{t=1}^{|y|} D_{\text{KL}}\!\big(\pi_\theta(\cdot\mid y_{<t}, x)\,\big\|\,\pi_T(\cdot\mid y_{<t}, x)\big)\right]
$$

| 旋钮 | 作用 | TML OPD 取值 |
| ---- | ---- | ----------- |
| $\lambda$ | On-policy 比例。0 = 纯 SFT-带软标签；1 = 纯 on-policy。 | 1.0 |
| $\beta$（KL 方向） | 0 = 正向 KL（mean-seeking）；1 = 反向 KL（mode-seeking）；0.5 = 对称 JSD。 | 1.0（反向） |
| Discount $\gamma$ | 跨轨迹时间折扣。 | 0（reward 本来就 token 级稠密） |

### 与策略梯度的对偶 —— 完整推导

这是让 [[grpo|GRPO]] / [[ppo-for-llm|PPO]] 出身的人秒懂的结果。推导很短，完整列出。

**Step 1.** 从纯 OPD 目标出发（单个位置 $t$，下标暂时省略）：

$$
J(\theta) = \mathbb{E}_{y \sim \pi_\theta}\!\left[D_{\text{KL}}(\pi_\theta \| \pi_T)\right] = \mathbb{E}_{y \sim \pi_\theta}\!\left[\log \frac{\pi_\theta(y)}{\pi_T(y)}\right]
$$

期望对学生分布取。注意期望*和*被积函数都依赖 $\theta$。

**Step 2.** 求梯度。因为采样分布也依赖 $\theta$，要用 score function（REINFORCE）恒等式 $\nabla_\theta \mathbb{E}_{y \sim \pi_\theta}[f(y)] = \mathbb{E}_{y \sim \pi_\theta}[f(y) \nabla_\theta \log \pi_\theta(y) + \nabla_\theta f(y)]$：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{y \sim \pi_\theta}\!\left[\log\frac{\pi_\theta(y)}{\pi_T(y)} \cdot \nabla_\theta \log \pi_\theta(y) \;+\; \nabla_\theta \log \pi_\theta(y)\right]
$$

第二项期望为 0（$\mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta] = 0$，标准 score function 恒等式），所以去掉：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{y \sim \pi_\theta}\!\left[\log\frac{\pi_\theta(y)}{\pi_T(y)} \cdot \nabla_\theta \log \pi_\theta(y)\right]
$$

**Step 3.** 翻符号（我们是要 *最小化* KL，所以梯度*下降*方向是 $-\nabla_\theta J$）：

$$
-\nabla_\theta J(\theta) = \mathbb{E}_{y \sim \pi_\theta}\!\left[\log\frac{\pi_T(y)}{\pi_\theta(y)} \cdot \nabla_\theta \log \pi_\theta(y)\right]
$$

**Step 4.** 把所有位置加起来。完整 OPD 梯度（MiniLLM §3）：

$$
\boxed{\;-\nabla_\theta \mathcal{L}_{\text{OPD}} \;=\; \mathbb{E}_{y \sim \pi_\theta}\!\left[\sum_{t} \nabla_\theta \log \pi_\theta(y_t \mid y_{<t}) \cdot \underbrace{\log\frac{\pi_T(y_t \mid y_{<t})}{\pi_\theta(y_t \mid y_{<t})}}_{\text{每 token 稠密"reward"}}\right]\;}
$$

**Step 5.** 跟 REINFORCE 策略梯度对比。对 RL 目标 $J_{\text{RL}}(\theta) = \mathbb{E}_{y \sim \pi_\theta}[R(y)]$：

$$
\nabla_\theta J_{\text{RL}} = \mathbb{E}_{y \sim \pi_\theta}\!\left[\sum_t \nabla_\theta \log \pi_\theta(y_t \mid y_{<t}) \cdot R(y)\right]
$$

两个表达式**结构完全相同**。唯一差别是"奖励"是什么：

$$
R_{\text{OPD}}(s_t, a_t) \;=\; \log \frac{\pi_T(y_t \mid y_{<t})}{\pi_\theta(y_t \mid y_{<t})}
$$

**OPD 就是 REINFORCE，奖励 = teacher log-ratio 当作 per-token 稠密 reward**。

### 为什么这个对偶是 load-bearing

合成奖励 $\log(\pi_T/\pi_\theta)$ 的三个性质，决定了 RL critic 那一整套全部消失：

| 性质 | 后果 |
| ---- | ---- |
| **Dense（稠密）**：每 token 都有非平凡数字，不只是末尾 | 不需要 credit assignment。不需要 value head 来把末尾稀疏 reward 反传到各位置 |
| **Informative（信息丰富）**：幅值告诉学生该往哪走 —— 老师比学生更偏好这 token 时，ratio > 1，梯度推学生靠近 | 不用 baseline 方差就够低。GRPO 那个组均值 baseline 也不需要了 |
| **Self-bounded（自界）**：学生与老师对齐时 ratio → 1，log → 0，梯度消失 | 收敛到老师分布。没有 reward hacking —— reward 是相对老师 *定义* 出来的。KL 项天然兼任 trust-region 正则（PPO 风格），不需要外加 KL penalty |

从 RL 到 OPD 的结构性坍缩：

| RL 组件 | OPD 等价物 |
| ------- | ---------- |
| Reward model | 老师 LM forward |
| 稀疏 outcome reward $R(y) \in \{0, 1\}$ | 稠密 per-token $\log(\pi_T/\pi_\theta)$ |
| Value head（PPO） | 不需要 —— reward 已经是 token 级 |
| Group-mean baseline（GRPO） | 不需要 —— 方差已经很低 |
| 重要性比率 + clip $\min(r_t A_t, \text{clip}(r_t) A_t)$ | 不需要 —— 按构造完全 on-policy |
| KL-to-old-policy penalty | 已内置于 loss —— KL-to-teacher 本身就是 loss |
| Discount $\gamma$ | 设 0 —— 不需要 credit 传播 |

> [!quote] 心智模型
> OPD = GRPO 目标，把稀疏 outcome reward $R(y) \in \{0, 1\}$ 换成稠密 token 级信号 $\log(\pi_T/\pi_\theta)$，再把 value head 拿掉（reward 已经是 token 级）。KL-to-teacher **既产生**梯度信号，**又约束**每步策略移动的幅度。

### 为什么 "on-policy" 重要

期望 $\mathbb{E}_{y\sim\pi_\theta}$ 是在 *student 自己的* 轨迹分布上算的。把它换成 $\mathbb{E}_{y\sim\mathcal{D}}$（固定语料）就是 GKD 的 $\lambda{=}0$，等于带软标签的 SFT：

| 设置 | 采样 | 效果 |
| ---- | ---- | ---- |
| $\lambda = 0$ | $y \sim \mathcal{D}$ | SFT 带软标签；相对部署状态有偏；漂移 compound |
| $\lambda = 1$ | $y \sim \pi_\theta$ | On-policy；梯度在部署 student 实际经过的状态上 |
| $0 < \lambda < 1$ | 混合 | 稳定性 vs on-policy 相关性的折中 |

这也是 [DAGGER (Ross, Gordon, Bagnell, 2010)](https://arxiv.org/abs/1011.0686) 的核心思想 —— 见 [OPD vs RL 争论](#opd-vs-rl-争论)。

### Token 级 vs 全词表 KL

二阶旋钮但在规模上一阶后果。两种算 KL 的方式：

| 形式 | 在测什么 | 梯度形式 | 方差 | 内存 / 带宽 |
| ---- | -------- | -------- | ---- | ----------- |
| **Token 级**（TML、MiniLLM、多数 OPD 论文） | 仅 *被采到的 token* 上的 KL | $\nabla_\theta \log\pi_\theta(y_t)\cdot \log(\pi_T(y_t)/\pi_\theta(y_t))$ | 高（$V$ 维分布的单 token 采样） | 每 token $O(1)$ |
| **全词表**（[[deepseek-v4-opd|DeepSeek-V4]]） | 全部 $V$ 个 token 上的解析 KL | $\sum_v \pi_\theta(v) \log(\pi_\theta(v)/\pi_T(v))$ | 低（精确） | 每 token $O(V)$ |

Token 级是 HF TRL 实现的版本，也是 TML 用的；全词表是 DeepSeek-V4（2026-04）的主张 —— V4 认为旗舰规模下 token 级估计器的方差在长 rollout 上累积，必须用全词表。

## 变体分类

2025-26 在用的命名变体。每行标的是相对 vanilla GKD-with-$\lambda{=}1$ 的 *delta*。

| 变体 | 起源 | 相对 vanilla OPD 的关键差异 |
| ---- | ---- | -------------------------- |
| **OPSD**（Self-Distillation）([arXiv:2602.04942](https://arxiv.org/abs/2602.04942)) | 2025-26 | Teacher 是 student 自己的早期 checkpoint 或它的特权信息版本。Continual learning 原语。 |
| **KDRL** ([arXiv:2506.02208](https://arxiv.org/abs/2506.02208)) | Xu, Zhu 等, 2025-06 | 把 GRPO 的 KL-to-old-policy 换成 KL-to-teacher；同一个梯度步联合优化规则 reward + OPD。 |
| **dGRPO** ([survey](https://arxiv.org/abs/2604.00626)) | 2025-26 | GRPO advantage + per-token OPD loss 作为稠密辅助 head。 |
| **MOPD**（Multi-Domain）([Nemotron-Cascade 2](https://research.nvidia.com/labs/nemotron/nemotron-cascade-2/)) | NVIDIA, 2026-03 | 7-stage Cascade RL 内的单一稳定阶段；3 个 cascade 内部 teacher 按 prompt 路由；采样 token 反向 KL + 重要性裁剪。详见 [[mopd]]。**注意**：相同缩写两个月前被 Xiaomi MiMo-V2-Flash 用过，含义是 "Multi-**Teacher** OPD"。 |
| **MAD-OPD** ([arXiv:2605.01347](https://arxiv.org/abs/2605.01347)) | 2026 | 多智能体辩论当 teacher 信号。试图突破单 teacher 天花板。 |
| **Reward-Extrapolated OPD** ([arXiv:2602.12125](https://arxiv.org/abs/2602.12125)) | 2026 | 加 RL reward head 让 student 能学超过 teacher。 |
| **Black-Box OPD (GAD)** ([arXiv:2511.10643](https://arxiv.org/abs/2511.10643)) | Ye, Dong 等, 2025-11 | 只能拿到输出文本（看不到 logits）时的 OPD —— 用对抗判别器。OpenAI / Anthropic teacher 用得上。 |
| **多教师全词表 OPD** ([DeepSeek-V4](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)) | DeepSeek, 2026-04 | $\sum_i w_i D_{\text{KL}}(\pi_\theta\|\pi_{E_i})$ 跨 10+ specialist，全词表 KL。旗舰规模演示。详见 [[deepseek-v4-opd]]。 |

变体可以拆成三个轴：**teacher 是什么**（单 / 多教师 / 自蒸馏 / 辩论 / 黑盒）、**加了什么**（额外 RL reward、RL exploration 项、重要性裁剪）、**KL 怎么算**（token 级 / 全词表 / top-k 受限）。

## 生产部署

把 OPD 装进生产管线的旗舰 recipe。

| 部署 | Recipe | 来源 |
| ---- | ------ | ---- |
| **NVIDIA Nemotron-Cascade 2**（2026-03） | 30B-A3B MoE。7-stage Cascade RL 里 Multi-domain RL 和 RLHF 之间的单一 MOPD 阶段。3 个 cascade 内部 teacher（math SFT / RLHF 侧分支 / multi-domain RL best）。52 步恢复 160 步 RLHF 才能恢复的东西。**IMO/IOI/ICPC 2025 金牌** 在 3B 激活参数上。详见 [[mopd]]。 | [Nemotron-Cascade 2 页](https://research.nvidia.com/labs/nemotron/nemotron-cascade-2/) |
| **Alibaba Qwen3 小模型**（2025-05） | 0.6B–14B + 30B-A3B-MoE。离线蒸馏（teacher：更大 Qwen3）→ on-policy 蒸馏。替代完整 RL 管线的 stage 3–4。**报告 1/10 GPU-hour 成本**。 | [Qwen3 tech report](https://arxiv.org/abs/2505.09388) |
| **DeepSeek-V4**（2026-04） | 1.6T/49B MoE。Per-domain (SFT → GRPO) specialist → 多教师全词表 OPD merge。**完全替换** V3.2 mixed-RL 阶段。详见 [[deepseek-v4-opd]]。 | [V4 tech report](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf) |

### 标志数字（TML Qwen3-8B 复现）

TML 2025-10 博客用 Qwen3-8B-Base 当 student、Qwen3-32B 当 teacher 复现 Qwen3 OPD recipe：

| 方法 | AIME'24 | 计算量（GPU-h，约） |
| ---- | ------: | ------------------: |
| 仅 SFT (400K prompt) | 60 % | 未报告 |
| SFT 外推到 2M | ~70 % | 未报告 |
| Qwen3 RL recipe | 67.6 % | ~17,920 |
| **On-Policy Distillation** | **74.4 %** | **~1,800** |

> [!success] ~10× 计算降低 + AIME 反而更高
> TML 博客进一步声称自蒸馏 50–100×；**100× 这个数到 2026-05 没有独立复现**。Qwen3 tech report 的 1/10 GPU-h 印证了 ~10× 方向；更高那端是单实验室数字。

### 哪些地方 *没* 用 OPD

为了校准：**DeepSeek-R1 → 小学生**用 **SFT-only 离线**蒸馏（~800K 已验证 trace）。**Meta Llama 4** 用 codistillation 配动态软/硬目标加权 —— 公开材料没描述 student-rollout 形式的 on-policy。**Anthropic、OpenAI、Mistral、Cohere** —— 到 2026-05 公开材料里没有 on-policy distillation 的证据。

## 优势与限制

最强两点：(1) **token 级稠密梯度信号 + 不要 reward / value model**，有 teacher 时把多数 RL post-training 基础设施塌缩；(2) **on-policy 轨迹** 消除 SFT 的 compounding-error 病，student 在它实际部署会经过的状态上训。

诚实承认的限制：

- **被 teacher 能力上限封顶**。反向 KL 是 imitation learning —— student 集中在 teacher 高概率 token 上，发现不了 teacher 没掌握的解。要超越 teacher 时 OPD 充其量是 warm-start。
- **冷启动脆弱**。反向 KL 要求 student 支撑覆盖 teacher 高概率 token。没有先 SFT 就没有，梯度爆炸。TML recipe 依赖 Qwen3-Base 已经充分预训练。
- **跨家族负迁移**。"thinking pattern" 不同的 teacher（不同 RL 历史、不同家族）可能 *降低* student。[清华 Rethinking OPD](https://arxiv.org/abs/2604.13016) 把成功 OPD 描述为在 97–99 % 共享高概率 token 集上的对齐。同家族 teacher 或 GOLD 跨 tokenizer 对齐。
- **有偏的 token 级估计器**。Vanilla token 级反向 KL 不是 sequence 级反向 KL 的无偏估计；方差在长 rollout 上累积（16K+ token reasoning trace 严重，agentic 场景更严重）。[DeepSeek-V4 的全词表 KL](#token-级-vs-全词表-kl) 是主流解法。
- **熵塌缩**。没有合适 KL 正则下 student 可能 mode-collapse 到 teacher 某个模式。KDRL 风格联合目标配 RL exploration 缓解。
- **成本声明要小心**。"比 RL 便宜 100×" 取决于摊销 teacher inference 和忽略生成成本。基于独立 Qwen3 证据的可辩护区间：5–20×。
- **Teacher inference 成本**。OPD recipe 假设 teacher logits 拿得到。Teacher 是闭源 API（GPT-4、Claude）时用 Black-Box OPD (GAD)。

> [!warning] "这就是 2010 年的 DAGGER" 批评
> Guohao Li ([推文，2025-11](https://x.com/guohao_li/status/1987821200060625175))：*"Thinking Machines Lab 出 On Policy Distillation 博客时，我的第一反应是它就是 15 年前的 DAGGER … 果然，他们提到了 DAGGER。"* 实质：on-policy imitation learning 在 [DAGGER (Ross, Gordon, Bagnell, 2010)](https://arxiv.org/abs/1011.0686) 已经被解决；OPD 是 DAGGER 用到 LLM token 序列上的版本。新的是 LLM 规模的工程，不是算法。引申意：OPD 继承 DAGGER 的 imitation-learning 天花板。

### OPD vs RL 争论

2025 年底起的标志性争论。综合立场（多数生产团队）是 **"两个都用"**：RL 干 exploration，OPD 干稳定 + 回归恢复。KDRL 是最干净的写法 —— 同一梯度步联合 KL-to-teacher + GRPO reward，报告 +4.7 %（vs SFT）、+2.6 %（vs GRPO）、+1.1 %（vs KD-RKL）。NVIDIA Nemotron-Cascade 2 在规模上采取同样的架构立场：MOPD *与* cascade RL 交错，不是替代。"纯 OPD 替代 RL" 这边最强辩护是 DeepSeek-V4，完全砍掉 mixed-RL 阶段换成多教师全词表 OPD —— 但前提是先有 per-domain SFT→GRPO 的 specialist 训练阶段产出 teacher。

## 这意味着什么

三条值得跟踪的预测：

1. **有 teacher 时 OPD 会替代 reasoning 模型的 mixed-RL post-training**。[[deepseek-v4-opd|DeepSeek-V4]] 是第一个全押的旗舰。预期 Qwen、Mistral、开源社区跟进。RL 在每代 *第一个* 模型（没 teacher）和前沿能力扩展上继续主导。
2. **多教师 OPD 会变成新默认**。单教师 OPD 继承单个天花板；多教师（V4 风格）让你合并 specialist。预期 2026 H2 起标配。
3. **有意思的研究不再在 loss 函数**。GKD 和 MiniLLM 在 2023 把数学钉死了。2026 的研究在：(a) 方差减少（全词表 KL、sequence 级修正），(b) 跨 tokenizer 对齐 (GOLD)，(c) 经济 teacher 服务（logit 缓存、FP4 QAT、隐藏状态缓存），(d) OPD+RL 混合目标 (KDRL、dGRPO)。

这 *不是*：万能 RL 杀手。当目标超越最强可得 teacher 时 OPD 只能 warm-start。RL 仍有不可替代的角色。

## 源码与复现

### HuggingFace TRL —— `GKDTrainer`

最权威的开源参考。文件：[`trl/trainer/gkd_trainer.py`](https://github.com/huggingface/trl/blob/main/trl/trainer/gkd_trainer.py)。两个关键部分：

```python
# lines 226-295 — generalized_jsd_loss
def generalized_jsd_loss(student_logits, teacher_logits, labels=None,
                        beta=0.5, temperature=1.0, reduction="batchmean"):
    # beta == 0 → 正向 KL；beta == 1 → 反向 KL；其它 → mixture via logsumexp
    ...

# lines 421-449 — training_step
def training_step(self, model, inputs, ...):
    if random.random() <= self.lmbda:                # lmbda = 1.0 ⇒ 纯 OPD
        inputs = self.generate_on_policy_outputs(...)
    ...
```

| `GKDConfig` 字段 | 作用 |
| ---------------- | ---- |
| `lmbda` | 每 batch 现采 student rollout 的概率。`1.0` = 纯 OPD。 |
| `beta` | KL 方向。TML OPD 取 `beta=1.0`（反向）。 |
| `temperature` | softmax 温度。 |
| `seq_kd` | 从 *teacher* 出 rollout —— teacher 样本上的 sequence 级 KD。 |
| `use_liger_kernel` | 融合 linear+JSD kernel 省内存。 |

> [!warning] TRL 正在弃用 `GKDTrainer`
> 当前 TRL 提示 *"This trainer will soon be moved to `trl.experimental` and is a candidate for removal."* 一个更新的 `DistillationTrainer` 在开发中：带 generation buffer（rollout batch 与 train microbatch 解耦，号称 ~40× 加速）、支持外部 teacher server。跟进 [TRL releases](https://github.com/huggingface/trl/releases) 和 [issue #4390](https://github.com/huggingface/trl/issues/4390)。

### 其它实现

| 项目 | 路径 | 备注 |
| ---- | ---- | ---- |
| **veRL** | [`algo/opd` docs](https://verl.readthedocs.io/en/latest/algo/opd.html) | `distillation.*` 配置命名空间。按 `data_source` 多教师路由（支持 MOPD）。`loss_mode={forward_kl_topk, k1, k3}`、`use_policy_gradient`、`use_task_rewards`。vLLM-hosted teacher via ZeroMQ。 |
| **NVIDIA NeMo-RL** | [Discussion #1445](https://github.com/NVIDIA-NeMo/RL/discussions/1445) | 正向 / 反向 / 混合 KL。**Top-k 限制 KL**（带宽优化）。Qwen3-4B-Base AIME'25 Avg@16：47.71 %（SFT+OPD）vs 30.42 %（SFT+离线）。 |
| **TML `tinker-cookbook`** | [github.com/thinking-machines-lab/tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) | `tinker_cookbook/recipes/distillation/` —— 单/多教师、on/off-policy、多轮 tool-use 变体。最接近 TML 博客 recipe。 |
| **HF H4 GOLD** | [HF Space](https://huggingface.co/spaces/HuggingFaceH4/on-policy-distillation) | 跨 tokenizer OPD（token-merge 对齐 + 乘积规则合并 logits）。会作为 `GOLDTrainer` 进 TRL。 |
| **Tsinghua `thunlp/OPD`** | [GitHub](https://github.com/thunlp/OPD) | "Rethinking OPD" 论文官方代码。冷启动修复、teacher 对齐 prompt 选择。 |

### 最小复现 recipe（TML 风格，用 TRL）

```python
from trl import GKDConfig, GKDTrainer
from transformers import AutoModelForCausalLM

student = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B-Base")
teacher = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-32B")

config = GKDConfig(
    output_dir="./opd_qwen3",
    lmbda=1.0,           # 纯 on-policy
    beta=1.0,            # 反向 KL
    temperature=1.0,
    max_new_tokens=16384,
    learning_rate=1e-6,
    per_device_train_batch_size=1,
)

trainer = GKDTrainer(
    model=student,
    teacher_model=teacher,
    args=config,
    train_dataset=load_math_prompts(),
)
trainer.train()
```

复现 TML AIME 数字的最小骨架 —— 实际生产用多教师路由、全词表 KL（DeepSeek-V4 路径）、`tinker-cookbook` / veRL 里的工程技巧。

## 相关阅读

- [[deepseek-v4-opd]] —— DeepSeek-V4 的多教师全词表 OPD recipe；旗舰规模实例。
- [[mopd]] —— NVIDIA Nemotron-Cascade 2 的 Multi-Domain OPD；生产中与 cascade RL 交错。
- [[grpo]] —— OPD 最常被对比 / 组合的 RL 算法；OPD 结构上是去稀疏 reward 的 GRPO。
- [[ppo-for-llm]] —— OPD KL-to-teacher 项共享的 trust-region 直觉。
- [[rlhf-overview]] —— OPD 替代的标准 RL post-training 管线。
- [[dpo]] —— 另一种 preference 风格 RL 替代品；与 OPD 正交。
- [[rl-training-frameworks]] —— 容纳 OPD 实现的 trainer 端库（OpenRLHF、TRL、veRL、NeMo-RL）。
- [[das-spec-rl]] —— RL / OPD rollout 阶段的投机解码加速；推理层互补。
- [[prorl-agent]] —— 容纳 RL + OPD 工作负载的 rollout-即-服务基础设施。

## 参考文献

- **GKD（OPD 原 paper）**：Agarwal et al., *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes*, ICLR 2024。[arXiv:2306.13649](https://arxiv.org/abs/2306.13649)
- **MiniLLM**：Gu et al., NeurIPS 2024。[arXiv:2306.08543](https://arxiv.org/abs/2306.08543)
- **TML 博客**：Kevin Lu, 2025-10-27。[thinkingmachines.ai/blog/on-policy-distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)
- **Qwen3 tech report**：[arXiv:2505.09388](https://arxiv.org/abs/2505.09388)
- **DeepSeek-V4 tech report**：[HF PDF](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf)
- **KDRL**：Xu, Zhu 等 (2025-06)。[arXiv:2506.02208](https://arxiv.org/abs/2506.02208)
- **OPD Survey**：[arXiv:2604.00626](https://arxiv.org/abs/2604.00626)
- **Rethinking OPD**：清华 (2026-04)。[arXiv:2604.13016](https://arxiv.org/abs/2604.13016) —— 代码 [thunlp/OPD](https://github.com/thunlp/OPD)
- **Revisiting OPD Failure Modes**：[arXiv:2603.25562](https://arxiv.org/abs/2603.25562)
- **Black-Box OPD (GAD)**：[arXiv:2511.10643](https://arxiv.org/abs/2511.10643)
- **NVIDIA Nemotron-Cascade 2**：[research.nvidia.com/labs/nemotron/nemotron-cascade-2](https://research.nvidia.com/labs/nemotron/nemotron-cascade-2/)
- **HF TRL GKDTrainer**：[trl/trainer/gkd_trainer.py](https://github.com/huggingface/trl/blob/main/trl/trainer/gkd_trainer.py)
- **veRL OPD docs**：[verl.readthedocs.io/en/latest/algo/opd.html](https://verl.readthedocs.io/en/latest/algo/opd.html)
- **NeMo-RL Discussion #1445**：[github.com/NVIDIA-NeMo/RL/discussions/1445](https://github.com/NVIDIA-NeMo/RL/discussions/1445)
- **Tinker cookbook**：[github.com/thinking-machines-lab/tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)
- **DAGGER（祖宗）**：Ross, Gordon, Bagnell (2010)。[arXiv:1011.0686](https://arxiv.org/abs/1011.0686)
