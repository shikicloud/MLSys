---
title: "On-Policy Distillation (OPD)：用稠密的教师信号替代 RL"
category: rl-infra
tags: [on-policy-distillation, opd, gkd, minillm, distillation, rl-post-training, reverse-kl, knowledge-transfer]
created: 2026-05-19
updated: 2026-05-19
status: mature
---

# On-Policy Distillation (OPD)：用稠密的教师信号替代 RL

> [!info] 本页范围
> 覆盖：(1) **技术本身**与谱系（GKD / MiniLLM → Thinking Machines Lab 重新包装的 "OPD"）；(2) 让 OPD 在 [[grpo|GRPO]] 一族里看得懂的 **策略梯度对偶**；(3) 2025–2026 在用的 **变体**（OPSD、KDRL、MOPD、MAD-OPD、Black-Box OPD）；(4) HF TRL / veRL / NeMo-RL 里的 **源码**；(5) **生产部署**（NVIDIA Nemotron-Cascade 2、Alibaba Qwen3 小模型）；(6) "OPD 替代 RL 还是仅 warm-start RL" 的 **争论**。DeepSeek-V4 的多教师全词表 OPD 单独成页：[[deepseek-v4-opd]]。

> [!abstract]+ TL;DR
> **On-Policy Distillation** 是反向 KL 蒸馏，但 student 自己采 rollout，**冻结**的 teacher 按 token 打分。数学上，这是 **策略梯度，每 token 的稠密 reward 等于 $\log(\pi_T/\pi_\theta)$** —— 形状跟 [[grpo|GRPO]] 一模一样，只是把稀疏的 outcome reward 和 value head 拿掉了，换成 KL-to-teacher，这个 KL 同时充当 reward *和* trust-region 正则项。Thinking Machines Lab 2025-10 博客的标志性数字：Qwen3-8B-Base 上 **74.4 % AIME'24 @ ~1,800 GPU-h**（Qwen3 自己的 RL 配方 67.6 % @ ~17,920 GPU-h）—— ~10× 计算效率提升；自蒸馏时官方声称 50–100×。技术本身 **不算新** —— 是 [GKD (Agarwal et al., 2023)](https://arxiv.org/abs/2306.13649) 在 $\lambda{=}1$、反向 KL 下的特例，也是 [MiniLLM (Gu et al., 2023)](https://arxiv.org/abs/2306.08543) 反向 KL 形式的重现。新的是 **从业者把它重新定位成 reasoning post-training 的 RL 替代品**。到 2026 年中，最强的生产部署是 **NVIDIA Nemotron-Cascade 2**（MOPD 与 cascade RL 交错）、**Alibaba Qwen3** 小模型、以及 **DeepSeek-V4**（多教师全词表 OPD **完全替代** V3.2 的 mixed-RL 阶段 —— 见 [[deepseek-v4-opd]]）。

---

## 起源

### 它声称解决什么问题

[Thinking Machines Lab 博客](https://thinkingmachines.ai/blog/on-policy-distillation/)（Kevin Lu, 2025-10-27）的开篇：LLM post-training 现有两条路，结构上各有缺陷：

- **SFT / 离线 KD** 是 *off-policy* —— 你在一个固定的 teacher 轨迹语料上训练，但部署后的 student 走到的 *是不同的* 状态分布。Student 训着训着漂走，梯度方向相对部署 student 实际会经过的状态就有偏，结果就是 compounding errors 和"学了风格、没学行为"。
- **RL（[[grpo|GRPO]]、[[ppo-for-llm|PPO]]、DPO）** 是 *on-policy* —— 轨迹相关性对了，但 reward 信号是 **每 episode O(1) bit**。16K-token 的推理 rollout 里绝大多数 token 没有逐 token 的 credit assignment；模型必须从单个稀疏 reward 反推哪些 token 起作用。代价昂贵。

OPD 的卖点：**让 on-policy 的相关性配上 token 级稠密监督** —— 把稀疏 scalar reward 换成 teacher 的每 token log-prob。每 token 都被打分；on-policy 状态分布留着；value head 和 reward model 完全省掉。

### 谱系 —— 不新但被新武器化

TML 博客自己点了两条祖宗：

- **GKD —— *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes*** (Agarwal, Vieillard, Zhou, Stańczyk, Ramos, Geist, Bachem；DeepMind；ICLR 2024)。[arXiv:2306.13649](https://arxiv.org/abs/2306.13649)。引入 $\lambda$-混合 loss、student rollout 采样、generalized JSD 在 forward / reverse / 混合 KL 之间插值。
- **MiniLLM —— *Knowledge Distillation of Large Language Models*** (Gu, Dong, Wei, Huang；Microsoft / Tsinghua；NeurIPS 2024 —— v3 改名为 *MiniLLM: On-Policy Distillation of Large Language Models*)。[arXiv:2306.08543](https://arxiv.org/abs/2306.08543)。把 OPD 写成 **policy gradient 配 teacher log-ratio 当 reward** 的最干净的数学形式。

这两篇都比 TML 博客早 **~2.5 年**。TML 的贡献不在算法 —— 是 **在 Qwen3 规模上的有立场的重新包装**，明确把它定位成"如果有强 teacher，多数 RL post-training 都没必要"。光是这条叙事 —— 从"KD 是压缩技术"到"KD 是 RL 替代品" —— 就是 2025 年底它能病毒式传播的原因。

> [!note] TML 那篇 *真正* 新的东西在哪
> Loss 就是 GKD-with-$\lambda{=}1$-and-reverse-KL —— 2023 年那个框架的一个配置。新的在三处：(a) **Qwen3-32B 教 8B-Base 的生产规模演示**；(b) **自蒸馏作为 continual learning / forgetting 缓解工具** 的操作模式（mid-train 灌入领域文档，再用 mid-train 之前的 chat 模型快照当 teacher 跑 OPD）；(c) **"比 RL 便宜 50–100×"标题党** 把 OPD 从成本对话里硬塞进 RL 替代品的位置。

### 标志性数字

TML 博客（复现 Qwen3 小模型 recipe，见 [Qwen3 tech report](https://arxiv.org/abs/2505.09388)）：

| 方法 | AIME'24 | 计算量（GPU-h，约） | 来源 |
|------|---------|-------------------|------|
| 仅 SFT (400K prompt) | 60 % | 未报告 | TML |
| SFT 外推到 2M | ~70 % | 未报告 | TML |
| Qwen3 RL recipe | 67.6 % | ~17,920 | TML |
| **On-Policy Distillation** | **74.4 %** | **~1,800** | TML |

Student: Qwen3-8B-Base，Teacher: Qwen3-32B。**~10× 计算降低，AIME 反而更高**。Qwen3 tech report 自己也在同样 0.6B–14B + 30B-A3B-MoE 的蒸馏管线上印证了这个量级（"仅为四阶段训练方法的 1/10 GPU hours"）。

博客进一步声称自蒸馏时 **50–100×**。但 100× 这个数字 *没有* 独立复现 —— 到 2026-05 公开文献里没人重复出来。基于 Qwen3 报告的可辩护区间是 5–20×，取决于 teacher inference 成本怎么摊。

---

## 算法

### Loss 函数

单 teacher 反向 KL on-policy 蒸馏（TML / GKD-$\lambda=1$）：

$$
\mathcal{L}_{\text{OPD}}(\theta) \;=\; \mathbb{E}_{x,\,y\sim\pi_\theta(\cdot\mid x)}\!\left[\sum_{t=1}^{|y|} D_{\text{KL}}\!\big(\pi_\theta(\cdot\mid y_{<t}, x) \,\big\|\, \pi_T(\cdot\mid y_{<t}, x)\big)\right]
$$

两个固定设计选择：

- **On-policy**：$y \sim \pi_\theta$，*每个训练步* 从 *当前* student 现采。Off-policy 版本（一次采完反复用）退化成"带软标签的 SFT"。
- **反向 KL**（mode-seeking）：student 把概率质量集中在 teacher 高概率 token 上，而不是去覆盖 teacher 的尾部。正向 KL（mean-seeking）也合法，是经典 SFT-style KD 用的；方向变了 student 行为质上不同。反向 KL 是 GKD 的 $\beta{=}1$ 分支。

GKD 的 generalized JSD 把所有这些当成特例：

$$
\mathcal{L}_{\text{GKD}}(\theta) = (1-\lambda)\,\mathbb{E}_{(x,y)\sim \mathcal{D}}\!\left[D(\pi_T\|\pi_\theta)(y\mid x)\right] + \lambda\,\mathbb{E}_{x,\,y\sim \pi_\theta}\!\left[D(\pi_T\|\pi_\theta)(y\mid x)\right]
$$

$\lambda$ 控制 on-policy 比例，$D$ 是按 $\beta$ 参数化的 generalized JSD。TML 的 OPD = GKD 在 $(\lambda, \beta, \text{方向}) = (1.0,\, \text{N/A},\, \text{反向 KL})$ 下的取值。HuggingFace TRL 把两个都暴露成 knob（见 [源码](#源码)）。

### 与策略梯度的对偶

这是让 [[grpo|GRPO]] / [[ppo-for-llm|PPO]] 出身的人秒懂的结果。MiniLLM §3：

$$
\nabla_\theta \,\mathbb{E}_{y\sim\pi_\theta}\!\big[D_{\text{KL}}(\pi_\theta\|\pi_T)\big] \;=\; -\,\mathbb{E}_{y\sim\pi_\theta}\!\left[\sum_{t} \nabla_\theta \log\pi_\theta(y_t\mid y_{<t}) \cdot \underbrace{\log\frac{\pi_T(y_t\mid y_{<t})}{\pi_\theta(y_t\mid y_{<t})}}_{\text{每 token 稠密"reward"}}\right]
$$

这就是 **普通 REINFORCE**，每 token reward 换成 teacher log-ratio。三个直接推论：

- **OPD 继承 PPO 的稳定性**。KL-to-teacher 同时充当 trust-region 正则项（student 离 teacher 支撑越远惩罚越大），跟 PPO 的 clip / KL 项一个角色。
- **OPD 是去 baseline 去 value 的 GRPO**。GRPO 已经用 group-relative advantage 砍掉 critic；OPD 再进一步，把稀疏 outcome reward 换成 token 级稠密 reward。无 critic、无 group 归一化、无 advantage 估计。
- **discount $\gamma$ 不重要**。Reward 已经是每 token 稠密的，没有要靠 $\gamma$ 解决的 credit assignment。TML 实验上发现 $\gamma=0$ 最好 —— token $t$ 的梯度只依赖当下教师不同意的程度。

> [!quote] 一句话心智模型
> *"On-policy distillation 是把 GRPO 的稀疏 outcome reward $R(y)\in\{0,1\}$ 换成稠密 token 级信号 $\log(\pi_T/\pi_\theta)$，再把 value head 拿掉（因为 reward 已经是 token 级的）后的 GRPO 目标。"*

### 为什么 "on-policy" 重要

期望 $\mathbb{E}_{y\sim\pi_\theta}$ 是在 *student 自己的* 轨迹分布上算的。把它换成 $\mathbb{E}_{y\sim\mathcal{D}}$（固定语料）就是 GKD 的 $\lambda{=}0$，等于带软标签的 SFT：

| 设置 | Loss 期望 | 干什么 |
|------|----------|--------|
| $\lambda = 0$ | $y \sim \mathcal{D}$ (固定数据集) | SFT 带软标签；离线；漂移累积 |
| $\lambda = 1$ | $y \sim \pi_\theta$ (现采的 student rollout) | On-policy 蒸馏；走的是 student 实际部署的分布 |
| $0 < \lambda < 1$ | 混合 | 在稳定性和 on-policy 相关性之间平衡 |

on-policy 重要的根本原因是 **分布偏移**。Off-policy 设置在 teacher 生成的、student 永远不会经过的状态上训；推理时小误差通过"模型从没见过自己上一个 token 之后会发生什么"放大。On-policy distillation **直接在 student 部署时会经过的状态上训**，每个梯度信号都有现实意义。这也是 [DAGGER (Ross, Gordon, Bagnell, 2010)](https://arxiv.org/abs/1011.0686) 的核心思想 —— 见 [争论那一节](#opd-vs-rl-争论)。

---

## 2025–2026 在用的变体

| 变体 | 起源 | 加了什么 |
|------|------|---------|
| **GKD** ([Agarwal 2023](https://arxiv.org/abs/2306.13649)) | DeepMind / ICLR 2024 | 总伞形式。$(\lambda, \beta)$ 旋钮；$\lambda{=}1$ 退化成纯 OPD。 |
| **MiniLLM** ([Gu 2023](https://arxiv.org/abs/2306.08543)) | Microsoft / Tsinghua / NeurIPS 2024 | OPD-as-PG 最干净的推导；teacher 混采、单步方差减少技巧。 |
| **OPD (TML)** ([博客](https://thinkingmachines.ai/blog/on-policy-distillation/)) | Thinking Machines Lab, 2025-10 | 从业者框架。反向 KL，$\gamma{=}0$，"RL 替代品"叙事。 |
| **OPSD**（On-Policy Self-Distillation）([Privileged-Info OPD](https://arxiv.org/abs/2602.04942)) | 多组工作，2025–26 | Teacher 是 *student 自己的早期 checkpoint* 或它的 *特权信息* 版本。适合 continual learning 和 personalization。 |
| **KDRL** ([arXiv:2506.02208](https://arxiv.org/abs/2506.02208)) | Xu, Zhu 等, 2025-06 | 把 [[grpo|GRPO]] 的 KL-to-old-policy 换成 KL-to-teacher；同一个梯度步里联合优化规则 reward + OPD 信号。 |
| **dGRPO** ([survey arXiv:2604.00626](https://arxiv.org/abs/2604.00626)) | 2025–26 | GRPO advantage + 每 token OPD loss 作为稠密辅助 head；与 KDRL 同 recipe 家族。 |
| **MOPD**（Multi-Domain OPD）([Nemotron-Cascade 2](https://research.nvidia.com/labs/nemotron/nemotron-cascade-2/)) | NVIDIA, 2026-03 | 按 domain 取最好 checkpoint 当 teacher；OPD 把 student 拉回各 domain 最优，在 cascade RL 中恢复回归。生产已用。 |
| **MAD-OPD** ([arXiv:2605.01347](https://arxiv.org/abs/2605.01347)) | 2026 | 多智能体辩论当 teacher 信号。试图突破单 teacher 能力天花板。研究阶段。 |
| **Asymmetric / Reward-Extrapolated OPD** ([arXiv:2602.12125](https://arxiv.org/abs/2602.12125), [arXiv:2605.06387](https://arxiv.org/abs/2605.06387)) | 2026 | 加一个 RL reward head，让 student 能 *超越 teacher*。对 teacher 天花板批评的直接回应。 |
| **Black-Box OPD (GAD)** ([arXiv:2511.10643](https://arxiv.org/abs/2511.10643)) | Ye, Dong 等, 2025-11 | 当只能拿到 teacher 的输出文本（看不到 logits）时的 OPD —— 用对抗判别器替代 teacher log-prob。OpenAI / Anthropic teacher 用得上。 |
| **多教师全词表 OPD** ([DeepSeek-V4 §5.1.2](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)) | DeepSeek, 2026-04 | $\sum_i w_i D_{\text{KL}}(\pi_\theta \| \pi_{E_i})$ 跨 $> 10$ 个专家 teacher，用 **全词表 logit KL** 替代 token 级近似。旗舰规模的演示。详见 [[deepseek-v4-opd]]。 |

---

## 已验证的生产部署

这里只列有 primary source 支持的部署，跳过普遍意义上的"用了 distillation"（人人都用），只标注真正用 student rollout + 每 step teacher 打分的 *on-policy* 版本。

### NVIDIA Nemotron-Cascade 2 — MOPD（2026-03）

到 2026 年中，开放出来的、OPD-作为-RL-组件的 recipe 里最经得起检验的一个。来源：[Nemotron-Cascade 2 页面](https://research.nvidia.com/labs/nemotron/nemotron-cascade-2/) 和 [PDF](https://research.nvidia.com/labs/nemotron/files/Nemotron-Cascade-2.pdf)。30B-active MoE，IMO / IOI / ICPC 金牌。

**Cascade recipe（每个 domain $D_k$）：**

```
对每个 domain D_k in {math, code, agent, IF, alignment, ...}：
  (a) GRPO RL 在 D_k 上                              ← exploration / reward 信号
  (b) 保存该 domain 最佳 checkpoint  →  teacher_D_k
  (c) MOPD：student 跨所有已访问 domain 出 rollout；
            按 prompt 路由到对应 domain 的 best-checkpoint
            teacher；在 student 走出的状态上做 token 级
            反向 KL（~30 个优化步）                    ← 回归恢复
  (d) 进下一个 domain。
```

**为什么这么做：** 顺序 RL 跨 domain 漂移 —— student 在 $D_k$ 上变好但在 $D_{k-1}, D_{k-2}, \dots$ 上退化。MOPD 用 *token 级、不是轨迹级* 的梯度信号把它拉回 per-domain 最优 —— 30 步 OPD 能恢复大部分性能，而在旧 domain 上重做 RL 要花几个数量级以上的开销。

这个模式 —— **RL 干 exploration，OPD 干稳定 + 回归恢复** —— 是 NVIDIA 栈最有建筑学价值的 OPD 用法。交叉链接：[[rl-training-frameworks]]。

### Alibaba Qwen3（2025-05）

*首例* 大规模公开部署，比 TML 博客早 5 个月。[Qwen3 tech report](https://arxiv.org/abs/2505.09388)：Qwen3 的 0.6B / 1.7B / 4B / 8B / 14B + 30B-A3B-MoE 小模型线把完整 RL 管线的第 3–4 阶段换成两阶段蒸馏：

```
Stage 1: Pretrain (base model)
Stage 2: 离线蒸馏，从更大的 Qwen3 teacher
         (teacher 同时在 /think 和 /no_think 模式下产出)
Stage 3: On-policy 蒸馏，从同一个 teacher
         (student 现采，teacher logits，反向 KL)
         -- 跳过完整 RL 管线的 stage 3-4 --
```

**报告成本：** Qwen3 自己声称是四阶段 GPU-hour 的 1/10。TML 博客就是把这套 recipe 在 Qwen3-8B-Base 上复现得到 74.4 % AIME 那个数字。

### DeepSeek-V4 — 完整管线替换（2026-04）

迄今为止最激进的 OPD 部署 —— V4 的 post-training **完全替换** V3.2 的 mixed-RL 阶段。完整细节在 [[deepseek-v4-opd]]；简版：V4 把训练拆成 per-domain specialist（每个用 GRPO 训），再用一个加权反向 KL 目标把 10+ 个 specialist 合并成一个统一 policy。算法层面相对之前 OPD 的新点是 **全词表 logit KL** 而不是 token 级近似。

### 哪些地方 *没* 用 on-policy distillation

为了校准认知：

- **DeepSeek-R1 → 小学生模型** 用的是 **SFT-only 离线蒸馏**，~800K 条已验证 trace（"仅用标准 SFT 不做 RL"，见 [R1 paper](https://arxiv.org/abs/2501.12948)）。是 off-policy 不是 OPD。这是 V3.2 时代；V4 才是 OPD 转折点。
- **Meta Llama 4** 用了 codistillation 配 "动态软 / 硬目标加权" loss，但公开材料没描述 student rollout 形式的 on-policy 蒸馏。
- **Anthropic、OpenAI、Mistral、Cohere** —— 公开材料里没有 on-policy distillation 的证据。蒸馏作为大类肯定内部都在用；"on-policy" 这个具体形式没有公开证据。

---

## 源码

### HuggingFace TRL —— `GKDTrainer`

最权威的开源实现。文件：[`trl/trainer/gkd_trainer.py`](https://github.com/huggingface/trl/blob/main/trl/trainer/gkd_trainer.py)。文档：[huggingface.co/docs/trl/en/gkd_trainer](https://huggingface.co/docs/trl/en/gkd_trainer)。

**关键部分（已读源验证）：**

```python
# trl/trainer/gkd_trainer.py — generalized_jsd_loss (lines 226-295)
def generalized_jsd_loss(
    student_logits,
    teacher_logits,
    labels=None,
    beta=0.5,            # 0 = 正向 KL, 1 = 反向 KL, 0.5 = 对称 JSD
    temperature=1.0,
    reduction="batchmean",
):
    # 温度缩放后的 log-softmax，两边都做
    # 然后分支：纯 KL（beta == 0 或 1），或 mixture via
    # torch.logsumexp([student + log(1-beta), teacher + log(beta)])
    # 返回 beta * KL(M||teacher) + (1-beta) * KL(M||student)
    ...

# training_step (lines 421-449)
def training_step(self, model, inputs, num_items_in_batch=None):
    # 以概率 self.lmbda 替换输入为 on-policy rollout：
    if random.random() <= self.lmbda:
        inputs = self.generate_on_policy_outputs(...)
    # 然后过 student + teacher，算 GJSD loss
    ...
```

配置（`GKDConfig`）：

| 字段 | 作用 |
|------|------|
| `lmbda` | 每 batch 现采 student rollout 的概率。`1.0` = 纯 OPD；`0.0` = 普通离线 KD；中间 = 混合。 |
| `beta` | KL 方向。`0` = 正向 KL，`1` = 反向 KL，`0.5` = JSD。TML OPD 取 `beta=1.0`。 |
| `temperature` | log-softmax 的温度。 |
| `seq_kd` | 从 *teacher* 出 rollout，给你 teacher 样本上的 sequence 级 KD。 |
| `use_liger_kernel` | 用 Liger 融合的 linear+JSD kernel 省内存。 |

> [!warning] TRL 正在弃用 GKDTrainer
> 当前 TRL 会提示：*"This trainer will soon be moved to `trl.experimental` and is a candidate for removal."* 一个更新的 `DistillationTrainer` 在开发中：带 generation buffer（rollout batch 和 train microbatch 解耦，号称 ~40× 加速）、支持外部 teacher server、用二进制压缩 logprob 传输。跟进 [TRL releases](https://github.com/huggingface/trl/releases) 和 [issue #4390](https://github.com/huggingface/trl/issues/4390)。

### veRL —— OPD recipe

文档：[verl.readthedocs.io/en/latest/algo/opd.html](https://verl.readthedocs.io/en/latest/algo/opd.html)。配置命名空间 `distillation.*`：

- `enabled` —— 开关。
- `teacher_models.<name>.model_path` —— 多教师路由；每个 teacher 按 `data_source` keyed（用于 MOPD 风格 per-domain 路由）。
- `distillation_loss.loss_mode` —— `forward_kl_topk`（top-k 限制 KL —— 内存优化）/ `k1`（logp 比）/ `k3`。
- `use_policy_gradient` —— 在 GKD 风格直接 KL 与 PG 风格（REINFORCE-with-KL-as-reward）之间切换。
- `use_task_rewards` —— 与 PPO/GRPO outcome reward 联合（这就是 KDRL recipe 的形状）。

vLLM-hosted teacher 走 ZeroMQ server 是标准基础设施模式：见 [Zoey Li 的实现 walkthrough](https://zoeyli.com/reinforcement%20learning/implementing-on-policy-distillation/)。K1 ($\log\pi_S - \log\pi_T$) 和 K2 ($0.5(\log\pi_S - \log\pi_T)^2$) 估计器只要 *scalar* logprob，绕过 vLLM vocab 分布 API 的限制。

### NVIDIA NeMo-RL

[NVIDIA-NeMo/RL Discussion #1445](https://github.com/NVIDIA-NeMo/RL/discussions/1445)（作者 zpqiu、sharonyu-115、shuo-nvidia、sharathts、snowmanwwg）跟进 NeMo-RL 的 OPD 支持：

- 正向 / 反向 / 混合 KL。
- **Top-k 限制 KL** —— 只有 top-k 个 teacher token 进 loss。NVIDIA 基础设施为了让多教师 OPD 跑得起来必须的带宽优化。
- Student rollout 用 vLLM，teacher 单独 parallelize（更大 teacher 用更大 TP）。
- 数据集：DeepScaler（on-policy）、AceReason-1.1-SFT（off-policy）。
- 报告 Qwen3-4B-Base 上 AIME 2025 Avg@16：**47.71 %（SFT+OPD）** vs **30.42 %（SFT+离线蒸馏）**。

相关指南：[NeMo-Aligner KD docs](https://github.com/NVIDIA/NeMo-Aligner/blob/main/docs/user-guide/knowledge-distillation.rst)、[NeMo-AutoModel KD guide](https://docs.nvidia.com/nemo/automodel/latest/guides/llm/knowledge-distillation.html)。

### Thinking Machines Lab —— `tinker-cookbook`

[github.com/thinking-machines-lab/tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) 在 `tinker_cookbook/recipes/distillation/` 下有蒸馏 recipe。单 / 多教师、on-policy / off-policy 都有。也带多轮 tool-use 蒸馏样例。是最接近 TML 博客的 recipe 参考实现。

### HuggingFace H4 —— GOLD（跨 tokenizer OPD）

[GOLD (General On-policy Logit Distillation)](https://huggingface.co/spaces/HuggingFaceH4/on-policy-distillation) 把 OPD 扩到跨 tokenizer —— SmolLM ↔ Llama ↔ Qwen ↔ Gemma 互蒸，通过 token merge 对齐 + 乘积规则合并 logits。将作为 `GOLDTrainer` 进 TRL。当你的 teacher 跟 student 不同家族（如 GPT-4 级别 teacher 教 Llama student）时有用。[Lewis Tunstall 公告](https://x.com/_lewtun/status/1983620843952328726)。

### 其他

- **`thunlp/OPD`** ([GitHub](https://github.com/thunlp/OPD)) —— "Rethinking OPD" 论文（清华，2026-04；[arXiv:2604.13016](https://arxiv.org/abs/2604.13016)）的官方代码，包括 cold-start 修复和 teacher 对齐的 prompt 选择。
- **`HJSang/OPSD_OnPolicyDistillation`** ([GitHub](https://github.com/HJSang/OPSD_OnPolicyDistillation)) —— veRL 上的 OPSD 社区 fork。
- **Unsloth** —— 没有原生 OPD trainer；社区模式是 TRL `GKDTrainer` 配 Unsloth model loader。
- **OPenRLHF** —— 到 2026-05 没有原生 OPD trainer；只有 MiniLLM 风格的离线 KD 引用。

---

## OPD vs RL 争论

2025 年底 / 2026 年初让 OPD 站到聚光灯下的现役争论。

### 替代 RL 那派

Kevin Lu / TML：*"on-policy distillation 能用约 7–10× 更少的梯度步学到 RL 训练的 policy，对应 50–100× 计算效率提升。"* 实质论点：

- 多数 RL post-training 在为 **稀疏 outcome reward 下的 credit assignment** 付钱。模型从 16K-token rollout 收到一个 0/1 信号，要自己反推哪些 token 起作用。
- 如果有一个 teacher 已经解决了这个问题（如同家族更大的 RL 训过的模型），**你不需要重做 credit assignment** —— 你可以通过 token 级 teacher log-prob 直接抄过来。
- 所以：有强 teacher 时 RL 是过度复杂的基础设施。

### "这就是 15 年前的 DAGGER"那派

Guohao Li ([推文](https://x.com/guohao_li/status/1987821200060625175)，2025-11)：*"Thinking Machines Lab 出 On Policy Distillation 博客时，我的第一反应是它就是 15 年前的 DAGGER … 果然，他们提到了 DAGGER。"*

实质：**on-policy imitation learning** 在 [DAGGER (Ross, Gordon, Bagnell, 2010)](https://arxiv.org/abs/1011.0686) 已经被解决。OPD 是 DAGGER 用到 LLM token 序列上的版本。新的是在 LLM 规模上做这件事的 *工程*，不是算法。引申意：OPD 继承 DAGGER 的 **imitation learning 天花板** —— 你超不过 teacher。

### Teacher 天花板与能力扩展那派

多篇 2026 论文（[arXiv:2604.00626 survey](https://arxiv.org/abs/2604.00626)、[arXiv:2602.12125 "Learning beyond Teacher"](https://arxiv.org/abs/2602.12125)、[arXiv:2605.01347 MAD-OPD](https://arxiv.org/abs/2605.01347)）打同一个点：*"existing methods are capped by a single-teacher capability ceiling: when the teacher errs, the student inherits the error."* RL with verifiable rewards *原则上* 能发现 teacher 没掌握的解决方案；OPD 构造性上做不到。当目标是 *超越* teacher 时（前沿推理、新颖科学发现），OPD 充其量是 warm start。

### 不稳定 / 负迁移那派

[Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes](https://arxiv.org/abs/2603.25562)（2026-03）：vanilla OPD 可能 **熵塌缩**、**从强 teacher 负迁移**（[Rethinking OPD, 2026-04](https://arxiv.org/abs/2604.13016)）、**biased token-level estimator**（token 级反向 KL 不是 sequence 级反向 KL 的无偏估计）。三个具体 failure：

1. **冷启动塌缩** —— 反向 KL 要求 student 支撑覆盖 teacher 支撑；没有先 SFT 就没有这个，梯度爆炸。
2. **从强 teacher 的负迁移** —— "thinking pattern" 与 student 偏离的 teacher（不同 RL 历史、不同家族）伤害大于帮助。清华那篇把成功 OPD 描述为在 97–99 % 共享高概率 token 集合上的对齐。
3. **Tokenizer / special-token mismatch** —— 跨家族 OPD 在 tokenizer 不对齐时 silent fail；GOLD 通过跨 tokenizer logit 对齐解决。

### 综合："两个都用"

2026 年主流观点，跟 NVIDIA 实际部署一致：**RL 干 exploration，OPD 干稳定 + 回归恢复，两个信号都有时联合优化**。[KDRL (Xu, Zhu 等, 2025-06)](https://arxiv.org/abs/2506.02208) 是最干净的写法 —— 把 [[grpo|GRPO]] 的 KL-to-old-policy 换成 KL-to-teacher，把 OPD 项加到 GRPO advantage 上。报告 +4.7 %（vs SFT）、+2.6 %（vs GRPO）、+1.1 %（vs KD-RKL）的 reasoning benchmark 提升。这就是 dGRPO、MOPD、DeepSeek-V4 完整管线替代所属的 recipe 家族。

---

## 何时选 OPD vs RL —— 决策树

> [!tip] 速查

```
                  我有一个已经把目标任务做到我想要的
                  质量的 teacher 吗？
                          /                   \
                        有                     无
                        /                       \
            Teacher 与 student 在同家族 /        Reward 可验证吗
            同 tokenizer 吗？                     （数学、代码、形式化）？
                /         \                       /         \
              是          否                    是          否
              /             \                   /             \
        OPD 胜出       试 GOLD              GRPO /          DPO /
        (从 TML       (跨 tokenizer)       DAPO /          preference RL
        recipe 起步)   或回落到             outcome RL      或干脆别 RL
                      off-policy KD                       （只做 SFT）
```

**经验启发：**

- **有 teacher 时 OPD warm-start 胜过冷 RL**；先 OPD 再 RL 上面做能力扩展。
- **自蒸馏 (OPSD) 是回归恢复原语** —— 任何有可能造成灾难性遗忘的 mid-training 之后用一下。
- **多教师 OPD（MOPD / DeepSeek-V4 风格）**是合并多个 specialist 时的对策。
- **OPD 单独不足以做前沿推理** —— 想超过 teacher 必须有 RL 才能提供的探索。两者叠加（KDRL / dGRPO 风格）当你都要时。

---

## 限制

- **被 teacher 能力上限封顶**。没有辅助信号（RL reward、debate、reward extrapolation）就超不过 teacher。
- **冷启动脆弱**。反向 KL 需要 student 先有支撑覆盖 teacher 高概率 token。要先 SFT 或正向 KL 暖。
- **跨家族负迁移**。Tokenizer 不同、预训练语料不同的强 teacher 可能 *降低* student。用 GOLD 或同家族 teacher。
- **成本声明要小心**。"比 RL 便宜 100×"取决于 teacher inference 怎么摊；忽略生成成本会不对。Qwen3 自己说 ~10×；100× 没有独立复现。
- **Biased token-level estimator**。Vanilla token 级反向 KL 不是 sequence 级反向 KL 的无偏估计 —— 长 rollout 上方差累积（16K+ token 推理 trace 严重，agentic 场景更严重）。DeepSeek-V4 的全词表 KL 是一个解法；sequence 级方差界是另一条活跃研究方向。
- **Teacher inference 成本**。开放 OPD recipe 假设 teacher 服务便宜（logits 可见）。Teacher 是闭源 API（GPT-4 / Claude）时用 Black-Box OPD (GAD)。
- **熵塌缩**。没有合适的 KL 正则下 student 可能 mode-collapse 到 teacher 某个模式。KDRL 风格联合目标配 RL exploration 缓解。

---

## 这意味着什么

三条值得跟踪的预测：

1. **有 teacher 时 OPD 会替代 reasoning 模型的 mixed-RL post-training**。[[deepseek-v4-opd|DeepSeek-V4]] 是第一个全押的旗舰模型。预期 Qwen、Mistral、开源社区跟进。RL 在每代 *第一个* 模型上（没 teacher）和前沿能力扩展上继续主导。
2. **多教师 OPD 会变成新默认**。单教师 OPD 继承单个天花板；多教师（DeepSeek-V4 风格）让你合并 specialist。预期 2026 H2 起标配。
3. **有意思的研究不再是 loss 函数**。GKD / MiniLLM 在 2023 把数学钉死了。2026 的研究在：(a) 方差减少（全词表 KL、sequence 级修正），(b) 跨 tokenizer 对齐 (GOLD)，(c) 经济 teacher 服务（logit 缓存、FP4 QAT），(d) OPD+RL 混合目标 (KDRL、dGRPO)。

这 *不是*：万能 RL 杀手。当目标是超越最强可得 teacher 时 OPD 只能 warm-start。RL 仍有不可替代的角色。

---

## 相关阅读

- [[deepseek-v4-opd]] —— DeepSeek-V4 的多教师全词表 OPD recipe。
- [[grpo]] —— OPD 最常被对比 / 组合的 RL 算法。
- [[ppo-for-llm]] —— OPD KL-to-teacher 项分享的 trust-region 直觉。
- [[rlhf-overview]] —— 被 OPD 替代的标准 RL post-training 管线。
- [[dpo]] —— 另一种 preference 风格 RL 替代品；与 OPD 正交。
- [[rl-training-frameworks]] —— 容纳 OPD 的 trainer 端库（OpenRLHF、TRL、veRL、NeMo-RL）。
- [[das-spec-rl]] —— RL rollout 的投机解码加速；推理层互补。
- [[prorl-agent]] —— 容纳 RL + OPD 工作负载的 rollout-即-服务基础设施。

## 参考文献

- **Thinking Machines Lab 博客** (Kevin Lu, 2025-10-27)：https://thinkingmachines.ai/blog/on-policy-distillation/
- **GKD 论文**：Agarwal et al., *On-Policy Distillation of Language Models* (ICLR 2024)。[arXiv:2306.13649](https://arxiv.org/abs/2306.13649)
- **MiniLLM 论文**：Gu et al., *MiniLLM: On-Policy Distillation of Large Language Models* (NeurIPS 2024)。[arXiv:2306.08543](https://arxiv.org/abs/2306.08543)
- **Qwen3 tech report**：[arXiv:2505.09388](https://arxiv.org/abs/2505.09388)
- **DeepSeek-V4 tech report**：[huggingface.co/deepseek-ai/DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) （无 arXiv）
- **KDRL**：Xu, Zhu 等 (2025-06)。[arXiv:2506.02208](https://arxiv.org/abs/2506.02208)
- **OPD Survey**：[arXiv:2604.00626](https://arxiv.org/abs/2604.00626)
- **Rethinking OPD**：清华 (2026-04)。[arXiv:2604.13016](https://arxiv.org/abs/2604.13016) —— 代码 [thunlp/OPD](https://github.com/thunlp/OPD)
- **Revisiting OPD Failure Modes**：[arXiv:2603.25562](https://arxiv.org/abs/2603.25562)
- **Black-Box OPD (GAD)**：[arXiv:2511.10643](https://arxiv.org/abs/2511.10643)
- **NVIDIA Nemotron-Cascade 2**：[research.nvidia.com/labs/nemotron/nemotron-cascade-2](https://research.nvidia.com/labs/nemotron/nemotron-cascade-2/)
- **HuggingFace TRL GKDTrainer**：[trl/trainer/gkd_trainer.py](https://github.com/huggingface/trl/blob/main/trl/trainer/gkd_trainer.py)
- **veRL OPD docs**：[verl.readthedocs.io/en/latest/algo/opd.html](https://verl.readthedocs.io/en/latest/algo/opd.html)
- **NeMo-RL Discussion #1445**：[github.com/NVIDIA-NeMo/RL/discussions/1445](https://github.com/NVIDIA-NeMo/RL/discussions/1445)
- **Tinker cookbook**：[github.com/thinking-machines-lab/tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)
- **DAGGER**（祖宗）：Ross, Gordon, Bagnell (2010)。[arXiv:1011.0686](https://arxiv.org/abs/1011.0686)
