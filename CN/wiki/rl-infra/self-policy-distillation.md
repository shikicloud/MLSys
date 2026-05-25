---
title: "SPD：通过能力选择性子空间投影的自策略蒸馏"
category: rl-infra
tags: [self-policy-distillation, spd, self-distillation, opd, subspace-projection, kv-steering, paper-review]
created: 2026-05-22
updated: 2026-05-22
status: mature
paper: arXiv:2605.22675
code: not-yet-released
---

# SPD：通过能力选择性子空间投影的自策略蒸馏

> [!info] 论文信息
> - **论文**：[arXiv:2605.22675](https://arxiv.org/abs/2605.22675) — *Self-Policy Distillation via Capability-Selective Subspace Projection*，2026-05-21 预印本
> - **作者**：Guangya Hao¹, Yitong Shang¹², Yunbo Long¹, Zhuokai Zhao³†, Hanxue Liang¹†
> - **单位**：¹剑桥大学，²香港科技大学，³芝加哥大学（†共同最后作者）
> - **代码**：预印本阶段尚未发布
> - **通讯邮箱**：hl589@cantab.ac.uk, zhuokai@uchicago.edu, ytshang@ust.hk

---

## 摘要（2 分钟读完这一节就够）

**SPD 是什么**。一种**无老师自蒸馏**方法：让 LLM 在自己的生成上训练 —— 但**先**通过一个学习到的 KV 激活投影**引导**这些生成，让它们集中在你想提升的*能力*上，而不是风格伪迹和模型特有错误上。不需要外部 verifier、不需要 reward model、不需要 RL。

**核心思想**。现有自蒸馏有两种失败模式：(a) 需要外部信号（correctness filter、exec feedback、reward search），这些既贵又对 frontier 模型不可用；或 (b) 在原始输出上训练，继承模型坏习惯，因为*自生成输出把能力跟格式/风格/错误纠缠在一起*。SPD 的答案是从一个小校准集上**correctness-defining tokens** 的梯度做 SVD，找一个**低秩能力子空间**（在 KV 激活空间里），自生成时把 KV 投影到这个子空间上。挂了 hook 的模型生成更干净的输出；之后做标准 next-token SFT 就够了。两个阶段：**Phase 1** 从校准梯度提取 $P_K^{(\ell)}, P_V^{(\ell)}$；**Phase 2** 用 $\tilde K = K P_K, \tilde V = V P_V$ hook 生成，然后在 $(q, \hat y)$ 上微调没挂 hook 的原模型。

**标志数字**。Qwen2.5-0.5B-Instruct 跨 3 个领域（code / math / QA）6 个 benchmark 上：平均 **+8.9 % vs base，+9.3 % vs Plain Self-Retraining，+6.4 % vs Simple Self-Distillation**。**vs SOTA 无外部信号自蒸馏 +13 %**，**vs 预训练 baseline +16 %**。最关键的是 **out-of-domain 迁移**：用 QA（MMLU）校准的子空间能把 GSM8K 从 11 提到 26 %、SVAMP 从 16 提到 21 % —— *能力 filter 跨越校准 domain 泛化了*。

**为什么重要**。

- **不需要 verifier 的自蒸馏**。之前唯一无老师且效果好的自蒸馏（SSD，Zhang et al.）只针对代码。SPD 用一套机制覆盖 math 和 QA。
- **子空间引导作为基本操作**。把"correctness token 上梯度长什么样"看成一个*可以投影进去的子空间*，把表征工程（RepE, ITI）从推理时引导扩展到训练数据生成。
- **对 frontier 模型友好**。没有更强老师可用时，OPD（[[on-policy-distillation]]）撞上模仿学习天花板。SPD 提供了一条不需要老师的自我提升路径。
- **关键消融**。子空间提取用 full-sequence loss 只能让 MBPP 拿到 11.9 %（比 base 还差！），用 correctness-aligned loss 拿到 25.5 %。"哪些 token 算数"这个选择是 load-bearing 的，不是修饰。

---

# 深度部分（往下展开细节）

## 背景：为什么无老师自蒸馏老是停滞

SPD 定位的 post-training 全景：

| 路线 | 分布 | 监督来源 | 失败模式 |
| ---- | ---- | ------- | ------- |
| **Off-policy KD**（Hinton 2015、sequence-KD） | 老师 rollout $y \sim f_T$ | 老师 logits | 训推分布不匹配 → 错误累积 |
| **On-policy 蒸馏**（[[on-policy-distillation\|OPD]] / GKD / MiniLLM） | 学生 rollout $y \sim \pi_\theta$ | 每 token 老师 logits | 需要老师；天花板就是老师；老师 serving 贵 |
| **带外部信号的自蒸馏** | 学生 rollout | correctness filter / verifier / reward search / exec feedback | 信号有成本 + 基础设施；frontier 模型上不可用 |
| **Simple Self-Distillation (SSD)**（Zhang et al.） | 学生 rollout（截断式解码） | 直接在原始输出上训 | Domain-specific（仅 code）；不泛化 |
| **SPD（本文）** | 经过 KV 投影 hook 的学生 rollout | 直接在引导后的输出上训 | — |

SPD 提出的统一批评：**自生成输出是混合监督**。0.5B-Instruct 写 MBPP 答案会同时产出 (i) 答案逻辑、(ii) 冗长解释、(iii) 格式伪迹、(iv) 模型特有错误。在全部输出上训会稀释能力信号；事后过滤（SSD 风格的截断）太粗糙。SPD 的操作：**引导生成本身**，让能力轴主导、噪声轴被压制。

为什么"无外部信号"是 load-bearing：对 frontier 模型（GPT-5、Claude Opus 4.x、DeepSeek-V4），correctness filter 和 reward model 要么很贵（要更强的 judge），要么压根不存在 —— 没有比模型更强的能力 oracle。SPD 瞄准的就是这个场景。

## 方法详解

![SPD 两阶段总览（论文 Fig. 2）](CN/wiki/rl-infra/self-policy-distillation-figs/spd-overview.png)

两个阶段。Phase 1 一次性提取投影矩阵；Phase 2 把它们当 inference hook 用，然后微调。

### Phase 1 —— 能力子空间提取

**校准集**。$D_{\text{cal}} = \{(q^{(i)}, y^{(i)})\}_{i=1}^{N_{\text{cal}}}$ —— prompt 配正确答案。小：**20-500 个例子就够**（论文 Fig. 4；该区间内性能稳定）。

**Correctness-aligned loss**。不在整段输出上算标准 next-token loss，而是 mask 掉除一个 **correctness-defining 位置**集合 $S^{(i)}$ 之外的所有 token —— 这些位置的 token 预测"直接关系任务成败"（论文 App. A.1 有 per-domain 规则）：

$$
\mathcal{L}_{\text{align}}(q^{(i)}, y^{(i)}) = -\frac{1}{|S^{(i)}|} \sum_{t \in S^{(i)}} \log p_{\theta_{\text{old}}}(z_t^{(i)} \mid z_{<t}^{(i)})
$$

这就是把梯度集中在能力相关方向、避开风格噪声的地方。**它也是工作量隐藏的地方** —— $S$ 怎么选放在附录，正文不暴露 MBPP / GSM8K / MMLU 上具体怎么挑位置。

**梯度收集**。每个校准例子，**冻结**学生做一次 forward + 一次 backward。在每个目标层 $\ell \in \mathcal{L}$ 收集 $\mathcal{L}_{\text{align}}$ 对 K、V 激活的 token 级梯度：

$$
g_{K,t}^{(\ell,i)} = \frac{\partial \mathcal{L}_{\text{align}}(q^{(i)}, y^{(i)})}{\partial K_t^{(\ell,i)}} \in \mathbb{R}^{d_k}, \qquad
g_{V,t}^{(\ell,i)} = \frac{\partial \mathcal{L}_{\text{align}}(q^{(i)}, y^{(i)})}{\partial V_t^{(\ell,i)}} \in \mathbb{R}^{d_v}
$$

虽然所有 token 位置都有梯度定义，只有在 $S^{(i)}$ 内的位置才收到任务相关信号 —— 其余被 mask 掉。

**Stack + SVD**。把全部校准 token 拼接起来，形成 $G_K^{(\ell)}, G_V^{(\ell)} \in \mathbb{R}^{M \times d_k}$，其中 $M = \sum_i T_i$。每行是 KV 特征空间里的一个 token 级梯度方向。每个矩阵做 SVD：

$$
G_K^{(\ell)} = U_K^{(\ell)} \Sigma_K^{(\ell)} V_K^{(\ell)\top}
$$

保留 top-$r$ 个右奇异向量 $V_{K,r}^{(\ell)} \in \mathbb{R}^{d_k \times r}$。这些是 K 特征空间里 correctness-aligned loss 变化最大的主方向。**秩 $r$ 正交投影矩阵**：

$$
P_K^{(\ell)} = V_{K,r}^{(\ell)} V_{K,r}^{(\ell)\top} \in \mathbb{R}^{d_k \times d_k}
$$

$P_V^{(\ell)}$ 同构造。这些投影矩阵**计算一次**之后冻结。

> [!example] 这里为什么用 SVD 合理
> 梯度 $\partial \mathcal{L}_{\text{align}} / \partial K_t$ 指向 K 激活的某个方向 —— 沿这个方向变 K 最影响任务成败。所有 calibration token 的此类梯度堆起来后做 SVD，top-$r$ 奇异向量就是校准集上的*共识*方向 —— 那些一致影响 correctness 的 K 轴。把 K 投影到这个 $r$ 维子空间，**保留能力对齐的方差，把其余的清零**（风格、格式、模型特有噪声 —— 梯度不关心的方向）。

### Phase 2 —— 能力选择性蒸馏

**自生成时的投影 hook**。插在目标层 $\ell \in \mathcal{L}$。每次 forward 时：

$$
\tilde K^{(\ell)} = K^{(\ell)} P_K^{(\ell)}, \qquad \tilde V^{(\ell)} = V^{(\ell)} P_V^{(\ell)}
$$

挂了 hook 的模型 $f_{\theta_{\text{old}}}^{\text{hook}}$ 做标准 autoregressive 生成：$\hat y \sim f_{\theta_{\text{old}}}^{\text{hook}}(\cdot \mid q)$，对每个训练 prompt 都生成。**模型参数没动** —— hook 纯粹是 inference 时的激活重写。

**默认目标层**。中间 + 最后：$\mathcal{L} = \{L, \lfloor L/2 \rfloor\}$。这是出于之前工作显示中间表征带有用信号；具体选择留给用户。

**在引导后的语料上微调**。摘 hook；对*原* $f_{\theta_{\text{old}}}$ 做 LoRA，在 $(q^{(i)}, \hat y^{(i)})$ 上用标准 next-token loss：

$$
\min_\theta \;-\mathbb{E}_{q \sim D, \hat y \sim f_{\theta_{\text{old}}}^{\text{hook}}(\cdot \mid q)} \left[ \sum_t \log p_\theta(z_t \mid z_{<t}) \right], \quad z = T(q, \hat y)
$$

就这样。没有 KL、没有老师 logits、没有 reward、没有 RL。全部新意都在*数据生成*那一步；微调就是普通的 SFT-on-self-rollouts。

**简洁视角**。$f_{\theta_{\text{old}}} \xrightarrow{T_{\theta_{\text{old}}}^{\text{hook}}} f_{\theta_{\text{old}}}^{\text{hook}} \xrightarrow{\text{generate}} \hat y \xrightarrow{\text{distill}} f_\theta$。SPD 是"把自己内部选择过的版本蒸馏回自己"。

### SPD 在蒸馏图景中的位置

论文自己的框架公式（Eq. 1）：off-policy 和 on-policy 蒸馏都最小化老师 KL，rollout $y \sim f_{\text{roll}}$，其中 $f_{\text{roll}} \in \{f_T, D_{\text{offline}}, f_{\theta_{\text{old}}}\}$ —— *都需要外部老师* $f_T$。自蒸馏（Eq. 2）去掉老师但加了外部评分 $S(q, y)$ —— *仍然外部*。SPD 的贡献是把 $S(q, y)$（输出层 filter）换成 $T_{\theta_{\text{old}}}^{\text{hook}}$（激活层变换）：

| 方法 | Rollout 来源 | 监督 | 外部依赖 |
| ---- | ----------- | --- | -------- |
| Off-policy KD | $y \sim D_{\text{offline}}$ | 老师 logits | 老师 $f_T$ + 数据集 |
| [[on-policy-distillation\|OPD]] / GKD | $y \sim \pi_\theta$ | 老师 logits | 老师 $f_T$ |
| 自蒸馏 + filter | $y \sim \pi_\theta$ | $\log p_\theta(y)$ 加 $S(q,y)$ 权重 | Verifier / RM / exec env |
| SSD（截断） | $y \sim \pi_\theta$（截断解码） | $\log p_\theta(y)$ | 无（但仅 code） |
| **SPD** | $y \sim \pi_\theta^{\text{hook}}$（引导后） | $\log p_\theta(y)$ | 一个小标注校准集（20-500 例） |

SPD 的"外部依赖"*小很多*：20-500 标注例子提取子空间，推理时不需要 judge 或 RM。代价是子空间 per-capability —— 每个目标能力需要一个校准集。

## 标志证据

**配置**。5 个 backbone，跨 2 个模型家族 3 种规模：

- Qwen2.5-0.5B / 7B / 14B-Instruct
- Qwen3-4B-Instruct
- Llama-3.1-8B-Instruct

6 个数据集，3 个能力域：**Code**（MBPP, CodeAlpaca-20k），**Math**（GSM8K, SVAMP），**QA**（MMLU, BBH）。指标：Pass@1（MBPP）、exact-match（GSM8K, SVAMP）、字母准确率（MMLU）、归一化 exact-match（BBH）、NLL（CodeAlpaca，越低越好）。

Baseline：**Base**（预训练学生）、**PSR**（Plain Self-Retraining，在原始自输出上微调）、**SSD**（Simple Self-Distillation，Zhang et al.）。

### 主结果 —— Qwen2.5-0.5B-Instruct（Table 1 / Fig. 1 摘要）

| 数据集 | Base | PSR | SSD | **SPD** | Δ vs Base |
| ----- | ---: | --: | --: | ------: | --------: |
| MBPP (code) | 17.0 % | 29.0 % | 18.3 % | **25.5 %** | +8.5 pp |
| GSM8K (math) | 11.0 % | 17.0 % | 12.0 % | **22.0 %** | +11.0 pp |
| MMLU (QA) | 46.0 % | 43.0 % | 48.0 % | **49.0 %** | +3.0 pp |
| SVAMP (math, 同域迁移) | 16.0 % | 19.0 % | 16.0 % | **32.0 %** | +16.0 pp |
| BBH (QA, 同域迁移) | 32.7 % | 33.7 % | 36.0 % | **38.7 %** | +6.0 pp |
| CodeAlpaca NLL ↓ | 0.683 | 0.683 | 0.682 | **0.679** | better |

> [!success] 5-backbone 平均
> SPD 平均 **+8.9 % over Base，+9.3 % over PSR，+6.4 % over SSD**，跨 Qwen2.5-0.5B / 7B / 14B、Qwen3-4B、Llama-3.1-8B。0.5B 是最大涨幅；大模型涨得少（Qwen3-4B +2.1 %、Llama-3.1-8B +1.9 %）。跟"小模型更受数据 curation 技巧帮助"这个一般规律一致。

### Out-of-domain 迁移（Table 2）

能力选择性的证据。只用 **QA 校准**（MMLU）提取子空间；在 math 和 code 上评估：

| 方法 | MMLU | BBH | GSM8K | SVAMP | MBPP | CodeAlpaca ↓ |
| ---- | ---: | --: | ----: | ----: | ---: | -----------: |
| Base | 46.0 % | 32.7 % | 11.0 % | 16.0 % | 17.0 % | 0.683 |
| SSD-QA | 48.0 % | 36.0 % | 19.0 % | 14.0 % | 12.0 % | 0.676 |
| **SPD-QA** | **49.0 %** | **38.7 %** | **26.0 %** | **17.0 %** | **21.0 %** | 0.680 |

只用 QA 梯度建立的子空间能把 GSM8K 提 **+15 pp**、MBPP 提 **+4 pp** —— *能力 filter 跨越了校准域*。作者的解读："校准域决定了提取的能力子空间，让我们能朝目标能力引导生成，并实现更有效的跨任务迁移。" 更冷静的解读：QA 推理跟 math/code 推理在 KV 激活空间里共享低秩结构，子空间捕捉到了这个共享结构。

### 关键消融 —— correctness-aligned vs full-sequence loss（Table 5）

这是 load-bearing 的实验。SPD 用 full-sequence loss（梯度在*所有*输出 token 上算，不只是 correctness-defining 那些）vs correctness-aligned 版本：

| 方法 | MBPP | CodeAlpaca ↓ | GSM8K | SVAMP | MMLU | BBH |
| ---- | ---: | -----------: | ----: | ----: | ---: | --: |
| Base | 17.0 % | 0.683 | 11.0 % | 16.0 % | 46.0 % | 32.7 % |
| SPD w/ Full-sequence | **11.9 %** | 0.681 | 13.0 % | 24.0 % | 48.0 % | 35.7 % |
| **SPD w/ Correctness-aligned** | **25.5 %** | 0.679 | 22.0 % | 32.0 % | 49.0 % | 38.7 % |

> [!important] Full-sequence loss 在 MBPP 上比 Base 还差
> 17 % → 11.9 %。"哪些 token 算数"的选择在干活，不是 SVD 机器单独的功劳。没有 correctness-aligned mask，梯度会捡起风格和格式方差，投影矩阵就投到*那些*噪声上。"能力"框架需要 correctness mask 才成立。

> [!example]- 完整消融结果（drill-down）
>
> **微调前自生成数据质量**（Table 3）：SPD 生成的输出在大多数 benchmark 上*微调前就已经*打过 Base 和 SSD 的生成 —— GSM8K 18 %（vs SSD 10 %、Base 11 %），BBH 35.3 %（vs SSD 28.7 %、Base 32.7 %）。hook 本身就改善了生成质量；微调进一步放大。
>
> **校准集规模敏感性**（Fig. 4）：SPD 数据高效 —— 20-500 校准例子性能差不多。区间内 GSM8K 19-22 %、SVAMP 26-32 %、MMLU 48-50 %、BBH 32-37 %。子空间估计在很少标注下就稳定。
>
> **微调分析**（Table 4）：PSR 在 MBPP 上 overfit（29.0 %）但 CodeAlpaca 没改善（NLL 0.683）。SSD 在 MBPP 上弱（18.3 %）也没 CodeAlpaca 改善。SPD 平衡到 MBPP 25.5 % + 最好的 CodeAlpaca（0.679）。
>
> **定性例子**（Fig. 3）：MBPP `remove_Occ` 上 Base 输出冗长有解释和 print 语句；SSD 短一些但逻辑仍错；SPD 输出最紧凑，去掉了装饰文本也修了 bug。hook *肉眼可见*把生成风格推向任务专注。

## 优势与限制

两个真实优势：(1) 训练时**不需外部信号**（只要小校准集），让它在没有 oracle 的 frontier 模型上可行；(2) **子空间跨域迁移** —— QA 校准帮 math/code —— 说明"能力子空间"捕捉的是结构性推理 pattern，不只是数据集表面特征。

可推敲的地方：

- **小模型秀场**。Qwen2.5-0.5B 有 +8.9 % 标题；Qwen3-4B 有 +2.1 %；Llama-3.1-8B 有 +1.9 %。报告范围内方法收益随规模衰减。能不能扩到 70B+（这恰恰是自蒸馏在安全 / 风格清理上最有用的规模）没测。
- **"Correctness-defining tokens" 在附录**。整个框架最重要的超参 —— $S^{(i)}$ 怎么定 —— 正文一句话点到 App. A.1。Math 答案位置很明显；code（`def remove_Occ(s, c):` 哪些 token 算数？）和 QA（字母？推理链？）都不明显。可复现性挂在正文没列的 per-domain 规则上。
- **Rank $r$ 没说**。正文从没说 $V_{K,r}^{(\ell)}$ 选的 $r$ 是多少。大概率敏感：太小子空间没法表示能力，太大噪声又溜进来。这放在 App. A.3，应该当一级消融。
- **只两层**。默认 $\mathcal{L} = \{L, \lfloor L/2 \rfloor\}$。没有"哪几层重要"的消融，没有探索全层投影或只投 attention 层。"中间 + 最后"有先前工作背书，但没针对 *本方法* 论证。
- **SVD 成本没在规模上量化**。70B 模型上每头 $d_k \approx 128$、校准 token 多，每头梯度堆很小（$M \times 128$），SVD 便宜。但梯度*收集*要冻结模型做 backward —— 20-500 例可以接受，但每能力都要校准就会累计。
- **没跟 on-policy distillation 比**。最自然的对照是"OPD/GKD 用同一个模型同时当学生和老师，$\lambda=1$"。这是最干净的"自蒸馏" baseline，SPD 没跑。
- **"Self-policy" 名字误导**。哪里都没策略梯度或 RL。是 SFT，在激活引导后的自 rollout 上。"带能力子空间引导的自蒸馏"更诚实。"Policy" 框架可能是想搭 [[on-policy-distillation|OPD]] / [[grpo|GRPO]] 命名风的车。
- **还没有 GitHub**。2026-05 预印本阶段无代码，附录 A.1 的 token 选择规则、A.3 的超参都无法独立验证。

> [!warning] 作者自承的限制
> "SPD 在三个能力域和多个 backbone 上评估；在更多样化、高风险任务上的更广验证会进一步加强其经验稳健性。" 翻译：math/code/QA 覆盖好了，但 agentic 任务、长上下文推理、多轮对话、多语言都没测。

## 这意味着什么

方法之外更有意思的框架：**自蒸馏的效果跟你能多干净地把能力轴从噪声轴分离开成正比**。SSD 的 trick 是截断式解码（粗糙的输出层 filter）；SPD 的 trick 是激活层投影（更精细的 filter）。两者都管用，因为它们减少了自生成训练数据被风格 / 格式 / 错误方差污染的程度。

12 个月预测三条：

1. **激活引导成为标准数据 curation 工具**。一旦大家接受"自生成输出是混合监督"，更多方法会作用于生成过程本身而非事后过滤。期待表征工程（RepE, ITI, CAA）和 SPD 风格的梯度子空间合流成统一的自我提升数据生成工具集。
2. **Correctness-token 选择自成一个子领域**。SPD 最大的隐藏超参 —— "哪些 token 算能力定义" —— 会得到更多关注。可能方向：用小 reward model 学、从 chain-of-thought 结构推导、用信息论度量。
3. **SPD-OPD 混合**。SPD 的子空间可以当 OPD 里的*老师侧* filter：不是蒸馏老师的原始分布，而是蒸馏老师的*引导后*分布。能力集中度复合：老师给信号，SPD 给聚焦。如果下一波 OPD 变体（KDRL、dGRPO、MOPD —— 见 [[on-policy-distillation#变体分类]]）吸收这个不奇怪。

它*不是*：frontier 能力扩展方法。SPD 把模型压向它自己的能力子空间 —— 这是"少噪声地做你已经擅长的事"的正则化。它教不了模型校准数据没覆盖的事情。要真正的能力提升（base 模型解不出的 math），verifiable reward 的 RL 或更强老师的 [[on-policy-distillation|OPD]] 仍然必需。

## 源码与复现

论文 2026-05 预印本时**没有公开代码**。论文里足够尝试复现的关键实现细节：

| 组件 | 设定 |
| ---- | ---- |
| 校准集大小 | 20-500 例（Fig. 4 显示该区间稳定） |
| 目标层 | $\mathcal{L} = \{\lfloor L/2 \rfloor, L\}$ —— 中间 + 最后 |
| 投影范围 | 每个目标层的 K 和 V 都投 |
| 子空间提取 loss | 在 $S^{(i)}$ token 上的 correctness-aligned NLL（Eq. 6） |
| 子空间秩 $r$ | 正文未指明，App. A.3 |
| Backbone | Qwen2.5-0.5B/7B/14B、Qwen3-4B、Llama-3.1-8B（全 -Instruct） |
| 微调方法 | LoRA |
| 微调 loss | 在 $(q^{(i)}, \hat y^{(i)})$ 上的标准 next-token NLL |

完整流程伪代码：

```python
# Phase 1 —— 提取投影矩阵（一次性）
def extract_subspace(model, cal_set, target_layers, rank_r):
    grad_K = {ell: [] for ell in target_layers}
    grad_V = {ell: [] for ell in target_layers}
    for q, y in cal_set:
        loss = correctness_aligned_loss(model, q, y)  # mask 掉 non-S token
        loss.backward()
        for ell in target_layers:
            grad_K[ell].append(model.layers[ell].K.grad)  # shape [T, d_k]
            grad_V[ell].append(model.layers[ell].V.grad)
        model.zero_grad()
    P_K, P_V = {}, {}
    for ell in target_layers:
        G_K = torch.cat(grad_K[ell], dim=0)               # [M, d_k]
        _, _, V_K = torch.linalg.svd(G_K, full_matrices=False)
        V_K_r = V_K[:rank_r].T                            # [d_k, r]
        P_K[ell] = V_K_r @ V_K_r.T                        # [d_k, d_k]
        # V 同理
    return P_K, P_V

# Phase 2 —— 挂 hook 生成，然后微调
def spd(model, cal_set, train_prompts, target_layers, rank_r):
    P_K, P_V = extract_subspace(model, cal_set, target_layers, rank_r)
    hooks = install_projection_hooks(model, P_K, P_V, target_layers)
    self_gen = [(q, model.generate(q)) for q in train_prompts]
    remove_hooks(hooks)
    finetune_lora(model, self_gen, loss="next_token_nll")
    return model
```

有意思的工程问题：hook 跟 RoPE / KV-cache 写入的顺序。论文没说 $\tilde K = K P_K$ 在 RoPE 之前还是之后 —— 大概是之后（保留位置编码），但没明示。

## 相关阅读

- [[on-policy-distillation]] —— 需要老师的同族；SPD 结构上类似 OPSD（自蒸馏 OPD），但加了子空间引导层。SPD 的变体条目应该住在 [[on-policy-distillation#变体分类]]。
- [[deepseek-v4-opd]] —— 旗舰级多老师全词表 OPD；跟 SPD 的无老师单模型路径形成对比。
- [[mopd]] —— NVIDIA 的多域 OPD；两篇都在乎 per-domain 校准，但 MOPD 切换的是老师，SPD 切换的是同一模型内的子空间。
- [[grpo]] —— RL 替代品；SPD 瞄准同一个"改善自 rollout"问题，但不要 reward model 和 value head 的开销。
- [[rlhf-overview]] —— SPD 部分替代的 pipeline（当目标是能力锐化、不是偏好对齐时）。

## 参考文献

- 论文：Hao et al., *Self-Policy Distillation via Capability-Selective Subspace Projection*, 2026-05-21. [arXiv:2605.22675](https://arxiv.org/abs/2605.22675)
- SSD（baseline）：Zhang et al., simple self-distillation for code. SPD 引为 [3]。
- On-policy distillation: [arXiv:2306.13649](https://arxiv.org/abs/2306.13649)（GKD）、[arXiv:2306.08543](https://arxiv.org/abs/2306.08543)（MiniLLM）。
- 表征工程背景 —— RepE、ITI、CAA —— SPD 把激活引导扩展到训练数据生成场景的脉络。
