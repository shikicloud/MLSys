---
title: "DeepSeek-V4 OPD：多教师全词表 On-Policy Distillation 替代 RL"
category: rl-infra
tags: [deepseek-v4, opd, on-policy-distillation, multi-teacher-kl, full-vocabulary-kl, post-training, moe, paper-review]
created: 2026-05-19
updated: 2026-05-19
status: mature
---

# DeepSeek-V4 OPD：多教师全词表 On-Policy Distillation 替代 RL

> [!info] 模型元信息
> - **发布**：2026-04-24（DeepSeek-V4-Pro 和 DeepSeek-V4-Flash 同时）
> - **Tech report**：仅以 PDF 形式托管在 HF：[huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf)（无 arXiv 提交）
> - **Model cards**：[V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)（1.6T 总 / 49B 激活）· [V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)（284B 总 / 13B 激活）
> - **License**：MIT，开放权重
> - **Context**：1M tokens
> - **API**：同日上线，见 [api-docs.deepseek.com/updates](https://api-docs.deepseek.com/updates)
> - **媒体**：[CNBC](https://www.cnbc.com/2026/04/24/deepseek-v4-llm-preview-open-source-ai-competition-china.html)、[MIT Technology Review](https://www.technologyreview.com/2026/04/24/1136422/why-deepseeks-v4-matters/)、[Bloomberg](https://www.bloomberg.com/news/articles/2026-04-24/deepseek-unveils-newest-flagship-a-year-after-ai-breakthrough)

> [!abstract]+ TL;DR
> DeepSeek-V4 相对 V3.2 做了一项 **关键方法学替换**：*"the mixed Reinforcement Learning (RL) stage was entirely replaced by On-Policy Distillation (OPD)"*（§5.1）。管线变成 **base → per-domain（SFT → GRPO）specialist → 多教师 OPD merge → V4**。OPD 损失是 $> 10$ 个 specialist teacher 的加权和：$\mathcal{L}_{\text{OPD}}(\theta) = \sum_i w_i\, D_{\text{KL}}(\pi_\theta \| \pi_{E_i})$，student 现采 rollout 上的反向 KL。相对 Thinking Machines Lab 博客（[[on-policy-distillation]]）和 MiniLLM 的算法新点是 **全词表 logit KL** —— V4 明确拒绝先前 on-policy distillation 工作的 token 级 KL 近似，理由是近似有高梯度方差、训练不稳。基础设施新点是 **在 1.6T-MoE 规模上让全词表 KL 跑起来** 的工程：teacher 在中央存储里只缓存隐藏状态（不是 logits）、按 teacher 排序的 sample 调度让 GPU 任一时刻最多挂一个 teacher 的 LM head、自研 TileLang exact-KL kernel、teacher 权重 FP4 QAT。**论文没报告 OPD vs GRPO 的 GPU-hour 对比** —— 网上 V4 评论里流传的成本声明继承自 TML 的 Qwen3 博客，不是 DeepSeek 自己的数字。

---

## 为什么重要

V4 是 **第一个完全押注在 OPD 上来做 post-training merge** 的旗舰级、开放权重模型。之前公开的 OPD 工作要么小规模（[[on-policy-distillation|Qwen3]] 0.6B–14B + 30B-A3B-MoE），要么把 OPD 当 RL 的组件（[Nemotron-Cascade 2 MOPD](https://research.nvidia.com/labs/nemotron/nemotron-cascade-2/)）。V4 是第一次有人在 1.6T 万亿级 MoE 上赌 OPD-替代-RL 假说并把 recipe 写出来。

如果 V4 推理表现稳得住（论文声称跟 GPT-5.2 / Gemini-3.0-Pro 在标准 reasoning benchmark 上具备竞争力），从 2026 起所有人在"是的，OPD 能在前沿规模替代 RL"这件事上都会引用 V4。如果守不住，V4 会成为 [[on-policy-distillation#opd-vs-rl-争论|OPD-replaces-RL 框架]] 在没有足够 RL 探索去超越 teacher 能力的情况下的反面案例。

---

## 架构（简）

只讲到能撑起 OPD 讨论的程度。完整架构分析不在本页范围内。

**两个同时发布的 SKU：**

| 模型 | 总参数 | 激活参数 | 架构 | Context |
|------|--------|---------|------|---------|
| DeepSeek-V4-Pro | 1.6T | 49B | MoE + hybrid CSA + HCA attention, mHC residuals, Muon optimizer | 1M |
| DeepSeek-V4-Flash | 284B | 13B | 同形状，更小 | 1M |

相对 V3 的关键架构变化：

- **Hybrid attention** (CSA + HCA) —— 在 1M 上下文规模上替换了 V3 的纯 MLA。
- **mHC residuals** —— 论文记号，社区解读尚不普及。
- **Muon optimizer** —— 部分训练阶段从 AdamW 切到 Muon。
- **MoE 专家数** —— 比 V3 增加；tech report 未公开具体数，但激活分数 (49/1600 ≈ 3 %) 跟 V3 的稀疏度持平。

训练基础设施一节明说 infra **复用 V3.2 栈**（rollout 引擎、容错生成服务、KV 池）。新东西在 post-training 不在 pre-training。

---

## OPD 管线

### Stage 1 —— Specialist 训练（§5.1.1）

对每个 domain $D_k \in$ {数学、代码、agent、instruction-following、alignment、...}：

1. 从 V4 base 起步。
2. **Domain 特定 SFT**，curated trace。
3. **GRPO RL** 配 domain 特定 reward。*"超参基本与之前研究对齐"*（V3.2 / R1 时代）。
4. 保存 domain specialist 为 $E_{D_k}$。

**三种推理力度模式** —— Non-think / Think High / Think Max —— 通过 **在不同长度惩罚和上下文窗口约束下训练不同 specialist** 得到，然后在 Stage 2 一起合并。

注意这一阶段 GRPO 还在用 —— V4 不是全局替换 GRPO，是替换 *V3.2 用来合并 specialist 的统一 mixed-RL post-training 阶段*。Per-domain RL 仍然做；变成 OPD 的是 *合并*。

**Generative Reward Model (GRM)：** V4 引入一个自评机制：*"the actor network natively functions as the GRM"* —— 联合优化打分 + 生成，替代难验证任务（alignment、instruction-following、agentic）的 scalar RLHF reward。这本身就是值得跟进的方法贡献；GRM 产出 specialist GRPO 训练用的稠密 reward 信号。

### Stage 2 —— 多教师 OPD merge（§5.1.2）

V4 之所以是 V4 的那次替换。Tech report 原话（line 1583）：

> *"Although the training pipeline largely mirrored that of DeepSeek-V3.2, a critical methodological substitution was made: the mixed Reinforcement Learning (RL) stage was entirely replaced by On-Policy Distillation (OPD)."*

Loss（论文 Eq. 29, p. 32）：

$$
\boxed{\,\mathcal{L}_{\text{OPD}}(\theta) \;=\; \sum_{i=1}^{N} w_i \cdot D_{\text{KL}}\!\left(\pi_\theta \,\Big\|\, \pi_{E_i}\right)\,}
$$

性质：

- **反向 KL**：student 在 KL 第一项 —— mode-seeking，跟 TML/MiniLLM 一致（见 [[on-policy-distillation#loss-函数|loss 推导]]）。
- **On-policy**：轨迹每步从 $\pi_\theta$ 现采（论文 §5.1.2：*"Computing the reverse KL loss … requires sampling training trajectories from the student π_θ to maintain on-policy learning."*）。
- **多教师**：$N > 10$ 个 specialist。
- **权重 $w_i$**：*"typically determined by the relative importance of the expert"* —— 没公布固定 schedule。
- **选择性对齐**：*"the unified policy π_θ selectively learns from the specialized expert relevant to the current task context"* —— prompt → 相关 teacher 的路由隐式在 per-prompt 的 domain 识别里；只有相关 teacher 在某 prompt 上提供有意义的梯度。

### 论文引用的祖宗

V4 OPD 一节显式引两条根（PDF line 1779）：

> *"we employ multi-teacher On-Policy Distillation (OPD; Gu et al. 2024; Lu and Lab 2025) as the primary technique for merging expert capabilities into the final model."*

- **Gu et al. 2024** = [MiniLLM](https://arxiv.org/abs/2306.08543) —— 反向 KL 作为策略梯度的推导。
- **Lu and Lab 2025** = [Thinking Machines Lab 博客](https://thinkingmachines.ai/blog/on-policy-distillation/) —— "RL 替代品"从业者叙事。

DeepSeek 没声称发明 OPD。新的是把它扩到万亿级 MoE 规模的多教师全词表部署。

---

## 真正的算法新点：全词表 KL

这是 V4 OPD 跟 TML recipe 不同的地方。论文论据（lines 1803-1812）：

> *"prior works usually simplify the full-vocabulary KL loss into a token-level KL estimate … reuse RL framework by replacing $\text{sg}\,\log(\pi_E / \pi_\theta)$ as the per-token advantage estimate … this approach … leads to high variance in gradient estimation and often causes training instability. Therefore, we adopt **full-vocabulary logit distillation in our OPD**. Preserving the complete logit distribution … yields more stable gradient estimates and ensures faithful distillation."*

两种配置：

| | Token 级 OPD（先前工作、TML、MiniLLM） | 全词表 OPD（DeepSeek-V4） |
|---|---|---|
| 比较什么 | teacher / student 在 *被采到的那个 token* 上的概率 | teacher / student 在 *所有词表 token* 上的概率分布 |
| 梯度形式 | $\nabla_\theta \log\pi_\theta(y_t) \cdot \log\frac{\pi_T(y_t)}{\pi_\theta(y_t)}$（REINFORCE 风格） | $\sum_v \pi_\theta(v) \log\frac{\pi_\theta(v)}{\pi_T(v)}$ 直接 KL 梯度 |
| 方差 | 高 —— $V$ 维 KL 单 token 采样 | 低 —— 完整分布上解析 KL |
| 每 token 内存 | $O(1)$ —— 两个 scalar logprob | $O(V)$ —— 两端的完整 softmax |
| 网络带宽（teacher 远程时） | 每 token $O(1)$ | 每 token $O(V)$，$V \approx 100$K |
| 是否忠实 | sequence 级 KL 的有偏估计 | 精确的 per-token KL |

全词表形式从 GKD（2023）起在数学上一直是显然的。**没人在规模上用是因为 $O(V \times \text{tokens} \times N_{\text{teachers}})$ 的内存和带宽不可行。** V4 用 §5.2.2 的基础设施 recipe 让它可行。

---

## 让全词表 OPD 跑起来的基础设施（§5.2.2）

对 NVIDIA reader 来说真正有意思的部分。引论文 "Efficient Teacher Scheduling for Full-Vocabulary OPD" 一节：

### 1. 隐藏状态缓存（不是 logits）

朴素做法 —— 对所有训练数据缓存每 token 每 teacher 的 logits —— 是 $O(|V| \times N_{\text{tokens}} \times N_{\text{teachers}})$。$|V| = 100$K、$N_{\text{tokens}} = $ 百万级、$N_{\text{teachers}} \geq 10$，是 TB 级 per epoch —— 不可行。

V4 的招：**只缓存 teacher 最后一层的隐藏状态**（不是 logits），训练时再 **在线跑预测 head**：

```
预训练前 per teacher：
  对每个训练 prompt x：
    h_E = teacher_E.last_hidden_states(x)    # 一次 forward，不过 LM head
    把 h_E 存入中央 buffer                    # 每 token O(d_model)，比 O(V) 小 ~50×

训练时：
  对每个 minibatch：
    sample student rollout y ~ π_θ
    加载该 rollout prompt 的 h_E
    teacher_logits = teacher_E.lm_head(h_E)  # 在缓存的隐藏状态上跑 head
    算全词表 KL(π_θ || softmax(teacher_logits))
```

相对 logit 缓存省 ~50× 内存（$d_{\text{model}} = 2048$、$V \approx 100$K 时）。

### 2. 按 teacher 索引排 sample

$> 10$ 个 teacher 时不可能所有 teacher 的 LM head 同时驻留 GPU。V4 在数据 dispatcher 按 sample 主 teacher 排序，让任一 microbatch 内只需要挂一个 teacher 的 LM head：

```
Microbatch dispatcher：
  按主 teacher index 排 sample
  在每个 "teacher chunk" 内部：
    把 teacher_E.lm_head 加载到 GPU
    对 chunk 内所有 sample 算 KL
  进下一个 teacher chunk
```

任一时刻每个 data-parallel rank 上 GPU 内最多一个 teacher head。代价是每 epoch 要加载 $\geq 10$ 个 LM head，靠 critical path 之外的 async I/O 摊。

### 3. Critical path 之外的 async I/O

Teacher 权重住在 **中央分布式存储**，**ZeRO 风格参数分片**，按需取。隐藏状态缓存同上。前一个 teacher chunk 算的时候后一个 async 加载。

### 4. 自研 TileLang 精确 KL kernel

TileLang 是 NVIDIA 最近开发的 tile-based 编程模型（类似 Triton）。V4 用自研 TileLang kernel 算精确的 teacher-student KL，融合 softmax —— 避开中间 logit materialization 和标准 "先算 logits 再算 loss" 的两遍模式。

### 5. Teacher FP4 QAT

Teacher 权重 **FP4 量化**（QAT，不是后训练量化）以塞进存储、让按需加载的带宽可行。Student 保持全精度。论文论据：teacher 提供 *目标分布*（不是梯度），FP4 推理可接受精度。

### 6. Rollout infra 复用

OPD rollout 跑在 **Stage 1 GRPO 同一套 preemptible、容错、WAL-based 生成服务**上。从基础设施视角看 OPD 就是 RL 减 reward —— 同 scheduler、同 rollout 引擎、同容错。

> [!important] Recipe 洞察
> V4 OPD 这一节读起来不像算法论文，更像 *一篇把一个先前只是假设的算法做到能跑的系统论文*。数学（Eq. 29）一行；工程（隐藏状态缓存 + 按 teacher 排 sample + TileLang + FP4 QAT + 复用 RL infra）才是把 "1.6T 规模多教师全词表 KL" 从研究梦变成生产管线的部分。

---

## 论文 *没有* 报告的

为了校准认知，V4 tech report 里缺的三块：

1. **OPD vs GRPO 的 GPU-hour 对比**。论文 *没* 公布"OPD 成本 vs mixed-RL 阶段成本"。V4 评论里流传的"便宜 10×"声明 **继承自 TML 的 Qwen3 博客**，不是 V4 自己的数。
2. **消融："V4 用 mixed RL 会比 V4 用 OPD 差"**。替换的合理性靠定性论证（避免 weight-merging 退化、全词表 KL 梯度方差更低）支撑，没有 A/B。
3. **具体权重 $w_i$**。只说"相对重要性"，没 schedule、没学习路由机制描述。
4. **OPD trainer 源码**。没公开。训练栈基于 V3.2 基础设施，那也是闭源的。

跟 DeepSeek 一贯的发布模式一致（权重开、训练栈闭），但如果想复现 recipe 要注意。

---

## V4 OPD 跟其他的关系

简谱：

| | GKD (2023) | MiniLLM (2023) | TML OPD (2025) | Qwen3 小 (2025) | Nemotron-Cascade 2 MOPD (2026) | **DeepSeek-V4 OPD (2026)** |
|---|---|---|---|---|---|---|
| Teacher 数 | 1 | 1 | 1 | 1 (更大 Qwen3) | 多 (per-domain best ckpt) | 多 (per-domain specialist) |
| 采样 | $\lambda$ 旋钮 | student | student | student | student | student |
| KL 方向 | $\beta$ 旋钮 (gen. JSD) | 反向 | 反向 | 反向 | 反向 | 反向 |
| KL 形式 | token 级 | token 级 (as PG) | token 级 | token 级 | token 级 | **全词表** |
| 管线角色 | KD 原语 | KD 原语 | RL 替代品 (博客框架) | 替代完整 RL 的 stage 3-4 | 与 cascade RL 交错做回归恢复 | **整段 post-training merge 阶段** |
| 模型规模 | T5 / PaLM-2 | medium dense | Qwen3-8B | Qwen3 0.6B-30B | 30B 激活 MoE | **1.6T MoE** |
| 源码 | google-deepmind | thu-coai/MiniLLM | (cookbook) | (Qwen) | (NVIDIA) | (闭源) |

V4 在每个维度上都是最激进位置 —— 模型最大、管线角色最广、teacher 最多、全词表 KL。

---

## 评论与延伸阅读

发布后窗口（2026-04 至 2026-05）我找到的几篇值得读：

- **Andrew Lukyanenko** —— [V4 Review](https://artgor.medium.com/deepseek-v4-review-why-million-token-context-needs-efficient-attention-not-just-larger-windows-6dc8e74a00b1)。讲替换：*"V4 replaces the unified GRPO pipeline from DeepSeek-R1 with a compositional alternative … decomposing into specialists and merging via full-vocabulary KL."*
- **OutcomeSchool** —— [Decoding DeepSeek V4](https://outcomeschool.com/blog/decoding-deepseek-v4)。讲为什么 on-policy 在合并阶段重要：*"the student never sees the kind of outputs it actually produces at inference time. With OPD, the student samples its own trajectories and the teacher corrects each token … more stable and faithful knowledge transfer than weight merging or mixed RL."*
- **BSWEN** —— [Two-Stage Post-Training Pipeline of DeepSeek V4](https://docs.bswen.com/blog/2026-04-25-deepseek-v4-two-stage-post-training/)。独立印证 Eq. 29 与全词表论据。
- **qingkeai.online（中文）** —— [DeepSeek V4 OPD 分析](https://qingkeai.online/archives/DeepSeek-V4-OPD)。把 OPD 框成 specialist 合并时对抗灾难性遗忘的工具。
- **Fireworks AI** —— [What DeepSeek V4 Says About Training Platforms](https://fireworks.ai/blog/what-deepseek-v4-says-about-training-platforms)。从基础设施视角写（对 NVIDIA 训练 infra 读者最相关）。

**Sebastian Raschka V3 → V3.2 深度解析**（[magazine post](https://magazine.sebastianraschka.com/p/technical-deepseek)）**没有覆盖 V4 / OPD** —— 比 V4 早几个月。他出 V4 OPD 跟进会很有价值；2026-05 中旬还没出。

---

## 诚实评估

**V4 OPD 真正新的点**（跟先前 on-policy distillation 工作相比）：

1. **旗舰规模的全词表 KL**。TML 和 MiniLLM 用 token 级近似是因为便宜；V4 证明配合正确的 infra（隐藏状态缓存 + 按 teacher 排 sample + TileLang + FP4 QAT）*精确* 的 per-token KL 在 1.6T MoE 规模也可行。方差降低在大规模下更重要，因为梯度不稳定会累积。
2. **多教师合并作为 post-training 范式**。先前 OPD 是单教师。V4 的 $\sum_i w_i D_{\text{KL}}$ 跨 $> 10$ 个 specialist 是 *质上* 不同的用法 —— 不是"压缩一个 teacher"，是"合并 specialist" —— 这一形式可以推广到任何你有正交能力想合并的场景。
3. **按推理力度分 specialist**。给 Non-think / Think High / Think Max 各训一个 specialist 再用 OPD 合并，对"一个模型多种推理力度"是一个干净的架构答案。之前的做法（一个模型按 mode token 条件化）有能力稀释问题。

**回收的部分（论文正确引用）：**

- 反向 KL on-policy 形式（MiniLLM 2023）。
- "OPD 作为 RL 替代品"叙事（TML 2025-10）。
- GRPO specialist 训练（DeepSeek-V3 / R1 栈）。

**Marketing 或没被验证的：**

- 成本声明。论文没发 GPU-hour 对比；评论里"比 RL 便宜 10×"的标题来自把 TML 的 Qwen3 数字洗到 V4 讨论里。
- 替换决策靠定性，没实证。没有 V4-with-mixed-RL 的消融。
- 1.6T-MoE 规模声明独立无法验证因为 trainer 闭源。

**总结。** V4 OPD recipe 是迄今为止 on-policy distillation 最雄心勃勃的部署，工程足够具体到可复现（如果有基础设施），从 token 级 KL 转向全词表 KL 的方向值得认真对待。"完整管线替换"的声明大胆，独立验证还没出，但作为 *recipe 来研究* 它是 2026 的标准参考。

---

## 训模型时该带走什么

1. **如果你有同家族的强 teacher（或一组 specialist）**：试 OPD 再考虑 RL。从 HF TRL 的 `GKDTrainer`（token 级）开始原型；上规模时考虑全词表 KL。
2. **要合并 specialist**：V4 是标准 recipe —— 各 domain 用 GRPO 训，再用多教师反向 KL OPD 合并。加权和 $\sum_i w_i D_{\text{KL}}$ 结构组合性好。
3. **要前沿能力扩展**：不要指望 OPD 独自解决。当目标超过 teacher 时叠 OPD warm-start + GRPO（或 [KDRL](https://arxiv.org/abs/2506.02208) 联合目标）。
4. **要基础设施**：全词表 KL 在缓存 teacher 隐藏状态（不缓存 logits）、按 teacher 排 sample、teacher FP4 QAT 之下可行。可复用的洞察是 **OPD 在基础设施层看就是 RL 减 reward** —— 同 rollout 引擎、同容错、同 scheduler。

---

## 相关阅读

- [[on-policy-distillation]] —— 总伞技术（算法 + 变体 + 争论）；V4 OPD 是其旗舰规模实例。
- [[grpo]] —— V4 Stage 1 specialist 训练用的 RL 算法，也是 Stage 2 merge 被替换掉的那个。
- [[ppo-for-llm]] —— 两阶段共享的 KL 正则的 trust-region 直觉。
- [[rlhf-overview]] —— V4 在 merge 阶段颠覆的标准 post-training 管线。
- [[parallelism-strategies-deep-dive#14. 实战案例：DeepSeek-V3]] —— V4 借鉴的 V3 架构和并行基础。
- [[kv-cache-optimization]] —— 1M context 下的 KV 管理，跟 V4 推理基础设施相关。
- [[das-spec-rl]] —— GRPO / OPD 管线 rollout 阶段的投机解码加速；推理层互补。

## 参考文献

- **DeepSeek-V4 tech report**：[HF PDF](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf)
- **DeepSeek-V4-Pro model card**：[huggingface.co/deepseek-ai/DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
- **DeepSeek-V4-Flash model card**：[huggingface.co/deepseek-ai/DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
- **API 更新**：[api-docs.deepseek.com/updates](https://api-docs.deepseek.com/updates)
- **Thinking Machines Lab — On-Policy Distillation 博客**（V4 引用）：[thinkingmachines.ai/blog/on-policy-distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)
- **MiniLLM**（V4 引用）：Gu et al., [arXiv:2306.08543](https://arxiv.org/abs/2306.08543)
- **GKD**（总数学）：Agarwal et al., [arXiv:2306.13649](https://arxiv.org/abs/2306.13649)
- **独立 V4 解读**：[Lukyanenko](https://artgor.medium.com/deepseek-v4-review-why-million-token-context-needs-efficient-attention-not-just-larger-windows-6dc8e74a00b1)、[OutcomeSchool](https://outcomeschool.com/blog/decoding-deepseek-v4)、[BSWEN](https://docs.bswen.com/blog/2026-04-25-deepseek-v4-two-stage-post-training/)、[qingkeai.online](https://qingkeai.online/archives/DeepSeek-V4-OPD)、[Fireworks AI](https://fireworks.ai/blog/what-deepseek-v4-says-about-training-platforms)
- **媒体**：[CNBC](https://www.cnbc.com/2026/04/24/deepseek-v4-llm-preview-open-source-ai-competition-china.html)、[MIT Technology Review](https://www.technologyreview.com/2026/04/24/1136422/why-deepseeks-v4-matters/)、[Bloomberg](https://www.bloomberg.com/news/articles/2026-04-24/deepseek-unveils-newest-flagship-a-year-after-ai-breakthrough)
