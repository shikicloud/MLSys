---
title: "DeepSeek-V4 OPD：多教师全词表 On-Policy Distillation 替代 RL"
category: rl-infra
tags: [deepseek-v4, opd, on-policy-distillation, multi-teacher-kl, full-vocabulary-kl, post-training, moe, paper-review]
created: 2026-05-19
updated: 2026-05-21
status: mature
paper: DeepSeek-V4 technical report
# Figures TODO: tech report PDF (HF only) 没有 OPD 相关的图；pdfimages 只产出 Fig. 1（benchmark bar chart）的碎片。报告里没有 pipeline / KL 对比 / cache 基础设施图。
---

# DeepSeek-V4 OPD：多教师全词表 On-Policy Distillation 替代 RL

> [!info] 论文元信息
> - **Paper**：[DeepSeek-V4 tech report (HF PDF)](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf) —— DeepSeek, 2026-04-24（无 arXiv 提交）
> - **Models**：[DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)(1.6T 总 / 49B 激活)· [DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)(284B / 13B)
> - **License**：MIT,开放权重
> - **Context**:1M tokens
> - **API**:同日上线 —— [api-docs.deepseek.com/updates](https://api-docs.deepseek.com/updates)
> - **OPD trainer 源码**:未发布
> - **媒体**:[CNBC](https://www.cnbc.com/2026/04/24/deepseek-v4-llm-preview-open-source-ai-competition-china.html)· [MIT Tech Review](https://www.technologyreview.com/2026/04/24/1136422/why-deepseeks-v4-matters/)· [Bloomberg](https://www.bloomberg.com/news/articles/2026-04-24/deepseek-unveils-newest-flagship-a-year-after-ai-breakthrough)

---

## 摘要(2 分钟读完这一节就够)

**它是什么**。DeepSeek-V4(1.6T 参数 MoE,49B 激活)相对 V3.2 做了一项关键方法学替换:tech report §5.1 原文 *"the mixed Reinforcement Learning (RL) stage was entirely replaced by On-Policy Distillation (OPD)."* Post-training 管线变成 **base → per-domain (SFT → GRPO) specialist → 多教师 OPD merge → V4**。

**核心思想**。Per-domain specialist 用 GRPO 训完后,通过 **精确每 token 全词表反向 KL** 对 $> 10$ 个 teacher 做 on-policy 蒸馏来合并成一个统一模型。三个支柱:

1. **反向 KL 多教师合并**,不是权重平均 —— 保留 specialist 的行为指纹。
2. **全词表 KL** 替代 TML / MiniLLM / GKD 用的 token 级 Monte Carlo 估计器 —— 长 rollout 规模上关键的方差减少手段。
3. **工程 recipe**(缓存 teacher 隐藏状态、按 teacher 排 sample、TileLang 精确 KL kernel、teacher FP4 QAT、复用 V3.2 rollout / WAL 基础设施)让上述在 1.6T-MoE 规模上可行。

去掉任何一个:失去多教师就退化成单教师 OPD 带 TML 的所有限制;失去全词表就重引入方差病;失去工程就跑不起来。

**头条结果**。V4-Pro 在 AIME、GPQA、code 等推理 benchmark 上与 GPT-5.2、Gemini-3.0-Pro 具备竞争力,声称开源 SimpleQA SOTA。**论文没报告 OPD vs GRPO 的 GPU-hour 对比** —— 网上流传的"比 RL 便宜 10×"声明继承自 TML 的 Qwen3 博客,**对 V4 没被验证**。

**为什么这重要**。

- **第一个旗舰规模演示** OPD 能完全替代 post-training 里的 mixed-RL merge 阶段。2026 起这就是标准引用。
- **全词表 KL 是规模上的新默认**。Token 级估计器方差随 rollout 长度增长 —— 对 16K+ token reasoning 模型恰好是错误的 scaling 方向。
- **OPD-作为-merge-阶段** 回答了任何训大 MoE 模型的人都面临的真实问题(统一 RL 下的能力稀释);预期 Qwen / Mistral / 开源社区在 6–12 个月内跟进。
- **这 *不是* 万能 RL 杀手**。V4 在 Stage 1 specialist 训练里还在用 GRPO。替换只发生在 *merge* 阶段。

---

# 深度部分(往下展开细节)

上面摘要是 executive 层。下面是给愿意细读管线、算法论证、工程 recipe 和实证范围的人准备的。

## 背景:为什么 V4 要替换 V3.2 的 mixed-RL 阶段

V3.2 / R1 用的是 "四阶段 mixed-RL" post-training recipe:SFT → reasoning RL → mixed RL → alignment RL。两个结构性问题让它在 V4 想达到的规模上既贵又不稳:

| 问题 | 为什么在规模上显现 | 代价 |
| ---- | ----------------- | ---- |
| **稀疏 outcome reward** | 16K+ token 推理 rollout 只收一个 0/1 信号 | 多数 token 没 credit assignment;从 scalar 反推用 GPU 贵 |
| **mixed RL 中 specialist 回归** | 不同 domain(math、code、agent、IF)要不同 reward 信号;训一个回归另一些 | 要么重训(贵)要么接受回归 |
| **没有干净的 reasoning 模式分离** | 单模型靠 mode token 处理 Non-think / Think High / Think Max 会能力稀释 | 需要架构层 workaround |

V3.2 → V4 的设计思路:**拆成 specialist 独立用 GRPO 训,再用 OPD 合并**。RL 提供 per-domain exploration 与 reward 驱动训练;OPD 提供稠密 token 级信号,让合并 specialist 又快又稳。结果是迄今为止最激进的 [[on-policy-distillation|OPD]] 部署 —— V4 是第一个把 1.6T MoE *全押* 在 OPD-替换-合并-阶段假说上的旗舰。

交叉参考:[[on-policy-distillation#背景为什么需要发明-on-policy-蒸馏|OPD 的 SFT-vs-RL 框架]] 给这个奠定了基础。

## 两阶段管线详解

Trainer 的 API 契约延续自 V3.2;变的是 merge 那一步跑什么。

```
                  ┌──────────────────────────────────────────────┐
V4 base ────►     │  Stage 1 (§5.1.1) —— Specialist 训练           │
                  │                                                │
                  │  for domain D in {math, code, agent, IF, ...}: │
                  │      SFT(D)  →  GRPO(D)  →  best ckpt E_D     │
                  │                                                │
                  │  GRM(生成式奖励模型):actor 自己也充当       │
                  │  judge,给难验证任务出 reward                 │
                  │                                                │
                  │  3 种推理模式(Non-think / Think High /       │
                  │  Think Max)训不同 specialist,分别用不同     │
                  │  长度惩罚 + 上下文窗口                        │
                  └─────────────────────┬──────────────────────────┘
                                        │
                                        ▼  10+ specialist
                  ┌──────────────────────────────────────────────┐
                  │  Stage 2 (§5.1.2) —— 多教师 OPD merge           │
                  │                                                │
                  │  student rollout y ~ π_θ                       │
                  │  loss = Σ_i w_i · D_KL(π_θ || π_{E_i})         │
                  │  对每个 π_{E_i} 跑全词表反向 KL                │
                  │                                                │
                  │  路由:prompt → 相关 specialist               │
                  │ (选择性对齐;每个 prompt 上只有一个 teacher  │
                  │  给得出有意义梯度)                           │
                  └─────────────────────┬──────────────────────────┘
                                        │
                                        ▼
                                   DeepSeek-V4
```

### Stage 1 —— Specialist 训练 (§5.1.1)

对每个 domain $D_k$ ∈ `{math, coding, agent, IF, alignment, ...}`:

1. 从 V4 base 起步(架构:hybrid CSA + HCA attention、mHC residuals、部分阶段 Muon optimizer)。
2. **Domain 特定 SFT**,curated trace。
3. **GRPO RL** 配 domain 特定 reward。论文说 *"超参基本与之前研究对齐"*(V3.2 / R1 时代)。
4. 保存 domain specialist 为 $E_{D_k}$。

**GRM 贡献**:V4 引入一个 Generative Reward Model,*"the actor network natively functions as the GRM"* —— 联合优化打分 + 生成,替代难验证任务(alignment、IF、agentic)的 scalar RLHF reward。GRM 给 Stage 1 内的 GRPO 训练提供稠密 reward 信号。

**推理模式 specialist**:Non-think / Think High / Think Max 由 **不同长度惩罚 + 不同上下文窗口下训的不同 specialist** 实现,而不是 mode token 条件化。Stage 2 的 OPD merge 把它们融合。

> [!note] GRPO 仍在使用 —— 只是不再用于合并
> V4 不是全局替换 GRPO。替换的是 V3.2 用来合并 specialist 的统一 *mixed-RL* 阶段。Per-domain RL 仍然做;变成 OPD 的是合并阶段。

### Stage 2 —— 多教师 OPD merge (§5.1.2)

让 V4 之所以是 V4 的那次替换。Tech report 原话(line 1583):

> *"Although the training pipeline largely mirrored that of DeepSeek-V3.2, a critical methodological substitution was made: the mixed Reinforcement Learning (RL) stage was entirely replaced by On-Policy Distillation (OPD)."*

> [!quote] 一句话总结贡献
> Per-domain specialist 用 GRPO 训完后,通过 **精确每 token 全词表反向 KL** 对 $> 10$ 个 teacher 做 on-policy 蒸馏来合并成一个统一模型 —— 由一套缓存 teacher 隐藏状态而非 logits 的工程 recipe 在 1.6T 规模上跑起来。

Loss(论文 Eq. 29, p. 32):

$$
\boxed{\,\mathcal{L}_{\text{OPD}}(\theta) \;=\; \sum_{i=1}^{N} w_i \cdot D_{\text{KL}}\!\left(\pi_\theta \,\Big\|\, \pi_{E_i}\right)\,}
$$

性质:

| 维度 | 取值 | 备注 |
| ---- | ---- | ---- |
| KL 方向 | 反向(student 在第一项) | Mode-seeking;与 TML/MiniLLM 一致 |
| 采样 | $y \sim \pi_\theta$(student rollout) | On-policy;匹配部署分布 |
| Teacher 数 | $N > 10$ | 一个 domain specialist 一个 |
| 权重 $w_i$ | "Relative importance of the expert" | 没公开 schedule |
| 路由 | Per-prompt 到相关 teacher | 选择性对齐 —— 每 prompt 上只有一个 teacher 给得出强梯度 |

### 全词表 KL 的选择

相对 TML / MiniLLM 的真正算法差异点。论文论据(lines 1803–1812):

> *"prior works usually simplify the full-vocabulary KL loss into a token-level KL estimate … this approach … leads to high variance in gradient estimation and often causes training instability. Therefore, we adopt **full-vocabulary logit distillation in our OPD**."*

| | Token 级 OPD(TML、MiniLLM、GKD 默认) | 全词表 OPD(V4) |
| --- | --- | --- |
| 估计器 | $V$ 维 KL 的单 token 采样 | 解析精确 KL |
| 梯度 | $\nabla \log\pi_\theta(y_t) \cdot \log(\pi_T(y_t)/\pi_\theta(y_t))$ | $\sum_v \pi_\theta(v) \log(\pi_\theta(v)/\pi_T(v))$ |
| 方差 | 高;在长 rollout 上 compound | 低 |
| 每 token 内存 | $O(1)$ | $O(V)$,$V \approx 100$K |
| 每 token 带宽 | $O(1)$ | $O(V)$ —— 没工程就不可行 |
| 1.6T 规模可行性 | trivial | 难 —— 需要 §5.2.2 基础设施 |

全词表形式从 GKD(2023)起在数学上就显然。没人在规模上用是因为内存 / 带宽不可行。**V4 的算法贡献是这个选择;V4 的系统贡献是让它跑起来。**

## 让全词表 OPD 可行的基础设施 (§5.2.2)

这是把 Eq. 29 从研究梦变成生产管线的部分。五项工程动作让 10+ teacher × 全词表 KL × 1.6T 模型真的能跑。

### 1. 缓存 teacher 隐藏状态,不是 logits

朴素是 $O(|V| \times N_{\text{tokens}} \times N_{\text{teachers}})$ —— 每 epoch TB 级。V4 只缓存 teacher **最后一层隐藏状态**(不缓存 LM head 输出),训练时再 **在线跑 LM head**:

```python
# 预训练前 per teacher specialist
for prompt in training_data:
    h_E = teacher_E.last_hidden_states(prompt)   # 每 token O(d_model)
    store(h_E)                                   # 比 O(V) 小 ~50×

# 训练时
for batch in dataloader:
    y = sample_rollout(student)                  # on-policy
    h_E = load_cached_hidden_states(batch)
    teacher_logits = teacher_E.lm_head(h_E)      # 在缓存 h 上跑 head
    loss = full_vocab_kl(student_logits, teacher_logits)
```

在 $d_{\text{model}} = 2048$、$V \approx 100$K 时省 ~50× 内存。

### 2. 按主 teacher 排 sample

$> 10$ 个 teacher 时不可能所有 LM head 同时驻 GPU。V4 在 data dispatcher 按 sample 主 teacher 索引排序,让一个 microbatch 内 **最多一个 teacher LM head 驻 GPU**:

```
Dispatcher:
  group(samples, key=primary_teacher_index)
  for teacher_chunk in groups:
      gpu_load(teacher_chunk.teacher.lm_head)
      compute_kl_for_all(teacher_chunk.samples)
```

代价:每 epoch 加载 $\geq 10$ 个不同 LM head —— 靠 critical path 外的 async I/O 摊。

### 3. TileLang 精确 KL kernel + teacher FP4 QAT

> [!note]- Async I/O、TileLang kernel、FP4 QAT、复用 V3.2 栈 —— 展开看 recipe 剩下部分
>
> **Critical path 之外的 async I/O**。Teacher 权重住中央分布式存储,ZeRO 风格参数分片,按需取。隐藏状态缓存同上。前一个 teacher chunk 算的时候后一个 async 加载。
>
> **自研 TileLang 精确 KL kernel**。TileLang 是 NVIDIA 最近开发的 tile-based 编程模型(类似 Triton)。V4 用自研 kernel 算精确的 teacher-student KL **融合 softmax**,避开 "logits → loss" 两遍模式。
>
> **Teacher FP4 QAT**。Teacher 权重 **FP4 量化 via QAT**(不是后训练量化),以塞进存储 + 让按需加载带宽可行。Student 保持全精度。论据:teacher 提供 *目标分布*(不是梯度),FP4 推理精度可接受。
>
> **复用 V3.2 rollout / WAL 栈**。OPD rollout 跑在 **Stage 1 GRPO 同一套 preemptible、容错、WAL-based 生成服务**上。从基础设施视角看 OPD 就是 RL 减 reward —— 同 scheduler、同 rollout 引擎、同容错。

> [!important] 系统洞察
> V4 §5.2.2 读起来不像算法论文,更像 *一篇把一个先前只是假设的算法做到能跑的系统论文*。数学(Eq. 29)一行;工程(隐藏状态缓存 + 按 teacher 排 sample + TileLang + FP4 QAT + 复用 RL infra)才是把 "1.6T 规模多教师全词表 KL" 从研究梦变成生产管线的部分。

## 头条证据

V4 是模型发布不是受控研究,所以这一节报告论文报告了什么、缺什么。

**论文报告的**。

- **推理质量**:V4-Pro 在 AIME、GPQA、code 上与 GPT-5.2、Gemini-3.0-Pro 具备竞争力。V4-Pro 声称开源 SimpleQA SOTA。
- **架构**:hybrid CSA + HCA attention、mHC residuals、Muon optimizer 下的 1M 上下文 —— 但架构故事独立于 OPD 故事。
- **OPD 管线验证**:合并产出一个单一统一模型,论文声称保留了 per-domain specialist 的能力。

**论文 *没* 报告的**(决定要不要相信 OPD pitch 的关键):

| 缺什么 | 为什么重要 | 流传声明的来源 |
| ------ | ---------- | -------------- |
| GPU-hour 对比:OPD merge vs mixed-RL merge | 替换的全部理由本应是效率 | "便宜 10×" 标题继承自 [TML 的 Qwen3 博客](https://thinkingmachines.ai/blog/on-policy-distillation/),不是 V4 |
| 消融:V4-with-mixed-RL vs V4-with-OPD | 证明 OPD *更好* 而不只是 *不同* | 无 —— 替换靠定性论证 |
| 具体 $w_i$ schedule | 路由是学的还是手调的 | Tech report 只说 "relative importance" |
| OPD trainer 源码 | 可复现性 | DeepSeek `github.com/deepseek-ai` 到 2026-05 没有 V4 / OPD repo |

> [!warning] marketing 与真正起作用的部分
> *起作用*:V4(post-OPD)达到具备竞争力的 reasoning 数字。全词表 KL + 多教师合并 recipe 具体到可重新实现。*marketing 或未验证*:成本声明、"OPD 严格胜过 mixed RL" 的暗示。任何 V4 成本声明都当成继承自 TML 而不是 V4 结果来对待。

> [!example]- V4 OPD 与其它 OPD 部署的关系(对比表)
>
> | | GKD (2023) | MiniLLM (2023) | TML OPD (2025) | Qwen3 小 (2025) | Nemotron-Cascade 2 MOPD (2026) | **DeepSeek-V4 OPD (2026)** |
> | --- | --- | --- | --- | --- | --- | --- |
> | Teacher 数 | 1 | 1 | 1 | 1 | 多(per-domain ckpt) | **多**(per-domain specialist) |
> | 采样 | $\lambda$ 旋钮 | student | student | student | student | **student** |
> | KL 方向 | $\beta$ 旋钮 | 反向 | 反向 | 反向 | 反向 | **反向** |
> | KL 形式 | token 级 | token 级 (as PG) | token 级 | token 级 | token 级 | **全词表** |
> | 管线角色 | KD 原语 | KD 原语 | RL 替代品(博客) | 替代 RL stage 3-4 | 与 cascade RL 交错做回归恢复 | **整段 post-training merge 阶段** |
> | 模型规模 | T5 / PaLM-2 | medium dense | Qwen3-8B | Qwen3 0.6B-30B | 30B 激活 MoE | **1.6T MoE** |
> | 源码 | open | open | partial (cookbook) | partial (Qwen) | partial (NVIDIA) | **闭源** |
>
> V4 在每个维度上都是最激进位置。

## 优势与限制

最强两点:(1) **第一个旗舰规模、完全开放权重的演示**,证明 OPD 能完全替代 mixed-RL post-training —— 2026 起这就是标准引用;(2) **全词表 KL 作为方差减少手段** 是真正的算法主张,不是 marketing —— 让它在 1.6T 规模可行的工程本身就是贡献。

诚实承认的限制:

- **没有 vs GRPO 的成本消融**。论文不公布 GPU-hour 对比。V4 评论里任何"比 RL 便宜 10×"声明都来自 TML 的 Qwen3 博客,**对 V4 没被验证**。
- **没有 vs mixed-RL 的 A/B**。替换靠定性合理化(避免 weight-merging 退化、全词表 KL 梯度方差更低)。没有 V4 规模上 OPD 击败继续 mixed RL 的数据。
- **OPD trainer 闭源**。V4 OPD 构建于其上的 V3.2 post-training 栈也是闭源的。要重新实现得从头搭建 hidden-state-cache + TileLang kernel + FP4 QAT + sample-sort dispatcher。
- **Specialist 路由不透明**。$w_i$ 权重和 prompt-to-teacher 路由逻辑没详细到能复现。
- **继承 OPD 的结构限制**。[[on-policy-distillation#优势与限制|OPD 作为技术的限制]] 全部适用:被 teacher 能力上限封顶、冷启动脆弱等。V4 的 specialist 都从同一 base GRPO 训出来,跨家族负迁移风险弱化,但能力天花板论点仍成立。

## 启示

三条值得跟踪的预测:

1. **OPD-作为-merge-阶段会变成 MoE post-training 的默认**。V4 是 proof-of-concept。预期 Qwen、Mistral、开源社区在 6–12 个月内跟进。"per-domain-or-mode specialist + OPD merge" 架构回答了任何训大 MoE 模型的人都面临的真实问题(统一 RL 下的能力稀释)。
2. **规模上的 OPD 默认会用全词表 KL**。Token 级估计器的方差病真实存在且随 rollout 长度增长 —— 对 16K+ token reasoning 模型恰好是错误的 scaling 方向。等 V4 工程 recipe 被 HF TRL / veRL / NeMo-RL 复现,token 级 OPD 会退到小规模原型阶段。
3. **有意思的前沿移到 teacher 多样性**。多教师 OPD 标配后,下个问题是合并哪些 teacher —— 同家族 specialist(V4 现状)、跨家族(GOLD 风格)、多智能体辩论(MAD-OPD)、还是 RL reward head(KDRL)。多样性工程变成差异化点。

这 *不是*:万能 RL 杀手。V4 在 Stage 1 specialist 训练里还在用 GRPO。前沿能力扩展 —— 推过最强可得 teacher —— 仍然需要 RL exploration。V4 证明 OPD 能 *替换合并*,不是 *消除 RL*。

## 源码与复现

> [!note]- 公开发布状态 —— 什么开了、什么没开
>
> | Artifact | 状态 |
> | -------- | ---- |
> | 模型权重(V4-Pro, V4-Flash) | ✓ 开放,MIT |
> | Tech report | ✓ HF 上的 PDF |
> | OPD trainer 源码 | ✗ 未发布 |
> | 隐藏状态缓存 / TileLang KL kernel | ✗ 未发布 |
> | Specialist checkpoint / 权重 $w_i$ | ✗ 未发布 |
> | V3.2 post-training 基础设施 | ✗ 闭源(V4 OPD 构建在其上) |
>
> 含义:V4 OPD recipe 是 **paper-only**。要复现得在现有 trainer(HF TRL `GKDTrainer`、veRL、NeMo-RL)上重新实现 §5.2.2 的完整基础设施。

**最小复现草图**。

```python
# 概念草图 —— 完整实现需要 §5.2.2 基础设施
from trl import GKDConfig, GKDTrainer
from transformers import AutoModelForCausalLM

student = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-V4-Base")  # 假设有

teachers = [
    AutoModelForCausalLM.from_pretrained(f"deepseek-ai/V4-specialist-{d}")
    for d in ["math", "code", "agent", "IF", "alignment", ...]   # 10+ specialist
]
weights = {d: w_d for d, w_d in importance_per_domain.items()}

# V4 风格多教师全词表 KL(stock TRL 不支持 —— 需要自定义 training_step)
def opd_loss(student_logits, teacher_logits_per_specialist, weights):
    return sum(
        weights[i] * full_vocab_kl(student_logits, teacher_logits_per_specialist[i])
        for i in range(len(teachers))
    )
```

需要在 stock TRL 上建的基础设施部分:

1. **多 teacher logit 拉取** —— 当前 `GKDTrainer` 只支持单 teacher。
2. **隐藏状态缓存层** —— 预算 teacher 隐藏状态,存中央 buffer。
3. **Sample-sort-by-teacher dispatcher** —— GPU 内最多一个 teacher LM head。
4. **全词表精确 KL kernel** —— Triton 或 TileLang 实现;不是 `generalized_jsd_loss` 里的 JSD 分支。
5. **Teacher FP4 QAT** —— 存储 / 加载用。

> [!note]- 最接近 V4 的可用参考实现
>
> | 项目 | 给你什么 | 距 V4 还差什么 |
> | ---- | -------- | -------------- |
> | [HF TRL `GKDTrainer`](https://github.com/huggingface/trl/blob/main/trl/trainer/gkd_trainer.py) | 单 teacher 反向 KL OPD,token 级 | 多教师、全词表、基础设施缺 |
> | [veRL `algo/opd`](https://verl.readthedocs.io/en/latest/algo/opd.html) | 多教师路由 via `data_source` | 仅 token 级;无隐藏状态缓存 |
> | [NeMo-RL OPD (#1445)](https://github.com/NVIDIA-NeMo/RL/discussions/1445) | Top-k 限制 KL(带宽优化) | 最接近全词表;无公开多教师管线 |
> | [Tinker cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) | 单 + 多教师 OPD recipe | Token 级,无全词表路径 |

## 相关阅读

- [[on-policy-distillation]] —— OPD 总伞页(起源 paper、数学、变体、争论);V4 OPD 是其旗舰规模实例。
- [[grpo]] —— V4 Stage 1 specialist 训练用的 RL 算法;Stage 2 merge 被替换掉的那个。
- [[ppo-for-llm]] —— 两阶段共享的 KL 正则的 trust-region 直觉。
- [[rlhf-overview]] —— V4 在 merge 阶段颠覆的标准 post-training 管线。
- [[parallelism-strategies-deep-dive#14. 实战案例:DeepSeek-V3]] —— V4 借鉴的 V3 架构和并行基础。
- [[kv-cache-optimization]] —— 1M context 下的 KV 管理,跟 V4 推理基础设施相关。
- [[das-spec-rl]] —— GRPO / OPD 管线 rollout 阶段的投机解码加速;推理层互补。

## 参考文献

- **DeepSeek-V4 tech report**:[HF PDF](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf)(无 arXiv)
- **DeepSeek-V4-Pro model card**:[huggingface.co/deepseek-ai/DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
- **DeepSeek-V4-Flash model card**:[huggingface.co/deepseek-ai/DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
- **API 更新**:[api-docs.deepseek.com/updates](https://api-docs.deepseek.com/updates)
- **GKD(V4 引用的 OPD 原 paper)**:Agarwal et al., ICLR 2024。[arXiv:2306.13649](https://arxiv.org/abs/2306.13649)
- **MiniLLM(V4 引用)**:Gu et al., NeurIPS 2024。[arXiv:2306.08543](https://arxiv.org/abs/2306.08543)
- **Thinking Machines Lab — On-Policy Distillation 博客**(V4 引用):[thinkingmachines.ai/blog/on-policy-distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)
- **独立 V4 解读**:[Lukyanenko](https://artgor.medium.com/deepseek-v4-review-why-million-token-context-needs-efficient-attention-not-just-larger-windows-6dc8e74a00b1)· [OutcomeSchool](https://outcomeschool.com/blog/decoding-deepseek-v4)· [BSWEN](https://docs.bswen.com/blog/2026-04-25-deepseek-v4-two-stage-post-training/)· [qingkeai.online](https://qingkeai.online/archives/DeepSeek-V4-OPD)· [Fireworks AI](https://fireworks.ai/blog/what-deepseek-v4-says-about-training-platforms)
- **媒体**:[CNBC](https://www.cnbc.com/2026/04/24/deepseek-v4-llm-preview-open-source-ai-competition-china.html)· [MIT Technology Review](https://www.technologyreview.com/2026/04/24/1136422/why-deepseeks-v4-matters/)· [Bloomberg](https://www.bloomberg.com/news/articles/2026-04-24/deepseek-unveils-newest-flagship-a-year-after-ai-breakthrough)
