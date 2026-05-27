---
title: "Search-R1：用 RL 训练 LLM 推理 + 调搜索引擎"
category: agentic-rl
tags: [search-r1, agentic-rl, retrieval-augmented, ppo, grpo, r1-zero-lineage, retrieved-token-masking, 论文精读]
created: 2026-05-26
updated: 2026-05-26
status: mature
paper: arXiv:2503.09516
code: https://github.com/PeterGriffinJin/Search-R1
---

# Search-R1：用 RL 训练 LLM 推理 + 调搜索引擎

> [!info] 论文信息
> - **论文**：[arXiv:2503.09516](https://arxiv.org/abs/2503.09516) —— *Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning*，COLM 2025（v1 2025-03-12，最新 v5 2025-08-05）
> - **作者**：Bowen Jin¹, Hansi Zeng², Zhenrui Yue¹, Jinsung Yoon³, Sercan Ö. Arık³, Dong Wang¹, Hamed Zamani², Jiawei Han¹
> - **单位**：¹UIUC，²UMass Amherst CIIR，³Google Cloud AI Research
> - **代码**：[PeterGriffinJin/Search-R1](https://github.com/PeterGriffinJin/Search-R1)
> - **发布模型**：[PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo](https://huggingface.co/PeterJinGo)

> [!tip] 入门 agentic RL 选这一篇没错
> 推荐阅读路径见 [[agentic-rl-foundations]]。参考实现逐文件 walkthrough 见 [[search-r1-codebase-walkthrough]]。

---

## 摘要（2 分钟读完这一节就够）

**Search-R1 是什么**。把 **DeepSeek-R1-Zero 从纯推理扩展到 tool-use 的标志论文**：4B-7B LLM 用 **outcome-only 规则化 reward**（EM 跟 ground truth 比对）+ 标准 PPO/GRPO 训练，**自主在多轮推理中调用搜索引擎**。训出的模型在 `<think>` 推理之间穿插 `<search>query</search>` 调用和 `<information>retrieved</information>` 注入，最终输出 `<answer>final</answer>`。

**核心思想**。**把搜索引擎当作环境的一部分**，从学生策略采样交错的（LLM-token、retrieved-token）轨迹，**端到端**地用 PPO/GRPO 训练 —— 但加 **retrieved-token loss masking**，让梯度只流过 LLM 生成的 token，搜索引擎注入的文档位置不算 loss。三件事撑起来：

1. **多轮 rollout 协议** —— system prompt + 4 个 tag（`<think>` / `<search>` / `<information>` / `<answer>`）定义 agent loop。生成停在 `</search>` 触发真实检索调用，结果包在 `<information>` tag 里拼回前缀，继续生成。**纯 R1-Zero 路线**：无 SFT 暖身、无手工"好 rollout"，模型靠 RL 信号自学协议
2. **Retrieved-token loss masking** —— policy gradient 和 KL penalty *只*在 LLM-generated 位置上算；retrieved-token 位置 `loss_mask = 0`。没有这个，模型会"学着"模仿检索段落文风，训练崩溃、EM 掉约 9 个绝对点
3. **极简 outcome-only reward** —— 规则化 EM 跟 `ground_truth.target` 对比；提取的 `<answer>` 匹配 = 1，否则 = 0。无 format reward、无神经 RM、无 process reward、无 search-quality reward

**标志数字**。Qwen2.5-7B-base + PPO，7 个 QA 数据集：

| 方法类型 | 方法 | Avg EM（7 数据集） |
| -------- | ---- | -----------------: |
| 无检索 | Direct / CoT | 0.181 / 0.106 |
| 提示式检索 | RAG / IRCoT / Search-o1 | 0.304 / 0.239 / 0.206 |
| 微调 | SFT / Rejection Sampling | 0.207 / 0.348 |
| 纯 RL（无搜索） | R1-base | 0.276 |
| **Search-R1-base (PPO)** | | **0.431** |

相对提升：**比 RAG +42 %**、**比 Rejection Sampling +24 %**、**比纯 RL 无搜索 +56 %**。5 个 backbone 测试（Qwen2.5-3B/7B/14B base+instruct、Llama-3.1/3.2）。作者总结：**Qwen2.5-7B 平均提升 24 %，3B 提升 20 %**。

**为什么重要**。

- **验证了 outcome-only RL 可用于 tool-use**。DeepSeek-R1-Zero 证明纯 RL（无 SFT 数据）能引出复杂推理。Search-R1 把这个扩展到*工具调用*：模型从 1 bit 最终答案 reward 出发，自主学习*何时*搜索、*查什么*、*怎么消化结果*
- **`retrieved-token loss masking` 是这套技巧的首次公开提出**。Polar 的 `prefix_merging`、NeMo Gym 的 response-API agent、之后所有多轮 tool-use RL 系统都做某种"屏蔽环境注入 token"的变体。Search-R1 是第一次系统化提出并通过消融证明必要性
- **参考代码让 agentic RL 对任何有 8×H100 的人都可达**。veRL fork 很小（~600 行 Search-R1 特有代码 + 标准 PPO/GRPO），跑 Qwen2.5-3B/7B 几天就能复现。这是最接近"agentic-RL 入门作业"的资源
- **范式快速泛化**。R1-Searcher、ReSearch、DeepResearcher、ToolRL、ReTool、WebGPT-RL、Computer-Use Agents —— 全是 Search-R1 骨架的变体。到 2026 年中 `<think>/<tool>/<obs>/<answer>` 4-tag 协议已是 agentic-RL 的*事实标准*

---

# 深度部分（往下展开细节）

## 背景：为什么 RAG 和 search-as-tool 都不够

LLM 需要外部知识，Search-R1 之前有两条接搜索引擎的路，各自有自己的毛病：

| 路线 | 做什么 | 失败模式 |
| ---- | ----- | -------- |
| **RAG**（Lewis 2020） | 静态先检索后生成，一次调用，所有检索结果塞进 context，LLM 生成 | 模型永远学不会*何时*搜索、*查什么*、*怎么组合多次查询*。检索与推理脱钩 |
| **Search-as-tool 提示**（ReAct、IRCoT） | 多轮、模型自己决定何时调 | OOD 任务失败；prompting 教不了没见过的工具协议 |
| **Search-as-tool SFT**（Toolformer） | 在带工具调用标注的轨迹上训 | 需要大规模高质量多轮标注；难以扩展；搜索不可微 → 端到端梯度不通 |

第三条路 —— RL —— 当时还没被证明能用在搜索上。Search-R1 论文的实际贡献就是证明第三条路可行，并指出让它工作所需的具体技巧。

### 论文明说的 3 个挑战

引用论文 Intro：

1. **RL 框架与稳定性** —— 怎么把搜索引擎集成进 PPO/GRPO 循环，让训练在引入检索 context 后还能稳？
2. **多轮交错推理与搜索** —— 模型怎么动态决定何时检索、何时推理？
3. **Reward 设计** —— Outcome-only reward 够吗，还是需要 process reward？

答案分别是：retrieved-token loss masking、4-tag 协议、是的（这个规模下 outcome reward 够用，连 format reward 都不要）。

### 跟其他工作对比

论文跟 8 个 baseline 对比（Table 2）。理解 Search-R1 位置最关键的几行：

| Baseline | 代表什么 | Avg EM（Qwen2.5-7B） |
| -------- | -------- | -------------------: |
| Direct Inference | 完全不搜索 | 0.181 |
| CoT | "想后再答"，无搜索 | 0.106 |
| RAG | 静态一次性检索 | 0.304 |
| IRCoT（Trivedi 2022） | 提示式多轮搜索 | 0.239 |
| Search-o1（Li 2025） | 提示式推理 + 搜索，R1 风格格式 | 0.206 |
| SFT | 在 rollout 上训，但无 RL | 0.207 |
| R1（无搜索） | DeepSeek-R1 风格 RL，无检索 | 0.276 |
| **Rejection Sampling**（带搜索） | 在 RL 风格 rollout 上做过滤正确答案的 SFT —— *最强的非-RL baseline* | **0.348** |
| **Search-R1** | 本工作 | **0.431** |

Rejection Sampling 是最干净的反事实："我们做了同样的多轮 rollout，挑出正确的，做 SFT。" Search-R1 比它再多 **+8.3 pp（24 % 相对）** —— 说明 **RL 本身有贡献**，不只是生成 rollout。

## 方法详解

### RL 目标函数

标准 RLHF 目标，唯一改动是策略多了一个搜索引擎 $\mathcal{R}$ 作条件：

$$
\max_{\pi_\theta} \;\mathbb{E}_{x \sim \mathcal{D},\; y \sim \pi_\theta(\cdot \mid x; \mathcal{R})}\!\left[r_\phi(x, y)\right] - \beta \, D_{\text{KL}}\!\left[\pi_\theta(y \mid x; \mathcal{R}) \,\|\, \pi_{\text{ref}}(y \mid x; \mathcal{R})\right]
$$

记号 $\pi_\theta(\cdot \mid x; \mathcal{R}) = \pi_\theta(\cdot \mid x) \otimes \mathcal{R}$ 形式化了"交错的推理-检索" —— rollout $y$ 同时包含 LLM 采样 token 和搜索引擎注入 token。

### 4 个 token tag 协议

```
<think> 推理步骤 </think>
<search> 查询 </search>
<information> 检索到的 passage </information>
<answer> 最终答案 </answer>
```

- `<think>...</think>` —— 模型的推理。System prompt 强制要求（base 模型之前没见过这些 tag）
- `<search>...</search>` —— 模型生成 `</search>` 时系统暂停生成，提取 query 调真实检索 server，结果包在 `<information>...</information>` 里追加到前缀，继续生成
- `<information>...</information>` —— **环境注入**，*不*是模型生成。受 `loss_mask = 0` 约束
- `<answer>...</answer>` —— 模型生成 `</answer>` 时 rollout 结束、算 reward

System prompt 原文（[`scripts/data_process/nq_search.py`](https://github.com/PeterGriffinJin/Search-R1/blob/main/scripts/data_process/nq_search.py)）：

```text
Answer the given question. You must conduct reasoning inside <think> and
</think> first every time you get new information. After reasoning, if you
find you lack some knowledge, you can call a search engine by <search>
query </search> and it will return the top searched results between
<information> and </information>. You can search as many times as your
want. If you find no further external knowledge needed, you can directly
provide the answer inside <answer> and </answer>, without detailed
illustrations. For example, <answer> Beijing </answer>. Question: {question}
```

注意**故意做得很简**：没有 in-context examples、没有偏好的 query 风格、没有鼓励反思推理。**R1-Zero 哲学**：让策略只靠 RL 信号自己想清楚该做什么。

### Rollout 算法

论文 Algorithm 1，等价于 [`generation.py:run_llm_loop`](https://github.com/PeterGriffinJin/Search-R1/blob/main/search_r1/llm_agent/generation.py)：

```text
y ← ∅
for step in range(max_turns):
    自回归生成 token，停在 </search>、</answer> 或 <eos>
    把生成的 token y_b 追加到 y
    if 检测到 </search>:
        从 y_b 提取 query
        retrieved = R(query)        ← 真实搜索调用（HTTP）
        把 "<information>{retrieved}</information>" 追加到 y
    elif 检测到 </answer>:
        return y
    else:
        追加 "My action is not correct. Let me rethink." 到 y

# 强制最终答案（不再 search）
if not done:
    再生成一次，do_search=False
return y
```

action 预算 `max_turns`（论文默认 `2`）限制每个 rollout 的总搜索次数。

### Retrieved-token loss masking —— load-bearing 的关键

PPO 和 GRPO 在 rollout 上算 token 级 loss。Search-R1 里 rollout 同时含 LLM 生成 token 和检索 token。**对检索 token 算 PPO loss 等于训练策略"模仿检索段落"** —— 但那些 token 不在策略的 action space 里！检索内容是环境状态，不是行为。

修复：indicator $I(y_t) \in \{0, 1\}$ 标记 LLM 生成位置，应用到每个 loss 项：

$$
\mathcal{J}_{\text{PPO}}(\theta) = \mathbb{E}_{x, y}\left[\frac{1}{\sum_t I(y_t)} \sum_{t:\, I(y_t)=1} \min\!\left(\frac{\pi_\theta(y_t | \cdot)}{\pi_{\text{old}}(y_t | \cdot)} A_t,\; \text{clip}(\cdot, 1{-}\epsilon, 1{+}\epsilon) A_t\right)\right]
$$

同样的 mask 用在 **KL penalty** 上 —— 检索 token 也不进 KL-to-reference。（如果进了，reference policy 在那些位置上的分布是无意义噪声，会作为 gradient 传播。）

> [!important] 这是 load-bearing 的消融
> Table 4 显示：同模型、同超参，唯一区别 `state_masking=true`。**不开 mask 模型从 0.431 → 0.343 EM** —— 25 % 相对下降。MBPP 上不开 mask 训练的模型实际*比 base 还差*。这个 "trick" 不是优化，是**多轮 RL 训练能成立的前提**。

### Reward 设计

论文坚持极简：

$$
r_\phi(x, y) = \text{EM}(a_{\text{pred}}, a_{\text{gold}})
$$

从 rollout 提取 `<answer>...</answer>`，归一化（小写、去冠词、去标点），跟数据集的 `ground_truth.target`（可接受答案列表）比对。匹配 = 1，否则 = 0。

论文特意*不加*的东西：

- **No format reward** —— R1-Zero 加了 format 惩罚防止 `<think>` 格式错；Search-R1 说"我们的模型已经按格式输出了，format reward 没必要"
- **No process reward / 步级评分** —— 不训 PRM；outcome-only 是设计约束
- **No search-quality reward** —— 不给"你搜对了"信号
- **No search-count penalty** —— 模型想搜几次搜几次
- **No neural reward model** —— 只用规则化 EM，消除 reward hacking 面 + 省训 RM 的成本

500-1500 token trajectory 末尾的 1 bit 信号在这个规模（Qwen 3B-14B）下够用。更大规模或更难任务可能失效，但超出论文范围。

## PPO vs GRPO —— 反直觉的结果

论文两个都测：

**Table 3 结果**（Qwen2.5-7B-base、7 数据集 EM）：

| RL 方法 | Avg EM |
| ------- | -----: |
| GRPO | 0.350 |
| **PPO** | **0.431** |

Qwen2.5-3B-base：GRPO 0.312 vs PPO 0.303 —— GRPO 略好。Qwen2.5-7B-base：PPO 明显更好。base vs instruct 上结果也混杂。

**训练动力学**（Figure 2a）：GRPO 收敛更快（不需要 critic 暖身），但部分 run **训 500 步后崩**。PPO 更慢但稳。

这是 Search-R1 特有的**反直觉发现**。数学 / code 领域 GRPO 常赢（DeepSeek-Math、DeepSeek-R1）。多轮 search-interleaved RL 里，PPO 的 value function 在长 rollout（多次 `<search>` 调用 + retrieved passage = 1000+ token 序列）上帮助 credit assignment；GRPO group-mean baseline 的方差长到足以让训练崩溃。

**Lesson**：agentic RL 的算法选择**任务相关**。DeepSeek-R1 让 GRPO 成默认，但 Search-R1 明确推荐多轮搜索场景用 PPO。

## 训练动力学涌现（Figure 2c/d）

agentic-RL 入门讲稿最常引的图。Qwen2.5-7B-base + PPO，训 200 步：

**Response 长度（Fig 2c）** —— 三阶段：

1. **0-100 步（下降）** —— base 模型一开始啰嗦、塞填充词（~1150 token/rollout）；RL 教它简洁（降到 ~900 token）。Reward 这阶段只微涨
2. **100+ 步（上升）** —— 模型学会**主动调搜索**，retrieved passage 把序列拉长（~900 → ~1100），reward 急涨
3. **后期（稳定）** —— 策略收敛，长度平台

**有效搜索调用次数（Fig 2d）** —— 稳定从 ~1.4 增加到 ~2.0 次/rollout

这是 **agentic-RL "涌现"图**。没人教模型多搜，它从 reward 学到这些 QA 任务上多搜更好。**这是 R1-Zero "aha moment" / 自反思涌现在工具调用上的对应物**。之后每篇 agentic-RL 论文都会展示类似图。

## 标志消融

### Retrieved-token loss masking（Table 4）

| 方法 | NQ | TriviaQA | PopQA | HotpotQA | 2wiki | Musique | Bamboogle | **Avg** |
| ---- | -: | -------: | ----: | -------: | ----: | ------: | --------: | ------: |
| Search-R1 **w/ mask** | 0.480 | 0.638 | 0.457 | 0.433 | 0.382 | 0.196 | 0.432 | **0.431** |
| Search-R1 **w/o mask** | 0.388 | 0.567 | 0.391 | 0.325 | 0.321 | 0.108 | 0.304 | 0.343 |

> [!success] 仅靠 masking 就 +25.6 % 相对
> 0.343 → 0.431。没 mask 训练不稳，模型部分学会模仿检索段落，EM 崩。

### Base vs Instruct（Fig 2b）

| 模型变体 | 初始 reward | 最终 reward |
| -------- | ----------: | ----------: |
| Qwen2.5-7B-base | 低 | 跟 instruct 持平 |
| Qwen2.5-7B-instruct | 高 | 跟 base 持平 |

**Instruct 收敛快，但 base 最后追上**。跟 R1-Zero 显示 RL 能从纯 base 起步一致。

## 优势与不足

两个真实优势：(1) **outcome-only RL 对 tool calling 可行** —— 这是 R1-Zero 的非平凡扩展，开启了 agentic-RL 领域。论文干净地验证了这点。(2) **代码是市面上最干净的 agentic-RL 参考实现**，~600 行论文特有代码加在标准 veRL PPO/GRPO 之上。

可推敲的地方：

- **只测 QA / 单工具**。所有实验是事实型或多跳 QA 对 Wikipedia。没 web-agent、OS-agent、scientific reasoning、code agent、多工具组合。"agentic" framing 略夸 —— 其实是"单工具检索 agent"。后续工作（ToolRL、ReSearch、DeepResearcher）扩到更多工具和任务
- **EM 是噪声 reward**。"Albert Einstein" vs "A. Einstein" —— 都对，但只一个匹配 EM。模型被训成最贴合 EM 的格式，未必等同"最有信息量的答案"。后续工作用 LLM-as-judge 或 F1
- **`max_turns=2` 太少**。大多 rollout 最多 2 次搜索。"多轮"声明在这个深度上很弱。长 horizon agentic RL（10+ 轮）会暴露 Search-R1 没碰的失败模式（context 长度、KV cache 管理、reward 稀疏）
- **静态 Wikipedia 检索**。无时变语料、无噪声 / 对抗 passage、无限速真实搜索引擎。现实更乱
- **搜索引擎成本不在 reward 里**。生产调用要 $/延迟；Search-R1 模型没有节俭激励。训出的模型倾向过度搜索
- **`Bamboogle` 测试集只 125 题**。小 held-out 集统计显著性可疑
- **只测 3-14B**。没 30B+ run。涌现 pattern 在 frontier 规模是否持续未知
- **输出格式比论文承认得更重要**。`extract_solution` 要求*至少 2 个* `<answer>...</answer>` match（一个在 system prompt 例子里，一个在真实响应里）—— 这是脆弱的耦合。如果模型输出有 0 或 1 个 match，reward 自动 0，不管是否真正答对

> [!warning] "valid_search count 上升 = 更好 agent" 的解读部分循环
> Figure 2d 显示搜索调用次数上升，被解读成"模型学会用搜索"。但 reward 函数本身就*偏好*带搜索的序列（在知识重的问题上搜索能解锁正确答案）。所以搜索次数增加部分是"任何收敛到高 reward 的策略都会多搜"的产物。不是反驳 emergence 的故事，但让它没那么神奇

## 这意味着什么

3 条预测，多数 2025-26 已经实现：

1. **4-tag 协议成为 agentic-RL 默认**。✅ 已发生。R1-Searcher、ReSearch、ToolRL 都用 `<think>/<tool_or_search>/<observation_or_information>/<answer>` 模式。2026 年写 agentic-RL 论文默认大家知道这套
2. **Retrieved-token loss masking 变通用**。✅ Polar 的 `prefix_merging` 是最精细版本；veRL 上游了 `state_masking` 作为一等配置项；之后所有多轮 tool-use RL 系统都做某种变体
3. **PPO/GRPO 不再是 agentic RL 的算法前沿**。✅ 有趣的工作搬离 RL 算法本身，去到 rollout 基础设施（ProRL Agent、Polar）、reward 设计（process reward、LLM-as-judge）、工具协议泛化。Search-R1 的贡献是搭**台**，不是在台上获胜

Search-R1 *不是*：frontier paper。它的终身贡献是作为 agentic RL 的**标准入门教学例**。读它学 framing 和消融；用它的代码搭起步；然后到系统论文（Polar）和更难的任务（computer use、长 horizon agent）做真前沿研究。

## 源码与复现

[GitHub: PeterGriffinJin/Search-R1](https://github.com/PeterGriffinJin/Search-R1) —— Apache-2.0，~600 行 Search-R1 特有 Python，加一个 veRL fork。

| 文件 | 角色 |
| ---- | ---- |
| `train_ppo.sh` / `train_grpo.sh` | 顶层配置 |
| `retrieval_launch.sh` | 启 FAISS 检索 server |
| `search_r1/llm_agent/generation.py` | **多轮 rollout 循环**（469 行） |
| `search_r1/search/retrieval_server.py` | FastAPI E5+FAISS dense retriever（392 行） |
| `scripts/data_process/nq_search.py` | NQ 数据集 → parquet + template |
| `verl/utils/reward_score/qa_em.py` | Outcome reward（EM） |
| `verl/trainer/ppo/ray_trainer.py` | 主 PPO/GRPO 循环（867 行） |
| `infer.py` | 参考推理路径（130 行） |

最小复现（Qwen2.5-3B，单机 8×A100，约 2 天）：

```bash
# 1. 一次性建 Wikipedia 2018 FAISS index（几小时）
bash example/build_e5_index.sh

# 2. 启动检索 server（后台）
bash retrieval_launch.sh

# 3. 准备数据
python scripts/data_process/nq_search.py

# 4. 训练
bash train_ppo.sh
```

完整逐文件 walkthrough（包括底下 veRL 的部分）见 [[search-r1-codebase-walkthrough]]。

## 相关阅读

- [[agentic-rl-foundations]] —— 入门 hub；Search-R1 是推荐入门论文
- [[search-r1-codebase-walkthrough]] —— 逐文件代码教程，覆盖 Search-R1 600 行 + 底下 veRL 机器
- [[grpo]] —— Search-R1 用的 RL 算法之一（另一个是 PPO；Search-R1 实验里 PPO 更稳）
- [[ppo-for-llm]] —— PPO-for-LLM 基础参考
- [[on-policy-distillation]] —— 类似问题的 non-RL 替代品；Search-R1 的 rollout 结构（学生采样、每 token 来自环境的稠密信号）结构上类似 OPD 把搜索引擎当退化"老师"
- [[prorl-agent]] —— 首篇"agentic RL 作为基础设施"的论文。ProRL Agent 的 `AgentHandler` ABC 是 Search-R1 手写 rollout loop 的生产泛化
- [[polar]] —— 当前最先进的 rollout 基础。Polar 的 `prefix_merging` 是 Search-R1 retrieved-token loss masking 的精细版
- [[nemo-gym]] —— NVIDIA 的环境 catalog 框架；Search-R1 风格 QA 任务是它 84-benchmark 库存的一部分
- [[tool-use-rl]] —— 更广的工具使用 RL 图景
- [[multi-step-reasoning-rl]] —— 长 horizon 推理的邻近 RL 设置
- [[rl-training-frameworks]] —— Search-R1 所在的 veRL/OpenRLHF/TRL 框架图景

## 参考文献

- 论文：Jin et al., *Search-R1: Training LLMs to Reason and Leverage Search Engines with RL*, COLM 2025. [arXiv:2503.09516](https://arxiv.org/abs/2503.09516)
- DeepSeek-R1 / R1-Zero（直接前作）：[arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
- GRPO：Shao et al., 2024. [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
- PPO：Schulman et al., 2017. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
- veRL：[github.com/volcengine/verl](https://github.com/volcengine/verl) —— Search-R1 fork 的 RL 框架
- Search-o1（baseline）：Li et al., 2025
- IRCoT（baseline）：Trivedi et al., 2022
- ReAct（最早的 search-as-tool 提示）：Yao et al., 2023
- Toolformer：Schick et al., 2023
- 数据集：NQ (Kwiatkowski 2019)、TriviaQA (Joshi 2017)、PopQA (Mallen 2022)、HotpotQA (Yang 2018)、2WikiMultiHopQA (Ho 2020)、Musique (Trivedi 2022b)、Bamboogle (Press 2022)
- E5 retriever：Wang et al., 2022. [arXiv:2212.03533](https://arxiv.org/abs/2212.03533)
