---
title: "智能体强化学习概述"
category: agentic-rl
tags: [agentic-rl, 多轮rl, 智能体训练, agent-r1, agentrl, rlhf, grpo, ppo]
created: 2026-04-13
updated: 2026-04-14
status: mature
---

# 智能体强化学习概述

> [!abstract]+ TL;DR
> 智能体强化学习（Agentic RL）专门训练 AI 智能体与外部环境/工具交互 —— 在多轮交互中优化**动作序列、工具调用和任务完成**，而不是像传统 [[rlhf-overview|RLHF]] 那样仅优化单轮文本质量。把 LLM 视为 MDP 中的策略：状态 = 对话历史 + 环境状态，动作 = 文本/工具调用，奖励 = 任务完成信号。**范式转变**：从"训练模型说什么"到"训练模型做什么"。驱动 DeepSeek-R1 式推理、WebGPT 式搜索、ReTool 式代码解释器智能体。

## 概述

智能体强化学习（Agentic RL）是一类专门用于训练 AI 智能体与外部环境/工具交互的强化学习方法。与传统的 [[rlhf-overview|RLHF]]（主要优化单轮文本生成质量）不同，Agentic RL 关注的是在多轮交互中优化动作序列、工具调用和任务完成。

**核心定义**：Agentic RL 将 LLM 视为一个在马尔可夫决策过程（MDP）中运行的策略（policy），其中：
- **状态（State）**：对话历史 + 环境状态（文件系统、网页、数据库等）
- **动作（Action）**：文本生成，包括自然语言回复和结构化工具调用
- **转移（Transition）**：环境执行动作并返回观察（observation）
- **奖励（Reward）**：任务完成信号、效率指标、安全约束

这是从"训练模型说什么"到"训练模型做什么"的根本性范式转变。2025-2026 年，随着 DeepSeek-R1、WebGPT、ReTool 等系统的成功，Agentic RL 已成为构建下一代 AI 智能体的核心技术路线。

### 为什么需要 Agentic RL？

传统的 SFT（监督微调）方法通过模仿专家轨迹来训练智能体，但存在根本限制：

1. **分布偏移（Distribution Shift）**：SFT 模型在训练分布外的状态下表现急剧下降
2. **探索不足**：模仿学习无法发现专家未展示的更优策略
3. **错误累积**：多步决策中，每步的小偏差会指数级累积
4. **泛化能力弱**：SFT 智能体难以适应新的环境配置

RL 通过试错学习解决了这些问题——智能体在交互中学习从错误中恢复、探索新策略、并在长时间跨度内优化行为。

## 与传统 RLHF 的区别

### 范式对比

| 维度 | 传统 RLHF | 智能体 RL |
|------|-----------|----------|
| **优化目标** | 文本质量（有用、安全、诚实） | 动作序列与任务完成 |
| **交互轮次** | 单轮（prompt → response） | 多轮（数十到数百轮交互） |
| **状态空间** | 仅文本（prompt + response） | 文本 + 环境状态 |
| **动作空间** | 自然语言文本 | 文本 + 工具调用 + 代码执行 |
| **奖励信号** | 人类偏好（密集、主观） | 任务完成（稀疏、客观可验证） |
| **奖励来源** | 奖励模型（RM） | 环境执行反馈（RLEF） |
| **时间跨度** | 单步 | 长期（episode 可达数百步） |
| **信用分配** | 简单（整个回复） | 复杂（哪个动作导致了成功？） |
| **环境依赖** | 无 | 需要沙箱化执行环境 |
| **安全约束** | 内容安全 | 内容安全 + 行为安全 |

### 流程对比（ASCII 图）

```
传统 RLHF 流程：
┌──────────────────────────────────────────────────┐
│                                                  │
│  User Prompt ──> LLM ──> Response ──> Reward Model ──> Score
│       │                                    │           │
│       │                                    │           │
│       └────────────── Policy Update <──────┘───────────┘
│                                                  │
│  特点：单轮交互，优化文本质量                      │
└──────────────────────────────────────────────────┘

智能体 RL 流程：
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  Task ──> LLM Agent ──┬──> Text Response                     │
│    ^                  │                                      │
│    │                  ├──> Tool Call ──> Environment ──> Obs  │
│    │                  │                      │                │
│    │                  ├──> Code Exec ──> Sandbox ──> Result   │
│    │                  │                      │                │
│    │                  └──> ... (多轮循环) ...                  │
│    │                                         │                │
│    │         Task Completion? ──> Reward ─────┘                │
│    │              │                                           │
│    └──── Policy Update <──────────────────────────────────────┘
│                                                              │
│  特点：多轮交互，环境执行，稀疏奖励                             │
└──────────────────────────────────────────────────────────────┘
```

### 关键差异深入分析

**1. 奖励延迟与稀疏性**

在 RLHF 中，每个回复都能获得奖励模型的即时评分。但在 Agentic RL 中，奖励往往只在任务结束时才出现：

```
RLHF:    Prompt → Response → Reward ✓ (即时)
Agentic: Task → Action1 → Obs1 → Action2 → Obs2 → ... → ActionN → Reward ✓ (延迟)
```

这使得信用分配（credit assignment）成为核心挑战：当一个 50 步的任务最终成功时，模型需要学习是哪些步骤起了关键作用。

**2. 动作空间的混合性**

RLHF 的动作空间是纯文本 token。Agentic RL 的动作空间是混合的：

```
动作空间 = {
    自然语言文本,          # "让我分析这个问题..."
    工具调用(name, args),  # search("quantum computing")
    代码执行(code),        # python: result = solve(equation)
    特殊动作              # <submit>, <give_up>, <ask_user>
}
```

**3. 环境交互的不可逆性**

RLHF 中，生成的文本可以随时丢弃重来。但在 Agentic RL 中，某些动作是不可逆的（删除文件、发送邮件、执行交易），这引入了安全约束的需求。

## 核心范式

### LLM 作为策略（Policy）

在 Agentic RL 框架中，LLM 被形式化为一个策略函数：

$$\pi_\theta(a_t | s_t) = \text{LLM}_\theta(\text{action} | \text{history}_t, \text{env\_state}_t)$$

其中：
- $s_t = (h_t, e_t)$：状态由对话历史 $h_t$ 和环境状态 $e_t$ 组成
- $a_t$：在时间步 $t$ 的动作（文本或工具调用）
- $\theta$：LLM 的参数

**关键特性**：
- 动作是自回归生成的 token 序列
- 一个"动作"可能包含多个 token（例如完整的函数调用）
- 观察（observation）被注入到上下文中作为下一状态的一部分

### 回合（Episode）结构

一个典型的 Agentic RL 回合：

```
Episode = {
    (s_0, a_0, r_0, s_1),   # 初始状态，第一个动作
    (s_1, a_1, r_1, s_2),   # 观察工具输出，第二个动作
    ...
    (s_T, a_T, R_T, done)   # 最终动作，任务级奖励
}

其中：
  s_t = concat(system_prompt, task, history_{0:t-1}, env_obs_t)
  a_t = LLM_θ(s_t)  # 可能是文本或工具调用
  r_t = 0 (中间步骤) 或 R(task_result) (最终步骤)
```

### 奖励设计

奖励设计是 Agentic RL 中最关键也最困难的部分：

**1. 任务完成奖励（Outcome Reward）**
$$R_{\text{outcome}} = \begin{cases} +1 & \text{任务成功完成} \\ 0 & \text{任务失败} \\ -1 & \text{产生有害结果} \end{cases}$$

**2. 效率奖励（Efficiency Reward）**
$$R_{\text{efficiency}} = -\alpha \cdot \text{num\_steps} - \beta \cdot \text{num\_tool\_calls}$$

鼓励用更少的步骤和工具调用完成任务。

**3. 过程奖励（Process Reward）**
$$R_{\text{process}} = \sum_{t=0}^{T} r_t^{\text{progress}}$$

对每一步的进展给予中间奖励（例如：正确选择了工具、生成了有效的参数）。

**4. 安全惩罚（Safety Penalty）**
$$R_{\text{safety}} = -\gamma \cdot \mathbb{1}[\text{unsafe\_action}]$$

对危险动作施加惩罚（例如：尝试删除系统文件、访问未授权的 API）。

**5. 复合奖励**
$$R_{\text{total}} = R_{\text{outcome}} + \lambda_1 R_{\text{efficiency}} + \lambda_2 R_{\text{process}} + \lambda_3 R_{\text{safety}}$$

### 探索挑战

Agentic RL 中的探索问题比传统 RL 更为严峻：

1. **动作空间巨大**：LLM 的动作空间是所有可能 token 序列的组合，本质上是无限的
2. **稀疏奖励**：只有完成任务才能获得正奖励，大量探索轨迹得到零奖励
3. **长时间跨度**：需要在数十到数百步中做出连贯的决策序列
4. **组合爆炸**：工具选择 x 参数选择 x 执行顺序的组合空间呈指数增长

**应对策略**：
- **课程学习（Curriculum Learning）**：从简单任务开始，逐步增加难度
- **SFT 热启动**：用专家轨迹进行初始训练，缩小探索范围
- **密集奖励塑形（Reward Shaping）**：添加中间奖励信号引导探索
- **经验回放**：存储成功轨迹，反复学习
- **分层探索**：先学习子技能，再组合成复杂行为

## 关键研究方向

### 1. 工具使用 RL

[[tool-use-rl|工具使用 RL]] 研究如何通过 RL 训练 LLM 学会何时以及如何调用外部工具。

**核心问题**：
- 何时调用工具 vs. 何时纯推理？
- 如何选择正确的工具和参数？
- 如何从工具输出中提取有用信息？

**代表性工作**：ReTool、Toolformer、RLEF

### 2. 多步推理 RL

[[multi-step-reasoning-rl|多步推理 RL]] 通过 RL 训练 LLM 生成扩展的思维链来解决复杂问题。

**核心问题**：
- 如何激励模型发展出自我验证、回溯等高级推理策略？
- 过程奖励 vs. 结果奖励的权衡？
- 如何防止推理长度膨胀而不增加深度？

**代表性工作**：DeepSeek-R1、OpenAI o1/o3、STaR

### 3. 环境设计

[[environment-design|环境设计]] 决定了智能体能学到什么以及学得多好。

**核心问题**：
- 如何设计高保真且可大规模并行的训练环境？
- 如何平衡环境多样性和真实性？
- 如何设计有效的课程从简单到复杂？

**代表性工作**：SWE-bench、WebArena、OpenReward

### 4. 智能体范式

不同的任务领域催生了不同的智能体范式：

| 范式 | 描述 | 代表系统 |
|------|------|----------|
| **WebAgent** | 在网页上导航、点击、填表 | WebGPT, WebArena, MindAct |
| **CodeAgent** | 编写、调试、执行代码 | SWE-Agent, OpenHands, Devin |
| **SearchAgent** | 搜索、检索、综合信息 | Perplexity, SearchGPT |
| **ToolAgent** | 调用各种 API 和工具 | Gorilla, ToolLLM, API-Bank |
| **OSAgent** | 操作桌面 GUI 和命令行 | OSWorld, CogAgent |
| **MultiAgent** | 多个智能体协作完成任务 | MARTI, AutoGen, CrewAI |

## 代表性工作

### DeepSeek-R1：RL 训练涌现推理能力

**论文**：[DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL](https://arxiv.org/abs/2501.12948)（2025 年 1 月）

**核心发现**：
- 纯 RL 训练（无 SFT）可以让模型涌现出复杂的推理行为
- 使用 [[grpo|GRPO]] 算法，基于 DeepSeek-V3-Base 训练
- 观察到"顿悟时刻"（Aha Moment）：模型自发学会自我反思和错误修正
- 两阶段训练：(1) RL 热启动，(2) RL + SFT 混合训练 + 二次 RL

**训练流程**：
```
Stage 1: Cold Start（冷启动）
  DeepSeek-V3-Base → 少量高质量 CoT 数据 SFT → 初始策略

Stage 2: Reasoning RL（推理 RL）
  初始策略 → GRPO 训练（数学/代码任务）→ 推理能力涌现

Stage 3: Rejection Sampling + SFT
  RL 模型生成推理轨迹 → 过滤 → 混合通用 SFT 数据训练

Stage 4: All-scenario RL（全场景 RL）
  混合模型 → 二次 RL（推理 + 对齐）→ 最终模型
```

**结果**：在 AIME 2024 上 pass@1 达到 79.8%，与 OpenAI o1 相当。

### WebGPT：通过 RL 进行网页浏览

**论文**：[WebGPT: Browser-Assisted Question-Answering with Human Feedback](https://arxiv.org/abs/2112.09332)（OpenAI, 2021）

**核心设计**：
- 赋予 GPT-3 一组网页浏览动作：搜索、点击链接、滚动、引用
- 通过 RL from Human Feedback 训练浏览策略
- 模型学会搜索相关信息、组合多个来源、生成带引用的回答

**训练流程**：
1. 收集人类网页浏览示范 → 行为克隆（BC）
2. 训练奖励模型（基于人类偏好对比）
3. 用 PPO 优化浏览策略

**意义**：WebGPT 是最早的"LLM + 环境交互 + RL"范式的成功案例之一。

### Toolformer：自监督工具学习

**论文**：[Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761)（Meta, 2023）

**核心方法**：
- 不使用 RL，而是通过自监督方式让模型学习在文本中插入工具调用
- 模型自行标注哪些位置插入工具调用能降低困惑度
- 支持计算器、搜索引擎、翻译器、日历等工具

**与 RL 的关系**：
- Toolformer 是 SFT 方法，后续工作（EMNLP 2025）证明同样的能力可通过纯 RL 从头学习
- ReTool 将其思路扩展到 RL 框架，实现了更强的性能

### ReAct：推理 + 行动

**论文**：[ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)（2022）

**核心思想**：
- 交替生成推理步骤（Thought）和行动步骤（Action）
- 推理帮助模型规划、跟踪状态、处理异常
- 行动帮助模型获取外部信息、执行操作

**格式示例**：
```
Thought 1: 我需要搜索苹果公司的最新市值
Action 1: search("Apple Inc market cap 2026")
Observation 1: Apple's market cap is $4.2 trillion as of April 2026...
Thought 2: 找到了信息，让我提取关键数据
Action 2: extract_number("$4.2 trillion")
Observation 2: 4200000000000
Thought 3: 现在我可以回答用户的问题了
Action 3: finish("Apple's market cap is approximately $4.2 trillion.")
```

**意义**：ReAct 定义了现代 LLM 智能体的基本交互模式，几乎所有后续工作都采用了类似的 Thought-Action-Observation 循环。

### RLEF：来自执行反馈的强化学习

**核心思想**：用环境执行的实际结果作为奖励信号，而非人类偏好或学习到的奖励模型。

**优势**：
- 奖励信号客观、可验证（代码是否通过测试、API 是否返回正确结果）
- 无需昂贵的人类标注
- 避免奖励模型的偏差和过拟合

**应用场景**：
```
代码执行 RLEF:  代码 → 执行 → 测试通过/失败 → 奖励
SQL RLEF:       SQL → 执行 → 结果正确/错误 → 奖励
API RLEF:       API 调用 → 执行 → 返回值有效/无效 → 奖励
Web RLEF:       浏览动作 → 执行 → 到达目标页面/未到达 → 奖励
```

## 主要框架（2025-2026）

### AgentGym-RL（ICLR 2026 Oral）

训练长期决策智能体的统一框架，支持网页导航、深度搜索、多步推理等任务。提出了标准化的智能体环境接口和训练流程。

[arXiv:2509.08755](https://arxiv.org/abs/2509.08755)

### AgentRL

多轮、多任务智能体 RL 框架：
- 全异步的生成-训练流水线
- 跨策略采样（cross-policy sampling）：允许不同任务使用不同的采样策略
- 任务优势归一化：解决不同任务奖励尺度不一致的问题

[arXiv:2510.04206](https://arxiv.org/abs/2510.04206)

### Agent-R1

将单轮 RL 扩展到多轮智能体任务的模块化框架：
- 支持 PPO、GRPO、REINFORCE++ 等算法
- 多工具协调机制
- 灵活的环境接口

[arXiv:2511.14460](https://arxiv.org/abs/2511.14460)

### MARTI（ICLR 2026，清华）

多智能体强化训练与推理：
- 异步工具使用
- 多智能体 RL 工作流
- 智能体间的协作与竞争

### ProRL Agent（NVIDIA, 2026 年 3 月）

"Rollout-as-a-Service"架构：
- 将智能体 rollout 编排与训练循环解耦
- 支持大规模并行 rollout
- 灵活的环境后端

## 技术挑战

### 1. 信用分配（Credit Assignment）

在长时间跨度的交互中，确定哪个动作对最终结果起了关键作用是极其困难的：

```
Episode: a_1, a_2, a_3, ..., a_50 → Reward = +1

问题：哪些 a_i 是"好"的？哪些是"无关"的？哪些实际上是"有害但被后续动作补救"的？
```

**现有方法**：
- 轮级奖励（turn-level reward）：由辅助奖励模型为每轮评分
- 过程奖励模型（PRM）：训练模型评估每个中间步骤
- 时间差分学习（TD Learning）：使用价值函数进行逐步信用分配
- GAE（广义优势估计）：平衡偏差和方差的优势估计

### 2. 稀疏奖励

大多数真实任务只在最终才给出奖励：

```
编程任务：50 步交互 → 代码是否通过所有测试？ → +1 或 0
网页导航：20 步浏览 → 是否到达目标页面？ → +1 或 0
```

**应对策略**：
- 奖励塑形（Reward Shaping）：添加启发式中间奖励
- 好奇心驱动探索（Curiosity-Driven Exploration）
- 层级 RL（Hierarchical RL）：分解为子目标
- 密集过程奖励模型

### 3. 安全约束

智能体在真实环境中行动会引入严重的安全风险：

- **不可逆动作**：删除文件、发送消息、执行交易
- **信息泄露**：访问敏感数据、暴露系统信息
- **资源滥用**：无限循环、大量 API 调用、占用系统资源
- **对抗攻击**：恶意环境试图欺骗智能体执行危险操作

**安全机制**：
- 沙箱化执行环境
- 动作白名单/黑名单
- 安全审核模型（Safety Critic）
- 人在环中（Human-in-the-Loop）审批

### 4. 环境保真度

训练环境与真实环境之间的差距（sim-to-real gap）会严重影响策略的迁移：

- 模拟环境可能缺少真实世界的噪声和不确定性
- API 行为可能随时间变化
- 用户行为难以精确模拟

### 5. 可扩展性

训练智能体需要大量的环境交互：

- 每个 rollout 需要实际执行工具（代码运行、API 调用、网页请求）
- 工具执行延迟远高于纯文本生成
- 需要数千到数万个并行环境实例

### 6. 评估困难

智能体能力的评估比文本生成质量评估复杂得多：

- 需要端到端的任务完成评估
- 同一任务可能有多条有效路径
- 部分完成如何评分？
- 效率、安全性等多维度评估

## 代码示例

### 简单的智能体 RL 训练循环（伪代码）

```python
import torch
from typing import List, Dict, Tuple

class AgenticRLTrainer:
    """智能体 RL 训练器：训练 LLM 在环境中完成任务"""

    def __init__(self, policy_model, ref_model, env, reward_fn,
                 lr=1e-6, kl_coeff=0.01, clip_eps=0.2):
        self.policy = policy_model        # 当前策略 (LLM)
        self.ref_model = ref_model        # 参考模型 (用于 KL 约束)
        self.env = env                     # 交互环境
        self.reward_fn = reward_fn         # 奖励函数
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.kl_coeff = kl_coeff
        self.clip_eps = clip_eps

    def collect_rollouts(self, tasks: List[str], n_samples: int = 4
                        ) -> List[Dict]:
        """为每个任务收集多条交互轨迹"""
        rollouts = []
        for task in tasks:
            for _ in range(n_samples):
                trajectory = self._run_episode(task)
                rollouts.append(trajectory)
        return rollouts

    def _run_episode(self, task: str) -> Dict:
        """运行一个完整的交互回合"""
        obs = self.env.reset(task)
        history = [{"role": "system", "content": "You are a helpful agent."},
                   {"role": "user", "content": task}]
        trajectory = {"task": task, "turns": [], "total_reward": 0.0}

        for step in range(self.max_steps):
            # 1. 策略生成动作
            action, log_prob = self.policy.generate(history)

            # 2. 检查是否是工具调用
            if self._is_tool_call(action):
                tool_name, tool_args = self._parse_tool_call(action)
                obs, done, info = self.env.step(tool_name, tool_args)
                history.append({"role": "assistant", "content": action})
                history.append({"role": "tool", "content": obs})
            else:
                # 纯文本回复
                obs, done, info = action, True, {}
                history.append({"role": "assistant", "content": action})

            # 3. 记录轨迹
            trajectory["turns"].append({
                "action": action,
                "log_prob": log_prob,
                "observation": obs,
            })

            if done:
                break

        # 4. 计算回合奖励
        trajectory["total_reward"] = self.reward_fn(task, trajectory)
        return trajectory

    def compute_advantages(self, rollouts: List[Dict]) -> List[Dict]:
        """计算每条轨迹的优势值（GRPO 风格：组内归一化）"""
        # 按任务分组
        task_groups = {}
        for r in rollouts:
            task_groups.setdefault(r["task"], []).append(r)

        for task, group in task_groups.items():
            rewards = [r["total_reward"] for r in group]
            mean_r = sum(rewards) / len(rewards)
            std_r = (sum((r - mean_r)**2 for r in rewards) / len(rewards))**0.5
            std_r = max(std_r, 1e-8)

            for r in group:
                # 组内归一化优势
                r["advantage"] = (r["total_reward"] - mean_r) / std_r

        return rollouts

    def update_policy(self, rollouts: List[Dict]):
        """使用 PPO/GRPO 风格的策略更新"""
        rollouts = self.compute_advantages(rollouts)

        total_loss = 0
        for rollout in rollouts:
            advantage = rollout["advantage"]

            for turn in rollout["turns"]:
                # 当前策略的 log prob
                curr_log_prob = self.policy.log_prob(turn["action"])
                old_log_prob = turn["log_prob"]

                # 重要性采样比率
                ratio = torch.exp(curr_log_prob - old_log_prob)

                # PPO clip
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio,
                                    1 - self.clip_eps,
                                    1 + self.clip_eps) * advantage
                policy_loss = -torch.min(surr1, surr2)

                # KL 散度惩罚
                ref_log_prob = self.ref_model.log_prob(turn["action"])
                kl_penalty = curr_log_prob - ref_log_prob

                total_loss += policy_loss + self.kl_coeff * kl_penalty

        # 梯度更新
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

    def train(self, task_dataset, n_epochs=100, batch_size=32):
        """主训练循环"""
        for epoch in range(n_epochs):
            # 1. 采样任务
            tasks = task_dataset.sample(batch_size)

            # 2. 收集 rollouts
            rollouts = self.collect_rollouts(tasks)

            # 3. 策略更新
            self.update_policy(rollouts)

            # 4. 评估
            if epoch % 10 == 0:
                eval_score = self.evaluate(task_dataset.eval_set)
                print(f"Epoch {epoch}: eval_score = {eval_score:.3f}")
```

### 奖励函数示例

```python
def agentic_reward(task: str, trajectory: Dict) -> float:
    """复合奖励函数"""
    # 1. 任务完成奖励
    task_reward = 1.0 if trajectory["task_completed"] else 0.0

    # 2. 效率奖励（步数惩罚）
    num_steps = len(trajectory["turns"])
    efficiency_reward = -0.01 * num_steps

    # 3. 工具使用正确性
    tool_calls = [t for t in trajectory["turns"] if t.get("is_tool_call")]
    valid_calls = sum(1 for t in tool_calls if t.get("tool_success"))
    tool_reward = valid_calls / max(len(tool_calls), 1)

    # 4. 安全惩罚
    safety_violations = sum(1 for t in trajectory["turns"]
                           if t.get("safety_violation"))
    safety_penalty = -1.0 * safety_violations

    # 复合奖励
    return (task_reward
            + 0.1 * efficiency_reward
            + 0.2 * tool_reward
            + safety_penalty)
```

## 参考文献

### 核心论文

- DeepSeek-AI (2025). [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948). arXiv:2501.12948.
- Nakano et al. (2021). [WebGPT: Browser-Assisted Question-Answering with Human Feedback](https://arxiv.org/abs/2112.09332). arXiv:2112.09332.
- Schick et al. (2023). [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761). arXiv:2302.04761.
- Yao et al. (2022). [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629). arXiv:2210.03629.

### 框架与系统

- Xi et al. (2025). [AgentGym: Evolving Large Language Model-based Agents across Diverse Environments](https://arxiv.org/abs/2509.08755). arXiv:2509.08755. (ICLR 2026 Oral)
- AgentRL Team (2025). [AgentRL: Training Language Model Agents with Reinforcement Learning](https://arxiv.org/abs/2510.04206). arXiv:2510.04206.
- Agent-R1 Team (2025). [Agent-R1: Training Powerful LLM Agents with End-to-End Reinforcement Learning](https://arxiv.org/abs/2511.14460). arXiv:2511.14460.

### 综述与指南

- NeurIPS 2025. A Practitioner's Guide to Multi-turn Agentic RL.
- HuggingFace (2026). [When LLMs Grow Hands and Feet: Agentic RL Systems](https://huggingface.co/blog/AmberLJC/agentic-rl-systems).

## 相关页面

- [[tool-use-rl]] -- 工具使用与 API 调用的 RL
- [[multi-step-reasoning-rl]] -- 多步推理的强化学习
- [[environment-design]] -- 智能体 RL 的环境设计
- [[rl-training-frameworks]] -- 训练基础设施（veRL, OpenRLHF 等）
- [[ai-agent-overview]] -- AI 智能体架构
- [[rlhf-overview]] -- RLHF 概述（对比）
- [[grpo]] -- GRPO 算法（DeepSeek-R1 使用）
- [[ppo-for-llm]] -- PPO 在 LLM 训练中的应用
- [[reward-modeling]] -- 奖励建模（ORM/PRM）
