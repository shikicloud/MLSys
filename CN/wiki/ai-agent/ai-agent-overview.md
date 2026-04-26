---
title: "AI 智能体架构与模式"
category: ai-agent
tags: [智能体, react, 规划执行, 思维树, reflexion, 认知架构, 多智能体, 框架, 编码智能体, 记忆, 评估]
created: 2026-04-13
updated: 2026-04-14
status: mature
---

# AI 智能体架构与模式

## 概述

AI 智能体（Agent）是以 LLM 为核心推理引擎，在感知-推理-行动循环中自主运行的系统。与单轮问答的聊天机器人不同，智能体能够：

- **自主规划**：将复杂目标分解为子任务序列
- **工具使用**：调用外部 API、执行代码、读写文件
- **环境感知**：处理来自工具执行结果、用户反馈、系统状态的观察
- **迭代修正**：基于反馈调整策略，从错误中恢复

智能体的核心循环可以表示为：

```
┌─────────────────────────────────────────┐
│              Agent Loop                  │
│                                          │
│   ┌──────────┐    ┌──────────┐          │
│   │ Perceive │───>│  Reason  │          │
│   │ (观察)    │    │  (推理)   │          │
│   └──────────┘    └────┬─────┘          │
│        ^               │                │
│        │               v                │
│   ┌──────────┐    ┌──────────┐          │
│   │ Observe  │<───│   Act    │          │
│   │ (接收结果) │    │  (执行)   │          │
│   └──────────┘    └──────────┘          │
│                                          │
└─────────────────────────────────────────┘
```

截至 2026 年初，推理时计算扩展（Inference-Time Compute Scaling）成为主流趋势，分析师预测推理将在 2030 年占 AI 总计算量的 75%。Gartner 预测到 2026 年底，40% 的企业应用将嵌入智能体。

---

## 核心架构

### ReAct（Reason + Act）

**来源**：Yao et al., ICLR 2023

ReAct 是最基础也是应用最广泛的智能体架构，核心思想是交替进行推理（Thought）和行动（Action）：

```
Thought 1: 我需要查找用户的订单状态
Action 1:  search_orders(user_id="12345")
Observation 1: 找到订单 #ORD-789, 状态: 已发货
Thought 2: 订单已发货，需要获取物流信息
Action 2:  get_tracking(order_id="ORD-789")
Observation 2: 物流单号 SF12345, 预计明天送达
Thought 3: 现在可以回复用户了
Action 3:  respond("您的订单已发货，物流单号 SF12345，预计明天送达")
```

**核心优势**：
- 推理过程透明可追踪
- 自然地将推理与工具调用结合
- 实现简单，适合大多数场景

**扩展变体**：
| 变体 | 核心改进 | 适用场景 |
|------|---------|---------|
| PreAct | 在推理前先预测结果 | 减少不必要的工具调用 |
| RP-ReAct | 分离规划器和执行器 | 复杂多步任务 |
| ReSpAct | 增加用户澄清步骤 | 需要用户交互的场景 |

**Python 实现示意**：

```python
def react_loop(llm, tools, task, max_steps=10):
    """基础 ReAct 循环实现"""
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    history.append({"role": "user", "content": task})

    for step in range(max_steps):
        # 推理 + 行动决策
        response = llm.generate(history, tools=tools)

        if response.is_final_answer:
            return response.content

        # 执行工具调用
        tool_name = response.tool_call.name
        tool_args = response.tool_call.arguments
        observation = tools[tool_name].execute(**tool_args)

        # 将观察结果加入历史
        history.append({"role": "assistant", "content": response.content})
        history.append({"role": "tool", "content": observation})

    return "达到最大步数限制"
```

### Plan-and-Execute（规划-执行）

**核心思想**：将规划与执行解耦为两个独立阶段。

```
┌─────────────────────────────────────────────────┐
│                Plan-and-Execute                   │
│                                                   │
│  ┌─────────┐     ┌──────────────────────────┐    │
│  │ Planner │────>│ Step 1: 查询数据库        │    │
│  │ (规划器) │     │ Step 2: 分析数据          │    │
│  │ GPT-4o  │     │ Step 3: 生成报告          │    │
│  └─────────┘     │ Step 4: 发送邮件          │    │
│       ^          └──────────┬───────────────┘    │
│       │                     │                     │
│       │  replan if needed   v                     │
│       │          ┌──────────────────────────┐    │
│       └──────────│  Executor (执行器)        │    │
│                  │  可以使用更便宜的模型       │    │
│                  └──────────────────────────┘    │
└─────────────────────────────────────────────────┘
```

**优势**：
- 规划和执行可以使用不同模型（规划用强模型，执行用便宜模型）
- 整体成本更低，速度更快
- 计划失败时可以 replan，不需要重新执行已完成的步骤

**适用场景**：多步骤的复杂任务，如数据分析流水线、自动化测试、文档生成。

### 思维树 (Tree of Thoughts, ToT)

**来源**：Yao et al., NeurIPS 2023

```
              [问题]
             /   |   \
           /     |     \
        [思路A] [思路B] [思路C]
        /  \      |      /  \
      [A1] [A2] [B1]  [C1] [C2]
       X    |     X     |    X
            v           v
          [A2展开]    [C1展开]
            |           |
            v           v
          [解答1]     [解答2]
                        ^
                     最优解
```

**核心机制**：
- 同时探索多条推理路径
- 对每条路径进行评估打分
- 支持回溯（backtracking），剪枝低质量路径
- BFS 或 DFS 搜索策略

**扩展**：TouT 增加了不确定性量化。

### LATS（Language Agent Tree Search）

**来源**：ICML 2024

将蒙特卡洛树搜索（MCTS）与 LLM 价值函数结合：

```
                   [初始状态]
                   /    |    \
                [s1]  [s2]  [s3]     ← 扩展 (Expansion)
                 |     |     |
               [v=0.8][v=0.3][v=0.6] ← LLM 评估 (Evaluation)
                 |                    
            UCB 选择最优              ← 选择 (Selection)
               / | \
           [s1a][s1b][s1c]           ← 继续扩展
             |    |    |
           [0.9][0.4][0.7]          ← 再次评估
             |
           [最终解]                   ← 反向传播 (Backpropagation)
```

**特点**：能力最强但计算成本最高，适合需要高质量解的场景（如竞赛编程）。

### Reflexion（反思）

**来源**：Shinn et al., NeurIPS 2023

```
┌──────────────────────────────────────────┐
│              Reflexion 循环               │
│                                          │
│   [任务] ──> [尝试解决] ──> [执行/测试]   │
│                                ^    │    │
│                                │    v    │
│               [存入记忆] <── [反思失败原因] │
│               (自然语言)       (如果失败)   │
│                                          │
│   记忆内容示例：                          │
│   "上次我忘了处理边界情况 n=0"            │
│   "应该先验证输入再进行计算"              │
│                                          │
└──────────────────────────────────────────┘
```

**核心创新**：
- 不需要更新模型权重
- 通过自然语言反思来学习
- 在编码基准上提升 10-20 个百分点

### CoALA（Cognitive Architectures for Language Agents）

将 LLM 智能体映射到经典认知架构（ACT-R、Soar），提出模块化记忆组件框架：

```
┌─────────────────────────────────────────┐
│              CoALA 架构                  │
│                                          │
│  ┌──────────┐  ┌──────────┐  ┌────────┐ │
│  │ 工作记忆  │  │ 情景记忆  │  │ 语义记忆│ │
│  │(Working) │  │(Episodic)│  │(Semantic)│ │
│  └────┬─────┘  └────┬─────┘  └───┬────┘ │
│       │              │            │      │
│       v              v            v      │
│  ┌──────────────────────────────────┐    │
│  │         决策过程 (LLM)            │    │
│  └──────────────┬───────────────────┘    │
│                 │                         │
│       ┌─────────┴─────────┐              │
│       v                   v              │
│  ┌──────────┐       ┌──────────┐         │
│  │ 内部行动  │       │ 外部行动  │         │
│  │(推理/检索)│       │(工具调用) │         │
│  └──────────┘       └──────────┘         │
└─────────────────────────────────────────┘
```

---

## 智能体核心组件

### 1. 感知 (Perception)

智能体从多种来源获取输入：

| 来源类型 | 示例 |
|---------|------|
| 用户输入 | 自然语言指令、代码、图片 |
| 工具返回 | API 响应、数据库查询结果、文件内容 |
| 环境状态 | 系统日志、错误信息、测试结果 |
| 其他智能体 | 多智能体系统中的消息传递 |

### 2. 推理 (Reasoning)

LLM 作为核心推理引擎，负责：
- **任务理解**：解析用户意图
- **规划分解**：将复杂任务拆分为子步骤
- **工具选择**：决定使用哪个工具、传递什么参数
- **结果解读**：理解工具返回的信息
- **错误诊断**：识别问题并制定修复策略

推理模型（如 o1/o3, DeepSeek-R1）通过推理时计算扩展提升复杂推理能力。

### 3. 行动 (Action)

智能体可执行的动作类型：

```python
# 动作类型定义
class ActionType:
    TOOL_CALL = "tool_call"       # 调用外部工具
    CODE_EXEC = "code_execution"  # 执行代码
    RESPOND = "respond"           # 回复用户
    DELEGATE = "delegate"         # 委托给子智能体
    WAIT = "wait"                 # 等待外部事件
    TERMINATE = "terminate"       # 结束任务
```

### 4. 记忆 (Memory)

详见下方 [[#智能体记忆系统]] 章节。

---

## 智能体模式

### 单智能体模式 (Single Agent)

单个 LLM 实例独立完成所有任务。适合：
- 明确定义的任务（代码修复、问答）
- 延迟敏感的场景
- 成本受限的部署

```
用户 ──> [单智能体 + 工具集] ──> 结果
```

### 多智能体模式 (Multi-Agent)

多个专业化智能体协作完成复杂任务。

**常见拓扑**：

```
模式 1: 主从/委托
┌─────────┐
│ 编排智能体 │──> [研究智能体]
│ (Orchestrator)│──> [编码智能体]
│           │──> [测试智能体]
└─────────┘

模式 2: 辩论/对抗
[提出者] <──> [评审者] <──> [总结者]

模式 3: 流水线
[分析] ──> [设计] ──> [实现] ──> [测试] ──> [部署]

模式 4: 民主投票
[Agent A] ──┐
[Agent B] ──┼──> [投票/共识] ──> 最终决策
[Agent C] ──┘
```

**Google A2A 协议**：定义了智能体间通信的标准协议（2025 年 4 月发布），与 [[mcp-protocol|MCP]] 互补 -- MCP 负责智能体-工具通信，A2A 负责智能体间通信。

---

## 框架对比

### 主流框架详细对比（2025-2026）

| 框架 | 开发者 | 核心特点 | 最佳场景 | 语言 |
|------|--------|---------|---------|------|
| **LangGraph** | LangChain | 基于图的工作流定义，状态机 | 复杂可控流程 | Python/JS |
| **CrewAI** | CrewAI | 角色驱动的多智能体，任务委托 | 多角色协作 | Python |
| **AutoGen** | Microsoft | 多智能体对话，代码执行 | 研究/原型 | Python |
| **OpenAI Agents SDK** | OpenAI | 原生工具调用，Handoff 机制 | OpenAI 生态 | Python |
| **Anthropic Agents SDK** | Anthropic | Claude 原生，MCP 集成 | Claude 生态 | Python |
| **Bee Agent** | IBM | 开源，TypeScript 原生 | 企业 TS 项目 | TypeScript |
| **smolagents** | HuggingFace | 轻量级，代码智能体 | 研究/小型项目 | Python |

### LangGraph 示例

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated

class AgentState(TypedDict):
    messages: list
    plan: list[str]
    current_step: int

def planner(state: AgentState) -> AgentState:
    """规划阶段：分解任务"""
    plan = llm.invoke(
        f"将以下任务分解为步骤: {state['messages'][-1]}"
    )
    return {"plan": plan.steps, "current_step": 0}

def executor(state: AgentState) -> AgentState:
    """执行阶段：完成当前步骤"""
    step = state["plan"][state["current_step"]]
    result = llm.invoke(f"执行: {step}", tools=available_tools)
    return {
        "messages": state["messages"] + [result],
        "current_step": state["current_step"] + 1
    }

def should_continue(state: AgentState) -> str:
    if state["current_step"] >= len(state["plan"]):
        return END
    return "executor"

# 构建图
graph = StateGraph(AgentState)
graph.add_node("planner", planner)
graph.add_node("executor", executor)
graph.add_edge("planner", "executor")
graph.add_conditional_edges("executor", should_continue)
graph.set_entry_point("planner")

agent = graph.compile()
```

### OpenAI Agents SDK 示例

```python
from openai import agents

# 定义工具
@agents.tool
def search_web(query: str) -> str:
    """搜索网页获取信息"""
    return web_search(query)

# 定义智能体
research_agent = agents.Agent(
    name="Research Agent",
    instructions="你是一个研究助手，善于查找和总结信息。",
    tools=[search_web],
    model="gpt-4o"
)

# 定义 Handoff（委托）
triage_agent = agents.Agent(
    name="Triage Agent",
    instructions="根据用户请求，决定委托给哪个专业智能体。",
    handoffs=[research_agent, coding_agent, writing_agent]
)

# 运行
result = agents.run(triage_agent, "帮我调研 Transformer 架构的最新进展")
```

---

## 编码智能体 (Coding Agents)

编码智能体是 2025-2026 年最成功的智能体应用类别，直接在代码库上执行任务。

### 主要产品对比

| 产品 | 开发者 | 特点 | 模式 |
|------|--------|------|------|
| **Claude Code** | Anthropic | CLI 工具，直接操作终端和文件系统 | 终端/IDE |
| **Cursor** | Cursor Inc. | AI-native IDE，内置智能体模式 | IDE |
| **Devin** | Cognition | 首个"AI 软件工程师"，全自主 | Web/自主 |
| **GitHub Copilot** | GitHub/MS | 最大用户基础，Agent 模式 2025 发布 | IDE 插件 |
| **Windsurf** | Codeium | AI IDE，Cascade 多步智能体 | IDE |
| **Codex CLI** | OpenAI | 开源 CLI 编码智能体 | 终端 |
| **Augment Code** | Augment | 企业级编码智能体 | IDE 插件 |

### Claude Code 工作流示例

```bash
# Claude Code 的典型工作流
$ claude "修复 auth 模块中的 JWT 过期问题"

# Claude Code 会：
# 1. 搜索代码库找到相关文件
# 2. 阅读代码理解逻辑
# 3. 识别 bug（如缺少过期时间检查）
# 4. 修改代码
# 5. 运行测试验证修复
# 6. 总结所做更改
```

### 编码智能体的核心能力

```
┌────────────────────────────────────────────────┐
│             编码智能体核心能力                    │
│                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │ 代码阅读  │  │ 代码生成  │  │ 代码编辑  │     │
│  │ & 理解    │  │ & 补全    │  │ & 重构    │     │
│  └──────────┘  └──────────┘  └──────────┘     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │ 测试编写  │  │ Bug 诊断  │  │ 终端操作  │     │
│  │ & 运行    │  │ & 修复    │  │ & 命令    │     │
│  └──────────┘  └──────────┘  └──────────┘     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │ 代码搜索  │  │ Git 操作  │  │ 文档生成  │     │
│  │ & 导航    │  │ & PR      │  │ & 注释    │     │
│  └──────────┘  └──────────┘  └──────────┘     │
└────────────────────────────────────────────────┘
```

---

## 智能体记忆系统

智能体的记忆是其"持续学习"和"上下文保持"的关键。

### 记忆类型

```
┌─────────────────────────────────────────────────┐
│              智能体记忆系统                       │
│                                                  │
│  ┌───────────────────────────────────────────┐  │
│  │  短期记忆 (Short-Term / Working Memory)    │  │
│  │  - 当前对话历史                             │  │
│  │  - 上下文窗口内的信息                        │  │
│  │  - 生命周期: 单次会话                        │  │
│  └───────────────────────────────────────────┘  │
│                                                  │
│  ┌───────────────────────────────────────────┐  │
│  │  长期记忆 (Long-Term Memory)               │  │
│  │  - 向量数据库存储的知识                      │  │
│  │  - 用户偏好和历史摘要                        │  │
│  │  - 生命周期: 跨会话持久化                    │  │
│  └───────────────────────────────────────────┘  │
│                                                  │
│  ┌───────────────────────────────────────────┐  │
│  │  情景记忆 (Episodic Memory)                │  │
│  │  - 过去任务的执行经验                        │  │
│  │  - 成功/失败案例及反思                       │  │
│  │  - 类似 Reflexion 的学习记录                 │  │
│  └───────────────────────────────────────────┘  │
│                                                  │
│  ┌───────────────────────────────────────────┐  │
│  │  程序记忆 (Procedural Memory)              │  │
│  │  - 工具使用方法                              │  │
│  │  - 标准操作流程 (SOP)                        │  │
│  │  - 编码规范和最佳实践                        │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

### 实现方式

```python
import chromadb
from datetime import datetime

class AgentMemory:
    """智能体记忆系统示例实现"""

    def __init__(self):
        self.client = chromadb.Client()
        # 长期记忆 - 向量数据库
        self.long_term = self.client.create_collection("long_term")
        # 情景记忆 - 任务执行经验
        self.episodic = self.client.create_collection("episodic")
        # 短期记忆 - 当前会话
        self.working_memory = []

    def store_experience(self, task: str, result: str, success: bool):
        """存储任务执行经验到情景记忆"""
        self.episodic.add(
            documents=[f"Task: {task}\nResult: {result}\nSuccess: {success}"],
            metadatas=[{
                "timestamp": datetime.now().isoformat(),
                "success": success
            }],
            ids=[f"exp_{datetime.now().timestamp()}"]
        )

    def recall_similar(self, query: str, n_results: int = 3) -> list:
        """检索与当前任务相关的历史经验"""
        results = self.episodic.query(
            query_texts=[query],
            n_results=n_results
        )
        return results["documents"]

    def add_to_working(self, message: dict):
        """添加到工作记忆"""
        self.working_memory.append(message)
        # 滑动窗口管理，防止超出上下文长度
        if len(self.working_memory) > 50:
            self._summarize_and_compress()

    def _summarize_and_compress(self):
        """压缩工作记忆：将旧对话摘要化"""
        old_messages = self.working_memory[:30]
        summary = llm.summarize(old_messages)
        self.working_memory = [
            {"role": "system", "content": f"之前的对话摘要: {summary}"}
        ] + self.working_memory[30:]
```

---

## 评估基准

### 主要基准测试

| 基准 | 评估内容 | 最佳成绩 (2026 Q1) | 说明 |
|------|---------|-------------------|------|
| **SWE-bench Verified** | 真实 GitHub issue 修复 | ~65% (Claude 3.5 + scaffold) | 500 个人工验证的 Python 仓库问题 |
| **SWE-bench Full** | 完整版 2294 题 | ~50% | 更大规模更多样化 |
| **WebArena** | 网页浏览与操作 | ~42% | 在真实网站上完成任务 |
| **GAIA** | 通用 AI 助手能力 | ~75% (Level 1) | 需要多步推理+工具使用 |
| **HumanEval** | 代码生成 | ~95%+ | 函数级代码补全 |
| **MATH** | 数学问题求解 | ~95%+ | 竞赛级数学 |
| **TAU-bench** | 真实工具使用 | ~50% (airline domain) | 模拟真实客服场景 |
| **ToolBench** | 工具使用能力 | ~70% | 16,000+ 真实 API |

### SWE-bench 进展趋势

```
SWE-bench Verified 解决率 (%)
│
65%│                                          ●  2026 Q1 (最佳)
60%│                                     ●
55%│                                ●
50%│                           ●
45%│                      ●
40%│                 ●
35%│            ●
30%│       ●
25%│  ●
20%│●
   └──────────────────────────────────────────
    2024    2024     2025    2025    2025   2026
    Q2      Q4       Q1      Q2      Q4     Q1
```

---

## 当前最先进水平 (2025-2026)

### 关键趋势

1. **推理时计算扩展**：推理模型（o1/o3, DeepSeek-R1, Claude 3.5 Extended Thinking）分配可变的测试时计算，在复杂任务上显著提升性能

2. **双范式融合**：符号/经典方法 + 神经/生成方法的结合
   - 传统规划算法 + LLM 推理
   - 形式化验证 + 神经生成

3. **编码智能体主导**：编码是智能体最成功的应用领域
   - 可自动验证（运行测试）
   - 明确的成功标准
   - 丰富的训练数据

4. **多模态智能体**：结合视觉、代码、网页浏览的综合能力

5. **长时间自主运行**：从分钟级到小时级的自主任务执行

### "Underthinking" 问题

推理模型存在"想多了"的问题：
- 在中间步骤达到正确方向后偏离
- 过度探索低质量的推理路径
- 在简单问题上浪费推理时间

---

## 局限性与挑战

### 核心限制

1. **错误传播**：多步骤链中，早期错误会级联放大
2. **成本-精度权衡**：搜索型智能体（ToT, LATS）性能好但成本极高
3. **可靠性**：即使 95% 的单步成功率，20 步后整体成功率仅 ~36%
4. **幻觉风险**：工具调用参数错误、对观察结果的误解
5. **安全性**：智能体执行任意代码和 API 调用带来的安全风险
6. **评估困难**：端到端评估成本高，可复现性差
7. **上下文窗口限制**：长任务积累大量上下文，导致性能退化

### 可靠性计算

```
多步智能体可靠性:

单步成功率  5步    10步    20步    50步
  99%      95.1%  90.4%  81.8%  60.5%
  95%      77.4%  59.9%  35.8%  7.7%
  90%      59.0%  34.9%  12.2%  0.5%
  80%      32.8%  10.7%   1.2%  0.0%

公式: P(全部成功) = p^n
结论: 智能体的每一步都必须非常可靠
```

---

## 参考文献

- Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models," ICLR 2023
- Yao et al., "Tree of Thoughts: Deliberate Problem Solving with Large Language Models," NeurIPS 2023
- Shinn et al., "Reflexion: Language Agents with Verbal Reinforcement Learning," NeurIPS 2023
- Zhou et al., "Language Agent Tree Search," ICML 2024
- Sumers et al., "Cognitive Architectures for Language Agents," 2023
- Xi et al., "The Rise and Potential of Large Language Model Based Agents: A Survey," 2023
- Wang et al., "A Survey on Large Language Model based Autonomous Agents," 2023
- SWE-bench, https://swe-bench.github.io
- Zhou et al., "WebArena: A Realistic Web Environment for Building Autonomous Agents," 2023

---

## 相关页面

- [[tool-use]] -- 智能体如何与工具交互
- [[mcp-protocol]] -- 模型上下文协议标准化工具集成
- [[multi-agent-systems]] -- 多智能体协作
- [[agent-frameworks]] -- 实现框架
- [[agent-memory]] -- 记忆架构
- [[agentic-rl-overview]] -- 通过 RL 训练智能体
- [[compound-ai-systems]] -- 复合 AI 系统
- [[agent-serving-challenges]] -- 智能体服务挑战
- [[multi-turn-optimization]] -- 多轮对话优化
