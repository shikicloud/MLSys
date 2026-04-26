---
title: "工具使用与 API 调用的强化学习"
category: agentic-rl
tags: [工具使用, rl, retool, 代码解释器, api调用, toolformer, gorilla, function-calling]
created: 2026-04-13
updated: 2026-04-14
status: mature
---

# 工具使用与 API 调用的强化学习

## 概述

工具使用 RL（Tool-Use RL）是 [[agentic-rl-overview|智能体 RL]] 的核心研究方向之一，专注于通过强化学习训练 LLM 学会**何时**（when）、**调用哪个**（which）工具、**如何**（how）格式化调用参数、以及**如何解读**（interpret）工具返回结果。

传统方法通过监督微调（SFT）在专家标注的工具调用示例上训练模型，但这种方法存在明显局限：
- 无法探索专家未覆盖的工具使用策略
- 难以适应新工具或 API 变更
- 错误积累导致多步工具链失败

RL 方法通过试错学习克服了这些限制——模型在交互中学习何时使用工具更高效、何时纯推理更优，并能够从工具执行的实际反馈中持续改进。

### 工具使用的价值

为什么让 LLM 学会使用工具如此重要？

1. **克服固有限制**：LLM 的数学计算、实时信息获取、代码执行等能力天生有限
2. **扩展能力边界**：通过工具访问，LLM 可以操作数据库、控制软件、搜索互联网
3. **提高可靠性**：工具提供确定性结果（计算器不会算错），减少幻觉
4. **实现落地**：真实世界的任务几乎都需要与外部系统交互

## 工具使用的形式化

### MDP 建模

工具使用可以被形式化为一个马尔可夫决策过程（MDP）：

$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \gamma)$$

**状态空间** $\mathcal{S}$：
$$s_t = (\text{task}, h_{1:t-1}, \text{tool\_results}_{1:t-1})$$

状态由任务描述、历史交互记录和之前的工具执行结果组成。

**动作空间** $\mathcal{A}$：
$$a_t \in \begin{cases} \mathcal{A}_{\text{text}} & \text{生成自然语言文本} \\ \mathcal{A}_{\text{tool}} = \{(\text{tool\_name}, \text{args})\} & \text{调用工具} \\ \mathcal{A}_{\text{special}} = \{\text{submit, give\_up}\} & \text{特殊动作} \end{cases}$$

动作空间是混合的：模型在每一步可以选择生成纯文本（例如推理、总结）或调用工具。

**转移函数** $\mathcal{T}$：
$$s_{t+1} = \begin{cases} s_t \oplus a_t & \text{if } a_t \in \mathcal{A}_{\text{text}} \\ s_t \oplus a_t \oplus \text{env}(a_t) & \text{if } a_t \in \mathcal{A}_{\text{tool}} \end{cases}$$

当动作是文本时，状态简单追加。当动作是工具调用时，环境执行工具并返回结果，结果被追加到状态中。

**奖励函数** $\mathcal{R}$：通常是稀疏的任务完成奖励加上中间过程奖励。

### ASCII 图：工具使用 MDP

```
工具使用 MDP 交互流程：

     ┌─────────────────────────────────────────────────────────┐
     │                                                         │
     │   State s_t                    LLM Policy π_θ           │
     │   ┌─────────────┐            ┌─────────────────┐        │
     │   │ Task        │            │                 │        │
     │   │ History     │───────────>│  Decision:      │        │
     │   │ Tool Results│            │  Text or Tool?  │        │
     │   └─────────────┘            └────────┬────────┘        │
     │                                       │                 │
     │                          ┌────────────┼────────────┐    │
     │                          │            │            │    │
     │                          v            v            v    │
     │                     ┌────────┐  ┌──────────┐  ┌──────┐ │
     │                     │ Text   │  │ Tool Call│  │Submit│ │
     │                     │ Output │  │(name,arg)│  │Answer│ │
     │                     └───┬────┘  └────┬─────┘  └──┬───┘ │
     │                         │            │           │     │
     │                         │            v           │     │
     │                         │    ┌──────────────┐    │     │
     │                         │    │  Environment │    │     │
     │                         │    │  ┌─────────┐ │    │     │
     │                         │    │  │ Execute  │ │    │     │
     │                         │    │  │ Tool     │ │    │     │
     │                         │    │  └────┬────┘ │    │     │
     │                         │    │       │      │    │     │
     │                         │    │  Observation │    │     │
     │                         │    └──────┬───────┘    │     │
     │                         │           │            │     │
     │                         v           v            v     │
     │                     ┌──────────────────────────────┐   │
     │                     │   s_{t+1} = s_t + action     │   │
     │                     │            + observation      │   │
     │                     └──────────────┬───────────────┘   │
     │                                    │                   │
     │                         ┌──────────┴──────────┐        │
     │                         │  Done?              │        │
     │                         │  No → next turn     │        │
     │                         │  Yes → compute R    │        │
     │                         └─────────────────────┘        │
     │                                                         │
     └─────────────────────────────────────────────────────────┘
```

### 回合（Episode）示例

```
Task: "计算 2^100 + 3^50 的结果并判断是否为质数"

Turn 1:
  State: [task description]
  Action: <think>这个计算量很大，我应该使用 Python 解释器</think>
          <tool>python
          result = 2**100 + 3**50
          print(f"Result: {result}")
          </tool>
  Observation: Result: 1267650600228229401496703205975

Turn 2:
  State: [task + turn1 + observation]
  Action: <tool>python
          from sympy import isprime
          n = 1267650600228229401496703205975
          print(f"Is prime: {isprime(n)}")
          </tool>
  Observation: Is prime: False

Turn 3:
  State: [task + turn1 + turn2 + observations]
  Action: 2^100 + 3^50 = 1,267,650,600,228,229,401,496,703,205,975。
          这个数不是质数。<submit>
  Reward: +1 (正确答案)
```

## 训练方法

### 1. Toolformer 方法：自监督工具标注

**论文**：[Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761)（Meta, 2023）

**核心思想**：让模型自行决定在文本的哪些位置插入工具调用能够降低后续 token 的困惑度（perplexity）。

**训练流程**：

```
Step 1: 候选位置采样
  对文本中每个位置，采样可能的工具调用 c_i
  例如：在 "The population is [calculator(2.5 * 10^9)] large" 中的 [calculator] 位置

Step 2: 执行并过滤
  实际执行工具调用，获得结果 r_i
  构造带结果的文本：x_i = "... [c_i → r_i] ..."

Step 3: 困惑度比较
  如果 L(x_with_tool) < L(x_without_tool) - τ:
    保留这个工具调用标注
  否则：
    丢弃

Step 4: 微调
  用过滤后的标注数据对模型进行 SFT
```

**支持的工具**：计算器、问答系统、维基百科搜索、机器翻译、日历

**局限性**：
- 本质上是 SFT 而非 RL，无法从试错中学习
- 工具调用模式固定，不适应动态变化
- 无法处理多步工具链

### 2. RLEF：执行反馈作为奖励

**核心思想**：用工具执行的实际结果作为 RL 的奖励信号。

```
传统 RLHF:  Action → 奖励模型评分 → 可能不准确
RLEF:       Action → 实际执行 → 客观结果 → 精确奖励
```

**RLEF 的奖励信号类型**：

| 奖励类型 | 描述 | 示例 |
|----------|------|------|
| 二值奖励 | 成功/失败 | 代码测试通过/失败 |
| 连续奖励 | 部分正确程度 | SQL 查询返回 80% 正确行 |
| 多维奖励 | 多个评估维度 | 正确性 + 效率 + 安全性 |
| 差分奖励 | 与基准比较 | 比朴素方法快 3 倍 |

**优势**：
- 零人工标注成本
- 奖励信号完全客观、可验证
- 自然适应工具 API 的变化

**挑战**：
- 需要可靠的沙箱执行环境
- 某些工具调用有副作用（不可逆）
- 执行延迟增加 rollout 时间

### 3. 过程奖励用于工具选择质量

对工具使用的每个决策点给予中间奖励，而不仅仅是最终任务完成奖励：

```python
def tool_process_reward(turn):
    reward = 0.0

    # 1. 工具选择是否合理？
    if turn.chose_correct_tool:
        reward += 0.3
    elif turn.chose_wrong_tool:
        reward -= 0.2

    # 2. 参数格式是否正确？
    if turn.is_tool_call and turn.valid_syntax:
        reward += 0.1
    elif turn.is_tool_call and not turn.valid_syntax:
        reward -= 0.3

    # 3. 是否应该使用工具？
    if turn.used_tool_unnecessarily:
        reward -= 0.1  # 能推理解决却用了工具
    elif turn.should_have_used_tool:
        reward -= 0.2  # 应该用工具却选择推理

    return reward
```

### 4. RL 微调用于函数调用（Function Calling）

现代 LLM API 的函数调用（function calling）功能通常通过 SFT 训练，但 RL 可以进一步提升：

**SFT 阶段**：在大量函数调用示例上训练基本能力

```json
{
  "function": "get_weather",
  "arguments": {"city": "Beijing", "unit": "celsius"}
}
```

**RL 阶段**：通过实际执行反馈优化

```
Reward signals:
  +1.0: 函数调用成功且结果正确
  +0.5: 函数调用成功但非最优选择
  -0.5: 函数调用语法错误
  -1.0: 调用了不存在的函数
  -0.2: 不必要的函数调用（浪费 token/时间）
```

### 5. ReTool：工具增强 RL 的里程碑

**论文**：[ReTool: Reinforcement Learning for Strategic Tool Use in LLMs](https://arxiv.org/abs/2504.11536)（2025）

ReTool 是将 RL 用于工具使用训练的里程碑式工作：

**核心创新**：将"调用 Python 解释器还是继续纯文本推理"建模为一个显式的 RL 决策问题。

**两阶段训练**：

```
Stage 1: 冷启动 SFT
  - 收集少量高质量的工具使用示例
  - 标注何时应该调用代码解释器
  - SFT 训练给模型基本的工具调用能力

Stage 2: 工具增强 RL
  - 使用 veRL + PPO
  - 每次 rollout 中，模型可以选择：
    (a) 继续文本推理：<think>...</think>
    (b) 调用 Python：<tool>python\n...\n</tool>
  - 实时执行代码，将结果注入上下文
  - 根据最终答案正确性计算奖励
```

**关键结果**：
- ReTool-32B 在 AIME 上达到 67%（400 步）vs. 纯文本 RL 40%（1080 步）
- 最终 72.5%，超越 o1-preview 27.9%
- 涌现出代码自我修正行为：模型运行代码发现错误后，会自动修改并重新执行

**涌现行为示例**：
```
Turn 1: <tool>python
        def solve():
            # 初始尝试
            return naive_solution()
        print(solve())
        </tool>
Obs: Error: overflow

Turn 2: <think>代码溢出了，我需要用大数处理</think>
        <tool>python
        from decimal import Decimal
        def solve():
            # 修正后的方案
            return improved_solution()
        print(solve())
        </tool>
Obs: 42

Turn 3: 答案是 42。
```

这种"发现错误 → 反思 → 修正"的行为是纯 RL 训练涌现的，未在训练数据中显式演示。

## 奖励设计

奖励设计是工具使用 RL 的关键挑战。一个好的奖励函数需要平衡多个目标：

### 1. 任务完成奖励（Binary Task Reward）

$$R_{\text{task}} = \begin{cases} +1 & \text{最终答案正确} \\ 0 & \text{最终答案错误} \end{cases}$$

最简单但也最稀疏的奖励。对于复杂任务，大量轨迹可能都得到 0 奖励，导致学习效率极低。

### 2. 工具效率奖励（Tool Efficiency Reward）

$$R_{\text{eff}} = -\alpha \cdot N_{\text{tool\_calls}} - \beta \cdot N_{\text{total\_steps}}$$

鼓励模型用更少的工具调用和步骤完成任务。参数 $\alpha$ 和 $\beta$ 控制惩罚力度：

```
好的行为：2 次工具调用解决问题 → 效率奖励高
差的行为：10 次冗余工具调用 → 效率奖励低
```

### 3. 工具正确性奖励（Tool Correctness Reward）

$$R_{\text{correct}} = \frac{1}{N_{\text{calls}}} \sum_{i=1}^{N_{\text{calls}}} \mathbb{1}[\text{call}_i \text{ is valid}]$$

评估工具调用的质量：参数格式是否正确、工具名是否存在、调用是否成功执行。

### 4. 复合奖励函数

```python
def composite_tool_reward(trajectory):
    """实际使用的复合奖励函数"""

    # 任务完成（主要信号）
    task_score = 1.0 if check_answer(trajectory) else 0.0

    # 工具效率
    n_tools = count_tool_calls(trajectory)
    n_steps = len(trajectory.turns)
    efficiency = -0.01 * n_tools - 0.005 * n_steps

    # 工具正确性
    valid_calls = count_valid_calls(trajectory)
    total_calls = max(count_tool_calls(trajectory), 1)
    correctness = 0.2 * (valid_calls / total_calls)

    # 格式规范性（避免格式错误的工具调用）
    format_violations = count_format_errors(trajectory)
    format_penalty = -0.1 * format_violations

    # 安全性（避免危险操作）
    safety_violations = count_unsafe_actions(trajectory)
    safety_penalty = -1.0 * safety_violations

    return (task_score
            + efficiency
            + correctness
            + format_penalty
            + safety_penalty)
```

### 奖励设计的陷阱

| 陷阱 | 描述 | 解决方案 |
|------|------|----------|
| **奖励黑客** | 模型找到获得高奖励但不完成任务的捷径 | 多维度奖励 + 对抗性测试 |
| **效率过惩罚** | 模型为避免工具调用惩罚而不用工具 | 仅在任务成功时计算效率奖励 |
| **格式过拟合** | 模型学会完美格式但内容无意义 | 增加内容正确性权重 |
| **稀疏奖励困境** | 奖励太稀疏导致学不到东西 | 添加过程奖励 + 课程学习 |

## 代表性系统

### WebGPT

**出处**：OpenAI（2021）

WebGPT 是最早将 RL 用于工具使用的大规模系统之一：
- 赋予 GPT-3 网页浏览工具集：搜索、点击、滚动、回退、引用
- 通过 RLHF 训练浏览策略
- 最终模型能搜索信息、综合多个来源、生成带引用的回答

**动作空间**：
```
Actions = {
    Search(query),      # 搜索
    Click(element_id),  # 点击链接/按钮
    Scroll(direction),  # 上下滚动
    Quote(text),        # 引用当前页面文本
    Back(),             # 返回上一页
    Submit(answer)      # 提交最终答案
}
```

### Gorilla

**出处**：UC Berkeley（NeurIPS 2024）

Gorilla 专注于正确生成 API 调用：
- 训练数据来自 API 文档（Torch Hub, TensorFlow Hub, HuggingFace）
- 模型学会根据自然语言描述生成准确的 API 调用
- 引入 AST 准确率评估指标

**贡献**：证明了 LLM 可以学会精确的 API 调用语法，而不仅仅是近似。

### ToolLLM

**核心设计**：
- 构建了包含 16,000+ 真实世界 API 的大规模数据集
- 设计了 DFSDT（深度优先搜索决策树）推理策略
- 支持单工具和多工具场景

**动作空间**：
```
Single-tool: search_weather(city="Beijing")
Multi-tool:  search_flights(from="Beijing", to="Tokyo", date="2026-05-01")
             → book_hotel(city="Tokyo", checkin="2026-05-01", nights=3)
             → get_directions(from="Narita Airport", to="Hotel")
```

### API-Bank

**核心设计**：
- 314 个工具 API 的基准测试集
- 三个层级的评估：API 检索、API 调用、API 融合
- 用于评估 LLM 的工具使用能力

## 代码示例

### 工具使用 RL 训练伪代码

```python
class ToolUseRLTrainer:
    """工具使用 RL 训练器（简化版）"""

    def __init__(self, policy, ref_model, env):
        self.policy = policy
        self.ref_model = ref_model
        self.env = env

    def collect_rollout(self, task, max_turns=20):
        """收集一条完整的工具使用轨迹"""
        obs = self.env.reset(task)
        messages = [{"role": "system", "content": TOOL_PROMPT},
                    {"role": "user", "content": obs}]
        trajectory = {"task": task, "turns": []}

        for _ in range(max_turns):
            action, log_probs = self.policy.generate(messages)
            obs, done, info = self.env.step(action)
            trajectory["turns"].append(
                {"action": action, "log_probs": log_probs, "info": info})
            messages.append({"role": "assistant", "content": action})
            if obs:
                messages.append({"role": "tool", "content": obs})
            if done:
                break
        return trajectory

    def compute_reward(self, task, trajectory):
        """复合奖励：任务完成 + 工具质量 + 效率"""
        task_reward = 1.0 if check_answer(trajectory) else 0.0
        tool_turns = [t for t in trajectory["turns"] if is_tool_call(t["action"])]
        tool_quality = (sum(t["info"].get("tool_success", 0) for t in tool_turns)
                       / max(len(tool_turns), 1))
        efficiency = max(0, 1.0 - 0.02 * len(trajectory["turns"]))
        return 1.0 * task_reward + 0.2 * tool_quality + 0.1 * efficiency

    def grpo_update(self, task_batch, n_samples=8):
        """GRPO 策略更新"""
        all_groups = []
        for task in task_batch:
            group = []
            for _ in range(n_samples):
                traj = self.collect_rollout(task)
                traj["reward"] = self.compute_reward(task, traj)
                group.append(traj)
            # 组内归一化优势
            rewards = [r["reward"] for r in group]
            mean_r, std_r = mean(rewards), max(std(rewards), 1e-8)
            for r in group:
                r["advantage"] = (r["reward"] - mean_r) / std_r
            all_groups.append(group)

        # 策略梯度更新（Clipped PPO + KL 惩罚）
        loss = 0
        for group in all_groups:
            for rollout in group:
                for turn in rollout["turns"]:
                    ratio = exp(self.policy.log_prob(turn["action"]) - turn["log_probs"])
                    adv = rollout["advantage"]
                    clipped = clamp(ratio, 0.8, 1.2) * adv
                    kl = self.policy.log_prob(turn["action"]) - self.ref_model.log_prob(turn["action"])
                    loss += -min(ratio * adv, clipped) + 0.01 * kl
        loss.backward()
        self.optimizer.step()
```

## 挑战

### 1. 工具调用幻觉（Hallucinated Tool Calls）

模型可能生成不存在的工具名、错误的参数格式或虚构的工具输出：

```
常见幻觉类型：
- 调用不存在的工具：<tool>quantum_solver\n...</tool>
- 参数类型错误：search(query=42) # 期望字符串，给了数字
- 假装收到了工具输出：（不等待实际执行就编造结果）
- 使用过时的 API 签名
```

**应对策略**：
- 在系统 prompt 中明确列出可用工具和参数格式
- 对无效工具调用给予负奖励
- 使用约束解码确保工具名在有效集合中
- 训练格式检查器在执行前验证调用

### 2. 动作空间爆炸

工具使用的有效动作空间远大于纯文本生成：

$$|\mathcal{A}_{\text{effective}}| = |\mathcal{A}_{\text{text}}| + \sum_{t \in \text{Tools}} |\text{Args}(t)|$$

对于拥有数百个 API 端点的系统，探索空间变得极其庞大。

**应对策略**：
- 分层动作空间：先选工具类别，再选具体工具，再填参数
- 检索增强：根据任务检索相关工具子集
- 课程学习：从少量工具开始，逐步增加

### 3. 信用分配

在多步工具调用链中，难以确定哪个调用是成功/失败的关键：

```
Task: 分析股票数据并给出建议
  Turn 1: search("AAPL stock price") → ✓
  Turn 2: python(parse_data(results)) → ✓
  Turn 3: search("AAPL earnings report") → ✓ (但搜索词不够精确)
  Turn 4: python(analyze(data)) → Bug in code → ✗
  Turn 5: python(fix_and_analyze(data)) → ✓
  Final: 给出了错误的建议 → Reward = 0

  问题：哪一步最应该为失败负责？Turn 3 的不精确搜索？Turn 4 的 bug？
```

### 4. 工具执行延迟

工具调用需要实际执行时间，显著增加 rollout 成本：

| 动作类型 | 典型延迟 |
|----------|----------|
| 文本生成 | ~100ms/token |
| 代码执行 | 100ms - 30s |
| API 调用 | 200ms - 5s |
| 网页搜索 | 1s - 10s |
| 数据库查询 | 50ms - 5s |

在大规模 RL 训练中（数千条 rollout），工具执行延迟成为训练瓶颈。

**应对策略**：
- 异步并行执行多条 rollout
- 工具结果缓存（相同调用返回缓存结果）
- 沙箱池预热（预先创建沙箱实例）

### 5. 安全性

代码执行和 API 调用引入严重的安全风险：

- **代码注入**：模型可能生成恶意代码
- **资源耗尽**：无限循环、内存溢出
- **数据泄露**：读取敏感文件、环境变量
- **网络攻击**：通过 API 调用发起请求

**安全机制**：
```python
class SafeToolExecutor:
    """安全工具执行器"""

    FORBIDDEN_PATTERNS = [
        r"import\s+os",
        r"import\s+subprocess",
        r"open\(.*/etc/",
        r"requests\.delete",
        r"rm\s+-rf",
    ]

    def execute(self, tool_name, args, timeout=30):
        # 1. 安全检查
        for pattern in self.FORBIDDEN_PATTERNS:
            if re.search(pattern, str(args)):
                raise SecurityError(f"Forbidden pattern: {pattern}")

        # 2. 沙箱执行
        with Sandbox(
            network=False,       # 禁用网络
            filesystem="readonly", # 只读文件系统
            memory_limit="512MB",  # 内存限制
            cpu_time=timeout       # CPU 时间限制
        ) as sandbox:
            return sandbox.run(self.tools[tool_name], args)
```

## 参考文献

### 核心论文

- Schick et al. (2023). [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761). arXiv:2302.04761.
- Feng et al. (2025). [ReTool: Reinforcement Learning for Strategic Tool Use in LLMs](https://arxiv.org/abs/2504.11536). arXiv:2504.11536.
- Nakano et al. (2021). [WebGPT: Browser-Assisted Question-Answering with Human Feedback](https://arxiv.org/abs/2112.09332). arXiv:2112.09332.
- Patil et al. (2024). [Gorilla: Large Language Model Connected with Massive APIs](https://arxiv.org/abs/2305.15334). NeurIPS 2024.
- Qin et al. (2023). [ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs](https://arxiv.org/abs/2307.16789). arXiv:2307.16789.
- Li et al. (2023). [API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs](https://arxiv.org/abs/2304.08244). arXiv:2304.08244.

### 综述与基准

- Qu et al. (2025). [Tool Learning with Large Language Models: A Survey](https://arxiv.org/abs/2405.17935). arXiv:2405.17935.

## 相关页面

- [[agentic-rl-overview]] -- 智能体 RL 全景
- [[environment-design]] -- 沙箱与执行环境设计
- [[tool-use]] -- 从智能体架构角度看工具使用
- [[multi-step-reasoning-rl]] -- 推理与工具使用的结合
- [[rl-training-frameworks]] -- RL 训练框架（veRL 等）
- [[grpo]] -- GRPO 算法
- [[ppo-for-llm]] -- PPO 算法
