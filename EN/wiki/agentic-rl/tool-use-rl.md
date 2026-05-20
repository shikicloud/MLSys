---
title: "Reinforcement Learning for Tool Use and API Calling"
category: agentic-rl
tags: [tool-use, rl, retool, code-interpreter, api-calling, toolformer, gorilla, function-calling]
created: 2026-04-13
updated: 2026-05-20
status: mature
---

# Reinforcement Learning for Tool Use and API Calling

> [!abstract]+ TL;DR
> One of the core directions of [[agentic-rl-overview|agentic RL]] -- training LLMs via RL and execution feedback to learn **when** to call tools, **which** to call, **how** to format arguments, and **how to interpret** results, rather than relying on expert SFT. Milestone system: **ReTool** (2025) -- an RL-trained 32B reaches **72.5% on AIME**, surpassing OpenAI o1-preview by 27.9 pp; trained with [[rl-training-frameworks#veRL|veRL]] + PPO, two-stage cold-start SFT + tool-augmented RL. Emergent behavior: code self-correction (the "aha moment" of adaptive tool use). EMNLP 2025 Findings shows pure RL (no SFT) from scratch is also viable.

## Overview

Tool-Use RL is one of the core research directions in [[agentic-rl-overview|agentic RL]], focused on using reinforcement learning to train LLMs to learn **when** to call a tool, **which** tool to call, **how** to format the call arguments, and **how to interpret** the tool's return value.

The traditional approach uses supervised fine-tuning (SFT) on expert-labeled tool-call examples, but this method has clear limitations:
- Cannot explore tool-use strategies not covered by the experts
- Hard to adapt to new tools or API changes
- Error accumulation causes multi-step tool chains to fail

RL methods overcome these limitations through trial-and-error learning -- the model learns through interaction when tools are more efficient than pure reasoning, and can continually improve from the actual feedback of tool execution.

### The value of tool use

Why is it so important for LLMs to learn to use tools?

1. **Overcome inherent limitations**: LLMs' abilities in math, real-time information retrieval, code execution, etc. are inherently limited
2. **Extend the capability frontier**: Through tool access, LLMs can operate databases, control software, and search the internet
3. **Improve reliability**: Tools provide deterministic results (a calculator does not miscalculate), reducing hallucination
4. **Enable grounding**: Real-world tasks almost always require interacting with external systems

## Formalization of tool use

### MDP modeling

Tool use can be formalized as a Markov Decision Process (MDP):

$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \gamma)$$

**State space** $\mathcal{S}$:
$$s_t = (\text{task}, h_{1:t-1}, \text{tool\_results}_{1:t-1})$$

The state consists of the task description, history of interaction, and previous tool execution results.

**Action space** $\mathcal{A}$:
$$a_t \in \begin{cases} \mathcal{A}_{\text{text}} & \text{generate natural-language text} \\ \mathcal{A}_{\text{tool}} = \{(\text{tool\_name}, \text{args})\} & \text{call a tool} \\ \mathcal{A}_{\text{special}} = \{\text{submit, give\_up}\} & \text{special actions} \end{cases}$$

The action space is hybrid: at each step the model can choose to generate pure text (e.g. reasoning, summary) or call a tool.

**Transition function** $\mathcal{T}$:
$$s_{t+1} = \begin{cases} s_t \oplus a_t & \text{if } a_t \in \mathcal{A}_{\text{text}} \\ s_t \oplus a_t \oplus \text{env}(a_t) & \text{if } a_t \in \mathcal{A}_{\text{tool}} \end{cases}$$

When the action is text, the state is simply appended. When the action is a tool call, the environment executes the tool and returns a result, which is appended to the state.

**Reward function** $\mathcal{R}$: typically a sparse task-completion reward plus intermediate process rewards.

### ASCII diagram: tool-use MDP

```
Tool-use MDP interaction flow:

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

### Episode example

```
Task: "Compute 2^100 + 3^50 and determine whether the result is prime"

Turn 1:
  State: [task description]
  Action: <think>This is a huge calculation; I should use a Python interpreter</think>
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
  Action: 2^100 + 3^50 = 1,267,650,600,228,229,401,496,703,205,975.
          This number is not prime.<submit>
  Reward: +1 (correct answer)
```

## Training methods

### 1. Toolformer approach: self-supervised tool annotation

**Paper**: [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761) (Meta, 2023)

**Core idea**: let the model itself decide at which positions in the text inserting a tool call lowers the perplexity of subsequent tokens.

**Training pipeline**:

```
Step 1: Candidate position sampling
  For each position in the text, sample possible tool calls c_i
  e.g. the [calculator] position in "The population is [calculator(2.5 * 10^9)] large"

Step 2: Execute and filter
  Actually execute the tool call to obtain result r_i
  Construct text with result: x_i = "... [c_i → r_i] ..."

Step 3: Perplexity comparison
  If L(x_with_tool) < L(x_without_tool) - τ:
    Keep this tool-call annotation
  Otherwise:
    Discard

Step 4: Fine-tune
  Run SFT on the model with the filtered annotated data
```

**Supported tools**: calculator, QA system, Wikipedia search, machine translation, calendar

**Limitations**:
- Fundamentally SFT, not RL -- cannot learn from trial and error
- Fixed tool-call pattern, does not adapt to dynamic change
- Cannot handle multi-step tool chains

### 2. RLEF: execution feedback as reward

**Core idea**: use the actual result of tool execution as the RL reward signal.

```
Classic RLHF:  Action → reward model score → may be inaccurate
RLEF:          Action → actual execution → objective result → precise reward
```

**Types of reward signals in RLEF**:

| Reward type | Description | Example |
|-------------|-------------|---------|
| Binary reward | Success / failure | Code tests pass / fail |
| Continuous reward | Degree of partial correctness | SQL query returns 80% of correct rows |
| Multi-dimensional reward | Multiple evaluation axes | Correctness + efficiency + safety |
| Differential reward | Compared with a baseline | 3x faster than the naive method |

**Advantages**:
- Zero human annotation cost
- Reward signal is fully objective and verifiable
- Naturally adapts to changes in tool APIs

**Challenges**:
- Requires a reliable sandbox execution environment
- Some tool calls have side effects (irreversible)
- Execution latency increases rollout time

### 3. Process reward for tool-selection quality

Give intermediate rewards at each decision point in tool use, not only the final task-completion reward:

```python
def tool_process_reward(turn):
    reward = 0.0

    # 1. Is the tool choice reasonable?
    if turn.chose_correct_tool:
        reward += 0.3
    elif turn.chose_wrong_tool:
        reward -= 0.2

    # 2. Is the argument format correct?
    if turn.is_tool_call and turn.valid_syntax:
        reward += 0.1
    elif turn.is_tool_call and not turn.valid_syntax:
        reward -= 0.3

    # 3. Should a tool be used at all?
    if turn.used_tool_unnecessarily:
        reward -= 0.1  # could reason it out but used a tool
    elif turn.should_have_used_tool:
        reward -= 0.2  # should have used a tool but chose to reason

    return reward
```

### 4. RL fine-tuning for function calling

The function-calling feature of modern LLM APIs is usually trained via SFT, but RL can push it further:

**SFT stage**: train basic capability on a large set of function-call examples

```json
{
  "function": "get_weather",
  "arguments": {"city": "Beijing", "unit": "celsius"}
}
```

**RL stage**: optimize via actual execution feedback

```
Reward signals:
  +1.0: function call succeeds and the result is correct
  +0.5: function call succeeds but is not the optimal choice
  -0.5: function-call syntax error
  -1.0: called a non-existent function
  -0.2: unnecessary function call (wasted tokens/time)
```

### 5. ReTool: the milestone of tool-augmented RL

**Paper**: [ReTool: Reinforcement Learning for Strategic Tool Use in LLMs](https://arxiv.org/abs/2504.11536) (2025)

ReTool is the milestone work that applies RL to training tool use:

**Core innovation**: cast "call the Python interpreter or continue pure-text reasoning" as an explicit RL decision problem.

**Two-stage training**:

```
Stage 1: Cold-start SFT
  - Collect a small set of high-quality tool-use examples
  - Annotate when the code interpreter should be called
  - SFT to give the model basic tool-call capability

Stage 2: Tool-augmented RL
  - Use veRL + PPO
  - Within each rollout, the model can choose:
    (a) Continue text reasoning: <think>...</think>
    (b) Call Python: <tool>python\n...\n</tool>
  - Code is executed in real time and the result is injected into context
  - Reward is computed from final-answer correctness
```

**Key results**:
- ReTool-32B reaches 67% on AIME (400 steps) vs. pure-text RL 40% (1080 steps)
- Final 72.5%, surpassing o1-preview by 27.9%
- Code self-correction emerges: after running code and discovering an error, the model automatically modifies and re-executes

**Emergent behavior example**:
```
Turn 1: <tool>python
        def solve():
            # initial attempt
            return naive_solution()
        print(solve())
        </tool>
Obs: Error: overflow

Turn 2: <think>The code overflowed; I need big-number handling</think>
        <tool>python
        from decimal import Decimal
        def solve():
            # corrected plan
            return improved_solution()
        print(solve())
        </tool>
Obs: 42

Turn 3: The answer is 42.
```

This "discover error → reflect → correct" behavior emerged from pure RL training; it was not explicitly demonstrated in the training data.

## Reward design

Reward design is the key challenge in tool-use RL. A good reward function must balance multiple objectives:

### 1. Binary task reward

$$R_{\text{task}} = \begin{cases} +1 & \text{final answer is correct} \\ 0 & \text{final answer is incorrect} \end{cases}$$

The simplest but also the sparsest reward. For complex tasks a huge number of trajectories may all get 0 reward, making learning extremely inefficient.

### 2. Tool efficiency reward

$$R_{\text{eff}} = -\alpha \cdot N_{\text{tool\_calls}} - \beta \cdot N_{\text{total\_steps}}$$

Encourages the model to finish a task with fewer tool calls and steps. Parameters $\alpha$ and $\beta$ control the strength of the penalty:

```
Good behavior: solves the problem in 2 tool calls → high efficiency reward
Bad behavior:  10 redundant tool calls → low efficiency reward
```

### 3. Tool correctness reward

$$R_{\text{correct}} = \frac{1}{N_{\text{calls}}} \sum_{i=1}^{N_{\text{calls}}} \mathbb{1}[\text{call}_i \text{ is valid}]$$

Evaluates the quality of tool calls: whether the argument format is correct, whether the tool name exists, whether the call executes successfully.

### 4. Composite reward function

```python
def composite_tool_reward(trajectory):
    """Composite reward function used in practice"""

    # Task completion (primary signal)
    task_score = 1.0 if check_answer(trajectory) else 0.0

    # Tool efficiency
    n_tools = count_tool_calls(trajectory)
    n_steps = len(trajectory.turns)
    efficiency = -0.01 * n_tools - 0.005 * n_steps

    # Tool correctness
    valid_calls = count_valid_calls(trajectory)
    total_calls = max(count_tool_calls(trajectory), 1)
    correctness = 0.2 * (valid_calls / total_calls)

    # Format compliance (penalize malformed tool calls)
    format_violations = count_format_errors(trajectory)
    format_penalty = -0.1 * format_violations

    # Safety (penalize dangerous operations)
    safety_violations = count_unsafe_actions(trajectory)
    safety_penalty = -1.0 * safety_violations

    return (task_score
            + efficiency
            + correctness
            + format_penalty
            + safety_penalty)
```

### Pitfalls in reward design

| Pitfall | Description | Mitigation |
|---------|-------------|------------|
| **Reward hacking** | Model finds shortcuts that earn high reward without completing the task | Multi-dimensional reward + adversarial testing |
| **Over-penalizing efficiency** | Model avoids tool calls to dodge efficiency penalty | Compute efficiency reward only when the task succeeds |
| **Format overfitting** | Model learns perfect format but content is meaningless | Increase weight of content correctness |
| **Sparse-reward dilemma** | Reward is too sparse and nothing is learned | Add process reward + curriculum learning |

## Representative systems

### WebGPT

**Source**: OpenAI (2021)

WebGPT is one of the earliest large-scale systems to apply RL to tool use:
- Equips GPT-3 with a web-browsing toolset: search, click, scroll, back, quote
- Trains the browsing policy via RLHF
- The resulting model can search for information, synthesize multiple sources, and generate answers with citations

**Action space**:
```
Actions = {
    Search(query),      # search
    Click(element_id),  # click link / button
    Scroll(direction),  # scroll up / down
    Quote(text),        # quote text from current page
    Back(),             # go back one page
    Submit(answer)      # submit final answer
}
```

### Gorilla

**Source**: UC Berkeley (NeurIPS 2024)

Gorilla focuses on generating correct API calls:
- Training data comes from API documentation (Torch Hub, TensorFlow Hub, HuggingFace)
- The model learns to generate accurate API calls from natural-language descriptions
- Introduces the AST-accuracy evaluation metric

**Contribution**: demonstrates that LLMs can learn the exact syntax of API calls, not merely an approximation.

### ToolLLM

**Core design**:
- Built a large-scale dataset containing 16,000+ real-world APIs
- Designed a DFSDT (Depth-First Search Decision Tree) reasoning strategy
- Supports both single-tool and multi-tool scenarios

**Action space**:
```
Single-tool: search_weather(city="Beijing")
Multi-tool:  search_flights(from="Beijing", to="Tokyo", date="2026-05-01")
             → book_hotel(city="Tokyo", checkin="2026-05-01", nights=3)
             → get_directions(from="Narita Airport", to="Hotel")
```

### API-Bank

**Core design**:
- A benchmark set of 314 tool APIs
- Three evaluation levels: API retrieval, API call, API fusion
- Used to evaluate the tool-use ability of LLMs

## Code example

### Tool-use RL training pseudocode

```python
class ToolUseRLTrainer:
    """Tool-use RL trainer (simplified)"""

    def __init__(self, policy, ref_model, env):
        self.policy = policy
        self.ref_model = ref_model
        self.env = env

    def collect_rollout(self, task, max_turns=20):
        """Collect a full tool-use trajectory"""
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
        """Composite reward: task completion + tool quality + efficiency"""
        task_reward = 1.0 if check_answer(trajectory) else 0.0
        tool_turns = [t for t in trajectory["turns"] if is_tool_call(t["action"])]
        tool_quality = (sum(t["info"].get("tool_success", 0) for t in tool_turns)
                       / max(len(tool_turns), 1))
        efficiency = max(0, 1.0 - 0.02 * len(trajectory["turns"]))
        return 1.0 * task_reward + 0.2 * tool_quality + 0.1 * efficiency

    def grpo_update(self, task_batch, n_samples=8):
        """GRPO policy update"""
        all_groups = []
        for task in task_batch:
            group = []
            for _ in range(n_samples):
                traj = self.collect_rollout(task)
                traj["reward"] = self.compute_reward(task, traj)
                group.append(traj)
            # In-group normalized advantage
            rewards = [r["reward"] for r in group]
            mean_r, std_r = mean(rewards), max(std(rewards), 1e-8)
            for r in group:
                r["advantage"] = (r["reward"] - mean_r) / std_r
            all_groups.append(group)

        # Policy-gradient update (clipped PPO + KL penalty)
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

## Challenges

### 1. Hallucinated tool calls

The model may produce non-existent tool names, wrong argument formats, or fabricated tool outputs:

```
Common hallucination types:
- Calling a non-existent tool: <tool>quantum_solver\n...</tool>
- Wrong argument type: search(query=42)  # expected string, got int
- Pretending to receive tool output (fabricating results without waiting for actual execution)
- Using an outdated API signature
```

**Mitigations**:
- Explicitly list available tools and argument formats in the system prompt
- Penalize invalid tool calls with negative reward
- Use constrained decoding to ensure tool names fall in the valid set
- Train a format checker that validates calls before execution

### 2. Action-space explosion

The effective action space for tool use is much larger than for pure-text generation:

$$|\mathcal{A}_{\text{effective}}| = |\mathcal{A}_{\text{text}}| + \sum_{t \in \text{Tools}} |\text{Args}(t)|$$

For systems with hundreds of API endpoints the exploration space becomes enormous.

**Mitigations**:
- Hierarchical action space: pick tool category first, then specific tool, then fill arguments
- Retrieval-augmented: retrieve a relevant tool subset for each task
- Curriculum learning: start with a small set of tools and gradually grow

### 3. Credit assignment

In a multi-step tool chain it is hard to determine which call is responsible for success/failure:

```
Task: analyze stock data and give advice
  Turn 1: search("AAPL stock price") → ✓
  Turn 2: python(parse_data(results)) → ✓
  Turn 3: search("AAPL earnings report") → ✓ (but the query is not precise enough)
  Turn 4: python(analyze(data)) → Bug in code → ✗
  Turn 5: python(fix_and_analyze(data)) → ✓
  Final: produced wrong advice → Reward = 0

  Question: which step should bear the most blame? Turn 3's imprecise search? Turn 4's bug?
```

### 4. Tool-execution latency

Tool calls require actual execution time, significantly inflating rollout cost:

| Action type | Typical latency |
|-------------|-----------------|
| Text generation | ~100 ms / token |
| Code execution | 100 ms - 30 s |
| API call | 200 ms - 5 s |
| Web search | 1 s - 10 s |
| Database query | 50 ms - 5 s |

In large-scale RL training (thousands of rollouts), tool-execution latency becomes the training bottleneck.

**Mitigations**:
- Asynchronously execute many rollouts in parallel
- Cache tool results (return cached results for identical calls)
- Pre-warm a sandbox pool (pre-create sandbox instances)

### 5. Safety

Code execution and API calls introduce serious security risks:

- **Code injection**: model may generate malicious code
- **Resource exhaustion**: infinite loops, memory overflow
- **Data leakage**: reading sensitive files or environment variables
- **Network attacks**: launching requests via API calls

**Safety mechanisms**:
```python
class SafeToolExecutor:
    """Safe tool executor"""

    FORBIDDEN_PATTERNS = [
        r"import\s+os",
        r"import\s+subprocess",
        r"open\(.*/etc/",
        r"requests\.delete",
        r"rm\s+-rf",
    ]

    def execute(self, tool_name, args, timeout=30):
        # 1. Safety check
        for pattern in self.FORBIDDEN_PATTERNS:
            if re.search(pattern, str(args)):
                raise SecurityError(f"Forbidden pattern: {pattern}")

        # 2. Sandboxed execution
        with Sandbox(
            network=False,         # disable network
            filesystem="readonly", # read-only filesystem
            memory_limit="512MB",  # memory limit
            cpu_time=timeout       # CPU-time limit
        ) as sandbox:
            return sandbox.run(self.tools[tool_name], args)
```

## References

### Core papers

- Schick et al. (2023). [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761). arXiv:2302.04761.
- Feng et al. (2025). [ReTool: Reinforcement Learning for Strategic Tool Use in LLMs](https://arxiv.org/abs/2504.11536). arXiv:2504.11536.
- Nakano et al. (2021). [WebGPT: Browser-Assisted Question-Answering with Human Feedback](https://arxiv.org/abs/2112.09332). arXiv:2112.09332.
- Patil et al. (2024). [Gorilla: Large Language Model Connected with Massive APIs](https://arxiv.org/abs/2305.15334). NeurIPS 2024.
- Qin et al. (2023). [ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs](https://arxiv.org/abs/2307.16789). arXiv:2307.16789.
- Li et al. (2023). [API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs](https://arxiv.org/abs/2304.08244). arXiv:2304.08244.

### Surveys and benchmarks

- Qu et al. (2025). [Tool Learning with Large Language Models: A Survey](https://arxiv.org/abs/2405.17935). arXiv:2405.17935.

## Related pages

- [[agentic-rl-overview]] -- panorama of agentic RL
- [[environment-design]] -- sandbox and execution-environment design
- [[tool-use]] -- tool use from an agent-architecture perspective
- [[multi-step-reasoning-rl]] -- combining reasoning with tool use
- [[rl-training-frameworks]] -- RL training frameworks (veRL etc.)
- [[grpo]] -- the GRPO algorithm
- [[ppo-for-llm]] -- the PPO algorithm
