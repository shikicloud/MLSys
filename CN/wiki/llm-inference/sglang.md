---
title: "SGLang：快速结构化生成与服务"
category: llm-inference
tags: [sglang, radix-attention, 结构化生成, 推理引擎, 约束解码]
created: 2026-04-13
updated: 2026-04-14
status: mature
---

# SGLang：快速结构化生成与服务

> [!abstract]+ TL;DR
> SGLang（Structured Generation Language）由 LMSYS（UC Berkeley）开发，结合**前端 DSL** 与**后端运行时（SRT）** 的协同设计：**RadixAttention**（基数树驱动的跨请求 KV 复用）、压缩 FSM（jump-forward 约束解码）、XGrameur 集成。2023 年 12 月发布，NeurIPS 2024 正式发表。截至 2026 年驱动 xAI（Grok 3）、Microsoft Azure、LinkedIn、Cursor —— 生产部署超 40 万 GPU。H100 上吞吐量比 [[vllm|vLLM]] **高 29 %**（16,200 vs 12,500 tok/s），前缀密集工作负载提升达 **6.4 倍**。在多轮对话、结构化输出、智能体工具调用场景表现尤其突出。

**关键特性**：
- **RadixAttention**：基于基数树的跨请求 [[kv-cache-optimization|KV 缓存]]自动复用
- **压缩有限状态机**：约束解码加速，jump-forward 优化
- **前端 DSL**：声明式 LLM 编程原语（gen、select、fork 等）
- **高度并行**：支持 TP / PP / DP / EP / CP 五种并行策略组合
- **XGrammar 集成**：高性能结构化输出后端

**生态位置**：SGLang 与 [[vllm|vLLM]] 是当前最主流的两个开源 LLM 推理引擎。SGLang 在前缀密集型工作负载上性能领先（吞吐量高 29%），而 vLLM 在硬件覆盖面和社区规模上占优。截至 2026 年，SGLang 已在 xAI（Grok 3）、Microsoft Azure、LinkedIn、Cursor 等生产环境中大规模部署，覆盖超过 400,000 张 GPU。

[GitHub](https://github.com/sgl-project/sglang) | [论文](https://arxiv.org/abs/2312.07104)

---

## 核心架构

SGLang 采用前端-后端分离的协同设计架构（co-design）：

```
+------------------------------------------------------------------+
|                        用户应用层                                  |
|  @sgl.function 装饰的 Python 程序 / OpenAI 兼容 API 请求          |
+------------------------------------------------------------------+
         |                                    |
         v                                    v
+--------------------+          +-----------------------------+
|   SGLang 前端 DSL  |          |    HTTP / gRPC API 网关     |
|                    |          |  (FastAPI, OpenAI 兼容)      |
| - gen() / select() |          +-----------------------------+
| - fork() / join()  |                    |
| - extend (+=)      |                    v
| - 角色包装器       |   +-----------------------------------+
+--------------------+   |        TokenizerManager            |
         |               |   (独立进程, 分词/去分词)           |
         v               +-----------------------------------+
+--------------------+                    |
|   IR / 解释器      |                    | ZeroMQ IPC
|  (数据流图编译)     |                    v
+--------------------+   +-----------------------------------+
         |               |          Scheduler 调度器          |
         |               |   - 缓存感知调度 (cache-aware)     |
         +-------------->|   - RadixCache 管理                |
                         |   - 内存预算控制                    |
                         |   - 批次组装 (ScheduleBatch)        |
                         +-----------------------------------+
                                          |
                                          v
                         +-----------------------------------+
                         |         ModelRunner / Worker       |
                         |   - FlashInfer / Triton 注意力内核 |
                         |   - CUDA Graph 优化                |
                         |   - ForwardBatch GPU 执行          |
                         +-----------------------------------+
                                          |
                                          v
                         +-----------------------------------+
                         |           GPU 集群                 |
                         |   TP / PP / DP / EP / CP           |
                         +-----------------------------------+
```

### 多进程架构

SGLang 采用多进程架构避免 Python GIL 瓶颈：

| 组件 | 进程 | 职责 |
|------|------|------|
| **Server** | 主进程 | FastAPI 应用，路由 HTTP/gRPC 请求 |
| **TokenizerManager** | 独立进程 | 分词、去分词，CPU 密集 |
| **Scheduler** | 独立进程 | GPU 内存管理、批次调度、RadixCache |
| **ModelRunner** | 独立进程 | 模型前向计算，GPU 密集 |
| **DetokenizerManager** | 独立进程 | 输出去分词，流式返回 |

进程间通过 **ZeroMQ** 通信，使用专用端口：`tokenizer_ipc`、`scheduler_input_ipc`、`detokenizer_ipc`。

### 请求流程

```
用户请求 --> FastAPI --> TokenizerManager(分词) --ZMQ--> Scheduler(前缀匹配,
批次组装) --> ModelRunner(GPU 前向) --> Scheduler(采样, 更新 RadixCache)
--ZMQ--> DetokenizerManager(去分词) --> 流式返回
```

数据结构逐级转换：**Req** --> **ScheduleBatch**（CPU） --> **ModelWorkerBatch**（Worker） --> **ForwardBatch**（GPU）

---

## RadixAttention

RadixAttention 是 SGLang 的核心创新，实现了**跨请求的 KV 缓存自动复用**。

### 问题背景

传统推理引擎（如早期 vLLM）在请求处理完成后会丢弃其 KV 缓存。这意味着：
- 多轮对话中，每一轮都需要重新计算之前所有轮次的 prefill
- 共享系统提示（system prompt）的不同请求无法复用缓存
- Few-shot 示例在每个请求中都要重新计算

### 基数树数据结构

**基数树（Radix Tree）** 是一种空间高效的前缀树变体。与标准 trie 不同，基数树的边可以标记**可变长度的序列**（而非单个元素），大幅提高了存储效率。

SGLang 使用基数树建立 **token 序列 --> KV 缓存张量** 的映射关系：

```
                        [root]
                       /      \
                  [system      [system
                  prompt A]     prompt B]
                 /    \              \
           [user      [user         [user
            msg 1]     msg 2]        msg X]
           /    \         \
     [asst      [asst     [asst
      rsp 1]     rsp 1']   rsp 2]
       |
  [user msg 2]
       |
  [asst rsp 2]

  绿色 = 新插入节点
  蓝色 = 缓存命中（复用 KV cache）
  红色 = LRU 驱逐节点

每个节点存储：
  - token 序列片段（边标签）
  - 对应的 KV cache 张量引用（分页 GPU 内存）
  - 引用计数 + 最后访问时间戳
```

### 工作流程

1. **前缀匹配**：新请求到达时，从根节点遍历基数树，找到最长匹配的缓存前缀
2. **增量计算**：仅从匹配点之后的 token 开始 prefill（命中 = 跳过已缓存部分）
3. **缓存插入**：新计算的 KV 缓存作为新节点插入树中
4. **LRU 驱逐**：GPU 内存不足时，递归驱逐引用计数为零的叶节点（最近最少使用）
5. **缓存感知调度**：优先调度前缀匹配最长的请求（近似最优的 DFS 顺序）

### 与 vLLM 哈希前缀缓存的对比

| 特性 | SGLang RadixAttention | vLLM 哈希前缀缓存 |
|------|----------------------|-------------------|
| **粒度** | Token 级（1 token = 1 page） | Block 级（如 16 tokens/block） |
| **匹配方式** | 树遍历，自然支持前缀 | 哈希查找，需完整块匹配 |
| **配置** | 零配置，自动发现 | 需手动启用 `--enable-prefix-caching` |
| **调度协同** | 缓存感知调度器原生集成 | 调度器与缓存独立 |
| **内存开销** | 基数树本身占 CPU 内存 | 哈希表占内存较少 |
| **适用场景** | 前缀密集型（多轮、RAG）最优 | 通用场景足够好 |

### 四种复用模式

RadixAttention 支持自动发现以下四种 KV 缓存复用模式：

1. **Few-shot 示例共享**：多个请求共享相同的 few-shot 示例前缀
2. **自一致性采样（Self-consistency）**：同一问题的多次采样复用问题前缀
3. **多轮对话历史**：后续轮次复用之前所有轮次的 KV 缓存
4. **思维树搜索（Tree-of-Thought）**：搜索树的不同分支复用共同祖先路径

### 缓存命中率

实测缓存命中率：
- Few-shot 任务：**85-95%**
- 多轮对话：**75-90%**
- RAG 工作负载：**50-80%**（取决于文档复用率）
- 整体基准测试：**50-99%**

### 性能影响

RadixAttention 经过消融实验验证，在无缓存命中的情况下也**不会引入可测量的开销**。它与 [[continuous-batching|连续批处理]]和[[kv-cache-optimization#PagedAttention|PagedAttention]] 完全兼容，可以无缝集成。

---

## SGLang 编程语言 / DSL

SGLang 的前端是一个嵌入在 Python 中的领域特定语言（DSL），提供声明式原语来编程复杂的 LLM 应用。

### 核心原语

| 原语 | 语法 | 功能 |
|------|------|------|
| **extend** | `s += "text"` | 向当前提示追加文本 |
| **gen** | `gen(name, ...)` | 调用 LLM 生成，非阻塞，结果存入变量 |
| **select** | `gen(name, choices=[...])` | 从候选列表中选择概率最高的选项 |
| **fork** | `s.fork(n)` | 创建 n 个并行执行分支 |
| **角色包装器** | `s.system()` / `s.user()` / `s.assistant()` | 自动管理聊天模板格式 |

### gen() 参数详解

```python
gen(
    name="variable_name",    # 变量名，后续通过 s["variable_name"] 获取
    max_tokens=512,          # 最大生成 token 数
    stop="\n",               # 停止序列
    temperature=0.7,         # 采样温度（0 = 确定性）
    regex=r"pattern",        # 正则表达式约束
    choices=["A", "B"],      # 选项约束（select 模式）
)
```

### @sgl.function 装饰器

SGLang 程序通过 `@sgl.function`（或 `@function`）装饰器定义。第一个参数 `s` 是状态对象，跟踪提示历史和生成的变量。

### 执行模式

- **解释器模式**：逐步执行原语，通过 `RuntimeEndpoint` 与后端通信
- **编译器模式**：将程序编译为数据流图（IR），启用优化（如异步执行、批次合并）

### 代码示例

#### 基础问答

```python
from sglang import function, gen, system, user, assistant

@function
def basic_qa(s, question):
    s += system("You are a helpful assistant.")
    s += user(question)
    s += assistant(gen("answer", max_tokens=512))

state = basic_qa("What is the capital of France?")
print(state["answer"])
```

#### 多轮对话

```python
@function
def multi_turn(s):
    s += system("You are a helpful assistant.")
    s += user("List 3 countries and their capitals.")
    s += assistant(gen("first_answer", max_tokens=256))
    s += user("Now list 3 more, different from the above.")
    s += assistant(gen("second_answer", max_tokens=256))
    # 第二轮自动复用第一轮的 KV cache（RadixAttention）
```

#### 条件分支（工具选择）

```python
@function
def tool_use(s, question):
    s += user(question)
    s += assistant(
        "To answer this question, I need to use a " +
        gen("tool", choices=["calculator", "search engine"]) + ". "
    )
    if s["tool"] == "calculator":
        s += assistant("The expression is: " + gen("expression"))
    elif s["tool"] == "search engine":
        s += assistant("Search keyword: " + gen("keyword"))
```

#### 并行分支（Fork）

```python
@function
def parallel_tips(s):
    s += user("Give me 3 tips for learning Python.")
    s += assistant(gen("tips", max_tokens=128))

    forks = s.fork(3)
    for i, f in enumerate(forks):
        f += user(f"Expand tip {i+1} in detail.")
        f += assistant(gen("detail", max_tokens=256))

    # 收集所有分支的结果
    for i, f in enumerate(forks):
        print(f"Tip {i+1} detail:", f["detail"])
```

#### 正则约束生成

```python
@function
def ip_address_gen(s):
    s += user("What is the IP address of Google DNS?")
    s += assistant(gen(
        "answer",
        temperature=0,
        regex=r"((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)"
    ))
```

#### 批量推理

```python
states = basic_qa.run_batch(
    [
        {"question": "What is the capital of France?"},
        {"question": "What is the capital of Japan?"},
        {"question": "What is the capital of Brazil?"},
    ],
    progress_bar=True
)
for state in states:
    print(state["answer"])
```

#### 流式输出

```python
state = basic_qa.run(
    question="Explain quantum computing.",
    temperature=0.1,
    stream=True
)
for chunk in state.text_iter():
    print(chunk, end="", flush=True)
```

### 与原始 API 对比

SGLang DSL 将命令式 API 调用转为声明式程序，后端自动管理 KV cache 复用：

```python
# 原始 API：每次调用传输完整历史，无 KV cache 复用
# SGLang DSL：声明式定义，KV cache 跨 gen() 调用自动保留
@function
def dialog(s, q1, q2):
    s += system("...")
    s += user(q1)
    s += assistant(gen("a1"))   # KV cache 自动保留
    s += user(q2)
    s += assistant(gen("a2"))   # 自动复用 a1 的 KV cache
```

---

## 约束解码 (Constrained Decoding)

SGLang 在约束解码方面具有两大核心技术：**压缩有限状态机（Compressed FSM）** 和 **XGrammar 集成**。

### 基本原理

约束解码的目标是确保 LLM 输出符合预定义的格式（如 JSON schema、正则表达式等）。核心方法是在每一步解码时，根据当前 FSM 状态计算允许的 token 集合，对不合法的 token 施加 logit 掩码（logit bias masking）。

```
传统约束解码流程：

JSON Schema --> 正则表达式 --> 有限状态机 (FSM)
                                    |
                                    v
每步解码: FSM 当前状态 --> 计算允许的 token 集合
                         --> logit masking
                         --> 采样 --> 更新 FSM 状态
```

### 压缩有限状态机 (Compressed FSM)

SGLang 的关键创新是**压缩 FSM**：分析 FSM 中相邻的单一转移边（singular-transition edges），将连续的确定性转移压缩为单条边。

```
原始 FSM（token 级）：
  S0 --{--> S1 --"--> S2 --n--> S3 --a--> S4 --m--> S5 --e--> S6 --"--> S7

压缩后 FSM：
  S0 ------{"name"------> S7

  (整个 {"name" 序列在一步中完成)
```

**效果**：当 FSM 处于确定性路径上时，可以**一步解码多个 token**，无需逐个调用 LLM forward pass。

### Jump-Forward 解码优化

Jump-forward 是压缩 FSM 的运行时优化：

1. 当检测到当前 FSM 状态进入确定性路径时
2. **直接 prefill** 整个确定性 token 序列（而非逐 token 解码）
3. 利用 RadixAttention 的 extend 原语，自动复用已有 KV 缓存
4. 实现方式：终止当前请求，用扩展后的前缀入队新请求

```
普通解码：  [prompt] -> t1 -> t2 -> t3 -> t4 -> ... (逐 token)
Jump-forward: [prompt] -> [{"name":] -> t_free -> t_free -> ... (跳过确定性部分)
                           ^^^^^^^^^
                           一次 prefill 完成
```

### XGrammar 集成

从 v0.4 起，SGLang 默认使用 **XGrammar** 作为结构化输出后端。XGrammar 是一个独立的高性能约束解码库，支持：

| 约束类型 | 示例 |
|---------|------|
| **JSON Schema** | `{"type": "object", "properties": {...}}` |
| **正则表达式** | `r"[A-Z][a-z]+ \d{4}"` |
| **EBNF 语法** | 自定义 BNF 格式的语法规则 |
| **结构化标签** | 推理模型的 think/answer 标签约束 |

XGrammar 的核心优势是将语法掩码生成与 LLM 前向传播**重叠执行**（overlap），消除约束解码的额外延迟。

### 性能数据

| 指标 | SGLang + XGrammar | 传统方法 |
|------|-------------------|---------|
| JSON 解码延迟 | 降低 **2x** | 基准 |
| JSON 解码吞吐量 | 提升 **2.5x** | 基准 |
| JSON 合规率 | **96-99.8%** | 90-94% |
| 对比无约束解码 | 约束解码**更快** | 约束解码更慢 |

关键发现：在 SGLang 中，约束解码（使用压缩 FSM + jump-forward）甚至比无约束解码更快，因为跳过了确定性 token 的 LLM 计算。

### JSON 约束生成示例

```python
# 方式 1：DSL + regex 约束
@sgl.function
def json_gen(s, prompt):
    s += sgl.user(prompt)
    s += sgl.assistant(sgl.gen("output", max_tokens=256,
        regex=r'\{"name": "[^"]+", "age": \d+, "city": "[^"]+"\}'))

# 方式 2：OpenAI 兼容 API + JSON schema
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Generate a person's info"}],
    response_format={"type": "json_schema", "json_schema": {
        "name": "person",
        "schema": {"type": "object",
                   "properties": {"name": {"type": "string"},
                                  "age": {"type": "integer"},
                                  "city": {"type": "string"}},
                   "required": ["name", "age", "city"]}
    }}
)
```

---

## FlashInfer 集成

SGLang 深度集成 **FlashInfer** 作为默认注意力计算后端，这是其高性能的关键支柱之一。

### FlashInfer 简介

FlashInfer 是一个专为 LLM 推理设计的高性能注意力内核库，提供：
- **PagedAttention 内核**：优化的分页 KV 缓存注意力计算
- **块稀疏注意力**：将 PagedAttention 实现为块稀疏注意力内核
- **可定制变体**：通过 Jinja 模板和 JIT 编译支持自定义注意力变体

### 核心优化

| 优化 | 说明 |
|------|------|
| **GPU 共享内存预取** | 在 GPU shared memory 中预取 page indices，消除 page size 对性能的影响 |
| **JIT 编译** | 通过 PyTorch JIT 编译自定义注意力内核，零额外开发成本 |
| **FlexAttention 启发** | 用户可通过 LogitsTransform/QueryTransform 等函子定义注意力变体 |
| **FP8 支持** | 原生 FP8 KV 缓存注意力计算 |

### 性能影响

FlashInfer 相比 Triton 后端在 SGLang 中的表现（Llama-3 8B/70B）：
- **中位 ITL（Inter-Token Latency）降低 29-69%**
- **TTFT（Time-To-First-Token）提升最高 21%**
- 支持 1 token/page 的细粒度分页（RadixAttention 的基础）

SGLang 也支持 **Flash Attention 3** 后端作为替代选项，在特定场景下可进一步优化性能。

---

## 并行策略

SGLang 支持五种正交并行策略，可以灵活组合：

```
总 GPU 数 = TP_size × PP_size × EP_size × DP_size

示例：32 GPU 部署
  --tp 4 --pp 2 --ep 2 --dp 2
  每张 GPU 被分配多个维度的 rank 索引
```

### 各并行策略说明

| 策略 | 参数 | 说明 | 适用场景 |
|------|------|------|---------|
| **张量并行 (TP)** | `--tp N` | 水平切分权重矩阵，all-reduce 同步 | 单模型太大放不下一张卡 |
| **流水线并行 (PP)** | `--pp N` | 按层分配到不同 GPU | 超大模型跨节点 |
| **数据并行 (DP)** | `--dp N` | 多副本并行服务，提升吞吐 | 内存充足时最大化吞吐 |
| **专家并行 (EP)** | `--ep N` | MoE 模型专家分布到不同 GPU | DeepSeek V3/R1 等 MoE 模型 |
| **上下文并行 (CP)** | `--cp N` | 序列维度切分（超长上下文） | 超长文档处理 |

### 专家并行 (EP) 详解

EP 对于 MoE 模型（如 DeepSeek V3/R1）至关重要：
- 专家权重分布到多个 GPU，通过 **all-to-all 通信** 路由 token
- 使用优化的 **分组矩阵乘法 (grouped GEMM)** 减少空闲 GPU 时间
- SGLang 实现了首个开源的 DeepSeek V3/R1 专家并行 + PD 分离方案

### 多节点部署

```bash
# 双节点 TP=8 部署
# 节点 0（master）
python -m sglang.launch_server --model meta-llama/Llama-3.1-70B \
    --tp 8 --nnodes 2 --node-rank 0 --master-addr <IP>

# 节点 1
python -m sglang.launch_server --model meta-llama/Llama-3.1-70B \
    --tp 8 --nnodes 2 --node-rank 1 --master-addr <IP>
```

---

## 性能分析

### 吞吐量对比 (H100 GPU)

| 引擎 | 标准吞吐量 (tok/s) | 前缀密集吞吐量 | DeepSeek V3 |
|------|---------------------|---------------|-------------|
| **SGLang** | **16,200** | **基准 x 6.4** | **3.1x vs vLLM** |
| vLLM | 12,500 | 基准 | 基准 |
| TensorRT-LLM | ~14,000 | - | - |

### 延迟对比 (Llama 3.1 8B, H100)

| 指标 | SGLang | vLLM |
|------|--------|------|
| **TTFT (首 token 延迟)** | **79 ms** | 103 ms |
| **ITL (token 间延迟)** | **6.0 ms** | 7.1 ms |
| **ITL 范围** | 4-21 ms（最稳定） | 更大波动 |
| **输出吞吐** | **894 tok/s** | 413 tok/s |

### 结构化输出性能

| 指标 | SGLang + XGrammar | 传统引导解码 |
|------|-------------------|-------------|
| 吞吐量 | **4,200 tok/s** | ~1,400 tok/s |
| JSON 合规率 | **99.8%** | 90-94% |
| 延迟 | **0.4s** | ~1.2s |

### 大规模分离部署 (96 H100)

SGLang 首创开源 DeepSeek V3/R1 PD 分离 + EP 部署：
- **输入吞吐**：52,300 tok/s per node
- **输出吞吐**：22,300 tok/s per node
- 相比普通 TP 部署：**5x 提升**

### GPU 利用率 (v0.4+)

零开销批调度器：**GPU 利用率 95-98%**（传统 70-80%），CPU 开销 <2%（传统 15-25%）。

---

## 代码示例

### 启动服务器与基本调用

```bash
# 基本启动
python -m sglang.launch_server \
    --model meta-llama/Llama-3.1-8B-Instruct --port 30000

# 多 GPU: TP + DP
python -m sglang.launch_server \
    --model meta-llama/Llama-3.1-70B-Instruct --tp 4 --dp 2

# DeepSeek V3: EP
python -m sglang.launch_server \
    --model deepseek-ai/DeepSeek-V3 --tp 4 --ep 4 --trust-remote-code
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:30000/v1", api_key="none")

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain transformer attention in 3 sentences."}
    ],
    temperature=0.7, max_tokens=256
)
```

### 多轮对话（自动前缀缓存复用）

```python
# SGLang 自动通过 RadixAttention 复用前缀 KV cache，无需额外配置
messages = [{"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "What is a binary tree?"}]
r1 = client.chat.completions.create(model="...", messages=messages)

messages.append({"role": "assistant", "content": r1.choices[0].message.content})
messages.append({"role": "user", "content": "How to implement it in Python?"})
r2 = client.chat.completions.create(model="...", messages=messages)
# 第二轮自动匹配前缀，跳过已缓存部分的 prefill
```

### Python Engine API（离线批量推理）

```python
import sglang as sgl
engine = sgl.Engine(model_path="meta-llama/Llama-3.1-8B-Instruct", tp_size=2)

outputs = engine.generate(
    [{"role": "user", "content": p} for p in [
        "Summarize relativity.", "Explain quantum entanglement.", "What is Higgs boson?"
    ]],
    sampling_params={"max_new_tokens": 256, "temperature": 0.7}
)
engine.shutdown()
```

---

## SGLang vs vLLM 详细对比

| 维度 | SGLang | vLLM |
|------|--------|------|
| **核心优化** | RadixAttention（基数树） | PagedAttention（分页内存） |
| **KV 缓存复用** | 跨请求自动复用（零配置） | 需手动启用 `--enable-prefix-caching` |
| **缓存粒度** | Token 级 | Block 级 (默认 16 tokens) |
| **调度策略** | 缓存感知调度 | FIFO 连续批处理 |
| **结构化输出** | 压缩 FSM + XGrammar（~3x 快） | XGrammar / Outlines |
| **前端 DSL** | 有（gen/select/fork 原语） | 无（纯 API 调用） |
| **并行策略** | TP + PP + DP + EP + CP | TP + PP + DP + EP |
| **H100 吞吐量** | **16,200 tok/s** | 12,500 tok/s |
| **前缀密集场景** | **最高 6.4x 提升** | 基准 |
| **DeepSeek 性能** | **3.1x 更快** | 基准 |
| **硬件支持** | NVIDIA、AMD（有限） | NVIDIA、AMD、TPU、Trainium、Gaudi |
| **模型覆盖** | 解码器、多模态、MoE | 解码器、编码器-解码器、多模态、MoE |
| **社区** | ~25K stars, 600 贡献者 | ~75K stars, 2,400 贡献者 |
| **TTFT** | **79 ms** | 103 ms |
| **ITL** | **6.0 ms** | 7.1 ms |
| **投机解码** | EAGLE-2/3 | Eagle、Medusa、多种方法 |
| **PD 分离** | 原生支持（first-class API） | 实验性支持 |

### 选择建议

- **选 SGLang**：多轮对话、RAG、结构化输出、DeepSeek 部署、智能体工作负载、极致吞吐
- **选 vLLM**：独立提示批处理、非 NVIDIA 硬件、编码器-解码器模型、最广模型支持、大社区

---

## 部署实践

### 关键服务器参数

```yaml
# config.yaml 示例
model: meta-llama/Llama-3.1-70B-Instruct
port: 30000
host: 0.0.0.0

# 并行
tp: 4
dp: 2

# 内存
mem_fraction_static: 0.85        # GPU 内存中用于 KV cache 的比例
max_running_requests: 256        # 最大并发请求数

# 批处理
max_num_reqs: 1024               # 最大排队请求数
schedule_policy: lpm             # lpm = longest prefix match

# 量化
quantization: fp8                # 支持 fp4/fp8/int4

# 投机解码
speculative_algorithm: EAGLE     # EAGLE-2/3
speculative_num_steps: 3         # 投机深度
speculative_eagle_topk: 4        # 每步候选数
```

### 部署模式

```bash
# 单机多卡
python -m sglang.launch_server \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --tp 4 --dp 2 --mem-fraction-static 0.85

# PD 分离 (Prefill 节点)
python -m sglang.launch_server \
    --model deepseek-ai/DeepSeek-V3 \
    --tp 8 --ep 4 --disaggregation-mode prefill --port 30000

# PD 分离 (Decode 节点)
python -m sglang.launch_server \
    --model deepseek-ai/DeepSeek-V3 \
    --tp 8 --ep 4 --disaggregation-mode decode --port 30001

# Docker
docker run --gpus all -p 30000:30000 lmsysorg/sglang:latest \
    python -m sglang.launch_server \
    --model meta-llama/Llama-3.1-8B-Instruct --host 0.0.0.0
```

### 生产环境调优建议

| 参数 | 建议值 | 说明 |
|------|--------|------|
| `mem_fraction_static` | 0.80-0.90 | 留出内存给模型权重和临时缓冲 |
| `schedule_policy` | `lpm` | 最长前缀匹配，最大化缓存命中 |
| `max_running_requests` | 根据 GPU 内存调整 | 过高会导致 OOM，过低浪费吞吐 |
| `chunked_prefill_size` | 8192 | 长输入的分块 prefill，避免卡顿 |
| `disable_radix_cache` | 否（保持开启） | 仅在无共享前缀时考虑关闭 |

---

## 其他重要特性

| 特性 | 说明 |
|------|------|
| **[[prefill-decode-disaggregation\|PD 分离]]** | Prefill 节点（计算密集）与 Decode 节点（内存密集）分开部署，通过 KV cache 传输通信 |
| **[[speculative-decoding\|投机解码]]** | EAGLE-2/3，投机深度 3-5 步，分支因子 4-8，DeepSeek 实测 1.4x 吞吐提升 |
| **多模态** | 支持视觉-语言模型（DSL 提供 `sgl.image()` 原语） |
| **量化** | FP4、FP8、INT4（AWQ/GPTQ），启动参数 `--quantization fp8` |
| **Multi-LoRA** | 单引擎同时服务多个 LoRA 适配器 |

---

## 不足与局限

### 硬件支持有限

- **主要支持 NVIDIA GPU**，AMD 支持在持续改进但仍不如 vLLM 成熟
- **不支持 TPU、Trainium、Gaudi** 等非 GPU 加速器（vLLM 支持）
- TPU 支持通过 SGLang-Jax 后端有初步进展（2025 年 10 月）

### 模型架构覆盖

- **不支持编码器-解码器模型**（T5、BART 等）
- 模型支持列表比 vLLM 少

### 社区与生态

- GitHub stars ~25K（vLLM ~75K），贡献者 ~600（vLLM ~2,400）
- Issue 响应时间 3-5 天（vLLM 12 小时）
- 文档和教程相对 vLLM 较少

### 技术限制

- **Python GIL 瓶颈**：CPU 路由管线在高并发时可能受限于单核 GIL，限制多线程扩展
- **RadixAttention 内存开销**：当前缀重叠率低时，基数树缓存消耗的 GPU 内存可能得不偿失
- **无共享前缀场景**：当所有请求完全独立（零前缀重叠）时，RadixAttention 的优势消失，性能与 vLLM 持平或略低

### 成熟度

- 项目较 vLLM 年轻，API 稳定性仍在演进中
- 部分高级功能（PD 分离、CP）仍处于快速迭代阶段

---

## 发展历程

2023.12 论文发布 --> 2024.01 RadixAttention 博客 --> 2024.02 压缩 FSM --> 2024.10 NeurIPS 发表 --> 2024.12 v0.4（零开销调度器） --> 2025.01 DeepSeek V3/R1 首日支持 --> 2025.03 加入 PyTorch 生态 --> 2025.05 96 H100 PD+EP 部署 --> 2025.10 SGLang-Jax TPU 初步支持 --> 2025.12 EAGLE-3 draft 模型

---

## 参考文献

- Zheng, L., Yin, L., Xie, Z., et al. **"SGLang: Efficient Execution of Structured Language Model Programs."** NeurIPS 2024. [arXiv:2312.07104](https://arxiv.org/abs/2312.07104)
- LMSYS Blog. **"Fast and Expressive LLM Inference with RadixAttention and SGLang."** 2024-01-17. [链接](https://www.lmsys.org/blog/2024-01-17-sglang/)
- LMSYS Blog. **"Fast JSON Decoding for Local LLMs with Compressed Finite State Machine."** 2024-02-05. [链接](https://www.lmsys.org/blog/2024-02-05-compressed-fsm/)
- LMSYS Blog. **"SGLang v0.4: Zero-Overhead Batch Scheduler, Cache-Aware Load Balancer."** 2024-12-04. [链接](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/)
- LMSYS Blog. **"Deploying DeepSeek with PD Disaggregation and Large-Scale EP on 96 H100s."** 2025-05-05. [链接](https://www.lmsys.org/blog/2025-05-05-large-scale-ep/)
- Dong, Y., et al. **"XGrammar: Flexible and Efficient Structured Generation Engine."** [arXiv:2411.15100](https://arxiv.org/abs/2411.15100)
- Ye, Z., et al. **"FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving."** [arXiv:2501.01005](https://arxiv.org/abs/2501.01005)

---

## 相关页面

- [[vllm]] — 替代推理引擎，更广泛的硬件支持
- [[continuous-batching]] — SGLang 使用的连续批处理技术
- [[kv-cache-optimization]] — KV 缓存技术总览（PagedAttention、前缀缓存等）
- [[structured-output-serving]] — 约束解码与结构化输出
- [[multi-turn-optimization]] — 多轮对话优化，SGLang 擅长领域
- [[prefill-decode-disaggregation]] — PD 分离部署策略
- [[speculative-decoding]] — 投机解码技术（EAGLE 等）
- [[flashinfer]] — FlashInfer 注意力内核库
