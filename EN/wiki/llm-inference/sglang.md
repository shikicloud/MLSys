---
title: "SGLang: Fast Structured Generation and Serving"
category: llm-inference
tags: [sglang, radix-attention, structured-generation, inference-engine, constrained-decoding]
created: 2026-04-13
updated: 2026-05-20
status: mature
---

# SGLang: Fast Structured Generation and Serving

> [!abstract]+ TL;DR
> SGLang (Structured Generation Language), from LMSYS (UC Berkeley), is a co-design of a **frontend DSL** and a **backend runtime (SRT)**: **RadixAttention** (radix-tree-driven cross-request KV reuse), Compressed FSM (jump-forward constrained decoding), XGrammar integration. Released December 2023, accepted at NeurIPS 2024. As of 2026 it powers xAI (Grok 3), Microsoft Azure, LinkedIn, Cursor — deployed across 400k+ GPUs in production. Throughput on H100 is **29% higher** than [[vllm|vLLM]] (16,200 vs 12,500 tok/s), and as much as **6.4×** higher on prefix-heavy workloads. It shines in multi-turn chat, structured output, and agent tool-use scenarios.

**Key features**:
- **RadixAttention**: radix-tree-based automatic cross-request [[kv-cache-optimization|KV cache]] reuse
- **Compressed finite-state machine**: faster constrained decoding via jump-forward optimization
- **Frontend DSL**: declarative primitives for LLM programming (gen, select, fork, ...)
- **Highly parallel**: combines five parallelism strategies (TP / PP / DP / EP / CP)
- **XGrammar integration**: high-performance structured-output backend

**Position in the ecosystem**: SGLang and [[vllm|vLLM]] are currently the two most prominent open-source LLM inference engines. SGLang leads on prefix-heavy workloads (29% higher throughput), while vLLM wins on hardware coverage and community size. By 2026 SGLang has been deployed at scale by xAI (Grok 3), Microsoft Azure, LinkedIn, Cursor and others, spanning over 400,000 GPUs.

[GitHub](https://github.com/sgl-project/sglang) | [Paper](https://arxiv.org/abs/2312.07104)

---

## Core architecture

SGLang uses a frontend-backend co-design:

```
+------------------------------------------------------------------+
|                     User application layer                       |
|  @sgl.function-decorated Python programs / OpenAI-compatible API |
+------------------------------------------------------------------+
         |                                    |
         v                                    v
+--------------------+          +-----------------------------+
|   SGLang frontend  |          |     HTTP / gRPC gateway     |
|   DSL              |          |  (FastAPI, OpenAI-compat)   |
| - gen() / select() |          +-----------------------------+
| - fork() / join()  |                    |
| - extend (+=)      |                    v
| - role wrappers    |   +-----------------------------------+
+--------------------+   |        TokenizerManager            |
         |               |   (separate process, tokenize/detok)|
         v               +-----------------------------------+
+--------------------+                    |
|   IR / interpreter |                    | ZeroMQ IPC
|  (dataflow compile)|                    v
+--------------------+   +-----------------------------------+
         |               |          Scheduler                 |
         |               |   - cache-aware scheduling         |
         +-------------->|   - RadixCache management          |
                         |   - memory budget control          |
                         |   - batch assembly (ScheduleBatch) |
                         +-----------------------------------+
                                          |
                                          v
                         +-----------------------------------+
                         |         ModelRunner / Worker       |
                         |   - FlashInfer / Triton attn       |
                         |   - CUDA Graph optimization        |
                         |   - ForwardBatch GPU execution     |
                         +-----------------------------------+
                                          |
                                          v
                         +-----------------------------------+
                         |           GPU cluster              |
                         |   TP / PP / DP / EP / CP           |
                         +-----------------------------------+
```

### Multi-process architecture

SGLang adopts a multi-process design to sidestep the Python GIL:

| Component | Process | Responsibility |
|------|------|------|
| **Server** | Main | FastAPI, routes HTTP/gRPC requests |
| **TokenizerManager** | Separate | Tokenize/detokenize, CPU-bound |
| **Scheduler** | Separate | GPU memory management, batch scheduling, RadixCache |
| **ModelRunner** | Separate | Model forward, GPU-bound |
| **DetokenizerManager** | Separate | Output detokenization, streaming back |

Inter-process communication uses **ZeroMQ** with dedicated ports: `tokenizer_ipc`, `scheduler_input_ipc`, `detokenizer_ipc`.

### Request flow

```
User request --> FastAPI --> TokenizerManager (tokenize) --ZMQ-->
Scheduler (prefix match, batch assembly) --> ModelRunner (GPU forward) -->
Scheduler (sampling, update RadixCache) --ZMQ--> DetokenizerManager
(detokenize) --> streamed back
```

Data structures transform step by step: **Req** --> **ScheduleBatch** (CPU) --> **ModelWorkerBatch** (Worker) --> **ForwardBatch** (GPU).

---

## RadixAttention

RadixAttention is SGLang's signature innovation: **automatic cross-request KV-cache reuse**.

### Problem background

Traditional inference engines (including early vLLM) discard a request's KV cache once it finishes. This means:
- In multi-turn chat, every turn re-prefills the entire prior history
- Different requests sharing the same system prompt cannot share its cache
- Few-shot examples are recomputed for every request

### Radix tree data structure

A **radix tree** is a space-efficient variant of a trie. Unlike a standard trie, a radix tree's edges can be labeled with **variable-length sequences** (not just single elements), which sharply improves storage efficiency.

SGLang uses a radix tree as the mapping from **token sequences --> KV cache tensors**:

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

  green = newly inserted node
  blue  = cache hit (reuse KV cache)
  red   = LRU-evicted node

Each node stores:
  - the token sequence segment (edge label)
  - reference to the corresponding KV cache tensor (paged GPU memory)
  - reference count + last access timestamp
```

### Workflow

1. **Prefix matching**: when a new request arrives, walk the radix tree from the root to find the longest matching cached prefix
2. **Incremental compute**: prefill only starts after the match point (hit = skip the cached portion)
3. **Cache insertion**: the newly computed KV cache is inserted as a new node
4. **LRU eviction**: when GPU memory is tight, recursively evict leaf nodes with refcount zero (least recently used)
5. **Cache-aware scheduling**: requests with the longest prefix match are scheduled first (a near-optimal DFS order)

### vs vLLM's hash-based prefix cache

| Aspect | SGLang RadixAttention | vLLM hash prefix cache |
|------|----------------------|-------------------|
| **Granularity** | Token level (1 token = 1 page) | Block level (e.g. 16 tokens/block) |
| **Match method** | Tree traversal, prefix-native | Hash lookup, requires full-block match |
| **Configuration** | Zero-config, automatic | Manual `--enable-prefix-caching` |
| **Scheduler coupling** | Cache-aware scheduler native | Scheduler and cache independent |
| **Memory overhead** | Radix tree itself uses CPU memory | Hash table is lightweight |
| **Best for** | Prefix-heavy (multi-turn, RAG) | Good enough for generic loads |

### Four reuse patterns

RadixAttention automatically discovers the following four KV-cache reuse patterns:

1. **Few-shot example sharing**: many requests share the same few-shot prefix
2. **Self-consistency sampling**: many samples of the same question share the question prefix
3. **Multi-turn dialogue history**: later turns reuse all prior turns' KV caches
4. **Tree-of-Thought search**: search-tree branches share the ancestor path

### Cache hit rates

Measured hit rates:
- Few-shot tasks: **85-95%**
- Multi-turn chat: **75-90%**
- RAG workloads: **50-80%** (depends on document reuse)
- Overall benchmarks: **50-99%**

### Performance impact

Ablations show RadixAttention introduces **no measurable overhead** even at zero hit rate. It composes cleanly with [[continuous-batching|continuous batching]] and [[kv-cache-optimization#PagedAttention|PagedAttention]].

---

## SGLang programming language / DSL

The frontend is a Python-embedded DSL with declarative primitives for programming complex LLM applications.

### Core primitives

| Primitive | Syntax | Purpose |
|------|------|------|
| **extend** | `s += "text"` | Append text to the current prompt |
| **gen** | `gen(name, ...)` | Call the LLM to generate; non-blocking, result stored in a variable |
| **select** | `gen(name, choices=[...])` | Pick the highest-probability option from a list |
| **fork** | `s.fork(n)` | Create n parallel execution branches |
| **Role wrappers** | `s.system()` / `s.user()` / `s.assistant()` | Automatically format chat templates |

### gen() parameters

```python
gen(
    name="variable_name",    # Variable name; retrieve via s["variable_name"]
    max_tokens=512,          # Max generated tokens
    stop="\n",               # Stop sequence
    temperature=0.7,         # Sampling temperature (0 = deterministic)
    regex=r"pattern",        # Regex constraint
    choices=["A", "B"],      # Option constraint (select mode)
)
```

### @sgl.function decorator

SGLang programs are defined with the `@sgl.function` (or `@function`) decorator. The first argument `s` is the state object that tracks prompt history and generated variables.

### Execution modes

- **Interpreter mode**: execute primitives step-by-step via `RuntimeEndpoint`
- **Compiler mode**: compile the program to a dataflow graph (IR) and enable optimizations (async execution, batch merging)

### Code examples

#### Basic Q&A

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

#### Multi-turn chat

```python
@function
def multi_turn(s):
    s += system("You are a helpful assistant.")
    s += user("List 3 countries and their capitals.")
    s += assistant(gen("first_answer", max_tokens=256))
    s += user("Now list 3 more, different from the above.")
    s += assistant(gen("second_answer", max_tokens=256))
    # The second turn automatically reuses the first turn's KV cache (RadixAttention)
```

#### Conditional branching (tool selection)

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

#### Parallel branches (fork)

```python
@function
def parallel_tips(s):
    s += user("Give me 3 tips for learning Python.")
    s += assistant(gen("tips", max_tokens=128))

    forks = s.fork(3)
    for i, f in enumerate(forks):
        f += user(f"Expand tip {i+1} in detail.")
        f += assistant(gen("detail", max_tokens=256))

    # Collect results from all branches
    for i, f in enumerate(forks):
        print(f"Tip {i+1} detail:", f["detail"])
```

#### Regex-constrained generation

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

#### Batch inference

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

#### Streaming output

```python
state = basic_qa.run(
    question="Explain quantum computing.",
    temperature=0.1,
    stream=True
)
for chunk in state.text_iter():
    print(chunk, end="", flush=True)
```

### Vs. raw API

The SGLang DSL converts imperative API calls into declarative programs; the backend manages KV cache reuse automatically:

```python
# Raw API: full history transferred each call, no KV cache reuse
# SGLang DSL: declarative, KV cache preserved across gen() calls automatically
@function
def dialog(s, q1, q2):
    s += system("...")
    s += user(q1)
    s += assistant(gen("a1"))   # KV cache automatically retained
    s += user(q2)
    s += assistant(gen("a2"))   # Automatically reuses a1's KV cache
```

---

## Constrained decoding

SGLang has two core techniques for constrained decoding: **Compressed FSM** and **XGrammar integration**.

### Principle

Constrained decoding ensures the LLM output conforms to a predefined format (JSON schema, regex, etc.). At every decode step, the legal token set is computed from the current FSM state and a logit bias mask is applied to illegal tokens.

```
Traditional constrained decoding flow:

JSON Schema --> regex --> finite-state machine (FSM)
                              |
                              v
each decode step: FSM state --> legal token set
                              --> logit masking
                              --> sample --> update FSM state
```

### Compressed FSM

SGLang's key innovation is the **Compressed FSM**: analyze adjacent singular-transition edges in the FSM and collapse contiguous deterministic transitions into a single edge.

```
Original FSM (token level):
  S0 --{--> S1 --"--> S2 --n--> S3 --a--> S4 --m--> S5 --e--> S6 --"--> S7

Compressed FSM:
  S0 ------{"name"------> S7

  (the entire {"name" sequence finished in one step)
```

**Effect**: when the FSM is on a deterministic path, **multiple tokens are decoded in one step** without a per-token LLM forward.

### Jump-forward decoding

Jump-forward is the runtime optimization that exploits the compressed FSM:

1. When the current FSM state enters a deterministic path
2. **Directly prefill** the entire deterministic token sequence (rather than decoding token by token)
3. Use RadixAttention's extend primitive to reuse existing KV cache
4. Mechanism: terminate the current request, enqueue a new request with the extended prefix

```
Plain decoding:  [prompt] -> t1 -> t2 -> t3 -> t4 -> ... (token by token)
Jump-forward:    [prompt] -> [{"name":] -> t_free -> t_free -> ... (skip deterministic parts)
                              ^^^^^^^^^
                              one prefill covers it
```

### XGrammar integration

Since v0.4, SGLang uses **XGrammar** as the default structured-output backend. XGrammar is an independent high-performance constrained-decoding library that supports:

| Constraint type | Example |
|---------|------|
| **JSON Schema** | `{"type": "object", "properties": {...}}` |
| **Regex** | `r"[A-Z][a-z]+ \d{4}"` |
| **EBNF grammar** | Custom BNF grammar rules |
| **Structured tags** | think/answer tag constraints for reasoning models |

XGrammar's core advantage: it **overlaps** grammar mask generation with the LLM forward pass, eliminating the latency tax of constrained decoding.

### Performance

| Metric | SGLang + XGrammar | Traditional |
|------|-------------------|---------|
| JSON decoding latency | **2x lower** | Baseline |
| JSON decoding throughput | **2.5x higher** | Baseline |
| JSON compliance rate | **96-99.8%** | 90-94% |
| Vs. unconstrained decoding | Constrained decoding is **faster** | Constrained decoding is slower |

Key finding: with SGLang's compressed FSM + jump-forward, constrained decoding is actually faster than unconstrained decoding because deterministic-token LLM compute is skipped.

### JSON-constrained generation example

```python
# Method 1: DSL + regex constraint
@sgl.function
def json_gen(s, prompt):
    s += sgl.user(prompt)
    s += sgl.assistant(sgl.gen("output", max_tokens=256,
        regex=r'\{"name": "[^"]+", "age": \d+, "city": "[^"]+"\}'))

# Method 2: OpenAI-compatible API + JSON schema
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

## FlashInfer integration

SGLang deeply integrates **FlashInfer** as its default attention backend — a key pillar of its performance.

### What is FlashInfer

FlashInfer is a high-performance attention-kernel library purpose-built for LLM inference, providing:
- **PagedAttention kernels**: optimized paged-KV-cache attention
- **Block-sparse attention**: implements PagedAttention as a block-sparse attention kernel
- **Customizable variants**: custom attention variants via Jinja templates and JIT compilation

### Core optimizations

| Optimization | Description |
|------|------|
| **GPU shared-memory prefetch** | Prefetch page indices in GPU shared memory; eliminates page-size sensitivity |
| **JIT compilation** | Compile custom attention kernels via PyTorch JIT, zero extra dev cost |
| **FlexAttention inspired** | Users define attention variants via functors (LogitsTransform/QueryTransform) |
| **FP8 support** | Native FP8 KV-cache attention |

### Performance impact

FlashInfer vs. Triton backend inside SGLang (Llama-3 8B/70B):
- **Median ITL (inter-token latency) 29-69% lower**
- **TTFT up to 21% lower**
- Supports 1-token-per-page fine-grained paging (foundation for RadixAttention)

SGLang also supports **FlashAttention 3** as an alternative backend, which can deliver further wins in certain settings.

---

## Parallelism strategies

SGLang supports five orthogonal parallelism strategies that can be combined freely:

```
Total GPUs = TP_size × PP_size × EP_size × DP_size

Example: 32-GPU deployment
  --tp 4 --pp 2 --ep 2 --dp 2
  Each GPU is assigned rank indices across multiple dimensions
```

### Per-strategy notes

| Strategy | Flag | Description | When to use |
|------|------|------|---------|
| **Tensor parallelism (TP)** | `--tp N` | Horizontally shard weight matrices, all-reduce sync | Single model doesn't fit one GPU |
| **Pipeline parallelism (PP)** | `--pp N` | Layers split across GPUs | Very large models, cross-node |
| **Data parallelism (DP)** | `--dp N` | Multi-replica parallel serving for throughput | Memory available, maximize throughput |
| **Expert parallelism (EP)** | `--ep N` | MoE experts distributed across GPUs | DeepSeek V3/R1 and other MoE models |
| **Context parallelism (CP)** | `--cp N` | Sequence-dimension sharding (very long context) | Very long document processing |

### Expert parallelism (EP)

EP is essential for MoE models (e.g. DeepSeek V3/R1):
- Expert weights are distributed across GPUs and tokens are routed via **all-to-all communication**
- Optimized **grouped GEMMs** reduce GPU idle time
- SGLang ships the first open-source DeepSeek V3/R1 EP + PD-disaggregation deployment

### Multi-node deployment

```bash
# Two-node TP=8 deployment
# Node 0 (master)
python -m sglang.launch_server --model meta-llama/Llama-3.1-70B \
    --tp 8 --nnodes 2 --node-rank 0 --master-addr <IP>

# Node 1
python -m sglang.launch_server --model meta-llama/Llama-3.1-70B \
    --tp 8 --nnodes 2 --node-rank 1 --master-addr <IP>
```

---

## Performance analysis

### Throughput (H100)

| Engine | Standard throughput (tok/s) | Prefix-heavy throughput | DeepSeek V3 |
|------|---------------------|---------------|-------------|
| **SGLang** | **16,200** | **6.4x baseline** | **3.1x vs vLLM** |
| vLLM | 12,500 | Baseline | Baseline |
| TensorRT-LLM | ~14,000 | - | - |

### Latency (Llama 3.1 8B, H100)

| Metric | SGLang | vLLM |
|------|--------|------|
| **TTFT (first-token latency)** | **79 ms** | 103 ms |
| **ITL (inter-token latency)** | **6.0 ms** | 7.1 ms |
| **ITL range** | 4-21 ms (most stable) | Larger variance |
| **Output throughput** | **894 tok/s** | 413 tok/s |

### Structured output

| Metric | SGLang + XGrammar | Traditional guided decoding |
|------|-------------------|-------------|
| Throughput | **4,200 tok/s** | ~1,400 tok/s |
| JSON compliance rate | **99.8%** | 90-94% |
| Latency | **0.4s** | ~1.2s |

### Large-scale disaggregated deployment (96 H100)

SGLang pioneered open-source DeepSeek V3/R1 PD disaggregation + EP deployment:
- **Input throughput**: 52,300 tok/s per node
- **Output throughput**: 22,300 tok/s per node
- vs. plain TP deployment: **5x improvement**

### GPU utilization (v0.4+)

Zero-overhead batch scheduler: **GPU utilization 95-98%** (vs. traditional 70-80%), CPU overhead <2% (vs. traditional 15-25%).

---

## Code examples

### Server launch and basic calls

```bash
# Basic launch
python -m sglang.launch_server \
    --model meta-llama/Llama-3.1-8B-Instruct --port 30000

# Multi-GPU: TP + DP
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

### Multi-turn chat (automatic prefix-cache reuse)

```python
# SGLang automatically reuses prefix KV cache via RadixAttention — no extra config
messages = [{"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "What is a binary tree?"}]
r1 = client.chat.completions.create(model="...", messages=messages)

messages.append({"role": "assistant", "content": r1.choices[0].message.content})
messages.append({"role": "user", "content": "How to implement it in Python?"})
r2 = client.chat.completions.create(model="...", messages=messages)
# The second turn matches the prefix automatically and skips prefill of the cached part
```

### Python Engine API (offline batch inference)

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

## SGLang vs vLLM detailed comparison

| Dimension | SGLang | vLLM |
|------|--------|------|
| **Core optimization** | RadixAttention (radix tree) | PagedAttention (paged memory) |
| **KV cache reuse** | Automatic cross-request (zero config) | Manual via `--enable-prefix-caching` |
| **Cache granularity** | Token level | Block level (default 16 tokens) |
| **Scheduling policy** | Cache-aware scheduling | FIFO continuous batching |
| **Structured output** | Compressed FSM + XGrammar (~3x faster) | XGrammar / Outlines |
| **Frontend DSL** | Yes (gen/select/fork primitives) | None (pure API calls) |
| **Parallelism strategies** | TP + PP + DP + EP + CP | TP + PP + DP + EP |
| **H100 throughput** | **16,200 tok/s** | 12,500 tok/s |
| **Prefix-heavy workloads** | **Up to 6.4x** | Baseline |
| **DeepSeek performance** | **3.1x faster** | Baseline |
| **Hardware support** | NVIDIA, AMD (limited) | NVIDIA, AMD, TPU, Trainium, Gaudi |
| **Model coverage** | Decoder, multi-modal, MoE | Decoder, encoder-decoder, multi-modal, MoE |
| **Community** | ~25K stars, 600 contributors | ~75K stars, 2,400 contributors |
| **TTFT** | **79 ms** | 103 ms |
| **ITL** | **6.0 ms** | 7.1 ms |
| **Speculative decoding** | EAGLE-2/3 | Eagle, Medusa, multiple methods |
| **PD disaggregation** | Native (first-class API) | Experimental |

### Selection guide

- **Pick SGLang**: multi-turn chat, RAG, structured output, DeepSeek deployment, agent workloads, peak throughput
- **Pick vLLM**: standalone prompt batching, non-NVIDIA hardware, encoder-decoder models, broadest model support, large community

---

## Deployment in practice

### Key server parameters

```yaml
# config.yaml example
model: meta-llama/Llama-3.1-70B-Instruct
port: 30000
host: 0.0.0.0

# Parallelism
tp: 4
dp: 2

# Memory
mem_fraction_static: 0.85        # Fraction of GPU memory used for KV cache
max_running_requests: 256        # Max concurrent requests

# Batching
max_num_reqs: 1024               # Max queued requests
schedule_policy: lpm             # lpm = longest prefix match

# Quantization
quantization: fp8                # supports fp4/fp8/int4

# Speculative decoding
speculative_algorithm: EAGLE     # EAGLE-2/3
speculative_num_steps: 3         # Speculative depth
speculative_eagle_topk: 4        # Candidates per step
```

### Deployment patterns

```bash
# Single node, multiple GPUs
python -m sglang.launch_server \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --tp 4 --dp 2 --mem-fraction-static 0.85

# PD disaggregation (prefill node)
python -m sglang.launch_server \
    --model deepseek-ai/DeepSeek-V3 \
    --tp 8 --ep 4 --disaggregation-mode prefill --port 30000

# PD disaggregation (decode node)
python -m sglang.launch_server \
    --model deepseek-ai/DeepSeek-V3 \
    --tp 8 --ep 4 --disaggregation-mode decode --port 30001

# Docker
docker run --gpus all -p 30000:30000 lmsysorg/sglang:latest \
    python -m sglang.launch_server \
    --model meta-llama/Llama-3.1-8B-Instruct --host 0.0.0.0
```

### Production tuning suggestions

| Parameter | Suggested value | Note |
|------|--------|------|
| `mem_fraction_static` | 0.80-0.90 | Leave room for weights and scratch buffers |
| `schedule_policy` | `lpm` | Longest prefix match, maximize cache hits |
| `max_running_requests` | Tune to GPU memory | Too high → OOM, too low → wasted throughput |
| `chunked_prefill_size` | 8192 | Chunked prefill for long inputs, avoid stalls |
| `disable_radix_cache` | No (keep enabled) | Only consider disabling if no shared prefixes |

---

## Other notable features

| Feature | Description |
|------|------|
| **[[prefill-decode-disaggregation\|PD disaggregation]]** | Prefill nodes (compute-bound) and decode nodes (memory-bound) deployed separately, KV cache transferred between them |
| **[[speculative-decoding\|Speculative decoding]]** | EAGLE-2/3, speculation depth 3-5, branching 4-8; 1.4x throughput uplift measured on DeepSeek |
| **Multi-modal** | Vision-language models supported (DSL provides the `sgl.image()` primitive) |
| **Quantization** | FP4, FP8, INT4 (AWQ/GPTQ); launch flag `--quantization fp8` |
| **Multi-LoRA** | One engine serves many LoRA adapters concurrently |

---

## Limitations

### Limited hardware support

- **Primarily NVIDIA GPUs**; AMD support is improving but still less mature than vLLM
- **No TPU, Trainium, or Gaudi support** (vLLM has them)
- TPU support has early progress via the SGLang-Jax backend (October 2025)

### Model architecture coverage

- **No encoder-decoder models** (T5, BART, ...)
- Smaller model support list than vLLM

### Community and ecosystem

- ~25K GitHub stars (vLLM ~75K), ~600 contributors (vLLM ~2,400)
- Issue response time 3-5 days (vLLM 12 hours)
- Less docs and tutorials than vLLM

### Technical limits

- **Python GIL bottleneck**: at high concurrency, the CPU routing pipeline can hit a single-core GIL ceiling, limiting multi-threaded scaling
- **RadixAttention memory overhead**: when prefix overlap is low, the radix-tree cache's GPU memory cost may not pay off
- **No-shared-prefix scenarios**: when all requests are fully independent (zero prefix overlap), RadixAttention's edge disappears, performance equals or slightly trails vLLM

### Maturity

- Younger project than vLLM, API stability still evolving
- Some advanced features (PD disaggregation, CP) remain in rapid iteration

---

## Timeline

2023.12 paper release --> 2024.01 RadixAttention blog --> 2024.02 Compressed FSM --> 2024.10 NeurIPS publication --> 2024.12 v0.4 (zero-overhead scheduler) --> 2025.01 DeepSeek V3/R1 day-one support --> 2025.03 joined PyTorch ecosystem --> 2025.05 96 H100 PD+EP deployment --> 2025.10 SGLang-Jax initial TPU support --> 2025.12 EAGLE-3 draft model

---

## References

- Zheng, L., Yin, L., Xie, Z., et al. **"SGLang: Efficient Execution of Structured Language Model Programs."** NeurIPS 2024. [arXiv:2312.07104](https://arxiv.org/abs/2312.07104)
- LMSYS Blog. **"Fast and Expressive LLM Inference with RadixAttention and SGLang."** 2024-01-17. [link](https://www.lmsys.org/blog/2024-01-17-sglang/)
- LMSYS Blog. **"Fast JSON Decoding for Local LLMs with Compressed Finite State Machine."** 2024-02-05. [link](https://www.lmsys.org/blog/2024-02-05-compressed-fsm/)
- LMSYS Blog. **"SGLang v0.4: Zero-Overhead Batch Scheduler, Cache-Aware Load Balancer."** 2024-12-04. [link](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/)
- LMSYS Blog. **"Deploying DeepSeek with PD Disaggregation and Large-Scale EP on 96 H100s."** 2025-05-05. [link](https://www.lmsys.org/blog/2025-05-05-large-scale-ep/)
- Dong, Y., et al. **"XGrammar: Flexible and Efficient Structured Generation Engine."** [arXiv:2411.15100](https://arxiv.org/abs/2411.15100)
- Ye, Z., et al. **"FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving."** [arXiv:2501.01005](https://arxiv.org/abs/2501.01005)

---

## Related pages

- [[vllm]] — Alternative inference engine with broader hardware support
- [[continuous-batching]] — Continuous batching technique used by SGLang
- [[kv-cache-optimization]] — KV cache techniques overview (PagedAttention, prefix cache, ...)
- [[structured-output-serving]] — Constrained decoding and structured output
- [[multi-turn-optimization]] — Multi-turn dialogue optimization, SGLang's home turf
- [[prefill-decode-disaggregation]] — PD disaggregation deployment strategy
- [[speculative-decoding]] — Speculative decoding (EAGLE, ...)
- [[flashinfer]] — FlashInfer attention-kernel library
