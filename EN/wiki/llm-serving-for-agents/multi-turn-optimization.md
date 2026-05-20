---
title: "Multi-Turn Conversation Serving Optimization"
category: llm-serving-for-agents
tags: [multi-turn, kv-cache-reuse, prefix-caching, session-management, lmcache, prompt-caching, context-management, sticky-sessions]
created: 2026-04-13
updated: 2026-05-20
status: mature
---

# Multi-Turn Conversation Serving Optimization

> [!abstract]+ TL;DR
> Agent sessions are inherently multi-turn. Each turn appends LLM output + tool result to the conversation history, and every LLM call must re-process the full context. **Without optimization, prefill cost grows quadratically with turns**; KV cache reuse lets each turn process only the newly added tokens — linear scaling. Core systems: **[[vllm|vLLM]]** (block-level prefix cache), **[[sglang|SGLang]] RadixAttention** (token-level radix tree, multi-turn throughput 10% above vLLM), **LMCache** (cross-engine KV sharing across GPU/CPU/disk/S3, **15× throughput / 2× lower latency**), **Continuum** (KV cache TTL across tool-execution windows, 1.12–3.66× latency reduction).

## Overview

Agent sessions are inherently multi-turn. Every step appends new content (LLM output + tool result) to the conversation history, and every LLM call must re-process the entire context. **Without optimization, prefill cost grows quadratically with the number of turns**.

### Essence of the Multi-Turn Problem

```
Turn 1: [sys prompt + user request]
         |---- prefill 2000 tokens ----|

Turn 2: [sys prompt + user request + T1 output + tool result]
         |---- prefill 5000 tokens -------------------|

Turn 3: [sys prompt + user request + T1 + T2 + tool result]
         |---- prefill 10000 tokens --------------------------------|

Turn N: [all accumulated history]
         |---- prefill grows and grows ──────────────────────────────|

Prefill cost growth:
  Tokens
  │
50k│                                          ●
40k│                                    ●
30k│                              ●
20k│                        ●
15k│                  ●
10k│            ●
 5k│      ●
 2k│●
   └──────────────────────────────────────────
    T1  T2  T3  T4  T5  T6  T7  T8  T9  T10

  Total prefill cost without optimization:
  C_total = Σ(i=1 to N) c_i ≈ O(N^2)

  because c_i (the i-th turn's prefill length) grows roughly linearly
```

### Core Optimization Idea

```
Goal: drop O(N^2) toward O(N)

  Key observation:
  Turn i input = Turn (i-1) input + Turn (i-1) output + new content

  i.e., each turn's input is a prefix extension of the previous turn's input

  If the previous turn's KV cache is retained:
  Turn i only needs to process the newly added tokens (typically <2000)
  rather than reprocessing the entire context (potentially >20000)

  After optimization:
  Per-turn prefill cost ≈ constant (only new tokens)
  Total cost C_total ≈ O(N)
```

---

## Cross-Turn KV Cache Reuse

### Mechanics

```
KV cache reuse:

Turn 1:
  Input: [sys_prompt][user_msg_1]
  KV:    [████████████████████]  (2000 tokens)
  Output: [assistant_reply_1]

Turn 2:
  Input: [sys_prompt][user_msg_1][assistant_reply_1][tool_result_1][user_msg_2]
          ^^^^^^^^^^^^^^^^^^^^^^^^ identical to Turn 1 (prefix match)

  If KV cache is retained:
  KV:    [████████████████████][■■■■■■■■■■■■■■■]
          ^cached (reused)^      ^new compute (1500 tokens)^

  Savings: 2000 / 3500 = 57% of prefill compute

Turn 5 (15000 tokens accumulated):
  With cache: only ~1500 new tokens
  Without:    all 15000 tokens
  Savings: 90%
```

### Core Tension

```
Retain vs. evict trade-off:

  Retain KV cache:
  ✓ Next-turn TTFT drops sharply
  ✗ Holds GPU memory (~0.5-2 MB per token)
  ✗ Fewer concurrent requests can be served
  ✗ Memory fragmentation

  Evict KV cache:
  ✓ Frees GPU memory
  ✓ More concurrent requests
  ✗ Next turn must fully re-prefill
  ✗ Latency adds seconds
  ✗ GPU wasted on recompute

  Memory-footprint estimate (LLaMA 70B, FP16):
  - KV cache per token: ~1.25 MB
  - 10000-token session: ~12.5 GB
  - 100 concurrent sessions: ~1.25 TB (far beyond a single GPU)

  → Smart cache management is mandatory
```

---

## Prefix-Cache Implementations

### Comparison of Major Systems

| System | Approach | Granularity | Pro | Con |
|--------|----------|-------------|-----|-----|
| [[vllm\|vLLM]] | Block-level hashing (APC) | Block (16 tokens) | Great for templated prompts, simple | Block-boundary alignment |
| [[sglang\|SGLang]] | Token-level radix tree (RadixAttention) | Token | Auto-discovers cache opportunities, ~10% better on multi-turn | Tree maintenance overhead |
| LMCache | Cross-engine connector | Flexible | Tiered storage (GPU/CPU/disk/S3) | Extra component |
| TensorRT-LLM | KV cache reuse | Block | NVIDIA-native optimization | NVIDIA ecosystem lock-in |

### vLLM Automatic Prefix Caching (APC)

```
vLLM APC mechanics:

  1. Partition KV cache into fixed-size blocks (typically 16 tokens)
  2. Hash the content of each block
  3. For a new request, if a prefix block's hash matches → reuse directly

  Example:
  Request A: [sys_prompt | user_1 | asst_1 | user_2]
  Blocks:    [block_0    | block_1| block_2| block_3]
  Hashes:    [h0=abc     | h1=def | h2=ghi | h3=jkl]

  Request B: [sys_prompt | user_1 | asst_1 | user_3]
  Blocks:    [block_0    | block_1| block_2| block_3']
  Hashes:    [h0=abc     | h1=def | h2=ghi | h3=xyz]
              ^match^     ^match^  ^match^   ^no-match^

  Result: first 3 blocks' KV cache reused; only block_3' is computed

Enable:
  vllm serve model_name --enable-prefix-caching
```

### SGLang RadixAttention

```
SGLang radix tree:

  RadixAttention stores all requests' KV cache in a radix tree:

           [root]
           /    \
     [sys_prompt] [sys_prompt_B]
         |
    [user_msg_1]
       /     \
  [asst_1]  [asst_1']   ← divergent reply branches
     |         |
  [user_2]  [user_2']
     |
  [asst_2]

  Lookup:
  1. The new request's token sequence walks down the tree
  2. Match the longest common prefix
  3. Compute new KV from the divergence point onward

  Advantages over vLLM APC:
  - Token-level granularity (vs. block-level) → more precise matches
  - Natively supports branching (beam search, multi-sample)
  - ~10% better on multi-turn conversation
  - Auto-discovers and exploits cache opportunities

Enable (on by default):
  python -m sglang.launch_server --model model_name
```

### LMCache Tiered Caching

```
LMCache architecture:

  ┌──────────────────────────────────────────────┐
  │                  LMCache                      │
  │                                               │
  │  Layer 1: GPU HBM (fastest, smallest)         │
  │  ┌──────────────────────────────┐             │
  │  │  Hot: KV of active sessions  │             │
  │  │  Latency: ~0.1 ms            │             │
  │  └──────────────┬───────────────┘             │
  │                 │ evict                        │
  │                 v                              │
  │  Layer 2: CPU DRAM (fast, medium)              │
  │  ┌──────────────────────────────┐             │
  │  │  Warm: recently-used sessions│             │
  │  │  Latency: ~1 ms              │             │
  │  └──────────────┬───────────────┘             │
  │                 │ evict                        │
  │                 v                              │
  │  Layer 3: Disk/NVMe (medium, large)            │
  │  ┌──────────────────────────────┐             │
  │  │  Cold: dormant but resumable │             │
  │  │  Latency: ~10 ms             │             │
  │  └──────────────┬───────────────┘             │
  │                 │ evict                        │
  │                 v                              │
  │  Layer 4: S3/object store (slow, largest)      │
  │  ┌──────────────────────────────┐             │
  │  │  Archive: all session KVs    │             │
  │  │  Latency: ~100 ms            │             │
  │  └──────────────────────────────┘             │
  │                                               │
  │  Perf: 15× throughput, 2× lower latency       │
  │  Integrations: vLLM, SGLang, KServe, NVIDIA Dynamo │
  └──────────────────────────────────────────────┘
```

```python
# LMCache usage example
from lmcache import LMCacheEngine

# Configure the tiered cache
cache = LMCacheEngine(
    layers=[
        {"type": "gpu", "capacity_gb": 4},
        {"type": "cpu", "capacity_gb": 32},
        {"type": "disk", "path": "/data/kv_cache", "capacity_gb": 500},
        {"type": "s3", "bucket": "kv-cache-store"},
    ],
    eviction_policy="lru",        # least recently used
    compression="fp8",            # KV cache compression
    chunk_size=256,               # token granularity
)

# Integrate with vLLM
from vllm import LLM
llm = LLM(
    model="meta-llama/Llama-3-70B",
    kv_cache_engine=cache,        # use LMCache to manage KV
    enable_prefix_caching=True,
)
```

---

## Multi-Turn Challenges in Disaggregated Architectures

In [[prefill-decode-disaggregation|P-D disaggregated architectures]], prefill nodes and decode nodes are separated, which raises special challenges for multi-turn KV cache reuse.

```
Multi-turn issue in P-D disaggregation:

  Turn 1:
  [Prefill node A] ──> [Decode node B]
                         KV cache lives on B

  Turn 2:
  [Prefill node A] needs Turn 1's KV cache
  But the KV cache is on Decode node B!

  Solutions:
  1. KV cache transfer: B -> A (network overhead)
  2. Recompute on A (wasted compute)
  3. Shared KV cache layer (e.g. LMCache)

┌──────────────────────────────────────────────┐
│         P-D disaggregation + multi-turn       │
│                                               │
│  Turn 1:                                      │
│  [Prefill A] ──KV──> [Decode B]              │
│       │                   │                   │
│       │       KV deposited to shared layer    │
│       │                   │                   │
│       v                   v                   │
│  ┌─────────────────────────────┐             │
│  │   Shared KV cache (LMCache) │             │
│  └─────────────────────────────┘             │
│       │                                       │
│  Turn 2:                                      │
│  [Prefill A'] ──load KV from shared──> [Decode B']│
│  (may be a different node)                    │
│                                               │
└──────────────────────────────────────────────┘
```

### Specialized Solutions

**PrefillShare (Feb 2026)**:
- Shared prefill module that enables cross-model KV cache reuse
- Multiple decode instances share prefill compute

**Cache-Aware P-D (CPD, Together AI)**:
- Introduces a pre-prefill node for cold starts
- Prefill node prioritizes requests with hot caches
- Cache-aware request routing

---

## Context-Window Management Strategies

When conversation history grows close to or beyond the context window, management strategies are required.

### Strategy Comparison

```
┌──────────────────────────────────────────────────────┐
│              Context-window management                │
│                                                       │
│  Strategy 1: Truncation                               │
│  ┌────────────────────────────────────────┐          │
│  │ [sys prompt][...drop old history...][last N turns]│
│  └────────────────────────────────────────┘          │
│  ✓ Simple   ✗ Loses important history                 │
│                                                       │
│  Strategy 2: Summarization                            │
│  ┌────────────────────────────────────────┐          │
│  │ [sys prompt][old summary][last N turns in full]   │
│  └────────────────────────────────────────┘          │
│  ✓ Keeps key info  ✗ Summary may miss details         │
│                                                       │
│  Strategy 3: Sliding window                           │
│  ┌────────────────────────────────────────┐          │
│  │ [sys prompt][sliding window: last K tokens]       │
│  └────────────────────────────────────────┘          │
│  ✓ Fixed memory  ✗ Out-of-window info fully lost      │
│                                                       │
│  Strategy 4: Hierarchical summary                     │
│  ┌────────────────────────────────────────┐          │
│  │ [sys prompt]                            │          │
│  │ [global summary: high-level of all conv]│          │
│  │ [mid-term summary: detailed for last 5-10 turns]  │
│  │ [recent verbatim: last 2-3 turns]       │          │
│  └────────────────────────────────────────┘          │
│  ✓ Best info retention  ✗ Most complex, extra LLM calls│
│                                                       │
│  Strategy 5: Memory-augmented                         │
│  ┌────────────────────────────────────────┐          │
│  │ [sys prompt][retrieved relevant history][curr turn]│
│  └────────────────────────────────────────┘          │
│  ✓ Dynamic relevance  ✗ Needs extra retrieval infra   │
└──────────────────────────────────────────────────────┘
```

### Implementation Example

```python
class ContextManager:
    """Context manager for multi-turn conversations."""

    def __init__(
        self,
        max_context_tokens: int = 128000,
        strategy: str = "hierarchical",
        reserve_for_output: int = 4096,
        recent_turns_to_keep: int = 3,
    ):
        self.max_tokens = max_context_tokens - reserve_for_output
        self.strategy = strategy
        self.recent_turns = recent_turns_to_keep
        self.full_history = []
        self.summaries = []  # hierarchical summaries

    def add_turn(self, role: str, content: str):
        """Add a new conversation turn."""
        self.full_history.append({"role": role, "content": content})
        self._maybe_compress()

    def get_context(self, system_prompt: str) -> list[dict]:
        """Return current context, fit to the window."""
        if self.strategy == "truncation":
            return self._truncation_context(system_prompt)
        elif self.strategy == "sliding_window":
            return self._sliding_window_context(system_prompt)
        elif self.strategy == "hierarchical":
            return self._hierarchical_context(system_prompt)
        elif self.strategy == "summarization":
            return self._summarization_context(system_prompt)

    def _truncation_context(self, system_prompt: str) -> list[dict]:
        """Naive truncation: keep system prompt + last N turns."""
        messages = [{"role": "system", "content": system_prompt}]
        recent = self.full_history[-self.recent_turns * 2:]
        messages.extend(recent)
        return messages

    def _sliding_window_context(self, system_prompt: str) -> list[dict]:
        """Sliding window: keep history within last K tokens."""
        messages = [{"role": "system", "content": system_prompt}]
        sys_tokens = count_tokens(system_prompt)
        remaining = self.max_tokens - sys_tokens

        selected = []
        for msg in reversed(self.full_history):
            msg_tokens = count_tokens(msg["content"])
            if remaining - msg_tokens < 0:
                break
            selected.insert(0, msg)
            remaining -= msg_tokens

        messages.extend(selected)
        return messages

    def _hierarchical_context(self, system_prompt: str) -> list[dict]:
        """Hierarchical: global summary + mid summary + recent verbatim."""
        messages = [{"role": "system", "content": system_prompt}]

        # Global summary (if any)
        if self.summaries:
            global_summary = self.summaries[-1]
            messages.append({
                "role": "system",
                "content": f"Conversation summary:\n{global_summary}"
            })

        # Recent verbatim
        recent = self.full_history[-self.recent_turns * 2:]
        messages.extend(recent)

        return messages

    def _maybe_compress(self):
        """Trigger compression once history is too long."""
        total = sum(count_tokens(m["content"]) for m in self.full_history)
        if total > self.max_tokens * 0.7:  # compress at 70%
            old_messages = self.full_history[:-self.recent_turns * 2]
            if old_messages:
                summary = self._summarize(old_messages)
                self.summaries.append(summary)
                # Keep recent history
                self.full_history = self.full_history[-self.recent_turns * 2:]

    def _summarize(self, messages: list[dict]) -> str:
        """Summarize old history with an LLM."""
        content = "\n".join(
            f"{m['role']}: {m['content']}" for m in messages
        )
        return llm.generate(
            f"Concisely summarize the key information from the conversation below:\n{content}"
        )
```

---

## Prompt Caching

Prompt caching is an **API-layer** optimization (provided by LLM vendors); it operates on a different layer than KV cache reuse.

### Prompt Caching Across Vendors

| Vendor | Feature | Granularity | Price benefit | TTL |
|--------|---------|-------------|---------------|-----|
| **Anthropic** | Prompt Caching | cache_control markers | 90% off on cache reads | 5 minutes |
| **OpenAI** | Automatic Caching | Auto prefix matching | 50% off on cache hits | ~5–10 minutes |
| **Google** | Context Caching | Explicit cache creation | 75% off on cache reads | Configurable |

### Anthropic Prompt Caching in Detail

```
Anthropic Prompt Caching mechanics:

  Request 1:
  ┌──────────────────────────────────────────────┐
  │ system: "You are an assistant..." (with cache_control) │ ← create cache
  │ user: "What is a KV cache?"                   │
  └──────────────────────────────────────────────┘

  Pricing: cache write $3.75/M (1.25x)

  Request 2 (within 5 min):
  ┌──────────────────────────────────────────────┐
  │ system: "You are an assistant..." (with cache_control) │ ← cache hit!
  │ user: "What is prefix caching?"               │
  └──────────────────────────────────────────────┘

  Pricing: cache read $0.30/M (0.1x) ← 90% off!

  Multi-turn application:
  ┌──────────────────────────────────────────────┐
  │ system: "..." (cached)                        │ ← 100% hit
  │ tools: [...] (cached)                         │ ← 100% hit
  │ conversation Turn 1-5 (cached)                │ ← prefix match
  │ new user message Turn 6                       │ ← new compute
  └──────────────────────────────────────────────┘

  Note: cache-break position matters
  - Put breakpoints before content that changes
  - System prompt + tool defs → best cache targets
  - Place dynamic content at the end
```

```python
# Anthropic prompt caching example
import anthropic

client = anthropic.Anthropic()

# Cache usage in multi-turn conversation
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=4096,
    system=[
        {
            "type": "text",
            "text": "You are a professional coding assistant... (long system prompt)",
            "cache_control": {"type": "ephemeral"}  # mark as cacheable
        }
    ],
    messages=[
        # Conversation history (can be marked as cached)
        {"role": "user", "content": "Help me write a sorting algorithm"},
        {"role": "assistant", "content": "Sure, here is quicksort..."},
        # Earlier history can be cached
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Help me write a sorting algorithm\nSure, here is quicksort...",
                    "cache_control": {"type": "ephemeral"}  # cache breakpoint
                }
            ]
        },
        # New message
        {"role": "user", "content": "Can you optimize space complexity?"},
    ]
)

# Inspect cache usage
print(f"Cache create: {response.usage.cache_creation_input_tokens}")
print(f"Cache read:   {response.usage.cache_read_input_tokens}")
print(f"New input:    {response.usage.input_tokens}")
```

### Prompt Caching vs. KV Cache Reuse

```
How the two caches relate:

  ┌──────────────────────────────────────────────────┐
  │                                                   │
  │  Prompt caching (API layer)                       │
  │  - Managed by the LLM vendor                      │
  │  - Cross-request prompt prefix matching           │
  │  - Price benefit (50-90% off)                     │
  │  - User just marks cache_control                  │
  │  - TTL: 5-10 minutes                              │
  │                                                   │
  │         ↕ Complementary, different layers          │
  │                                                   │
  │  KV cache reuse (inference-engine layer)          │
  │  - Managed by the inference engine (vLLM, SGLang) │
  │  - KV tensor reuse inside the engine              │
  │  - Perf benefit (lower TTFT)                      │
  │  - Transparent to the user                        │
  │  - TTL: depends on memory pressure                │
  │                                                   │
  │  Self-hosted: use KV cache reuse                  │
  │  Vendor API:  use prompt caching                  │
  │  Both benefit from prefix stability               │
  └──────────────────────────────────────────────────┘
```

---

## Session Management and Routing

### Sticky Sessions

To maximize KV cache hit rate, requests from the same session should route to the same inference node.

```
Non-sticky routing (round-robin):

  Turn 1 ──> [Node A] (computes KV cache)
  Turn 2 ──> [Node B] (KV cache not on B, recompute!)
  Turn 3 ──> [Node C] (KV cache not on C, recompute!)
  Turn 4 ──> [Node A] (KV cache likely evicted, recompute!)

  Result: full prefill every turn, zero cache reuse

Sticky routing (session affinity):

  Turn 1 ──> [Node A] (computes KV cache)
  Turn 2 ──> [Node A] (reuses KV cache, only new tokens!)
  Turn 3 ──> [Node A] (reuses KV cache, only new tokens!)
  Turn 4 ──> [Node A] (reuses KV cache, only new tokens!)

  Result: only incremental work per turn, maximum cache reuse
```

### KV-Cache-Aware Routing

```
┌──────────────────────────────────────────────────┐
│          KV-cache-aware router                    │
│                                                   │
│  ┌──────────┐                                    │
│  │ New req  │                                    │
│  │ session=X│                                    │
│  └────┬─────┘                                    │
│       │                                           │
│       v                                           │
│  ┌──────────────────────────┐                    │
│  │  Routing logic:           │                    │
│  │                          │                    │
│  │  1. Find session X's     │                    │
│  │     last node             │                    │
│  │  2. Check if that node    │                    │
│  │     has a hot KV cache    │                    │
│  │  3. Check node load       │                    │
│  │  4. Decide:               │                    │
│  │     hot cache + load OK   │                    │
│  │     → route to that node  │                    │
│  │     hot cache + load high │                    │
│  │     → weigh wait vs recompute │                │
│  │     no hot cache          │                    │
│  │     → route to least-loaded   │                │
│  └──────────┬───────────────┘                    │
│             │                                     │
│     ┌───────┼───────┐                            │
│     v       v       v                            │
│  [Node A] [Node B] [Node C]                      │
│  session X          session Y                     │
│  KV: ████           KV: ████                     │
└──────────────────────────────────────────────────┘
```

### llm-d KV-Aware Routing

```python
# llm-d KV-cache-aware router (conceptual)
class KVAwareRouter:
    """Smart router based on KV cache state."""

    def __init__(self, nodes: list[str]):
        self.nodes = nodes
        self.session_map = {}   # session_id -> node
        self.cache_status = {}  # node -> {session_id: cache_info}

    def route(self, request) -> str:
        """Route a request to the best node."""
        session_id = request.session_id

        # 1. Check if there's a known node
        if session_id in self.session_map:
            node = self.session_map[session_id]
            cache_info = self.cache_status.get(node, {}).get(session_id)

            if cache_info and cache_info["warm"]:
                # Hot cache present; check load
                if self._node_load(node) < 0.9:
                    return node  # route to cached node

                # Load too high: weigh wait vs recompute
                wait_cost = self._estimate_wait(node)
                recompute_cost = self._estimate_recompute(request)

                if wait_cost < recompute_cost:
                    return node  # waiting beats recompute
                # else fall through to load balancing

        # 2. No cache or stale cache: route to least-loaded
        best_node = min(self.nodes, key=self._node_load)
        self.session_map[session_id] = best_node
        return best_node

    def _estimate_recompute(self, request) -> float:
        """Estimate cost of re-computing the KV cache."""
        context_length = request.context_tokens
        # Rough estimate: ~100 ms prefill per 1000 tokens
        return context_length / 1000 * 0.1
```

---

## Memory-Augmented Conversations

When conversation history exceeds the context window or information must persist long-term, use an external memory system.

```
Memory-augmented architecture:

  ┌──────────────────────────────────────────┐
  │           Memory-augmented chat            │
  │                                          │
  │  User msg ──> [memory retrieval] ──> [context build]│
  │                  │              │         │
  │                  v              v         │
  │          ┌──────────────┐  ┌────────┐    │
  │          │ Long-term mem│  │  LLM   │    │
  │          │              │  │        │    │
  │          │ - User prefs │  └───┬────┘    │
  │          │ - Hist summary│      │         │
  │          │ - Key facts   │      v         │
  │          │ - Task XP     │  [generate response]│
  │          └──────────────┘      │         │
  │                                v         │
  │                          [update memory]  │
  │                          - Extract new facts │
  │                          - Update user profile │
  │                          - Store key decisions │
  └──────────────────────────────────────────┘
```

### Real-World Uses

- **Claude Memory**: Anthropic's memory feature, auto-extracts and stores cross-session info
- **ChatGPT Memory**: OpenAI's memory feature
- **MemGPT / Letta**: programmable memory-management frameworks

```python
# Memory-augmented chat example
class MemoryAugmentedChat:
    def __init__(self, llm, memory_store):
        self.llm = llm
        self.memory = memory_store
        self.context_manager = ContextManager(
            strategy="hierarchical"
        )

    async def chat(self, user_message: str, session_id: str):
        """Memory-augmented chat handler."""

        # 1. Retrieve relevant info from long-term memory
        relevant_memories = await self.memory.retrieve(
            query=user_message,
            session_id=session_id,
            top_k=5
        )

        # 2. Build an enhanced system prompt
        memory_context = "\n".join([
            f"- {m['content']}" for m in relevant_memories
        ])
        enhanced_system = f"""You are an intelligent assistant.

Known user information:
{memory_context}
"""

        # 3. Manage the context window
        self.context_manager.add_turn("user", user_message)
        messages = self.context_manager.get_context(enhanced_system)

        # 4. Generate response
        response = await self.llm.generate(messages)

        # 5. Update memory
        self.context_manager.add_turn("assistant", response)
        await self._update_memory(user_message, response, session_id)

        return response

    async def _update_memory(
        self, user_msg: str, assistant_msg: str, session_id: str
    ):
        """Extract and store new information from the conversation."""
        extraction_prompt = f"""Extract information worth long-term memorization from the conversation below:
User: {user_msg}
Assistant: {assistant_msg}

Extracted info (answer "none" if there is nothing):"""

        new_info = await self.llm.generate(extraction_prompt)
        if new_info.strip().lower() != "none":
            await self.memory.store(
                content=new_info,
                session_id=session_id,
                metadata={"type": "extracted_fact"}
            )
```

---

## Cost Analysis

### Per-Turn Token Growth

```
Token-cost growth (Claude 3.5 Sonnet pricing):

  Input: $3/M tokens, Output: $15/M tokens
  Cache read: $0.30/M tokens (90% off)

  No caching:
  Turn │ Input tokens│ Output│ Input cost │ Output cost │ Total
  ─────┼─────────────┼───────┼────────────┼─────────────┼──────
    1  │    2,000    │  500  │  $0.006    │  $0.0075    │ $0.014
    2  │    5,500    │  500  │  $0.017    │  $0.0075    │ $0.024
    3  │    9,000    │  500  │  $0.027    │  $0.0075    │ $0.035
    4  │   13,000    │  500  │  $0.039    │  $0.0075    │ $0.047
    5  │   17,000    │  500  │  $0.051    │  $0.0075    │ $0.059
  ─────┼─────────────┼───────┼────────────┼─────────────┼──────
  Total│   46,500    │ 2,500 │  $0.140    │  $0.038     │ $0.177

  With prompt caching (system prompt 2000 tokens cached):
  Turn │ Cache read  │ New in │ Cache cost │ New in cost │ Total
  ─────┼─────────────┼────────┼────────────┼─────────────┼──────
    1  │      0      │ 2,000  │  $0        │  $0.006     │ $0.014
    2  │   2,000     │ 3,500  │  $0.0006   │  $0.011     │ $0.018
    3  │   2,000     │ 7,000  │  $0.0006   │  $0.021     │ $0.029
    4  │   2,000     │ 11,000 │  $0.0006   │  $0.033     │ $0.041
    5  │   2,000     │ 15,000 │  $0.0006   │  $0.045     │ $0.053
  ─────┼─────────────┼────────┼────────────┼─────────────┼──────
  Total│   8,000     │ 38,500 │  $0.002    │  $0.116     │ $0.155

  Savings: ($0.177 - $0.155) / $0.177 = 12.4%

  With prompt caching (full prefix cache):
  Turn │ Cache read  │ New in │ Cache cost │ New in cost │ Total
  ─────┼─────────────┼────────┼────────────┼─────────────┼──────
    1  │      0      │ 2,000  │  $0        │  $0.006     │ $0.014
    2  │   2,500     │ 3,000  │  $0.0008   │  $0.009     │ $0.017
    3  │   6,000     │ 3,000  │  $0.0018   │  $0.009     │ $0.018
    4  │   9,500     │ 3,500  │  $0.0029   │  $0.011     │ $0.021
    5  │  13,500     │ 3,500  │  $0.0041   │  $0.011     │ $0.022
  ─────┼─────────────┼────────┼────────────┼─────────────┼──────
  Total│  31,500     │ 15,000 │  $0.010    │  $0.045     │ $0.092

  Savings: ($0.177 - $0.092) / $0.177 = 48%
```

---

## Benchmarks

### Multi-Turn Optimization Effect Comparison

```
10-turn agent conversation benchmark (relative to no-optimization baseline):

Metric                    │ No opt │ APC   │ Radix │ +LMCache │ +Continuum
──────────────────────────┼────────┼───────┼───────┼──────────┼──────────
Cumulative TTFT           │ 100%   │  40%  │  35%  │   25%    │   20%
End-to-end latency        │ 100%   │  70%  │  65%  │   55%    │   45%
GPU memory usage          │ 100%   │ 110%  │ 115%  │  130%    │  105%
Throughput (req/s)        │ 100%   │ 130%  │ 140%  │  250%    │  180%
Cost (API pricing)        │ 100%   │  -    │   -   │    -     │    -
Cost (prompt caching)     │ 100%   │  -    │   -   │    -     │   52%

Notes: APC = vLLM Auto Prefix Caching
       Radix = SGLang RadixAttention
       LMCache = tiered KV cache
       Continuum = agent-aware KV TTL

       APC and Radix are inference-engine-layer optimizations
       LMCache is a cache-layer optimization
       Continuum is a scheduler-layer optimization
       Layers can be stacked
```

### TTFT vs. Turn Number

```
TTFT (ms) vs. conversation turn:

       │ No opt    Prefix cache  +Session affinity
  5000 │ ●
  4500 │ │
  4000 │ │
  3500 │ │  ●
  3000 │ │  │
  2500 │ │  │  ●
  2000 │ │  │  │                      No opt: TTFT grows linearly
  1500 │ │  │  │  ●     ●
  1000 │ │  │  │  │     │  ●
   500 │ ● ─●──●──●─────●──●──────── Prefix cache: TTFT roughly constant
   200 │ ●──●──●──●─────●──●──────── + session affinity: lowest, constant
       └──────────────────────────
        T1  T2  T3  T4  T5  T6

  Takeaways:
  - No optimization: TTFT grows linearly with turn (re-prefill longer context)
  - Prefix caching: TTFT roughly constant (only new tokens processed)
  - Session affinity: lowest TTFT (100% KV cache hit)
```

---

## References

- Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs," 2024
- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," SOSP 2023
- LMCache, "KV Cache Management for LLM Serving," https://github.com/LMCache/LMCache
- Qin et al., "Continuum: KV Cache TTL for Agent Workloads," arXiv 2511.02230
- Anthropic, "Prompt Caching," https://docs.anthropic.com/claude/docs/prompt-caching
- OpenAI, "Prompt Caching," https://platform.openai.com/docs/guides/prompt-caching
- Zhong et al., "Don't Break the Cache: Prompt Caching for Agentic Workloads," 2026
- Liu et al., "PrefillShare: Shared Prefill Modules for Disaggregated Serving," 2026
- Together AI, "Cache-Aware Prefill-Decode Disaggregation," 2025

---

## Related Pages

- [[agent-serving-challenges]] -- why agent serving is different
- [[kv-cache-optimization]] -- KV cache techniques overview
- [[sglang]] -- RadixAttention prefix caching
- [[vllm]] -- vLLM automatic prefix caching
- [[prefill-decode-disaggregation]] -- P-D disaggregated architecture
- [[long-context-serving]] -- long-context serving
- [[compound-ai-systems]] -- compound AI systems
- [[ai-agent-overview]] -- agent architecture overview
