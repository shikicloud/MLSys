---
title: "Multi-Turn Conversation Serving Optimization"
category: llm-serving-for-agents
tags: [multi-turn, kv-cache-reuse, prefix-caching, session-management, lmcache, prompt-caching, context-management, sticky-sessions]
created: 2026-04-13
updated: 2026-05-07
status: mature
---

# Multi-Turn Conversation Serving Optimization

> [!abstract]+ TL;DR
> Agent sessions are multi-turn by definition. Each turn adds LLM output + tool results to the conversation history, and every LLM call must reprocess the full context. **Without optimization, prefill cost grows quadratically with turns.** With KV cache reuse, each turn only processes new tokens — linear scaling. Core systems: **[[vllm|vLLM]]** (block-level prefix caching), **[[sglang|SGLang]] RadixAttention** (token-level radix tree, +10 % multi-turn throughput vs vLLM), **LMCache** (cross-engine KV sharing GPU/CPU/disk/S3, **15× throughput / 2× lower latency**), **Continuum** (KV cache TTL during tool execution, 1.12–3.66× delay reduction).

```
Turn 1:  prefill 2K tokens
Turn 2:  prefill 5K tokens    (includes Turn 1)
Turn 3:  prefill 10K tokens   (includes Turns 1-2)
...
Turn 10: prefill 30K+ tokens  (includes all previous)

Total without optimization:  C_total = Sum(c_i) ~ O(N^2)
With KV cache reuse:         Each turn only processes new tokens ~ O(N)
```

---

## KV Cache Reuse Across Turns

Current request = prefix extension of previous turn. If KV cache retained, only new tokens need processing -- dramatic TTFT reduction.

```
Turn 2 input: [sys_prompt][user_1][asst_1][tool_result][user_2]
               ^^^^^^^^^^^^^^^^^^^^^^^^^ exact prefix match with Turn 1

With cache:  process only ~1500 new tokens (TTFT ~300ms)
Without:     process all 5000 tokens       (TTFT ~2000ms)

By Turn 10:  cache saves 90%+ of prefill compute
```

**Core tension**: Retaining cache consumes GPU memory (~1.25 MB/token for 70B model); evicting forces recomputation.

---

## Prefix Caching Implementations

| System | Method | Granularity | Advantage |
|--------|--------|-------------|-----------|
| [[vllm\|vLLM]] APC | Block-level hash | 16-token blocks | Good for templated prompts |
| [[sglang\|SGLang]] RadixAttention | Token-level radix tree | Token | Auto-discovers opportunities, ~10% better multi-turn |
| LMCache | Cross-engine connectors | Flexible | GPU/CPU/disk/S3 tiered hierarchy |
| TensorRT-LLM | KV cache reuse | Block | NVIDIA-native optimization |

**vLLM APC**: Hashes KV cache blocks; new requests with matching prefix hashes reuse cached blocks. Enable: `vllm serve model --enable-prefix-caching`.

**SGLang RadixAttention**: Stores all KV caches in a token-level radix tree. New requests match along the tree to the longest common prefix. Naturally supports branching (beam search, multi-sampling).

---

## LMCache: Multi-Tiered KV Caching

```
┌────────────────────────────────┐
│           LMCache              │
│  Layer 1: GPU HBM   (~0.1ms)  │
│  Layer 2: CPU DRAM   (~1ms)   │
│  Layer 3: Disk/NVMe  (~10ms)  │
│  Layer 4: S3/Object  (~100ms) │
│                                │
│  Performance: 15x throughput   │
│               2x lower latency │
│  Integrations: vLLM, SGLang,  │
│    KServe, NVIDIA Dynamo      │
└────────────────────────────────┘
```

---

## Disaggregated Serving Challenge

In [[prefill-decode-disaggregation|P-D architectures]], KV cache from Turn 1 lives on decode nodes, inaccessible to prefill nodes for Turn 2.

**Solutions**:
- **PrefillShare** (Feb 2026): Shared prefill modules for cross-model KV reuse
- **Cache-Aware P-D (CPD)** (Together AI): Pre-prefill nodes handle cold, prefill prioritizes warm
- **Shared KV layer** (LMCache): External KV store accessible by all nodes

---

## Context Window Management Strategies

| Strategy | Mechanism | Pros | Cons |
|----------|-----------|------|------|
| **Truncation** | Drop old history, keep recent N turns | Simple | Loses important history |
| **Summarization** | LLM-summarize old turns | Preserves key info | Summary may miss details; extra LLM call |
| **Sliding Window** | Keep last K tokens | Fixed memory | Information outside window completely lost |
| **Hierarchical Summary** | Global + mid-term + recent full | Best information retention | Most complex, extra LLM calls |
| **Memory-Augmented** | Retrieve relevant history from external memory | Dynamic relevance | Requires retrieval infrastructure |

```python
class ContextManager:
    def __init__(self, max_tokens=128000, recent_turns=3):
        self.max_tokens = max_tokens
        self.recent_turns = recent_turns
        self.full_history = []
        self.summaries = []

    def get_context(self, system_prompt):
        messages = [{"role": "system", "content": system_prompt}]
        if self.summaries:
            messages.append({"role": "system",
                           "content": f"History summary:\n{self.summaries[-1]}"})
        messages.extend(self.full_history[-self.recent_turns * 2:])
        return messages

    def _maybe_compress(self):
        total = sum(count_tokens(m["content"]) for m in self.full_history)
        if total > self.max_tokens * 0.7:
            old = self.full_history[:-self.recent_turns * 2]
            self.summaries.append(llm.summarize(old))
            self.full_history = self.full_history[-self.recent_turns * 2:]
```

---

## Prompt Caching (API-Level)

API-level optimization by LLM providers, complementary to engine-level KV cache reuse.

| Provider | Feature | Discount | TTL |
|----------|---------|----------|-----|
| **Anthropic** | Prompt Caching (`cache_control`) | 90% read discount | 5 min |
| **OpenAI** | Automatic Caching | 50% hit discount | ~5-10 min |
| **Google** | Context Caching | 75% read discount | Configurable |

```python
# Anthropic prompt caching for multi-turn
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    system=[{
        "type": "text",
        "text": "Long system prompt...",
        "cache_control": {"type": "ephemeral"}  # Cache this
    }],
    messages=[...previous_turns..., new_message]
)
# cache_read_input_tokens -> 90% cheaper
```

**Prompt caching vs KV cache reuse**: Prompt caching is API-level (pricing benefit from providers); KV cache reuse is engine-level (TTFT performance benefit in self-hosted). Both benefit from prefix stability.

---

## Session Management and Routing

### Sticky Sessions (Session Affinity)

Same-session requests must route to the same inference node for maximum KV cache hit rate.

```
Round-robin:  Turn1->A, Turn2->B, Turn3->C  (0% cache reuse)
Sticky:       Turn1->A, Turn2->A, Turn3->A  (maximum reuse)
```

### KV-Aware Routing

```python
class KVAwareRouter:
    def route(self, request):
        session_id = request.session_id
        if session_id in self.session_map:
            node = self.session_map[session_id]
            if self._has_warm_cache(node, session_id):
                if self._node_load(node) < 0.9:
                    return node  # Warm cache, acceptable load
                # Compare wait cost vs recompute cost
                if self._wait_cost(node) < self._recompute_cost(request):
                    return node
        return min(self.nodes, key=self._node_load)  # Fallback: least loaded
```

**llm-d**: KV-aware routing that routes requests to pods with warm cache.

---

## Cost Analysis

```
Token cost per turn (Claude 3.5 Sonnet, $3/M in, $15/M out):

Without caching:
  5 turns cumulative input: 46,500 tokens = $0.140 input + $0.038 output = $0.177

With full prefix caching (90% discount on cached):
  5 turns: $0.010 cached + $0.045 new input + $0.038 output = $0.092

Savings: 48%
```

---

## Benchmarks

```
10-turn agent, relative to unoptimized baseline:

                   No opt   APC    Radix   +LMCache  +Continuum
Cumulative TTFT:   100%     40%    35%     25%       20%
End-to-end:        100%     70%    65%     55%       45%
Throughput:        100%     130%   140%    250%      180%

TTFT vs turns:
  No opt:     linearly increasing (recompute growing context)
  Prefix:     roughly constant (only new tokens)
  +Affinity:  lowest and constant (100% KV cache hit)
```

---

## References

- Zheng et al., "SGLang: Efficient Execution of Structured LM Programs," 2024
- Kwon et al., "PagedAttention: Efficient Memory Management for LLM Serving," SOSP 2023
- LMCache, https://github.com/LMCache/LMCache
- Qin et al., "Continuum: KV Cache TTL for Agent Workloads," arXiv 2511.02230
- Anthropic, "Prompt Caching," https://docs.anthropic.com/claude/docs/prompt-caching
- OpenAI, "Prompt Caching," https://platform.openai.com/docs/guides/prompt-caching
- Zhong et al., "Don't Break the Cache," 2026

---

## Related Pages

- [[agent-serving-challenges]] -- Why agent serving is different
- [[kv-cache-optimization]] -- KV cache techniques
- [[sglang]] -- RadixAttention for prefix caching
- [[vllm]] -- vLLM auto prefix caching
- [[prefill-decode-disaggregation]] -- P-D architecture
- [[long-context-serving]] -- Long context serving
- [[compound-ai-systems]] -- Compound AI systems
- [[ai-agent-overview]] -- Agent architecture overview
