---
title: "多轮对话服务优化"
category: llm-serving-for-agents
tags: [多轮, kv-cache复用, 前缀缓存, 会话管理, lmcache, 提示缓存, 上下文管理, 粘性会话]
created: 2026-04-13
updated: 2026-04-14
status: mature
---

# 多轮对话服务优化

## 概述

智能体会话天然是多轮的（multi-turn）。每一步都向对话历史中添加新内容（LLM 输出 + 工具结果），而每次 LLM 调用都必须重新处理整个上下文。**如果不做优化，预填充（prefill）成本会随轮次呈二次增长**。

### 多轮问题的本质

```
Turn 1: [系统提示 + 用户请求]
         |---- prefill 2000 tokens ----|
         
Turn 2: [系统提示 + 用户请求 + Turn1输出 + 工具结果]  
         |---- prefill 5000 tokens -------------------|

Turn 3: [系统提示 + 用户请求 + Turn1 + Turn2 + 工具结果]
         |---- prefill 10000 tokens --------------------------------|

Turn N: [所有历史累积]
         |---- prefill 越来越长 ──────────────────────────────────────|

预填充成本增长:
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

  无优化的总 prefill 成本:
  C_total = Σ(i=1 to N) c_i ≈ O(N^2)
  
  因为 c_i (第 i 轮的 prefill 长度) 近似线性增长
```

### 核心优化思路

```
优化目标: 将 O(N^2) 降低到接近 O(N)

  关键观察: 
  Turn i 的输入 = Turn (i-1) 的输入 + Turn (i-1) 的输出 + 新内容
  
  即: 每轮的输入是上一轮输入的前缀扩展

  如果保留上一轮的 KV 缓存:
  Turn i 只需处理新增的 token (通常 <2000)
  而不是重新处理整个上下文 (可能 >20000)

  优化后:
  每轮 prefill 成本 ≈ 常数 (仅新增 token)
  总成本 C_total ≈ O(N)
```

---

## 跨轮 KV 缓存复用

### 原理

```
KV 缓存复用机制:

Turn 1:
  输入: [sys_prompt][user_msg_1]
  KV:   [████████████████████]  (2000 tokens)
  输出: [assistant_reply_1]

Turn 2:
  输入: [sys_prompt][user_msg_1][assistant_reply_1][tool_result_1][user_msg_2]
         ^^^^^^^^^^^^^^^^^^^^^^^^ 与 Turn 1 完全相同 (前缀匹配)
  
  如果 KV 缓存被保留:
  KV:   [████████████████████][■■■■■■■■■■■■■■■]  
         ^已有缓存(复用)^     ^仅需新计算 (1500 tokens)^

  节省: 2000 / 3500 = 57% 的 prefill 计算

Turn 5 (累积 15000 tokens):
  如果有缓存: 仅处理 ~1500 新 tokens
  如果无缓存: 处理全部 15000 tokens
  节省: 90%
```

### 核心矛盾

```
保留 vs 驱逐的权衡:

  保留 KV 缓存:
  ✓ 下一轮 TTFT 大幅降低
  ✗ 占用 GPU 显存 (每 token 约 0.5-2 MB)
  ✗ 减少可服务的并发请求数
  ✗ 内存碎片化

  驱逐 KV 缓存:
  ✓ 释放 GPU 显存
  ✓ 可服务更多并发请求
  ✗ 下一轮需要完全重新预填充
  ✗ 延迟增加数秒
  ✗ GPU 浪费在重复计算上

  显存占用估算 (LLaMA 70B, FP16):
  - 每 token KV 缓存: ~1.25 MB
  - 10000 token 会话: ~12.5 GB
  - 100 个并发会话: ~1.25 TB (远超单 GPU 显存)
  
  → 必须有智能的缓存管理策略
```

---

## 前缀缓存实现

### 主要系统对比

| 系统 | 方法 | 粒度 | 优势 | 局限 |
|------|------|------|------|------|
| [[vllm\|vLLM]] | 块级哈希 (APC) | 块 (16 tokens) | 适合模板化提示，实现简单 | 块边界对齐要求 |
| [[sglang\|SGLang]] | Token 级基数树 (RadixAttention) | Token | 自动发现缓存机会，多轮约好 10% | 树维护开销 |
| LMCache | 跨引擎连接器 | 灵活 | 多层级存储 (GPU/CPU/磁盘/S3) | 额外组件 |
| TensorRT-LLM | KV 缓存复用 | 块 | NVIDIA 原生优化 | NVIDIA 生态锁定 |

### vLLM 自动前缀缓存 (APC)

```
vLLM APC 工作原理:

  1. 将 KV 缓存划分为固定大小的块 (通常 16 tokens)
  2. 计算每个块的内容哈希
  3. 新请求的前缀块如果哈希匹配 → 直接复用

  示例:
  请求 A: [sys_prompt | user_1 | asst_1 | user_2]
  块:     [block_0   | block_1| block_2 | block_3]
  哈希:   [h0=abc    | h1=def | h2=ghi  | h3=jkl ]

  请求 B: [sys_prompt | user_1 | asst_1 | user_3]
  块:     [block_0   | block_1| block_2 | block_3']
  哈希:   [h0=abc    | h1=def | h2=ghi  | h3=xyz ]
           ^匹配^     ^匹配^   ^匹配^    ^不匹配^

  结果: 前 3 个块的 KV 缓存被复用, 仅需计算 block_3'

启用方式:
  vllm serve model_name --enable-prefix-caching
```

### SGLang RadixAttention

```
SGLang 基数树 (Radix Tree):

  RadixAttention 使用基数树存储所有请求的 KV 缓存:

           [root]
           /    \
     [sys_prompt] [sys_prompt_B]
         |
    [user_msg_1]
       /     \
  [asst_1]  [asst_1']   ← 不同的回复分支
     |         |
  [user_2]  [user_2']
     |
  [asst_2]

  查找过程:
  1. 新请求的 token 序列沿树向下匹配
  2. 匹配到最长的共同前缀
  3. 从分叉点开始计算新的 KV

  相比 vLLM APC 的优势:
  - Token 级粒度 (vs 块级), 更精确的匹配
  - 自然支持树状分支 (如 beam search, 多路采样)
  - 多轮对话性能约好 10%
  - 自动发现和利用缓存机会

启用方式 (默认启用):
  python -m sglang.launch_server --model model_name
```

### LMCache 多层级缓存

```
LMCache 架构:

  ┌──────────────────────────────────────────────┐
  │                  LMCache                      │
  │                                               │
  │  Layer 1: GPU HBM (最快, 最小)                │
  │  ┌──────────────────────────────┐             │
  │  │  热数据: 当前活跃会话的 KV     │             │
  │  │  延迟: ~0.1ms                │             │
  │  └──────────────┬───────────────┘             │
  │                 │ 驱逐                         │
  │                 v                              │
  │  Layer 2: CPU DRAM (快, 中等)                  │
  │  ┌──────────────────────────────┐             │
  │  │  温数据: 最近使用的会话 KV     │             │
  │  │  延迟: ~1ms                  │             │
  │  └──────────────┬───────────────┘             │
  │                 │ 驱逐                         │
  │                 v                              │
  │  Layer 3: 磁盘/NVMe (中等, 大)                │
  │  ┌──────────────────────────────┐             │
  │  │  冷数据: 不活跃但可能恢复的会话 │             │
  │  │  延迟: ~10ms                 │             │
  │  └──────────────┬───────────────┘             │
  │                 │ 驱逐                         │
  │                 v                              │
  │  Layer 4: S3/对象存储 (慢, 最大)               │
  │  ┌──────────────────────────────┐             │
  │  │  归档: 所有历史会话 KV         │             │
  │  │  延迟: ~100ms                │             │
  │  └──────────────────────────────┘             │
  │                                               │
  │  性能: 吞吐量提升 15x, 延迟降低 2x             │
  │  集成: vLLM, SGLang, KServe, NVIDIA Dynamo    │
  └──────────────────────────────────────────────┘
```

```python
# LMCache 使用示例
from lmcache import LMCacheEngine

# 配置多层级缓存
cache = LMCacheEngine(
    layers=[
        {"type": "gpu", "capacity_gb": 4},
        {"type": "cpu", "capacity_gb": 32},
        {"type": "disk", "path": "/data/kv_cache", "capacity_gb": 500},
        {"type": "s3", "bucket": "kv-cache-store"},
    ],
    eviction_policy="lru",        # 最近最少使用
    compression="fp8",            # KV 缓存压缩
    chunk_size=256,               # token 粒度
)

# 与 vLLM 集成
from vllm import LLM
llm = LLM(
    model="meta-llama/Llama-3-70B",
    kv_cache_engine=cache,        # 使用 LMCache 管理 KV
    enable_prefix_caching=True,
)
```

---

## 分离式架构下的多轮挑战

在 [[prefill-decode-disaggregation|P-D 分离架构]] 中，prefill 节点和 decode 节点是分开的，这给多轮 KV 缓存复用带来了特殊挑战。

```
P-D 分离架构的多轮问题:

  Turn 1:
  [Prefill 节点 A] ──> [Decode 节点 B]
                         KV 缓存在 B 上

  Turn 2:
  [Prefill 节点 A] 需要 Turn 1 的 KV 缓存
  但 KV 缓存在 Decode 节点 B 上!

  解决方案:
  1. KV 缓存传输: B -> A (网络开销)
  2. A 重新计算 (浪费计算)
  3. 共享 KV 缓存层 (如 LMCache)

┌──────────────────────────────────────────────┐
│         P-D 分离 + 多轮优化                    │
│                                               │
│  Turn 1:                                      │
│  [Prefill A] ──KV──> [Decode B]              │
│       │                   │                   │
│       │          KV 存入共享层                  │
│       │                   │                   │
│       v                   v                   │
│  ┌─────────────────────────────┐             │
│  │    共享 KV 缓存层 (LMCache)  │             │
│  └─────────────────────────────┘             │
│       │                                       │
│  Turn 2:                                      │
│  [Prefill A'] ──从共享层加载 KV──> [Decode B']│
│  (可能是不同节点)                               │
│                                               │
└──────────────────────────────────────────────┘
```

### 专用解决方案

**PrefillShare (2026 年 2 月)**：
- 共享预填充模块，支持跨模型 KV 缓存复用
- 多个 decode 实例共享 prefill 计算

**Cache-Aware P-D (CPD, Together AI)**：
- 引入 pre-prefill 节点处理冷启动
- Prefill 节点优先处理有热缓存的请求
- 缓存感知的请求路由

---

## 上下文窗口管理策略

当对话历史增长到接近或超过上下文窗口限制时，需要采用管理策略。

### 策略对比

```
┌──────────────────────────────────────────────────────┐
│              上下文窗口管理策略                         │
│                                                       │
│  策略 1: 截断 (Truncation)                            │
│  ┌────────────────────────────────────────┐          │
│  │ [系统提示][...丢弃旧历史...][最近N轮]    │          │
│  └────────────────────────────────────────┘          │
│  ✓ 简单   ✗ 丢失重要历史                              │
│                                                       │
│  策略 2: 摘要 (Summarization)                         │
│  ┌────────────────────────────────────────┐          │
│  │ [系统提示][旧历史摘要][最近N轮完整历史]   │          │
│  └────────────────────────────────────────┘          │
│  ✓ 保留关键信息  ✗ 摘要可能遗漏细节                    │
│                                                       │
│  策略 3: 滑动窗口 (Sliding Window)                    │
│  ┌────────────────────────────────────────┐          │
│  │ [系统提示][滑动窗口: 最近 K tokens]      │          │
│  └────────────────────────────────────────┘          │
│  ✓ 固定内存  ✗ 窗口外信息完全丢失                      │
│                                                       │
│  策略 4: 分层摘要 (Hierarchical Summary)              │
│  ┌────────────────────────────────────────┐          │
│  │ [系统提示]                              │          │
│  │ [全局摘要: 整个对话的高级总结]            │          │
│  │ [中期摘要: 最近 5-10 轮的详细摘要]        │          │
│  │ [近期完整: 最近 2-3 轮的完整内容]         │          │
│  └────────────────────────────────────────┘          │
│  ✓ 信息保留最好  ✗ 实现最复杂, 需要额外 LLM 调用       │
│                                                       │
│  策略 5: 记忆增强 (Memory-Augmented)                  │
│  ┌────────────────────────────────────────┐          │
│  │ [系统提示][从记忆中检索的相关历史][当前轮]  │          │
│  └────────────────────────────────────────┘          │
│  ✓ 动态相关性  ✗ 需要额外的检索基础设施                 │
└──────────────────────────────────────────────────────┘
```

### 实现示例

```python
class ContextManager:
    """多轮对话的上下文管理器"""

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
        self.summaries = []  # 分层摘要

    def add_turn(self, role: str, content: str):
        """添加新的对话轮次"""
        self.full_history.append({"role": role, "content": content})
        self._maybe_compress()

    def get_context(self, system_prompt: str) -> list[dict]:
        """获取当前上下文，适配上下文窗口"""
        if self.strategy == "truncation":
            return self._truncation_context(system_prompt)
        elif self.strategy == "sliding_window":
            return self._sliding_window_context(system_prompt)
        elif self.strategy == "hierarchical":
            return self._hierarchical_context(system_prompt)
        elif self.strategy == "summarization":
            return self._summarization_context(system_prompt)

    def _truncation_context(self, system_prompt: str) -> list[dict]:
        """简单截断: 保留系统提示 + 最近 N 轮"""
        messages = [{"role": "system", "content": system_prompt}]
        recent = self.full_history[-self.recent_turns * 2:]
        messages.extend(recent)
        return messages

    def _sliding_window_context(self, system_prompt: str) -> list[dict]:
        """滑动窗口: 保留最近 K tokens 的历史"""
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
        """分层摘要: 全局摘要 + 中期摘要 + 近期完整"""
        messages = [{"role": "system", "content": system_prompt}]

        # 全局摘要 (如果有)
        if self.summaries:
            global_summary = self.summaries[-1]
            messages.append({
                "role": "system",
                "content": f"对话历史摘要:\n{global_summary}"
            })

        # 近期完整历史
        recent = self.full_history[-self.recent_turns * 2:]
        messages.extend(recent)

        return messages

    def _maybe_compress(self):
        """当历史过长时触发压缩"""
        total = sum(count_tokens(m["content"]) for m in self.full_history)
        if total > self.max_tokens * 0.7:  # 70% 时开始压缩
            old_messages = self.full_history[:-self.recent_turns * 2]
            if old_messages:
                summary = self._summarize(old_messages)
                self.summaries.append(summary)
                # 保留近期历史
                self.full_history = self.full_history[-self.recent_turns * 2:]

    def _summarize(self, messages: list[dict]) -> str:
        """使用 LLM 对旧历史进行摘要"""
        content = "\n".join(
            f"{m['role']}: {m['content']}" for m in messages
        )
        return llm.generate(
            f"请简洁地摘要以下对话历史的关键信息:\n{content}"
        )
```

---

## 提示缓存 (Prompt Caching)

提示缓存是一种 **API 层面**的优化（由 LLM 提供商实现），与 KV 缓存复用在不同层面工作。

### 各提供商的提示缓存

| 提供商 | 功能 | 缓存粒度 | 定价优势 | TTL |
|--------|------|---------|---------|-----|
| **Anthropic** | Prompt Caching | 标记 cache_control | 缓存读取 90% 折扣 | 5 分钟 |
| **OpenAI** | Automatic Caching | 自动前缀匹配 | 缓存命中 50% 折扣 | ~5-10 分钟 |
| **Google** | Context Caching | 显式缓存创建 | 缓存读取 75% 折扣 | 可配置 |

### Anthropic 提示缓存详解

```
Anthropic Prompt Caching 工作原理:

  请求 1:
  ┌──────────────────────────────────────────────┐
  │ system: "你是一个助手..." (带 cache_control)  │ ← 创建缓存
  │ user: "什么是 KV 缓存?"                       │
  └──────────────────────────────────────────────┘
  
  定价: 创建缓存写入 $3.75/M (1.25x)
  
  请求 2 (5分钟内):
  ┌──────────────────────────────────────────────┐
  │ system: "你是一个助手..." (带 cache_control)  │ ← 缓存命中!
  │ user: "什么是前缀缓存?"                       │
  └──────────────────────────────────────────────┘
  
  定价: 缓存读取 $0.30/M (0.1x) ← 90% 折扣!

  多轮应用:
  ┌──────────────────────────────────────────────┐
  │ system: "..." (缓存)                          │ ← 100% 命中
  │ tools: [...] (缓存)                           │ ← 100% 命中
  │ 对话历史 Turn 1-5 (缓存)                      │ ← 前缀匹配
  │ 新用户消息 Turn 6                              │ ← 新计算
  └──────────────────────────────────────────────┘
  
  注意: 缓存断点位置很重要
  - 在变化的内容之前设置断点
  - 系统提示 + 工具定义 → 最佳缓存对象
  - 动态变化的内容放在最后
```

```python
# Anthropic Prompt Caching 使用示例
import anthropic

client = anthropic.Anthropic()

# 多轮对话中的缓存使用
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=4096,
    system=[
        {
            "type": "text",
            "text": "你是一个专业的编程助手...(很长的系统提示)",
            "cache_control": {"type": "ephemeral"}  # 标记为可缓存
        }
    ],
    messages=[
        # 对话历史 (可以标记为缓存)
        {"role": "user", "content": "请帮我写一个排序算法"},
        {"role": "assistant", "content": "好的，这里是快速排序..."},
        # 前面的历史可以被缓存
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "请帮我写一个排序算法\n好的，这里是快速排序...",  
                    "cache_control": {"type": "ephemeral"}  # 缓存断点
                }
            ]
        },
        # 新消息
        {"role": "user", "content": "能优化一下空间复杂度吗?"},
    ]
)

# 查看缓存使用情况
print(f"缓存创建: {response.usage.cache_creation_input_tokens}")
print(f"缓存读取: {response.usage.cache_read_input_tokens}")
print(f"新输入:   {response.usage.input_tokens}")
```

### 提示缓存 vs KV 缓存复用

```
两种缓存机制的关系:

  ┌──────────────────────────────────────────────────┐
  │                                                   │
  │  提示缓存 (API 层)                                │
  │  - 由 LLM 提供商管理                              │
  │  - 跨请求的提示前缀匹配                           │
  │  - 价格优惠 (50-90% 折扣)                         │
  │  - 用户只需标记 cache_control                     │
  │  - TTL: 5-10 分钟                                │
  │                                                   │
  │         ↕ 互补但不同层面                           │
  │                                                   │
  │  KV 缓存复用 (推理引擎层)                          │
  │  - 由推理引擎管理 (vLLM, SGLang)                  │
  │  - 推理引擎内部的 KV 张量复用                      │
  │  - 性能提升 (TTFT 降低)                           │
  │  - 对用户透明                                     │
  │  - TTL: 取决于内存压力                            │
  │                                                   │
  │  自建服务: 使用 KV 缓存复用                        │
  │  使用 API: 使用提示缓存                           │
  │  两者都可以从前缀稳定性中获益                      │
  └──────────────────────────────────────────────────┘
```

---

## 会话管理与路由

### 粘性会话 (Sticky Sessions)

为了最大化 KV 缓存命中率，同一个会话的请求应该路由到同一个推理节点。

```
非粘性路由 (Round-Robin):

  Turn 1 ──> [Node A] (计算 KV 缓存)
  Turn 2 ──> [Node B] (KV 缓存不在 B 上, 重新计算!)
  Turn 3 ──> [Node C] (KV 缓存不在 C 上, 重新计算!)
  Turn 4 ──> [Node A] (KV 缓存可能已被驱逐, 重新计算!)

  结果: 每轮都需要完整预填充, 零缓存复用

粘性路由 (Session Affinity):

  Turn 1 ──> [Node A] (计算 KV 缓存)
  Turn 2 ──> [Node A] (复用 KV 缓存, 仅处理新 token!)
  Turn 3 ──> [Node A] (复用 KV 缓存, 仅处理新 token!)
  Turn 4 ──> [Node A] (复用 KV 缓存, 仅处理新 token!)

  结果: 每轮仅需处理增量, 最大缓存复用
```

### KV 缓存感知路由

```
┌──────────────────────────────────────────────────┐
│          KV 缓存感知路由器                         │
│                                                   │
│  ┌──────────┐                                    │
│  │ 新请求    │                                    │
│  │ session=X │                                    │
│  └────┬─────┘                                    │
│       │                                           │
│       v                                           │
│  ┌──────────────────────────┐                    │
│  │  路由决策逻辑:            │                    │
│  │                          │                    │
│  │  1. 查找 session X 的    │                    │
│  │     上次节点              │                    │
│  │  2. 检查该节点是否有      │                    │
│  │     热 KV 缓存            │                    │
│  │  3. 检查该节点负载        │                    │
│  │  4. 综合决策:             │                    │
│  │     有热缓存+负载OK       │                    │
│  │     → 路由到该节点        │                    │
│  │     有热缓存+负载高       │                    │
│  │     → 权衡等待 vs 重计算  │                    │
│  │     无热缓存              │                    │
│  │     → 路由到最空闲节点    │                    │
│  └──────────┬───────────────┘                    │
│             │                                     │
│     ┌───────┼───────┐                            │
│     v       v       v                            │
│  [Node A] [Node B] [Node C]                      │
│  session X          session Y                     │
│  KV: ████           KV: ████                     │
└──────────────────────────────────────────────────┘
```

### llm-d 的 KV 感知路由

```python
# llm-d KV 缓存感知路由示例 (概念)
class KVAwareRouter:
    """基于 KV 缓存状态的智能路由器"""

    def __init__(self, nodes: list[str]):
        self.nodes = nodes
        self.session_map = {}   # session_id -> node
        self.cache_status = {}  # node -> {session_id: cache_info}

    def route(self, request) -> str:
        """路由请求到最优节点"""
        session_id = request.session_id

        # 1. 检查是否有已知节点
        if session_id in self.session_map:
            node = self.session_map[session_id]
            cache_info = self.cache_status.get(node, {}).get(session_id)

            if cache_info and cache_info["warm"]:
                # 有热缓存，检查负载
                if self._node_load(node) < 0.9:
                    return node  # 路由到有缓存的节点

                # 负载过高，计算等待成本 vs 重计算成本
                wait_cost = self._estimate_wait(node)
                recompute_cost = self._estimate_recompute(request)

                if wait_cost < recompute_cost:
                    return node  # 等待比重计算便宜
                # 否则 fall through 到负载均衡

        # 2. 无缓存或缓存已失效，路由到最空闲节点
        best_node = min(self.nodes, key=self._node_load)
        self.session_map[session_id] = best_node
        return best_node

    def _estimate_recompute(self, request) -> float:
        """估算重新计算 KV 缓存的成本"""
        context_length = request.context_tokens
        # 粗略估算: 每 1000 token 约 100ms prefill
        return context_length / 1000 * 0.1
```

---

## 记忆增强对话 (Memory-Augmented Conversations)

当对话历史超出上下文窗口或需要长期保持信息时，使用外部记忆系统。

```
记忆增强架构:

  ┌──────────────────────────────────────────┐
  │              记忆增强对话                  │
  │                                          │
  │  用户消息 ──> [记忆检索] ──> [上下文构建]  │
  │                  │              │         │
  │                  v              v         │
  │          ┌──────────────┐  ┌────────┐    │
  │          │ 长期记忆库    │  │  LLM   │    │
  │          │              │  │        │    │
  │          │ - 用户偏好    │  └───┬────┘    │
  │          │ - 历史摘要    │      │         │
  │          │ - 关键事实    │      v         │
  │          │ - 任务经验    │  [响应生成]     │
  │          └──────────────┘      │         │
  │                                v         │
  │                          [更新记忆]       │
  │                          - 提取新事实     │
  │                          - 更新用户画像   │
  │                          - 存储重要决策   │
  └──────────────────────────────────────────┘
```

### 实际应用

- **Claude Memory**：Anthropic 的记忆功能，自动提取和存储跨会话信息
- **ChatGPT Memory**：OpenAI 的记忆功能
- **MemGPT / Letta**：可编程的记忆管理框架

```python
# 记忆增强对话示例
class MemoryAugmentedChat:
    def __init__(self, llm, memory_store):
        self.llm = llm
        self.memory = memory_store
        self.context_manager = ContextManager(
            strategy="hierarchical"
        )

    async def chat(self, user_message: str, session_id: str):
        """记忆增强的对话处理"""

        # 1. 从长期记忆中检索相关信息
        relevant_memories = await self.memory.retrieve(
            query=user_message,
            session_id=session_id,
            top_k=5
        )

        # 2. 构建增强的系统提示
        memory_context = "\n".join([
            f"- {m['content']}" for m in relevant_memories
        ])
        enhanced_system = f"""你是一个智能助手。

已知用户信息:
{memory_context}
"""

        # 3. 管理上下文窗口
        self.context_manager.add_turn("user", user_message)
        messages = self.context_manager.get_context(enhanced_system)

        # 4. 生成回复
        response = await self.llm.generate(messages)

        # 5. 更新记忆
        self.context_manager.add_turn("assistant", response)
        await self._update_memory(user_message, response, session_id)

        return response

    async def _update_memory(
        self, user_msg: str, assistant_msg: str, session_id: str
    ):
        """从对话中提取并存储新信息"""
        extraction_prompt = f"""从以下对话中提取值得长期记忆的信息:
用户: {user_msg}
助手: {assistant_msg}

提取的信息 (如果没有则回答"无"):"""

        new_info = await self.llm.generate(extraction_prompt)
        if new_info.strip() != "无":
            await self.memory.store(
                content=new_info,
                session_id=session_id,
                metadata={"type": "extracted_fact"}
            )
```

---

## 成本分析

### 每轮 Token 增长

```
Token 成本增长分析 (Claude 3.5 Sonnet 定价):

  输入: $3/M tokens, 输出: $15/M tokens
  缓存读取: $0.30/M tokens (90% 折扣)

  无任何缓存:
  Turn │ 输入 Tokens │ 输出 │ 输入成本   │ 输出成本  │ 总计
  ─────┼─────────────┼──────┼───────────┼──────────┼────────
    1  │    2,000    │  500 │  $0.006   │ $0.0075  │ $0.014
    2  │    5,500    │  500 │  $0.017   │ $0.0075  │ $0.024
    3  │    9,000    │  500 │  $0.027   │ $0.0075  │ $0.035
    4  │   13,000    │  500 │  $0.039   │ $0.0075  │ $0.047
    5  │   17,000    │  500 │  $0.051   │ $0.0075  │ $0.059
  ─────┼─────────────┼──────┼───────────┼──────────┼────────
  总计 │   46,500    │2,500 │  $0.140   │ $0.038   │ $0.177

  使用提示缓存 (系统提示 2000 tokens 缓存):
  Turn │ 缓存读取  │ 新输入  │ 缓存成本   │ 新输入成本 │ 总计
  ─────┼──────────┼────────┼───────────┼──────────┼────────
    1  │    0     │ 2,000  │  $0       │ $0.006   │ $0.014
    2  │  2,000   │ 3,500  │  $0.0006  │ $0.011   │ $0.018
    3  │  2,000   │ 7,000  │  $0.0006  │ $0.021   │ $0.029
    4  │  2,000   │ 11,000 │  $0.0006  │ $0.033   │ $0.041
    5  │  2,000   │ 15,000 │  $0.0006  │ $0.045   │ $0.053
  ─────┼──────────┼────────┼───────────┼──────────┼────────
  总计 │  8,000   │ 38,500 │  $0.002   │ $0.116   │ $0.155

  节省: ($0.177 - $0.155) / $0.177 = 12.4%

  使用提示缓存 (全前缀缓存):
  Turn │ 缓存读取  │ 新输入  │ 缓存成本   │ 新输入成本 │ 总计
  ─────┼──────────┼────────┼───────────┼──────────┼────────
    1  │    0     │ 2,000  │  $0       │ $0.006   │ $0.014
    2  │  2,500   │ 3,000  │  $0.0008  │ $0.009   │ $0.017
    3  │  6,000   │ 3,000  │  $0.0018  │ $0.009   │ $0.018
    4  │  9,500   │ 3,500  │  $0.0029  │ $0.011   │ $0.021
    5  │  13,500  │ 3,500  │  $0.0041  │ $0.011   │ $0.022
  ─────┼──────────┼────────┼───────────┼──────────┼────────
  总计 │  31,500  │ 15,000 │  $0.010   │ $0.045   │ $0.092

  节省: ($0.177 - $0.092) / $0.177 = 48%
```

---

## 基准测试

### 多轮优化效果对比

```
10 轮智能体对话基准 (相对无优化基线):

指标                    │ 无优化 │ APC   │ Radix │ +LMCache │ +Continuum
────────────────────────┼────────┼───────┼───────┼──────────┼──────────
总 TTFT (累积)           │ 100%  │  40%  │  35%  │   25%    │   20%
端到端延迟               │ 100%  │  70%  │  65%  │   55%    │   45%
GPU 显存使用             │ 100%  │ 110%  │ 115%  │  130%    │  105%
吞吐量 (req/s)          │ 100%  │ 130%  │ 140%  │  250%    │  180%
成本 (API 定价)          │ 100%  │  -    │   -   │    -     │    -
成本 (提示缓存)          │ 100%  │  -    │   -   │    -     │   52%

注: APC = vLLM Auto Prefix Caching
    Radix = SGLang RadixAttention
    LMCache = 多层级 KV 缓存
    Continuum = 智能体感知 KV TTL
    
    APC 和 Radix 是推理引擎层优化
    LMCache 是缓存层优化
    Continuum 是调度层优化
    各层优化可以叠加
```

### 不同轮次下的 TTFT 对比

```
TTFT (ms) vs 对话轮次:

       │ 无优化    前缀缓存   +会话亲和
  5000 │ ●
  4500 │ │
  4000 │ │         
  3500 │ │  ●
  3000 │ │  │
  2500 │ │  │  ●
  2000 │ │  │  │                      无优化: TTFT 线性增长
  1500 │ │  │  │  ●     ●
  1000 │ │  │  │  │     │  ●
   500 │ ● ─●──●──●─────●──●──────── 前缀缓存: TTFT 基本恒定
   200 │ ●──●──●──●─────●──●──────── +会话亲和: TTFT 最低且恒定
       └──────────────────────────
        T1  T2  T3  T4  T5  T6

  结论:
  - 无优化: TTFT 随轮次线性增长 (重新预填充越来越长的上下文)
  - 前缀缓存: TTFT 基本恒定 (仅处理新增 token)
  - 会话亲和: TTFT 最低 (100% KV 缓存命中)
```

---

## 参考文献

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

## 相关页面

- [[agent-serving-challenges]] -- 为何智能体服务不同
- [[kv-cache-optimization]] -- KV 缓存技术总览
- [[sglang]] -- RadixAttention 前缀缓存
- [[vllm]] -- vLLM 自动前缀缓存
- [[prefill-decode-disaggregation]] -- P-D 分离架构
- [[long-context-serving]] -- 长上下文服务
- [[compound-ai-systems]] -- 复合 AI 系统
- [[ai-agent-overview]] -- 智能体架构总览
