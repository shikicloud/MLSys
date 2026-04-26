---
title: "PagedAttention：KV 缓存的虚拟内存管理"
category: llm-inference
tags: [paged-attention, kv-cache, 内存管理, vllm, 虚拟内存]
created: 2026-04-13
updated: 2026-04-14
status: mature
---

# PagedAttention：KV 缓存的虚拟内存管理

## 概述

PagedAttention 是 2023 年由 UC Berkeley 的 Kwon 等人提出的 KV 缓存内存管理算法，灵感来源于操作系统的**虚拟内存分页机制**。它解决了 LLM 推理中最关键的资源瓶颈之一——KV 缓存的内存碎片化和浪费问题。

在 PagedAttention 出现之前，LLM 服务系统因为内存管理低效，实际可用 GPU 内存中有 **60-80%** 被浪费。PagedAttention 将这一浪费降低到 **不足 4%**，使得相同硬件上可以服务的并发请求数量大幅增加，吞吐量提升可达 **2-4 倍**。

PagedAttention 是 [[vllm|vLLM]] 的核心创新，并已被几乎所有主流 LLM 服务框架采用，包括 [[sglang|SGLang]]、[[tensorrt-llm|TensorRT-LLM]]、HuggingFace TGI 等。它与 [[continuous-batching|连续批处理]] 的结合使得现代 LLM 服务系统的效率达到了全新水平。

---

## 传统 KV 缓存的问题

### KV 缓存基础

在自回归解码过程中，每生成一个新 token，都需要对所有之前 token 的 Key 和 Value 进行注意力计算。为避免重复计算，系统会将每层的 K 和 V 张量缓存下来，称为 **KV 缓存**。

单个请求的 KV 缓存大小：

```
KV_size = 2 × num_layers × num_kv_heads × head_dim × seq_len × dtype_bytes
```

以 LLaMA-13B 为例（40 层，40 heads，128 head_dim，FP16）：
- 最大序列长度 2048：`2 × 40 × 40 × 128 × 2048 × 2 bytes = 1.6 GB`
- 一个请求就可能占据大量 GPU 内存

### 预分配浪费 (60-80%)

传统系统采用**预分配策略**：在请求到达时，为其 KV 缓存预分配一块能容纳最大序列长度的连续内存。

```
传统预分配方式（max_seq_len = 2048 tokens）

请求 A（实际用 327 tokens）：
┌─────────┬──────────────────────────────────────────────────┐
│ 已使用  │              预留但浪费的内存                       │
│ 327 tok │            1721 tokens 的空间被浪费                 │
└─────────┴──────────────────────────────────────────────────┘
 ←─ 16% ─→←──────────── 84% 浪费 ──────────────────────────→

请求 B（实际用 1150 tokens）：
┌──────────────────────────┬─────────────────────────────────┐
│        已使用             │         预留但浪费的内存          │
│      1150 tokens          │       898 tokens 被浪费          │
└──────────────────────────┴─────────────────────────────────┘
 ←──────── 56% ───────────→←──────── 44% 浪费 ─────────────→

请求 C（实际用 89 tokens）：
┌──┬─────────────────────────────────────────────────────────┐
│用│                 预留但浪费的内存                           │
│89│              1959 tokens 的空间被浪费                      │
└──┴─────────────────────────────────────────────────────────┘
 4%←───────────────── 96% 浪费 ─────────────────────────────→
```

平均浪费率通常在 **60-80%**，因为：
1. 大多数请求的实际输出长度远小于最大序列长度
2. 在生成开始时就必须锁定全部内存
3. 系统无法预知每个请求的实际输出长度

### 内部碎片（Internal Fragmentation）

预分配的内存块中，已分配但未使用的部分称为内部碎片。由于无法将这部分内存回收给其他请求使用，即使 GPU 内存总量充足，也无法接受更多并发请求。

```
GPU 内存
┌─────────────────────────────────────────────────────┐
│ 请求A的KV缓存（预分配）  [██░░░░░░░░░░░░░░░░░░░░]  │  ██ = 已用
│ 请求B的KV缓存（预分配）  [██████████░░░░░░░░░░░░]  │  ░░ = 内部碎片
│ 请求C的KV缓存（预分配）  [█░░░░░░░░░░░░░░░░░░░░░]  │
│                                                     │
│ ╳ 无法容纳新请求！（虽然总空闲内存 > 所需量）        │
│ ╳ 因为找不到足够大的 *连续* 空闲块                   │
└─────────────────────────────────────────────────────┘
```

### 外部碎片（External Fragmentation）

当多个请求完成并释放内存后，空闲内存被分散为多个不连续的小块。虽然总量可能足够，但没有任何一块足够大来服务新请求。

```
请求完成后的 GPU 内存
┌─────────────────────────────────────────────────────┐
│ [空闲1] [请求D ██████] [空闲2] [请求E ████] [空闲3] │
│                                                     │
│ 空闲1 + 空闲2 + 空闲3 = 3GB（总量足够）             │
│ 但每个空闲块都不足以分配给新请求（需要 2GB 连续）     │
│                                                     │
│ 这就是外部碎片问题！                                 │
└─────────────────────────────────────────────────────┘
```

### 预留策略的失败

为缓解上述问题，业界尝试过多种预留策略：

| 策略 | 做法 | 问题 |
|------|------|------|
| **最大长度预分配** | 为每个请求预分配 max_seq_len | 浪费最严重，60-80% |
| **预测长度预分配** | 根据历史统计预测输出长度 | 预测不准时仍然浪费或 OOM |
| **增量扩展** | 按需逐步扩大内存块 | 需要内存拷贝，引入延迟 |
| **内存池** | 预分配固定大小的 slab | 粒度不匹配，仍有内部碎片 |

这些方案都无法从根本上解决问题，因为它们都受限于**连续内存分配**的约束。PagedAttention 的关键洞察是：打破连续性约束。

---

## PagedAttention 原理

### 核心思想：借鉴操作系统虚拟内存

操作系统通过虚拟内存机制解决了物理内存碎片化问题：
- 进程看到的是连续的**虚拟地址空间**
- 物理内存被分为固定大小的**页帧**（page frame）
- **页表**（page table）将虚拟页映射到物理页帧
- 应用无需关心物理内存是否连续

PagedAttention 将同样的思想应用于 KV 缓存管理：

| OS 概念 | PagedAttention 对应 |
|---------|---------------------|
| 虚拟页 (Virtual Page) | 逻辑块 (Logical Block) |
| 物理页帧 (Physical Frame) | 物理块 (Physical Block) |
| 页表 (Page Table) | 块表 (Block Table) |
| 进程 (Process) | 序列 (Sequence) |
| 页大小 (Page Size) | 块大小 (Block Size) |

### 块抽象

PagedAttention 将 KV 缓存划分为固定大小的**块**（block），每个块存储固定数量 token 的 Key 和 Value 向量。

**逻辑块（Logical Block）**：
- 从模型计算的角度，每个序列的 KV 缓存被视为一系列连续的逻辑块
- 逻辑块编号从 0 开始，按序列中 token 的位置顺序排列
- 模型计算时通过逻辑块号 + 块内偏移来定位

**物理块（Physical Block）**：
- GPU 内存中实际分配的内存块
- 大小与逻辑块相同
- 物理上可以分散在 GPU 内存的任何位置
- 通过空闲块列表（free list）管理

单个物理块的大小：

```python
block_size_bytes = block_size × num_layers × num_kv_heads × head_dim × dtype_bytes × 2
# 例如 block_size=16, LLaMA-7B (32 层, 32 heads, 128 dim, FP16):
# 16 × 32 × 32 × 128 × 2 × 2 = 32 MB
# 注意：在实际实现中，每层独立分配，所以单层的一个块为：
# 16 × 32 × 128 × 2 × 2 = 1 MB (包含 K 和 V)
```

### 块表机制

块表是 PagedAttention 的核心数据结构，维护从逻辑块到物理块的映射关系。

```
序列 "The cat sat on the mat and then ..." (假设 block_size = 4)

逻辑块视图（序列看到的）：
┌──────────────┬──────────────┬──────────────┬──────────────┐
│ 逻辑块 0     │ 逻辑块 1     │ 逻辑块 2     │ 逻辑块 3     │
│ The cat sat  │ on the mat   │ and then the │ dog ...      │
│ on           │              │              │ (部分填充)    │
└──────────────┴──────────────┴──────────────┴──────────────┘
   4 tokens       4 tokens       4 tokens      2/4 tokens

块表 (Block Table):
┌──────────┬──────────────┐
│ 逻辑块号  │ 物理块号      │
├──────────┼──────────────┤
│    0     │     7        │
│    1     │     3        │
│    2     │    12        │
│    3     │     1        │
└──────────┴──────────────┘

GPU 物理内存布局（物理块不连续）：
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│  0  │ *1* │  2  │ *3* │  4  │  5  │  6  │ *7* │  8  │  9  │ 10  │ 11  │*12* │
│其他 │逻辑3│其他 │逻辑1│空闲 │空闲 │其他 │逻辑0│其他 │其他 │空闲 │其他 │逻辑2│
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
  * 标记的块属于该序列

注意：逻辑上连续的块 0,1,2,3 对应物理块 7,3,12,1 —— 完全不连续！
```

### 按需分配流程

PagedAttention 的内存分配完全按需进行：

```
时间线（block_size = 4）：

t=0: 请求到达，prompt = "The cat sat on"
     分配物理块 7 → 逻辑块 0   [The, cat, sat, on]  (满)
     
t=1: 生成 "the" → 需要新块
     分配物理块 3 → 逻辑块 1   [the, _, _, _]  (1/4)
     
t=2: 生成 "mat"
     逻辑块 1 未满，直接追加   [the, mat, _, _]  (2/4)
     
t=3: 生成 "and"
     逻辑块 1 未满，直接追加   [the, mat, and, _]  (3/4)
     
t=4: 生成 "then"
     逻辑块 1 填满             [the, mat, and, then]  (4/4 满)
     
t=5: 生成 "the" → 需要新块
     分配物理块 12 → 逻辑块 2  [the, _, _, _]  (1/4)
     
... 请求完成时，释放物理块 7, 3, 12 回到空闲列表
```

这种按需分配的关键优势：
1. **零预分配浪费**：不需要预测输出长度
2. **最后一个块的浪费**：平均只有 `block_size / 2` 个 token 的空间被浪费
3. **即时回收**：请求完成时立即释放所有物理块

### 注意力计算中的非连续内存访问

传统注意力计算假设 KV 缓存在内存中连续存储。PagedAttention 需要修改注意力计算内核，使其能够从非连续的物理块中正确获取数据。

注意力计算的基本公式：

```
Attention(Q, K, V) = softmax(Q × K^T / sqrt(d_k)) × V
```

在 PagedAttention 中，Q 来自当前 token（连续），但 K 和 V 分散在多个物理块中。注意力计算需要：

1. 根据块表找到所有相关的物理块
2. 从各物理块中加载对应的 K、V 向量
3. 正确组合各块的注意力分数

### 自定义 CUDA 内核设计

PagedAttention 的 CUDA 内核是其技术实现的核心。内核设计需要解决的关键挑战：

```
PagedAttention CUDA 内核的工作流程：

输入：query (当前token), block_table, kv_cache_pool
输出：attention_output

对于每个注意力头 (并行)：
  ┌──────────────────────────────────────────────┐
  │ 1. 从 block_table 查找该序列的物理块列表       │
  │                                              │
  │ 2. 对每个物理块 (并行):                       │
  │    ├─ 加载该块的 K 向量                       │
  │    ├─ 计算 Q × K^T / sqrt(d)                 │
  │    └─ 保存部分注意力分数 (partial scores)     │
  │                                              │
  │ 3. 跨所有块进行 safe softmax:                 │
  │    ├─ 找全局最大值 (数值稳定性)               │
  │    ├─ 计算归一化的注意力权重                   │
  │    └─ 处理最后一个块的 padding mask           │
  │                                              │
  │ 4. 对每个物理块 (并行):                       │
  │    ├─ 加载该块的 V 向量                       │
  │    └─ 计算加权求和 (attention_weights × V)    │
  │                                              │
  │ 5. 累加所有块的结果得到最终输出                │
  └──────────────────────────────────────────────┘
```

内核的关键优化：
- **分块归约（Block-wise Reduction）**：每个 CUDA 线程块处理一个或多个 KV 块，通过共享内存进行归约
- **在线 Softmax**：使用 Milakov & Gimelshein 的在线 softmax 算法，无需两遍扫描
- **内存合并访问（Coalesced Access）**：尽管物理块不连续，但块内访问是连续的
- **与 FlashAttention 结合**：后续版本支持在 FlashAttention 框架内使用分页

### 块大小选择与权衡

块大小是 PagedAttention 最重要的超参数：

| 块大小 | 优势 | 劣势 |
|--------|------|------|
| 小 (1-4) | 极低浪费，细粒度分配 | 块表大，内核效率低，更多间接寻址 |
| 中 (16) | 平衡浪费和效率 | **通常是最佳选择** |
| 大 (64-256) | 内核高效，接近连续 | 最后一个块浪费大，灵活性下降 |

vLLM 默认使用 **block_size = 16**，这是经过大量基准测试得出的平衡点：

```python
# 浪费分析
# 每个序列最后一个块的平均浪费 = block_size / 2 个 token
# block_size = 16 时：
#   平均浪费 8 token 的 KV 缓存
#   对于 2048 长序列：8/2048 = 0.4% 浪费
#   对于 128 长序列：8/128 = 6.25% 浪费（但仍远好于传统方案）

# 相比传统预分配（max_seq_len=2048, 实际128 tokens）：
#   (2048 - 128) / 2048 = 93.75% 浪费
```

---

## 写时复制 (Copy-on-Write)

### 并行采样的内存挑战

在 LLM 推理中，许多场景需要从同一个前缀生成多个不同的输出：

- **并行采样**：为一个 prompt 生成 N 个候选回答
- **束搜索（Beam Search）**：维护 K 个最优候选序列
- **Best-of-N**：生成 N 个回答，选择最优

这些场景中，多个序列共享相同的前缀 KV 缓存。传统方法需要为每个序列复制一份完整的前缀 KV 缓存，造成巨大的内存浪费。

### 引用计数机制

PagedAttention 借鉴 OS 的写时复制（Copy-on-Write, CoW）机制，通过引用计数实现 KV 缓存共享：

```
并行采样 (n=3) 初始状态：

Prompt: "Write a poem about spring"
          (占 2 个逻辑块)

                     块表
序列 1:  逻辑块0 → 物理块 5 (ref_count=3)
         逻辑块1 → 物理块 9 (ref_count=3)

序列 2:  逻辑块0 → 物理块 5 (ref_count=3)  ← 共享！
         逻辑块1 → 物理块 9 (ref_count=3)  ← 共享！

序列 3:  逻辑块0 → 物理块 5 (ref_count=3)  ← 共享！
         逻辑块1 → 物理块 9 (ref_count=3)  ← 共享！

物理内存：只存储一份前缀 KV 缓存
传统方法：需要存储三份！
```

### CoW 触发过程

当某个序列需要修改共享块时，CoW 机制被触发：

```
Step 1: 序列 1 生成新 token，需要修改逻辑块 1（最后一个块，部分填充）

检查：物理块 9 的 ref_count = 3 > 1 → 需要 CoW！

Step 2: 分配新物理块 14，将物理块 9 的内容复制过去

Step 3: 更新映射和引用计数

序列 1:  逻辑块0 → 物理块 5  (ref_count=3)
         逻辑块1 → 物理块 14 (ref_count=1)  ← 新的私有块
         逻辑块2 → 物理块 20 (ref_count=1)  ← 新分配（如果需要）

序列 2:  逻辑块0 → 物理块 5  (ref_count=3)
         逻辑块1 → 物理块 9  (ref_count=2)  ← 引用计数减1

序列 3:  逻辑块0 → 物理块 5  (ref_count=3)
         逻辑块1 → 物理块 9  (ref_count=2)

只有被修改的块才会被复制，前面的共享块继续共享！
```

### 束搜索中的 CoW

束搜索是 CoW 收益最大的场景，因为束搜索会频繁地淘汰和复制候选序列：

```
Beam Search (beam_width=4) 示例：

Step 0 (初始): 所有 beam 共享 prompt 的 KV 缓存
┌──────────────────────────────────────────────────────┐
│  Beam 0 ──→ [物理块A][物理块B]                        │
│  Beam 1 ──→ [物理块A][物理块B]  (所有 beam 共享)       │
│  Beam 2 ──→ [物理块A][物理块B]                        │
│  Beam 3 ──→ [物理块A][物理块B]                        │
│                                                      │
│  物理块A: ref_count=4                                 │
│  物理块B: ref_count=4                                 │
│  总物理块数: 2（而非传统方法的 8）                      │
└──────────────────────────────────────────────────────┘

Step 5: 各 beam 分叉后
┌──────────────────────────────────────────────────────┐
│  Beam 0 ──→ [A][B][C][D0]                            │
│  Beam 1 ──→ [A][B][C][D1]   (前3个块仍共享)           │
│  Beam 2 ──→ [A][B][E][F]    (前2个块共享)              │
│  Beam 3 ──→ [A][B][E][G]    (前2个块共享)              │
│                                                      │
│  物理块A: ref_count=4, B: ref_count=4                 │
│  物理块C: ref_count=2, E: ref_count=2                 │
│  其余: ref_count=1                                    │
│  总物理块数: 9（传统方法需要 16）                       │
└──────────────────────────────────────────────────────┘

Step 10: Beam 2 被淘汰，Beam 1 扩展
┌──────────────────────────────────────────────────────┐
│  释放 Beam 2 的私有块 F                               │
│  E 的 ref_count: 2→1                                 │
│  无需复制——只需更新引用计数                            │
└──────────────────────────────────────────────────────┘
```

### 内存节省分析

CoW 在不同场景下的内存节省：

| 场景 | 传统方法 | PagedAttention + CoW | 节省 |
|------|---------|---------------------|------|
| 并行采样 n=4, 前缀占比 50% | 4x 前缀 + 4x 输出 | 1x 前缀 + 4x 输出 | ~38% |
| 束搜索 beam=8, 长序列 | 8x 完整序列 | ~3x 等效 (大量共享) | ~55% |
| Best-of-16 | 16x 完整序列 | ~6x 等效 | ~60% |

论文中的实验结果：
- 并行采样场景：**吞吐量提升 2.2 倍**（仅来自内存共享，不含其他优化）
- 束搜索场景：**内存减少高达 55%**

---

## 前缀缓存 (Prefix Caching)

### 动机

在实际生产环境中，大量请求共享相同的前缀：

- **系统提示词（System Prompt）**：所有请求使用相同的系统指令
- **Few-shot 示例**：相同的示例被附加到多个请求前
- **多轮对话**：每轮对话都包含之前所有轮次的历史
- **RAG 场景**：检索到的文档片段可能重复

如果能识别并复用这些共享前缀的 KV 缓存，就可以避免大量的重复计算。

### 基于哈希的前缀缓存（vLLM 方式）

vLLM V1 采用基于内容哈希的前缀缓存方案：

```
前缀缓存工作原理：

请求 1: [系统提示词 tokens] + [用户问题 A]
         hash(block0) = 0xAB12  → 计算并缓存物理块
         hash(block1) = 0xCD34  → 计算并缓存物理块
         hash(block2) = 0xEF56  → 计算并缓存物理块（用户问题）

请求 2: [系统提示词 tokens] + [用户问题 B]
         hash(block0) = 0xAB12  → 缓存命中！直接复用 ✓
         hash(block1) = 0xCD34  → 缓存命中！直接复用 ✓
         hash(block2) = 0x7890  → 未命中，需要计算

节省：跳过了系统提示词的 prefill 计算！
```

哈希策略：
- 每个逻辑块根据其包含的 **token 内容** 计算哈希
- 哈希还包含块的**位置信息**（因为位置编码会影响 KV 值）
- 使用 LRU（最近最少使用）策略淘汰缓存块

### 基于基数树的前缀缓存（SGLang RadixAttention）

[[sglang|SGLang]] 使用基数树（Radix Tree）实现更高效的前缀匹配：

```
RadixAttention 的基数树结构：

                        [root]
                       /      \
            [系统提示词...]     [另一系统提示词...]
            /     |     \
    [用户问题A] [用户问题B] [Few-shot前缀]
        |          |         /    \
    [回答A]    [回答B]  [问题C]  [问题D]

优势：
- O(n) 的前缀匹配（n 为共享前缀长度）
- 支持 token 级别的精确匹配
- 自然支持多轮对话的层次结构
- 缓存驱逐可以精确到子树
```

### 两种方案对比

| 特性 | 哈希方案 (vLLM) | 基数树方案 (SGLang) |
|------|-----------------|-------------------|
| 匹配粒度 | 块级别 | Token 级别 |
| 查找复杂度 | O(1) 哈希查找 | O(n) 树遍历 |
| 前缀密集场景性能 | 好 | 更好（快 ~29%）|
| 实现复杂度 | 较低 | 较高 |
| 多轮对话优化 | 好 | 更自然 |

### 系统提示词优化

前缀缓存对系统提示词的优化效果显著：

```python
# 典型场景：系统提示词 1000 tokens, 用户消息 200 tokens
# 假设 prefill 速度 10,000 tok/s

# 无前缀缓存：
#   每个请求: prefill 1200 tokens → 120ms TTFT

# 有前缀缓存（系统提示词命中）：
#   首个请求: prefill 1200 tokens → 120ms TTFT
#   后续请求: prefill 200 tokens → 20ms TTFT
#   TTFT 降低 83%！

# vLLM V1 中，前缀缓存默认开启
# 缓存未命中时的额外开销 < 1%（几乎免费）
```

### 性能收益

前缀缓存的实际收益取决于工作负载特性：

| 工作负载 | 前缀命中率 | TTFT 降低 | 吞吐量提升 |
|---------|-----------|----------|-----------|
| 单系统提示词 + 短用户消息 | >90% | 60-85% | 1.5-3x |
| 多轮对话（3-5轮） | 70-90% | 40-70% | 1.3-2x |
| RAG（共享文档） | 30-60% | 20-40% | 1.1-1.5x |
| 完全随机请求 | ~0% | ~0% | ~0%（<1% 开销）|

---

## 代码示例

### 简化版 PagedAttention 内核伪代码

```python
import torch

def paged_attention_forward(
    query: torch.Tensor,         # [batch, num_heads, 1, head_dim] (单token解码)
    key_cache: torch.Tensor,     # [num_physical_blocks, block_size, num_kv_heads, head_dim]
    value_cache: torch.Tensor,   # [num_physical_blocks, block_size, num_kv_heads, head_dim]
    block_tables: torch.Tensor,  # [batch, max_num_blocks] 逻辑→物理块映射
    context_lens: torch.Tensor,  # [batch] 每个序列的当前长度
    block_size: int = 16,
) -> torch.Tensor:
    """
    简化版 PagedAttention 前向计算（实际为 CUDA 内核实现）。
    此处用 Python 展示逻辑流程。
    """
    batch_size, num_heads, _, head_dim = query.shape
    scale = head_dim ** -0.5
    output = torch.zeros_like(query)
    
    for b in range(batch_size):
        seq_len = context_lens[b].item()
        num_blocks = (seq_len + block_size - 1) // block_size
        
        # 收集该序列的所有 K, V（从非连续物理块中）
        keys = []
        values = []
        for logical_idx in range(num_blocks):
            physical_idx = block_tables[b, logical_idx].item()
            
            # 确定该块中有效的 token 数
            if logical_idx == num_blocks - 1:
                # 最后一个块可能只部分填充
                valid_tokens = seq_len - logical_idx * block_size
            else:
                valid_tokens = block_size
            
            # 从物理块中加载 K, V
            keys.append(key_cache[physical_idx, :valid_tokens])
            values.append(value_cache[physical_idx, :valid_tokens])
        
        # 拼接所有块的 K, V
        k = torch.cat(keys, dim=0)   # [seq_len, num_kv_heads, head_dim]
        v = torch.cat(values, dim=0) # [seq_len, num_kv_heads, head_dim]
        
        # 标准注意力计算
        # (实际 CUDA 内核中会逐块计算并使用在线 softmax)
        for h in range(num_heads):
            kv_head = h // (num_heads // k.shape[1])  # GQA 支持
            attn_scores = (query[b, h] @ k[:, kv_head].T) * scale  # [1, seq_len]
            attn_weights = torch.softmax(attn_scores, dim=-1)
            output[b, h] = attn_weights @ v[:, kv_head]
    
    return output
```

### vLLM BlockSpaceManager 简化实现

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

@dataclass
class PhysicalBlock:
    """物理块：GPU 内存中的实际存储单元"""
    block_id: int
    ref_count: int = 0          # 引用计数，用于 CoW
    
    def is_shared(self) -> bool:
        return self.ref_count > 1

class BlockSpaceManager:
    """
    简化版 vLLM 块空间管理器。
    实际实现更复杂，包含 CPU/GPU 交换、前缀缓存等。
    """
    
    def __init__(
        self,
        block_size: int = 16,
        num_gpu_blocks: int = 1024,
    ):
        self.block_size = block_size
        self.num_gpu_blocks = num_gpu_blocks
        
        # 初始化所有物理块
        self.gpu_blocks = [
            PhysicalBlock(block_id=i) for i in range(num_gpu_blocks)
        ]
        
        # 空闲块列表
        self.free_blocks: List[PhysicalBlock] = list(self.gpu_blocks)
        
        # 每个序列的块表：seq_id → [物理块列表]
        self.block_tables: Dict[int, List[PhysicalBlock]] = {}
    
    def can_allocate(self, num_blocks_needed: int) -> bool:
        """检查是否有足够的空闲块"""
        return len(self.free_blocks) >= num_blocks_needed
    
    def allocate_block(self) -> PhysicalBlock:
        """分配一个物理块"""
        if not self.free_blocks:
            raise RuntimeError("GPU 内存不足！无空闲物理块")
        block = self.free_blocks.pop()
        block.ref_count = 1
        return block
    
    def free_block(self, block: PhysicalBlock) -> None:
        """释放一个物理块（减少引用计数）"""
        block.ref_count -= 1
        if block.ref_count == 0:
            self.free_blocks.append(block)
    
    def allocate_sequence(self, seq_id: int, num_initial_tokens: int) -> None:
        """为新序列分配初始块"""
        num_blocks = (num_initial_tokens + self.block_size - 1) // self.block_size
        blocks = [self.allocate_block() for _ in range(num_blocks)]
        self.block_tables[seq_id] = blocks
    
    def append_token(self, seq_id: int, num_new_tokens: int = 1) -> None:
        """追加 token 时按需分配新块"""
        blocks = self.block_tables[seq_id]
        current_tokens = len(blocks) * self.block_size  # 简化：假设之前都满
        
        # 检查最后一个块是否还有空间
        last_block = blocks[-1]
        remaining_in_last = self.block_size - (current_tokens % self.block_size)
        if remaining_in_last == self.block_size:
            remaining_in_last = 0
        
        if remaining_in_last < num_new_tokens:
            # 需要新的物理块
            new_blocks_needed = (num_new_tokens - remaining_in_last + 
                                self.block_size - 1) // self.block_size
            for _ in range(new_blocks_needed):
                blocks.append(self.allocate_block())
    
    def fork_sequence(self, parent_seq_id: int, child_seq_id: int) -> None:
        """Fork 序列（用于并行采样/束搜索）——通过 CoW 共享块"""
        parent_blocks = self.block_tables[parent_seq_id]
        # 子序列共享所有物理块（增加引用计数）
        child_blocks = []
        for block in parent_blocks:
            block.ref_count += 1
            child_blocks.append(block)
        self.block_tables[child_seq_id] = child_blocks
    
    def cow_if_needed(self, seq_id: int, logical_block_idx: int) -> None:
        """写时复制：如果要修改的块是共享的，先复制"""
        block = self.block_tables[seq_id][logical_block_idx]
        if block.is_shared():
            # 分配新块并复制内容
            new_block = self.allocate_block()
            # 实际实现中会 GPU memcpy 复制 KV 数据
            # copy_kv_data(src=block, dst=new_block)
            self.block_tables[seq_id][logical_block_idx] = new_block
            block.ref_count -= 1
    
    def free_sequence(self, seq_id: int) -> None:
        """释放序列的所有块"""
        for block in self.block_tables[seq_id]:
            self.free_block(block)
        del self.block_tables[seq_id]
    
    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)
    
    @property
    def gpu_utilization(self) -> float:
        used = self.num_gpu_blocks - self.num_free_blocks
        return used / self.num_gpu_blocks
```

### 块大小配置

```python
# vLLM 中配置块大小
from vllm import LLM, SamplingParams

# 默认 block_size = 16
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    block_size=16,           # 默认值，通常无需修改
    gpu_memory_utilization=0.90,  # 允许使用 90% GPU 内存
    # swap_space=4,          # GB, CPU 交换空间
    # enable_prefix_caching=True,  # V1 中默认开启
)

# 查看 KV 缓存信息
# vLLM 启动时会打印类似信息：
# INFO: # GPU blocks: 7890, # CPU blocks: 512
# INFO: Maximum concurrency: ~120 requests (depends on seq length)
```

---

## 性能分析

### 与基准系统对比

论文中的基准测试结果（A100-40GB）：

| 系统 | 模型 | 吞吐量 (req/s) | 相对提升 |
|------|------|----------------|---------|
| HuggingFace Transformers | OPT-13B | 1.0x（基准）| - |
| HuggingFace TGI | OPT-13B | 3.4x | +240% |
| vLLM (PagedAttention) | OPT-13B | **14.0x** | +1300% |
| vLLM (PagedAttention) | OPT-175B | **24.3x** | +2330% |

### 内存效率提升

```
传统方案 vs PagedAttention 的内存使用对比（LLaMA-13B, max_seq=2048）：

传统预分配：
┌──────────────────────────────────────────────────────────┐
│ ████░░░░░░░░░░░░░░░ ████████░░░░░░░░░░░░ ██░░░░░░░░░░░ │
│ req1 (25% used)     req2 (50% used)      req3 (10% used)│
│ 总共可服务 3 个请求                                       │
│ 有效利用率: ~28%                                         │
└──────────────────────────────────────────────────────────┘

PagedAttention：
┌──────────────────────────────────────────────────────────┐
│ ████ ████████ ██ ██████ ████ ████████████ ██████ ██████  │
│ r1   r2       r3 r4     r5   r6           r7     r8     │
│ 总共可服务 8 个请求（同样的内存！）                        │
│ 有效利用率: >96%                                         │
└──────────────────────────────────────────────────────────┘
```

### 不同场景的性能提升

| 场景 | 吞吐量提升 | 主要原因 |
|------|-----------|---------|
| 标准请求服务 | 2-4x | 更高内存利用率 → 更大批量 |
| 长序列 (>4K tokens) | 4-8x | 长序列浪费更严重，PagedAttention 收益更大 |
| 并行采样 (n=4) | 3-6x | CoW 内存共享 |
| 束搜索 (beam=8) | 5-10x | CoW + 大量共享前缀 |
| 共享系统提示词 | 2-5x | 前缀缓存 |

---

## 后续发展

### vAttention (ASPLOS 2025)

vAttention 提出了一种不同的方案：利用 CUDA 的**低级虚拟内存管理 API** 实现按需分配，同时保持虚拟地址连续。

```
vAttention 的核心思想：

传统方案：物理连续 + 虚拟连续
  虚拟地址：[0x1000─────────────────0x5000]
  物理地址：[0x1000─────────────────0x5000]  (必须预分配)

PagedAttention：物理不连续 + 虚拟不连续（通过块表间接寻址）
  虚拟地址：不使用
  物理块：  [块A] ... [块C] ... [块B] ... (分散)
  块表：    0→A, 1→C, 2→B (软件层面的间接寻址)

vAttention：物理不连续 + 虚拟连续（通过 CUDA VMM API）
  虚拟地址：[0x1000─────────────────0x5000]  (连续)
  物理地址：[帧X] ... [帧Z] ... [帧Y] ...   (不连续)
  映射：    由 CUDA VMM 硬件管理
```

vAttention 的优势：
- **兼容所有现有注意力内核**：无需修改 FlashAttention 等
- 预填充阶段快 **3.92 倍**（因为可以直接使用 FlashAttention 的连续内存路径）
- Token 生成快 **1.97 倍**
- 无需自定义 CUDA 内核维护开销

vAttention 的局限：
- 依赖 CUDA VMM API（仅 NVIDIA GPU）
- 虚拟地址空间有限（需要预留较大的虚拟地址范围）
- CoW 支持需要额外的 VMM 操作

### TokenAttention

TokenAttention 将管理粒度从块级别细化到 **token 级别**：

- 每个 token 的 KV 缓存独立管理
- 无最后一个块的浪费问题
- 但管理开销更大（更大的映射表）
- 适合长序列场景

### 动态块大小

研究方向之一是根据负载动态调整块大小：

- 短序列多时使用小块
- 长序列多时使用大块
- 在运行时自适应切换

### 硬件级支持

未来硬件可能直接支持分页：
- NVIDIA 的 CUDA VMM API 已经提供基础设施（vAttention 利用了这一点）
- 未来 GPU 可能集成专门的 KV 缓存地址转换单元

---

## 不足与局限

### 块粒度的浪费

虽然 PagedAttention 大幅减少了浪费，但仍然存在最后一个块的浪费：

```
block_size = 16 时：
- 平均每个序列浪费 8 个 token 的空间
- 对短序列（如 32 token）：浪费 25%
- 对长序列（如 4096 token）：浪费 0.2%
- 如果有 1000 个并发序列：浪费 8000 token 的空间

虽然比传统方案好得多，但仍有优化空间
→ TokenAttention 等方案试图解决这一问题
```

### 间接寻址开销

块表引入了额外的间接寻址层：

- 每次注意力计算需要查表
- 内存访问模式不如连续内存高效（缓存行利用率下降）
- 对于 prefill 阶段（大量连续访问），开销相对更明显
  - 这也是 vAttention 提出的动机之一

### 自定义内核维护成本

PagedAttention 需要维护自定义 CUDA 内核：

- 不能直接使用标准的 FlashAttention（需要修改版）
- 每当新的注意力优化出现（如 FlashAttention-3），都需要适配
- 不同硬件平台（AMD、Intel、TPU）需要各自的实现
- vAttention 通过保持虚拟地址连续避免了这一问题

### 跨设备扩展

在多 GPU 场景下，PagedAttention 的块管理需要额外考虑：

- Tensor Parallelism 下，同一序列的块需要在所有 GPU 上同步
- 预填充-解码分离时，块需要跨设备传输
- 分布式场景下的块表同步开销

---

## 参考文献

1. **Kwon et al.** "Efficient Memory Management for Large Language Model Serving with PagedAttention" — SOSP 2023. [论文](https://arxiv.org/abs/2309.06180) [代码](https://github.com/vllm-project/vllm)
   - 提出 PagedAttention 算法和 vLLM 系统

2. **Panwar et al.** "vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention" — ASPLOS 2025. [论文](https://arxiv.org/abs/2405.04437)
   - 使用 CUDA VMM API 替代软件分页

3. **Yu et al.** "Orca: A Distributed Serving System for Transformer-Based Generative Models" — OSDI 2022.
   - 连续批处理的开创性工作，PagedAttention 在此基础上进一步优化

4. **Dao et al.** "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" — NeurIPS 2022.
   - FlashAttention 与 PagedAttention 的结合是现代推理系统的标准配置

5. **Zheng et al.** "SGLang: Efficient Execution of Structured Language Model Programs" — 2024.
   - RadixAttention 前缀缓存方案

---

## 相关页面

- [[vllm]] — 基于 PagedAttention 的服务引擎
- [[kv-cache-optimization]] — 更广泛的 KV 缓存优化技术
- [[continuous-batching]] — PagedAttention 支持的调度方式
- [[prefill-decode-disaggregation]] — 分离架构中的 KV 缓存传输
- [[sglang]] — RadixAttention 前缀缓存
- [[quantization]] — KV 缓存量化与 PagedAttention 结合
