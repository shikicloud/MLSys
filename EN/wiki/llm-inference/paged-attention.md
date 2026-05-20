---
title: "PagedAttention: Virtual Memory for KV Cache"
category: llm-inference
tags: [paged-attention, kv-cache, memory-management, vllm, virtual-memory]
created: 2026-04-13
updated: 2026-05-20
status: mature
---

# PagedAttention: Virtual Memory for KV Cache

> [!abstract]+ TL;DR
> PagedAttention (Kwon et al., SOSP 2023) ports the **OS virtual-memory paging** mechanism to KV cache management. Before it, LLM serving systems wasted **60-80%** of GPU memory to fragmentation and pre-allocation; PagedAttention drives the waste to **< 4%** and delivers **2-4×** throughput on the same hardware. It is the core innovation of [[vllm|vLLM]] and is now adopted by virtually every mainstream serving framework ([[sglang|SGLang]], [[tensorrt-llm|TensorRT-LLM]], HuggingFace TGI). Combined with [[continuous-batching|continuous batching]], it defines the efficiency baseline of modern LLM serving.

---

## Problems with the traditional KV cache

### KV cache basics

During autoregressive decoding, generating each new token requires attention over the Keys and Values of all previous tokens. To avoid recomputation, every layer's K and V tensors are cached — the **KV cache**.

KV cache size per request:

```
KV_size = 2 × num_layers × num_kv_heads × head_dim × seq_len × dtype_bytes
```

For LLaMA-13B (40 layers, 40 heads, 128 head_dim, FP16):
- Max sequence length 2048: `2 × 40 × 40 × 128 × 2048 × 2 bytes = 1.6 GB`
- A single request can consume a large chunk of GPU memory

### Pre-allocation waste (60-80%)

Traditional systems use a **pre-allocation policy**: on request arrival, a contiguous block of memory big enough for the maximum sequence length is reserved for the request's KV cache.

```
Traditional pre-allocation (max_seq_len = 2048 tokens)

Request A (uses 327 tokens):
┌─────────┬──────────────────────────────────────────────────┐
│  in use │            reserved but wasted memory            │
│ 327 tok │            space for 1721 tokens wasted          │
└─────────┴──────────────────────────────────────────────────┘
 ←─ 16% ─→←──────────── 84% wasted ─────────────────────────→

Request B (uses 1150 tokens):
┌──────────────────────────┬─────────────────────────────────┐
│        in use             │       reserved but wasted       │
│      1150 tokens          │       space for 898 tokens      │
└──────────────────────────┴─────────────────────────────────┘
 ←──────── 56% ───────────→←──────── 44% wasted ────────────→

Request C (uses 89 tokens):
┌──┬─────────────────────────────────────────────────────────┐
│in│                 reserved but wasted memory               │
│89│              space for 1959 tokens wasted                 │
└──┴─────────────────────────────────────────────────────────┘
 4%←───────────────── 96% wasted ───────────────────────────→
```

Average waste is typically **60-80%** because:
1. Most requests have actual output much shorter than the maximum
2. All memory must be locked in at the start of generation
3. The system cannot predict each request's actual output length

### Internal fragmentation

Within a pre-allocated block, the allocated-but-unused portion is internal fragmentation. Because this memory cannot be reclaimed for other requests, even when total free GPU memory is abundant, new concurrent requests cannot be admitted.

```
GPU memory
┌─────────────────────────────────────────────────────┐
│ Req A KV cache (pre-alloc)  [██░░░░░░░░░░░░░░░░░░░░]│  ██ = used
│ Req B KV cache (pre-alloc)  [██████████░░░░░░░░░░░░]│  ░░ = internal frag
│ Req C KV cache (pre-alloc)  [█░░░░░░░░░░░░░░░░░░░░░]│
│                                                     │
│ ╳ Cannot fit a new request! (total free > needed)  │
│ ╳ Because no *contiguous* free block is big enough  │
└─────────────────────────────────────────────────────┘
```

### External fragmentation

When multiple requests finish and free their memory, the free regions become a set of disjoint small pieces. Total free memory may suffice, but no single piece is large enough to serve a new request.

```
GPU memory after requests finish
┌─────────────────────────────────────────────────────┐
│ [free1] [Req D ██████] [free2] [Req E ████] [free3] │
│                                                     │
│ free1 + free2 + free3 = 3GB (total enough)         │
│ But no single piece is big enough for a new request │
│ (needs 2GB contiguous)                              │
│                                                     │
│ This is the external fragmentation problem!         │
└─────────────────────────────────────────────────────┘
```

### Why reservation strategies fail

Many reservation strategies have been tried:

| Strategy | Approach | Problem |
|------|------|------|
| **Max-length pre-allocation** | Reserve max_seq_len per request | Worst waste, 60-80% |
| **Predicted-length pre-alloc** | Predict output length from history | Either still wasted or OOM when prediction misses |
| **Incremental expansion** | Grow the block on demand | Requires memory copy, adds latency |
| **Memory pool** | Pre-allocate fixed-size slabs | Granularity mismatch, still internal frag |

None solves the problem fundamentally, because they are all bound by the constraint of contiguous allocation. PagedAttention's key insight: break the contiguity constraint.

---

## How PagedAttention works

### Core idea: borrow OS virtual memory

Operating systems solved physical-memory fragmentation with virtual memory:
- Processes see a contiguous **virtual address space**
- Physical memory is divided into fixed-size **page frames**
- A **page table** maps virtual pages to physical frames
- Applications do not need to worry about physical contiguity

PagedAttention applies the same idea to KV cache management:

| OS concept | PagedAttention analog |
|---------|---------------------|
| Virtual page | Logical block |
| Physical page frame | Physical block |
| Page table | Block table |
| Process | Sequence |
| Page size | Block size |

### Block abstraction

PagedAttention partitions the KV cache into fixed-size **blocks**, each block storing the Key and Value vectors of a fixed number of tokens.

**Logical block**:
- From the model's perspective, each sequence's KV cache is a list of contiguous logical blocks
- Logical block IDs start at 0 and follow token order
- During computation, the model uses (logical block ID, offset-within-block) to locate data

**Physical block**:
- Actual storage units in GPU memory
- Same size as logical blocks
- Physically scattered anywhere in GPU memory
- Managed via a free-block list

Single physical block size:

```python
block_size_bytes = block_size × num_layers × num_kv_heads × head_dim × dtype_bytes × 2
# Example: block_size=16, LLaMA-7B (32 layers, 32 heads, 128 dim, FP16):
# 16 × 32 × 32 × 128 × 2 × 2 = 32 MB
# Note: in practice each layer is allocated independently, so one block per layer:
# 16 × 32 × 128 × 2 × 2 = 1 MB (includes K and V)
```

### Block table

The block table is PagedAttention's core data structure: a mapping from logical blocks to physical blocks.

```
Sequence "The cat sat on the mat and then ..." (assume block_size = 4)

Logical-block view (as seen by the sequence):
┌──────────────┬──────────────┬──────────────┬──────────────┐
│ logical 0    │ logical 1    │ logical 2    │ logical 3    │
│ The cat sat  │ on the mat   │ and then the │ dog ...      │
│ on           │              │              │ (partial)    │
└──────────────┴──────────────┴──────────────┴──────────────┘
   4 tokens       4 tokens       4 tokens      2/4 tokens

Block table:
┌──────────┬──────────────┐
│ logical  │ physical     │
├──────────┼──────────────┤
│    0     │     7        │
│    1     │     3        │
│    2     │    12        │
│    3     │     1        │
└──────────┴──────────────┘

GPU physical memory layout (physical blocks are not contiguous):
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│  0  │ *1* │  2  │ *3* │  4  │  5  │  6  │ *7* │  8  │  9  │ 10  │ 11  │*12* │
│other│ L3  │other│ L1  │free │free │other│ L0  │other│other│free │other│ L2  │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
  * marked blocks belong to this sequence

Note: logically contiguous blocks 0,1,2,3 map to physical 7,3,12,1 — totally non-contiguous!
```

### On-demand allocation

PagedAttention allocates memory strictly on demand:

```
Timeline (block_size = 4):

t=0: request arrives, prompt = "The cat sat on"
     allocate physical block 7 → logical 0  [The, cat, sat, on]  (full)
     
t=1: generate "the" → need a new block
     allocate physical block 3 → logical 1  [the, _, _, _]  (1/4)
     
t=2: generate "mat"
     logical 1 not full, append in place    [the, mat, _, _]  (2/4)
     
t=3: generate "and"
     logical 1 not full, append in place    [the, mat, and, _]  (3/4)
     
t=4: generate "then"
     logical 1 now full                     [the, mat, and, then]  (4/4 full)
     
t=5: generate "the" → need a new block
     allocate physical block 12 → logical 2 [the, _, _, _]  (1/4)
     
... when the request finishes, release physical blocks 7, 3, 12 back to the free list
```

Key advantages of on-demand allocation:
1. **Zero pre-allocation waste**: no need to predict output length
2. **Last-block waste only**: on average `block_size / 2` tokens of space are wasted
3. **Immediate reclamation**: every physical block is freed the moment the request finishes

### Non-contiguous memory access in attention computation

Traditional attention assumes the KV cache is contiguous in memory. PagedAttention needs to modify the attention kernel so that it correctly reads data from non-contiguous physical blocks.

The attention formula:

```
Attention(Q, K, V) = softmax(Q × K^T / sqrt(d_k)) × V
```

In PagedAttention, Q comes from the current token (contiguous), but K and V are scattered across multiple physical blocks. Attention computation must:

1. Find all relevant physical blocks via the block table
2. Load K and V vectors from each block
3. Correctly combine partial attention scores from different blocks

### Custom CUDA kernel design

The PagedAttention CUDA kernel is the technical centerpiece. The kernel design has to solve several challenges:

```
PagedAttention CUDA kernel workflow:

Input:  query (current token), block_table, kv_cache_pool
Output: attention_output

For each attention head (in parallel):
  ┌──────────────────────────────────────────────┐
  │ 1. Look up the sequence's physical blocks    │
  │    from the block_table                      │
  │                                              │
  │ 2. For each physical block (in parallel):    │
  │    ├─ Load the block's K vectors             │
  │    ├─ Compute Q × K^T / sqrt(d)              │
  │    └─ Save partial attention scores          │
  │                                              │
  │ 3. Safe softmax across all blocks:           │
  │    ├─ Find global max (numerical stability)  │
  │    ├─ Compute normalized attention weights   │
  │    └─ Handle the last block's padding mask   │
  │                                              │
  │ 4. For each physical block (in parallel):    │
  │    ├─ Load the block's V vectors             │
  │    └─ Compute weighted sum (attn × V)        │
  │                                              │
  │ 5. Accumulate per-block results into output  │
  └──────────────────────────────────────────────┘
```

Key kernel optimizations:
- **Block-wise reduction**: each CUDA thread block processes one or more KV blocks and reduces via shared memory
- **Online softmax**: use the Milakov & Gimelshein online-softmax algorithm to avoid two passes
- **Coalesced access**: even though physical blocks are not contiguous, within-block access is contiguous
- **Combined with FlashAttention**: later versions support paging inside the FlashAttention framework

### Choosing the block size

Block size is PagedAttention's most important hyperparameter:

| Block size | Pros | Cons |
|--------|------|------|
| Small (1-4) | Minimal waste, fine-grained allocation | Large block table, kernel inefficient, more indirection |
| Medium (16) | Balance of waste and efficiency | **Usually the sweet spot** |
| Large (64-256) | Kernel efficient, close to contiguous | Last-block waste, less flexibility |

vLLM defaults to **block_size = 16**, the balance point established by extensive benchmarking:

```python
# Waste analysis
# Average waste of last block per sequence = block_size / 2 tokens
# For block_size = 16:
#   average waste of 8 tokens worth of KV cache
#   sequence of length 2048: 8/2048 = 0.4% waste
#   sequence of length 128:  8/128 = 6.25% waste (still far better than tradition)

# Compared to traditional pre-allocation (max_seq_len=2048, actual 128 tokens):
#   (2048 - 128) / 2048 = 93.75% waste
```

---

## Copy-on-Write

### Memory challenge of parallel sampling

Many LLM inference scenarios require generating multiple different outputs from the same prefix:

- **Parallel sampling**: generate N candidate answers for one prompt
- **Beam search**: maintain K best candidate sequences
- **Best-of-N**: generate N answers and pick the best

In these scenarios multiple sequences share the same prefix KV cache. The traditional approach copies the entire prefix KV cache for each sequence — a massive waste of memory.

### Reference counting

PagedAttention borrows the OS Copy-on-Write (CoW) mechanism, using reference counts to share KV-cache blocks:

```
Parallel sampling (n=3), initial state:

Prompt: "Write a poem about spring"
          (occupies 2 logical blocks)

                     Block tables
Seq 1:   logical 0 → physical 5 (ref_count=3)
         logical 1 → physical 9 (ref_count=3)

Seq 2:   logical 0 → physical 5 (ref_count=3)  ← shared!
         logical 1 → physical 9 (ref_count=3)  ← shared!

Seq 3:   logical 0 → physical 5 (ref_count=3)  ← shared!
         logical 1 → physical 9 (ref_count=3)  ← shared!

Physical memory: only one copy of the prefix KV cache
Traditional approach: three copies!
```

### CoW trigger

When a sequence tries to modify a shared block, CoW is triggered:

```
Step 1: Seq 1 generates a new token and needs to modify logical 1 (last block, partially filled)

Check: physical block 9's ref_count = 3 > 1 → CoW needed!

Step 2: allocate a new physical block 14, copy contents of block 9 into it

Step 3: update mappings and ref counts

Seq 1:   logical 0 → physical 5  (ref_count=3)
         logical 1 → physical 14 (ref_count=1)  ← new private block
         logical 2 → physical 20 (ref_count=1)  ← newly allocated (if needed)

Seq 2:   logical 0 → physical 5  (ref_count=3)
         logical 1 → physical 9  (ref_count=2)  ← ref count -1

Seq 3:   logical 0 → physical 5  (ref_count=3)
         logical 1 → physical 9  (ref_count=2)

Only the modified block is copied; preceding shared blocks remain shared!
```

### CoW in beam search

Beam search is where CoW pays off the most, because beams are frequently pruned and duplicated:

```
Beam search (beam_width=4) example:

Step 0 (initial): all beams share the prompt's KV cache
┌──────────────────────────────────────────────────────┐
│  Beam 0 ──→ [block A][block B]                       │
│  Beam 1 ──→ [block A][block B]  (all beams share)    │
│  Beam 2 ──→ [block A][block B]                       │
│  Beam 3 ──→ [block A][block B]                       │
│                                                      │
│  Block A: ref_count=4                                │
│  Block B: ref_count=4                                │
│  Total physical blocks: 2 (vs. 8 traditional)        │
└──────────────────────────────────────────────────────┘

Step 5: beams have diverged
┌──────────────────────────────────────────────────────┐
│  Beam 0 ──→ [A][B][C][D0]                            │
│  Beam 1 ──→ [A][B][C][D1]   (first 3 blocks shared)  │
│  Beam 2 ──→ [A][B][E][F]    (first 2 blocks shared)  │
│  Beam 3 ──→ [A][B][E][G]    (first 2 blocks shared)  │
│                                                      │
│  Block A: ref_count=4, B: ref_count=4                │
│  Block C: ref_count=2, E: ref_count=2                │
│  Others: ref_count=1                                 │
│  Total physical blocks: 9 (vs. 16 traditional)       │
└──────────────────────────────────────────────────────┘

Step 10: Beam 2 is pruned, Beam 1 expands
┌──────────────────────────────────────────────────────┐
│  Release Beam 2's private block F                    │
│  E's ref_count: 2 → 1                                │
│  No copy needed — just update ref counts             │
└──────────────────────────────────────────────────────┘
```

### Memory-savings analysis

CoW's memory savings in different scenarios:

| Scenario | Traditional | PagedAttention + CoW | Savings |
|------|---------|---------------------|------|
| Parallel sampling n=4, prefix 50% | 4× prefix + 4× output | 1× prefix + 4× output | ~38% |
| Beam search beam=8, long sequences | 8× full sequences | ~3× equivalent (heavy sharing) | ~55% |
| Best-of-16 | 16× full sequences | ~6× equivalent | ~60% |

Paper results:
- Parallel sampling: **2.2× throughput** (from memory sharing alone, no other optimizations)
- Beam search: **up to 55% memory reduction**

---

## Prefix caching

### Motivation

In real production deployments, many requests share the same prefix:

- **System prompt**: every request uses the same system instruction
- **Few-shot examples**: the same examples prepended to multiple requests
- **Multi-turn dialogue**: every turn contains all prior turns
- **RAG**: retrieved document chunks may be repeated

Identifying and reusing the KV cache of these shared prefixes avoids a great deal of redundant computation.

### Hash-based prefix cache (vLLM approach)

vLLM V1 uses content-hash-based prefix caching:

```
How prefix caching works:

Request 1: [system-prompt tokens] + [user question A]
            hash(block0) = 0xAB12  → compute and cache the physical block
            hash(block1) = 0xCD34  → compute and cache the physical block
            hash(block2) = 0xEF56  → compute and cache the physical block (user question)

Request 2: [system-prompt tokens] + [user question B]
            hash(block0) = 0xAB12  → cache hit! reuse ✓
            hash(block1) = 0xCD34  → cache hit! reuse ✓
            hash(block2) = 0x7890  → miss, must compute

Savings: skip the system prompt's prefill!
```

Hashing strategy:
- Each logical block is hashed from its **token content**
- The hash also incorporates the block's **position** (since positional encoding affects KV)
- LRU policy is used to evict cached blocks

### Radix-tree-based prefix cache (SGLang RadixAttention)

[[sglang|SGLang]] uses a radix tree for more efficient prefix matching:

```
RadixAttention's radix-tree structure:

                        [root]
                       /      \
            [system prompt...] [another system prompt...]
            /     |     \
    [user Q A] [user Q B] [Few-shot prefix]
        |          |         /    \
    [answer A] [answer B] [Q C]  [Q D]

Advantages:
- O(n) prefix matching (n = shared prefix length)
- Token-level precision
- Naturally supports multi-turn hierarchy
- Eviction can be precise down to subtrees
```

### Comparing the two approaches

| Aspect | Hash (vLLM) | Radix tree (SGLang) |
|------|-----------------|-------------------|
| Match granularity | Block level | Token level |
| Lookup complexity | O(1) hash lookup | O(n) tree traversal |
| Prefix-heavy performance | Good | Better (~29% faster) |
| Implementation complexity | Lower | Higher |
| Multi-turn optimization | Good | More natural |

### System-prompt optimization

Prefix caching delivers a striking win for system prompts:

```python
# Typical: system prompt 1000 tokens, user message 200 tokens
# Assume prefill speed 10,000 tok/s

# No prefix cache:
#   every request: prefill 1200 tokens → 120ms TTFT

# With prefix cache (system prompt hits):
#   first request:  prefill 1200 tokens → 120ms TTFT
#   later requests: prefill 200 tokens  → 20ms TTFT
#   TTFT down 83%!

# In vLLM V1, prefix caching is on by default
# Miss-rate overhead < 1% (effectively free)
```

### Performance benefit

Real-world prefix-cache gains depend on workload characteristics:

| Workload | Prefix hit rate | TTFT reduction | Throughput gain |
|---------|-----------|----------|-----------|
| Single system prompt + short user messages | >90% | 60-85% | 1.5-3x |
| Multi-turn chat (3-5 turns) | 70-90% | 40-70% | 1.3-2x |
| RAG (shared documents) | 30-60% | 20-40% | 1.1-1.5x |
| Fully random requests | ~0% | ~0% | ~0% (<1% overhead) |

---

## Code examples

### Simplified PagedAttention kernel pseudocode

```python
import torch

def paged_attention_forward(
    query: torch.Tensor,         # [batch, num_heads, 1, head_dim] (single-token decode)
    key_cache: torch.Tensor,     # [num_physical_blocks, block_size, num_kv_heads, head_dim]
    value_cache: torch.Tensor,   # [num_physical_blocks, block_size, num_kv_heads, head_dim]
    block_tables: torch.Tensor,  # [batch, max_num_blocks] logical→physical mapping
    context_lens: torch.Tensor,  # [batch] current length per sequence
    block_size: int = 16,
) -> torch.Tensor:
    """
    Simplified PagedAttention forward pass (the real implementation is a CUDA kernel).
    Shown in Python for clarity.
    """
    batch_size, num_heads, _, head_dim = query.shape
    scale = head_dim ** -0.5
    output = torch.zeros_like(query)
    
    for b in range(batch_size):
        seq_len = context_lens[b].item()
        num_blocks = (seq_len + block_size - 1) // block_size
        
        # Gather K, V from non-contiguous physical blocks
        keys = []
        values = []
        for logical_idx in range(num_blocks):
            physical_idx = block_tables[b, logical_idx].item()
            
            # Determine valid tokens in this block
            if logical_idx == num_blocks - 1:
                # The last block may be partially filled
                valid_tokens = seq_len - logical_idx * block_size
            else:
                valid_tokens = block_size
            
            # Load K, V from physical block
            keys.append(key_cache[physical_idx, :valid_tokens])
            values.append(value_cache[physical_idx, :valid_tokens])
        
        # Concatenate K, V from all blocks
        k = torch.cat(keys, dim=0)   # [seq_len, num_kv_heads, head_dim]
        v = torch.cat(values, dim=0) # [seq_len, num_kv_heads, head_dim]
        
        # Standard attention computation
        # (real CUDA kernels compute block by block with online softmax)
        for h in range(num_heads):
            kv_head = h // (num_heads // k.shape[1])  # GQA support
            attn_scores = (query[b, h] @ k[:, kv_head].T) * scale  # [1, seq_len]
            attn_weights = torch.softmax(attn_scores, dim=-1)
            output[b, h] = attn_weights @ v[:, kv_head]
    
    return output
```

### Simplified vLLM BlockSpaceManager

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

@dataclass
class PhysicalBlock:
    """Physical block: actual storage unit in GPU memory"""
    block_id: int
    ref_count: int = 0          # reference count, used for CoW
    
    def is_shared(self) -> bool:
        return self.ref_count > 1

class BlockSpaceManager:
    """
    Simplified vLLM block-space manager.
    The real implementation is more complex with CPU/GPU swap, prefix cache, etc.
    """
    
    def __init__(
        self,
        block_size: int = 16,
        num_gpu_blocks: int = 1024,
    ):
        self.block_size = block_size
        self.num_gpu_blocks = num_gpu_blocks
        
        # Initialize all physical blocks
        self.gpu_blocks = [
            PhysicalBlock(block_id=i) for i in range(num_gpu_blocks)
        ]
        
        # Free block list
        self.free_blocks: List[PhysicalBlock] = list(self.gpu_blocks)
        
        # Per-sequence block table: seq_id → [physical blocks]
        self.block_tables: Dict[int, List[PhysicalBlock]] = {}
    
    def can_allocate(self, num_blocks_needed: int) -> bool:
        """Check whether enough free blocks are available"""
        return len(self.free_blocks) >= num_blocks_needed
    
    def allocate_block(self) -> PhysicalBlock:
        """Allocate one physical block"""
        if not self.free_blocks:
            raise RuntimeError("Out of GPU memory! No free physical blocks")
        block = self.free_blocks.pop()
        block.ref_count = 1
        return block
    
    def free_block(self, block: PhysicalBlock) -> None:
        """Free one physical block (decrement ref count)"""
        block.ref_count -= 1
        if block.ref_count == 0:
            self.free_blocks.append(block)
    
    def allocate_sequence(self, seq_id: int, num_initial_tokens: int) -> None:
        """Allocate initial blocks for a new sequence"""
        num_blocks = (num_initial_tokens + self.block_size - 1) // self.block_size
        blocks = [self.allocate_block() for _ in range(num_blocks)]
        self.block_tables[seq_id] = blocks
    
    def append_token(self, seq_id: int, num_new_tokens: int = 1) -> None:
        """Allocate new blocks on demand when appending tokens"""
        blocks = self.block_tables[seq_id]
        current_tokens = len(blocks) * self.block_size  # simplified: assume previous are full
        
        # Check whether the last block has room
        last_block = blocks[-1]
        remaining_in_last = self.block_size - (current_tokens % self.block_size)
        if remaining_in_last == self.block_size:
            remaining_in_last = 0
        
        if remaining_in_last < num_new_tokens:
            # Need new physical blocks
            new_blocks_needed = (num_new_tokens - remaining_in_last + 
                                self.block_size - 1) // self.block_size
            for _ in range(new_blocks_needed):
                blocks.append(self.allocate_block())
    
    def fork_sequence(self, parent_seq_id: int, child_seq_id: int) -> None:
        """Fork a sequence (parallel sampling / beam search) — share blocks via CoW"""
        parent_blocks = self.block_tables[parent_seq_id]
        # Child sequence shares all physical blocks (increment ref counts)
        child_blocks = []
        for block in parent_blocks:
            block.ref_count += 1
            child_blocks.append(block)
        self.block_tables[child_seq_id] = child_blocks
    
    def cow_if_needed(self, seq_id: int, logical_block_idx: int) -> None:
        """Copy-on-Write: if the block to modify is shared, copy first"""
        block = self.block_tables[seq_id][logical_block_idx]
        if block.is_shared():
            # Allocate a new block and copy contents
            new_block = self.allocate_block()
            # In real impl, GPU memcpy of KV data
            # copy_kv_data(src=block, dst=new_block)
            self.block_tables[seq_id][logical_block_idx] = new_block
            block.ref_count -= 1
    
    def free_sequence(self, seq_id: int) -> None:
        """Free all blocks of a sequence"""
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

### Configuring block size

```python
# Configure block size in vLLM
from vllm import LLM, SamplingParams

# Default block_size = 16
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    block_size=16,           # default, usually no need to change
    gpu_memory_utilization=0.90,  # allow up to 90% of GPU memory
    # swap_space=4,          # GB, CPU swap space
    # enable_prefix_caching=True,  # on by default in V1
)

# Inspect KV cache info
# vLLM prints information like this on startup:
# INFO: # GPU blocks: 7890, # CPU blocks: 512
# INFO: Maximum concurrency: ~120 requests (depends on seq length)
```

---

## Performance analysis

### Vs. baseline systems

Benchmark results from the paper (A100-40GB):

| System | Model | Throughput (req/s) | Relative |
|------|------|----------------|---------|
| HuggingFace Transformers | OPT-13B | 1.0x (baseline) | - |
| HuggingFace TGI | OPT-13B | 3.4x | +240% |
| vLLM (PagedAttention) | OPT-13B | **14.0x** | +1300% |
| vLLM (PagedAttention) | OPT-175B | **24.3x** | +2330% |

### Memory-efficiency gain

```
Traditional vs PagedAttention memory usage (LLaMA-13B, max_seq=2048):

Traditional pre-allocation:
┌──────────────────────────────────────────────────────────┐
│ ████░░░░░░░░░░░░░░░ ████████░░░░░░░░░░░░ ██░░░░░░░░░░░ │
│ req1 (25% used)     req2 (50% used)      req3 (10% used)│
│ Serves 3 requests total                                  │
│ Effective utilization: ~28%                              │
└──────────────────────────────────────────────────────────┘

PagedAttention:
┌──────────────────────────────────────────────────────────┐
│ ████ ████████ ██ ██████ ████ ████████████ ██████ ██████  │
│ r1   r2       r3 r4     r5   r6           r7     r8     │
│ Serves 8 requests with the same memory!                  │
│ Effective utilization: >96%                              │
└──────────────────────────────────────────────────────────┘
```

### Gains across scenarios

| Scenario | Throughput gain | Main driver |
|------|-----------|---------|
| Standard request serving | 2-4x | Higher memory utilization → bigger batches |
| Long sequences (>4K tokens) | 4-8x | Larger absolute waste in tradition, bigger paged-attention win |
| Parallel sampling (n=4) | 3-6x | CoW memory sharing |
| Beam search (beam=8) | 5-10x | CoW + heavy prefix sharing |
| Shared system prompts | 2-5x | Prefix cache |

---

## Follow-up work

### vAttention (ASPLOS 2025)

vAttention takes a different approach: it uses CUDA's **low-level virtual-memory management API** to implement on-demand allocation while keeping virtual addresses contiguous.

```
Core idea of vAttention:

Traditional: physical contiguous + virtual contiguous
  virtual: [0x1000─────────────────0x5000]
  physical:[0x1000─────────────────0x5000]  (must pre-allocate)

PagedAttention: physical non-contiguous + virtual non-contiguous (block-table indirection)
  virtual: not used
  physical:  [block A] ... [block C] ... [block B] ... (scattered)
  block table: 0→A, 1→C, 2→B (software-level indirection)

vAttention: physical non-contiguous + virtual contiguous (via CUDA VMM API)
  virtual: [0x1000─────────────────0x5000]  (contiguous)
  physical:[frame X] ... [frame Z] ... [frame Y] ...   (non-contiguous)
  mapping: managed by CUDA VMM hardware
```

vAttention's advantages:
- **Compatible with every existing attention kernel**: no need to modify FlashAttention etc.
- Prefill phase **3.92×** faster (can use FlashAttention's contiguous-memory path directly)
- Token generation **1.97×** faster
- No maintenance burden for custom CUDA kernels

vAttention's limitations:
- Depends on the CUDA VMM API (NVIDIA GPU only)
- Limited virtual-address space (needs a large reserved virtual range)
- CoW support requires extra VMM operations

### TokenAttention

TokenAttention pushes the management granularity from block level to **token level**:

- Each token's KV cache is managed independently
- No last-block waste
- But higher management overhead (larger mapping table)
- Best fit for long-sequence scenarios

### Dynamic block size

One research direction is to adapt block size to the workload:

- Use smaller blocks when short sequences dominate
- Use larger blocks when long sequences dominate
- Switch adaptively at runtime

### Hardware-level support

Future hardware may directly support paging:
- NVIDIA's CUDA VMM API already provides the infrastructure (exploited by vAttention)
- Future GPUs may integrate dedicated KV-cache address-translation units

---

## Limitations

### Block-granularity waste

PagedAttention drastically reduces waste, but the last block still wastes space:

```
With block_size = 16:
- Average waste per sequence: 8 tokens of space
- Short sequences (e.g. 32 tokens): 25% wasted
- Long sequences (e.g. 4096 tokens): 0.2% wasted
- With 1000 concurrent sequences: 8000 tokens of wasted space

Much better than tradition, but still room for improvement
→ TokenAttention and others target this gap
```

### Indirection overhead

The block table adds an extra layer of indirection:

- Every attention computation requires a table lookup
- Memory-access patterns are less efficient than contiguous memory (cache-line utilization drops)
- During prefill (large contiguous accesses) the overhead becomes more visible
  - This is one of vAttention's motivations

### Custom kernel maintenance cost

PagedAttention requires custom CUDA kernels:

- Cannot use standard FlashAttention directly (need a modified version)
- Every new attention optimization (e.g. FlashAttention-3) needs to be ported
- Each hardware platform (AMD, Intel, TPU) needs its own implementation
- vAttention sidesteps this by keeping virtual addresses contiguous

### Cross-device scaling

Multi-GPU deployments add concerns for PagedAttention's block management:

- Under tensor parallelism, a sequence's blocks must be synchronized across all GPUs
- Under prefill-decode disaggregation, blocks must be transferred across devices
- Block-table synchronization overhead in distributed settings

---

## References

1. **Kwon et al.** "Efficient Memory Management for Large Language Model Serving with PagedAttention" — SOSP 2023. [Paper](https://arxiv.org/abs/2309.06180) [Code](https://github.com/vllm-project/vllm)
   - Proposed the PagedAttention algorithm and the vLLM system

2. **Panwar et al.** "vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention" — ASPLOS 2025. [Paper](https://arxiv.org/abs/2405.04437)
   - Uses the CUDA VMM API instead of software paging

3. **Yu et al.** "Orca: A Distributed Serving System for Transformer-Based Generative Models" — OSDI 2022.
   - The seminal continuous-batching work; PagedAttention builds on it

4. **Dao et al.** "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" — NeurIPS 2022.
   - FlashAttention combined with PagedAttention is the standard configuration in modern inference systems

5. **Zheng et al.** "SGLang: Efficient Execution of Structured Language Model Programs" — 2024.
   - The RadixAttention prefix-cache scheme

---

## Related pages

- [[vllm]] — Serving engine built on PagedAttention
- [[kv-cache-optimization]] — Broader KV-cache optimization techniques
- [[continuous-batching]] — The scheduling style that PagedAttention enables
- [[prefill-decode-disaggregation]] — KV-cache transfer in disaggregated serving
- [[sglang]] — RadixAttention prefix cache
- [[quantization]] — KV-cache quantization combined with PagedAttention
