---
title: "KV 缓存优化技术"
category: llm-inference
tags: [kv-cache, mqa, gqa, mla, 量化, 稀疏注意力, 内存优化]
created: 2026-04-13
updated: 2026-05-07
status: mature
---

# KV 缓存优化技术

> [!abstract]+ TL;DR
> KV 缓存是 LLM 服务的**首要内存瓶颈** —— 可达 **GPU 总显存的 70 %** —— 大小随序列长度 × 批量大小线性增长。优化栈从架构到字节：**架构层**（MHA → GQA → MQA → MLA，~MHA 的 3 %）、**内存管理**（[[paged-attention|PagedAttention]] 把浪费从 60–80 % 降到 < 4 %）、**量化**（FP8 → INT4 → INT4+BDR 旋转）、**压缩与驱逐**（H2O、StreamingLLM、KVTC）、**前缀缓存**（[[vllm|vLLM]] 哈希、[[sglang|SGLang]] RadixAttention）、**分布式**（LMCache、Mooncake）。现代生产栈：GQA + PagedAttention + FP8 KV + 前缀缓存。

## 概述

KV 缓存（Key-Value Cache）是 Transformer 自回归解码中最关键的数据结构。它存储已计算 token 的 Key 和 Value 向量，避免在每个解码步骤重复计算之前所有 token 的注意力。

KV 缓存是 LLM 推理的**首要内存瓶颈**，在长序列、大批量场景下可消耗 **高达 70% 的 GPU 总内存**。其大小随序列长度线性增长、随批量大小线性增长，直接限制了系统的最大并发数和可处理的上下文长度。

优化 KV 缓存是提升 LLM 服务效率的核心挑战，涉及从模型架构设计到系统层面的全栈优化。本文系统梳理当前主流的 KV 缓存优化技术。

---

## KV 缓存大小分析

### 计算公式

单个请求的 KV 缓存大小：

```
KV_cache_size = 2 × num_layers × num_kv_heads × head_dim × seq_len × dtype_bytes
```

其中：
- `2`：Key 和 Value 各一份
- `num_layers`：Transformer 层数
- `num_kv_heads`：KV 注意力头数（MHA 中等于 num_heads，GQA/MQA 中更少）
- `head_dim`：每个注意力头的维度
- `seq_len`：当前序列长度
- `dtype_bytes`：数据类型字节数（FP16=2，FP8=1，INT4=0.5）

### 典型模型的 KV 缓存大小

```python
def kv_cache_size_gb(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
    batch_size: int = 1,
    dtype_bytes: float = 2.0,  # FP16
) -> float:
    """计算 KV 缓存大小 (GB)"""
    size_bytes = (
        2 * num_layers * num_kv_heads * head_dim 
        * seq_len * batch_size * dtype_bytes
    )
    return size_bytes / (1024 ** 3)

# LLaMA-3.1-8B (GQA: 32层, 8 KV heads, 128 dim)
print(f"LLaMA-8B, seq=4K, bs=1:  {kv_cache_size_gb(32, 8, 128, 4096):.2f} GB")
print(f"LLaMA-8B, seq=4K, bs=32: {kv_cache_size_gb(32, 8, 128, 4096, 32):.2f} GB")
print(f"LLaMA-8B, seq=128K, bs=1: {kv_cache_size_gb(32, 8, 128, 131072):.2f} GB")

# LLaMA-3.1-70B (GQA: 80层, 8 KV heads, 128 dim)
print(f"LLaMA-70B, seq=4K, bs=1:  {kv_cache_size_gb(80, 8, 128, 4096):.2f} GB")
print(f"LLaMA-70B, seq=4K, bs=16: {kv_cache_size_gb(80, 8, 128, 4096, 16):.2f} GB")

# DeepSeek-V3 (MLA: 61层, 等效很小的 KV heads, 但需要特殊计算)
# MLA 的 KV 缓存 ≈ 每层 512 维压缩向量 (而非常规 KV)
deepseek_v3_kv = 2 * 61 * 512 * 4096 * 2 / (1024**3)  # 简化估算
print(f"DeepSeek-V3 (MLA), seq=4K, bs=1: {deepseek_v3_kv:.3f} GB")

# 输出：
# LLaMA-8B, seq=4K, bs=1:  0.50 GB
# LLaMA-8B, seq=4K, bs=32: 16.00 GB
# LLaMA-8B, seq=128K, bs=1: 16.00 GB
# LLaMA-70B, seq=4K, bs=1:  1.25 GB
# LLaMA-70B, seq=4K, bs=16: 20.00 GB
# DeepSeek-V3 (MLA), seq=4K, bs=1: 0.476 GB (远小于等效 MHA)
```

### KV 缓存增长可视化

```
KV 缓存大小随序列长度线性增长（LLaMA-8B, FP16, batch=1）：

内存
(GB)
 16 ┤                                                    ●  128K
    │                                                 ╱
 12 ┤                                              ╱
    │                                           ╱
  8 ┤                                        ╱
    │                                     ╱
  4 ┤                                  ╱
    │                               ╱
  2 ┤                        ●  32K
    │                  ●  16K
  1 ┤           ●  8K
0.5 ┤    ●  4K
0.25┤ ● 2K
    └──┬────┬────┬────┬────┬────┬────┬────┬──→ 序列长度
       2K   8K   16K  32K  64K  96K  128K

批量大小的乘数效应（seq=4K）：
  bs=1:   0.5 GB
  bs=8:   4.0 GB
  bs=32:  16.0 GB    ← 已占满大部分 H100 80GB 的 KV 预算
  bs=128: 64.0 GB    ← 超出单张 H100 内存！
```

### KV 缓存 vs 模型权重

```
LLaMA-3.1-70B (FP16) 内存分配示例：

模型权重:  140 GB (2 张 H100 的 TP)
                    ┌─────────────────────────────────┐
H100 #1 (80GB):    │ 权重 70GB │ KV缓存 ~8GB │ 其他 2GB │
                    └─────────────────────────────────┘
H100 #2 (80GB):    │ 权重 70GB │ KV缓存 ~8GB │ 其他 2GB │
                    └─────────────────────────────────┘

→ KV 缓存可用空间仅 ~8GB/卡
→ 限制了最大 batch_size × seq_len
→ 这就是为什么 KV 缓存优化如此重要！
```

---

## 架构级优化

架构级优化从模型设计层面减少每个注意力头需要存储的 KV 数据量，是最根本的优化方式。

### Multi-Head Attention (MHA) — 基线

标准 MHA 中，每个注意力头有独立的 Q、K、V 投影：

```
MHA (标准多头注意力):

Query heads:  Q1  Q2  Q3  Q4  Q5  Q6  Q7  Q8
               │   │   │   │   │   │   │   │
Key heads:    K1  K2  K3  K4  K5  K6  K7  K8    ← 8 个独立 KV heads
Value heads:  V1  V2  V3  V4  V5  V6  V7  V8

KV 缓存: 8 × 2 × head_dim × seq_len = 16 × head_dim × seq_len
```

### Multi-Query Attention (MQA)

MQA（Shazeer, 2019）将所有注意力头**共享同一对** K 和 V：

```
MQA (多查询注意力):

Query heads:  Q1  Q2  Q3  Q4  Q5  Q6  Q7  Q8
               │   │   │   │   │   │   │   │
               └───┴───┴───┼───┴───┴───┴───┘
                           │
Key head:                  K1                    ← 仅 1 个 KV head！
Value head:                V1

KV 缓存: 1 × 2 × head_dim × seq_len = 2 × head_dim × seq_len
KV 缓存减少: 8x (相比 MHA)
```

MQA 的特点：
- KV 缓存减少为 `1/num_heads`（如 32 头模型减少 32 倍）
- 解码速度显著提升（内存带宽需求降低）
- 可能影响模型质量（所有头共享相同 KV，表达能力下降）
- 代表模型：PaLM, StarCoder, Falcon

### Grouped-Query Attention (GQA)

GQA（Ainslie et al., 2023）是 MHA 和 MQA 的折中：将注意力头分组，每组共享一对 K、V。

```
GQA (分组查询注意力, 2 groups):

Query heads:  Q1  Q2  Q3  Q4 │ Q5  Q6  Q7  Q8
               │   │   │   │ │  │   │   │   │
               └───┴───┼───┘ │  └───┴───┼───┘
                       │     │          │
Key heads:             K1    │         K2         ← 2 个 KV heads
Value heads:           V1    │         V2

KV 缓存: 2 × 2 × head_dim × seq_len = 4 × head_dim × seq_len
KV 缓存减少: 4x (相比 MHA)
```

GQA 的特点：
- KV 缓存减少为 `num_kv_groups / num_heads`
- 质量接近 MHA（通过适当选择组数）
- 目前**最主流**的方案
- 代表模型：LLaMA-2/3, Mistral, Gemma, Qwen-2

### Multi-head Latent Attention (MLA)

MLA（DeepSeek-V2/V3, 2024）采用更激进的压缩策略：将 KV 投影到低维**潜在空间**，只缓存压缩后的潜在向量。

```
MLA (多头潜在注意力):

              ┌─── 原始 KV (高维) ───┐
              │                       │
              ▼                       ▼
         ┌─────────┐           ┌─────────┐
         │ 下投影 W_d│           │         │
         │ (压缩)   │           │         │
         └────┬────┘           └─────────┘
              │
              ▼
         ┌─────────┐
         │ 潜在向量  │  ← 只缓存这个！(维度: d_c << n_h × d_h)
         │ c_t      │     DeepSeek-V3: d_c = 512 vs 原始 16384
         └────┬────┘
              │
         ┌────┴────┐
         │ 上投影 W_u│
         │ (解压缩)  │
         └────┬────┘
              │
              ▼
         ┌─────────┐
         │ K, V heads│  ← 注意力计算时实时解压
         └─────────┘

KV 缓存: d_c × seq_len × dtype_bytes (远小于 MHA/GQA)
```

MLA 的特点：
- 压缩比极高：DeepSeek-V3 中 KV 缓存仅为 MHA 的 **~3%**
- 计算时需要上投影解压（额外计算，但可被吸收到注意力 GEMM 中）
- 需要特殊处理 RoPE 位置编码（因为压缩潜在向量不能直接应用 RoPE）
  - DeepSeek 的解决方案：对 RoPE 相关维度单独存储不压缩的分量
- 代表模型：DeepSeek-V2, DeepSeek-V3

### 架构对比总结

```
MHA vs MQA vs GQA vs MLA 对比（8 Query heads）：

         Q heads    KV heads    KV 缓存    质量
MHA:    Q1..Q8      K1..K8      8x          最佳
         ││││││││    ││││││││
         ↓↓↓↓↓↓↓↓    ↓↓↓↓↓↓↓↓
         8对1映射     8份KV

GQA-2:  Q1..Q8      K1, K2      2x          接近MHA
         ↓↓↓↓↓↓↓↓    ↓    ↓
         4对1映射     2份KV

MQA:    Q1..Q8      K1          1x          略低
         ↓↓↓↓↓↓↓↓    ↓
         8对1映射     1份KV

MLA:    Q1..Q8      c_t(压缩)   ~0.25x      接近MHA
         ↓↓↓↓↓↓↓↓    ↓
         解压后映射   1份压缩向量
```

| 架构 | KV 缓存大小 (相对MHA) | 模型质量 | 推理速度 | 代表模型 |
|------|----------------------|---------|---------|---------|
| MHA | 1x (基线) | 最好 | 最慢 | GPT-3 |
| MQA | 1/n_heads (~3%) | 略低 | 最快 | PaLM, Falcon |
| GQA | n_groups/n_heads (~25%) | 接近MHA | 快 | LLaMA-3, Mistral |
| MLA | ~3-5% | 接近MHA | 需解压 | DeepSeek-V3 |

### GQA 减少 KV 缓存的代码示例

```python
import torch
import torch.nn as nn

class GroupedQueryAttention(nn.Module):
    """GQA 实现，展示如何减少 KV 缓存"""
    
    def __init__(
        self,
        hidden_dim: int = 4096,
        num_q_heads: int = 32,
        num_kv_heads: int = 8,  # GQA: 每 4 个 Q head 共享 1 个 KV head
        head_dim: int = 128,
    ):
        super().__init__()
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_groups = num_q_heads // num_kv_heads  # = 4
        
        # Q 投影: 完整的 num_q_heads
        self.q_proj = nn.Linear(hidden_dim, num_q_heads * head_dim, bias=False)
        # K, V 投影: 只有 num_kv_heads (少很多!)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_q_heads * head_dim, hidden_dim, bias=False)
    
    def forward(self, x, kv_cache=None):
        batch, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch, seq_len, self.num_q_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)
        
        # KV 缓存只需存储 num_kv_heads 份（不是 num_q_heads 份）
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=1)
            v = torch.cat([v_cache, v], dim=1)
        new_kv_cache = (k, v)
        
        # 扩展 KV heads 以匹配 Q heads（只在计算时扩展，不存储）
        # [batch, seq, num_kv_heads, dim] → [batch, seq, num_q_heads, dim]
        k = k.repeat_interleave(self.num_groups, dim=2)
        v = v.repeat_interleave(self.num_groups, dim=2)
        
        # 标准注意力计算...
        # (实际使用 FlashAttention 等优化内核)
        
        return output, new_kv_cache
    
    def kv_cache_size_per_token(self) -> int:
        """每个 token 的 KV 缓存大小 (bytes, FP16)"""
        return 2 * self.num_kv_heads * self.head_dim * 2  # 2 for K,V; 2 for FP16
        # MHA 版本: 2 * self.num_q_heads * self.head_dim * 2
        # GQA 节省: num_q_heads / num_kv_heads = 4x

# 内存对比
mha_per_token = 2 * 32 * 128 * 2  # 16,384 bytes
gqa_per_token = 2 * 8 * 128 * 2   # 4,096 bytes
print(f"MHA per token: {mha_per_token:,} bytes")
print(f"GQA per token: {gqa_per_token:,} bytes")
print(f"GQA savings: {mha_per_token / gqa_per_token:.0f}x")
```

---

## 内存管理优化

### PagedAttention

[[paged-attention|PagedAttention]] 是内存管理层面最重要的优化，将 KV 缓存的内存浪费从 60-80% 降低到 <4%。详见 [[paged-attention]] 深度解析。

核心思想：将 KV 缓存分为固定大小的块（block），通过块表实现非连续物理内存的虚拟连续管理，完全按需分配。

### Token 级别 vs 块级别管理

| 管理粒度 | 代表方案 | 内存浪费 | 管理开销 | 适用场景 |
|---------|---------|---------|---------|---------|
| 最大长度预分配 | 传统方案 | 60-80% | 最低 | 已淘汰 |
| 块级别 (block) | PagedAttention | <4% | 低 | 主流标准 |
| Token 级别 | TokenAttention | ~0% | 较高 | 长序列优化 |
| 硬件虚拟内存 | vAttention | ~0% | 极低 | NVIDIA GPU |

### 动态内存分配策略

现代推理框架采用的内存管理策略：

```
动态分配 + 抢占机制：

1. 正常运行：按需分配物理块
   请求到达 → 分配 prompt 块 → 逐步分配生成块

2. 内存压力：当空闲块不足时
   ├─ 方案A: 暂停新请求（等待现有请求完成释放内存）
   ├─ 方案B: 抢占优先级低的请求
   │         ├─ Swap: 将 KV 缓存交换到 CPU 内存
   │         └─ Recompute: 丢弃 KV 缓存，需要时重新计算
   └─ 方案C: 压缩现有 KV 缓存（量化/驱逐）

3. 内存回收：请求完成后立即释放所有物理块
```

---

## KV 缓存量化

### FP8 KV 缓存

将 KV 缓存从 FP16（16 bit）量化到 FP8（8 bit），内存减半且精度损失极小。

```
FP8 E4M3 格式：
┌─────┬──────────┬─────────┐
│符号位│ 指数(4位) │ 尾数(3位)│
└─────┴──────────┴─────────┘
  1 bit   4 bits    3 bits = 8 bits total

vs FP16：
┌─────┬──────────┬──────────────┐
│符号位│ 指数(5位) │ 尾数(10位)    │
└─────┴──────────┴──────────────┘
  1 bit   5 bits    10 bits = 16 bits total

精度范围：
  FP16: ±65504, 精度 ~0.001
  FP8:  ±448,   精度 ~0.125
  对于注意力分数已足够！
```

FP8 KV 缓存特点：
- **2x 压缩**
- 精度损失通常 **< 0.5%**（在困惑度和下游任务上）
- Hopper (H100) 及以后架构原生支持 FP8
- 几乎所有框架已默认支持

```python
# vLLM 中启用 FP8 KV 缓存
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    kv_cache_dtype="fp8",       # FP8 KV 缓存
    # kv_cache_dtype="fp8_e4m3", # 或显式指定 E4M3 格式
    quantization="fp8",          # 权重也用 FP8
    tensor_parallel_size=4,
)
```

### INT4/INT8 KV 缓存

更激进的量化方案：

**INT8 KV 缓存**：
- 2x 压缩（同 FP8）
- 使用 per-channel 或 per-token 缩放因子
- 精度损失与 FP8 相当

**INT4 KV 缓存**（如 KIVI）：
- **4x 压缩**
- 需要精心设计的量化策略：
  - Key cache 使用 per-channel 量化（不同 channel 值域差异大）
  - Value cache 使用 per-token 量化（不同 token 值域差异大）
  - 保留少量最近 token 的 KV 为全精度（作为"锚点"）

```
KIVI 量化策略：

Key cache (per-channel quantization):
每个 channel 有独立的 scale 和 zero-point
┌──────────────────────────────────────────┐
│ ch0: [全精度scale] [INT4 values...]      │
│ ch1: [全精度scale] [INT4 values...]      │
│ ...                                      │
│ ch_d: [全精度scale] [INT4 values...]     │
└──────────────────────────────────────────┘

Value cache (per-token quantization):
每个 token 有独立的 scale 和 zero-point
┌──────────────────────────────────────────┐
│ tok0: [全精度scale] [INT4 values...]     │
│ tok1: [全精度scale] [INT4 values...]     │
│ ...                                      │
└──────────────────────────────────────────┘

最近 W 个 token 保持 FP16（保护质量）
```

### NVFP4 (Blackwell)

NVIDIA Blackwell 架构引入的 4-bit 浮点格式：

- **4x 压缩**（相比 FP16）
- 硬件原生支持，两级缩放机制
- 精度损失 **< 1%**
- 需要 B200/GB200 等 Blackwell GPU

### 量化精度 vs 内存权衡

| 格式 | 压缩比 | 精度损失 | 硬件需求 | 状态 |
|------|--------|---------|---------|------|
| FP16 (基线) | 1x | 无 | 任何 | 默认 |
| FP8 E4M3 | 2x | 极小 (<0.5%) | Hopper+ | 广泛使用 |
| INT8 | 2x | 极小 | 任何 | 广泛使用 |
| NVFP4 | 4x | 小 (<1%) | Blackwell | 新兴 |
| INT4（原始 scale+zero） | ~3.5x | **在推理模型上灾难性崩溃** | 任何 | 不要单独用 |
| **INT4 + BDR**（[[saw-int4]]） | ~3.5x | **GPQA <1 %** | 任何（仅 Triton MHA） | 新 |
| KIVI (2-bit, 混合粒度) | 8x | 较小 (~1-2%) | 任何（需自定义内核） | 研究阶段 |

### RoPE-aware 量化

一个重要的实现细节：RoPE（旋转位置编码）会把能量集中到特定的维度对上，让 Key 在这些通道上跨所有 token 携带显著更大的值。

解决方案：
- 将 Key 分为 RoPE 相关维度和非 RoPE 维度
- RoPE 维度保持较高精度或使用特殊量化
- 非 RoPE 维度可以更激进地量化

### 基于旋转的 KV 缓存量化

K 离群点问题更深层的修复方法是在量化**之前**对 K（可选 V）做**旋转**，把 per-channel 的离群点跨 head 维度摊平，让结果张量对量化均匀友好。乘上一个正交矩阵保持 L2 范数不变，但把能量重新分布；接下来的 per-token scale-and-zero 量化任务就轻松得多。

这与 QuIP/QuIP# 和 QuaRot 用于**权重 + 激活**量化的思想是同一回事（参见 [[quantization#基于旋转的量化（QuIP → QuaRot → SpinQuant → BDR）]] 和 [[rotation-based-quantization]]） —— SAW-INT4 论文把它具体化到了**KV 缓存**在服务约束下：

- **沿头维度的块对角 Hadamard 旋转**，固定块大小（如 16 或 128）→ 内核友好且与分页布局兼容。
- **与 INT4 写入融合**：旋转 + 归一化 + per-token scale/zero + INT4 打包发生在一个 Triton 内核里，所以旋转开销摊薄进 INT4 本来就需要的内存通过。
- **解码时的 Q 修正**：同一旋转在解码内核内对 Q 应用，注意力数学保持不变。

具体效果（Qwen3-4B-Thinking-2507，GPQA）：原始 INT4 把模型崩到 0%；INT4 + BDR 恢复到 65.82%（vs. BF16 的 66.67%）。端到端吞吐与原始 INT4 不可区分。完整的论文精读与内核走读见 [[saw-int4]]。

**注意事项。** 目前仅 MHA（MLA 架构需要不同的形式化）、Triton GQA 解码后端，且仅在单一精度基准上验证。

---

## KV 缓存压缩

### Token 驱逐（Eviction）策略

当 KV 缓存过大时，可以选择性地移除对注意力贡献最小的 token。

#### H2O (Heavy-Hitter Oracle)

H2O 观察到注意力分数高度集中在少数"重击者"（Heavy Hitter）token 上：

```
H2O 的 token 保留策略：

注意力分数分布（典型 pattern）：
Token位置:  [0] [1] [2] [3] [4] [5] [6] ... [95] [96] [97] [98] [99]
注意力分:   0.3 0.1 0.01 0.02 0.01 0.15 0.01 ... 0.01 0.05 0.08 0.12 0.14

H2O 保留三类 token：
┌─────────────────────────────────────────────────────────────────────┐
│ [Initial Tokens]      [Heavy Hitters]           [Recent Window]    │
│ ████                  ████  ████                 ████████████████   │
│ token 0-3            token 5, 98-99             token 85-99        │
│ (注意力 sink)         (累积注意力最高)            (最近上下文)        │
│                                                                     │
│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  被驱逐的 token      │
└─────────────────────────────────────────────────────────────────────┘

保留比例: ~20-40% 的 token
内存节省: 2.5-5x
质量损失: 长文本任务可能有 1-3% 降级
```

#### Scissorhands

与 H2O 类似，但使用不同的"重要性"度量：
- 基于注意力模式的**持久性**（不仅是当前步的注意力高低）
- 如果一个 token 在连续多步都获得低注意力，则被驱逐
- 避免驱逐"暂时被忽视但后续重要"的 token

#### Token 合并 (Token Merging)

不是直接丢弃 token，而是将相似 token 的 KV 向量合并：

```
Token 合并：

原始 KV: [t0] [t1] [t2] [t3] [t4] [t5] [t6] [t7]

相似度检测后合并：
合并后:  [t0] [t1+t2] [t3] [t4+t5+t6] [t7]

优势: 比直接丢弃损失更少信息
劣势: 合并操作有额外计算开销
```

### Attention Sink

StreamingLLM 发现，Transformer 中**第一个 token** 往往获得异常高的注意力分数，即使其内容无关紧要。这个现象称为 **Attention Sink**。

```
Attention Sink 现象：

注意力分数 (典型 pattern)：
         ┃
    0.25 ┃ █
         ┃ █
    0.20 ┃ █
         ┃ █
    0.15 ┃ █                                           ██
         ┃ █                                          ████
    0.10 ┃ █                                        ████████
         ┃ █          █                            ████████████
    0.05 ┃ █ ░ ░ ░ █ █ ░ ░ ░ ░ ░ ░ ░ ░ ░ ░ ░ ██████████████
         ┃ █ ░ ░ ░ █ █ ░ ░ ░ ░ ░ ░ ░ ░ ░ ░ ░ ██████████████
    0.00 ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
         t0 t1 t2 t3 t4 t5 ... ... ... ... ... t95 t96 t97 t98 t99
         ↑                                      └──────────────┘
      attention sink                              近期 tokens

模型将"多余"的注意力分数倾倒到第一个 token 上
即使第一个 token 是 <bos> 或无意义的 padding
```

### StreamingLLM

StreamingLLM 利用 Attention Sink 现象实现了无限长度的流式推理：

```
StreamingLLM 策略：

只保留两部分 KV 缓存：
┌──────────────┬─────────────────────────────────┐
│ Sink tokens  │      Sliding Window              │
│ (前 4 个)    │    (最近 N 个 token)              │
│ ref=0.25     │    ref: 保持上下文连贯            │
└──────────────┴─────────────────────────────────┘

固定内存预算 = sink_size + window_size
例如: 4 + 2044 = 2048 tokens

处理序列 token 100,000 时的缓存状态：
[t0][t1][t2][t3] ... [t97,956][t97,957]...[t99,999]
└──sink tokens──┘     └────── window (2044) ──────┘

优势:
- 固定内存，可处理无限长序列
- 困惑度稳定（不会随序列增长而降级）
- 实现简单

劣势:
- 丢失中间上下文信息
- 不适合需要全文理解的任务（如长文档 QA）
```

### 滑动窗口注意力 (Sliding Window Attention)

Mistral 在模型架构中直接使用滑动窗口注意力：

```
滑动窗口注意力 (window_size = 4096):

标准注意力（full attention）:
每个 token 关注所有之前的 token
Token 100: 关注 token 0 ~ 99
Token 10000: 关注 token 0 ~ 9999  ← KV 缓存很大！

滑动窗口注意力:
每个 token 只关注最近 W 个 token
Token 100: 关注 token 0 ~ 99 (不足W个，全部关注)
Token 10000: 关注 token 5904 ~ 9999  ← KV 缓存固定为 W！

KV 缓存大小：固定为 window_size（不随序列长度增长）

实际效果:
- Mistral-7B (window=4096): 质量接近全注意力
- 原因: 多层叠加后，信息可以通过层间传递覆盖更长范围
  - 例如 32 层 × 4096 窗口 ≈ 理论上可覆盖 131,072 token 的信息
```

### 驱逐策略对比

```
不同驱逐/压缩策略对比：

策略              内存节省   质量损失   适用场景
────────────────────────────────────────────────
Full KV cache     0%         无        短序列
Sliding Window    固定上限   中等      流式推理, 模型内置
StreamingLLM      固定上限   中等      无限流式对话
H2O               2-5x      小        通用长序列
Scissorhands      2-5x      小        通用长序列
Token Merging     2-3x      极小      需要保留信息的场景
KVTC              10-20x    小        极端压缩场景
```

---

## 前缀缓存与共享

### 自动前缀缓存

前缀缓存在多个请求间共享相同前缀的 KV 缓存，避免重复计算。详见 [[paged-attention#前缀缓存 (Prefix Caching)]]。

### RadixAttention (SGLang)

[[sglang|SGLang]] 的 RadixAttention 使用基数树（Radix Tree）实现 token 级别的精确前缀匹配：

```
RadixAttention 基数树示例：

多轮对话场景：
User: "What is Python?" → Assistant: "Python is..."
User: "What are its features?" → Assistant: "Key features..."

                    [root]
                      │
              [System prompt KV]
                   /        \
        [User: "What is     [User: "What are
         Python?"]           its features?"]
            │                      │
        [Asst: "Python        [Asst: "Key
         is..."]               features..."]

新请求: User: "What is Java?"
  → 匹配 [System prompt KV]（命中！跳过 prefill）
  → 从 "What is Java?" 开始计算

收益: 系统提示词 + 对话历史的 prefill 完全跳过
```

### 跨请求 KV 共享 (LMCache)

LMCache 将 KV 缓存管理从推理引擎中解耦，支持跨引擎、跨节点的 KV 缓存共享：

```
LMCache 架构：

┌──────────┐   ┌──────────┐   ┌──────────┐
│ vLLM 实例1│   │ vLLM 实例2│   │ SGLang   │
└─────┬────┘   └─────┬────┘   └─────┬────┘
      │              │              │
      └──────────┬───┴──────────────┘
                 │
         ┌───────┴────────┐
         │   LMCache      │
         │  Connector     │
         └───────┬────────┘
                 │
    ┌────────┬───┴───┬──────────┐
    │GPU Cache│CPU DRAM│  SSD    │  Redis/S3
    └────────┴───────┴──────────┘

特点：
- 多级缓存层次 (GPU → CPU → SSD → 远程)
- 跨引擎实例共享 KV
- 吞吐量提升可达 15x (高前缀重用率场景)
```

---

## 分布式 KV 缓存

### 数据并行注意力中的 KV 缓存

在 DP Attention（DeepSeek-V3 使用的方案）中，不同 DP rank 持有不同请求的 KV 缓存：

```
DP Attention 中的 KV 缓存分区：

4-way DP Attention:
┌────────────────────────────────────────────────────────┐
│ GPU 0: KV cache for requests {r0, r4, r8, ...}        │
│ GPU 1: KV cache for requests {r1, r5, r9, ...}        │
│ GPU 2: KV cache for requests {r2, r6, r10, ...}       │
│ GPU 3: KV cache for requests {r3, r7, r11, ...}       │
└────────────────────────────────────────────────────────┘

注意力计算时:
- 每个 GPU 只计算本地请求的注意力 (无通信)
- FFN 层使用 EP (Expert Parallelism) 需要 All-to-All 通信
- KV 缓存不需要跨 GPU 同步 (各自独立)

优势: KV 缓存可扩展性极好
      每个 GPU 只存储 1/DP_size 的请求 KV
```

### 预填充-解码分离中的 KV 缓存传输

在 [[prefill-decode-disaggregation|预填充-解码分离]] 架构中，预填充节点生成的 KV 缓存需要传输到解码节点：

```
KV 缓存传输流程：

Prefill Node                      Decode Node
┌─────────────┐                  ┌─────────────┐
│ 1. 执行 Prefill │                │              │
│ 2. 生成 KV cache │──── 传输 ──→│ 3. 接收 KV   │
│ 3. 释放本地 KV  │              │ 4. 继续解码   │
└─────────────┘                  └─────────────┘

传输量 = KV_cache_size(prompt_length)
例如 LLaMA-70B, prompt=4K tokens, FP8:
  = 2 × 80 × 8 × 128 × 4096 × 1 = 640 MB

传输优化：
├─ RDMA/InfiniBand: 高带宽低延迟
├─ Mooncake Transfer Engine: 专为 KV 传输优化
├─ KV 压缩传输: KVTC 等压缩算法减少传输量
└─ 流水线传输: 边计算边传输 (layer-by-layer)
```

### CacheBlend 与 CacheGen

**CacheGen**：将 KV 缓存编码压缩后存储和传输
- 使用学习到的编码器压缩 KV 缓存
- 压缩比 3-5x
- 适合跨网络传输场景

**CacheBlend**：
- 混合本地计算和远程 KV 缓存
- 对远程获取的 KV 缓存进行局部修正
- 平衡传输延迟和计算开销

### Mooncake

[[prefill-decode-disaggregation#Mooncake|Mooncake]]（FAST 2025 最佳论文）将 KV 缓存视为系统核心，池化集群中所有 CPU/DRAM/SSD 资源：

```
Mooncake KV Cache Pool：

┌─────────────────────────────────────────────────┐
│                 KV Cache Pool                    │
│                                                  │
│  Node 0          Node 1          Node 2         │
│  ┌──────┐       ┌──────┐       ┌──────┐       │
│  │GPU HBM│       │GPU HBM│       │GPU HBM│       │
│  │ (热)  │       │ (热)  │       │ (热)  │       │
│  ├──────┤       ├──────┤       ├──────┤       │
│  │CPU RAM│       │CPU RAM│       │CPU RAM│       │
│  │ (温)  │       │ (温)  │       │ (温)  │       │
│  ├──────┤       ├──────┤       ├──────┤       │
│  │ SSD   │       │ SSD   │       │ SSD   │       │
│  │ (冷)  │       │ (冷)  │       │ (冷)  │       │
│  └──────┘       └──────┘       └──────┘       │
│                                                  │
│  统一管理，按热度自动分层迁移                     │
│  容量提升 59-498% (相比仅用 GPU HBM)             │
└─────────────────────────────────────────────────┘
```

---

## 代码示例

### KV 缓存大小计算器

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    name: str
    num_layers: int
    num_q_heads: int
    num_kv_heads: int      # GQA: < num_q_heads; MQA: = 1
    head_dim: int
    mla_latent_dim: Optional[int] = None  # MLA 压缩维度

# 常见模型配置
MODELS = {
    "llama-3.1-8b": ModelConfig("LLaMA-3.1-8B", 32, 32, 8, 128),
    "llama-3.1-70b": ModelConfig("LLaMA-3.1-70B", 80, 64, 8, 128),
    "llama-3.1-405b": ModelConfig("LLaMA-3.1-405B", 126, 128, 8, 128),
    "mistral-7b": ModelConfig("Mistral-7B", 32, 32, 8, 128),
    "deepseek-v3": ModelConfig("DeepSeek-V3", 61, 128, 128, 128, mla_latent_dim=512),
    "qwen-2.5-72b": ModelConfig("Qwen-2.5-72B", 80, 64, 8, 128),
}

def kv_cache_size(
    model: ModelConfig,
    seq_len: int,
    batch_size: int = 1,
    dtype_bytes: float = 2.0,  # FP16=2, FP8=1, INT4=0.5
) -> dict:
    """计算 KV 缓存大小"""
    
    if model.mla_latent_dim:
        # MLA: 只存储压缩后的潜在向量
        # 需要额外存储 RoPE 相关维度 (通常 64-128 维)
        rope_dim = 64  # 近似值
        size_bytes = (
            model.num_layers * (model.mla_latent_dim + rope_dim) 
            * seq_len * batch_size * dtype_bytes
        )
    else:
        # MHA/GQA/MQA
        size_bytes = (
            2 * model.num_layers * model.num_kv_heads * model.head_dim
            * seq_len * batch_size * dtype_bytes
        )
    
    size_gb = size_bytes / (1024 ** 3)
    
    # MHA 等效大小（用于比较）
    mha_equiv_bytes = (
        2 * model.num_layers * model.num_q_heads * model.head_dim
        * seq_len * batch_size * dtype_bytes
    )
    mha_equiv_gb = mha_equiv_bytes / (1024 ** 3)
    
    return {
        "model": model.name,
        "size_gb": size_gb,
        "mha_equiv_gb": mha_equiv_gb,
        "compression_vs_mha": mha_equiv_gb / size_gb if size_gb > 0 else float('inf'),
        "per_token_bytes": size_bytes / seq_len / batch_size,
    }

# 打印各模型的 KV 缓存大小
print(f"{'模型':<20} {'seq_len':>8} {'bs':>4} {'KV缓存':>10} {'MHA等效':>10} {'压缩比':>8}")
print("─" * 70)

for name, model in MODELS.items():
    for seq_len in [4096, 32768, 131072]:
        result = kv_cache_size(model, seq_len, batch_size=1)
        print(f"{result['model']:<20} {seq_len:>8} {1:>4} "
              f"{result['size_gb']:>9.2f}G {result['mha_equiv_gb']:>9.2f}G "
              f"{result['compression_vs_mha']:>7.1f}x")
    print()
```

### FP8 KV 缓存配置

```python
# ========== vLLM FP8 KV 缓存配置 ==========

from vllm import LLM, SamplingParams

# 方式 1：直接指定 KV 缓存类型
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    kv_cache_dtype="fp8",            # KV 缓存使用 FP8
    tensor_parallel_size=4,
    gpu_memory_utilization=0.92,     # FP8 允许更高利用率
)

# 方式 2：配合权重量化
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    quantization="fp8",              # 权重 FP8
    kv_cache_dtype="fp8",            # KV 缓存也 FP8
    tensor_parallel_size=2,          # FP8 减少内存，可用更少卡
)

# 方式 3：使用量化配置文件
# 创建 quantization_config.json:
# {
#     "kv_cache": {
#         "dtype": "fp8_e4m3",
#         "static": false,           # 动态量化（per-token scale）
#         "scaling_factor": null      # 自动计算
#     }
# }

# ========== 内存节省对比 ==========
# LLaMA-70B, seq=4K, batch=16:
# FP16 KV: 20.0 GB
# FP8 KV:  10.0 GB  ← 节省 10GB → 可以增加 ~80% 的并发请求
```

---

## 优化选择指南

根据不同场景选择合适的 KV 缓存优化策略：

```
KV 缓存优化决策树：

你的主要瓶颈是什么？
│
├─ 内存不足（无法容纳目标 batch_size）
│   │
│   ├─ 使用 GQA/MQA 模型了吗？
│   │   ├─ 否 → 切换到 GQA 模型（如 LLaMA-3）
│   │   └─ 是 → 继续下一步
│   │
│   ├─ 启用了 FP8 KV 缓存吗？
│   │   ├─ 否 → 启用 FP8（最简单的 2x 优化）
│   │   └─ 是 → 继续下一步
│   │
│   ├─ 使用了 PagedAttention 吗？
│   │   ├─ 否 → 使用 vLLM/SGLang（自动启用）
│   │   └─ 是 → 继续下一步
│   │
│   ├─ 序列很长 (>32K)？
│   │   ├─ 是 → 考虑：
│   │   │   ├─ 滑动窗口模型 (Mistral)
│   │   │   ├─ KV 驱逐 (H2O)
│   │   │   └─ KV 卸载到 CPU (LMCache)
│   │   └─ 否 → 考虑 INT4 KV 量化（旋转 + INT4 见 [[saw-int4]] / BDR，或 KIVI）
│   │
│   └─ 以上都用了？→ 加 GPU 或用模型并行
│
├─ TTFT 过高（首 token 延迟大）
│   │
│   ├─ 有重复前缀吗？
│   │   ├─ 是 → 启用前缀缓存
│   │   │   ├─ vLLM: 默认开启
│   │   │   └─ SGLang: RadixAttention (更快)
│   │   └─ 否 → 考虑预填充-解码分离
│   │
│   └─ 使用分离架构 + KV 缓存传输优化
│
└─ 吞吐量不足
    │
    ├─ 启用连续批处理 ✓
    ├─ 增大 batch_size（需要更多 KV 内存 → 回到上面）
    └─ 考虑投机解码提高生成速度
```

### 各优化技术的兼容性

| 优化技术 | + PagedAttn | + FP8 KV | + 前缀缓存 | + GQA |
|---------|-------------|----------|-----------|-------|
| PagedAttention | - | 兼容 | 互补 | 兼容 |
| FP8 KV cache | 兼容 | - | 兼容 | 兼容 |
| 前缀缓存 | 互补 | 兼容 | - | 兼容 |
| GQA 架构 | 兼容 | 兼容 | 兼容 | - |
| Token 驱逐 | 需适配 | 可组合 | 冲突风险 | 兼容 |
| 滑动窗口 | 兼容 | 兼容 | 部分兼容 | 兼容 |

大多数优化是正交的，可以组合使用。典型的生产配置：
- **GQA 模型 + PagedAttention + FP8 KV + 前缀缓存** = 最佳通用配置

---

## 参考文献

1. **Kwon et al.** "Efficient Memory Management for Large Language Model Serving with PagedAttention" — SOSP 2023. [论文](https://arxiv.org/abs/2309.06180)

2. **Shazeer, N.** "Fast Transformer Decoding: One Write-Head is All You Need" — 2019. [论文](https://arxiv.org/abs/1911.02150)
   - Multi-Query Attention (MQA) 原始论文

3. **Ainslie et al.** "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" — EMNLP 2023. [论文](https://arxiv.org/abs/2305.13245)
   - Grouped-Query Attention (GQA)

4. **DeepSeek-AI.** "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model" — 2024. [论文](https://arxiv.org/abs/2405.04434)
   - Multi-head Latent Attention (MLA)

5. **Liu et al.** "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache" — 2024. [论文](https://arxiv.org/abs/2402.02750)
12. **Jia et al.** "SAW-INT4: System-Aware 4-Bit KV-Cache Quantization for Real-World LLM Serving" — 2026. [论文](https://arxiv.org/abs/2604.19157) —— 块对角 Hadamard 旋转让原始 INT4 KV 在推理模型上变得可用。
13. **Ashkboos et al.** "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs" — NeurIPS 2024. [论文](https://arxiv.org/abs/2404.00456) —— 用于权重 + 激活的完整 Hadamard 旋转；BDR 的 KV 变体的源流。

6. **Zhang et al.** "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models" — NeurIPS 2023. [论文](https://arxiv.org/abs/2306.14048)

7. **Xiao et al.** "Efficient Streaming Language Models with Attention Sinks" — ICLR 2024. [论文](https://arxiv.org/abs/2309.17453)
   - StreamingLLM

8. **Zheng et al.** "SGLang: Efficient Execution of Structured Language Model Programs" — 2024.
   - RadixAttention

9. **Qin et al.** "Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving" — FAST 2025 Best Paper.

10. **Panwar et al.** "vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention" — ASPLOS 2025. [论文](https://arxiv.org/abs/2405.04437)

---

## 相关页面

- [[paged-attention]] — 基于块的内存管理深度解析
- [[vllm]] — 前缀缓存与 FP8 KV 的实现
- [[sglang]] — RadixAttention 前缀缓存
- [[quantization]] — 更广泛的量化技术（含权重/激活旋转）
- [[saw-int4]] — 块对角 Hadamard 旋转 + INT4 KV（论文精读）
- [[rotation-based-quantization]] — QuIP / QuaRot / SpinQuant / BDR 家族脉络
- [[continuous-batching]] — 调度与 KV 缓存管理的交互
- [[prefill-decode-disaggregation]] — KV 缓存传输挑战
- [[long-context-serving]] — 长上下文场景的 KV 缓存挑战
- [[multi-turn-optimization]] — 跨轮 KV 缓存复用
