---
title: "LLM 推理量化"
category: llm-inference
tags: [量化, gptq, awq, fp8, int4, 权重量化, kv缓存量化]
created: 2026-04-13
updated: 2026-04-14
status: mature
---

# LLM 推理量化

## 概述

大语言模型的参数通常以 FP16 或 BF16（16 位浮点数）存储，单个参数占 2 字节。一个 70B 参数的模型仅权重就需要约 140GB 显存，超过了单卡（如 A100 80GB）的容量。

**量化（Quantization）** 将模型的数值精度从高位（FP16/BF16）降低到低位（INT8、INT4、FP8 等），从而：

1. **减少内存占用**：INT4 量化可将模型大小压缩约 4 倍（70B 模型从 140GB 降至 ~35GB）
2. **提高推理吞吐**：更少的数据从显存加载 → 内存带宽受限的 decode 阶段显著加速
3. **降低硬件需求**：原本需要多卡的模型可以在更少的 GPU 上运行

量化是 **在更少 GPU 上运行更大模型** 的最有效单一优化手段。

```
模型精度与显存需求对比 (Llama-3.3-70B):

  FP16:   ████████████████████████████████  140 GB  (2x H100)
  FP8:    ████████████████                   70 GB  (1x H100)
  INT4:   ████████                           35 GB  (1x A100-40G)
  NVFP4:  ████████                           35 GB  (1x B200)
```

量化的核心挑战在于：如何在降低精度的同时 **最小化对模型输出质量的影响**。不同方法在精度损失、量化速度、推理加速之间做出不同权衡。


## 量化基础

### 数值表示基础

```
FP16 (半精度浮点): 1 位符号 + 5 位指数 + 10 位尾数
  范围: -65504 ~ 65504
  精度: ~3.3 位十进制数

BF16 (Brain Float): 1 位符号 + 8 位指数 + 7 位尾数
  范围: 同 FP32 (~3.4e38)
  精度: ~2.4 位十进制数

FP8 E4M3: 1 位符号 + 4 位指数 + 3 位尾数
  范围: -448 ~ 448
  精度: ~1.4 位十进制数

FP8 E5M2: 1 位符号 + 5 位指数 + 2 位尾数
  范围: -57344 ~ 57344
  精度: ~0.9 位十进制数

INT8: 8 位整数
  范围: -128 ~ 127 (有符号)
  256 个离散值

INT4: 4 位整数
  范围: -8 ~ 7 (有符号)
  仅 16 个离散值
```

### 对称量化 vs 非对称量化

**对称量化**：零点固定为 0，使用统一缩放因子。

```
量化:    q = round(x / scale)
反量化:  x' = q * scale

其中 scale = max(|x|) / (2^(b-1) - 1)

示例 (FP16 → INT8, 对称):
  原始值: [-1.2, 0.5, 3.7, -0.1, 2.8]
  max(|x|) = 3.7
  scale = 3.7 / 127 ≈ 0.02913
  量化值: [-41, 17, 127, -3, 96]
  反量化: [-1.194, 0.495, 3.700, -0.087, 2.796]
  误差:   [0.006, 0.005, 0.000, 0.013, 0.004]
```

**非对称量化**：引入零点偏移，更好地利用量化范围。

```
量化:    q = round(x / scale) + zero_point
反量化:  x' = (q - zero_point) * scale

其中 scale = (max(x) - min(x)) / (2^b - 1)
     zero_point = round(-min(x) / scale)

适用场景: 数据分布不以零为中心（如 ReLU 后的激活值）
```

### 量化粒度

```
┌─────────────────────────────────────────────────────────────┐
│                     量化粒度对比                              │
├─────────────┬───────────────────────────────────────────────┤
│             │  权重矩阵 W (shape: out_dim x in_dim)         │
│             │  ┌──────────────────────────────────┐         │
│ Per-Tensor  │  │ 整个矩阵共享一个 scale            │         │
│ (逐张量)    │  │ scale_tensor = max(|W|) / 127     │         │
│             │  │ 精度最差, 速度最快                  │         │
│             │  └──────────────────────────────────┘         │
│             │                                               │
│             │  ┌──────────────────────────────────┐         │
│ Per-Channel │  │ 每行（输出通道）一个 scale          │         │
│ (逐通道)    │  │ scale[i] = max(|W[i,:]|) / 127   │         │
│             │  │ 精度较好, 最常用                    │         │
│             │  └──────────────────────────────────┘         │
│             │                                               │
│             │  ┌──────────────────────────────────┐         │
│ Per-Group   │  │ 每 G 个元素一个 scale (G=128常用)  │         │
│ (逐组)      │  │ 精度最好, 开销稍大                  │         │
│             │  │ GPTQ/AWQ 默认使用 group_size=128   │         │
│             │  └──────────────────────────────────┘         │
└─────────────┴───────────────────────────────────────────────┘
```

per-group 量化在精度和效率之间取得了最佳平衡。典型设置是 group_size=128，即每 128 个权重共享一组量化参数（scale 和 zero_point）。

### 量化误差分析

量化引入的误差可以从两个角度分析：

**权重层面**：
```
MSE = E[(W - W_q)^2]      -- 权重重建误差
```

**输出层面**（更重要）：
```
||WX - W_q X||^2           -- 输出重建误差（取决于激活值 X）
```

关键观察：**不是所有权重同等重要**。如果某些权重总是被大的激活值乘，量化这些权重的误差会被放大。这就是 AWQ 等方法的出发点。


## 权重量化方法

### GPTQ (ICLR 2023)

GPTQ 基于 **最优脑量化（Optimal Brain Quantization, OBQ）** 算法，是第一个能在几小时内将 175B 模型量化到 3-4 位的方法。

**核心思想**：逐列量化权重矩阵，每量化一列，用 Hessian 信息补偿未量化列的权重，最小化整体输出误差。

```
算法流程:
  输入: 权重矩阵 W, 校准数据集 → Hessian H = 2X^T X
  
  for col in range(columns):
      1. 量化当前列: w_q = quantize(W[:, col])
      2. 计算量化误差: delta = W[:, col] - w_q
      3. 补偿后续未量化列:
         W[:, col+1:] -= delta * H[col, col+1:] / H[col, col]
      
  关键: 补偿步骤利用 Hessian 的逆来最小化输出误差
```

**特点**：
- 一次性量化（post-training），不需要重新训练
- 需要少量校准数据（128-256 条样本）
- 175B 模型约 4 GPU 小时
- 3-4 bit 量化，perplexity 增加 < 0.5
- 支持 group quantization（group_size=128）

**局限**：
- 量化精度依赖校准数据的代表性
- 对异常值敏感
- 串行逐列处理较慢

论文：[arXiv:2210.17323](https://arxiv.org/abs/2210.17323)

### AWQ (MLSys 2024 最佳论文)

AWQ（Activation-Aware Weight Quantization）的核心观察：

> 不到 1% 的"显著权重"（salient weights）对模型输出有决定性影响，而这些权重可以通过观察 **激活值大小** 来识别。

```
关键发现:

权重重要性 ∝ 激活值大小

  权重 w1 = 0.3, 对应激活 a1 = 100 → 输出贡献 = 30    ← 重要!
  权重 w2 = 0.5, 对应激活 a2 = 0.1 → 输出贡献 = 0.05  ← 不重要

因此: 保护那些对应大激活值的权重通道
```

**方法**：

1. 用校准数据计算每个权重通道对应的平均激活值大小
2. 对显著通道应用 **逐通道缩放**（per-channel scaling），放大这些权重再量化
3. 反量化时相应缩小，在推理时无额外开销

```
原始:        W_q = quantize(W)           → 显著权重被严重量化
AWQ:         W_q = quantize(W * s) / s   → 显著权重缩放后更好保留

其中 s (per-channel scale) 基于激活值大小确定
最优 s*: 通过网格搜索在 [0, 1] 范围内找到最小化量化误差的缩放因子
```

**性能**（配合 Marlin 内核）：

| 方法 | 吞吐 (tok/s) | 相对加速 |
|------|-------------|----------|
| FP16 基线 | 68 | 1.0x |
| GPTQ-4bit + Marlin | 179 | 2.6x |
| AWQ-4bit + Marlin | 741 | 10.9x |

AWQ + Marlin 的巨大加速来自 Marlin 内核针对 AWQ 量化格式的极致优化。

论文：[arXiv:2306.00978](https://arxiv.org/abs/2306.00978)

### SqueezeLLM (ICML 2024)

**核心思想**：将权重分解为"正常值"和"异常值"两部分。

```
W = W_normal + W_sparse

W_normal: 非均匀量化（使用 K-means 找最优量化级别）
W_sparse: 稀疏矩阵, 保存异常值（全精度）
```

- **非均匀量化**：不使用均匀间隔的量化级别，而是通过 K-means 聚类找到最优的量化码本
- **稠密-稀疏分解**：异常值用稀疏矩阵存储（全精度），避免它们破坏正常值的量化范围
- 在 sub-3-bit（如 2-bit）量化时表现突出
- 适合极端压缩场景

论文：[arXiv:2306.07629](https://arxiv.org/abs/2306.07629)

### QuIP / QuIP# 

**核心思想**：通过随机正交变换使权重"不相干"，消除异常值。

```
原始权重:   有异常值, 分布不均 → 量化困难
变换后:     W' = U^T W V  (U, V 为随机正交矩阵)
            → 权重变得更均匀 → 量化更容易
推理时:     通过修改层的输入输出来补偿变换
```

- QuIP# 使用 lattice codebook 进一步提升编码效率
- 在 2-bit 量化时质量显著优于 GPTQ/AWQ
- 实际推理速度受限于解码开销

### 方法对比

| 方法 | 位宽 | 校准数据 | 量化时间 (70B) | Perplexity 增加 | 推理速度 | 特点 |
|------|------|----------|---------------|-----------------|----------|------|
| GPTQ | 3-4 bit | 需要 | ~4h | <0.5 | 快（Marlin） | 最成熟 |
| AWQ | 4 bit | 需要 | ~1h | <0.3 | 最快（Marlin） | 最佳平衡 |
| SqueezeLLM | 2-4 bit | 需要 | ~8h | <1.0 (2-bit) | 中等 | 极端压缩 |
| QuIP# | 2-4 bit | 需要 | ~6h | <0.5 (2-bit) | 较慢 | 理论最优 |
| Round-to-nearest | 任意 | 不需要 | 即时 | 较大 | 快 | 基线 |


## FP8 量化

### E4M3 vs E5M2

FP8 有两种标准格式：

```
FP8 E4M3:                        FP8 E5M2:
┌───┬────────┬─────────┐          ┌───┬──────────┬───────┐
│ S │  Exp   │ Mantissa │          │ S │   Exp    │ Mant  │
│ 1 │  4 bit │  3 bit   │          │ 1 │  5 bit   │ 2 bit │
└───┴────────┴─────────┘          └───┴──────────┴───────┘

  范围: ±448                        范围: ±57344
  精度: 较高 (8 级尾数)              精度: 较低 (4 级尾数)
  
  用途: 前向推理 (权重+激活)          用途: 梯度（训练中）
```

**推理通常使用 E4M3**：精度更高，动态范围对推理够用。

### 硬件支持

| GPU | FP8 支持 | 性能 |
|-----|---------|------|
| A100 (Ampere) | 不支持 | N/A |
| H100 (Hopper) | 原生 | ~2x vs FP16 |
| H200 (Hopper) | 原生 | ~2x vs FP16 |
| B100/B200 (Blackwell) | 原生 | ~2x vs FP16 |

### 动态 vs 静态缩放

FP8 量化需要将 FP16/BF16 值映射到 FP8 范围，需要缩放因子：

**静态缩放**：
```
在校准阶段确定固定的 scale
优点: 推理时无额外计算
缺点: 如果实际值超出校准范围, 会截断 (clipping)
```

**动态缩放**：
```
每次推理时根据实际张量值计算 scale
优点: 始终最优利用 FP8 范围
缺点: 需要额外的 reduction 操作计算 max
实际开销: 很小 (~1-2%)，通常值得
```

**per-tensor vs per-channel vs block-wise scaling**：

```
Per-tensor:   整个张量一个 scale → 最快，精度最低
Per-channel:  每个输出通道一个 scale → 平衡
Block-wise:   每个小块一个 scale → 最精确，用于 DeepSeek V3
```

### FP8 vs INT8：何时选哪个？

| 维度 | FP8 E4M3 | INT8 |
|------|----------|------|
| 动态范围 | 大（±448） | 小（-128~127） |
| 硬件支持 | Hopper+ | Ampere+ |
| 量化难度 | 低（范围大，更 robust） | 中（需要仔细校准） |
| 推理速度 | 快（Tensor Core 原生） | 快（Tensor Core 原生） |
| 精度保持 | 优秀 | 良好（需要 SmoothQuant 等技巧） |
| 推荐 | Hopper+ 的首选 | Ampere 上的选择 |


## KV 缓存量化

KV 缓存是推理内存的另一大消耗。对于长序列和大 batch，KV 缓存可能占总显存的 30-50%。

### KV 缓存显存计算

```
KV 缓存大小 = 2 × num_layers × num_kv_heads × head_dim × seq_len × batch_size × bytes_per_element

示例 (Llama-3.3-70B, FP16):
  = 2 × 80 × 8 × 128 × 4096 × 1 × 2 bytes
  = ~13.4 GB (单请求!)
  
  batch_size=16: ~214 GB → 远超模型权重
```

### FP8 KV 缓存

```
KV 缓存从 FP16 → FP8:
  内存减半: 13.4 GB → 6.7 GB
  精度影响: 极小 (perplexity 增加 < 0.1)
  
vLLM 配置:
  --kv-cache-dtype fp8
```

- 几乎无质量损失
- 内存减半 → 可服务更多并发请求或更长序列
- Hopper+ 上推荐默认开启

### INT4 KV 缓存

```
KV 缓存从 FP16 → INT4 (per-group quantization):
  内存 1/4: 13.4 GB → ~3.4 GB (含 scale/zero_point 开销实际约 4 GB)
  精度影响: 较小但可测量 (perplexity 增加 ~0.3-0.5)
  
需要 per-group scaling (group_size=32 或 64) 来维持质量
```

- 内存节省更激进
- 需要仔细选择 group_size
- 对长上下文场景特别有价值（128K+ token 的 KV 缓存巨大）

### KV 缓存量化的质量影响

| 方法 | 内存节省 | PPL 增加 (Llama-70B, Wiki) | 推荐场景 |
|------|----------|---------------------------|----------|
| FP16 (基线) | 0% | 0 | 质量第一 |
| FP8 | 50% | <0.1 | 默认推荐 |
| INT8 | 50% | <0.1 | Ampere 上 |
| INT4 (g=64) | ~70% | 0.3-0.5 | 显存极紧张 |
| INT4 (g=32) | ~65% | 0.2-0.3 | 质量敏感 |


## 激活量化

### 为什么激活量化更难？

权重是静态的（量化一次即可），但激活值在每次推理时动态变化，且存在显著的异常值问题：

```
典型激活值分布:
                  ▌
                  ▌
            ▌▌▌▌▌▌▌▌▌▌▌▌▌
     ▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌
  ───────────────────────────────────────  ●  ←  异常值!
  -2        -1         0         1    2      100
  
  这些异常值（outliers）虽然稀少，但幅度是正常值的 50-100 倍
  如果用统一的 scale 量化，正常值的精度会被极度压缩
```

### SmoothQuant (ICML 2023)

**核心思想**：将量化困难从激活值"迁移"到权重。

```
观察:
  激活 X: 有异常值 → 量化困难
  权重 W: 分布均匀 → 量化容易
  
SmoothQuant: 用 per-channel 缩放把难度从 X 迁移到 W

  Y = X * W
    = (X * diag(s)^{-1}) * (diag(s) * W)
    = X_smooth * W_smooth

  s = max(|X_j|)^α / max(|W_j|)^(1-α)
  其中 α ∈ [0, 1] 控制迁移程度 (通常 α=0.5)

效果:
  X_smooth: 异常值被 s^{-1} 缩小 → 量化更容易
  W_smooth: 吸收了部分异常值 → 仍然可以量化
```

- 实现 **W8A8**（权重 INT8 + 激活 INT8）量化
- 无需训练，只需校准数据
- 在 175B 模型上精度保持与 FP16 接近
- INT8 矩阵乘法在 GPU 上有硬件加速

论文：[arXiv:2211.10438](https://arxiv.org/abs/2211.10438)

### W8A8 vs W4A16 范式

```
W8A8 (SmoothQuant):
  权重: INT8, 激活: INT8
  计算: INT8 矩阵乘法 (Tensor Core 加速)
  优点: 计算和内存都加速
  缺点: 激活量化引入额外误差
  适用: 大 batch, 计算密集场景

W4A16 (GPTQ/AWQ):
  权重: INT4, 激活: FP16
  计算: 权重反量化 → FP16 矩阵乘法 (或专用内核)
  优点: 无激活量化误差
  缺点: 仅内存加速, 计算速度不提升 (除非用 Marlin 等专用内核)
  适用: 小 batch, 内存带宽受限场景
```


## 代码示例

### 使用 AutoAWQ 量化模型

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# 加载 FP16 模型
model_path = "meta-llama/Llama-3.1-8B-Instruct"
quant_path = "llama-3.1-8b-instruct-awq-4bit"

model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 量化配置
quant_config = {
    "zero_point": True,       # 非对称量化
    "q_group_size": 128,      # 每 128 个权重一组
    "w_bit": 4,               # 4-bit 量化
    "version": "GEMM",        # 内核版本
}

# 执行量化 (需要校准数据, 默认使用 C4 数据集的子集)
model.quantize(tokenizer, quant_config=quant_config)

# 保存量化后的模型
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
print(f"量化完成! 模型保存至 {quant_path}")
```

### 在 vLLM 中加载量化模型

```python
from vllm import LLM, SamplingParams

# ---- GPTQ 模型 ----
llm_gptq = LLM(
    model="TheBloke/Llama-3-70B-Instruct-GPTQ",
    quantization="gptq",
    tensor_parallel_size=2,
    # Marlin 内核自动启用 (如果模型兼容)
    # 手动指定: --quantization gptq_marlin
)

# ---- AWQ 模型 ----
llm_awq = LLM(
    model="casperhansen/llama-3-70b-instruct-awq",
    quantization="awq",
    tensor_parallel_size=2,
)

# ---- FP8 在线量化 (无需预量化模型, Hopper+) ----
llm_fp8 = LLM(
    model="meta-llama/Llama-3.3-70B-Instruct",
    quantization="fp8",
    tensor_parallel_size=4,
    kv_cache_dtype="fp8",  # KV 缓存也用 FP8
)

# ---- FP8 预量化模型 ----
llm_fp8_pre = LLM(
    model="neuralmagic/Llama-3.3-70B-Instruct-FP8",
    tensor_parallel_size=4,
    kv_cache_dtype="fp8",
)

# 使用方式完全相同
params = SamplingParams(temperature=0.7, max_tokens=512)
outputs = llm_fp8.generate(["Explain quantum computing"], params)
```

### 质量评估：Perplexity 对比

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def evaluate_perplexity(model, tokenizer, dataset_name="wikitext",
                        dataset_config="wikitext-2-raw-v1", max_length=2048):
    """计算模型在给定数据集上的 perplexity"""
    dataset = load_dataset(dataset_name, dataset_config, split="test")
    
    # 拼接所有文本
    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(model.device)
    
    nlls = []
    for i in range(0, input_ids.size(1) - 1, max_length):
        begin = max(i + max_length - input_ids.size(1), 0)
        end = min(i + max_length, input_ids.size(1))
        target_len = end - (i if i > 0 else 0)
        
        input_chunk = input_ids[:, begin:end]
        with torch.no_grad():
            outputs = model(input_chunk)
            # 计算 NLL
            shift_logits = outputs.logits[:, -(target_len):, :]
            shift_labels = input_ids[:, (end - target_len):end]
            loss = torch.nn.functional.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                reduction="none"
            )
            nlls.append(loss.sum())
    
    ppl = torch.exp(torch.stack(nlls).sum() / input_ids.size(1))
    return ppl.item()

# 使用示例
# model_fp16: perplexity ≈ 5.68
# model_awq4: perplexity ≈ 5.82  (+0.14)
# model_gptq4: perplexity ≈ 5.85 (+0.17)
# model_fp8: perplexity ≈ 5.70   (+0.02)
```


## 量化选择指南

```
                          量化方法选择决策树
                          
                          你的 GPU 是什么?
                              │
                ┌─────────────┼─────────────┐
                ▼             ▼             ▼
           Blackwell      Hopper         Ampere/更早
           (B100/B200)   (H100/H200)    (A100/A10G)
                │             │             │
                ▼             ▼             ▼
           NVFP4 首选      FP8 首选      INT4/INT8
           (原生 4-bit)   (原生 8-bit)   (软件量化)
                │             │             │
                │             │             ├─→ 模型小? → INT8 (SmoothQuant)
                │             │             └─→ 模型大? → INT4 (AWQ > GPTQ)
                │             │
                │             ├─→ 质量第一? → FP8 (近乎无损)
                │             └─→ 显存紧张? → FP8 权重 + FP8 KV cache
                │
                └─→ 质量第一? → NVFP4 (硬件优化)
                    显存紧张? → NVFP4 + INT4 KV cache
                    
CPU 推理 (llama.cpp):
  └─→ GGUF Q4_K_M (4-bit, ~92% 质量) 或 Q5_K_M (5-bit, ~95% 质量)
```

**快速推荐总结**：

| 场景 | 推荐方法 | 理由 |
|------|----------|------|
| H100/H200，不想损失质量 | FP8 (自动量化) | 近乎无损，2x 加速 |
| A100，需要跑 70B | AWQ-4bit + Marlin | 最佳速度/质量平衡 |
| A100，质量重要 | INT8 SmoothQuant | 精度损失极小 |
| B200，最大吞吐 | NVFP4 | 硬件原生支持 |
| 极端压缩（2-bit） | SqueezeLLM / QuIP# | 牺牲质量换极致压缩 |
| CPU/边缘设备 | GGUF Q4_K_M | llama.cpp 生态 |
| KV 缓存太大 | FP8 KV cache | 内存减半，质量几乎不变 |


## 性能分析

### 速度基准

| 模型 | 方法 | GPU | 吞吐 (tok/s) | 相对 FP16 | 质量损失 |
|------|------|-----|-------------|-----------|----------|
| Llama-3-70B | FP16 | 4xA100 | ~68 | 1.0x | 基线 |
| Llama-3-70B | GPTQ-4bit | 2xA100 | ~179 | 2.6x | 小 |
| Llama-3-70B | AWQ-4bit + Marlin | 2xA100 | ~741 | 10.9x | 小 |
| Llama-3-70B | FP8 | 2xH100 | ~380 | 5.6x | 极小 |
| Llama-3-70B | NVFP4 | 1xB200 | ~900* | ~13x | 小 |

*NVFP4 数据为估计值，基于 NVIDIA 公开基准。

### 质量 vs 压缩曲线

```
Perplexity
  增加 ▲
  2.0  │                                         ● 2-bit RTN
       │
  1.5  │                              ● 2-bit GPTQ
       │
  1.0  │                    ● 3-bit GPTQ
       │                           ○ 2-bit QuIP#
  0.5  │          ● 4-bit GPTQ
       │              ● 4-bit AWQ
  0.2  │     ● INT8 SmoothQuant
  0.1  │  ● FP8
  0.0  │● FP16 ──────────────────────────────────────
       └──────────────────────────────────────────── 压缩率 ▶
        1x     2x       4x        8x       16x
```


## 不足与局限

1. **质量不可避免地下降**：任何量化都会引入误差。低于 4-bit 时质量下降加速。对于需要极高精度的任务（数学推理、代码生成），建议使用 FP8 或更高精度。

2. **校准数据依赖**：GPTQ、AWQ 等方法需要校准数据，校准数据的代表性直接影响量化质量。如果校准数据与实际使用场景差距大，量化模型的表现会下降。

3. **异常值问题**：某些模型（尤其是较老的架构如 OPT、BLOOM）存在极端的激活异常值，使量化特别困难。

4. **任务特异性**：在通用基准（如 PPL）上表现良好的量化模型，在特定任务上可能退化更严重。建议在目标任务上评估。

5. **内核支持碎片化**：不同量化格式需要不同的推理内核。并非所有格式都有高度优化的内核（如 2-bit 量化的内核通常不够快）。

6. **量化不适用于所有层**：embedding 层和 LM head 通常保持较高精度，因为这些层对量化更敏感。

7. **MoE 模型的特殊考虑**：混合专家模型的各专家可能有不同的权重分布，统一量化可能不是最优的。


## 参考文献

- Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers," ICLR 2023. [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)
- Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration," MLSys 2024. [arXiv:2306.00978](https://arxiv.org/abs/2306.00978)
- Kim et al., "SqueezeLLM: Dense-and-Sparse Quantization," ICML 2024. [arXiv:2306.07629](https://arxiv.org/abs/2306.07629)
- Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models," ICML 2023. [arXiv:2211.10438](https://arxiv.org/abs/2211.10438)
- Chee et al., "QuIP: 2-Bit Quantization of Large Language Models with Guarantees," NeurIPS 2023. [arXiv:2307.13304](https://arxiv.org/abs/2307.13304)
- Chee et al., "QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks," ICML 2024. [arXiv:2402.04396](https://arxiv.org/abs/2402.04396)
- Dettmers et al., "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale," NeurIPS 2022. [arXiv:2208.07339](https://arxiv.org/abs/2208.07339)
- NVIDIA, "FP8 Formats for Deep Learning," 2022. [arXiv:2209.05433](https://arxiv.org/abs/2209.05433)


## 相关页面

- [[kv-cache-optimization]] -- KV 缓存量化的详细讨论
- [[vllm]] -- 支持 GPTQ、AWQ、FP8 等所有主要量化格式
- [[tensorrt-llm]] -- NVIDIA 原生量化支持（FP8、NVFP4）
- [[model-parallelism]] -- 量化可减少模型并行的需求
- [[speculative-decoding]] -- 量化草稿模型可进一步减少开销
- [[continuous-batching]] -- 量化提升单卡吞吐，影响批处理策略
