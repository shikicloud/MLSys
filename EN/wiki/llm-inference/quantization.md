---
title: "Quantization for LLM Inference"
category: llm-inference
tags: [quantization, gptq, awq, fp8, int4, weight-quantization, kv-cache-quantization]
created: 2026-04-13
updated: 2026-05-20
status: mature
---

# Quantization for LLM Inference

> [!abstract]+ TL;DR
> LLM weights in FP16/BF16 take 2 bytes/parameter -- a 70B model needs ~140 GB, more than a single GPU can hold. Quantization drops precision from FP16 to INT8/INT4/FP8/NVFP4, cutting memory 2-4x and lifting throughput. **The single most effective optimization for running larger models on fewer GPUs**. Methods covered: weight-only (**GPTQ**, **AWQ**, **SqueezeLLM**, **QuIP#**); weight + activation (**SmoothQuant**, **QuaRot**, **SpinQuant**); hardware-native formats (Hopper+ **FP8 E4M3**, Blackwell **NVFP4**); KV cache quantization (FP8/INT8/INT4 + [[saw-int4|BDR]]). For an overview of the family, see [[rotation-based-quantization]].

## Overview

LLM parameters are typically stored as FP16 or BF16 (16-bit floats), 2 bytes per parameter. A 70B-parameter model's weights alone need around 140 GB of GPU memory, beyond the capacity of a single card (e.g., A100 80GB).

**Quantization** lowers the numeric precision from high bit-widths (FP16/BF16) to lower ones (INT8, INT4, FP8, etc.), which:

1. **Reduces memory footprint**: INT4 quantization shrinks a model by ~4x (70B model: 140GB -> ~35GB)
2. **Improves inference throughput**: less data streamed from memory -> the memory-bandwidth-bound decode stage speeds up significantly
3. **Lowers hardware requirements**: models that needed many GPUs now fit on fewer

Quantization is **the single most effective optimization for running larger models on fewer GPUs**.

```
Precision vs. memory (Llama-3.3-70B):

  FP16:   ################################  140 GB  (2x H100)
  FP8:    ################                   70 GB  (1x H100)
  INT4:   ########                           35 GB  (1x A100-40G)
  NVFP4:  ########                           35 GB  (1x B200)
```

The core challenge is **minimizing the impact on model output quality** while reducing precision. Different methods strike different trade-offs between accuracy loss, quantization speed, and inference acceleration.


## Quantization Fundamentals

### Numeric Representation Basics

```
FP16 (half precision): 1 sign + 5 exponent + 10 mantissa bits
  Range: -65504 to 65504
  Precision: ~3.3 decimal digits

BF16 (Brain Float):    1 sign + 8 exponent + 7 mantissa bits
  Range: same as FP32 (~3.4e38)
  Precision: ~2.4 decimal digits

FP8 E4M3: 1 sign + 4 exponent + 3 mantissa bits
  Range: -448 to 448
  Precision: ~1.4 decimal digits

FP8 E5M2: 1 sign + 5 exponent + 2 mantissa bits
  Range: -57344 to 57344
  Precision: ~0.9 decimal digits

INT8: 8-bit integer
  Range: -128 to 127 (signed)
  256 discrete values

INT4: 4-bit integer
  Range: -8 to 7 (signed)
  Only 16 discrete values
```

### Symmetric vs. Asymmetric Quantization

**Symmetric quantization**: zero-point fixed at 0, one uniform scale.

```
Quantize:    q = round(x / scale)
Dequantize:  x' = q * scale

with scale = max(|x|) / (2^(b-1) - 1)

Example (FP16 -> INT8, symmetric):
  Original: [-1.2, 0.5, 3.7, -0.1, 2.8]
  max(|x|) = 3.7
  scale = 3.7 / 127 ~= 0.02913
  Quantized: [-41, 17, 127, -3, 96]
  Dequantized: [-1.194, 0.495, 3.700, -0.087, 2.796]
  Error:       [0.006, 0.005, 0.000, 0.013, 0.004]
```

**Asymmetric quantization**: includes a zero-point offset, making better use of the integer range.

```
Quantize:    q = round(x / scale) + zero_point
Dequantize:  x' = (q - zero_point) * scale

with scale = (max(x) - min(x)) / (2^b - 1)
     zero_point = round(-min(x) / scale)

When to use: distributions not centered on zero (e.g., post-ReLU activations)
```

### Quantization Granularity

```
+-------------------------------------------------------------+
|                  Quantization granularity                   |
+-------------+-----------------------------------------------+
|             |  Weight matrix W (shape: out_dim x in_dim)    |
|             |  +----------------------------------+         |
| Per-Tensor  |  | One scale for the whole matrix   |         |
|             |  | scale_tensor = max(|W|) / 127    |         |
|             |  | Worst accuracy, fastest          |         |
|             |  +----------------------------------+         |
|             |                                               |
|             |  +----------------------------------+         |
| Per-Channel |  | One scale per row (output chan)  |         |
|             |  | scale[i] = max(|W[i,:]|) / 127   |         |
|             |  | Better accuracy, most common     |         |
|             |  +----------------------------------+         |
|             |                                               |
|             |  +----------------------------------+         |
| Per-Group   |  | One scale per G elements (G=128) |         |
|             |  | Best accuracy, slight overhead   |         |
|             |  | GPTQ/AWQ default group_size=128  |         |
|             |  +----------------------------------+         |
+-------------+-----------------------------------------------+
```

Per-group quantization is the sweet spot between accuracy and efficiency. A typical setting is group_size=128: every 128 weights share a quantization scale (and zero point).

### Error Analysis

Quantization error can be viewed at two levels:

**Weight level**:
```
MSE = E[(W - W_q)^2]      -- weight reconstruction error
```

**Output level** (more important):
```
||WX - W_q X||^2           -- output reconstruction error (depends on X)
```

Key observation: **not all weights are equally important**. If a weight is repeatedly multiplied by a large activation, its quantization error gets amplified. This is the starting point for AWQ and related methods.


## Weight-Only Quantization

### GPTQ (ICLR 2023)

GPTQ is based on **Optimal Brain Quantization (OBQ)** and was the first method capable of quantizing 175B models to 3-4 bits in a few hours.

**Core idea**: quantize the weight matrix column by column; after each column is quantized, use Hessian information to compensate the remaining columns, minimizing the overall output error.

```
Algorithm:
  Input: weight matrix W, calibration set -> Hessian H = 2X^T X
  
  for col in range(columns):
      1. Quantize this column: w_q = quantize(W[:, col])
      2. Compute error: delta = W[:, col] - w_q
      3. Compensate remaining columns:
         W[:, col+1:] -= delta * H[col, col+1:] / H[col, col]
      
  Key: the compensation step uses the inverse Hessian to minimize output error
```

**Features**:
- One-shot (post-training), no retraining required
- Needs only a small calibration set (128-256 samples)
- ~4 GPU hours for a 175B model
- At 3-4 bits, perplexity increase < 0.5
- Supports group quantization (group_size=128)

**Limitations**:
- Quality depends on how representative the calibration set is
- Sensitive to outliers
- Serial column-by-column processing is slow

Paper: [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)

### AWQ (MLSys 2024 Best Paper)

AWQ (Activation-Aware Weight Quantization) is built around a key observation:

> Less than 1% of "salient weights" dominate the model output, and they can be identified by looking at **activation magnitudes**.

```
Key insight:

weight importance ~ activation magnitude

  weight w1 = 0.3, activation a1 = 100 -> contribution = 30    <- important!
  weight w2 = 0.5, activation a2 = 0.1 -> contribution = 0.05  <- not important
  
So: protect weight channels that correspond to large activations
```

**Method**:

1. Use calibration data to measure the mean activation magnitude per weight channel
2. Apply **per-channel scaling** to salient channels: scale up the weight before quantization
3. Compensate by scaling down at dequantize time -- no runtime cost

```
Original:    W_q = quantize(W)           -> salient weights heavily quantized
AWQ:         W_q = quantize(W * s) / s   -> salient weights better preserved

where s (per-channel scale) is set based on activation magnitude
optimal s*: grid search on [0, 1] to minimize quantization error
```

**Performance** (with the Marlin kernel):

| Method | Throughput (tok/s) | Relative speedup |
|------|-------------|----------|
| FP16 baseline | 68 | 1.0x |
| GPTQ-4bit + Marlin | 179 | 2.6x |
| AWQ-4bit + Marlin | 741 | 10.9x |

The huge AWQ + Marlin speedup comes from Marlin's heavy optimization for the AWQ quantization layout.

Paper: [arXiv:2306.00978](https://arxiv.org/abs/2306.00978)

### SqueezeLLM (ICML 2024)

**Core idea**: decompose weights into "normal" and "outlier" parts.

```
W = W_normal + W_sparse

W_normal: non-uniform quantization (K-means to find optimal levels)
W_sparse: sparse matrix storing outliers (full precision)
```

- **Non-uniform quantization**: instead of evenly spaced quantization levels, use K-means clustering to find the optimal codebook
- **Dense-sparse decomposition**: store outliers as a sparse matrix in full precision so they don't blow up the normal quantization range
- Stands out at sub-3-bit (e.g., 2-bit) quantization
- Suited to extreme compression scenarios

Paper: [arXiv:2306.07629](https://arxiv.org/abs/2306.07629)

### QuIP / QuIP#

**Core idea**: apply a random orthogonal transformation to make weights "incoherent" and squash outliers.

```
Original weights:      outliers, uneven distribution -> hard to quantize
After transformation:  W' = U^T W V  (U, V random orthogonal matrices)
                       -> weights become more uniform -> easier to quantize
At inference:          compensate by modifying the layer inputs and outputs
```

- QuIP# uses a lattice codebook for further encoding efficiency
- At 2-bit, quality is significantly better than GPTQ/AWQ
- Inference speed is limited by decoding overhead in practice

QuIP/QuIP# is the foundational paper of the **rotation-based quantization** family; follow-up work (QuaRot, SpinQuant, SAW-INT4/BDR) is covered in a dedicated section below.

### Comparison

| Method | Bit-width | Calibration | Quant time (70B) | PPL increase | Inference speed | Notes |
|------|------|----------|---------------|-----------------|----------|------|
| GPTQ | 3-4 bit | Yes | ~4h | <0.5 | Fast (Marlin) | Most mature |
| AWQ | 4 bit | Yes | ~1h | <0.3 | Fastest (Marlin) | Best balance |
| SqueezeLLM | 2-4 bit | Yes | ~8h | <1.0 (2-bit) | Medium | Extreme compression |
| QuIP# | 2-4 bit | Yes | ~6h | <0.5 (2-bit) | Slower | Theoretically optimal |
| Round-to-nearest | Any | No | Instant | Large | Fast | Baseline |


## Rotation-Based Quantization (QuIP -> QuaRot -> SpinQuant -> BDR)

This is a coherent family of techniques whose contribution is **not** a new quantizer but applying an *orthogonal transformation* **before** standard quantization, flattening outliers and making the resulting tensors easier to quantize. Multiplying by an orthogonal matrix preserves the L2 norm but redistributes energy across all dimensions; the rotated tensor has a more uniform distribution, and per-token (or per-channel) scale-and-zero quantization works much better on it.

Family timeline:

| Year | Method | Rotation location | Rotation type | Notes |
|------|------|-------------|---------|------|
| 2023 | **QuIP** ([arXiv:2307.13304](https://arxiv.org/abs/2307.13304)) | Weights | Random orthogonal | Introduces "incoherence processing"; first formalization of "random rotation makes low-bit quantization feasible". |
| 2024 | **QuIP#** ([arXiv:2402.04396](https://arxiv.org/abs/2402.04396)) | Weights | Random Hadamard + lattice codebook | Vector-quantize rotated weights; SOTA for 2-bit weights. |
| 2024 | **QuaRot** ([arXiv:2404.00456](https://arxiv.org/abs/2404.00456)) | Weights **and activations** | Random Hadamard, fused into adjacent linear weights | NeurIPS 2024. Shows the rotation can be *absorbed* into the neighboring linear layers (zero inference cost), enabling INT4-weight + INT4-activation Llama with nearly no quality loss. |
| 2024 | **SpinQuant** ([arXiv:2405.16406](https://arxiv.org/abs/2405.16406)) | Weights and activations | **Learned** rotation matrix | Replaces random Hadamard with a rotation trained on a calibration set. Larger accuracy gains; needs offline training. |
| 2026 | **SAW-INT4 / BDR** ([arXiv:2604.19157](https://arxiv.org/abs/2604.19157)) | **KV cache** | Block-diagonal Hadamard fused with INT4 writes | First production-friendly version for KV cache. Recovers Qwen3-4B-Thinking's GPQA from 0% under naive INT4 to 65.82%. See [[saw-int4]]. |

A few observations that cut across the family:

- **Where the rotation is absorbed matters.** QuaRot's contribution over QuIP is fusing the rotation into adjacent linear weights so inference cost stays the same. SAW-INT4 fuses the rotation into both the Triton KV-write kernel and the decode-side Q-attention kernel. Both are forms of "system-aware" rotation.
- **Random vs. learned vs. block-diagonal.** Learned (SpinQuant) > random Hadamard > no rotation, in terms of accuracy. Block-diagonal trades some rotation power for kernel cache locality and paged-layout compatibility.
- **Rotation is orthogonal to the quantizer.** GPTQ, AWQ, plain scale-and-zero, K-means -- any quantizer can be stacked on top of a rotation. Most of the literature pairs rotation with simple per-channel/per-token scale-zero quantization, because the rotation has already done the hard work.

For a deeper family synthesis (math foundations and trade-offs) see [[rotation-based-quantization]].

## FP8 Quantization

### E4M3 vs. E5M2

FP8 comes in two standard formats:

```
FP8 E4M3:                        FP8 E5M2:
+---+--------+----------+         +---+----------+-------+
| S |  Exp   | Mantissa |         | S |   Exp    | Mant  |
| 1 |  4 bit |  3 bit   |         | 1 |  5 bit   | 2 bit |
+---+--------+----------+         +---+----------+-------+

  Range: +/-448                    Range: +/-57344
  Precision: higher (8 mantissa)   Precision: lower (4 mantissa)
  
  Use: forward inference (W+A)     Use: gradients (training)
```

**Inference generally uses E4M3**: higher precision, dynamic range sufficient for inference.

### Hardware Support

| GPU | FP8 support | Performance |
|-----|---------|------|
| A100 (Ampere) | None | N/A |
| H100 (Hopper) | Native | ~2x vs FP16 |
| H200 (Hopper) | Native | ~2x vs FP16 |
| B100/B200 (Blackwell) | Native | ~2x vs FP16 |

### Dynamic vs. Static Scaling

FP8 quantization maps FP16/BF16 values into the FP8 range, requiring a scale:

**Static scaling**:
```
Determine a fixed scale during calibration
Pros: no extra compute at inference time
Cons: values outside the calibration range get clipped
```

**Dynamic scaling**:
```
Compute scale per inference based on actual tensor values
Pros: always uses the FP8 range optimally
Cons: requires a reduction to compute max
Real cost: small (~1-2%), usually worth it
```

**Per-tensor vs. per-channel vs. block-wise scaling**:

```
Per-tensor:   one scale for the whole tensor -> fastest, lowest accuracy
Per-channel:  one scale per output channel  -> balanced
Block-wise:   one scale per small block      -> most accurate; used by DeepSeek V3
```

### FP8 vs. INT8: Which to Pick?

| Aspect | FP8 E4M3 | INT8 |
|------|----------|------|
| Dynamic range | Large (+/-448) | Small (-128 to 127) |
| Hardware support | Hopper+ | Ampere+ |
| Quantization difficulty | Low (range is forgiving) | Medium (needs careful calibration) |
| Inference speed | Fast (native Tensor Core) | Fast (native Tensor Core) |
| Accuracy preservation | Excellent | Good (needs SmoothQuant or similar) |
| Recommended | Default on Hopper+ | Choice on Ampere |


## KV Cache Quantization

The KV cache is the other large consumer of inference memory. For long sequences and large batches, KV cache can account for 30-50% of total memory.

### KV Cache Memory Math

```
KV cache size = 2 x num_layers x num_kv_heads x head_dim x seq_len x batch_size x bytes_per_elem

Example (Llama-3.3-70B, FP16):
  = 2 x 80 x 8 x 128 x 4096 x 1 x 2 bytes
  = ~13.4 GB (per request!)
  
  batch_size=16: ~214 GB -- far exceeds the model weights
```

### FP8 KV Cache

```
KV cache from FP16 -> FP8:
  Memory halved: 13.4 GB -> 6.7 GB
  Quality impact: very small (perplexity increase < 0.1)
  
vLLM config:
  --kv-cache-dtype fp8
```

- Virtually no quality loss
- Halves memory -> serve more concurrent requests or longer sequences
- Recommended default on Hopper+

### INT4 KV Cache

```
KV cache from FP16 -> INT4 (per-group quantization):
  Memory 1/4: 13.4 GB -> ~3.4 GB (~4 GB with scale/zero_point overhead)
  Quality impact: small but measurable (perplexity increase ~0.3-0.5)
  
Needs per-group scaling (group_size=32 or 64) to preserve quality
```

- More aggressive memory savings
- Careful group_size selection required
- Particularly valuable for long-context workloads (KV cache for 128K+ tokens is enormous)

### Quality Impact of KV Cache Quantization

| Method | Memory savings | PPL increase (Llama-70B, Wiki) | When to use |
|------|----------|---------------------------|----------|
| FP16 (baseline) | 0% | 0 | Quality first |
| FP8 | 50% | <0.1 | Default recommended |
| INT8 | 50% | <0.1 | On Ampere |
| INT4 (g=64) | ~70% | 0.3-0.5 | Memory very tight |
| INT4 (g=32) | ~65% | 0.2-0.3 | Quality sensitive |


## Activation Quantization

### Why Is Activation Quantization Harder?

Weights are static (quantize once), but activations change with every inference and have severe outlier issues:

```
Typical activation distribution:
                  |
                  |
            #############
     ############################
  -----------------------------------------  <-  outlier!
  -2        -1         0         1    2      100
  
  Outliers are rare but 50-100x the normal magnitude
  Using a single scale crushes the precision of normal values
```

### SmoothQuant (ICML 2023)

**Core idea**: shift the quantization difficulty from activations into weights.

```
Observation:
  Activation X: has outliers -> hard to quantize
  Weight W:     fairly uniform -> easy to quantize
  
SmoothQuant: use a per-channel scale to migrate difficulty from X to W

  Y = X * W
    = (X * diag(s)^{-1}) * (diag(s) * W)
    = X_smooth * W_smooth

  s = max(|X_j|)^alpha / max(|W_j|)^(1-alpha)
  where alpha in [0, 1] controls migration strength (usually alpha=0.5)

Effect:
  X_smooth: outliers scaled down by s^{-1} -> easier to quantize
  W_smooth: absorbs some of the outliers -> still quantizable
```

- Enables **W8A8** (INT8 weight + INT8 activation) quantization
- No training required, only calibration data
- On a 175B model, accuracy stays close to FP16
- INT8 matmul has hardware acceleration on GPUs

Paper: [arXiv:2211.10438](https://arxiv.org/abs/2211.10438)

### W8A8 vs. W4A16 Paradigms

```
W8A8 (SmoothQuant):
  Weights: INT8, activations: INT8
  Compute: INT8 matmul (Tensor Core accelerated)
  Pros: compute and memory both accelerated
  Cons: activation quantization adds error
  Best for: large batch, compute-intensive

W4A16 (GPTQ/AWQ):
  Weights: INT4, activations: FP16
  Compute: dequantize weights -> FP16 matmul (or specialized kernel)
  Pros: no activation quantization error
  Cons: only memory accelerated, no compute boost (unless using Marlin etc.)
  Best for: small batch, memory-bandwidth-bound
```


## Code Examples

### Quantizing a Model with AutoAWQ

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# Load FP16 model
model_path = "meta-llama/Llama-3.1-8B-Instruct"
quant_path = "llama-3.1-8b-instruct-awq-4bit"

model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Quantization config
quant_config = {
    "zero_point": True,       # asymmetric quantization
    "q_group_size": 128,      # one group per 128 weights
    "w_bit": 4,               # 4-bit quantization
    "version": "GEMM",        # kernel version
}

# Run quantization (needs calibration data, defaults to a C4 subset)
model.quantize(tokenizer, quant_config=quant_config)

# Save the quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
print(f"Quantization complete! Model saved to {quant_path}")
```

### Loading a Quantized Model in vLLM

```python
from vllm import LLM, SamplingParams

# ---- GPTQ model ----
llm_gptq = LLM(
    model="TheBloke/Llama-3-70B-Instruct-GPTQ",
    quantization="gptq",
    tensor_parallel_size=2,
    # Marlin kernel is auto-enabled if the model is compatible
    # Manually: --quantization gptq_marlin
)

# ---- AWQ model ----
llm_awq = LLM(
    model="casperhansen/llama-3-70b-instruct-awq",
    quantization="awq",
    tensor_parallel_size=2,
)

# ---- FP8 online quantization (no pre-quantized model needed, Hopper+) ----
llm_fp8 = LLM(
    model="meta-llama/Llama-3.3-70B-Instruct",
    quantization="fp8",
    tensor_parallel_size=4,
    kv_cache_dtype="fp8",  # KV cache also in FP8
)

# ---- FP8 pre-quantized model ----
llm_fp8_pre = LLM(
    model="neuralmagic/Llama-3.3-70B-Instruct-FP8",
    tensor_parallel_size=4,
    kv_cache_dtype="fp8",
)

# Usage is identical
params = SamplingParams(temperature=0.7, max_tokens=512)
outputs = llm_fp8.generate(["Explain quantum computing"], params)
```

### Quality Evaluation: Perplexity Comparison

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def evaluate_perplexity(model, tokenizer, dataset_name="wikitext",
                        dataset_config="wikitext-2-raw-v1", max_length=2048):
    """Compute perplexity on a given dataset"""
    dataset = load_dataset(dataset_name, dataset_config, split="test")
    
    # Concatenate all text
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
            # Compute NLL
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

# Example numbers
# model_fp16: perplexity ~= 5.68
# model_awq4: perplexity ~= 5.82  (+0.14)
# model_gptq4: perplexity ~= 5.85 (+0.17)
# model_fp8: perplexity ~= 5.70   (+0.02)
```


## Picking a Quantization Method

```
                          Quantization decision tree
                          
                          What GPU are you running?
                              |
                +-------------+-------------+
                v             v             v
           Blackwell      Hopper         Ampere or older
           (B100/B200)   (H100/H200)    (A100/A10G)
                |             |             |
                v             v             v
           NVFP4 first      FP8 first    INT4/INT8
           (native 4-bit) (native 8-bit) (software quant)
                |             |             |
                |             |             +-> Small model? -> INT8 (SmoothQuant)
                |             |             +-> Large model? -> INT4 (AWQ > GPTQ)
                |             |
                |             +-> Quality first? -> FP8 (near-lossless)
                |             +-> Memory tight?  -> FP8 weights + FP8 KV cache
                |
                +-> Quality first? -> NVFP4 (hardware-optimized)
                    Memory tight?  -> NVFP4 + INT4 KV cache
                    
CPU inference (llama.cpp):
  +-> GGUF Q4_K_M (4-bit, ~92% quality) or Q5_K_M (5-bit, ~95% quality)
```

**Quick recommendations**:

| Scenario | Recommended method | Why |
|------|----------|------|
| H100/H200, no quality loss | FP8 (auto-quantize) | Near-lossless, 2x speedup |
| A100, run 70B | AWQ-4bit + Marlin | Best speed / quality balance |
| A100, quality matters | INT8 SmoothQuant | Minimal accuracy loss |
| B200, max throughput | NVFP4 | Hardware-native |
| Extreme compression (2-bit) | SqueezeLLM / QuIP# | Trade quality for compression |
| CPU / edge devices | GGUF Q4_K_M | llama.cpp ecosystem |
| KV cache too large | FP8 KV cache | Halve memory, minimal quality cost |


## Performance Analysis

### Speed Benchmarks

| Model | Method | GPU | Throughput (tok/s) | Relative to FP16 | Quality loss |
|------|------|-----|-------------|-----------|----------|
| Llama-3-70B | FP16 | 4xA100 | ~68 | 1.0x | Baseline |
| Llama-3-70B | GPTQ-4bit | 2xA100 | ~179 | 2.6x | Small |
| Llama-3-70B | AWQ-4bit + Marlin | 2xA100 | ~741 | 10.9x | Small |
| Llama-3-70B | FP8 | 2xH100 | ~380 | 5.6x | Minimal |
| Llama-3-70B | NVFP4 | 1xB200 | ~900* | ~13x | Small |

*NVFP4 figures are estimates based on NVIDIA public benchmarks.

### Quality vs. Compression Curve

```
Perplexity
 increase ^
  2.0  |                                         * 2-bit RTN
       |
  1.5  |                              * 2-bit GPTQ
       |
  1.0  |                    * 3-bit GPTQ
       |                           o 2-bit QuIP#
  0.5  |          * 4-bit GPTQ
       |              * 4-bit AWQ
  0.2  |     * INT8 SmoothQuant
  0.1  |  * FP8
  0.0  |* FP16 --------------------------------------
       +---------------------------------------------> compression ratio
        1x     2x       4x        8x       16x
```


## Limitations

1. **Quality always drops**: every quantization introduces error. Below 4-bit the drop accelerates. For accuracy-critical tasks (math reasoning, code generation), use FP8 or higher precision.

2. **Calibration data dependence**: GPTQ, AWQ, and similar methods rely on calibration data; how representative it is directly determines quantization quality. A poor match between calibration data and actual usage degrades the quantized model.

3. **Outlier issues**: some models (especially older architectures like OPT, BLOOM) have extreme activation outliers that make quantization especially hard.

4. **Task-specific behavior**: a quantized model that does well on general benchmarks (like PPL) may degrade more on specific tasks. Evaluate on the target task.

5. **Fragmented kernel support**: different quantization formats need different inference kernels. Not all formats have highly optimized kernels (2-bit kernels are usually not fast enough).

6. **Quantization isn't for every layer**: embedding layers and the LM head are usually kept at higher precision because they're more sensitive to quantization.

7. **MoE models need special care**: each expert in a mixture-of-experts model can have its own weight distribution; uniform quantization may not be optimal.


## References

- Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers," ICLR 2023. [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)
- Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration," MLSys 2024. [arXiv:2306.00978](https://arxiv.org/abs/2306.00978)
- Kim et al., "SqueezeLLM: Dense-and-Sparse Quantization," ICML 2024. [arXiv:2306.07629](https://arxiv.org/abs/2306.07629)
- Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models," ICML 2023. [arXiv:2211.10438](https://arxiv.org/abs/2211.10438)
- Chee et al., "QuIP: 2-Bit Quantization of Large Language Models with Guarantees," NeurIPS 2023. [arXiv:2307.13304](https://arxiv.org/abs/2307.13304)
- Chee et al., "QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks," ICML 2024. [arXiv:2402.04396](https://arxiv.org/abs/2402.04396)
- Ashkboos et al., "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs," NeurIPS 2024. [arXiv:2404.00456](https://arxiv.org/abs/2404.00456)
- Liu et al., "SpinQuant: LLM Quantization with Learned Rotations," 2024. [arXiv:2405.16406](https://arxiv.org/abs/2405.16406)
- Jia et al., "SAW-INT4: System-Aware 4-Bit KV-Cache Quantization for Real-World LLM Serving," 2026. [arXiv:2604.19157](https://arxiv.org/abs/2604.19157)
- Dettmers et al., "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale," NeurIPS 2022. [arXiv:2208.07339](https://arxiv.org/abs/2208.07339)
- NVIDIA, "FP8 Formats for Deep Learning," 2022. [arXiv:2209.05433](https://arxiv.org/abs/2209.05433)


## Related Pages

- [[kv-cache-optimization]] -- detailed discussion of KV cache quantization
- [[saw-int4]] -- block-diagonal Hadamard rotation + INT4 KV quantization (paper deep-dive)
- [[rotation-based-quantization]] -- QuIP / QuaRot / SpinQuant / BDR family overview
- [[vllm]] -- supports GPTQ, AWQ, FP8, and all major quantization formats
- [[tensorrt-llm]] -- NVIDIA's native quantization support (FP8, NVFP4)
- [[model-parallelism]] -- quantization reduces the need for model parallelism
- [[speculative-decoding]] -- quantizing the draft model further reduces overhead
- [[continuous-batching]] -- quantization lifts per-GPU throughput and affects batching strategy
