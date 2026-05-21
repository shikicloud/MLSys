---
title: "Data Parallelism: DDP, ZeRO, FSDP, HSDP, DWDP — and the memory account they fight over"
category: ml-infra
tags: [data-parallelism, ddp, zero, fsdp, hsdp, dwdp, optimizer-states, mixed-precision, memory, concept]
created: 2026-05-22
updated: 2026-05-22
status: growing
---

# Data Parallelism: DDP, ZeRO, FSDP, HSDP, DWDP — and the memory account they fight over

> [!info] Page scope
> Concept-synthesis page covering the **DP family**: DDP, the ZeRO stages, PyTorch FSDP (1 and 2), HSDP, and the inference-side cousin DWDP. Also covers the memory account they're all attacking — what optimizer states are, why gradients have the same count as weights, why FFN dominates a Transformer's parameter budget, and what "activation" actually means in this context.

---

## Summary (read this if you have 2 minutes)

**What DP is.** Data Parallelism = "split the *batch* across workers, replicate the *model*, AllReduce gradients at the end of each step". DDP is the canonical implementation. DP is **one axis** of parallelism — orthogonal to Tensor Parallel (TP), Pipeline Parallel (PP), Context Parallel (CP), Expert Parallel (EP).

**The DP family — variants that trade comm for memory.** ZeRO and FSDP are **still DP** — they still split the batch the same way — they just **shard the model state** (params / grads / optimizer states) across the DP ranks to eliminate redundancy. Each variant shards one more component:

| Variant | Splits batch? | Shards opt states | Shards grads | Shards params |
| ------- | :-----------: | :---------------: | :----------: | :-----------: |
| **DDP** | ✓ | ✗ | ✗ | ✗ |
| **ZeRO-1** | ✓ | ✓ | ✗ | ✗ |
| **ZeRO-2 / FSDP `SHARD_GRAD_OP`** | ✓ | ✓ | ✓ | ✗ |
| **ZeRO-3 / FSDP `FULL_SHARD`** | ✓ | ✓ | ✓ | ✓ |

**Headline savings.** For a 100B model with Adam mixed precision, each rank goes from **1600 GB (DDP)** → **200 GB (ZeRO-3)** at 8-way DP. Cost: extra AllGather communication per layer.

**Why it matters.** Without ZeRO/FSDP, training anything above ~7B on a single 80GB GPU is impossible. With ZeRO-3 / FSDP, you can train 70B-class models on a single 8×H100 node. With ZeRO-Infinity (CPU + NVMe offload) you can train 1T-class on a single node, just slower.

---

# Depth (drill-down starts here)

## Background: the memory account in modern training

Mixed-precision Adam training stores **16 bytes per parameter** on every DP rank under vanilla DDP:

| Component | Precision | Bytes / param | Notes |
| --------- | --------- | -------------:| ----- |
| Working weight | FP16 / BF16 | 2 | what forward/backward see |
| Gradient | FP16 / BF16 | 2 | one per param (same count as weights) |
| **Master weight (FP32)** | FP32 | **4** | persistent FP32 copy — updates happen here |
| **Adam first moment $m$** | FP32 | **4** | $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$ |
| **Adam second moment $v$** | FP32 | **4** | $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$ |
| **Total** | | **16** | of which **12 = optimizer states** |

The boldface rows ("optimizer states") = **12 bytes/param = 75% of total memory**. That's the prize ZeRO is going after.

For a 100B model: 100B × 16 = **1.6 TB per rank**. Doesn't fit on any GPU today. Without DP-internal sharding, training stops at ~7B on a single 80GB H100.

Source paper: [ZeRO (arXiv:1910.02054)](https://arxiv.org/abs/1910.02054) §3.2.

### Gradients have the same count as weights — why

For each parameter scalar $\theta_i$, backprop produces one gradient scalar $\partial L / \partial \theta_i$. Loss $L$ is a scalar; its gradient w.r.t. a tensor has the *same shape* as the tensor:

$$
\frac{\partial L}{\partial W} \in \mathbb{R}^{d_1 \times d_2}, \quad \text{where } W \in \mathbb{R}^{d_1 \times d_2}.
$$

Geometric intuition: the parameter space is $\Psi$-dimensional ($\Psi$ = total param count). Loss is a scalar function $L: \mathbb{R}^\Psi \to \mathbb{R}$. Its gradient $\nabla L \in \mathbb{R}^\Psi$ lives in the same space. **Same number of components by definition.**

Dtypes can differ (FP16 weight + FP16 grad in mixed precision; FP32 grad accumulator in some setups), but the **count is identical**.

### Optimizer states by optimizer

| Optimizer | State per param | Bytes (FP32) | Comment |
| --------- | --------------- | ------------:| ------- |
| **SGD** (no momentum) | — | 0 | stateless |
| **SGD with momentum** | velocity | 4 | one EMA |
| **Adam / AdamW** | master + $m$ + $v$ | **12** | the default for LLMs |
| **Lion** ([Chen et al. 2023](https://arxiv.org/abs/2302.06675)) | momentum only | 4 | ⅓ of Adam |
| **Adafactor** | factored $v$ (row + col vectors) | ~5 | Google's memory-saver |
| **8-bit Adam** ([Dettmers et al. 2022](https://arxiv.org/abs/2110.02861)) | INT8-quantized $m$, $v$ | ~6 | ~½ Adam |

PyTorch's `torch.optim.AdamW` keeps `state['exp_avg']` ($m$) and `state['exp_avg_sq']` ($v$):
[`torch/optim/adamw.py`](https://github.com/pytorch/pytorch/blob/main/torch/optim/adamw.py)

`bitsandbytes` 8-bit Adam:
[`bitsandbytes/optim/adamw.py`](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/bitsandbytes/optim/adamw.py)

---

## The DP family in detail

### DDP — distributed data parallel

The baseline. Each rank holds a complete model replica. Forward + backward happen locally on the rank's batch slice; gradients are AllReduced at the end of backward; optimizer step happens identically on every rank.

- Code: [`torch/nn/parallel/distributed.py`](https://github.com/pytorch/pytorch/blob/main/torch/nn/parallel/distributed.py)
- Docs: [PyTorch DDP notes](https://docs.pytorch.org/docs/stable/notes/ddp.html)
- Communication per step: **1× AllReduce(gradients)** — that's it.
- Memory per rank: full $16 \Psi$ under Adam mixed precision.

### ZeRO-1 — shard optimizer states ($P_{os}$)

The 75% optimizer-state slice is split $N_d$ ways across DP ranks. Each rank only updates the parameters it "owns" the optimizer state for; after the step, an **AllGather** broadcasts the updated weights so everyone's working copy stays in sync.

- Per-rank memory: $4\Psi + 4\Psi + \frac{12\Psi}{N_d}$ (weight + grad + sharded opt state)
- Communication: AllReduce(grad) + AllGather(updated param)
- Code: [`deepspeed/runtime/zero/stage_1_and_2.py`](https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/runtime/zero/stage_1_and_2.py) — `partition_grads=False` branch
- Megatron-Core's distributed optimizer is essentially ZeRO-1: [`megatron/core/optimizer/distrib_optimizer.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/optimizer/distrib_optimizer.py)

### ZeRO-2 — also shard gradients ($P_{os+g}$)

Now the 2-byte-per-param gradient buffer is also sharded. Reduce-Scatter replaces AllReduce — each rank only ends up with the gradient slice for the parameters it owns.

- Per-rank memory: $4\Psi + \frac{2\Psi}{N_d} + \frac{12\Psi}{N_d}$
- Communication: **Reduce-Scatter(grad) + AllGather(param)** = same total bytes as DDP's AllReduce, just routed differently
- Code: same file, `partition_grads=True`

### ZeRO-3 — shard parameters too ($P_{os+g+p}$)

The most aggressive. Each rank holds only $\frac{1}{N_d}$ of the parameters at rest. Right before each layer's forward, the full parameters of that layer are **AllGathered** from peer ranks; immediately after compute, they're freed.

- Per-rank memory: $\frac{16\Psi}{N_d}$ + max-layer-params transient buffer
- Communication: **per-layer AllGather** during forward + symmetric in backward + Reduce-Scatter(grad). ~1.5× DDP's bytes.
- Code: [`deepspeed/runtime/zero/stage3.py`](https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/runtime/zero/stage3.py) — see `_pre_forward_module_hook` (AllGather on entry) and `_post_forward_module_hook` (partition on exit). This is the trick that makes the memory account work.

### ZeRO offload variants

| Variant | What it offloads | Where to | Paper |
| ------- | ---------------- | -------- | ----- |
| **ZeRO-Offload** | optimizer states + master weights | CPU DRAM | [arXiv:2101.06840](https://arxiv.org/abs/2101.06840) |
| **ZeRO-Infinity** | params + grads + opt states + activations | CPU + **NVMe** | [arXiv:2104.07857](https://arxiv.org/abs/2104.07857) |
| **ZeRO++** | quantized AllGather (INT8 weights) + hierarchical sharding | — | [arXiv:2306.10209](https://arxiv.org/abs/2306.10209) |

ZeRO-Infinity hierarchy: **GPU HBM → CPU DRAM → NVMe SSD**, with [`deepspeed/runtime/swap_tensor/`](https://github.com/microsoft/DeepSpeed/tree/master/deepspeed/runtime/swap_tensor) handling the I/O.

### FSDP — PyTorch's take

FSDP is PyTorch's adoption of ZeRO-3 ideas into the native framework. There have been **two implementations** ("FSDP1" and "FSDP2"); both are still callable, but the framework is moving everyone to FSDP2.

> **No official "FSDP3" exists** as of 2026-05. Future plans tracked in [TorchTitan](https://github.com/pytorch/torchtitan), but no new numbered generation has been announced.

#### FSDP1 (PyTorch 1.11+, 2022)

- Paper: [*PyTorch FSDP* (arXiv:2304.11277)](https://arxiv.org/abs/2304.11277)
- Internal data structure: `FlatParameter` — a module's parameters are flattened into one 1D tensor, then chunked across ranks.
- Communication granularity: one AllGather per `FlatParameter` (coarser, fewer kernels)
- Problems: doesn't compose well with TP / LoRA / partial freezing (the flat representation hides per-parameter semantics); coarse mixed-precision config; rough `torch.compile` interaction.
- Code (still callable, deprecation-track): [`torch/distributed/fsdp/fully_sharded_data_parallel.py`](https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/fully_sharded_data_parallel.py)
- Flat-param implementation: [`torch/distributed/fsdp/_flat_param.py`](https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/_flat_param.py)

#### FSDP2 (PyTorch 2.4+, 2024)

Complete rewrite. **Per-parameter sharding** using [`DTensor`](https://github.com/pytorch/pytorch/tree/main/torch/distributed/tensor) — each `nn.Parameter` is independently sharded.

- API changed from class-wrapping (`FSDP(model, ...)`) to functional (`fully_shard(model, ...)`)
- Composes natively with TP (both use DTensor) → 2D parallelism is a one-liner
- Mixed precision configurable per-parameter
- Friendly with PEFT (LoRA, frozen params) — just set `requires_grad=False`
- Design RFC: [pytorch/pytorch#114299](https://github.com/pytorch/pytorch/issues/114299)
- Tutorial: [PyTorch FSDP2 tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- Code:
  - API: [`torch/distributed/_composable/fsdp/_fully_shard/_fully_shard.py`](https://github.com/pytorch/pytorch/blob/main/torch/distributed/_composable/fsdp/_fully_shard/_fully_shard.py)
  - Param sharding: [`torch/distributed/_composable/fsdp/_fully_shard/_fsdp_param.py`](https://github.com/pytorch/pytorch/blob/main/torch/distributed/_composable/fsdp/_fully_shard/_fsdp_param.py)
  - Pre/post-forward hooks (the AllGather + partition machinery): [`_fsdp_param_group.py`](https://github.com/pytorch/pytorch/blob/main/torch/distributed/_composable/fsdp/_fully_shard/_fsdp_param_group.py)

#### FSDP1 vs FSDP2 cheat sheet

| Dimension | FSDP1 | FSDP2 |
| --------- | ----- | ----- |
| Internal repr | FlatParameter (1D flatten) | DTensor (per-param) |
| API | `FSDP(model, ...)` | `fully_shard(model, ...)` |
| TP composition | Manual via HSDP/2D mesh | Native DTensor mesh |
| LoRA / partial freeze | Hard | Easy |
| Mixed precision granularity | Per-FlatParameter | Per-Parameter |
| `torch.compile` | Partial | First-class |
| Status | Deprecation-track | **Default since PyTorch 2.5** |

PyTorch 2.5 release: [pytorch.org blog](https://pytorch.org/blog/pytorch2-5/) (search "FSDP2").

#### FSDP ShardingStrategy mapping to ZeRO

| ZeRO stage | FSDP `ShardingStrategy` | Notes |
| ---------- | ----------------------- | ----- |
| ZeRO-1 | (no exact match) | closest via mixed-stage config |
| ZeRO-2 | `SHARD_GRAD_OP` | shards opt states + grads |
| ZeRO-3 | `FULL_SHARD` | shards everything |
| — | `NO_SHARD` | DDP |
| — | `HYBRID_SHARD` | full shard intra-node, replicate inter-node (HSDP) |

Enum source: [`torch/distributed/fsdp/api.py`](https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/api.py).

### HSDP — Hybrid Sharded Data Parallel

Shard *within* a node (NVLink, ~600 GB/s), replicate *across* nodes (IB, ~25 GB/s). Lets you keep the AllGather/Reduce-Scatter traffic on the fast intra-node fabric and only do AllReduce across the slower inter-node link.

- Pre-condition: model fits in a single node (8 GPUs × 80 GB = 640 GB)
- FSDP2 config: pass a 2D `DeviceMesh` with dims `("replicate", "shard")`
- Tutorial: [FSDP advanced tutorial — HSDP section](https://docs.pytorch.org/tutorials/intermediate/FSDP_advanced_tutorial.html)
- DeepSpeed's equivalent is ZeRO++'s **hpZ** (hierarchical partitioning) — see [`deepspeed/runtime/zero/config.py`](https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/runtime/zero/config.py) `zero_hpz_partition_size`

### DWDP — the inference-side cousin

Not training-related, but conceptually adjacent. **DWDP (Distributed Weight Data Parallelism)** is an *inference* parallelization for MoE models on NVL72-class hardware that borrows ZeRO/FSDP's "shard weights, fetch on demand" pattern.

- Paper: [arXiv:2604.01621](https://arxiv.org/abs/2604.01621) (Li et al., 2026-04, NVIDIA)
- Key difference: removes the inter-rank collective synchronization that EP MoE needs at every layer, lets each GPU progress independently and fetch missing expert weights via peer-to-peer NVLink
- Targets MoE inference on GB200 NVL72; shows +8.8% TPS/GPU on DeepSeek-R1
- **Not training-side** — there are no optimizer states or gradients to shard at inference. The thing being sharded is MoE expert weights, fetched on demand during forward.

---

## DP at inference time

At inference, there are no optimizer states or gradients. The DP-relevant problem becomes "how to handle batches across GPUs without replicating large weights". For dense models, ordinary DP works. For MoE models, the standard pattern is **DP attention + EP MoE**.

### DP attention + EP MoE

Production-default for MoE inference (DeepSeek-V3 + vLLM + SGLang all use it):

| Layer type | Parallelism | Why |
| ---------- | ----------- | --- |
| Attention | **DP** (each rank holds full attention weights, sees its batch slice) | Attention weights are small relative to MoE; DP avoids TP's KV-cache splitting headaches |
| MoE FFN | **EP** (experts sharded across ranks) | MoE total weights too big to replicate (DeepSeek-V3: 671B total, 37B active) |

Per-step flow within one layer:

```
batched tokens (DP-distributed)
    │
    ├── DP attention   ── each rank does attention on its own tokens locally
    │
    ├── router          ── decides which experts each token goes to
    │
    ├── AllToAll dispatch ── shuffle tokens to the GPU holding the chosen expert
    │
    ├── EP MoE          ── each rank runs its experts on the tokens routed to it
    │
    └── AllToAll combine  ── shuffle outputs back to original rank
```

Code references:
- vLLM: [`vllm/distributed/parallel_state.py`](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/parallel_state.py) for `ep_group`, [`vllm/v1/worker/gpu_model_runner.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu_model_runner.py) for `dp_size` handling.
- SGLang: [`sglang/srt/layers/moe/`](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/layers/moe).
- DeepSeek-V3 paper describing this architecture: [arXiv:2412.19437](https://arxiv.org/abs/2412.19437).

DWDP attacks the AllToAll synchronization in this pattern.

---

## Where the memory actually lives: Transformer block anatomy

To know what DP-sharding is *worth* sharding, you need to know where the parameter mass sits in a Transformer.

### Attention vs FFN — division of labor

Each transformer block has two sub-layers, both shaped `[B, S, H] → [B, S, H]`, both wrapped in residual + LayerNorm:

| Property | Attention | FFN |
| -------- | --------- | --- |
| Information flow | **mixes across sequence dim** (token-to-token) | **mixes within hidden dim** (per-token) |
| Math | $\text{softmax}(QK^T/\sqrt{d}) V$ | $\text{down}(\sigma(\text{gate}(x)) \odot \text{up}(x))$ |
| Role analogy | **communication** between tokens | **computation** per token |
| Complexity | $O(S^2 H)$ | $O(S \cdot H \cdot H_{\text{ffn}})$ |
| Memory growth | $O(S^2)$ peak (FlashAttention → $O(S)$) | $O(S)$ |

They alternate: attention pulls context in, FFN processes the per-token result.

### Activation — two distinct meanings

The word is overloaded:

- **Activation function** (singular): the nonlinearity *inside* the FFN. Llama / Mistral / DeepSeek use **SwiGLU**; older GPT-2/3 use GeLU; BERT uses GeLU; Gemma uses GeGLU. Without it, two linear layers collapse to one.
- **Activations** (plural noun): the intermediate tensors stored during forward — every layer's input/output, attention scores, FFN intermediate `[B, S, H_ffn]` (this one's the biggest single tensor). These are *data*, not weights: they depend on the input batch, and they're what backward needs for the chain rule. **This is what activation checkpointing trades off** — don't store them, recompute on backward.

The memory account split:

```
Total per-rank memory =
    Weights        (depends on model size; static; what DP/FSDP shards)
  + Gradients      (same count as weights; static during a step)
  + Optimizer state (depends on optimizer; static)
  + Activations    (depends on batch × seq; lives during forward; biggest single chunk often)
  + Buffers / workspace (smaller)
```

ZeRO/FSDP focus on the first three. Activations are managed separately by activation checkpointing — see [`torch.distributed.algorithms._checkpoint.checkpoint_wrapper`](https://github.com/pytorch/pytorch/tree/main/torch/distributed/algorithms/_checkpoint) and [`torch.utils.checkpoint`](https://github.com/pytorch/pytorch/blob/main/torch/utils/checkpoint.py).

### Why attention parameters are small, FFN parameters are large

Hidden dim $H$, num Q-heads $n_q$, num KV-heads $n_{kv}$, FFN width $H_{\text{ffn}}$.

**Attention** (multi-head with GQA ratio $r = n_q / n_{kv}$):

| Matrix | Shape | Params |
| ------ | ----- | -----: |
| $W_Q$ | $H \times H$ | $H^2$ |
| $W_K$ | $H \times H/r$ | $H^2/r$ |
| $W_V$ | $H \times H/r$ | $H^2/r$ |
| $W_O$ | $H \times H$ | $H^2$ |
| **Total** | | $H^2(2 + 2/r)$ |

For full MHA ($r=1$): $4 H^2$. For Llama 3 70B GQA ($r=8$): $2.25 H^2$.

**FFN** (SwiGLU, three matrices because of the gate):

| Matrix | Shape | Params |
| ------ | ----- | -----: |
| $W_{\text{gate}}$ | $H \times H_{\text{ffn}}$ | $H \cdot H_{\text{ffn}}$ |
| $W_{\text{up}}$ | $H \times H_{\text{ffn}}$ | $H \cdot H_{\text{ffn}}$ |
| $W_{\text{down}}$ | $H_{\text{ffn}} \times H$ | $H \cdot H_{\text{ffn}}$ |
| **Total** | | $3 H \cdot H_{\text{ffn}}$ |

Modern $H_{\text{ffn}} \approx 8H/3$ (keeps total FLOPs ≈ vanilla 2-matrix MLP with $4H$ width), giving FFN $\approx 8 H^2$.

**Ratio**:

| Config | Attention | FFN | FFN / Attention |
| ------ | --------- | --- | ---------------: |
| MHA + vanilla GPT FFN | $4 H^2$ | $8 H^2$ | **2×** |
| MHA + SwiGLU | $4 H^2$ | $8 H^2$ | **2×** |
| GQA(r=8) + SwiGLU | $2.25 H^2$ | $8 H^2$ | **3.6×** |

**FFN dominates by 2-4×.** GQA makes it worse (attention shrinks; FFN doesn't).

### A real 100B-class breakdown: Llama 3.1 70B

Config from [HF model card](https://huggingface.co/meta-llama/Llama-3.1-70B/blob/main/config.json):

```json
{
  "hidden_size": 8192,
  "intermediate_size": 28672,
  "num_hidden_layers": 80,
  "num_attention_heads": 64,
  "num_key_value_heads": 8,
  "vocab_size": 128256
}
```

**Per layer** (head_dim = 128, GQA 1:8 so KV out_dim = 8 × 128 = 1024):

| Block | Matrix | Shape | Params |
| ----- | ------ | ----- | -----: |
| Attention | $W_Q$ | $8192 \times 8192$ | 67.1 M |
|  | $W_K$ | $8192 \times 1024$ | 8.4 M |
|  | $W_V$ | $8192 \times 1024$ | 8.4 M |
|  | $W_O$ | $8192 \times 8192$ | 67.1 M |
|  | input layernorm | $8192$ | 8 K |
|  | **Attention subtotal** | | **151.0 M** |
| FFN | $W_{\text{gate}}$ | $8192 \times 28672$ | 234.9 M |
|  | $W_{\text{up}}$ | $8192 \times 28672$ | 234.9 M |
|  | $W_{\text{down}}$ | $28672 \times 8192$ | 234.9 M |
|  | post-attn layernorm | $8192$ | 8 K |
|  | **FFN subtotal** | | **704.6 M** |
| **Per-layer total** | | | **855.6 M** |

**Whole model**:

| Component | Params | Share |
| --------- | -----: | -----: |
| 80 × Attention | 12.1 B | **17.1 %** |
| 80 × FFN | 56.4 B | **79.9 %** |
| Input embedding ($128256 \times 8192$) | 1.05 B | 1.5 % |
| LM head ($8192 \times 128256$, untied in Llama 3) | 1.05 B | 1.5 % |
| Final RMSNorm | 8 K | ~0 % |
| **Total** | **70.55 B** | 100 % |

**FFN is ~80% of the parameter budget.** This is why DP-sharding pays off most for the FFN matrices — they're the bulk of what's being replicated. It's also why MoE designs (DeepSeek-V3, Mixtral) target FFN — making it sparse via expert routing gets far more bang per FLOP than messing with attention.

### MoE flips the picture: DeepSeek-V3

DeepSeek-V3 is **671 B total / 37 B active** ([arXiv:2412.19437](https://arxiv.org/abs/2412.19437), [config.json](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json)):

- $H = 7168$, 60 layers
- Each layer: 1 shared expert + **256 routed experts**, top-8 active per token
- Each expert is a small FFN ($H \to H_{\text{ffn},e} \to H$ with $H_{\text{ffn},e} = 2048$)
- Almost the entire 671 B parameter count lives in the 256-expert pool; only the 8 routed experts contribute compute per token

This is why MoE inference needs **EP** rather than DP for the FFN — you can't replicate 671 B of weights, but you can route per-token to the right shard.

---

## DP vs other parallel axes

DP is **one axis** in 3D / 5D parallelism. It's *orthogonal* to TP / PP / CP / EP — you can stack them all.

```
Total GPUs = DP × TP × PP × CP × EP
```

| Axis | Splits | Communicates | Code reference |
| ---- | ------ | ------------ | -------------- |
| **DP** | batch (and optionally model state) | AllReduce / Reduce-Scatter / AllGather (grads) | this page |
| **TP** | layer-internal matrices | AllReduce (output) per layer | [Megatron-LM TP](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/tensor_parallel) |
| **PP** | layers across stages | P2P (activation across stage boundary) | [Megatron-LM PP](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/pipeline_parallel) |
| **CP** | sequence dimension | Ring P2P or AllToAll (KV) | [[ring-attention]], [[deepspeed-ulysses]] |
| **EP** | MoE experts | AllToAll (tokens) | [vLLM EP](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/parallel_state.py) |
| **SP** (Megatron version) | LayerNorm/Dropout activations | extra AllGather/ReduceScatter | [Megatron-LM SP](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/layers.py) |

**Common misconception**: "FSDP is model parallel." No — FSDP is DP. Each rank still runs the *whole model's* forward, params are just gathered on the fly. TP is true model parallel — each rank only runs its slice of each matrix.

External link comparing the two: [PyTorch TP docs](https://docs.pytorch.org/docs/stable/distributed.tensor.parallel.html) vs [FSDP2 docs](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html).

Megatron-Core's `parallel_state.py` is the canonical reference for stacking these axes: [link](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py) — DP is the residual dimension after the others are configured.

---

## How to pick

| Situation | Recommendation |
| --------- | -------------- |
| Model fits in single-GPU memory (with optimizer states) | DDP — simplest, fastest |
| Model ≤ 70B class, single 8×80GB node | **HSDP (FSDP2 + 2D mesh)** — sharding intra-node, replicate inter-node |
| Model > 100B, single node can't fit | **FSDP2 FULL_SHARD** (= ZeRO-3) across the cluster |
| Extreme scale, even cluster can't hold params | **ZeRO-Infinity** (CPU + NVMe offload) — slower but works |
| Already on Megatron stack | Megatron-Core's [`distributed_optimizer`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/optimizer/distrib_optimizer.py) (ZeRO-1 in Megatron clothing) |
| RL training (PPO/GRPO/DAPO) of LLMs | **FSDP2** — veRL / NeMo-RL / OpenRLHF all default to it; better PyTorch-native integration than DeepSpeed for this workload. See e.g. veRL's [`fsdp_workers.py`](https://github.com/volcengine/verl/blob/main/verl/workers/fsdp_workers.py) |
| MoE inference on NVL72 | **DWDP** (the inference cousin) or DP attention + EP MoE |

---

## References

**Primary papers**:
- [ZeRO (arXiv:1910.02054)](https://arxiv.org/abs/1910.02054) — the foundational paper introducing stages 1/2/3
- [ZeRO-Offload (arXiv:2101.06840)](https://arxiv.org/abs/2101.06840) — CPU offload of optimizer states
- [ZeRO-Infinity (arXiv:2104.07857)](https://arxiv.org/abs/2104.07857) — NVMe offload, 1T on single node
- [ZeRO++ (arXiv:2306.10209)](https://arxiv.org/abs/2306.10209) — quantized AllGather + hierarchical partitioning
- [PyTorch FSDP (arXiv:2304.11277)](https://arxiv.org/abs/2304.11277) — FSDP1 design
- [DWDP (arXiv:2604.01621)](https://arxiv.org/abs/2604.01621) — inference-side cousin for MoE on NVL72

**Frameworks**:
- DeepSpeed: [github.com/microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed)
- PyTorch FSDP2: [`torch.distributed._composable.fsdp`](https://github.com/pytorch/pytorch/tree/main/torch/distributed/_composable/fsdp)
- Megatron-Core distributed optimizer: [`megatron/core/optimizer/distrib_optimizer.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/optimizer/distrib_optimizer.py)
- TorchTitan (PyTorch native pretrain reference): [github.com/pytorch/torchtitan](https://github.com/pytorch/torchtitan)

**Tutorials**:
- [PyTorch FSDP2 tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [PyTorch HSDP tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_advanced_tutorial.html)
- [DeepSpeed ZeRO tutorial](https://www.deepspeed.ai/tutorials/zero/)

## Related reading

- [[distributed-training]] — broader survey of distributed training (3D parallelism, mixed precision, fault tolerance, fault tolerance)
- [[ring-attention]], [[deepspeed-ulysses]] — the two main flavors of Context Parallelism (the CP axis, orthogonal to DP)
- [[parallelism-strategies-deep-dive]] — full canonical reference for all parallel axes
- [[training-frameworks]] — Megatron-LM, DeepSpeed, FSDP, NeMo — the actual training stacks
- [[grpo]], [[ppo-for-llm]], [[rlhf-overview]] — RL training stacks that consume FSDP2 underneath
- [[das-spec-rl]], [[aurora]] — speculative decoding (inference-side parallelism, different from DP)
