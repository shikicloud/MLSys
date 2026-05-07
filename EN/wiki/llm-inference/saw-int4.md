---
title: "SAW-INT4: System-Aware 4-Bit KV-Cache Quantization (Block-Diagonal Rotation)"
category: llm-inference
tags: [saw-int4, kv-cache, quantization, int4, hadamard-rotation, bdr, sglang, paper-review]
created: 2026-05-06
updated: 2026-05-07
status: mature
paper: arXiv:2604.19157
code: https://github.com/togethercomputer/saw-int4
---

# SAW-INT4: System-Aware 4-Bit KV-Cache Quantization

> [!info] Paper metadata
> - **Paper**: [arXiv:2604.19157](https://arxiv.org/abs/2604.19157) — Together AI et al.
> - **Code**: [togethercomputer/saw-int4](https://github.com/togethercomputer/saw-int4) (branch `main`)
> - **SGLang fork**: [jindajia/sglang-fork @ colm_rotation_fast](https://github.com/jindajia/sglang-fork) (commit `0fcc241`)
> - **Authors**: Jinda Jia, Jisen Li, Zhongzhu Zhou, Jung Hwan Heo, Jue Wang, Tri Dao, Shuaiwen Leon Song, Ben Athiwaratkun, Chenfeng Xu, Tianyi Zhang, Xiaoxia Wu

> [!abstract]+ TL;DR
> INT4 [[kv-cache-optimization|KV-cache]] quantization quadruples capacity vs. BF16 but breaks reasoning models — Qwen3-4B-Thinking-2507 collapses from **66.67 % → 0 %** on GPQA under naive INT4. SAW-INT4 fixes it by applying a **block-diagonal Hadamard rotation (BDR)** to K (and optionally V) along the head dimension before per-token INT4 quantization, fused into one Triton kernel that writes the paged KV layout. Q is rotated by the same matrix inside the decode kernel so attention math is unchanged. Result on the same model: **65.82 % GPQA**, with **end-to-end throughput indistinguishable from plain INT4** across concurrencies 32–256 on H100.

---

## Background: why INT4 KV breaks reasoning models

A KV cache stores key/value tensors per token, per layer, per head. For long contexts and high concurrency it dominates GPU memory (see [[kv-cache-optimization]]); compressing it to INT4 cuts memory by 4× vs. BF16 and is the headline knob for serving long-context [[long-context-serving|workloads]]. But there's a known failure mode: K and V tensors have **outlier channels** — a small number of dimensions with magnitudes one or two orders larger than the rest. A few well-studied causes:

1. **RoPE-aligned channels.** Rotary positional embeddings concentrate energy in specific dimension pairs, which then carry consistently larger values across all tokens.
2. **"Massive activations" tokens.** A handful of tokens (often early system-prompt tokens or `<bos>`-like markers) carry orders-of-magnitude larger activations than the rest, and those activations show up in K/V.
3. **Channel-specialized heads.** Trained attention heads develop preferred channels — features used for retrieval, induction, or copying — that are systematically wider in distribution.

Per-token scale-and-zero INT4 quantization computes one `(scale, zero)` pair per token vector and divides the 16 INT4 levels evenly across $[\min, \max]$. When two or three channels carry 90 % of the magnitude, those channels get most of the resolution and the bulk of the head dimension is quantized to zero or near-zero. On surface tasks the rounding survives — the model still produces fluent text. On multi-step reasoning the per-attention error compounds across hundreds of attention rounds, the model loses the ability to track distinctions, and accuracy collapses.

The paper's GPQA numbers on `Qwen/Qwen3-4B-Thinking-2507` make this concrete:

| KV dtype                          | GPQA score   |
| --------------------------------- | ------------ |
| BF16 (baseline)                   | **66.67 %**  |
| Plain INT4                        | **0.00 %**   |
| INT4 + BDR (K-only, order=128)    | **65.82 %**  |

> [!important] Plain INT4 doesn't degrade the model — it disables it.
> The drop is from 66.67 % to 0 %, not a graceful slide. That cliff is the problem SAW-INT4 attacks.

> [!question]+ Shiki — What is an outlier channel and why does plain INT4 KV collapse? (2026-05-07)
>
> *(Quoted)*: "Per-token scale-and-zero INT4 quantization computes one `(scale, zero)` pair per token vector and divides the 16 INT4 levels evenly across `[min, max]`. When two or three channels carry 90 % of the magnitude, those channels eat most of the resolution and the rest of the head dim is quantized to zero or near-zero. On surface tasks the rounding survives — the model still produces fluent text. On multi-step reasoning the per-attention error compounds across hundreds of attention rounds, the model loses the ability to track distinctions, and accuracy collapses." What does this paragraph mean?
>
> An **outlier channel** is one of the `head_dim` dimensions in a K (or V) token vector whose magnitude is systematically 1–2 orders larger than the rest, *across all tokens*. With `head_dim = 128`, you might have 2–3 channels at magnitude ~1.0 while the other 125 sit at ~0.01. The three usual sources are: RoPE pairs (rotary positional embeddings concentrate energy in specific dimension pairs), "massive activation" tokens (a few tokens — typically system-prompt openers or `<bos>` markers — carry orders-of-magnitude larger activations whole-row), and trained head specialization (some channels become preferred for retrieval, induction, or copying).
>
> Per-token scale-and-zero quantization compresses the whole row of `head_dim` values to INT4 with a single `(scale, zero)` pair: $\text{scale} = (\max(x) - \min(x)) / 15$ and $\text{zero} = -\min(x) / \text{scale}$. The arithmetic comes from the row's max and min, but if 2–3 outlier channels carry 90 % of the magnitude the row's max and min are *almost entirely determined by them*. `scale` gets stretched to about 0.1 magnitude (to fit those big values), while the 125 ordinary channels live at ±0.01.
>
> Divided by a `scale` that's ten times larger than they need, the ordinary channels all round to the same level — usually `zero` itself — i.e. they're effectively quantized to 0. So **>95 % of channels lose almost all their information**; only the few outlier channels keep usable resolution. That's exactly what "the rest of the head dim is quantized to 0 or near-zero" means.
>
> A single attention $\text{softmax}(QK^\top / \sqrt{d}) \cdot V$ is dominated by the outlier channels (they're biggest), so the attention distribution stays roughly correct and the model still produces fluent text on surface tasks — LLM training redundancy covers the blur. But GPQA-style multi-step reasoning needs the model to discriminate fine differences across dozens or hundreds of attention rounds. The per-attention error compounds independently round after round; eventually the right-vs-wrong signal is buried under noise. That compounding is why Qwen3-4B-Thinking's GPQA drops from 66.67 % to 0 % in one step rather than degrading gradually.
>
> What [[#The key idea: block-diagonal Hadamard rotation before INT4|BDR]] does is apply a Hadamard rotation per head_dim block *before* quantization, redistributing each block's outlier energy uniformly across the block's dimensions. After the rotation the row's max/min is no longer dominated by a few channels, `scale` shrinks to a reasonable value, every dimension gets several INT4 levels of resolution, and the per-attention error stops compounding catastrophically — recovering 65.82 % of the 66.67 % BF16 baseline.

### Why rotations fix this — and what's already been tried

Multiplying a vector by an orthonormal matrix doesn't change its $L_2$ norm but redistributes its energy across all dimensions. If outliers live in a few coordinates and a rotation mixes those coordinates with the rest, the post-rotation tensor has a more uniform distribution and quantization becomes much easier. The Hadamard matrix is the natural choice: it's orthogonal, all entries are $\pm 1$, and the matrix-vector product runs in $O(d \log d)$ via the Fast Walsh-Hadamard Transform.

This idea is not new in 2026:

- **QuaRot** (Ashkboos et al., NeurIPS 2024) introduced random Hadamard rotation for **weight + activation** quantization in LLaMA, achieving INT4 weight + INT4 activation with little quality loss.
- **SpinQuant** (Liu et al., 2024) replaced random rotations with *learned* rotation matrices, getting better quantization at cost of an offline calibration step.
- **HALO**, **DuQuant**, and others extended these ideas to specific places in the model.

SAW-INT4's contribution is to apply this lineage to **KV cache** specifically (not weights, not activations) under **production serving constraints** ([[paged-attention|paged]] memory, fused [[continuous-batching|continuous batching]], no offline calibration). The block-diagonal restriction (rotate inside fixed-size head-dim blocks rather than the full head dim) is what makes it kernel-friendly; the fused Triton implementation is what makes it free at runtime. Tri Dao being on this paper is consistent with the framing — this is FlashAttention-pedigree systems work, not algorithmic novelty for its own sake.

For the broader family overview — QuIP / QuIP# / QuaRot / SpinQuant / BDR — see [[rotation-based-quantization]].

---

## The key idea: block-diagonal Hadamard rotation before INT4

> [!quote] The contribution in one sentence
> Apply a Hadamard rotation to the KV tensors *along the head dimension*, in fixed-size blocks (e.g. 16 or 128), before per-token INT4 quantization. The rotation redistributes outlier energy across the block. Q is rotated by the same matrix at decode time so the attention math is unchanged.

The "block-diagonal" qualifier is doing real work. A full Hadamard over the entire head dimension would be $O(d^2)$ memory traffic to apply (or $O(d \log d)$ with FWHT but with worse cache behaviour on small head dims) and incompatible with paged KV layouts that group by head. Splitting `head_dim` into blocks of size $H$ (where $H \mid \text{head\_dim}$) and rotating each block independently has three properties:

1. **Cost.** Rotation is $O(d \log H)$ instead of $O(d \log d)$ — for $d = 128$, $H = 16$ this is 4 butterfly stages instead of 7.
2. **Cache locality.** Each block is small enough to live in shared memory / registers throughout the FWHT.
3. **Channel-aligned outliers stay localized.** Because outlier channels are at fixed positions (e.g., RoPE pairs), and because each Hadamard block is a contiguous slice of the head dim, the outlier energy is mixed *within* its block. That's exactly enough — per-token quantization operates per token across the full head dim, but the per-block Hadamard already flattens the variance enough for a single per-token `(scale, zero)` pair to cover the whole row.

For Qwen3 (`head_dim = 128`), valid orders are powers of two dividing 128: 16, 32, 64, 128. The README's primary results use **`HADAMARD_ORDER = 128`**, while the env-var docs example uses **`16`**. Both are validated to work; the trade-off is rotation strength vs. kernel-size cost.

Three sub-claims hold the contribution up:

- **Token-wise INT4 is enough** *if* you fix the input distribution. The paper compares against k-means clustering, vector quantization, and Hessian-aware methods, and shows BDR matches or exceeds them under real serving constraints.
- **The kernel must be fused.** A split implementation (rotate → write) costs an extra global memory pass; the BDR contribution is rotation, normalization ($\div \sqrt{H}$), per-token min/max/scale/zero compute, and INT4 pack-and-write **in one Triton kernel** that targets the paged buffer directly.
- **K alone is usually enough.** The default mode rotates K only; rotating V (`ROTATE_V=1`) requires an inverse rotation in the attention output, costs extra memory traffic, and the paper claims marginal accuracy gain.

> [!tip] Recommended primary mode
> `HADAMARD=1`, `HADAMARD_ORDER=128`, `ROTATE_V=0`, `--kv-cache-dtype int4`. K-only is the default for a reason — V rotation costs extra wall-time for marginal accuracy.

### A two-line refresher on Hadamard rotation

The Sylvester-Hadamard matrix is defined recursively:

$$
H_2 = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}, \qquad H_{2n} = H_2 \otimes H_n
$$

$H_d$ is orthogonal: $H_d^\top H_d = I$. The $1/\sqrt{d}$ normalization is what makes it an isometry — without it, applying $H$ scales every component by $\sqrt{d}$. The Fast Walsh-Hadamard Transform (FWHT) computes $H_d \cdot x$ in $O(d \log d)$ using $\log_2 d$ butterfly stages: at stage $s$, every element pairs with its partner at distance $2^s$, and the pair becomes $(a + b,\ a - b)$. The kernel does exactly this — see the `_fwht_blocked_segments_tensor` walkthrough below.

---

## How it works

### Where BDR sits in the inference pipeline

```
                      ┌─ standard SGLang INT4 path ─────────────────────┐
prefill / decode ────►│ compute K,V (BF16)                              │
                      │ quantized_set_kv_int4_triton(...)               │
                      │   = per-token scale/zero + INT4 pack + paged    │
                      │     write into k_buffer / v_buffer              │
                      └────────────────────────────────────────────────┘

                      ┌─ BDR path (HADAMARD=1) ─────────────────────────┐
prefill / decode ────►│ compute K,V (BF16)                              │
                      │ quantized_set_kv_int4_hadamard_fused_triton(...)│
                      │   one kernel does, per (token, head):           │
                      │     1. Load BF16 row → cast FP32 → × 1/√H        │
                      │     2. In-register block-FWHT (LOG stages)       │
                      │     3. Per-token min/max → scale, zero           │
                      │     4. Round + pack two 4-bit values per byte    │
                      │     5. Store into paged k_buffer + scales_zeros  │
                      │                                                 │
                      │ At decode the Triton GQA kernel applies the     │
                      │ same in-register Hadamard to Q (fuse_q_hadamard │
                      │ flag → triton_backend.py)                       │
                      └────────────────────────────────────────────────┘
```

### Configuration interface — 4 env vars + 1 CLI flag

The whole switchable behaviour is exposed as environment variables read once at server start (`memory_pool.py`). The simplicity is intentional — no model surgery, no calibration step for the primary mode.

```python
# memory_pool.py
_hadamard_enabled  = 1 if os.environ.get("HADAMARD",  "0") in ("1","true","True") else 0
_rotate_v_enabled  = 1 if os.environ.get("ROTATE_V",  "0") in ("1","true","True") else 0
_hadamard_order    = int(os.environ.get("HADAMARD_ORDER", "16"))
_fuse_hadamard_int4_kv = os.environ.get(
    "SGLANG_FUSE_HADAMARD_INT4_KV", "1"
).lower() in ("1", "true", "yes")
```

The full mode matrix (from `docs/bdr_env_vars.md`):

| Mode                            | `HADAMARD` | `ROTATE_V` | `HADAMARD_ORDER`     | `--kv-cache-dtype` |
| ------------------------------- | ---------- | ---------- | -------------------- | ------------------ |
| BF16 KV (baseline)              | `0`        | `0`        | unset                | `auto`             |
| INT4 KV (no rotation)           | `0`        | `0`        | unset                | `int4`             |
| INT4 + BDR (K only, default)    | `1`        | `0`        | e.g. `16` or `128`   | `int4`             |
| INT4 + BDR (K + V)              | `1`        | `1`        | e.g. `16`            | `int4`             |

> [!note] Constraints on `HADAMARD_ORDER`
> Must be a power of two **and** divide `head_dim`. For Qwen3 (`head_dim = 128`), 16 / 32 / 64 / 128 all qualify. The fused kernel additionally caps the order at `MAX_HADAMARD_ORDER = 4096` to keep `tl.arange(0, order)` from blowing up Triton's compile time.

### The dispatch site (`set_kv_buffer`)

Three paths are available in `set_kv_buffer` of the INT4 KV pool. The fast path is the default; the slow path exists for debugging:

```python
# memory_pool.py — INT4 set_kv_buffer with BDR
if self.dtype == "int4":
    if _hadamard_enabled:
        hadamard_order = _hadamard_order
        assert cache_k.shape[-1] % hadamard_order == 0, \
            f"head_dim must be divisible by {hadamard_order}"

        if _fuse_hadamard_int4_kv:
            # FAST PATH: one Triton kernel does rotate + normalize + quantize + paged write.
            validate_hadamard_order_for_kv_fuse(hadamard_order, cache_k.shape[-1])
            quantized_set_kv_int4_hadamard_fused_triton(
                cache_k, cache_v, loc,
                self.k_buffer        [layer_id - self.start_layer],
                self.v_buffer        [layer_id - self.start_layer],
                self.k_scales_zeros  [layer_id - self.start_layer],
                self.v_scales_zeros  [layer_id - self.start_layer],
                hadamard_order,
                rotate_v=bool(_rotate_v_enabled),
            )
            return

        # SLOW PATH: split rotate → quantize via fast_hadamard_transform.
        # Reshape last dim into (n_blocks, block).
        orig_shape = cache_k.shape                               # (..., head_dim)
        cache_k = cache_k.view(*orig_shape[:-1],
                               orig_shape[-1] // hadamard_order,
                               hadamard_order)                   # (..., n_blocks, block)
        cache_k = hadamard_transform(cache_k / math.sqrt(hadamard_order))
        cache_k = cache_k.view(orig_shape)
        if _rotate_v_enabled:
            # same block reshape + transform for V
            ...

    # Common: per-token scale/zero + INT4 pack + paged write.
    quantized_set_kv_int4_triton(
        cache_k, cache_v, loc,
        self.k_buffer       [layer_id - self.start_layer],
        self.v_buffer       [layer_id - self.start_layer],
        self.k_scales_zeros [layer_id - self.start_layer],
        self.v_scales_zeros [layer_id - self.start_layer],
    )
    return
```

Three things to pull out:

- **The reshape is the rotation's domain definition.** `(..., head_dim) → (..., head_dim // H, H)` says "treat the head dim as `head_dim/H` independent groups of size $H$". The Hadamard transform mixes within each group only; nothing crosses block boundaries. Outliers in one block don't leak into the next, which is exactly what makes paged KV layouts compatible.
- **The $1/\sqrt{H}$ normalization** keeps the Hadamard transform an *isometry* (preserves the $L_2$ norm). Without it, BF16 magnitudes would shift and the per-token scale calibration would be off. The slow path is explicit about this; the fused kernel has it baked into `PRE_SCALE`.
- **`scales_zeros` is a separate buffer.** The paged KV layout stores the INT4-packed bytes in `k_buffer` / `v_buffer` and the `(scale, zero)` pair *per (token, head)* in `k_scales_zeros` / `v_scales_zeros`. The kernel writes both buffers atomically — there's no stale-pair race.

### Inside the fused Triton kernel

The fused kernel is in `python/sglang/QuantKernel/fused_hadamard_int4_kv.py`. Three logical pieces matter: the FWHT butterfly, the per-token min/max + INT4 pack, and the launch grid that processes multiple heads per program.

#### The butterfly

Each program loads the whole padded head-dim row into a register vector and runs `LOG = log₂(hadamard_order)` Sylvester FWHT stages in place:

```python
@triton.jit
def _fwht_blocked_segments_tensor(x, head_dim_: tl.constexpr, LOG: tl.constexpr):
    """FWHT on each contiguous block of size 2**LOG tiling head_dim_ (vectorized).

    Uses 1 gather per butterfly stage (down from 4): fetch partner via
    x[i ^ stride] (compile-time permutation), then select add vs subtract by
    testing bit s of i.
    """
    i = tl.arange(0, head_dim_)
    for s in tl.static_range(0, LOG):
        stride  = 1 << s
        partner = i ^ stride                # compile-time index permutation
        x_p     = tl.gather(x, partner, 0)  # 1 gather (was 4)
        is_lo   = ((i >> s) & 1) == 0       # compile-time mask
        x       = tl.where(is_lo, x + x_p, x_p - x)
    return x
```

What's happening: at stage $s$, element $i$ pairs with element $i \oplus 2^s$. Lower elements of each pair become $a + b$, upper elements become $b - a$. Because `i` and `stride` are both compile-time `tl.arange` constants, the partner permutation and the lo/hi mask are entirely resolved at JIT time — no runtime address arithmetic. The author comments this brings the butterfly down from 4 gathers per stage to 1, which on Hopper is the difference between memory-bound and ALU-bound for small `head_dim`.

> [!note] Block-diagonal Hadamard for free
> The "blocked segments" name comes from the fact that this single 1-D butterfly *implicitly* does block-diagonal Hadamard whenever $\text{LOG} < \log_2(\text{head\_dim\_pad\_})$ — the $\log_2(\text{order})$ stages only mix within power-of-two-aligned blocks of size $2^{\text{LOG}}$. So `HADAMARD_ORDER = 128` runs 7 stages on a 128-element row; `HADAMARD_ORDER = 16` runs 4 stages and leaves the higher-order block boundaries unmixed. Block-diagonal achieved by stopping the butterfly early.

#### The full kernel body (per token, per head group)

```python
@triton.autotune(configs=autotune_cfgs, key=["head_dim_"])
@triton.jit
def _fused_hadamard_int4_set_kv_kernel(
    input_ptr, loc_ptr, cache_ptr, scales_zeros_ptr,
    num_tokens, num_heads,
    head_dim_:    tl.constexpr,   # true (unpadded) head dim
    head_dim_pad_: tl.constexpr,  # next power-of-2 ≥ head_dim_
    input_stride_token, input_stride_head, input_stride_dim,
    cache_stride_loc,   cache_stride_head, cache_stride_dim,
    sz_stride_loc,      sz_stride_head,    sz_stride_dim,
    LOG:        tl.constexpr,
    PRE_SCALE:  tl.constexpr,
    BLOCK_HALF: tl.constexpr,
    HEADS_PER_PROGRAM: tl.constexpr,
):
    token_idx  = tl.program_id(0)
    head_group = tl.program_id(1)
    if token_idx >= num_tokens: return
    cache_loc = tl.load(loc_ptr + token_idx)

    for hh in tl.static_range(0, HEADS_PER_PROGRAM):
        head_idx = head_group * HEADS_PER_PROGRAM + hh
        if head_idx < num_heads:
            # 1. Load BF16 row into power-of-2 register buffer; cast FP32; pre-scale 1/√H.
            dim_full = tl.arange(0, head_dim_pad_)
            input_off = token_idx * input_stride_token + head_idx * input_stride_head
            x = tl.load(
                input_ptr + input_off + dim_full * input_stride_dim,
                mask=dim_full < head_dim_, other=0.0,
            ).to(tl.float32) * PRE_SCALE                   # PRE_SCALE = 1 / sqrt(H)

            # 2. In-register block-FWHT.
            x = _fwht_blocked_segments_tensor(x, head_dim_pad_, LOG)

            # 3. Round-trip through BF16 (matches CUDA path numerics).
            half_dim = head_dim_ // 2
            dim_off  = tl.arange(0, BLOCK_HALF)
            dim_mask = dim_off < half_dim
            vals1 = tl.where(dim_mask, tl.gather(x, dim_off,            0), 0.0).to(tl.bfloat16).to(tl.float32)
            vals2 = tl.where(dim_mask, tl.gather(x, dim_off + half_dim, 0), 0.0).to(tl.bfloat16).to(tl.float32)

            # 4. Per-token min/max → INT4 scale/zero (range = 15 levels = 0..15).
            val_min   = tl.minimum(tl.min(vals1, 0), tl.min(vals2, 0))
            val_max   = tl.maximum(tl.max(vals1, 0), tl.max(vals2, 0))
            val_range = tl.maximum(val_max - val_min, 1e-8)
            scale     = val_range / 15.0
            zero      = -val_min / scale

            # 5. Round, clip to uint8 (already 0..15), pack two 4-bit values per byte.
            q1 = (vals1 / scale + zero + 0.5).to(tl.uint8)
            q2 = (vals2 / scale + zero + 0.5).to(tl.uint8)
            packed = q1 | (q2 << 4)

            # 6. Store packed bytes into paged KV buffer; store (scale, zero) sidecar.
            cache_off = (cache_loc * cache_stride_loc
                         + head_idx * cache_stride_head
                         + dim_off  * cache_stride_dim)
            tl.store(cache_ptr + cache_off, packed, mask=dim_mask)
            sz_base = cache_loc * sz_stride_loc + head_idx * sz_stride_head
            tl.store(scales_zeros_ptr + sz_base + 0 * sz_stride_dim, scale)
            tl.store(scales_zeros_ptr + sz_base + 1 * sz_stride_dim, zero)
```

A few details worth highlighting:

- **`head_dim_pad_` handles non-power-of-2 head dims** (e.g. 320, 576, 768) by padding with zeros; because `hadamard_order | head_dim`, the padded portion always falls on a block boundary and never contaminates real blocks.
- **The BF16 round-trip** before quantization (`.to(tl.bfloat16).to(tl.float32)`) matches the numerics of the slow $(\text{bf16\_tensor} / \sqrt{\text{order}})$ path — which is why the kernel docstring warns: *"that scaling differs slightly from `(bf16_tensor / sqrt(order))` before CUDA Hadamard, so packed bytes can differ in rare cases from the unfused path."* In practice the GPQA results match.
- **The pack `q1 | (q2 << 4)`** is the standard INT4 layout — element $2i$ lives in the low nibble of byte $i$, element $2i + 1$ in the high nibble. The `BLOCK_HALF = head_dim/2` constant is the number of bytes per row.
- **`HEADS_PER_PROGRAM`** lets one program process multiple heads sequentially, cutting grid size and amortizing the `loc_ptr` load. Default is `min(8, num_heads)` for small head dims, dropped to `1` once `next_power_of_2(head_dim) ≥ 512` (huge per-program register pressure makes multiple heads per program counter-productive). The autotune table also shrinks at large head dims to avoid multi-minute compile times.

#### The launcher

```python
def quantized_set_kv_int4_hadamard_fused_triton(
    cache_k, cache_v, loc,
    k_cache_buffer, v_cache_buffer,
    k_scales_zeros_buffer, v_scales_zeros_buffer,
    hadamard_order: int,
    work_k=None, work_v=None,                  # legacy; ignored
    rotate_v: bool = True,
    heads_per_program: Optional[int] = None,
) -> None:
    num_tokens, num_heads, head_dim = cache_k.shape
    assert cache_v.shape == cache_k.shape
    assert head_dim % 2 == 0
    _validate_hadamard_order_impl(hadamard_order, head_dim)

    hpp = (heads_per_program
           if heads_per_program is not None
           else _fused_default_heads_per_program(head_dim, num_heads))
    hpp = min(max(1, hpp), num_heads)

    kernel, cfg = _get_kernel(head_dim, hadamard_order)   # JIT cache keyed on (head_dim, order, rev)
    fused_grid = (num_tokens, triton.cdiv(num_heads, hpp))

    def _launch(inp, cache_buf, sz_buf):
        kernel[fused_grid](
            inp, loc, cache_buf, sz_buf,
            num_tokens, num_heads,
            cfg["head_dim_"], cfg["head_dim_pad_"],
            inp.stride(0),       inp.stride(1),       inp.stride(2),
            cache_buf.stride(0), cache_buf.stride(1), cache_buf.stride(2),
            sz_buf.stride(0),    sz_buf.stride(1),    sz_buf.stride(2),
            LOG=cfg["LOG"], PRE_SCALE=cfg["PRE_SCALE"],
            BLOCK_HALF=cfg["BLOCK_HALF"], HEADS_PER_PROGRAM=hpp,
        )

    _launch(cache_k, k_cache_buffer, k_scales_zeros_buffer)

    if rotate_v:
        _launch(cache_v, v_cache_buffer, v_scales_zeros_buffer)
    else:
        # ROTATE_V=0: V goes through the plain INT4 kernel — same tiling, no rotation.
        _quantized_set_kv_int4_kernel[(num_tokens, triton.cdiv(num_heads, hpp))](
            cache_v, loc, v_cache_buffer, v_scales_zeros_buffer,
            num_tokens, num_heads, head_dim,
            cache_v.stride(0), cache_v.stride(1), cache_v.stride(2),
            v_cache_buffer.stride(0), v_cache_buffer.stride(1), v_cache_buffer.stride(2),
            v_scales_zeros_buffer.stride(0), v_scales_zeros_buffer.stride(1), v_scales_zeros_buffer.stride(2),
            BLOCK_SIZE_DIM=triton.next_power_of_2(head_dim // 2),
            HEADS_PER_PROGRAM=hpp, num_warps=1, num_stages=1,
        )
```

The clever bit: when `rotate_v=False`, V is *not* skipped — it's quantized into the same paged buffer using the existing `_quantized_set_kv_int4_kernel`, with the same `HEADS_PER_PROGRAM` tiling so the V launch perfectly matches the K launch's grid. The two launches share `loc` and the per-token scale-zero layout, so KV remain consistent.

### Q-correction at decode

A Hadamard rotation on K only preserves the attention dot product $Q \cdot K^\top$ if Q is rotated by the same matrix. The fork applies this in the GQA decode kernel itself, gated by the same `SGLANG_FUSE_HADAMARD_INT4_KV` env (`triton_backend.py:1042-1058`):

```python
# triton_backend.py — decode path
if hasattr(kv_pool, "dtype") and kv_pool.dtype in ("int4", "int8"):
    fuse_q_hadamard_in_kernel = (
        kv_pool.dtype == "int4"
        and _hadamard_enabled
        and _fuse_hadamard_int4_kv
    )
    if kv_pool.dtype == "int4" and _hadamard_enabled and not fuse_q_hadamard_in_kernel:
        # SLOW PATH: explicit Q rotation before the decode kernel call.
        q = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
        orig_shape = q.shape
        q = q.view(*orig_shape[:-1], orig_shape[-1] // _hadamard_order, _hadamard_order)
        q = hadamard_transform(q / math.sqrt(_hadamard_order))
        q = q.view(orig_shape)

    self.decode_attention_fwd_quantized(
        q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
        kv_pool.get_raw_key_buffer(layer.layer_id),
        kv_pool.get_raw_value_buffer(layer.layer_id),
        kv_pool.get_key_scales_zeros(layer.layer_id),
        kv_pool.get_value_scales_zeros(layer.layer_id),
        o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
        ...,
        kv_pool.dtype,
        fuse_q_hadamard=fuse_q_hadamard_in_kernel,
        hadamard_order=_hadamard_order,
    )

    if kv_pool.dtype == "int4" and _hadamard_enabled and _rotate_v_enabled:
        # ROTATE_V=1: undo the V rotation on the attention output.
        orig_shape = o.shape
        o = o.view(*orig_shape[:-1], orig_shape[-1] // _hadamard_order, _hadamard_order)
        o = hadamard_transform(o / math.sqrt(_hadamard_order))
        o = o.view(orig_shape)
```

Three pieces stitch this together:

1. **In-kernel Q rotation** is the default — `decode_attention_fwd_quantized` accepts `fuse_q_hadamard=True` and `hadamard_order` and applies the same FWHT inside the GQA dot-product before dequantizing K. No extra global memory pass.
2. **Out-of-kernel slow path** rotates Q with `fast_hadamard_transform` before the call. Used when `SGLANG_FUSE_HADAMARD_INT4_KV=0`. Slower because Q is touched twice.
3. **Output un-rotation when `ROTATE_V=1`.** If V was rotated at write time, the attention output $o = \text{softmax}(QK^\top) \cdot V$ carries the V rotation; an inverse Hadamard restores it. This is done outside the decode kernel as a separate pass — which is one reason `ROTATE_V=1` costs more wall time even when accuracy gains are marginal.

### Per-token INT4 quantization formulas

The kernel uses standard asymmetric-zero-point unsigned 4-bit quantization. Per token, per head:

$$
\begin{aligned}
\text{range} &= \max(x) - \min(x) \quad \text{over the head\_dim row} \\
\text{scale} &= \text{range} / 15 \quad \text{(4-bit unsigned has 16 levels, 15 intervals)} \\
\text{zero}  &= -\min(x) / \text{scale} \quad \text{(int level that maps back to 0)} \\
q            &= \mathrm{round}(x / \text{scale} + \text{zero}) \quad \text{element-wise, clipped to } [0, 15]
\end{aligned}
$$

Dequantization (inside the decode kernel):

$$
x_{\text{dequant}} = (q - \text{zero}) \cdot \text{scale} = q \cdot \text{scale} - \min(x)
$$

Storage per `(token, head)`:
- `head_dim / 2` bytes of packed nibbles (one 4-bit element per nibble),
- 2 floats for `(scale, zero)`.

> [!example] Memory math for Qwen3
> With `head_dim = 128` and 4 KV heads (GQA), one token costs $4 \times (64 + 8) = 288$ bytes for KV cache after BDR + INT4 — vs. $4 \times 128 \times 2 = 1024$ bytes for BF16, a **3.55× reduction** end-to-end. Not quite 4× because of the scales/zeros sidecar.

### Running it

The user-facing surface is a single env-var flip on top of an SGLang launch:

```bash
# BF16 baseline
python -m sglang.launch_server \
  --prefill-attention-backend fa3 --decode-attention-backend triton \
  --model-path "Qwen/Qwen3-4B-Thinking-2507" --port 30000 \
  --kv-cache-dtype auto

# Plain INT4 KV (the model collapses on reasoning tasks)
python -m sglang.launch_server ... --kv-cache-dtype int4

# INT4 + BDR (K-only, block size 128) — the recommended primary mode
HADAMARD=1 HADAMARD_ORDER=128 \
python -m sglang.launch_server ... --kv-cache-dtype int4

# INT4 + BDR (K + V, block size 16)
HADAMARD=1 ROTATE_V=1 HADAMARD_ORDER=16 \
python -m sglang.launch_server ... --kv-cache-dtype int4

# Slow reference path (fast-hadamard-transform CUDA + plain INT4 kernel) — for debugging
HADAMARD=1 HADAMARD_ORDER=128 SGLANG_FUSE_HADAMARD_INT4_KV=0 \
python -m sglang.launch_server ... --kv-cache-dtype int4
```

A short OpenAI-client smoke test (`scripts/bdr_smoke_test.py`) sends a GPQA question to verify the install:

```python
from openai import OpenAI
client = OpenAI(api_key="EMPTY", base_url=f"http://0.0.0.0:{port}/v1")
response = client.chat.completions.create(
    model="Qwen/Qwen3-4B-Thinking-2507",
    messages=[{"role": "user", "content": GPQA_SAMPLE}],
    temperature=0.6, top_p=0.95, max_tokens=32768, stream=True,
)
```

A coherent answer to the GPQA chemistry problem (about TLC polarities) confirms BDR is wired through.

### K-means ablation pipeline

A separate sub-repo (`third_party/sglang-kmeans`, branch `jinda_kmeans_rotation_dump` of the same fork) implements an alternative quantizer: instead of scale-and-zero, cluster KV vectors into $N$ centroids per layer and store cluster indices. Calibration is offline:

```bash
# 1. Dump KV activations from a BF16 server.
DUMP_KVCACHE=true DUMP_KVCACHE_TOKENS=512 DUMP_KVCACHE_DIR=/path/to/dumps \
python -m sglang.launch_server ... --kv-cache-dtype auto

# 2. Fit per-layer centroids (tools/fit_kv_centroids.py).
python tools/fit_kv_centroids.py \
  --dump-dir /path/to/dumps \
  --out-dir  /path/to/centroids \
  --n-clusters 16 --seed 0

# 3. Serve INT4 + k-means.
N_CLUSTERS=16 SGLANG_KV_CENTROIDS_PATH=/path/to/centroids \
python -m sglang.launch_server ... --kv-cache-dtype int4
```

`fit_kv_centroids.py` is small and concrete:

```python
# Per layer:
blob = torch.load(f"kv_calibration_layer_{L}.pt")  # {'k': [T,H,D], 'v': [T,H,D]}
xk = blob["k"].reshape(T, H * D).float().numpy()    # flatten heads × dims
xv = blob["v"].reshape(T, H * D).float().numpy()
km_k = KMeans(n_clusters=16, n_init=10, max_iter=300).fit(xk)
km_v = KMeans(n_clusters=16, n_init=10, max_iter=300).fit(xv)
torch.save(km_k.cluster_centers_, f"k_layer_{L}_clusters_16_centers.pt")
torch.save(km_v.cluster_centers_, f"v_layer_{L}_clusters_16_centers.pt")
```

Whole script ≈100 lines. Optional rotation can stack on top of k-means via the same `HADAMARD` / `ROTATE_V` env vars — the README documents this matrix:

| Method            | `HADAMARD` | `ROTATE_V`     | `--kv-cache-dtype` | `SGLANG_KV_CENTROIDS_PATH` |
| ----------------- | ---------- | -------------- | ------------------ | -------------------------- |
| K-means + INT4    | `0`        | `0`            | `int4`             | required                   |
| K-means + BDR     | `1`        | `0` or `1`     | `int4`             | required                   |

> [!warning] Empty ablation table
> The README's ablation accuracy table for these methods is empty (placeholder rows), which is honest but means the published comparison rests on the GPQA-only primary result.

---

## Experiments

**Hardware.** 1× H100 80 GB, TP = 1.

### Accuracy

Qwen3-4B-Thinking-2507, GPQA, `temp=0.6`, `top_p=0.95`, 3 repeats, 32 K context:

| Config                              | GPQA       |
| ----------------------------------- | ---------- |
| BF16 KV                             | 66.67 %    |
| INT4 KV                             | 0 %        |
| **INT4 + BDR (K-only, ord=128)**    | **65.82 %** |

### Throughput

Qwen3-8B, GenAI-Bench, traffic `D(256, 1024)` short and `D(16384, 1024)` long.

**Short context** (256 input / 1024 output), concurrency sweep — job-level `output_tps` (tokens/s aggregated across requests) and TTFT in ms:

| Concurrency | BF16            | INT4            | INT4 + BDR              |
| ----------: | --------------: | --------------: | ----------------------: |
|  32         | 3,795 / 196     | 3,687 / 225     | 3,689 / 226             |
|  64         | 5,950 / 369     | 6,371 / 370     | 6,235 / 377             |
| 128         | 8,410 / 657     | 9,544 / 665     | 9,350 / 655             |
| 256         | 11,195 / 1,224  | 11,624 / 1,237  | **11,732 / 1,148**      |

**Long context** (16,384 input / 1,024 output), concurrency sweep:

| Concurrency | BF16             | INT4             | INT4 + BDR        |
| ----------: | ---------------: | ---------------: | ----------------: |
|   8         |   414 / 2,636    |   458 / 2,631    |   457 / 2,523     |
|  16         |   481 / 5,104    |   571 / 4,956    |   568 / 4,875     |
|  32         |   570 / 18,047   |   618 / 9,568    |   616 / 9,350     |
|  64         |   471 / 44,798   |   666 / 19,398   |   663 / **18,371** |
| 128         |   559 / 113,583  |   701 / 57,654   |   701 / **57,054** |

The pattern: BDR's throughput numbers are within noise of plain INT4, and at concurrency ≥ 256 / long context they actually edge ahead on `output_tps` and TTFT. At high concurrency BF16 falls behind dramatically (113 s TTFT at conc-128 long context vs. ~57 s for INT4 / BDR) because its 4× larger KV cache pushes the system into memory pressure.

> [!note] Why BDR sometimes *beats* plain INT4 on TTFT
> The BDR kernel touches `cache_k` and writes the scales/zeros buffer in the same memory pass that plain INT4 would have done, so the *added* rotation cost is amortized into work that already had to happen. Kernel fusion turns a potential 5–10 % overhead into a free lunch — exactly the framing the paper wants.

The accuracy story is the headline: BDR is the difference between a usable reasoning model and a broken one, and it costs essentially nothing at runtime.

---

## Strengths and limitations

The two strongest points: (1) the technique is **minimally invasive** — four env vars and a forked kernel, no model retraining or calibration; (2) the **fused kernel** turns what could have been a 5–10 % overhead into measurement noise, because rotation, normalization, and quantization share the same memory pass that INT4 already needed. The system-aware framing in the title is earned.

Where the work is honest about its scope but the limits matter:

- **MHA only.** The README explicitly disallows MLA. DeepSeek-V3-style architectures (where MLA cuts KV cache by another huge factor by storing a low-rank projection of K) can't use BDR as written — the rotation would have to interact with the up-projection from the compressed representation. Whether the same idea translates is an open question the paper does not take on.
- **Backend constraints.** Decode uses Triton GQA; prefill uses FA3. Switching attention backends elsewhere in SGLang or porting to vLLM is non-trivial — the Q-correction has to land inside whichever decode kernel you use. The slow path (`SGLANG_FUSE_HADAMARD_INT4_KV=0`) gives portability at a cost.
- **One head_dim block size, one model family.** Primary numbers are Qwen3-4B-Thinking-2507 (accuracy) and Qwen3-8B (throughput). The repo notes `HADAMARD_ORDER=128` as the primary BDR result but the env-var docs example uses `16`. The paper does not appear to systematically sweep block size against accuracy across multiple model families.
- **Single benchmark for accuracy.** GPQA is the *only* accuracy result in the README. GPQA stresses long-form scientific reasoning, which is where INT4 fails hardest, so it's a fair stress test — but a single benchmark for a quantization paper is thin. MMLU, MATH, HumanEval, and longer-context retrieval suites (e.g., RULER) are all reasonable next steps.
- **Ablation table is empty.** The k-means ablation matrix in the README has placeholder cells. Whether BDR strictly dominates k-means or only matches it isn't shown — the paper claims more sophisticated methods give "marginal gains" but the table to back this up isn't published in the repo.
- **No comparison to alternatives in published systems.** [[quantization|KIVI]], NVFP4, FP8 KV, ShadowKV, KVTC — all live in the same problem space (see [[kv-cache-optimization]]). Comparing throughput and accuracy across these is the obvious follow-up; the paper restricts itself to BF16 vs. plain INT4 vs. BDR.
- **Random Hadamard, not learned.** SpinQuant showed that learned rotations beat random Hadamard for weight + activation quantization. The paper doesn't try a learned per-layer rotation matrix, which would be an obvious accuracy lever at the cost of an offline calibration step.
- **Python-level Hadamard fallback exists for debugging.** When `SGLANG_FUSE_HADAMARD_INT4_KV=0`, BDR uses `Dao-AILab/fast-hadamard-transform` then writes; this path is slower because it adds a global memory round-trip. The default is fast, but anyone porting the idea to a serving stack without a Triton-friendly attention backend will hit this overhead.

> [!bug] Documentation port mismatch
> `scripts/bdr_smoke_test.py` defaults to `--port 30000` (matching the launch examples), but the README's smoke-test snippet shows `--port 30001` for no apparent reason. Trivial to fix, but a sign the OSS release was rushed.

---

## What this means

The bigger lesson here is the same as in [[paged-attention|PagedAttention]] or [[sglang|RadixAttention]]: **the right granularity for inference optimization is the kernel, not the model.** SAW-INT4 doesn't propose a new quantization scheme so much as it proves that *plain* per-token INT4 is fine if you do one thing right at the kernel level — rotate before you quantize, in blocks the paged layout can swallow. That has two implications I'd watch:

1. **Block-diagonal Hadamard is going to spread.** It's small enough to be a free addition to any INT4 KV path, the math is well-known (FWHT has been textbook material since the 1970s), and the kernel work is mostly done in the SGLang fork. Expect [[vllm|vLLM]] and [[tensorrt-llm|TensorRT-LLM]] to pick this up; expect the same trick to be tried on weight quantization (where QuaRot already shows it helps) and activation quantization for FP8.
2. **The "system-aware" framing is the more durable contribution.** The paper's recurring point — that vector quantization and Hessian-aware methods give "marginal gains under real serving constraints" — is really an argument that algorithmic sophistication has run into a wall, and the remaining gains live in the kernels and memory layout. That argument is going to keep being right; expect more 2026 papers to be about *fusing* known techniques into the right kernel rather than inventing new techniques.

What this is *not*: a solution for [[long-context-serving|long-context serving]] of MLA-style models, an answer for non-NVIDIA hardware, or evidence that INT4 is "solved." It's a clean, narrow result on a real problem.

---

## Source code & reproduction

```bash
# Clone with submodules.
git clone --recurse-submodules https://github.com/togethercomputer/saw-int4.git
cd saw-int4

# Install primary BDR fork.
cd third_party/sglang-fast-rotation/python
pip install -e ".[all]"
pip install --no-build-isolation \
  "git+https://github.com/Dao-AILab/fast-hadamard-transform.git"

# Launch with BDR.
HADAMARD=1 HADAMARD_ORDER=128 \
python -m sglang.launch_server \
  --prefill-attention-backend fa3 \
  --decode-attention-backend triton \
  --model-path "Qwen/Qwen3-4B-Thinking-2507" \
  --port 30000 \
  --kv-cache-dtype int4

# Smoke test.
python scripts/bdr_smoke_test.py --port 30000 \
  --model Qwen/Qwen3-4B-Thinking-2507
```

Files worth reading next, with the role of each:

| File                                                                                       | Role                                                                                                                                                            |
| ------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `third_party/sglang-fast-rotation/python/sglang/srt/mem_cache/memory_pool.py`              | env-var gate (lines 92–98), `set_kv_buffer` BDR branch (lines 1136–1190), slow-path reference using `fast_hadamard_transform`.                                  |
| `third_party/sglang-fast-rotation/python/sglang/QuantKernel/fused_hadamard_int4_kv.py`     | fused kernel: `_fwht_blocked_segments_tensor` (the butterfly), `_fused_hadamard_int4_set_kv_kernel` (full per-token kernel), launcher, `validate_hadamard_order_for_kv_fuse`. |
| `third_party/sglang-fast-rotation/python/sglang/srt/layers/attention/triton_backend.py`    | decode-side Q rotation (lines 1042–1091); the `fuse_q_hadamard` flag passed into `decode_attention_fwd_quantized`.                                              |
| `third_party/sglang-fast-rotation/python/sglang/srt/layers/attention/triton_ops/decode_attention.py` | the GQA decode kernel that accepts `fuse_q_hadamard` and `hadamard_order` and applies the in-register Q FWHT.                                                   |
| `tools/fit_kv_centroids.py`                                                                | k-means centroid calibration (ablation only).                                                                                                                    |
| `docs/bdr_env_vars.md`                                                                     | env var reference and mode matrix.                                                                                                                               |
| `scripts/bdr_smoke_test.py`                                                                | minimal OpenAI-client GPQA verification.                                                                                                                         |
| `scripts/run_genai_bench_example.sh`                                                       | throughput sweep helper.                                                                                                                                         |
| `scripts/run_primary_eval_matrix.sh`                                                       | primary accuracy/speed sweep helper.                                                                                                                             |

---

## Related reading

- [[kv-cache-optimization]] — KV cache compression landscape (paging, quantization, eviction, offloading).
- [[quantization]] — weight/activation quantization (GPTQ, AWQ, SmoothQuant, FP8, NVFP4) — orthogonal to KV quantization, but Hadamard rotation has the same role across them (cf. QuaRot, SpinQuant).
- [[rotation-based-quantization]] — full QuIP / QuIP# / QuaRot / SpinQuant / BDR family overview with comparison table.
- [[sglang]] — the serving engine SAW-INT4 forks.
- [[long-context-serving]] — where KV-cache compression matters most.
- [[paged-attention]] — the paged KV layout BDR has to be compatible with.
- [[vllm]] — alternative serving engine; an obvious port target for BDR.
- [[multi-turn-optimization]] — multi-turn KV reuse interacts with quantization quality at the prefix-cache level.
