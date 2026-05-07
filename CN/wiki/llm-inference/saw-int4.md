---
title: "SAW-INT4：系统感知的 4 位 KV 缓存量化（块对角旋转）"
category: llm-inference
tags: [saw-int4, kv-cache, 量化, int4, hadamard旋转, bdr, sglang, 论文精读]
created: 2026-05-06
updated: 2026-05-07
status: mature
paper: arXiv:2604.19157
code: https://github.com/togethercomputer/saw-int4
---

# SAW-INT4：系统感知的 4 位 KV 缓存量化

> [!info] 论文元信息
> - **论文**：[arXiv:2604.19157](https://arxiv.org/abs/2604.19157) — Together AI 等
> - **代码**：[togethercomputer/saw-int4](https://github.com/togethercomputer/saw-int4)（分支 `main`）
> - **SGLang fork**：[jindajia/sglang-fork @ colm_rotation_fast](https://github.com/jindajia/sglang-fork)（commit `0fcc241`）
> - **作者**：Jinda Jia, Jisen Li, Zhongzhu Zhou, Jung Hwan Heo, Jue Wang, Tri Dao, Shuaiwen Leon Song, Ben Athiwaratkun, Chenfeng Xu, Tianyi Zhang, Xiaoxia Wu

> [!abstract]+ TL;DR
> INT4 [[kv-cache-optimization|KV 缓存]]量化把容量翻 4 倍（相对 BF16），但会把推理模型搞崩 —— Qwen3-4B-Thinking-2507 在原始 INT4 下 GPQA 从 **66.67 % 跌到 0 %**。SAW-INT4 通过对 K（可选 V）沿头维度做**块对角 Hadamard 旋转 (BDR)**、再做 per-token INT4 量化来修复，整套操作融合进一个写入分页 KV 布局的 Triton 内核。Q 在解码内核里被同一矩阵旋转，注意力数学保持不变。同模型上 BDR 恢复到 **65.82 % GPQA**，并且**端到端吞吐与原始 INT4 在 32–256 并发下不可区分**（H100 单卡）。

---

## 背景：为什么 INT4 KV 会崩掉推理模型

KV 缓存按 token、层、头存放 K/V 张量。对长上下文 + 高并发场景而言，它主导 GPU 显存（参见 [[kv-cache-optimization]]）；压缩到 INT4 相对 BF16 内存减 4 倍，是长上下文 [[long-context-serving|服务]]最关键的旋钮。但有一个已知的失效模式：K/V 张量存在**离群通道** —— 少数维度的幅值比其余大 1–2 个数量级。文献里几个常见的诱因：

1. **RoPE 对齐的通道。** 旋转位置编码把能量集中到特定的维度对，导致这些维度在所有 token 上都一致地携带较大值。
2. **"Massive activations" token。** 少数 token（常见为系统提示开头的 token 或 `<bos>` 类标记）携带数量级大于其他 token 的激活，并体现在 K/V 上。
3. **通道专用化的注意力头。** 训练好的注意力头会发展出偏好通道 —— 用于检索、归纳或拷贝的特征 —— 这些通道在分布上系统性更宽。

Per-token scale-and-zero INT4 量化为每个 token 向量计算一对 `(scale, zero)`，把 16 个 INT4 等级均匀分配到 $[\min, \max]$。当两三个通道携带 90 % 的幅值时，这些通道吃掉绝大多数分辨率，head dim 的其余部分被量化到 0 或近似 0。表面任务上还能撑住 —— 模型仍然能产出流畅文本。多步推理上，每次注意力的误差跨数百轮累积，模型逐步丧失辨别能力，精度直接崩溃。

论文在 `Qwen/Qwen3-4B-Thinking-2507` 上的 GPQA 数字让这一点很具体：

| KV 数据类型                       | GPQA 分     |
| --------------------------------- | ---------- |
| BF16（基线）                      | **66.67 %** |
| 原始 INT4                         | **0.00 %**  |
| INT4 + BDR（仅 K，order=128）     | **65.82 %** |

> [!important] 原始 INT4 不是让模型变差 —— 是直接让它失效
> 跌幅是 66.67 % → 0 %，不是渐进退化，是悬崖。这就是 SAW-INT4 攻击的鸿沟。

> [!question]+ Shiki — 离群通道是什么？为什么原始 INT4 KV 会崩？(2026-05-07)
>
> *（引用）*："Per-token scale-and-zero INT4 量化为每个 token 向量计算一对 `(scale, zero)`，把 16 个 INT4 等级均匀分配到 `[min, max]`。当两三个通道携带 90 % 的幅值时，这些通道吃掉绝大多数分辨率，head dim 的其余部分被量化到 0 或近似 0。表面任务上还能撑住 —— 模型仍然能产出流畅文本。多步推理上，每次注意力的误差跨数百轮累积，模型逐步丧失辨别能力，精度直接崩溃。"这段话是什么意思？
>
> **离群通道**指的是 K（或 V）的 token 向量里的 `head_dim` 个通道中，幅值在所有 token 上**系统性地**比其余通道大 1–2 个数量级的那几个。`head_dim = 128` 时，可能有 2–3 个通道的幅值在 ~1.0 量级，其他 125 个在 ~0.01 量级。三类常见来源：RoPE 把能量集中到特定维度对上（这几对在所有 token 上都偏大）、"massive activations" token（少数 token，如 system prompt 开头或 `<bos>` 类标记，整行幅值数量级更大）、训练后的注意力头专用化某些通道（用于检索、归纳、拷贝）。
>
> Per-token scale-and-zero 量化把整行 `head_dim` 个值用同一对 `(scale, zero)` 压到 INT4：$\text{scale} = (\max(x) - \min(x)) / 15$，$\text{zero} = -\min(x) / \text{scale}$。这两个值都来自整行的 max/min，但如果 2–3 个离群通道携带了 90 % 的幅值，行的 max 和 min *几乎只由它们决定*。`scale` 被撑成 ~0.1 量级（去匹配那些大值），而剩下 125 个普通通道的真实值只在 ±0.01 范围。
>
> 普通通道除以一个比它们需要的尺度大 10 倍的 `scale`，舍入后全部塌缩到同一个等级 —— 通常就是 `zero` 本身 —— 也就是被有效量化成 0。**95 % 以上的通道丢失了几乎全部信息**；只有那几个离群通道保留可用分辨率。这正是"head dim 的其余部分被量化到 0 或近似 0"的准确含义。
>
> 单次 attention $\text{softmax}(QK^\top / \sqrt{d}) \cdot V$ 的内积幅值被离群通道主导（它们最大），所以 softmax 后的注意力分布大致正确，模型仍能在表面任务上产出流畅文本 —— LLM 训练的冗余足以盖过这层模糊。但 GPQA 这种多步推理要求模型跨数十、上百次 attention 精确辨别细微差别，每次的量化误差独立累积，几百轮后区分对错的信号被噪声彻底盖住。这种"累积"是 Qwen3-4B-Thinking 在 GPQA 上从 66.67 % 一步掉到 0 % 的原因 —— 不是渐进退化，是雪崩。
>
> [[#核心思想：在 INT4 之前做块对角 Hadamard 旋转|BDR]] 的作用就是在量化前对每个 head_dim 块做 Hadamard 旋转，把每块内的离群能量均匀打散到块内所有维度上。旋转后行的 max/min 不再被少数通道主导，`scale` 缩小到合理范围，每个维度都能分到几个 INT4 等级，每次 attention 的误差也不再灾难性累积 —— 恢复到 BF16 基线 66.67 % 中的 65.82 %。

### 为什么旋转能修复这个问题，以及之前已经有人尝试过什么

把向量乘上一个正交矩阵不改变 $L_2$ 范数，但会把它的能量重新分布到所有维度上。如果离群点住在少数坐标里，且旋转把这些坐标与其余坐标混合，旋转后的张量分布会更均匀，量化也就容易得多。Hadamard 矩阵是自然选择：它是正交的、所有元素是 $\pm 1$，且通过 Fast Walsh-Hadamard Transform (FWHT) 做矩阵-向量积只需 $O(d \log d)$。

这个想法在 2026 年并不是新的：

- **QuaRot**（Ashkboos et al., NeurIPS 2024）首次把随机 Hadamard 旋转用在 LLaMA 的**权重 + 激活**量化上，做到了 INT4 权重 + INT4 激活而几乎没有质量损失。
- **SpinQuant**（Liu et al., 2024）把随机旋转换成**学习的**旋转矩阵，量化效果更好但需要离线校准。
- **HALO**、**DuQuant** 等把这些想法扩展到模型中其他位置。

SAW-INT4 的贡献是把这条线路具体应用到 **KV 缓存**（不是权重，不是激活）上，并且在**生产服务约束**下（[[paged-attention|分页]]内存、融合的[[continuous-batching|连续批处理]]、无离线校准）。**块对角**这个限制（在固定大小的 head-dim 块内旋转，而不是整个 head dim）是让它对内核友好的关键；融合的 Triton 实现是让它在运行时几乎免费的关键。Tri Dao 列在作者里也合乎这个框架 —— 这是 FlashAttention 血统的系统工作，不是为新而新的算法。

更广的家族综览（QuIP / QuIP# / QuaRot / SpinQuant / BDR）见 [[rotation-based-quantization]]。

---

## 核心思想：在 INT4 之前做块对角 Hadamard 旋转

> [!quote] 一句话贡献
> 在 per-token INT4 量化之前，对 KV 张量沿头维度做固定块大小（如 16 或 128）的 Hadamard 旋转。旋转把离群能量在块内重新分布。Q 在解码时由同一矩阵旋转，从而保持注意力数学不变。

"块对角"这个限定承担了真正的工作量。在整个 head_dim 上做完整 Hadamard 是 $O(d^2)$ 的内存通量（或 FWHT 的 $O(d \log d)$，但在小 head_dim 上缓存行为更糟），且与按头分组的分页 KV 布局不兼容。把 `head_dim` 切成大小 $H$ 的块（要求 $H \mid \text{head\_dim}$），每块独立旋转，有三个性质：

1. **代价。** 旋转是 $O(d \log H)$ 而非 $O(d \log d)$ —— 对 $d = 128$、$H = 16$，这是 4 个蝶形阶段而非 7 个。
2. **缓存局部性。** 每块小到能在整个 FWHT 期间住在共享内存 / 寄存器里。
3. **通道对齐的离群点保持局域。** 因为离群通道在固定位置（如 RoPE 对），且每个 Hadamard 块是 head dim 的一个连续切片，离群能量在**块内**被混合。这刚好够 —— per-token 量化对整个 head dim 跨 token 操作，但每块的 Hadamard 已经把方差摊得足够平，单一的 per-token `(scale, zero)` 就能覆盖整行。

Qwen3 的 `head_dim = 128`，可选 order 是 128 的 2 的幂因子（16、32、64、128）。README 主结果用 **`HADAMARD_ORDER = 128`**，env-var 文档示例用 **`16`**。两者都被验证可用；权衡是旋转强度 vs. 内核大小成本。

支撑这个贡献的三个子主张：

- **Token 级 INT4 已经够用** —— 只要先固定输入分布。论文与 k-means 聚类、向量量化、Hessian-aware 方法对比，BDR 在真实服务约束下匹配或超越它们。
- **内核必须融合**。分裂实现（旋转 → 写入）多一次全局内存通过；BDR 的贡献是把"旋转 + 归一化（$\div \sqrt{H}$）+ per-token min/max/scale/zero 计算 + INT4 打包写入分页 buffer"放进**一个 Triton 内核**。
- **K-only 通常就够**。默认只旋转 K；旋转 V（`ROTATE_V=1`）需要在注意力输出处做反旋转，多一次内存通量，论文称带来的精度增益很小。

> [!tip] 推荐主模式
> `HADAMARD=1`、`HADAMARD_ORDER=128`、`ROTATE_V=0`、`--kv-cache-dtype int4`。K-only 是默认是有原因的 —— 旋转 V 多花墙钟时间，精度收益却边际。

### Hadamard 旋转的两行复习

Sylvester-Hadamard 矩阵的递归定义：

$$
H_2 = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}, \qquad H_{2n} = H_2 \otimes H_n
$$

$H_d$ 是正交的：$H_d^\top H_d = I$。$1/\sqrt{d}$ 归一化是让它成为等距变换的关键 —— 没有它，$Hx$ 会把每个分量放大 $\sqrt{d}$。Fast Walsh-Hadamard Transform (FWHT) 用 $\log_2 d$ 个蝶形阶段在 $O(d \log d)$ 时间里算 $H_d \cdot x$：在第 $s$ 阶，每个元素与距离 $2^s$ 的伙伴配对，配对结果变成 $(a + b,\ a - b)$。下面 `_fwht_blocked_segments_tensor` 的代码就是这件事的具体实现。

---

## 系统是如何工作的

### BDR 在推理流水线中的位置

```
                      ┌─ 标准 SGLang INT4 路径 ────────────────────────┐
prefill / decode ────►│ 计算 K,V (BF16)                              │
                      │ quantized_set_kv_int4_triton(...)            │
                      │   = per-token scale/zero + INT4 打包 +       │
                      │     分页写入 k_buffer / v_buffer             │
                      └────────────────────────────────────────────┘

                      ┌─ BDR 路径（HADAMARD=1）────────────────────────┐
prefill / decode ────►│ 计算 K,V (BF16)                              │
                      │ quantized_set_kv_int4_hadamard_fused_triton  │
                      │   单内核完成（每个 token、每个头）：           │
                      │     1. 加载 BF16 行 → 转 FP32 → × 1/√H        │
                      │     2. 寄存器内 block-FWHT（LOG 个阶段）       │
                      │     3. per-token min/max → scale, zero        │
                      │     4. 取整 + 每字节打包两个 4-bit 值          │
                      │     5. 写入分页 k_buffer + scales_zeros        │
                      │                                              │
                      │ 解码时 Triton GQA 内核对 Q 应用同样的         │
                      │ 寄存器内 Hadamard（fuse_q_hadamard 标志       │
                      │ → triton_backend.py）                         │
                      └────────────────────────────────────────────┘
```

### 配置接口 —— 4 个环境变量 + 1 个 CLI 参数

整套行为开关用环境变量暴露，服务启动时读一次（`memory_pool.py`）。简洁是有意为之 —— 主模式不需要改模型，也不需要校准。

```python
# memory_pool.py
_hadamard_enabled  = 1 if os.environ.get("HADAMARD",  "0") in ("1","true","True") else 0
_rotate_v_enabled  = 1 if os.environ.get("ROTATE_V",  "0") in ("1","true","True") else 0
_hadamard_order    = int(os.environ.get("HADAMARD_ORDER", "16"))
_fuse_hadamard_int4_kv = os.environ.get(
    "SGLANG_FUSE_HADAMARD_INT4_KV", "1"
).lower() in ("1", "true", "yes")
```

完整模式矩阵（来自 `docs/bdr_env_vars.md`）：

| 模式                            | `HADAMARD` | `ROTATE_V` | `HADAMARD_ORDER`    | `--kv-cache-dtype` |
| ------------------------------- | ---------- | ---------- | ------------------- | ------------------ |
| BF16 KV（基线）                  | `0`        | `0`        | 不设                 | `auto`             |
| INT4 KV（无旋转）                | `0`        | `0`        | 不设                 | `int4`             |
| INT4 + BDR（仅 K，默认）         | `1`        | `0`        | 例如 `16` 或 `128`   | `int4`             |
| INT4 + BDR（K + V）              | `1`        | `1`        | 例如 `16`            | `int4`             |

> [!note] `HADAMARD_ORDER` 的约束
> 必须是 2 的幂**且**整除 `head_dim`。Qwen3（`head_dim = 128`）下 16 / 32 / 64 / 128 都行。融合内核额外把 order 上限设为 `MAX_HADAMARD_ORDER = 4096`，避免 `tl.arange(0, order)` 把 Triton 编译时间撑爆。

### 分派点（`set_kv_buffer`）

INT4 KV 池的 `set_kv_buffer` 里有三条路径。快路径是默认；慢路径用于调试：

```python
# memory_pool.py — INT4 set_kv_buffer 的 BDR 路径
if self.dtype == "int4":
    if _hadamard_enabled:
        hadamard_order = _hadamard_order
        assert cache_k.shape[-1] % hadamard_order == 0, \
            f"head_dim must be divisible by {hadamard_order}"

        if _fuse_hadamard_int4_kv:
            # 快路径：单 Triton 内核做 旋转 + 归一化 + 量化 + 分页写入。
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

        # 慢路径：通过 fast_hadamard_transform 分裂式 旋转 → 量化。
        # 把最后一维切成 (n_blocks, block)。
        orig_shape = cache_k.shape                               # (..., head_dim)
        cache_k = cache_k.view(*orig_shape[:-1],
                               orig_shape[-1] // hadamard_order,
                               hadamard_order)                   # (..., n_blocks, block)
        cache_k = hadamard_transform(cache_k / math.sqrt(hadamard_order))
        cache_k = cache_k.view(orig_shape)
        if _rotate_v_enabled:
            # 同样的 reshape + transform 应用到 V
            ...

    # 共同路径：per-token scale/zero + INT4 打包 + 分页写入。
    quantized_set_kv_int4_triton(
        cache_k, cache_v, loc,
        self.k_buffer       [layer_id - self.start_layer],
        self.v_buffer       [layer_id - self.start_layer],
        self.k_scales_zeros [layer_id - self.start_layer],
        self.v_scales_zeros [layer_id - self.start_layer],
    )
    return
```

三点值得点出来：

- **reshape 就是旋转的作用域定义。** `(..., head_dim) → (..., head_dim // H, H)` 即"把 head dim 当作 `head_dim/H` 个独立的、大小为 $H$ 的组"。Hadamard 变换只在每组内部混合；任何跨块的混合都不存在。一个块里的离群点不会泄漏到下一个块 —— 这正是分页 KV 布局兼容性的保证。
- **$1/\sqrt{H}$ 归一化** 让 Hadamard 变换保持**等距**（保 $L_2$ 范数）。没有这一项，BF16 幅值会偏移，per-token scale 校准会跑偏。慢路径里这一项是显式的；融合内核里它已固化为 `PRE_SCALE`。
- **`scales_zeros` 是独立的 buffer。** 分页 KV 布局把 INT4 打包字节放在 `k_buffer` / `v_buffer`，把*每个 (token, head)* 的 `(scale, zero)` 对放在 `k_scales_zeros` / `v_scales_zeros`。内核原子地写两个 buffer —— 不存在 stale-pair 竞态。

### 融合 Triton 内核内部

融合内核位于 `python/sglang/QuantKernel/fused_hadamard_int4_kv.py`。三个逻辑片段重要：FWHT 蝶形、per-token min/max + INT4 打包、以及"每个 program 处理多个头"的 launch grid。

#### 蝶形

每个 program 把整个 padded head-dim 行加载到寄存器向量里，原地跑 `LOG = log₂(hadamard_order)` 个 Sylvester FWHT 阶段：

```python
@triton.jit
def _fwht_blocked_segments_tensor(x, head_dim_: tl.constexpr, LOG: tl.constexpr):
    """FWHT on each contiguous block of size 2**LOG tiling head_dim_ (vectorized).

    每个蝶形阶段只用 1 次 gather（之前是 4）：通过 x[i ^ stride]
    取 partner（编译期排列），再用 i 的第 s 位决定加还是减。
    """
    i = tl.arange(0, head_dim_)
    for s in tl.static_range(0, LOG):
        stride  = 1 << s
        partner = i ^ stride                # 编译期索引置换
        x_p     = tl.gather(x, partner, 0)  # 1 次 gather（之前是 4）
        is_lo   = ((i >> s) & 1) == 0       # 编译期 mask
        x       = tl.where(is_lo, x + x_p, x_p - x)
    return x
```

发生了什么：在第 $s$ 阶，元素 $i$ 与 $i \oplus 2^s$ 配对。每对的下半部变成 $a + b$，上半部变成 $b - a$。因为 `i` 与 `stride` 都是编译期 `tl.arange` 常量，partner 置换与 lo/hi 掩码在 JIT 时完全求解 —— 没有运行时地址算术。作者评论说这把每阶段的蝶形从 4 次 gather 降到 1 次，在 Hopper 上这是"内存受限"vs."ALU 受限"的差别（对小 head_dim 而言）。

> [!note] 块对角 Hadamard 是免费的副产品
> "blocked segments"这个名字来自一个事实：当 $\text{LOG} < \log_2(\text{head\_dim\_pad\_})$ 时，这条单一的 1-D 蝶形*隐式地*完成了块对角 Hadamard —— 那 $\log_2(\text{order})$ 个阶段只在 2 的幂对齐的、大小为 $2^{\text{LOG}}$ 的块内混合。所以 `HADAMARD_ORDER = 128` 在 128 元素行上跑 7 个阶段；`HADAMARD_ORDER = 16` 跑 4 个阶段并保留更高块边界不混合。块对角性质 —— 通过提早结束蝶形免费拿到。

#### 完整的内核体（按 token、按头组）

```python
@triton.autotune(configs=autotune_cfgs, key=["head_dim_"])
@triton.jit
def _fused_hadamard_int4_set_kv_kernel(
    input_ptr, loc_ptr, cache_ptr, scales_zeros_ptr,
    num_tokens, num_heads,
    head_dim_:    tl.constexpr,   # 真实（未 padded）head dim
    head_dim_pad_: tl.constexpr,  # 大于等于 head_dim_ 的下一个 2 的幂
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
            # 1. 加载 BF16 行到 2 的幂寄存器 buffer；转 FP32；预缩 1/√H。
            dim_full = tl.arange(0, head_dim_pad_)
            input_off = token_idx * input_stride_token + head_idx * input_stride_head
            x = tl.load(
                input_ptr + input_off + dim_full * input_stride_dim,
                mask=dim_full < head_dim_, other=0.0,
            ).to(tl.float32) * PRE_SCALE                   # PRE_SCALE = 1 / sqrt(H)

            # 2. 寄存器内 block-FWHT。
            x = _fwht_blocked_segments_tensor(x, head_dim_pad_, LOG)

            # 3. 经过 BF16 round-trip（与 CUDA 路径数值一致）。
            half_dim = head_dim_ // 2
            dim_off  = tl.arange(0, BLOCK_HALF)
            dim_mask = dim_off < half_dim
            vals1 = tl.where(dim_mask, tl.gather(x, dim_off,            0), 0.0).to(tl.bfloat16).to(tl.float32)
            vals2 = tl.where(dim_mask, tl.gather(x, dim_off + half_dim, 0), 0.0).to(tl.bfloat16).to(tl.float32)

            # 4. per-token min/max → INT4 scale/zero（4-bit 无符号有 16 个等级 = 15 个间隔）。
            val_min   = tl.minimum(tl.min(vals1, 0), tl.min(vals2, 0))
            val_max   = tl.maximum(tl.max(vals1, 0), tl.max(vals2, 0))
            val_range = tl.maximum(val_max - val_min, 1e-8)
            scale     = val_range / 15.0
            zero      = -val_min / scale

            # 5. 取整、剪到 uint8（已经在 0..15）、每字节打包两个 4-bit 值。
            q1 = (vals1 / scale + zero + 0.5).to(tl.uint8)
            q2 = (vals2 / scale + zero + 0.5).to(tl.uint8)
            packed = q1 | (q2 << 4)

            # 6. 把打包字节写入分页 KV buffer；写 (scale, zero) sidecar。
            cache_off = (cache_loc * cache_stride_loc
                         + head_idx * cache_stride_head
                         + dim_off  * cache_stride_dim)
            tl.store(cache_ptr + cache_off, packed, mask=dim_mask)
            sz_base = cache_loc * sz_stride_loc + head_idx * sz_stride_head
            tl.store(scales_zeros_ptr + sz_base + 0 * sz_stride_dim, scale)
            tl.store(scales_zeros_ptr + sz_base + 1 * sz_stride_dim, zero)
```

几点细节值得点出：

- **`head_dim_pad_` 处理非 2 的幂的 head dim**（如 320、576、768），方法是用 0 填充；因为 `hadamard_order | head_dim`，padding 部分总是落在块边界上，绝不会污染真实块。
- **量化前的 BF16 round-trip**（`.to(tl.bfloat16).to(tl.float32)`）匹配慢路径 $(\text{bf16\_tensor} / \sqrt{\text{order}})$ 的数值 —— 这也是为什么内核 docstring 警告："那个缩放与 CUDA Hadamard 之前的 `(bf16_tensor / sqrt(order))` 略有不同，所以打包字节在罕见情况下可能与未融合路径不同。" 实际上 GPQA 结果一致。
- **`q1 | (q2 << 4)` 打包**是标准的 INT4 布局 —— 元素 $2i$ 在字节 $i$ 的低半字节，元素 $2i + 1$ 在高半字节。`BLOCK_HALF = head_dim/2` 常量是每行的字节数。
- **`HEADS_PER_PROGRAM`** 让一个 program 顺序处理多个头，缩减 grid 大小并摊薄 `loc_ptr` 加载。小 head dim 默认 `min(8, num_heads)`；当 `next_power_of_2(head_dim) ≥ 512` 时降到 `1`（per-program 寄存器压力变大让多头 per program 反而劣）。autotune 表也在大 head dim 时缩减以避免长达数分钟的编译时间。

#### Launcher

```python
def quantized_set_kv_int4_hadamard_fused_triton(
    cache_k, cache_v, loc,
    k_cache_buffer, v_cache_buffer,
    k_scales_zeros_buffer, v_scales_zeros_buffer,
    hadamard_order: int,
    work_k=None, work_v=None,                  # 历史遗留；忽略
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

    kernel, cfg = _get_kernel(head_dim, hadamard_order)   # JIT 缓存以 (head_dim, order, rev) 为键
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
        # ROTATE_V=0：V 走原始 INT4 内核 —— 同样的分块、不旋转。
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

巧妙的地方：当 `rotate_v=False` 时，V 不是被跳过 —— 而是用现有的 `_quantized_set_kv_int4_kernel` 写入同一个分页 buffer，使用同样的 `HEADS_PER_PROGRAM` 分块以匹配 K launch 的 grid。两次 launch 共享 `loc` 与每 token 的 scale-zero 布局，KV 保持一致。

### 解码侧的 Q 修正

只对 K 做 Hadamard 旋转保持 $Q \cdot K^\top$ 不变的前提是 Q 也被同一矩阵旋转。Fork 把这个在 GQA 解码内核内部完成，由同一个 `SGLANG_FUSE_HADAMARD_INT4_KV` 控制（`triton_backend.py:1042-1058`）：

```python
# triton_backend.py — 解码路径
if hasattr(kv_pool, "dtype") and kv_pool.dtype in ("int4", "int8"):
    fuse_q_hadamard_in_kernel = (
        kv_pool.dtype == "int4"
        and _hadamard_enabled
        and _fuse_hadamard_int4_kv
    )
    if kv_pool.dtype == "int4" and _hadamard_enabled and not fuse_q_hadamard_in_kernel:
        # 慢路径：在 decode 调用前显式旋转 Q。
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
        # ROTATE_V=1：在注意力输出上撤销 V 旋转。
        orig_shape = o.shape
        o = o.view(*orig_shape[:-1], orig_shape[-1] // _hadamard_order, _hadamard_order)
        o = hadamard_transform(o / math.sqrt(_hadamard_order))
        o = o.view(orig_shape)
```

三块拼起来：

1. **内核内 Q 旋转**是默认 —— `decode_attention_fwd_quantized` 接收 `fuse_q_hadamard=True` 与 `hadamard_order`，在 GQA 点积内反量化 K 之前对 Q 应用同一 FWHT。无额外全局内存通过。
2. **内核外慢路径**用 `fast_hadamard_transform` 在调用前旋转 Q。当 `SGLANG_FUSE_HADAMARD_INT4_KV=0` 时使用。慢的原因是 Q 被多碰一次。
3. **`ROTATE_V=1` 时的输出反旋转。** 如果写入时 V 被旋转，注意力输出 $o = \text{softmax}(QK^\top) \cdot V$ 携带了 V 的旋转；反 Hadamard 把它还原。这是在解码内核外作为单独一遍做的 —— 这也是 `ROTATE_V=1` 即便精度增益边际仍然多花墙钟的一个原因。

### Per-token INT4 量化公式

内核使用标准的非对称零点无符号 4-bit 量化。每 token、每头：

$$
\begin{aligned}
\text{range} &= \max(x) - \min(x) \quad \text{沿 head\_dim 行} \\
\text{scale} &= \text{range} / 15 \quad \text{(4-bit 无符号有 16 个等级、15 个间隔)} \\
\text{zero}  &= -\min(x) / \text{scale} \quad \text{(反量化后回到 0 的整数等级)} \\
q            &= \mathrm{round}(x / \text{scale} + \text{zero}) \quad \text{元素级，剪到 } [0, 15]
\end{aligned}
$$

反量化（在解码内核里）：

$$
x_{\text{dequant}} = (q - \text{zero}) \cdot \text{scale} = q \cdot \text{scale} - \min(x)
$$

每 `(token, head)` 的存储：
- `head_dim / 2` 字节的打包半字节（每个半字节一个 4-bit 元素），
- `(scale, zero)` 各 1 个 float。

> [!example] Qwen3 的内存账
> 对 Qwen3（`head_dim = 128` 且 4 个 KV 头，GQA），BDR + INT4 后单 token 的 KV 缓存成本是 $4 \times (64 + 8) = 288$ 字节 —— 而 BF16 是 $4 \times 128 \times 2 = 1024$ 字节，端到端**减少 3.55 倍**。不到 4 倍的原因是 scales/zeros sidecar。

### 怎么跑

用户可见的接口就是在 SGLang 启动命令上翻一个 env 变量：

```bash
# BF16 基线
python -m sglang.launch_server \
  --prefill-attention-backend fa3 --decode-attention-backend triton \
  --model-path "Qwen/Qwen3-4B-Thinking-2507" --port 30000 \
  --kv-cache-dtype auto

# 原始 INT4 KV（在推理任务上模型崩溃）
python -m sglang.launch_server ... --kv-cache-dtype int4

# INT4 + BDR（K-only，块大小 128）—— 推荐主模式
HADAMARD=1 HADAMARD_ORDER=128 \
python -m sglang.launch_server ... --kv-cache-dtype int4

# INT4 + BDR（K + V，块大小 16）
HADAMARD=1 ROTATE_V=1 HADAMARD_ORDER=16 \
python -m sglang.launch_server ... --kv-cache-dtype int4

# 慢参考路径（fast-hadamard-transform CUDA + 原始 INT4 内核）—— 调试用
HADAMARD=1 HADAMARD_ORDER=128 SGLANG_FUSE_HADAMARD_INT4_KV=0 \
python -m sglang.launch_server ... --kv-cache-dtype int4
```

一个简短的 OpenAI-client 冒烟测试（`scripts/bdr_smoke_test.py`）发送 GPQA 题验证安装：

```python
from openai import OpenAI
client = OpenAI(api_key="EMPTY", base_url=f"http://0.0.0.0:{port}/v1")
response = client.chat.completions.create(
    model="Qwen/Qwen3-4B-Thinking-2507",
    messages=[{"role": "user", "content": GPQA_SAMPLE}],
    temperature=0.6, top_p=0.95, max_tokens=32768, stream=True,
)
```

对那道 GPQA 化学题（关于 TLC 极性）给出连贯回答即可确认 BDR 接通了。

### K-means 消融管线

一个独立的子仓（`third_party/sglang-kmeans`，是同一 fork 的 `jinda_kmeans_rotation_dump` 分支）实现了一个替代量化器：不用 scale-and-zero，而是把每层的 KV 向量聚成 $N$ 个质心，存簇索引。校准是离线的：

```bash
# 1. 从 BF16 服务 dump KV 激活。
DUMP_KVCACHE=true DUMP_KVCACHE_TOKENS=512 DUMP_KVCACHE_DIR=/path/to/dumps \
python -m sglang.launch_server ... --kv-cache-dtype auto

# 2. 拟合每层质心（tools/fit_kv_centroids.py）。
python tools/fit_kv_centroids.py \
  --dump-dir /path/to/dumps \
  --out-dir  /path/to/centroids \
  --n-clusters 16 --seed 0

# 3. 用 INT4 + k-means 起服务。
N_CLUSTERS=16 SGLANG_KV_CENTROIDS_PATH=/path/to/centroids \
python -m sglang.launch_server ... --kv-cache-dtype int4
```

`fit_kv_centroids.py` 短小且具体：

```python
# 每层：
blob = torch.load(f"kv_calibration_layer_{L}.pt")  # {'k': [T,H,D], 'v': [T,H,D]}
xk = blob["k"].reshape(T, H * D).float().numpy()    # 把 heads × dims 拍平
xv = blob["v"].reshape(T, H * D).float().numpy()
km_k = KMeans(n_clusters=16, n_init=10, max_iter=300).fit(xk)
km_v = KMeans(n_clusters=16, n_init=10, max_iter=300).fit(xv)
torch.save(km_k.cluster_centers_, f"k_layer_{L}_clusters_16_centers.pt")
torch.save(km_v.cluster_centers_, f"v_layer_{L}_clusters_16_centers.pt")
```

整脚本约 100 行。可选的旋转可通过同样的 `HADAMARD` / `ROTATE_V` 叠在 k-means 之上 —— README 文档化的矩阵：

| 方法              | `HADAMARD` | `ROTATE_V`      | `--kv-cache-dtype` | `SGLANG_KV_CENTROIDS_PATH` |
| ----------------- | ---------- | --------------- | ------------------ | -------------------------- |
| K-means + INT4    | `0`        | `0`             | `int4`             | 必需                        |
| K-means + BDR     | `1`        | `0` 或 `1`      | `int4`             | 必需                        |

> [!warning] 消融表是空的
> README 里这两种方法的消融精度表是空的（占位行），这点很诚实，但意味着公开的对比目前只靠 GPQA 主结果支撑。

---

## 实验

**硬件。** 1× H100 80 GB，TP = 1。

### 精度

Qwen3-4B-Thinking-2507，GPQA，`temp=0.6`，`top_p=0.95`，3 次重复，32 K 上下文：

| 配置                                | GPQA       |
| ----------------------------------- | ---------- |
| BF16 KV                             | 66.67 %    |
| INT4 KV                             | 0 %        |
| **INT4 + BDR（仅 K，ord=128）**      | **65.82 %** |

### 吞吐

Qwen3-8B，GenAI-Bench，流量 `D(256, 1024)` 短和 `D(16384, 1024)` 长。

**短上下文**（256 输入 / 1024 输出），并发扫描 —— job 级 `output_tps`（汇总 token/s）和 TTFT（ms）：

| 并发 | BF16             | INT4             | INT4 + BDR              |
| ---: | ---------------: | ---------------: | ----------------------: |
|  32  | 3,795 / 196      | 3,687 / 225      | 3,689 / 226             |
|  64  | 5,950 / 369      | 6,371 / 370      | 6,235 / 377             |
| 128  | 8,410 / 657      | 9,544 / 665      | 9,350 / 655             |
| 256  | 11,195 / 1,224   | 11,624 / 1,237   | **11,732 / 1,148**      |

**长上下文**（16,384 输入 / 1,024 输出），并发扫描：

| 并发 | BF16              | INT4              | INT4 + BDR         |
| ---: | ----------------: | ----------------: | -----------------: |
|   8  |   414 / 2,636     |   458 / 2,631     |   457 / 2,523      |
|  16  |   481 / 5,104     |   571 / 4,956     |   568 / 4,875      |
|  32  |   570 / 18,047    |   618 / 9,568     |   616 / 9,350      |
|  64  |   471 / 44,798    |   666 / 19,398    |   663 / **18,371** |
| 128  |   559 / 113,583   |   701 / 57,654    |   701 / **57,054** |

规律：BDR 的吞吐数字基本与原始 INT4 在噪声范围内，并发 ≥ 256 / 长上下文时甚至在 `output_tps` 与 TTFT 上反超。在高并发下 BF16 急剧落后（长上下文 conc-128 的 TTFT 113s vs. INT4 / BDR 约 57s），因为它 4 倍大的 KV 缓存把系统推进了内存压力。

> [!note] 为什么 BDR 偶尔 *快过* 原始 INT4 的 TTFT
> BDR 内核在与原始 INT4 相同的内存通过中触碰 `cache_k` 并写 scales/zeros buffer，所以*额外*的旋转成本被摊薄进本来就要做的工作里。内核融合把潜在的 5–10 % 开销变成白送 —— 正是论文想要的框架。

精度故事是重头戏：BDR 让一个能用的推理模型从"可用"变成"完全失效"或反过来，而运行时几乎零代价。

---

## 优点与不足

两个最强的点：(1) 这个技术**侵入极小** —— 4 个 env var + 一个 fork 的内核，不需要重训也不需要校准；(2) **融合内核**把本来可能 5–10 % 的开销摊进 INT4 本来就需要的同一次内存通过，让它隐没在测量噪声里。标题里的"系统感知"是名副其实的。

工作对自身范围的诚实，但限制本身仍值得关注：

- **仅 MHA。** README 明确禁用 MLA。DeepSeek-V3 风格架构（MLA 通过存 K 的低秩投影把 KV 缓存又压一大截）无法直接用 BDR —— 旋转必须与从压缩表示到上投影的过程交互。这个想法是否能迁移过去是开放问题，论文未涉及。
- **后端约束。** 解码用 Triton GQA，预填充用 FA3。换 SGLang 内的其他注意力后端、或往 vLLM 移植都不平凡 —— Q 修正必须落到你所用的解码内核里。慢路径（`SGLANG_FUSE_HADAMARD_INT4_KV=0`）以代价换取可移植性。
- **一个 head_dim、一个块大小、一个模型家族。** 主要数字来自 Qwen3-4B-Thinking-2507（精度）和 Qwen3-8B（吞吐）。仓库给主 BDR 结果用的是 `HADAMARD_ORDER=128`，env-var 文档示例用 `16`。论文似乎没有跨模型家族系统扫描块大小对精度的影响。
- **精度只一个基准。** GPQA 是 README 里**唯一**的精度结果。GPQA 是长形科学推理，正是 INT4 失效最严重的地方，作为 stress test 公平 —— 但量化论文只跑一个基准偏薄。MMLU、MATH、HumanEval、长上下文检索套件（如 RULER）都是合理的下一步。
- **消融表是空的。** README 里 k-means 消融矩阵全是占位。BDR 是严格优于 k-means 还是只是相当并未展示 —— 论文宣称更复杂的方法增益"边际"，但支撑这一点的表格在仓库中并未发布。
- **未与已发布系统对比。** [[quantization|KIVI]]、NVFP4、FP8 KV、ShadowKV、KVTC —— 都在同一问题空间（参见 [[kv-cache-optimization]]）。跨方法做吞吐与精度对比是显然的下一步；论文把自己限制在 BF16 vs. 原始 INT4 vs. BDR。
- **随机 Hadamard，而非学习的。** SpinQuant 表明学习的旋转优于随机 Hadamard（在权重 + 激活量化上）。论文没尝试每层学习的旋转矩阵，这本是个明显的精度杠杆 —— 代价是离线校准。
- **存在 Python 级 Hadamard 回退路径用于调试。** 当 `SGLANG_FUSE_HADAMARD_INT4_KV=0`，BDR 走 `Dao-AILab/fast-hadamard-transform` 然后写入；这条路因为多一次全局内存通过会更慢。默认是快路径，但任何想把这套移到无 Triton 友好注意力后端的服务栈的人会撞到这个开销。

> [!bug] 文档端口不一致
> `scripts/bdr_smoke_test.py` 的默认端口是 `--port 30000`（与启动样例一致），但 README 的冒烟测试代码片段无故写成 `--port 30001`。修起来很小，但说明开源发布有点仓促。

---

## 这意味着什么

更大的教训和 [[paged-attention|PagedAttention]] 或 [[sglang|RadixAttention]] 一致：**推理优化的正确粒度是内核，不是模型**。SAW-INT4 与其说是提了一个新量化方案，不如说是证明了"**普通的** per-token INT4 就够了"，前提是你在内核层面做对一件事 —— 在量化前旋转，按分页布局能吞下的块大小做。这有两条我会盯着的推论：

1. **块对角 Hadamard 会扩散开来。** 它小到可以白送给任何 INT4 KV 路径，数学是熟知的（FWHT 自 1970 年代就是教科书内容），内核工作 SGLang fork 已经做得差不多。预期 [[vllm|vLLM]] 和 [[tensorrt-llm|TensorRT-LLM]] 会很快接入；同样的招数预计会被试在权重量化（QuaRot 已经证明有效）和 FP8 激活量化上。
2. **"系统感知"框架是更耐用的贡献。** 论文反复指出向量量化与 Hessian-aware 方法在真实服务约束下"增益边际" —— 这其实是在说算法层面的精巧已经撞墙，剩余增益在内核与内存布局里。这个论断之后会持续正确；预计 2026 年更多论文会是关于把已知技术*融合*进正确的内核，而不是发明新技术。

它*不是*什么：MLA 风格模型 [[long-context-serving|长上下文服务]]的解；非 NVIDIA 硬件的答案；也不是"INT4 已解决"的证据。它是一个干净、窄的真实问题上的真实结果。

---

## 源码与复现

```bash
# 带子模块克隆。
git clone --recurse-submodules https://github.com/togethercomputer/saw-int4.git
cd saw-int4

# 安装主 BDR fork。
cd third_party/sglang-fast-rotation/python
pip install -e ".[all]"
pip install --no-build-isolation \
  "git+https://github.com/Dao-AILab/fast-hadamard-transform.git"

# 用 BDR 启动。
HADAMARD=1 HADAMARD_ORDER=128 \
python -m sglang.launch_server \
  --prefill-attention-backend fa3 \
  --decode-attention-backend triton \
  --model-path "Qwen/Qwen3-4B-Thinking-2507" \
  --port 30000 \
  --kv-cache-dtype int4

# 冒烟测试。
python scripts/bdr_smoke_test.py --port 30000 \
  --model Qwen/Qwen3-4B-Thinking-2507
```

值得继续阅读的源码（带各文件的角色）：

| 文件                                                                                         | 角色                                                                                                                                              |
| -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `third_party/sglang-fast-rotation/python/sglang/srt/mem_cache/memory_pool.py`                | env-var gate（92–98 行）、`set_kv_buffer` 的 BDR 分支（1136–1190 行）、用 `fast_hadamard_transform` 的慢路径参考。                                  |
| `third_party/sglang-fast-rotation/python/sglang/QuantKernel/fused_hadamard_int4_kv.py`       | 融合内核：`_fwht_blocked_segments_tensor`（蝶形）、`_fused_hadamard_int4_set_kv_kernel`（完整 per-token 内核）、launcher、`validate_hadamard_order_for_kv_fuse`。 |
| `third_party/sglang-fast-rotation/python/sglang/srt/layers/attention/triton_backend.py`      | 解码侧 Q 旋转（1042–1091 行）；`fuse_q_hadamard` 标志传给 `decode_attention_fwd_quantized`。                                                       |
| `third_party/sglang-fast-rotation/python/sglang/srt/layers/attention/triton_ops/decode_attention.py` | 接收 `fuse_q_hadamard` 与 `hadamard_order` 并对 Q 应用寄存器内 FWHT 的 GQA 解码内核。                                                              |
| `tools/fit_kv_centroids.py`                                                                  | k-means 质心校准（仅消融）。                                                                                                                       |
| `docs/bdr_env_vars.md`                                                                       | env var 参考与模式矩阵。                                                                                                                            |
| `scripts/bdr_smoke_test.py`                                                                  | 最简 OpenAI-client GPQA 验证。                                                                                                                      |
| `scripts/run_genai_bench_example.sh`                                                         | 吞吐扫描辅助。                                                                                                                                     |
| `scripts/run_primary_eval_matrix.sh`                                                         | 主精度/速度扫描辅助。                                                                                                                              |

---

## 相关阅读

- [[kv-cache-optimization]] —— KV 缓存压缩的全景（分页、量化、驱逐、卸载）。
- [[quantization]] —— 权重/激活量化（GPTQ、AWQ、SmoothQuant、FP8、NVFP4） —— 与 KV 量化正交，但 Hadamard 旋转在它们之间扮演同一角色（参考 QuaRot、SpinQuant）。
- [[rotation-based-quantization]] —— 完整的 QuIP / QuIP# / QuaRot / SpinQuant / BDR 家族综览，含对比表。
- [[sglang]] —— SAW-INT4 fork 的服务引擎。
- [[long-context-serving]] —— KV 压缩最重要的场景。
- [[paged-attention]] —— BDR 必须兼容的分页 KV 布局。
- [[vllm]] —— 替代服务引擎；BDR 的明显移植目标。
- [[multi-turn-optimization]] —— 多轮 KV 复用与量化质量在前缀缓存层面交互。
