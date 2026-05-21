---
title: "SAW-INT4：系统感知的 4 位 KV 缓存量化（块对角旋转）"
category: llm-inference
tags: [saw-int4, kv-cache, 量化, int4, hadamard旋转, bdr, sglang, 论文精读]
created: 2026-05-06
updated: 2026-05-21
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

---

## 摘要（2 分钟读完这一节就够）

**它是什么**。SAW-INT4（Together AI，2026）是一套面向生产级 LLM 服务的*系统感知* 4 位 [[kv-cache-optimization|KV 缓存]]量化方案。它在原始 per-token INT4 量化之前插入**块对角 Hadamard 旋转 (BDR)**，把"旋转 + 归一化 + 量化 + 分页写入"全部融合进一个 Triton 内核，并在 GQA 解码内核里对 Q 应用同一旋转，从而保持注意力数学不变。

**核心思想**。原始 INT4 KV 会把推理模型搞崩，是因为 K/V 里的**离群通道**几乎吃掉了所有量化分辨率。在量化前先做旋转 —— 用分页布局能吞下的固定块大小 —— 离群能量就被均匀打散。三个支柱：

1. **沿 `head_dim` 的块对角 Hadamard 旋转**（Qwen3 下 order = 16 / 32 / 64 / 128）—— 小到能住在寄存器里、对分页 KV 兼容、FWHT 复杂度 $O(d \log H)$。
2. **融合 Triton 内核**（`_fused_hadamard_int4_set_kv_kernel`）—— 单次内存通过完成 旋转 + $\div\sqrt{H}$ + per-token min/max/scale/zero + INT4 打包 + 分页写入。
3. **解码侧寄存器内 Q 旋转** —— `decode_attention_fwd_quantized` 在 GQA 点积内对 Q 应用同一 FWHT，保持 $QK^\top$ 而无需额外的全局内存通过。

去掉旋转：推理崩溃；去掉融合：吃下 5–10 % 开销；去掉 Q 修正：注意力数学错了。

**头条结果**。Qwen3-4B-Thinking-2507 GPQA，悬崖案例：

| KV 数据类型                       | GPQA 分     |
| --------------------------------- | ---------- |
| BF16（基线）                      | 66.67 %    |
| 原始 INT4                         | **0.00 %** |
| **INT4 + BDR（仅 K，order=128）** | **65.82 %** |

> [!important] 原始 INT4 不是让推理模型变差 —— 是直接让它失效
> 66.67 % → 0 % 是悬崖不是渐进。BDR 在几乎零运行时代价下恢复到 65.82 % —— Qwen3-8B 在 32–256 并发下端到端吞吐与**原始 INT4 不可区分**，且在高并发下 BDR 甚至在 TTFT 上反超，因为旋转成本被摊薄进 INT4 本来就要做的那次内存通过。

**为什么这重要**。

- **内存：** 端到端 KV 缓存比 BF16 小 3.55×（不到 4× 是因为 per-token `(scale, zero)` sidecar）。同样上下文、同样硬件，~4× 的并发量才触发 KV 压力。
- **侵入性极小：** 4 个 env var + 一个 fork 的内核。不重训、不校准。今天就可以接入 [[sglang|SGLang]]。
- **"系统感知"名副其实：** 贡献是内核工作，不是算法创新。数学（FWHT）从 1970 年代起就是教科书内容；新东西是把它*正确地融合*在分页 KV 写入里。
- **12 个月预测：** 块对角 Hadamard 会扩散到 [[vllm|vLLM]] 和 TensorRT-LLM；同样的招数会被试在 FP8 KV 和激活量化上。"在真实服务约束下增益边际"这个框架会继续打脸算法创新派的论文。

---

# 深度部分（往下展开细节）

上面摘要是 executive 层。下面是给愿意细读内核走读和完整代码路径的人准备的。

## 背景：为什么 INT4 KV 会崩掉推理模型

KV 缓存按 token、层、头存放 K/V 张量。对长上下文 + 高并发场景而言，它主导 GPU 显存（参见 [[kv-cache-optimization]]）；压缩到 INT4 相对 BF16 内存减 4 倍，是长上下文 [[long-context-serving|服务]]最关键的旋钮。但有一个已知的失效模式：K/V 张量存在**离群通道** —— 少数维度的幅值比其余大 1–2 个数量级。文献里几个常见的诱因：

1. **RoPE 对齐的通道。** 旋转位置编码把能量集中到特定的维度对，导致这些维度在所有 token 上都一致地携带较大值。
2. **"Massive activations" token。** 少数 token（常见为系统提示开头的 token 或 `<bos>` 类标记）携带数量级大于其他 token 的激活，并体现在 K/V 上。
3. **通道专用化的注意力头。** 训练好的注意力头会发展出偏好通道 —— 用于检索、归纳或拷贝的特征 —— 这些通道在分布上系统性更宽。

Per-token scale-and-zero INT4 量化为每个 token 向量计算一对 `(scale, zero)`，把 16 个 INT4 等级均匀分配到 $[\min, \max]$。当两三个通道携带 90 % 的幅值时，这些通道吃掉绝大多数分辨率，head dim 的其余部分被量化到 0 或近似 0。表面任务上还能撑住 —— 模型仍然能产出流畅文本。多步推理上，每次注意力的误差跨数百轮累积，模型逐步丧失辨别能力，精度直接崩溃。

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

![BDR 模块：带离群点的 token × 块对角 Hadamard → 平滑后的 token → INT4 打包（论文 Fig. 1，BDR 子组件）](CN/wiki/llm-inference/saw-int4-figs/bdr-module-rotation.png)

论文 BDR 模块示意图是最清晰的一图解释：左侧带离群能量的 head-dim 行（冷暖色条），乘上块对角 Hadamard（沿对角线的洋红方块），得到能量均匀的"平滑后的 token"，*这个* 行才被 per-token `(scale, zero)` 和 4-bit 打包接管。

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

端到端的图景是两条路径穿过同一个分页 KV 缓冲区 —— 一条带 BDR、一条不带。论文的总览图同时画出了两条：BDR 模块坐在每一次 K（和可选 V）写入之前，同样的旋转被应用到 Q 上、发生在 Triton 解码内核内部：

![SAW-INT4 系统总览：预填充和解码 KV 写入处的 BDR + Triton 注意力内核里的融合 Q 旋转（论文 Fig. 1）](CN/wiki/llm-inference/saw-int4-figs/system-architecture.png)

同一画面的 ASCII 流程：

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

### 组件 1 —— 配置接口（4 个环境变量 + 1 个 CLI 参数）

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

### 组件 2 —— 分派点（`set_kv_buffer`）

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

### 组件 3 —— 融合 Triton 内核内部

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

当 `rotate_v=False` 时，V 不是被跳过 —— 而是用现有的 `_quantized_set_kv_int4_kernel` 写入同一个分页 buffer，使用同样的 `HEADS_PER_PROGRAM` 分块以匹配 K launch 的 grid。两次 launch 共享 `loc` 与每 token 的 scale-zero 布局，KV 保持一致。完整 launcher 见 `fused_hadamard_int4_kv.py`。

### 组件 4 —— 解码侧的 Q 修正

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

### 辅助机制（可跳读）

> [!note]- 怎么跑 —— env-var 矩阵 → 启动命令
> 用户可见的接口就是在 SGLang 启动命令上翻一个 env 变量：
>
> ```bash
> # BF16 基线
> python -m sglang.launch_server \
>   --prefill-attention-backend fa3 --decode-attention-backend triton \
>   --model-path "Qwen/Qwen3-4B-Thinking-2507" --port 30000 \
>   --kv-cache-dtype auto
>
> # 原始 INT4 KV（在推理任务上模型崩溃）
> python -m sglang.launch_server ... --kv-cache-dtype int4
>
> # INT4 + BDR（K-only，块大小 128）—— 推荐主模式
> HADAMARD=1 HADAMARD_ORDER=128 \
> python -m sglang.launch_server ... --kv-cache-dtype int4
>
> # INT4 + BDR（K + V，块大小 16）
> HADAMARD=1 ROTATE_V=1 HADAMARD_ORDER=16 \
> python -m sglang.launch_server ... --kv-cache-dtype int4
>
> # 慢参考路径（fast-hadamard-transform CUDA + 原始 INT4 内核）—— 调试用
> HADAMARD=1 HADAMARD_ORDER=128 SGLANG_FUSE_HADAMARD_INT4_KV=0 \
> python -m sglang.launch_server ... --kv-cache-dtype int4
> ```
>
> 一个简短的 OpenAI-client 冒烟测试（`scripts/bdr_smoke_test.py`）发送 GPQA 题验证安装。对那道 GPQA 化学题（关于 TLC 极性）给出连贯回答即可确认 BDR 接通了。

> [!note]- K-means 消融管线 —— 想对比 centroid 量化器再展开
> 一个独立的子仓（`third_party/sglang-kmeans`，是同一 fork 的 `jinda_kmeans_rotation_dump` 分支）实现了一个替代量化器：不用 scale-and-zero，而是把每层的 KV 向量聚成 $N$ 个质心，存簇索引。校准是离线的：
>
> ```bash
> # 1. 从 BF16 服务 dump KV 激活。
> DUMP_KVCACHE=true DUMP_KVCACHE_TOKENS=512 DUMP_KVCACHE_DIR=/path/to/dumps \
> python -m sglang.launch_server ... --kv-cache-dtype auto
>
> # 2. 拟合每层质心（tools/fit_kv_centroids.py）。
> python tools/fit_kv_centroids.py \
>   --dump-dir /path/to/dumps \
>   --out-dir  /path/to/centroids \
>   --n-clusters 16 --seed 0
>
> # 3. 用 INT4 + k-means 起服务。
> N_CLUSTERS=16 SGLANG_KV_CENTROIDS_PATH=/path/to/centroids \
> python -m sglang.launch_server ... --kv-cache-dtype int4
> ```
>
> `fit_kv_centroids.py` 约 100 行。可选的旋转可通过同样的 `HADAMARD` / `ROTATE_V` 叠在 k-means 之上。README 里 k-means 消融精度表是空的（占位行），这点很诚实，但意味着公开的对比目前只靠 GPQA 主结果支撑。

## 头条证据

**硬件。** 1× H100 80 GB，TP = 1（精度跑）；Qwen3-8B / 32B 和 GLM-4.7 的吞吐扫描在一个小型 H100 集群上跑。

**精度 —— 悬崖和恢复。** Qwen3-4B-Thinking-2507，GPQA，`temp=0.6`，`top_p=0.95`，3 次重复，32 K 上下文：

| 配置                                | GPQA       |
| ----------------------------------- | ---------- |
| BF16 KV                             | 66.67 %    |
| INT4 KV                             | 0 %        |
| **INT4 + BDR（仅 K，ord=128）**      | **65.82 %** |

> [!success] BDR 把模型救了回来
> 原始 INT4 让 Qwen3-Thinking 失效，BDR 把它拉回到距 BF16 1 pp 以内。这一行就是论文的全部。

**吞吐 —— BDR 是免费的，原始 INT4 是坏掉的，BF16 在规模下跌出悬崖。** 论文头版吞吐图（Qwen3-8B，GenAI-Bench）与 Hugging Face static/dynamic FP16、HF KIVI INT4、Kitty-Pro、SGLang BF16 以及 **SGLang INT4 + BDR** 对比：

![Per-GPU 吞吐 vs batch size（Qwen3-8B）：SGLang INT4 + BDR 在任何 batch size 都是最上方曲线（论文 Fig. 2/3）](CN/wiki/llm-inference/saw-int4-figs/throughput-vs-batchsize.png)

这张图有两个值得点出的读法：(1) 在任何 batch size，INT4 + BDR 都达到或超过 SGLang BF16；(2) 旋转代价是隐形的 —— 一旦离开小 batch 区，BDR 曲线就贴在 SGLang INT4 上。连续 batching 加上分页让 INT4 + BDR 把内存节省真正兑现为吞吐。

**四个模型上的延迟-吞吐 Pareto。** 同样的权衡，以 TPS-per-request（延迟）对 per-GPU TPS（吞吐）展示，并发 1 / 8 / 16 / 32 / 256：

![Qwen3-4B/8B/32B 与 GLM-4.7 上 TPS_req vs per-GPU TPS_sys（论文 Fig. 4）](CN/wiki/llm-inference/saw-int4-figs/throughput-latency-curves.png)

四个模型的规律一致：BDR 的 Pareto 曲线在原始 INT4 之上或与其重合；BF16 在每张图右侧高并发处都跌出去 —— 4× 大的 KV 缓存把系统推进了内存压力。

> [!example]- 完整吞吐表格（展开）
> **短上下文**（256 输入 / 1024 输出），Qwen3-8B —— job 级 `output_tps`（汇总 token/s）和 TTFT（ms）：
>
> | 并发 | BF16             | INT4             | INT4 + BDR              |
> | ---: | ---------------: | ---------------: | ----------------------: |
> |  32  | 3,795 / 196      | 3,687 / 225      | 3,689 / 226             |
> |  64  | 5,950 / 369      | 6,371 / 370      | 6,235 / 377             |
> | 128  | 8,410 / 657      | 9,544 / 665      | 9,350 / 655             |
> | 256  | 11,195 / 1,224   | 11,624 / 1,237   | **11,732 / 1,148**      |
>
> **长上下文**（16,384 输入 / 1,024 输出）：
>
> | 并发 | BF16              | INT4              | INT4 + BDR         |
> | ---: | ----------------: | ----------------: | -----------------: |
> |   8  |   414 / 2,636     |   458 / 2,631     |   457 / 2,523      |
> |  16  |   481 / 5,104     |   571 / 4,956     |   568 / 4,875      |
> |  32  |   570 / 18,047    |   618 / 9,568     |   616 / 9,350      |
> |  64  |   471 / 44,798    |   666 / 19,398    |   663 / **18,371** |
> | 128  |   559 / 113,583   |   701 / 57,654    |   701 / **57,054** |
>
> 并发 ≥ 256 / 长上下文时 BDR 甚至在 `output_tps` 与 TTFT 上反超原始 INT4。BF16 在长上下文 conc-128 的 TTFT 113s vs. INT4 / BDR 约 57s，正是 4× 大的 KV 缓存遭遇内存压力时会发生的事。

> [!note] 为什么 BDR 偶尔 *快过* 原始 INT4 的 TTFT
> BDR 内核在与原始 INT4 相同的内存通过中触碰 `cache_k` 并写 scales/zeros buffer，所以*额外*的旋转成本被摊薄进本来就要做的工作里。内核融合把潜在的 5–10 % 开销变成白送 —— 正是论文想要的框架。

精度故事是重头戏：BDR 让一个能用的推理模型从"可用"变成"完全失效"或反过来，而运行时几乎零代价。

---

## 优点与不足

两个最强的点：(1) 这个技术**侵入极小** —— 4 个 env var + 一个 fork 的内核，不需要重训也不需要校准；(2) **融合内核**把本来可能 5–10 % 的开销摊进 INT4 本来就需要的同一次内存通过，让它隐没在测量噪声里。

工作对自身范围的诚实，但限制本身仍值得关注：

- **仅 MHA。** README 明确禁用 MLA —— DeepSeek-V3 风格架构无法直接用 BDR。这个想法是否能迁移过去是开放问题，论文未涉及。
- **后端约束。** 解码用 Triton GQA，预填充用 FA3。往 vLLM 移植不平凡 —— Q 修正必须落到你所用的解码内核里。
- **一个块大小、精度上主要一个模型家族。** Qwen3-4B-Thinking-2507 是唯一的精度结果；吞吐在 Qwen3-8B / 32B 与 GLM-4.7 上扫了，但这些模型上没给精度。README 的 `HADAMARD_ORDER=128` 与 env-var 文档示例的 `16` 之间没解释为什么不同。
- **精度只一个基准。** GPQA 是唯一数字。MMLU、MATH、HumanEval、RULER 都是合理的下一步。
- **空的 k-means 消融表。** BDR 是严格优于 k-means 还是只是相当并未展示；论文宣称更复杂的方法"边际收益"，但支撑这一点的表是占位。
- **未与已发布系统对比。** [[quantization|KIVI]]、NVFP4、FP8 KV、ShadowKV、KVTC —— 都在同一问题空间；SAW-INT4 把自己限制在 BF16 vs. 原始 INT4 vs. BDR。
- **随机 Hadamard，而非学习的。** SpinQuant 表明学习的旋转优于随机 Hadamard（在权重 + 激活量化上）。这里没尝试。
- **Python 级 Hadamard 回退路径更慢。** `SGLANG_FUSE_HADAMARD_INT4_KV=0` 多一次全局内存通过；任何想把这套移到无 Triton 友好注意力后端的服务栈的人会撞到这个开销。

> [!warning] 空的 k-means 消融表是 load-bearing 的
> README 消融矩阵都是占位。"更复杂的方法只给边际增益"这一句正是论文选择更简单的 BDR + per-token INT4 设计的依据 —— 没有支撑表格，这个架构选择只是断言，不是证明。

> [!bug] 文档端口不一致
> `scripts/bdr_smoke_test.py` 的默认端口是 `--port 30000`（与启动样例一致），但 README 的冒烟测试代码片段无故写成 `--port 30001`。修起来很小，但说明开源发布有点仓促。

## 这意味着什么

更大的教训和 [[paged-attention|PagedAttention]] 或 [[sglang|RadixAttention]] 一致：**推理优化的正确粒度是内核，不是模型**。SAW-INT4 与其说是提了一个新量化方案，不如说是证明了"**普通的** per-token INT4 就够了"，前提是你在内核层面做对一件事 —— 在量化前旋转，按分页布局能吞下的块大小做。这有两条我会盯着的推论：

1. **块对角 Hadamard 会扩散开来。** 它小到可以白送给任何 INT4 KV 路径，数学是熟知的（FWHT 自 1970 年代就是教科书内容），内核工作 SGLang fork 已经做得差不多。预期 [[vllm|vLLM]] 和 TensorRT-LLM 会很快接入；同样的招数预计会被试在权重量化（QuaRot 已经证明有效）和 FP8 激活量化上。
2. **"系统感知"框架是更耐用的贡献。** 论文反复指出向量量化与 Hessian-aware 方法在真实服务约束下"增益边际" —— 这其实是在说算法层面的精巧已经撞墙，剩余增益在内核与内存布局里。这个论断之后会持续正确；预计 2026 年更多论文会是关于把已知技术*融合*进正确的内核，而不是发明新技术。

它*不是*什么：MLA 风格模型 [[long-context-serving|长上下文服务]]的解；非 NVIDIA 硬件的答案；也不是"INT4 已解决"的证据。它是一个干净、窄的真实问题上的真实结果。

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
| `third_party/sglang-fast-rotation/python/sglang/srt/mem_cache/memory_pool.py`                | env-var gate（92–98 行）、`set_kv_buffer` 的 BDR 分支(1136–1190 行)、用 `fast_hadamard_transform` 的慢路径参考。                                  |
| `third_party/sglang-fast-rotation/python/sglang/QuantKernel/fused_hadamard_int4_kv.py`       | 融合内核：`_fwht_blocked_segments_tensor`（蝶形）、`_fused_hadamard_int4_set_kv_kernel`（完整 per-token 内核）、launcher、`validate_hadamard_order_for_kv_fuse`。 |
| `third_party/sglang-fast-rotation/python/sglang/srt/layers/attention/triton_backend.py`      | 解码侧 Q 旋转（1042–1091 行）；`fuse_q_hadamard` 标志传给 `decode_attention_fwd_quantized`。                                                       |
| `third_party/sglang-fast-rotation/python/sglang/srt/layers/attention/triton_ops/decode_attention.py` | 接收 `fuse_q_hadamard` 与 `hadamard_order` 并对 Q 应用寄存器内 FWHT 的 GQA 解码内核。                                                              |
| `tools/fit_kv_centroids.py`                                                                  | k-means 质心校准（仅消融）。                                                                                                                       |
| `docs/bdr_env_vars.md`                                                                       | env var 参考与模式矩阵。                                                                                                                            |
| `scripts/bdr_smoke_test.py`                                                                  | 最简 OpenAI-client GPQA 验证。                                                                                                                      |
| `scripts/run_genai_bench_example.sh`                                                         | 吞吐扫描辅助。                                                                                                                                     |
| `scripts/run_primary_eval_matrix.sh`                                                         | 主精度/速度扫描辅助。                                                                                                                              |

## 相关阅读

- [[kv-cache-optimization]] —— KV 缓存压缩的全景（分页、量化、驱逐、卸载）。
- [[quantization]] —— 权重/激活量化（GPTQ、AWQ、SmoothQuant、FP8、NVFP4） —— 与 KV 量化正交，但 Hadamard 旋转在它们之间扮演同一角色（参考 QuaRot、SpinQuant）。
- [[rotation-based-quantization]] —— 完整的 QuIP / QuIP# / QuaRot / SpinQuant / BDR 家族综览，含对比表。
- [[sglang]] —— SAW-INT4 fork 的服务引擎。
- [[long-context-serving]] —— KV 压缩最重要的场景。
- [[paged-attention]] —— BDR 必须兼容的分页 KV 布局。
- [[vllm]] —— 替代服务引擎；BDR 的明显移植目标。
- [[multi-turn-optimization]] —— 多轮 KV 复用与量化质量在前缀缓存层面交互。
