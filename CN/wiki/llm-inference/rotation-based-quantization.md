---
title: "基于旋转的量化（QuIP / QuaRot / SpinQuant / BDR）"
category: llm-inference
tags: [量化, 旋转, hadamard, quip, quarot, spinquant, saw-int4, bdr, 家族综览]
created: 2026-05-06
updated: 2026-05-07
status: mature
---

# 基于旋转的量化

> [!abstract]+ 家族总览
> 一个低位量化技术的家族，共享同一个想法：在量化之前先应用一个**正交变换**把异常值打平。旋转保持 $L_2$ 范数不变，但把能量重新分布到所有维度上，使旋转后的张量对 per-channel 或 per-token scale-and-zero 量化友好得多。家族成员的差异在于：*旋转什么*（权重、激活、KV 缓存）、*如何选择旋转矩阵*（随机 Hadamard、学习的、块对角的）、以及*推理时旋转住在哪里*（吸收进相邻线性层、还是融合进量化内核）。

> [!info] 接下来去哪里看
> - SAW-INT4 / BDR 论文专题精读见 [[saw-int4]]。
> - 更广的量化全景见 [[quantization]]。

---

## 共同的洞察

LLM 的权重与激活都存在**离群通道** —— 少数维度的幅值比其余大 1–2 个数量级。Per-token 或 per-channel 的 scale-and-zero 量化把动态范围分给了离群点，剩余通道分辨率严重不足。在 INT4（16 个等级）下误差严重；在多步推理上，误差跨注意力轮次累积，模型崩溃。

> [!example] 具体失效
> 原始 INT4 KV 下 Qwen3-4B-Thinking-2507 在 GPQA 上从 66.67 % 跌到 0 %。见 [[saw-int4#背景：为什么 INT4 KV 会崩掉推理模型]]。

把向量乘以一个正交矩阵 $R$（满足 $R^\top R = I$）不改变 $L_2$ 范数，但把分量在所有维度上**重新分布**。如果异常值原本集中在少数坐标里，旋转后向量分布更均匀。量化误差大致正比于每个量化组的*值域*，所以分布越平 → 误差越小。

Hadamard 矩阵是 $R$ 的自然选择：它正交，所有元素是 $\pm 1$（旋转只需加减，不需要乘法），且通过 Fast Walsh-Hadamard Transform 可以在 $O(d \log d)$ 时间内做矩阵-向量积。配上 $1/\sqrt{d}$ 归一化它就是等距变换；不归一化则把幅值放大 $\sqrt{d}$。

让这件事变得实用的诀窍是：要么让旋转**被吸收进已有的计算**（推理无额外成本），要么**融合进本来就要跑的内核**。下面家族里两种做法都出现了。

---

## 家族脉络

| 年份 | 方法 | 旋转作用位置 | 旋转类型 | 旋转被什么吸收 | 头条结果 |
|------|------|-------------|---------|---------------|---------|
| 2023 | **QuIP** ([arXiv:2307.13304](https://arxiv.org/abs/2307.13304)) | 权重 | 随机正交（非相干处理） | 修改层的输入/输出 | 首次形式化"旋转使 2-bit 权重可行"。 |
| 2024 | **QuIP#** ([arXiv:2402.04396](https://arxiv.org/abs/2402.04396)) | 权重 | 随机 Hadamard | 同上 | 在旋转后的权重上加 lattice-codebook 向量量化。2-bit 权重的 SOTA。 |
| 2024 | **QuaRot** ([arXiv:2404.00456](https://arxiv.org/abs/2404.00456)) | 权重、激活、**与 KV 缓存** | 随机 Hadamard | **吸收进相邻线性层权重** —— 推理零成本 | NeurIPS 2024。LLaMA-2-70B 上 W4A4KV4，WikiText-2 PPL 增加 ≤0.47，零样本性能保留 99%。 |
| 2024 | **SpinQuant** ([arXiv:2405.16406](https://arxiv.org/abs/2405.16406)) | 权重、激活、KV | **学习的**旋转矩阵（梯度训练） | 同 QuaRot（吸收进线性层权重） | 在难量化的 LLaMA-3 8B 上把"与 FP 的差距"缩小 45%；LLaMA-2-7B W4A4KV4 在零样本推理任务上仅落后 FP 2.9pp。 |
| 2026 | **SAW-INT4 / BDR** ([arXiv:2604.19157](https://arxiv.org/abs/2604.19157)) | 仅 KV 缓存 | 块对角 Hadamard（小固定块） | **融合进 INT4 KV 写入 Triton 内核 + 解码侧 Q 旋转内核** | 把 Qwen3-4B-Thinking 的 GPQA 从原始 INT4 的 0% 恢复到 65.82%。吞吐与原始 INT4 不可区分。见 [[saw-int4]]。 |

跨家族的观察：

- **旋转吸收的位置就是系统贡献。** QuIP 提出算法；QuaRot 通过吸收进线性层权重让推理*零成本*；BDR 通过融合进 Triton 写入内核让 KV 也*零成本*。每一代都找到一个新的"免费"位置安放旋转。
- **随机 vs. 学习。** 学习的旋转（SpinQuant）在精度上优于随机 Hadamard，尤其在难量化模型（LLaMA-3 8B）上，代价是离线校准（最小化量化感知损失）。随机 Hadamard 无需校准，对许多模型已够用。
- **完整 vs. 块对角。** 在头维度上做完整 Hadamard 是 $O(d \log d)$，矩阵数学干净，但对 KV 缓存来说与按头分页的布局和 FA3 风格的融合注意力不兼容。块对角 Hadamard 把混合限制在头维度内的固定大小块里，用一些旋转强度换内核友好性。
- **旋转与量化器正交。** 一旦输入分布被打平，**任何**量化器都更好用 —— GPTQ、AWQ、原始 scale-and-zero、k-means。文献里多数把旋转与简单的 per-token/per-channel scale-zero 叠加，因为旋转已经把硬活干完了。

---

## 各方法旋转的具体位置

**QuIP / QuIP#**：仅权重矩阵。每个线性层 $y = Wx$ 变成 $y = (U^\top W V)(V^\top x)$，其中 $U$、$V$ 是随机正交矩阵；输入侧旋转通过修改前一层吸收掉，输出侧通过修改后一层吸收掉。量化在旋转后的 $W'$ 上做。这有效因为 LLM 权重也存在类似激活的离群点分布。

**QuaRot**：扩展到**激活与 KV 缓存**，方法是在精心选择的位置插入 Hadamard 旋转 —— 在每个 Transformer 块的输入/输出、RMSNorm 边界附近、注意力内部。旋转的位置安排好后，运行时被吸收进相邻线性层权重时正好相互抵消。结果是 LLaMA-2 上的 `W4A4KV4`（4-bit 权重 + 4-bit 激活 + 4-bit KV），质量损失极小。

**SpinQuant**：插入点与 QuaRot 相同，但旋转矩阵 $R_1, R_2, \ldots$ 不再是随机 Hadamard —— 它们是一个**离线优化问题**的参数，在校准集上最小化旋转后网络的量化误差。学到的旋转近似正交，但比随机 Hadamard 多一些结构。

**SAW-INT4 / BDR**：仅旋转 **KV 缓存**沿头维度，固定块大小（16 或 128）。旋转融合进写入分页 INT4 KV 缓存的 Triton 内核（`quantized_set_kv_int4_hadamard_fused_triton`）；同样的块对角 Hadamard 在 GQA 解码内核内对 Q 应用。不影响权重与激活，无离线校准。内核走读见 [[saw-int4#融合 Triton 内核内部]]。

---

## 实践指南

| 你想要... | 用 |
|----------|-----|
| 无校准数据的 INT4 权重 | QuIP#（或 AWQ，如果觉得随机 Hadamard 复杂度不值） |
| LLaMA-2 系列上 INT4 权重 + INT4 激活 | QuaRot |
| 在难量化模型上把 INT4 W+A 推到最高精度 | SpinQuant（接受离线校准成本） |
| 今天就在服务系统上把 KV 缓存压到 INT4 | SAW-INT4 / BDR（仅 MHA，SGLang fork） |
| 同一栈中组合：INT4 权重 + INT4 激活 + INT4 KV | QuaRot（三者一次到位）或 QuaRot/SpinQuant 用于 W+A + BDR 用于 KV（但要小心 —— 两者都旋转 KV，必须二选一） |
| 想保持对非 NVIDIA 硬件的可移植性 | QuaRot/SpinQuant（旋转在 PyTorch 里，不在 Triton 里）；BDR 当前需要 Triton GQA 解码 |

> [!warning] 陷阱：不要把 BDR 的 KV 旋转叠在 QuaRot 的 KV 旋转之上
> QuaRot 已经旋转过 KV 并吸收进相邻权重；再叠 BDR 等于旋转两次。如果你既要权重 + 激活量化也要 KV 量化，要么 (a) 单独用 QuaRot / SpinQuant 处理三者，要么 (b) QuaRot 仅用于 W+A（KV 走非量化路径），然后在 KV 上叠 BDR —— 这意味着 QuaRot 的 W+A 旋转*不能*触碰 KV。

---

## 开放问题

- **MLA 架构。** QuaRot、SpinQuant、BDR 全部针对 MHA/GQA。DeepSeek-V3 风格的 MLA 存 K 的低秩投影，旋转必须与上投影交互 —— 至今没有公开方法干净地处理这一点。
- **学习的块对角。** SpinQuant 学习了完整旋转；BDR 用随机块对角 Hadamard。一个学习的块对角旋转（小到对内核友好、又比随机更聪明）是显然的下一步。
- **非 2 的幂头维度的旋转。** 现有方法都假设 2 的幂块大小。Llama 风格的 128 维头没问题；不太常见的维度（如某些研究模型的 80）则不行。
- **旋转 + 稀疏。** 旋转打平分布，让 Top-K 稀疏注意力效果变差（现在每个通道都重要）。这层交互未被研究。
- **FP4 / NVFP4 + 旋转。** [[quantization|NVFP4]] 通过两级缩放有自己的离群缓解；NVFP4 下旋转是否还有用是开放问题。

---

## 相关阅读

- [[saw-int4]] —— KV 缓存的块对角 Hadamard 旋转（完整论文精读）。
- [[quantization]] —— 更广的量化全景（GPTQ、AWQ、FP8、NVFP4、SmoothQuant）。
- [[kv-cache-optimization]] —— KV 缓存优化栈；基于旋转的 KV 是众多技术之一。
- [[sglang]] —— SAW-INT4 fork 所在的服务引擎。
- [[paged-attention]] —— 约束 BDR 块对角设计的分页 KV 布局。

---

## 参考文献

- Chee et al., "QuIP: 2-Bit Quantization of Large Language Models with Guarantees," NeurIPS 2023. [arXiv:2307.13304](https://arxiv.org/abs/2307.13304)
- Chee et al., "QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks," ICML 2024. [arXiv:2402.04396](https://arxiv.org/abs/2402.04396)
- Ashkboos et al., "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs," NeurIPS 2024. [arXiv:2404.00456](https://arxiv.org/abs/2404.00456)
- Liu et al., "SpinQuant: LLM Quantization with Learned Rotations," 2024. [arXiv:2405.16406](https://arxiv.org/abs/2405.16406)
- Jia et al., "SAW-INT4: System-Aware 4-Bit KV-Cache Quantization for Real-World LLM Serving," 2026. [arXiv:2604.19157](https://arxiv.org/abs/2604.19157)
