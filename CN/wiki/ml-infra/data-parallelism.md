---
title: "数据并行（DP）：DDP、ZeRO、FSDP、HSDP、DWDP —— 以及它们争夺的内存账本"
category: ml-infra
tags: [数据并行, ddp, zero, fsdp, hsdp, dwdp, 优化器状态, 混合精度, 显存, 概念]
created: 2026-05-22
updated: 2026-05-22
status: growing
---

# 数据并行（DP）：DDP、ZeRO、FSDP、HSDP、DWDP —— 以及它们争夺的内存账本

> [!info] 页面范围
> 概念综览：**DP 家族** —— DDP、ZeRO 三个阶段、PyTorch FSDP（1 与 2 两代）、HSDP，以及推理侧的近亲 DWDP。也讲清楚它们都在打的那本内存账：优化器状态是什么、梯度参数量为什么跟权重一样、为什么 FFN 在 Transformer 里占大头、"activation" 到底指什么。

---

## 摘要（2 分钟读完这一节就够）

**什么是 DP**。Data Parallelism = "**把 batch 切到不同 worker、复制 model、每 step 用 AllReduce 同步梯度**"。DDP 是经典实现。DP 是并行的**一根轴** —— 跟 Tensor Parallel (TP)、Pipeline Parallel (PP)、Context Parallel (CP)、Expert Parallel (EP) 正交。

**DP 家族 —— 用通信换内存的几种变体**。ZeRO 和 FSDP **仍然是 DP** —— batch 切法没变 —— 它们只是把 **模型状态**（参数 / 梯度 / 优化器状态）按 DP rank 切片，消除冗余复制。每个变体多切一类：

| 变体 | 切 batch | 切 opt states | 切梯度 | 切参数 |
| ---- | :------: | :-----------: | :----: | :----: |
| **DDP** | ✓ | ✗ | ✗ | ✗ |
| **ZeRO-1** | ✓ | ✓ | ✗ | ✗ |
| **ZeRO-2 / FSDP `SHARD_GRAD_OP`** | ✓ | ✓ | ✓ | ✗ |
| **ZeRO-3 / FSDP `FULL_SHARD`** | ✓ | ✓ | ✓ | ✓ |

**头条收益**。100B 参数 + Adam 混合精度，8 卡 DP 下每卡显存从 **1600 GB（DDP）** → **200 GB（ZeRO-3）**。代价：每层多一次 AllGather 通信。

**为什么这重要**。没有 ZeRO/FSDP，单张 80GB GPU 上训 7B 以上就不可能。有了 ZeRO-3 / FSDP，单台 8×H100 节点能训 70B 级别。有 ZeRO-Infinity（CPU + NVMe offload），单节点能训 1T 级别，只是慢点。

---

# 深度部分（往下展开细节）

## 背景：现代训练的内存账本

混合精度 Adam 训练，DDP 下每卡每个参数要占 **16 字节**：

| 组件 | 精度 | bytes / param | 备注 |
| ---- | ---- | -------------:| ---- |
| Working weight（工作权重） | FP16 / BF16 | 2 | forward/backward 用 |
| Gradient（梯度） | FP16 / BF16 | 2 | 每个参数一个（数量跟权重一样） |
| **Master weight（FP32 主权重）** | FP32 | **4** | 持久 FP32 副本 —— update 在它上面发生 |
| **Adam first moment $m$** | FP32 | **4** | $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$ |
| **Adam second moment $v$** | FP32 | **4** | $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$ |
| **合计** | | **16** | 其中 **12 = 优化器状态** |

加粗那三行 ("optimizer states") = **12 bytes/param = 总显存的 75%**。这正是 ZeRO 攻击的目标。

100B 模型：100B × 16 = **每卡 1.6 TB**。今天没有 GPU 能装下。没有 DP 内部分片的话，单张 80GB H100 上训不动 7B 以上。

来源：[ZeRO paper (arXiv:1910.02054)](https://arxiv.org/abs/1910.02054) §3.2。

### 为什么梯度的参数量 = 权重参数量

对每个参数 scalar $\theta_i$，backprop 算出一个梯度 scalar $\partial L / \partial \theta_i$。Loss $L$ 是标量；对张量的梯度跟张量*形状相同*：

$$
\frac{\partial L}{\partial W} \in \mathbb{R}^{d_1 \times d_2}, \quad \text{其中 } W \in \mathbb{R}^{d_1 \times d_2}.
$$

几何直觉：参数空间是 $\Psi$ 维（$\Psi$ = 总参数量）。Loss 是这个 $\Psi$ 维空间上的标量函数 $L: \mathbb{R}^\Psi \to \mathbb{R}$。它的梯度 $\nabla L \in \mathbb{R}^\Psi$ 住在同一空间。**按定义分量数相同**。

dtype 可以不同（混合精度下 FP16 weight + FP16 grad，有时 FP32 grad 累加），但**数量严格相等**。

### 各种优化器的状态对比

| 优化器 | 状态 | bytes/param (FP32) | 备注 |
| ------ | ---- | ------------------:| ---- |
| **SGD**（无 momentum） | — | 0 | 无状态 |
| **SGD with momentum** | velocity | 4 | 一个 EMA |
| **Adam / AdamW** | master + $m$ + $v$ | **12** | LLM 默认 |
| **Lion**（[Chen et al. 2023](https://arxiv.org/abs/2302.06675)） | momentum 一份 | 4 | Adam 的 1/3 |
| **Adafactor** | factored $v$（行 + 列向量） | ~5 | Google 的省内存方案 |
| **8-bit Adam**（[Dettmers et al. 2022](https://arxiv.org/abs/2110.02861)） | INT8 量化 $m$, $v$ | ~6 | Adam 的一半 |

PyTorch AdamW 维护 `state['exp_avg']`（$m$）和 `state['exp_avg_sq']`（$v$）：
[`torch/optim/adamw.py`](https://github.com/pytorch/pytorch/blob/main/torch/optim/adamw.py)

bitsandbytes 8-bit Adam：
[`bitsandbytes/optim/adamw.py`](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/bitsandbytes/optim/adamw.py)

---

## DP 家族详解

### DDP —— 分布式数据并行

Baseline。每张卡持有完整模型副本。Forward + backward 本地跑各自的 batch slice；backward 末尾 AllReduce 梯度；optimizer step 在每张卡上各跑一遍。

- 代码：[`torch/nn/parallel/distributed.py`](https://github.com/pytorch/pytorch/blob/main/torch/nn/parallel/distributed.py)
- 文档：[PyTorch DDP notes](https://docs.pytorch.org/docs/stable/notes/ddp.html)
- 每 step 通信：**1× AllReduce(梯度)**，就这一次
- 每卡显存：Adam 混合精度下 $16 \Psi$

### ZeRO-1 —— 分片优化器状态（$P_{os}$）

把占 75% 的 optimizer states 按 $N_d$ 个 DP rank 切。每张卡只 update 自己"持有" optimizer state 那部分参数；step 之后做一次 **AllGather** 把更新过的权重广播让大家工作副本保持同步。

- 每卡显存：$4\Psi + 4\Psi + \frac{12\Psi}{N_d}$（权重 + 梯度 + 分片 opt state）
- 通信：AllReduce(梯度) + AllGather(更新后的参数)
- 代码：[`deepspeed/runtime/zero/stage_1_and_2.py`](https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/runtime/zero/stage_1_and_2.py) —— `partition_grads=False` 分支
- Megatron-Core 的 distributed optimizer 本质就是 ZeRO-1：[`megatron/core/optimizer/distrib_optimizer.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/optimizer/distrib_optimizer.py)

### ZeRO-2 —— 再切梯度（$P_{os+g}$）

2 bytes/param 的梯度 buffer 也切了。Reduce-Scatter 替代 AllReduce —— 每张卡最终只拿到自己所属参数的梯度切片。

- 每卡显存：$4\Psi + \frac{2\Psi}{N_d} + \frac{12\Psi}{N_d}$
- 通信：**Reduce-Scatter(梯度) + AllGather(参数)** = 总字节数等同于 DDP 的 AllReduce，只是路由不同
- 代码：同一个文件，`partition_grads=True`

### ZeRO-3 —— 把参数也切了（$P_{os+g+p}$）

最激进。每张卡静态时只持有 $\frac{1}{N_d}$ 的参数。每层 forward 前，**AllGather** 从对等 rank 凑齐这一层的完整参数；算完立刻释放。

- 每卡显存：$\frac{16\Psi}{N_d}$ + 最大层参数的瞬时 buffer
- 通信：**每层 AllGather**（forward 中）+ 对称的 backward + Reduce-Scatter(梯度)。约 1.5× DDP 字节数
- 代码：[`deepspeed/runtime/zero/stage3.py`](https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/runtime/zero/stage3.py) —— 看 `_pre_forward_module_hook`（进入时 AllGather）和 `_post_forward_module_hook`（退出时 partition）。这就是让账本成立的核心 trick

### ZeRO offload 系列

| 变体 | 卸载什么 | 卸到哪 | 论文 |
| ---- | -------- | ------ | ---- |
| **ZeRO-Offload** | optimizer states + master weights | CPU DRAM | [arXiv:2101.06840](https://arxiv.org/abs/2101.06840) |
| **ZeRO-Infinity** | 参数 + 梯度 + opt states + activations | CPU + **NVMe** | [arXiv:2104.07857](https://arxiv.org/abs/2104.07857) |
| **ZeRO++** | 量化 AllGather（INT8 weights）+ 分层 sharding | — | [arXiv:2306.10209](https://arxiv.org/abs/2306.10209) |

ZeRO-Infinity 三级存储：**GPU HBM → CPU DRAM → NVMe SSD**，由 [`deepspeed/runtime/swap_tensor/`](https://github.com/microsoft/DeepSpeed/tree/master/deepspeed/runtime/swap_tensor) 负责 I/O。

### FSDP —— PyTorch 原生版

FSDP 是 PyTorch 把 ZeRO-3 思想吸进框架的产物。**实现两代**（"FSDP1" 和 "FSDP2"）；都还能调，但框架在推大家全迁 FSDP2。

> **没有官方 "FSDP3"**，截至 2026-05。未来路线在 [TorchTitan](https://github.com/pytorch/torchtitan) 讨论，但没有新一代命名。

#### FSDP1（PyTorch 1.11+，2022）

- 论文：[*PyTorch FSDP* (arXiv:2304.11277)](https://arxiv.org/abs/2304.11277)
- 内部数据结构：`FlatParameter` —— 一个 module 的所有参数被 flatten 拼成 1D 张量，再按 rank chunk
- 通信粒度：一个 `FlatParameter` 一次 AllGather（粒度粗，kernel 数少）
- 问题：跟 TP / LoRA / 部分参数冻结 等组合不好（flat 表示掩盖了 per-param 语义）；混合精度配置粒度粗；`torch.compile` 集成有挑战
- 代码（仍可用，deprecation 路线）：[`torch/distributed/fsdp/fully_sharded_data_parallel.py`](https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/fully_sharded_data_parallel.py)
- FlatParam 实现：[`torch/distributed/fsdp/_flat_param.py`](https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/_flat_param.py)

#### FSDP2（PyTorch 2.4+，2024）

完全重写。**Per-parameter sharding** 用 [`DTensor`](https://github.com/pytorch/pytorch/tree/main/torch/distributed/tensor) —— 每个 `nn.Parameter` 独立分片。

- API 改了：从类包裹 (`FSDP(model, ...)`) 改成函数式 (`fully_shard(model, ...)`)
- 跟 TP 原生组合（都基于 DTensor）→ 2D 并行一行配置
- 混合精度可 per-parameter 配
- PEFT 友好（LoRA、冻参）—— 直接 `requires_grad=False`
- 设计 RFC：[pytorch/pytorch#114299](https://github.com/pytorch/pytorch/issues/114299)
- 教程：[PyTorch FSDP2 tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- 代码：
  - API：[`torch/distributed/_composable/fsdp/_fully_shard/_fully_shard.py`](https://github.com/pytorch/pytorch/blob/main/torch/distributed/_composable/fsdp/_fully_shard/_fully_shard.py)
  - 参数分片：[`torch/distributed/_composable/fsdp/_fully_shard/_fsdp_param.py`](https://github.com/pytorch/pytorch/blob/main/torch/distributed/_composable/fsdp/_fully_shard/_fsdp_param.py)
  - Pre/post-forward hooks（AllGather + partition 机制）：[`_fsdp_param_group.py`](https://github.com/pytorch/pytorch/blob/main/torch/distributed/_composable/fsdp/_fully_shard/_fsdp_param_group.py)

#### FSDP1 vs FSDP2 速查表

| 维度 | FSDP1 | FSDP2 |
| ---- | ----- | ----- |
| 内部表示 | FlatParameter（1D flatten） | DTensor（per-param） |
| API | `FSDP(model, ...)` | `fully_shard(model, ...)` |
| 与 TP 组合 | 通过 HSDP/2D mesh 手工组装 | 原生 DTensor mesh |
| LoRA / 部分冻参 | 难 | 容易 |
| 混合精度粒度 | per-FlatParameter | per-Parameter |
| `torch.compile` | 部分支持 | 一等公民 |
| 状态 | deprecation 路线 | **PyTorch 2.5+ 默认** |

PyTorch 2.5 release：[官方 blog](https://pytorch.org/blog/pytorch2-5/)（搜 "FSDP2"）。

#### FSDP ShardingStrategy 对应 ZeRO 阶段

| ZeRO stage | FSDP `ShardingStrategy` | 备注 |
| ---------- | ----------------------- | ---- |
| ZeRO-1 | (没有精确对应) | 通过混合阶段配置最接近 |
| ZeRO-2 | `SHARD_GRAD_OP` | 切 opt states + 梯度 |
| ZeRO-3 | `FULL_SHARD` | 都切 |
| — | `NO_SHARD` | DDP |
| — | `HYBRID_SHARD` | 节点内 full shard、节点间 replicate（HSDP）|

Enum 源：[`torch/distributed/fsdp/api.py`](https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/api.py)。

### HSDP —— 混合分片数据并行

**节点内分片**（NVLink，~600 GB/s），**节点间复制**（IB，~25 GB/s）。让 AllGather / Reduce-Scatter 流量留在快的节点内 fabric 上，节点间只跑 AllReduce。

- 前提：模型能装进单节点（8 张 GPU × 80 GB = 640 GB）
- FSDP2 配置：传入 2D `DeviceMesh`，维度 `("replicate", "shard")`
- 教程：[FSDP advanced tutorial 的 HSDP 部分](https://docs.pytorch.org/tutorials/intermediate/FSDP_advanced_tutorial.html)
- DeepSpeed 的对应物是 ZeRO++ 的 **hpZ**（hierarchical partitioning）—— 见 [`deepspeed/runtime/zero/config.py`](https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/runtime/zero/config.py) 的 `zero_hpz_partition_size`

### DWDP —— 推理侧的近亲

不是训练相关，但概念上邻近。**DWDP（Distributed Weight Data Parallelism）** 是给 NVL72 级硬件上 MoE 模型设计的 *推理* 并行策略，借用了 ZeRO/FSDP "切权重、按需 fetch" 的思路。

- 论文：[arXiv:2604.01621](https://arxiv.org/abs/2604.01621)（Li et al., 2026-04, NVIDIA）
- 关键差异：去掉了 EP MoE 每层的 inter-rank 集体同步，让每张 GPU 独立推进，缺失的 expert weight 通过 peer-to-peer NVLink 拉
- 目标：GB200 NVL72 上的 MoE 推理；在 DeepSeek-R1 上 +8.8% TPS/GPU
- **不是训练侧** —— 推理时根本没有 optimizer states 和梯度可切。切的是 MoE expert weights，forward 时按需 fetch

---

## 推理时的 DP

推理时没有 optimizer states 和梯度。DP 相关的问题变成 "如何在 GPU 间分发 batch 而不复制大权重"。稠密模型上普通 DP 就行。MoE 模型上的标准模式是 **DP attention + EP MoE**。

### DP attention + EP MoE

MoE 推理的生产默认（DeepSeek-V3 + vLLM + SGLang 都在用）：

| 层类型 | 并行方式 | 为什么 |
| ------ | -------- | ------ |
| Attention | **DP**（每张卡持有完整 attention 权重，处理自己的 batch slice） | Attention 权重相对 MoE 小；DP 避免 TP 切 KV cache 的复杂度 |
| MoE FFN | **EP**（experts 切到不同 rank） | MoE 总权重太大不能复制（DeepSeek-V3：671B 总，37B 激活） |

一层内的 per-step 流程：

```
batched tokens（DP 分散在各 rank）
    │
    ├── DP attention   ── 每 rank 本地做自己 token 的 attention
    │
    ├── router          ── 决定每个 token 路由到哪些 expert
    │
    ├── AllToAll dispatch ── 把 token shuffle 到持有目标 expert 的 GPU
    │
    ├── EP MoE          ── 每 rank 在被路由过来的 token 上跑自己的 expert
    │
    └── AllToAll combine  ── 把输出 shuffle 回原 rank
```

代码参考：
- vLLM：[`vllm/distributed/parallel_state.py`](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/parallel_state.py) 看 `ep_group`，[`vllm/v1/worker/gpu_model_runner.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu_model_runner.py) 看 `dp_size` 处理
- SGLang：[`sglang/srt/layers/moe/`](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/layers/moe)
- DeepSeek-V3 论文描述这套架构：[arXiv:2412.19437](https://arxiv.org/abs/2412.19437)

DWDP 攻击的就是这套里的 AllToAll 同步。

---

## 显存到底住在哪：Transformer block 解剖

要知道 DP-sharding *值得* 切什么，先得知道参数量住在 Transformer 的哪里。

### Attention vs FFN —— 分工

每个 transformer block 有两个 sub-layer，shape 都是 `[B, S, H] → [B, S, H]`，都包在 residual + LayerNorm 里：

| 属性 | Attention | FFN |
| ---- | --------- | --- |
| 信息流向 | **沿 sequence 维度混合**（token 之间） | **沿 hidden 维度展开**（每 token 独立） |
| 数学 | $\text{softmax}(QK^T/\sqrt{d}) V$ | $\text{down}(\sigma(\text{gate}(x)) \odot \text{up}(x))$ |
| 角色比喻 | **"通信"** —— token 之间交换信息 | **"思考"** —— 每 token 内部加工 |
| 复杂度 | $O(S^2 H)$ | $O(S \cdot H \cdot H_{\text{ffn}})$ |
| 显存增长 | $O(S^2)$ peak（FlashAttention 降到 $O(S)$） | $O(S)$ |

两者交替起作用：attention 把 context 拉进来，FFN 加工 per-token 结果。

### Activation —— 两个不同含义

中英文里 "activation" 都歧义：

- **激活函数**（单数）：FFN *内部* 的非线性。Llama / Mistral / DeepSeek 用 **SwiGLU**；老 GPT-2/3 用 GeLU；BERT 用 GeLU；Gemma 用 GeGLU。没它两个 linear 就坍缩成一个
- **激活值**（复数名词）：forward 过程中存的中间张量 —— 每层的输入/输出、attention scores、FFN 中间的 `[B, S, H_ffn]` 张量（最大的单个）。这些是 *数据*，不是权重：取决于输入 batch、backward 链式法则要它们。**这就是 activation checkpointing 在 trade-off 的东西** —— 不存，backward 时重算

内存账本切分：

```
每卡总显存 =
    Weights        （取决于模型大小；静态；DP/FSDP 切的就是它）
  + Gradients      （数量等同于权重；step 内静态）
  + Optimizer state（取决于优化器；静态）
  + Activations    （取决于 batch × seq；forward 时存活；常常是单个最大块）
  + Buffers / workspace（小）
```

ZeRO/FSDP 攻前三项。Activation 由 activation checkpointing 单独管 —— 见 [`torch.distributed.algorithms._checkpoint.checkpoint_wrapper`](https://github.com/pytorch/pytorch/tree/main/torch/distributed/algorithms/_checkpoint) 和 [`torch.utils.checkpoint`](https://github.com/pytorch/pytorch/blob/main/torch/utils/checkpoint.py)。

### 为什么 attention 参数小、FFN 参数大

Hidden dim $H$、Q 头数 $n_q$、KV 头数 $n_{kv}$、FFN 宽度 $H_{\text{ffn}}$。

**Attention**（多头 + GQA 比率 $r = n_q / n_{kv}$）：

| 矩阵 | 形状 | 参数 |
| ---- | ---- | ----:|
| $W_Q$ | $H \times H$ | $H^2$ |
| $W_K$ | $H \times H/r$ | $H^2/r$ |
| $W_V$ | $H \times H/r$ | $H^2/r$ |
| $W_O$ | $H \times H$ | $H^2$ |
| **合计** | | $H^2(2 + 2/r)$ |

完整 MHA（$r=1$）：$4 H^2$。Llama 3 70B GQA（$r=8$）：$2.25 H^2$。

**FFN**（SwiGLU，三个矩阵 —— 多了 gate）：

| 矩阵 | 形状 | 参数 |
| ---- | ---- | ----:|
| $W_{\text{gate}}$ | $H \times H_{\text{ffn}}$ | $H \cdot H_{\text{ffn}}$ |
| $W_{\text{up}}$ | $H \times H_{\text{ffn}}$ | $H \cdot H_{\text{ffn}}$ |
| $W_{\text{down}}$ | $H_{\text{ffn}} \times H$ | $H \cdot H_{\text{ffn}}$ |
| **合计** | | $3 H \cdot H_{\text{ffn}}$ |

现代 $H_{\text{ffn}} \approx 8H/3$（保持总 FLOPs 跟老式 2 矩阵 MLP $4H$ 宽度接近），所以 FFN $\approx 8 H^2$。

**比例**：

| 配置 | Attention | FFN | FFN / Attention |
| ---- | --------- | --- | ---------------:|
| MHA + 老式 GPT FFN | $4 H^2$ | $8 H^2$ | **2×** |
| MHA + SwiGLU | $4 H^2$ | $8 H^2$ | **2×** |
| GQA(r=8) + SwiGLU | $2.25 H^2$ | $8 H^2$ | **3.6×** |

**FFN 占 2-4 倍**。GQA 让差距更大（attention 缩水，FFN 不变）。

### 一个真实的 100B 级别拆解：Llama 3.1 70B

配置（来自 [HF model card](https://huggingface.co/meta-llama/Llama-3.1-70B/blob/main/config.json)）：

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

**每层**（head_dim = 128，GQA 1:8 所以 KV 输出维度 = 8 × 128 = 1024）：

| 块 | 矩阵 | 形状 | 参数 |
| -- | ---- | ---- | ----:|
| Attention | $W_Q$ | $8192 \times 8192$ | 67.1 M |
|  | $W_K$ | $8192 \times 1024$ | 8.4 M |
|  | $W_V$ | $8192 \times 1024$ | 8.4 M |
|  | $W_O$ | $8192 \times 8192$ | 67.1 M |
|  | input layernorm | $8192$ | 8 K |
|  | **Attention 小计** | | **151.0 M** |
| FFN | $W_{\text{gate}}$ | $8192 \times 28672$ | 234.9 M |
|  | $W_{\text{up}}$ | $8192 \times 28672$ | 234.9 M |
|  | $W_{\text{down}}$ | $28672 \times 8192$ | 234.9 M |
|  | post-attn layernorm | $8192$ | 8 K |
|  | **FFN 小计** | | **704.6 M** |
| **每层总计** | | | **855.6 M** |

**整模型**：

| 组件 | 参数 | 占比 |
| ---- | ----:| ----:|
| 80 × Attention | 12.1 B | **17.1 %** |
| 80 × FFN | 56.4 B | **79.9 %** |
| Input embedding（$128256 \times 8192$） | 1.05 B | 1.5 % |
| LM head（$8192 \times 128256$，Llama 3 是 untied） | 1.05 B | 1.5 % |
| 最后 RMSNorm | 8 K | ~0 % |
| **总计** | **70.55 B** | 100 % |

**FFN 占了参数预算的 80%**。这就是为什么 DP-sharding 对 FFN 矩阵收益最大 —— 它们是被复制的主体。也是为什么 MoE 设计（DeepSeek-V3、Mixtral）都瞄准 FFN —— 把它通过 expert routing 变稀疏，比改 attention 性价比高得多。

### MoE 把这个图景翻过来：DeepSeek-V3

DeepSeek-V3 是 **671 B 总参 / 37 B 激活**（[arXiv:2412.19437](https://arxiv.org/abs/2412.19437)、[config.json](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json)）：

- $H = 7168$，60 层
- 每层：1 个 shared expert + **256 个 routed expert**，每 token top-8 激活
- 每个 expert 是个小 FFN（$H \to H_{\text{ffn},e} \to H$，$H_{\text{ffn},e} = 2048$）
- 几乎整个 671 B 参数都住在 256 个 routed expert 池里；推理时每 token 只走 8 个

这就是为什么 MoE 推理 FFN 部分要用 **EP** 而不是 DP —— 671 B 权重没法复制，但可以 per-token 路由到对应分片。

---

## DP vs 其它并行轴

DP 是 3D / 5D 并行里的**一根轴**。跟 TP / PP / CP / EP 正交，可以叠加。

```
总 GPU = DP × TP × PP × CP × EP
```

| 轴 | 切什么 | 通信 | 代码参考 |
| -- | ------ | ---- | -------- |
| **DP** | batch（外加可选的模型状态） | AllReduce / Reduce-Scatter / AllGather（梯度） | 本页 |
| **TP** | 层内矩阵 | 每层 AllReduce(output) | [Megatron-LM TP](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/tensor_parallel) |
| **PP** | 跨 stage 的层 | P2P（跨 stage 边界的 activation） | [Megatron-LM PP](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/pipeline_parallel) |
| **CP** | sequence 维度 | Ring P2P 或 AllToAll(KV) | [[ring-attention]]、[[deepspeed-ulysses]] |
| **EP** | MoE expert | AllToAll(tokens) | [vLLM EP](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/parallel_state.py) |
| **SP**（Megatron 版） | LayerNorm/Dropout activations | 额外 AllGather/ReduceScatter | [Megatron-LM SP](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/layers.py) |

**常见误解**："FSDP 是 model parallel"。错的 —— FSDP 是 DP。每张卡仍然跑整个模型 forward，参数只是 on-the-fly AllGather 凑齐。TP 才是真 model parallel —— 每张卡只跑自己那块矩阵切片。

外部对比链接：[PyTorch TP 文档](https://docs.pytorch.org/docs/stable/distributed.tensor.parallel.html) vs [FSDP2 文档](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)。

Megatron-Core 的 `parallel_state.py` 是叠加这些轴的标准参考：[link](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py) —— DP 是其它轴配完之后剩下的维度。

---

## 怎么选

| 情况 | 推荐 |
| ---- | ---- |
| 模型能装进单卡（含 optimizer states） | DDP —— 最简单最快 |
| 模型 ≤ 70B，单台 8×80GB 节点装得下 | **HSDP（FSDP2 + 2D mesh）** —— 节点内 sharding，节点间 replicate |
| 模型 > 100B，单节点装不下 | **FSDP2 FULL_SHARD**（= ZeRO-3）跨集群 |
| 极端规模，连集群都装不下参数 | **ZeRO-Infinity**（CPU + NVMe offload）—— 慢但能跑 |
| 已经在 Megatron 栈上 | Megatron-Core 的 [`distributed_optimizer`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/optimizer/distrib_optimizer.py)（Megatron 版本的 ZeRO-1） |
| LLM 的 RL 训练（PPO/GRPO/DAPO） | **FSDP2** —— veRL / NeMo-RL / OpenRLHF 都默认用它；这种工作负载 PyTorch 原生集成比 DeepSpeed 好。看 veRL 的 [`fsdp_workers.py`](https://github.com/volcengine/verl/blob/main/verl/workers/fsdp_workers.py) |
| NVL72 上的 MoE 推理 | **DWDP**（推理近亲）或 DP attention + EP MoE |

---

## 参考文献

**主要论文**：
- [ZeRO (arXiv:1910.02054)](https://arxiv.org/abs/1910.02054) —— 引入 stage 1/2/3 的奠基论文
- [ZeRO-Offload (arXiv:2101.06840)](https://arxiv.org/abs/2101.06840) —— optimizer states CPU offload
- [ZeRO-Infinity (arXiv:2104.07857)](https://arxiv.org/abs/2104.07857) —— NVMe offload，单节点 1T
- [ZeRO++ (arXiv:2306.10209)](https://arxiv.org/abs/2306.10209) —— 量化 AllGather + 分层 sharding
- [PyTorch FSDP (arXiv:2304.11277)](https://arxiv.org/abs/2304.11277) —— FSDP1 设计
- [DWDP (arXiv:2604.01621)](https://arxiv.org/abs/2604.01621) —— NVL72 上 MoE 的推理近亲

**框架**：
- DeepSpeed：[github.com/microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed)
- PyTorch FSDP2：[`torch.distributed._composable.fsdp`](https://github.com/pytorch/pytorch/tree/main/torch/distributed/_composable/fsdp)
- Megatron-Core distributed optimizer：[`megatron/core/optimizer/distrib_optimizer.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/optimizer/distrib_optimizer.py)
- TorchTitan（PyTorch 原生预训练参考实现）：[github.com/pytorch/torchtitan](https://github.com/pytorch/torchtitan)

**教程**：
- [PyTorch FSDP2 tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [PyTorch HSDP tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_advanced_tutorial.html)
- [DeepSpeed ZeRO tutorial](https://www.deepspeed.ai/tutorials/zero/)

## 相关阅读

- [[distributed-training]] —— 更广义的分布式训练综述（3D 并行、混合精度、容错）
- [[ring-attention]]、[[deepspeed-ulysses]] —— Context Parallelism 的两条主流（CP 轴，跟 DP 正交）
- [[parallelism-strategies-deep-dive]] —— 所有并行轴的完整规范参考
- [[training-frameworks]] —— Megatron-LM、DeepSpeed、FSDP、NeMo —— 实际训练栈
- [[grpo]]、[[ppo-for-llm]]、[[rlhf-overview]] —— 底层用 FSDP2 的 RL 训练栈
- [[das-spec-rl]]、[[aurora]] —— 投机解码（推理侧并行，跟 DP 不同）
