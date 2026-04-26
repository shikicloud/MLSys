---
title: "训练框架：Megatron-LM、DeepSpeed、FSDP"
category: ml-infra
tags: [megatron-lm, deepspeed, fsdp, torchtitan, 训练, 框架, megatron-core, nemo]
created: 2026-04-13
updated: 2026-04-14
status: mature
---

# 训练框架：Megatron-LM、DeepSpeed、FSDP

---

## 1. 概述

大模型训练框架的核心使命是：**将 [[distributed-training]] 中的各种并行策略和优化技术整合为可用的工程系统**。从 2020 年 GPT-3 时代的"每个团队自建轮子"到 2025-2026 年，框架生态逐渐收敛为几个主流选项：

```
                    LLM 训练框架生态 (2025-2026)
┌──────────────────────────────────────────────────────────┐
│                                                          │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │ Megatron-LM │  │  DeepSpeed   │  │ PyTorch FSDP2  │  │
│  │ /Megatron-  │  │  (Microsoft) │  │ + TorchTitan   │  │
│  │  Core       │  │              │  │ (Meta/PyTorch) │  │
│  │ (NVIDIA)    │  │              │  │                │  │
│  └──────┬──────┘  └──────┬───────┘  └───────┬────────┘  │
│         │                │                   │           │
│    最大吞吐量       内存效率/易用         原生PyTorch      │
│    TP+PP+SP+EP     ZeRO 1-3 + 卸载     FSDP+TP+PP       │
│                                                          │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  上层封装 / 其他框架                                  │ │
│  │  NeMo (NVIDIA)  |  Composer (Databricks/MosaicML)   │ │
│  │  Nanotron (HF)  |  Colossal-AI  |  Fairscale (Meta)│ │
│  └─────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

**选择框架的核心考量**：
1. **训练吞吐量** — 单位时间处理多少 token
2. **显存效率** — 能在给定硬件上训练多大的模型
3. **并行支持** — 支持哪些并行维度及其组合
4. **易用性** — 学习曲线、调试难度、社区支持
5. **可组合性** — 与其他 PyTorch 生态工具的兼容性

---

## 2. Megatron-LM / Megatron-Core

### 2.1 架构概述

Megatron 是 NVIDIA 开发的大模型训练框架，分为两个层次：

```
┌──────────────────────────────────────────────────┐
│                   NeMo Framework                  │
│  (高层 API: 配置驱动, Hydra config, 训练 recipe)   │
├──────────────────────────────────────────────────┤
│                  Megatron-Core                     │
│  (核心库: 并行原语, 分布式策略, 融合算子)            │
├──────────────────────────────────────────────────┤
│              Megatron-LM (原始仓库)                │
│  (训练脚本, 模型定义, 数据处理 pipeline)            │
├──────────────────────────────────────────────────┤
│  PyTorch  |  NCCL  |  CUDA  |  Transformer Engine │
└──────────────────────────────────────────────────┘
```

**Megatron-LM vs Megatron-Core**：

| 维度 | Megatron-LM | Megatron-Core |
|------|-------------|---------------|
| 定位 | 完整训练方案（模型+数据+训练） | 可复用的并行原语库 |
| 仓库 | `NVIDIA/Megatron-LM` | `NVIDIA/Megatron-LM` (子模块) |
| 模型定义 | 内置 GPT/BERT/T5 | 提供可组合的模块 |
| 适用场景 | 直接训练 NVIDIA 支持的模型 | 被其他框架集成 (如 NeMo) |
| API 稳定性 | 较低，频繁重构 | 较高，设计为库 |

### 2.2 核心并行支持

Megatron-LM 支持最全面的并行维度组合：

```
Megatron-LM 5D 并行:

  Dense 层: TP × CP × DP × PP
  MoE 层:   ETP × EP × EDP × PP

  ┌──────────────────────────────────────────┐
  │              并行维度                      │
  │                                          │
  │  TP  — 张量并行 (Megatron-style 切分)     │
  │        列并行: Y = GeLU(X · A_col)       │
  │        行并行: Y = X · A_row             │
  │        每层 2× AllReduce                  │
  │                                          │
  │  SP  — 序列并行 (与 TP 配合)              │
  │        LayerNorm 和 Dropout 在序列        │
  │        维度分片, 减少激活值显存             │
  │                                          │
  │  PP  — 流水线并行                         │
  │        支持: GPipe, 1F1B,                 │
  │        Interleaved 1F1B (virtual stages)  │
  │                                          │
  │  CP  — 上下文并行                         │
  │        长序列: Ring Attention 风格         │
  │                                          │
  │  EP  — 专家并行 (MoE 模型)               │
  │                                          │
  │  DP  — 数据并行 (可叠加 ZeRO-1)          │
  └──────────────────────────────────────────┘
```

### 2.3 关键技术特性

**1. Fused Kernels (融合算子)**

Megatron 大量使用 CUDA 融合算子减少显存访问和 kernel launch 开销：

```
标准 PyTorch:
  x = LayerNorm(x)          # kernel 1: 读写 x
  qkv = Linear(x)           # kernel 2: 读写 x
  q, k, v = split(qkv)      # kernel 3: 读写 qkv
  attn = softmax(q @ k^T)   # kernel 4-5
  out = attn @ v             # kernel 6

Megatron Fused:
  x = FusedLayerNorm(x)                    # 融合 LN 内部操作
  out = FlashAttention(x, fused_qkv=True)  # 融合 QKV + Attention
  → 更少的 kernel, 更少的显存读写, 更快
```

**2. Transformer Engine + FP8**

NVIDIA Transformer Engine 提供 FP8 精度的矩阵乘法，与 Megatron 深度集成：

```python
# Megatron 中使用 Transformer Engine
import transformer_engine.pytorch as te

class TETransformerLayer(te.TransformerLayer):
    """使用 Transformer Engine 的 FP8 层"""
    def __init__(self, config):
        super().__init__(
            hidden_size=config.hidden_size,
            ffn_hidden_size=config.ffn_hidden_size,
            num_attention_heads=config.num_heads,
            fuse_qkv_params=True,           # 融合 QKV
            fp8_format=te.recipe.Format.HYBRID,  # E4M3 前向, E5M2 反向
        )
```

**3. 选择性激活重计算**

```
完整重计算: 每层所有激活都不保存，反向时全部重算
  → 最省显存，但计算开销 ~33%

选择性重计算 (Megatron 默认):
  保留: LayerNorm 输出, Linear 输出 (计算便宜，读/写多)
  重算: Attention 中的 QK^T, Softmax (显存大，计算容易恢复)
  → 显存节省 ~70%, 计算开销仅 ~5-10%
```

**4. 分布式优化器**

Megatron 的分布式优化器将优化器状态分片到 DP 组（类似 ZeRO-1），减少显存占用。

### 2.4 配置示例

```bash
# Megatron-LM 训练 70B 模型 (8 节点 × 8 GPU = 64 GPU)
python pretrain_gpt.py \
    # 模型配置
    --num-layers 80 \
    --hidden-size 8192 \
    --num-attention-heads 64 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    \
    # 并行配置
    --tensor-model-parallel-size 8 \      # TP=8 (节点内)
    --pipeline-model-parallel-size 4 \     # PP=4
    --num-layers-per-virtual-pipeline-stage 2 \  # Interleaved PP
    # DP = 64 / (8 × 4) = 2
    \
    # 精度
    --bf16 \
    --use-flash-attn \
    \
    # 优化
    --use-distributed-optimizer \          # ZeRO-1 分布式优化器
    --recompute-activations \              # 激活检查点
    --recompute-granularity selective \    # 选择性重计算
    --overlap-grad-reduce \               # 梯度通信与计算重叠
    --overlap-param-gather \              # 参数收集重叠
    \
    # 训练
    --micro-batch-size 1 \
    --global-batch-size 1024 \
    --lr 1.5e-4 \
    --min-lr 1.5e-5 \
    --lr-decay-style cosine \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --train-iters 500000 \
    \
    # 检查点
    --save /checkpoint/70b \
    --save-interval 1000 \
    --async-save \                        # 异步检查点
    \
    # 数据
    --data-path /data/tokenized_dataset \
    --tokenizer-type GPT2BPETokenizer
```

### 2.5 NeMo Framework 集成

NeMo 是 NVIDIA 基于 Megatron-Core 的上层框架，提供配置驱动的训练体验：

```python
# NeMo 2.0 训练配置 (使用 Hydra)
# nemo/collections/llm/recipes/llama3_70b.py
from nemo.collections.llm import PreTrainingRecipe

recipe = PreTrainingRecipe(
    model="llama3_70b",
    trainer=dict(
        num_nodes=8,
        devices=8,
        precision="bf16-mixed",
        max_steps=500000,
    ),
    data=dict(
        paths=["/data/tokenized"],
        seq_length=8192,
        global_batch_size=1024,
        micro_batch_size=1,
    ),
    parallelism=dict(
        tensor_model_parallel_size=8,
        pipeline_model_parallel_size=4,
        context_parallel_size=1,
        expert_model_parallel_size=1,  # MoE 时设置
    ),
    optim=dict(
        name="distributed_fused_adam",  # Megatron 分布式优化器
        lr=1.5e-4,
        weight_decay=0.1,
    ),
)
```

### 2.6 优缺点总结

| 维度 | 评价 |
|------|------|
| **吞吐量** | 业界最高，是其他框架的 benchmark |
| **并行支持** | 最全面：TP+PP+SP+CP+EP+DP |
| **融合算子** | 大量手写 CUDA 算子，与 Transformer Engine/FP8 深度集成 |
| **学习曲线** | 陡峭：代码库庞大，文档不足，经常重构 |
| **可移植性** | 强绑定 NVIDIA 生态（NCCL, NVLink, Transformer Engine） |
| **社区** | 较封闭，主要由 NVIDIA 内部维护 |
| **调试** | 困难：多维并行交叉，错误信息不友好 |

---

## 3. DeepSpeed

### 3.1 架构概述

DeepSpeed 是 Microsoft 开发的深度学习优化库，核心定位是**突破显存墙**：

```
┌──────────────────────────────────────────────────┐
│                 DeepSpeed 架构                     │
│                                                   │
│  ┌─────────────────────────────────────────────┐  │
│  │  Training API                                │  │
│  │  deepspeed.initialize(model, optimizer, ...) │  │
│  └──────────────────────┬──────────────────────┘  │
│                         │                         │
│  ┌──────────┬───────────┼───────────┬──────────┐  │
│  │ ZeRO     │ Pipeline  │ MoE       │ Sequence │  │
│  │ 1/2/3    │ Parallel  │ Support   │ Parallel │  │
│  │ ++/Offload│          │ DeepSpeed │ Ulysses  │  │
│  │ Infinity │          │ -MoE      │          │  │
│  └──────────┴───────────┴───────────┴──────────┘  │
│                                                   │
│  ┌─────────────────────────────────────────────┐  │
│  │  底层优化                                     │  │
│  │  Fused Optimizers | Activation Checkpointing │  │
│  │  Sparse Attention | Communication Optim      │  │
│  │  ZenFlow (Offload) | Quantization            │  │
│  └─────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────┘
```

### 3.2 ZeRO 系列

ZeRO 是 DeepSpeed 最核心的贡献（详细原理见 [[distributed-training#3. ZeRO 优化]]）：

| 变体 | 分区内容 | 通信量 | 特点 |
|------|---------|--------|------|
| ZeRO-1 | 优化器状态 | $2\Phi$ | 最易用，几乎无性能损失 |
| ZeRO-2 | + 梯度 | $2\Phi$ | 进一步节省显存 |
| ZeRO-3 | + 参数 | $3\Phi$ | 最大节省，通信增加 50% |
| ZeRO-Offload | ZeRO-2 + CPU 卸载 | 同 ZeRO-2 + PCIe | 单 GPU 训练大模型 |
| ZeRO-Infinity | ZeRO-3 + CPU/NVMe 卸载 | 同 ZeRO-3 + PCIe/NVMe | 极端显存场景 |
| ZeRO++ | ZeRO-3 + 量化通信 | ~$0.75\Phi$ | 跨节点通信减少 4× |

**ZeRO-Offload / ZeRO-Infinity 工作原理**：

```
┌──────────────────────────────────────────────┐
│            ZeRO-Infinity 三级存储              │
│                                              │
│  GPU 显存 (80GB):                             │
│  ┌──────────────────────────────────────┐    │
│  │ 当前计算所需的参数分片 (FP16)          │    │
│  │ 当前计算的激活值                       │    │
│  └──────────────┬───────────────────────┘    │
│                 │ PCIe 4.0: ~32 GB/s         │
│  CPU 内存 (1TB+):│                            │
│  ┌──────────────┴───────────────────────┐    │
│  │ 优化器状态 (FP32 master weights + m,v)│    │
│  │ 暂时不需要的参数分片                   │    │
│  └──────────────┬───────────────────────┘    │
│                 │ NVMe SSD: ~3-7 GB/s        │
│  NVMe SSD (数TB):│                            │
│  ┌──────────────┴───────────────────────┐    │
│  │ 全部参数分片的溢出部分                  │    │
│  │ 优化器状态溢出                         │    │
│  └──────────────────────────────────────┘    │
└──────────────────────────────────────────────┘
```

**ZenFlow (2025 年 8 月)** — DeepSpeed 最新的卸载优化：
- 传统 offload 需要在 GPU↔CPU 传输时暂停计算
- ZenFlow 实现"无停顿"卸载：计算和传输完全重叠
- 利用 pinned memory + 异步流实现流水线化

### 3.3 DeepSpeed-Chat (RLHF 训练)

DeepSpeed-Chat 提供完整的 RLHF 训练管线：

```
┌──────────────────────────────────────────────┐
│           DeepSpeed-Chat RLHF 管线            │
│                                              │
│  Step 1: SFT (监督微调)                       │
│  ┌──────────────────────────────────────┐    │
│  │ Base Model + Instruction Data → SFT   │    │
│  │ 使用 ZeRO-3 分布式训练                 │    │
│  └──────────────────────────────────────┘    │
│                                              │
│  Step 2: Reward Model 训练                    │
│  ┌──────────────────────────────────────┐    │
│  │ SFT Model → Reward Model              │    │
│  │ 使用人类偏好数据 (chosen/rejected)     │    │
│  └──────────────────────────────────────┘    │
│                                              │
│  Step 3: PPO 训练                             │
│  ┌──────────────────────────────────────┐    │
│  │ 4 模型同时训练:                        │    │
│  │   Actor (SFT) | Critic (RM)           │    │
│  │   Ref Model   | Reward Model          │    │
│  │ Hybrid Engine: 训练+推理混合调度        │    │
│  └──────────────────────────────────────┘    │
└──────────────────────────────────────────────┘
```

**Hybrid Engine** 是 DeepSpeed-Chat 的核心创新：
- PPO 中 Actor 和 Ref Model 需要做推理（生成 response）
- Hybrid Engine 在训练（ZeRO）和推理（TP）模式间动态切换
- 推理时自动应用 kernel fusion、tensor parallelism
- 生成阶段切换到推理模式 → 3-4× 加速

### 3.4 DeepSpeed-MoE

DeepSpeed-MoE 提供 MoE 模型的训练支持：

```python
# DeepSpeed MoE 层配置
import deepspeed
from deepspeed.moe.layer import MoE

moe_layer = MoE(
    hidden_size=4096,
    expert=FeedForward(4096, 11008),  # 单个专家的 FFN
    num_experts=64,
    ep_size=8,                # 专家并行度
    use_residual=True,        # 残差 MoE
    k=2,                      # Top-K 路由
    capacity_factor=1.25,     # 容量因子
    eval_capacity_factor=2.0,
    min_capacity=4,
    use_rts=True,             # 随机 token 选择
    use_tutel=True,           # Tutel 加速 AllToAll
)
```

### 3.5 配置示例

DeepSpeed 使用 JSON 配置文件：

```json
{
    "train_batch_size": 1024,
    "train_micro_batch_size_per_gpu": 2,
    "gradient_accumulation_steps": 8,

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1.5e-4,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.1
        }
    },

    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 1.5e-4,
            "warmup_num_steps": 2000,
            "total_num_steps": 500000
        }
    },

    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "none"
        },
        "offload_param": {
            "device": "none"
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": 5e8,
        "stage3_prefetch_bucket_size": 5e8,
        "stage3_param_persistence_threshold": 1e6,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },

    "bf16": {
        "enabled": true
    },

    "gradient_clipping": 1.0,

    "activation_checkpointing": {
        "partition_activations": true,
        "cpu_checkpointing": false,
        "contiguous_memory_optimization": true,
        "number_checkpoints": null,
        "synchronize_checkpoint_boundary": false
    },

    "wall_clock_breakdown": false,
    "steps_per_print": 100
}
```

```python
# DeepSpeed 训练脚本
import deepspeed

model = MyLargeModel()

# DeepSpeed 初始化 — 自动处理分布式、优化器、混合精度
model_engine, optimizer, _, scheduler = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config="ds_config.json",
)

for step, batch in enumerate(dataloader):
    loss = model_engine(batch)
    model_engine.backward(loss)       # 自动处理 loss scaling、梯度同步
    model_engine.step()               # 自动处理优化器更新

    if step % save_interval == 0:
        model_engine.save_checkpoint("/checkpoint", tag=f"step_{step}")
```

### 3.6 DeepSpeed 推理优化

DeepSpeed-Inference 不是本页重点（详见 [[vllm]] 等推理框架），但值得一提：
- 自动 TP 划分
- Kernel fusion (QKV fusion, bias add + residual + LayerNorm)
- 量化推理 (INT8, INT4)
- 与 Hugging Face `transformers` 深度集成

### 3.7 优缺点总结

| 维度 | 评价 |
|------|------|
| **显存效率** | 业界最佳: ZeRO 全系列 + Offload + Infinity |
| **易用性** | JSON 配置驱动，与 HuggingFace 集成好 |
| **并行支持** | ZeRO + PP + EP + SP (Ulysses)。TP 支持弱于 Megatron |
| **吞吐量** | 略低于 Megatron-LM (特别是大规模 TP 场景) |
| **RLHF** | DeepSpeed-Chat 提供完整管线 |
| **社区** | 活跃，文档好，但 bug 修复有时较慢 |
| **可组合性** | 与 HuggingFace Trainer 深度集成 |
| **调试** | ZeRO-3 调试困难 (参数分片后难以 inspect) |

---

## 4. PyTorch FSDP / FSDP2

### 4.1 FSDP 基本原理

FSDP (Fully Sharded Data Parallel) 是 PyTorch 原生的 ZeRO-3 实现：

```
FSDP 工作流 (以 2 GPU 为例):

         GPU 0                    GPU 1
     ┌──────────┐            ┌──────────┐
     │ Param 1/2│            │ Param 2/2│  ← 参数分片存储
     └────┬─────┘            └────┬─────┘
          │                       │
     AllGather (收集完整参数)     AllGather
          │                       │
     ┌────┴─────┐            ┌────┴─────┐
     │ Full     │            │ Full     │
     │ Param    │            │ Param    │  ← 临时持有完整参数
     ├──────────┤            ├──────────┤
     │ Forward  │            │ Forward  │
     │ (data 0) │            │ (data 1) │
     ├──────────┤            ├──────────┤
     │ 丢弃全量  │            │ 丢弃全量  │  ← 前向完成后释放
     │ 参数     │            │ 参数     │
     └────┬─────┘            └────┬─────┘
          │                       │
     AllGather (再次收集)        AllGather
          │                       │
     ┌────┴─────┐            ┌────┴─────┐
     │ Backward │            │ Backward │
     │ (data 0) │            │ (data 1) │
     ├──────────┤            ├──────────┤
     │ 丢弃全量  │            │ 丢弃全量  │
     │ 参数     │            │ 参数     │
     └────┬─────┘            └────┬─────┘
          │                       │
     ReduceScatter (梯度分片同步)
          │                       │
     ┌────┴─────┐            ┌────┴─────┐
     │ Grad 1/2 │            │ Grad 2/2 │  ← 只保留自己的梯度分片
     │ Update   │            │ Update   │
     │ Param 1/2│            │ Param 2/2│  ← 只更新自己的参数分片
     └──────────┘            └──────────┘
```

### 4.2 FSDP vs DeepSpeed ZeRO-3

两者本质上实现同样的分片策略（都是 ZeRO-3），但实现方式和工程决策不同：

| 维度 | PyTorch FSDP | DeepSpeed ZeRO-3 |
|------|-------------|------------------|
| **实现方式** | `FlatParameter` (FSDP1) / DTensor (FSDP2) | 自定义 hook + partitioning |
| **参数分组** | 按 FSDP wrapping unit | 按 parameter group |
| **通信** | 原生 PyTorch c10d | DeepSpeed 自有通信层 |
| **CPU Offload** | 支持 (基本) | 更成熟 (ZeRO-Offload/Infinity) |
| **与 PyTorch 集成** | 原生，torch.compile 友好 | 外部库，部分不兼容 compile |
| **混合精度** | `MixedPrecision` policy | JSON config |
| **TP 组合** | FSDP2 + TP composable (DeviceMesh) | 需要 Megatron-DeepSpeed |
| **调试** | PyTorch 原生工具 | DeepSpeed profiler |

### 4.3 FSDP2 改进

FSDP2 是 PyTorch 2.x 中对 FSDP 的重大重写：

```
FSDP1 vs FSDP2:

FSDP1:
  - FlatParameter: 所有参数拼成一个大 tensor
  - 问题: 参数形状变化, 无法与 TP 组合, checkpoint 需要特殊处理
  - 所有参数作为一个整体分片

FSDP2:
  - DTensor (Distributed Tensor): 每个参数独立管理
  - per-parameter sharding: 参数粒度的分片
  - 通过 DeviceMesh 定义多维并行拓扑
  - 与 TP/PP/SP 自由组合
  - 通信无关检查点 (communication-free checkpoints)
```

**核心改进**：

| 改进 | FSDP1 | FSDP2 |
|------|-------|-------|
| 参数表示 | FlatParameter (拼接) | DTensor (独立) |
| 检查点 | 需要 resharding | 通信无关 |
| TP 组合 | 不支持 | 原生 composable |
| torch.compile | 部分支持 | 完整支持 |
| 显存 | 基线 | **低 7%** |
| 灵活性 | 受限 | per-parameter 策略 |

### 4.4 TorchTitan

TorchTitan 是 Meta/PyTorch 团队的"一站式"LLM 预训练方案（ICLR 2025）：

```
┌──────────────────────────────────────────────────┐
│                  TorchTitan                       │
│                                                  │
│  ┌────────────────────────────────────────────┐  │
│  │  4D Parallelism                             │  │
│  │  FSDP2 × TP × PP × SP/CP                   │  │
│  │  通过 DeviceMesh 灵活配置                    │  │
│  └────────────────────────────────────────────┘  │
│                                                  │
│  ┌────────────────────────────────────────────┐  │
│  │  关键组件                                    │  │
│  │  - Float8 训练 (with Transformer Engine)    │  │
│  │  - torch.compile 加速                       │  │
│  │  - 分布式检查点 (DCP)                        │  │
│  │  - 弹性训练 (torchrun)                      │  │
│  │  - 混合精度 (BF16/FP8)                      │  │
│  │  - 选择性激活检查点                          │  │
│  └────────────────────────────────────────────┘  │
│                                                  │
│  性能:                                            │
│  - 128 GPU 加速 65.08% (vs naive baseline)       │
│  - Float8 + FSDP2: 额外 50% 吞吐提升            │
│  - SimpleFSDP: 28.5% 显存降低, 68.7% 吞吐提升   │
└──────────────────────────────────────────────────┘
```

**DeviceMesh 多维并行配置**：

```python
from torch.distributed.device_mesh import init_device_mesh

# 4D 并行: (DP, PP, TP, SP)
# 64 GPU = 2 DP × 4 PP × 8 TP
mesh = init_device_mesh(
    "cuda",
    (2, 4, 8),   # (DP, PP, TP)
    mesh_dim_names=("dp", "pp", "tp"),
)

# 每个并行维度的 ProcessGroup 自动创建
dp_mesh = mesh["dp"]    # DP 通信组
pp_mesh = mesh["pp"]    # PP 通信组
tp_mesh = mesh["tp"]    # TP 通信组
```

### 4.5 SimpleFSDP (编译器驱动)

SimpleFSDP 是 PyTorch 团队探索的下一代方案，通过编译器自动实现 FSDP：

```
传统 FSDP:
  用户手动指定 wrapping policy → 框架插入 AllGather/ReduceScatter

SimpleFSDP:
  用户只标记模型 → torch.compile 自动分析计算图 →
  编译器决定最优通信调度 → 自动插入 AllGather/ReduceScatter

优势:
  - 28.5% 显存降低 (编译器优化内存布局)
  - 68.7% 吞吐提升 (更优的通信调度)
  - 更简单的用户 API
```

### 4.6 配置示例

```python
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

# ============ FSDP1 配置 ============
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    ),
    auto_wrap_policy=transformer_auto_wrap_policy(
        transformer_layer_cls={LlamaDecoderLayer},
    ),
    device_id=torch.cuda.current_device(),
    use_orig_params=True,  # 需要 torch.compile 时必须开启
)

# 可选: torch.compile
model = torch.compile(model)
```

```python
# ============ FSDP2 + TP (via TorchTitan style) ============
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel

# 步骤 1: 张量并行
for layer in model.layers:
    parallelize_module(
        layer.self_attn,
        tp_mesh,
        {
            "q_proj": ColwiseParallel(),
            "k_proj": ColwiseParallel(),
            "v_proj": ColwiseParallel(),
            "o_proj": RowwiseParallel(),
        },
    )
    parallelize_module(
        layer.mlp,
        tp_mesh,
        {
            "gate_proj": ColwiseParallel(),
            "up_proj": ColwiseParallel(),
            "down_proj": RowwiseParallel(),
        },
    )

# 步骤 2: FSDP2 分片 (在 TP 之上)
for layer in model.layers:
    fully_shard(layer, mesh=dp_mesh)  # per-layer sharding
fully_shard(model, mesh=dp_mesh)      # 顶层 sharding
```

### 4.7 优缺点总结

| 维度 | 评价 |
|------|------|
| **PyTorch 原生** | 最大优势: 无外部依赖, 与 compile/DCP/DTensor 完美集成 |
| **可组合性** | FSDP2 + TP + PP 自由组合 (DeviceMesh) |
| **torch.compile** | 完整支持，可获得额外加速 |
| **吞吐量** | 接近 Megatron (TorchTitan 差距 5-10%) |
| **显存** | ZeRO-3 级别，SimpleFSDP 进一步优化 |
| **TP 实现** | 不如 Megatron 成熟 (无 SP, CP 仍在开发) |
| **MoE** | EP 支持较新，不如 Megatron/DeepSpeed 成熟 |
| **文档** | PyTorch 官方文档 + TorchTitan 教程 |
| **未来方向** | PyTorch 官方路线图，活跃开发中 |

---

## 5. 框架对比表

### 5.1 详细特性对比

| 特性 | Megatron-LM/Core | DeepSpeed | FSDP2/TorchTitan |
|------|------------------|-----------|------------------|
| **开发方** | NVIDIA | Microsoft | Meta / PyTorch |
| **首次发布** | 2019 | 2020 | 2022 (FSDP1), 2024 (FSDP2) |
| **核心语言** | Python + CUDA | Python + C++ | Python (with C++ backend) |
| | | | |
| **--- 并行支持 ---** | | | |
| 数据并行 | DP + ZeRO-1 | ZeRO 1/2/3/++/Offload/Infinity | FSDP (= ZeRO-3) |
| 张量并行 | 最佳 (Megatron-style) | 基本支持 | DTensor-based (改进中) |
| 流水线并行 | 1F1B, Interleaved 1F1B | PP + ZeRO | Schedule-based PP |
| 序列/上下文并行 | SP + CP | Ulysses SP | 开发中 |
| 专家并行 | EP + ETP + EDP | DeepSpeed-MoE | 基本支持 |
| | | | |
| **--- 性能 ---** | | | |
| 最大吞吐量 | 最高 (基准) | ~85-95% of Megatron | ~90-95% of Megatron |
| FP8 支持 | Transformer Engine | 实验性 | Float8 + compile |
| 融合算子 | 最多 (手写 CUDA) | 较多 | 通过 compile 自动融合 |
| Flash Attention | 集成 | 集成 | 集成 |
| | | | |
| **--- 工程 ---** | | | |
| 易用性 | 困难 | 中等 | 中等 |
| HuggingFace 集成 | 需要适配 | Trainer 原生支持 | Trainer 支持 (加速中) |
| torch.compile | 部分支持 | 部分支持 | 完整支持 |
| 检查点 | 自有格式 | 自有格式 | DCP (原生) |
| 弹性训练 | 手动 | 有限 | torchrun |
| | | | |
| **--- 生态 ---** | | | |
| 社区活跃度 | 中 | 高 | 高 (PyTorch 官方) |
| 文档质量 | 中 | 中-高 | 中 (快速改善) |
| 模型支持 | GPT/LLaMA/MoE | 几乎所有 HF 模型 | 通用 |
| 上层封装 | NeMo | HF Trainer, DeepSpeed-Chat | TorchTitan |

### 5.2 性能基准 (参考值)

训练 LLaMA-like 70B 模型，64× H100 80GB NVLink：

| 框架 | 配置 | 吞吐量 (tokens/s/GPU) | MFU | 相对性能 |
|------|------|----------------------|-----|---------|
| Megatron-LM | TP=8, PP=4, DP=2 | ~3,800 | ~42% | 100% (基准) |
| TorchTitan | FSDP2+TP=8, PP=4, DP=2 | ~3,500 | ~39% | ~92% |
| DeepSpeed | ZeRO-3 + PP=4 | ~3,200 | ~36% | ~84% |
| FSDP1 | FULL_SHARD | ~2,900 | ~32% | ~76% |

> **注意**: 以上数据为近似参考值，实际性能取决于具体配置、硬件互联、batch size、序列长度等。不同 benchmark 结论可能不同。

---

## 6. 其他框架

### 6.1 Colossal-AI

HPC-AI Tech 开发的开源框架：

| 维度 | 描述 |
|------|------|
| **核心特性** | Gemini (ZeRO-like) + TP + PP + SP |
| **优势** | API 简单, 自动并行 (Alpha), HuggingFace 集成好 |
| **劣势** | 社区较小, 大规模验证不足 |
| **适用** | 快速实验, 中等规模训练 |

```python
# Colossal-AI 简单示例
import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin

plugin = GeminiPlugin(
    precision='bf16',
    initial_scale=2**16,
    max_norm=1.0,
)
booster = Booster(plugin=plugin)
model, optimizer, _, dataloader, _ = booster.boost(
    model, optimizer, dataloader=dataloader
)
```

### 6.2 Composer / MosaicML (Databricks)

Databricks (原 MosaicML) 的训练框架：

| 维度 | 描述 |
|------|------|
| **核心特性** | 算法加速 (MixUp, Label Smoothing, Stochastic Depth 等) |
| **并行** | 通过 FSDP 和 DeepSpeed 集成 |
| **优势** | 训练 recipe 标准化, Llama/MPT 训练脚本开源 |
| **劣势** | 主要服务 Databricks 生态 |
| **适用** | 快速实验, 中等规模预训练 |
| **代表模型** | DBRX (132B MoE), MPT 系列 |

### 6.3 Nanotron (HuggingFace)

HuggingFace 内部的预训练框架：

| 维度 | 描述 |
|------|------|
| **核心特性** | TP + PP + DP, 3D 并行 |
| **优势** | 代码简洁, 与 HuggingFace 生态集成 |
| **劣势** | 功能不如 Megatron 完整, 社区小 |
| **适用** | HuggingFace 生态内的预训练 |
| **代表模型** | SmolLM 系列训练 |

### 6.4 Fairscale (Meta)

Meta 的分布式训练库，FSDP 的前身：

| 维度 | 描述 |
|------|------|
| **核心特性** | FSDP 原型, OSS (Optimizer State Sharding), Pipeline |
| **现状** | 已被 PyTorch FSDP 吸收，维护模式 |
| **历史意义** | ZeRO 的 PyTorch 原生实现先驱 |

### 6.5 Megatron-DeepSpeed

Megatron-LM + DeepSpeed 的混合方案（曾广泛使用）：

```
Megatron 提供: TP + PP + 数据处理
DeepSpeed 提供: ZeRO 优化器 + 卸载 + 混合精度管理

曾用于: BigScience BLOOM (176B), 多个开源大模型
现状: 随着 Megatron-Core 和 FSDP2 成熟，使用逐渐减少
```

---

## 7. 选择指南

### 7.1 决策树

```
                      选择训练框架
                          │
              ┌───────────┴───────────┐
              │                       │
         模型 > 100B?            模型 < 100B?
              │                       │
       ┌──────┴──────┐          ┌─────┴─────┐
       │             │          │           │
    追求极致     快速上手?     显存紧张?    快速实验?
    吞吐量?         │          │           │
       │        TorchTitan   DeepSpeed   Composer
       │        /FSDP2       ZeRO-3      或
   Megatron-LM              + Offload   FSDP + HF
   + NeMo                               Trainer
       │
       │
   有 NVIDIA 专家支持?
   ┌──────┴──────┐
   │             │
   是            否
   │             │
  NeMo +      TorchTitan
  Megatron     (更易调试)
```

### 7.2 按场景推荐

| 场景 | 推荐方案 | 理由 |
|------|---------|------|
| **大规模预训练 (>100B)** | Megatron-LM + NeMo | 最高吞吐量, 最成熟的 5D 并行 |
| **中等规模预训练 (7-70B)** | TorchTitan / FSDP2 | PyTorch 原生, 好调试, 性能接近 Megatron |
| **显存受限 (少 GPU)** | DeepSpeed ZeRO-3 + Offload | ZeRO-Infinity 可用极少 GPU 训练大模型 |
| **RLHF / 对齐** | DeepSpeed-Chat 或 [[rl-training-frameworks]] | 完整 RLHF 管线 |
| **HuggingFace 模型微调** | DeepSpeed + HF Trainer | JSON 配置, 一行代码启用 |
| **MoE 模型** | Megatron-LM | EP + ETP + EDP 最完整 |
| **学术研究 / 原型** | FSDP2 或 Composer | 上手快, 代码可读性好 |
| **生产部署 (NVIDIA 基础设施)** | NeMo + Megatron-Core | 与 NVIDIA 硬件深度优化 |

### 7.3 组合使用

框架并非互斥，常见组合：

```
1. Megatron-LM (TP+PP) + DeepSpeed (ZeRO) = Megatron-DeepSpeed
   → 经典组合, BLOOM 训练使用

2. FSDP2 (数据并行分片) + PyTorch TP (张量并行) = TorchTitan 4D
   → PyTorch 原生全栈

3. DeepSpeed (训练) + vLLM (推理) + DeepSpeed-Chat (RLHF)
   → 端到端管线

4. NeMo (训练封装) + Megatron-Core (底层) + TensorRT-LLM (推理)
   → NVIDIA 全栈
```

---

## 8. 代码示例：三框架对比

### 8.1 最小训练循环对比

**PyTorch DDP (基线)**:
```python
model = DDP(model.cuda(), device_ids=[local_rank])
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for batch in dataloader:
    loss = model(batch.cuda())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

**DeepSpeed**:
```python
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model, config="ds_config.json"
)
for batch in dataloader:
    loss = model_engine(batch.cuda())
    model_engine.backward(loss)
    model_engine.step()
```

**FSDP**:
```python
model = FSDP(model.cuda(), sharding_strategy=ShardingStrategy.FULL_SHARD)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for batch in dataloader:
    loss = model(batch.cuda())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

**Megatron-LM**: (更复杂，不是简单 loop)
```python
# Megatron 有自己的训练循环, 用户需要定义:
def forward_step(data_iterator, model):
    batch = next(data_iterator)
    tokens = batch['tokens'].cuda()
    labels = batch['labels'].cuda()
    output = model(tokens)
    loss = cross_entropy(output, labels)
    return loss, {'lm_loss': loss}

# 框架自动处理: 并行 setup, 优化器, 梯度同步, PP 调度
pretrain(
    train_valid_test_datasets_provider=get_datasets,
    model_provider=get_model,
    forward_step_func=forward_step,
    args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
)
```

### 8.2 检查点保存对比

```python
# --- DeepSpeed ---
model_engine.save_checkpoint("/ckpt", tag="step_1000")
# 自动处理 ZeRO 分片, 保存完整模型和优化器状态

# --- FSDP + DCP ---
import torch.distributed.checkpoint as dcp
dcp.save(
    {"model": model.state_dict(), "optim": FSDP.optim_state_dict(model, optimizer)},
    storage_writer=dcp.FileSystemWriter("/ckpt/step_1000"),
)
# 并行写入, 支持 resharding

# --- Megatron-LM ---
# 内置在训练循环中:
# --save /ckpt --save-interval 1000 --async-save
```

---

## 9. 参考文献

- Shoeybi et al., *"Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"* (arXiv 2019)
- Narayanan et al., *"Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM"* (SC 2021)
- Rajbhandari et al., *"ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"* (SC 2020)
- Rajbhandari et al., *"ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning"* (SC 2021)
- Yao et al., *"DeepSpeed-Chat: Easy, Fast and Affordable RLHF Training of ChatGPT-like Models at All Scales"* (2023)
- Zhao et al., *"PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel"* (VLDB 2023)
- Liang et al., *"TorchTitan: One-stop PyTorch Native Solution for Production Ready LLM Pre-training"* (ICLR 2025)
- NVIDIA, *"Megatron-Core Documentation"* (2024-2025)
- Wang et al., *"ZeRO++: Extremely Efficient Collective Communication for Giant Model Training"* (2023)
- Bian et al., *"Colossal-AI: A Unified Deep Learning System for Big Model Era"* (ICPP 2021)
- MosaicML, *"Composer: A Library for Training Neural Networks Better, Faster, and Cheaper"* (2022)

---

## 10. 相关页面

- [[distributed-training]] — 分布式训练的并行策略与优化原理
- [[parallelism-strategies-deep-dive]] — DP / TP / EP / EDP / ETP 切分详解
- [[model-parallelism]] — 推理中的并行策略
- [[checkpointing]] — 检查点保存与恢复
- [[rl-training-frameworks]] — RL 专用训练框架 (OpenRLHF, veRL 等)
- [[gpu-cluster-management]] — GPU 集群管理
- [[ray-ecosystem]] — Ray 分布式计算生态
