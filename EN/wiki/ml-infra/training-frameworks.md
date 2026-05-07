---
title: "Training Frameworks: Megatron-LM, DeepSpeed, FSDP"
category: ml-infra
tags: [megatron-lm, deepspeed, fsdp, torchtitan, training, frameworks, megatron-core, nemo]
created: 2026-04-13
updated: 2026-04-14
status: mature
---

# Training Frameworks: Megatron-LM, DeepSpeed, FSDP

> [!abstract]+ TL;DR
> LLM training frameworks integrate the parallelism strategies and optimizations from [[distributed-training]] into usable engineering systems. The ecosystem has converged to three main options: **Megatron-LM/Core** (NVIDIA, max throughput, TP+PP+SP+CP+EP+DP), **DeepSpeed** (Microsoft, memory efficiency, ZeRO 1-3+++ and PP/EP/Ulysses-SP), **FSDP2/TorchTitan** (Meta/PyTorch, native PyTorch, FSDP+TP+PP+SP). Production large-scale runs typically combine Megatron's TP/PP with DeepSpeed's ZeRO via Megatron-DeepSpeed.

---

## 1. Overview

LLM training frameworks integrate the parallelism strategies and optimizations from [[distributed-training]] into usable engineering systems. The ecosystem has converged to three main options:

| Framework | Developer | Best For | Parallelism |
|-----------|-----------|----------|-------------|
| **Megatron-LM/Core** | NVIDIA | Max throughput | TP+PP+SP+CP+EP+DP |
| **DeepSpeed** | Microsoft | Memory efficiency | ZeRO 1-3+++, PP, EP, Ulysses SP |
| **FSDP2/TorchTitan** | Meta/PyTorch | Native PyTorch | FSDP+TP+PP+SP |

---

## 2. Megatron-LM / Megatron-Core

### Architecture

```
NeMo Framework (high-level, config-driven)
  -> Megatron-Core (reusable parallelism primitives)
    -> Megatron-LM (training scripts, model definitions)
      -> PyTorch + NCCL + CUDA + Transformer Engine
```

**Megatron-Core** is the library of composable parallel primitives; **Megatron-LM** is the full training solution. NeMo wraps both with config-driven recipes.

### Key Features

- **Best TP implementation**: Megatron-style column/row parallel with sequence parallelism
- **Full 5D parallelism**: TP + PP (1F1B, interleaved) + SP + CP + EP + DP
- **Fused CUDA kernels**: Fused LayerNorm, fused QKV, Flash Attention
- **Transformer Engine + FP8**: Deep integration with NVIDIA FP8 hardware
- **Selective activation recomputation**: ~70% memory savings with only ~5-10% compute overhead
- **Distributed optimizer**: ZeRO-1 style optimizer state sharding

### Configuration Example

```bash
python pretrain_gpt.py \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 4 \
    --num-layers-per-virtual-pipeline-stage 2 \
    --bf16 --use-flash-attn \
    --use-distributed-optimizer \
    --recompute-granularity selective \
    --overlap-grad-reduce --overlap-param-gather \
    --micro-batch-size 1 --global-batch-size 1024
```

### Pros & Cons

| + | - |
|---|---|
| Highest throughput (benchmark for others) | Steep learning curve, poor docs |
| Most complete parallelism support | Tightly coupled to NVIDIA ecosystem |
| Production-proven at extreme scale | Frequent refactors, unstable API |

---

## 3. DeepSpeed

### Core: ZeRO Family

| Variant | Partitions | Special |
|---------|-----------|---------|
| ZeRO-1 | Optimizer states | Zero overhead vs DDP |
| ZeRO-2 | + Gradients | Zero overhead vs DDP |
| ZeRO-3 | + Parameters | +50% comm, max memory savings |
| ZeRO-Offload | ZeRO-2 + CPU | Train large models on few GPUs |
| ZeRO-Infinity | ZeRO-3 + CPU/NVMe | Extreme memory scenarios |
| ZeRO++ | ZeRO-3 + quantized comm | 4x less cross-node traffic |

See [[distributed-training#3. ZeRO Optimization]] for memory analysis.

### DeepSpeed-Chat (RLHF)

Complete 3-stage pipeline: SFT -> Reward Model -> PPO. The **Hybrid Engine** dynamically switches between ZeRO training mode and TP inference mode during PPO generation, achieving 3-4x speedup.

### DeepSpeed-MoE

Expert parallelism with configurable EP size, capacity factors, and Tutel-accelerated AllToAll.

### Configuration

```json
{
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_bucket_size": 5e8,
        "stage3_prefetch_bucket_size": 5e8
    },
    "bf16": {"enabled": true},
    "gradient_clipping": 1.0,
    "train_micro_batch_size_per_gpu": 2,
    "gradient_accumulation_steps": 8
}
```

```python
model_engine, optimizer, _, _ = deepspeed.initialize(model=model, config="ds_config.json")
for batch in dataloader:
    loss = model_engine(batch.cuda())
    model_engine.backward(loss)
    model_engine.step()
```

### Pros & Cons

| + | - |
|---|---|
| Best memory efficiency (ZeRO full suite) | TP weaker than Megatron |
| JSON config, HuggingFace Trainer integration | ZeRO-3 debugging is hard |
| Complete RLHF pipeline | Bug fixes sometimes slow |

---

## 4. PyTorch FSDP / FSDP2

### FSDP = PyTorch-native ZeRO-3

Each FSDP unit: AllGather params before forward -> compute -> discard full params -> AllGather before backward -> ReduceScatter gradients -> update local shard.

### FSDP2 Improvements over FSDP1

| | FSDP1 | FSDP2 |
|---|---|---|
| Param representation | FlatParameter (concat) | DTensor (per-parameter) |
| TP composability | Not supported | Native via DeviceMesh |
| torch.compile | Partial | Full |
| Checkpointing | Needs resharding | Communication-free (DCP) |
| Memory | Baseline | **7% lower** |

### TorchTitan (ICLR 2025)

Meta's one-stop LLM pre-training solution built on FSDP2:
- 4D parallelism: FSDP2 x TP x PP x SP via DeviceMesh
- Float8 training integration
- torch.compile acceleration
- 65% speedup at 128 GPUs; SimpleFSDP: 28.5% memory reduction, 68.7% throughput improvement

### Configuration Example (FSDP2 + TP)

```python
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel

# Step 1: Tensor Parallel
for layer in model.layers:
    parallelize_module(layer.self_attn, tp_mesh, {
        "q_proj": ColwiseParallel(), "k_proj": ColwiseParallel(),
        "v_proj": ColwiseParallel(), "o_proj": RowwiseParallel(),
    })

# Step 2: FSDP2 on top
for layer in model.layers:
    fully_shard(layer, mesh=dp_mesh)
fully_shard(model, mesh=dp_mesh)
```

### Pros & Cons

| + | - |
|---|---|
| PyTorch native, no external deps | TP less mature than Megatron |
| Full torch.compile support | EP/MoE support still developing |
| Best composability (DeviceMesh) | Smaller model zoo than DeepSpeed |
| Official PyTorch roadmap | |

---

## 5. Framework Comparison

| Feature | Megatron-LM | DeepSpeed | FSDP2/TorchTitan |
|---------|-------------|-----------|------------------|
| Max throughput | Highest (baseline) | ~85-95% | ~90-95% |
| Memory efficiency | Good (ZeRO-1 + selective ckpt) | Best (ZeRO-3 + Offload) | Good (ZeRO-3 level) |
| TP quality | Best | Basic | Improving |
| PP support | 1F1B, Interleaved | Basic | Schedule-based |
| MoE / EP | Most complete | Good | Basic |
| HuggingFace integration | Needs adapters | Native (Trainer) | Growing |
| torch.compile | Partial | Partial | Full |
| Ease of use | Hard | Medium | Medium |

---

## 6. Other Frameworks

| Framework | Developer | Notes |
|-----------|-----------|-------|
| **Colossal-AI** | HPC-AI Tech | Simple API, auto-parallelism (alpha), smaller community |
| **Composer** | Databricks | Algorithmic speedups, used for DBRX/MPT |
| **Nanotron** | HuggingFace | Clean 3D parallel code, used for SmolLM |
| **Fairscale** | Meta | FSDP precursor, maintenance mode |
| **Megatron-DeepSpeed** | Hybrid | Classic combo (used for BLOOM), declining usage |

---

## 7. Selection Guide

| Scenario | Recommendation | Reason |
|----------|---------------|--------|
| Pre-training >100B | Megatron-LM + NeMo | Max throughput, mature 5D parallel |
| Pre-training 7-70B | TorchTitan / FSDP2 | Native PyTorch, good debugging |
| Few GPUs, large model | DeepSpeed ZeRO-3 + Offload | ZeRO-Infinity for extreme cases |
| RLHF / Alignment | DeepSpeed-Chat or [[rl-training-frameworks]] | Complete pipeline |
| HF model fine-tuning | DeepSpeed + HF Trainer | One-line config |
| MoE models | Megatron-LM | EP + ETP + EDP most complete |
| Research / prototyping | FSDP2 or Composer | Fast iteration |

---

## 8. Code: Side-by-Side Comparison

### Minimal Training Loop

```python
# --- DDP (baseline) ---
model = DDP(model.cuda(), device_ids=[rank])
for batch in dl: loss = model(batch); loss.backward(); opt.step(); opt.zero_grad()

# --- DeepSpeed ---
engine, opt, _, _ = deepspeed.initialize(model=model, config="ds.json")
for batch in dl: loss = engine(batch); engine.backward(loss); engine.step()

# --- FSDP ---
model = FSDP(model.cuda(), sharding_strategy=ShardingStrategy.FULL_SHARD)
for batch in dl: loss = model(batch); loss.backward(); opt.step(); opt.zero_grad()

# --- Megatron-LM --- (framework-controlled loop)
def forward_step(data_iter, model):
    batch = next(data_iter); return model(batch['tokens'].cuda())
pretrain(model_provider=get_model, forward_step_func=forward_step, ...)
```

### Checkpointing

```python
# DeepSpeed: engine.save_checkpoint("/ckpt", tag="step_1000")
# FSDP + DCP: dcp.save({"model": model.state_dict()}, FileSystemWriter("/ckpt"))
# Megatron: --save /ckpt --save-interval 1000 --async-save
```

---

## 9. References

- Shoeybi et al., *"Megatron-LM: Training Multi-Billion Parameter Language Models"* (2019)
- Narayanan et al., *"Efficient Large-Scale Language Model Training on GPU Clusters"* (SC 2021)
- Rajbhandari et al., *"ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"* (SC 2020)
- Yao et al., *"DeepSpeed-Chat: Easy, Fast and Affordable RLHF Training"* (2023)
- Zhao et al., *"PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel"* (VLDB 2023)
- Liang et al., *"TorchTitan: One-stop PyTorch Native Solution for Production Ready LLM Pre-training"* (ICLR 2025)

---

## 10. Related Pages

- [[distributed-training]] -- Parallelism strategies and optimization principles
- [[parallelism-strategies-deep-dive]] -- DP/TP/EP/EDP/ETP mechanics
- [[model-parallelism]] -- Inference parallelism
- [[checkpointing]] -- Checkpoint save/restore
- [[rl-training-frameworks]] -- RL training frameworks (OpenRLHF, veRL)
- [[gpu-cluster-management]] -- GPU cluster management
- [[ray-ecosystem]] -- Ray distributed computing
