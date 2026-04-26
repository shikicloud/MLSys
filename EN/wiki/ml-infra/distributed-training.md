---
title: "Distributed Training: Data/Model/Pipeline Parallelism"
category: ml-infra
tags: [distributed-training, data-parallelism, tensor-parallelism, pipeline-parallelism, zero, fsdp, mixed-precision, fault-tolerance]
created: 2026-04-13
updated: 2026-04-14
status: mature
---

# Distributed Training: Data/Model/Pipeline Parallelism

> **Companion page**: [[parallelism-strategies-deep-dive]] covers DP/TP/EP/EDP/ETP splitting mechanics and communication primitives (shared by training and inference). This page focuses on **training-specific** concerns: gradient synchronization, ZeRO memory optimization, mixed-precision training, activation checkpointing, communication optimization, and fault tolerance.

---

## 1. Overview

### Why Distributed Training

Modern LLMs far exceed single-GPU capacity:

| Model | Params | GPUs | Tokens | Training Time |
|-------|--------|------|--------|---------------|
| GPT-3 (2020) | 175B | ~1,000 V100 | 300B | ~34 days |
| LLaMA 3.1 405B (2024) | 405B | 16,384 H100 | 15.6T | ~54 days |
| DeepSeek-V3 (2024) | 671B MoE (37B active) | 2,048 H800 | 14.8T | ~55 days |

### Training Memory Breakdown

For a model with $\Phi$ parameters using AdamW + mixed precision:

```
Model params (mixed precision):  2Phi (BF16) + 4Phi (FP32 master)
Gradients:                       2Phi (BF16)
Optimizer (AdamW):               4Phi (m) + 4Phi (v) in FP32
Total (excl. activations):       16Phi bytes

Example: 70B model -> 1,120 GB (far exceeds single GPU)
```

---

## 2. Data Parallel (DP) Training Details

> See [[parallelism-strategies-deep-dive#2. DP]] for DP fundamentals.

### DDP (DistributedDataParallel)

Each GPU holds a full model copy. Data is sharded across workers. After backward pass, gradients are synchronized via AllReduce so all replicas update identically.

### Ring AllReduce

The core algorithm for gradient synchronization. N GPUs form a logical ring; communication proceeds in two phases:

1. **ReduceScatter** (N-1 rounds): each GPU ends up with 1/N of the globally-summed gradient
2. **AllGather** (N-1 rounds): each GPU broadcasts its chunk to all others

Communication per GPU: $2 \cdot \frac{N-1}{N} \cdot D$ -- nearly independent of GPU count for large N.

### Gradient Accumulation

When GPU memory cannot fit the desired micro-batch size:

```python
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    # Skip gradient sync on intermediate steps
    ctx = model.no_sync() if (i+1) % accumulation_steps != 0 else nullcontext()
    with ctx:
        loss.backward()
    if (i+1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

Effective batch size = `micro_batch * accumulation_steps * world_size`.

### Synchronous vs Asynchronous

| | Synchronous (BSP) | Asynchronous (ASP) |
|---|---|---|
| Gradients | Identical across workers | Stale gradients |
| Convergence | Equivalent to single-GPU | May diverge |
| Usage | **Mainstream** (all LLM training) | Recommendation systems |

---

## 3. ZeRO Optimization

ZeRO eliminates memory redundancy in data parallelism. Standard DDP replicates all of {params, gradients, optimizer states} on every GPU.

### Three Stages

| Component | DDP | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|-----------|-----|--------|--------|--------|
| BF16 Params | $2\Phi$ | $2\Phi$ | $2\Phi$ | $2\Phi/N_d$ |
| BF16 Grads | $2\Phi$ | $2\Phi$ | $2\Phi/N_d$ | $2\Phi/N_d$ |
| FP32 Master + Adam m,v | $12\Phi$ | $12\Phi/N_d$ | $12\Phi/N_d$ | $12\Phi/N_d$ |
| **Total** | **$16\Phi$** | **$4\Phi + 12\Phi/N_d$** | **$2\Phi + 14\Phi/N_d$** | **$16\Phi/N_d$** |

For 70B params, $N_d=64$: DDP needs 1,120 GB/GPU; ZeRO-3 needs only 17.5 GB/GPU.

### Communication Overhead

| Stage | Comm Volume | Notes |
|-------|------------|-------|
| DDP / ZeRO-1 / ZeRO-2 | $2\Phi$ | Same as standard AllReduce |
| ZeRO-3 | $3\Phi$ | +50% due to forward AllGather |

ZeRO-1/2 save memory with **zero communication overhead** vs DDP. ZeRO-3's extra AllGather can be hidden via prefetching.

### ZeRO++

Reduces cross-node communication 4x via: INT8 quantized weight comm, hierarchical partitioning (full replicas within node), and INT4 quantized gradient comm.

---

## 4. Mixed Precision Training

### Data Type Comparison

| Type | Bits | Exponent | Mantissa | Range | Use |
|------|------|----------|----------|-------|-----|
| FP32 | 32 | 8 | 23 | ~1e38 | Master weights, optimizer |
| BF16 | 16 | 8 | 7 | ~1e38 | Forward/backward (recommended) |
| FP16 | 16 | 5 | 10 | ~6.5e4 | Forward/backward (needs loss scaling) |
| FP8 (E4M3) | 8 | 4 | 3 | ~448 | Forward (emerging) |

### Training Flow

1. Cast FP32 master weights to BF16
2. Forward pass in BF16 (Tensor Cores: 2-3x speedup)
3. Backward pass in BF16
4. AllReduce gradients in BF16
5. Cast gradients to FP32, update master weights with Adam

**BF16 advantage**: Same dynamic range as FP32, so no loss scaling needed (unlike FP16).

### FP8 Training

Leverages H100 FP8 Tensor Cores for 2x theoretical throughput over BF16. E4M3 for forward, E5M2 for backward. DeepSeek-V3 was the first large-scale success. TorchTitan + Float8 + FSDP2 achieves 50% throughput improvement.

---

## 5. 3D Parallelism

> See [[parallelism-strategies-deep-dive]] for TP/EP details and [[model-parallelism]] for PP scheduling.

### Composition: TP x PP x DP

Single strategies have limitations; 3D parallelism combines their strengths:

| Dimension | Maps To | Communication |
|-----------|---------|---------------|
| **TP** | Intra-node NVLink (900 GB/s) | AllReduce per layer |
| **PP** | Inter-node or intra-node | Point-to-point |
| **DP** | Cross-replica InfiniBand | AllReduce per step |

### Real-World Configurations

**LLaMA 3.1 405B**: TP=8, PP=16, DP=128 across 16,384 H100s. ~38-43% MFU.

**DeepSeek-V3**: TP=1(!), PP=16, DP=128, EP=64 on 2,048 H800s. No TP because MoE activates only 37B params/token. Uses FP8 + DualPipe. Cost ~$5.5M vs >$100M for LLaMA 3.1.

### 4D/5D Parallelism

- 4D = TP + PP + DP + CP/SP
- 5D = TP + PP + DP + CP + EP (for MoE)

NVIDIA's MoE Parallel Folding (2025) decouples dense and MoE parallelism entirely.

---

## 6. Activation Checkpointing

Trades compute for memory by recomputing intermediate activations during backward instead of storing them.

| Strategy | Activation Memory | Extra Compute | When to Use |
|----------|------------------|---------------|-------------|
| None | $O(L)$ | 0 | Memory-rich |
| Full checkpointing | $O(L/k)$ | ~33% | Memory-constrained |
| Selective (Megatron-style) | ~$O(L \cdot 0.3)$ | ~5-10% | Best trade-off |

Selective checkpointing: keep cheap-to-recompute activations (LayerNorm output), recompute expensive ones (attention QK^T, softmax).

---

## 7. Communication Optimization

### DDP Bucketing

Gradients are packed into ~25 MB buckets for AllReduce, starting from the last layer to overlap with backward computation.

### Computation-Communication Overlap

- **DDP**: AllReduce starts as soon as a bucket's gradients are ready (during backward)
- **FSDP**: AllGather for next layer prefetched during current layer's forward/backward
- **Pipeline**: Micro-batches in flight overlap computation across stages

### NCCL Tuning

Key settings: GPU Direct RDMA (`NCCL_NET_GDR_LEVEL`), Ring vs Tree algorithm selection, multi-rail InfiniBand, SHARP in-network reduction.

---

## 8. Fault Tolerance

At 16K GPUs, expect ~1.5-2.5 GPU failures per day. LLaMA 3.1 training (~54 days) faced ~80-135 failures.

### Checkpointing Strategies

- **Synchronous**: Pause training, save, resume. Simple but slow for large models.
- **Asynchronous**: Copy to pinned memory (fast), then write to disk in background. No training pause.
- **Distributed Checkpoint (DCP)**: Each rank writes its shard in parallel; supports resharding on load.

### Elastic Training (torchrun)

Dynamically adjusts worker count on failure/recovery:
```bash
torchrun --nnodes=15:16 --nproc_per_node=8 --rdzv_backend=c10d train.py
```

### Silent Data Corruption (SDC)

Most dangerous failure type: GPU produces wrong results without errors. LLaMA 3 team reported multiple SDC incidents. Detection via loss anomaly monitoring and cross-replica comparison.

---

## 9. Code Examples

### PyTorch DDP Setup

```python
dist.init_process_group("nccl", rank=rank, world_size=world_size)
model = DDP(model.to(rank), device_ids=[rank])
sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
# Training loop: loss.backward() auto-triggers AllReduce
```

### FSDP Configuration

```python
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3
    mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16),
    auto_wrap_policy=transformer_auto_wrap_policy(transformer_layer_cls={LlamaDecoderLayer}),
)
```

### Mixed Precision + Gradient Accumulation

```python
for step, batch in enumerate(dataloader):
    ctx = model.no_sync() if (step+1) % accum_steps != 0 else nullcontext()
    with ctx:
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss = model(batch) / accum_steps
        loss.backward()
    if (step+1) % accum_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
```

---

## 10. Large-Scale Training Case Studies

### LLaMA 3.1 405B (Meta, 2024)

405B dense, 16,384 H100s, TP=8 PP=16 DP=128, BF16, 15.6T tokens, ~38-43% MFU. Key lesson: network failures more frequent than GPU failures at 16K scale; SDC requires dedicated monitoring.

### DeepSeek-V3 (DeepSeek, 2024)

671B MoE (37B active), 2,048 H800s, TP=1 PP=16 DP=128 EP=64, FP8, 14.8T tokens. Training cost ~$5.5M (10-20x cheaper than comparable dense models). Innovations: DualPipe scheduling, FP8 training, auxiliary-loss-free load balancing.

---

## 11. References

- Rajbhandari et al., *"ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"* (SC'20)
- Micikevicius et al., *"Mixed Precision Training"* (ICLR 2018)
- Narayanan et al., *"Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM"* (SC'21)
- Zhao et al., *"PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel"* (VLDB 2023)
- Liang et al., *"TorchTitan: One-stop PyTorch Native Solution for Production Ready LLM Pre-training"* (ICLR 2025)
- Chen et al., *"Training Deep Nets with Sublinear Memory Cost"* (2016) -- Activation checkpointing
- Meta, *"Llama 3.1 Technical Report"* (2024)
- DeepSeek, *"DeepSeek-V3 Technical Report"* (2024)

---

## 12. Related Pages

- [[parallelism-strategies-deep-dive]] -- DP/TP/EP/EDP/ETP mechanics (training + inference)
- [[model-parallelism]] -- Inference parallelism and PP scheduling
- [[training-frameworks]] -- Megatron-LM, DeepSpeed, FSDP frameworks
- [[gpu-cluster-management]] -- Hardware infrastructure
- [[checkpointing]] -- Checkpoint save/restore details
- [[rl-training-frameworks]] -- Distributed strategies for RL training
