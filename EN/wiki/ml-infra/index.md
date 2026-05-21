---
title: ML Infrastructure
---

# ML Infrastructure

Distributed training, training frameworks, and ML systems plumbing.

## Distributed training

- [[data-parallelism]] — **DP family deep-dive**: DDP / ZeRO 1-3 / FSDP 1-2 / HSDP / DWDP, optimizer-states memory math, Transformer parameter anatomy (attention vs FFN), Llama 3 70B real breakdown
- [[distributed-training]] — distributed training deep dive

  - [[distributed-training#2. Data Parallelism (DP) Training Details|Data parallelism (DDP, Ring AllReduce)]]
  - [[distributed-training#3. ZeRO Optimization|ZeRO 1/2/3 (memory partitioning)]]
  - [[distributed-training#4. Mixed Precision Training|Mixed precision (FP16/BF16/FP8)]]
  - [[distributed-training#5. 3D Parallelism|3D parallelism composition]]
  - [[distributed-training#6. Gradient Checkpointing (Activation Checkpointing)|Gradient / activation checkpointing]]
  - [[distributed-training#7. Communication Optimization|Communication optimization (bucketing, NCCL tuning)]]
  - [[distributed-training#8. Fault Tolerance|Fault tolerance & elastic training]]
  - [[distributed-training#10. Large-Scale Training Case Studies|Large-scale case studies (Llama 3, DeepSeek-V3)]]

## Training frameworks

- [[training-frameworks]] — Megatron / DeepSpeed / FSDP / TorchTitan

  - [[training-frameworks#2. Megatron-LM / Megatron-Core|Megatron-LM and 5D parallelism]]
  - [[training-frameworks#3. DeepSpeed|DeepSpeed (ZeRO suite, Chat, MoE)]]
  - [[training-frameworks#4. PyTorch FSDP / FSDP2|FSDP / FSDP2]]
  - [[training-frameworks#5. Framework Comparison Table|Framework comparison table]]
  - [[training-frameworks#6. Other Frameworks|Other frameworks (Colossal-AI, Nanotron, Fairscale)]]
  - [[training-frameworks#7. Selection Guide|Selection guide]]
