---
title: ML 基础设施
---

# ML 基础设施

分布式训练、训练框架、ML 系统底层。

## 分布式训练

- [[distributed-training]] —— 分布式训练深度解析

  - [[distributed-training#2. 数据并行 (DP) 训练细节|数据并行（DDP、Ring AllReduce）]]
  - [[distributed-training#3. ZeRO 优化|ZeRO 1/2/3（内存分片）]]
  - [[distributed-training#4. 混合精度训练|混合精度（FP16 / BF16 / FP8）]]
  - [[distributed-training#5. 3D 并行|3D 并行组合]]
  - [[distributed-training#6. 梯度检查点 (Activation Checkpointing)|梯度 / 激活检查点]]
  - [[distributed-training#7. 通信优化|通信优化（bucketing、NCCL 调优）]]
  - [[distributed-training#8. 容错|容错与弹性训练]]
  - [[distributed-training#10. 大规模训练案例|大规模训练实战（Llama 3、DeepSeek-V3）]]

## 训练框架

- [[training-frameworks]] —— Megatron / DeepSpeed / FSDP / TorchTitan

  - [[training-frameworks#2. Megatron-LM / Megatron-Core|Megatron-LM 与 5D 并行]]
  - [[training-frameworks#3. DeepSpeed|DeepSpeed（ZeRO 全系列、Chat、MoE）]]
  - [[training-frameworks#4. PyTorch FSDP / FSDP2|FSDP / FSDP2]]
  - [[training-frameworks#5. 框架对比表|框架对比表]]
  - [[training-frameworks#6. 其他框架|其他框架（Colossal-AI、Nanotron、Fairscale）]]
  - [[training-frameworks#7. 选择指南|选择指南]]
