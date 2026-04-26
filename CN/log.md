---
title: 变更日志
updated: 2026-04-13
---

# 变更日志

## 2026-04-14
- [扩展] [[distributed-training]] — 全面扩展为深度文章（~1090行），涵盖训练显存分析、DDP 梯度同步与 Ring AllReduce 算法图解、ZeRO 1/2/3 各阶段显存公式与 ASCII 图、混合精度训练（FP16/BF16/FP8）、3D/4D/5D 并行组合与硬件拓扑映射、梯度检查点、通信优化（Bucketing/NCCL）、容错与弹性训练、代码示例、LLaMA 3.1 / DeepSeek-V3 训练案例
- [扩展] [[training-frameworks]] — 全面扩展为深度文章（~1050行），涵盖 Megatron-LM/Megatron-Core 架构与 5D 并行、DeepSpeed ZeRO 全系列与 Chat/MoE、FSDP/FSDP2 与 TorchTitan、框架详细对比表与性能基准、其他框架（Colossal-AI/Composer/Nanotron/Fairscale）、选择指南决策树、三框架代码对比
- [扩展] [[rlhf-overview]] — 全面扩展为深度文章（~600行），涵盖三阶段流水线详解（SFT/RM/PPO）、Bradley-Terry 模型推导、RM 损失函数、RL 目标函数与 GAE、RLHF 变体（Online/Offline/RLAIF/RLVR/迭代/Best-of-N）、TRL 代码示例、挑战与开放问题
- [扩展] [[ppo-for-llm]] — 全面扩展为深度文章（~760行），涵盖 TRPO→PPO 演进、裁剪替代目标详解与几何直觉、四模型架构与内存分析、GAE 公式推导与实现、Token 级更新机制、KL 惩罚整合、实现技巧与常见陷阱、PPO 训练步骤完整伪代码、替代方案对比表
- [扩展] [[grpo]] — 全面扩展为深度文章（~750行），涵盖去掉 Critic 的动机分析、组相对优势完整推导、GRPO 流水线图解、与 PPO 系统性对比、DeepSeek-R1-Zero 涌现现象、R1 完整训练流水线、GRPO 伪代码与 TRL 使用、DAPO/Dr.GRPO/RLOO 变体、性能基准对比
- [扩展] [[continuous-batching]] — 全面扩展为深度文章（~1000行），涵盖静态批处理问题分析、Orca 论文贡献、分块预填充（Sarathi-Serve）、调度策略（FCFS/抢占/优先级）、vLLM V1 统一调度器、SGLang 对比、内存管理交互、完整调度器伪代码、性能分析
- [扩展] [[prefill-decode-disaggregation]] — 全面扩展为深度文章（~1100行），涵盖 Prefill vs Decode 特性对比、Roofline 分析、Splitwise/DistServe/Mooncake 架构详解、KV 缓存传输机制、DeepSeek-V3 分离部署案例、分离架构伪代码、性能与成本分析
- [扩展] [[vllm]] — 全面扩展为深度技术文章（~700行），涵盖 V1 架构详解、EngineCore 隔离、统一调度器、持久化批处理、前缀缓存机制、DP Attention + EP、投机解码、量化、基准测试、部署实践、引擎对比
- [新增] [[parallelism-strategies-deep-dive]] — LLM 并行策略深度解析（DP/TP/EP/EDP/ETP），含原理、代码、通信分析、DeepSeek-V3 案例
- [扩展] [[model-parallelism]] — 全面重写为深度文章（~800行），聚焦 PP 和 CP：GPipe/1F1B/Interleaved/Zero Bubble/DualPipe 调度策略详解、Ring Attention/Ulysses 上下文并行、推理应用、混合并行配置、性能分析

## 2026-04-13
- [初始化] Wiki Vault 创建完毕，初始结构就绪
- [初始化] Schema（CLAUDE.md）定义完成
- [初始化] 目录索引创建完毕，规划了所有主题覆盖范围
- [摄入] 所有主题领域的初始研究与填充
