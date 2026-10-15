---
title: Agent Serving 与 RL Infra 经典阅读路线
category: ml-sys
tags: [reading-roadmap]
created: '2026-10-15'
updated: '2026-10-15'
status: growing
---

# Agent Serving 与 RL Infra 经典阅读路线

## 概述

这条路线从请求调度和显存管理出发，逐步连接 agent workload、分布式训练、MoE 与 RL 训练流水线。适合希望补齐系统基础的读者；不预设已经掌握任何论文。

选材不限制发表时间，优先基础性贡献与当前系统问题的直接关系，兼顾会议、期刊及作者团队。以下发表信息由原始论文或正式出版页核实，不使用未经核实的引用次数。**推荐不等于已读，路线顺序也不代表已经完成。**

每日推荐采用 **1 篇主读 + 2–4 篇候选**。主读用于建立一个概念，其余可以比较、略读或留待以后；无需每天精读所有候选。先读摘要、动机、核心设计和关键实验，感兴趣后再深入证明、代码与实现细节。

## 第一阶段：请求怎样执行，KV 为什么占满显存

**先修：** 了解自回归生成每次产生一个 token；能区分模型参数与每个请求的 KV cache。若这些概念还不熟悉，可带着问题先读论文背景部分。

### 1. Orca — OSDI 2022

[Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)

- **核心贡献：** 以每次生成迭代为调度单位，配合 selective batching，让请求能够动态加入和退出批次。
- **系统联系：** 理解 rollout 长短不一时，固定批次为什么浪费 GPU 时间，以及 continuous batching 的基本思路。
- **阅读建议：优先精读。** 先画一个长请求和两个短请求的执行时间线，对比请求级与迭代级调度。

### 2. PagedAttention / vLLM — SOSP 2023

[Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)

- **核心贡献：** 用分页思想管理动态增长的 KV cache，减少碎片，并支持请求内与请求间的缓存共享。
- **系统联系：** 解释 rollout 并发为何受到 KV 容量限制，以及共享前缀怎样影响可容纳的请求数量。
- **阅读建议：优先精读。** 先理解逻辑块、物理块和共享关系，再看 kernel；可手画两个共享前缀、随后分叉的请求。

### 3. FlashAttention — NeurIPS 2022

[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://papers.nips.cc/paper_files/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html)

- **核心贡献：** 通过分块与重计算减少 HBM 和片上 SRAM 之间的数据搬运，计算精确 attention 时不物化完整 attention 矩阵。
- **系统联系：** 建立 GPU 存储层次与数据搬运成本的直觉；区分 attention 执行优化和 KV 容量管理这两类问题。
- **阅读建议：先精读设计。** 关注数据流和显存访问；online softmax 的推导可以第二轮再读。

**阶段检查：** 尝试解释“更多空闲 KV 空间怎样提高吞吐”和“更少计算量为什么不一定更快”。

## 第二阶段：从单请求走向 agent 工作负载

**先修：** 第一阶段的批次调度与 KV 管理概念；能够区分处理输入的 prefill 和逐 token 输出的 decode。

### 4. SGLang — NeurIPS 2024

[SGLang: Efficient Execution of Structured Language Model Programs](https://proceedings.nips.cc/paper_files/paper/2024/hash/724be4472168f31ba1c9ac630f15dec8-Abstract-Conference.html)

- **核心贡献：** 联合设计语言前端和运行时，通过 RadixAttention 复用 KV 前缀，并优化结构化输出。
- **系统联系：** 直接连接多轮调用、分支生成与共享提示词，帮助分析 agent workload 中重复计算发生在哪里。
- **阅读建议：优先精读。** 从一个多轮或分支调用例子画前缀树，再看运行时如何复用缓存。

### 5. DistServe — OSDI 2024

[DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)

- **核心贡献：** 分离 prefill 与 decode，按首 token 延迟和后续 token 延迟要求分别配置资源与并行方式。
- **系统联系：** 为 serving 资源隔离和 KV 传输提供分析框架；优化目标是满足延迟约束的有效吞吐，而非只看总吞吐。
- **阅读建议：值得精读。** 先理解 TTFT、TPOT 与 goodput，再追踪分离后新增的 KV 搬运成本。

**阶段检查：** 尝试为一个多轮 agent 标出可复用的前缀，以及 prefill、decode 和工具等待各自的时间。论文结论还需要按具体 workload 验证。

## 第三阶段：模型怎样跨 GPU 执行

**先修：** 矩阵乘法、前向与反向传播、数据并行的基本概念。先读 Megatron-LM 与 ZeRO，再进入 MoE；MoE 两篇也可以稍后补读。

### 6. Megatron-LM — 2019，NVIDIA 技术报告

[Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) · [NVIDIA 作者主页](https://research.nvidia.com/labs/adlr/MegatronLM/)

- **核心贡献：** 针对 Transformer 的矩阵运算设计模型并行，将大层拆到多个 GPU，同时控制同步开销。
- **系统联系：** 理解张量并行的通信边界，以及训练与 rollout 为什么可能需要不同的并行配置。
- **阅读建议：值得精读。** 手推一个 MLP 两次矩阵乘法的行切分、列切分和 all-reduce。此处将原始工作标为技术报告，不声称其发表于某顶会。

### 7. ZeRO — SC 2020

[ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) · [正式出版记录](https://doi.org/10.1109/SC41405.2020.00024)

- **核心贡献：** 对数据并行副本中的优化器状态、梯度和参数逐步分片，以通信换取显存容量。
- **系统联系：** 建立训练 worker 的显存账本，也是理解 FSDP 与训练／生成权重布局差异的基础。
- **阅读建议：值得精读。** 自己列出参数、梯度和 Adam 状态分别占多少字节，再对比三个分片阶段。

### 8. Switch Transformers — JMLR 2022；预印本 2021

[Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://www.jmlr.org/papers/v23/21-0998.html)

- **核心贡献：** 简化为 top-1 专家路由，并处理专家容量、负载均衡和训练稳定性。
- **系统联系：** 建立 expert、routing 与 capacity factor 的直觉，理解激活参数量、总参数量与系统成本的区别。
- **阅读建议：先读概念。** 重点问：每 token 计算量相近时，专家分布不均和跨设备通信为什么仍会拖慢执行？

### 9. GShard — ICLR 2021；预印本 2020

[GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668) · [会议版全文](https://openreview.net/pdf/cdb90e31da8446076492c5aef0c8c6ae35dd472a.pdf)

- **核心贡献：** 将条件计算 MoE 与自动分片结合，在扩展模型时同时处理设备划分、路由和跨设备通信。
- **系统联系：** 为专家并行、负载不均与 all-to-all 通信提供系统层面的起点。
- **阅读建议：选择性精读。** 在理解 Switch 后，追踪 token 发往专家再返回的路径；编译器细节可以后读。这里按学习难度排序，并非发表时间排序。

**阶段检查：** 尝试说明张量并行、数据并行分片与专家并行分别切分什么、节省什么、增加什么通信。

## 第四阶段：把生成、训练和控制流接起来

**先修：** 第一阶段的 serving 概念、第三阶段的 Megatron-LM 与 ZeRO，以及“采样 → 评估 → 更新策略”的 RL 基本循环。MoE 不是这一阶段的必要先修。

### 10. Ray — OSDI 2018

[Ray: A Distributed Framework for Emerging AI Applications](https://www.usenix.org/conference/osdi18/presentation/moritz)

- **核心贡献：** 在同一执行引擎中支持 task 与 actor，并结合分布式调度和容错状态管理执行动态 AI 工作负载。
- **系统联系：** 理解 rollout worker 的状态、资源占用和调度；将计算任务与长生命周期服务区分开。
- **阅读建议：选择性精读。** 先思考一个 rollout worker 适合 task 还是 actor，再看设计。论文描述当时的架构，不直接作为当前 Ray 的操作手册。

### 11. HybridFlow / verl — EuroSys 2025；预印本 2024

[HybridFlow: A Flexible and Efficient RLHF Framework](https://i.cs.hku.hk/~cwu/papers/gmsheng-eurosys25.pdf) · [arXiv](https://arxiv.org/abs/2409.19256)

- **核心贡献：** 结合单控制器和多控制器方式组织 RLHF 数据流，并用 3D-HybridEngine 降低 actor 在训练与生成之间切换分片的成本。
- **系统联系：** 直接连接 rollout、训练、设备放置和权重同步，是理解 verl 系统设计的核心入口。
- **阅读建议：优先精读。** 先画 generation → preparation → training 数据流，再标出权重布局变化、通信和设备占用；第一轮无需逐行推导 PPO。

**阶段检查：** 尝试为一次训练迭代画出关键路径，区分生成慢、控制开销大、权重切换慢和训练慢这几种瓶颈。

## 相关工作与后续阅读

结合 [[CN/wiki/llm-inference/index|LLM 推理与服务]]、[[CN/wiki/rl-infra/index|RL 基础设施]] 和 [[CN/wiki/llm-serving-for-agents/index|面向智能体的 LLM 服务]] 继续补充概念。新论文和深度分析可围绕以上问题加入候选，不必为了“新”跳过基础，也不把普通版本更新当作研究贡献。

论文链接列在各条目标题下；会议年份与预印本年份分别标注。若后续讨论发现先修不足，可以调整阅读顺序，而无需把整条路线一次读完。

返回 [[CN/reading/index|阅读推荐首页]]。
