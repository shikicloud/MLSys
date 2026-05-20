---
title: LLM 推理与服务
---

# LLM 推理与服务

本类别的主题目录。基础技术直接跳到深度页对应章节；论文则跳到论文精读页。

## 推理引擎

- [[vllm]] —— vLLM：基于 PagedAttention 的高吞吐 LLM 服务
- [[sglang]] —— SGLang：结构化生成 + RadixAttention 前缀缓存
- [[tensorrt-llm]] —— TensorRT-LLM：NVIDIA 编译图推理引擎

## 内存与 KV 缓存

- [[paged-attention]] —— PagedAttention：KV 缓存的虚拟内存管理
- [[kv-cache-optimization]] —— KV 缓存优化全景

  - [[kv-cache-optimization#架构级优化|架构层（MQA / GQA / MLA）]]
  - [[kv-cache-optimization#内存管理优化|内存管理（PagedAttention、VirtualAttention）]]
  - [[kv-cache-optimization#KV 缓存量化|KV 缓存量化]]
  - [[kv-cache-optimization#KV 缓存压缩|KV 缓存压缩（淘汰、稀疏化）]]
  - [[kv-cache-optimization#前缀缓存与共享|前缀缓存与共享]]
  - [[kv-cache-optimization#分布式 KV 缓存|分布式 KV 缓存]]

## 并行策略（单一深度页，按章节跳转）

权威参考：[[parallelism-strategies-deep-dive]]。章节锚点：

- [[parallelism-strategies-deep-dive#1. 概述|总览 + 组合公式]]
- [[parallelism-strategies-deep-dive#2. DP — 数据并行 (Data Parallelism)|数据并行 (DP)]]
- [[parallelism-strategies-deep-dive#3. ZeRO / FSDP — 分片数据并行 (Sharded Data Parallelism)|ZeRO / FSDP]]
- [[parallelism-strategies-deep-dive#4. TP — 张量并行 (Tensor Parallelism)|张量并行 (TP)]]
- [[parallelism-strategies-deep-dive#5. SP — 序列并行 (Sequence Parallelism)|序列并行 (SP)]] —— Megatron-v2 风格
- [[parallelism-strategies-deep-dive#6. PP — 流水线并行 (Pipeline Parallelism)|流水线并行 (PP)]] —— GPipe、1F1B、DualPipe、Zero Bubble
- [[parallelism-strategies-deep-dive#7. CP — 上下文并行 (Context Parallelism)|上下文并行 (CP)]] —— 序列 > 128K 时
- [[parallelism-strategies-deep-dive#8. EP — 专家并行 (Expert Parallelism)|专家并行 (EP)]]
- [[parallelism-strategies-deep-dive#9. EDP/DEP — 专家数据并行 (Expert Data Parallelism)|EDP / DEP]]
- [[parallelism-strategies-deep-dive#10. ETP/TEP — 专家张量并行 (Expert Tensor Parallelism)|ETP / TEP]]
- [[parallelism-strategies-deep-dive#11. DP Attention — 数据并行注意力 (Data-Parallel Attention for MoE Inference)|DP Attention（MoE 推理）]]
- [[parallelism-strategies-deep-dive#12. 混合并行策略：如何组合|混合策略（TP × CP、4D / 5D 并行）]]
- [[parallelism-strategies-deep-dive#13. 选择指南：决策流程|选择指南与决策树]]
- [[parallelism-strategies-deep-dive#14. 实战案例：DeepSeek-V3|DeepSeek-V3 实战]]

配套论文精读：

- [[ring-attention]] —— Ring Attention（Liu et al., ICLR 2024）
- [[deepspeed-ulysses]] —— DeepSpeed Ulysses（Microsoft, 2023）

## 批处理与调度

- [[continuous-batching]] —— 连续批处理（Orca / 迭代级调度）

  - [[continuous-batching#静态批处理的问题|为什么静态批处理不行]]
  - [[continuous-batching#连续批处理 (Continuous / Iteration-Level Batching)|连续批处理原理]]
  - [[continuous-batching#分块预填充 (Chunked Prefill)|分块预填充（Sarathi-Serve）]]
  - [[continuous-batching#调度策略|调度策略：FCFS、抢占、recompute vs swap]]

## 量化

- [[quantization]] —— 量化全景

  - [[quantization#权重量化方法|权重量化（GPTQ、AWQ）]]
  - [[quantization#基于旋转的量化（QuIP → QuaRot → SpinQuant → BDR）|基于旋转的量化（QuIP / QuaRot / SpinQuant）]]
  - [[quantization#FP8 量化|FP8 量化]]
  - [[quantization#KV 缓存量化|KV 缓存量化]]
  - [[quantization#激活量化|激活量化]]

- [[rotation-based-quantization]] —— 基于旋转的 KV 缓存量化家族综览
- [[saw-int4]] —— SAW-INT4 论文精读（块对角 Hadamard 旋转，Together AI）

## 投机解码

- [[speculative-decoding]] —— 投机解码总览（EAGLE、Medusa、lookahead）
- [[das-spec-rl]] —— DAS 论文精读（面向 RL 训练的分布感知投机解码）
- [[aurora]] —— Aurora 论文精读（在线投机解码训练做成 SGLang 实时流量上的异步 RL；Day-0 上线；Tree Attention）

## 分离式推理

- [[prefill-decode-disaggregation]] —— PD 分离（Splitwise、DistServe、Mooncake）

  - [[prefill-decode-disaggregation#Prefill vs Decode 特性对比|计算特性不对称]]
  - [[prefill-decode-disaggregation#为什么要分离|为什么要分离]]
  - [[prefill-decode-disaggregation#分离架构设计|Splitwise / DistServe / Mooncake 架构]]
  - [[prefill-decode-disaggregation#KV 缓存传输|KV 缓存传输机制]]
  - [[prefill-decode-disaggregation#与 chunked prefill 的组合|与 chunked prefill 的组合]]

- [[af-disaggregation]] —— 注意力-FFN 分离（MegaScale-Infer，下一个轴）

## 模型并行（旧 stub）

- [[model-parallelism]] —— 已重定向到 [[parallelism-strategies-deep-dive]]
