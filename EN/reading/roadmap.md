---
title: A Foundational Reading Roadmap for Agent Serving and RL Infrastructure
category: ml-sys
tags: [reading-roadmap]
created: '2026-10-15'
updated: '2026-10-15'
status: growing
---

# A Foundational Reading Roadmap for Agent Serving and RL Infrastructure

## Overview

This roadmap starts with request scheduling and memory management, then connects agent workloads, distributed training, MoE, and RL training pipelines. It is intended for readers building their systems foundations; it does not assume that any paper has already been mastered.

Selection is not limited by publication date. Foundational contributions and relevance to concrete systems problems come first, with venues and research teams as additional signals. Publication details below were checked against original papers or official publication pages. No unverified citation counts are used. **A recommendation is not a record of having read a paper, and the sequence does not imply completion.**

Each daily selection should contain **one main read and two to four alternatives**. The main read develops one concept; alternatives can be compared, skimmed, or saved. There is no expectation to read every candidate in depth each day. Start with the abstract, motivation, main design, and key experiments before proofs, code, and implementation details.

## Stage 1: How requests execute and why KV fills GPU memory

**Prerequisites:** Understand that autoregressive generation produces one token per step, and distinguish model parameters from each request's KV cache. If these ideas are unfamiliar, use the papers' background sections to develop them.

### 1. Orca — OSDI 2022

[Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)

- **Contribution:** Schedules individual generation iterations and uses selective batching so requests can enter and leave a batch dynamically.
- **Systems connection:** Explains why fixed batches waste GPU time when rollout lengths differ, and introduces the core idea behind continuous batching.
- **Reading advice: prioritize a close read.** Draw a timeline for one long request and two short ones, then compare request-level and iteration-level scheduling.

### 2. PagedAttention / vLLM — SOSP 2023

[Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)

- **Contribution:** Applies paging to a growing KV cache, reducing fragmentation and enabling sharing within and across requests.
- **Systems connection:** Explains how KV capacity limits concurrent rollouts and how shared prefixes change the number of requests that fit.
- **Reading advice: prioritize a close read.** Understand logical blocks, physical blocks, and sharing before inspecting kernels. Draw two requests that share a prefix and then branch.

### 3. FlashAttention — NeurIPS 2022

[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://papers.nips.cc/paper_files/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html)

- **Contribution:** Uses tiling and recomputation to reduce transfers between HBM and on-chip SRAM, computing exact attention without materializing the full attention matrix.
- **Systems connection:** Builds intuition for GPU memory hierarchies and data movement, and helps distinguish attention execution from KV capacity management.
- **Reading advice: study the design first.** Focus on data flow and memory accesses; return to the online softmax derivation on a second pass.

**Checkpoint:** Try explaining how additional free KV space can improve throughput, and why fewer arithmetic operations do not always mean faster execution.

## Stage 2: From individual requests to agent workloads

**Prerequisites:** Stage 1's batching and KV concepts, plus the distinction between processing an input during prefill and producing tokens during decode.

### 4. SGLang — NeurIPS 2024

[SGLang: Efficient Execution of Structured Language Model Programs](https://proceedings.nips.cc/paper_files/paper/2024/hash/724be4472168f31ba1c9ac630f15dec8-Abstract-Conference.html)

- **Contribution:** Co-designs a language frontend and runtime, reuses KV prefixes with RadixAttention, and optimizes structured output.
- **Systems connection:** Connects multi-turn calls, branching generation, and shared prompts to the redundant computation in agent workloads.
- **Reading advice: prioritize a close read.** Draw a prefix tree for a multi-turn or branching program, then follow how the runtime reuses cached work.

### 5. DistServe — OSDI 2024

[DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)

- **Contribution:** Separates prefill and decode, allocating resources and parallelism according to their distinct first-token and subsequent-token latency requirements.
- **Systems connection:** Provides a framework for reasoning about serving isolation and KV transfers. The objective is throughput that meets latency constraints, rather than aggregate throughput alone.
- **Reading advice: worth a close read.** Understand TTFT, TPOT, and goodput before tracing the additional KV movement introduced by disaggregation.

**Checkpoint:** Mark reusable prefixes and the time spent in prefill, decode, and tool waits for a multi-turn agent. Applying a paper's conclusions still requires checking the particular workload.

## Stage 3: How models execute across GPUs

**Prerequisites:** Matrix multiplication, forward and backward passes, and basic data parallelism. Read Megatron-LM and ZeRO before the MoE papers; the latter can also be deferred.

### 6. Megatron-LM — 2019, NVIDIA technical report

[Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) · [NVIDIA author page](https://research.nvidia.com/labs/adlr/MegatronLM/)

- **Contribution:** Designs model parallelism around Transformer matrix operations, splitting large layers across GPUs while controlling synchronization overhead.
- **Systems connection:** Explains communication boundaries in tensor parallelism and why training and rollout may prefer different parallel configurations.
- **Reading advice: worth a close read.** Work through row and column partitions and all-reduce for the two matrix multiplications of an MLP. This original work is labeled a technical report here; no conference publication is claimed.

### 7. ZeRO — SC 2020

[ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) · [Official publication record](https://doi.org/10.1109/SC41405.2020.00024)

- **Contribution:** Progressively shards optimizer states, gradients, and parameters across data-parallel replicas, trading communication for memory capacity.
- **Systems connection:** Establishes a training worker's memory budget and prepares the reader to understand FSDP and differences between training and generation weight layouts.
- **Reading advice: worth a close read.** Calculate bytes for parameters, gradients, and Adam states, then compare the three sharding stages.

### 8. Switch Transformers — JMLR 2022; preprint 2021

[Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://www.jmlr.org/papers/v23/21-0998.html)

- **Contribution:** Simplifies MoE to top-1 expert routing while addressing expert capacity, load balancing, and training stability.
- **Systems connection:** Introduces experts, routing, and capacity factors, separating active parameters, total parameters, and systems costs.
- **Reading advice: start with the concepts.** Ask why uneven expert loads and cross-device communication can slow execution even when per-token arithmetic is similar.

### 9. GShard — ICLR 2021; preprint 2020

[GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668) · [Conference paper](https://openreview.net/pdf/cdb90e31da8446076492c5aef0c8c6ae35dd472a.pdf)

- **Contribution:** Combines conditional MoE computation with automatic sharding to scale models while managing device partitioning, routing, and communication.
- **Systems connection:** Provides a systems starting point for expert parallelism, load imbalance, and all-to-all communication.
- **Reading advice: read selectively in depth.** After understanding Switch, trace tokens to their experts and back. Compiler details can wait. This ordering reflects learning difficulty, not publication chronology.

**Checkpoint:** Try explaining what tensor parallelism, data-parallel sharding, and expert parallelism each divide, what they save, and which communication they introduce.

## Stage 4: Connecting generation, training, and control flow

**Prerequisites:** Stage 1's serving concepts, Megatron-LM and ZeRO from Stage 3, and the basic RL loop of sampling, evaluation, and policy updates. MoE is not a prerequisite for this stage.

### 10. Ray — OSDI 2018

[Ray: A Distributed Framework for Emerging AI Applications](https://www.usenix.org/conference/osdi18/presentation/moritz)

- **Contribution:** Supports tasks and actors within one execution engine, combining distributed scheduling and fault-tolerant state management for dynamic AI workloads.
- **Systems connection:** Helps reason about rollout workers' state, resource use, and scheduling, distinguishing individual computations from long-lived services.
- **Reading advice: read selectively in depth.** Consider whether a rollout worker should be a task or an actor before reading the design. The paper describes the architecture of its time, not an operating manual for current Ray.

### 11. HybridFlow / verl — EuroSys 2025; preprint 2024

[HybridFlow: A Flexible and Efficient RLHF Framework](https://i.cs.hku.hk/~cwu/papers/gmsheng-eurosys25.pdf) · [arXiv](https://arxiv.org/abs/2409.19256)

- **Contribution:** Combines single-controller and multi-controller execution to organize RLHF data flow, and uses a 3D-HybridEngine to reduce actor resharding costs between training and generation.
- **Systems connection:** Directly connects rollout, training, device placement, and weight synchronization, making it a central entry point into verl's systems design.
- **Reading advice: prioritize a close read.** Draw generation → preparation → training, then mark weight-layout changes, communication, and device use. A first pass need not derive PPO line by line.

**Checkpoint:** Draw the critical path of one training iteration and distinguish slow generation, control overhead, weight switching, and training computation as possible bottlenecks.

## Related work and further reading

Use [[EN/wiki/llm-inference/index|LLM Inference and Serving]], [[EN/wiki/rl-infra/index|RL Infrastructure]], and [[EN/wiki/llm-serving-for-agents/index|LLM Serving for Agents]] to fill conceptual gaps. Add new papers and in-depth analyses around the questions above. Newness need not displace foundations, and routine version updates are not research contributions by themselves.

Original sources are linked under each paper heading, with publication and preprint years distinguished. The order can change when a discussion reveals missing prerequisites; finishing the entire list at once is unnecessary.

Return to the [[EN/reading/index|reading recommendations home]].
