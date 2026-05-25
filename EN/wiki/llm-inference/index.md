---
title: LLM Inference & Serving
---

# LLM Inference & Serving

Topical index for this category. For foundational techniques the links go directly to the relevant section of the deep-dive page; for papers, links go to the paper-review page.

## Inference engines

- [[vllm]] — vLLM: high-throughput LLM serving with PagedAttention
- [[sglang]] — SGLang: structured generation + RadixAttention prefix cache
- [[tensorrt-llm]] — TensorRT-LLM: NVIDIA's compiled-graph inference engine

## Memory & KV cache

- [[paged-attention]] — PagedAttention: virtual memory for KV cache
- [[kv-cache-optimization]] — KV cache optimization landscape

  - [[kv-cache-optimization#Architecture-Level Optimization|Architecture-level (MQA, GQA, MLA)]]
  - [[kv-cache-optimization#Memory Management Optimization|Memory management (PagedAttention, VirtualAttention)]]
  - [[kv-cache-optimization#KV Cache Quantization|KV cache quantization]]
  - [[kv-cache-optimization#KV Cache Compression|KV cache compression (eviction, sparsification)]]
  - [[kv-cache-optimization#Prefix Caching and Sharing|Prefix caching and sharing]]
  - [[kv-cache-optimization#Distributed KV Cache|Distributed KV cache]]

## Parallelism strategies (single deep-dive, jump by section)

Canonical reference: [[parallelism-strategies-deep-dive]]. Section anchors:

- [[parallelism-strategies-deep-dive#1. Overview|Overview & combining formula]]
- [[parallelism-strategies-deep-dive#2. DP — Data Parallelism|Data Parallelism (DP)]]
- [[parallelism-strategies-deep-dive#3. ZeRO / FSDP — Sharded Data Parallelism|ZeRO / FSDP (Sharded DP)]]
- [[parallelism-strategies-deep-dive#4. TP — Tensor Parallelism|Tensor Parallelism (TP)]]
- [[parallelism-strategies-deep-dive#5. SP — Sequence Parallelism|Sequence Parallelism (SP)]] — Megatron-v2 style
- [[parallelism-strategies-deep-dive#6. PP — Pipeline Parallelism|Pipeline Parallelism (PP)]] — GPipe, 1F1B, DualPipe, Zero Bubble
- [[parallelism-strategies-deep-dive#7. CP — Context Parallelism|Context Parallelism (CP)]] — for sequences > 128K
- [[parallelism-strategies-deep-dive#8. EP — Expert Parallelism|Expert Parallelism (EP)]]
- [[parallelism-strategies-deep-dive#9. EDP/DEP — Expert Data Parallelism|EDP / DEP (Expert Data Parallelism)]]
- [[parallelism-strategies-deep-dive#10. ETP/TEP — Expert Tensor Parallelism|ETP / TEP (Expert Tensor Parallelism)]]
- [[parallelism-strategies-deep-dive#11. DP Attention — Data-Parallel Attention (MoE Inference)|DP Attention (MoE inference)]]
- [[parallelism-strategies-deep-dive#12. Combining Strategies|Combining strategies (TP × CP, 4D / 5D parallelism)]]
- [[parallelism-strategies-deep-dive#13. Selection Guide|Selection guide & decision tree]]
- [[parallelism-strategies-deep-dive#14. Case Study: DeepSeek-V3|DeepSeek-V3 case study]]

Companion paper-review pages:

- [[ring-attention]] — Ring Attention (Liu et al., ICLR 2024)
- [[deepspeed-ulysses]] — DeepSpeed Ulysses (Microsoft, 2023)

## Batching & scheduling

- [[continuous-batching]] — continuous batching (Orca / iteration-level scheduling)

  - [[continuous-batching#The Problem with Static Batching|Why static batching fails]]
  - [[continuous-batching#Continuous (Iteration-Level) Batching|How continuous batching works]]
  - [[continuous-batching#Chunked Prefill|Chunked prefill (Sarathi-Serve)]]
  - [[continuous-batching#Scheduling Strategies|Scheduling: FCFS, preemption, recompute vs swap]]

## Quantization

- [[quantization]] — quantization landscape

  - [[quantization#Weight-Only Quantization|Weight-only (GPTQ, AWQ)]]
  - [[quantization#Rotation-Based Quantization (QuIP -> QuaRot -> SpinQuant -> BDR)|Rotation-based (QuIP / QuaRot / SpinQuant)]]
  - [[quantization#FP8 Quantization|FP8 quantization]]
  - [[quantization#KV Cache Quantization|KV cache quantization]]
  - [[quantization#Activation Quantization|Activation quantization]]

- [[rotation-based-quantization]] — Rotation-based KV cache quantization family overview
- [[saw-int4]] — SAW-INT4 paper review (block-diagonal Hadamard rotation, Together AI)

## Speculative decoding

- [[speculative-decoding]] — speculative decoding overview (EAGLE, Medusa, lookahead)
- [[das-spec-rl]] — DAS paper review (distribution-aware speculative decoding for RL training)
- [[aurora]] — Aurora paper review (online speculative-decoding training as async RL on live SGLang traffic; day-0 deployment; Tree Attention)

## Disaggregated inference

- [[prefill-decode-disaggregation]] — PD disaggregation (Splitwise, DistServe, Mooncake)

  - [[prefill-decode-disaggregation#Prefill vs Decode Profiles|Compute-profile asymmetry]]
  - [[prefill-decode-disaggregation#Why Disaggregate|Why disaggregate]]
  - [[prefill-decode-disaggregation#Architecture Designs|Splitwise / DistServe / Mooncake architectures]]
  - [[prefill-decode-disaggregation#KV Cache Transfer|KV cache transfer mechanics]]
  - [[prefill-decode-disaggregation#Composing with chunked prefill|Composing with chunked prefill]]

- [[af-disaggregation]] — Attention-FFN disaggregation (MegaScale-Infer, the next axis)
- [[prfaas]] — Prefill-as-a-Service: cross-datacenter PD disaggregation over commodity Ethernet (Moonshot/Tsinghua, arXiv 2604.15039) — paper review

## Model parallelism (legacy stub)

- [[model-parallelism]] — redirects to [[parallelism-strategies-deep-dive]]
