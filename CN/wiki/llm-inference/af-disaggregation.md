---
title: "注意力-FFN 分离：切的是算子，不是阶段"
category: llm-inference
tags: [af-disaggregation, attention-ffn, megascale-infer, moe-serving, disaggregated-inference, hardware-heterogeneity]
created: 2026-05-13
updated: 2026-05-13
status: mature
---

# 注意力-FFN 分离：切的是算子，不是阶段

> [!abstract]+ TL;DR
> **AF 分离（Attention-FFN Disaggregation）** 把同一个模型的 *attention 层* 和 *FFN 层* 跑在**不同的 GPU 池**上，每层之间通过网络传递 hidden state。为什么这么做：attention 在 decode 阶段是内存带宽瓶颈（算术强度 ≈ 1–2 FLOPs/byte），FFN 在批量计算下是算力瓶颈（算术强度随 batch 增长，轻易超过 1000 FLOPs/byte）。两者最优硬件不同 —— attention 要 HBM 带宽，FFN 要原始算力。这种模式**在 DP-attention + EP-MoE 部署里结构性地隐含存在**（DeepSeek-V3 推理、SGLang/vLLM MoE serving），并被 **MegaScale-Infer**（ByteDance, 2024）显式做成系统。与 [[prefill-decode-disaggregation|PD 分离]]截然不同：PD 切 *阶段*，AF 切 *单次 forward pass 内部的算子*。

---

## 让 AF 分离值得做的"不对称性"

标准推理把 attention 和 FFN 打包在同一 GPU 上跑 —— 这是 Transformer 层结构使然。但两个算子要的东西很不一样：

| 性质 | Attention（decode 时） | FFN（大 batch 矩阵乘） |
|------|----------------------|----------------------|
| 瓶颈 | HBM 带宽（读 KV cache） | FLOPs（稠密 matmul） |
| 算术强度 | 1–2 FLOPs / byte | 取决于 batch；轻易 ≥ 1000 FLOPs / byte |
| 随 batch 扩展性 | 亚线性（每请求 KV 读主导） | 线性（更多 token = 更多 matmul） |
| 随 sequence 扩展性 | $O(S)$（每步要读全 KV cache） | 每 token 常数 |
| 最优硬件 | 高 HBM、高带宽 GPU（H100/H200、MI300X） | 高 FLOPs 加速器；算力比 HBM 重要 |
| KV cache | 自己持有 | 不碰 |
| 受益的并行 | DP attention（按请求切 KV） | TP / EP（按权重 / 专家切） |

H100 的 roofline（HBM3 3.35 TB/s，FP16 989 TFLOPs/s）拐点在 **≈295 FLOPs/byte**。低于这个点 → 内存带宽瓶颈；高于 → 算力瓶颈。Attention 在 ~1–2 处（远低于拐点），FFN 在 batched workload 下远高于拐点。同一张卡跑两个瓶颈完全相反的算子意味着**任何时刻两种资源至少有一种被浪费**：

- Attention 跑时，FP16 tensor core 闲着（等 HBM 读）。
- FFN 跑时，HBM 带宽富余（算力是瓶颈）。

AF 分离的想法是：与其在每张 GPU 上同时配满带宽 *和* 算力，不如**专业化** —— 建"attention 节点"（HBM 富）和"FFN 节点"（算力富），各自按真实瓶颈定硬件规格，激活在两池之间穿梭。

---

## 架构一张图

```
传统（层内原地执行）：

  GPU 池 ──► [LayerNorm → Attention → Add → LayerNorm → FFN → Add] × 60 层 ──► 输出
            \____________________________________________________/
                            同一硬件，同一显存

AF 分离：

  Attention 节点（HBM 富）                  FFN 节点（算力富）
  ┌──────────────────────────┐               ┌──────────────────────────┐
  │ KV cache 池               │               │ FFN 权重                  │
  │ Attention 计算            │  hidden       │ FFN 计算                  │
  │ 跨请求 DP                 │  state        │ 跨专家 TP / EP            │
  │                           │  ─────►       │                          │
  │（Layer N attention 输出） │               │（Layer N FFN 计算）       │
  │                           │  ◄─────       │（返回 Layer N 输出）      │
  └──────────────────────────┘   激活          └──────────────────────────┘
       │                                                │
       └──── 60 层每层都重复一遍此过程 ──────────────────┘
                （每 forward 60 次跨池）
```

每一层 hidden state **跨网络往返两次**（attention 输出 → FFN 输入，FFN 输出 → 下一层 attention 输入）。batch $B$、sequence $S$、hidden dim $H$、dtype 2 字节，每层单方向：

$$
\text{字节} = B \cdot S \cdot H \cdot 2
$$

具体：Llama-70B 量级，$H = 8192$，batch 64，decode（每步 1 token），每层每方向：

$$
64 \times 1 \times 8192 \times 2 = 1\,\text{MiB}
$$

× 2（往返）× 80 层 = **每 forward step 160 MiB**。在 400 Gb/s NVLink 级 fabric 上 ~3 ms；在 200 Gb/s InfiniBand 上 ~6 ms。可做，但只在胖管道上 —— 这个设计的可行空间真实但狭窄。

---

## 为什么 MoE 让 AF 分离更自然

稠密模型上 AF 分离有意思但稍勉强 —— attention 和 FFN 想要的硬件有差异但单卡也能兼顾。MoE 模型上不对称性激化到几乎是结构性强制的程度：

**(1) Attention 稠密、MoE FFN 稀疏。** Attention 每个 token、每一层都跑。MoE FFN 把每个 token 路由到 top-$k$ 个专家（如 DeepSeek-V3 是 256 选 8）。FFN 的计算模式根本不同 —— 任何时候大多数专家在自己的 GPU 上闲着，但聚合起来 FFN 计算量主导整个推理。

**(2) 两者的并行策略不兼容。**
- Attention 要 **DP**（每张 GPU 处理一部分 *请求*，配自己那片 KV cache）—— 见 [[parallelism-strategies-deep-dive#11. DP Attention — 数据并行注意力 (Data-Parallel Attention for MoE Inference)|DP Attention]]。
- MoE FFN 要 **EP**（每张 GPU 持有一部分 *专家*，AllToAll dispatch 路由 token）。
单一 GPU 池既想 DP 跑 attention 又想 EP 跑 FFN，结果就是在同一层内部的两个算子之间走 AllToAll —— 这 *本身* 就是 AF-shape 结构，只是在同一硬件上。

**(3) Attention → FFN 的过渡在 EP-MoE 部署里已经是 AllToAll。** MoE 推理每层都有这个激活传输步骤（用于专家路由）。在同一个过渡上加一道 *物理* 节点边界，复杂度增量很小。

所以实践中，**"同集群上的 DP-attention + EP-MoE" 和 "显式 AF 分离" 之间的边界很薄** —— 前者是后者的结构投影。

---

## 具体系统

### MegaScale-Infer（ByteDance, 2024）

显式 AF 分离系统。报道的设计：

- **两个物理 GPU 池** —— attention 池和 FFN 池 —— 由 RDMA 连接。
- **每层激活穿梭**（每层一次往返）。
- **Attention 和 FFN 之间 pipeline 并行**，把传输与计算重叠。层 $L$ 的 attention 与层 $L{-}1$ 的 FFN 在各自池上并发跑。
- **池内并行方式不同**：attention 池跑 DP attention（每张 attention GPU 分一片 KV cache），FFN 池跑 EP 跨专家。
- **硬件异构（可选）**：attention 池可以上 H100/H200（HBM3e 带宽）；FFN 池可以上更便宜的算力密集型硬件。

报道指标：同等硬件成本下相对标准 MoE 推理 1.7×–2.5× 吞吐提升，靠按真实瓶颈给每个池定规格。

### DeepSeek-V3 推理（结构上是 AF-shape，未显式分离）

DeepSeek-V3 生产推理在同一个集群里用 **DP attention + EP MoE**（见 [[parallelism-strategies-deep-dive#14. 实战案例：DeepSeek-V3|DeepSeek-V3 案例]]）：

- Prefill：TP=4 + SP，EP=32，DP=8，32 个冗余专家。
- Decode：TP=4 + SP，EP=320，DP=80。

Attention 和 FFN 层在集群内通过 AllToAll 通信。这是**逻辑上的 AF 分离** —— 并行边界沿着算子边界走 —— 但 *物理* 硬件是统一的，attention 和 FFN 都跑在同样的 NVL72 / H800 节点上。

MegaScale-Infer 架构是把这个再推一步：同样的逻辑结构，但在**异构物理硬件**上。

### Mooncake（Moonshot AI, FAST 2025）

Mooncake 主要是 PD 分离系统，但其 KVCache-Centric 架构以一种与 AF 分离邻近的方式把 KV 存储与计算解耦：

- 集中式 KV cache 池（CPU DRAM + SSD），多个计算节点共享。
- 计算节点按需从池里拉 KV 块。
- 这把 KV 存储硬件规格与 attention 计算硬件规格分开 —— 部分 AF 逻辑。

不是完整 AF，但走在同一条轨迹上。

---

## 激活传输代价 —— AF 何时划算

经济性取决于激活传输代价是否相对专业化收益小。

**代价侧（每层、单方向）：**

$$
t_{\text{transfer}} = \frac{B \cdot S_{\text{step}} \cdot H \cdot 2}{\text{互联带宽}}
$$

decode 下（$S_{\text{step}} = 1$），$H = 8192$、$B = 64$、NVLink (400 Gb/s = 50 GB/s)：

$$
t_{\text{transfer}} = \frac{64 \times 8192 \times 2}{50 \times 10^9} \approx 21\,\mu\text{s 每层每方向}
$$

× 2 方向 × 80 层 = **每 forward step 3.4 ms**。对比典型 50–80 ms 一个 decode forward → ~5% 开销，如果专业化带来的收益更大就值得。

**收益侧：**

如果 attention 节点能按 HBM 带宽专配（如 H200 上 HBM3e 4.8 TB/s，单位 FLOP 贵但单位字节便宜），FFN 节点用 H100 级或算力更密的加速器（单位 FLOP 便宜），总成本-单位吞吐就下降。收支平衡点取决于：

- **MoE 稀疏度** —— 越稀疏 → 计算越集中在 FFN → FFN 专配收益越大。
- **序列长度** —— 越长 → KV cache 越大 → attention 专配 HBM 收益越大。
- **Batch size** —— 越大 → FFN 越计算瓶颈 → 收益越大。
- **互联带宽** —— 越慢 → 传输开销越多吃掉收益。

经验法则（粗略，不能死扣）：AF 在以下条件齐备时划算 —— **集群足够大让硬件分级有意义**（≥ 100 GPU）、**MoE 高稀疏度**（active fraction ≤ 5%）、**池间 NVLink 级或 400+ Gbps 互联**。

---

## PD 对比 AF

| 维度 | [[prefill-decode-disaggregation\|PD 分离]] | AF 分离 |
|------|--------------------------------------|---------|
| 切什么 | 请求生命周期的**阶段**（prefill vs decode） | 单次 forward pass 内部的**算子**（attention vs FFN） |
| 跨池传输 | KV cache，每请求一次（prefill → decode 交接） | Hidden state，每**层每方向**一次（每 forward 60–80 次） |
| 传输总量 | 大但稀疏（每请求 ~MB，一次） | 小但密集（每层 ~MB，每 step） |
| 传输节奏 | 按请求 | 按层 |
| 硬件专业化动机 | 不同阶段计算特性不同（prefill 算力瓶颈、decode 内存瓶颈） | 不同算子计算特性不同（attention 内存瓶颈、FFN 算力瓶颈） |
| 不做的代价 | 长 prefill 推高 decode TBT | 瓶颈相反的硬件半层时间闲置 |
| 没有专配硬件还有用吗？ | 有 —— 仅做流量隔离也值得 | 用处变小 |
| 生产成熟度 | 主流（Splitwise、DistServe、Mooncake、DeepSeek-V3） | 新兴（MegaScale-Infer；DP-attention + EP-MoE 是结构性隐含） |
| 两者可叠加吗？ | ✓ —— 见 [[#AF 与 PD 的叠加]] | ✓ |

核心概念差别：**PD 沿时间轴切**（prefill 先做，decode 后做），跨池传输是一次性的 KV 交接。**AF 在单次 forward pass 内部切**（每层都过一次），跨池传输是每层持续的流量。

### AF 与 PD 的叠加

两者可以叠成 2×2 硬件矩阵：

```
                    │  Attention 池         │  FFN 池
────────────────────┼──────────────────────┼──────────────────
Prefill 节点         │  HBM 富，prefill      │  算力富，
                    │  attention            │  prefill FFN
                    │ （大 batch matmul）    │ （大 batch matmul）
────────────────────┼──────────────────────┼──────────────────
Decode 节点          │  HBM 富，decode       │  算力富，
                    │  attention（重 KV）    │  decode FFN
                    │                       │ （较小 batch）
```

四个池而不是两个。每池按自己的瓶颈精确定规格。这是 MegaScale-Infer 式系统正在朝大规模 MoE serving 推的配置。运维更重（要 scale 和监控更多池），但发表 benchmark 上 cluster ≥ 256 GPU 时单位吞吐成本能撑住这个复杂度。

---

## 限制

- **依赖互联带宽**：每层都要往返 hidden state。低于 NVLink 级（或 200+ Gbps RDMA）的话传输开销吃掉专业化收益。
- **Pipeline 调度复杂**：要把激活传输与计算重叠，attention 与 FFN 之间需要细粒度 PP。bubble 管理不简单。
- **异构硬件的运维代价**：生产里同时跑两种 GPU 把基础设施复杂度乘倍（驱动版本、NCCL 拓扑、监控、故障处理）。
- **主要价值在 MoE**：稠密模型受益较小 —— attention/FFN 不对称真实但比 MoE 上的弱。
- **尾延迟交互**：激活传输加一道 per-layer 延迟下限，慢的那个池决定整 forward；任一池卡住整 forward 都卡住。
- **仍然早期**：MegaScale-Infer 之外，公开的 AF 部署很少见。很多"看着像 AF"的系统（DeepSeek-V3 推理、SGLang MoE）实际上是同样硬件上的 DP-attention + EP-MoE —— *逻辑* 结构在，但 *物理* 专业化不在。

---

## 这意味着什么

AF 分离是从 PD 分离开始的"该用什么硬件就用什么硬件"这条线的自然终点。两条值得跟踪的预测：

1. **MoE serving 会把 AF 推进主流**。随着 MoE 模型变更稀疏（DeepSeek-V3 是 256 选 8，未来推向 1024+），attention/FFN 计算失衡会大到忽略它的代价超过分池运维复杂度。
2. **"硬件 tier"边界会移动**。当下大家把 H100 / H200 视为一档 GPU。随着 HBM 与算力扩展解耦（HBM4 偏带宽；tensor core 演进偏 FLOPs），attention/FFN 切分会跟着硬件切分走，AF 会成为新模型的默认部署形状。

这 *不是*：任意部署都能拿到 2× 加速。它是大规模下的成本-效率论点，前提是拓扑对、模型架构对。小稠密模型在单节点部署上拿不到任何东西。

---

## 相关阅读

- [[prefill-decode-disaggregation]] —— 跨阶段的分离（prefill vs decode）；前置模式。可与 AF 叠加。
- [[parallelism-strategies-deep-dive#11. DP Attention — 数据并行注意力 (Data-Parallel Attention for MoE Inference)]] —— DP attention 是让 AF 在单集群内变自然的并行形状。
- [[parallelism-strategies-deep-dive#14. 实战案例：DeepSeek-V3]] —— 生产 DP-attention + EP-MoE 部署；同硬件上的逻辑 AF 结构。
- [[kv-cache-optimization]] —— KV cache 压缩降低 attention 的内存带宽代价、改变 AF 的成本算式。
- [[continuous-batching]] —— 每个 AF 池 *内部* 仍然要用的调度层负载平滑。

## 参考文献

- **MegaScale-Infer**：ByteDance (2024)。*Attention/FFN-Disaggregated MoE Inference*。显式 AF 分离论文。
- **DeepSeek-V3 Technical Report** (2024)。生产 DP-attention + EP-MoE 部署，是同硬件上的结构性 AF。
- **Mooncake**：Qin et al., *Mooncake: A KVCache-Centric Disaggregated Architecture for LLM Serving*, FAST 2025。把 KV 存储与计算解耦 —— AF 的近邻。
- **DistServe** (Zhong et al., OSDI 2024) 和 **Splitwise** (Patel et al., ISCA 2024)：基础的 PD 分离系统，AF 是其架构逻辑的延伸。
