---
title: "注意力-FFN 分离：切的是算子，不是阶段"
category: llm-inference
tags: [af-disaggregation, attention-ffn, megascale-infer, moe-serving, disaggregated-inference, hardware-heterogeneity, concept]
created: 2026-05-13
updated: 2026-05-21
status: mature
---

# 注意力-FFN 分离：切的是算子，不是阶段

> [!info] 页面元信息
> - **类型**：技术总伞页（概念综合，不是单论文精读）
> - **代表论文**：[MegaScale-Infer（ByteDance, ICML 2025）](https://arxiv.org/abs/2504.02263) —— 第一个显式 AF 分离系统
> - **相邻 / 结构性 AF 系统**：DeepSeek-V3（同硬件上 DP attention + EP MoE 的逻辑 AF）、Mooncake（KV 存储解耦架构）
> - **配套页**：[[prefill-decode-disaggregation]] —— 前置模式（PD 切阶段；AF 切算子）

---

## 摘要（2 分钟读完这一节就够）

**它是什么**。**AF 分离（Attention-FFN Disaggregation）** 把同一个模型的 *attention 层* 和 *FFN 层* 跑在**不同的 GPU 池**上，每层之间通过网络传递 hidden state。动机是一个 roofline 不对称：decode attention 在 $1\text{–}2$ FLOPs/byte（内存带宽瓶颈，远低于 H100 的 295 FLOPs/byte 拐点），而 $B \geq 64$ 的批量 FFN 轻易超过 $1000$ FLOPs/byte（算力瓶颈）。同一张卡跑两个瓶颈相反的算子 —— 一类资源永远在被浪费。

**核心思想**。把 attention 和 FFN 当成两个瓶颈相反的算子，分别放到按各自真实资源需求配规格的硬件池上。三个支柱：

1. **算子级专业化** —— attention 节点按 HBM 带宽配规格；FFN / Expert 节点按原始算力配规格。MoE 稀疏让不对称更尖锐（FFN 在低 batch 也变内存瓶颈，高 batch 才回到算力瓶颈；attention 全程保持内存瓶颈）。
2. **每层激活传输** —— 每层在两池之间传 hidden state 一来一回。NVLink 级 fabric 下代价约 forward 的 5%，专业化收益可以覆盖。
3. **乒乓 pipeline 并行** —— 把 batch 切成 micro-batch 在 attention 与 FFN 池之间往返穿梭，让任一池都在计算另一池不在的那一部分（MegaScale-Infer 的贡献）。

少任一支柱 AF 就崩：失去专业化 → 没有收益；失去快互联 → 传输开销吃掉收益；失去乒乓 → 任一池一半时间在等。

**代表实现**。

| 系统 | 做什么 | AF 化程度 |
| ---- | ------ | -------- |
| [MegaScale-Infer](https://arxiv.org/abs/2504.02263)（ByteDance, ICML 2025） | 两个物理池（Attention Node + Expert Node）由自研 **M2N** RDMA 库连接；乒乓 PP；可选异构硬件 | 显式 |
| [DeepSeek-V3](https://arxiv.org/abs/2412.19437) 生产推理 | 单集群 DP attention + EP MoE；跨算子边界 AllToAll | 逻辑（同构硬件） |
| [Mooncake](https://arxiv.org/abs/2407.00079)（Moonshot AI, FAST 2025） | KV-cache-centric PD 系统，把 KV 存储与计算解耦 | 部分（KV 存储轴） |

MegaScale-Infer 在 Mixtral 8×22B / DBRX / Scarlett / DeepSeek-V2 上对比 SOTA 报告 **最高 1.90× 每 GPU 吞吐**（Ampere 和 Hopper 上都做了实验）。

**为什么这重要**。

- **MoE serving 成本将被 AF-shape 的部署主导**。MoE 稀疏度在变大（DeepSeek-V3 是 256 选 8，未来推向 1024+），attention/FFN 计算失衡会大到忽略它的代价超过分池运维复杂度。
- **"硬件 tier" 边界会跟着移动**。HBM 型（HBM4）和算力型（下一代 tensor core）加速器分化，AF 给你天然的接缝把正确的硅片放到正确的算子下面。
- **可以与 PD 叠加**。AF 与 [[prefill-decode-disaggregation|PD]] 切在正交的两个轴上（算子 vs. 阶段），在旗舰规模上叠成 2×2 硬件矩阵。

---

# 深度部分（往下展开细节）

上面摘要是 executive 层。下面是 roofline 数学、三种系统设计、以及成本算式，给愿意细读的人。

## 背景：为什么需要发明 AF 分离

标准推理把 attention 和 FFN 打包在同一 GPU 上跑 —— 这是 Transformer 层结构使然：`attention → FFN → attention → FFN → ...`。但两个算子的硬件需求差别很大，在规模上这种不对称性在统一硬件上要付真金白银。

H100 的 roofline（HBM3 3.35 TB/s，FP16 989 TFLOPs/s）拐点在 **≈ 295 FLOPs/byte**。低于拐点的算子是内存带宽瓶颈；高于拐点是算力瓶颈：

| 性质 | Attention（decode 时） | FFN（大 batch 矩阵乘） |
|------|----------------------|----------------------|
| 算术强度 | 1–2 FLOPs / byte | 取决于 batch；$B = 64$ 时轻易 $\geq 1000$ FLOPs / byte |
| 瓶颈 | HBM 带宽（读 KV cache） | FLOPs（稠密 matmul） |
| 随 batch 扩展性 | 亚线性（每请求 KV 读主导） | 线性（更多 token = 更多 matmul） |
| 随 sequence 扩展性 | $O(S)$（每步要读全 KV cache） | 每 token 常数 |
| 最优硬件 | 高 HBM、高带宽 GPU（H100/H200、MI300X） | 高 FLOPs 加速器；算力比 HBM 重要 |
| KV cache | 自己持有 | 不碰 |
| 受益的并行 | DP attention（按请求切 KV） | TP / EP（按权重 / 专家切） |

同一张卡跑两个瓶颈完全相反的算子意味着 **任何时刻两种资源至少有一种被浪费**：

- Attention 跑时，FP16 tensor core 闲着（等 HBM 读）。
- FFN 跑时，HBM 带宽富余（算力是瓶颈）。

MegaScale-Infer Fig. 1 直接把这种不对称画了出来：

![Attention 与 FFN 在稠密 / MoE / AF 部署下的 GPU 利用率（MegaScale-Infer Fig. 1）](CN/wiki/llm-inference/af-disaggregation-figs/gpu-utilization-comparison.png)

稠密模型 (a) 里 FFN 在 batch 不大时就把 GPU 算力打满，attention 一直在低位。MoE (b) 里差距更大：每个 expert 只看到 batch 的一小片，FFN 需要 *更大* 的全局 batch 才能打满，但 attention 算术强度不变。MegaScale-Infer (c) 里把两个池分开，各自往自己的 `max bs` 推 —— FFN 曲线被往左拉（单位成本的利用率更高），attention 曲线靠 DP 扩容往上抬。

自然回应是 **专业化** —— 建 "attention 节点"（按 HBM 带宽配规格）和 "expert 节点"（按算力配规格），各自按真实瓶颈以最优单位资源成本配规格，激活在两池之间穿梭。这正是 [[prefill-decode-disaggregation|PD 分离]] 在 *阶段* 层面做的事 —— AF 把同样的想法延伸到单次 forward pass 内部的 *算子* 层面。

| 与 X 对比 | 切什么 | 跨池传输节奏 | 硬件专业化动机 |
| --------- | ------ | ----------- | -------------- |
| **PD 分离** | 阶段（prefill vs decode） | 按请求（KV 交接，一次） | prefill 算力瓶颈 vs decode 内存瓶颈 |
| **AF 分离** | 算子（attention vs FFN） | 每层每方向（每 forward 60–80 次） | attention 内存瓶颈 vs FFN 算力瓶颈 |

## 核心思想：按算子瓶颈专业化硬件

> [!quote] 一句话总结贡献
> 把 attention 和 FFN 当成两个瓶颈相反的算子，分别放到按各自真实资源需求配规格的硬件池上，付每层一次激活往返的代价，换不在每张 GPU 上浪费一半资源。

三个支撑次级声明：

- **算力 / 带宽不对称是真实的**。Decode attention 在 $\sim 1\text{–}2$ FLOPs/byte（远低于 H100 的 295 拐点）；batched FFN 远高于拐点。统一硬件下必有一类资源浪费。
- **MoE 让不对称更尖锐**。Attention 稠密；MoE FFN 稀疏。每个 expert 只看到 batch 的一小片，所以 FFN 需要更大的 *全局* batch 才打到 roofline，但 attention 强度不变。DP attention 和 EP MoE 的并行不兼容，本来就需要跨算子 AllToAll —— AF 的结构缝隙已经在了。
- **每层传输代价在 NVLink 级 fabric 上可接受**。$H = 8192$、$B = 64$、decode 时每层每方向 ~1 MiB。× 80 层 × 2 方向 = 每 forward step ~160 MiB。400 Gb/s NVLink 上 ~3.4 ms —— 相对典型 50–80 ms forward 是 ~5 %，专业化收回更多就值。

去掉任何一个：失去不对称论点 AF 不省什么；失去 MoE 稠密模型收益缩小；失去快速互联 transfer 开销吃掉收益。

## 实现细节

### 架构一张图

MegaScale-Infer runtime instance 架构（论文 Fig. 3）是 AF 的标准物理布局：

![MegaScale-Infer runtime instance 架构（论文 Fig. 3）](CN/wiki/llm-inference/af-disaggregation-figs/megascale-infer-architecture.png)

两个物理池 —— **Attention Node**（按请求 $M$ 重复，持有 attention 参数 + KV cache，节点内 TP + 节点间 DP）和 **Expert Node**（按 $N$ 个 expert 分片，只持有该 expert 的参数，节点内 TP + 节点间 EP）。两池由自研的两个 RDMA primitive 连接 —— **M2N**（Attention → Expert dispatch）和 **N2M**（Expert → Attention combine），跑在 IB/Ethernet 上。整体作为 **乒乓 Pipeline** 运行：层 $L$ 的 attention micro-batch 在一个池上跑，同时层 $L{-}1$ 的 FFN micro-batch 在另一个池上跑。

或者用 ASCII 画出每层数据流相对传统 in-place 执行的差别：

```
传统（层内原地执行）：

  GPU 池 ──► [LayerNorm → Attention → Add → LayerNorm → FFN → Add] × 60 层 ──► 输出
            \____________________________________________________/
                            同一硬件，同一显存

AF 分离：

  Attention 节点（HBM 富）                  FFN / Expert 节点（算力富）
  ┌──────────────────────────┐               ┌──────────────────────────┐
  │ KV cache 池              │               │ FFN / expert 权重         │
  │ Attention 计算           │   M2N         │ Expert 计算               │
  │ 跨请求 DP                │  ─────►       │ 跨专家 EP                 │
  │                          │               │                          │
  │（Layer N attention 输出）│   N2M         │（Layer N FFN 计算）       │
  │                          │  ◄─────       │（返回 Layer N 输出）      │
  └──────────────────────────┘   激活        └──────────────────────────┘
       │                                                │
       └──── 60 层每层都重复一遍此过程 ──────────────────┘
                （每 forward 60 次跨池）
```

### 激活传输代价

batch $B$、sequence step $S_{\text{step}}$、hidden dim $H$、dtype 2 字节，每层单方向：

$$
\text{每层每方向字节} = B \cdot S_{\text{step}} \cdot H \cdot 2
$$

具体（Llama-70B 量级，$H = 8192$，batch 64，decode step）：

$$
64 \times 1 \times 8192 \times 2 = 1\,\text{MiB 每层每方向}
$$

× 2 方向 × 80 层 = **每 forward step 160 MiB**。在 400 Gb/s NVLink 级 fabric（~50 GB/s）上 ~3.4 ms；在 200 Gb/s InfiniBand 上 ~6.8 ms。NVLink 上可行，IB 边缘，没有胖管道不可行。

> [!example] decode 每 step 传输数学
> | 量 | 值 |
> | -- | -: |
> | 400 Gb/s NVLink 上每层单方向（$B = 64$、$H = 8192$） | $\approx 21$ μs |
> | 每 forward step 总传输（× 2 × 80 层） | $\approx 3.4$ ms |
> | 典型 decode forward 时间 | 50–80 ms |
> | 传输开销 | ~5 % |

### 为什么 MoE 让 AF 自然

稠密模型上 AF 有意思但稍勉强 —— attention 和 FFN 想要的硬件有差异但单卡也兼顾。MoE 上不对称性激化到 AF 几乎是结构性强制：

**1. Attention 稠密、MoE FFN 稀疏**。Attention 每个 token、每一层都跑。MoE FFN 把每个 token 路由到 top-$k$ 个专家（如 DeepSeek-V3 是 256 选 8）。两个算子计算模式根本不同。

**2. 两者的并行策略不兼容**。

- Attention 要 **DP**（每张 GPU 处理一部分 *请求*，配自己那片 KV cache） —— 见 [[parallelism-strategies-deep-dive#11. DP Attention — 数据并行注意力 (Data-Parallel Attention for MoE Inference)|DP Attention]]。
- MoE FFN 要 **EP**（每张 GPU 持有一部分 *专家*；AllToAll dispatch 路由 token）。

单一 GPU 池既想 DP 跑 attention 又想 EP 跑 FFN，结果就是在同一层的两个算子之间走 AllToAll —— 这 *本身* 就是 AF-shape 结构，只是在同一硬件上。

**3. attention → FFN 过渡本来就是跨算子 AllToAll**（EP-MoE 部署里）。在同一过渡上加一道物理节点边界，复杂度增量很小。

所以 "同集群上的 DP-attention + EP-MoE" 和 "显式 AF 分离" 之间的边界很薄 —— 前者是后者的结构投影。

### 乒乓 pipeline 并行（MegaScale-Infer）

朴素地把 attention 和 FFN 放到两个池，必然有一个池在另一个算的时候空闲。MegaScale-Infer 用 **micro-batch** 把 batch 切片，在两池之间做乒乓往返：

- Micro-batch A：attention(L) → M2N → expert(L) → N2M → attention(L+1) → …
- Micro-batch B：A 在 expert 池时，B 在 attention 池 —— 反之亦然。

两个 micro-batch 互错开，两池都一直在算，M2N/N2M 的传输被压在计算下面。两池的池内并行（TP 内 + DP/EP 外）相互独立，所以每池都能独立配规格打满自己的瓶颈，互不影响。

代价是延迟：乒乓把 pipeline 填满，但要等一个 micro-batch 时间才出第一个 token。对高并发 decode（吞吐是关键指标）这是好买卖。

### AF 与 PD 的叠加

两者可以叠成 2×2 硬件矩阵：

```
                    │  Attention 池         │  FFN / Expert 池
────────────────────┼──────────────────────┼──────────────────
Prefill 节点         │  HBM 富，prefill      │  算力富，
                    │  attention            │  prefill FFN
                    │ （大 batch matmul）    │ （大 batch matmul）
────────────────────┼──────────────────────┼──────────────────
Decode 节点          │  HBM 富，decode       │  算力富，
                    │  attention（重 KV）    │  decode FFN
                    │                       │ （较小 batch）
```

四个池而不是两个。每池按自己的瓶颈精确定规格。这是 MegaScale-Infer 式系统朝大规模 MoE serving 推的配置。运维更重，cluster $\geq 256$ GPU 时单位吞吐成本撑得住。

## 具体系统

技术总伞页的 "Experiments" 槽 —— 已知部署及其 AF 化程度，从最显式到最相邻排列。

### MegaScale-Infer（ByteDance, ICML 2025）

显式 AF 分离系统。[arXiv:2504.02263](https://arxiv.org/abs/2504.02263)。

| 维度 | 设计 |
| ---- | ---- |
| 池布局 | 两个物理池 —— *Attention Node*（$M$ 倍复制，内部 TP）和 *Expert Node*（按 $N$ 个 expert 分片，内部 TP + 跨 expert EP），由 RDMA 连接 |
| 激活传输 | 每层一次往返，通过自研 **M2N**（attention → expert dispatch）和 **N2M**（expert → attention combine）库 —— 消除了拖累 MoE AllToAll 的 GPU↔CPU 拷贝、group init 开销、GPU 同步 |
| Pipeline | 乒乓 pipeline 并行 —— 把 batch 切成 micro-batch 在两池之间往返，确保两池都一直在算 |
| 池内并行 | Attention：跨请求 TP + DP；Expert：内部 TP + 跨 expert EP |
| 硬件 | 可选异构 —— attention 池可用 HBM-rich tier（H100/H200），expert 池可用算力密集 tier；同构也可以 |
| 报告增益 | 对比 SOTA 在 Mixtral 8×22B / DBRX / Scarlett / DeepSeek-V2 上 **最高 1.90× 每 GPU 吞吐**；Ampere、Hopper 都做了实验 |

M2N 库是最深的工程贡献：原生 MoE token dispatch 走 `torch.distributed` 的 AllToAll 有尾延迟病态（论文 Fig. 5），团队基于 GDRCopy 用独立的 Sender/Receiver 状态机从零重写了这个 primitive（Fig. 6–7）。

### DeepSeek-V3 推理（结构上是 AF，统一硬件）

DeepSeek-V3 生产推理在同一集群里用 **DP attention + EP MoE**（见 [[parallelism-strategies-deep-dive#14. 实战案例：DeepSeek-V3]]）：

| 阶段 | TP | EP | DP | 备注 |
| ---- | -- | -- | -- | ---- |
| Prefill | 4 + SP | 32 | 8 | 32 个冗余专家 |
| Decode | 4 + SP | 320 | 80 | — |

Attention 和 FFN 层在集群内通过 AllToAll 通信。这是 **逻辑上的 AF 分离** —— 并行边界沿着算子边界走 —— 但物理硬件是统一的 NVL72 / H800 节点。MegaScale-Infer 把同样的逻辑结构推到异构物理硬件上，并加上乒乓 PP 来隐藏传输代价。

### Mooncake（Moonshot AI, FAST 2025）

主要是 [[prefill-decode-disaggregation|PD 分离]] 系统，但其 KVCache-Centric 架构以一种与 AF 邻近的方式把 KV 存储与计算解耦：

- 集中式 KV cache 池（CPU DRAM + SSD），多个计算节点共享。
- 计算节点按需从池里拉 KV 块。
- 把 KV 存储硬件规格与 attention 计算硬件规格分开 —— 存储轴上的部分 AF 逻辑。

不是完整 AF，但走在同一条轨迹上。

### AF 何时划算

经济性取决于激活传输代价是否相对专业化收益小。

**代价侧**。每层单方向传输时间（decode，$S_{\text{step}} = 1$）：

$$
t_{\text{transfer}} = \frac{B \cdot H \cdot 2}{\text{互联带宽}}
$$

$H = 8192$、$B = 64$、400 Gb/s NVLink（= 50 GB/s）：$t_{\text{transfer}} \approx 21$ μs 每层每方向。

**收益侧**。取决于：

- **MoE 稀疏度** —— 越稀疏 → 每个活跃 expert 的 FFN 计算越集中 → FFN 专配收益越大。
- **序列长度** —— 越长 → KV cache 越大 → attention 专配 HBM 收益越大。
- **Batch size** —— 越大 → FFN 越计算瓶颈 → 收益越大。
- **互联带宽** —— 越慢 → 传输开销越多吃掉收益。

> [!tip] 粗略法则
> AF 在 **集群足够大让硬件分级有意义**（$\geq 100$ GPU）、**MoE 高稀疏度**（active fraction $\leq 5$%）、**池间 NVLink 级或 400+ Gbps RDMA** 时划算。

## 优势与限制

最强两点：(1) **硬件专业化机会真实且在变大** —— HBM 型（HBM4）和 FLOPs 型（下一代 tensor core）加速器分级分化时，AF 的根本不对称会更尖锐；(2) **MoE 的结构性 AllToAll** 意味着 "单集群 DP-attention + EP-MoE" 和 "双集群显式 AF" 之间的距离很小 —— 工程上大部分已经搭好。

诚实承认的限制：

- **依赖互联带宽**。每层都要往返 hidden state。低于 NVLink 级（或 200+ Gbps RDMA）的话传输开销吃掉专业化收益。
- **乒乓加一道延迟下限**。Micro-batch 切分意味着要等一个 micro-batch 时间才出第一个 token；对吞吐友好，对 TTFT 不友好。
- **异构硬件的运维代价**。生产里同时跑两种 GPU tier 把基础设施复杂度乘倍（驱动版本、NCCL 拓扑、监控、故障处理）。
- **主要价值在 MoE**。稠密模型受益较小 —— attention/FFN 不对称在它们上面更弱。
- **尾延迟交互**。激活传输加一道 per-layer 延迟下限，慢的那个池决定整 forward；任一池卡住整 forward 都卡住。
- **仍然早期**。MegaScale-Infer 之外，公开的 AF 部署很少见。很多 "看着像 AF" 的系统（DeepSeek-V3 推理、SGLang MoE）实际上是同样硬件上的 DP-attention + EP-MoE —— *逻辑* 结构在，但 *物理* 专业化不在。

> [!warning] PD vs AF 一张表
> 都带 "分离" 两个字容易混，但切在正交的两个轴上。
>
> | 维度 | [[prefill-decode-disaggregation\|PD 分离]] | AF 分离 |
> | ---- | --------------------------------------- | ------- |
> | 切什么 | 时间轴上的**阶段** | 单次 forward pass 内部的**算子** |
> | 跨池传输 | KV cache，每请求一次 | Hidden state，每层 × 每 step |
> | 传输节奏 | 按请求 | 按层 |
> | 没专配硬件还有用吗 | 仍然有用（流量隔离） | 用处变小 |
> | 成熟度 | 主流（Splitwise、DistServe、Mooncake） | 新兴（MegaScale-Infer） |
> | 两者可叠加吗 | ✓ —— 见 *AF 与 PD 的叠加* | ✓ |

## 这意味着什么

两条值得跟踪的预测：

1. **MoE serving 会把 AF 推进主流**。随着 MoE 模型变更稀疏（DeepSeek-V3 是 256 选 8，未来推向 1024+），attention/FFN 计算失衡会大到忽略它的代价超过分池运维复杂度。
2. **"硬件 tier" 边界会移动**。当下大家把 H100 / H200 视为一档 GPU。随着 HBM 扩展（HBM4 偏带宽）与算力扩展（下一代 tensor core 偏 FLOPs）解耦，attention/FFN 切分会跟着硬件切分走，AF 会成为新旗舰 MoE 模型的默认部署形状。

这 *不是*：任意部署都能拿到 2× 加速。它是大规模下的成本-效率论点，前提是拓扑对、模型架构对。小稠密模型在单节点部署上拿不到任何东西。

## 源码与复现

到 2026 年中没有广泛部署的开源 AF 分离框架。相关系统状态：

| 系统 | 你能跑什么 | 闭源部分 |
| ---- | ---------- | -------- |
| MegaScale-Infer (ByteDance) | 仅论文 —— [arXiv:2504.02263](https://arxiv.org/abs/2504.02263) | 所有代码，M2N 库 |
| DeepSeek-V3 推理（逻辑 AF） | 开放权重；DP-attention + EP-MoE via vLLM / SGLang 单集群 | 生产部署拓扑 |
| Mooncake | 开放：[github.com/kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake) —— KV 池 / Transfer Engine | 完整 PD + KV 池在托管服务里的集成 |
| vLLM + SGLang DP-attention | DP attention 存在；AF 分离不存在（暂时） | — |

**今天用统一硬件近似 AF 的做法**：在单集群上用 vLLM 或 SGLang 部署 DP-attention + EP-MoE。并行边界沿算子边界走，所以你拿到逻辑 AF 结构，但没有异构硬件的物理专业化。[[parallelism-strategies-deep-dive#14. 实战案例：DeepSeek-V3|DeepSeek-V3 公布的拓扑]] 是标准 recipe。

**要近似真正 AF**：得 fork 一个推理引擎加 (a) 跨池 RDMA hidden-state 传输（相当于 M2N/N2M）、(b) 乒乓 micro-batch 调度器把激活传输与计算重叠、(c) per-pool 并行配置。工程量不小 —— 不是周末项目。

## 相关阅读

- [[prefill-decode-disaggregation]] —— 跨阶段的分离（prefill vs decode）；前置模式。可与 AF 叠加成 2×2 硬件矩阵。
- [[parallelism-strategies-deep-dive#11. DP Attention — 数据并行注意力 (Data-Parallel Attention for MoE Inference)]] —— DP attention 是让 AF 在单集群内变自然的并行形状。
- [[parallelism-strategies-deep-dive#14. 实战案例：DeepSeek-V3]] —— 生产 DP-attention + EP-MoE 部署；同硬件上的逻辑 AF 结构。
- [[kv-cache-optimization]] —— KV cache 压缩降低 attention 的内存带宽代价、改变 AF 的成本算式。
- [[continuous-batching]] —— 每个 AF 池 *内部* 仍然要用的调度层负载平滑。
- [[vllm]] / [[sglang]] —— 你要加 AF 支持时会 fork 的推理引擎。

## 参考文献

- **MegaScale-Infer**：Zhu et al., *MegaScale-Infer: Serving Mixture-of-Experts at Scale with Disaggregated Expert Parallelism*, ICML 2025 / ByteDance。[arXiv:2504.02263](https://arxiv.org/abs/2504.02263) —— 显式 AF 分离论文。
- **DeepSeek-V3 tech report**：生产 DP-attention + EP-MoE 部署，是同硬件上的结构性 AF。[arXiv:2412.19437](https://arxiv.org/abs/2412.19437)
- **Mooncake**：Qin et al., *Mooncake: A KVCache-Centric Disaggregated Architecture for LLM Serving*, FAST 2025。[arXiv:2407.00079](https://arxiv.org/abs/2407.00079) —— KV 存储与计算解耦，与 AF 邻近。
- **DistServe**：Zhong et al., OSDI 2024。[arXiv:2401.09670](https://arxiv.org/abs/2401.09670) —— 基础 PD 分离系统，AF 是其架构逻辑的延伸。
- **Splitwise**：Patel et al., ISCA 2024。[arXiv:2311.18677](https://arxiv.org/abs/2311.18677) —— 原 PD 分离论文。
