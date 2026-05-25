---
title: "PrfaaS：下一代模型的 KVCache 可以跨数据中心"
category: llm-inference
tags: [prfaas, pd-disaggregation, cross-datacenter, kvcache, hybrid-attention, mooncake, moonshot, paper-review]
created: 2026-05-22
updated: 2026-05-22
status: mature
paper: arXiv:2604.15039
code: not-released
---

# PrfaaS：下一代模型的 KVCache 可以跨数据中心

> [!info] 论文信息
> - **论文**：[arXiv:2604.15039](https://arxiv.org/abs/2604.15039) —— *Prefill-as-a-Service: KVCache of Next-Generation Models Could Go Cross-Datacenter*，v1 2026-04-16，v2 2026-04-22
> - **作者**：Ruoyu Qin¹², Weiran He¹, Yaoyu Wang¹, Zheming Li¹, Xinran Xu¹, Yongwei Wu², Weimin Zheng², Mingxing Zhang²
> - **单位**：¹Moonshot AI，²清华大学
> - **通讯**：Mingxing Zhang（zhang_mingxing@mail.tsinghua.edu.cn）—— 跟 [[prefill-decode-disaggregation|Mooncake]] 同一位主导
> - **代码**：未发布；论文提到 "in-house vLLM"，基于 [[vllm]] 的 hybrid KVCache manager fork

---

## 摘要（2 分钟读完这一节就够）

**PrfaaS 是什么**。一个**跨数据中心**的 PD 分离架构：长上下文 prefill 跑在算力密集 GPU 集群上（H200、Rubin CPX），生成的 KVCache 通过**普通以太网**传到另一个 PD 集群（H20、LPU 这类内存带宽优化的硬件）去做 decode。背后是 Moonshot/清华那个 [[prefill-decode-disaggregation|Mooncake]] 团队，把 PD 分离再往外推一层 —— 从"单 DC 内 RDMA 紧耦合"推到"DC 间以太网松耦合"。

**核心思想**。Hybrid attention 模型（Kimi Linear、MiMo-V2-Flash、Qwen3.5-397B、Ring-2.5-1T）把 KVCache 大小相比 dense GQA 砍了大约 13×，**把 PD 分离的可部署网络边界从 RDMA 类 fabric 推到普通以太网**。但单靠 KVCache 变小不够 —— 生产流量是 bursty 的、长度分布严重偏斜、prefix cache 局部性不均、DC 间带宽抖动。PrfaaS 用三件系统侧的东西补齐：(1) **长度阈值选择性外放** —— 只有 uncached prefix $l > t$ 的请求跨 DC，短请求留本地；(2) **混合 prefix cache 池** —— 把 full-attention 的 block 级 KV 和 linear-attention 的 request 级 recurrent state 分组管理但共享 block 池，区分 prefix-cache 块和 transfer-cache 块；(3) **双时间尺度调度** —— 短期带宽 + cache 感知路由响应拥塞，长期按流量重新分配 PD 集群的 prefill/decode 比例。

**标志数字**。1T Kimi-Linear 风格 hybrid 模型（KDA:MLA 3:1）的案例研究，32 H200 PrfaaS + 64 H20 本地 PD vs 96 H20 同构 baseline：

| 指标 | 同构 PD (96 H20) | Naive 异构（无调度） | **PrfaaS-PD** |
| ---- | --------------: | -------------------: | ------------: |
| $\Lambda_{\max}$ (req/s) | 2.11 | 2.45 | **3.24** |
| 加速比 | 1.00× | 1.16× | **1.54×** |
| Mean / P90 TTFT (s) | 4.44 / 9.73 | 1.74 / 3.51 | 2.22 / **3.51** |
| 跨 DC 带宽消耗 | — | — | **13 Gbps（100 Gbps 链路的 13 %）** |

**吞吐 +54 %，P90 TTFT −64 %** vs 同构；等成本下吞吐增益约 **+15 %**。最关键的 naive-heterogeneous 对照：拿掉调度器损失约 25 % 的吞吐增益 —— 选择性外放 + 长度阈值路由是实在干活的，不只是硬件混搭的红利。

**为什么重要**。

- **填上"异构 PD"的坑**。各家在出 phase-specialized 芯片（[NVIDIA Rubin CPX](https://www.nvidia.com/en-us/data-center/products/rubin-cpx/) 为 prefill 优化、[Groq LPU](https://wow.groq.com/lpu-inference-engine/) 和 Taalas HC1 为 decode 优化）。在这之前没有 RDMA 级别 fabric 横跨异构加速器，部署不到一起。PrfaaS 让普通以太网够用了。
- **Mooncake → PrfaaS 是连贯论题**。把 KVCache 当一等系统资源从 2024 [[prefill-decode-disaggregation|Mooncake]] 开始；PrfaaS 把同一思路推到 DC 边界之外。跨集群 KV manager 是 Mooncake 全局 KV 池的自然续作。
- **首个 hybrid-attention 感知的 PD 调度公开方案**。混合 prefix cache 池（linear attention recurrent state + full attention block KV 在统一 manager 里）是让新一代 hybrid 模型 PD 可服务的关键。
- **2026-27 开放系统议程**。跨 DC KV 传输成为真实负载；预期 vLLM / SGLang / Dynamo 加入跨集群 KV 连接器；DC 间网络设计（10G→100G→400G 以太网）开始针对 KV 流量模式优化。

---

# 深度部分（往下展开细节）

## 背景：传统 PD 的带宽墙

PD 分离（Splitwise、DistServe、[[prefill-decode-disaggregation|Mooncake]]）把 prefill（compute-bound、$O(S^2)$ FLOPs）和 decode（memory-bandwidth-bound、每步 $O(S)$ DRAM 读）干净分离。这种分离允许 phase 特定优化，但 KVCache 从 prefill 导出到 decode 现在变成了*跨节点传输问题*。DC 内部署里 NVLink + RDMA 把这部分流量无形吸收掉。一旦想把 prefill 和 decode 推到不同 DC，KVCache 传输就成了约束。

论文把这量化成模型实例的 **KV 吞吐**：

$$
\Phi_{kv}(l) = \frac{S_{kv}(l)}{T_{\text{prefill}}(l)}
$$

对一个 $N$-GPU 集群、每实例并行度 $P$，避免 GPU 空闲所需的最小出口带宽：

$$
B_{\text{out}} = \frac{N}{P} \cdot \frac{\mathbb{E}[S_{kv}]}{\mathbb{E}[T_{\text{prefill}}]} \approx \frac{N}{P} \cdot \Phi_{kv}(L_{\text{avg}})
$$

对一个 512-GPU prefill 集群在 $L_{\text{avg}}=32K$ 上跑 dense 模型（GQA 的 MiniMax-M2.5），需要约 **3.8 Tbps** 出口带宽 —— 跨 DC 完全不现实。Qwen3-235B 需要 **2.1 Tbps**。传统 PD 因此被锁在单一 RDMA 级 fabric 岛里。

### Hybrid attention 怎么改变这幅图

变化在模型侧，不是系统侧。Hybrid stack 把少量 full-attention 层和大量 linear-complexity 层（linear attention、SWA 等）交错。只有 full-attention 层产出随序列长度增长的 KVCache；linear 层维护固定大小的 recurrent state。

KV 吞吐对比表（8×H200，SGLang v0.5.9，论文 Table 3）：

| 序列长 | **Kimi Linear** | **MiMo-V2-Flash** | **Qwen3.5-397B** | **Ring-2.5-1T** | MiniMax-M2.5 (dense) | Qwen3-235B (dense) |
| ----- | --------------: | ----------------: | ---------------: | --------------: | -------------------: | -----------------: |
| 1K | 1.19 Gbps | 0.82 | 4.13 | 7.27 | 4.94 | 4.12 |
| 8K | 2.29 | 2.85 | 6.28 | 4.47 | 32.87 | 22.42 |
| 32K | 3.87 | 4.66 | 8.25 | 2.59 | **59.93** | **33.35** |
| 128K | 4.88 | 4.71 | 7.47 | 1.46 | 47.82 | 21.50 |

32K tokens 上 **MiMo-V2-Flash 4.66 Gbps vs MiniMax-M2.5 59.93 Gbps —— 13× 减少**。Ring-2.5-1T 在 128K 上降到 1.46 Gbps。论文还指出 Ring-2.5-1T MLA 比 GQA 压缩约 4.5×，7:1 hybrid 比例再压 ~8×，**总共约 36× KV 内存节省**。

重新算 512-GPU 部署：hybrid Ring-2.5-1T 只需要 **170 Gbps**；把 $l>128K$ 的请求才路由到 PrfaaS 还能把带宽降到 **100 Gbps 以下**。10,000-GPU DC 的聚合 KV 出口带宽约 **1.8 Tbps**，舒适地落在现代 DC 间链路能力之内。

> [!note] 锚定全文的两轴 attention 分类（Table 2）
> | 机制 | Prefill 延迟 | KV 吞吐 |
> | ---- | ----------- | ------- |
> | GQA | 高 | 高 |
> | MLA | 高 | 低 |
> | Sparse attention | 低 | 高 |
> | SWA | 低 | 低 |
> | Linear attention | 低 | 低 |
>
> Hybrid stack 选"低 / 低" —— 这正是跨 DC PrfaaS 工作的前提。Sparse attention 在这架构里恰恰是错的 trade-off（compute 便宜，但网络仍贵）。

### 异构硬件为什么需要这个

phase 特化芯片已经存在：
- **Prefill 侧**：NVIDIA Rubin CPX（compute-dense）、H200（案例研究用的）
- **Decode 侧**：Groq LPU、Taalas HC1、H20（案例研究用的）—— 极致内存带宽

但高性能互联紧绑定到芯片 form factor 和部署环境上。把异构加速器塞进同一个 RDMA 岛需要定制工程，而且会继承固定的 prefill-decode 硬件比例 —— 这在生产里是致命的，请求 mix 不停变。PrfaaS 绕开这个，接受 prefill 和 decode 可以*完全在不同 DC*。

## PrfaaS-PD 架构

![PrfaaS-PD 部署拓扑（论文 Fig. 3）](CN/wiki/llm-inference/prfaas-figs/prfaas-topology.png)

三个子系统：

| 子系统 | 组成 |
| ----- | ---- |
| **Compute** | PrfaaS 集群（同质硬件、算力密集、只跑 prefill）+ 本地 PD 集群（同质硬件、带宽优化、走常规 PD） |
| **Network** | 集群内 RDMA（延迟敏感的集合通信 + PD KV 传输）+ 集群间以太网，靠 VPC peering 或专线（跨 DC KV 传输） |
| **Storage** | 每个集群一个分布式 hybrid prefix cache 池 + 全局 KVCache manager 跨集群跟踪元数据 |

之上有一个全局调度器按长度 / cache 局部性 / 网络状态路由请求。

### 组件 1 —— 长度阈值路由

让选择性外放说得通的核心机制：

```
请求到达，uncached prefill 长度 l：
  if l > t：路由到 PrfaaS 集群       （长上下文、compute-bound）
  if l ≤ t：路由到本地 PD-P          （短的、memory/comm-bound）
```

直觉：短 prefill 在本地 PD 规模下通常是 memory-bound 或 communication-bound，PrfaaS 的高算力加速器跑它们是浪费。只有长 prefill 真正能榨出 compute-dense H200/Rubin-CPX 集群。PrfaaS 本身是个**无状态 KVCache 生产者**，吞吐 = $\min(\text{prefill 计算}, \text{出口带宽})$。

对 prefix cache 命中的请求（agentic 工作负载里很常见），全局 KVCache manager 追踪每条缓存的存放位置，**只把增量部分跨集群**。这是带宽计算在实践里能成立的关键 —— agentic 流量大多是 incremental prefill。

### 组件 2 —— 混合 prefix cache 池

Hybrid 模型打破了 KV cache 的传统假设：

| Attention 类型 | KV 状态形态 | 复用语义 |
| ------------- | ---------- | ------- |
| Linear / SWA | **Request 级** recurrent state，**固定大小** | 只支持精确匹配（长度必须完全相同） |
| Full attention (MLA, GQA) | **Block 级** KV，**随序列长增长** | 标准 prefix matching，支持部分复用 |

PrfaaS 基于 [vLLM hybrid KVCache manager (PR #29427)](https://github.com/vllm-project/vllm/pull/29427)，再为跨集群传输改造。设计：

- **分开的 KVCache group** 给 linear state 和 full-attention KV，但 **block size 对齐**，所有 group 共享一个 **block pool**
- Cache 块分两类：
  - **Prefix-cache 块** —— 必须填满才能跨请求复用、只限集群内、按 block 对齐
  - **Transfer-cache 块** —— 在 prefill 请求末端产生、用于 PD 分离传输、**任意长度**、**传输完丢弃**

新请求到来时，全局 KVCache manager 算出每个集群的 prefix-match 信息，路由器据此选 prefill 集群 + 集群内 cache-affine 节点。当集群间带宽允许时还做 cache 再平衡。

### 组件 3 —— 双时间尺度调度

这是系统贡献住的地方。光有选择性外放扛不住真实流量 —— 拥塞会堆、队列膨胀、P90 TTFT 爆炸。调度器在两个时间尺度上运行：

**短期：带宽 + cache 感知路由**。调度器持续监控 PrfaaS 出口利用率和队列深度。利用率接近带宽上限 $B_{\text{out}}/S_{kv}(l_{\text{long}})$ 时，提高路由阈值 $t$，少送（更长的）请求过 DC，从而减少 per-request 带宽需求。

对 prefix cache 命中请求，路由按带宽还是计算是 binding：
- **带宽稀缺态** —— 每集群独立评估 cache：`if l_total − l_pd ≤ t → PD-P；else → PrfaaS`
- **带宽充裕态** —— 跨集群看最好 cache：`l_prefix = max(l_prfaas, l_pd)；if l_total − l_prefix ≤ t → PD-P；else → PrfaaS`。如果最好 cache 集群跟计算集群不同，就先做一次跨集群 cache 转移。

**长期：流量驱动的分配重优化**。流量 mix 和量在小时 / 天尺度变化。当 $\Theta_{\text{prfaas}} + \Theta_{\text{pd-p}} \ll \Theta_{\text{pd-d}}$ 时是 prefill 瓶颈；反之是 decode 瓶颈。调度器周期性重评估负载平衡，**在 PD 集群里把节点在 prefill 和 decode 角色之间互转**（$N_p \leftrightarrow N_d$），然后重新算最优 $t$。PrfaaS 集群 GPU 保持只跑 prefill，因为它们 compute-dense，跑 decode 会浪费。

### 吞吐模型

案例研究的所有数都来自这个解析模型（论文 §3.4）。三阶段，最慢的决定 $\Lambda_{\max}$：

$$
\Theta_{\text{prfaas}} = \min\!\left(\frac{N_{\text{prfaas}}}{T_{\text{prefill}}(l_{\text{long}})}, \;\frac{B_{\text{out}}}{S_{kv}(l_{\text{long}})}\right)
$$

$$
\Theta_{\text{pd-p}} = \frac{N_p}{T_{\text{prefill}}(l_{\text{short}})}, \qquad \Theta_{\text{pd-d}} = \frac{N_d \cdot \text{BS}_{\max}}{T_{\text{decode}} \cdot L_{\text{out}}}
$$

$$
\Lambda_{\max} = \min\!\left(\frac{\Theta_{\text{prfaas}}}{p}, \;\frac{\Theta_{\text{pd-p}}}{1-p}, \;\Theta_{\text{pd-d}}\right)
$$

其中 $p = \Pr(L > t)$ 是路由到 PrfaaS 的比例。两个优化旋钮：$t$（决定 $p$、$l_{\text{long}}$、$l_{\text{short}}$）和 $N_p/N_d$（PD 集群 prefill/decode 分配）。最优条件：

$$
\frac{\Theta_{\text{prfaas}}}{p} = \frac{\Theta_{\text{pd-p}}}{1-p} \quad\text{（阈值平衡）}
$$

$$
\Theta_{\text{prfaas}} + \Theta_{\text{pd-p}} = \Theta_{\text{pd-d}} \quad\text{（生产者-消费者平衡）}
$$

两个未知数（$t$、$N_p/N_d$）两个方程 —— 在 profiling 出的 $T_{\text{prefill}}(l)$、$S_{kv}(l)$ 曲线上做 2D grid search 求解。

> [!example] 阈值最优的直觉
> 增大 $t$ 把 PrfaaS 限制在更长的请求上。对这些请求 $T_{\text{prefill}}(l)$ 几乎是二次增长（hybrid 模型里 full-attention 层仍占主导），$S_{kv}(l)$ 线性增长 —— 所以 $\Phi_{kv}$ 随 $l$ *下降*。$t$ 越大，每个被外放请求的带宽压力越小，$B_{\text{out}}$ 下的余量越大。降低 $t$ 会让 PrfaaS 涌入高 $\Phi_{kv}$ 的短请求，撞上带宽天花板。最优点是 PrfaaS 和 PD-P 同时饱和的位置。

### 维持吞吐的网络工程

即使带宽需求降了，bursty 流量还能在以太网上引发瞬时拥塞。论文的网络侧设计：

- **逐层 prefill 流水** —— KV 生成跟传输重叠，每层的 KV 一产出就开始 stream，不等 prefill 整个结束
- **多连接 TCP 传输** —— 多个并行连接充分利用以太网带宽，单 flow 拥塞可容忍
- **拥塞监控与调度集成** —— 早期检测丢包 / 重传信号，反馈给短期路由器，让它在队列堆积前节流 PrfaaS 路由

注意：**是 TCP，不是 RDMA**。这就是系统贡献 —— 让普通基础设施承载 KV 流量。

## 标志证据 —— 1T-hybrid 案例研究

**配置**。内部 1T 参数 Kimi-Linear 风格 hybrid（KDA:MLA 3:1），8 GPU/实例。

| 集群 | GPU | 角色 | 网络 |
| ---- | --- | --- | --- |
| PrfaaS | 32× H200 | 长上下文 prefill（$l > t$） | 跨 DC：100 Gbps 以太网 |
| 本地 PD | 64× H20 | 短 prefill + 全部 decode | 集群内：800 Gbps RDMA |
| **Baseline** | **96× H20** | **同构 PD** | **只有集群内 RDMA** |

模型 profile（8×H200，in-house vLLM，论文 Table 5）：

| 序列长 | KV 大小 | Prefill 延迟 | $\Phi_{kv}$ |
| ----- | ------: | ----------: | ----------: |
| 1K | 190.8 MiB | 0.44 s | 3.61 Gbps |
| 8K | 308.9 MiB | 0.72 s | 3.59 Gbps |
| 32K | 701.3 MiB | 1.84 s | 3.19 Gbps |
| 128K | 2316.3 MiB | 7.40 s | 2.62 Gbps |

工作负载：输入长度 log-normal 分布（$\mu=9.90$、$\sigma=1.00$，截断 $[128, 128K]$，均值约 27K tokens），输出固定 1024 token，SLO 40 tok/s decode。

**优化结果**。Grid search 给出：

| 参数 | 值 |
| ---- | -: |
| 路由阈值 $t$ | **19.4K tokens** |
| PrfaaS 实例 $N_{\text{prfaas}}$ | 4 |
| PD-P / PD-D 实例数 | 3 / 5 |
| PrfaaS 路由比例 $p$ | **49.6 %** |
| $\mathbb{E}[L \mid L > t]$ | ~44K tokens |

**三路对比**（论文 Table 6）：

| 指标 | 同构 PD | Naive 异构 | **PrfaaS-PD** |
| ---- | -----: | --------: | ------------: |
| 阈值 $t$ | — | — | **19.4K** |
| 布局 ($N_{\text{prfaas}}/N_p/N_d$) | —/9/3 | 4/—/8 | **4/3/5** |
| Mean / P90 TTFT (s) | 4.44 / 9.73 | 1.74 / 3.51 | **2.22 / 3.51** |
| $\Theta_{\text{prfaas}}/\Theta_{\text{pd-p}}/\Theta_{\text{pd-d}}$ (req/s) | —/2.11/2.35 | 2.45/—/6.25 | **1.61/1.64/3.91** |
| $\Lambda_{\max}$ (req/s) | 2.11 | 2.45 | **3.24** |
| 加速比 | 1.00× | 1.16× | **1.54×** |

> [!success] 13 Gbps 这个标题
> 最优运行点上跨 DC 出口带宽 **13 Gbps —— 只占 100 Gbps 以太网的 13 %**。Burst 余量充足。对比一下，同规模的 dense MiniMax-M2.5 部署稳态就要 ~60 Gbps，burst 起来直接打穿任何普通链路。

> [!important] Naive 异构是关键对照
> vs 同构 1.54× 告诉你"用 H200+H20 配 PrfaaS 调度"。1.54× / 1.16× = **比 naive 异构再高 1.33×**，这告诉你 *调度本身贡献约 25 % 吞吐增益*。架构离了选择性外放 + 长度阈值路由就 load-bearing 不起来。

### P90 TTFT 为什么降 64 %

两个效应叠加：

1. **长请求隔离**。同构 baseline 里长短请求共享同一个 prefill 池，长请求把*所有人*的排队时间拉高 —— prefill 端的 head-of-line blocking。PrfaaS-PD 把长请求转到单独集群，短请求 TTFT 保持低。
2. **更快的 prefill 计算**。H200 在长上下文 prefill 上明显比 H20 快；即使吃掉跨集群传输延迟，模型化的长请求 prefill 在 PrfaaS 上比在 H20 baseline 上更快完成。

Mean TTFT 降 50 %（2.22 vs 4.44 s）；P90 降 64 %（3.51 vs 9.73 s）。P90 涨幅更大是因为同构 baseline 的尾部正是被长请求排队病理主导 —— 这正是 PrfaaS 消灭的东西。

## 优势与限制

两个真实优势：(1) **首篇把跨 DC PD 分离工程化落地的系统论文** —— 不是"我们应该做这个"的散文，而是有完整吞吐模型、调度器、cache 架构和案例数字的端到端设计；(2) **模型-系统协同设计的 framing 诚实** —— 论文反复强调"KVCache 变小不足够"，并通过 naive 异构对照实打实证明了这一点。太多"模型 X 让系统 Y 成为可能"的论文在后半部分塌方，这篇没有。

可推敲的地方：

- **是案例研究，不是部署**。所有数字来自 profiling 数据（$T_{\text{prefill}}(l)$、$S_{kv}(l)$、$T_{\text{decode}}$）喂进稳态解析模型。没有 1T 模型的生产实际跑、没有 burst pattern 下实测 TTFT、没有跨 DC 链路抖动实验。13 Gbps / 13 % 余量这个说法是在 log-normal 工作负载假设下*模型化*的。
- **没有源码**。论文承认是"in-house vLLM"；hybrid KV cache 池基于 vLLM PR #29427，但跨集群胶水代码没开源。layer-wise pipelining + 多连接 TCP 这些 claim 不跑代码很难验证。
- **输出长度定死 1024**。长输出推理工作负载（DeepSeek-R1、o3/o4 风格）经常生成 8K-32K token。Decode 阶段吞吐按 $N_d \cdot \text{BS}_{\max}/(T_{\text{decode}} \cdot L_{\text{out}})$ 缩放，所以更长输出会把生产者-消费者平衡推向需要*更多* PD-D，改变最优分配。案例研究没扫 $L_{\text{out}}$。
- **单一工作负载分布**。一条 log-normal $(\mu=9.90, \sigma=1.00)$。真实生产流量是多模态的（chat + 长上下文 + agentic + RAG），最优 $t$ 依赖联合分布。双时间尺度调度器据说能处理这个，但没实验展示它在分布漂移下的自适应。
- **网络故障模式不讨论**。跨 DC 以太网丢包*是常态*（带宽抖动、链路 flap）。一块 in-flight KVCache 丢了怎么办？有重传，还是请求直接失败？"拥塞监控与调度集成"暗示了降级路由但没定义 per-request fallback。
- **隐私 / 多租户没讨论**。KVCache 包含从请求 prompt 派生的信息。通过普通以太网跨 DC 传输 —— 即使加密 —— 有合规和信息流意涵（GDPR、residency 要求）。论文沉默。生产部署里这是监管会问的第一个问题。
- **完全帮不到 dense GQA 模型**。这点很明确但值得重申：PrfaaS 工作是因为 hybrid attention 把 $\Phi_{kv}$ 降了一个数量级。Dense GQA 模型（Llama-3、Qwen3-235B）每实例还要约 30 Gbps，64 实例就是 2 Tbps，完全超出设计范围。整个架构契约式地依赖 hybrid attention 这个行业趋势。
- **1T 模型是内部的、未公开**。架构跟 Kimi Linear 接近但不公开。Moonshot 外的复现性受限于"去试 Kimi Linear 开源版本"（小于 1T）。
- **没跟 [[af-disaggregation|AF 分离]] 对比**。PD 和 AF 都是 operator-vs-phase 切分策略，都处理跨网络状态传输。论文没把 PrfaaS 与 production 已经落地的 AF 分离工作（MegaScale-Infer / DP-attention + EP-MoE）做对照。一个刚入门的读者会想问 "PD-vs-AF 在跨 DC 场景下"。

> [!warning] 论文（虽然标题这么说）*没有* 声称的东西
> "Could Go Cross-Datacenter" 是诚实的 —— 论文展示了在现实假设下的可行性，不是有人已经在生产跑跨 DC PD。Moonshot 实际的 Kimi-2 serving 栈用的是 DC 内 Mooncake；PrfaaS-PD 是面向未来 Rubin-CPX 级硬件部署的前瞻设计。把这当成"生产现在在做的事"会过度解读。

## 这意味着什么

大的系统趋势：**KVCache 越来越是你围绕设计的资源**。Mooncake 让 KVCache 成为单集群里的一等系统对象。PrfaaS 把它推到集群边界之外。接下来明显的几步：

1. **跨 DC KV 连接器标准化**。2026 H2 期待 vLLM / SGLang / Dynamo 加入跨集群 KV 传输适配器 —— 可插拔的 TCP / RDMA-over-DCI / 专线变体。接口会像 Mooncake `Transfer Engine` 扩展到 WAN。会有一阵竞争协议（NIXL、Mooncake、KVCache-over-gRPC）。
2. **KV-aware 的 DC 选址决策**。云厂商开始宣传 "Rubin CPX DC" 和 "LPU DC"，带文档化的 DC 间带宽保证。Prefill 重的工作负载（长文档摘要、代码理解）会跟 decode 重的（agentic 链、推理）走不同路由。PrfaaS 给出了这些路由决策的样板。
3. **Hybrid attention 成为跨 DC 部署的必要条件**。Dense-GQA-only 模型家族跨 DC 部署不了。这对模型设计方是一个有意义的竞争劣势 —— 期待 Llama-5 / Qwen-4 等等都搭载某种 hybrid attention 变体。Linear attention + SWA 不只是"长上下文效率"，而是"部署架构使能"。
4. **Phase 特化 DC 成为自然单位**。今天 DC 是"装满 GPU"。明天可能是"这个 DC 是 prefill 优化 —— compute-dense H200/Rubin、网络中等、给每个客户的长上下文任务用"；"那个 DC 是 decode 优化 —— 带宽怪兽 GPU、更大内存池、给推理工作负载用"。PrfaaS 是让工作负载跨越两个 DC 的系统架构。

它*不是* —— 论文 §5 也诚实承认 —— DC 间 LLM serving 的完全通用方案。它瞄准的是 hybrid attention 模型的长上下文 prefill 瓶颈，那是 serving 大问题里的一个切片（重要的切片）。

## 源码与复现

v2 预印本时无公开发布。论文基于：

| 组件 | 引用 |
| ---- | ---- |
| Hybrid KVCache manager（单集群） | [vLLM PR #29427](https://github.com/vllm-project/vllm/pull/29427) —— per-cluster hybrid prefix pool 的基础 |
| Mooncake KVCache 池 | [Mooncake repo](https://github.com/kvcache-ai/Mooncake) —— 单集群前身；全局 KVCache manager 把 Mooncake 的元数据层泛化到跨集群 |
| SGLang 用于 $\Phi_{kv}$ 基准（Table 3） | [SGLang v0.5.9](https://github.com/sgl-project/sglang) |

双时间尺度调度器伪代码（从 §3.4.3 提取）：

```python
class PrfaaSScheduler:
    def __init__(self):
        self.t = initial_threshold       # 来自 grid-search 最优
        self.Np, self.Nd = init_alloc    # PD 集群 prefill/decode 分配

    def route(self, request):
        # 短期：带宽 + cache 感知
        l_total = request.input_length
        l_pd    = prefix_cache_hit(request, cluster="local-pd")
        l_prfaas = prefix_cache_hit(request, cluster="prfaas")

        if egress_util() > BANDWIDTH_CEILING:
            # 带宽稀缺：每集群独立评估 cache
            incremental = l_total - l_pd
            return "PD-P" if incremental <= self.t else "PrfaaS"
        else:
            # 带宽充裕：可以跨集群转移 cache
            l_best = max(l_pd, l_prfaas)
            incremental = l_total - l_best
            target = "PD-P" if incremental <= self.t else "PrfaaS"
            if l_best == l_prfaas and target == "PD-P":
                transfer_cache_to_pd(request)
            return target

    def periodic_reoptimize(self):
        # 长期：重平衡 N_p / N_d、重新 grid-search t
        profile = collect_recent_profile()
        if Θ_prfaas() + Θ_pd_p() < Θ_pd_d():    # prefill 瓶颈
            self.Np += 1; self.Nd -= 1
        elif Θ_prfaas() + Θ_pd_p() > Θ_pd_d():  # decode 瓶颈
            self.Np -= 1; self.Nd += 1
        self.t, _ = grid_search_optimum(profile, self.Np, self.Nd)
```

要复现案例数字，你需要：(1) 接近 Kimi Linear 或 Ring-2.5-1T 的 hybrid attention 模型，(2) H200 上 $l \in \{1K, 8K, 32K, 128K\}$ 的 profiling 数据 $T_{\text{prefill}}(l)$ 和 $S_{kv}(l)$，(3) §3.4 解析模型 + 工作负载假设。论文给的数字足够让你从 Table 5 手算出 Table 6。

## 相关阅读

- [[prefill-decode-disaggregation]] —— 单集群 PD 基础；PrfaaS 把这个推过 DC 边界。Mooncake 是直接前身，同一位清华主导（Mingxing Zhang）。
- [[af-disaggregation]] —— 同族分离策略：PD 沿时间轴切*阶段*（prefill vs decode）；AF 在每次 forward 内切*算子*（attention vs FFN）。PrfaaS 把 PD 推过 DC；AF 在单集群内把工作铺到异构芯片上，每层都是。原则上可以组合。
- [[paged-attention]] —— Paged KV cache 管理；vLLM 的 hybrid KVCache manager（作为 PrfaaS 存储层）继承自 PagedAttention。
- [[kv-cache-optimization]] —— H2O / KIVI / KVQuant 这类从架构之外压缩 KV 的家族；PrfaaS §5 当互补技术引用。
- [[vllm]] —— PrfaaS 扩展的 serving 框架。
- [[sglang]] —— Table 3 测 $\Phi_{kv}$ 用的。
- [[long-context-serving]] —— PrfaaS 增益最大的工作负载类别。
- [[parallelism-strategies-deep-dive#7. CP — 上下文并行 (Context Parallelism)]] —— Context parallelism 是长上下文 prefill 的*集群内*答案；PrfaaS 是*集群间*答案。两者可以组合（PrfaaS 集群内用 CP，DC 间用 PrfaaS）。

## 参考文献

- 论文：Qin et al., *Prefill-as-a-Service: KVCache of Next-Generation Models Could Go Cross-Datacenter*, 2026-04-16. [arXiv:2604.15039](https://arxiv.org/abs/2604.15039)
- **Mooncake**（前身）：Qin et al., FAST 2025. [arXiv:2407.00079](https://arxiv.org/abs/2407.00079)
- **Splitwise**: Patel et al., MICRO 2024. [arXiv:2311.18677](https://arxiv.org/abs/2311.18677)
- **DistServe**: Zhong et al., OSDI 2024. [arXiv:2401.09670](https://arxiv.org/abs/2401.09670)
- **NVIDIA Rubin CPX**（prefill 专用硬件目标）：[nvidia.com/en-us/data-center/products/rubin-cpx](https://www.nvidia.com/en-us/data-center/products/rubin-cpx/)
- **Groq LPU**（decode 专用对手）：[wow.groq.com/lpu-inference-engine](https://wow.groq.com/lpu-inference-engine/)
- **Kimi Linear**（1T 模型遵循的架构家族）：[arXiv:2511.06257](https://arxiv.org/abs/2511.06257)
- **vLLM hybrid KVCache manager**: [github.com/vllm-project/vllm/pull/29427](https://github.com/vllm-project/vllm/pull/29427)
- 互补 KV 压缩：**H2O** ([arXiv:2306.14048](https://arxiv.org/abs/2306.14048))、**KIVI** ([arXiv:2402.02750](https://arxiv.org/abs/2402.02750))、**KVQuant** ([arXiv:2401.18079](https://arxiv.org/abs/2401.18079))
- 互补 KV 复用：**CacheGen** ([arXiv:2310.07240](https://arxiv.org/abs/2310.07240))、**CacheBlend** ([arXiv:2405.16444](https://arxiv.org/abs/2405.16444))
