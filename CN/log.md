---
title: 变更日志
updated: 2026-05-13
---

# 变更日志

## 2026-05-13
- [新增] [[nemo-gym]] —— 给 NVIDIA NeMo Gym 写了一篇框架深度解析（RL 环境 / rollout 一侧的库；对应训练侧的 NeMo RL / VeRL）。页面覆盖：三 server 的 FastAPI 架构（resources / model / agent）、"环境 = 数据集 + harness + verifier + state"心智模型、自带的 84 个 benchmark（SWE-bench、GPQA、BigCodeBench、math、IFBench、GDPVal、Newton Bench 等）、19 种 agent harness（simple、OpenHands 风格 SWE、Mini-SWE、LangGraph、Verifiers、Aviary、Harbor …）、Hydra/OmegaConf 配置树、JSONL 数据 schema + GitLab MLflow 数据集仓。**容器 / sandbox 一段是页面核心**（这是 Shiki 一开始问的问题，前面三次没说清）：两条互不相干的路径 —— Apptainer 路径走生产 HPC（通过 `docker://...` URI 消费 Docker 镜像，但用 Apptainer 跑，因为集群节点本身就在 enroot 容器里；引自 `docs/infrastructure/engineering-notes/swe-rl-case-study.md`），Python 进程级 sandbox 路径走简单任务（`newton_bench`：受限 builtins + AST 检查 + SIGALRM）。另外把非显然的工程选择都记下来了（高并发用 aiohttp 而不是 httpx；Lustre 上的 `RAY_TMPDIR` 坑；锁定 `openai<=2.6.1`；全异步；配置走 config 而不是环境变量）。两张 Mermaid 图（系统架构 + 容器路径分支）遵守 WIKI-Format-Skill 的单方向 TB 规则。已交叉链接至 [[prorl-agent]]、[[environment-design]]、[[rl-training-frameworks]]、[[grpo]]、[[ppo-for-llm]]、[[rlhf-overview]]、[[das-spec-rl]]、[[vllm]]、[[sglang]]、[[multi-step-reasoning-rl]]。EN/CN 双语对齐。
- [摄入] arXiv:2511.13841 "Beat the long tail: Distribution-Aware Speculative Decoding for RL Training"（Shao, Srivatsa, Srivastava 等，2025-11-17，MLSys 投稿）—— 论文精读已添加于 `wiki/llm-inference/` 下的 [[das-spec-rl]]。引用元数据位于 `sources/papers/das-spec-rl/`。报告涵盖三块：(1) **基于 Ukkonen 在线 suffix tree 的 per-problem drafter**，配 prefix trie 路由与滑动窗口刷新 —— drafter 不占 GPU 显存且自动跟随策略；(2) **长度感知投机策略**，从显式 makespan 模型 `t_total = c_base · N_fwd + c_tok · N_toks + C` 推导出闭式最优预算 `p_i* = -(l_i/α_i) · ln(1 - k_i(1 - N_fwd/l_i))`（Eq. 7）与每位置 acceptance 指数衰减 `a_{i,k} = a_{i,0} · e^(-β_i(k-1))`，打包成 Long/Med/Short 三桶启发式（Short 桶完全关掉投机）；(3) **实验结果** —— 数学 RL（DeepSeek-R1-Distill-Qwen-7B on DSR-sub, 1× 8×H100, batch 128, 30 步）rollout 时间下降 >50 %，代码 RL（Qwen3-8B on DeepCoder, 2× 8×H100）下降 ~25 %；消融显示分布感知预算比无限预算最多快 15 %，8K 序列长度下仍能 >30 % 加速。Mermaid 图展示 DAS 装配后的 rollout 循环（单列 TB，按 ProRL 课上学到的规则把 Signals 放成侧边子图）。已交叉链接至 [[speculative-decoding]]、[[prorl-agent]]、[[grpo]]、[[continuous-batching]]、[[kv-cache-optimization]]、[[vllm]]、[[sglang]]、[[long-context-serving]]。EN/CN 双语对齐。

## 2026-05-08
- [Q&A] [[prorl-agent]] —— Shiki 提了 4 个关于论文的概念性问题：(1) scaffold 是什么、(2) "稳定的 HTTP 契约"指什么、(3) token-in/token-out 是什么以及去掉之后为什么 off-policy 不稳、(4) "rootless 沙箱"是什么。Q1+Q2+Q4 合并为一个 `[!question]+` callout，置于 Background 章节比较表之后；Q3 单独 callout 置于 Token-in/token-out 子节。EN/CN 双语对齐。
- [扩展] [[prorl-agent]] —— 把原本 3 框 ASCII 架构图替换为 Mermaid `flowchart TB`，展示 FastAPI 父进程 / multiprocessing 子进程拆分、三阶段队列流水线与各自的 worker 池、AgentHandler 分派、per-job 与共享状态，以及两个外部资源（Singularity 沙箱 + vLLM 后端池）。章节标题从 *三组件架构* 改为 *系统架构*。EN/CN 双语对齐。
- [修订] [[prorl-agent]] 架构图（同日）—— 第一版 Mermaid 每个节点塞 5–6 行 `<br/>`，Obsidian 渲染成几屏高的巨图（Shiki 反馈）。压缩到每节点 2–3 行，把 API 调用序列（① ② ③ ④）从 Trainer 节点挪到图前的散文里，加上 `%%{init: ...}%%` 指令把 `fontSize` 设为 13px 并收紧 `nodeSpacing` / `rankSpacing`。
- [修订 #2] [[prorl-agent]] 架构图（同日）—— Shiki 澄清：实际问题不是图太高，而是**横向溢出** —— v2 在 `flowchart TB` 父图里嵌了一个 `direction LR` 的 `subgraph Pipeline`，Mermaid 把它甩到一侧，需要左右滚动才能看到 EVAL。修复：把 LR pipeline 展开成 `Server` subgraph 内的三个顺序 TB 节点，完全去掉那层嵌套 LR subgraph，恢复更丰富的 3 行节点内容（布局对了就没问题）。整图全竖向单列流，Sandbox / vLLM 作为 Worker 底端的两条侧支。
- [扩展] WIKI-Format-Skill（skills/wiki-format-skill/SKILL.md）—— 在 *Visual hierarchy* 下新增 *Diagrams* 子节，并随着 ProRL Agent 图的两轮失败迭代两次细化。最终规则：**（布局）整图统一一个 direction、不要嵌套混合方向的 subgraph**（TB 里嵌 LR 会被甩到一侧）；**（内容）每节点 3–4 行可以**，`<b>名字</b>` + 角色 + 关键开关，调用序列挪到图前散文；init 指令设 `fontSize: 12px` 加紧间距；仍太大就拆成两张图。参考模板现在示范"单列 TB + 侧支从锚点节点连出在 subgraph 外"。反模式：三框加箭头、5+ 行节点、嵌套 subgraph 混合方向、裸 flowchart、仅标识符标签、无 init 指令。

## 2026-05-06
- [摄入] arXiv:2603.18815 "ProRL Agent"（NVIDIA, 2026 年 3 月）— 论文精读已添加于 `wiki/agentic-rl/` 下的 [[prorl-agent]]。引用元数据位于 `sources/papers/prorl-agent/`。报告涵盖论文分析以及源码阅读：`start_server.py`（FastAPI + multiprocessing）、`openhands/nvidia/registry.py`（AgentHandler ABC + JobDetails dataclass + 注册表模式）、`openhands/nvidia/async_server.py`（三队列流水线、min-heap 负载均衡器、统一 _worker）、`openhands/llm/nvidia/`（token-in/out 动机）。已交叉链接至 [[agentic-rl-overview]]、[[environment-design]]、[[rl-training-frameworks]]、[[grpo]]、[[kv-cache-optimization]]、[[multi-turn-optimization]]。
- [摄入] arXiv:2604.19157 "SAW-INT4"（Together AI 等, 2026 年 5 月）— 论文精读已添加于 `wiki/llm-inference/` 下的 [[saw-int4]]。引用元数据位于 `sources/papers/saw-int4/`。报告涵盖论文分析以及 SGLang fork 源码阅读：`memory_pool.py` 的 env-var 门控与 `set_kv_buffer` 的 BDR 分支、融合 Triton 内核 `quantized_set_kv_int4_hadamard_fused_triton`、`tools/fit_kv_centroids.py` k-means 校准。块对角 Hadamard 旋转使 Qwen3-4B-Thinking 的 GPQA 从原始 INT4 的 0% 恢复到 65.82%。已交叉链接至 [[kv-cache-optimization]]、[[quantization]]、[[sglang]]、[[long-context-serving]]、[[paged-attention]]、[[vllm]]。
- [新增] [[rotation-based-quantization]] —— 综合页，涵盖 QuIP / QuIP# / QuaRot / SpinQuant / SAW-INT4-BDR 家族，含数学基础、对比表、"旋转吸收位置"分析、实践指南与开放问题。已交叉链接至 [[saw-int4]]、[[quantization]]、[[kv-cache-optimization]]。
- [扩展] [[quantization]] —— 增加"基于旋转的量化"章节综合 QuIP/QuaRot/SpinQuant/BDR。与已有 QuIP# 内容打通。更新 KV 缓存表区分原始 INT4 vs INT4+BDR。在参考文献与相关页面里加入 QuaRot/SpinQuant/SAW-INT4。
- [扩展] [[kv-cache-optimization]] —— 增加"基于旋转的 KV 缓存量化"子章节并引用 [[saw-int4]]。更新量化表加入 INT4（原始）和 INT4 + BDR 两行展示 GPQA 崩溃与恢复。改进 KIVI 描述（非对称混合粒度，per-channel K + per-token V + FP16 锚点）。更新决策树与参考文献。

## 2026-05-07
- [Q&A] [[saw-int4]] —— Shiki 询问"离群通道"段落的含义。Q&A 以 Obsidian `[!question]+` callout 格式**就地嵌入**于 Background 章节里 GPQA 崩溃表格之后（不再放页面末尾的统一 Q&A 节）。紧凑段落格式（4 段）解释离群通道三类来源（RoPE / massive activations / 专用头）、per-token scale-zero 失效模式（~95% 通道塌缩到 0）、为什么表面任务能撑住但多步推理累积误差崩溃、以及 BDR 如何修复。约定记录到记忆里：论文 Q&A 就地放在被讨论的位置、callout 格式、紧凑段落。

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
