---
title: "Polar：任意 harness 上的可扩展智能体 RL（ProRL Agent 的续作）"
category: agentic-rl
tags: [polar, prorl-agent, nvidia, rollout即服务, agentic-rl, llm-api-proxy, nemo-gym, swe-bench, 论文精读]
created: 2026-05-26
updated: 2026-05-27
status: mature
paper: arXiv:2605.24220
code: https://github.com/NVIDIA-NeMo/ProRL-Agent-Server
---

# Polar：任意 harness 上的可扩展智能体 RL（ProRL Agent 的续作）

> [!info] 论文信息
> - **论文**：[arXiv:2605.24220](https://arxiv.org/abs/2605.24220) —— *Polar: Agentic RL on Any Harness at Scale*，2026-05-22
> - **代码**：[NVIDIA-NeMo/ProRL-Agent-Server](https://github.com/NVIDIA-NeMo/ProRL-Agent-Server) —— **跟 [[prorl-agent|ProRL Agent]] 是同一个 repo；Polar 是原地重写**
> - **作者**：Binfeng Xu, Hao Zhang, Shaokun Zhang, Songyang Han, Mingjie Liu, Jian Hu, Shizhe Diao, Zhenghui Jin, Yunheng Zou, Michael Demoret, Jan Kautz, Yi Dong
> - **名字来源**：从 "Pr**O**rL Agent serv**R**" 里取字母 → **Polar**，又同时呼应"agent 训练"与"产品部署"两个极点
> - **状态**：已注册成 [[nemo-gym|NeMo Gym]] 的一个环境 —— 这就是 2026-05 之前缺的那座 ProRL Agent ↔ NeMo Gym 桥

> [!important] 取代 [[prorl-agent|ProRL Agent]]
> 论文原话："Polar rewrites its preceding work, ProRL Agent, and has been registered as one of NeMo Gym environments." 同一个 NVIDIA 团队（作者重合度 ~75%），同一个 GitHub repo。[[prorl-agent|ProRL Agent]] 页面记录的是前代架构；这一页是 NVIDIA 智能体 RL rollout 基础设施的**当前状态**。

---

## 摘要（2 分钟读完这一节就够）

**Polar 是什么**。NVIDIA 第二代智能体 RL rollout 框架，接替 [[prorl-agent|ProRL Agent]]。ProRL Agent 要求你给每个 agent harness 写一个 Python `AgentHandler` ABC 适配器；Polar 让 **任何未修改的 agent harness —— Codex、Claude Code、Qwen Code、Pi、Gemini CLI、OpenCode —— 当作黑盒运行**，通过 proxy 拦截它的 LLM API 调用。捕获的 tokens + log-probs 被重建成 token-faithful 轨迹给 trainer。

**核心思想**。**把集成边界从 agent 的 Python API 上推到 LLM provider API 上**。每个基于 LLM 的 agent 都必须跟模型说话 —— 那是通用接口。在 harness 和 inference server 之间坐着、记录一切、重建轨迹。三件事撑起来：

1. **provider 兼容 proxy** —— 接 Anthropic Messages、OpenAI Chat Completions、OpenAI Responses、Google `generateContent` 各种形状；翻译到本地 inference；记 prompt token IDs、采样 token、log-probs
2. **Token-faithful 前缀合并** —— 多轮对话被重建成轨迹，只有 behavior-policy 采样的 token 才参与训练（loss mask = 1）；canonical interstitial tokens（harness 对历史轮次的渲染 + 注入的上下文）被屏蔽（loss mask = 0）。子 agent、上下文压缩、prompt 改写天然形成独立的 chain
3. **Gateway 级异步分级** —— rollout server + gateway 节点；每个 gateway 有 INIT / RUNNING / POSTRUN worker 池 + READY buffer，让 CPU-bound runtime 准备和长尾 evaluation 不堵 GPU-bound agent 执行

去掉 proxy 就退回到 ProRL Agent 的 plugin-per-harness 模式；去掉 prefix-merging trainer 被 1000+ 个碎片 trace 淹掉；去掉异步分级 rollout 跟 training 串行化。

**标志数字**。Qwen3.5-4B base + simple GRPO 在 SkyRL-v0-293-data 上训练，SWE-Bench Verified pass@1 评估：

| Harness | Base | Polar RL | 增益 |
| ------- | ---: | -------: | ---: |
| **Codex** | 3.8 % | **26.4 %** | **+22.6** |
| Claude Code | 29.8 % | 34.6 % | +4.8 |
| Qwen Code | 34.6 % | 35.2 % | +0.6 |
| Pi | 34.2 % | 40.4 % | +6.2 |

同一个 base、四个不同 harness、全都涨。**Codex 的 +22.6 pp 是 showcase 数字**：Qwen3.5-4B 在 Codex 不熟悉的 action protocol 下几乎不可用，harness-native RL 把协议教给了它。最小增益（Qwen Code +0.6）正好是 base 本来就对齐的那个 harness —— 形状对。

**关键消融**。轨迹重建策略极其重要：`per_request`（每次 model call = 一条 trace）vs `prefix_merging`（按 append-only 链合并），同样的 workload、同样的 3 个训练步：

| 策略 | Trainer 更新次数 | Wall-clock | Rollout GPU 利用率 |
| ---- | ---------------: | ---------: | -----------------: |
| `per_request` | 1185 | 189.5 min | 20.4 % |
| **`prefix_merging`** | **218** | **35.2 min** | **87.7 %** |

**Wall-clock 加速 5.39×，GPU 利用率 4.3×**。`per_request` + outcome reward 广播*还*会触发严重的 reward hacking —— request 级 trace 拿到 session 级信用，没做归一化。重建算法不是装饰，是 load-bearing 的。

**为什么重要**。

- **可训练的 harness 范围扩大 10×**。任何会跟 LLM API 通话的东西都能训练了 —— 包括闭源二进制（Codex）、TypeScript CLI（Claude Code）、Go agent（Pi）。再也不用等"等谁有空写 AgentHandler 时再说"
- **ProRL Agent vs NeMo Gym 合并完成了**。昨天 wiki 里说"两者间还没有公开 adapter"已经过时 —— Polar 就是 adapter，注册为 NeMo Gym 环境。详见 [[prorl-agent#ProRL Agent vs NeMo Gym —— 同族、不同层]]（本节现在被本页超越）
- **公开了 agentic SFT 语料**。[`nvidia/polar-swegym-pi-qwen35-122b-a10b-trajectories`](https://huggingface.co/datasets/nvidia/polar-swegym-pi-qwen35-122b-a10b-trajectories)，HF，Apache-2.0 —— 504 条通过 SWE-Bench 的轨迹，Qwen3.5-122B-A10B + pi-coding-agent 生成，平均每条 104 messages / 51 assistant turns。可复现 offline SFT
- **2027 预测**。"agent-as-blackbox + API proxy" 模式成为默认。每个主要 RL 框架（veRL、NeMo RL、OpenRLHF）会在 12 个月内发布自家 proxy gateway；预期 Anthropic/OpenAI 推出"RL 友好"的公开 API 扩展，直接暴露 logprobs

---

# 深度部分（往下展开细节）

## 背景：为什么 ProRL Agent 的 plugin 模式撞墙了

[[prorl-agent|ProRL Agent]]（NVIDIA，2026-03）让 rollout 成为 HTTP 服务 —— 这是它的核心贡献。但它的集成契约是：**写一个 Python `AgentHandler` 子类，在 rollout 服务进程里驱动 agent loop**。具体长这样：

```python
class AgentHandler(ABC):
    @abstractmethod
    async def initialize(self, task): ...
    @abstractmethod
    async def run(self, model_client) -> Trajectory: ...
    @abstractmethod
    async def evaluate(self, trajectory) -> float: ...
```

每出一个新 harness —— OpenHands、Mini-SWE、LangGraph 风格 agent、Aviary —— 都需要有人：
1. 读 harness 的 Python 源码
2. 把它的 event loop、tool 定义、上下文管理、reward 评估翻译到 `AgentHandler` plugin 里
3. 上游 harness 演进时维护这个 plugin

对 NVIDIA 关心的 ~5 个 harness 这能跑，但对更广 ecosystem 失败有三种很具体的方式（论文里点了）：

| 失败 | 具体例子 |
| ---- | ------- |
| **闭源 / 二进制 harness** | Codex CLI 是二进制；在 `AgentHandler` 里重实现它根本不可能 |
| **非 Python harness** | Claude Code 是 TypeScript，Pi-coding-agent 是 Go；要翻译整个 event loop |
| **快速演化的 harness** | Claude Code 的 prompt 结构每月都变；Python 重实现追同步是个跑步机 |

同类系统（SkyRL-Agent、PRIME-RL、rLLM、Agent Lightning）都有同问题的变体：即使是"降低"集成成本的系统（Agent Lightning 的 tracing、rLLM 的 decorator）也仍然要求 harness *配合* —— 调一个被装饰的方法、emit 一个 span、遵守 SDK。**Polar 的核心问题**（论文原话）：

> *Can we train agents with RL without opening the box?*（不打开盒子能不能用 RL 训练 agent？）

转变在概念上很小、在架构上很大：**不集成 agent**（cooperative API），**听 LLM API 流量**（universal API）。每个基于 LLM 的 agent 都必须跟 model 说话。这是最低公分母。

| 前作系统（要 cooperate） | Polar（黑盒） |
| ----------------------- | ------------ |
| SkyRL-Agent、PRIME-RL —— harness 适配 RL 基础设施 | Harness 不动 |
| Agent Lightning —— tracing SDK 钩进 harness 代码 | 在 harness 和 LLM API 之间放 proxy；零 harness 代码改 |
| rLLM —— 装饰函数、跟踪 client | provider-API 协议检测（Anthropic / OpenAI / Google） |
| ProRL Agent —— `AgentHandler` ABC plugin per harness | 极小的 *adapter*：写配置 + 返回 shell 命令 |

> [!question]+ Shiki —— ProRL Agent 是不是就是加 HTTP 层去获取 rollout engine 的数据？(2026-05-27)
>
> 不是 —— 常见误解。**ProRL Agent 自己就 *是* rollout engine，不是包裹外部 engine 的 wrapper**。它进程里包含 vLLM（LLM 推理）、AgentHandler Python plugin（agent 循环）、rootless Apptainer 沙箱，全在一个进程里。HTTP 层是*朝向 trainer* 的契约 —— 让 trainer（veRL / NeMo-RL / slime）跟 rollout 解耦。Trainer 发 `POST /process`，ProRL Agent **主动控制**整条 rollout：起 sandbox、跑 AgentHandler 循环（每轮决定 search vs answer）、驱动 vLLM、调工具、评估、返回 trajectory。
>
> Polar 也是一样 —— 它*就是* rollout engine，只是集成边界不同（LLM-API proxy 代替进程内 AgentHandler plugin）。
>
> "Rollout engine" ≠ "LLM engine"。Rollout engine 跑 *完整 trajectory*（LLM forward + agent loop + tool calling + sandbox + 评分）。vLLM 只是里面 LLM forward 这一块。

## Proxy-as-boundary 架构

![Polar 的 proxy 边界（论文 Fig. 2）](CN/wiki/agentic-rl/polar-figs/polar-proxy-boundary.png)

左边：经典"harness as components"模型（Gymnasium / `env.init/step/reset` 风格）要求你逆向 harness 的内部组件 —— sys prompt、tool-call 格式、多 agent 协调、context engineering 技巧、cron jobs —— 然后在 `env.step()` 里重新拼起来。右边：Polar 把 harness 视为黑盒，它向 `v1/chat/completions` / `v1/responses` / `v1/messages` / `googleapis` 发请求，**API proxy** 在那里拦截，轨迹在 harness *外面*重建。

### Polar 的两层架构

| 层 | 角色 | 住在这里的东西 |
| -- | --- | ------------- |
| **Rollout server** | 任务调度 | 接 `TaskRequest`，展开成 `num_samples` 个 session，分发到 gateway，持久化结果，暴露 status 轮询，接 gateway 回调 |
| **Gateway node** | Session 生命周期 | 起 runtime、准备 harness、跑 harness 命令、host model proxy、构建轨迹、评估、撤资源 |

切分点是"持久任务状态" vs "per-session 执行"。Trainer（Slime、NeMo RL、veRL）打 rollout server 的异步端点；rollout server fan-out 到 gateway。

### Proxy 四步协议

Harness 每发一次 model 请求，gateway proxy 都做：

1. **检测 provider API**。路径 + header 区分：
   - Anthropic Messages（`/v1/messages`）
   - OpenAI Chat Completions（`/v1/chat/completions`）
   - OpenAI Responses（`/v1/responses`）
   - Google `generateContent`
2. **归一化请求**。provider transformer 转成 OpenAI Chat Completions schema（本地 inference server 的原生形状）。加上 `logprobs=true` 拿训练信号
3. **抓 token 级数据**。转发给 inference server（vLLM / SGLang）；存：prompt token IDs、response token IDs、finish reason、log 概率、请求/响应 messages
4. **以 provider 形状返回**。把响应转回 harness 期望的 schema。**streaming**：proxy 拿非流式上游响应，再合成 provider 形状的事件流 —— 简化 token 抓取、不破坏吃 SSE 的 harness

proxy 不需要懂 agent 的 planning、tool 选择、终止逻辑。它只要保 API 兼容、记够重建训练样本的信息就行。

### Harness adapter（很小的部分）

```python
class HarnessAdapter:
    def prepare_runtime(self, runtime, session): ...   # 装配置、注册 MCP servers
    def write_provider_settings(self, runtime): ...    # 把 model base URL 指到 gateway proxy
    def run_command(self, session) -> List[str]: ...   # 启 agent 的 shell 命令
```

就这。adapter 是配置 + shell 命令，不是 agent 重实现。论文自带 `claude_code`、`codex`、`gemini_cli`、`qwen_code`、`opencode`、`pi` 的快捷 adapter，外加一个通用 shell-command harness。

### Runtime 接口 —— Docker + rootless Apptainer

跟 ProRL Agent 同样的隔离选择：**rootless Apptainer** 用于 HPC / Slurm 集群（没有 Docker daemon）。初版也支持 Docker。接口（`start, stop, exec, upload, download, cancel`）让换隔离后端没摩擦。

> [!question]+ Shiki —— Polar 看得见 tool call 吗？工具在哪？(2026-05-27)
>
> **Polar 看不见 tool 调用直接**。工具发生在**未修改的 harness 内部** —— Codex CLI 的 bash 调用、Claude Code 的文件编辑、Pi 的 repo 读取。Polar 只看 **LLM API 边界**上流过的东西。
>
> 具体什么意思：
>
> 1. Harness（比如 Codex）决定"我应该读这个文件" —— **内部**调 `bash` 工具 → Polar 看不见
> 2. Harness 把文件内容作为下一次 API 调用的一部分发给 LLM → Polar 的 proxy **拦截**，捕 token IDs，转发到 vLLM
> 3. LLM（vLLM）响应"我应该编辑这行" → proxy 捕 token IDs，返回 text 给 harness
> 4. Harness 执行编辑（更多 tool 调用） → Polar 看不见
> 5. 下一次 LLM API 调用带新状态 → Polar 再次 capture
>
> Tool 的输入和输出**在 LLM API 调用里出现**，作为 prompt 的一部分（比如 `{"role": "tool", "content": "file contents..."}`）。Polar 的 `prefix_merging` 重建 trajectory 时：
>
> - Policy 实际采样的 token → `loss_mask = 1`（可训）
> - Harness/系统注入的 token（tool 结果、prompt 渲染等） → `loss_mask = 0`（interstitial）
>
> 所以 tool 输出**确实被 mask 掉了** —— 但 Polar 是通过 *diff* 相邻 API 调用反推 mask 的（前一个 response vs 这次 prompt 多出的是什么），而不是被显式告知"这是 tool 调用"。
>
> 这跟 [[search-r1#Retrieved-token loss masking — the load-bearing trick|Search-R1 的 retrieved-token loss masking]] 结构上类似，但 Polar 在 LLM-API 层做，Search-R1 在进程内 Python plugin 做。两者达到同一目标（梯度只流过 policy 采样的 token），但机制完全不同。

## Token-faithful 轨迹重建

这是论文的技术贡献。Polar 在 registry 里提供两个策略；prefix-merging 是 load-bearing 的那个。

> [!question]+ Shiki —— "Token-faithful" 跟 ProRL Agent 说的 token drift 是同一回事吗？(2026-05-27)
>
> **同样的问题，完全不同的实现**。ProRL Agent 和 Polar 都关心保留 policy 实际采样的 token IDs，永远不 retokenize text。
>
> **ProRL Agent 的方案：避开问题**。AgentHandler 跟 vLLM 在同一 Python 进程。直接调 vLLM，token IDs 进、token IDs 出，**根本不经过 text 这一步**。没有协议层 = 没有 retokenize 机会。
>
> ```python
> # ProRL Agent 的 AgentHandler（同进程）
> output_token_ids = await vllm_engine.generate(input_token_ids)
> # token IDs 是原生交换格式
> ```
>
> **Polar 的方案：解决问题**。Harness 是*独立进程*（Codex 是二进制、Claude Code 是 TypeScript），通过 text-based LLM API 通话。Token IDs 在这些 API 里不暴露。Polar 的 proxy 坐在中间：
>
> 1. Harness 发请求（text） → proxy 拦截
> 2. Proxy 转发给本地 vLLM **加 `logprobs=true`** → 拿到 text + token IDs + logprobs
> 3. Proxy 把 token IDs 存进 session log
> 4. Proxy 把 text 返回给 harness（保持 API 兼容）
> 5. Harness 完全不知道 token IDs 被捕获了
>
> Harness 看到的是**标准 LLM API text 响应**，Polar 在背后悄悄累积 token-faithful 轨迹数据。比 ProRL Agent 的方案难，但让 harness 可以是**任何会调 LLM API 的东西** —— 闭源二进制、TypeScript CLI、Go agent 都行。这种泛化是 Polar 的全部意义。

### Token-fidelity 问题

Provider API 返回 *text 或 tool-call JSON 或 reasoning 字段或流式事件*，不是 inference 后端实际采样的 token ID。把响应 text 解码再重新 tokenize 可能产生**跟原始生成不一样的 token ID** —— 这叫 "retokenization drift"，[vLLM 和 Agent Lightning 的分析里有详述](https://github.com/microsoft/agent-lightning)。在重 tokenize 文本上算的 RL 梯度是 off-policy 的。

**Polar 的不变量**（论文原话）：

> *Every trainable token matches the behavior policy during rollout, and any non-generated tokens are masked out.*

机制：直接从 inference server 响应里拷贝采样到的 assistant token（精确 token ID 在那里），interstitial / 非生成 token 用 canonical prompt tokenization 填，通过 `loss_mask = 0` 屏蔽。

### 策略 1：`per_request` —— 保守 baseline

每次 model call → 一条 trace。对单次 completion **无损**，但把多轮 agent session 切碎成几百条短样本。论文说一个 SWE-Bench 问题能产出 **1000+ 条 trace**，淹掉 trainer。

### 策略 2：`prefix_merging` —— 新贡献

数学设定。一个 session 抓到的 completion 序列 $C_1, \ldots, C_T$，每个有：

- Prompt token 序列 $p_i$
- 采样响应 token 序列 $a_i$
- 响应 log 概率 $\ell_i$
- 请求/响应 messages $m_i$

Polar 把 completion 分到有序 chain $\mathcal{G} = \{G_1, \ldots, G_J\}$，每个 $G_j = (C_{i_{j1}}, C_{i_{j2}}, \ldots, C_{i_{jK_j}})$。新 completion 加入已有 chain 的条件**仅当**：

1. 规范化的 message 级 grouping key 把它认成候选 continuation
2. **严格 token 前缀关系**成立：链内相邻 completion 之间，
   $$
   p_{i_{m+1}}[1 : |p_{i_m}|] = p_{i_m}
   $$

这让链成员检查很便宜（整数数组前缀对比），**自然处理**：

- **子 agent** → 起独立的链（它们的 prompt 不是父 prompt 的延伸）
- **并行分支** → 独立的链
- **上下文压缩** → 压缩后的 prompt 不再是原 prompt 的前缀，新链开始
- **prompt 改写** → 同压缩

### 链内合并怎么做

链 $G = (C_{i_1}, \ldots, C_{i_K})$，简写 $p_m, a_m, \ell_m$。令 $e$ = end-of-turn token ID。相邻 completion $m, m+1$ 之间的 canonical tail：

$$
t_m = p_{m+1}[|p_m| + 1 :]
$$

在 $t_m$ 里找第一个 $e$：
- 如果 $a_m$ 已经以 $e$ 结尾：interstitial $u_m$ = 那个 $e$ 之后的后缀
- 否则：$u_m$ 从那个 $e$ 开始，保证下一个 prompt context 之前 assistant turn 是闭合的

这条链表示的 token 序列：

$$
z^{(j)} = p_1 \,\|\, a_1 \,\|\, u_1 \,\|\, a_2 \,\|\, u_2 \,\|\, \cdots \,\|\, a_K
$$

emit 的 trace：
- **Trace prompt** = $p_1$
- **Trace response** = $a_1 \| u_1 \| \cdots \| a_K$
- **Loss mask** = 1 对应 $a_m$ token（可训练）；0 对应 $u_m$ token（interstitial / 非生成）
- **Log-probs** = $a_m$ token 用真实 $\ell_m$ 项；$u_m$ 位置填合成占位符让 `response_logprobs` 跟 `response_ids` 对齐

> [!note]- 占位符为什么重要
> Trainer 期待 `response_logprobs` 和 `response_ids` 同长度。如果在 logprobs 数组里*跳过* interstitial 位，下游梯度计算就坏了。Polar 填占位符，再由 `loss_mask = 0` 让它们不影响梯度 —— 对齐保住、梯度不流过。

### 哪些被训、哪些被屏蔽

```
trace response = [a₁]  [u₁]   [a₂]  [u₂]   ...   [a_K]
loss_mask     =  1s     0s     1s    0s    ...    1s
                ▲      ▲      ▲     ▲             ▲
                │      │      │     │             │
                │      │      │     │             behavior-policy tokens
                │      │      │     │             （最后一轮 assistant）
                │      │      │     canonical interstitial
                │      │      │     （轮次之间的系统渲染）
                │      │      behavior-policy tokens
                │      canonical interstitial
                behavior-policy tokens
```

每个可训练的 token 都是模型实际采样过的。每个被屏蔽的 token 都是 harness 或 server *塞给*模型的上下文。梯度按构造就是 on-policy 的。

> [!question]+ Shiki —— Gateway 级异步分级跟 LLM prefill 相关吗？(2026-05-27)
>
> **不相关**。这里的 "prefill" 跟 LLM inference 的 prefill 阶段含义完全不一样。LLM **prefill** 是 vLLM 内部概念：一次 forward 把整段 prompt 算完、填好 KV cache，然后才开始 decode。
>
> Polar 的"分级"是**编排层**的并发设计 —— gateway 节点内部多 worker pool 让 rollout pipeline 不同阶段并行跑，跟 LLM forward 形状完全无关。
>
> 一次 SWE-Bench rollout 有 3 个成本结构很不同的阶段：
>
> | 阶段 | 时间 | 资源 |
> | ---- | ---- | ---- |
> | **INIT** —— 起 Apptainer 容器、装 harness、配 git repo | 30-90 秒 | CPU + 磁盘 |
> | **RUNNING** —— harness 跑 agent loop（LLM 调用 + tool 执行） | 1-5 分钟 | GPU（LLM）+ CPU（tool） |
> | **POSTRUN** —— Verifier 跑（SWE-Bench 跑测试套件）+ 撤资源 | 30-180 秒 | CPU |
>
> **串行跑这三段，GPU 在 INIT 和 POSTRUN 期间完全闲置**，浪费总时间的 ~30-50%。Polar gateway 内部有 4 个池子并发跑：
>
> - **INIT 池**在背景初始化新 session
> - **READY buffer** 装着已初始化但还没开跑的 session
> - **RUNNING 池**执行 harness（GPU 活跃阶段）
> - **POSTRUN 池**给已完成的 session 评分撤资源
>
> 效果：GPU 几乎一直在做 LLM 推理（87.7% 利用率），CPU 在背景做容器管理和评估。**这是 Polar 从 20.4% 提升到 87.7% rollout GPU 利用率的关键** —— pipeline-stage 并发，不是更快 LLM kernel。
>
> 类比：vLLM prefill ≈ 厨师切菜（一个菜的内部步骤）；gateway 分级 ≈ 厨房里几个工位（备菜 / 炒菜 / 洗碗）并行运作（整个 pipeline 编排）。

## 异步 rollout 分级

每个 gateway 有**三个 worker 池 + 一个 buffer**（ProRL Agent INIT→RUN→EVAL 的精炼）：

| 阶段 | 角色 | 为什么要独立池 |
| ---- | ---- | -------------- |
| **INIT** | 起 runtime、跑 prepare 动作 | CPU 重，能耗时几分钟 |
| **READY（buffer）** | 装着已初始化的 runtime，直到有运行槽 | 让 INIT 在关键路径外推进 |
| **RUNNING** | 执行 harness | GPU-bound（驱动 LLM inference） |
| **POSTRUN** | 构建轨迹、跑 evaluator、发回调、撤资源 | 可能包含长尾 patch 验证 |

**READY buffer 是相对 ProRL Agent 的新增**：把 runtime 准备跟 agent 执行解耦。Agent 跑的时候下一批 runtime 已经在预热。配合 agent run 期间预热 evaluator，这就是 `prefix_merging` 拿到 87.7% rollout GPU 利用率（vs ProRL Agent 较低数字）的关键。

**Per-session 死线**。每个 session 一个 timeout 预算。如果 harness 在 model call 已经被抓后才超时，gateway 仍然进 POSTRUN，partial trace 能恢复 —— 部分 RL 信号比丢掉 RL 信号好。

## 标志证据

### 在线 RL：四个 coding harness 上的 SWE-Gym GRPO

**配置**。Qwen3.5-4B base、SkyRL-v0-293-data（训练）、SWE-Bench Verified（评估）、标准 GRPO、Polar + Slime trainer。所有 run 都用 `prefix_merging` 构建轨迹、`swebench_harness` 给最终 patch 评分。

**SWE-Bench Verified pass@1**（Table 1）：

| Harness | Base | Polar RL | 增益 |
| ------- | ---: | -------: | ---: |
| Codex | 3.8 % | **26.4 %** | **+22.6** |
| Claude Code | 29.8 % | 34.6 % | +4.8 |
| Qwen Code | 34.6 % | 35.2 % | +0.6 |
| Pi | 34.2 % | 40.4 % | +6.2 |

训练曲线四个 harness 都稳定涨。前 10 步 vs 后 10 步均值：

| Harness | 前 10 步 | 后 10 步 |
| ------- | -------: | -------: |
| Codex | 9.5 % | 54.5 % |
| Claude Code | 28.8 % | 67.0 % |
| Qwen Code | 61.6 % | 66.0 % |
| Pi | 61.6 % | 76.2 % |

> [!success] Codex 那个数字到底说明什么
> Qwen3.5-4B 在 Codex 下 **3.8 % pass@1** 是一个根本不会用 Codex 协议的模型 —— patch 格式错、tool schema 错、停止条件错。Polar 的贡献是**奖励挂到 Codex 执行路径上实际采样到的 token 上** —— GRPO 优化的是模型在评估时必须用的行为，不是 `AgentHandler` 里重实现版本的行为。在 Qwen 原生 harness（Qwen Code）下，base 已经懂协议；+0.6 pp 说"Polar 没破坏本来在 work 的东西"。两个端点合在一起才是"harness-native RL"声明的正确形状。

### 关键消融：prefix_merging vs per_request

同模型、同硬件、同拓扑，只换轨迹构建器。3 个训练步：

| 策略 | Trainer 更新数 | Wall-clock | Rollout GPU 利用率 |
| ---- | -------------: | ---------: | -----------------: |
| `per_request` | 1,185 | 189.5 min | 20.4 % |
| **`prefix_merging`** | **218** | **35.2 min** | **87.7 %** |

`per_request` 在同物理 work 下产生 ~5× 多的 trainer 更新。Wall-clock 5.39× 加速来自 trainer 批量梯度计算主导：1185 个独立的 trainer iteration 比 218 个慢 ~5×，即使每个单独更便宜。

> [!important] per_request + outcome reward 广播会触发 reward hacking
> 给每条 `per_request` trace 同一个 session 级 outcome reward（最自然 baseline），论文观察到**严重 reward hacking**：request 级 trace 拿到 session 级信用，没做归一化，所以幸运结局的 noisy trace 也被强化。论文把修复推到 future work（"PRM-style credit assignment 在 roadmap 上"）。当前你要么用 `prefix_merging`、要么接受 hacking 风险；意味着无法合并的 workload（重写上下文很多的 harness）目前用 Polar 做 outcome-only RL 不容易。

### 离线数据生成：HF 上的 SFT 语料

同样的 Polar 基础设施跑离线。论文 case study：

| 设置 | 值 |
| ---- | -- |
| 硬件 | 8× H100 SGLang serve（TP=8、max_model_len=32K） |
| 模型 | Qwen3.5-122B-A10B |
| Harness | pi-coding-agent v0.67.68 |
| 任务 | 1,638 个 SWE-Gym 实例，跨 7 个 repo |
| 并发 | 每 gateway 5-8 session，重试 1 次，timeout 3,600s |
| 接受 | **504 / 1,638 = 30.8 %**（通过完整 FAIL_TO_PASS + PASS_TO_PASS） |
| GPU 小时 | ~64 |
| 平均轨迹 | 104 messages、51 assistant turns（长尾超过 200） |

每个 repo 的接受率（Table 2）：

| Repo | 尝试 | 接受 | 率 |
| ---- | ---: | ---: | -: |
| getmoto/moto | 343 | 184 | 53.6 % |
| python/mypy | 257 | 101 | 39.3 % |
| conan-io/conan | 71 | 27 | 38.0 % |
| pydantic/pydantic | 81 | 24 | 29.6 % |
| iterative/dvc | 219 | 45 | 20.5 % |
| pandas-dev/pandas | 477 | 98 | 19.7 % |
| dask/dask | 141 | 25 | 17.7 % |

bug-fix 偏多的 repo（moto、mypy）接受率高；dataframe / dataflow 工作负载测试套件长、接受率低于 20%。发布在 **[`nvidia/polar-swegym-pi-qwen35-122b-a10b-trajectories`](https://huggingface.co/datasets/nvidia/polar-swegym-pi-qwen35-122b-a10b-trajectories)** ，Apache-2.0，按 repo stratified 90/10 train/test 切分。

## 优势与不足

两个真实优势：(1) **从 adapter 到 observer 的架构转变是真泛化** —— 它让可训练 harness 的集合本质上等于"所有基于 LLM 的 agent"，这是该瞄准的天花板；(2) **轨迹重建的数学做对了** —— token 前缀不变量优雅处理子 agent 和上下文压缩，masked-interstitial loss 在非平凡多轮场景里保住了 on-policy 正确性。

可推敲：

- **所有实验都是 4B 模型**。Codex 的 +22.6 pp 是 showcase，但 Qwen3.5-4B 在 Codex 下从 3.8 % 起步*部分*是"模型不懂协议"问题，不全是"harness 有隐藏能力"。70B+ 上 base 模型本就协议流利时，harness-native RL 的差距可能急剧缩小。论文没扫模型规模，"能 scale to size"的说法是隐含的
- **所有实验都是 coding 任务**。整篇论文是 SWE-Bench / SWE-Gym + 4 个 coding harness。"any harness"的 framing 跟"没评估任何 web-agent（BrowserGym、Mind2Web）、OS-agent（OSWorld）、scientific agent（Aviary）、多模态 harness"是矛盾的。论文引这些作动机但没跑
- **prefix_merging 假设 sub-session 内是 append-only 对话链**。如果 harness *持续*重写历史轮次（有些上下文压缩策略会，比如丢/总结旧 tool 输出），每条 completion 自成单元素链，那段就退回 `per_request`。论文说"压缩天然形成独立链"但没量化真实 harness 上链碎片化的程度
- **Interstitial token 代价**。每轮注入大量上下文的 harness（当前文件内容、检索文档、之前 tool 输出）会产出很长的 $u_m$。合并的 trace 里大量 token 是被 mask 的 —— trainer 还是要为它们付 attention 二次代价。32K context 模型上跑一个每轮注入 20K 状态的 agent，大部分 context window 浪费在训练 masked 位置上
- **`per_request` 下的 reward hacking 留给 future work**。论文承认（"PRM-style credit assignment 在 roadmap 上"）但没修。当前要么 `prefix_merging`、要么接受 hacking 风险；难合并的 workload（重写上下文很多的 harness）目前不容易用 Polar 做 outcome-only RL
- **没跟 [[prorl-agent|ProRL Agent]] 直接对比**。论文修辞上 supersede ProRL Agent（"Polar rewrites..."）但从没做一对一实验：同任务、同硬件，ProRL Agent `AgentHandler` plugin vs Polar proxy。直觉说 proxy 必然比同进程 plugin 慢（网络一跳、请求转换、响应合成），但慢多少？没测
- **闭源 provider 兼容比听起来脆弱**。Polar 的 proxy 在 Anthropic / OpenAI Responses / OpenAI Chat / Google 之间翻译。这些 API *会变*。Anthropic 的 tool-use 格式 2025 改过两次；OpenAI Responses 还在 beta。论文没说 provider transformer 层怎么跟上游 API 演化保持同步
- **有内建重试的 harness 怎么办**？很多生产 harness 在 LLM call 失败时重试退避。Polar 的 proxy 会记每一次尝试，包括失败的。轨迹重建过滤掉重试的 call，还是失败尝试污染 trace？没说
- **`opencode` 列在支持但实验不测**。§3.2.1 的"流行 harness"列表（`claude_code`、`codex`、`gemini_cli`、`qwen_code`、`opencode`、`pi`）比实验集（Codex、Claude Code、Qwen Code、Pi）更广。Gemini CLI 和 OpenCode 缺席 —— 很可能因为需要 Anthropic / Google API 翻译路径还没打磨好

> [!warning] 架构风险：harness 厂商可以故意打破 Polar
> Codex、Claude Code、Qwen Code、Gemini CLI 是产品。如果厂商决定*阻止*第三方模型替换（比如签名请求方案、首方推理 endpoint 证书绑定），Polar 在那些 harness 上就废了。Polar 的整个前提是 LLM API 边界 *通用且未守*。第二条依赖商业行为，不是技术设计

## 这意味着什么

值得追踪的三条声明：

1. **黑盒-via-API-proxy 模式成为 agentic-RL 默认**。Polar 的核心洞察 ——"LLM API 是通用接口"—— 是那种事后看显然、采纳必然的想法。期待 veRL、NeMo RL、OpenRLHF、OpenHands 在 12 个月内发布自家的 proxy gateway 层。期待 [Agent Lightning](https://github.com/microsoft/agent-lightning) 吸收 proxy 模式；期待 rLLM 加 `proxy_mode` 开关。
2. **"什么算可训练 harness"的边界戏剧性挪移**。Polar 之前：任何有 Python AgentHandler adapter 的（小列表、NVIDIA-curated）。Polar 之后：任何打 HTTP LLM API 的（基本上是 2025-26 出货的每个 agent）。这就解锁了对 OpenHands、Claude Code、Codex 这些产品的 RL 训练 —— 它们之前不可碰因为不是 Python 库。agent-as-product 趋势（闭源、二进制分发）变得 RL 兼容。
3. **Token-fidelity 成为一等关切**。Retokenization-drift 问题多年来悄悄毁掉多轮场景的 RL 梯度（vLLM 团队和 Agent Lightning 团队都写过）。Polar 的 framing —— "behavior-policy token vs canonical interstitial，加显式 loss mask" —— 是正确的词汇。期待这套术语扩散，期待未来 RL 框架在公开 API 里给 token-fidelity 保证。

这*不是*：通用 RL trainer（论文自己说 —— Polar 是 rollout 基础设施、不是 trainer；它喂 Slime、NeMo RL、veRL）。也*还不是* `per_request` reward hacking 问题的修复 —— 那仍是开放系统问题。

## 三层 agentic-RL 栈 —— Polar / ProRL Agent / NeMo Gym

这是大家最容易混淆的问题。三个 NVIDIA 项目（ProRL Agent、Polar、NeMo Gym）加上 trainer（veRL / NeMo-RL / slime），共同构成一个**三层架构**。每一层回答不同的问题。

### 一句话定位

| 层级 | 系统 | 回答的问题 |
| ---- | ---- | ---------- |
| **Trainer** | veRL / NeMo-RL / slime | "拿到 trajectory 之后怎么更新 policy？" |
| **Rollout-driver** | ProRL Agent → Polar | "怎么实际跑 agent 把任务做一遍，抓 trajectory？" |
| **Environment catalog** | NeMo Gym | "任务是什么？怎么打分？" |

ProRL Agent 和 Polar 是**同一层、新旧版本**（Polar 取代 ProRL Agent，同一 NVIDIA 团队、同一 repo）。NeMo Gym 是**不同层**，给 rollout-driver 提供输入。

### 栈图

```
┌──────────────────────────────────────────────────────────┐
│       Trainer (veRL / slime / NeMo-RL / OpenRLHF)         │
│   ─ PPO/GRPO ─ 梯度更新 ─ 分布式训练                       │
└──────────────────────────┬───────────────────────────────┘
                           │
                           │  HTTP rollout 请求：
                           │  "用 policy π 跑 task T"
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│      Rollout-driver (ProRL Agent → Polar, 2026-05)         │
│  ─ 起 sandbox ─ 跑 harness ─ 拦 LLM API ─                │
│  ─ 重建 trajectory ─ 跑 verifier ─ 算 reward               │
└────┬────────────────────┬───────────────────────────────┘
     │                    │
     │ 需要 task          │ harness 调 LLM
     │ 需要 runtime       │ (proxy 拦截)
     │ 需要 verifier      │
     ▼                    ▼
┌─────────────────┐  ┌────────────────────────────────┐
│   NeMo Gym       │  │  vLLM (训练的 policy π)         │
│  ─ 84 benchmark │  │  + reference policy                │
│  ─ runtime 镜像 │  │  + critic (PPO 用)                  │
│  ─ verifier     │  │  ─ hybrid engine 跟 FSDP actor 共享 │
│  ─ 数据切分     │  │                                    │
└─────────────────┘  └────────────────────────────────┘
```

**注意**：NeMo Gym 在 rollout-driver *旁边*，提供输入 —— 不在上面也不在下面。Trainer 不直接跟 NeMo Gym 通话；Polar 读 NeMo Gym 的 task / runtime / verifier 规格作为自己工作的输入。

### 历史脉络

```
2026-03  ProRL Agent       ┐
                            ├─ 平行 NVIDIA 项目，无公开 adapter
2026-03  NeMo Gym          ┘     （我之前说的 "must pick one" gap）

2026-05  Polar              ─ 取代 ProRL Agent（同一 NVIDIA repo）
         + 注册成 NeMo Gym 环境 ─ 桥出来了
```

**关键事件**：2026-03 ProRL Agent 和 NeMo Gym 作为平行项目出货，没有正式连接。2026-05 Polar 取代 ProRL Agent **并且**注册成 NeMo Gym 环境，正式把两层连起来。这是 NVIDIA agentic-RL stack 整合的时刻。

### 各自单独能做什么

| 单独用 | 能做 | 不能做 |
| ------ | ---- | ------ |
| **NeMo Gym 单独** | benchmark 评测（在 84 个任务上跑现有模型）；给别的 RL 框架提供 task | 不能训练（没 trainer）；harness 集成留给用户 |
| **ProRL Agent / Polar 单独** | rollout 服务（给 task 跑 harness 抓 trajectory）；offline SFT 数据生成 | 不能定义 task（task 你自己给）；不能训练 |
| **Trainer 单独** | PPO/GRPO 在简单 RL 任务上；经典 RLHF | 不能做 agentic（没多轮 rollout 基础设施）；没 env catalog |

**只有三个一起 = 完整生产 agentic-RL 训练 pipeline**。

### 具体训练运行 —— 每一步谁干什么

假设你要"训 Qwen2.5-7B 在 SWE-Bench Verified 上做 GRPO"：

```
1. 用户：./train_swebench.sh
              │
              ▼
2. Trainer (slime) 启动：
   ─ 加载 SWE-Bench Verified task list ←── 从 NeMo Gym
   ─ 加载 Apptainer runtime spec      ←── 从 NeMo Gym
   ─ 知道 verifier 是 swebench_harness ←── 从 NeMo Gym
              │
              ▼ 每个 RL step：
3. Trainer 选一个 batch 的 task instance
              │
              ▼ HTTP POST /process(task_batch)
4. Polar gateway 收到请求：
   ─ INIT pool：起 N 个 Apptainer 容器（用 NeMo Gym 镜像规格）
   ─ 在每个容器里装 Codex CLI / Claude Code / Pi
   ─ 把 task 的 git repo + 题目灌进去
              │
              ▼
5. Polar RUN pool：在每个容器里启动 Codex CLI 进程
              │
              ▼
6. Codex CLI 跑自己的 agent loop：
   ─ 读文件（bash tool）       ←── Polar 看不见
   ─ 调 LLM API ───────────────┐
                              │
                              ▼ Polar API proxy 拦截
                          ┌────────────────────────────┐
                          │ proxy 转发给本地 vLLM        │
                          │ 录 token IDs + logprobs     │
                          │ 把 text 响应按 OpenAI/Anthropic │
                          │ 格式返回给 harness          │
                          └────────────────────────────┘
   ─ 编辑文件（edit tool）     ←── Polar 看不见
   ─ ... 多轮 ...
   ─ 提交 patch
              │
              ▼
7. Polar POSTRUN pool：
   ─ 跑 swebench_harness（NeMo Gym 的 verifier）
   ─ 拿到 reward (0 或 1)
   ─ 跑 prefix_merging 重建 trajectory（含 token IDs + loss_mask）
              │
              ▼ HTTP 响应：(token_ids, logprobs, loss_mask, reward)
8. Trainer 收集所有 trajectory，拼 batch
              │
              ▼
9. Trainer 算 GRPO advantage、PPO loss，update Qwen 参数
              │
              ▼
10. 新 Qwen 权重 sync 到 vLLM（hybrid engine 切换）
              │
              ▼
（回到 step 3，下一个 RL step）
```

**每一步谁干什么**：

| 步骤 | NeMo Gym | Polar | Trainer | vLLM |
| ---- | -------- | ----- | ------- | ---- |
| 1-2 启动 | task list + runtime spec + verifier | – | 从 NeMo Gym 加载 | – |
| 3 选 batch | – | – | ✓ | – |
| 4 sandbox | runtime spec | ✓ 起容器 | – | – |
| 5 harness | – | ✓ | – | – |
| 6 agent loop | – | proxy 拦 LLM API | – | ✓ LLM forward |
| 7 评分 | 提供 verifier | ✓ 跑 verifier | – | – |
| 8-9 更新 | – | – | ✓ | – |
| 10 sync | – | – | – | ✓ |

三个独立系统、三个干净的 concern。

### "同族、不同层" 在这里具体含义

- **ProRL Agent ↔ Polar**：**同层、版本升级**。同一 NVIDIA 团队、同一 GitHub repo（`NVIDIA-NeMo/ProRL-Agent-Server`）、同一架构哲学（rollout-as-a-service）。Polar 把 ProRL Agent 的 Python `AgentHandler` plugin 换成 LLM-API proxy —— 这是唯一根本变化。

- **Polar ↔ NeMo Gym**：**不同层、协作**。一个是 rollout-driver、一个是 environment catalog。Polar 注册成 NeMo Gym 环境 = "trainer 通过 NeMo Gym 访问 Polar" 成为标准路径。

Web stack 类比：
- ProRL Agent → Polar 像 **Apache → nginx**（同层 web 服务器，新旧迭代）
- Polar ↔ NeMo Gym 像 **nginx ↔ PostgreSQL**（web 服务器跟数据库，不同层、互相依赖）

### 常见误解："NeMo Gym 就是 SWE-agent 那种 agent 框架吗？"

不是。NeMo Gym 是 *catalog*；SWE-agent / Codex / Claude Code 是 *agent*（harness）。NeMo Gym **引用**了 19 个 agent harness（包括 SWE-agent 风格的）作为可用 harness 列表，但自己**不实现** agent 的推理循环。

| 跟谁混淆 | 实际是 | 在哪一层 |
| -------- | ------ | -------- |
| NeMo Gym = SWE-agent？ | 不 —— SWE-agent 是 *harness*（NeMo Gym 引用的 19 个 harness 之一）。NeMo Gym 是 catalog | Harness ≠ Environment |
| NeMo Gym = Codex / Claude Code？ | 不 —— 它们是 *harness*（可能注册到 NeMo Gym 里，肯定能在 Polar 里用） | Harness 层 |
| NeMo Gym = trainer？ | 不 —— NeMo-RL 是 trainer（不同名字，同一 NVIDIA 家族） | Trainer ≠ Environment |
| NeMo Gym = vLLM？ | 不 —— vLLM 是 LLM 推理 | LLM-engine 层 |

NeMo Gym **以上都不是** —— 它是连接组织：说"这里有 84 个 task，配着 Apptainer 镜像和 verifier，准备好被任何 rollout-driver 或 trainer 消费"。

### "必须二选一时" —— 更新

旧答案（ProRL Agent 和 NeMo Gym 没连接时，见 [[prorl-agent#ProRL Agent vs NeMo Gym —— 同族、不同层|prorl-agent]]）：基于你看重什么选 —— token 级 off-policy 正确性 vs benchmark catalog 广度。

**新答案（2026-05 起）**：**两个都用**。Polar 填 rollout-driver 层，NeMo Gym 填 environment-catalog 层，Polar 注册成 NeMo Gym 环境所以二者自然连接。

## 源码与复现

跟 ProRL Agent 同一个 repo（团队保留了 GitHub URL 但重写了代码）：

| 路径 | 角色 |
| ---- | ---- |
| [`server/gateway/`](https://github.com/NVIDIA-NeMo/ProRL-Agent-Server) | Gateway 节点实现 —— proxy、INIT/RUN/POSTRUN 池 |
| `server/proxy/` | Provider API transformer（Anthropic / OpenAI Chat / OpenAI Responses / Google） |
| `server/trajectory/` | per-request + prefix-merging 轨迹构建器 |
| `harnesses/` | 预构建的 `claude_code`、`codex`、`gemini_cli`、`qwen_code`、`opencode`、`pi` adapter，外加通用 shell harness |
| `trainer_integration/slime/` | 跟 Slime 异步 RL trainer 的参考集成 |

最小复现 recipe —— Qwen3.5-4B + Codex + GRPO + Slime（Codex 标志实验）：

```bash
# 1. 启 Polar
docker compose -f deploy/polar.yaml up   # 或 apptainer/，HPC 用

# 2. 配 Codex harness adapter
cat > harness.yaml <<EOF
harness: codex
adapter:
  install:
    - npm install -g @openai/codex@latest
  env:
    OPENAI_BASE_URL: ${POLAR_GATEWAY_URL}
    OPENAI_API_KEY: dummy
  command: ["codex", "exec", "--task", "${TASK}"]
EOF

# 3. 训练（Slime、simple GRPO）
slime train \
  --model Qwen/Qwen3.5-4B-Base \
  --dataset NovaSky-AI/SkyRL-v0-293-data \
  --rollout polar \
  --polar-endpoint http://localhost:8000 \
  --polar-harness harness.yaml \
  --trajectory-builder prefix_merging \
  --evaluator swebench_harness
```

发布的 SFT 语料：

```bash
huggingface-cli download nvidia/polar-swegym-pi-qwen35-122b-a10b-trajectories \
  --repo-type dataset \
  --local-dir ./polar-sft-corpus
```

## 相关阅读

- [[prorl-agent]] —— 直接前作；同一个 NVIDIA repo，2026-05 被 Polar 替换。那里记录的 `AgentHandler` ABC plugin 架构是 Polar 取代的设计；更广的 rollout-as-a-service framing 仍然准确
- [[nemo-gym]] —— Polar 注册进的环境 catalog。NeMo Gym 拥有 84-benchmark + 19-harness 库存；Polar 是执行它们的 rollout 驱动层
- [[agentic-rl-overview]] —— 智能体 RL 框架更广的图景
- [[grpo]] —— Polar 实验用的 RL 算法
- [[rl-training-frameworks]] —— Slime、NeMo RL、VeRL、OpenRLHF —— 消费 Polar rollout 的 trainer
- [[environment-design]] —— 沙箱基础设施设计（Apptainer、OpenReward、ARES、Daytona）；Polar 继承 ProRL Agent 的 rootless-HPC 沙箱
- [[tool-use-rl]] —— 工具使用 agent 的 RL；Polar 实验直接瞄准这个
- [[das-spec-rl]] —— RL rollout 的投机解码加速；在推理层互补
- [[aurora]] —— 2026 另一篇"rollout 作为活系统"论文（在线投机解码 draft 训练）；Polar 和 Aurora 在同一架构脉络里攻不同瓶颈

## 参考文献

- 论文：Xu et al., *Polar: Agentic RL on Any Harness at Scale*, 2026-05-22. [arXiv:2605.24220](https://arxiv.org/abs/2605.24220)
- 前作：Zhang et al., *ProRL Agent*, 2026-03. [arXiv:2603.18815](https://arxiv.org/abs/2603.18815) —— [[prorl-agent]]
- 发布的 SFT 语料：[`nvidia/polar-swegym-pi-qwen35-122b-a10b-trajectories`](https://huggingface.co/datasets/nvidia/polar-swegym-pi-qwen35-122b-a10b-trajectories)（Apache-2.0）
- 代码：[github.com/NVIDIA-NeMo/ProRL-Agent-Server](https://github.com/NVIDIA-NeMo/ProRL-Agent-Server)
- SWE-Bench Verified：Jimenez et al. [arXiv:2310.06770](https://arxiv.org/abs/2310.06770)
- SWE-Gym：Pan et al.（软件工程 agent 训练环境）
- SkyRL-Agent：Cao et al.（多轮 agent 全栈 RL）；[SkyRL-v0-293-data](https://huggingface.co/datasets/NovaSky-AI/SkyRL-v0-293-data)
- Agent Lightning：Luo et al.（基于 tracing 的 agent RL，含 retokenization-drift 讨论）—— [microsoft/agent-lightning](https://github.com/microsoft/agent-lightning)
- rLLM：Tan et al.（跨框架 agent RL，跟踪 client）
- Slime：Zheng et al. & Zhu et al.（Megatron 训练 + SGLang rollout）
- PRIME-RL：Prime Intellect（异步 RL，stale-policy 语义）
- Harbor：Harbor Framework Team（容器化 agent 评估）
