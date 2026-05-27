---
title: "Polar: Agentic RL on Any Harness at Scale (the ProRL Agent successor)"
category: agentic-rl
tags: [polar, prorl-agent, nvidia, rollout-as-a-service, agentic-rl, llm-api-proxy, nemo-gym, swe-bench, paper-review]
created: 2026-05-26
updated: 2026-05-26
status: mature
paper: arXiv:2605.24220
code: https://github.com/NVIDIA-NeMo/ProRL-Agent-Server
---

# Polar: Agentic RL on Any Harness at Scale (the ProRL Agent successor)

> [!info] Paper metadata
> - **Paper**: [arXiv:2605.24220](https://arxiv.org/abs/2605.24220) — *Polar: Agentic RL on Any Harness at Scale*, 2026-05-22
> - **Code**: [NVIDIA-NeMo/ProRL-Agent-Server](https://github.com/NVIDIA-NeMo/ProRL-Agent-Server) — **same repo as [[prorl-agent|ProRL Agent]]; Polar rewrites it in-place**
> - **Authors**: Binfeng Xu, Hao Zhang, Shaokun Zhang, Songyang Han, Mingjie Liu, Jian Hu, Shizhe Diao, Zhenghui Jin, Yunheng Zou, Michael Demoret, Jan Kautz, Yi Dong
> - **Name origin**: "Pr**O**rL Agent serv**R**" → **Polar**, also evoking the two "poles" of agent training and product deployment
> - **Status**: registered as a [[nemo-gym|NeMo Gym]] environment — this is the consolidation bridge that was missing as of May 2026

> [!important] Replaces [[prorl-agent|ProRL Agent]]
> The paper says it explicitly: "Polar rewrites its preceding work, ProRL Agent, and has been registered as one of NeMo Gym environments." Same NVIDIA team (~75% author overlap), same GitHub repo. The [[prorl-agent|ProRL Agent]] page documents the predecessor architecture; this page is the current state of NVIDIA's agentic-RL rollout substrate.

---

## Summary (read this if you have 2 minutes)

**What it is.** Polar is NVIDIA's second-generation agentic-RL rollout framework, succeeding [[prorl-agent|ProRL Agent]]. Where ProRL Agent required you to write a Python `AgentHandler` ABC adapter for every agent harness, Polar runs **any unmodified agent harness — Codex, Claude Code, Qwen Code, Pi, Gemini CLI, OpenCode — as a black box** and intercepts its LLM API calls through a proxy. Captured tokens + log-probs are reconstructed into token-faithful trajectories for the trainer.

**The one idea.** **Move the integration boundary from the agent's Python API to the LLM provider API.** Every LLM-based agent has to talk to a model — that's the universal interface. Sit between the harness and the inference server, record everything, reconstruct trajectories. Three sub-pieces hold this up:

1. **Provider-compatible proxy** — accepts Anthropic Messages, OpenAI Chat Completions, OpenAI Responses, and Google `generateContent` shapes; translates to local inference; records prompt token IDs, sampled tokens, log-probs.
2. **Token-faithful prefix merging** — multi-turn conversations are reconstructed into traces where only behavior-policy sampled tokens are trainable (loss mask = 1); canonical interstitial tokens (the harness's rendering of prior turns + injected context) are masked out (loss mask = 0). Subagents, context compaction, prompt rewriting naturally form separate chains.
3. **Gateway-level async staging** — rollout server + gateway nodes; each gateway has INIT / RUNNING / POSTRUN worker pools + READY buffer so CPU-bound runtime setup and long-tail evaluation don't block GPU-bound agent execution.

Remove the proxy and you're back to ProRL Agent's plugin-per-harness model; remove prefix-merging and your trainer drowns in 1000s of short fragmented traces; remove async staging and rollout serializes against training.

**Headline result.** Qwen3.5-4B base + simple GRPO on SkyRL-v0-293-data, evaluated on SWE-Bench Verified pass@1:

| Harness | Base | Polar RL | Gain |
| ------- | ---: | -------: | ---: |
| **Codex** | 3.8 % | **26.4 %** | **+22.6** |
| Claude Code | 29.8 % | 34.6 % | +4.8 |
| Qwen Code | 34.6 % | 35.2 % | +0.6 |
| Pi | 34.2 % | 40.4 % | +6.2 |

Same base model, four different harnesses, all improve. **Codex's +22.6 pp is the showcase number**: Qwen3.5-4B starts barely functional under Codex's unfamiliar action protocol, and harness-native RL teaches it the protocol. The smallest gain (Qwen Code +0.6) is on the harness the base model was already aligned with — exactly the right shape.

**Critical ablation.** Trajectory reconstruction strategy matters enormously: `per_request` (every model call = one trace) vs `prefix_merging` (chains of append-only completions merged), same workload, same 3 training steps:

| Strategy | Trainer updates | Wall-clock | Rollout GPU util |
| -------- | --------------: | ---------: | ---------------: |
| `per_request` | 1185 | 189.5 min | 20.4 % |
| **`prefix_merging`** | **218** | **35.2 min** | **87.7 %** |

**5.39× wall-clock speedup**, **4.3× GPU utilization improvement**. `per_request` with outcome-reward broadcasting *also* causes significant reward hacking — request-level traces get session-level credit without proper normalization. The reconstruction algorithm isn't polish; it's load-bearing.

**Why it matters.**

- **Trainable harnesses just expanded by 10×.** Anything that talks to an LLM API can now be trained — including closed-source binaries (Codex), TypeScript CLIs (Claude Code), Go agents (Pi). No more "we'll integrate it when someone writes the AgentHandler."
- **The ProRL Agent vs NeMo Gym consolidation happened.** Yesterday's wiki claim that no adapter exists between them was already stale — Polar is the adapter, registered as a NeMo Gym environment. See [[prorl-agent#ProRL Agent vs NeMo Gym — same family, different layer]] (now superseded by this).
- **Released agentic SFT corpus.** [`nvidia/polar-swegym-pi-qwen35-122b-a10b-trajectories`](https://huggingface.co/datasets/nvidia/polar-swegym-pi-qwen35-122b-a10b-trajectories) on HuggingFace, Apache-2.0 — 504 SWE-Bench-passing trajectories from Qwen3.5-122B-A10B + pi-coding-agent, average 104 messages / 51 assistant turns each. Reproducible offline SFT.
- **2027 prediction.** The "agent-as-black-box via API proxy" pattern becomes the default. Expect every major RL framework (veRL, NeMo RL, OpenRLHF) to ship a Polar-style proxy gateway; expect Anthropic/OpenAI to publish "RL-friendly" extensions to their public APIs that surface logprobs directly.

---

# Depth (drill-down starts here)

## Background: why ProRL Agent's plugin model hit a ceiling

[[prorl-agent|ProRL Agent]] (NVIDIA, March 2026) made rollout an HTTP service — its key contribution. But its integration contract was: **write a Python `AgentHandler` subclass that drives the agent loop inside the rollout service**. Concretely:

```python
class AgentHandler(ABC):
    @abstractmethod
    async def initialize(self, task): ...
    @abstractmethod
    async def run(self, model_client) -> Trajectory: ...
    @abstractmethod
    async def evaluate(self, trajectory) -> float: ...
```

Every new harness — OpenHands, Mini-SWE, LangGraph-style agents, Aviary — required someone to:
1. Read the harness's Python source.
2. Port its event loop, tool definitions, context-management logic, and reward-eval logic into an `AgentHandler` plugin.
3. Maintain that plugin as the upstream harness evolves.

This worked for ~5 harnesses NVIDIA cared about, but failed for the broader ecosystem in three concrete ways the paper names:

| Failure | Concrete examples |
| ------- | ----------------- |
| **Closed-source / binary harnesses** | Codex CLI ships as a binary; reimplementing it inside `AgentHandler` is impossible. |
| **Non-Python harnesses** | Claude Code is TypeScript, Pi-coding-agent is Go; you'd have to translate the entire event loop. |
| **Fast-evolving harnesses** | Claude Code's prompt structure changes monthly; keeping a Python reimplementation in sync is a treadmill. |

Comparable systems (SkyRL-Agent, PRIME-RL, rLLM, Agent Lightning) had variants of the same problem: even systems that "reduce" integration cost (Agent Lightning's tracing, rLLM's decorators) still required the harness to *cooperate* — call a decorated method, emit a span, conform to an SDK. **Polar's central question** (quoting the paper):

> *Can we train agents with RL without opening the box?*

The shift is conceptually small but architecturally enormous: instead of integrating with the agent **(the cooperative API)**, listen to its **LLM API traffic** (the universal API). Every LLM-based agent must talk to a model. That's the lowest common denominator.

| Prior systems (cooperate-required) | Polar (black-box) |
| ---------------------------------- | ----------------- |
| SkyRL-Agent, PRIME-RL — harness adapts to RL infra | Harness runs unchanged |
| Agent Lightning — tracing SDK hooks inside harness code | Proxy between harness and LLM API; no harness code change |
| rLLM — decorated functions, tracked clients | Provider-API protocol detection (Anthropic / OpenAI / Google) |
| ProRL Agent — `AgentHandler` ABC plugin per harness | Tiny *adapter* that writes config + returns shell command |

## The proxy-as-boundary architecture

![Polar's proxy boundary (paper Fig. 2)](EN/wiki/agentic-rl/polar-figs/polar-proxy-boundary.png)

Left: the classic "harness as components" model (Gymnasium / `env.init/step/reset`-style) requires you to reverse-engineer the harness's internal pieces — sys prompt, tool-call format, multi-agent coordination, context-engineering tricks, cron jobs — and reconstruct them inside `env.step()`. Right: Polar treats the harness as a blackbox emitting requests to `v1/chat/completions` / `v1/responses` / `v1/messages` / `googleapis`, intercepts those at an **API proxy**, and reconstructs trajectories outside the harness.

### Polar's two-tier architecture

| Tier | Role | What lives here |
| ---- | ---- | --------------- |
| **Rollout server** | Task scheduling | Accepts `TaskRequest`, expands into `num_samples` sessions, dispatches to gateways, persists results, exposes status polling, accepts gateway callbacks |
| **Gateway node** | Session lifecycle | Starts runtime, prepares harness, runs harness command, hosts model proxy, builds trajectories, evaluates, tears down |

The split is durable-task-state vs per-session-execution. Trainers (Slime, NeMo RL, veRL) hit the rollout server's async endpoint; the rollout server fans out to gateways.

### The four-step proxy protocol

For each incoming model request the harness makes, the gateway proxy:

1. **Detects the provider API.** Path + headers distinguish:
   - Anthropic Messages (`/v1/messages`)
   - OpenAI Chat Completions (`/v1/chat/completions`)
   - OpenAI Responses (`/v1/responses`)
   - Google `generateContent`
2. **Normalizes the request.** Provider transformer converts to OpenAI Chat Completions schema (the local inference server's native shape). Adds `logprobs=true` for training signal.
3. **Captures token-level data.** Forwards to inference server (vLLM / SGLang); stores: prompt token IDs, response token IDs, finish reason, log-probabilities, request/response messages.
4. **Returns the provider shape.** Transforms back to the schema the harness expects. **Streaming**: the proxy obtains a non-streaming upstream response and emits a synthetic provider-shaped event stream — simplifies token capture while staying compatible with SSE-consuming harnesses.

The proxy doesn't understand the agent's planning, tool selection, or stopping logic. It only preserves API compatibility and records enough to reconstruct trainable samples.

### Harness adapter (the small part)

```python
class HarnessAdapter:
    def prepare_runtime(self, runtime, session): ...   # install config, register MCP servers
    def write_provider_settings(self, runtime): ...    # point model base URL at gateway proxy
    def run_command(self, session) -> List[str]: ...   # shell command to launch the agent
```

That's it. The adapter is configuration + shell command, not an agent reimplementation. The paper ships shortcuts for `claude_code`, `codex`, `gemini_cli`, `qwen_code`, `opencode`, `pi`, plus a generic shell-command harness.

### Runtime interface — Docker + rootless Apptainer

Same isolation choice as ProRL Agent: **rootless Apptainer** for HPC / Slurm clusters where Docker daemons aren't available. Initial release also supports Docker. The interface (`start, stop, exec, upload, download, cancel`) means swapping isolation backends is friction-free.

## Token-faithful trajectory reconstruction

This is the technical contribution of the paper. Polar provides two strategies in a registry; the prefix-merging one is the load-bearing one.

### The token-fidelity problem

Provider APIs return *text or tool-call JSON or reasoning fields or streamed events*, not the exact token IDs the inference backend sampled. Decoding response text and re-tokenizing it can produce **different token IDs from the original generation** — this is "retokenization drift," documented in [vLLM and Agent Lightning analyses](https://github.com/microsoft/agent-lightning). RL gradients computed on retokenized text are off-policy.

**Polar's invariant** (stated in the paper):

> *Every trainable token matches the behavior policy during rollout, and any non-generated tokens are masked out.*

Mechanism: copy sampled assistant tokens directly from inference-server responses (where the exact token IDs live), and use canonical prompt tokenization for interstitial / non-generated tokens, masked out via `loss_mask = 0`.

### Strategy 1: `per_request` — the conservative baseline

Every model call → one trace. **Lossless** with respect to individual completions but fragments a multi-turn agent session into hundreds of short samples. For a single SWE-Bench problem, the paper reports this can produce **1000+ traces**, overwhelming the trainer.

### Strategy 2: `prefix_merging` — the new contribution

The mathematical setup. A session has captured completions $C_1, \ldots, C_T$, each with:

- Prompt token sequence $p_i$
- Sampled response token sequence $a_i$
- Response log-probs $\ell_i$
- Prompt/response messages $m_i$

Polar partitions completions into ordered chains $\mathcal{G} = \{G_1, \ldots, G_J\}$ where each $G_j = (C_{i_{j1}}, C_{i_{j2}}, \ldots, C_{i_{jK_j}})$. A new completion joins an existing chain **only when**:

1. A normalized message-level grouping key identifies it as a candidate continuation.
2. The **strict token-prefix relation** holds: for adjacent completions in a chain,
   $$
   p_{i_{m+1}}[1 : |p_{i_m}|] = p_{i_m}
   $$

This makes the chain-membership test cheap (prefix comparison on integer arrays) and **automatically handles**:

- **Subagents** → spawn separate chains (their prompts don't extend the parent's prompt)
- **Parallel branches** → separate chains
- **Context compaction** → the compacted prompt is no longer a prefix of the original, so a new chain starts
- **Prompt rewriting** → same as compaction

### How merging works within a chain

For chain $G = (C_{i_1}, \ldots, C_{i_K})$ with $p_m, a_m, \ell_m$ shorthand. Let $e$ = end-of-turn token ID. The canonical tail between completions $m$ and $m+1$:

$$
t_m = p_{m+1}[|p_m| + 1 :]
$$

Locate the first $e$ in $t_m$:
- If $a_m$ already ends with $e$: interstitial $u_m$ = suffix after that $e$
- Otherwise: $u_m$ starts at that $e$ so the assistant turn is closed before the next prompt

The token sequence for the chain:

$$
z^{(j)} = p_1 \,\|\, a_1 \,\|\, u_1 \,\|\, a_2 \,\|\, u_2 \,\|\, \cdots \,\|\, a_K
$$

The emitted trace has:
- **Trace prompt** = $p_1$
- **Trace response** = $a_1 \| u_1 \| \cdots \| a_K$
- **Loss mask** = 1 on tokens copied from sampled $a_m$ (trainable); 0 on tokens copied from canonical $u_m$ (interstitial / not generated)
- **Log-probs** = real $\ell_m$ entries for $a_m$ tokens; synthetic placeholders for $u_m$ slots so `response_logprobs` stays aligned

> [!note]- Why the placeholders matter
> Trainers expect `response_logprobs` and `response_ids` to be the same length. If you just *skipped* the interstitial slots in the logprobs array, downstream gradient computation breaks. Polar fills them with placeholders that the `loss_mask = 0` then makes irrelevant — alignment preserved, no gradient flow.

### What gets trained vs what gets masked

```
trace response = [a₁]  [u₁]   [a₂]  [u₂]   ...   [a_K]
loss_mask     =  1s     0s     1s    0s    ...    1s
                ▲      ▲      ▲     ▲             ▲
                │      │      │     │             │
                │      │      │     │             behavior-policy tokens
                │      │      │     │             (the last assistant turn)
                │      │      │     canonical interstitial
                │      │      │     (system rendering between turns)
                │      │      behavior-policy tokens
                │      canonical interstitial
                behavior-policy tokens
```

Every trainable token is one the model actually sampled. Every masked token is part of context the harness or server *gave* the model. The gradient is on-policy by construction.

## Asynchronous rollout staging

Each gateway has **three worker pools + one buffer** (a refinement of ProRL Agent's INIT→RUN→EVAL):

| Stage | Role | Why it's its own pool |
| ----- | ---- | --------------------- |
| **INIT** | Start runtime, run prepare actions | CPU-heavy, can take minutes |
| **READY (buffer)** | Holds initialized runtimes until a run slot opens | Lets INIT proceed off the critical path |
| **RUNNING** | Execute the harness | GPU-bound (drives LLM inference) |
| **POSTRUN** | Build trajectories, run evaluators, send callbacks, tear down | Can include long-tail patch validation |

The **READY buffer** is the addition vs ProRL Agent: it decouples runtime preparation from agent execution. While agents are running, the next batch of runtimes is being warmed up. Combined with evaluator prewarm during the agent run, this is what gets `prefix_merging`'s 87.7% rollout GPU utilization vs ProRL Agent's lower numbers.

**Per-session deadline.** Each session has a single timeout budget. If the harness times out *after* model calls have been captured, the gateway still enters POSTRUN with the partial traces — partial RL signal beats lost RL signal.

## Headline evidence

### Online RL: SWE-Gym GRPO on four coding harnesses

**Setup.** Qwen3.5-4B base, SkyRL-v0-293-data (training), SWE-Bench Verified (eval), standard GRPO, Polar + Slime trainer. All runs use `prefix_merging` for trajectory construction and `swebench_harness` for the final patch evaluator.

**Pass@1 on SWE-Bench Verified** (Table 1):

| Harness | Base | Polar RL | Gain |
| ------- | ---: | -------: | ---: |
| Codex | 3.8 % | **26.4 %** | **+22.6** |
| Claude Code | 29.8 % | 34.6 % | +4.8 |
| Qwen Code | 34.6 % | 35.2 % | +0.6 |
| Pi | 34.2 % | 40.4 % | +6.2 |

Training curves show steady reward improvement across all four. First-10-step vs last-10-step averages:

| Harness | First-10 steps | Last-10 steps |
| ------- | -------------: | ------------: |
| Codex | 9.5 % | 54.5 % |
| Claude Code | 28.8 % | 67.0 % |
| Qwen Code | 61.6 % | 66.0 % |
| Pi | 61.6 % | 76.2 % |

> [!success] What the Codex number actually means
> Qwen3.5-4B at **3.8 % pass@1 under Codex** is a model that essentially doesn't know how to use Codex's protocol — wrong patch format, wrong tool schemas, wrong stopping conditions. Polar's contribution is that the **reward attaches to the actual sampled tokens flowing through Codex's execution path** — so GRPO optimizes the behavior the model needs at evaluation time, not the behavior a reimplemented harness in `AgentHandler` would produce. Under the Qwen-native harness (Qwen Code), the base model already knows the protocol; the +0.6 pp says "Polar didn't break what was working." Together these two endpoints are the right shape for a "harness-native RL" claim.

### Critical ablation: prefix_merging vs per_request

Same model, same hardware, same topology, only the trajectory builder changes. Three training steps:

| Strategy | Trainer updates | Wall-clock | Rollout GPU util |
| -------- | --------------: | ---------: | ---------------: |
| `per_request` | 1,185 | 189.5 min | 20.4 % |
| **`prefix_merging`** | **218** | **35.2 min** | **87.7 %** |

`per_request` produces ~5× more trainer updates than `prefix_merging` does for the same physical work. The wall-clock 5.39× comes from the trainer's batched gradient computation dominating: 1185 separate trainer iterations is ~5× slower than 218 even if each iteration is cheaper per-trace.

> [!important] Per-request with outcome-reward broadcasting causes reward hacking
> When you give every `per_request` trace the same session-level outcome reward (the natural baseline), the paper observes **significant reward hacking**: request-level traces get session-level credit without proper normalization, so noisy traces get reinforced by lucky-final-patch sessions. They punt on the fix ("PRM-style credit assignment is on our roadmap"). For now, `prefix_merging` is the only safe option for outcome-reward RL.

### Offline data generation: SFT corpus on HF

The same Polar infrastructure runs offline. The paper case study:

| Setting | Value |
| ------- | ----- |
| Hardware | 8× H100 SGLang serve (TP=8, max_model_len=32K) |
| Model | Qwen3.5-122B-A10B |
| Harness | pi-coding-agent v0.67.68 |
| Tasks | 1,638 SWE-Gym instances across 7 repos |
| Concurrency | 5-8 sessions per gateway, retry once, 3,600s timeout |
| Accepted | **504 / 1,638 = 30.8%** (full FAIL_TO_PASS + PASS_TO_PASS) |
| GPU-hours | ~64 |
| Avg trajectory | 104 messages, 51 assistant turns (long tail >200) |

Per-repo acceptance rates (Table 2):

| Repo | Attempts | Accepted | Rate |
| ---- | -------: | -------: | ---: |
| getmoto/moto | 343 | 184 | 53.6 % |
| python/mypy | 257 | 101 | 39.3 % |
| conan-io/conan | 71 | 27 | 38.0 % |
| pydantic/pydantic | 81 | 24 | 29.6 % |
| iterative/dvc | 219 | 45 | 20.5 % |
| pandas-dev/pandas | 477 | 98 | 19.7 % |
| dask/dask | 141 | 25 | 17.7 % |

Bug-fix-heavy repos (moto, mypy) accept at high rates; dataframe / dataflow workloads with long test suites accept below 20%. Released as **[`nvidia/polar-swegym-pi-qwen35-122b-a10b-trajectories`](https://huggingface.co/datasets/nvidia/polar-swegym-pi-qwen35-122b-a10b-trajectories)** on HuggingFace, Apache-2.0, 90/10 train/test split stratified by repo.

## Strengths and limitations

The two genuine strengths: (1) **the architectural shift from adapter to observer is a real generalization** — it makes the universe of trainable harnesses essentially equal to "all LLM-based agents", which is the right ceiling to aim at; (2) **the trajectory-reconstruction math is properly worked out** — the token-prefix invariant cleanly handles subagents and context compaction, and the masked-interstitial loss formulation preserves on-policy correctness in a non-trivial multi-turn setting.

What I'd push back on:

- **All experiments are on a 4B model.** Codex's +22.6 pp is the showcase, but Qwen3.5-4B starting at 3.8 % under Codex is *partially* a "the model doesn't know the protocol" problem, not a "the harness has hidden capability" problem. At 70B+ where base models are already protocol-fluent, the gap from harness-native RL may shrink dramatically. The paper doesn't sweep model size, so the "scales with model size" claim is implicit.
- **All experiments are coding tasks.** The whole paper is SWE-Bench / SWE-Gym + 4 coding harnesses. The "any harness" framing is contradicted by the fact that no web-agent (BrowserGym, Mind2Web), no OS-agent (OSWorld), no scientific-agent (Aviary), no multi-modal harness is evaluated. The paper cites these as motivation but doesn't run them.
- **prefix_merging assumes append-only conversation chains within sub-sessions.** If a harness *consistently* rewrites prior turns (some context-compaction strategies do this, e.g. dropping/summarizing old tool outputs), every completion starts its own one-element chain and you're back to `per_request` for those sections. The paper says "compaction naturally forms separate chains" but doesn't quantify how much chain fragmentation this causes on real harnesses.
- **Interstitial token cost.** Harnesses that inject heavy context per turn (current file contents, retrieved docs, prior tool outputs) produce very long $u_m$ sequences. The merged trace then has a lot of masked tokens — the trainer pays attention-quadratic cost on them anyway. For a 32K-token model running an agent that injects 20K tokens of state per turn, much of the context window is wasted on training masked positions.
- **Reward hacking under `per_request` is left for future work.** The paper acknowledges it (PRM-style credit assignment "is on our roadmap") but doesn't ship a fix. For now you have to use `prefix_merging` or accept the hacking risk; that means workloads that can't be merged (heavy context-rewriting harnesses) can't easily use outcome-only RL with Polar.
- **No direct comparison with [[prorl-agent|ProRL Agent]].** The paper supersedes ProRL Agent rhetorically ("Polar rewrites...") but never runs an apples-to-apples experiment: same task, same hardware, ProRL Agent's `AgentHandler` plugin vs Polar's proxy. The intuition says proxy must be slower than direct in-process plugin (network hop, request transformation, response synthesis), but how much? Unmeasured.
- **Closed-source provider compatibility is more fragile than it sounds.** Polar's proxy translates between Anthropic / OpenAI Responses / OpenAI Chat / Google. These APIs *change*. Anthropic's tool-use format changed twice in 2025; OpenAI Responses is still beta. The paper doesn't address how the provider transformer layer is maintained as upstream APIs evolve.
- **What happens to harnesses with built-in retries?** Many production harnesses retry failed LLM calls with backoff. Polar's proxy will record every attempt, including failures. Does the trajectory reconstruction filter out retried calls, or do failed attempts pollute the trace? Not addressed.
- **`opencode` mentioned as supported but the experiments don't test it.** The "popular harnesses" list in §3.2.1 (`claude_code`, `codex`, `gemini_cli`, `qwen_code`, `opencode`, `pi`) is broader than the experiments (Codex, Claude Code, Qwen Code, Pi). Gemini CLI and OpenCode are absent — likely because they'd require Anthropic / Google API translation paths that aren't yet polished.

> [!warning] Architectural risk: harness vendors can break Polar deliberately
> Codex, Claude Code, Qwen Code, Gemini CLI are products. If their vendors decide to *prevent* third-party model substitution (e.g., signed-request schemes, certificate pinning to first-party inference endpoints), Polar breaks for those harnesses. Polar's whole premise is that the LLM API boundary is universal *and unguarded*. That second clause is contingent on commercial behavior, not technical design.

## What this means

Three claims worth tracking:

1. **The black-box-via-API-proxy pattern becomes the agentic-RL default.** Polar's core insight — "the LLM API is the universal interface" — is one of those ideas that's obvious in retrospect and inevitable in adoption. Expect veRL, NeMo RL, OpenRLHF, OpenHands to publish their own proxy-gateway layers within 12 months. Expect [Agent Lightning](https://github.com/microsoft/agent-lightning) to absorb the proxy pattern; expect rLLM to add a `proxy_mode` flag.
2. **The "what is a trainable harness" boundary shifts dramatically.** Before Polar: anything with a Python AgentHandler adapter (small list, NVIDIA-curated). After Polar: anything that hits an HTTP LLM API (essentially every agent shipped in 2025-26). This unlocks RL training over OpenHands, Claude Code, Codex, etc. — products that were previously off-limits because they're not Python libraries. The agent-as-product trend (closed-source, binary distribution) becomes RL-compatible.
3. **Token-fidelity becomes a first-class concern.** The retokenization-drift problem has been quietly killing RL gradients in multi-turn settings for years (vLLM team and Agent Lightning team have both written about it). Polar's framing — "behavior-policy tokens vs canonical interstitials, with explicit loss masks" — is the right vocabulary. Expect this terminology to spread, and expect future RL frameworks to expose token-fidelity guarantees in their public APIs.

What this is *not*: a general-purpose RL trainer (the paper says so — Polar is a rollout substrate, not a trainer; it feeds Slime, NeMo RL, veRL). And it's not yet a fix for the per-request reward-hacking problem — that remains an open systems question.

## How this changes the ProRL Agent vs NeMo Gym picture

[Yesterday's wiki section](#related-reading) compared [[prorl-agent|ProRL Agent]] and [[nemo-gym|NeMo Gym]] as "same family, different layer" with no public adapter bridging them. **Polar IS that bridge**, and shipped before the ink dried on that section. The updated picture:

```
Trainer (NeMo RL / VeRL / Slime / Unsloth)
    │
    │  async rollout-as-a-service contract     ← Polar's gateway endpoints
    ▼
Polar (rollout server + gateway nodes)
    │
    │  small harness adapter (config + shell command)
    │  + provider-API proxy (Anthropic / OpenAI / Google)
    │
    ▼
Unmodified agent harness (Codex / Claude Code / Qwen Code / Pi / ...)
    │
    │  registered as one of  ──────────────────┐
    ▼                                          │
NeMo Gym environment                           │
    (Polar as the rollout backend) ────────────┘
```

So:
- **ProRL Agent** is now legacy — the same NVIDIA-NeMo/ProRL-Agent-Server repo, but the codebase has been rewritten.
- **Polar** is the production rollout-driver layer.
- **NeMo Gym** still owns the environment catalog (84 benchmarks, 19 harnesses) — Polar registers as one of its environments.

The "must pick one" guidance in [[prorl-agent#ProRL Agent vs NeMo Gym — same family, different layer|the ProRL Agent comparison section]] is now: **use both**, with Polar as the rollout layer.

## Source code & reproduction

Same repository as ProRL Agent (the team kept the GitHub URL but rewrote the code):

| Path | Role |
| ---- | ---- |
| [`server/gateway/`](https://github.com/NVIDIA-NeMo/ProRL-Agent-Server) | Gateway node implementation — proxy, INIT/RUN/POSTRUN pools |
| `server/proxy/` | Provider-API transformers (Anthropic / OpenAI Chat / OpenAI Responses / Google) |
| `server/trajectory/` | Per-request + prefix-merging trajectory builders |
| `harnesses/` | Pre-built adapters for `claude_code`, `codex`, `gemini_cli`, `qwen_code`, `opencode`, `pi`, plus the generic shell harness |
| `trainer_integration/slime/` | Reference integration with Slime async RL trainer |

Minimal reproduction recipe — Qwen3.5-4B + Codex + GRPO + Slime, the Codex headline experiment:

```bash
# 1. Bring up Polar
docker compose -f deploy/polar.yaml up   # or apptainer/, for HPC

# 2. Configure the Codex harness adapter
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

# 3. Run training (Slime, simple GRPO)
slime train \
  --model Qwen/Qwen3.5-4B-Base \
  --dataset NovaSky-AI/SkyRL-v0-293-data \
  --rollout polar \
  --polar-endpoint http://localhost:8000 \
  --polar-harness harness.yaml \
  --trajectory-builder prefix_merging \
  --evaluator swebench_harness
```

The released SFT corpus:

```bash
huggingface-cli download nvidia/polar-swegym-pi-qwen35-122b-a10b-trajectories \
  --repo-type dataset \
  --local-dir ./polar-sft-corpus
```

## Related reading

- [[prorl-agent]] — The direct predecessor; same NVIDIA repo, replaced by Polar in May 2026. The "AgentHandler ABC plugin" architecture documented there is the design Polar supersedes; the broader rollout-as-a-service framing remains accurate.
- [[nemo-gym]] — The environment catalog that Polar registers into. NeMo Gym owns the 84-benchmark + 19-harness inventory; Polar is the rollout-driver layer that executes them.
- [[agentic-rl-overview]] — Broader landscape of agentic RL frameworks.
- [[grpo]] — The RL algorithm Polar uses in its experiments.
- [[rl-training-frameworks]] — Slime, NeMo RL, VeRL, OpenRLHF — the trainers that consume Polar rollouts.
- [[environment-design]] — Sandbox infrastructure design (Apptainer, OpenReward, ARES, Daytona); Polar inherits ProRL Agent's rootless-HPC sandbox.
- [[tool-use-rl]] — RL for tool-using agents; Polar's experiments target this directly.
- [[das-spec-rl]] — Speculative-decoding speedup for RL rollouts; complementary at the inference layer.
- [[aurora]] — The other 2026 "rollout as live system" paper (online spec-decoding draft training); Polar and Aurora attack different bottlenecks in the same rollout-architecture lineage.

## References

- Paper: Xu et al., *Polar: Agentic RL on Any Harness at Scale*, 2026-05-22. [arXiv:2605.24220](https://arxiv.org/abs/2605.24220)
- Predecessor: Zhang et al., *ProRL Agent*, 2026-03. [arXiv:2603.18815](https://arxiv.org/abs/2603.18815) — [[prorl-agent]]
- Released SFT corpus: [`nvidia/polar-swegym-pi-qwen35-122b-a10b-trajectories`](https://huggingface.co/datasets/nvidia/polar-swegym-pi-qwen35-122b-a10b-trajectories) (Apache-2.0)
- Code: [github.com/NVIDIA-NeMo/ProRL-Agent-Server](https://github.com/NVIDIA-NeMo/ProRL-Agent-Server)
- SWE-Bench Verified: Jimenez et al. [arXiv:2310.06770](https://arxiv.org/abs/2310.06770)
- SWE-Gym: Pan et al. (training environment for software-engineering agents)
- SkyRL-Agent: Cao et al. (full-stack RL for multi-turn agents); [SkyRL-v0-293-data](https://huggingface.co/datasets/NovaSky-AI/SkyRL-v0-293-data)
- Agent Lightning: Luo et al. (tracing-based agent RL with retokenization-drift discussion) — [microsoft/agent-lightning](https://github.com/microsoft/agent-lightning)
- rLLM: Tan et al. (cross-framework agent RL with tracked clients)
- Slime: Zheng et al. & Zhu et al. (Megatron training + SGLang rollout)
- PRIME-RL: Prime Intellect (async RL with stale-policy semantics)
- Harbor: Harbor Framework Team (containerized agent evaluation)
