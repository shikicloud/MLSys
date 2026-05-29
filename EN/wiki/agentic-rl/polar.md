---
title: "Polar: Agentic RL on Any Harness at Scale (the ProRL Agent successor)"
category: agentic-rl
tags: [polar, prorl-agent, nvidia, rollout-as-a-service, agentic-rl, llm-api-proxy, nemo-gym, swe-bench, paper-review]
created: 2026-05-26
updated: 2026-05-27
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

> [!question]+ Shiki — Is ProRL Agent just an HTTP layer wrapping an external rollout engine? (2026-05-27)
>
> No — common misconception. **ProRL Agent IS the rollout engine, not a wrapper for one.** It contains vLLM (LLM inference), the AgentHandler Python plugin (agent loop), and the rootless Apptainer sandbox, all in one process. The HTTP layer is the *trainer-facing* contract — it lets the trainer (veRL / NeMo-RL / slime) decouple from rollout. The trainer sends `POST /process` and ProRL Agent **actively controls** the whole rollout: starts the sandbox, runs the AgentHandler loop (deciding to search vs answer at each turn), drives vLLM, calls tools, evaluates, returns the trajectory.
>
> The same is true of Polar — it IS the rollout engine, just with a different integration boundary (LLM-API proxy instead of in-process AgentHandler plugin).
>
> "Rollout engine" ≠ "LLM engine". A rollout engine runs a *complete trajectory* (LLM forward + agent loop + tool calling + sandbox + scoring). vLLM is just the LLM forward piece inside.

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

> [!question]+ Shiki — Does Polar see tool calls? Where do tools fit? (2026-05-27)
>
> **Polar does NOT directly see tool calls.** Tools happen *entirely inside the unmodified harness* — Codex CLI's bash invocations, Claude Code's file edits, Pi's repository reads. Polar only sees what flows through the **LLM API boundary**.
>
> What this means concretely:
>
> 1. Harness (e.g. Codex) decides "I should read this file" — calls `bash` tool **internally** → Polar doesn't see this
> 2. Harness sends the file content to LLM as the next API call → Polar's proxy **intercepts**, captures token IDs, forwards to vLLM
> 3. LLM (vLLM) responds with "I should edit this line" → proxy captures token IDs, returns text response to harness
> 4. Harness executes the edit (more tool calls) → Polar doesn't see
> 5. Next API call to LLM with new state → Polar captures again
>
> Tool inputs and outputs **appear inside LLM API calls** as part of the prompt (e.g. `{"role": "tool", "content": "file contents..."}`). When Polar's `prefix_merging` reconstructs the trajectory:
>
> - Tokens the policy actually sampled → `loss_mask = 1` (trainable)
> - Tokens the harness/system injected (tool results, prompt rendering, etc.) → `loss_mask = 0` (interstitial)
>
> So tool outputs ARE masked out — but Polar derives the mask by *diffing* successive API calls (what was the previous response vs what's new in this prompt), not by being told "this is a tool call."
>
> This is structurally similar to [[search-r1#Retrieved-token loss masking — the load-bearing trick|Search-R1's retrieved-token loss masking]] but at the LLM-API layer instead of within an in-process Python plugin. Both achieve the same goal (gradients only flow through policy-sampled tokens) by very different mechanisms.

## Token-faithful trajectory reconstruction

This is the technical contribution of the paper. Polar provides two strategies in a registry; the prefix-merging one is the load-bearing one.

> [!question]+ Shiki — Is "token-faithful" the same as ProRL Agent's avoid-retokenization? (2026-05-27)
>
> **Same problem, very different implementation.** Both ProRL Agent and Polar care about preserving the exact token IDs the policy sampled, never re-tokenizing text.
>
> **ProRL Agent's solution: avoid the problem.** AgentHandler runs in the same Python process as vLLM. It calls vLLM directly, getting token IDs in and out without ever crossing a text boundary. No protocol layer means no retokenization opportunity.
>
> ```python
> # ProRL Agent's AgentHandler (in-process)
> output_token_ids = await vllm_engine.generate(input_token_ids)
> # token IDs are the native exchange format
> ```
>
> **Polar's solution: solve the problem.** The harness is a *separate process* (Codex is a binary, Claude Code is TypeScript), communicating via text-based LLM APIs. Token IDs aren't exposed in those APIs. Polar's proxy sits between:
>
> 1. Harness sends request (text) → proxy intercepts
> 2. Proxy forwards to local vLLM **with `logprobs=true`** → gets back text + token IDs + logprobs
> 3. Proxy stores token IDs in session log
> 4. Proxy returns text to harness (preserving API compatibility)
> 5. Harness has no idea token IDs were captured
>
> The harness sees **standard LLM API text responses** while Polar quietly accumulates token-faithful trajectory data. This is harder than ProRL Agent's approach but lets the harness be **anything that talks to an LLM API** — closed-source binaries, TypeScript CLIs, Go agents. That generalization is the whole point.

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

> [!question]+ Shiki — Is gateway-level async staging related to LLM prefill? (2026-05-27)
>
> **No, unrelated.** "Prefill" in LLM inference is vLLM's internal concept: ingesting a long prompt's tokens in one forward pass to populate the KV cache before decoding begins. Polar's "async staging" is at a completely different layer — it's the **orchestration of rollout pipeline stages within a gateway node**, not about LLM forward shape.
>
> The motivation: a typical SWE-Bench rollout has three cost-distinct phases:
>
> | Phase | Time | Resource |
> | ----- | ---- | -------- |
> | **INIT** — start Apptainer container, install harness, configure git repo | 30-90 s | CPU + disk |
> | **RUNNING** — harness executes the agent loop (LLM calls + tool execution) | 1-5 min | GPU (LLM) + CPU (tools) |
> | **POSTRUN** — run verifier (SWE-Bench test suite), tear down container | 30-180 s | CPU |
>
> **Serial execution would idle the GPU during INIT and POSTRUN**, wasting ~30-50% of total time. Polar's gateway runs 4 pools concurrently:
>
> - **INIT pool** initializes new sessions in the background
> - **READY buffer** holds initialized sessions waiting for a run slot
> - **RUNNING pool** executes harnesses (the GPU-active phase)
> - **POSTRUN pool** scores and tears down completed sessions
>
> Effect: GPU stays busy on LLM inference (87.7% utilization) while CPU does container management and verification in the background. **This is what gets Polar from 20.4% to 87.7% rollout GPU utilization** — pipeline-stage parallelism, not better LLM kernels.
>
> Analogy: vLLM prefill is "the chef chopping vegetables" (one dish's internal step); gateway staging is "the kitchen running prep / cooking / dishwashing stations in parallel" (whole-pipeline orchestration).

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

> [!question]+ Shiki — Why is the Codex gain (+22.6 pp) so much larger than the others? (2026-05-27)
>
> The 4-harness gain asymmetry is striking: Codex +22.6, Pi +6.2, Claude Code +4.8, Qwen Code +0.6. **A 38× range from highest to lowest**. This isn't accidental — it tells you something important about what Polar is actually doing.
>
> ### Reason 1: Codex's protocol is "foreign" to Qwen2.5-4B
>
> Codex is OpenAI's CLI, designed for the GPT-4/5 family. Its internal protocol is **calibrated to GPT-family behavior**:
>
> - **Tool-calling format**: OpenAI function-calling JSON schema. GPT models saw millions of training examples in this format.
> - **Patch submission format**: specific unified-diff markers, specific newline rules.
> - **System prompts**: phrasing and structure calibrated for GPT-style assistants.
> - **Multi-turn patterns**: when to function-call vs reason vs submit — aligned with GPT training distribution.
>
> **Qwen2.5-4B has essentially never seen this protocol in pretraining.** Qwen's instruction tuning uses Alibaba's own formats. So when the base model tries to use Codex:
> - It tries to call a tool but the JSON format is malformed → Codex parser fails → wasted turn
> - It generates a patch but the diff marker is wrong → Codex rejects → wasted turn
> - It doesn't recognize Codex's "stop here" signal → keeps generating or stops early
>
> **The 3.8 % base ≈ "model gets lucky occasionally"**, not "model knows how to use Codex." There's massive headroom for protocol learning.
>
> ### Reason 2: Qwen Code is the opposite extreme
>
> Qwen Code is built **by Alibaba for Qwen models**. Its prompts, tool schemas, and response formats are tuned to Qwen2.5's natural output style. The model picks up Qwen Code like a native speaker — 34.6 % base is the model's real coding ability shining through.
>
> Remaining +0.6 % gain reflects "fine-polishing an already-fluent protocol." **No protocol learning room left**, so the gain is tiny.
>
> ### Reason 3: Claude Code and Pi are in between
>
> Claude Code (Anthropic, calibrated for Claude) is closer to general LLM patterns than Codex is, so Qwen can semi-follow (29.8 % base) but with mismatch costs that RL fixes (+4.8 %). Pi is open-source with broad LLM compatibility in mind — works on Qwen (34.2 %) with some protocol overhead RL polishes (+6.2 %).
>
> ### The unstated insight: Codex's +22.6 is largely "protocol learning", not "coding skill"
>
> This is what the paper doesn't explicitly say but you should read between the lines:
>
> The +22.6 gain is mostly the model learning **how to talk to Codex**:
> 1. JSON tool-call formatting
> 2. Patch submission marker conventions
> 3. When to function-call vs reason
> 4. Wasting fewer turns
>
> Compare with Qwen Code's +0.6 % — on a harness the model is already fluent with, RL has to optimize the **actual coding decisions** (which file to edit, what bug to fix). That's the harder problem, and the gains reflect it.
>
> **So +22.6 pp does NOT mean "Polar made Qwen a much better coder" — it means "Polar taught Qwen to talk to a foreign agent."** Important distinction.
>
> ### What this means for adoption
>
> The Codex result demonstrates Polar's strongest claim: **you can RL-train a model to use a harness it was never designed for**. This is real and valuable. But:
>
> 1. **At 70B+ scale**, the base model has likely seen Codex / Claude Code outputs in pretraining data. Base rates won't be 3.8 % — they'll be 20-30 %+. The +22.6 dramatic gain is a **small-model phenomenon**. Expect 70B Codex gains around +5-8 pp, similar to Pi.
>
> 2. **The asymmetry IS the point** of Polar's "any harness" claim: training works on the harness the base model knows worst. But it also means the headline number oversells general capability — most of the gain is unlocking a specific protocol, not making the model fundamentally better.
>
> ### Where the paper acknowledges this
>
> §4.2: "The largest absolute gain appears in Codex, likely due to unfamiliar tool schemas. ... Codex presents an unfamiliar action protocol, context policy, and patch-submission style to a Qwen model that was not originally trained as a Codex-native policy." The paper says "unfamiliar tool schemas" but doesn't unpack the implication that **most of +22.6 is protocol adaptation, not coding improvement**.

### Critical ablation: prefix_merging vs per_request

Same model, same hardware, same topology, only the trajectory builder changes. Three training steps:

| Strategy | Trainer updates | Wall-clock | Rollout GPU util |
| -------- | --------------: | ---------: | ---------------: |
| `per_request` | 1,185 | 189.5 min | 20.4 % |
| **`prefix_merging`** | **218** | **35.2 min** | **87.7 %** |

`per_request` produces ~5× more trainer updates than `prefix_merging` does for the same physical work. The wall-clock 5.39× comes from the trainer's batched gradient computation dominating: 1185 separate trainer iterations is ~5× slower than 218 even if each iteration is cheaper per-trace.

> [!question]+ Shiki — Why do BOTH trainer-updates and wall-clock improve? Are these independent metrics? (2026-05-27)
>
> They're **highly correlated, not independent** — three numbers measuring the same underlying physics from different angles. The root cause is **5× fewer trainer triggers**, and everything else follows.
>
> ### Time-line view
>
> ```
> per_request mode (rollout GPU's view):
>   [gen trace1] [wait trainer ack] [wait weight sync] [gen trace2] [wait...] ...
>        ↑busy        ↑idle              ↑idle              ↑busy
>   1185 active-idle alternations  →  average util 20.4%
>
> prefix_merging mode:
>   [gen 5 completions back-to-back into 1 session] [wait trainer ack] [next session] ...
>        ↑long busy stretch                              ↑short idle         ↑long busy
>   218 long-active / short-idle cycles  →  average util 87.7%
> ```
>
> ### Why fewer trainer updates → faster wall-clock
>
> Each trainer update has **fixed overhead independent of batch size**:
> - Network: ship the trace from rollout worker to trainer worker
> - NCCL synchronization across data-parallel ranks
> - Optimizer state load + step
> - Weight sync back to vLLM (hybrid engine swap)
>
> All of these cost ~constant time per update, regardless of whether the update is on 1 trace or 5 traces. So **1185 updates pay 1185× the fixed overhead; 218 updates pay 218×**.
>
> Each `prefix_merging` trace IS ~5× longer (since it contains 5 completions worth of tokens). But GPU forward/backward is **sublinear** in sequence length (within reason — tensor parallelism + flash attention amortize the cost). So one update on a 5×-longer trace is much less than 5× the cost of one update on a short trace.
>
> Net: 218 long-trace updates is ~5× faster total wall-clock than 1185 short-trace updates, despite roughly equivalent total token throughput.
>
> ### Why fewer updates → higher rollout GPU utilization
>
> Each trainer update is a **synchronization point**: rollout has to wait for the trainer to finish updating weights before generating the next trace (otherwise the new trace would use stale policy, which is off-policy noise).
>
> - per_request: 1185 sync points → rollout GPU sits idle 1185 times waiting for trainer
> - prefix_merging: 218 sync points → rollout GPU sits idle 218 times
>
> 80 % idle on rollout side in per_request = the GPU is just waiting for trainer to keep up.
>
> ### The unified story
>
> Three metrics, one root cause:
>
> ```
>  5× fewer trainer triggers (218 vs 1185)
>         │
>         ├──► 5× less fixed overhead → wall-clock 189.5 → 35.2 min (5.39×)
>         ├──► 5× less network traffic between rollout / trainer
>         ├──► 5× fewer weight-sync cycles
>         └──► 5× less rollout-GPU idle time → 20.4 → 87.7 % util (4.3×)
> ```
>
> This is also why `prefix_merging` is the **default** in all Polar production runs, not just because of reward-hacking safety (next callout) but because it's straightforwardly faster.

> [!important] Per-request with outcome-reward broadcasting causes reward hacking
> When you give every `per_request` trace the same session-level outcome reward (the natural baseline), the paper observes **significant reward hacking**: request-level traces get session-level credit without proper normalization, so noisy traces get reinforced by lucky-final-patch sessions. They punt on the fix ("PRM-style credit assignment is on our roadmap"). For now, `prefix_merging` is the only safe option for outcome-reward RL.

> [!warning]+ Shiki — What "reward hacking" specifically means here, and the paper's open problem (2026-05-27)
>
> This is the most important caveat in the Polar paper. **Not classical reward hacking** (model exploits reward function loopholes) — this is **credit misassignment** that the paper observed in ablation but explicitly punted on.
>
> ### The concrete failure mode
>
> A Codex session on SWE-Bench typically makes 5-10+ LLM API calls. Suppose one session goes:
>
> | LLM call | What the agent did | Actual quality |
> | --- | --- | --- |
> | 1 | "Let me explore the repo structure" → lists files | OK |
> | 2 | "I'll open `main.py`" → opens **wrong** file | **bad — off-task** |
> | 3 | "Let me look at `utils.py`" → another wrong file | **bad — random exploration** |
> | 4 | "Wait, I should check the tests" → gets back on track | good |
> | 5 | "Found the bug, here's the patch" → submits correct patch | excellent |
>
> Verifier passes the patch → **session reward = 1**.
>
> Under `per_request` + outcome reward broadcasting, **every one of those 5 traces gets reward = 1**, including calls 2 and 3 which were objectively bad exploration. PPO sees:
>
> - Trace 2 "open wrong file" → reward 1.0 → gradient: **increase probability of this action**
> - Trace 3 "random exploration" → reward 1.0 → gradient: **increase probability of this action**
>
> Result: the model learns **"actions that appeared in successful sessions"** rather than **"actions that caused success"**. This is a **credit-assignment failure**, but it manifests as classic reward hacking — the model finds a cheap exploit:
>
> 1. Make more exploratory LLM calls (each one might land in a successful session and get reward = 1)
> 2. Each individual call quality can be lower (one bad call won't ruin the session)
> 3. Training reward looks great; **test-time task completion drops**
>
> ### Why this is a credit-assignment issue
>
> RL should learn "actions that *cause* high reward". Outcome-broadcast `per_request` makes the signal "actions that *correlate* with high reward". Spurious correlations get reinforced.
>
> Compare with `prefix_merging`: the entire session becomes **one** trace, reward sits at the end token, GAE backpropagates it with $\gamma < 1$, so early tokens get less credit. A single "open wrong file" action inside a longer trajectory doesn't get an isolated reward = 1.
>
> ### Did the paper solve it?
>
> **No — explicitly punted to future work**. Paper §4.1 final paragraph:
>
> > "We also tried per_request with outcome-reward broadcasting to every emitted trace, but observed significant reward hacking. The issue is noisy credit assignment: request-level traces can receive session-level credit without proper session normalization or an advanced process reward model. **Those mechanisms are outside the scope of this work**, but providing examples and tools for session normalization and PRM-style credit assignment is on our roadmap."
>
> ### Workaround they actually ship
>
> **Use `prefix_merging` instead of `per_request`**. This is why prefix_merging is the default in all main experiments — not just for the 5× wall-clock speedup, but because per_request isn't safe to train with on outcome-only reward.
>
> ### Where the workaround leaks
>
> `prefix_merging` only works when the harness maintains **append-only conversation history** within a chain. When the harness does:
>
> - **Context compaction** (rewrite history to summarize old turns)
> - **Subagent spawning** (new conversation thread)
> - **Prompt rewriting** (any heavy mutation of the prompt context)
>
> ...the prefix check ($p_{m+1}[:|p_m|] = p_m$) fails, and Polar starts a **new chain**. Each new chain reverts to per-request-like fragmentation for that section. **In context-rewriting harnesses, you get a hybrid mode where parts of the session are clean and parts are fragmented** — reward hacking creeps back in proportionally.
>
> ### What the "real fix" would look like
>
> Three directions the paper gestures at but doesn't implement:
>
> 1. **Session-level reward normalization** — instead of broadcasting reward=1 to all 10 traces, normalize: each gets 1/10. Or discount by recency: $r_i = R \cdot \gamma^{n-1-i}$ so later traces get more credit. Cheap, no new models, but assumes "later = more important" which isn't always true.
> 2. **Process Reward Models (PRM)** — train a separate model to score each step. Mature in math ([MathShepherd](https://arxiv.org/abs/2312.08935), [OmegaPRM](https://arxiv.org/abs/2406.06592), [PRIME](https://arxiv.org/abs/2502.01456)) but immature for code/agentic. Expensive: needs labeled per-step data + adds an extra model to training and serving.
> 3. **Token-level advantage on merged trajectories** — what `prefix_merging` already does well. The job is making this robust to context compaction, subagent boundaries, and prompt rewriting.
>
> ### Why this matters for adoption
>
> Polar's "any harness" claim is **structurally limited by this open problem**. For harnesses with clean append-only conversation (Codex, simple agents) → `prefix_merging` works → no issue. For sophisticated context-managing harnesses (some Claude Code workflows, modern long-horizon agents) → `prefix_merging` degrades → reward hacking returns. The deeper an agent's internal context management, the worse Polar handles its RL training. This is the **open frontier** of Polar's research agenda.
>
> Reference: contrast with [[search-r1]], which side-stepped this entirely by maintaining one continuous trajectory in `generation.py:run_llm_loop` and applying GAE token-level — but it pays for this with the "AgentHandler-per-harness" cost that Polar specifically tried to eliminate. **There is a tension between "universal harness support" and "clean credit assignment"; nobody has solved both yet.**

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

## The three-layer agentic-RL stack — Polar / ProRL Agent / NeMo Gym

This is the question people get most confused about. Three NVIDIA projects (ProRL Agent, Polar, NeMo Gym) plus the trainer (veRL / NeMo-RL / slime) work together in a **3-layer architecture**. Each layer answers a different question.

### One-sentence positioning

| Layer | System | Answers the question |
| ----- | ------ | -------------------- |
| **Trainer** | veRL / NeMo-RL / slime | "How do I use a trajectory to update the policy?" |
| **Rollout-driver** | ProRL Agent → Polar | "How do I run the agent against a task and capture a trajectory?" |
| **Environment catalog** | NeMo Gym | "What's the task and how do I score it?" |

ProRL Agent and Polar are **the same layer, new and old versions** (Polar supersedes ProRL Agent, same NVIDIA team, same repo). NeMo Gym is **a different layer**, providing inputs to the rollout-driver.

### The stack diagram

```
┌──────────────────────────────────────────────────────────┐
│        Trainer (veRL / slime / NeMo-RL / OpenRLHF)        │
│   ─ PPO/GRPO ─ gradient update ─ distributed training    │
└──────────────────────────┬───────────────────────────────┘
                           │
                           │  HTTP rollout request:
                           │  "run task T with policy π"
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│      Rollout-driver (ProRL Agent → Polar, May 2026)        │
│  ─ spin sandbox ─ run harness ─ capture LLM API ─        │
│  ─ build trajectory ─ run verifier ─ compute reward       │
└────┬────────────────────┬───────────────────────────────┘
     │                    │
     │ needs task         │ harness invokes LLM
     │ needs runtime      │ (proxy intercepts)
     │ needs verifier     │
     ▼                    ▼
┌─────────────────┐  ┌────────────────────────────────┐
│   NeMo Gym      │  │  vLLM (training policy π)         │
│  ─ 84 benchmarks│  │  + reference policy                │
│  ─ runtimes     │  │  + critic (PPO only)               │
│  ─ verifiers    │  │  ─ hybrid engine with FSDP actor  │
│  ─ data splits  │  │                                    │
└─────────────────┘  └────────────────────────────────┘
```

**Note**: NeMo Gym sits *alongside* the rollout-driver, providing inputs — not above or below. The trainer doesn't talk to NeMo Gym directly; Polar reads NeMo Gym's task/runtime/verifier specs as inputs to do its job.

### History

```
2026-03  ProRL Agent       ┐
                            ├─ Independent NVIDIA projects, no public adapter
2026-03  NeMo Gym          ┘     (the original "must pick one" gap)

2026-05  Polar              ─ Supersedes ProRL Agent (same NVIDIA repo)
         + registered as a NeMo Gym environment  ─ The bridge ships
```

**The key event**: in March 2026 ProRL Agent and NeMo Gym shipped as parallel projects with no formal connection. In May 2026 Polar replaced ProRL Agent **and** registered as a NeMo Gym environment, formally connecting the two layers. This is the moment NVIDIA's agentic-RL stack consolidated.

### What each does, standalone

| Used alone | Can do | Can't do |
| ---------- | ------ | -------- |
| **NeMo Gym only** | Benchmark eval (run existing models against 84 tasks); provide tasks to other RL frameworks | Can't train (no trainer); harness integration left to user |
| **ProRL Agent / Polar only** | Rollout service (given a task, run harness + capture trajectory); offline SFT data generation | Can't define tasks (you bring them); can't train |
| **Trainer only** | PPO/GRPO on simple RL tasks; classic RLHF | Can't do agentic (no multi-turn rollout); no env catalog |

**Only all three together = a complete production agentic-RL training pipeline.**

### A concrete training run — who does what at each step

Suppose you want to "train Qwen2.5-7B on SWE-Bench Verified with GRPO":

```
1. User: ./train_swebench.sh
              │
              ▼
2. Trainer (slime) starts:
   ─ loads SWE-Bench Verified task list ←── from NeMo Gym
   ─ loads Apptainer runtime spec      ←── from NeMo Gym
   ─ knows verifier is swebench_harness ←── from NeMo Gym
              │
              ▼ each RL step:
3. Trainer picks a batch of task instances
              │
              ▼ HTTP POST /process(task_batch)
4. Polar gateway receives the request:
   ─ INIT pool: spin N Apptainer containers (using NeMo Gym's image spec)
   ─ install Codex CLI / Claude Code / Pi in each container
   ─ inject task's git repo + problem statement
              │
              ▼
5. Polar RUN pool: launch Codex CLI process in each container
              │
              ▼
6. Codex CLI runs its OWN agent loop:
   ─ reads files (bash tool)       ←── Polar can't see this
   ─ calls LLM API ────────────────┐
                                   │
                                   ▼ Polar's API proxy intercepts
                               ┌────────────────────────────┐
                               │ proxy forwards to local vLLM │
                               │ captures token IDs + logprobs│
                               │ returns text response in     │
                               │ OpenAI/Anthropic format       │
                               └────────────────────────────┘
   ─ edits files (edit tool)       ←── Polar can't see this
   ─ ... multiple turns ...
   ─ submits patch
              │
              ▼
7. Polar POSTRUN pool:
   ─ runs swebench_harness (NeMo Gym's verifier)
   ─ obtains reward (0 or 1)
   ─ runs prefix_merging to reconstruct trajectory (token IDs + loss_mask)
              │
              ▼ HTTP response: (token_ids, logprobs, loss_mask, reward)
8. Trainer collects all trajectories, builds batch
              │
              ▼
9. Trainer computes GRPO advantages, PPO loss, updates Qwen weights
              │
              ▼
10. New Qwen weights sync to vLLM (hybrid engine swap)
              │
              ▼
(back to step 3 for the next RL step)
```

**Who does what at each step**:

| Step | NeMo Gym | Polar | Trainer | vLLM |
| ---- | -------- | ----- | ------- | ---- |
| 1-2 setup | task list + runtime spec + verifier | – | loads from NeMo Gym | – |
| 3 batch | – | – | ✓ | – |
| 4 sandbox | runtime spec | ✓ spin container | – | – |
| 5 harness | – | ✓ | – | – |
| 6 agent loop | – | proxy captures LLM API | – | ✓ LLM forward |
| 7 score | provides verifier | ✓ runs verifier | – | – |
| 8-9 update | – | – | ✓ | – |
| 10 sync | – | – | – | ✓ |

Three independent systems, three clean concerns.

### "Same family, different layer" — concretely

- **ProRL Agent ↔ Polar**: **Same layer, version upgrade**. Same NVIDIA team, same GitHub repo (`NVIDIA-NeMo/ProRL-Agent-Server`), same architectural philosophy (rollout-as-a-service). Polar replaces ProRL Agent's Python `AgentHandler` plugin with an LLM-API proxy — that's the only fundamental change.

- **Polar ↔ NeMo Gym**: **Different layers, collaborating**. One is a rollout-driver, the other is an environment catalog. Polar registering as a NeMo Gym environment is what makes "trainer accesses Polar through NeMo Gym" a standard path.

A web-stack analogy:
- ProRL Agent → Polar is like **Apache → nginx** (same-layer web server, new-vs-old generation)
- Polar ↔ NeMo Gym is like **nginx ↔ PostgreSQL** (web server ↔ database, different layers, mutually dependent)

### Common misconception: "NeMo Gym is an agent framework like SWE-agent"

It's not. NeMo Gym is the *catalog*; SWE-agent / Codex / Claude Code are *agents* (harnesses). NeMo Gym **references** agent harnesses in its 19-harness inventory but doesn't itself implement an agent's reasoning loop.

| Confused with | Actually | Lives at layer |
| ------------- | -------- | -------------- |
| NeMo Gym = SWE-agent? | No — SWE-agent is a *harness* (one of NeMo Gym's 19 referenced harnesses). NeMo Gym is the catalog | Harness ≠ Environment |
| NeMo Gym = Codex / Claude Code? | No — those are *harnesses* (potentially registered with NeMo Gym, definitely usable inside Polar) | Harness layer |
| NeMo Gym = trainer? | No — NeMo-RL is a trainer (different name, same NVIDIA family) | Trainer ≠ Environment |
| NeMo Gym = vLLM? | No — vLLM is LLM inference | LLM-engine layer |

NeMo Gym is **none of the above** — it's the connective tissue that says "here are 84 tasks with their Apptainer images and verifiers, ready to be consumed by any rollout-driver or trainer."

### "Must pick one" guidance — updated

Old answer (from when ProRL Agent and NeMo Gym were unconnected, [[prorl-agent#ProRL Agent vs NeMo Gym — same family, different layer|see prorl-agent]]): pick based on what you valued more — token-level off-policy fidelity vs benchmark catalog breadth.

**New answer (May 2026 onward)**: **use both**. Polar fills the rollout-driver layer, NeMo Gym fills the environment-catalog layer, and Polar is registered as a NeMo Gym environment so they connect naturally.

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
