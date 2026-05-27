---
title: "Agentic RL Foundations — onboarding hub"
category: agentic-rl
tags: [agentic-rl, hub, foundations, onboarding, family-overview]
created: 2026-05-26
updated: 2026-05-26
status: mature
---

# Agentic RL Foundations — onboarding hub

> [!abstract]+ What this page is
> A navigation hub for entering agentic-RL research. Walks through the canonical papers and infrastructure in the order most useful for a newcomer, with a 4-phase reading path, mini-summaries of every important paper, and pointers to dedicated wiki pages.
>
> **Use this page** if you're new to agentic RL and want a curated path through the field. For specific deep dives, follow the wiki links — every paper mentioned here has its own page (or will, soon).

> [!info] Snapshot — what is the state of agentic RL as of May 2026?
> - **Algorithm side**: PPO and GRPO are still the workhorse RL algos; no fundamentally new algorithm has displaced them. Variants (DAPO, RLOO, Dr.GRPO, KDRL) tune around the edges.
> - **Architecture side**: The field consolidated around "rollout-as-a-service" + "harness-as-blackbox" patterns. [[polar|Polar]] (NVIDIA, May 2026) is the current state of the art.
> - **Frontier tasks**: SWE-Bench (code), WebArena (browser), OSWorld (operating system), GAIA (general agent). Multi-tool composition, long-horizon planning, computer use.
> - **Frontier method**: Pure outcome-reward RL still works at small-medium scale; process rewards (PRMs) and LLM-as-judge are the open question at frontier scale.

## What is agentic RL

**Agentic RL** is reinforcement-learning fine-tuning of LLMs where the policy isn't just generating text but **interacting with a stateful environment over multiple turns**. The environment can be:

- A search engine ([[search-r1]])
- A code execution sandbox (SWE agents)
- A web browser (BrowserGym / Mind2Web)
- An operating system (OSWorld)
- A multi-tool composition (calculator + search + code)

The standard ingredients vs vanilla RLHF:

| Aspect | Vanilla RLHF | Agentic RL |
| ------ | ------------ | ---------- |
| Rollout | Single LLM forward pass | Multi-turn (LLM → tool call → observation → LLM → ...) |
| Environment | None (LLM generates conditional on prompt) | Real, often stateful (sandbox, DB, browser) |
| Trajectory composition | Pure model-generated tokens | LLM tokens + environment-injected tokens (interleaved) |
| Reward | RM scalar or rule-based | Usually sparse outcome reward (correct? Y/N) |
| Loss masking | Pad-only | **Must mask environment-injected tokens** |
| Infrastructure | Trainer + rollout (vLLM) | Trainer + rollout + environment server(s) + sandbox |

The "agentic" qualifier flags that the **gradient must flow correctly across the LLM/environment boundary**, which is what makes the field interesting (and harder than RLHF).

## The three core challenges

Every agentic-RL paper grapples with some subset of these. Recognize them to read papers fluently:

1. **Heterogeneous trajectory composition**. Rollouts contain LLM tokens (sampled from the policy) and environment tokens (injected by tools / retrievers / sandbox). PPO loss naively applied to both teaches the model to imitate environment output, destabilizing training. The fix is **retrieved-token loss masking** (or its generalizations).
2. **Sparse outcome rewards across long horizons**. A 10-turn 8K-token trajectory with one bit of reward at the end. PPO's value function (or GRPO's group baseline) has to do all the credit-assignment work. This breaks at long horizons.
3. **Environment plumbing scales differently than training**. Rollout is I/O-bound (containers spawn, network calls, tool latency), training is GPU-bound. Coupling them in one process wastes compute and limits scale. The fix is **service-oriented architecture** (rollout-as-a-service, ProRL Agent / Polar).

## Recommended reading path

A 4-week curriculum for someone who knows LLMs and a little bit of PPO but is new to agentic RL.

### Phase 1 — RL foundations (1 week)

Skip if you already know PPO/GRPO. Otherwise read in this order:

1. **[[ppo-for-llm]]** — How PPO is adapted to LLM token-level optimization. Read for: 4-model architecture (actor/critic/RM/ref), GAE, clipped objective, KL penalty.
2. **[[grpo]]** — The "PPO without critic" variant DeepSeek made famous. Read for: group-mean baseline, why no value function, when GRPO wins vs loses.
3. **[[on-policy-distillation]]** (Preliminaries section) — Friendly explanations of: KL divergence, forward vs reverse KL, on-policy vs off-policy, credit assignment, value head. **The §Preliminaries section is the best 20-minute primer on these concepts in the wiki**.

By end of Phase 1: you can explain PPO loss, GAE, GRPO, KL penalty, the difference between PPO-style (KL-in-reward) and GRPO-style (KL-as-loss) KL handling.

### Phase 2 — The DeepSeek-R1 lineage (3-5 days)

Read DeepSeek-R1 / R1-Zero ([arXiv:2501.12948](https://arxiv.org/abs/2501.12948)) carefully. Key takeaways:

- Pure RL (no SFT data) can elicit complex reasoning capabilities
- Outcome-only reward (correct answer = 1, else = 0) is sufficient on math/code tasks
- "Aha moment" / self-reflection emergence is real
- GRPO + outcome reward is the simplest working recipe

This is the **conceptual foundation** of agentic RL. Search-R1 is R1-Zero extended from pure reasoning to tool calling.

### Phase 3 — Entry paper: Search-R1 (1 week)

The canonical agentic-RL entry-point paper.

1. **Read [[search-r1]]** (the paper review). Understand: the multi-turn rollout protocol, retrieved-token loss masking, outcome-only EM reward, the PPO-vs-GRPO surprise result, the emergence plots (Fig 2c/d).
2. **Read [[search-r1-codebase-walkthrough]]** (the code walkthrough). Understand: how `generation.py` orchestrates multi-turn rollout, how `info_mask` becomes `loss_mask`, how veRL plugs it all together.
3. **Run the reference code** — clone the repo, build the FAISS index over Wikipedia, train Qwen2.5-3B for 200-500 steps. Watch the response-length and search-call-count curves emerge.
4. **Run the `state_masking=false` ablation** — see retrieved-token masking become necessary in practice.

By end of Phase 3: you understand the **agentic-RL training-loop pattern** end-to-end and have hands-on experience.

### Phase 4 — Production infrastructure (1 week)

The infrastructure that scales Phase 3's pattern to real deployment.

1. **[[prorl-agent]]** — The first "rollout-as-a-service" agentic-RL system. NVIDIA, March 2026. Read for: HTTP `POST /process` contract, INIT/RUN/EVAL pipeline, rootless HPC sandbox.
2. **[[polar]]** — The successor to ProRL Agent (same repo, same team, May 2026). Read for: LLM-API proxy paradigm (treat harness as black box), token-faithful prefix merging, why it generalizes to Codex / Claude Code / Qwen Code / Pi harnesses.
3. **[[nemo-gym]]** — NVIDIA's environment catalog (84 benchmarks, 19 agent harnesses). Read for: three-server architecture (resources / model / agent), Apptainer sandbox, how it connects to trainers.
4. **The relationship**: read [[prorl-agent#ProRL Agent vs NeMo Gym — same family, different layer|the ProRL-vs-NeMo-Gym section]] to understand which layer each fills, and read [[polar#How this changes the ProRL Agent vs NeMo Gym picture|Polar's section]] to see how the gap was bridged.

By end of Phase 4: you understand how agentic RL works at production scale — what the layer boundaries are, who provides which service, where to extend if you want to add your own.

### Phase 5 — Frontiers (open-ended)

Once you have the foundation, follow whichever direction interests you:

**Multi-tool / general agents** — extending beyond single-tool retrieval:
- ToolRL — multi-tool extension of Search-R1 pattern
- ReTool — same lineage, code tools
- Agent Lightning (Microsoft) — tracing-based agent RL
- rLLM — cross-framework agent RL

**Browser / OS agents** — harder environments:
- WebGPT-RL — browser agents trained with RL
- OSWorld — operating-system agents
- BrowserGym / Mind2Web

**Reward design** — past outcome-only:
- Process Reward Models (PRM) for math
- LLM-as-judge for open-ended tasks
- KDRL — combining RL with on-policy distillation

**Inference-time agentic** — bridging training and serving:
- [[das-spec-rl]] — speculative decoding for RL rollouts
- [[aurora]] — online spec-decoding training framed as RL

**Long-horizon / open-ended** — beyond benchmarks:
- DeepResearcher — multi-source web research
- Computer-use agents (CUA-RL)
- Long-context agents (100K+ token trajectories)

## Canonical references — short summaries

### The foundational papers (read these first)

- **DeepSeek-R1 / R1-Zero** (DeepSeek, Jan 2025, [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)) — Pure outcome-only RL elicits complex reasoning. The conceptual ancestor of agentic RL. No dedicated wiki page yet (covered in [[grpo]] context).
- **[[search-r1]]** (UIUC + UMass + Google, COLM 2025, arXiv:2503.09516) — R1-Zero extended to tool use. **The canonical entry-point paper.** Multi-turn rollout, retrieved-token loss masking, outcome-only EM reward.
- **[[grpo]]** — DeepSeek's PPO-without-critic. Most widely-used algo in 2025-26 agentic RL.
- **[[ppo-for-llm]]** — The foundational algorithm everything else specializes from.

### Infrastructure papers (read after foundation)

- **[[prorl-agent]]** (NVIDIA, March 2026, arXiv:2603.18815) — First "rollout-as-a-service" agentic-RL framework. Service-oriented design, rootless HPC sandbox, token-in/token-out wire protocol. Superseded by Polar in May 2026.
- **[[polar]]** (NVIDIA, May 2026, arXiv:2605.24220) — Successor to ProRL Agent. LLM-API proxy paradigm; trains *any unmodified harness* (Codex, Claude Code, Qwen Code, Pi). Registered as a NeMo Gym environment.
- **[[nemo-gym]]** (NVIDIA, 2026) — Environment-catalog framework. 84 benchmarks, 19 agent harnesses, three-server architecture.
- **[[rl-training-frameworks]]** — veRL, OpenRLHF, TRL — the underlying RL framework landscape.

### Adjacent / supporting papers

- **[[on-policy-distillation]]** — The non-RL cousin of agentic RL. When a teacher exists, OPD provides dense per-token signal without the credit-assignment burden.
- **[[das-spec-rl]]** — Distribution-Aware Speculative decoding for RL rollouts. Speeds up training-time rollouts 1.5-2×.
- **[[aurora]]** — Online speculative-decoding draft training framed as agentic RL on live serving traffic.
- **[[ring-attention]]** / **[[deepspeed-ulysses]]** — Long-context attention parallelism, relevant for long-horizon agentic rollouts.

### Algorithms covered in the wiki

- **[[grpo]]** — Group Relative Policy Optimization.
- **[[ppo-for-llm]]** — Proximal Policy Optimization for LLMs.
- **[[dpo]]** — Direct Preference Optimization (less relevant to agentic RL but useful comparison).
- **[[on-policy-distillation]]** — OPD family.
- **[[reward-modeling]]** — How reward models are built and where they fail.

## Common confusions (FAQ)

> [!question] Q: Is Search-R1 still the state of the art?
>
> No — but it's still the best **entry point**. As of May 2026 the SOTA in agentic-RL infrastructure is [[polar|Polar]]; the SOTA in trained models depends on the task (DeepSeek-V4 for general agents, Nemotron-Cascade 2 for competition math/code). Search-R1's role is teaching the foundations cleanly.

> [!question] Q: PPO or GRPO for agentic RL?
>
> Depends on the task:
> - **Short-horizon, single-answer (math, code)**: GRPO often wins (DeepSeek-R1, GRPO paper). Group-mean baseline is fine when trajectories are short.
> - **Multi-turn, long-horizon (Search-R1, agentic tasks)**: PPO often more stable (Search-R1 Table 3). Value function helps with long-trajectory credit assignment.
> - **When in doubt**: PPO is more conservative; GRPO is faster to iterate. Try both in your specific setting.

> [!question] Q: Do I need GPUs the size of NVIDIA's research cluster?
>
> No. Search-R1 runs on a single 8×H100 / 8×A100 node for Qwen2.5-3B in ~2 days. The smallest interesting agentic-RL run is **single 8×A100 / 8×H100 + 256GB RAM + 2TB SSD**.

> [!question] Q: When do I use OPD instead of RL?
>
> When you have a stronger teacher model and want to compress its behavior into a smaller student. [[on-policy-distillation|OPD]] is much more sample-efficient than RL (10× less compute) but capped at the teacher's ceiling. RL is unbounded but expensive. **Hybrid (KDRL, dGRPO) is becoming the default for production**.

> [!question] Q: What is "retrieved-token loss masking" really doing?
>
> It tells the optimizer "these tokens were injected by the environment, not generated by the policy — don't compute gradients on them, and don't compute KL on them either." Without this, PPO loss on environment tokens trains the model to imitate retrieved content (wrong behavior), and KL on environment tokens compares against meaningless reference distributions. See [[search-r1#Retrieved-token loss masking — the load-bearing trick|Search-R1's section]] for the math, and [[search-r1-codebase-walkthrough#4.5 ★ THE function: \`_info_masked_concatenate_with_padding\`|the code walkthrough]] for the implementation.

> [!question] Q: How is agentic RL different from RLHF?
>
> RLHF is **single-turn** (one prompt, one response) with a **learned reward model**. Agentic RL is **multi-turn** (LLM ↔ environment loop) with **rule-based or environment-derived reward**. The infrastructure overlap is large (same PPO, same actor-critic split, same KL-to-reference) but the rollout phase is fundamentally different.

> [!question] Q: What's the most important paper to read first if I only have time for one?
>
> **[[search-r1]]**. It teaches the protocol, the math, the ablation, the emergence pattern, and the codebase architecture in one shot. Everything else (Polar, ToolRL, ReSearch, etc.) is variations on its theme.

## Open research directions

What's actively being worked on as of mid-2026:

1. **Long-horizon agentic RL** (10+ turns, 32K+ tokens) — token budget, KV cache management, reward sparsity at extreme horizons. Open: how do we get useful gradients past horizon ~20?
2. **Multi-tool composition** — Search-R1 single-tool extended to 3-10 tools (search + calculator + code + browser + ...). Open: how does the model learn the right tool at the right step?
3. **Process reward models (PRMs)** — replacing outcome-only with step-level supervision. Open: how do you train reliable PRMs without infinite labeled data?
4. **Self-improvement loops** — model generates rollouts, judges them with itself, trains on its own judgments. Open: how to avoid drift / reward hacking?
5. **Computer-use agents** — visual + text agents that operate desktop applications (OSWorld, Anthropic Computer Use). Open: how to bridge vision-language models to RL training stacks.
6. **Cost-aware rollouts** — production search has $$/latency. Open: how to train models that are parsimonious with tool calls.
7. **Trainer-rollout disaggregation at scale** — [[polar]] is the current frontier. Open: how does this scale to 100B+ frontier models?
8. **Off-policy correction in multi-turn settings** — environment state changes, model versions update, how do you reuse old rollouts? Open: importance-weighted multi-turn RL.
9. **Hybrid OPD + RL** — KDRL, dGRPO showed early gains. Open: what's the cleanest formulation?

## Related reading (the broader landscape)

- [[agentic-rl-overview]] — Higher-level survey of agentic RL (older page, more conceptual)
- [[tool-use-rl]] — Tool-use RL specifically
- [[multi-step-reasoning-rl]] — Long-horizon reasoning
- [[environment-design]] — How to design good RL environments
- [[rl-training-frameworks]] — veRL / OpenRLHF / TRL / NeMo-RL
- [[rlhf-overview]] — RLHF basics, the predecessor

## Quick-reference: which page covers what

| You want to learn | Read |
| ----------------- | ---- |
| **Whole field at a glance** | This page |
| **What Search-R1 does and why** | [[search-r1]] |
| **Search-R1 code line-by-line** | [[search-r1-codebase-walkthrough]] |
| **How PPO works on LLMs** | [[ppo-for-llm]] |
| **Why GRPO replaces PPO** | [[grpo]] |
| **KL divergence, on-policy, credit assignment** | [[on-policy-distillation]] Preliminaries section |
| **Production rollout architecture** | [[polar]] (current SOTA) or [[prorl-agent]] (predecessor) |
| **What environments exist** | [[nemo-gym]] |
| **Which framework to pick** | [[rl-training-frameworks]] |
| **Speed up rollouts** | [[das-spec-rl]] |
| **Avoid RL entirely (have a teacher?)** | [[on-policy-distillation]] |

---

If anything's missing from this hub, the wiki is open — see something not linked or not yet a page, suggest it.
