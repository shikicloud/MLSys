---
title: "Aurora: Online Speculative Decoding Training via RL"
category: llm-inference
tags: [speculative-decoding, online-learning, reinforcement-learning, sglang, draft-model, together-ai, paper-review]
created: 2026-05-20
updated: 2026-05-21
status: growing
paper: arXiv:2602.06932
code: https://github.com/togethercomputer/aurora
---

# Aurora: Online Speculative Decoding Training via RL

> [!info] Paper metadata
> - **Paper**: [arXiv:2602.06932](https://arxiv.org/abs/2602.06932) — *Aurora: When RL Meets Adaptive Speculative Training*
> - **Code**: [github.com/togethercomputer/aurora](https://github.com/togethercomputer/aurora)
> - **Models**: Qwen3-Coder Spec, MiniMax M2.1/M2.5 Spec on Hugging Face
> - **Authors**: Junxiong Wang*, Fengxiang Bie*, Jisen Li, Zhongzhu Zhou, Yinghui Liu, Yubo Wang, Avner May, Sri Yamamatra, Tri Dao, Percy Liang, Ce Zhang, Ben Athiwaratkun, Shuaiwen Leon Song, Chenfeng Xu, Xiaoxia Wu
> - **Affiliation**: Together AI (primary), with Stanford, CMU collaborators
> - **Venue**: ICML 2026
> - **Project page**: https://aurora-spec-ai.github.io/

---

## Summary (read this if you have 2 minutes)

**What it is.** Aurora (Together AI, ICML 2026) is a unified training-serving framework that turns speculative-decoding draft-model training into an **online RL problem** running *during* live serving. An SGLang inference server streams every accepted and rejected token proposal to a distributed buffer; an asynchronous training server learns from that buffer and hot-swaps improved draft-model weights back into the serving instance without dropping traffic.

**The one idea.** Treat the draft model as a **policy** trained by online RL against the verifier's accept/reject signal — *the same signal the verifier already produces for free on every request*. Three pieces hold it up:

1. **Async RL loop** — Inference Server and Training Server share a data buffer + hot-swap RPC, fully decoupled in time.
2. **Two-term loss with Discard Sampling** — accept tokens pull `pdraft` toward `ptarget`; rejected tokens *push it away* (a KL term most online-distillation methods skip).
3. **Tree Attention** — a custom attention mask that lets one batched forward/backward cover all accepted *and* rejected branches in the speculation tree.

Remove any one and the draft either lags serving, learns only positives, or can't train fast enough to keep up.

**Headline result.** Two consequences fall out:

| Setting | Throughput vs no spec | Acc. Length (BS=8) |
| ------- | --------------------- | ------------------ |
| **Day-0** cold-start, Qwen3-Coder-Next (FP8) | **1.21×** | 3.0 |
| **Day-0** cold-start, MiniMax M2.1 | **1.45×** | 2.8 |
| Adapted vs **static** speculator under domain shift | **1.25×** over static | recovers in ~10K req |

The Day-0 numbers are from an **untrained** draft that only saw production traffic — no offline pretraining. The 1.25× over a static-trained competitor *after* a distribution shift is the strongest claim: Aurora keeps improving while static drafts decay.

**Why it matters.**

- **Eliminates draft pretraining.** Multi-day offline draft pretraining is a real cost on every new target model — Aurora retires that line item.
- **Self-healing under distribution shift.** Production traffic mix changes; static drafts decay; Aurora compounds.
- **Uses signal the verifier already produces.** Every existing speculative-decoding deployment is throwing away the rejection signal. Aurora picks it up.
- **12-month prediction.** vLLM and TRT-LLM ship analogous online-spec features; the "draft pretraining as a service" market gets squeezed.

---

# Depth (drill-down starts here)

The summary above is the executive layer. Everything below is for the careful reader who wants full architecture and code-level detail.

## Background: why speculative-decoding training is a serving bottleneck

[[speculative-decoding|Speculative decoding]] gives 2–3× inference throughput by letting a small *draft model* propose K tokens that the large *target model* verifies in one parallel pass. The throughput gain is gated by **acceptance length (AL)** — how many of the K proposed tokens the target accepts. AL depends critically on **distribution match** between draft and target outputs.

In production, this match is hard to maintain:

1. **Offline-trained drafts go stale.** Drafts pretrained on a generic corpus drift away from the target's behavior on production prompts (coding, agent traces, long-context RAG). EAGLE-3 and similar approaches spend days of GPU time pretraining a draft, and re-train when the target swaps.
2. **Distribution shifts mid-deployment.** A traffic mix that's 40 % code one week and 70 % agent traces the next breaks AL on the old draft.
3. **No feedback loop.** The serving stack has the data needed to fix this — the verifier knows exactly which tokens were accepted and which were rejected — but it throws that signal away.

[[das-spec-rl|DAS]] partially addresses this for *RL training* (where rollouts are the workload), by training a draft online against the rollout-policy distribution. Aurora generalizes the idea to **arbitrary production inference traffic**.

| System | Draft training | Adapts to shift | Workload focus |
| ------ | -------------- | --------------- | -------------- |
| EAGLE-3 / vanilla SD | offline, static after deploy | no | general inference |
| Online Speculative Decoding (2023) | online KD on accept only | partial | general inference |
| [[das-spec-rl\|DAS]] | online RL | yes | RL training rollouts |
| **Aurora** | **online RL, accept + reject** | **yes** | **production inference** |

## Three components in detail

Aurora's framework consists of two coupled servers connected by a shared data buffer; the loss has two terms; and the kernel makes the tree-shaped training tractable. The paper figure below shows the high-level placement.

![Aurora unified training-serving framework, paper Fig. 1](EN/wiki/llm-inference/aurora-figs/system-architecture.png)

The Inference Server hosts a fixed target (verifier) and a hot-swappable draft (speculator), runs the speculative-decoding loop, and streams *all* accept/reject branches to a distributed Data Buffer. The Training Server pulls batches asynchronously, runs the two-term loss, and pushes a new speculator back via off-policy update — without ever pausing serving.

### Component 1 — Async RL loop with hot-swap weights

The two servers run continuously. The inference side never pauses for training; the training side never blocks on a forward pass for serving. Weight hot-swap happens via GPU-aware RPC — the draft's parameter buffer is updated in-place while live requests are in flight.

The minimal architecture sketch:

```
┌─────────────────────────────────────┐    ┌─────────────────────────────┐
│   SGLang Inference Server           │    │   Async Training Server     │
│                                     │    │                             │
│   target (verifier) + draft π       │    │   pull (accept, reject)     │
│   → verify → emit tokens            │    │   → two-term loss           │
│   → log (Q, accepted, rejected)     ├───►│   → update π                │
│                                     │    │   → push new draft          │
│   ◄── hot-swap draft weights ───────┤    │                             │
└─────────────────────────────────────┘    └─────────────────────────────┘
```

**Day-0 serving.** Because training runs *during* serving, the draft can start *uninitialized* and improve in flight. The paper makes this concrete: AL converges to useful values within hours of serving on a cold-start speculator, not days of offline pretraining.

**Hot-swap is a real engineering ask.** Aurora uses `torch.distributed.rpc` over TensorPipe for GPU-direct transfers, plus expandable CUDA memory segments to prevent fragmentation. The training server holds a thread-safe transmitted-data cache, so backward passes happen on the previously transmitted (stable) micro-batch, not the in-flight one.

### Component 2 — Two-term loss: Discard Sampling on rejected tokens

For each speculative step, the verifier partitions the K proposed tokens into an *accepted prefix* and a *rejected suffix*. Aurora trains the draft on both:

- **Acceptance term** — KL `p_target || p_draft` on accepted tokens. Standard imitation: pull `p_draft` toward what the target chose.
- **Rejection term** — a KL on the *rejected* branches, via what the authors call **Discard Sampling**. The intuition: if the target rejected a branch, the draft should put less mass there next time, even if the target hasn't explicitly chosen an alternative.

The combined objective:

$$
\mathcal{L} = \mathbb{E}_{x \sim p_{\text{accept}}} \left[\, \mathrm{KL}(p_{\text{target}} \,\|\, p_{\text{draft}}) \,\right] + \lambda_{\text{discard}} \, \mathbb{E}_{x \sim p_{\text{discard}}} \left[\, \mathrm{KL}(p_{\text{target}} \,\|\, p_{\text{draft}}) \,\right]
$$

Naïve online distillation only learns from accepted tokens (positive examples). Aurora makes the rejected branches a first-class learning signal — and the ablation in [[#Headline evidence]] shows this is where the gain over Static comes from under domain shift.

### Component 3 — Tree Attention: batching speculative branches

Speculative decoding produces a *tree* of token continuations, not a linear sequence. A single request might propose multiple branches; verifiers reject some at each depth. Naïvely, computing loss on this tree requires multiple forward/backward passes — one per branch.

Aurora's **Tree Attention** uses a custom attention mask so a single batched forward/backward pass covers all accepted *and* rejected branches simultaneously. The figure shows the construction: the full token sequence is the union of the accepted prefix and the rejected siblings, and the attention mask is block-shaped so each branch can attend to its ancestors only.

![Tree Attention mask: a single forward covers accepted and rejected branches, paper Fig. 2](EN/wiki/llm-inference/aurora-figs/tree-attention.png)

This is the kernel-level enabler — without it, training throughput can't keep up with serving throughput, and the training server lags arbitrarily behind. Expect this kernel to be ported into other speculative-decoding stacks even by teams that don't adopt the rest of Aurora's RL framing.

### Supporting machinery (skim or skip)

> [!note]- Async synchronization policy — open if you're tuning push frequency
> Aurora exposes the policy refresh interval (how often a new speculator is hot-swapped into the Inference Server) as a tuning knob. The paper's Figure 5 sweeps it and finds the trade-off:
>
> - **Aggressive** (every 48 requests): higher post-shift adaptation (the draft catches up faster on a new distribution) but more synchronization overhead — net throughput suffers.
> - **Lazy** (every 1600 requests): minimal overhead, but loses some of the adaptation benefit.
> - **Moderate** (~every 80 requests): a strong Pareto point — retains most of the adaptation while delivering the best overall throughput.
>
> The system defaults to the moderate schedule and exposes the knob through config.

> [!note]- Loss variants explored — open if you care about RKL/FKL/NTP
> The paper ablates several training objectives in Section 5: Frozen Draft (Static Baseline, no online updates), Aurora (FKL) using forward KL, Aurora (RKL) using reverse KL on accepted tokens, Aurora (RKL + NTP) adding an auxiliary next-token-prediction loss on accepted tokens, and Aurora (w discard) with Discard Sampling on rejected branches. The full combination — **RKL + tree/discard + NTP** — wins consistently in Figure 6, and the ablation in [[#Headline evidence]] uses these labels.

## Headline evidence

**Setup.** Three configurations sweep across model sizes and serving conditions:

1. **Day-0 cold-start** on Qwen3-Coder-Next (FP8) and MiniMax M2.1 (a 230B-parameter Transformer-MoE).
2. **Mixed streams** — Day-0 adaptation against alternating domains (math, code, finance, instruction).
3. **Ordered streams** — Day-0 adaptation against sharp domain transitions to stress-test recovery.

Algorithm: RKL + tree/discard + NTP. Lookahead K=5 (Qwen3-8B), K=10 (Llama3.1-8B).

**Headline numbers.** Aurora's Day-0 untrained speculator reaches:

| Target model | Throughput vs no spec | AL at BS=8 |
| ------------ | --------------------- | ---------- |
| Qwen3-Coder-Next (FP8) | **1.21×** | 3.0 |
| MiniMax M2.1 | **1.45×** | 2.8 |

Per the paper: *"on top of the trained model drops at first but achieves better results after some training"* — i.e. Day-0 catches a *pretrained* baseline within hours.

**Adaptation under domain shift.** Aurora vs Static on ordered streams:

![Ordered streams: Day-0 adaptation of an untrained speculator vs Static + No-Speculator (paper Fig. 4)](EN/wiki/llm-inference/aurora-figs/ordered-streams.png)

Reading: each step on the x-axis is ~1K serving requests. Aurora (Trained, blue) sits above Static (green) throughout, recovers within ~10K requests after a sharp transition (the cliffs at ~10K/20K/30K), and improves *on top* of pretraining rather than plateauing.

> [!success] The shift-recovery number
> Across ordered streams, Aurora delivers ~**1.25× speedup over a static speculator** and recovers to pre-shift AL within ~**10K requests** after a forced distribution shift. Static drafts cannot recover — once AL drops, it stays dropped.

**The critical ablation: Discard Sampling on rejected tokens.** Removing the rejection term (Aurora RKL only) leaves headroom on the table; adding it (Aurora RKL + tree/discard + NTP) closes the gap. Figure 6 shows this on Qwen3-8B-Instruct:

![Discard sampling closes the gap when lookahead is large enough to leave headroom, paper Fig. 6](EN/wiki/llm-inference/aurora-figs/discard-tokens-ablation.png)

The paper's framing: when discard tokens help is gated by lookahead. At a small lookahead (5), the pretrained speculator already achieves close to its native AL ceiling — discard tokens have nowhere to add value. At lookahead 10, headroom opens up and the rejected-branch signal becomes valuable. This is a subtle scope claim worth internalizing before you ship.

**Scaling to a 230B target.** MiniMax M2.1 (BS=4 / BS=8):

![Scalability on MiniMax M2.1 — Aurora (Scratch) vs No-Speculator over time (paper Fig. 9)](EN/wiki/llm-inference/aurora-figs/scalability-minimax.png)

> [!example]- All experimental results (drill-down)
> **End-to-end throughput numbers**, varying batch sizes (paper Table 1):
>
> | Model | Config | Speedup | Acc. Length |
> | ----- | ------ | ------- | ----------- |
> | MiniMax M2.1 | BS=4, H100 GPUs w/ TP=4 | 1.45× (Scratch) | 2.8 |
> | Qwen3-Coder-Next (FP8) | BS=4 | 1.21× (Scratch) | 3.0 |
> | Llama3.1-8B | K=10, lookahead | strongest discard-sample gain | ~3.8 |
> | Qwen3-8B-Instruct | K=5 | margin closes at K=10 | ~3.0 |
>
> **Batch-size sensitivity** (paper Figure 8): tested at BS=4 and BS=12. Aurora's AL gain over static persists across batch sizes, but the speedup multiplier compresses at larger batches — because the target model is already better amortized/efficient and speculative overhead becomes a larger fraction of the pipeline. Acceptance improves; net speedup is less dramatic.
>
> **Top-k strategy on discard tokens** (paper Figure 7): training on all discard tokens vs top-k (k=0, 10, 50) yields *only marginal* differences. Top-k saves memory and keeps performance; not a fragile hyperparameter.
>
> **Async policy refresh schedule** (paper Figure 5): aggressive (48 req) improves adaptation but cuts throughput via sync overhead; moderate (80 req) is the Pareto-best; lazy (1600 req) loses the adaptation benefit. Default = moderate.

### What's missing from the evaluation

- **No head-to-head against [[das-spec-rl|DAS]]** on RL-rollout workloads, despite obvious overlap.
- **Coding-heavy benchmark mix.** Evaluation skews to coding workloads (Qwen3-Coder, MiniMax — also code-strong). Long-context summarization, math reasoning, multi-modal — not run.
- **No throughput numbers under sustained churn.** "Recovers in ~10K requests" is for one injected shift; what happens if the mix continuously oscillates?
- **Production batch-size sweep limited.** BS=4 and BS=12 reported; BS=32 / BS=64 (more common production batch sizes) not tested.

## Strengths and limitations

The standout strengths are the three components — async loop, two-term loss, Tree Attention — each addresses a real failure mode in prior systems and each lands with measurable evidence.

Where the paper is less convincing:

- **Training-serving coupling is operationally risky.** Hot-swapping weights mid-flight: a bad gradient update can tank AL and the draft can't be rolled back cleanly without rolling back the entire weight buffer. The paper doesn't discuss safeguards (gradient clipping, AL-regression rollback, A/B-split training).
- **GPU-aware RPC dependency.** Hot-swap requires Inference and Training Servers to share a GPU fabric. Deployments where training runs on a separate cluster need non-trivial retrofit.
- **Convergence depends on traffic volume.** Day-0 works *because* there's enough traffic to drive learning. Low-QPS deployments might never learn fast enough to outperform a generic offline-pretrained baseline.
- **Target stays fixed.** If the target itself is swapped (new release, fine-tuned variant), the draft has to relearn from scratch. Aurora handles distribution shift in *prompts*, not in *target*.

> [!warning] Discard tokens only help with enough lookahead
> Figure 6 shows discard tokens yield *no* gain at lookahead 5 because the pretrained speculator already operates near its AL ceiling. The win only materializes at lookahead 10 where there's headroom. This is a useful-scope caveat; a deployment that picks the wrong K will see Aurora's most novel piece (Discard Sampling) buy nothing.

## What this means

Aurora is the second canonical 2026 paper (after [[das-spec-rl|DAS]]) arguing the same structural thing: **speculative decoding should be a closed-loop system, not an open-loop one**. The draft is a *learned policy* that should learn from production, not a frozen artifact baked offline.

Two trajectories likely follow:

1. **Inference engines absorb the loop.** SGLang underpins Aurora; expect vLLM and TRT-LLM to ship analogous online-spec features within 6–12 months. Training-server-as-sidecar is the path of least resistance.
2. **The "draft pretraining as a service" market collapses.** Together itself and small vendors sell pretrained drafts per target model. Aurora's claim — *from-scratch online beats pretrained* — undermines that business line if it generalizes beyond coding workloads.

The most underrated piece is **Tree Attention** as a kernel. It's unglamorous infrastructure work that makes the rest possible. Expect the kernel (separated from the RL framing) to be ported into other speculative-decoding stacks regardless of whether they adopt the RL loop.

## Source code & pointers

```bash
git clone https://github.com/togethercomputer/aurora
# the repo ships:
#   - SGLang fork with the Inference Server hot-swap hook
#   - Training Server with Tree Attention kernel + Discard Sampling
#   - example launch configs for Qwen3-Coder-Next and MiniMax M2.1
```

Pre-trained speculators are released on Hugging Face under `togethercomputer/Tougyuan/qwen3_8b_eagle3` and analogous slugs for MiniMax variants.

Files worth reading first (paths are illustrative — names may shift):

| Path | Role |
| ---- | ---- |
| `aurora/inference/sglang_patch.py` | Hot-swap hook into SGLang's draft model |
| `aurora/training/server.py` | Async training loop, batches accept/reject |
| `aurora/training/loss.py` | Two-term loss with Discard Sampling |
| `aurora/kernels/tree_attention.py` | Custom attention mask for the speculation tree |
| `aurora/rpc/torch_distributed.py` | TensorPipe-based hot-swap RPC |

## Related reading

- [[speculative-decoding]] — broader technique and its acceptance-length / distribution-match fundamentals.
- [[das-spec-rl|DAS]] — the analogous online-spec system for **RL training rollouts** (not production inference). Aurora and DAS likely converge in implementation despite different target workloads.
- [[sglang]] — the inference backend Aurora builds on; the training-server-as-sidecar pattern leans on SGLang's continuous-batching primitives.
- [[kv-cache-optimization]] — the orthogonal serving-side throughput axis; speculative decoding multiplies, KV-cache optimization sustains the multiplier across long context.

## References

- Aurora paper, arXiv:2602.06932 — [paper](https://arxiv.org/abs/2602.06932), [project page](https://aurora-spec-ai.github.io/), [code](https://github.com/togethercomputer/aurora).
- **Online Speculative Decoding** (arXiv:2310.07177, 2023) — knowledge-distillation-only precursor; Aurora extends to RL.
- **EAGLE-3** (arXiv:2503.01840, 2025) — strong offline-pretrained speculator baseline that Aurora positions against.
- [SGLang at NeurIPS 2024](https://arxiv.org/abs/2312.07104) — the serving infrastructure Aurora extends.
