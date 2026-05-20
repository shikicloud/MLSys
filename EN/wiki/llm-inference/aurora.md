---
title: "Aurora: Online Speculative Decoding Training via RL"
category: llm-inference
tags: [speculative-decoding, online-learning, reinforcement-learning, sglang, draft-model, together-ai, paper-review]
created: 2026-05-20
updated: 2026-05-20
status: growing
---

# Aurora: Online Speculative Decoding Training via RL

> [!abstract]+ TL;DR
> **Aurora** (Together AI, ICML 2026) turns speculative-decoding draft-model training into an **online RL problem** that runs *during* live serving. An SGLang inference server streams every accepted *and* rejected token proposal to a distributed buffer; an asynchronous training server learns from that buffer and hot-swaps improved draft-model weights back into the serving instance without dropping traffic. Two consequences: (1) **day-0 deployment** — an *untrained* draft model can serve immediately and improve in flight, eliminating the multi-day offline pretraining bottleneck; (2) **continuous adaptation** — the draft tracks distribution shift in production traffic, recovering acceptance length within ~10K requests after a shift. Result: **1.21–1.45× throughput** on Qwen3-Coder-Next (FP8) and MiniMax M2.1 from scratch, **1.25×** over static speculators on adapted workloads. Sits alongside [[das-spec-rl|DAS]] (which solves the analogous problem for RL training rollouts, not general inference) as the canonical 2026 example of "speculator that learns from its own traffic."

> [!info]+ Paper metadata
> - **Paper**: [arXiv:2602.06932](https://arxiv.org/abs/2602.06932) — *Aurora: When RL Meets Adaptive Speculative Training*
> - **Code**: [github.com/togethercomputer/aurora](https://github.com/togethercomputer/aurora)
> - **Models**: Qwen3-Coder Spec, MiniMax M2.1/M2.5 Spec on Hugging Face
> - **Authors**: Junxiong Wang*, Fengxiang Bie*, Jisen Li, Zhongzhu Zhou, … Xiaoxia Wu, Chenfeng Xu (project leads)
> - **Affiliation**: Together AI (primary), with Stanford, CMU collaborators
> - **Venue**: ICML 2026
> - **Project page**: https://aurora-spec-ai.github.io/

## Background: why speculative-decoding training is a serving bottleneck

[[speculative-decoding|Speculative decoding]] gives 2–3× inference throughput by letting a small *draft model* propose K tokens that the large *target model* verifies in one parallel pass. The throughput gain is gated by **acceptance length** (AL) — how many of the K proposed tokens the target accepts. AL depends critically on **distribution match** between draft and target outputs.

In production, this match is hard to maintain:

1. **Offline-trained drafts go stale.** Drafts pretrained on a generic corpus drift away from the target's behavior on production prompts (coding, agent traces, long-context RAG). EAGLE-3 and similar approaches spend days of GPU time pretraining a draft, and re-train when the target swaps.
2. **Distribution shifts mid-deployment.** A traffic mix that's 40 % code one week and 70 % agent traces the next breaks AL on the old draft.
3. **No feedback loop.** The serving stack has the data needed to fix this — the verifier knows exactly which tokens were accepted and which were rejected — but it throws that signal away.

[[das-spec-rl|DAS]] partially addresses this for *RL training* (where rollouts are the workload), by training a draft online against the rollout-policy distribution. Aurora generalizes the idea to **arbitrary production inference traffic**.

## The key idea: speculator-as-policy in an async RL loop

Aurora frames the draft model as a **policy π** in an RL setting:

- **Action**: emit a K-token continuation.
- **Environment**: target model + verifier.
- **Reward signal**: which tokens were accepted (`paccept`), which got rejected (`pdiscard`).

The verifier already produces this signal *for free* on every request — Aurora just stops discarding it. Accepted and rejected branches stream into a shared distributed buffer; an async training server consumes the buffer and pushes updated draft-model weights back to the inference server via GPU-aware RPC. No request blocks on training, no training step blocks on serving.

The framing matters because it lets Aurora claim something stronger than prior online-distillation work: **online training *from scratch* can exceed an offline-pretrained speculator**. You don't need the multi-day pretraining phase at all — start the draft cold, let production traffic train it.

## How it works

### Architecture: SGLang server + async training server

```
┌─────────────────────────────────────┐    ┌─────────────────────────────┐
│   SGLang inference server           │    │   Async training server     │
│                                     │    │                             │
│   target model + draft model π      │    │   collect accept / reject   │
│   → verifier → emit tokens          │    │   → compute loss            │
│   → log (Q, accepted, rejected)     ├───►│   → update π                │
│                                     │    │   → push new weights        │
│   ◄── hot-swap draft weights ───────┤    │                             │
└─────────────────────────────────────┘    └─────────────────────────────┘
```

Both sides run continuously. The inference server never pauses for training; the training server never blocks on a forward pass to be served. Weight hot-swap happens via GPU-aware RPC — the draft model parameter buffer is updated in-place while live requests are in flight.

### The two-term loss

For each speculative step, the verifier partitions the K proposed tokens into an *accepted prefix* and a *rejected suffix*. Aurora trains the draft on both:

- **Acceptance loss** — cross-entropy on accepted tokens. Standard imitation-learning objective: pull `pdraft` toward what the target chose.

- **Rejection loss** — a KL term that *pushes* `pdraft` away from rejected branches, via what the authors call **Discard Sampling**. The intuition: if the target rejected a branch, the draft should put less mass there next time, even if the target hasn't explicitly chosen an alternative.

The combined objective is:

$$
\mathcal{L} = \mathbb{E}_{x\sim p_{\text{accept}}}\!\left[ \mathrm{KL}(p_{\text{target}} \,\|\, p_{\text{draft}}) \right] + \lambda_{\text{discard}}\, \mathbb{E}_{x\sim p_{\text{discard}}}\!\left[ \mathrm{KL}(p_{\text{target}} \,\|\, p_{\text{draft}}) \right]
$$

The rejection term is the non-trivial contribution. Naïve online distillation only learns from accepted tokens (positive examples). Aurora makes the rejected branches a first-class learning signal.

### Tree Attention: batching the speculative branches

Speculative decoding produces a *tree* of token continuations, not a linear sequence. A single request might propose multiple branches; verifiers reject some at each depth. Naïvely, computing loss on this tree requires multiple forward/backward passes — one per branch.

Aurora's **Tree Attention** uses custom attention masks so a single batched forward/backward pass covers all accepted *and* rejected branches simultaneously. This is the kernel-level enabler — without it, training throughput can't keep up with serving throughput, and the training server lags arbitrarily behind.

## Experiments

### Day-0: starting from a cold draft

| Target model | Throughput vs no spec | AL at batch=8 |
| ------------ | --------------------- | ------------- |
| Qwen3-Coder-Next (FP8) | **1.21×** | 3.0 |
| MiniMax M2.1 | **1.45×** | 2.8 |

These numbers are from an *untrained* draft that only saw production traffic. No offline pretraining. The headline: AL converges to useful values within hours of serving, not days.

### Adaptation after distribution shift

On Qwen3 / Llama3 with mixed coding + agent traffic:

- **1.25×** speedup over a *static* speculator that doesn't update.
- After an injected distribution shift, AL drops, then recovers to the pre-shift value within **~10K requests**.

This is the strongest claim: Aurora's gain over the prior art is *not* just a different initial training scheme. It's that the draft *keeps* improving and *recovers from* shifts, while a static-trained competitor's AL degrades and stays degraded.

### What's missing from the evaluation

- **No comparison to [[das-spec-rl|DAS]]** on RL-rollout workloads, despite the obvious overlap.
- **Coding-heavy benchmark mix.** The evaluation skews to coding workloads (Qwen3-Coder, MiniMax which is also code-strong). Long-context summarization, math reasoning, multi-modal — not run.
- **No throughput numbers under sustained distribution-shift churn.** The "recovers in 10K requests" claim is for one injected shift; what happens when the workload mix continuously oscillates?
- **No batch-size sweep at production scale.** AL=3 at batch=8 is reported; at batch=32 or batch=64 (common production batch sizes), what's the AL?

## Strengths and limitations

**Strengths:**

- **Eliminates the offline-pretraining phase.** This is a real cost — multi-day GPU spend on draft pretraining is a meaningful fraction of total inference infra cost for a new target model. Aurora retires that line item.
- **Self-healing under distribution shift.** Production traffic mix changes. Static-trained drafts decay. Aurora compounds.
- **Uses signal the verifier already produces.** No extra forward passes, no extra labels. The rejection signal is sitting on the floor in every existing speculative-decoding deployment.

**Limitations:**

- **Training-serving coupling.** Hot-swapping weights mid-flight is operationally risky. A bad gradient update can degrade AL and the draft can't be rolled back cleanly without rolling back the entire weight buffer. The paper doesn't discuss safeguards (gradient clipping, AL-regression rollback, A/B-split training).
- **GPU-aware RPC dependency.** Aurora's weight-hot-swap path requires the inference and training servers to share a GPU fabric. For deployments where training runs on a separate cluster, this is non-trivial to retrofit.
- **Convergence depends on traffic volume.** Day-0 works *because* there's enough traffic to drive learning. For low-QPS deployments, the draft might not learn fast enough to outperform a generic offline pretrained baseline.
- **Implicit assumption: target stays fixed.** If the target model itself gets swapped (new release, fine-tuned variant), the draft has to relearn from scratch. The system handles distribution shift in *prompts*, not in *target*.

## What this means

Aurora is the second canonical 2026 paper (after [[das-spec-rl|DAS]]) saying the same structural thing: **speculative decoding should be a closed-loop system, not an open-loop one**. The draft is a *learned policy* that should learn from production, not a frozen artifact baked offline.

Two trajectories likely follow:

1. **Inference engines absorb the loop.** SGLang already underpins Aurora; expect vLLM and TRT-LLM to ship analogous online-spec features within 6-12 months. The training-server-as-sidecar pattern is the path of least resistance.
2. **The "draft pretraining" market collapses.** Companies (e.g. Together itself, but also small vendors selling pretrained speculators per target model) sell offline-pretrained drafts as a service. Aurora's claim — *from-scratch online beats pretrained* — undermines that whole business line if it generalizes beyond coding workloads.

The most underrated piece is **Tree Attention** as a kernel. It's the unglamorous infrastructure work that makes the rest possible. Expect the kernel itself, separated from the RL framing, to be ported into other speculative-decoding stacks regardless of whether they adopt Aurora's RL loop.

## Related reading

- [[speculative-decoding]] — the broader technique and its acceptance-length / distribution-match fundamentals.
- [[das-spec-rl|DAS]] — the analogous online-spec system for **RL training rollouts** (not production inference). Aurora and DAS likely converge in implementation despite different target workloads.
- [[sglang]] — the inference backend Aurora builds on; the training-server-as-sidecar pattern leans on SGLang's continuous-batching primitives.

## References

- Aurora paper, arXiv:2602.06932 — [paper](https://arxiv.org/abs/2602.06932), [project page](https://aurora-spec-ai.github.io/), [code](https://github.com/togethercomputer/aurora).
- **Online Speculative Decoding** (arXiv:2310.07177, 2023) — knowledge-distillation-only precursor; Aurora extends to RL.
- **EAGLE-3** (arXiv:2503.01840, 2025) — strong offline-pretrained speculator baseline that Aurora positions against.
- [SGLang at NeurIPS 2024](https://arxiv.org/abs/2312.07104) — the serving infrastructure Aurora extends.
