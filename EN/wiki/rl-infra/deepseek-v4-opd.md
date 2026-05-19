---
title: "DeepSeek-V4 OPD: Multi-Teacher Full-Vocabulary On-Policy Distillation as RL Replacement"
category: rl-infra
tags: [deepseek-v4, opd, on-policy-distillation, multi-teacher-kl, full-vocabulary-kl, post-training, moe, paper-review]
created: 2026-05-19
updated: 2026-05-19
status: mature
---

# DeepSeek-V4 OPD: Multi-Teacher Full-Vocabulary On-Policy Distillation as RL Replacement

> [!info] Model metadata
> - **Release**: 2026-04-24 (DeepSeek-V4-Pro and DeepSeek-V4-Flash, simultaneous)
> - **Technical report**: PDF only at [huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf) (no arXiv submission)
> - **Model cards**: [V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) (1.6T total / 49B active) · [V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) (284B total / 13B active)
> - **License**: MIT, open weights
> - **Context**: 1M tokens
> - **API**: launched same day, see [api-docs.deepseek.com/updates](https://api-docs.deepseek.com/updates)
> - **Press**: [CNBC](https://www.cnbc.com/2026/04/24/deepseek-v4-llm-preview-open-source-ai-competition-china.html), [MIT Technology Review](https://www.technologyreview.com/2026/04/24/1136422/why-deepseeks-v4-matters/), [Bloomberg](https://www.bloomberg.com/news/articles/2026-04-24/deepseek-unveils-newest-flagship-a-year-after-ai-breakthrough)

> [!abstract]+ TL;DR
> DeepSeek-V4 makes one **critical methodological substitution** vs V3.2: *"the mixed Reinforcement Learning (RL) stage was entirely replaced by On-Policy Distillation (OPD)"* (§5.1). The pipeline becomes **base → per-domain (SFT → GRPO) specialists → multi-teacher OPD merge → V4**. The OPD loss is a weighted sum over $> 10$ specialist teachers, $\mathcal{L}_{\text{OPD}}(\theta) = \sum_i w_i\, D_{\text{KL}}(\pi_\theta \| \pi_{E_i})$, with reverse-KL on student-sampled rollouts. The algorithmic novelty vs Thinking Machines Lab's blog ([[on-policy-distillation]]) and MiniLLM is **full-vocabulary logit KL** — V4 explicitly rejects the token-level KL surrogate that prior on-policy distillation work used, arguing the surrogate has high gradient variance and causes training instability. The infrastructure novelty is **how they make full-vocab KL feasible at 1.6T-MoE scale**: hidden-state caching (not full logits) of teachers in centralized storage, sample-sort-by-teacher to keep at most one teacher head in GPU memory per microbatch, a custom TileLang exact-KL kernel, and FP4 QAT applied to teacher weights to fit them. **No GPU-hour comparison vs GRPO is reported** in the paper — cost claims circulating online inherit from TML's Qwen3 blog, not from DeepSeek.

---

## Why this matters

V4 is the **first flagship-scale, fully-open-weight model** to commit *exclusively* to OPD for the post-training merge step that previously required RL. Prior public OPD work was either small-scale ([[on-policy-distillation|Qwen3]] 0.6B–14B + 30B-A3B-MoE) or used OPD as a component alongside RL ([Nemotron-Cascade 2 MOPD](https://research.nvidia.com/labs/nemotron/nemotron-cascade-2/)). V4 is the first time someone has bet a 1.6T trillion-parameter MoE on the OPD-replaces-RL hypothesis and published the recipe.

If V4's reasoning numbers hold (the paper claims competitive with GPT-5.2 / Gemini-3.0-Pro on standard reasoning benchmarks), this is the canonical example everyone in the field will cite from 2026 onwards for "yes, OPD can replace RL at frontier scale." If they don't, V4 will be the cautionary tale of trusting [[on-policy-distillation#The opd-vs-rl-debate|the OPD-as-RL-replacement framing]] without enough RL exploration to push past teacher capabilities.

---

## The architecture (briefly)

Just enough to ground the OPD discussion. Full architecture analysis is out of scope for this page.

**Two simultaneous releases:**

| Model | Total params | Active params | Architecture | Context |
|-------|-------------|---------------|-------------|---------|
| DeepSeek-V4-Pro | 1.6T | 49B | MoE + hybrid CSA + HCA attention, mHC residuals, Muon optimizer | 1M |
| DeepSeek-V4-Flash | 284B | 13B | Same shape, smaller | 1M |

Key architectural moves vs V3:

- **Hybrid attention** (CSA + HCA) — replaces V3's pure MLA at 1M-context scale.
- **mHC (multi-head conditional?) residuals** — the paper's notation; not yet widely interpreted in commentary.
- **Muon optimizer** — switches from AdamW for some training stages.
- **MoE expert count** — increased over V3; the technical report doesn't publish the exact number publicly, but the active fraction (49/1600 ≈ 3 %) matches V3's sparsity profile.

The training-infrastructure section explicitly says infrastructure is **reused from V3.2's stack** (rollout engine, fault-tolerant generation service, KV pool). The novel piece is post-training, not pre-training.

---

## The OPD pipeline

### Stage 1 — Specialist training (§5.1.1)

For each domain $D_k \in$ {math, coding, agent, instruction-following, alignment, ...}:

1. Start from V4 base.
2. **Domain-specific SFT** on curated traces.
3. **GRPO RL** with domain-specific reward signals. *"Hyperparameters closely aligned with our prior research"* (V3.2 / R1-era).
4. Save the resulting domain specialist as $E_{D_k}$.

**Three reasoning effort modes** — Non-think / Think High / Think Max — are obtained by training **different specialists under different length penalties and context window allowances**, then merged together in Stage 2.

Note that GRPO is still used in this stage — V4 doesn't replace GRPO globally, it replaces the *unified mixed-RL post-training stage that V3.2 used to merge specialists*. Per-domain RL still happens; it's the merge that becomes OPD.

**Generative Reward Model (GRM):** V4 introduces a self-judging mechanism: *"the actor network natively functions as the GRM"* — joint optimization of judging + generation, replacing scalar RLHF rewards for hard-to-verify tasks (alignment, instruction-following, agentic). This is its own methodological contribution worth tracking; the GRM produces the dense reward signal for the specialist's GRPO training.

### Stage 2 — Multi-teacher OPD merge (§5.1.2)

The substitution that defines V4. Quote from the technical report (line 1583):

> *"Although the training pipeline largely mirrored that of DeepSeek-V3.2, a critical methodological substitution was made: the mixed Reinforcement Learning (RL) stage was entirely replaced by On-Policy Distillation (OPD)."*

The loss (paper's Eq. 29, p. 32):

$$
\boxed{\,\mathcal{L}_{\text{OPD}}(\theta) \;=\; \sum_{i=1}^{N} w_i \cdot D_{\text{KL}}\!\left(\pi_\theta \,\Big\|\, \pi_{E_i}\right)\,}
$$

Properties:

- **Reverse KL**: student first in the KL — mode-seeking, consistent with TML/MiniLLM (see [[on-policy-distillation#Loss function|loss derivation]]).
- **On-policy**: trajectories sampled from $\pi_\theta$ at each training step (paper §5.1.2: *"Computing the reverse KL loss … requires sampling training trajectories from the student π_θ to maintain on-policy learning."*).
- **Multi-teacher**: $N > 10$ specialist teachers, one per domain.
- **Weights $w_i$**: *"typically determined by the relative importance of the expert"* — no fixed schedule published.
- **Selective alignment**: *"the unified policy π_θ selectively learns from the specialized expert relevant to the current task context"* — the routing of prompt → relevant teacher is implicit in the per-prompt domain identification; only the relevant teacher contributes meaningful gradient on a given prompt.

### Citations the paper relies on

V4's OPD section explicitly cites two lineage roots (PDF line 1779):

> *"we employ multi-teacher On-Policy Distillation (OPD; Gu et al. 2024; Lu and Lab 2025) as the primary technique for merging expert capabilities into the final model."*

- **Gu et al. 2024** = [MiniLLM](https://arxiv.org/abs/2306.08543) — the reverse-KL-as-policy-gradient derivation.
- **Lu and Lab 2025** = [Thinking Machines Lab blog](https://thinkingmachines.ai/blog/on-policy-distillation/) — the practitioner "RL replacement" framing.

The DeepSeek paper makes no claim of inventing OPD as a technique. The novelty is the multi-teacher full-vocabulary deployment at trillion-parameter scale.

---

## The actual algorithmic novelty: full-vocabulary KL

This is what makes V4's OPD different from TML's recipe. The paper's argument (lines 1803-1812):

> *"prior works usually simplify the full-vocabulary KL loss into a token-level KL estimate … reuse RL framework by replacing $\text{sg}\,\log(\pi_E / \pi_\theta)$ as the per-token advantage estimate … this approach … leads to high variance in gradient estimation and often causes training instability. Therefore, we adopt **full-vocabulary logit distillation in our OPD**. Preserving the complete logit distribution … yields more stable gradient estimates and ensures faithful distillation."*

Two configurations under discussion:

| | Token-level OPD (prior work, TML, MiniLLM) | Full-vocabulary OPD (DeepSeek-V4) |
|---|---|---|
| What's compared | Teacher / student probabilities at *the sampled token* only | Teacher / student probability distributions over *all* vocab tokens |
| Gradient form | $\nabla_\theta \log\pi_\theta(y_t) \cdot \log\frac{\pi_T(y_t)}{\pi_\theta(y_t)}$ (REINFORCE-style) | $\sum_v \pi_\theta(v) \log\frac{\pi_\theta(v)}{\pi_T(v)}$ direct KL gradient |
| Variance | High — single-token sample of a $V$-dimensional KL | Low — analytic KL over full distribution |
| Memory per token | $O(1)$ — just two scalar logprobs | $O(V)$ — full softmax outputs from both models |
| Network bandwidth (if teacher is remote) | $O(1)$ per token | $O(V)$ per token, $V \approx 100$ K |
| Faithful | Biased estimator of sequence-level KL | Exact per-token KL |

The full-vocabulary form has been mathematically obvious since GKD (2023). **Nobody used it at scale because the $O(V \times \text{tokens} \times N_{\text{teachers}})$ memory and bandwidth made it infeasible.** V4 makes it feasible via the infrastructure recipe in §5.2.2.

---

## The infrastructure that makes full-vocab OPD feasible (§5.2.2)

This is the practically interesting part for an NVIDIA reader. Quoting from the technical report's section on "Efficient Teacher Scheduling for Full-Vocabulary OPD":

### 1. Hidden-state caching (not logits)

The naive approach — cache per-token logits for every teacher across all training data — is $O(|V| \times N_{\text{tokens}} \times N_{\text{teachers}})$. For $|V| = 100$K, $N_{\text{tokens}} = $ millions, $N_{\text{teachers}} \geq 10$, that's terabytes per epoch — prohibitive.

V4's trick: **cache only the teacher's last-layer hidden states** (not its logits), then **re-run the prediction head on-the-fly** at training time:

```
Pre-training pass per teacher specialist:
  for each training prompt x:
    h_E = teacher_E.last_hidden_states(x)    # one forward pass, no LM head
    store h_E to centralized buffer          # O(d_model) per token, ~50x smaller than O(V)

Training time:
  for each minibatch:
    sample student rollouts y ~ π_θ
    load h_E for the rollout's prompt
    teacher_logits = teacher_E.lm_head(h_E)   # re-run head on cached hidden states
    compute full-vocab KL(π_θ || softmax(teacher_logits))
```

Saves ~50× memory vs logit caching (at $d_{\text{model}} = 2048$ and $V \approx 100$K).

### 2. Sample sorting by teacher index

With $> 10$ teachers, you can't have all teacher LM heads resident in GPU memory simultaneously. V4 sorts samples in the data dispatcher by which teacher they need, so within any microbatch only one teacher's LM head needs to be loaded:

```
Microbatch dispatcher:
  Sort samples by primary teacher index
  Within each "teacher chunk":
    Load teacher_E.lm_head onto GPU
    Compute KL for all samples in this chunk
  Move to next teacher chunk
```

At most one teacher head in GPU memory per moment per data-parallel rank. The cost is loading $\geq 10$ different LM heads per epoch, paid by async I/O off the critical path.

### 3. Async I/O off the critical path

Teacher weights live in **centralized distributed storage** with **ZeRO-like parameter sharding**, fetched on demand. Hidden-state caches similarly. Loads run async while the previous teacher chunk's compute happens.

### 4. Custom TileLang kernel for exact KL

TileLang is a recent NVIDIA-developed tile-based programming model (similar to Triton). V4 ships a custom TileLang kernel computing the exact teacher-student KL fused with the softmax — avoiding intermediate logit materialization and the standard "compute logits, then loss" two-pass pattern.

### 5. FP4 teacher QAT

Teacher weights are **FP4-quantized** (QAT, not post-training) to fit in storage and to make on-demand loading bandwidth-feasible. Student stays full precision. The paper argues that since the teacher provides the *target distribution* (not gradients), FP4 inference is acceptable accuracy-wise.

### 6. Rollout infrastructure reuse

OPD rollouts ride on the **same preemptible, fault-tolerant WAL-based generation service that handles GRPO** in Stage 1. From the infra perspective, OPD looks like RL minus the reward — same scheduler, same rollout engine, same fault tolerance.

> [!important] The recipe insight
> V4's OPD section reads less like an algorithm paper and more like a *systems paper that happens to make a previously-hypothetical algorithm feasible*. The math (Eq. 29) is one line; the engineering (hidden-state cache + sample-sort + TileLang + FP4 QAT + reused RL infra) is the part that turns "full-vocab multi-teacher KL at 1.6T scale" from a research dream into a shipping pipeline.

---

## What the paper does NOT report

For calibration, three pieces of information absent from the V4 technical report:

1. **GPU-hour comparison vs GRPO.** The paper does *not* publish "OPD cost vs the mixed-RL stage cost." Any "10× cheaper" claim circulating in V4 commentary is **inherited from TML's Qwen3 blog**, not from V4 itself.
2. **Ablation: "V4 with mixed RL would have been worse than V4 with OPD."** The substitution is justified qualitatively (avoids weight-merging degradation, lower variance gradients via full-vocab KL) but no A/B is published.
3. **Concrete weights $w_i$.** Stated as "relative importance" — no schedule, no learned routing mechanism described.
4. **Source code for the OPD trainer.** No public release. The training stack is built on V3.2's infrastructure, which is also closed.

This is consistent with DeepSeek's overall release pattern (weights open, training stack closed) but worth flagging for anyone planning to reproduce the recipe.

---

## How V4's OPD relates to others

A short genealogy table:

| | GKD (2023) | MiniLLM (2023) | TML OPD (2025) | Qwen3 small (2025) | Nemotron-Cascade 2 MOPD (2026) | **DeepSeek-V4 OPD (2026)** |
|---|---|---|---|---|---|---|
| Teacher count | 1 | 1 | 1 | 1 (larger Qwen3) | many (per-domain best checkpoint) | many (per-domain specialist) |
| Sampling | $\lambda$ knob | student | student | student | student | student |
| KL direction | $\beta$ knob (gen. JSD) | reverse | reverse | reverse | reverse | reverse |
| KL form | token-level | token-level (as PG) | token-level | token-level | token-level | **full-vocabulary** |
| Role in pipeline | KD primitive | KD primitive | RL replacement (blog framing) | replaces stages 3–4 of full RL | regression recovery interleaved with cascade RL | **entire post-training merge stage** |
| Model scale | T5 / PaLM-2 | medium dense | Qwen3-8B | Qwen3 0.6B–30B | 30B-active MoE | **1.6T MoE** |
| Source code | google-deepmind | thu-coai/MiniLLM | (cookbook) | (Qwen) | (NVIDIA) | (closed) |

V4 occupies the most aggressive position on every dimension — biggest model, broadest pipeline role, most teachers, full-vocab KL.

---

## Reception and reading

A small set of writeups I found informative for the post-release window (Apr–May 2026):

- **Andrew Lukyanenko** — [V4 Review: Why Million Token Context Needs Efficient Attention](https://artgor.medium.com/deepseek-v4-review-why-million-token-context-needs-efficient-attention-not-just-larger-windows-6dc8e74a00b1). On the substitution: *"V4 replaces the unified GRPO pipeline from DeepSeek-R1 with a compositional alternative … decomposing into specialists and merging via full-vocabulary KL."*
- **OutcomeSchool** — [Decoding DeepSeek V4](https://outcomeschool.com/blog/decoding-deepseek-v4). On why on-policy matters in the merge: *"the student never sees the kind of outputs it actually produces at inference time. With OPD, the student samples its own trajectories and the teacher corrects each token … more stable and faithful knowledge transfer than weight merging or mixed RL."*
- **BSWEN** — [Two-Stage Post-Training Pipeline of DeepSeek V4](https://docs.bswen.com/blog/2026-04-25-deepseek-v4-two-stage-post-training/). Independent verification of Eq. 29 and the full-vocab framing.
- **qingkeai.online (CN)** — [DeepSeek V4 OPD analysis](https://qingkeai.online/archives/DeepSeek-V4-OPD). Frames OPD as catastrophic-forgetting countermeasure during specialist merging.
- **Fireworks AI** — [What DeepSeek V4 Says About Training Platforms](https://fireworks.ai/blog/what-deepseek-v4-says-about-training-platforms). Useful from the infra perspective (relevant for NVIDIA training infra readers).

**Sebastian Raschka's pre-V4 deep-dive on V3 → V3.2** ([magazine post](https://magazine.sebastianraschka.com/p/technical-deepseek)) does NOT cover V4 / OPD — it pre-dates V4 by a few months. A follow-up from him on V4's OPD would be valuable; not yet out as of mid-May 2026.

---

## Honest assessment

**What's actually novel in V4's OPD section** (as distinct from prior on-policy distillation work):

1. **Full-vocabulary KL at flagship scale.** TML and MiniLLM use token-level surrogates because they're cheap; V4 demonstrates that with the right infra (hidden-state cache + sample-sort + TileLang + FP4 QAT) the *exact* per-token KL is feasible at 1.6T MoE scale. The variance reduction matters more at large scale because gradient instability compounds.
2. **Multi-teacher merging as the post-training paradigm.** Prior OPD was single-teacher. V4's $\sum_i w_i D_{\text{KL}}$ with $> 10$ specialists is *qualitatively* a different use case — it's not "compress a teacher" but "merge specialists" — and it generalizes to any setting where you have orthogonal capabilities you want to combine.
3. **Specialist-per-reasoning-mode.** Training distinct specialists per Non-think / Think High / Think Max effort level and merging them via OPD is a clean architectural answer to "one model, multiple effort levels." Previous approaches (single model conditioned on a mode token) struggle with capability dilution.

**What is recycled (correctly attributed by the paper):**

- The reverse-KL on-policy formulation (MiniLLM 2023).
- The "OPD as RL replacement" narrative (TML Oct 2025).
- The GRPO specialist training (DeepSeek-V3 / R1 stack).

**What is marketing or untested:**

- Cost claims. The paper publishes no GPU-hour comparison; "10× cheaper than RL" headlines in coverage come from TML's Qwen3 number being laundered into the V4 discussion.
- The substitution is justified qualitatively, not empirically. There's no V4-with-mixed-RL ablation.
- The 1.6T-MoE scaling claim is unverifiable independently because the trainer is closed.

**Bottom line.** V4's OPD recipe is the most ambitious deployment of on-policy distillation to date, the engineering is concrete enough to replicate (given infrastructure), and the algorithmic shift toward full-vocabulary KL is genuinely worth taking seriously. The pipeline-replacement claim is bold and remains to be independently validated, but as a recipe-to-study it's the canonical reference for 2026.

---

## What to take away if you're training a model

1. **If you have a strong teacher (or a set of specialists) in the same family**: try OPD before doing RL. Start with HF TRL's `GKDTrainer` (token-level) for prototyping; consider full-vocab KL when scaling up.
2. **For specialist merging**: V4 is the canonical recipe — train per-domain via GRPO, then merge via multi-teacher reverse-KL OPD. The weighted sum $\sum_i w_i D_{\text{KL}}$ structure cleanly composes.
3. **For frontier capability extension**: don't expect OPD alone. Stack OPD warm-start + GRPO (or [KDRL](https://arxiv.org/abs/2506.02208) joint objective) for tasks where the goal is to surpass the teacher.
4. **For infrastructure**: full-vocab KL is feasible if you cache teacher hidden states (not logits), sort samples by teacher, and use FP4 QAT on teachers. The reusable insight is that **OPD looks like RL minus reward at the infra level** — same rollout engine, same fault tolerance, same scheduler.

---

## Related reading

- [[on-policy-distillation]] — The umbrella technique covered as algorithm + variants + debate; V4's OPD is the flagship-scale instantiation.
- [[grpo]] — The RL algorithm used in V4 Stage 1 specialist training and replaced in the Stage 2 merge.
- [[ppo-for-llm]] — The trust-region intuition underlying the KL regularization in both stages.
- [[rlhf-overview]] — Standard post-training pipeline that V4 disrupts at the merge stage.
- [[parallelism-strategies-deep-dive#13. DeepSeek-V3 Case Study]] — V3 architecture and parallelism foundation V4 builds on.
- [[kv-cache-optimization]] — KV management at 1M context, relevant to V4's inference infrastructure.
- [[das-spec-rl]] — Speculative decoding speedup for the rollout phase of any GRPO / OPD pipeline; complementary at the inference layer.

## References

- **DeepSeek-V4 technical report**: [HF PDF](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf)
- **DeepSeek-V4-Pro model card**: [huggingface.co/deepseek-ai/DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
- **DeepSeek-V4-Flash model card**: [huggingface.co/deepseek-ai/DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
- **API changelog**: [api-docs.deepseek.com/updates](https://api-docs.deepseek.com/updates)
- **Thinking Machines Lab — On-Policy Distillation blog** (cited by V4): [thinkingmachines.ai/blog/on-policy-distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)
- **MiniLLM** (cited by V4): Gu et al., [arXiv:2306.08543](https://arxiv.org/abs/2306.08543)
- **GKD** (the umbrella math): Agarwal et al., [arXiv:2306.13649](https://arxiv.org/abs/2306.13649)
- **Independent V4 writeups**: [Lukyanenko](https://artgor.medium.com/deepseek-v4-review-why-million-token-context-needs-efficient-attention-not-just-larger-windows-6dc8e74a00b1), [OutcomeSchool](https://outcomeschool.com/blog/decoding-deepseek-v4), [BSWEN](https://docs.bswen.com/blog/2026-04-25-deepseek-v4-two-stage-post-training/), [qingkeai.online](https://qingkeai.online/archives/DeepSeek-V4-OPD), [Fireworks AI](https://fireworks.ai/blog/what-deepseek-v4-says-about-training-platforms)
- **Press coverage**: [CNBC](https://www.cnbc.com/2026/04/24/deepseek-v4-llm-preview-open-source-ai-competition-china.html), [MIT Technology Review](https://www.technologyreview.com/2026/04/24/1136422/why-deepseeks-v4-matters/), [Bloomberg](https://www.bloomberg.com/news/articles/2026-04-24/deepseek-unveils-newest-flagship-a-year-after-ai-breakthrough)
