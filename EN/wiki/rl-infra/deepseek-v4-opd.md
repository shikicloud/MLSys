---
title: "DeepSeek-V4 OPD: Multi-Teacher Full-Vocabulary On-Policy Distillation as RL Replacement"
category: rl-infra
tags: [deepseek-v4, opd, on-policy-distillation, multi-teacher-kl, full-vocabulary-kl, post-training, moe, paper-review]
created: 2026-05-19
updated: 2026-05-19
status: mature
paper: DeepSeek-V4 technical report
---

# DeepSeek-V4 OPD: Multi-Teacher Full-Vocabulary On-Policy Distillation as RL Replacement

> [!info] Paper metadata
> - **Paper**: [DeepSeek-V4 technical report (HF PDF)](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf) — DeepSeek, 2026-04-24 (no arXiv submission)
> - **Models**: [DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) (1.6T total / 49B active) · [DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) (284B / 13B)
> - **License**: MIT, open weights
> - **Context**: 1M tokens
> - **API**: launched same day — [api-docs.deepseek.com/updates](https://api-docs.deepseek.com/updates)
> - **OPD trainer code**: not released
> - **Press**: [CNBC](https://www.cnbc.com/2026/04/24/deepseek-v4-llm-preview-open-source-ai-competition-china.html) · [MIT Tech Review](https://www.technologyreview.com/2026/04/24/1136422/why-deepseeks-v4-matters/) · [Bloomberg](https://www.bloomberg.com/news/articles/2026-04-24/deepseek-unveils-newest-flagship-a-year-after-ai-breakthrough)

> [!abstract]+ TL;DR
> DeepSeek-V4 makes one **critical methodological substitution** vs V3.2 (per §5.1 of the tech report): *"the mixed Reinforcement Learning (RL) stage was entirely replaced by On-Policy Distillation (OPD)."* The post-training pipeline becomes **base → per-domain (SFT → GRPO) specialists → multi-teacher OPD merge → V4**. The OPD loss sums reverse-KL over $> 10$ specialist teachers, $\mathcal{L}_{\text{OPD}}(\theta) = \sum_i w_i\, D_{\text{KL}}(\pi_\theta \| \pi_{E_i})$, on student-sampled rollouts. The algorithmic novelty vs [Thinking Machines Lab's blog](https://thinkingmachines.ai/blog/on-policy-distillation/) and [[on-policy-distillation|MiniLLM/GKD]] is **full-vocabulary logit KL** — V4 explicitly rejects the token-level KL surrogate that prior on-policy distillation work used, arguing the surrogate has high gradient variance and causes training instability at scale. The systems contribution is **how they make full-vocab KL feasible at 1.6T-MoE scale**: cache teacher hidden states (not logits), sort samples by teacher index, custom TileLang exact-KL kernel, FP4 QAT on teachers, reuse the V3.2 rollout / WAL infrastructure. **No GPU-hour comparison vs GRPO is reported** — cost claims circulating online inherit from TML's Qwen3 blog, not from DeepSeek.

---

## Background: why V4 replaced V3.2's mixed-RL stage

V3.2 / R1 used a "four-stage mixed-RL" post-training recipe: SFT → reasoning RL → mixed RL → alignment RL. Two structural problems made it expensive and unstable at the scale V4 wanted to hit:

| Problem | Why it shows up at scale | What it costs you |
| ------- | ------------------------ | ----------------- |
| **Sparse outcome reward** | 16 K+ token reasoning rollouts get one 0/1 signal | Most tokens have no credit assignment; reconstructing it from a scalar is GPU-expensive |
| **Specialist regression during mixed RL** | Different domains (math, code, agent, IF) need different reward signals — training one regresses on the others | Either re-train (expensive) or accept regression |
| **No clean reasoning-mode separation** | One model handling Non-think / Think High / Think Max via a mode token suffers capability dilution | Architectural workaround needed |

The V3.2 → V4 design move: **decompose into specialists trained independently with GRPO, then merge via OPD**. RL provides per-domain exploration and reward-driven training; OPD provides the dense token-level signal that makes merging the specialists fast and stable. The result is the most aggressive deployment of [[on-policy-distillation|OPD]] to date — V4 is the first flagship to bet a 1.6T MoE *entirely* on the OPD-replaces-merge-stage hypothesis.

Cross-reference: see [[on-policy-distillation#Background-why-on-policy-distillation-needed-inventing|the OPD background]] for the SFT-vs-RL framing this builds on.

---

## The key idea: multi-teacher full-vocabulary OPD

> [!quote] The contribution in one sentence
> Train per-domain specialists with GRPO, then merge them into a single unified model via on-policy distillation against $> 10$ teachers using **exact per-token full-vocabulary reverse KL**, made feasible at 1.6T scale by an engineering recipe that caches teacher hidden states rather than logits.

Three sub-claims hold the contribution up:

- **Multi-teacher merging > weight averaging.** Specialists trained with GRPO on different domains can't be naively averaged — capabilities cancel. Reverse-KL OPD into a single unified policy preserves the specialists' behavioral fingerprints because the student is graded against each teacher's distribution.
- **Full-vocabulary KL beats token-level surrogate.** Prior OPD work uses a single sampled token to estimate $V$-dimensional KL — a high-variance Monte Carlo estimator that destabilizes long-rollout training. Full-vocab gives the exact KL with $V$× more compute, but trades it for much lower gradient variance.
- **Infrastructure makes the impossible feasible.** Full-vocab KL × 10+ teachers × 1.6T model = naive impossible. The engineering recipe (hidden-state cache, sample-sort, FP4 teacher QAT, TileLang kernel) is what turns the algorithm into a shipping pipeline.

Remove any one: lose multi-teacher and you have single-teacher OPD with all of TML's limitations; lose full-vocab and you re-introduce the variance pathology; lose the infrastructure and the recipe doesn't run.

---

## How it works

### Pipeline overview

```
                  ┌──────────────────────────────────────────────┐
V4 base ────►     │  Stage 1 (§5.1.1) — Specialist training       │
                  │                                                │
                  │  for domain D in {math, code, agent, IF, ...}: │
                  │      SFT(D)  →  GRPO(D)  →  best ckpt E_D     │
                  │                                                │
                  │  GRM (generative reward model): actor itself  │
                  │  doubles as judge for hard-to-verify tasks    │
                  │                                                │
                  │  3 reasoning modes (Non-think / Think High /   │
                  │  Think Max) get separate specialists with     │
                  │  different length penalties + context windows │
                  └─────────────────────┬──────────────────────────┘
                                        │
                                        ▼  10+ specialists
                  ┌──────────────────────────────────────────────┐
                  │  Stage 2 (§5.1.2) — Multi-teacher OPD merge   │
                  │                                                │
                  │  student rollout y ~ π_θ                       │
                  │  loss = Σ_i w_i · D_KL(π_θ || π_{E_i})         │
                  │  full-vocabulary reverse KL on each π_{E_i}   │
                  │                                                │
                  │  routing: prompt → relevant specialist        │
                  │  (selective alignment; only the right teacher │
                  │  gives meaningful gradient per prompt)        │
                  └─────────────────────┬──────────────────────────┘
                                        │
                                        ▼
                                   DeepSeek-V4
```

### Stage 1 — Specialist training (§5.1.1)

For each domain $D_k$ from `{math, coding, agent, IF, alignment, ...}`:

1. Start from V4 base (architecture: hybrid CSA + HCA attention, mHC residuals, Muon optimizer for some stages).
2. **Domain-specific SFT** on curated traces.
3. **GRPO RL** with domain-specific reward signal. The paper says *"hyperparameters closely aligned with prior research"* (V3.2 / R1 era).
4. Save the resulting specialist as $E_{D_k}$.

**GRM contribution.** V4 introduces a Generative Reward Model where *"the actor network natively functions as the GRM"* — joint optimization of judging + generation, replacing scalar RLHF reward for hard-to-verify tasks (alignment, IF, agentic). The GRM produces the dense reward signal for GRPO training inside Stage 1.

**Reasoning mode specialists.** Non-think / Think High / Think Max are realized by **training different specialists under different length penalties + context windows**, rather than mode-token conditioning. The OPD merge in Stage 2 fuses them.

Note: **GRPO is still used inside Stage 1.** V4 doesn't replace GRPO globally — it replaces the unified *mixed-RL* stage that V3.2 used to merge specialists together. Per-domain RL still happens; the merge is what becomes OPD.

### Stage 2 — Multi-teacher OPD merge (§5.1.2)

The substitution that defines V4. Direct quote from the tech report (line 1583):

> *"Although the training pipeline largely mirrored that of DeepSeek-V3.2, a critical methodological substitution was made: the mixed Reinforcement Learning (RL) stage was entirely replaced by On-Policy Distillation (OPD)."*

The loss (paper's Eq. 29, p. 32):

$$
\boxed{\,\mathcal{L}_{\text{OPD}}(\theta) \;=\; \sum_{i=1}^{N} w_i \cdot D_{\text{KL}}\!\left(\pi_\theta \,\Big\|\, \pi_{E_i}\right)\,}
$$

Properties:

| Aspect | Value | Note |
| ------ | ----- | ---- |
| KL direction | Reverse (student first) | Mode-seeking; consistent with TML/MiniLLM |
| Sampling | $y \sim \pi_\theta$ (student rollout) | On-policy; matches deployment distribution |
| Teacher count | $N > 10$ | One per domain specialist |
| Weights $w_i$ | "Relative importance of the expert" | Not published as a schedule |
| Routing | Per-prompt to relevant teacher | Selective alignment — only one teacher gives strong gradient on a given prompt |

### The full-vocabulary KL choice

The actual algorithmic differentiator vs TML / MiniLLM. The paper's argument (lines 1803–1812):

> *"prior works usually simplify the full-vocabulary KL loss into a token-level KL estimate … this approach … leads to high variance in gradient estimation and often causes training instability. Therefore, we adopt **full-vocabulary logit distillation in our OPD**."*

| | Token-level OPD (TML, MiniLLM, GKD default) | Full-vocabulary OPD (V4) |
| --- | --- | --- |
| Estimator | Single sample of $V$-dim KL | Exact analytic KL |
| Gradient | $\nabla \log\pi_\theta(y_t) \cdot \log(\pi_T(y_t)/\pi_\theta(y_t))$ | $\sum_v \pi_\theta(v) \log(\pi_\theta(v)/\pi_T(v))$ |
| Variance | High; compounds on long rollouts | Low |
| Memory / token | $O(1)$ | $O(V)$, $V \approx 100$ K |
| Bandwidth / token | $O(1)$ | $O(V)$ — prohibitive without engineering |
| Feasibility at 1.6T | Trivial | Hard — needs the §5.2.2 infrastructure |

The full-vocabulary form has been mathematically obvious since GKD (2023). Nobody used it at scale because the memory / bandwidth made it infeasible. V4's contribution is making it run.

### Infrastructure that makes full-vocab OPD feasible (§5.2.2)

Five engineering moves let $10+$ teachers × full-vocab KL × 1.6T model actually run:

**1. Cache teacher hidden states, not logits.**

The naive approach is $O(|V| \times N_{\text{tokens}} \times N_{\text{teachers}})$ — terabytes per epoch. V4 caches only the teacher's **last-layer hidden states** (not the LM-head output) and **re-runs the LM head on-the-fly** at training time:

```python
# Pre-training pass per teacher specialist
for prompt in training_data:
    h_E = teacher_E.last_hidden_states(prompt)   # O(d_model) per token
    store(h_E)                                   # ~50× smaller than O(V)

# Training time
for batch in dataloader:
    y = sample_rollout(student)                  # on-policy
    h_E = load_cached_hidden_states(batch)
    teacher_logits = teacher_E.lm_head(h_E)      # re-run head on cached h
    loss = full_vocab_kl(student_logits, teacher_logits)
```

~50× memory reduction at $d_{\text{model}} = 2048$, $V \approx 100$ K.

**2. Sort samples by primary teacher.**

With $> 10$ teachers, not all LM heads fit on GPU simultaneously. V4 sorts samples in the data dispatcher by primary teacher index so within a microbatch **at most one teacher LM head is GPU-resident**:

```
Dispatcher:
  group(samples, key=primary_teacher_index)
  for teacher_chunk in groups:
      gpu_load(teacher_chunk.teacher.lm_head)
      compute_kl_for_all(teacher_chunk.samples)
```

Cost: loading $\geq 10$ different LM heads per epoch — paid by async I/O off the critical path.

**3. Async I/O off the critical path.**

Teacher weights live in centralized distributed storage with ZeRO-like parameter sharding. Hidden-state caches similarly. Loads run async while the previous teacher chunk computes.

**4. Custom TileLang kernel for exact KL.**

TileLang is a recent NVIDIA-developed tile-based programming model (similar to Triton). V4 ships a custom kernel computing exact teacher-student KL **fused with the softmax**, avoiding the two-pass "logits then loss" pattern.

**5. FP4 teacher QAT.**

Teacher weights are **FP4-quantized via QAT** (not post-training quantization) to fit in storage and make on-demand loading bandwidth-feasible. Student stays full precision. Argument: teacher provides the *target distribution* (not gradients), so FP4 inference is accuracy-acceptable.

**6. Reuse the V3.2 rollout / WAL stack.**

OPD rollouts ride on the **same preemptible, fault-tolerant WAL-based generation service that handles GRPO** in Stage 1. From the infrastructure perspective, OPD looks like RL minus the reward — same scheduler, same rollout engine, same fault tolerance.

> [!important] The systems insight
> V4 §5.2.2 reads less like an algorithm paper and more like a *systems paper that happens to make a previously-hypothetical algorithm feasible*. The math (Eq. 29) is one line; the engineering (hidden-state cache + sample-sort + TileLang + FP4 QAT + reused RL infra) is what turns "full-vocab multi-teacher KL at 1.6T scale" from a research dream into a shipping pipeline.

---

## Experiments

V4 is a model release, not a controlled study, so this section reports what the paper publishes and flags what it doesn't.

### What the paper reports

- **Reasoning quality**: V4-Pro is positioned as competitive with GPT-5.2 and Gemini-3.0-Pro on standard reasoning benchmarks (AIME, GPQA, code, etc.). V4-Pro claims SimpleQA SOTA among open-source models.
- **Architecture validation**: 1M context with hybrid CSA + HCA attention, mHC residuals, Muon optimizer — but the architecture story is separate from the OPD story.
- **OPD pipeline validation**: the merge produces a single unified model that the paper claims preserves the per-domain specialist capabilities. No ablation showing "V4 with mixed RL would have been worse" is given.

### What the paper does NOT report

| Missing | Why it matters | Source for circulating claim |
| ------- | -------------- | ---------------------------- |
| GPU-hour comparison: OPD merge vs mixed-RL merge | The substitution's whole point is supposedly efficiency | "10× cheaper" headlines inherit from [TML's Qwen3 blog](https://thinkingmachines.ai/blog/on-policy-distillation/), not from V4 |
| Ablation: V4-with-mixed-RL vs V4-with-OPD | Establishes that OPD is *better*, not just *different* | None — substitution justified qualitatively |
| Concrete weights $w_i$ schedule | Whether routing is learned or hand-tuned | Tech report only says "relative importance" |
| OPD trainer source code | Reproducibility | DeepSeek's `github.com/deepseek-ai` has no V4 / OPD repo as of May 2026 |

### How V4's OPD relates to other OPD deployments

| | GKD (2023) | MiniLLM (2023) | TML OPD (2025) | Qwen3 small (2025) | Nemotron-Cascade 2 MOPD (2026) | **DeepSeek-V4 OPD (2026)** |
| --- | --- | --- | --- | --- | --- | --- |
| Teacher count | 1 | 1 | 1 | 1 | many (per-domain ckpt) | **many** (per-domain specialist) |
| Sampling | $\lambda$ knob | student | student | student | student | **student** |
| KL direction | $\beta$ knob | reverse | reverse | reverse | reverse | **reverse** |
| KL form | token-level | token-level (as PG) | token-level | token-level | token-level | **full-vocabulary** |
| Pipeline role | KD primitive | KD primitive | RL replacement (blog) | replaces RL stages 3–4 | regression recovery in cascade RL | **entire post-training merge stage** |
| Model scale | T5 / PaLM-2 | medium dense | Qwen3-8B | Qwen3 0.6B–30B | 30B-active MoE | **1.6T MoE** |
| Source code | open | open | partial (cookbook) | partial (Qwen) | partial (NVIDIA) | **closed** |

V4 occupies the most aggressive position on every dimension.

---

## Strengths and limitations

The two strongest points: (1) **first flagship-scale, fully-open-weight demonstration** that OPD can entirely replace mixed-RL post-training — this is the canonical reference for 2026 onwards; (2) **full-vocabulary KL as a variance-reduction lever** is a real algorithmic claim, not just a marketing move — the engineering infrastructure to make it feasible at 1.6T scale is itself a contribution.

Where the work is honest about scope but the limits matter:

- **No cost ablation vs GRPO.** The paper publishes no GPU-hour comparison. Any "10× cheaper than RL" claim in V4 coverage is borrowed from TML's Qwen3 blog and **not validated for V4**.
- **No A/B vs mixed-RL.** The substitution is justified qualitatively (avoids weight-merging degradation, lower variance gradients). No data showing OPD beats continued mixed RL on V4-scale.
- **OPD trainer is closed-source.** The V3.2 post-training stack — on which V4's OPD is built — is also closed. Reimplementation requires rebuilding hidden-state-cache + TileLang kernel + FP4 QAT + sample-sort dispatcher from scratch.
- **Specialist routing is opaque.** The $w_i$ weights and the prompt-to-teacher routing logic aren't detailed enough to reproduce.
- **Inherits OPD's structural limits.** All the [[on-policy-distillation#Strengths-and-limitations|limitations of OPD as a technique]] apply: teacher-bounded capability, cold-start fragility, etc. V4's specialists are GRPO-trained from the same base, so the cross-family negative-transfer risk is muted, but the capability ceiling argument still holds.

> [!warning] What's marketing vs what's working
> *Working*: V4 (post-OPD) reaches competitive reasoning numbers. The full-vocab KL + multi-teacher merge recipe is concrete enough to reimplement. *Marketing or unverified*: the cost claim, the "OPD strictly beats mixed RL" implication. Treat any V4 cost claim as inherited from TML, not as a V4 result.

---

## What this means

Three predictions worth tracking:

1. **OPD-as-merge-stage becomes the default for MoE post-training.** V4 is the proof-of-concept. Expect Qwen, Mistral, the open-source community to adopt within 6–12 months. The "specialist-per-domain-or-mode + OPD merge" architecture answers a real problem (capability dilution under unified RL) that anyone training large MoE models faces.
2. **Full-vocabulary KL becomes the default for OPD at scale.** Token-level estimator's variance pathology is real and grows with rollout length — exactly the wrong scaling behavior for 16 K+ token reasoning models. Once the V4 engineering recipe is replicated in HF TRL / veRL / NeMo-RL, expect token-level OPD to be relegated to small-scale prototyping.
3. **The interesting frontier moves to teacher diversity.** Once multi-teacher OPD is standard, the next question is what teachers to merge — same-family specialists (current V4), cross-family (GOLD-style), multi-agent debate (MAD-OPD), or RL reward heads (KDRL). Diversity engineering becomes the differentiator.

What this is *not*: a universal RL killer. V4 still uses GRPO inside Stage 1 specialist training. Frontier capability extension — pushing past the strongest available teacher — still requires RL exploration. V4 demonstrates OPD can *replace the merge*, not *eliminate RL*.

---

## Source code & reproduction

### Public release status

| Artifact | Status |
| -------- | ------ |
| Model weights (V4-Pro, V4-Flash) | ✓ Open, MIT license |
| Technical report | ✓ PDF on HF |
| OPD trainer source | ✗ Not released |
| Hidden-state cache / TileLang KL kernel | ✗ Not released |
| Specialist checkpoints / weights $w_i$ | ✗ Not released |
| V3.2 post-training infrastructure | ✗ Closed (V4's OPD is built on this) |

Implication: V4's OPD recipe is **paper-only**. Reproducing requires reimplementing the full §5.2.2 infrastructure on top of an existing trainer (HF TRL `GKDTrainer`, veRL, NeMo-RL).

### Minimum reproduction path

```python
# Conceptual sketch — full implementation requires §5.2.2 infrastructure
from trl import GKDConfig, GKDTrainer
from transformers import AutoModelForCausalLM

student = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-V4-Base")  # hypothetical

teachers = [
    AutoModelForCausalLM.from_pretrained(f"deepseek-ai/V4-specialist-{d}")
    for d in ["math", "code", "agent", "IF", "alignment", ...]   # 10+ specialists
]
weights = {d: w_d for d, w_d in importance_per_domain.items()}

# V4-style multi-teacher full-vocab KL (NOT in stock TRL — needs custom training_step)
def opd_loss(student_logits, teacher_logits_per_specialist, weights):
    return sum(
        weights[i] * full_vocab_kl(student_logits, teacher_logits_per_specialist[i])
        for i in range(len(teachers))
    )
```

The infrastructure pieces that need building on top of stock TRL:

1. **Multi-teacher logit fetching** — current `GKDTrainer` only supports one teacher.
2. **Hidden-state cache layer** — pre-compute teacher hidden states, store in centralized buffer.
3. **Sample-sort-by-teacher dispatcher** — keep at most one teacher LM head on GPU.
4. **Full-vocab exact KL kernel** — Triton or TileLang implementation; not the JSD branch in `generalized_jsd_loss`.
5. **FP4 teacher QAT** — for storage / loading.

### Closest available reference implementations

| Project | What it gives you | Gap to V4 |
| ------- | ----------------- | --------- |
| [HF TRL `GKDTrainer`](https://github.com/huggingface/trl/blob/main/trl/trainer/gkd_trainer.py) | Single-teacher reverse-KL OPD, token-level | Multi-teacher, full-vocab, infra missing |
| [veRL `algo/opd`](https://verl.readthedocs.io/en/latest/algo/opd.html) | Multi-teacher routing via `data_source` | Token-level only; no hidden-state cache |
| [NeMo-RL OPD (#1445)](https://github.com/NVIDIA-NeMo/RL/discussions/1445) | Top-k restricted KL (bandwidth opt) | Closest to full-vocab; no published multi-teacher pipeline |
| [Tinker cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) | Single + multi-teacher OPD recipes | Token-level, no full-vocab path |

---

## Related reading

- [[on-policy-distillation]] — Umbrella page on OPD (origin paper, math, variants, debate); V4 OPD is the flagship-scale instantiation.
- [[grpo]] — Used in V4 Stage 1 specialist training; replaced in Stage 2 merge.
- [[ppo-for-llm]] — The trust-region intuition underlying both stages' KL regularization.
- [[rlhf-overview]] — Standard post-training pipeline V4 disrupts at the merge stage.
- [[parallelism-strategies-deep-dive#13. DeepSeek-V3 Case Study]] — V3 architecture and parallelism that V4 builds on.
- [[kv-cache-optimization]] — KV management at 1M context, relevant to V4 inference infrastructure.
- [[das-spec-rl]] — Speculative-decoding speedup for the rollout phase of any GRPO / OPD pipeline; complementary at the inference layer.

## References

- **DeepSeek-V4 technical report**: [HF PDF](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf) (no arXiv)
- **DeepSeek-V4-Pro model card**: [huggingface.co/deepseek-ai/DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
- **DeepSeek-V4-Flash model card**: [huggingface.co/deepseek-ai/DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
- **API changelog**: [api-docs.deepseek.com/updates](https://api-docs.deepseek.com/updates)
- **GKD (the OPD origin paper, cited by V4)**: Agarwal et al., ICLR 2024. [arXiv:2306.13649](https://arxiv.org/abs/2306.13649)
- **MiniLLM (cited by V4)**: Gu et al., NeurIPS 2024. [arXiv:2306.08543](https://arxiv.org/abs/2306.08543)
- **Thinking Machines Lab — On-Policy Distillation blog** (cited by V4): [thinkingmachines.ai/blog/on-policy-distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)
- **Independent V4 writeups**: [Lukyanenko](https://artgor.medium.com/deepseek-v4-review-why-million-token-context-needs-efficient-attention-not-just-larger-windows-6dc8e74a00b1) · [OutcomeSchool](https://outcomeschool.com/blog/decoding-deepseek-v4) · [BSWEN](https://docs.bswen.com/blog/2026-04-25-deepseek-v4-two-stage-post-training/) · [qingkeai.online](https://qingkeai.online/archives/DeepSeek-V4-OPD) · [Fireworks AI](https://fireworks.ai/blog/what-deepseek-v4-says-about-training-platforms)
- **Press**: [CNBC](https://www.cnbc.com/2026/04/24/deepseek-v4-llm-preview-open-source-ai-competition-china.html) · [MIT Technology Review](https://www.technologyreview.com/2026/04/24/1136422/why-deepseeks-v4-matters/) · [Bloomberg](https://www.bloomberg.com/news/articles/2026-04-24/deepseek-unveils-newest-flagship-a-year-after-ai-breakthrough)
