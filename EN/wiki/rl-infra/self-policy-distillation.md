---
title: "SPD: Self-Policy Distillation via Capability-Selective Subspace Projection"
category: rl-infra
tags: [self-policy-distillation, spd, self-distillation, opd, subspace-projection, kv-steering, paper-review]
created: 2026-05-22
updated: 2026-05-22
status: mature
paper: arXiv:2605.22675
code: not-yet-released
---

# SPD: Self-Policy Distillation via Capability-Selective Subspace Projection

> [!info] Paper metadata
> - **Paper**: [arXiv:2605.22675](https://arxiv.org/abs/2605.22675) — *Self-Policy Distillation via Capability-Selective Subspace Projection*, 2026-05-21 preprint
> - **Authors**: Guangya Hao¹, Yitong Shang¹², Yunbo Long¹, Zhuokai Zhao³†, Hanxue Liang¹†
> - **Affiliations**: ¹University of Cambridge, ²HKUST, ³University of Chicago (†joint last author)
> - **Code**: not yet released as of preprint
> - **Correspondence**: hl589@cantab.ac.uk, zhuokai@uchicago.edu, ytshang@ust.hk

---

## Summary (read this if you have 2 minutes)

**What it is.** SPD is a **teacher-free self-distillation** method that improves an LLM by training on its own generations — but it first **steers** those generations through a learned KV-activation projection so they're concentrated on the *capability* one wants to improve, not on stylistic artifacts or model-specific errors. No external verifier, no reward model, no RL.

**The one idea.** Existing self-distillation has two failure modes: (a) it needs external signals (correctness filters, exec feedback, reward search) that are expensive and unavailable for frontier models, or (b) it trains on raw outputs and inherits the model's bad habits because *self-generated outputs entangle capability with formatting/style/errors*. SPD's answer is to identify a **low-rank capability subspace** in KV-activation space — extracted via SVD of gradients on **correctness-defining tokens** in a small calibration set — and project KV through it during self-generation. The hooked model produces cleaner outputs; standard next-token SFT on those outputs is enough. Two phases: **Phase 1** extracts $P_K^{(\ell)}, P_V^{(\ell)}$ from calibration gradients; **Phase 2** generates with $\tilde K = K P_K, \tilde V = V P_V$ hooks and fine-tunes the unhooked model on the resulting $(q, \hat y)$ pairs.

**Headline result.** On Qwen2.5-0.5B-Instruct across 3 domains (code / math / QA) and 6 benchmarks: average **+8.9 % vs base, +9.3 % vs Plain Self-Retraining, +6.4 % vs Simple Self-Distillation**. Up to **+13 % vs SOTA self-distillation without external signals** and **+16 % vs pretrained baseline**. Crucially, the **out-of-domain transfer** result: calibrating the subspace on QA alone (MMLU) lifts GSM8K from 11 → 26 % and SVAMP from 16 → 21 % — *the capability filter generalizes beyond its calibration domain*.

**Why it matters.**

- **Self-distillation without a verifier.** The only previous teacher-free self-distillation with strong results (SSD, Zhang et al.) was code-domain specific. SPD generalizes to math and QA with one mechanism.
- **Subspace steering as a primitive.** Treating "what the gradient on correctness tokens looks like" as a *subspace you can project into* extends representation-engineering (RepE, ITI) from inference-time steering into a training-data-generation tool.
- **Frontier-model friendly.** When no stronger teacher exists, OPD ([[on-policy-distillation]]) hits its imitation-learning ceiling. SPD provides a self-improvement path that doesn't need one.
- **The critical ablation.** Full-sequence loss for subspace extraction recovers only MBPP 11.9 % (worse than base!), while correctness-aligned loss gets 25.5 %. The "which tokens count" choice is load-bearing — not a polish.

---

# Depth (drill-down starts here)

## Background: why teacher-free self-distillation kept stalling

The post-training landscape SPD positions against:

| Lane | Distribution | Supervision source | Failure mode |
| ---- | ------------ | ------------------ | ------------ |
| **Off-policy KD** (Hinton 2015, sequence-KD) | Teacher rollouts $y \sim f_T$ | Teacher logits | Train-inference distribution mismatch → compounding errors |
| **On-policy distillation** ([[on-policy-distillation\|OPD]] / GKD / MiniLLM) | Student rollouts $y \sim \pi_\theta$ | Teacher logits per token | Needs a teacher; capped at teacher ceiling; expensive teacher serving |
| **Self-distillation with external signal** | Student rollouts | Correctness filter / verifier / reward search / exec feedback | Signal cost + infrastructure; not available for frontier models |
| **Simple Self-Distillation (SSD)** (Zhang et al.) | Student rollouts (truncation-decoded) | Just train on raw outputs | Domain-specific (code only); doesn't generalize |
| **SPD (this paper)** | Student rollouts through KV projection hook | Just train on the steered outputs | — |

The unifying critique SPD makes: **self-generated outputs are mixed supervision**. A 0.5B-Instruct model asked to solve MBPP produces (i) the answer logic, (ii) verbose explanations, (iii) format artifacts, (iv) model-specific errors. Training on all of it dilutes the capability signal; filtering after-the-fact (SSD-style truncation) is too coarse. SPD's move: **steer the generation itself** so the capability axis dominates and the noise axes are suppressed.

Why "without external signal" is load-bearing: for frontier models (GPT-5, Claude Opus 4.x, DeepSeek-V4), correctness filters and reward models are either expensive (need an even-stronger judge) or simply not available — there is no oracle for capability beyond the model itself. SPD targets exactly this regime.

## The method in detail

![SPD two-phase overview, paper Fig. 2](EN/wiki/rl-infra/self-policy-distillation-figs/spd-overview.png)

Two phases. Phase 1 derives the projection matrices once; Phase 2 uses them as inference hooks and then fine-tunes.

### Phase 1 — Capability Subspace Extraction

**Calibration set.** $D_{\text{cal}} = \{(q^{(i)}, y^{(i)})\}_{i=1}^{N_{\text{cal}}}$ — prompts paired with correct answers. Small: **20-500 examples is enough** (Fig. 4 in the paper; performance stable across this range).

**Correctness-aligned loss.** Instead of computing the standard next-token loss over the entire output, SPD masks all tokens except a set $S^{(i)}$ of **correctness-defining positions** — tokens whose prediction is "directly tied to task success" (paper App. A.1 has the per-domain rules):

$$
\mathcal{L}_{\text{align}}(q^{(i)}, y^{(i)}) = -\frac{1}{|S^{(i)}|} \sum_{t \in S^{(i)}} \log p_{\theta_{\text{old}}}(z_t^{(i)} \mid z_{<t}^{(i)})
$$

This is what concentrates the gradient on capability-relevant directions rather than stylistic noise. **It's also where the work hides** — the choice of $S$ is in the appendix; main text doesn't expose how MBPP / GSM8K / MMLU positions are picked.

**Gradient collection.** For each calibration example, single forward + single backward pass on the **frozen** student. At each target layer $\ell \in \mathcal{L}$, collect token-level gradients of $\mathcal{L}_{\text{align}}$ w.r.t. the K and V activations:

$$
g_{K,t}^{(\ell,i)} = \frac{\partial \mathcal{L}_{\text{align}}(q^{(i)}, y^{(i)})}{\partial K_t^{(\ell,i)}} \in \mathbb{R}^{d_k}, \qquad
g_{V,t}^{(\ell,i)} = \frac{\partial \mathcal{L}_{\text{align}}(q^{(i)}, y^{(i)})}{\partial V_t^{(\ell,i)}} \in \mathbb{R}^{d_v}
$$

Although gradients are defined for all token positions, only those in $S^{(i)}$ receive task-relevant signal — the rest are masked.

**Stack and SVD.** Concatenate across all calibration tokens to form $G_K^{(\ell)}, G_V^{(\ell)} \in \mathbb{R}^{M \times d_k}$ where $M = \sum_i T_i$. Each row is a token-level gradient direction in KV feature space. SVD each:

$$
G_K^{(\ell)} = U_K^{(\ell)} \Sigma_K^{(\ell)} V_K^{(\ell)\top}
$$

Keep the top-$r$ right-singular vectors $V_{K,r}^{(\ell)} \in \mathbb{R}^{d_k \times r}$. These are the dominant directions in K-feature space along which the correctness-aligned loss varies. The **rank-$r$ orthogonal projector** is:

$$
P_K^{(\ell)} = V_{K,r}^{(\ell)} V_{K,r}^{(\ell)\top} \in \mathbb{R}^{d_k \times d_k}
$$

Same construction for $P_V^{(\ell)}$. These projectors are **computed once** and frozen.

> [!example] Why SVD here makes sense
> The gradient $\partial \mathcal{L}_{\text{align}} / \partial K_t$ points in the K-activation direction that, if changed, would change task success the most. The top-$r$ singular vectors of all such gradients stacked are the *consensus* directions across the calibration set — the K-axes that consistently matter for correctness. Projecting K onto this $r$-dim subspace **keeps the capability-aligned variance and zeros out the rest** (style, format, model-specific noise — directions the gradient doesn't care about).

### Phase 2 — Capability-Selective Distillation

**Projection hooks during self-generation.** Insert at target layers $\ell \in \mathcal{L}$. At each forward pass:

$$
\tilde K^{(\ell)} = K^{(\ell)} P_K^{(\ell)}, \qquad \tilde V^{(\ell)} = V^{(\ell)} P_V^{(\ell)}
$$

The hooked model $f_{\theta_{\text{old}}}^{\text{hook}}$ then runs standard autoregressive generation: $\hat y \sim f_{\theta_{\text{old}}}^{\text{hook}}(\cdot \mid q)$ for each training prompt. **The model parameters are not modified** — the hook is purely an inference-time activation rewrite.

**Default target layers.** Middle and last: $\mathcal{L} = \{L, \lfloor L/2 \rfloor\}$. This is motivated by prior work showing intermediate representations carry useful signal; the choice is left to the user.

**Fine-tune on the steered corpus.** Remove hooks; apply LoRA to the *original* $f_{\theta_{\text{old}}}$ with standard next-token loss on $(q^{(i)}, \hat y^{(i)})$ pairs:

$$
\min_\theta \;-\mathbb{E}_{q \sim D, \hat y \sim f_{\theta_{\text{old}}}^{\text{hook}}(\cdot \mid q)} \left[ \sum_t \log p_\theta(z_t \mid z_{<t}) \right], \quad z = T(q, \hat y)
$$

That's it. No KL term, no teacher logits, no reward, no RL. The whole novelty lives in the *data-generation* step; fine-tuning is vanilla SFT-on-self-rollouts.

**Compact view.** $f_{\theta_{\text{old}}} \xrightarrow{T_{\theta_{\text{old}}}^{\text{hook}}} f_{\theta_{\text{old}}}^{\text{hook}} \xrightarrow{\text{generate}} \hat y \xrightarrow{\text{distill}} f_\theta$. SPD is "distilling an internally-selected version of your own policy back into yourself."

### Where SPD sits in the distillation landscape

The paper's own framing equation (Eq. 1): off-policy and on-policy distillation both minimize a teacher-KL across rollouts $y \sim f_{\text{roll}}$ where $f_{\text{roll}} \in \{f_T, D_{\text{offline}}, f_{\theta_{\text{old}}}\}$ — *all need an external teacher* $f_T$. Self-distillation (Eq. 2) drops the teacher but adds an external scoring signal $S(q, y)$ — *still external*. SPD's contribution is replacing $S(q, y)$ (an output-level filter) with $T_{\theta_{\text{old}}}^{\text{hook}}$ (an activation-level transformation):

| Method | Rollout source | Supervision | External thing needed |
| ------ | ------------- | ----------- | --------------------- |
| Off-policy KD | $y \sim D_{\text{offline}}$ | Teacher logits | Teacher $f_T$ + dataset |
| [[on-policy-distillation\|OPD]] / GKD | $y \sim \pi_\theta$ | Teacher logits | Teacher $f_T$ |
| Self-distill + filter | $y \sim \pi_\theta$ | Just $\log p_\theta(y)$, weighted by $S(q,y)$ | Verifier / RM / exec env |
| SSD (truncation) | $y \sim \pi_\theta$ (truncated decoding) | Just $\log p_\theta(y)$ | None (but code-only) |
| **SPD** | $y \sim \pi_\theta^{\text{hook}}$ (steered) | Just $\log p_\theta(y)$ | Small labeled calibration set (20-500 examples) |

The "external thing" for SPD is *much smaller*: 20-500 labeled examples to extract a subspace, no inference-time judge or RM. The trade-off is that the subspace is per-capability — you need a calibration set per target capability.

## Headline evidence

**Setup.** 5 backbones across 2 model families and 3 sizes:

- Qwen2.5-0.5B / 7B / 14B-Instruct
- Qwen3-4B-Instruct
- Llama-3.1-8B-Instruct

6 datasets across 3 capability domains: **Code** (MBPP, CodeAlpaca-20k), **Math** (GSM8K, SVAMP), **QA** (MMLU, BBH). Metrics: Pass@1 (MBPP), exact-match (GSM8K, SVAMP), letter accuracy (MMLU), normalized exact-match (BBH), NLL (CodeAlpaca, lower is better).

Baselines: **Base** (pretrained student), **PSR** (Plain Self-Retraining on raw self-outputs), **SSD** (Simple Self-Distillation, Zhang et al.).

### Main result — Qwen2.5-0.5B-Instruct (Table 1 / Fig. 1 highlights)

| Dataset | Base | PSR | SSD | **SPD** | Δ vs Base |
| ------- | ---: | --: | --: | ------: | --------: |
| MBPP (code) | 17.0 % | 29.0 % | 18.3 % | **25.5 %** | +8.5 pp |
| GSM8K (math) | 11.0 % | 17.0 % | 12.0 % | **22.0 %** | +11.0 pp |
| MMLU (QA) | 46.0 % | 43.0 % | 48.0 % | **49.0 %** | +3.0 pp |
| SVAMP (math, in-domain) | 16.0 % | 19.0 % | 16.0 % | **32.0 %** | +16.0 pp |
| BBH (QA, in-domain) | 32.7 % | 33.7 % | 36.0 % | **38.7 %** | +6.0 pp |
| CodeAlpaca NLL ↓ | 0.683 | 0.683 | 0.682 | **0.679** | better |

> [!success] The 5-backbone average
> SPD averages **+8.9 % over Base, +9.3 % over PSR, +6.4 % over SSD** across Qwen2.5-0.5B / 7B / 14B, Qwen3-4B, Llama-3.1-8B. The 0.5B model is the largest delta; larger models gain less (+2.1 % on Qwen3-4B, +1.9 % on Llama-3.1-8B). This is consistent with the general pattern that smaller models have more room to be improved by data-curation tricks.

### Out-of-domain transfer (Table 2)

The capability-selectivity claim. Extract the subspace from **QA calibration only** (MMLU); evaluate on math and code:

| Method | MMLU | BBH | GSM8K | SVAMP | MBPP | CodeAlpaca ↓ |
| ------ | ---: | --: | ----: | ----: | ---: | -----------: |
| Base | 46.0 % | 32.7 % | 11.0 % | 16.0 % | 17.0 % | 0.683 |
| SSD-QA | 48.0 % | 36.0 % | 19.0 % | 14.0 % | 12.0 % | 0.676 |
| **SPD-QA** | **49.0 %** | **38.7 %** | **26.0 %** | **17.0 %** | **21.0 %** | 0.680 |

A subspace built only from QA gradients lifts GSM8K by **+15 pp** and MBPP by **+4 pp** — *the capability filter generalizes*. The authors' framing: "the calibration domain determines the extracted capability subspace, allowing us to steer self-generation toward a target capability and enable more effective transfer." A more sober reading: QA reasoning shares low-rank structure with math and code reasoning in KV activation space, and the subspace captures that shared structure.

### The critical ablation — correctness-aligned vs full-sequence loss (Table 5)

This is the load-bearing experiment. SPD with full-sequence loss (compute gradient on *all* output tokens, not just correctness-defining ones) vs the correctness-aligned version:

| Method | MBPP | CodeAlpaca ↓ | GSM8K | SVAMP | MMLU | BBH |
| ------ | ---: | -----------: | ----: | ----: | ---: | --: |
| Base | 17.0 % | 0.683 | 11.0 % | 16.0 % | 46.0 % | 32.7 % |
| SPD w/ Full-sequence | **11.9 %** | 0.681 | 13.0 % | 24.0 % | 48.0 % | 35.7 % |
| **SPD w/ Correctness-aligned** | **25.5 %** | 0.679 | 22.0 % | 32.0 % | 49.0 % | 38.7 % |

> [!important] Full-sequence loss is worse than the Base model on MBPP
> 17 % → 11.9 %. The "which tokens count" choice is doing the work, not the SVD machinery alone. Without correctness-aligned masking, the gradient picks up stylistic and formatting variance and the projector projects *onto* that noise. The "capability" framing requires the correctness mask.

> [!example]- Full ablation results (drill-down)
>
> **Self-generated data quality before fine-tuning** (Table 3): SPD-generated outputs beat both Base- and SSD-generated outputs on most benchmarks *even without the fine-tuning step* — GSM8K 18 % (vs SSD 10 %, Base 11 %), BBH 35.3 % (vs SSD 28.7 %, Base 32.7 %). The hook alone improves the model's generation quality; fine-tuning compounds the gain.
>
> **Calibration size sensitivity** (Fig. 4): SPD is data-efficient — 20-500 calibration examples all give comparable performance. GSM8K 19-22 %, SVAMP 26-32 %, MMLU 48-50 %, BBH 32-37 % across this range. The subspace estimation is stable with very few labeled examples.
>
> **Fine-tuning analysis** (Table 4): PSR overfits MBPP (29.0 %) but fails to transfer to CodeAlpaca (NLL 0.683). SSD is weak on MBPP (18.3 %) and shows no CodeAlpaca improvement. SPD balances at MBPP 25.5 % + best CodeAlpaca (0.679).
>
> **Qualitative example** (Fig. 3): Base output for MBPP `remove_Occ` is verbose with explanations and print statements; SSD output is shorter but still has incorrect logic; SPD output is the most compact and removes both decorative text and the logic error. The hook *visibly* changes the generation style toward task focus.

## Strengths and limitations

The two genuine strengths: (1) **no external signal needed** at training time (only a small calibration set), making it viable for frontier-scale models with no available oracle; (2) **cross-domain transfer of the subspace** — calibrating on QA helps math/code — suggesting the "capability subspace" captures structural reasoning patterns, not just dataset-specific surface features.

Where I'd push back:

- **Small-model showcase.** Qwen2.5-0.5B has the +8.9 % headline; Qwen3-4B has +2.1 %; Llama-3.1-8B has +1.9 %. The method's benefit decays with scale in the reported range. Whether the trick survives at 70B+ (where capability is already high and self-distillation matters most for safety/style cleanup) is untested.
- **"Correctness-defining tokens" is in the appendix.** The single most important hyperparameter — which tokens count for $S^{(i)}$ — gets a one-line main-text mention pointing to App. A.1. For math the answer position is obvious; for code (which tokens of `def remove_Occ(s, c):` matter?) and QA (the letter? the reasoning chain?) it's not. The framework's reproducibility hinges on per-domain rules the main text doesn't lay out.
- **Rank $r$ is silent.** The main text never states the chosen rank $r$ for $V_{K,r}^{(\ell)}$. Likely sensitive: too small and the subspace can't represent the capability; too large and the noise creeps back. This goes in App. A.3 but should be a first-class ablation.
- **Two layers only.** Default $\mathcal{L} = \{L, \lfloor L/2 \rfloor\}$. No ablation on which layers matter, no exploration of all-layer projection or attention-layer-only. The "middle + last" choice has prior-work backing but isn't justified for *this* method specifically.
- **SVD cost not characterized at scale.** On a 70B model with $d_k \approx 128$ per head and many calibration tokens, the gradient stack is small per head ($M \times 128$), so SVD is cheap. But the gradient *collection* requires one backward pass per calibration example through a frozen model — fine for 20-500 examples, but the per-domain calibration adds up if you want many capabilities.
- **No comparison to on-policy distillation.** The natural counter-experiment is "use the *same* model as both student and teacher in OPD/GKD with $\lambda=1$." That's the cleanest "self-distillation" baseline and SPD doesn't run it.
- **"Self-policy" name is misleading.** There's no policy gradient or RL anywhere. It's SFT on activation-steered self-rollouts. "Self-distillation with capability-subspace steering" would be more honest. The "policy" framing seems chosen to ride the [[on-policy-distillation|OPD]] / [[grpo|GRPO]] terminology wave.
- **No GitHub yet.** As of the May 2026 preprint there's no code release, so the App. A.1 token-selection rules and App. A.3 hyperparameters aren't independently verifiable.

> [!warning] Author-acknowledged limitation
> "SPD is evaluated across three capability domains and multiple backbones; broader validation on more diverse and high-stakes tasks would further strengthen its empirical robustness." Translation: math/code/QA is well-covered, but agentic tasks, long-context reasoning, multi-turn dialog, multilingual all untested.

## What this means

The interesting framing here, beyond the method itself: **self-distillation works in proportion to how cleanly you can separate the capability axis from the noise axes in your own generations**. SSD's trick was truncation-decoding (a coarse output-level filter); SPD's trick is activation-level projection (a more surgical filter). Both work because they reduce the contamination of self-generated training data with style/format/error variance.

Three predictions for 12 months:

1. **Activation steering becomes a standard data-curation tool.** Once people accept that "self-generated outputs are mixed supervision", more methods will operate on the generation process itself rather than post-hoc filtering. Expect representation engineering (RepE, ITI, CAA) and SPD-style gradient subspaces to merge into a unified toolkit for self-improvement data generation.
2. **The correctness-token selection becomes its own subfield.** SPD's biggest hidden hyperparameter — "which tokens count as capability-defining" — will get more attention. Likely directions: learn it from a small reward model, derive it from chain-of-thought structure, use information-theoretic measures.
3. **SPD-OPD hybrids.** SPD's subspace can serve as a *teacher-side* filter inside OPD: instead of distilling from the teacher's raw distribution, distill from the teacher's *steered* distribution. This compounds capability concentration: teacher gives the signal, SPD gives the focus. Don't be surprised if the next round of OPD variants (KDRL, dGRPO, MOPD — see [[on-policy-distillation#Variant taxonomy]]) incorporates this.

What this is *not*: a frontier-capability-extension method. SPD compresses the model toward its own capability subspace — that's a regularization toward "do what you're already good at, less noisily." It can't teach the model things the calibration data doesn't already span. For genuine capability gain (math that the base model can't solve), RL with verifiable rewards or [[on-policy-distillation|OPD]] with a stronger teacher remain necessary.

## Source code & reproduction

The paper has **no public code release** as of the May 2026 preprint. Key implementation details from the paper, sufficient to attempt reproduction:

| Component | Setting |
| --------- | ------- |
| Calibration set size | 20-500 examples (Fig. 4 shows stability across this range) |
| Target layers | $\mathcal{L} = \{\lfloor L/2 \rfloor, L\}$ — middle and last |
| Projection scope | Both K and V at each target layer |
| Loss for subspace extraction | Correctness-aligned NLL on $S^{(i)}$ tokens (Eq. 6) |
| Subspace rank $r$ | Unspecified in main text, App. A.3 |
| Backbones | Qwen2.5-0.5B/7B/14B, Qwen3-4B, Llama-3.1-8B (all -Instruct) |
| Fine-tuning method | LoRA |
| Fine-tuning loss | Standard next-token NLL on $(q^{(i)}, \hat y^{(i)})$ |

Pseudocode of the full pipeline:

```python
# Phase 1 — Extract projection matrices (once)
def extract_subspace(model, cal_set, target_layers, rank_r):
    grad_K = {ell: [] for ell in target_layers}
    grad_V = {ell: [] for ell in target_layers}
    for q, y in cal_set:
        loss = correctness_aligned_loss(model, q, y)  # masks non-S tokens
        loss.backward()
        for ell in target_layers:
            grad_K[ell].append(model.layers[ell].K.grad)  # shape [T, d_k]
            grad_V[ell].append(model.layers[ell].V.grad)
        model.zero_grad()
    P_K, P_V = {}, {}
    for ell in target_layers:
        G_K = torch.cat(grad_K[ell], dim=0)               # [M, d_k]
        _, _, V_K = torch.linalg.svd(G_K, full_matrices=False)
        V_K_r = V_K[:rank_r].T                            # [d_k, r]
        P_K[ell] = V_K_r @ V_K_r.T                        # [d_k, d_k]
        # same for V
    return P_K, P_V

# Phase 2 — Generate with hooks, then fine-tune
def spd(model, cal_set, train_prompts, target_layers, rank_r):
    P_K, P_V = extract_subspace(model, cal_set, target_layers, rank_r)
    hooks = install_projection_hooks(model, P_K, P_V, target_layers)
    self_gen = [(q, model.generate(q)) for q in train_prompts]
    remove_hooks(hooks)
    finetune_lora(model, self_gen, loss="next_token_nll")
    return model
```

The interesting engineering question: hook ordering relative to RoPE / KV-cache writes. The paper doesn't specify whether $\tilde K = K P_K$ happens before or after RoPE — probably after (to preserve positional encoding), but unspecified.

## Related reading

- [[on-policy-distillation]] — The teacher-required cousin; SPD is structurally similar to OPSD (self-distillation OPD) but adds the subspace steering layer. SPD's variant-table entry would live in [[on-policy-distillation#Variant taxonomy]].
- [[deepseek-v4-opd]] — The flagship multi-teacher full-vocab OPD; contrasts with SPD's teacher-free single-model approach.
- [[mopd]] — NVIDIA's Multi-Domain OPD; both papers care about per-domain calibration, but MOPD switches teachers while SPD switches subspaces within one model.
- [[grpo]] — The RL alternative; SPD targets the same "improve self-rollouts" problem without the reward model or value head overhead.
- [[rlhf-overview]] — The pipeline SPD partially displaces (when the goal is capability sharpening, not preference alignment).

## References

- Paper: Hao et al., *Self-Policy Distillation via Capability-Selective Subspace Projection*, 2026-05-21. [arXiv:2605.22675](https://arxiv.org/abs/2605.22675)
- SSD (the baseline): Zhang et al., simple self-distillation for code. Cited as [3] in SPD.
- On-policy distillation: [arXiv:2306.13649](https://arxiv.org/abs/2306.13649) (GKD), [arXiv:2306.08543](https://arxiv.org/abs/2306.08543) (MiniLLM).
- Representation engineering background — RepE, ITI, CAA — for the activation-steering lineage SPD extends into the training-data-generation regime.
