---
title: "On-Policy Distillation (OPD): Dense Teacher Signal as an RL Replacement"
category: rl-infra
tags: [on-policy-distillation, opd, gkd, minillm, distillation, rl-post-training, reverse-kl, knowledge-transfer]
created: 2026-05-19
updated: 2026-05-19
status: mature
---

# On-Policy Distillation (OPD): Dense Teacher Signal as an RL Replacement

> [!info] Page scope
> Covers (1) the **technique** and its lineage (GKD / MiniLLM → Thinking Machines Lab's "OPD" reframing), (2) the **policy-gradient duality** that makes OPD viewable as a special case of GRPO with teacher-log-ratio rewards, (3) the **variants** in active 2025–2026 use (OPSD, KDRL, MOPD, MAD-OPD, Black-Box OPD), (4) **source code** in HF TRL / veRL / NeMo-RL, (5) **production deployments** (NVIDIA Nemotron-Cascade 2, Alibaba Qwen3 small models), and (6) the live debate over whether OPD replaces or merely warm-starts RL post-training. For DeepSeek-V4's specific multi-teacher full-vocabulary OPD recipe see [[deepseek-v4-opd]].

> [!abstract]+ TL;DR
> **On-Policy Distillation** is reverse-KL distillation where the student samples its own rollouts and a frozen teacher scores them per token. Mathematically, it is **policy gradient with a dense per-token reward equal to $\log(\pi_T/\pi_\theta)$** — same shape as [[grpo|GRPO]] minus the sparse outcome reward and value head, plus a KL-to-teacher signal that is *both* the reward and the trust-region regularizer. The headline practitioner claim from Thinking Machines Lab (Oct 2025) is **74.4 % AIME'24 at ~1,800 GPU-hours** for Qwen3-8B-Base (vs Qwen3's own RL recipe at 67.6 % / ~17,920 GPU-h) — a ~10× compute-efficiency improvement, sometimes quoted as 50–100× when self-distilling. The technique is **not new** — it is the $\lambda=1$, reverse-KL configuration of [GKD (Agarwal et al., 2023)](https://arxiv.org/abs/2306.13649) and the reverse-KL formulation of [MiniLLM (Gu et al., 2023)](https://arxiv.org/abs/2306.08543). What IS new is its **practitioner repositioning as an RL-stage replacement** for reasoning post-training. As of mid-2026, the strongest production deployments are **NVIDIA Nemotron-Cascade 2** (MOPD interleaved with cascade RL), **Alibaba Qwen3** small models, and **DeepSeek-V4** (which uses multi-teacher full-vocabulary OPD to *entirely replace* V3.2's mixed-RL stage — see [[deepseek-v4-opd]]).

---

## The origin story

### What problem it claims to solve

The framing in the [Thinking Machines Lab blog](https://thinkingmachines.ai/blog/on-policy-distillation/) (Kevin Lu, 2025-10-27): there are two existing lanes for LLM post-training, and both have a structural flaw:

- **SFT / off-policy KD** is *off-policy* — you train on a fixed corpus of teacher-generated trajectories, but the deployed student visits a *different* state distribution. As the student drifts during training, the gradient direction becomes biased w.r.t. states the deployed student will actually visit, producing compounding errors and style mimicry without behavioral transfer.
- **RL ([[grpo|GRPO]], [[ppo-for-llm|PPO]], DPO)** is *on-policy* — relevant trajectories, but the reward signal is **O(1) bits per episode**. Most of a 16 K-token reasoning rollout receives no per-token credit assignment; the model must learn what made the rollout correct from a single sparse reward. This is expensive.

OPD's pitch: **get on-policy relevance with token-dense supervision** by replacing the sparse scalar reward with the teacher's per-token log-probabilities. Every token gets graded; you keep the on-policy state distribution; you skip the value head and reward model entirely.

### Lineage — not new, but freshly weaponized

The TML blog itself credits two ancestors:

- **GKD — *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes*** (Agarwal, Vieillard, Zhou, Stańczyk, Ramos, Geist, Bachem; DeepMind; ICLR 2024). [arXiv:2306.13649](https://arxiv.org/abs/2306.13649). Introduces a $\lambda$-mixed loss with student-rollout sampling and a generalized JSD that interpolates forward / reverse / mixed KL.
- **MiniLLM — *Knowledge Distillation of Large Language Models*** (Gu, Dong, Wei, Huang; Microsoft / Tsinghua; NeurIPS 2024 — v3 retitled to *MiniLLM: On-Policy Distillation of Large Language Models*). [arXiv:2306.08543](https://arxiv.org/abs/2306.08543). Provides the cleanest formal statement of OPD as a **policy gradient with teacher log-ratio as reward**.

Both papers pre-date the TML blog by **~2.5 years**. The TML contribution is not algorithmic; it is an **opinionated repackaging at Qwen3 scale** with the explicit framing that *most RL post-training is unnecessary if a strong teacher exists*. That single narrative shift — from "KD is a compression technique" to "KD is an RL alternative" — is what made the post viral in late 2025.

> [!note] What's *actually* novel in TML's piece
> The loss is GKD-with-$\lambda{=}1$-and-reverse-KL — exactly one configuration of a 2023 framework. The novelty is in three places: (a) the **production demo at 8B scale with Qwen3-32B teacher**, (b) the framing of **self-distillation as a continual-learning / forgetting-mitigation tool** (mid-train on domain documents, then OPD with a snapshot of the pre-mid-train chat model as teacher), and (c) the **"50–100× cheaper than RL" headline** that re-positioned OPD in the cost conversation.

### The headline number

From the TML blog, replicating the Qwen3 small-model recipe ([Qwen3 tech report](https://arxiv.org/abs/2505.09388)):

| Method | AIME'24 score | Compute (GPU-h, ~) | Source |
|--------|--------------|--------------------|--------|
| SFT only (400 K prompts) | 60 % | not reported | TML blog |
| SFT extrapolated to 2 M | ~70 % | not reported | TML blog |
| Qwen3 RL recipe | 67.6 % | ~17,920 | TML blog |
| **On-Policy Distillation** | **74.4 %** | **~1,800** | TML blog |

Student: Qwen3-8B-Base. Teacher: Qwen3-32B. **~10× compute reduction vs RL, with a *better* AIME score**. Qwen3's tech report independently corroborates the magnitude ("only 1/10 of the GPU hours compared to the four-stage training method") for the same distillation pipeline applied to their full 0.6B–14B + 30B-A3B-MoE small-model line.

The blog further claims **50–100×** when self-distilling (teacher is an RL-trained version of the same model). Independent replication of the 100× number is **not** present in the public literature as of May 2026; the defensible band based on Qwen3's report is 5–20× depending on how teacher inference cost is amortized.

---

## The algorithm

### Loss function

Single-teacher reverse-KL on-policy distillation, per the TML / GKD-$\lambda=1$ formulation:

$$
\mathcal{L}_{\text{OPD}}(\theta) \;=\; \mathbb{E}_{x,\,y\sim\pi_\theta(\cdot\mid x)}\!\left[\sum_{t=1}^{|y|} D_{\text{KL}}\!\big(\pi_\theta(\cdot\mid y_{<t}, x) \,\big\|\, \pi_T(\cdot\mid y_{<t}, x)\big)\right]
$$

Two design choices baked in:

- **On-policy**: $y \sim \pi_\theta$, sampled at *each training step* from the *current* student. Off-policy variants (sample once and reuse) reduce to SFT-with-soft-targets.
- **Reverse KL** (mode-seeking): the student concentrates mass on teacher-likely tokens rather than covering the teacher's tails. Forward KL (mean-seeking) is also valid and is what classical SFT-style KD uses; the choice changes student behavior qualitatively. Reverse KL is GKD's $\beta{=}1$ branch.

The GKD generalized JSD covers all of these as special cases:

$$
\mathcal{L}_{\text{GKD}}(\theta) = (1-\lambda)\,\mathbb{E}_{(x,y)\sim \mathcal{D}}\!\left[D(\pi_T\|\pi_\theta)(y\mid x)\right] + \lambda\,\mathbb{E}_{x,\,y\sim \pi_\theta}\!\left[D(\pi_T\|\pi_\theta)(y\mid x)\right]
$$

with $\lambda$ controlling on-policy fraction and $D$ being a generalized JSD parameterized by $\beta$. TML's OPD = GKD with $(\lambda, \beta, \text{direction}) = (1.0,\, \text{N/A},\, \text{reverse KL})$. HuggingFace TRL exposes both as knobs (see [Source code](#source-code) below).

### The policy-gradient duality

This is the result that makes OPD legible to anyone fluent in [[grpo|GRPO]] / [[ppo-for-llm|PPO]]. Per MiniLLM §3:

$$
\nabla_\theta \,\mathbb{E}_{y\sim\pi_\theta}\!\big[D_{\text{KL}}(\pi_\theta\|\pi_T)\big] \;=\; -\,\mathbb{E}_{y\sim\pi_\theta}\!\left[\sum_{t} \nabla_\theta \log\pi_\theta(y_t\mid y_{<t}) \cdot \underbrace{\log\frac{\pi_T(y_t\mid y_{<t})}{\pi_\theta(y_t\mid y_{<t})}}_{\text{dense per-token "reward"}}\right]
$$

This is **vanilla REINFORCE** with the per-token reward replaced by the teacher log-ratio. Three immediate corollaries:

- **OPD inherits PPO's stability properties.** The KL-to-teacher term doubles as a trust-region regularizer (the student is penalized for drifting away from the teacher's support), exactly as PPO's clip or KL term does.
- **OPD is GRPO without baselines or value models.** GRPO already removes the critic by using group-relative advantages; OPD goes one step further and replaces the sparse outcome reward with a dense token-level reward. No critic, no rollout group normalization, no advantage estimation.
- **The discount factor doesn't matter.** Because the reward is dense ($\log\pi_T/\pi_\theta$ at every token), there's no credit-assignment problem to solve via $\gamma$. TML reports empirically that $\gamma=0$ works best — the gradient at token $t$ depends only on the immediate teacher disagreement.

> [!quote] One-liner mental model
> *"On-policy distillation is the GRPO objective where the sparse outcome reward $R(y)\in\{0,1\}$ is replaced by the dense per-token signal $\log(\pi_T/\pi_\theta)$, and the value head is removed because the reward is already token-level."*

### Why "on-policy" matters

The expectation $\mathbb{E}_{y\sim\pi_\theta}$ is over the *student's* trajectory distribution. If you replace this with $\mathbb{E}_{y\sim\mathcal{D}}$ over a fixed corpus (GKD's $\lambda{=}0$), you get standard SFT-with-soft-targets:

| Setting | Loss expectation | What it does |
|---------|------------------|--------------|
| $\lambda = 0$ | $y \sim \mathcal{D}$ (fixed dataset) | SFT with soft targets; off-policy; drift accumulates |
| $\lambda = 1$ | $y \sim \pi_\theta$ (fresh student rollouts) | On-policy distillation; visits the student's actual deployment distribution |
| $0 < \lambda < 1$ | mixture | Mixed-policy; balances stability and on-policy relevance |

The reason on-policy matters is **distribution shift**. The off-policy setting trains the student on teacher-generated states it will never visit; small errors compound during inference because the model has never seen "what happens after my own previous tokens." On-policy distillation **trains the student on the states it will actually visit at inference**, giving every gradient signal practical relevance. This is also the core insight of [DAGGER (Ross, Gordon, Bagnell, 2010)](https://arxiv.org/abs/1011.0686) — see the [debate section](#the-opd-vs-rl-debate) below.

---

## Variants in active 2025–2026 use

| Variant | Origin | What it adds |
|---------|--------|--------------|
| **GKD** ([Agarwal 2023](https://arxiv.org/abs/2306.13649)) | DeepMind / ICLR 2024 | The umbrella formulation. $(\lambda, \beta)$ knobs; $\lambda{=}1$ recovers pure OPD. |
| **MiniLLM** ([Gu 2023](https://arxiv.org/abs/2306.08543)) | Microsoft / Tsinghua / NeurIPS 2024 | Cleanest derivation of OPD-as-PG; teacher-mixed sampling and single-step variance-reduction tricks. |
| **OPD (TML)** ([blog](https://thinkingmachines.ai/blog/on-policy-distillation/)) | Thinking Machines Lab, Oct 2025 | The practitioner framing. Reverse KL, $\gamma{=}0$, "RL replacement" narrative. |
| **OPSD** (On-Policy Self-Distillation) ([Privileged-Info OPD](https://arxiv.org/abs/2602.04942)) | Multiple groups, 2025–26 | Teacher is the *student's own earlier checkpoint* or a *privileged-info* version of it. Useful for continual learning and personalization. |
| **KDRL** ([arXiv:2506.02208](https://arxiv.org/abs/2506.02208)) | Xu, Zhu et al., Jun 2025 | Replaces [[grpo|GRPO]]'s KL-to-old-policy regularizer with reverse-KL-to-teacher; jointly optimizes rule-based reward + OPD signal in one gradient step. |
| **dGRPO** ([surveyed in arXiv:2604.00626](https://arxiv.org/abs/2604.00626)) | 2025–26 | GRPO advantage + per-token OPD loss as a dense auxiliary head; same recipe family as KDRL. |
| **MOPD** (Multi-Domain OPD) ([Nemotron-Cascade 2](https://research.nvidia.com/labs/nemotron/nemotron-cascade-2/)) | NVIDIA, Mar 2026 | Per-domain best-checkpoint teachers; OPD pulls the student toward each domain's optimum to recover regressions during cascade RL. Used in production. |
| **MAD-OPD** ([arXiv:2605.01347](https://arxiv.org/abs/2605.01347)) | 2026 | Multi-agent debate as the teacher signal. Attempts to break the single-teacher capability ceiling. Research-stage. |
| **Asymmetric / Reward-Extrapolated OPD** ([arXiv:2602.12125](https://arxiv.org/abs/2602.12125), [arXiv:2605.06387](https://arxiv.org/abs/2605.06387)) | 2026 | Adds an RL reward head so the student can *learn beyond the teacher*. Direct response to the teacher-ceiling critique. |
| **Black-Box OPD (GAD)** ([arXiv:2511.10643](https://arxiv.org/abs/2511.10643)) | Ye, Dong et al., Nov 2025 | OPD when only completions (no logits) are available — relevant for OpenAI / Anthropic teachers. Uses an adversarial discriminator to replace teacher log-probs. |
| **Multi-teacher full-vocab OPD** ([DeepSeek-V4 §5.1.2](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)) | DeepSeek, Apr 2026 | $\sum_i w_i D_{\text{KL}}(\pi_\theta \| \pi_{E_i})$ over $> 10$ specialist teachers, with **full-vocabulary logit KL** instead of token-level surrogate. The flagship-scale demonstration. See [[deepseek-v4-opd]]. |

---

## Production deployments (verified)

This section sticks to deployments backed by primary sources. I exclude generic "uses distillation" claims (universal) and only flag *on-policy distillation* with student rollouts and per-step teacher scoring.

### NVIDIA Nemotron-Cascade 2 — MOPD (Mar 2026)

The most production-validated OPD-as-RL-component recipe in the open as of mid-2026. From the [Nemotron-Cascade 2 page](https://research.nvidia.com/labs/nemotron/nemotron-cascade-2/) and [PDF](https://research.nvidia.com/labs/nemotron/files/Nemotron-Cascade-2.pdf): a 30B-active MoE that won IMO / IOI / ICPC gold medals.

**Cascade recipe (per domain $D_k$):**

```
For each domain D_k in {math, code, agent, IF, alignment, ...}:
  (a) GRPO-style RL on D_k                              ← exploration / reward signal
  (b) Save best per-domain checkpoint  →  teacher_D_k
  (c) MOPD: student rolls out across all domains
            visited so far; for each prompt, route to
            that domain's best-checkpoint teacher;
            token-level reverse-KL on student-induced
            states (~30 optimization steps)             ← regression recovery
  (d) Proceed to next domain.
```

**Rationale:** Sequential RL across domains drifts — the student gets better at $D_k$ but regresses on $D_{k-1}, D_{k-2}, \dots$. MOPD pulls it back to per-domain optima cheaply *because the gradient signal is per-token, not per-trajectory* — a 30-step OPD pass recovers most of the lost performance, where re-doing RL on the old domain would cost orders of magnitude more.

This pattern — **RL for exploration, OPD for stability and regression recovery** — is the most architecturally important OPD use case the NVIDIA stack has shown. Cross-link: [[rl-training-frameworks]].

### Alibaba Qwen3 (May 2025)

The *first* large-scale published deployment, predating the TML blog by 5 months. Per the [Qwen3 tech report](https://arxiv.org/abs/2505.09388), Qwen3's 0.6B / 1.7B / 4B / 8B / 14B + 30B-A3B-MoE small-model line replaces stages 3–4 of the full RL pipeline with a two-phase distillation:

```
Stage 1: Pretrain (base model)
Stage 2: Off-policy distillation from larger Qwen3 teacher
         (teacher outputs in both /think and /no_think modes)
Stage 3: On-policy distillation from same teacher
         (student rollouts, teacher logits, reverse KL)
         -- skips stages 3-4 of the full RL pipeline --
```

**Reported cost:** ~1/10 of the four-stage GPU-hour budget per Qwen3's own claim. The TML blog explicitly replicates this recipe with Qwen3-8B-Base as the student, which is where the 74.4 % AIME number comes from.

### DeepSeek-V4 — full pipeline replacement (Apr 2026)

The most aggressive OPD deployment in any flagship model to date — V4's post-training **entirely replaces** the V3.2 mixed-RL stage with multi-teacher full-vocabulary OPD. Full details in [[deepseek-v4-opd]]; the short version is that V4 splits training into per-domain specialists (each trained with GRPO), then merges 10+ specialists into a single unified policy via a weighted reverse-KL objective. The novel piece vs prior OPD work is **full-vocabulary logit KL** instead of token-level surrogate.

### Where on-policy distillation is *not* used

Worth noting for calibration:

- **DeepSeek-R1 → smaller students** uses **SFT-only off-policy distillation** on ~800 K verified traces ("applying only standard SFT without RL", per [the R1 paper](https://arxiv.org/abs/2501.12948)). Off-policy, not OPD. This is V3.2-era; V4 is the OPD pivot point.
- **Meta Llama 4** uses codistillation with a "dynamic soft / hard target weighting" loss but does not document student-rollout-based on-policy distillation in public materials.
- **Anthropic, OpenAI, Mistral, Cohere** — no public evidence of on-policy distillation in their pipelines. Distillation broadly is presumably used internally; "on-policy" specifically is unsupported.

---

## Source code

### HuggingFace TRL — `GKDTrainer`

The canonical open reference implementation. File: [`trl/trainer/gkd_trainer.py`](https://github.com/huggingface/trl/blob/main/trl/trainer/gkd_trainer.py). Docs: [huggingface.co/docs/trl/en/gkd_trainer](https://huggingface.co/docs/trl/en/gkd_trainer).

**Key components (verified in source):**

```python
# trl/trainer/gkd_trainer.py — generalized_jsd_loss (lines 226-295)
def generalized_jsd_loss(
    student_logits,
    teacher_logits,
    labels=None,
    beta=0.5,            # 0 = forward KL, 1 = reverse KL, 0.5 = symmetric JSD
    temperature=1.0,
    reduction="batchmean",
):
    # temperature-scaled log-softmax for both
    # then either pure KL branch (beta == 0 or 1)
    # or mixture via torch.logsumexp([student + log(1-beta), teacher + log(beta)])
    # returning beta * KL(M||teacher) + (1-beta) * KL(M||student)
    ...

# training_step (lines 421-449)
def training_step(self, model, inputs, num_items_in_batch=None):
    # with probability self.lmbda, replace inputs with on-policy rollouts:
    if random.random() <= self.lmbda:
        inputs = self.generate_on_policy_outputs(...)
    # then forward through student + teacher, compute GJSD loss
    ...
```

Configuration knobs (`GKDConfig`):

| Field | Role |
|-------|------|
| `lmbda` | Probability of fresh student rollout per batch. `1.0` = pure OPD; `0.0` = vanilla off-policy KD; in between = mixed. |
| `beta` | KL direction. `0` = forward KL, `1` = reverse KL, `0.5` = JSD. TML's OPD setting: `beta=1.0`. |
| `temperature` | Temperature for log-softmax scaling. |
| `seq_kd` | Generate from the *teacher* instead, giving sequence-level KD on teacher samples. |
| `use_liger_kernel` | Use Liger-fused linear+JSD kernel for memory savings. |

> [!warning] TRL is deprecating GKDTrainer
> Current TRL emits: *"This trainer will soon be moved to `trl.experimental` and is a candidate for removal."* A newer `DistillationTrainer` with a generation buffer (decouples gen batch from train microbatch, ~40× claimed speedup), external teacher server support, and binary-encoded logprob payloads is being developed. Track via [TRL releases](https://github.com/huggingface/trl/releases) and [issue #4390](https://github.com/huggingface/trl/issues/4390).

### veRL — OPD recipe

Docs: [verl.readthedocs.io/en/latest/algo/opd.html](https://verl.readthedocs.io/en/latest/algo/opd.html). Config namespace `distillation.*` with the following key knobs:

- `enabled` — turns OPD on.
- `teacher_models.<name>.model_path` — multi-teacher routing; each teacher keyed by `data_source` (enables MOPD-style per-domain routing).
- `distillation_loss.loss_mode` — `forward_kl_topk` (top-k restricted KL — memory optimization) / `k1` (logp ratio) / `k3`.
- `use_policy_gradient` — toggle between GKD-style direct KL and PG-style (REINFORCE with KL-as-reward).
- `use_task_rewards` — combine with PPO/GRPO outcome rewards (this is the KDRL recipe pattern).

The vLLM-hosted teacher behind a ZeroMQ server is the standard infrastructure pattern: see [Zoey Li's implementation walkthrough](https://zoeyli.com/reinforcement%20learning/implementing-on-policy-distillation/). K1 ($\log\pi_S - \log\pi_T$) and K2 ($0.5(\log\pi_S - \log\pi_T)^2$) estimators only need *scalar* logprobs from the teacher, which sidesteps vLLM's vocab-distribution API cap.

### NVIDIA NeMo-RL

[NVIDIA-NeMo/RL Discussion #1445](https://github.com/NVIDIA-NeMo/RL/discussions/1445) (authors zpqiu, sharonyu-115, shuo-nvidia, sharathts, snowmanwwg) tracks NeMo-RL's OPD support:

- Forward / reverse / mixed KL.
- **Top-k restricted KL** — only the top-k teacher tokens contribute to the loss. Bandwidth optimization that NVIDIA's infra needed to make multi-teacher OPD feasible.
- Student rollouts via vLLM, teacher separately parallelized (higher TP for larger teacher).
- Datasets used in the writeup: DeepScaler (on-policy), AceReason-1.1-SFT (off-policy).
- Reported delta on Qwen3-4B-Base AIME 2025 Avg@16: **47.71 % (SFT+OPD)** vs **30.42 % (SFT+off-policy distillation)**.

Related guides: [NeMo-Aligner KD docs](https://github.com/NVIDIA/NeMo-Aligner/blob/main/docs/user-guide/knowledge-distillation.rst), [NeMo-AutoModel KD guide](https://docs.nvidia.com/nemo/automodel/latest/guides/llm/knowledge-distillation.html).

### Thinking Machines Lab — `tinker-cookbook`

[github.com/thinking-machines-lab/tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) has a distillation recipe under `tinker_cookbook/recipes/distillation/`. Single- and multi-teacher, on-policy and off-policy variants. Multi-turn tool-use distillation example exists. This is the closest thing to TML's reference implementation of the blog's recipe.

### HuggingFace H4 — GOLD (cross-tokenizer OPD)

[GOLD (General On-policy Logit Distillation)](https://huggingface.co/spaces/HuggingFaceH4/on-policy-distillation) extends OPD across tokenizers — distill SmolLM ↔ Llama ↔ Qwen ↔ Gemma via token-merge alignment + product-rule combination of logits. Lands as a `GOLDTrainer` in TRL. Relevant if you want to use a teacher from a different model family than your student (e.g., GPT-4-class teacher with a Llama student). [Lewis Tunstall's announcement](https://x.com/_lewtun/status/1983620843952328726).

### Other

- **`thunlp/OPD`** ([GitHub](https://github.com/thunlp/OPD)) — official code for the "Rethinking OPD" paper (Tsinghua, Apr 2026; [arXiv:2604.13016](https://arxiv.org/abs/2604.13016)), including their cold-start fixes and teacher-aligned prompt-selection.
- **`HJSang/OPSD_OnPolicyDistillation`** ([GitHub](https://github.com/HJSang/OPSD_OnPolicyDistillation)) — community OPSD-on-veRL fork.
- **Unsloth** — no native OPD trainer; community pattern is TRL `GKDTrainer` with Unsloth model loaders.
- **OpenRLHF** — no native OPD trainer as of May 2026; only MiniLLM-style off-policy KD references.

---

## The OPD-vs-RL debate

This is the live argument that puts OPD in the spotlight in late 2025 / early 2026.

### Argument FOR replacing RL with OPD

From Kevin Lu / TML: *"on-policy distillation can learn the RL-trained policy in approximately 7–10× fewer gradient steps, which corresponds to a compute efficiency of 50–100×."* The substantive claim under this rhetoric:

- Most RL post-training is paying for **credit assignment over sparse outcome rewards**. The model receives a single 0/1 signal per 16K-token rollout and has to figure out which tokens mattered.
- If you have a teacher that has already solved that problem (e.g., a larger RL-trained model in the same family), **you don't need to redo credit assignment** — you can copy it directly via per-token teacher log-probs.
- Therefore: when a strong teacher exists, RL is overcomplicated infrastructure.

### Argument AGAINST — "this is DAGGER from 15 years ago"

Guohao Li ([tweet](https://x.com/guohao_li/status/1987821200060625175), Nov 2025): *"When Thinking Machines Lab released their blog on On Policy Distillation, my first reaction was that it should be just like DAGGER from 15 years ago … sure enough, they mentioned DAGGER."*

The conceptual point: **on-policy imitation learning** was solved in [DAGGER (Ross, Gordon, Bagnell, 2010)](https://arxiv.org/abs/1011.0686). OPD is DAGGER applied to LLM token sequences. The novelty is the *engineering* of doing it efficiently at LLM scale, not the algorithmic idea. The implication is also that OPD inherits DAGGER's **imitation-learning ceiling**: you cannot exceed the teacher.

### Argument AGAINST — teacher ceiling and capability extension

Several 2026 papers ([arXiv:2604.00626 survey](https://arxiv.org/abs/2604.00626), [arXiv:2602.12125 "Learning beyond Teacher"](https://arxiv.org/abs/2602.12125), [arXiv:2605.01347 MAD-OPD](https://arxiv.org/abs/2605.01347)) hit the same note: *"existing methods are capped by a single-teacher capability ceiling: when the teacher errs, the student inherits the error."* RL with verifiable rewards can *in principle* discover solutions the teacher doesn't have; OPD by construction cannot. For tasks where the goal is to *surpass* the teacher (frontier reasoning, novel scientific discovery), OPD is at best a warm-start.

### Argument AGAINST — instability and negative transfer

["Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes"](https://arxiv.org/abs/2603.25562) (Mar 2026): vanilla OPD can suffer **entropy collapse**, **negative transfer from strong teachers** ([Rethinking OPD, Apr 2026](https://arxiv.org/abs/2604.13016)), and **biased token-level estimator** (token-level reverse KL is not unbiased for sequence-level reverse KL). Three concrete failure modes:

1. **Cold-start collapse** — reverse KL requires the student's support to cover the teacher's; without prior SFT the student doesn't have it, and gradient explodes.
2. **Negative transfer from strong teachers** — a teacher whose "thinking pattern" diverges from the student's (different RL history, different family) can hurt rather than help. The Tsinghua paper characterizes successful OPD as alignment on a 97–99 % shared high-probability token set.
3. **Tokenizer / special-token mismatch** — cross-family OPD silently fails when tokenizers don't align; GOLD addresses this via cross-tokenizer logit alignment.

### Synthesis position — "use both"

The mainstream 2026 view, consistent with how NVIDIA actually deploys it: **RL for exploration, OPD for stability and regression recovery, jointly optimized when both signals are available**. [KDRL (Xu, Zhu et al., Jun 2025)](https://arxiv.org/abs/2506.02208) is the cleanest formulation — replace [[grpo|GRPO]]'s KL-to-old-policy with reverse-KL-to-teacher, and add the OPD term to the GRPO advantage. Reports +4.7 % vs SFT, +2.6 % vs GRPO, +1.1 % vs KD-RKL on reasoning benchmarks. This is the recipe family that includes dGRPO, MOPD, and DeepSeek-V4's full pipeline replacement.

---

## When to reach for OPD vs RL — a practitioner's decision tree

> [!tip] Quick selection guide

```
                  Do I have a teacher that already
                  solved this task at quality I want?
                          /                   \
                       Yes                     No
                        /                       \
            Is the teacher in the                Are rewards verifiable
            same model family / tokenizer?       (math, code, formal tasks)?
                /         \                       /         \
              Yes          No                   Yes          No
              /             \                   /             \
        OPD wins        Try GOLD             GRPO /          DPO /
        (start with     (cross-tokenizer)    DAPO /          preference RL
        TML recipe)     or fall back         outcome RL      or skip RL entirely
                        to off-policy KD                     (do SFT only)
```

**Heuristics from the literature:**

- **OPD warm-starts beat cold RL** if a teacher exists; do OPD first, then RL on top for capability extension.
- **Self-distillation (OPSD) is a regression-recovery primitive** — use it after any mid-training that risks catastrophic forgetting.
- **Multi-teacher OPD (MOPD / DeepSeek-V4 style)** is the right play when you have multiple specialists you need to merge.
- **OPD alone is insufficient for frontier reasoning** — to push past the teacher you need exploration that only RL provides. Stack OPD + RL (KDRL / dGRPO style) when you need both.

---

## Limitations

- **Bounded by teacher capability.** Cannot exceed teacher quality without auxiliary signals (RL reward, debate, reward extrapolation).
- **Cold-start fragile.** Reverse KL needs the student to already have support over teacher-likely tokens. Requires prior SFT or forward-KL warm-up.
- **Cross-family negative transfer.** Different tokenizers / different pretraining corpora can produce a strong teacher that *degrades* the student. Use GOLD or a same-family teacher.
- **Compute claims need scrutiny.** The "100× cheaper than RL" headline depends on amortization of teacher inference and ignores generation cost. Qwen3's own report says ~10×; independent replication of the 100× has not been published.
- **Biased token-level estimator.** Vanilla token-level reverse KL is not an unbiased estimator of sequence-level reverse KL — variance compounds on long rollouts (a serious issue for 16 K+ token reasoning traces, very relevant for agentic settings). DeepSeek-V4's full-vocabulary KL is one workaround; sequence-level variance bounds are another active research direction.
- **Teacher inference cost.** Open OPD recipes assume the teacher is cheap to serve (logits available). When the teacher is a closed API (GPT-4 / Claude), use Black-Box OPD (GAD) instead.
- **Entropy collapse.** Without proper KL regularization the student can mode-collapse onto a particular teacher mode. KDRL-style joint objective with RL exploration helps.

---

## What this means

Three predictions worth tracking:

1. **OPD will replace mixed-RL post-training for reasoning models when a teacher exists.** [[deepseek-v4-opd|DeepSeek-V4]] is the first flagship to fully commit. Expect Qwen, Mistral, and the open-source community to follow. RL will remain dominant for the *first* model in a generation (no teacher available) and for frontier capability extension.
2. **Multi-teacher OPD is the next default.** Single-teacher OPD inherits a single ceiling; multi-teacher (DeepSeek-V4-style) lets you merge specialists. Expect this to become standard in 2026 H2 onwards.
3. **The interesting research is no longer the loss function.** GKD / MiniLLM nailed the math in 2023. The interesting work in 2026 is: (a) variance reduction (full-vocab KL, sequence-level corrections), (b) cross-tokenizer alignment (GOLD), (c) cost-effective teacher serving (logit caching, FP4 QAT), (d) hybrid OPD+RL objectives (KDRL, dGRPO).

What this is *not*: a universal RL killer. For tasks where the goal is to exceed the strongest available teacher, OPD has nothing to offer beyond warm-starting. RL retains a fundamental role.

---

## Related reading

- [[deepseek-v4-opd]] — DeepSeek-V4's specific multi-teacher full-vocabulary OPD recipe.
- [[grpo]] — The RL algorithm OPD is most often compared / combined with.
- [[ppo-for-llm]] — The trust-region intuition shared with OPD's KL-to-teacher penalty.
- [[rlhf-overview]] — Standard RL post-training pipeline that OPD displaces.
- [[dpo]] — Alternative preference-based RL replacement; orthogonal to OPD.
- [[rl-training-frameworks]] — The trainer-side libraries (OpenRLHF, TRL, veRL, NeMo-RL) where OPD lives.
- [[das-spec-rl]] — Speculative-decoding speedup for RL rollouts; complementary to OPD at the inference layer.
- [[prorl-agent]] — Rollout-as-a-service infrastructure that hosts both RL and OPD workloads.

## References

- **Thinking Machines Lab blog** (Kevin Lu, 2025-10-27): https://thinkingmachines.ai/blog/on-policy-distillation/
- **GKD paper**: Agarwal et al., *On-Policy Distillation of Language Models* (ICLR 2024). [arXiv:2306.13649](https://arxiv.org/abs/2306.13649)
- **MiniLLM paper**: Gu et al., *MiniLLM: On-Policy Distillation of Large Language Models* (NeurIPS 2024). [arXiv:2306.08543](https://arxiv.org/abs/2306.08543)
- **Qwen3 technical report**: [arXiv:2505.09388](https://arxiv.org/abs/2505.09388)
- **DeepSeek-V4 technical report**: hosted at [huggingface.co/deepseek-ai/DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) (no arXiv)
- **KDRL**: Xu, Zhu et al. (Jun 2025). [arXiv:2506.02208](https://arxiv.org/abs/2506.02208)
- **OPD Survey**: [arXiv:2604.00626](https://arxiv.org/abs/2604.00626)
- **Rethinking OPD**: Tsinghua (Apr 2026). [arXiv:2604.13016](https://arxiv.org/abs/2604.13016) — code at [thunlp/OPD](https://github.com/thunlp/OPD)
- **Revisiting OPD Failure Modes**: [arXiv:2603.25562](https://arxiv.org/abs/2603.25562)
- **Black-Box OPD (GAD)**: [arXiv:2511.10643](https://arxiv.org/abs/2511.10643)
- **NVIDIA Nemotron-Cascade 2**: [research.nvidia.com/labs/nemotron/nemotron-cascade-2](https://research.nvidia.com/labs/nemotron/nemotron-cascade-2/)
- **HuggingFace TRL GKDTrainer**: [trl/trainer/gkd_trainer.py](https://github.com/huggingface/trl/blob/main/trl/trainer/gkd_trainer.py)
- **veRL OPD docs**: [verl.readthedocs.io/en/latest/algo/opd.html](https://verl.readthedocs.io/en/latest/algo/opd.html)
- **NeMo-RL Discussion #1445**: [github.com/NVIDIA-NeMo/RL/discussions/1445](https://github.com/NVIDIA-NeMo/RL/discussions/1445)
- **Tinker cookbook**: [github.com/thinking-machines-lab/tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)
- **DAGGER** (the ancestor): Ross, Gordon, Bagnell (2010). [arXiv:1011.0686](https://arxiv.org/abs/1011.0686)
