---
title: "On-Policy Distillation (OPD): Dense Teacher Signal as an RL Replacement"
category: rl-infra
tags: [on-policy-distillation, opd, gkd, minillm, distillation, rl-post-training, reverse-kl, paper-review]
created: 2026-05-19
updated: 2026-05-19
status: mature
paper: arXiv:2306.13649
code: https://github.com/huggingface/trl/blob/main/trl/trainer/gkd_trainer.py
---

# On-Policy Distillation (OPD): Dense Teacher Signal as an RL Replacement

> [!info] Paper metadata
> - **Original paper**: [arXiv:2306.13649](https://arxiv.org/abs/2306.13649) — *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes* (Agarwal, Vieillard, Zhou, Stańczyk, Ramos, Geist, Bachem; DeepMind; ICLR 2024). The paper title is literally "on-policy distillation of language models" — this **is** the OPD paper.
> - **Companion paper**: [arXiv:2306.08543](https://arxiv.org/abs/2306.08543) — *MiniLLM: Knowledge Distillation of Large Language Models* (Gu, Dong, Wei, Huang; Microsoft / Tsinghua; NeurIPS 2024; v3 retitled *MiniLLM: On-Policy Distillation of Large Language Models*). Provides the policy-gradient derivation.
> - **2025 reframing**: [Thinking Machines Lab blog](https://thinkingmachines.ai/blog/on-policy-distillation/) (Kevin Lu, 2025-10-27). Not a new paper — a Qwen3-scale repackaging of GKD that popularized the "OPD" label and the "RL replacement" framing.
> - **Reference code**: [HF TRL `GKDTrainer`](https://github.com/huggingface/trl/blob/main/trl/trainer/gkd_trainer.py); also veRL `algo/opd`, NVIDIA NeMo-RL, TML `tinker-cookbook`.

> [!abstract]+ TL;DR
> On-Policy Distillation is the **technique introduced by Agarwal et al. 2023 (GKD paper, arXiv:2306.13649)** for knowledge distillation where the student samples its own rollouts and a frozen teacher scores each token via reverse KL. Mathematically, it is **policy gradient with a dense per-token reward equal to $\log(\pi_T/\pi_\theta)$** — same shape as [[grpo|GRPO]] minus the sparse outcome reward and value head, with KL-to-teacher serving as *both* the reward and the trust-region regularizer. The technique sat as a quiet KD method until **Thinking Machines Lab's 2025-10 blog** reframed it as an *RL post-training replacement*, with the headline number **74.4 % AIME'24 at ~1,800 GPU-h** on Qwen3-8B-Base (vs Qwen3's own RL recipe at 67.6 % / ~17,920 GPU-h). As of mid-2026 the strongest production deployments are **NVIDIA Nemotron-Cascade 2** (MOPD interleaved with cascade RL), **Alibaba Qwen3** small models, and **DeepSeek-V4** (multi-teacher full-vocabulary OPD entirely replacing the mixed-RL stage — see [[deepseek-v4-opd]]).

---

## Background: why on-policy distillation needed inventing

LLM post-training has two existing lanes for transferring capability from a strong teacher / verifiable rewards into a smaller / specialist student, and both have a structural flaw:

| Lane | Trajectory distribution | Reward density | Failure mode |
| ---- | ----------------------- | -------------- | ------------ |
| **SFT / off-policy KD** | $y \sim \mathcal{D}$ (fixed corpus from teacher) | dense (per-token soft targets) | Student visits states at inference that aren't in $\mathcal{D}$; small errors compound; style mimicry without behavior transfer |
| **RL ([[grpo|GRPO]], [[ppo-for-llm|PPO]], DPO)** | $y \sim \pi_\theta$ (student rollouts) | **sparse** — O(1) bit per episode | Most of a 16 K-token rollout has no per-token credit assignment; expensive credit reconstruction from a scalar |

The first lane gets the reward density right but the trajectory distribution wrong. The second lane gets the trajectory distribution right but the reward density wrong. GKD's contribution was the obvious-once-you-see-it third lane: **on-policy** trajectories ($y \sim \pi_\theta$) with **token-dense** supervision (per-token reverse KL to the teacher).

Five competing methods on the same axes:

| Method | On-policy ($y \sim \pi_\theta$)? | Per-token signal? | Needs reward model? | Needs value head? |
| ------ | -------------------------------- | ----------------- | ------------------- | ----------------- |
| SFT + soft targets | ✗ | ✓ | ✗ | ✗ |
| RLHF (PPO) | ✓ | ✗ | ✓ | ✓ |
| GRPO | ✓ | ✗ | ✓ | ✗ |
| DPO | ✗ (preference data) | ✗ | ✗ | ✗ |
| **OPD / GKD** | **✓** | **✓** | **✗** | **✗** |

OPD is the only row with both ticks in the first two columns and no ticks in the last two.

---

## The key idea: dense per-token teacher signal on student rollouts

> [!quote] The contribution in one sentence
> Replace the sparse scalar reward of RL with the teacher's per-token log-probabilities, while still sampling trajectories from the *current student* so the gradient direction matches the deployment distribution.

Three sub-claims hold this up:

- **Reverse KL is mode-seeking** — the student concentrates probability mass on teacher-likely tokens instead of spreading to cover the teacher's tails. Right choice for combinatorial LLM output spaces.
- **On-policy trajectories** — gradient computed on the states the deployed student will actually visit, eliminating SFT's compounding-error pathology.
- **No reward / value model needed** — per-token teacher log-prob *is* the dense reward signal, so the entire RL critic infrastructure collapses.

Remove any one: lose mode-seeking and you cover teacher tails wastefully (forward KL); lose on-policy and you re-introduce SFT's distribution-shift problem; bring back the reward model and you've reinvented RLHF.

---

## How it works

### The OPD loss

GKD's general form, with $\lambda \in [0, 1]$ controlling on-policy fraction and $D$ being a generalized JSD parameterized by $\beta$:

$$
\mathcal{L}_{\text{GKD}}(\theta) = (1{-}\lambda)\,\mathbb{E}_{(x,y)\sim \mathcal{D}}\!\left[D(\pi_T\|\pi_\theta)(y\mid x)\right] + \lambda\,\mathbb{E}_{x,\,y\sim \pi_\theta}\!\left[D(\pi_T\|\pi_\theta)(y\mid x)\right]
$$

Pure OPD = GKD at $(\lambda, \text{direction}) = (1.0, \text{reverse KL})$:

$$
\mathcal{L}_{\text{OPD}}(\theta) = \mathbb{E}_{x,\,y\sim\pi_\theta(\cdot\mid x)}\!\left[\sum_{t=1}^{|y|} D_{\text{KL}}\!\big(\pi_\theta(\cdot\mid y_{<t}, x)\,\big\|\,\pi_T(\cdot\mid y_{<t}, x)\big)\right]
$$

| Knob | What it does | TML's OPD setting |
| ---- | ------------ | ----------------- |
| $\lambda$ | On-policy fraction. 0 = pure SFT-with-soft-targets, 1 = pure on-policy. | 1.0 |
| $\beta$ (KL direction) | 0 = forward KL (mean-seeking), 1 = reverse KL (mode-seeking), 0.5 = symmetric JSD. | 1.0 (reverse) |
| Discount factor $\gamma$ | Time discount across the trajectory. | 0 (reward is already token-level dense) |

### The policy-gradient duality

This is the result that makes OPD legible to anyone fluent in [[grpo|GRPO]] / [[ppo-for-llm|PPO]]. Per MiniLLM §3, the gradient of the on-policy reverse-KL expectation is:

$$
\nabla_\theta\,\mathbb{E}_{y\sim\pi_\theta}\!\big[D_{\text{KL}}(\pi_\theta\|\pi_T)\big] \;=\; -\,\mathbb{E}_{y\sim\pi_\theta}\!\left[\sum_{t} \nabla_\theta \log\pi_\theta(y_t\mid y_{<t})\cdot \underbrace{\log\frac{\pi_T(y_t\mid y_{<t})}{\pi_\theta(y_t\mid y_{<t})}}_{\text{dense per-token "reward"}}\right]
$$

This is **vanilla REINFORCE** with the per-token reward replaced by the teacher log-ratio. Three immediate corollaries:

- **OPD inherits PPO's stability.** The KL-to-teacher term doubles as a trust-region regularizer.
- **OPD = GRPO minus the sparse reward and value head.** GRPO already removed the critic via group-relative advantages; OPD goes one further and makes the reward itself token-level.
- **The discount factor is irrelevant.** Reward is dense, so there's no credit-assignment problem $\gamma$ needs to solve.

> [!quote] Mental model
> OPD is the GRPO objective where the sparse outcome reward $R(y) \in \{0, 1\}$ is replaced by the dense per-token signal $\log(\pi_T/\pi_\theta)$, and the value head is removed because the reward is already token-level.

### Why "on-policy" matters

The expectation $\mathbb{E}_{y\sim\pi_\theta}$ is over the *student's* trajectory distribution. Replace this with $\mathbb{E}_{y\sim\mathcal{D}}$ over a fixed corpus and you get standard SFT-with-soft-targets:

| Setting | Sampling | Effect |
| ------- | -------- | ------ |
| $\lambda = 0$ | $y \sim \mathcal{D}$ | SFT with soft targets; biased w.r.t. deployment states; drift compounds |
| $\lambda = 1$ | $y \sim \pi_\theta$ | On-policy; gradient on states the deployed student actually visits |
| $0 < \lambda < 1$ | mixture | Trade stability vs on-policy relevance |

This is also the core insight of [DAGGER (Ross, Gordon, Bagnell, 2010)](https://arxiv.org/abs/1011.0686) — see [the OPD-vs-RL debate](#opd-vs-rl-debate) below.

### Token-level vs full-vocabulary KL

A second-order knob with first-order consequences at scale. Two ways to compute the KL:

| Form | What it measures | Gradient form | Variance | Memory / bandwidth |
| ---- | ---------------- | ------------- | -------- | ------------------ |
| **Token-level** (TML, MiniLLM, most OPD papers) | KL at the *sampled token* only | $\nabla_\theta \log\pi_\theta(y_t)\cdot \log(\pi_T(y_t)/\pi_\theta(y_t))$ | High (single sample of a $V$-dim distribution) | $O(1)$ per token |
| **Full-vocabulary** ([[deepseek-v4-opd|DeepSeek-V4]]) | Analytic KL over all $V$ vocab tokens | $\sum_v \pi_\theta(v) \log(\pi_\theta(v)/\pi_T(v))$ | Low (exact) | $O(V)$ per token |

Token-level OPD is what HF TRL implements and what TML used; full-vocabulary is what DeepSeek-V4 (Apr 2026) argues you need at flagship scale because the token-level estimator's variance compounds on long rollouts.

---

## Variants and production deployments

Verified deployments and named variants in 2025–2026 use. This is the "Experiments" section for a technique umbrella page.

### Variants

| Variant | Origin | Key delta vs GKD-with-$\lambda{=}1$ |
| ------- | ------ | ----------------------------------- |
| **OPSD** (Self-Distillation) ([arXiv:2602.04942](https://arxiv.org/abs/2602.04942)) | 2025–26 | Teacher is the student's earlier checkpoint or a privileged-info version of itself. Continual-learning primitive. |
| **KDRL** ([arXiv:2506.02208](https://arxiv.org/abs/2506.02208)) | Xu, Zhu et al., Jun 2025 | Replaces GRPO's KL-to-old-policy with reverse-KL-to-teacher; jointly optimizes rule-based reward + OPD. |
| **dGRPO** ([survey](https://arxiv.org/abs/2604.00626)) | 2025–26 | GRPO advantage + per-token OPD loss as a dense auxiliary head. |
| **MOPD** (Multi-Domain) ([Nemotron-Cascade 2](https://research.nvidia.com/labs/nemotron/nemotron-cascade-2/)) | NVIDIA, Mar 2026 | Single stabilization stage inside 7-stage Cascade RL; 3 cascade-internal teachers (math SFT / RLHF side-branch / multi-domain RL best) routed per-prompt; sampled-token reverse-KL with importance clipping. Detailed at [[mopd]]. **Note**: same acronym used 2 months earlier by Xiaomi MiMo-V2-Flash as "Multi-**Teacher** OPD". |
| **MAD-OPD** ([arXiv:2605.01347](https://arxiv.org/abs/2605.01347)) | 2026 | Multi-agent debate as the teacher signal. Attempts to break single-teacher ceiling. |
| **Reward-Extrapolated OPD** ([arXiv:2602.12125](https://arxiv.org/abs/2602.12125)) | 2026 | Adds RL reward head so student can learn beyond teacher. |
| **Black-Box OPD (GAD)** ([arXiv:2511.10643](https://arxiv.org/abs/2511.10643)) | Ye, Dong et al., Nov 2025 | OPD when only completions (no logits) are available — for OpenAI / Anthropic teachers. Uses adversarial discriminator. |
| **Multi-teacher full-vocab OPD** ([DeepSeek-V4](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)) | DeepSeek, Apr 2026 | $\sum_i w_i D_{\text{KL}}(\pi_\theta\|\pi_{E_i})$ over 10+ specialists, full-vocabulary KL. Flagship-scale demo. See [[deepseek-v4-opd]]. |

### Production deployments (verified from primary sources)

| Deployment | Recipe | Source |
| ---------- | ------ | ------ |
| **NVIDIA Nemotron-Cascade 2** (Mar 2026) | 30B-A3B MoE. Single MOPD stage between Multi-domain RL and RLHF in a 7-stage Cascade RL pipeline. 3 cascade-internal teachers (math SFT / RLHF side-branch / multi-domain RL best). 52 steps recover what 160 RLHF steps would. **IMO/IOI/ICPC 2025 gold medals** at 3B active params. See [[mopd]] for full details. | [Nemotron-Cascade 2 page](https://research.nvidia.com/labs/nemotron/nemotron-cascade-2/) |
| **Alibaba Qwen3** small models (May 2025) | 0.6B–14B + 30B-A3B-MoE. Off-policy distillation (teacher: larger Qwen3) → on-policy distillation. Skips stages 3–4 of full RL pipeline. **Reported 1/10 GPU-hour cost.** | [Qwen3 tech report](https://arxiv.org/abs/2505.09388) |
| **DeepSeek-V4** (Apr 2026) | 1.6T/49B MoE. Per-domain (SFT → GRPO) specialists → multi-teacher full-vocab OPD merge. **Entirely replaces** V3.2's mixed-RL stage. Detailed at [[deepseek-v4-opd]]. | [V4 tech report](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf) |

### Headline numbers from TML's reproduction

TML's 2025-10 blog replicates the Qwen3 OPD recipe with Qwen3-8B-Base / Qwen3-32B:

| Method | AIME'24 | Compute (GPU-h, ~) |
| ------ | ------- | ------------------ |
| SFT only (400 K prompts) | 60 % | not reported |
| SFT extrapolated to 2 M | ~70 % | not reported |
| Qwen3 RL recipe | 67.6 % | ~17,920 |
| **On-Policy Distillation** | **74.4 %** | **~1,800** |

~10× compute reduction with a *better* AIME score. The TML blog further claims 50–100× when self-distilling; **independent replication of the 100× number has not been published as of May 2026.** Qwen3's tech report (1/10 GPU-h) corroborates the ~10× direction; the higher end is single-lab.

### Where OPD is *not* used

For calibration: **DeepSeek-R1 → smaller students** uses **SFT-only off-policy** distillation (~800 K verified traces). **Meta Llama 4** uses codistillation with dynamic soft/hard weighting — public materials don't describe student-rollout-based on-policy. **Anthropic, OpenAI, Mistral, Cohere** — no public evidence of on-policy distillation in their pipelines.

---

## Strengths and limitations

The two strongest points: (1) it produces a **token-dense gradient signal with no reward / value model**, collapsing most RL post-training infrastructure when a teacher exists; (2) the **on-policy trajectory** eliminates SFT's compounding-error pathology, so the student trains on the states it will actually visit.

Where it falls down and the limits matter:

- **Bounded by teacher capability.** Reverse KL is imitation-learning — the student concentrates on teacher-likely tokens and cannot discover solutions the teacher lacks. For frontier reasoning where the goal is to *surpass* the teacher, OPD is at best a warm-start.
- **Cold-start fragile.** Reverse KL requires the student's support to cover teacher-likely tokens. Without prior SFT the student doesn't have it and the gradient explodes. TML's recipe relies on Qwen3-Base already being heavily pretrained.
- **Cross-family negative transfer.** A teacher with different "thinking patterns" (different RL history, different family) can *degrade* the student. [Tsinghua's Rethinking OPD paper](https://arxiv.org/abs/2604.13016) characterizes successful OPD as alignment on a 97–99 % shared high-probability token set. Use same-family teachers or GOLD-style cross-tokenizer alignment.
- **Biased token-level estimator.** Vanilla token-level reverse KL is not an unbiased estimator of sequence-level reverse KL; variance compounds on long rollouts (16 K+ token reasoning traces — directly relevant to agentic settings). [DeepSeek-V4's full-vocabulary KL](#token-level-vs-full-vocabulary-kl) is the leading workaround.
- **Entropy collapse.** Without proper KL regularization the student can mode-collapse onto a particular teacher mode. KDRL-style joint objective with RL exploration helps.
- **Compute claims need scrutiny.** "100× cheaper than RL" depends on amortizing teacher inference and ignoring generation cost. Defensible band based on independent Qwen3 evidence: 5–20×.
- **Teacher inference cost.** OPD recipes assume teacher logits are cheap to obtain. With closed-API teachers (GPT-4, Claude), use Black-Box OPD (GAD) instead.

> [!warning] The "this is just DAGGER from 2010" critique
> Guohao Li ([tweet, Nov 2025](https://x.com/guohao_li/status/1987821200060625175)): *"When Thinking Machines Lab released their blog on On Policy Distillation, my first reaction was that it should be just like DAGGER from 15 years ago … sure enough, they mentioned DAGGER."* The conceptual point: on-policy imitation learning was solved in [DAGGER (Ross, Gordon, Bagnell, 2010)](https://arxiv.org/abs/1011.0686); OPD is DAGGER on LLM token sequences. The novelty is engineering at LLM scale, not the algorithmic idea. The implication is that OPD inherits DAGGER's imitation-learning ceiling.

### OPD-vs-RL debate

The headline argument since late 2025. The synthesis position (most production teams) is **"use both"**: RL for exploration, OPD for stability and regression recovery. KDRL is the cleanest formulation — joint reverse-KL-to-teacher + GRPO reward in one gradient step, reporting +4.7 % vs SFT, +2.6 % vs GRPO, +1.1 % vs KD-RKL. NVIDIA's Nemotron-Cascade 2 takes the same architectural stance at scale: MOPD is *interleaved with* cascade RL, not a replacement.

---

## What this means

Three predictions worth tracking:

1. **OPD will replace mixed-RL post-training for reasoning models when a teacher exists.** [[deepseek-v4-opd|DeepSeek-V4]] is the first flagship to fully commit. Expect Qwen, Mistral, and the open-source community to follow. RL remains dominant for the *first* model in a generation (no teacher available) and for frontier capability extension.
2. **Multi-teacher OPD becomes the default.** Single-teacher OPD inherits one ceiling; multi-teacher (V4-style) lets you merge specialists. Expect this to be standard from 2026 H2 onwards.
3. **The interesting research moves off the loss function.** GKD and MiniLLM nailed the math in 2023. 2026 work is on (a) variance reduction (full-vocab KL, sequence-level corrections), (b) cross-tokenizer alignment (GOLD), (c) cost-effective teacher serving (logit caching, FP4 QAT, hidden-state caching), (d) hybrid OPD+RL objectives (KDRL, dGRPO).

What this is *not*: a universal RL killer. When the goal is to exceed the strongest available teacher, OPD has nothing to offer beyond warm-starting. RL retains a fundamental role.

---

## Source code & reproduction

### HuggingFace TRL — `GKDTrainer`

The canonical open reference. File: [`trl/trainer/gkd_trainer.py`](https://github.com/huggingface/trl/blob/main/trl/trainer/gkd_trainer.py). Two key pieces:

```python
# lines 226-295 — generalized_jsd_loss
def generalized_jsd_loss(student_logits, teacher_logits, labels=None,
                        beta=0.5, temperature=1.0, reduction="batchmean"):
    # beta == 0 → forward KL; beta == 1 → reverse KL; else mixture via logsumexp
    ...

# lines 421-449 — training_step
def training_step(self, model, inputs, ...):
    if random.random() <= self.lmbda:                # lmbda = 1.0 ⇒ pure OPD
        inputs = self.generate_on_policy_outputs(...)
    ...
```

| `GKDConfig` field | Role |
| ----------------- | ---- |
| `lmbda` | Probability of fresh student rollout per batch. `1.0` = pure OPD. |
| `beta` | KL direction. TML's OPD setting: `beta=1.0` (reverse). |
| `temperature` | Softmax temperature. |
| `seq_kd` | Generate from *teacher* — sequence-level KD on teacher samples. |
| `use_liger_kernel` | Fused linear+JSD kernel for memory. |

> [!warning] TRL is deprecating `GKDTrainer`
> Current TRL emits *"This trainer will soon be moved to `trl.experimental` and is a candidate for removal."* A newer `DistillationTrainer` with a generation buffer (gen batch decoupled from train microbatch, ~40× claimed speedup) and external teacher server support is in development. Track [TRL releases](https://github.com/huggingface/trl/releases) and [issue #4390](https://github.com/huggingface/trl/issues/4390).

### Other implementations

| Project | Path | Notes |
| ------- | ---- | ----- |
| **veRL** | [`algo/opd` docs](https://verl.readthedocs.io/en/latest/algo/opd.html) | `distillation.*` config namespace. Multi-teacher routing keyed by `data_source` (enables MOPD). `loss_mode={forward_kl_topk, k1, k3}`, `use_policy_gradient`, `use_task_rewards`. vLLM-hosted teachers via ZeroMQ. |
| **NVIDIA NeMo-RL** | [Discussion #1445](https://github.com/NVIDIA-NeMo/RL/discussions/1445) | Forward / reverse / mixed KL. **Top-k restricted KL** (bandwidth opt). Qwen3-4B-Base AIME'25 Avg@16: 47.71 % (SFT+OPD) vs 30.42 % (SFT+off-policy). |
| **TML `tinker-cookbook`** | [github.com/thinking-machines-lab/tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) | `tinker_cookbook/recipes/distillation/` — single/multi-teacher, on/off-policy, multi-turn tool-use variants. Closest to TML's blog recipe. |
| **HF H4 GOLD** | [HF Space](https://huggingface.co/spaces/HuggingFaceH4/on-policy-distillation) | Cross-tokenizer OPD via token-merge alignment + product-rule logit combination. Will land as `GOLDTrainer`. |
| **Tsinghua `thunlp/OPD`** | [GitHub](https://github.com/thunlp/OPD) | Official code for the "Rethinking OPD" paper. Cold-start fixes, teacher-aligned prompt selection. |

### Minimum reproduction recipe (TML-style, with TRL)

```python
from trl import GKDConfig, GKDTrainer
from transformers import AutoModelForCausalLM

student = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B-Base")
teacher = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-32B")

config = GKDConfig(
    output_dir="./opd_qwen3",
    lmbda=1.0,           # pure on-policy
    beta=1.0,            # reverse KL
    temperature=1.0,
    max_new_tokens=16384,
    learning_rate=1e-6,
    per_device_train_batch_size=1,
)

trainer = GKDTrainer(
    model=student,
    teacher_model=teacher,
    args=config,
    train_dataset=load_math_prompts(),
)
trainer.train()
```

This is the minimum to reproduce TML's AIME number — actual production runs use multi-teacher routing, full-vocab KL (DeepSeek-V4 path), and the engineering tricks in `tinker-cookbook` / veRL.

---

## Related reading

- [[deepseek-v4-opd]] — DeepSeek-V4's multi-teacher full-vocabulary OPD recipe; the flagship-scale instantiation.
- [[grpo]] — The RL algorithm OPD is most often compared / combined with; OPD is structurally GRPO minus the sparse reward.
- [[ppo-for-llm]] — The trust-region intuition shared with OPD's KL-to-teacher penalty.
- [[rlhf-overview]] — The standard RL post-training pipeline OPD displaces.
- [[dpo]] — Alternative preference-based RL replacement; orthogonal to OPD.
- [[rl-training-frameworks]] — The trainer-side libraries (OpenRLHF, TRL, veRL, NeMo-RL) where OPD implementations live.
- [[das-spec-rl]] — Speculative-decoding speedup for RL / OPD rollouts; complementary at the inference layer.
- [[prorl-agent]] — Rollout-as-a-service infrastructure that hosts both RL and OPD workloads.

## References

- **GKD** (the OPD paper): Agarwal et al., *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes*, ICLR 2024. [arXiv:2306.13649](https://arxiv.org/abs/2306.13649)
- **MiniLLM**: Gu et al., NeurIPS 2024. [arXiv:2306.08543](https://arxiv.org/abs/2306.08543)
- **TML blog**: Kevin Lu, 2025-10-27. [thinkingmachines.ai/blog/on-policy-distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)
- **Qwen3 technical report**: [arXiv:2505.09388](https://arxiv.org/abs/2505.09388)
- **DeepSeek-V4 technical report**: [HF PDF](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf)
- **KDRL**: Xu, Zhu et al. (Jun 2025). [arXiv:2506.02208](https://arxiv.org/abs/2506.02208)
- **OPD Survey**: [arXiv:2604.00626](https://arxiv.org/abs/2604.00626)
- **Rethinking OPD**: Tsinghua (Apr 2026). [arXiv:2604.13016](https://arxiv.org/abs/2604.13016) — code [thunlp/OPD](https://github.com/thunlp/OPD)
- **Revisiting OPD Failure Modes**: [arXiv:2603.25562](https://arxiv.org/abs/2603.25562)
- **Black-Box OPD (GAD)**: [arXiv:2511.10643](https://arxiv.org/abs/2511.10643)
- **NVIDIA Nemotron-Cascade 2**: [research.nvidia.com/labs/nemotron/nemotron-cascade-2](https://research.nvidia.com/labs/nemotron/nemotron-cascade-2/)
- **HF TRL GKDTrainer**: [trl/trainer/gkd_trainer.py](https://github.com/huggingface/trl/blob/main/trl/trainer/gkd_trainer.py)
- **veRL OPD docs**: [verl.readthedocs.io/en/latest/algo/opd.html](https://verl.readthedocs.io/en/latest/algo/opd.html)
- **NeMo-RL Discussion #1445**: [github.com/NVIDIA-NeMo/RL/discussions/1445](https://github.com/NVIDIA-NeMo/RL/discussions/1445)
- **Tinker cookbook**: [github.com/thinking-machines-lab/tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)
- **DAGGER** (the ancestor): Ross, Gordon, Bagnell (2010). [arXiv:1011.0686](https://arxiv.org/abs/1011.0686)
