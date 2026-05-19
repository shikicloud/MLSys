---
title: "MOPD: Multi-Domain On-Policy Distillation as a Cascade-RL Stabilizer"
category: rl-infra
tags: [mopd, on-policy-distillation, nemotron-cascade-2, cascade-rl, multi-teacher, post-training, moe, paper-review]
created: 2026-05-19
updated: 2026-05-19
status: mature
paper: arXiv:2603.19220
---

# MOPD: Multi-Domain On-Policy Distillation as a Cascade-RL Stabilizer

> [!info] Paper metadata
> - **Paper**: [arXiv:2603.19220](https://arxiv.org/abs/2603.19220) — *Nemotron-Cascade 2: Post-Training LLMs with Cascade RL and Multi-Domain On-Policy Distillation* (Yang et al., NVIDIA; v1 2026-03-19, v2 2026-03-22; corresponding author Wei Ping)
> - **Project page**: [research.nvidia.com/labs/nemotron/nemotron-cascade-2](https://research.nvidia.com/labs/nemotron/nemotron-cascade-2/)
> - **Model**: [Nemotron-Cascade-2-30B-A3B](https://huggingface.co/nvidia/Nemotron-Cascade-2-30B-A3B) (30B total / 3B active MoE, 1M context, NVIDIA Open Model License)
> - **SFT data**: [nvidia/Nemotron-Cascade-2-SFT-Data](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-SFT-Data)
> - **RL data**: [nvidia/Nemotron-Cascade-2-RL-data](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-RL-data)
> - **Reference framework**: [NVIDIA-NeMo/RL](https://github.com/NVIDIA-NeMo/RL) (`nemo_rl/algorithms/distillation.py`); multi-teacher routing for MOPD is **not yet primitive** in NeMo-RL as of May 2026
> - **Prior work**: [[on-policy-distillation|GKD (Agarwal 2024)]], [Thinking Machines OPD blog (2025-10)](https://thinkingmachines.ai/blog/on-policy-distillation/), Xiaomi MiMo-V2-Flash MOPD ([arXiv:2601.02780](https://arxiv.org/abs/2601.02780); see [naming collision](#naming-collision-with-xiaomi-mimo-v2-flash))

> [!abstract]+ TL;DR
> **MOPD** is a single dedicated stage inside NVIDIA's 7-stage Cascade RL pipeline that uses [[on-policy-distillation|on-policy distillation]] from **three same-family teachers** to *recover capability regressions* introduced by earlier specialized RL stages. Per-prompt routing picks one of the three teachers (math SFT checkpoint / RLHF side-branch / multi-domain RL best checkpoint) and computes a **sampled-token reverse-KL "advantage"** $a_t = \log\pi_T(y_t) - \log\pi_\theta(y_t)$ as a token-weight on a REINFORCE-style update, with truncated importance weighting for the train-vs-inference policy gap. **All three teachers are free by-products of the same cascade** — no extra training runs, no external models, no logit caches. Reported wins: AIME 2025 92.4 (gold-medal class), IMO 2025 35/42 (gold), IOI 2025 439.28/600 (gold), ICPC WF 2025 10/12 (gold) — and crucially **MOPD reaches ArenaHard v2 85.5 in 52 steps vs RLHF's 80.7 in 160 steps**, a ~3× per-step efficiency win that's the paper's main empirical argument. **MOPD is not novel as a technique** — it's GKD-reverse-KL-OPD plus per-prompt teacher routing, which Xiaomi MiMo-V2-Flash published two months earlier under the same acronym (re-purposed by NVIDIA from "Multi-Teacher" to "Multi-Domain"). The contribution is the **recipe**: which teachers, which stage, what LR schedule, on a 30B-A3B MoE student validated by IMO/IOI/ICPC gold medals.

---

## Background: why MOPD needed inventing

Cascade RL stacks specialized RL stages back-to-back, each tuned for a different capability (instruction following, math reasoning, RLHF helpfulness, long-context, code, SWE-agent). Within each stage, the trainer makes the model better at *that* domain. Across stages, **the model drifts on previously-trained domains**. The Nemotron-Cascade 2 paper acknowledges this directly: some RL stages (notably code RL) "reduce model entropy and shorten reasoning traces, which hurts math" (per [Labonne's writeup](https://maximelabonne.substack.com/p/nemotron-cascade-2-on-policy-distillation)).

The naive fix — re-running the regressed domain's RL — is expensive and unstable (you'll just drift back). Two structural problems:

| Problem | Why it shows up across cascade stages | Naive fix's cost |
| ------- | ------------------------------------- | ---------------- |
| **Capability drift** | Each stage's reward signal is domain-specific; optimizing one domain shifts policy away from others | Re-train regressed domain's RL — expensive, possibly re-introduces a different drift |
| **Entropy collapse** | Code/agentic RL tends to shorten reasoning traces; this hurts math | Increase exploration coef — can break the gains you just paid for |
| **Misalignment after IF-RL** | IF-RL improves instruction following but hurts human-preference scores | Run RLHF later — but how much, and where to put it? |

MOPD's reframe: don't re-train. **Distill** the regressed capability back into the student using a teacher that already has it. The teacher doesn't need to be external or large — it can be a *cascade-internal* checkpoint that was strong on that domain before the regression. So you get a per-token credit-assignment signal (dense, as in [[on-policy-distillation|OPD]]) for capability restoration, at a fraction of RL's per-step cost.

Compared to existing OPD recipes:

| Method | # Teachers | Teacher source | Per-prompt routing | KL form | Pipeline role |
| ------ | ---------- | -------------- | ------------------ | ------- | ------------- |
| [[on-policy-distillation\|TML OPD]] (2025-10) | 1 | External (Qwen3-32B) | n/a | Reverse-KL, sampled-token | RL replacement |
| Xiaomi MOPD ([MiMo-V2-Flash](https://arxiv.org/abs/2601.02780), 2026-01) | Multi | Independent specialists | Yes | Reverse-KL, sampled-token | Post-training merge |
| [[deepseek-v4-opd\|DeepSeek-V4 OPD]] (2026-04) | 10+ | Independently-trained specialists | Weighted sum (all teachers per prompt) | **Full-vocabulary** KL | Replaces entire mixed-RL stage |
| **NVIDIA MOPD** (this page, 2026-03) | 3 | **Free by-products of same cascade** | Yes (one teacher per prompt) | Reverse-KL, sampled-token | Single stabilization stage inside cascade |

The key differentiator: MOPD's teachers come for free. No extra training, no external models, no FP4 QAT or hidden-state caching infrastructure. The same tokenizer, same vocab, same base model — so the distillation can be done with the existing NeMo-RL OPD primitive plus a per-batch teacher swap.

---

## The key idea: route prompts to per-domain teachers and use OPD as a stabilizer

> [!quote] The contribution in one sentence
> Insert a single short on-policy-distillation stage between Multi-domain RL and RLHF, routed per-prompt to one of three cascade-internal teachers (math SFT / RLHF side-branch / multi-domain best checkpoint), to recover capabilities the earlier specialized RL stages drifted away from — for free, in ~50 steps.

Three sub-claims hold this up:

- **Cascade-internal teachers are sufficient.** You don't need an external larger model. The math SFT checkpoint already had strong reasoning before specialized RL touched it; the RLHF side-branch already has good human-preference alignment; the multi-domain RL best checkpoint has the instruction-following gains. Each is the *previous-stage best on its own domain* and that's exactly what you want to restore.
- **One teacher per prompt is enough.** Unlike [[deepseek-v4-opd|DeepSeek-V4]]'s weighted sum, MOPD picks exactly one teacher per training example by domain tag. Cheaper signal, no teacher conflict, simpler dispatcher.
- **Sampled-token reverse-KL with importance clipping is enough.** Full-vocabulary KL (V4 style) is unnecessary at 30B-A3B scale; per-token sampled signal plus truncated $r_t \in [0.5, 2.0]$ importance clip handles train/inference policy skew with stability.

Remove any one: lose internal teachers and MOPD becomes "yet another external-teacher OPD"; lose per-prompt routing and a single teacher pulls all domains toward its style; lose importance clipping and async on-policy training diverges.

---

## How it works

### Where MOPD sits in the cascade

The 7-stage Nemotron-Cascade 2 pipeline (Figure 2 of the paper):

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │                                                                       │
   │  SFT → IF-RL → Multi-domain RL → MOPD → RLHF → Long-context RL       │
   │                                                                       │
   │       → Code RL → SWE RL → Nemotron-Cascade 2                         │
   │                                                                       │
   └──────────────────────────────────────────────────────────────────────┘

   MOPD is a SINGLE stabilization stage between Multi-domain RL and RLHF.
   It is NOT a per-round loop interleaved with each RL stage.
   The cascade does not return to MOPD between later stages.
```

The placement is the key recipe choice. By the time MOPD runs:

- **SFT** has produced a strong math reasoner (becomes math teacher).
- **IF-RL** has built instruction-following capability but may have hurt human alignment.
- **Multi-domain RL** has consolidated MCQA / agentic tool calling / structured output (becomes multi-domain teacher).
- **A parallel RLHF side-branch from SFT** has been trained for human preference (becomes RLHF teacher).

MOPD then uses all three teachers to stabilize before launching into RLHF on the main cascade, and the cascade continues uninterrupted through Long-context / Code / SWE RL.

### Loss function (Eqs. 2–4 in the paper)

Let $\pi^{\text{inf}}$ be the inference-engine student used for rollout, $\pi^{\text{train}}$ the student being optimized, $\pi^{\text{domain}_i}$ the teacher routed for this prompt's domain, and $s_t = (x, y_{<t})$.

**Token-level distillation advantage** — reverse-KL form on the sampled token only:

$$
a_t^{\text{MOPD}} = \log \pi^{\text{domain}_i}(y_t \mid s_t) - \log \pi^{\text{train}}(y_t \mid s_t) \tag{Eq. 2}
$$

**Truncated importance weighting** for train-vs-inference policy mismatch:

$$
r_t = \frac{\pi^{\text{train}}(y_t \mid s_t)}{\pi^{\text{inf}}(y_t \mid s_t)}, \qquad w_t = \text{sg}[r_t] \cdot \mathbf{1}\bigl[\epsilon_{\text{low}} \leq r_t \leq \epsilon_{\text{high}}\bigr] \tag{Eq. 3}
$$

with $\epsilon_{\text{low}} = 0.5$, $\epsilon_{\text{high}} = 2.0$. Out-of-bounds tokens are zero-weighted (not gradient-clipped, just dropped).

**Surrogate objective:**

$$
\boxed{\,\mathcal{L}_{\text{MOPD}} = -\,\mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi^{\text{inf}}(\cdot \mid x)}\!\left[\frac{1}{|\mathcal{V}(y)|}\sum_{t \in \mathcal{V}(y)} w_t \cdot \text{sg}[a_t^{\text{MOPD}}] \cdot \log \pi^{\text{train}}(y_t \mid s_t)\right]\,} \tag{Eq. 4}
$$

Two crucial implementation details the paper is explicit about:

- **Per-token sampled-token KL, not full-vocab.** *"The log-probability difference is computed only on the student-sampled token rather than over the full vocabulary"* (paper p.13). This is the deliberate choice vs [[deepseek-v4-opd|DeepSeek-V4]]'s full-vocab.
- **Stop-gradient on both $a_t$ and $r_t$.** The only gradient path is through $\log\pi^{\text{train}}(y_t \mid s_t)$ — a token-weighted REINFORCE-style update where the "advantage" is the log-prob gap to the teacher.

> [!quote] How it relates to [[on-policy-distillation|GKD]]
> Eq. 4 is GKD-reverse-KL-OPD with two engineering modifications: (a) truncated importance weighting to handle the async on-policy gap ($\pi^{\text{inf}}$ may lag $\pi^{\text{train}}$), and (b) explicit stop-gradients to keep the implementation REINFORCE-shaped rather than direct-KL-shaped. The reward $\log(\pi_T/\pi_\theta)$ is unchanged from MiniLLM/GKD/TML.

### Teacher selection — three teachers, all free

The paper enumerates exactly three teachers (p.13):

| Teacher | Source | What it's good at | Cost |
| ------- | ------ | ----------------- | ---- |
| **Math teacher** | Original SFT checkpoint | Strong mathematical reasoning from SFT data curation | 0 (already exists) |
| **RLHF teacher** | RLHF side-branch from SFT init (25 RLHF steps with GenRM = Qwen3-235B-A22B-Thinking-2507) | Human-preference alignment | 25 steps of RLHF (cheap) |
| **Multi-domain teacher** | Best checkpoint after IF-RL + Multi-domain RL | Instruction following, MCQA, agentic tool calling, structured output | 0 (already trained for the cascade) |

Selection criterion: *"the strongest validation checkpoint for each benchmark category"* (paper p.12). All three teachers share the **same tokenizer, same vocab, same base model** as the student — which is why MOPD doesn't need DeepSeek-V4-style FP4 QAT, hidden-state caching, or cross-vocab logit projection. Just point the rollout server at the right checkpoint per prompt.

### Per-prompt routing — one teacher per training example

Prompts are tagged with a `teacher_id` reflecting their domain origin:

```
training pool composition (approximate, from paper §4.4):

  math prompts        ─── AceReason-Math      ─► routed to math teacher (SFT)
  IF / multi-domain   ─── from IF-RL / Multi-  ─► routed to multi-domain teacher
  prompts                  domain RL pools
  helpfulness         ─── from RLHF training   ─► routed to RLHF teacher
  prompts                  pool (HelpSteer3 etc.)
```

There is **no cross-teacher logit mixing**. Each token in a rollout is supervised by exactly one teacher's log-prob. This is the architectural choice vs DeepSeek-V4's $\sum_i w_i D_{\text{KL}}(\pi_\theta \| \pi_{E_i})$ weighted sum.

### Hyperparameters

A discrepancy between the **prose** (p.13) and the **appendix Table 8** worth knowing:

| Setting | Prose §4.4 | Table 8 (Appendix B) |
| ------- | ---------- | -------------------- |
| Learning rate | 2×10⁻⁶ with linear warmup over first 30 steps from 2×10⁻⁷ | 3×10⁻⁶ |
| Steps | "Typically converges within 40–50 steps" | 52 |
| Rollouts per prompt | 4 | 4 |
| Prompts per update (batch) | 128 | 128 |
| Effective batch (responses) | 512 | — |
| Max response length | — | 98K |
| Importance bounds | $\epsilon_{\text{low}} = 0.5$, $\epsilon_{\text{high}} = 2.0$ | — |
| Temperature / top-p | — | 1.0 / 1.0 |
| Overlong filtering | — | False |
| KL form | reverse-KL, sampled-token (not full-vocab) | — |

Lead with the prose numbers; flag the LR inconsistency to anyone reproducing. The paper makes no claim about which is canonical.

### Why this works at modest cost

Three structural reasons MOPD is cheap relative to RLHF or rerunning specialized RL:

- **Teachers are free.** Math teacher = SFT checkpoint (no extra training). Multi-domain teacher = the best checkpoint you saved anyway during Multi-domain RL. RLHF teacher = 25 steps of side-branch RLHF (smaller than the main RLHF stage).
- **Dense per-token signal.** Unlike RLHF's per-trajectory scalar reward, MOPD scores every token. ~52 steps with dense signal recovers what 160 RLHF steps with sparse reward would.
- **No infrastructure changes.** Same tokenizer, same vocab, same base. The NeMo-RL OPD primitive needs only a per-batch teacher swap to become MOPD — no DeepSeek-V4-style hidden-state cache, no FP4 QAT, no TileLang kernel.

---

## Experiments

### Headline numbers (Tables 1–2)

Nemotron-Cascade-2-30B-A3B vs same-active-parameter baselines:

| Benchmark | Nemotron-Cascade-2-30B-A3B | Qwen3.5-35B-A3B (2026-02) | Nemotron-3-Super-120B-A12B (2026-03) |
| --------- | -------------------------- | -------------------------- | ------------------------------------ |
| **IMO 2025** | **35/42 (Gold)** | — | — |
| IMO AnswerBench | **79.3** | 74.8 | 77.2 |
| IMO ProofBench | **72.9** | — | — |
| **AIME 2025** | **92.4** (98.6 TIR) | 91.9 | 90.2 |
| AIME 2026 | 90.9 (95.0 TIR) | **91.1** | 89.8 |
| HMMT Feb25 | **94.6** | 89.0 | 93.7 |
| **IOI 2025** | **439.28/600 (Gold)** | 348.6 | — |
| **ICPC WF 2025** | **10/12 (Gold)** | — | — |
| LiveCodeBench v6 | **87.2** (88.4 TIR) | 74.6 | 78.7 |
| LCB Pro 25Q2 Med | **27.6** (36.8 TIR) | 17.8 | 23.2 |
| MMLU-Pro | 79.8 | **85.3** | 83.7 |
| GPQA-Diamond | 76.1 | **84.2** | 79.2 |
| ArenaHard v2 (Avg) | **83.5** | 65.4 | — |
| IFBench (prompt) | **82.9** | 70.2 | 72.6 |
| SWE Verified (OpenHands) | 50.2 | **69.2** | 60.5 |
| Terminal Bench 2.0 | 21.1 | **40.5** | 31.0 |
| 𝜏²-Bench | 58.9 | **81.2** | 61.2 |

The paper's Footnote 1 (p.4): Nemotron-Cascade 2 is *"the second open-weight LLM, after DeepSeek-V3.2-Speciale-671B-A37B, to achieve gold-medal performance in both the IMO and IOI"* — at **3B active parameters** vs DeepSeek's 37B active.

### The MOPD-specific result: step efficiency vs RLHF

The paper's main argument for MOPD (Table 3, prose p.13):

| Stage on the cascade | Steps | ArenaHard v2 (hard / overall) | AIME 25 |
| -------------------- | ----- | ------------------------------ | ------- |
| Multi-domain RL output | — | — | 91.0 |
| **MOPD** (52 steps) | **52** | **85.5 / 71.0** | **92.0** |
| RLHF (160 steps) | 160 | 80.7 / 71.2 | — |

MOPD reaches higher ArenaHard scores in ~3× fewer steps than RLHF. On AIME 25 the +1.0 absolute improvement (91.0 → 92.0) is modest but came in 30 steps, where GRPO needed 25 for 91.0 — so on math MOPD is roughly compute-matched with GRPO, but on human-preference benchmarks it dramatically beats RLHF on step efficiency.

> [!important] What the paper does NOT report
> No GPU-hour or wall-clock comparison for MOPD specifically. No leave-MOPD-out ablation isolating its contribution per benchmark across the whole cascade. No ablation of teacher count (would 1 or 2 work? Would 4–6 be better?). The step-efficiency claim is **per-step, not per-second** — which matters because each MOPD step still requires a teacher forward pass.

### Where MOPD doesn't help

The benchmarks Nemotron-Cascade 2 loses on tell the story:

| Benchmark | Cascade 2 | Qwen3.5-35B-A3B | Δ |
| --------- | --------- | --------------- | -- |
| MMLU-Pro | 79.8 | 85.3 | **−5.5** |
| GPQA-Diamond | 76.1 | 84.2 | **−8.1** |
| SWE Verified (OpenHands) | 50.2 | 69.2 | **−19.0** |
| Terminal Bench 2.0 | 21.1 | 40.5 | **−19.4** |
| 𝜏²-Bench | 58.9 | 81.2 | **−22.3** |

The paper concedes (p.5) Cascade 2 *"underperforms Qwen3.5-35B-A3B primarily on knowledge-intensive and agentic tasks."* The structural reason ties back to MOPD: **the teacher pool has no GPQA teacher and no agentic-tool-use teacher**. MOPD can only restore capabilities for which a cascade-internal teacher exists. Knowledge gaps and agentic-task gaps require either an external teacher or longer RL — neither of which is in the recipe.

---

## Strengths and limitations

The two strongest points: (1) the **cascade-internal teacher** insight is the genuine recipe contribution — getting three high-quality teachers for the cost of one short side-branch RLHF run is the actual story behind MOPD's efficiency, not the loss function; (2) the **production validation** is unambiguous — IMO/IOI/ICPC gold at 3B active parameters is the strongest open-weight reasoning-model evidence point of early 2026.

Where the work is honest about scope but the limits matter:

- **Not algorithmically novel.** The loss is GKD reverse-KL OPD ([[on-policy-distillation|Agarwal 2024]]) plus truncated importance weighting (standard in async on-policy RL — DAPO, PPO-clip). The per-prompt teacher routing was published two months earlier by Xiaomi MiMo-V2-Flash under the same acronym (see [naming collision](#naming-collision-with-xiaomi-mimo-v2-flash)).
- **No teacher-count ablation.** Would 1 or 2 teachers work? Would adding a GPQA teacher fix the knowledge-task gap? The paper doesn't say.
- **No leave-MOPD-out ablation across the full cascade.** Table 3 / Figure 3 compare MOPD vs GRPO on AIME25 and ArenaHard v2 only. We can't isolate MOPD's contribution to IMO/IOI/ICPC gold.
- **Hyperparameter inconsistency.** Prose says LR 2e-6 + warmup; Table 8 says 3e-6. Suggests the schedule was tuned more than reported.
- **Sampled-token KL means signal is sparse on low-confidence tokens.** Full-vocab (V4-style) would catch more but the paper didn't try the comparison at this scale.
- **Teacher conflict not analyzed.** At domain boundaries (e.g., a math prompt that also tests instruction-following), only one teacher is picked — possibly the wrong one.
- **Capability ceiling = teacher ceiling.** MOPD cannot extend capabilities the teachers don't have. The GPQA / SWE Verified gaps are evidence.
- **Multi-teacher routing is not in NeMo-RL.** The reference framework only implements single-teacher OPD. Reproducing MOPD requires forking the trainer to accept a teacher dict and look up the right one per batch — non-trivial.

> [!warning] Naming collision with Xiaomi MiMo-V2-Flash
> The acronym **MOPD was first used by Xiaomi** in MiMo-V2-Flash ([arXiv:2601.02780](https://arxiv.org/abs/2601.02780), 2026-01-06), where it stands for **Multi-Teacher On-Policy Distillation**. The Xiaomi MiMo Twitter account: *"Beyond arch innovation, MiMo-V2-Flash is cooked via a NEW post-training paradigm Multi-Teacher On-Policy Distillation (MOPD)"* ([source](https://x.com/XiaomiMiMo/status/2000930865757741342)). NVIDIA's Nemotron-Cascade 2 (2026-03) re-purposes the acronym as **Multi-Domain On-Policy Distillation** with essentially the same algorithm (per-prompt teacher routing for token-level reverse-KL distillation). The Nemotron-Cascade 2 paper cites Xiao et al. 2026 as prior work but doesn't acknowledge the acronym overlap. When reading any 2026 MOPD reference, **check whether it's Xiaomi's "Multi-Teacher" or NVIDIA's "Multi-Domain"** — same idea, same letters, different framings.

---

## What this means

Two predictions worth tracking:

1. **OPD-as-cascade-stabilizer becomes a standard stage.** The argument is empirically grounded: 52 MOPD steps with cascade-internal teachers replace 160 RLHF steps for cross-domain consolidation. This is too cheap not to adopt. Expect Qwen, Mistral, and open-source labs to add a similar stage between specialized RL and final-stage alignment.
2. **The "free internal teacher" insight will spread.** The DeepSeek-V4-style approach — train 10+ specialists, weight-merge with full-vocab KL — is expensive and infrastructure-heavy. NVIDIA's "the teacher is just an earlier checkpoint in your pipeline" insight is dramatically cheaper. For most practitioners without DeepSeek-V4 infrastructure, MOPD's recipe is the realistic adoption path.

What this is *not*: a universal cure for cascade-RL drift. MOPD can only restore capabilities for which a cascade-internal teacher exists. Capability *extension* still requires either RL with verifiable reward or an external teacher.

---

## Source code & reproduction

### Released artifacts (open)

| Artifact | Status |
| -------- | ------ |
| Model weights ([Nemotron-Cascade-2-30B-A3B](https://huggingface.co/nvidia/Nemotron-Cascade-2-30B-A3B)) | ✓ Open, NVIDIA Open Model License |
| SFT data ([nvidia/Nemotron-Cascade-2-SFT-Data](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-SFT-Data)) | ✓ Open |
| RL data ([nvidia/Nemotron-Cascade-2-RL-data](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-RL-data)) | ✓ Open |
| Technical report ([arXiv:2603.19220](https://arxiv.org/abs/2603.19220)) | ✓ Open |
| End-to-end MOPD script / config | ✗ Not released |
| Multi-teacher OPD trainer in NeMo-RL | ✗ Not yet primitive (single-teacher only) |

### NeMo-RL OPD primitive (the closest first-party reference)

[NVIDIA-NeMo/RL](https://github.com/NVIDIA-NeMo/RL), file `nemo_rl/algorithms/distillation.py` (1,072 lines). [Discussion #1445](https://github.com/NVIDIA-NeMo/RL/discussions/1445). [Docs](https://docs.nvidia.com/nemo/rl/0.5.0/about/algorithms/on-policy-distillation.html).

What's there:
- Single teacher / single student — `MasterConfig.teacher: PolicyConfig` (line 117).
- KL options: `forward` / `reverse` / `mixed`.
- Top-k restricted KL (e.g. `topk_logits_k=64`) — **note this is already a deviation from the paper's Eq. 2, which uses only the sampled-token log-prob, not top-k logits**.
- Backends: DTensor + vLLM. Megatron generation/training not yet supported.

What's missing for MOPD specifically:
- Multi-teacher routing (per-batch / per-sample teacher swap).
- Truncated importance weighting (Eq. 3).
- The exact sampled-token KL (no top-k expansion).

To replicate MOPD on NeMo-RL, you'd extend `DistillationLossFn` to accept a teacher dict and look up `batch["teacher_id"]` per sample, plus add the importance-clip mask.

### Closest off-the-shelf alternative — veRL multi-teacher OPD

[veRL OPD docs](https://verl.readthedocs.io/en/latest/algo/opd.html) and [async on-policy distill](https://verl.readthedocs.io/en/latest/advance/async-on-policy-distill.html). veRL already supports multi-teacher routing via `data_source`:

```yaml
distillation:
  enabled: true
  teacher_models:
    math_teacher:
      model_path: /path/to/sft_ckpt
    rlhf_teacher:
      model_path: /path/to/rlhf_sidebranch_ckpt
    multi_domain_teacher:
      model_path: /path/to/multi_domain_rl_best_ckpt
  teacher_key: data_source     # routes each prompt to the matching teacher
  distillation_loss:
    loss_mode: k1              # K1 = log(π_S/π_T) — close to Eq. 2
  use_policy_gradient: true
  use_task_rewards: false      # MOPD-style: no outcome reward, only OPD signal
```

This is **closer to the MOPD paper than NeMo-RL** — set `teacher_key="data_source"`, tag each prompt with its domain, and the framework handles per-batch teacher switching. Less invasive than forking NeMo-RL.

### Minimum reproduction recipe (on NeMo-RL fork)

```python
# Conceptual sketch — full implementation requires extending NeMo-RL's
# DistillationLossFn for per-sample teacher lookup.

# 1. SFT → save as math_teacher_ckpt
# 2. Run IF-RL (180 steps, batch 128 × 16 rollouts, LR 3e-6)
# 3. Run Multi-domain RL (70 steps) → save best as multi_domain_teacher_ckpt
# 4. Side-branch: from SFT init, RLHF (25 steps, GenRM=Qwen3-235B, KL 0.03)
#    → save as rlhf_teacher_ckpt
# 5. MOPD stage:
#    - Tag prompts: math from AceReason-Math → math_teacher;
#                   multi-domain from RL pool → multi_domain_teacher;
#                   helpfulness from HelpSteer3 → rlhf_teacher
#    - Loss: sampled-token reverse-KL advantage, stop-grad on a_t and r_t,
#            importance clip [0.5, 2.0]
#    - LR 2e-6 with linear warmup from 2e-7 over 30 steps; ~52 total steps
#    - Batch 128 prompts × 4 rollouts
# 6. Continue cascade: RLHF (25 steps) → Long-context RL (30 steps)
#    → Code RL (22 steps) → SWE RL (40–50 steps)
```

### Other open implementations of the underlying OPD primitive

| Project | Path | Closest to MOPD? |
| ------- | ---- | ---------------- |
| [HF TRL `GKDTrainer`](https://github.com/huggingface/trl/blob/main/trl/trainer/gkd_trainer.py) | Single-teacher GKD with $\lambda$ + $\beta$ knobs | Furthest — needs custom data routing + multi-teacher wrapping |
| [veRL `algo/opd`](https://verl.readthedocs.io/en/latest/algo/opd.html) | Multi-teacher routing via `data_source` | **Closest off-the-shelf** |
| [NVIDIA NeMo-RL OPD](https://github.com/NVIDIA-NeMo/RL) | Single-teacher with top-k KL | Closest in spirit; needs multi-teacher extension |
| [Tinker cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) | Single + multi-teacher recipes | Multi-teacher but token-level KL, no per-prompt routing |

---

## Related reading

- [[on-policy-distillation]] — Umbrella page on OPD (the GKD paper, math, variants, the OPD-vs-RL debate). MOPD sits in the variants section there.
- [[deepseek-v4-opd]] — The other 2026 flagship multi-teacher OPD deployment. Architectural contrast in [the comparison table](#background-why-mopd-needed-inventing).
- [[grpo]] — Used inside Cascade 2's specialized RL stages (IF-RL, Multi-domain RL, Code RL, SWE RL).
- [[ppo-for-llm]] — Trust-region intuition underlying the importance-clip in Eq. 3.
- [[rlhf-overview]] — The stage MOPD partially replaces by step-efficiency.
- [[rl-training-frameworks]] — NeMo-RL is where MOPD would be implemented; veRL has the closest off-the-shelf multi-teacher primitive.
- [[nemo-gym]] — Hosts the rollout-side of any Cascade RL / MOPD pipeline.
- [[das-spec-rl]] — Speculative-decoding speedup for the rollout phase; complementary at the inference layer.
- [[prorl-agent]] — Rollout-as-a-service infrastructure adjacent to the SWE RL stage.

## References

- **Nemotron-Cascade 2 technical report**: Yang et al., NVIDIA. [arXiv:2603.19220](https://arxiv.org/abs/2603.19220) · [Project page](https://research.nvidia.com/labs/nemotron/nemotron-cascade-2/) · [PDF](https://research.nvidia.com/labs/nemotron/files/Nemotron-Cascade-2.pdf)
- **Model & data**: [Nemotron-Cascade-2-30B-A3B](https://huggingface.co/nvidia/Nemotron-Cascade-2-30B-A3B) · [SFT data](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-SFT-Data) · [RL data](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-RL-data)
- **Nemotron-Cascade 1** (prior work): Wang et al. [arXiv:2512.13607](https://arxiv.org/abs/2512.13607)
- **Xiaomi MiMo-V2-Flash** (the *other* MOPD): [arXiv:2601.02780](https://arxiv.org/abs/2601.02780) · [GitHub](https://github.com/XiaomiMiMo/MiMo-V2-Flash) · [Xiaomi MiMo tweet](https://x.com/XiaomiMiMo/status/2000930865757741342)
- **GKD** (the underlying OPD technique): Agarwal et al., ICLR 2024. [arXiv:2306.13649](https://arxiv.org/abs/2306.13649)
- **Thinking Machines Lab OPD blog** (cited as ref [42] by Cascade 2): [thinkingmachines.ai/blog/on-policy-distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)
- **DeepSeek-V4 OPD** (architectural contrast): [HF PDF](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf)
- **OPD Survey 2026**: [arXiv:2604.00626](https://arxiv.org/abs/2604.00626) — groups Nemotron-Cascade 2 / DeepSeek-V4 / MiMo-V2-Flash / GLM-5 / KAT-Coder-V2 / ORBIT / Uni-OPD as the "industrial multi-teacher/multi-domain OPD" cluster.
- **Independent writeups**: [Maxime Labonne, *Nemotron Cascade 2: On-policy distillation is back!*](https://maximelabonne.substack.com/p/nemotron-cascade-2-on-policy-distillation) · [VentureBeat](https://venturebeat.com/orchestration/nvidias-nemotron-cascade-2-wins-math-and-coding-gold-medals-with-3b-active) · [MarkTechPost](https://www.marktechpost.com/2026/03/20/nvidia-releases-nemotron-cascade-2-an-open-30b-moe-with-3b-active-parameters-delivering-better-reasoning-and-strong-agentic-capabilities/) · [Ritvik Rastogi explainer](https://ritvik19.medium.com/papers-explained-552-nemotron-cascade-2-1ac869c28c8c)
- **NeMo-RL implementation**: [GitHub](https://github.com/NVIDIA-NeMo/RL) · [Discussion #1445](https://github.com/NVIDIA-NeMo/RL/discussions/1445) · [OPD docs](https://docs.nvidia.com/nemo/rl/0.5.0/about/algorithms/on-policy-distillation.html)
- **veRL multi-teacher OPD**: [docs](https://verl.readthedocs.io/en/latest/algo/opd.html) · [async OPD docs](https://verl.readthedocs.io/en/latest/advance/async-on-policy-distill.html)
- **HF TRL GKDTrainer**: [trl/trainer/gkd_trainer.py](https://github.com/huggingface/trl/blob/main/trl/trainer/gkd_trainer.py)
