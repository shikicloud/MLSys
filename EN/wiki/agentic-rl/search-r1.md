---
title: "Search-R1: Training LLMs to Reason and Leverage Search Engines with RL"
category: agentic-rl
tags: [search-r1, agentic-rl, retrieval-augmented, ppo, grpo, r1-zero-lineage, retrieved-token-masking, paper-review]
created: 2026-05-26
updated: 2026-05-26
status: mature
paper: arXiv:2503.09516
code: https://github.com/PeterGriffinJin/Search-R1
---

# Search-R1: Training LLMs to Reason and Leverage Search Engines with RL

> [!info] Paper metadata
> - **Paper**: [arXiv:2503.09516](https://arxiv.org/abs/2503.09516) — *Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning*, COLM 2025 (v1 2025-03-12, latest v5 2025-08-05)
> - **Authors**: Bowen Jin¹, Hansi Zeng², Zhenrui Yue¹, Jinsung Yoon³, Sercan Ö. Arık³, Dong Wang¹, Hamed Zamani², Jiawei Han¹
> - **Affiliations**: ¹UIUC, ²UMass Amherst CIIR, ³Google Cloud AI Research
> - **Code**: [PeterGriffinJin/Search-R1](https://github.com/PeterGriffinJin/Search-R1)
> - **Released models**: [PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo](https://huggingface.co/PeterJinGo)

> [!tip] If you're new to agentic RL, this is the canonical entry-point paper
> Read [[agentic-rl-foundations]] for the recommended reading path. Read
> [[search-r1-codebase-walkthrough]] for a file-by-file walkthrough of the
> reference implementation.

---

## Summary (read this if you have 2 minutes)

**What it is.** Search-R1 is the **DeepSeek-R1-Zero approach extended from pure-reasoning to tool-use**: a 4B-7B LLM is trained with **outcome-only rule-based reward** (Exact Match against ground truth) and standard PPO/GRPO to **autonomously call a search engine during multi-turn reasoning**. The trained model interleaves `<think>` reasoning with `<search>query</search>` calls and `<information>retrieved</information>` injections, finally emitting `<answer>final</answer>`.

**The one idea.** **Treat the search engine as part of the environment**, sample interleaved (LLM-token, retrieved-token) trajectories from the student policy, and train PPO/GRPO **end-to-end** on the resulting sequences — but apply **retrieved-token loss masking** so gradients flow only through LLM-generated tokens, not through the documents the search engine injected. Three pieces hold it up:

1. **Multi-turn rollout protocol** — system prompt + four tags (`<think>` / `<search>` / `<information>` / `<answer>`) define the agent loop. Decoding stops at `</search>` to trigger a real retrieval call, results are wrapped in `<information>` tags and concatenated back into the prefix, generation continues. Pure R1-Zero in spirit: no SFT warm-up, no hand-crafted "good rollouts", model learns the protocol via RL alone.
2. **Retrieved-token loss masking** — the policy gradient and KL penalty are computed *only* on LLM-generated positions; retrieved-token positions get `loss_mask = 0`. Without this, the model "learns" to mimic retrieved passage style, training destabilizes, EM collapses by ~9 absolute points.
3. **Simple outcome-only reward** — rule-based EM against `ground_truth.target`; reward = 1 if the extracted `<answer>` matches, else 0. No format reward, no neural RM, no process reward, no search-quality reward.

**Headline result.** Qwen2.5-7B-base + PPO, 7 QA datasets:

| Method category | Method | Avg EM (7 datasets) |
| --------------- | ------ | ------------------: |
| No retrieval | Direct / CoT | 0.181 / 0.106 |
| Prompted retrieval | RAG / IRCoT / Search-o1 | 0.304 / 0.239 / 0.206 |
| Fine-tuning | SFT / Rejection Sampling | 0.207 / 0.348 |
| RL-only (no search) | R1-base | 0.276 |
| **Search-R1-base (PPO)** | | **0.431** |

Relative improvements: **+42 % over RAG**, **+24 % over Rejection Sampling**, **+56 % over RL-only-no-search**. Five backbones tested (Qwen2.5-3B/7B/14B base+instruct, Llama-3.1/3.2). Author summary: **24 % average lift on Qwen2.5-7B, 20 % on 3B**.

**Why it matters.**

- **Validates outcome-only RL for tool-use.** DeepSeek-R1-Zero showed pure RL (no SFT data) could elicit complex reasoning. Search-R1 extends this to *tool calling*: the model autonomously learns *when* to search, *what* to query, *how* to digest results — purely from a 1-bit final-answer reward.
- **`retrieved-token loss masking` is THE first published instance** of a now-universal trick. Polar's `prefix_merging`, NeMo Gym's response-API agent, and every multi-turn tool-use RL system after this all do some variant of "mask environment-injected tokens." Search-R1 is where this pattern was first systematized and shown necessary via ablation.
- **Reference codebase opens agentic-RL to anyone with 8×H100.** veRL fork is tiny (~600 lines of Search-R1-specific code + standard PPO/GRPO), runs Qwen2.5-3B/7B in days. This is the closest thing the field has to a "first agentic-RL homework set."
- **The pattern generalized rapidly.** R1-Searcher, ReSearch, DeepResearcher, ToolRL, ReTool, WebGPT-RL, Computer-Use Agents — all are variations on Search-R1's frame. By mid-2026 the `<think>/<tool>/<obs>/<answer>` 4-tag protocol is the *de facto* agentic-RL standard.

---

# Depth (drill-down starts here)

The summary above is the executive layer. Everything below is for the careful reader who wants the algorithm, the empirical evidence, and an honest critique.

## Background: why RAG and search-as-tool both fall short

LLMs need external knowledge for many tasks, but pre-Search-R1 there were two ways to plug a search engine in, each broken in its own way:

| Lane | What it does | Failure mode |
| ---- | ------------ | ------------ |
| **RAG** (Lewis et al., 2020) | Static retrieve-then-generate. One call, all retrievals stuffed into context, then LLM generates | Model never learns *when* to search, *what* to query, or *how* to compose multiple queries. Retrieval and reasoning are decoupled |
| **Search-as-tool prompting** (ReAct, IRCoT) | Multi-turn, model decides when to call | Out-of-distribution tasks fail; prompting can't teach unseen tool protocols |
| **Search-as-tool SFT** (Toolformer) | Train on labeled tool-call trajectories | Needs large-scale annotated multi-turn trajectories; hard to scale; search is non-differentiable so no end-to-end gradients |

The third lane — RL — was unproven for search at the time. The Search-R1 paper's actual contribution is showing the third lane works, and naming the specific tricks needed to make it work.

### Three challenges Search-R1 names explicitly

Quoting the paper's introduction:

1. **RL framework and stability** — How do you integrate a search engine into the PPO/GRPO loop without destabilizing training when retrieved context appears?
2. **Multi-turn interleaved reasoning and search** — How does the model dynamically decide when to retrieve vs reason?
3. **Reward design** — Is outcome-only reward sufficient, or do you need process rewards?

The answers respectively are: retrieved-token loss masking, the 4-tag protocol, and yes (outcome reward is enough at this scale, no format reward needed).

### Comparison vs prior art

The paper compares against 8 baselines (Table 2). The relevant ones for understanding Search-R1's position:

| Baseline | What it represents | Avg EM (Qwen2.5-7B) |
| -------- | ------------------ | ------------------: |
| Direct Inference | No search at all | 0.181 |
| CoT | "Think before answer", no search | 0.106 |
| RAG | Static one-shot retrieval | 0.304 |
| IRCoT (Trivedi 2022) | Prompted multi-turn search | 0.239 |
| Search-o1 (Li 2025) | Prompted reasoning + search, R1-like format | 0.206 |
| SFT | Trained on rollouts but no RL | 0.207 |
| R1 (no search) | DeepSeek-R1-style RL, no retrieval | 0.276 |
| **Rejection Sampling** (with search) | SFT on RL-style rollouts filtered for correct answers — *the strongest non-RL baseline* | **0.348** |
| **Search-R1** | This work | **0.431** |

The Rejection Sampling baseline is the cleanest counterfactual: "we did the same multi-turn rollout, picked correct ones, did SFT." Search-R1 beats it by **+8.3 pp (24 % relative)** — showing that **RL itself contributes** beyond just generating rollouts.

## The method in detail

### RL objective

Standard RLHF objective, only modification is the policy is conditioned on the search engine $\mathcal{R}$ in addition to the input $x$:

$$
\max_{\pi_\theta} \;\mathbb{E}_{x \sim \mathcal{D},\; y \sim \pi_\theta(\cdot \mid x; \mathcal{R})}\!\left[r_\phi(x, y)\right] - \beta \, D_{\text{KL}}\!\left[\pi_\theta(y \mid x; \mathcal{R}) \,\|\, \pi_{\text{ref}}(y \mid x; \mathcal{R})\right]
$$

The notation $\pi_\theta(\cdot \mid x; \mathcal{R}) = \pi_\theta(\cdot \mid x) \otimes \mathcal{R}$ formalizes "interleaved reasoning and retrieval" — the rollout $y$ contains LLM-sampled tokens AND search-engine-injected tokens.

### The four-tag token protocol

```
<think> reasoning step </think>
<search> query </search>
<information> retrieved passages </information>
<answer> final answer </answer>
```

- `<think>...</think>` — model's reasoning. Hand-mandated by the system prompt (model wasn't trained on these tags before).
- `<search>...</search>` — when the model generates `</search>`, the system pauses generation, extracts the query, calls a real retrieval server, wraps results in `<information>...</information>` tags, appends to the prefix, and resumes generation.
- `<information>...</information>` — **environment-injected**, NOT model-generated. Subject to `loss_mask = 0`.
- `<answer>...</answer>` — when the model generates `</answer>`, rollout ends and reward is computed.

System prompt (verbatim, [`scripts/data_process/nq_search.py`](https://github.com/PeterGriffinJin/Search-R1/blob/main/scripts/data_process/nq_search.py)):

```text
Answer the given question. You must conduct reasoning inside <think> and
</think> first every time you get new information. After reasoning, if you
find you lack some knowledge, you can call a search engine by <search>
query </search> and it will return the top searched results between
<information> and </information>. You can search as many times as your
want. If you find no further external knowledge needed, you can directly
provide the answer inside <answer> and </answer>, without detailed
illustrations. For example, <answer> Beijing </answer>. Question: {question}
```

Notice the **deliberate minimalism** — no in-context examples, no preferred query style, no reflective-reasoning encouragement. **R1-Zero philosophy**: let the policy figure out what to do via RL signal alone.

### The rollout algorithm

Algorithm 1 from the paper, equivalent to [`generation.py:run_llm_loop`](https://github.com/PeterGriffinJin/Search-R1/blob/main/search_r1/llm_agent/generation.py):

```text
y ← ∅
for step in range(max_turns):
    generate tokens autoregressively, stopping at </search>, </answer>, or <eos>
    append generated tokens y_b to y
    if </search> detected:
        extract query from y_b
        retrieved = R(query)        ← real search call (HTTP)
        append "<information>{retrieved}</information>" to y
    elif </answer> detected:
        return y
    else:
        append "My action is not correct. Let me rethink." to y
    
# force final answer (no more search)
if not done:
    generate one more time with do_search=False
return y
```

The action budget `max_turns` (论文 default `2`) caps total search calls per rollout.

### Retrieved-token loss masking — the load-bearing trick

Both PPO and GRPO compute token-level losses over the rollout. In Search-R1, the rollout contains LLM-generated AND retrieved tokens. **Applying PPO loss to retrieved tokens trains the policy to "imitate the retrieved passages"** — but those tokens aren't from the policy's action space! The retrieved content is environment state, not behavior.

The fix: an indicator $I(y_t) \in \{0, 1\}$ marking LLM-generated positions, applied to every loss term:

$$
\mathcal{J}_{\text{PPO}}(\theta) = \mathbb{E}_{x, y}\left[\frac{1}{\sum_t I(y_t)} \sum_{t:\, I(y_t)=1} \min\!\left(\frac{\pi_\theta(y_t | \cdot)}{\pi_{\text{old}}(y_t | \cdot)} A_t,\; \text{clip}(\cdot, 1{-}\epsilon, 1{+}\epsilon) A_t\right)\right]
$$

The same mask gates the **KL penalty** computation — retrieved tokens don't contribute to KL-to-reference either. (If they did, the reference policy's distribution on those positions is meaningless noise that propagates as gradient.)

> [!important] This is THE load-bearing ablation
> Table 4 shows: same model, same hyperparameters, only difference is whether `state_masking=true`. **Without mask the model drops from 0.431 → 0.343 EM avg** — 25 % relative degradation. On MBPP the model trained without mask actually scores *worse than the base model*. The "trick" isn't optimization; it's what makes training work at all in multi-turn settings.

### Reward design

The paper insists on minimalism:

$$
r_\phi(x, y) = \text{EM}(a_{\text{pred}}, a_{\text{gold}})
$$

Extract `<answer>...</answer>` from the rollout, normalize (lowercase, strip articles, remove punctuation), compare against the dataset's `ground_truth.target` (list of acceptable answers). Match = 1, no match = 0.

What the paper deliberately *doesn't* add:

- **No format reward** — R1-Zero adds a format penalty for malformed `<think>` blocks; Search-R1 says "our model adheres to format already, format reward is unnecessary."
- **No process reward / step-level scoring** — no PRM is trained; outcome-only is the design constraint.
- **No search-quality reward** — no signal for "you searched the right thing."
- **No search-count penalty** — model is free to search as many times as it wants.
- **No neural reward model** — rule-based EM only, eliminating reward-hacking surface and saving the cost of training/serving an RM.

The 1-bit signal at the end of a 500-1500-token trajectory is enough at the scales tested (Qwen 3B-14B). At larger scales or harder tasks this might break — but that's beyond the paper's scope.

## PPO vs GRPO — the surprise

The paper tests both:

**Table 3 results** (Qwen2.5-7B-base, 7-dataset EM):

| RL method | Avg EM |
| --------- | -----: |
| GRPO | 0.350 |
| **PPO** | **0.431** |

Qwen2.5-3B-base: GRPO 0.312 vs PPO 0.303 — GRPO marginally better. Qwen2.5-7B-base: PPO significantly better. Mixed results across model sizes / instruct vs base.

**Training dynamics** (Figure 2a): GRPO converges faster (no critic warm-up needed) but **collapses after ~500 steps** in some runs. PPO is slower but stable.

This is a **non-obvious finding** that's specific to Search-R1's setting. In math/code domains GRPO often dominates (DeepSeek-Math, DeepSeek-R1). In multi-turn search-interleaved RL, PPO's value function helps with credit assignment across long rollouts (multiple `<search>` calls + retrieved passages = sequences of 1000+ tokens), and the variance from GRPO's group-mean baseline grows enough to cause collapse.

**Lesson**: algorithm choice in agentic RL is **task-specific**. The DeepSeek-R1 frenzy made GRPO the default, but Search-R1 explicitly recommends PPO for multi-turn search settings.

## Training-dynamics emergence (Figure 2c/d)

The most cited figure in agentic-RL onboarding talks. Qwen2.5-7B-base + PPO, training over 200 steps:

**Response length (Fig 2c)** — three phases:

1. **Steps 0-100 (decrease)** — base model starts producing verbose, filler-heavy output (~1150 tokens/rollout); RL teaches it to be concise (drops to ~900 tokens). Reward only slightly increases here.
2. **Steps 100+ (increase)** — model learns to **actively call search**, retrieved passages inflate sequence length (~900 → ~1100), and reward climbs sharply.
3. **Later (stabilize)** — strategy converges, length plateaus.

**Number of valid search calls (Fig 2d)** — steady increase from ~1.4 to ~2.0 search calls per rollout over training.

This is the **agentic-RL "emergence" plot**. The model wasn't told to search more; it learned from rewards that more searches help on these QA tasks. **This is the equivalent of R1-Zero's "aha moment" / self-reflection emergence, but for tool calling**. Every subsequent agentic-RL paper shows a similar figure.

## Headline ablations

### Retrieved-token loss masking (Table 4)

| Method | NQ | TriviaQA | PopQA | HotpotQA | 2wiki | Musique | Bamboogle | **Avg** |
| ------ | -: | -------: | ----: | -------: | ----: | ------: | --------: | ------: |
| Search-R1 **w/ mask** | 0.480 | 0.638 | 0.457 | 0.433 | 0.382 | 0.196 | 0.432 | **0.431** |
| Search-R1 **w/o mask** | 0.388 | 0.567 | 0.391 | 0.325 | 0.321 | 0.108 | 0.304 | 0.343 |

> [!success] +25.6 % relative just from masking
> 0.343 → 0.431. Without masking, training is unstable, model partially learns to mimic retrieved passages, EM collapses.

### Base vs instruct (Fig 2b)

| Model variant | Initial reward | Final reward |
| ------------- | -------------: | -----------: |
| Qwen2.5-7B-base | low | matches instruct |
| Qwen2.5-7B-instruct | higher | matches base |

**Instruct converges faster but base catches up to similar final performance.** Consistent with R1-Zero showing that RL can succeed from pure base models.

## Strengths and limitations

The two genuine strengths: (1) **outcome-only RL works for tool calling** — a non-trivial extension of R1-Zero that opens the agentic-RL field. The paper validates this cleanly. (2) **The codebase is the cleanest agentic-RL reference implementation available**, ~600 lines of paper-specific code on top of standard veRL PPO/GRPO.

Where it falls short:

- **Only QA / single tool.** All experiments are factoid or multi-hop QA against Wikipedia. No web-agent, OS-agent, scientific reasoning, code agent, multi-tool composition. The "agentic" framing oversells — it's "single-tool retrieval agent." Followup papers (ToolRL, ReSearch, DeepResearcher) extend to more tools and tasks.
- **EM is a noisy reward.** "Albert Einstein" vs "A. Einstein" — both correct, only one matches EM. The model is trained to format answers in the most EM-matching way, which doesn't always equal "most informative answer." Later work uses LLM-as-judge or F1.
- **`max_turns=2` is tiny.** Most rollouts have at most 2 search calls. The "multi-turn" claim is weak at this depth. Long-horizon agentic RL (10+ turns) reveals different failure modes (context length, KV cache management, reward sparsity) that Search-R1 doesn't touch.
- **Static Wikipedia retrieval.** No time-varying corpus, no noisy / adversarial passages, no rate-limited real search engine. Reality is messier.
- **Search engine cost ignored in reward.** Production calls cost $/latency; Search-R1 model has no incentive to be parsimonious. Trained models tend to over-search.
- **`(Bamboogle)` test set is 125 questions.** Statistical significance on small held-out sets is shaky.
- **3-14B only.** No 30B+ runs. Whether the emergence pattern persists at frontier scale is unanswered.
- **Output formatting matters more than the paper admits.** The fact that `extract_solution` requires *at least 2 matches* of `<answer>...</answer>` (one in the system prompt example, one in the actual response) is a fragile coupling. If model output has 0 or 1 matches, reward is automatically 0, regardless of actual correctness.

> [!warning] The "valid_search count is up = better agent" interpretation is partially circular
> Figure 2d shows search-call count rising. This is interpreted as "model learns to use search." But the reward function inherently *prefers* sequences with searches (since searches enable correct answers on knowledge-heavy questions). So increased search use is partially an artifact of "any policy that converges to higher reward will search more." This doesn't refute the emergence story but makes it less surprising than the framing suggests.

## What this means

Three predictions that mostly already played out in 2025-26:

1. **The 4-tag protocol becomes the agentic-RL default.** ✅ Already happened. R1-Searcher, ReSearch, ToolRL all use `<think>/<tool_or_search>/<observation_or_information>/<answer>` patterns. By 2026 anyone writing an agentic-RL paper assumes you know this protocol.
2. **Retrieved-token loss masking becomes universal.** ✅ Polar's `prefix_merging` is the most sophisticated version; veRL upstreamed `state_masking` as a first-class config option; every multi-turn tool-use RL system since does some variant.
3. **PPO/GRPO are not the algorithmic frontier in agentic RL.** ✅ The interesting work moved off the RL algorithm and onto rollout infrastructure (ProRL Agent, Polar), reward design (process rewards, LLM-as-judge), and tool-protocol generalization. Search-R1's contribution was setting up the **stage**, not winning at the stage.

What Search-R1 is *not*: a frontier paper. Its lifetime contribution is as the **canonical entry-point teaching example** for agentic RL. Read it for the framing and the ablation; build on its codebase; then move to the systems papers (Polar) and the harder tasks (computer use, long-horizon agents) for actual research frontier.

## Source code & reproduction

[GitHub: PeterGriffinJin/Search-R1](https://github.com/PeterGriffinJin/Search-R1) — Apache-2.0, ~600 lines of Search-R1-specific Python plus a veRL fork.

| File | Role |
| ---- | ---- |
| `train_ppo.sh` / `train_grpo.sh` | Top-level config |
| `retrieval_launch.sh` | Start FAISS retrieval server |
| `search_r1/llm_agent/generation.py` | **The multi-turn rollout loop** (469 lines) |
| `search_r1/search/retrieval_server.py` | FastAPI E5+FAISS dense retriever (392 lines) |
| `scripts/data_process/nq_search.py` | NQ dataset → parquet with template |
| `verl/utils/reward_score/qa_em.py` | Outcome reward (EM) |
| `verl/trainer/ppo/ray_trainer.py` | Main PPO/GRPO loop (867 lines) |
| `infer.py` | Reference inference path (130 lines) |

Minimal reproduction (Qwen2.5-3B, single 8×A100 node, ~2 days):

```bash
# 1. Pre-build the FAISS index over Wikipedia 2018 dump (one-time, ~hours)
bash example/build_e5_index.sh

# 2. Launch the retrieval server (background)
bash retrieval_launch.sh

# 3. Prepare data
python scripts/data_process/nq_search.py

# 4. Train
bash train_ppo.sh
```

For a complete file-by-file walkthrough including the veRL machinery, see [[search-r1-codebase-walkthrough]].

## Related reading

- [[agentic-rl-foundations]] — Onboarding hub; Search-R1 is the recommended entry-point paper.
- [[search-r1-codebase-walkthrough]] — File-by-file code tutorial covering both Search-R1's 600 lines and the underlying veRL machinery.
- [[grpo]] — The RL algorithm Search-R1 uses (and the alternative; Search-R1 found PPO better in this setting).
- [[ppo-for-llm]] — Foundational PPO-for-LLMs reference.
- [[on-policy-distillation]] — The non-RL alternative for similar problems; Search-R1's rollout structure (student samples, dense per-token signal from environment) is structurally analogous to OPD with the search engine as a degenerate "teacher."
- [[prorl-agent]] — The first "agentic RL as infrastructure" paper. ProRL Agent's `AgentHandler` ABC is the production generalization of Search-R1's hand-coded rollout loop.
- [[polar]] — The current state-of-the-art rollout substrate; supersedes ProRL Agent. Polar's `prefix_merging` is the sophisticated version of Search-R1's retrieved-token loss masking.
- [[nemo-gym]] — NVIDIA's environment-catalog framework; Search-R1-style QA tasks are part of its 84-benchmark inventory.
- [[tool-use-rl]] — The broader tool-use RL landscape.
- [[multi-step-reasoning-rl]] — Adjacent RL setup for longer-horizon reasoning.
- [[rl-training-frameworks]] — The veRL/OpenRLHF/TRL landscape Search-R1 sits in.

## References

- Paper: Jin et al., *Search-R1: Training LLMs to Reason and Leverage Search Engines with RL*, COLM 2025. [arXiv:2503.09516](https://arxiv.org/abs/2503.09516)
- DeepSeek-R1 / R1-Zero (the direct lineage): [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
- GRPO: Shao et al., 2024. [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
- PPO: Schulman et al., 2017. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
- veRL: [github.com/volcengine/verl](https://github.com/volcengine/verl) — the RL framework Search-R1 forks
- Search-o1 (baseline): Li et al., 2025
- IRCoT (baseline): Trivedi et al., 2022
- ReAct (the original search-as-tool prompting): Yao et al., 2023
- Toolformer: Schick et al., 2023
- Datasets: NQ (Kwiatkowski 2019), TriviaQA (Joshi 2017), PopQA (Mallen 2022), HotpotQA (Yang 2018), 2WikiMultiHopQA (Ho 2020), Musique (Trivedi 2022b), Bamboogle (Press 2022)
- E5 retriever: Wang et al., 2022. [arXiv:2212.03533](https://arxiv.org/abs/2212.03533)
