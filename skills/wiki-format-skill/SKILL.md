---
name: wiki-format-skill
description: "Authoritative guide for reading papers, writing wiki entries, and formatting Markdown in Shiki's Knowledge Wiki (~/Desktop/Shiki's Knowledge Wiki/). Consolidates the paper-reading 8-axis framework, wiki-writing report style, and the Obsidian-callout + LaTeX format. Invoke whenever I'm asked to read/review/精读 a paper, add an entry, edit an existing wiki page, or maintain the wiki graph."
---

# WIKI-Format-Skill

Single source of truth for everything related to Shiki's Knowledge Wiki at `~/Desktop/Shiki's Knowledge Wiki/`. This skill consolidates three previously separate memories — *how I read papers*, *how I write wiki pages*, and *what the formatting standard is* — into one document. When in doubt about wiki work, this skill outranks any older memory note.

## Vault basics

- **Location**: `~/Desktop/Shiki's Knowledge Wiki/` (NOT `~/Shiki's Knowledge Wiki/` — that path is empty/stale).
- **Bilingual**: parallel `EN/` and `CN/` vaults — every page exists in both languages with identical structure.
- **Topic categories** (each is a folder under `EN/wiki/` and `CN/wiki/`):
  `llm-inference/`, `rl-infra/`, `ml-infra/`, `ml-sys/`, `agentic-rl/`, `ai-agent/`, `llm-serving-for-agents/`.
- **Auxiliary files** in each vault root: `CLAUDE.md` (vault schema), `index.md` (catalog), `log.md` (append-only changelog).
- **Sources**: raw papers, articles, code references go under `EN/sources/` and `CN/sources/`.
- **Git**: the vault is a git repo with remote `git@github.com:shikicloud/MLsys.git` on branch `main`.
- **Verify before acting**: paths get reorganized — always `find ~/Desktop -maxdepth 4 -name "*Knowledge Wiki*"` to confirm location before saving files.

---

## Part 1 — How I read a paper (8-axis framework)

The framework is the **private analytical lens** I use to *understand* a paper. It is NOT the structure of the published wiki page — see Part 2 for that. Mixing the two produces academic-sounding pages that read like notes-to-self.

When the user asks me to read, review, summarize, or 精读 a paper, I work through these eight orthogonal axes:

1. **Position** — what was the field doing? Why is the *status quo* unsatisfactory? Compare against named prior work in a table when multiple alternatives exist.
2. **Motivation (立意)** — what *general principle* do the authors think the field is missing? Distill it to one sentence. A great paper has a philosophy underneath the engineering or math.
3. **Core idea (the delta)** — strip everything else away. What is the smallest defensible new thing? Ask: "if I removed this one component, would the result still hold?"
4. **The how (method / system / proof)** — concrete mechanism: architecture for systems, algorithm for ML, theorem + proof structure for theory, methodology for empirical work.
5. **Implementation reality** — what does the code / proof appendix / supplementary material actually show that the abstract glosses over? *Always* fetch the source repo or appendix; never rely on the abstract alone. This axis frequently reveals hidden constraints and pragmatic choices.
6. **Evidence (experiments)** — does the empirical or theoretical evidence support the claimed delta? Critically: **what is NOT measured?** What ablations are missing? What baselines are absent? For theory: are the assumptions realistic?
7. **Limitations** — both what authors acknowledge AND what I notice they don't. Be specific — "only validates on DAPO, not PPO/GRPO/RLOO" beats "limited evaluation."
8. **Generalization** — what broader pattern does the work expose? Does it foreshadow how the field will evolve? What does it predict about the next 12 months?

### Adapt by paper type

- **Systems / infra**: "the how" = architecture; "implementation reality" = source code (read `start_server.py`, kernels, the Pydantic API models).
- **ML / RL methods**: "the how" = algorithm + loss function; "implementation reality" = training tricks in appendix, hyperparameters, data preprocessing.
- **Theory**: "the how" = main theorem + proof sketch; "implementation reality" = full proofs in appendix; "evidence" = does the bound match observed behavior?
- **Empirical / benchmark**: "the how" = experimental protocol; "implementation reality" = raw data, statistical tests; "evidence" = effect size, reproducibility, threats to validity.

### Reading order ≠ report order

- **Systems**: abstract → intro → conclusion → figures → method → experiments → related work → **code**.
- **Theory**: abstract → intro → main theorem statement → proof sketch → experiments → related → **full proofs**.
- **ML methods**: abstract → main results table → method → ablations → appendix hyperparameters → related work.

Reading order reveals weaknesses faster than the paper's own structure. The wiki page I produce uses Part 2's report skeleton, NOT this analytical-axis sequence.

---

## Part 2 — How I write a wiki page (report style, not framework dump)

A wiki page is a **report to others**, not my analytical scratchpad. Use content-named headers that name the *thing* discussed, not the analytical role. Never publish headers like "Position", "Motivation", "Core Idea", "Implementation Reality", "Generalization" — replace each with a content-descriptive header.

### Recommended report skeleton for paper reviews

```markdown
# <Paper Title>

> [!info] Paper metadata
> - **Paper**: arxiv link, venue, date
> - **Code**: github link, branch, commit
> - **Authors**: list

> [!abstract]+ TL;DR
> 2–3 sentences: what it is, why it matters, the headline result.

---

## Background — narrative covering Position + Motivation
What was the problem? Why was the existing approach unsatisfying?
Comparison table here if multiple prior works.

---

## The key idea — name the thing (e.g., "Rollout-as-a-Service", "RadixAttention")
> [!quote] The contribution in one sentence
> ...

Three or four sub-claims that hold the contribution up.

---

## How it works — content-descriptive subsections weaving in code
- Architecture diagram (ASCII or Mermaid)
- Real code from the source repo (the actual ABC class, the Pydantic model, the kernel)
- Inline implementation-reality notes — don't separate them into their own section

---

## Experiments
Setup + main result table + ablations. Note what isn't measured inline.

---

## Strengths and limitations
Blended prose. Author-acknowledged + my own critiques.
> [!bug] for OSS bugs / docs gaps if any.

---

## What this means
1–2 paragraphs of perspective: broader pattern, prediction. Be opinionated.

---

## Source code & reproduction
Quick-start commands. Table mapping file → role.

---

## Related reading
[[wikilinks]] to neighboring wiki pages.
```

### Style guide — non-negotiable

1. **Narrative first, structure second.** Paragraphs that explain. Lists/tables only when they communicate faster than prose.
2. **Content-named headers**, never analytical-role-named headers. ✅ "Token-In/Token-Out" — ❌ "Implementation Reality".
3. **Show real code and interfaces, not just descriptions.** Paper pages must include the actual ABC class signatures, dataclass fields, Pydantic request models, kernel snippets, config examples — taken directly from the source. A page that *describes* "there is an AgentHandler abstract class" is weaker than one that *shows* the class and its seven abstract methods. Always fetch the source repo and quote real code blocks; cite the file path inline (e.g., `openhands/nvidia/registry.py`).
4. **Opinions are welcome but signaled.** When I write "the real contribution is X, not Y," frame it as my view, not the paper's claim.
5. **Critiques are inline and specific.** Don't reserve a "limitations" section as a graveyard; weave them where relevant. Be specific — "only validates DAPO" beats "limited evaluation".
6. **Cross-link aggressively.** Every wiki page is part of a graph; cite [[neighbors]] when concepts overlap.
7. **No "How I Read Papers" preamble** in the wiki page. That's metacommentary; keep it in this skill, not in a published page.
8. **Bilingual mirror.** EN + CN parallel structure, same callout types, same math, translated prose.

### When a paper cites others — pull on the wiki graph

When the paper I'm reviewing cites or builds on other named work:

1. **Mention the related papers in the current page** — compare/contrast, position the new paper in the lineage, not just citation-list dump.
2. **Update existing wiki pages** that are conceptually adjacent. If an existing page doesn't yet mention the lineage, add a subsection that links to the new page.
3. **Create a new synthesis/family page** when the cited works form a coherent technique family with a clear development arc (e.g., QuIP → QuIP# → QuaRot → SpinQuant → BDR became `[[rotation-based-quantization]]`). This page is a navigation hub, not a paper review — it explains the shared insight, lineage, comparison table, practical guidance, and open questions.

The rule: **the wiki is a graph**. Every new paper should pull on the surrounding topology — adding nodes for cited works that deserve them and adding edges to existing nodes that should now reference the new work.

What does NOT require a new page: papers cited only once or twice in passing. Reserve new pages for (a) papers I do a full code-walking review on, or (b) coherent families/lineages worth a synthesis hub.

### Logging Q&A inline (the "Shiki:" / "Answer:" pattern)

When the user asks a follow-up question about a specific paper, **record the question AND the answer on that paper's wiki page**, but do NOT collect them in a `## Q&A` section at the bottom. Instead:

1. **Inline placement** — put the Q&A *at the place where the discussed text appears*. If the user is asking about a paragraph in the "Background" section, the Q&A goes right after that paragraph. The page becomes a stratified study record — original review + clarifications layered exactly where the confusion was.
2. **Distinct visual format** — use an Obsidian callout `> [!question]+ Shiki — <short title> (YYYY-MM-DD)`. The `+` keeps it expanded.
3. **Compact-paragraph answer style** — answer body should be **3–5 compact paragraphs**, each one short (2–4 sentences) and self-contained. Avoid heavy bullet lists for explanatory answers; prose flows better when the user is trying to understand a concept.
4. **Cross-link** to the paper's own sections (`[[#Section Name]]`) so the answer extends the page's internal graph.
5. **Mirror in both EN and CN** with identical structure; keep "Shiki" as the questioner label in both.
6. **Don't accumulate at the bottom** — the inline location IS the index. Chat reply can be terse; the wiki version is the authoritative complete answer.

Format template:

```markdown
> [!question]+ Shiki — <short topic title> (YYYY-MM-DD)
>
> *(Quoted)*: <user's verbatim question or quoted passage they're asking about>
>
> Paragraph 1 — what the term means / context.
>
> Paragraph 2 — why it matters / the mechanism.
>
> Paragraph 3 — the consequence / specific number or example.
>
> Paragraph 4 — how it connects to the rest of the page (with `[[#Section]]` cross-links).
```

---

## Part 3 — Markdown format standard

Validated on `[[saw-int4]]` and `[[saw-int4]]` (CN). Apply to all paper-review pages. For long pre-existing concept pages, apply at minimum the top-of-page callouts (info + abstract) and key-result callouts (important / example / warning); deep per-section reformat is optional.

### Obsidian callouts — standard set

| Callout                 | Use for                                                                                            |
| ----------------------- | -------------------------------------------------------------------------------------------------- |
| `> [!info]`             | Paper / repo metadata block at the top                                                             |
| `> [!abstract]+`        | TL;DR (`+` makes it open by default)                                                                |
| `> [!important]`        | Headline numbers, the cliff-edge collapse, "this is the point"                                     |
| `> [!quote]`            | One-line statement of the paper's contribution                                                     |
| `> [!tip]`              | Recommended config, default mode, practical guidance                                                |
| `> [!note]`             | Sidebar observations, constraints, kernel-fusion side effects                                       |
| `> [!example]`          | Concrete worked-out memory math, throughput math, dimensional analysis                              |
| `> [!warning]`          | Empty ablation tables, scope caveats, things the paper claims but doesn't show                      |
| `> [!bug]`              | Real bugs in the OSS release (port mismatches, doc errors, install gotchas)                         |
| `> [!question]+ Shiki — title (YYYY-MM-DD)` | User Q&A logged inline near discussed text                                              |
| `> [!success]`          | (rare) when something works as advertised against expectations                                      |

Don't overdo it — every callout that's also-ran loses distinctiveness. Aim for 4–8 callouts on a typical paper-review page.

### LaTeX math

Use display math for definitions / formulas:

```markdown
$$
H_2 = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}, \qquad H_{2n} = H_2 \otimes H_n
$$
```

Inline math for complexity, ratios, simple expressions:

| Source phrase           | Use                       |
| ----------------------- | ------------------------- |
| `O(d log d)`            | `$O(d \log d)$`            |
| `O(d²)`                 | `$O(d^2)$`                 |
| `1/√H`, `1/√d`          | `$1/\sqrt{H}$`, `$1/\sqrt{d}$`           |
| `H_d^T · H_d = I`       | `$H_d^\top H_d = I$`       |
| `softmax(QK^T / √d) · V`| `$\text{softmax}(QK^\top / \sqrt{d}) \cdot V$` |
| `i XOR 2^s`             | `$i \oplus 2^s$`           |
| `[min, max]`            | `$[\min, \max]$`           |

Use `\text{...}` inside math for plain words (variables that aren't single letters). Step-by-step computation chains are fine inside `aligned` blocks. **Don't** use math just to show off — code and prose render better for things like "a 7-element list of fields"; math wins for true formulas.

### Visual hierarchy

- **`---` horizontal rules** between top-level (`##`) sections. Major concepts get breathing room.
- **`####` sub-subsections** inside long `###` sections (e.g., the fused-kernel section in `[[saw-int4]]` splits butterfly / full kernel body / launcher).
- **Tables for "file → role" lists**, comparison matrices, mode matrices.
- **Right-align numeric columns** in tables (`| ----: |`).
- **Code blocks** with language hints (`python`, `bash`, `toml`, etc.) — never untyped fences.
- **ASCII diagrams** are fine for architecture; Mermaid only when the diagram benefits from it.
- **Inline code** for identifiers (`AgentHandler`, `head_dim`, `--kv-cache-dtype`); math for expressions; bold for emphasis on prose nouns.

### Page header / frontmatter

```yaml
---
title: "Paper Title or Concept Name"
category: <one of the seven topic categories>
tags: [tag1, tag2, paper-review or family-overview or concept]
created: YYYY-MM-DD
updated: YYYY-MM-DD
status: seed | growing | mature
paper: arXiv:NNNN.NNNNN     # for paper reviews only
code: github URL            # for paper reviews only
---
```

### Index and log discipline

- Adding a new wiki page: add a one-line entry under the right category in **both** `EN/index.md` and `CN/index.md`.
- Adding a new wiki page or major edit: append a `[INGEST]` / `[EXPANDED]` / `[NEW]` / `[Q&A]` line to **both** `EN/log.md` and `CN/log.md` under today's date.
- Citation files: paper-review pages get `EN/sources/papers/<slug>/citation.md` and `CN/sources/papers/<slug>/citation.md` with arxiv / repo / authors metadata.

---

## Operations cookbook

### "Read this paper and put it in the wiki"

1. Determine paper type (systems / methods / theory / empirical) → reading order.
2. Fetch the arXiv page; if it's a systems paper, also fetch the GitHub repo (README + key source files).
3. Mentally walk the 8-axis framework as I read.
4. Decide placement: which `wiki/<category>/` folder fits.
5. Write the page using the Part 2 report skeleton, Part 3 format. Show real code; don't just describe.
6. Update both `index.md`s and both `log.md`s.
7. Add a `citation.md` in `sources/papers/<slug>/` (both languages).
8. Pull on the wiki graph: update neighboring pages with cross-links to the new page; consider creating a synthesis page if the lineage is coherent.
9. Mirror everything in EN + CN with parallel structure.

### "Answer my question about this paper"

1. Locate the discussed paragraph in the paper's wiki page.
2. Reply in chat with a tight version of the answer.
3. Add an inline `> [!question]+ Shiki — <title> (date)` callout right after that paragraph in BOTH EN and CN versions.
4. Format the answer as 3–5 compact paragraphs with cross-links back into the page.
5. Append a `[Q&A]` line to both `log.md`s.

### "Update the wiki format"

1. Edit this `SKILL.md` (the canonical source).
2. Mirror the change in `~/Desktop/Shiki's Knowledge Wiki/skills/wiki-format-skill/SKILL.md` for git tracking.
3. If the change affects existing pages: apply at least the top-of-page callouts to high-traffic pages; apply full reformat opportunistically over time.

### "Push to GitHub"

1. `cd ~/Desktop/Shiki's Knowledge Wiki/`
2. `git status` to see uncommitted work.
3. Stage relevant files (avoid `.obsidian/workspace.json` if it's noisy).
4. Commit with a clear summary referencing what was added/expanded.
5. `git push origin main`.

---

## Memory pointers

- `MEMORY.md` should reference this skill rather than duplicating its content. The previous `feedback_paper_reading_framework.md` memory is superseded by Parts 1–3 of this skill.
- `reference_knowledge_wiki.md` and `user_learner_llm_inference.md` remain as separate reference/user memories.
