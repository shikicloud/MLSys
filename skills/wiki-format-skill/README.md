# wiki-format-skill

This folder contains the canonical Markdown source for the **WIKI-Format-Skill** Claude Code skill that governs Shiki's Knowledge Wiki.

## Install / sync

The active skill lives at `~/.claude/skills/wiki-format-skill/SKILL.md`. To keep it in sync with this repo's tracked copy, you can either:

```bash
# Option A: copy the tracked file into ~/.claude/skills/
cp skills/wiki-format-skill/SKILL.md ~/.claude/skills/wiki-format-skill/SKILL.md

# Option B: replace the active copy with a symlink (do this once)
rm ~/.claude/skills/wiki-format-skill/SKILL.md
ln -s "$PWD/skills/wiki-format-skill/SKILL.md" ~/.claude/skills/wiki-format-skill/SKILL.md
```

Option B keeps GitHub and the local Claude install always in sync.

## What's in the skill

`SKILL.md` consolidates three previously separate memories:

1. **Part 1 — How I read a paper.** The 8-axis analytical framework (Position / Motivation / Core idea / Method / Implementation / Evidence / Limitations / Generalization), adapted by paper type (systems vs. ML methods vs. theory vs. empirical), with reading-order vs. report-order distinction.
2. **Part 2 — How I write a wiki page.** Report-style skeleton with content-named headers, the bilingual EN+CN mirror rule, the cite-others-pull-the-graph principle, and the inline Shiki/Answer Q&A pattern.
3. **Part 3 — Markdown format standard.** The Obsidian callout vocabulary (`info`, `abstract`, `important`, `quote`, `tip`, `note`, `example`, `warning`, `bug`, `question`), LaTeX math conventions, visual hierarchy rules, frontmatter template, and index/log discipline.

Plus an operations cookbook for the four common workflows (ingest a paper, answer a Q&A, update format, push to GitHub).
