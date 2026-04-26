# Shiki's Knowledge Wiki — Schema (EN)

This is the schema file that defines how the LLM should maintain this wiki vault.

## Vault Purpose

A personal knowledge base covering: LLM Inference & Serving, RL Infrastructure, ML Infrastructure, ML Systems, Agentic RL, AI Agents, and LLM Serving for AI Agents.

## Language

This vault is maintained in **English**. A parallel Chinese vault exists at `../CN/`.
When updating this vault, always update the CN vault simultaneously.

## Folder Structure

```
EN/
├── CLAUDE.md          # This schema file
├── index.md           # Content-oriented catalog organized by category
├── log.md             # Append-only chronological log
├── sources/           # Raw immutable sources
│   ├── papers/        # Academic papers (PDF, notes)
│   ├── articles/      # Web articles, blog posts
│   ├── notes/         # Personal notes
│   └── code/          # Code snippets, repo references
└── wiki/              # LLM-maintained pages
    ├── llm-inference/       # LLM inference & serving
    ├── rl-infra/            # RL infrastructure (RLHF, PPO, GRPO, DPO)
    ├── ml-infra/            # ML infrastructure (distributed training, GPUs)
    ├── ml-sys/              # ML systems (MLOps, pipelines)
    ├── agentic-rl/          # Agentic RL
    ├── ai-agent/            # AI agents, tool use, multi-agent
    └── llm-serving-for-agents/  # LLM serving optimized for agents
```

## Wiki Page Format

Every wiki page should follow this template:

```markdown
---
title: Page Title
category: llm-inference | rl-infra | ml-infra | ml-sys | agentic-rl | ai-agent | llm-serving-for-agents
tags: [tag1, tag2]
created: YYYY-MM-DD
updated: YYYY-MM-DD
status: seed | growing | mature
---

# Page Title

## Overview
Brief summary of the topic.

## Key Concepts
Detailed explanation.

## Related Work
Links to related wiki pages using [[wikilinks]].

## References
- Source citations with links.
```

## Operations

### Ingest
1. Place raw source in `sources/` under the appropriate subfolder
2. Create or update relevant wiki pages in `wiki/`
3. Add cross-references using `[[wikilinks]]`
4. Update `index.md` with new entries
5. Append to `log.md` with timestamp

### Query
1. Search wiki pages for relevant information
2. Synthesize answers with citations to wiki pages
3. If the answer produces valuable new synthesis, file it as a new wiki page

### Lint
1. Check for contradictions between pages
2. Identify stale claims that need updating
3. Find orphaned pages (no incoming links)
4. Flag missing cross-references
5. Verify source citations are valid

## Cross-Reference Conventions
- Use Obsidian `[[wikilinks]]` for internal links
- Use `[[page#section]]` for section-specific links
- Tag pages with YAML frontmatter `tags` field
- Categories map to wiki subfolders
