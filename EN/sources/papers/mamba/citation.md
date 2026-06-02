---
title: "Mamba — Citation"
type: paper-citation
created: 2026-06-02
---

# Mamba: Linear-Time Sequence Modeling with Selective State Spaces

- **arXiv**: [2312.00752](https://arxiv.org/abs/2312.00752)
- **Code**: [state-spaces/mamba](https://github.com/state-spaces/mamba) (Apache-2.0)
- **Authors**: Albert Gu (CMU), Tri Dao (Princeton)
- **Date**: December 2023
- **Venue**: NeurIPS 2024
- **License**: Code Apache-2.0
- **Wiki page**: [[mamba]]

## Why this paper matters

The foundational architecture page for the hybrid-attention era. Every 2026 hybrid-attention production LLM — Qwen3-Next, Qwen3.5, Qwen3.6, Nemotron-H, Kimi Linear, MiMo-V2-Flash, DeepSeek-V3.2 NSA — has linear-attention layers descended from Mamba. Understanding Selective SSM, chunkwise parallel scan, and Mamba's specific weaknesses (associative recall, ICL) is prerequisite for understanding why the industry moved to *hybrid* architectures rather than going all-Mamba.

## Key technical contributions

1. Selective SSM (S6): input-dependent state-space transitions that break LTI, enabling content-aware filtering and selective memory.
2. Hardware-aware parallel scan: Blelloch scan + kernel fusion + SRAM-resident state + recompute-in-backward. 20-40× speedup over naive PyTorch.
3. Mamba block: single homogeneous block replacing both attention and FFN.
4. First SSM to match Transformer++ on language modeling at 3B parameters and 300B tokens.
5. 5× faster decoding throughput at seqlen 2K; linear-time sequence scaling.

## Successor lineage (2024-2026)

- **Mamba-2 / SSD** (arXiv:2405.21060, May 2024) — Structured State-Space Duality unifies SSM and attention as structured matrices.
- **Gated DeltaNet** (arXiv:2412.06464, ICLR 2025) — delta-rule variant; powers Qwen3-Next, Qwen3.5, Qwen3.6.
- **Jamba** (AI21, Mar 2024) — first production hybrid (1:7 attention:Mamba).
- **Lightning Attention** (Qin et al., 2024) — used by MiniMax.
- **KDA (Kimi Delta Attention)** — used by Kimi Linear, K2 family.
- **Mamba-3** (2026) — active research.

## BibTeX

```bibtex
@article{gu2023mamba,
  title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023}
}
```
