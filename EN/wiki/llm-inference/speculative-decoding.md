---
title: "Speculative Decoding: Draft-Verify Acceleration"
category: llm-inference
tags: [speculative-decoding, draft-model, eagle, medusa, lossless-acceleration]
created: 2026-04-13
updated: 2026-05-07
status: mature
---

# Speculative Decoding: Draft-Verify Acceleration

> [!abstract]+ TL;DR
> Autoregressive LLM decoding is **memory-bandwidth-bound** — the GPU's compute is vastly underutilized while waiting on weight loads. Speculative decoding exploits this: verifying $K$ candidate tokens in one forward pass costs about the same as generating 1, so a lightweight draft mechanism proposes $K$ tokens and the target model verifies them in parallel via rejection sampling. **Output distribution is mathematically identical** to the target model — lossless speedup. Independently proposed by Leviathan et al. (2023) and Chen et al. (2023). State of the art: **EAGLE-3** (NeurIPS 2025) at 3–6× speedup; production deployments in [[vllm|vLLM]] and [[sglang|SGLang]].

```
Traditional:  5 forward passes → 5 tokens
Speculative:  1 draft + 1 verify → up to 5+ tokens
```

Independently proposed by Leviathan et al. (2023) and Chen et al. (2023).

## Core Algorithm

```
 1. DRAFT:   Lightweight model generates gamma candidate tokens
 2. VERIFY:  Target model runs single forward pass on all candidates
 3. ACCEPT:  Rejection sampling compares distributions sequentially
 4. OUTPUT:  Accepted tokens + 1 resampled token from corrected distribution
```

**Modified rejection sampling** for token `x ~ q(x)`:
- Accept probability = `min(1, p(x)/q(x))`
- On rejection, resample from `p'(x) = max(0, p(x) - q(x)) / Z`
- This guarantees: `P(output x) = p(x)` (target distribution exactly)

**Expected tokens per step** (gamma = draft length, alpha = acceptance rate):

```
E[tokens] = (1 - alpha^(gamma+1)) / (1 - alpha)
```

| alpha | gamma=5 | E[tokens/step] |
|-------|---------|----------------|
| 0.6 | 5 | ~2.37 |
| 0.7 | 5 | ~2.94 |
| 0.8 | 5 | ~3.78 |
| 0.9 | 8 | ~6.13 |

## Token Tree Verification

Instead of a single chain, tree-structured speculation explores multiple candidates per position:

```
              [The]
             /  |  \
         [cat] [dog] [big]
         / \     |
      [sat] [is] [ran]
```

Verified efficiently using a custom causal attention mask where each node attends only to its root-to-node path. Even if one branch is rejected, others may succeed.

## Draft Model Approaches

| Method | Accept Rate | Speedup | Extra Memory | Training | Notes |
|--------|------------|---------|--------------|----------|-------|
| Independent draft | 0.6-0.8 | 1.5-2.5x | High | No | Simplest |
| LayerSkip | 0.5-0.7 | 1.3-1.8x | Zero | No | Self-speculative |
| EAGLE | 0.7-0.85 | 2.0-3.5x | Low | Yes | Feature-level AR |
| EAGLE-2 | 0.75-0.9 | 2.5-4.5x | Low | Yes | Dynamic draft trees |
| **EAGLE-3** | **0.8-0.95** | **3.0-6.5x** | Low | Yes | **SOTA**, multi-level fusion |
| Medusa | 0.6-0.8 | 2.2-3.6x | Very low | Yes | K independent heads |
| Lookahead | N/A | 1.5-2.0x | Zero | No | Jacobi iteration |
| N-gram | Varies | 1.0-3.0x | Zero | No | Best for code/summary |

**EAGLE series** (current SOTA): Autoregression at the *feature level* (penultimate hidden state) is far more predictable than at the token level. EAGLE-3 (NeurIPS 2025) fuses low/mid/high-level features with training-time test simulation. [arXiv:2503.01840](https://arxiv.org/abs/2503.01840)

**Medusa** (ICML 2024): K extra feed-forward heads predict positions +1 to +K from the current hidden state (non-autoregressive). Uses tree attention for candidate evaluation. [arXiv:2401.10774](https://arxiv.org/abs/2401.10774)

## Code Examples

### vLLM Configuration

```python
from vllm import LLM, SamplingParams

# EAGLE-3 speculative decoding
llm = LLM(
    model="meta-llama/Llama-3.3-70B-Instruct",
    speculative_model="yuhuili/EAGLE3-LLaMA3.3-70B-Instruct",
    speculative_method="eagle",
    num_speculative_tokens=5,
    tensor_parallel_size=4,
)

# N-gram (no extra model needed)
llm = LLM(
    model="meta-llama/Llama-3.3-70B-Instruct",
    speculative_model="[ngram]",
    num_speculative_tokens=5,
    ngram_prompt_lookup_max=4,
)

params = SamplingParams(temperature=0.7, max_tokens=512)
outputs = llm.generate(["Explain quantum computing"], params)
```

## Performance Analysis

**Speedup formula**:
```
Speedup = E[accepted_tokens] / (1 + draft_cost / verify_cost)
```

| Config | Model | Draft | Concurrency | Speedup |
|--------|-------|-------|-------------|---------|
| 4xA100 | Llama-3.3-70B | EAGLE-3 | 1 | 3.2x |
| 4xA100 | Llama-3.3-70B | EAGLE-3 | 16 | 1.4x |
| 1xH100 | Llama-3.3-70B | EAGLE-3+FP8 | 1 | 3.8x |

**When it helps**: Batch size <= 10, model >= 13B, predictable outputs, latency-sensitive.

**When it hurts**: High concurrency (32+), very short generations, creative/open-ended tasks (low acceptance), prefill-dominated workloads.

## Limitations

1. **Draft quality is everything**: Below 0.5 acceptance rate, little to no speedup.
2. **Memory overhead**: Independent draft models require extra GPU memory (EAGLE/Medusa heads are lightweight).
3. **Diminishing returns at scale**: High batch sizes shift decode from memory-bound to compute-bound, making parallel verification no longer "free".
4. **No prefill benefit**: Only accelerates decode; prefill needs other optimizations like [[prefill-decode-disaggregation]].
5. **Temperature sensitivity**: Higher temperature = lower acceptance rate (harder to predict random sampling).
6. **Domain mismatch**: Draft model needs to match the target model's fine-tuned distribution.

## References

- Leviathan et al., "Fast Inference from Transformers via Speculative Decoding," ICML 2023. [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)
- Chen et al., "Accelerating Large Language Model Decoding with Speculative Sampling," 2023. [arXiv:2302.01318](https://arxiv.org/abs/2302.01318)
- Li et al., "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty," ICML 2024. [arXiv:2401.15077](https://arxiv.org/abs/2401.15077)
- Li et al., "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees," EMNLP 2024. [arXiv:2406.16858](https://arxiv.org/abs/2406.16858)
- Li et al., "EAGLE-3: Scaling Up Speculative Decoding with Feature Fusion and Training-time Test," NeurIPS 2025. [arXiv:2503.01840](https://arxiv.org/abs/2503.01840)
- Cai et al., "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads," ICML 2024. [arXiv:2401.10774](https://arxiv.org/abs/2401.10774)
- Fu et al., "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding," 2024. [arXiv:2402.02057](https://arxiv.org/abs/2402.02057)
- Elhoushi et al., "LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding," 2024. [arXiv:2404.16710](https://arxiv.org/abs/2404.16710)

## Related Pages

- [[vllm]] -- Supports multiple speculative decoding strategies
- [[sglang]] -- Deep EAGLE-3 integration
- [[continuous-batching]] -- Scheduling interactions with speculation
- [[kv-cache-optimization]] -- KV cache management for draft models
- [[quantization]] -- Quantized draft models for reduced overhead
- [[prefill-decode-disaggregation]] -- Spec decoding only helps decode phase
- [[tensorrt-llm]] -- Supports EAGLE-3, MTP, N-gram speculation
