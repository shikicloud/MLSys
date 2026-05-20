---
title: "Speculative Decoding: Draft-Verify Acceleration"
category: llm-inference
tags: [speculative-decoding, draft-model, eagle, medusa, lossless-acceleration]
created: 2026-04-13
updated: 2026-05-20
status: mature
---

# Speculative Decoding: Draft-Verify Acceleration

> [!abstract]+ TL;DR
> Autoregressive LLM decoding is **memory-bandwidth bound** -- GPU compute sits idle while weights are streamed in. Speculative decoding exploits this: a single forward pass that verifies $K$ candidate tokens costs roughly the same as generating 1, so a lightweight drafting mechanism proposes $K$ tokens and the target model verifies them in parallel via rejection sampling. **The output distribution is mathematically identical to the target model's** -- lossless acceleration. Proposed independently by Leviathan et al. (2023) and Chen et al. (2023). Current SOTA: **EAGLE-3** (NeurIPS 2025) at 3-6x speedup; deployed in production by [[vllm|vLLM]] and [[sglang|SGLang]].

```
Traditional autoregressive decoding (1 token per step):
  Step 1: [The] -> target -> [cat]
  Step 2: [The cat] -> target -> [sat]
  Step 3: [The cat sat] -> target -> [on]
  Step 4: [The cat sat on] -> target -> [the]
  Step 5: [The cat sat on the] -> target -> [mat]
  Total: 5 forward passes -> 5 tokens

Speculative decoding (potentially many tokens per step):
  Draft model guesses fast: [cat, sat, on, the, mat]
  Target model verifies in one shot: all accepted!
  Total: 1 draft + 1 verify -> 5 tokens
```

Speculative decoding was independently introduced by Leviathan et al. (2023) and Chen et al. (2023) and has since become a standard optimization in production inference systems.


## Core Principle

### The Draft-Verify Paradigm

Speculative decoding proceeds in three stages:

```
+-------------------------------------------------------------+
|                  Speculative Decoding Pipeline              |
+-------------------------------------------------------------+
|                                                             |
|  Stage 1: Draft generation                                  |
|  +-----------+                                              |
|  | Draft     |--> t1, t2, t3, ..., tK  (K candidate tokens) |
|  | model     |    (fast, lower quality)                     |
|  | (small)   |                                              |
|  +-----------+                                              |
|        |                                                    |
|        v                                                    |
|  Stage 2: Parallel verification                             |
|  +-----------+                                              |
|  | Target    |--> Single forward pass on                    |
|  | model     |    [prefix, t1, t2, ..., tK], producing      |
|  | (full)    |    a distribution at every position          |
|  +-----------+                                              |
|        |                                                    |
|        v                                                    |
|  Stage 3: Accept / reject                                   |
|  +---------------------------------------+                  |
|  | Compare position by position:         |                  |
|  |   t1: P_target(t1) / P_draft(t1) >= r?|                  |
|  |     -> accept, move on to t2          |                  |
|  |   t2: P_target(t2) / P_draft(t2) >= r?|                  |
|  |     -> accept, move on to t3          |                  |
|  |   t3: P_target(t3) / P_draft(t3) <  r?|                  |
|  |     -> reject! resample t3' from      |                  |
|  |        the residual distribution      |                  |
|  |   Output: t1, t2, t3' (3 tokens this step)               |
|  +---------------------------------------+                  |
|                                                             |
|  Loop: keep going with a fresh draft-verify round           |
+-------------------------------------------------------------+
```

### Why Does Verifying K Tokens Cost About the Same as Generating 1?

This follows from the computational profile of LLM inference:

- **Prefill** (processing the input prompt) is **compute-bound**: batched matmuls saturate the GPU's compute units.
- **Decode** (token-by-token generation) is **memory-bandwidth bound**: each step is a single matrix-vector multiply, and most of the time is spent streaming weights from HBM.

Verifying K candidate tokens is essentially a tiny prefill: the K tokens form a small batch that runs in parallel. For small K (say 5-10), the extra compute is negligible because the bottleneck is loading weights, and the weights are loaded only once.

### The Mathematical Basis of the Lossless Guarantee

The defining property of speculative decoding is **losslessness**: the final token distribution is identical to what the target model would have produced on its own. This is achieved through **modified rejection sampling**.


## The Verification Algorithm

### A Quick Recap of Standard Rejection Sampling

Classical rejection sampling draws from a target distribution p(x) that is hard to sample from but easy to evaluate, using an easy-to-sample proposal q(x):

1. Sample x from q(x)
2. Compute acceptance probability alpha = p(x) / (M * q(x)), where M = max_x p(x)/q(x)
3. Accept x with probability alpha; otherwise reject and retry

### Speculative Decoding's Modified Rejection Sampling

Speculative decoding uses a modified version. Let:
- `p(x)` = the target model's distribution at the current position
- `q(x)` = the draft model's distribution at the current position

For a draft token `x ~ q`:

```
acceptance probability = min(1, p(x) / q(x))
```

**Step by step**:

1. Sample token x from the draft model with probability q(x)
2. Compute the ratio r = p(x) / q(x)
3. If r >= 1 (the target prefers x at least as much as the draft), **always accept**
4. If r < 1, accept with probability r, reject with probability 1-r
5. On rejection, resample from the **residual distribution**:

```
p'(x) = max(0, p(x) - q(x)) / Sigma_x max(0, p(x) - q(x))
```

**Why does this preserve the target distribution?**

The distribution of an accepted token is:

```
P(accept x) = q(x) * min(1, p(x)/q(x))
            = min(q(x), p(x))
```

On rejection, we sample from the residual `p'(x) prop max(0, p(x) - q(x))`. Combining the two cases:

```
P(output x) = min(q(x), p(x)) + [1 - Sigma_x min(q(x), p(x))] * p'(x)
            = min(q(x), p(x)) + max(0, p(x) - q(x))
            = p(x)
```

So the output distribution is exactly p(x).

### Multi-Token Verification

For K draft tokens (t1, t2, ..., tK):

```python
def speculative_verify(draft_tokens, draft_probs, target_probs):
    """
    draft_tokens: [t1, t2, ..., tK] draft token sequence
    draft_probs:  [q1, q2, ..., qK] draft model distributions per position
    target_probs: [p1, p2, ..., pK, pK+1] target distributions per position
    """
    accepted = []
    for i in range(K):
        ti = draft_tokens[i]
        r = random.uniform(0, 1)
        
        if r < min(1, target_probs[i][ti] / draft_probs[i][ti]):
            # Accept ti
            accepted.append(ti)
        else:
            # Reject ti, resample from the residual
            residual = np.maximum(0, target_probs[i] - draft_probs[i])
            residual /= residual.sum()
            new_token = np.random.choice(vocab, p=residual)
            accepted.append(new_token)
            return accepted  # discard remaining tokens
    
    # All K tokens accepted! Sample one bonus token from pK+1
    bonus_token = np.random.choice(vocab, p=target_probs[K])
    accepted.append(bonus_token)
    return accepted  # K+1 tokens produced
```

### Acceptance Rate Analysis

The **acceptance rate alpha** is the probability that a draft token gets accepted. It governs the speedup:

```
alpha = E_x~q [min(1, p(x)/q(x))]
      = Sigma_x min(p(x), q(x))
      = 1 - (1/2) * Sigma_x |p(x) - q(x)|    (relation to TV distance)
```

That is, acceptance rate = 1 - TV_distance(p, q). The closer the draft's distribution is to the target's, the higher the acceptance.

**Expected tokens per step** (draft length gamma):

```
E[tokens_per_step] = (1 - alpha^(gamma+1)) / (1 - alpha)
```

| alpha (acceptance) | gamma=5 | Expected tokens/step |
|---------------|---------|--------------|
| 0.5 | 5 | ~1.97 |
| 0.6 | 5 | ~2.37 |
| 0.7 | 5 | ~2.94 |
| 0.8 | 5 | ~3.78 |
| 0.9 | 5 | ~5.22 |
| 0.9 | 8 | ~6.13 |

Raising the acceptance rate from 0.7 to 0.9 nearly doubles per-step throughput. This is why draft quality matters so much.


## Choosing a Draft Model

Different drafting strategies trade off quality, speed, and memory in different ways.

### Standalone Draft Model

The most direct option: use a smaller model from the same family as the draft.

- **Example**: Llama-3-8B as the draft for Llama-3-70B
- **Pros**: simple to set up, decent draft quality
- **Cons**: extra GPU memory for the draft model; both models must share a vocabulary

### Self-Speculative Decoding (LayerSkip)

Use part of the target model itself as the draft by skipping layers.

- **Idea**: skip middle layers (e.g., for a 32-layer model, use the first 8 + last 4 layers as the draft)
- **Pros**: no extra model, zero memory overhead
- **Cons**: acceptance rate is typically low (0.5-0.7) because shallow models produce limited-quality drafts
- **Representative work**: LayerSkip (Meta, 2024)

### The EAGLE Family (Current SOTA)

EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency) is currently the most efficient speculative decoding approach.

#### EAGLE (ICML 2024)

Key insight: autoregressive prediction is much easier in **feature space** (the second-to-last layer's hidden states) than in token space.

```
Traditional draft model: token -> embedding -> ... -> logits -> token -> ...
EAGLE:                   feature(t) -> lightweight_head -> feature(t+1) -> ...
```

- Train a tiny network (typically 1-2 Transformer layers) that takes the current token's feature and predicts the next token's feature
- Feature-space autoregression is more predictable than token-space autoregression
- Drafting is extremely cheap, acceptance is high

#### EAGLE-2 (EMNLP 2024)

Builds on EAGLE by introducing a **dynamic draft tree**:

- Tree structure adapts based on each candidate's confidence
- High-confidence branches expand deeper; low-confidence branches are pruned early
- Adaptively allocates the "verification budget"

#### EAGLE-3 (NeurIPS 2025)

Latest version, three core improvements:

1. **Multi-layer feature fusion**: combines features from low and middle layers, not just the second-to-last
2. **Training-time test**: simulates the autoregressive error accumulation that occurs at inference time during training
3. **A better tree topology**

Performance: **3.0-6.5x speedup**, 20-40% over EAGLE-2.

Paper: [arXiv:2503.01840](https://arxiv.org/abs/2503.01840)

### Medusa (ICML 2024)

```
                       +--> Head 1 -> predicts token at position +1
                       |
target model's last  --+--> Head 2 -> predicts token at position +2
hidden state            |
                       +--> Head 3 -> predicts token at position +3
                       |
                       +--> Head K -> predicts token at position +K
```

- Adds K independent feed-forward prediction heads on top of the target model
- Each head predicts the k-th future token directly from the current hidden state (non-autoregressive)
- Uses **tree attention** to verify multiple candidate combinations efficiently
- Speedup: 2.2-3.6x
- **Pros**: only a few small heads to train, very few extra parameters
- **Cons**: non-autoregressive prediction is lower quality than EAGLE's feature-level autoregression

Paper: [arXiv:2401.10774](https://arxiv.org/abs/2401.10774)

### Lookahead Decoding

- Borrows ideas from Jacobi iteration to guess tokens at multiple positions in parallel
- Each forward pass refines multiple positions at once
- No additional model required
- Real-world speedup is modest (1.5-2x), but the implementation is simple

### Prompt Lookup / N-gram Matching

- Look for n-gram matches in the existing context (the prompt)
- If a suffix of the generated text was seen in the prompt, reuse the continuation as the draft
- **Zero overhead**: no additional model or compute
- Excellent for code completion, summarization, and other settings where output overlaps heavily with input
- Nearly useless for creative generation

### Comparison

| Method | Acceptance | Speedup | Extra memory | Training needed | Use case |
|------|--------|------|----------|----------|----------|
| Standalone draft | 0.6-0.8 | 1.5-2.5x | High | No | General |
| LayerSkip | 0.5-0.7 | 1.3-1.8x | Zero | No | Memory-constrained |
| EAGLE | 0.7-0.85 | 2.0-3.5x | Low | Yes | General (recommended) |
| EAGLE-2 | 0.75-0.9 | 2.5-4.5x | Low | Yes | General |
| EAGLE-3 | 0.8-0.95 | 3.0-6.5x | Low | Yes | General (SOTA) |
| Medusa | 0.6-0.8 | 2.2-3.6x | Very low | Yes | Tight memory |
| Lookahead | N/A | 1.5-2.0x | Zero | No | Quick to deploy |
| N-gram | Varies | 1.0-3.0x | Zero | No | Code / summarization |


## Token Tree Verification

### Why a Tree Instead of a Chain?

A linear (chain) draft only proposes a single candidate sequence at a time. If the 2nd token is rejected, every token after it is wasted.

**Tree-shaped speculation** instead explores multiple candidates at each position, forming a tree:

```
                        [The]
                       /  |  \
                    [cat] [dog] [big]
                    / \      |
               [sat] [is]  [ran]
                |     |      |
              [on]  [very] [fast]
```

Even if one branch is rejected, another may still be accepted.

### Efficient Tree Verification

The key question: how do we verify the whole tree in a single forward pass?

The answer is a variant of the **causal attention mask**:

```
Standard causal mask (chain):      Tree attention mask:
  t1 t2 t3 t4                  t1 t2 t3 t4 t5 t6 t7
  1  0  0  0  t1               1  0  0  0  0  0  0  t1 (root)
  1  1  0  0  t2               1  1  0  0  0  0  0  t2 (child of t1)
  1  1  1  0  t3               1  0  1  0  0  0  0  t3 (child of t1)
  1  1  1  1  t4               1  1  0  1  0  0  0  t4 (child of t2)
                                1  1  0  0  1  0  0  t5 (child of t2)
                                1  0  1  0  0  1  0  t6 (child of t3)
                                1  0  1  0  0  0  1  t7 (child of t3)
```

Each node only attends to the tokens on its path from the root, implemented via a custom attention mask.

### Tree Construction Strategies

Different methods build trees differently:

1. **Static tree**: a pre-defined fixed topology (e.g., top-k expansion)
2. **Dynamic tree (EAGLE-2/3)**: branches expand based on confidence scores
3. **Medusa tree**: a Cartesian product of candidates verified with tree attention

**Why dynamic trees win**:

```
High-confidence sequences -> expand deeper (more tokens)
             "The capital of France is" -> [Paris] -> [.] -> depth 3

Low-confidence sequences  -> stop early
             "The meaning of life is" -> [a/the/to/...] -> wide, depth 1
```


## Code Examples

### Basic Speculative Decoding Pseudocode

```python
import torch
import numpy as np

def speculative_decode(
    target_model,
    draft_model,
    input_ids,
    max_tokens=100,
    gamma=5,          # draft length
    temperature=1.0,
):
    """Basic speculative decoding implementation"""
    generated = list(input_ids)
    
    while len(generated) - len(input_ids) < max_tokens:
        prefix = torch.tensor([generated])
        
        # ---- Stage 1: draft generation ----
        draft_tokens = []
        draft_probs = []
        draft_input = prefix.clone()
        
        for _ in range(gamma):
            with torch.no_grad():
                logits = draft_model(draft_input).logits[:, -1, :]
                probs = torch.softmax(logits / temperature, dim=-1)
                token = torch.multinomial(probs, 1)
                draft_tokens.append(token.item())
                draft_probs.append(probs[0].cpu().numpy())
                draft_input = torch.cat([draft_input, token], dim=-1)
        
        # ---- Stage 2: target verification ----
        # Pass prefix + all draft tokens through the target in one shot
        verify_input = torch.tensor([generated + draft_tokens])
        with torch.no_grad():
            target_logits = target_model(verify_input).logits
        
        # Extract target distributions at each position
        # Positions: len(generated)-1 to len(generated)+gamma-1
        start_pos = len(generated) - 1
        target_probs = []
        for i in range(gamma + 1):
            p = torch.softmax(
                target_logits[:, start_pos + i, :] / temperature, dim=-1
            )
            target_probs.append(p[0].cpu().numpy())
        
        # ---- Stage 3: accept / reject ----
        n_accepted = 0
        for i in range(gamma):
            ti = draft_tokens[i]
            p_target = target_probs[i][ti]
            p_draft = draft_probs[i][ti]
            
            r = np.random.uniform()
            if r < min(1.0, p_target / p_draft):
                generated.append(ti)
                n_accepted += 1
            else:
                # Sample from the residual distribution
                residual = np.maximum(0, target_probs[i] - draft_probs[i])
                residual /= residual.sum()
                new_token = np.random.choice(len(residual), p=residual)
                generated.append(new_token)
                break
        else:
            # All gamma tokens accepted, draw a bonus token
            bonus = np.random.choice(
                len(target_probs[gamma]), p=target_probs[gamma]
            )
            generated.append(bonus)
    
    return generated
```

### Configuring Speculative Decoding in vLLM

```python
from vllm import LLM, SamplingParams

# ---- Option 1: standalone draft model ----
llm = LLM(
    model="meta-llama/Llama-3.3-70B-Instruct",
    speculative_model="meta-llama/Llama-3.2-1B-Instruct",
    num_speculative_tokens=5,
    tensor_parallel_size=4,
    # The draft model can also use tensor parallel
    speculative_model_tensor_parallel_size=1,
)

# ---- Option 2: EAGLE draft ----
llm = LLM(
    model="meta-llama/Llama-3.3-70B-Instruct",
    speculative_model="yuhuili/EAGLE3-LLaMA3.3-70B-Instruct",
    speculative_method="eagle",
    num_speculative_tokens=5,
    tensor_parallel_size=4,
)

# ---- Option 3: N-gram matching (no extra model) ----
llm = LLM(
    model="meta-llama/Llama-3.3-70B-Instruct",
    speculative_model="[ngram]",
    num_speculative_tokens=5,
    ngram_prompt_lookup_max=4,
    ngram_prompt_lookup_min=2,
)

# Standard usage
params = SamplingParams(temperature=0.7, max_tokens=512)
outputs = llm.generate(["Explain quantum computing"], params)
```

### Monitoring Acceptance Rate

```python
# vLLM's built-in metrics
# Prometheus endpoint: /metrics
# Key metrics:
#   vllm:spec_decode_draft_acceptance_rate  - draft token acceptance rate
#   vllm:spec_decode_efficiency             - mean tokens produced per step
#   vllm:num_spec_tokens                    - draft tokens per step

# SGLang monitoring
# Launch with --enable-metrics
# Metrics:
#   sglang:spec_accept_length_mean
#   sglang:spec_accept_length_histogram

# Practical guidance:
# - acceptance < 0.5: switch draft model or reduce gamma
# - acceptance 0.5-0.7: normal range, may still optimize
# - acceptance > 0.8: increase gamma for more speedup
# - Track per-prompt-type variations
```


## Performance Analysis

### Speedup Formula

The actual speedup depends on several factors:

```
                    E[accepted_tokens_per_step]
Speedup ~= --------------------------------------------
            1 + (draft_cost / target_verify_cost)
```

where:
- `E[accepted_tokens_per_step]` = (1 - alpha^(gamma+1)) / (1 - alpha)
- `draft_cost` = time to draft gamma tokens
- `target_verify_cost` = time for one target verification

**Ideal case** (negligible draft cost): speedup ~= E[accepted_tokens]

**In practice**: drafting typically costs 5-20% of the target verification.

### Benchmarks Across Configurations

| Config | Model | Draft method | Concurrency | Speedup |
|------|------|----------|------|------|
| 4xA100 | Llama-3.3-70B | EAGLE-3 | 1 | 3.2x |
| 4xA100 | Llama-3.3-70B | EAGLE-3 | 4 | 2.5x |
| 4xA100 | Llama-3.3-70B | EAGLE-3 | 16 | 1.4x |
| 4xA100 | Llama-3.3-70B | Standalone 8B | 1 | 2.1x |
| 1xA100 | Llama-3-8B | EAGLE-2 | 1 | 2.8x |
| 1xA100 | Llama-3-8B | Medusa | 1 | 2.3x |
| 1xH100 | Llama-3.3-70B | EAGLE-3 + FP8 | 1 | 3.8x |

### When Speculative Decoding Helps vs. Hurts

**Helps**:

| Condition | Why |
|------|------|
| Low concurrency (batch <=10) | Decode is most memory-bandwidth bound |
| Large models (>=13B) | Bigger models = slower forward pass = more savings |
| Predictable output (code / formatting / translation) | High acceptance rate |
| Latency-sensitive applications | Directly reduces single-request latency |

**Doesn't help (or slows things down)**:

| Condition | Why |
|------|------|
| High concurrency (batch 32+) | Decode becomes compute-bound; verification is no longer "free" |
| Prefill-dominated workloads | Spec decoding only accelerates decode |
| Very short generations (<50 tokens) | Drafting overhead dominates |
| Creative / open-ended generation | Acceptance is low, drafts often rejected |
| Mismatched draft model | Low acceptance can make things slower |

### Interaction with Batching

Combining speculative decoding with [[continuous-batching]] requires care:

- Different requests accept different numbers of tokens, so sequence lengths diverge
- During verification, effective lengths inside the batch vary
- Production systems ([[vllm]], [[sglang]]) use padding or bucketing to manage this irregularity
- At high concurrency, the bookkeeping cost erodes the speedup


## Limitations

1. **Draft quality is everything**: acceptance below 0.5 gives little to no speedup, and may even be a regression. The draft must be distributionally close to the target on the target task.

2. **Memory overhead**: standalone-draft setups load two models. A 70B target + 8B draft needs ~16GB extra (FP16). EAGLE/Medusa heads are much smaller (<1GB).

3. **Diminishing returns at high concurrency**: as batch size grows, decode shifts from memory-bound to compute-bound and verifying K tokens is no longer free. This is the main barrier in high-throughput production.

4. **No help for prefill**: speculative decoding only helps decode. Long-prompt workloads need other optimizations like [[prefill-decode-disaggregation]].

5. **Implementation complexity**: tree attention masks, KV cache bookkeeping, and integration with continuous batching all add system complexity.

6. **Temperature dependence**: higher temperature (more random) sampling lowers acceptance rate because the target distribution is harder to predict. Greedy decoding (temperature=0) gives the highest acceptance.

7. **Out-of-domain drafts**: if the draft model isn't trained on the target task's data (or the target has been specifically fine-tuned), acceptance drops. Custom EAGLE/Medusa heads usually need to be trained per target model.


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

- [[vllm]] -- supports multiple speculative decoding strategies
- [[sglang]] -- deeply integrated with EAGLE-3
- [[continuous-batching]] -- scheduler interactions with speculative decoding
- [[kv-cache-optimization]] -- KV cache handling for the draft model
- [[quantization]] -- quantization composes with speculative decoding
- [[prefill-decode-disaggregation]] -- spec decoding only helps decode; prefill needs other optimizations
- [[tensorrt-llm]] -- supports EAGLE-3 and N-gram speculative decoding
