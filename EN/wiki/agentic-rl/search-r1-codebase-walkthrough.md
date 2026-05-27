---
title: "Search-R1 codebase walkthrough — agentic RL from the ground up"
category: agentic-rl
tags: [search-r1, code-walkthrough, agentic-rl, tutorial, verl, ppo-implementation, grpo-implementation, retrieved-token-masking]
created: 2026-05-26
updated: 2026-05-26
status: mature
code: https://github.com/PeterGriffinJin/Search-R1
---

# Search-R1 codebase walkthrough — agentic RL from the ground up

> [!tip] This page accompanies [[search-r1]]
> Read the paper review first for **what** the work is. This page covers **how** it's implemented at the code level — file by file, function by function, with the actual data flow through one training step.

> [!abstract]+ What this page covers
> The Search-R1 repository at [PeterGriffinJin/Search-R1](https://github.com/PeterGriffinJin/Search-R1) is the cleanest reference implementation of agentic RL available. Its total size: ~33,000 lines of Python, but the genuinely Search-R1-specific portion is only ~600 lines (`search_r1/`); the rest is a fork of [veRL](https://github.com/volcengine/verl). This walkthrough takes you through:
>
> 1. **The 600 lines** that are Search-R1's direct contribution — the multi-turn rollout loop, the retrieval server, the entry-point scripts
> 2. **The ~5,000 lines** of veRL PPO/GRPO machinery that Search-R1 sits on — the trainer loop, the actor/critic update, the GAE math, the vLLM wrapper
> 3. **The end-to-end data flow** through one PPO training step, with the full call stack
>
> By the end you'll understand not just Search-R1 but the architectural pattern that ProRL Agent, Polar, NeMo Gym, and the entire 2025-26 agentic-RL field follows.

## Repo layout (rough overview)

```
Search-R1/
├── train_ppo.sh                ← entry #1: training
├── train_grpo.sh               ← entry #2: training (GRPO variant)
├── retrieval_launch.sh         ← entry #3: launch search server
├── infer.py                    ← entry #4: inference
│
├── search_r1/                  ← THE Search-R1-specific code (~600 lines)
│   ├── llm_agent/
│   │   ├── generation.py       ★ THE core file: multi-turn rollout (469)
│   │   └── tensor_helper.py      utility: padding/mask (74)
│   └── search/
│       ├── retrieval_server.py    FastAPI E5+FAISS server (392)
│       ├── google_search_server.py   real web search variant (202)
│       ├── rerank_server.py / retrieval_rerank_server.py
│       ├── serp_search_server.py
│       └── index_builder.py    FAISS index construction (349)
│
├── scripts/
│   └── data_process/
│       └── nq_search.py        NQ → parquet with template
│
└── verl/                       ← forked veRL framework (~32,000 lines)
    ├── trainer/main_ppo.py     entry point (202)
    ├── trainer/ppo/
    │   ├── ray_trainer.py      ★ main fit() loop (867)
    │   └── core_algos.py       ★ PPO/GRPO math: GAE, clip, KL (274)
    ├── workers/
    │   ├── actor/dp_actor.py   ★ actor update (290)
    │   ├── critic/dp_critic.py ★ critic update (204)
    │   ├── rollout/vllm_rollout/vllm_rollout.py  vLLM wrapper (226)
    │   └── fsdp_workers.py     orchestration (1054)
    ├── utils/
    │   ├── reward_score/qa_em.py   outcome reward (139)
    │   └── dataset/rl_dataset.py   dataloader (156)
    └── third_party/vllm/       forked vLLM patches (~4000 lines)
```

**Critical observation**: Search-R1 itself is **just a veRL plugin**. It doesn't modify the RL algorithm, doesn't touch PPO/GRPO core. It only inserts code in two places:

1. **`generation.py`** — replaces default `rollout()` with multi-turn search-interleaved generation
2. **`ray_trainer.py:_create_loss_mask`** — converts the `info_mask` `generation.py` returns into `loss_mask`, hands it to standard PPO loss

That's it. Once you internalize this pattern, you'll see it everywhere in agentic RL: **modify rollout, add a mask, leave the RL algorithm untouched**.

## Recommended reading order

The optimal path through the codebase for a newcomer:

1. **`train_ppo.sh`** — hyperparameters, identify what the trainer manages vs what Search-R1 adds
2. **`scripts/data_process/nq_search.py`** — prompt template, ground-truth format
3. **`retrieval_launch.sh` + `retrieval_server.py` FastAPI portion** — environment interface
4. **`generation.py:run_llm_loop`** — high-level rollout loop
5. **`generation.py:execute_predictions`** — environment step
6. **`generation.py:_info_masked_concatenate_with_padding`** ★ — the heart of loss-masking logic
7. **`generation.py:_compose_final_output`** — exposing `info_mask` to the trainer
8. **`ray_trainer.py:_create_loss_mask`** — info_mask → loss_mask
9. **`dp_actor.py:update_policy`** — where loss_mask is actually applied
10. **`qa_em.py`** — reward
11. (Optional deeper passes) `core_algos.py`, `vllm_rollout.py`, `fsdp_workers.py`

This order takes about 4-6 hours for a careful first read.

---

# Part A: Search-R1's own code (~600 lines)

## 1. Training entry: `train_ppo.sh`

```bash
export DATA_DIR='data/nq_search'
export BASE_MODEL='Qwen/Qwen2.5-3B'

python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train.parquet \
    data.max_prompt_length=4096 \                       # total rollout cap
    data.max_response_length=500 \                      # per-turn LLM gen cap
    data.max_start_length=2048 \                        # original prompt cap
    data.max_obs_length=500 \                           # retrieved obs cap
    algorithm.adv_estimator=gae \                       # PPO uses GAE
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \             # small LR
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.actor.state_masking=true \        # ★ enable info-mask
    algorithm.kl_ctrl.kl_coef=0.001 \                   # KL coef
    trainer.n_gpus_per_node=8 \
    trainer.total_training_steps=1005 \
    max_turns=2 \                                       # ★ max search calls per rollout
    retriever.url="http://127.0.0.1:8000/retrieve" \    # search server
    retriever.topk=3
```

Key hyperparameters to know:

| Parameter | Meaning | Paper's value |
| --------- | ------- | ------------- |
| `max_turns` | Max search calls per rollout | 2 |
| `max_obs_length` | Per-search injected token cap | 500 |
| `state_masking=true` | **Enable retrieved-token loss masking** | true |
| `kl_coef=0.001` | KL-to-reference coefficient | 0.001 |
| `topk=3` | Top-k passages per search | 3 |
| `lr=1e-6` | Actor LR | 1e-6 |
| `temperature=1.0` | Sampling temperature | 1.0 |

Note `max_prompt_length=4096` relates to `max_start_length + max_response_length × (max_turns - 1) + max_obs_length × max_turns` — total budget for the whole rollout sequence.

## 2. Data preparation: `scripts/data_process/nq_search.py`

The critical part is the prompt template:

```python
def make_prefix(dp, template_type):
    question = dp['question']
    if template_type == 'base':
        prefix = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""
    return prefix
```

This is the **R1-Zero-style template** — entirely prompted, no SFT warm-up, no in-context examples. The model learns to obey the token protocol via RL signal alone.

Output (saved as parquet):
```python
{
    "prompt": [{"role": "user", "content": prefix}],
    "data_source": "nq",
    "reward_model": {"style": "rule", "ground_truth": {"target": ["Albert Einstein", ...]}},
}
```

`ground_truth.target` is a list of acceptable answers (NQ has multiple valid answers per question); EM check passes if the prediction matches any of them.

## 3. Retrieval server: `search_r1/search/retrieval_server.py`

This is the **environment implementation**. A separate-process FastAPI server listening on port 8000:

```python
class DenseRetriever(BaseRetriever):
    def __init__(self, config):
        self.index = faiss.read_index(self.index_path)        # FAISS index
        if config.faiss_gpu:
            self.index = faiss.index_cpu_to_all_gpus(...)     # multi-GPU FAISS
        self.corpus = load_corpus(self.corpus_path)            # wiki-18.jsonl
        self.encoder = Encoder(model_path=...)                 # E5 encoder
    
    def _batch_search(self, query_list, num):
        # 1. encode queries to dense vectors
        batch_emb = self.encoder.encode(query_list)
        # 2. FAISS top-k search
        batch_scores, batch_idxs = self.index.search(batch_emb, k=num)
        # 3. lookup actual document text
        results = load_docs(self.corpus, flat_idxs)
        return results

@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    results, scores = retriever.batch_search(
        query_list=request.queries,
        num=request.topk,
        return_score=request.return_scores
    )
    return {"result": resp}
```

**Key design choices**:

1. **Separate process** — retrieval and training are decoupled, communicating via HTTP. Same pattern as [[prorl-agent|ProRL Agent]]'s "rollout as a service," in simpler form.
2. **E5 + FAISS** — queries encoded to 768-dim vectors by [intfloat/e5-base-v2](https://huggingface.co/intfloat/e5-base-v2), FAISS ANN over Wikipedia 2018 dump (~21M passages).
3. **GPU FAISS** — optional `--faiss_gpu` puts FAISS index on GPU, retrieval is ~ms.
4. **Batched interface** — rollout-time may trigger multiple `search` calls in one batch, server accepts a list.

**Real web search variant**: `google_search_server.py` implements the same interface but calls SerpAPI Google. Production-realistic but Wikipedia is cheaper / reproducible, so paper main results use Wikipedia.

```bash
# Launch
bash retrieval_launch.sh
# python search_r1/search/retrieval_server.py \
#     --index_path /path/to/e5_Flat.index \
#     --corpus_path /path/to/wiki-18.jsonl \
#     --topk 3 \
#     --retriever_name e5 \
#     --retriever_model intfloat/e5-base-v2 \
#     --faiss_gpu
```

Then `train_ppo.sh` connects via `retriever.url=http://127.0.0.1:8000/retrieve`.

## 4. THE core file: `search_r1/llm_agent/generation.py` (469 lines)

This is the soul of the repo. I'll cover it in order of importance.

### 4.1 `GenerationConfig`

```python
@dataclass
class GenerationConfig:
    max_turns: int                # multi-turn cap
    max_start_length: int         # original prompt length
    max_prompt_length: int        # total concatenated length
    max_response_length: int      # per-turn LLM gen length
    max_obs_length: int           # per-search injected length
    num_gpus: int
    no_think_rl: bool = False     # ablation
    search_url: str = None        # retrieval server URL
    topk: int = 3                 # per-search top-k
```

All turn-related hyperparameters in one place.

### 4.2 The main loop: `run_llm_loop` (high-level view)

```python
def run_llm_loop(self, gen_batch, initial_input_ids):
    # Left (unchanged) = original prompt
    original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
    # Right (grows) = model generation + injected observations
    original_right_side = {
        'responses': initial_input_ids[:, []],            # empty, will append
        'responses_with_info_mask': initial_input_ids[:, []]
    }
    
    active_mask = torch.ones(B, dtype=torch.bool)        # which trajectories still active
    turns_stats = torch.ones(B, dtype=torch.int)
    valid_search_stats = torch.zeros(B, dtype=torch.int) # search calls per trajectory
    rollings = gen_batch                                  # current input to LLM
    
    # ★ MAIN LOOP ★
    for step in range(self.config.max_turns):
        if not active_mask.sum(): break
        
        # 1) only take alive batch elements
        rollings_active = DataProto.from_dict({k: v[active_mask] for k, v in rollings.batch.items()})
        # 2) vLLM generate next segment
        gen_output = self._generate_with_gpu_padding(rollings_active)
        # 3) truncate at </search> or </answer>
        responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
        # 4) ★ environment step (call search) ★
        next_obs, dones, valid_action, is_search = self.execute_predictions(...)
        # 5) update active mask
        active_mask = active_mask * curr_active_mask
        # 6) append (response, observation) to rollings and right_side
        rollings = self._update_rolling_state(rollings, responses_ids, next_obs_ids)
        original_right_side = self._update_right_side(original_right_side, responses_ids, next_obs_ids)
    
    # Final turn: alive trajectories produce final answer (no search)
    if active_mask.sum():
        gen_output = self._generate_with_gpu_padding(rollings_active)
        # ... same as above but do_search=False
        original_right_side = self._update_right_side(original_right_side, responses_ids)
    
    return self._compose_final_output(original_left_side, original_right_side, meta_info)
```

**Key concepts**:

- **`left_side`**: original prompt, unchanged.
- **`right_side`**: model generation + injected observations, grows in the loop.
- **`rollings`**: current input to LLM (= left + right_so_far).
- **`active_mask`**: which trajectories still active — trajectories that emitted `<answer>` are done; don't continue them.
- **`responses_with_info_mask`**: same length as `responses` but **at `<information>` block positions, filled with `pad_id`**. This is the source of `info_mask`.

**Why active_mask?** Different trajectories in a batch are heterogeneous — some emit `<answer>` after 1 turn, some need 2 turns. Already-done ones shouldn't forward through the LLM again — wastes compute, adds noise.

**Why left/right split?** PPO loss only computes on response tokens (prompt doesn't get gradients). Splitting left/right makes "which tokens are response" precise. `_compose_final_output` re-concatenates at the end.

### 4.3 `_postprocess_responses` — early stop at stop tokens

```python
def _postprocess_responses(self, responses):
    responses_str = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
    responses_str = [resp.split('</search>')[0] + '</search>'   # cut at </search>
             if '</search>' in resp 
             else resp.split('</answer>')[0] + '</answer>'      # or at </answer>
             if '</answer>' in resp 
             else resp                                          # else keep all
             for resp in responses_str]
    responses = self._batch_tokenize(responses_str)
    return responses, responses_str
```

Model generates up to `max_response_length=500` tokens, but **logically stops at `</search>` or `</answer>`** — content after that is discarded. Reason:

- If `</search>` was emitted, the model wants to query — anything after is meaningless before retrieval.
- If `</answer>` was emitted, the model is done — anything after is unreliable continuation.

**This is the second agentic-RL engineering detail**: you must **detect stop signals in the token stream and stop immediately**. vLLM supports `stop=["</search>", "</answer>"]` directly, but Search-R1 uses post-hoc string splits (more general / tokenizer-agnostic).

### 4.4 `execute_predictions` — the "env step"

This is the Algorithm 1 from the paper, in code:

```python
def execute_predictions(self, predictions, pad_token, active_mask, do_search=True):
    """
    NOTE: the function is the actual `step` function in the environment
    """
    cur_actions, contents = self.postprocess_predictions(predictions)
    # cur_actions[i] ∈ {'search', 'answer', None}
    # contents[i] = content inside <search>...</search> or <answer>...</answer>
    
    next_obs, dones, valid_action, is_search = [], [], [], []
    
    # Collect all 'search' actions, batch the HTTP request
    search_queries = [content for action, content in zip(cur_actions, contents) if action == 'search']
    if do_search:
        search_results = self.batch_search(search_queries)   # ← one HTTP POST
    else:
        search_results = [''] * sum(1 for a in cur_actions if a == 'search')

    for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
        if not active:
            next_obs.append(''); dones.append(1); valid_action.append(0); is_search.append(0)
        else:
            if action == 'answer':
                next_obs.append('')                           # no obs after answer
                dones.append(1)                               # ← trajectory ends
                valid_action.append(1); is_search.append(0)
            elif action == 'search':
                next_obs.append(f'\n\n<information>{search_results.pop(0).strip()}</information>\n\n')
                dones.append(0)                               # ← continue
                valid_action.append(1); is_search.append(1)
            else:   # malformed output
                next_obs.append('\nMy previous action is invalid. ...')
                dones.append(0)
                valid_action.append(0); is_search.append(0)
    
    return next_obs, dones, valid_action, is_search
```

**Important details**:

1. **Action parsing via regex** (`postprocess_predictions`):
   ```python
   pattern = r'<(search|answer)>(.*?)</\1>'
   match = re.search(pattern, prediction, re.DOTALL)
   ```
   Only two actions allowed; content is between tags. Regex failure → action = None.

2. **Error handling**:
   ```python
   next_obs.append('My previous action is invalid. ...')
   dones.append(0)  # don't exit, let it retry
   ```
   Inject a "reminder" so the model retries. Corresponds to Algorithm 1 line 19.

3. **Batched search**: all `search` actions in one HTTP request. Critical for efficiency — a batch of 256 trajectories with 100 searches → one request vs 100. Orders of magnitude difference.

4. **`do_search=False` use case**: last turn (force-final) calls with `do_search=False` because even if the model wants to search, there's no more turns to digest results — so skip the call.

### 4.5 ★ THE function: `_info_masked_concatenate_with_padding`

This is the single most important function in the repo. **Retrieved-token loss masking at the token level**:

```python
def _info_masked_concatenate_with_padding(self, prompt, prompt_with_mask, response, info=None, pad_to_left=True):
    """
    Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to 
    cover the information block if it exists.
    """
    pad_id = self.tokenizer.pad_token_id
    tensors = [prompt, response]                      # real concatenation
    tensors_with_mask = [prompt_with_mask, response]  # masked version
    if info is not None:                              # if there's retrieved obs
        tensors.append(info)                          # real version: append info
        info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device)
        tensors_with_mask.append(info_mask)           # ★ masked version: pad_id at info positions ★
    
    concatenated = torch.cat(tensors, dim=1)                       # real tokens: prompt + response + info
    concatenated_with_info = torch.cat(tensors_with_mask, dim=1)   # masked: prompt + response + [pad pad pad...]
    
    # Pad alignment
    mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
    sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
    padded_tensor = concatenated.gather(1, sorted_indices)
    padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)
    
    return padded_tensor, padded_tensor_with_info
```

**The magic of this code**:

It **simultaneously maintains two tensors**:
- `concatenated`: real input sequence including prompt + response + retrieved info (for forward pass)
- `concatenated_with_info`: **same positions but with retrieved info replaced by pad_token_id** (for masking)

The two tensors **have identical token-length and identical alignment**, but at retrieved-info positions:
- `concatenated[..., info_position] = actual retrieved token IDs`
- `concatenated_with_info[..., info_position] = pad_id, pad_id, pad_id, ...`

Later in `_compose_final_output`, computing `concatenated_with_info != pad_id` gives exactly the right attention mask that **zeroes out info positions**. This is `info_mask`.

**Why this design instead of a separate binary mask?** Because different trajectories in a batch have info at different positions (depending on when they searched). Token-level pad_id replacement and then standard padding alignment handles this in one pass.

When reading agentic-RL code, **find this "synchronously maintain two tensors" pattern** — you've found the loss-masking logic. Polar's `prefix_merging` uses the same design (real token array, loss_mask array, aligned).

### 4.6 `_update_right_side` — appending one turn

```python
def _update_right_side(self, right_side, cur_responses, next_obs_ids=None):
    if next_obs_ids != None:
        responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                right_side['responses'],
                right_side['responses_with_info_mask'],
                cur_responses,
                next_obs_ids,
                pad_to_left=False                  # right-append
            )
    else:
        # Last turn, no obs
        responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                right_side['responses'],
                right_side['responses_with_info_mask'],
                cur_responses,
                pad_to_left=False
            )
    effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
    max_len = min(self.config.max_prompt_length, effective_len)
    return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len]}
```

Called once per turn, appends `(new_response, new_obs)` to `right_side`, **both versions grow synchronously**.

### 4.7 `_compose_final_output` — package for trainer

```python
def _compose_final_output(self, left_side, right_side, meta_info):
    final_output = right_side.copy()
    final_output['prompts'] = left_side['input_ids']
    final_output['input_ids'] = torch.cat([left_side['input_ids'], right_side['responses']], dim=1)
    
    # Normal attention mask: 1 everywhere except pad
    final_output['attention_mask'] = torch.cat([
        self.tensor_fn.create_attention_mask(left_side['input_ids']),
        self.tensor_fn.create_attention_mask(final_output['responses'])
    ], dim=1)
    
    # ★ info_mask: same shape, but right-side info positions are 0
    final_output['info_mask'] = torch.cat([
        self.tensor_fn.create_attention_mask(left_side['input_ids']),
        self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
    ], dim=1)
    
    final_output['position_ids'] = self.tensor_fn.create_position_ids(final_output['attention_mask'])
    return DataProto.from_dict(final_output)
```

Batch given to trainer contains:

- `input_ids`: full rollout (prompt + response_with_real_info)
- `attention_mask`: standard mask (pad=0, else=1) — for vLLM forward
- **`info_mask`**: same shape, **all retrieved info positions are 0** — for loss masking
- `position_ids`: derived from attention_mask

`attention_mask` is for model forward ("which tokens to attend"); `info_mask` is for loss ("which tokens count toward gradient"). **Decoupling these two is the key**.

### 4.8 `_generate_with_gpu_padding` — engineering detail

```python
def _generate_with_gpu_padding(self, active_batch):
    """if active_batch size is not divisible by num_gpus, pad with first sequence then remove"""
    num_gpus = self.config.num_gpus
    batch_size = active_batch.batch['input_ids'].shape[0]
    remainder = batch_size % num_gpus
    if remainder == 0:
        return self.actor_rollout_wg.generate_sequences(active_batch)
    
    # Pad to be divisible
    padding_size = num_gpus - remainder
    padded_batch = {}
    for k, v in active_batch.batch.items():
        pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
        padded_batch[k] = torch.cat([v, pad_sequence], dim=0)
    
    padded_output = self.actor_rollout_wg.generate_sequences(DataProto.from_dict(padded_batch))
    trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
    padded_output.batch = trimmed_batch
    return padded_output
```

**This engineering detail is agentic-RL-specific**:

Normal RL training has fixed batch size, all trajectories forward in sync per step. But in agentic RL **active_mask shrinks the batch** (early-done trajectories stop forwarding), and remaining trajectories may **not be divisible by num_gpus**. vLLM multi-GPU requires batch divisible by num_gpus (for even sharding), so this manually **pads to divisible, generates, then trims the padding**.

Newcomers to multi-turn RL trip on this constantly.

## 5. `tensor_helper.py` — small utility (74 lines)

```python
class TensorHelper:
    def cut_to_effective_len(self, tensor_dict, keys, cut_left=True):
        """Cut tensors to their effective length based on attention mask."""
        effective_len = tensor_dict['attention_mask'].sum(dim=1).max()
        result = tensor_dict.copy()
        for key in keys:
            if cut_left:
                result[key] = tensor_dict[key][:, -effective_len:]
            else:
                result[key] = tensor_dict[key][:, :effective_len]
        return result

    def create_attention_mask(self, input_ids):
        return torch.where(input_ids != self.config.pad_token_id, 1, 0)

    def create_position_ids(self, attention_mask):
        return (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

    def concatenate_with_padding(self, tensors, pad_to_left=True):
        concatenated = torch.cat(tensors, dim=1)
        padded_tensor, _ = self.convert_pad_structure(concatenated, pad_to_left)
        return padded_tensor

    def _example_level_pad(self, responses, responses_str, active_mask):
        """Pad responses for non-active examples with pad tokens."""
        # ...
```

Standard padding/mask utilities. The interesting one is `_example_level_pad` — when generating only for active examples but needing to keep the full batch shape (for `active_mask` to work cleanly), this pads inactive positions back in.

## 6. `infer.py` — reference inference (130 lines)

After training, use HF Transformers + `StoppingCriteria` for inference:

```python
class StopOnSequence(transformers.StoppingCriteria):
    def __init__(self, target_sequences, tokenizer):
        self.target_ids = [tokenizer.encode(s, add_special_tokens=False) for s in target_sequences]
        self.target_lengths = [len(t) for t in self.target_ids]
    
    def __call__(self, input_ids, scores, **kwargs):
        for i, target in enumerate(targets):
            if torch.equal(input_ids[0, -self.target_lengths[i]:], target):
                return True
        return False

# ★ multiple variants handle tokenizer fragmentation
target_sequences = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]
stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(target_sequences, tokenizer)])

while True:
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    outputs = model.generate(
        input_ids,
        max_new_tokens=1024,
        stopping_criteria=stopping_criteria,
        do_sample=True, temperature=0.7
    )
    
    if outputs[0][-1].item() in curr_eos:
        break    # EOS, done
    
    tmp_query = get_query(tokenizer.decode(outputs[0], skip_special_tokens=True))
    if tmp_query:
        search_results = search(tmp_query)
    
    prompt += f'\n\n{output_text}<information>{search_results}</information>\n\n'
```

**Engineering notes**:

1. **6 variants of `target_sequences`** — different tokenizer encodings: `</search>` with or without leading space, with or without trailing newlines. Tokenizers fragment differently per context — must enumerate all "real token sequences."
2. **`do_sample=True, temperature=0.7`** — inference uses temperature sampling, not greedy. Paper's rollout temperature=1.0; inference uses lower (0.7) for stability.
3. **Shape parallels the training rollout** — every turn stopped on `</search>` → extract query → call search → append `<information>...</information>` → continue. **Re-implements `generation.py`'s logic using HF generate**.

This file is the **easiest entry point** for beginners — only 130 lines, no veRL machinery.

---

# Part B: The veRL machinery (~5,000 lines)

## 7. `verl/trainer/main_ppo.py` — program entry (202 lines)

```python
@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})
    ray.get(main_task.remote(config))

@ray.remote
def main_task(config):
    # 1. Load base model + tokenizer
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)
    tokenizer = hf_tokenizer(local_path)
    
    # 2. Pick worker implementation: FSDP or Megatron
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        ray_worker_group_cls = RayWorkerGroup

    # 3. Three roles
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),   # ★ actor + rollout combined
        Role.Critic:       ray.remote(CriticWorker),
        Role.RefPolicy:    ray.remote(ActorRolloutRefWorker),   # ★ same class, forward-only
    }
    
    # 4. Allocate all 8 GPUs to global_pool
    resource_pool_spec = {global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes}
    mapping = {Role.ActorRollout: global_pool_id, Role.Critic: global_pool_id, Role.RefPolicy: global_pool_id}
    
    # 5. Reward function (rule-based, no learning)
    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0)
    
    # 6. Build trainer, start training
    trainer = RayPPOTrainer(config=config, tokenizer=tokenizer, ...)
    trainer.init_workers()
    trainer.fit()
```

**Key design**:

1. **`@ray.remote`**: entire training job as a Ray remote function. Ray is veRL's distributed orchestration.
2. **Three roles share 8 GPUs** (`global_pool_id`): actor/rollout/ref all on the same GPUs, but FSDP offload staggers memory peaks. veRL's "hybrid engine": actor params in GPU during training, swap to vLLM view during rollout.
3. **`Role.ActorRollout` in fsdp_workers.py instantiates both actor + vLLM rollout** — not two separate processes but one worker process with two views, sharing weights.

### `RewardManager.__call__` — scoring each trajectory

```python
class RewardManager():
    def __call__(self, data: DataProto):
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch['prompts']
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            
            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            
            # call qa_em.compute_score_em
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)
            score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, ...)
            
            # ★ reward only at last valid token position ★
            reward_tensor[i, valid_response_length - 1] = score
        return reward_tensor
```

**This is what "sparse outcome reward" looks like in implementation**:

`reward_tensor` is shape `(batch_size, response_length)`, **initialized to zeros**, **with a single scalar at the last valid token of each trajectory** (0 or 1). Every other position is zero.

This is the literal meaning of "outcome-only sparse reward" — **only the endpoint has signal, everything else is zero**. GAE will propagate this final reward backward into per-token advantages via the value function (credit assignment).

## 8. `verl/utils/dataset/rl_dataset.py` — dataloader (156 lines)

```python
class RLHFDataset(Dataset):
    def __getitem__(self, item):
        row_dict = self.dataframe.iloc[item].to_dict()
        chat = row_dict.pop(self.prompt_key)  # list of {role, content}
        
        if self.tokenizer.chat_template:
            prompt_with_chat_template = self.tokenizer.apply_chat_template(
                chat, add_generation_prompt=True, tokenize=False)
        else:
            prompt_with_chat_template = chat[0]['content']
        
        # tokenize + left padding
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt_with_chat_template,
            tokenizer=self.tokenizer,
            max_length=self.max_prompt_length,    # 4096
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,                         # ★ left padding ★
            truncation=self.truncation
        )
        
        position_ids = compute_position_id_with_mask(attention_mask)
        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]
        
        # for GRPO grouping
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index
        return row_dict
```

**Two things to notice**:

1. **Left padding (`left_pad=True`)**:
   - Standard SFT uses right padding (token order = generation order).
   - **Generation (rollout) requires left padding** because vLLM autoregressive generation only sees the rightmost non-pad position.
   - The `pad_to_left` parameter in `generation.py:_info_masked_concatenate_with_padding` exists because of this — downstream processing must preserve left padding.

2. **`row_dict["index"]`**: per-sample ID. **GRPO uses this to group N rollouts of the same prompt**. GRPO's baseline = "mean reward of N rollouts of the same prompt," needs group ID to compute. PPO doesn't.

`collate_fn` packages multiple samples into a batch, splitting tensor / non-tensor:
```python
def collate_fn(data_list):
    tensors = {}     # input_ids, attention_mask, position_ids
    non_tensors = {} # data_source, reward_model, index, extra_info
    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors.setdefault(key, []).append(val)
            else:
                non_tensors.setdefault(key, []).append(val)
    for key in tensors: tensors[key] = torch.stack(tensors[key], dim=0)
    for key in non_tensors: non_tensors[key] = np.array(non_tensors[key], dtype=object)
    return {**tensors, **non_tensors}
```

**`DataProto` = TensorDict + NumPy object dict**. This is veRL's core data container, used everywhere. A batch is a DataProto; `batch` is tensor part, `non_tensor_batch` is metadata.

## 9. `ray_trainer.py:fit()` — main training loop (867 lines)

The entire PPO/GRPO main loop. Full flow:

```python
def fit(self):
    for epoch in range(self.config.trainer.total_epochs):
        for batch_dict in self.train_dataloader:
            metrics = {}; timing_raw = {}
            
            batch = DataProto.from_single_dict(batch_dict)
            batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n_agent, interleave=True)
            
            gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
            
            with _timer('step', timing_raw):
                # ════════════════════════════════════════════════════════
                # 1. ROLLOUT — branches here
                # ════════════════════════════════════════════════════════
                if not self.config.do_search:
                    # Plain PPO: single vLLM forward
                    gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)
                else:
                    # ★ Search-R1 path: multi-turn rollout ★
                    first_input_ids = gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone().long()
                    with _timer('gen', timing_raw):
                        final_gen_batch_output = generation_manager.run_llm_loop(
                            gen_batch=gen_batch,
                            initial_input_ids=first_input_ids,
                        )
                    # Recompute old_log_prob (vLLM logprobs may differ slightly from FSDP forward)
                    with torch.no_grad():
                        output = self.actor_rollout_wg.compute_log_prob(final_gen_batch_output)
                        final_gen_batch_output = final_gen_batch_output.union(output)
                    batch.non_tensor_batch['uid'] = batch.non_tensor_batch['index'].copy()
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(final_gen_batch_output)
                
                self._balance_batch(batch, metrics=metrics)
                
                # ════════════════════════════════════════════════════════
                # 2. REFERENCE log-prob (for KL penalty)
                # ════════════════════════════════════════════════════════
                if self.use_reference_policy:
                    with _timer('ref', timing_raw):
                        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                        batch = batch.union(ref_log_prob)
                
                # ════════════════════════════════════════════════════════
                # 3. CRITIC values (PPO only, GRPO skips)
                # ════════════════════════════════════════════════════════
                if self.use_critic:
                    with _timer('values', timing_raw):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)
                
                # ════════════════════════════════════════════════════════
                # 4. REWARD + ADVANTAGE
                # ════════════════════════════════════════════════════════
                with _timer('adv', timing_raw):
                    reward_tensor = self.reward_fn(batch)
                    batch.batch['token_level_scores'] = reward_tensor
                    
                    # KL penalty (PPO style: add -β·KL into reward)
                    if not self.config.actor_rollout_ref.actor.use_kl_loss:
                        batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl, ...)
                    else:
                        # GRPO style: KL as separate loss, reward unchanged
                        batch.batch['token_level_rewards'] = batch.batch['token_level_scores']
                    
                    # Compute advantage
                    batch = compute_advantage(batch,
                        adv_estimator=self.config.algorithm.adv_estimator,   # 'gae' or 'grpo'
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam, ...)
                
                # ════════════════════════════════════════════════════════
                # 5. CRITIC update (PPO only)
                # ════════════════════════════════════════════════════════
                if self.use_critic:
                    with _timer('update_critic', timing_raw):
                        critic_output = self.critic_wg.update_critic(batch)
                
                # ════════════════════════════════════════════════════════
                # 6. ACTOR update — retrieved-token loss masking kicks in here
                # ════════════════════════════════════════════════════════
                if self.config.trainer.critic_warmup <= self.global_steps:
                    with _timer('update_actor', timing_raw):
                        if self.config.do_search and self.config.actor_rollout_ref.actor.state_masking:
                            batch, metrics = self._create_loss_mask(batch, metrics)
                        actor_output = self.actor_rollout_wg.update_actor(batch)
            
            # ════════════════════════════════════════════════════════
            # 7. Validate + save
            # ════════════════════════════════════════════════════════
            if self.val_reward_fn is not None and self.global_steps % self.config.trainer.test_freq == 0:
                val_metrics = self._validate()
            
            self.global_steps += 1
```

**7 stages fully laid out**. Each batch goes through this. This is "one PPO training step" in the paper, but multi-turn rollout makes step 1 complex.

**Easily confused concepts**:

- `token_level_scores` = raw reward output (only nonzero at last valid token)
- `token_level_rewards` = scores minus KL penalty (PPO style) or = scores (GRPO style)
- `advantages` = GAE / GRPO output
- `returns` = `advantages + values` (PPO uses this as critic target)

### `apply_kl_penalty` — PPO-style KL handling

```python
def apply_kl_penalty(data, kl_ctrl, kl_penalty='kl'):
    response_length = data.batch['responses'].size(1)
    token_level_scores = data.batch['token_level_scores']
    # ★ if info_mask exists, KL penalty also respects info_mask
    attention_mask = data.batch['info_mask'] if 'info_mask' in data.batch else data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]
    
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'], kl_penalty=kl_penalty)
        kld = kld * response_mask  # ← mask retrieved + pad tokens
        beta = kl_ctrl.value
    else:
        beta = 0; kld = torch.zeros_like(response_mask, dtype=torch.float32)
    
    # ★ PPO style: add -β·KL to reward
    token_level_rewards = token_level_scores - beta * kld
    
    data.batch['token_level_rewards'] = token_level_rewards
    return data, metrics
```

**Two styles of KL handling** (must distinguish):

- **PPO style (`use_kl_loss=False`)**: add `-β·KL` to reward → KL automatically propagates through GAE to token-level advantages. **No explicit KL loss term**, but affects gradients through reward. `kl_coef=0.001`, `kl_ctrl=fixed`.

- **GRPO style (`use_kl_loss=True`)**: reward unchanged, KL is a separate term in actor loss (in `dp_actor.py:update_policy`), weighted-added to PPO clip loss. `kl_loss_coef=0.001`, `kl_loss_type=low_var_kl`.

DeepSeek-R1 / GRPO paper uses the second; classic PPO uses the first. Search-R1 defaults to PPO style for PPO, switches to GRPO style for GRPO. **Important engineering distinction**, easily confused.

Note the first line `attention_mask = data.batch['info_mask'] if 'info_mask' in data.batch else data.batch['attention_mask']` — **info_mask starts taking effect here**, retrieved tokens don't enter KL.

### `compute_advantage` — GAE or GRPO branch

```python
def compute_advantage(data, adv_estimator, gamma=1.0, lam=1.0):
    if adv_estimator == 'gae':
        values = data.batch['values']
        response_length = data.batch['responses'].size(-1)
        response_mask = data.batch['attention_mask'][:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=token_level_rewards,
            values=values,
            eos_mask=response_mask,
            gamma=gamma, lam=lam)
    elif adv_estimator == 'grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']   # ★ uid as group ID
        response_length = data.batch['responses'].size(-1)
        response_mask = data.batch['attention_mask'][:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards,
            eos_mask=response_mask,
            index=index)
    data.batch['advantages'] = advantages
    data.batch['returns'] = returns
    return data
```

Simple dispatch: GAE goes through `compute_gae_advantage_return` (needs values), GRPO through `compute_grpo_outcome_advantage` (needs group index).

## 10. `core_algos.py` — the RL math (274 lines)

Only 274 lines but contains **every PPO-paper formula as code**. This is what RL infra engineers should be able to recite.

### 10.1 `compute_gae_advantage_return` — GAE in 30 lines

```python
def compute_gae_advantage_return(token_level_rewards, values, eos_mask, gamma, lam):
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]
        
        # ★ reverse iteration ★
        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            # TD error: δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            # GAE: A_t = δ_t + γλ·A_{t+1}
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        
        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, eos_mask)  # ★ normalize ★
    return advantages, returns
```

Core is the GAE recurrence:

$$
A_t^{GAE} = \delta_t + \gamma \lambda \cdot A_{t+1}^{GAE}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

Iterate backward from the last token.

**Points to grok**:

1. **`token_level_rewards`** in Search-R1 is **almost-all-zero** (only last valid token has a value). GAE propagates this terminal reward backward via `value function + discount` to per-token advantages. **This is credit assignment in math**.
2. **`returns = advantages + values`** — this is the critic's training target. So critic learns "what future return should I predict", not "what's the final reward."
3. **`masked_whiten`** = subtract mean, divide by std, over valid token positions. **Critical for PPO stability** — without advantage normalization PPO often goes NaN.

### 10.2 `compute_grpo_outcome_advantage` — GRPO with group-mean baseline

```python
def compute_grpo_outcome_advantage(token_level_rewards, eos_mask, index, epsilon=1e-6):
    response_length = token_level_rewards.shape[-1]
    non_zero_mask = (token_level_rewards != 0)
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)  # one scalar per trajectory
    
    id2score = defaultdict(list)
    id2mean = {}; id2std = {}
    
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])    # group by group ID
        for idx in id2score:
            if len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))   # ★ baseline = group mean
                id2std[idx]  = torch.std(torch.tensor([id2score[idx]]))  # group-normalized std
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        # Tile to each token position + apply mask
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask
    return scores, scores
```

**The entire GRPO math is these 30 lines**. Comparison vs GAE:

| Dimension | GAE (PPO) | GRPO |
| --------- | --------- | ---- |
| Needs value function? | Yes | **No** |
| Per-token advantage? | Yes (different per token) | **No (single scalar per trajectory)** |
| Baseline | $V(s_t)$ | **Mean reward of N rollouts of same prompt** |
| Normalization | Across whole batch | **Per-group std** |
| Multi-turn / long-sequence stability | Good | More variance |

**Note** the `scores.unsqueeze(-1).tile([1, response_length])` line — "GRPO advantage is the same scalar at every token." **Completely different from GAE's token-level advantages**. Also why Search-R1's GRPO collapsed in long rollouts — same advantage at every position across thousands of tokens, less variance control.

### 10.3 `compute_policy_loss` — PPO clipped objective

```python
def compute_policy_loss(old_log_prob, log_prob, advantages, eos_mask, cliprange):
    # ratio = π_new / π_old
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    
    # Standard PPO clipped objective
    pg_losses  = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    
    # Take max (pessimistic bound)
    pg_loss = verl_F.masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
    return pg_loss, pg_clipfrac, ppo_kl
```

**This is page 7 of the [PPO paper](https://arxiv.org/abs/1707.06347)**. 3 lines:

$$
L^{CLIP}(\theta) = \mathbb{E}_t\!\left[\min\!\left(r_t(\theta) A_t,\;\text{clip}(r_t(\theta), 1{-}\epsilon, 1{+}\epsilon) A_t\right)\right]
$$

where $r_t(\theta) = \pi_\theta / \pi_{\text{old}}$. Taking `max(-r·A, -clip(r)·A)` then negating equals taking `min(r·A, clip(r)·A)`.

**`eos_mask` is a misleading name** — actually it's `response_mask` (valid token positions), but **after Search-R1 replaces it with info_mask, this argument now marks "which tokens count toward loss"** — the ultimate landing point of retrieved-token loss masking at the PPO loss level.

### 10.4 `kl_penalty` — four KL estimators

```python
def kl_penalty(logprob, ref_logprob, kl_penalty):
    if kl_penalty == "kl":
        return logprob - ref_logprob   # simple diff, can be negative
    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()
    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()
    if kl_penalty == 'low_var_kl':  # ★ Schulman's unbiased low-variance estimator
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)
```

**`low_var_kl` is GRPO's default** (also DeepSeek-R1's), from [Schulman's blog](http://joschu.net/blog/kl-approx.html):

$$
\mathrm{KL}(p \,\|\, q) \approx \mathbb{E}_q\!\left[\frac{p}{q} - \log\frac{p}{q} - 1\right] = \mathbb{E}_q\!\left[e^{kl} - kl - 1\right]
$$

Advantages over plain `logprob - ref_logprob`:
1. **Unbiased** (in sample limit, converges to true KL)
2. **Always ≥ 0** (mathematically $e^x - x - 1 \ge 0$)
3. **Low variance** (especially when $p \approx q$)

This is why GRPO picks this KL estimator.

## 11. `dp_actor.py:update_policy` — actor training (290 lines)

```python
def update_policy(self, data: DataProto):
    self.actor_module.train()
    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size
    temperature = data.meta_info['temperature']
    
    select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages']
    if self.config.state_masking:
        select_keys.append('loss_mask')
    if self.config.use_kl_loss:
        select_keys.append('ref_log_prob')
    batch = data.select(batch_keys=select_keys).batch
    
    # mini-batch / micro-batch split
    dataloader = batch.split(self.config.ppo_mini_batch_size)
    for batch_idx, data in enumerate(dataloader):
        mini_batch = data
        micro_batches = mini_batch.split(self.config.ppo_micro_batch_size)
        self.actor_optimizer.zero_grad()
        
        for data in micro_batches:
            data = data.cuda()
            responses = data['responses']; response_length = responses.size(1)
            attention_mask = data['attention_mask']
            response_mask = attention_mask[:, -response_length:]
            if self.config.state_masking:
                response_mask = data['loss_mask']  # ★ override with loss_mask ★
            old_log_prob = data['old_log_probs']
            advantages = data['advantages']
            
            # forward
            entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature)
            
            # ★ PPO clip loss
            pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(
                old_log_prob=old_log_prob, log_prob=log_prob, advantages=advantages,
                eos_mask=response_mask, cliprange=self.config.clip_ratio)
            
            # entropy regularization
            entropy_loss = verl_F.masked_mean(entropy, response_mask)
            policy_loss = pg_loss - entropy_loss * self.config.entropy_coeff
            
            # ★ GRPO-style KL loss (if enabled)
            if self.config.use_kl_loss:
                ref_log_prob = data['ref_log_prob']
                kld = core_algos.kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob,
                                            kl_penalty=self.config.kl_loss_type)  # 'low_var_kl'
                kl_loss = masked_mean(kld, response_mask)
                policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
            
            loss = policy_loss / self.gradient_accumulation
            loss.backward()
        
        grad_norm = self._optimizer_step()  # FSDP grad clip + optimizer step
```

**This is where gradients actually flow**. Each micro batch:
1. Forward to get `log_prob` (per-token log probability under current policy) and `entropy`
2. PPO clip loss
3. Entropy regularization
4. Optional KL-to-reference loss (GRPO style)
5. Backward
6. After all micro batches, optimizer step

`response_mask = data['loss_mask']` is the **final landing** of Search-R1's retrieved-token loss masking at the actor side. Masked tokens:
- PPO clip loss's `eos_mask` is this → no PG gradient
- Entropy's `mask` is this → no entropy regularization
- KL loss's `mask` is this → no KL constraint

### `compute_log_prob` — recomputing `old_log_prob`

```python
def compute_log_prob(self, data: DataProto) -> torch.Tensor:
    self.actor_module.eval()
    select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
    batch = data.select(batch_keys=select_keys).batch
    micro_batches = batch.split(micro_batch_size)
    
    log_probs_lst = []
    for micro_batch in micro_batches:
        with torch.no_grad():
            _, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature)
        log_probs_lst.append(log_probs)
    log_probs = torch.concat(log_probs_lst, dim=0)
    return log_probs
```

**Initially looks redundant**: vLLM returned logprobs during rollout, why does the trainer rerun forward?

**Two reasons**:
1. **vLLM's sampling logprob may not be numerically identical to FSDP forward's logprob** (different kernels, batching, fp16/bf16). If we used vLLM's logprob directly as `old_log_prob`, PPO's importance ratio $r_t = \exp(\log\pi_\theta - \log\pi_{\text{old}})$ would deviate from 1 at step 0 and PPO would immediately be unstable.
2. **Search-R1's rollout is multi-turn**, vLLM's input ≠ FSDP forward's input (which contains retrieved info). Must recompute on the **full rollout** to match.

So `actor_rollout_wg.compute_log_prob(final_gen_batch_output)` is **rerunning the complete sequence through FSDP forward** to get trainer-side consistent `old_log_prob`. **Not cheap** — same order as training forward.

## 12. `dp_critic.py` — critic / value function (204 lines)

```python
class DataParallelPPOCritic(BasePPOCritic):
    def _forward_micro_batch(self, micro_batch):
        # Like actor forward, but model has 1-dim value head
        output = self.critic_module(input_ids=input_ids, ...)
        values = output.logits[:, -response_length - 1:-1].squeeze(-1)
        return values
    
    def compute_values(self, data):
        """After rollout, trainer calls this for per-position value estimates"""
        self.critic_module.eval()
        for micro_batch in micro_batches:
            with torch.no_grad():
                values = self._forward_micro_batch(micro_batch)
            values_lst.append(values)
        return values
    
    def update_critic(self, data):
        """Per PPO step, update value head (GRPO skips this)"""
        for data in dataloader:
            for micro in micro_batches:
                vpreds = self._forward_micro_batch(micro)
                vf_loss, vf_clipfrac = core_algos.compute_value_loss(
                    vpreds=vpreds, values=values, returns=returns,
                    eos_mask=eos_mask, cliprange_value=self.config.cliprange_value)
                loss = vf_loss / self.gradient_accumulation
                loss.backward()
            grad_norm = self._optimizer_step()
```

**Critic is a separate model** (same backbone as actor + 1-dim value head), trained via FSDP.
- `compute_values`: per-token value estimates after rollout (GAE uses these)
- `update_critic`: train value head to predict `returns`

**Critic doesn't use info_mask**: it needs value predictions at **every token, including retrieved info positions** (GAE backward pass needs next-token value estimates). Critic uses plain `attention_mask`.

PPO's ~30% memory pressure is from critic — why GRPO (no critic) is memory-friendly.

## 13. `vllm_rollout.py` — rollout engine (226 lines)

```python
class vLLMRollout(BaseRollout):
    def __init__(self, actor_module, config, ...):
        self.inference_engine = LLM(actor_module, ..., 
                                    tensor_parallel_size=tensor_parallel_size,
                                    gpu_memory_utilization=config.gpu_memory_utilization,
                                    max_model_len=config.prompt_length + config.response_length)
        self.inference_engine.offload_model_weights()    # ← default offload to save memory
        
        kwargs = dict(n=1, logprobs=1, max_tokens=config.response_length)
        self.sampling_params = SamplingParams(**kwargs)
    
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        idx = prompts.batch['input_ids']     # (bs, prompt_length)
        
        # ★ Convert each prompt from left-padded to List[int] (strip pad)
        idx_list = []
        for i in range(batch_size):
            idx_list.append(_pre_process_inputs(self.pad_token_id, idx[i]))
        
        # ★ Call vLLM generate
        output = self.inference_engine.generate(
            prompts=None, sampling_params=self.sampling_params,
            prompt_token_ids=idx_list, use_tqdm=False)
        
        response = output[0]; log_probs = output[1]
        
        # Right-pad response to fixed length
        if response.shape[1] < self.config.response_length:
            response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
        
        # Concat: prompt (left-padded) + response (right-padded)
        seq = torch.cat([idx, response], dim=-1)
        # ... build attention_mask and position_ids
        return DataProto(batch=batch)
```

**Key points**:

1. **vLLM as inference engine**: `LLM` is veRL's fork of vLLM v0.5.4, interface matches mainline vLLM
2. **`offload_model_weights()`** — by default offload vLLM's internal weight tensors to CPU, **load back to GPU during rollout**. Key to hybrid engine: actor and vLLM share the same weights, offload staggers usage
3. **Left padding → packed list**: vLLM doesn't accept left-padded tensors, must convert each prompt to `List[int]` (`_pre_process_inputs`), stripping pad
4. **Prompt left-padded, response right-padded**: final sequence is "pad...pad [prompt] [response] pad...pad" (could have pad on both ends)

**This layer is unaware of search**. Search-R1's multi-turn is orchestrated entirely in `generation.py:run_llm_loop`, each turn calls `vLLMRollout.generate_sequences` once, then appends previous response + retrieved info to input for the next call. **vLLM is single-turn**.

### Hybrid engine — FSDP ↔ vLLM weight sharing

`fsdp_workers.py:_build_rollout` and `sharding_manager/fsdp_vllm.py` implement:

```
Training state: actor params on GPU (FSDP sharded) + optimizer state on GPU (or CPU offload)
            │
            │  rollout time:
            ▼
Inference state: merge FSDP shards → reshape to vLLM TP layout → copy to vLLM weight pointers
            │
            │  rollout done:
            ▼
Training state: reshape back to FSDP → continue PPO update
```

Logic in `verl/workers/sharding_manager/fsdp_vllm.py`. **Most complex veRL module**, but invisible to users.

Why not two weight copies? 8B model in fp16 = 16GB. Actor + vLLM = 32GB, plus critic = 48GB. On 80GB H100 that leaves 32GB for activations + KV cache + optimizer state — way too tight. **Hybrid engine sharing weights is mandatory**.

## 14. `fsdp_workers.py` — orchestration (1054 lines)

`fsdp_workers.py` wraps actor/critic/rollout/ref into Ray actors. Lots of boilerplate; key methods:

```python
class ActorRolloutRefWorker(Worker):
    """One worker does actor training + vLLM rollout + reference policy forward"""
    
    def init_model(self):
        # 1. Build FSDP-wrapped actor module + optimizer
        self.actor_module_fsdp, self.actor_optimizer, ... = self._build_model_optimizer(...)
        
        # 2. DataParallelPPOActor wraps training logic
        if self._is_actor:
            self.actor = DataParallelPPOActor(...)
        
        # 3. vLLMRollout (shares weights, hybrid engine)
        if self._is_rollout:
            self.rollout, self.rollout_sharding_manager = self._build_rollout()
        
        # 4. Reference policy (same model, frozen weights)
        if self._is_ref:
            self.ref_module_fsdp = self._build_model_optimizer(...)
            self.ref_policy = DataParallelPPOActor(config=self.config.ref, ...)
    
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp, ...)
        metrics = self.actor.update_policy(data=data)
        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, ...)
        return output
    
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        # ★ Call vLLM rollout
        with self.rollout_sharding_manager:
            output = self.rollout.generate_sequences(prompts=prompts)
        # ★ Then recompute old_log_prob via FSDP forward
        if self._is_actor and recompute_log_prob:
            with self.ulysses_sharding_manager:
                old_log_probs = self.actor.compute_log_prob(data=output)
                output.batch['old_log_probs'] = old_log_probs
        return output
```

**Engineering details**:

1. **`load_fsdp_param_and_grad` / `offload_fsdp_param_and_grad`**: before each actor update load params + grad from CPU to GPU, offload back after. **Key to running 70B on single 8-GPU node**. Not needed for 8B but on by default.

2. **`@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)`**: veRL's decorator. `DP_COMPUTE_PROTO` means "split DataProto along DP dimension to workers, each computes its share, results merged." Distributed SPMD pattern — upper-level `actor_rollout_wg.update_actor(batch)` automatically fans out to N workers.

3. **Three `sharding_manager` contexts**:
   - `self.rollout_sharding_manager` — switch to vLLM view
   - `self.ulysses_sharding_manager` — switch to Ulysses sequence parallel view
   - default — FSDP view
   
   Each method enters the correct sharding context.

---

# End-to-end data flow

Putting it all together. One Search-R1 PPO training step's complete call stack:

```
ray_trainer.py:fit() main loop
│
├─ 1. Dataloader fetches batch
│    └─ rl_dataset.py:__getitem__ → tokenize + left padding
│    └─ collate_fn → pack into DataProto
│
├─ 2. ROLLOUT (search mode)
│    ├─ generation_manager.run_llm_loop(gen_batch, initial_input_ids)
│    │  │
│    │  ├─ for step in range(max_turns):
│    │  │  ├─ actor_rollout_wg.generate_sequences(rollings_active)
│    │  │  │  └─ fsdp_workers.py:generate_sequences
│    │  │  │     ├─ rollout_sharding_manager (switch to vLLM view)
│    │  │  │     └─ vllm_rollout.py:generate_sequences
│    │  │  │        └─ self.inference_engine.generate(...)
│    │  │  │           └─ [vLLM internals: sampling + autoregressive decode]
│    │  │  │
│    │  │  ├─ _postprocess_responses → truncate at </search>/</answer>
│    │  │  ├─ execute_predictions:
│    │  │  │  ├─ postprocess_predictions: regex extract action/content
│    │  │  │  ├─ batch_search(queries) → POST http://127.0.0.1:8000/retrieve
│    │  │  │  │  └─ retrieval_server.py:retrieve_endpoint
│    │  │  │  │     └─ DenseRetriever.batch_search:
│    │  │  │  │        ├─ Encoder.encode(queries) → e5-base-v2
│    │  │  │  │        ├─ faiss.search(query_emb, k=3)
│    │  │  │  │        └─ load_docs(corpus, idxs)
│    │  │  │  └─ return search_results
│    │  │  │
│    │  │  └─ _info_masked_concatenate_with_padding
│    │  │     ★ THE function ★ synchronously maintain (real_tokens, mask_version)
│    │  │
│    │  ├─ Last turn: do_search=False, force final answer
│    │  └─ _compose_final_output → expose info_mask to trainer
│    │
│    └─ actor_rollout_wg.compute_log_prob(final_gen_batch_output)
│       └─ ★ FSDP forward to recompute old_log_prob ★
│          └─ dp_actor.py:compute_log_prob
│             └─ _forward_micro_batch (with use_remove_padding)
│
├─ 3. REFERENCE log-prob
│    └─ ref_policy_wg.compute_ref_log_prob(batch)
│       └─ dp_actor.py:compute_log_prob (uses ref_module_fsdp weights)
│
├─ 4. CRITIC values (PPO only)
│    └─ critic_wg.compute_values(batch)
│       └─ dp_critic.py:compute_values
│          └─ _forward_micro_batch (value head, 1-dim output)
│
├─ 5. REWARD
│    └─ reward_fn(batch) = RewardManager.__call__
│       └─ for each item: qa_em.compute_score_em
│          ├─ extract_solution: regex `<answer>...</answer>` (≥2 matches)
│          ├─ em_check: normalize + string equality
│          └─ score ∈ {0, 1}
│       └─ ★ reward_tensor[i, valid_response_length - 1] = score ★ (sparse!)
│
├─ 6. KL PENALTY (PPO style)
│    └─ apply_kl_penalty(batch, kl_ctrl, kl_penalty='kl')
│       ├─ kld = core_algos.kl_penalty(old_log_probs, ref_log_prob, 'kl')
│       ├─ kld = kld * info_mask  ← retrieved tokens don't count
│       └─ token_level_rewards = token_level_scores - β * kld
│
├─ 7. ADVANTAGE
│    └─ compute_advantage(batch, adv_estimator='gae')
│       └─ core_algos.compute_gae_advantage_return
│          ├─ for t in reversed(range(gen_len)):
│          │  ├─ delta_t = r_t + γ·V(s_{t+1}) - V(s_t)
│          │  └─ A_t = delta_t + γλ·A_{t+1}
│          ├─ returns = advantages + values
│          └─ advantages = masked_whiten(advantages, response_mask)
│
├─ 8. CRITIC UPDATE
│    └─ critic_wg.update_critic(batch)
│       └─ dp_critic.py:update_critic
│          ├─ for each micro_batch:
│          │  ├─ vpreds = _forward_micro_batch
│          │  ├─ vf_loss = compute_value_loss(vpreds, returns, values, ...)
│          │  └─ loss.backward()
│          └─ _optimizer_step → FSDP grad clip + step
│
├─ 9. LOSS MASK
│    └─ _create_loss_mask(batch, metrics)
│       ├─ loss_mask = batch['info_mask'][:, -response_length:]
│       └─ batch['loss_mask'] = loss_mask
│
├─ 10. ACTOR UPDATE
│     └─ actor_rollout_wg.update_actor(batch)
│        └─ dp_actor.py:update_policy
│           ├─ select_keys += 'loss_mask'
│           ├─ for each micro_batch:
│           │  ├─ response_mask = data['loss_mask']  ★ retrieved-token masking takes effect
│           │  ├─ entropy, log_prob = _forward_micro_batch
│           │  ├─ pg_loss = compute_policy_loss(old_log_prob, log_prob, advantages, response_mask, ...)
│           │  ├─ entropy_loss = masked_mean(entropy, response_mask)
│           │  ├─ policy_loss = pg_loss - entropy_coeff * entropy_loss
│           │  └─ loss.backward()
│           └─ _optimizer_step
│
└─ 11. VALIDATE (every N steps)
      └─ _validate() → re-rollout val set, compute EM → log to wandb
```

**Search-R1's contributions land at**: step 2 entirely (replaces rollout), step 6's `info_mask` replacing `attention_mask`, step 9 (loss_mask conversion), step 10's `response_mask = data['loss_mask']` line.

**All other steps are standard veRL PPO**. This is why **agentic RL = standard PPO + multi-turn rollout + one mask**.

---

# How Search-R1 maps onto veRL's design

Concrete file-level attribution:

| Belongs to | Files |
| ---------- | ----- |
| **Search-R1's direct contribution** | `search_r1/llm_agent/generation.py` (469 lines)<br>`search_r1/llm_agent/tensor_helper.py` (74)<br>`search_r1/search/*.py` (1200+ lines for various retrievers) |
| **Search-R1's veRL patches** | `verl/trainer/ppo/ray_trainer.py:_create_loss_mask` (15 lines)<br>`verl/workers/actor/dp_actor.py` (2-3 lines for `state_masking` config)<br>`verl/utils/reward_score/qa_em.py` (139 lines, partly inherited from veRL stub) |
| **Standard veRL PPO/GRPO** | `verl/trainer/ppo/core_algos.py` (274)<br>`verl/workers/critic/dp_critic.py` (204)<br>`verl/workers/rollout/vllm_rollout/*.py` (226)<br>`verl/workers/fsdp_workers.py` (1054)<br>`verl/third_party/vllm/*` (~4000) |

Total Search-R1 contribution: **~600-700 lines of paper-specific code, plus a 15-line patch to ray_trainer.py**. Everything else is veRL.

This is a **clean modular architectural pattern**. The next agentic-RL paper / project can:
- Write their own `<their_specific_thing>.py` modeled on `generation.py`
- Add `info_mask` (or equivalent) to their rollout output
- Add a 1-line `state_masking=true` config to enable mask-aware loss
- Inherit everything else from veRL

This is exactly what [[polar|Polar]] does at a more general level (any harness, not just search) with `prefix_merging` (token-faithful traces from arbitrary tool-use sessions).

---

# Hands-on experiments — what to modify first

By modification difficulty:

### Difficulty 1: Change reward function

Easiest ablation. Edit `verl/utils/reward_score/qa_em.py` or register a new file in `__init__.py`:

```python
# Reward search usage
def compute_score_with_search_bonus(solution_str, ground_truth, format_score=0., score=1.):
    answer = extract_solution(solution_str)
    em = em_check(answer, ground_truth['target']) if answer else 0
    n_searches = len(re.findall(r'<search>', solution_str))
    return em + 0.05 * min(n_searches, 3)  # encourage search, cap at 3

# Penalize over-searching
def compute_score_with_search_penalty(solution_str, ground_truth):
    em = em_check(...)
    n_searches = len(re.findall(r'<search>', solution_str))
    return em - 0.1 * max(n_searches - 2, 0)  # penalty after 2 searches
```

Observe: how do these rewards change the model's search frequency (`valid_search_stats`) and EM? Best entry point for learning reward shaping.

### Difficulty 2: Change retriever / corpus

`retrieval_launch.sh` → change `--retriever_name` and `--index_path`. E5 → BM25 → SPLADE → BGE, see how the model adapts. BM25 has worse recall; will the model search more or give up? Best entry to retrieval-augmented learning.

### Difficulty 3: Add a new tool

In `execute_predictions` add a new action:

```python
elif action == 'calculate':
    result = safe_eval(content)  # call calculator
    next_obs.append(f'\n\n<information>Calculator result: {result}</information>\n\n')
    dones.append(0)
    valid_action.append(1)
    is_search.append(0)
```

`postprocess_predictions` regex must change too: `r'<(search|answer|calculate)>(.*?)</\1>'`. Prompt template needs the new instruction.

Run on GSM8K to see if the model learns **when to search vs when to compute**. This is multi-tool agentic RL entry.

### Difficulty 4: Use process reward instead of outcome

Hardest but most valuable. In `compute_score_em` consider not just final answer but:
- Did the model call search?
- Is the search query related to the question?
- Do the retrieved passages actually support the final answer?

Needs an RM or LLM-as-judge, but teaches you how process rewards work.

### Difficulty 5: Turn off `state_masking=true`

This is paper Table 4's ablation. Set `actor_rollout_ref.actor.state_masking=false`. Rerun training, see how fast it collapses and how much EM you lose. **Reproducing this ablation by hand is the only way to truly internalize why retrieved-token masking is necessary**.

---

# Connecting to the broader agentic-RL stack

After understanding Search-R1's code, you'll see [[prorl-agent|ProRL Agent]] and [[polar|Polar]] are doing **the same thing**, just more engineered:

| Component | Search-R1 (academic repo) | [[prorl-agent\|ProRL Agent]] | [[polar\|Polar]] |
| --------- | ------------------------- | ---------------------------- | ---------------- |
| Rollout loop | `generation.py:run_llm_loop` (469 lines) | HTTP `POST /process` + AgentHandler ABC | LLM-API proxy + trajectory reconstruction |
| Environment | FastAPI search server | Rootless Apptainer + AgentHandler | Rootless Apptainer + unmodified harness |
| Mask mechanism | `info_mask` → `loss_mask` | Token-in/out wire protocol | `loss_mask`: sampled=1, interstitial=0 |
| Multi-turn handling | Single `for step in range(max_turns)` | INIT/RUN/EVAL async pipeline | INIT/RUN/POSTRUN + READY buffer |
| Trainer integration | In-process (veRL fork) | HTTP, trainer-agnostic | HTTP, trainer-agnostic |

**Search-R1 is the earliest, cleanest implementation of this paradigm**. Reading it you'll find Polar's `loss_mask` description is **the same thing as Search-R1's `info_mask`, generalized** — Polar generalizes the concept to any harness and any tool.

---

# What's next

After reading the codebase, recommended next steps:

1. **Run a training reproducing Table 2** on Qwen2.5-3B. Single 8×A100 / H100 node, ~2 days.
2. **Run the `state_masking=false` ablation** to feel why the mask matters.
3. **Read [[prorl-agent]] and [[polar]] papers** — see how the same pattern scales to production.
4. **Pick one Difficulty-3+ modification** above and run it. Reading code is necessary but insufficient; running experiments is where understanding actually consolidates.
5. **Read follow-up papers** (ReSearch, ToolRL, R1-Searcher, DeepResearcher) to see how Search-R1's pattern was extended.

## Related reading

- [[search-r1]] — Paper review (what / why / experiments)
- [[agentic-rl-foundations]] — Onboarding hub with full reading path
- [[prorl-agent]] — Service-oriented agentic RL infrastructure
- [[polar]] — The current state-of-the-art rollout substrate
- [[nemo-gym]] — NVIDIA's environment-catalog framework
- [[grpo]] — The other RL algorithm Search-R1 uses
- [[ppo-for-llm]] — Foundational PPO-for-LLMs
- [[rl-training-frameworks]] — veRL / OpenRLHF / TRL landscape
- [[tool-use-rl]] — Broader tool-use RL family
