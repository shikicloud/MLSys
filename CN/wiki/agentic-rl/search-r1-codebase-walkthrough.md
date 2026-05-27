---
title: "Search-R1 代码完全解读 —— 从头入门 agentic RL"
category: agentic-rl
tags: [search-r1, code-walkthrough, agentic-rl, tutorial, verl, ppo-implementation, grpo-implementation, retrieved-token-masking]
created: 2026-05-26
updated: 2026-05-26
status: mature
code: https://github.com/PeterGriffinJin/Search-R1
---

# Search-R1 代码完全解读 —— 从头入门 agentic RL

> [!tip] 本页配合 [[search-r1]] 阅读
> 先看 paper review 了解 **是什么 / 为什么**。本页讲 **怎么实现** —— 逐文件、逐函数、加上一次训练 step 的真实数据流。

> [!abstract]+ 本页覆盖什么
> Search-R1 仓库（[PeterGriffinJin/Search-R1](https://github.com/PeterGriffinJin/Search-R1)）是市面上最干净的 agentic RL 参考实现。总规模 ~33,000 行 Python，但真正属于 Search-R1 的部分只有 ~600 行（`search_r1/`），其余是 [veRL](https://github.com/volcengine/verl) 的 fork。本 walkthrough 带你过：
>
> 1. **600 行** Search-R1 直接贡献 —— 多轮 rollout 循环、检索 server、入口脚本
> 2. **~5,000 行** Search-R1 倚靠的 veRL PPO/GRPO 机器 —— trainer 循环、actor/critic 更新、GAE 数学、vLLM 封装
> 3. **端到端数据流** —— 一次 PPO 训练 step 的完整调用栈
>
> 读完你不只懂 Search-R1，还懂 ProRL Agent、Polar、NeMo Gym 和整个 2025-26 agentic-RL 领域用的架构 pattern。

## 仓库结构总览

```
Search-R1/
├── train_ppo.sh                ← 入口 #1：训练
├── train_grpo.sh               ← 入口 #2：训练（GRPO）
├── retrieval_launch.sh         ← 入口 #3：启动 search server
├── infer.py                    ← 入口 #4：推理
│
├── search_r1/                  ← Search-R1 特有代码（~600 行）
│   ├── llm_agent/
│   │   ├── generation.py       ★ 核心文件：多轮 rollout（469）
│   │   └── tensor_helper.py      工具：padding/mask（74）
│   └── search/
│       ├── retrieval_server.py     FastAPI E5+FAISS server（392）
│       ├── google_search_server.py 真实 web 搜索变体（202）
│       ├── rerank_server.py / retrieval_rerank_server.py
│       ├── serp_search_server.py
│       └── index_builder.py    FAISS 索引构建（349）
│
├── scripts/
│   └── data_process/
│       └── nq_search.py        NQ → parquet + template
│
└── verl/                       ← fork 的 veRL 框架（~32,000 行）
    ├── trainer/main_ppo.py     程序入口（202）
    ├── trainer/ppo/
    │   ├── ray_trainer.py      ★ 主 fit() 循环（867）
    │   └── core_algos.py       ★ PPO/GRPO 数学：GAE、clip、KL（274）
    ├── workers/
    │   ├── actor/dp_actor.py   ★ actor 更新（290）
    │   ├── critic/dp_critic.py ★ critic 更新（204）
    │   ├── rollout/vllm_rollout/vllm_rollout.py  vLLM 封装（226）
    │   └── fsdp_workers.py     编排（1054）
    ├── utils/
    │   ├── reward_score/qa_em.py   outcome reward（139）
    │   └── dataset/rl_dataset.py   dataloader（156）
    └── third_party/vllm/       fork 的 vLLM 补丁（~4000 行）
```

**关键观察**：Search-R1 本身**就是 veRL 的一个 plugin**。它不改 RL 算法，不碰 PPO/GRPO 核心。它只在两个地方插代码：

1. **`generation.py`** —— 替换默认 `rollout()` 为多轮 search-interleaved 生成
2. **`ray_trainer.py:_create_loss_mask`** —— 把 `generation.py` 返回的 `info_mask` 转成 `loss_mask`，交给标准 PPO loss

就这。理解了这个 pattern，你会看到它在 agentic RL 里无处不在：**改 rollout、加一个 mask、RL 算法不动**。

## 推荐阅读顺序

新手读代码的最优路径：

1. **`train_ppo.sh`** —— 看超参，分清 trainer 管的 vs Search-R1 加的
2. **`scripts/data_process/nq_search.py`** —— prompt template、ground truth 格式
3. **`retrieval_launch.sh` + `retrieval_server.py` FastAPI 部分** —— 环境接口
4. **`generation.py:run_llm_loop`** —— 高层 rollout 循环
5. **`generation.py:execute_predictions`** —— 环境 step
6. **`generation.py:_info_masked_concatenate_with_padding`** ★ —— loss masking 心脏
7. **`generation.py:_compose_final_output`** —— 把 info_mask 暴露给 trainer
8. **`ray_trainer.py:_create_loss_mask`** —— info_mask → loss_mask
9. **`dp_actor.py:update_policy`** —— loss_mask 实际生效的地方
10. **`qa_em.py`** —— reward
11. （可选深读）`core_algos.py`、`vllm_rollout.py`、`fsdp_workers.py`

精读一遍大约 4-6 小时。

---

# Part A：Search-R1 自己的代码（~600 行）

## 1. 训练入口 `train_ppo.sh`

```bash
export DATA_DIR='data/nq_search'
export BASE_MODEL='Qwen/Qwen2.5-3B'

python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train.parquet \
    data.max_prompt_length=4096 \                       # 整段 rollout 上限
    data.max_response_length=500 \                      # 单轮 LLM 生成上限
    data.max_start_length=2048 \                        # 原始 prompt 上限
    data.max_obs_length=500 \                           # 检索注入上限
    algorithm.adv_estimator=gae \                       # PPO 用 GAE
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \             # LR 很小
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.actor.state_masking=true \        # ★ 开启 info-mask
    algorithm.kl_ctrl.kl_coef=0.001 \                   # KL 惩罚很弱
    trainer.n_gpus_per_node=8 \
    trainer.total_training_steps=1005 \
    max_turns=2 \                                       # ★ 多轮上限
    retriever.url="http://127.0.0.1:8000/retrieve" \    # 搜索 server
    retriever.topk=3
```

关键超参：

| 参数 | 含义 | 论文用值 |
| ---- | ---- | -------- |
| `max_turns` | 每 rollout 允许的最大搜索次数 | 2 |
| `max_obs_length` | 每次检索注入 token 上限 | 500 |
| `state_masking=true` | **开启 retrieved-token loss masking** | true |
| `kl_coef=0.001` | KL 到 reference 系数 | 0.001 |
| `topk=3` | 每次搜索返回 top-k | 3 |
| `lr=1e-6` | actor LR | 1e-6 |
| `temperature=1.0` | rollout 采样温度 | 1.0 |

`max_prompt_length=4096` 跟 `max_start_length + max_response_length × (max_turns-1) + max_obs_length × max_turns` 大致对应。

## 2. 数据准备 `scripts/data_process/nq_search.py`

最关键的是 prompt template：

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

这就是 **R1-Zero 风格 template** —— 全靠 prompt 告诉模型协议。无 SFT 暖身、无 in-context examples。模型靠 RL **自己学会遵守 token 协议**。

输出（存为 parquet）：
```python
{
    "prompt": [{"role": "user", "content": prefix}],
    "data_source": "nq",
    "reward_model": {"style": "rule", "ground_truth": {"target": ["Albert Einstein", ...]}},
}
```

`ground_truth.target` 是可接受答案列表，EM 跟里面任何一个对上就算正确。

## 3. 检索服务 `search_r1/search/retrieval_server.py`

这是 **environment 的实现**。独立进程跑的 FastAPI server，监听 8000 端口：

```python
class DenseRetriever(BaseRetriever):
    def __init__(self, config):
        self.index = faiss.read_index(self.index_path)        # FAISS index
        if config.faiss_gpu:
            self.index = faiss.index_cpu_to_all_gpus(...)     # 多卡 FAISS
        self.corpus = load_corpus(self.corpus_path)            # wiki-18.jsonl
        self.encoder = Encoder(model_path=...)                 # E5 encoder
    
    def _batch_search(self, query_list, num):
        batch_emb = self.encoder.encode(query_list)
        batch_scores, batch_idxs = self.index.search(batch_emb, k=num)
        results = load_docs(self.corpus, flat_idxs)
        return results

@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    results, scores = retriever.batch_search(
        query_list=request.queries, num=request.topk, return_score=request.return_scores)
    return {"result": resp}
```

**核心设计**：

1. **独立进程**：检索跟训练分离，HTTP 通信。跟 [[prorl-agent|ProRL Agent]] "rollout 作为服务" 是同一 pattern 的简化版
2. **E5 + FAISS**：query 用 [intfloat/e5-base-v2](https://huggingface.co/intfloat/e5-base-v2) 编码成 768-dim vector，FAISS ANN 在 Wikipedia 2018 dump (~21M passages) 上检索
3. **GPU FAISS**：可选 `--faiss_gpu`，FAISS index 放 GPU，检索 ~ms 级
4. **批量接口**：一个 batch 内多条 trajectory 可能同时触发 search，server 接受 list

**真实 web 搜索版**：`google_search_server.py` 实现同样接口，但底下调 SerpAPI 真实 Google。生产更接近"agentic"，但 Wikipedia 便宜、可复现，所以论文 main result 用 Wikipedia。

```bash
# 启动
bash retrieval_launch.sh
# python search_r1/search/retrieval_server.py \
#     --index_path /path/to/e5_Flat.index \
#     --corpus_path /path/to/wiki-18.jsonl \
#     --topk 3 \
#     --retriever_name e5 \
#     --retriever_model intfloat/e5-base-v2 \
#     --faiss_gpu
```

`train_ppo.sh` 通过 `retriever.url=http://127.0.0.1:8000/retrieve` 调它。

## 4. 核心：`search_r1/llm_agent/generation.py`（469 行）

这是整个 repo 的灵魂。按重要性顺序讲。

### 4.1 `GenerationConfig`

```python
@dataclass
class GenerationConfig:
    max_turns: int                # 多轮上限
    max_start_length: int         # 原始 prompt 长度
    max_prompt_length: int        # 整段拼接后总长度
    max_response_length: int      # 单轮 LLM 生成长度
    max_obs_length: int           # 单次检索注入长度
    num_gpus: int
    no_think_rl: bool = False     # ablation 用
    search_url: str = None        # 检索 server URL
    topk: int = 3                 # 每次检索 top-k
```

所有 turn-related 超参收在这里。

### 4.2 主循环 `run_llm_loop`（高层）

```python
def run_llm_loop(self, gen_batch, initial_input_ids):
    # 左边（不变）= 原始 prompt
    original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
    # 右边（增长）= 模型生成 + 注入的 observation
    original_right_side = {
        'responses': initial_input_ids[:, []],
        'responses_with_info_mask': initial_input_ids[:, []]
    }
    
    active_mask = torch.ones(B, dtype=torch.bool)        # 哪些 trajectory 还在活
    turns_stats = torch.ones(B, dtype=torch.int)
    valid_search_stats = torch.zeros(B, dtype=torch.int) # 每条调了几次 search
    rollings = gen_batch                                  # 当前喂给 LLM 的 input
    
    # ★ 主循环：最多 max_turns 次 ★
    for step in range(self.config.max_turns):
        if not active_mask.sum(): break
        
        # 1) 只取还活着的 batch 元素
        rollings_active = DataProto.from_dict({k: v[active_mask] for k, v in rollings.batch.items()})
        # 2) vLLM 生成下一段
        gen_output = self._generate_with_gpu_padding(rollings_active)
        # 3) 在 </search> 或 </answer> 处截断
        responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
        # 4) ★ 调环境（搜索）★
        next_obs, dones, valid_action, is_search = self.execute_predictions(...)
        # 5) 更新 active_mask
        active_mask = active_mask * curr_active_mask
        # 6) 把 (response, observation) 拼到 rollings 和 right_side 上
        rollings = self._update_rolling_state(rollings, responses_ids, next_obs_ids)
        original_right_side = self._update_right_side(original_right_side, responses_ids, next_obs_ids)
    
    # 最后一轮：让活着的 trajectory 出 final answer（不调 search）
    if active_mask.sum():
        gen_output = self._generate_with_gpu_padding(rollings_active)
        # ... 同上但 do_search=False
        original_right_side = self._update_right_side(original_right_side, responses_ids)
    
    return self._compose_final_output(original_left_side, original_right_side, meta_info)
```

**关键概念**：

- **left_side**：原始 prompt，不变
- **right_side**：模型生成 + 注入的 observation，循环中增长
- **rollings**：当前喂给 LLM 的完整 input（= left + right_so_far）
- **active_mask**：哪些 trajectory 还在活动 —— 已输出 `<answer>` 的就 done，不再继续
- **`responses_with_info_mask`**：跟 `responses` 同长，但**在 information block 位置填 pad_id**。这是 info_mask 的源头

**为什么要 active_mask？** 一个 batch 里不同 trajectory 不同 —— 有的 1 轮就 answer，有的要 2 轮。已经 done 的不该再让模型 forward。

**为什么 left/right 分开？** PPO loss 只在 response token 上算（prompt 不参与梯度）。分开能精确知道"response 是哪段"。最后 `_compose_final_output` 拼回去。

### 4.3 `_postprocess_responses` —— 提前停在 stop token

```python
def _postprocess_responses(self, responses):
    responses_str = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
    responses_str = [resp.split('</search>')[0] + '</search>'   # 找到 </search> 就截到这里
             if '</search>' in resp 
             else resp.split('</answer>')[0] + '</answer>'      # 同理 </answer>
             if '</answer>' in resp 
             else resp                                          # 都没有就保留全部
             for resp in responses_str]
    responses = self._batch_tokenize(responses_str)
    return responses, responses_str
```

模型在 `max_response_length=500` 内自由生成，但**逻辑上停止于 `</search>` 或 `</answer>`** —— 后面的丢掉。原因：

- `</search>` 后再生成无意义（要等检索结果再继续）
- `</answer>` 后再生成不可信（hallucinated continuation）

**这是 agentic RL 第二个工程细节**：你必须**在 token 流里检测 stop signal 然后立即停止**。vLLM 支持 `stop=["</search>", "</answer>"]` 直接做，但 Search-R1 用 post-hoc string split（更通用、tokenizer-agnostic）。

### 4.4 `execute_predictions` —— "env step" 函数

论文 Algorithm 1 的实际实现：

```python
def execute_predictions(self, predictions, pad_token, active_mask, do_search=True):
    """
    NOTE: the function is the actual `step` function in the environment
    """
    cur_actions, contents = self.postprocess_predictions(predictions)
    # cur_actions[i] ∈ {'search', 'answer', None}
    
    next_obs, dones, valid_action, is_search = [], [], [], []
    
    # 把所有 'search' action 的 queries 收起来一次性发请求
    search_queries = [content for action, content in zip(cur_actions, contents) if action == 'search']
    if do_search:
        search_results = self.batch_search(search_queries)   # ← 一次 HTTP POST
    else:
        search_results = [''] * sum(1 for a in cur_actions if a == 'search')

    for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
        if not active:
            next_obs.append(''); dones.append(1); valid_action.append(0); is_search.append(0)
        else:
            if action == 'answer':
                next_obs.append('')                           # answer 后没 obs
                dones.append(1)                               # ← trajectory 结束
                valid_action.append(1); is_search.append(0)
            elif action == 'search':
                next_obs.append(f'\n\n<information>{search_results.pop(0).strip()}</information>\n\n')
                dones.append(0)                               # ← 继续
                valid_action.append(1); is_search.append(1)
            else:   # 乱七八糟的输出
                next_obs.append('\nMy previous action is invalid. ...')
                dones.append(0)
                valid_action.append(0); is_search.append(0)
    
    return next_obs, dones, valid_action, is_search
```

**重要细节**：

1. **Action parsing 用 regex**（`postprocess_predictions`）：
   ```python
   pattern = r'<(search|answer)>(.*?)</\1>'
   match = re.search(pattern, prediction, re.DOTALL)
   ```
   只允许 `<search>` 或 `<answer>` 两种 action。Regex 失败 → action = None

2. **错误处理**：
   ```python
   next_obs.append('My previous action is invalid. ...')
   dones.append(0)  # 还不退出，让它再试
   ```
   注入"提醒"让模型重试。对应论文 Algorithm 1 line 19

3. **批量化搜索**：所有 search action 一次发请求。一个 256 trajectory 的 batch 里若 100 个要 search，一次请求 vs 100 次请求 —— 差几个数量级

4. **`do_search=False` 用在哪？** 最后一轮（force-final）调，因为最后一轮即使模型要 search 也没用 —— 后面没有更多 turn 让它消化检索结果

### 4.5 ★ 关键函数 `_info_masked_concatenate_with_padding`

这是整个 repo 最重要的一段代码。**retrieved-token loss masking 在 token 层的实现**：

```python
def _info_masked_concatenate_with_padding(self, prompt, prompt_with_mask, response, info=None, pad_to_left=True):
    """
    Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to 
    cover the information block if it exists.
    """
    pad_id = self.tokenizer.pad_token_id
    tensors = [prompt, response]                      # 真实拼接
    tensors_with_mask = [prompt_with_mask, response]  # 带 mask 的版本
    if info is not None:                              # 如果有检索回来的 obs
        tensors.append(info)                          # 真实版本：拼 info
        info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device)
        tensors_with_mask.append(info_mask)           # ★ mask 版本：info 位置填 pad_id ★
    
    concatenated = torch.cat(tensors, dim=1)                       # 真实 token: prompt + response + info
    concatenated_with_info = torch.cat(tensors_with_mask, dim=1)   # mask 版: prompt + response + [pad pad pad...]
    
    # Padding 对齐
    mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
    sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
    padded_tensor = concatenated.gather(1, sorted_indices)
    padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)
    
    return padded_tensor, padded_tensor_with_info
```

**这段代码的 magic**：

它**同时构造两个 tensor**：
- `concatenated`：真实 input 序列，含 prompt + response + retrieved info（forward 用）
- `concatenated_with_info`：**相同位置但 retrieved info 替换成 pad_token_id**（mask 用）

两个 tensor **token 长度完全一致、对齐方式完全一致**，但在 retrieved info 的位置上：
- `concatenated[..., info_position] = 真实检索到的 token IDs`
- `concatenated_with_info[..., info_position] = pad_id, pad_id, pad_id, ...`

后续 `_compose_final_output` 里 `concatenated_with_info != pad_id` 算出来的 attention mask **正好把 info 位置标 0**。这就是 `info_mask`。

**为什么这样做而不直接 binary mask？** 因为 batch 内不同 trajectory 的 info 出现在不同位置（取决于何时调 search），用 token 级替换为 pad_id 再算 attention mask 是最简洁的：左右 padding 整理一次就处理好。

读 agentic RL 代码时，**找到这种"同步维护两个张量"的 pattern** 你就找到了 mask logic。Polar 的 `prefix_merging` 也是同样设计（真实 token 数组 + loss_mask 数组对齐）。

### 4.6 `_update_right_side` —— 把新一轮拼上去

```python
def _update_right_side(self, right_side, cur_responses, next_obs_ids=None):
    if next_obs_ids != None:
        responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                right_side['responses'],
                right_side['responses_with_info_mask'],
                cur_responses,
                next_obs_ids,
                pad_to_left=False                  # 右边追加
            )
    else:                                          # 最后一轮没 obs
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

每轮调一次，把 `(new_response, new_obs)` 追加到 `right_side`，**两个版本同步增长**。

### 4.7 `_compose_final_output` —— 整理输出给 trainer

```python
def _compose_final_output(self, left_side, right_side, meta_info):
    final_output = right_side.copy()
    final_output['prompts'] = left_side['input_ids']
    final_output['input_ids'] = torch.cat([left_side['input_ids'], right_side['responses']], dim=1)
    
    # 普通 attention mask：左+右都是 1（除了 pad）
    final_output['attention_mask'] = torch.cat([
        self.tensor_fn.create_attention_mask(left_side['input_ids']),
        self.tensor_fn.create_attention_mask(final_output['responses'])
    ], dim=1)
    
    # ★ info_mask：左+右，但 right 的 info 位置是 0 ★
    final_output['info_mask'] = torch.cat([
        self.tensor_fn.create_attention_mask(left_side['input_ids']),
        self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
    ], dim=1)
    
    final_output['position_ids'] = self.tensor_fn.create_position_ids(final_output['attention_mask'])
    return DataProto.from_dict(final_output)
```

给 trainer 的 batch：

- `input_ids`：完整 rollout（prompt + response_with_real_info）
- `attention_mask`：普通 mask（pad=0，其它=1）—— vLLM forward 用
- **`info_mask`**：同形状，**所有 retrieved info 位置是 0** —— loss masking 用
- `position_ids`：从 attention_mask 推

`attention_mask` 给 forward 用（"哪些 token 该被 attend"），`info_mask` 给 loss 用（"哪些 token 该参与梯度"）。**两者解耦是关键**。

### 4.8 `_generate_with_gpu_padding` —— 工程细节

```python
def _generate_with_gpu_padding(self, active_batch):
    """if active_batch size is not divisible by num_gpus, pad with first sequence then remove"""
    num_gpus = self.config.num_gpus
    batch_size = active_batch.batch['input_ids'].shape[0]
    remainder = batch_size % num_gpus
    if remainder == 0:
        return self.actor_rollout_wg.generate_sequences(active_batch)
    
    # active_batch 不是 num_gpus 倍数 → 用第一条 sequence 填齐
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

**这个工程细节是 agentic-RL 特有的**：

正常 RL 训练 batch size 固定，全部 trajectory 每一步同步 forward。agentic 里 **active_mask 让 batch 缩水**（早期 answer 的不再 forward），剩下的 trajectory 数量可能**不是 num_gpus 的倍数**。vLLM 多卡跑要求 batch 整除 num_gpus（要均匀分片），所以手动**先 pad 到整除，generate 完再扔掉**。

新手很容易踩这个坑。

## 5. `tensor_helper.py` —— 小工具（74 行）

```python
class TensorHelper:
    def cut_to_effective_len(self, tensor_dict, keys, cut_left=True):
        """根据 attention mask 把 tensor 切到有效长度"""
        effective_len = tensor_dict['attention_mask'].sum(dim=1).max()
        # ...
    
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

标准 padding/mask 工具。有意思的是 `_example_level_pad` —— 只给 active examples 生成时为了保持全 batch 形状（active_mask 干净工作），把 inactive 位置 pad 回来。

## 6. `infer.py` —— 参考推理（130 行）

训练完后用 HF Transformers + `StoppingCriteria` 跑推理：

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

# ★ 多变体处理 tokenizer 分词差异
target_sequences = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]
stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(target_sequences, tokenizer)])

while True:
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    outputs = model.generate(
        input_ids, max_new_tokens=1024,
        stopping_criteria=stopping_criteria,
        do_sample=True, temperature=0.7
    )
    
    if outputs[0][-1].item() in curr_eos:
        break    # EOS，结束
    
    tmp_query = get_query(tokenizer.decode(outputs[0], skip_special_tokens=True))
    if tmp_query:
        search_results = search(tmp_query)
    
    prompt += f'\n\n{output_text}<information>{search_results}</information>\n\n'
```

**工程细节**：

1. **`target_sequences` 有 6 个变体**：`</search>`、` </search>`、`</search>\n`、...。Tokenizer 在不同上下文里分词不同（有时带前导空格、有时带换行），所以要枚举所有可能的"真实 token 序列"。**这是工业代码 vs 论文公式的差距**
2. **`do_sample=True, temperature=0.7`**：推理用温度采样，不是 greedy。论文 rollout temperature=1.0，推理时降到 0.7 让答案更稳
3. **跟训练 rollout 形状一致**：每轮 stopped on `</search>` → 提取 query → 调搜索 → 拼 `<information>...</information>` → 继续。**用 HF generate 重新实现 generation.py 的逻辑**

这个文件 ~130 行，**新手从这里读 Search-R1 入门** —— 比训练侧的 `generation.py` 简单一倍。

---

# Part B：veRL 机器（~5,000 行）

## 7. `verl/trainer/main_ppo.py` —— 程序入口（202 行）

```python
@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})
    ray.get(main_task.remote(config))

@ray.remote
def main_task(config):
    # 1. 加载 base model 和 tokenizer
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)
    tokenizer = hf_tokenizer(local_path)
    
    # 2. 选 worker 实现：FSDP 还是 Megatron
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        ray_worker_group_cls = RayWorkerGroup

    # 3. 三种 role
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),   # ★ actor + rollout 合体
        Role.Critic:       ray.remote(CriticWorker),
        Role.RefPolicy:    ray.remote(ActorRolloutRefWorker),   # ★ 跟 actor 同类，但只跑 forward
    }
    
    # 4. 把 8 张 GPU 全分到 global_pool
    resource_pool_spec = {global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes}
    mapping = {Role.ActorRollout: global_pool_id, Role.Critic: global_pool_id, Role.RefPolicy: global_pool_id}
    
    # 5. Reward 函数（规则化，不学习）
    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0)
    
    # 6. Build trainer，开训
    trainer = RayPPOTrainer(config=config, tokenizer=tokenizer, ...)
    trainer.init_workers()
    trainer.fit()
```

**关键设计**：

1. **`@ray.remote`**：整个训练 job 作为 Ray remote function 跑。Ray 是 veRL 的分布式编排框架
2. **三个 role 共享 8 张卡**（`global_pool_id`）：actor/rollout/ref 全在同一组 GPU 上，但用 FSDP offload 错开内存峰值。这是 veRL 的 "hybrid engine" 设计：训练时 actor 参数在 GPU，rollout 时切到 vLLM 视角
3. **`Role.ActorRollout` 在 fsdp_workers.py 里同时实例化 actor + vllm rollout** —— 不是两个独立 process，而是一个 worker 进程里两套 view，参数同一份

### `RewardManager.__call__` —— 给每条 trajectory 算分

```python
class RewardManager():
    def __call__(self, data: DataProto):
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        for i in range(len(data)):
            data_item = data[i]
            # ...
            # decode
            sequences_str = self.tokenizer.decode(sequences)
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)
            score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, ...)
            
            # ★ 关键：reward 只放在 response 最后一个 token 的位置 ★
            reward_tensor[i, valid_response_length - 1] = score
        return reward_tensor
```

**这段告诉你 sparse outcome reward 在实现层长什么样**：

`reward_tensor` 形状 `(batch_size, response_length)`，**初始化全 0**，**只在每条 trajectory 的最后一个有效 token 位置上填一个 scalar**（0 或 1）。其它位置全是 0。

这就是"outcome-only sparse reward"的字面意思 —— **只有终点有信号，中间全是零**。后续 GAE 会把这个最终 reward 反向传播到前面 token 上（通过 value function 做 credit assignment）。

## 8. `verl/utils/dataset/rl_dataset.py` —— 数据加载（156 行）

```python
class RLHFDataset(Dataset):
    def __getitem__(self, item):
        row_dict = self.dataframe.iloc[item].to_dict()
        chat = row_dict.pop(self.prompt_key)
        
        if self.tokenizer.chat_template:
            prompt_with_chat_template = self.tokenizer.apply_chat_template(
                chat, add_generation_prompt=True, tokenize=False)
        
        # tokenize + 左 padding
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt_with_chat_template,
            tokenizer=self.tokenizer,
            max_length=self.max_prompt_length,    # 4096
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,                         # ★ 左 padding ★
            truncation=self.truncation
        )
        
        # 用于 GRPO 分组的 index
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index
        return row_dict
```

**两件事要注意**：

1. **左 padding (`left_pad=True`)**：
   - 标准 SFT 训练用右 padding（token 顺序 = 生成顺序）
   - **生成（rollout）必须用左 padding**，vLLM autoregressive 只看右边的最后一个有效位置
   - `generation.py:_info_masked_concatenate_with_padding` 那个 `pad_to_left` 参数就是因为这个

2. **`row_dict["index"]`**：每条样本一个 ID，**GRPO 用这个把同一 prompt 的 N 条 rollout 分组**。GRPO 的 baseline = "同一 prompt 下 N 条 rollout 的 reward 均值"，必须有 group ID 才能做。PPO 不用

`collate_fn` 把多条 sample 打包成一个 batch。**`DataProto` = TensorDict + NumPy object dict**，veRL 的核心数据容器。

## 9. `ray_trainer.py:fit()` —— 主训练循环（867 行）

整个 PPO/GRPO 主循环：

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
                # 1. ROLLOUT —— 分叉
                # ════════════════════════════════════════════════════════
                if not self.config.do_search:
                    # 普通 PPO：单次 vLLM forward
                    gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                else:
                    # ★ Search-R1 的路径：multi-turn rollout ★
                    first_input_ids = gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone().long()
                    with _timer('gen', timing_raw):
                        final_gen_batch_output = generation_manager.run_llm_loop(
                            gen_batch=gen_batch, initial_input_ids=first_input_ids)
                    # 重算 old_log_prob
                    with torch.no_grad():
                        output = self.actor_rollout_wg.compute_log_prob(final_gen_batch_output)
                        final_gen_batch_output = final_gen_batch_output.union(output)
                
                # ════════════════════════════════════════════════════════
                # 2. REFERENCE log-prob（KL penalty 用）
                # ════════════════════════════════════════════════════════
                if self.use_reference_policy:
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                    batch = batch.union(ref_log_prob)
                
                # ════════════════════════════════════════════════════════
                # 3. CRITIC values（PPO 用，GRPO 跳）
                # ════════════════════════════════════════════════════════
                if self.use_critic:
                    values = self.critic_wg.compute_values(batch)
                    batch = batch.union(values)
                
                # ════════════════════════════════════════════════════════
                # 4. REWARD + ADVANTAGE
                # ════════════════════════════════════════════════════════
                with _timer('adv', timing_raw):
                    reward_tensor = self.reward_fn(batch)
                    batch.batch['token_level_scores'] = reward_tensor
                    
                    if not self.config.actor_rollout_ref.actor.use_kl_loss:
                        batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl, ...)
                    else:
                        batch.batch['token_level_rewards'] = batch.batch['token_level_scores']
                    
                    batch = compute_advantage(batch, adv_estimator=self.config.algorithm.adv_estimator, ...)
                
                # ════════════════════════════════════════════════════════
                # 5. CRITIC update（PPO 用）
                # ════════════════════════════════════════════════════════
                if self.use_critic:
                    critic_output = self.critic_wg.update_critic(batch)
                
                # ════════════════════════════════════════════════════════
                # 6. ACTOR update —— retrieved-token loss masking 起作用
                # ════════════════════════════════════════════════════════
                if self.config.trainer.critic_warmup <= self.global_steps:
                    with _timer('update_actor', timing_raw):
                        if self.config.do_search and self.config.actor_rollout_ref.actor.state_masking:
                            batch, metrics = self._create_loss_mask(batch, metrics)
                        actor_output = self.actor_rollout_wg.update_actor(batch)
            
            # 7. validate + save
            if self.val_reward_fn is not None and self.global_steps % self.config.trainer.test_freq == 0:
                val_metrics = self._validate()
            
            self.global_steps += 1
```

**7 个阶段全展开**。每个 batch 走一遍。

**易混淆概念**：

- `token_level_scores` = reward 函数算出来的原始分数（只在最后一个 token 非零）
- `token_level_rewards` = scores 减 KL penalty 后的"真实"reward（PPO 风格）或直接等于 scores（GRPO 风格）
- `advantages` = GAE / GRPO 算出来的优势
- `returns` = `advantages + values`（PPO 里给 critic 当 target）

### `apply_kl_penalty` —— PPO 风格 KL 处理

```python
def apply_kl_penalty(data, kl_ctrl, kl_penalty='kl'):
    response_length = data.batch['responses'].size(1)
    token_level_scores = data.batch['token_level_scores']
    # ★ 关键：如果有 info_mask，KL penalty 也按 info_mask 算
    attention_mask = data.batch['info_mask'] if 'info_mask' in data.batch else data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]
    
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'], kl_penalty=kl_penalty)
        kld = kld * response_mask  # ← 屏蔽 retrieved + pad token
        beta = kl_ctrl.value
    else:
        beta = 0; kld = torch.zeros_like(response_mask, dtype=torch.float32)
    
    # ★ PPO 风格：把 -β·KL 加到 reward 上
    token_level_rewards = token_level_scores - beta * kld
    return data, metrics
```

**两种 KL 处理风格**（一定要分清）：

- **PPO 风格（`use_kl_loss=False`）**：把 `-β·KL` 加进 reward → KL 自动通过 GAE 反向传播到 token 级 advantage 上。**KL 没显式 loss 项**，但通过 reward 影响梯度
- **GRPO 风格（`use_kl_loss=True`)**：reward 不动，KL 单独作为 actor loss 的一项（在 `dp_actor.py:update_policy` 里）

DeepSeek-R1 / GRPO 论文用第二种，PPO 经典做法是第一种。Search-R1 默认 PPO 用第一种，跑 GRPO 时切到第二种。**这是个重要工程区别**。

注意第一行 `attention_mask = data.batch['info_mask'] if 'info_mask' in data.batch else data.batch['attention_mask']` —— **info_mask 在这里就开始起作用**了，retrieved token 不进 KL。

### `compute_advantage` —— GAE 或 GRPO 分叉

简单分发：GAE 走 `compute_gae_advantage_return`（需要 values），GRPO 走 `compute_grpo_outcome_advantage`（需要 group index）。

## 10. `core_algos.py` —— 核心数学（274 行）

只有 274 行，但里面**全是 PPO 论文里那些公式的代码实现**。

### 10.1 `compute_gae_advantage_return` —— GAE 的 30 行实现

```python
def compute_gae_advantage_return(token_level_rewards, values, eos_mask, gamma, lam):
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]
        
        # ★ 从后往前递推 ★
        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            # TD error: δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            # GAE: A_t = δ_t + γλ·A_{t+1}
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        
        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, eos_mask)  # ★ 归一化 ★
    return advantages, returns
```

核心是 GAE 递推公式：

$$
A_t^{GAE} = \delta_t + \gamma \lambda \cdot A_{t+1}^{GAE}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

实现上从最后一个 token 反向递推。

**入门必须看懂的点**：

1. **`token_level_rewards`** 在 Search-R1 里是**几乎全 0 的张量**。GAE 把这个最终 reward 通过 `value function + discount` 反向传播分摊到每个 token 的 advantage 上。**这就是 credit assignment 在数学上的实现**
2. **`returns = advantages + values`** —— critic 的训练 target
3. **`masked_whiten`** = 减均值除标准差。**对 PPO 稳定性极其重要** —— 没有 advantage normalization PPO 经常 NaN

### 10.2 `compute_grpo_outcome_advantage` —— GRPO 用 group mean 当 baseline

```python
def compute_grpo_outcome_advantage(token_level_rewards, eos_mask, index, epsilon=1e-6):
    response_length = token_level_rewards.shape[-1]
    non_zero_mask = (token_level_rewards != 0)
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)  # 每条 trajectory 一个 scalar
    
    id2score = defaultdict(list)
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])    # 按 group 收集 reward
        for idx in id2score:
            if len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))   # ★ baseline = group mean
                id2std[idx]  = torch.std(torch.tensor([id2score[idx]]))
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask
    return scores, scores
```

**GRPO 的全部数学就这 30 行**。对比 GAE：

| 维度 | GAE (PPO) | GRPO |
| ---- | --------- | ---- |
| 需要 value function？ | 是 | **否** |
| Token 级 advantage？ | 是 | **否（整条 trajectory 同一个 advantage）** |
| Baseline | $V(s_t)$ | **同 prompt N 条 rollout 的 mean reward** |
| 多轮 / 长序列稳定性 | 好 | 容易方差爆炸 |

**注意** `scores.unsqueeze(-1).tile([1, response_length])` 这一行 —— "GRPO advantage 在所有 token 上是同一 scalar"。**跟 GAE 完全不同**。也是 Search-R1 GRPO 长 rollout 上崩的原因。

### 10.3 `compute_policy_loss` —— PPO clipped objective

```python
def compute_policy_loss(old_log_prob, log_prob, advantages, eos_mask, cliprange):
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    
    pg_losses  = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    
    pg_loss = verl_F.masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
    return pg_loss, pg_clipfrac, ppo_kl
```

**[PPO paper](https://arxiv.org/abs/1707.06347) 第 7 页那个公式**：

$$
L^{CLIP}(\theta) = \mathbb{E}_t\!\left[\min\!\left(r_t(\theta) A_t,\;\text{clip}(r_t(\theta), 1{-}\epsilon, 1{+}\epsilon) A_t\right)\right]
$$

**`eos_mask` 这个名字误导**：实际上是 `response_mask`（valid token 位置），但 Search-R1 用 info_mask 替换后，**这个参数现在标记"哪些 token 参与 loss"** —— retrieved-token loss masking 在 PPO loss 层的最终落点。

### 10.4 `kl_penalty` —— 四种 KL 估计

```python
def kl_penalty(logprob, ref_logprob, kl_penalty):
    if kl_penalty == "kl":
        return logprob - ref_logprob
    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()
    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()
    if kl_penalty == 'low_var_kl':  # ★ Schulman 的 unbiased low-variance estimator
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)
```

**`low_var_kl` 是 GRPO 默认用的**，[Schulman 推的](http://joschu.net/blog/kl-approx.html)：

$$
\mathrm{KL}(p \,\|\, q) \approx \mathbb{E}_q\!\left[e^{kl} - kl - 1\right]
$$

优势：无偏、始终 ≥ 0、低方差。

## 11. `dp_actor.py:update_policy` —— actor 训练（290 行）

```python
def update_policy(self, data: DataProto):
    self.actor_module.train()
    select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages']
    if self.config.state_masking:
        select_keys.append('loss_mask')
    if self.config.use_kl_loss:
        select_keys.append('ref_log_prob')
    
    dataloader = batch.split(self.config.ppo_mini_batch_size)
    for batch_idx, data in enumerate(dataloader):
        micro_batches = mini_batch.split(self.config.ppo_micro_batch_size)
        self.actor_optimizer.zero_grad()
        
        for data in micro_batches:
            response_mask = attention_mask[:, -response_length:]
            if self.config.state_masking:
                response_mask = data['loss_mask']  # ★ 用 loss_mask 覆盖 ★
            
            entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature)
            
            # ★ PPO clip loss
            pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(
                old_log_prob=old_log_prob, log_prob=log_prob, advantages=advantages,
                eos_mask=response_mask, cliprange=self.config.clip_ratio)
            
            entropy_loss = verl_F.masked_mean(entropy, response_mask)
            policy_loss = pg_loss - entropy_loss * self.config.entropy_coeff
            
            # ★ GRPO 风格的 KL loss（如果开了）
            if self.config.use_kl_loss:
                kld = core_algos.kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                kl_loss = masked_mean(kld, response_mask)
                policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
            
            loss = policy_loss / self.gradient_accumulation
            loss.backward()
        
        grad_norm = self._optimizer_step()
```

**梯度真正流动的地方**。`response_mask = data['loss_mask']` 就是 retrieved-token loss masking 的**最终落地**。被 mask 的 token：
- PPO clip loss 的 `eos_mask` 是它 → 不参与 PG 梯度
- Entropy 的 `mask` 是它 → 不参与 entropy regularization
- KL loss 的 `mask` 是它 → 不参与 KL 约束

### `compute_log_prob` —— "重算 old_log_prob"

```python
def compute_log_prob(self, data: DataProto) -> torch.Tensor:
    self.actor_module.eval()
    for micro_batch in micro_batches:
        with torch.no_grad():
            _, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature)
    return log_probs
```

**这一步初看冗余**：vLLM rollout 时已经返回了 logprobs，为什么还要重算？

**两个原因**：
1. **vLLM 的 sampling logprob 跟 FSDP forward 的 logprob 可能不完全数值一致**（不同 kernel、batching、fp16/bf16）。如果直接用 vLLM logprob 当 `old_log_prob`，PPO 的 importance ratio 在 step 0 就偏离 1，立刻不稳
2. **Search-R1 的 rollout 是 multi-turn 的**，vLLM 看到的 input 跟 FSDP forward 看到的（含 retrieved info）不一样

所以 `actor_rollout_wg.compute_log_prob(...)` 是把**vLLM 生成的完整序列重新过一遍 FSDP forward**。**开销不小** —— 跟训练 forward 同量级。

## 12. `dp_critic.py` —— critic / value function（204 行）

```python
class DataParallelPPOCritic(BasePPOCritic):
    def _forward_micro_batch(self, micro_batch):
        # 同 actor 的 forward，但模型是带 1-dim value head 的版本
        output = self.critic_module(input_ids=input_ids, ...)
        values = output.logits[:, -response_length - 1:-1].squeeze(-1)
        return values
    
    def compute_values(self, data):
        """rollout 完之后 trainer 调这个：得到当前 critic 对每个位置的 value 估计"""
        for micro_batch in micro_batches:
            with torch.no_grad():
                values = self._forward_micro_batch(micro_batch)
        return values
    
    def update_critic(self, data):
        """每个 PPO step 更新 critic（GRPO 不调用）"""
        for data in dataloader:
            for micro in micro_batches:
                vpreds = self._forward_micro_batch(micro)
                vf_loss, vf_clipfrac = core_algos.compute_value_loss(
                    vpreds=vpreds, values=values, returns=returns, eos_mask=eos_mask, cliprange_value=self.config.cliprange_value)
                loss = vf_loss / self.gradient_accumulation
                loss.backward()
```

**Critic 是独立的模型**（同 backbone + 1-dim value head），FSDP 分布式训练。
- `compute_values`：rollout 完后给每 token 估计 value（GAE 用）
- `update_critic`：用 `returns` 当 target 训 value head

**注意 critic 不用 info_mask**：它要给每 token 预测 value，**包括 retrieved info 位置**（GAE 反向需要 next-token value）。

PPO 总内存压力 30% 是 critic —— 这是 GRPO 省内存的原因。

## 13. `vllm_rollout.py` —— rollout 引擎（226 行）

```python
class vLLMRollout(BaseRollout):
    def __init__(self, actor_module, config, ...):
        self.inference_engine = LLM(actor_module, ...,
                                    gpu_memory_utilization=config.gpu_memory_utilization,
                                    max_model_len=config.prompt_length + config.response_length)
        self.inference_engine.offload_model_weights()    # ← 默认 offload
    
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs):
        idx = prompts.batch['input_ids']     # (bs, prompt_length)
        
        # ★ 把每条 prompt 从左 padding 转成 List[int]（去 pad）
        idx_list = []
        for i in range(batch_size):
            idx_list.append(_pre_process_inputs(self.pad_token_id, idx[i]))
        
        # ★ 调 vLLM generate
        output = self.inference_engine.generate(
            prompts=None, sampling_params=self.sampling_params,
            prompt_token_ids=idx_list, use_tqdm=False)
        
        response = output[0]; log_probs = output[1]
        # 不够长就右 padding
        if response.shape[1] < self.config.response_length:
            response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
        
        # 拼出完整 sequence: prompt (left-padded) + response (right-padded)
        seq = torch.cat([idx, response], dim=-1)
        # ... build attention_mask 和 position_ids
        return DataProto(batch=batch)
```

**核心要点**：

1. **vLLM 作为 inference engine** 调用：`LLM` 类是 veRL fork 的 vLLM v0.5.4
2. **`offload_model_weights()`** —— 默认把 vLLM 内部权重 offload 到 CPU，**等 rollout 时再 load 回 GPU**。Hybrid engine 关键：actor 和 vLLM 共享同一份权重，靠 offload 错峰
3. **左 padding → packed list**：vLLM 不接受左 padding tensor，要把每条 prompt 转 `List[int]` 形式（`_pre_process_inputs`），去掉 pad token
4. **Prompt 左 padding，response 右 padding**

**这一层不知道 search 的存在**。Search-R1 的 multi-turn 完全在 `generation.py:run_llm_loop` 里编排。**vLLM 本身是 single-turn 的**。

### Hybrid engine —— FSDP ↔ vLLM 怎么共享权重

`fsdp_workers.py:_build_rollout` 和 `sharding_manager/fsdp_vllm.py` 实现：

```
训练态：actor 参数在 GPU（FSDP 分片）+ optimizer state 也在 GPU（或 offload CPU）
            │
            │  rollout 时切换：
            ▼
推理态：把 FSDP 分片合并 → reshape 到 vLLM 的 TP 切分 → 拷给 vLLM weight pointers
            │
            │  rollout 完成后：
            ▼
训练态：reshape 回 FSDP → 继续 PPO 更新
```

逻辑在 `verl/workers/sharding_manager/fsdp_vllm.py`。**最复杂 veRL 模块**，但用户看不到。

为什么不直接两套 weight？8B 模型 fp16 = 16GB。Actor + vLLM = 32GB，再 + critic = 48GB。单 H100 80GB 还剩 32GB 给 activation + KV cache + optimizer，远远不够。**Hybrid engine 同一份权重在 train 和 rollout 间切换是必须的**。

## 14. `fsdp_workers.py` —— 编排层（1054 行）

`fsdp_workers.py` 把 actor/critic/rollout/ref 包装成 Ray actor。1054 行但大多是 boilerplate。最重要的方法：

```python
class ActorRolloutRefWorker(Worker):
    """一个 worker 同时承担 actor 训练 + vLLM rollout + reference policy forward"""
    
    def init_model(self):
        # 1. 建 FSDP-wrapped actor module + optimizer
        # 2. DataParallelPPOActor 封装训练逻辑
        # 3. vLLMRollout（共享权重，hybrid engine）
        # 4. Reference policy（同 model，权重 frozen）
    
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp, ...)
        metrics = self.actor.update_policy(data=data)
        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, ...)
    
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        with self.rollout_sharding_manager:
            output = self.rollout.generate_sequences(prompts=prompts)
        if self._is_actor and recompute_log_prob:
            with self.ulysses_sharding_manager:
                old_log_probs = self.actor.compute_log_prob(data=output)
                output.batch['old_log_probs'] = old_log_probs
        return output
```

**工程点**：

1. **`load_fsdp_param_and_grad` / `offload_fsdp_param_and_grad`**：每次 actor update 前把参数和 grad 从 CPU load 到 GPU，update 完再 offload。**单机 8 卡跑 70B 的关键**

2. **`@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)`**：veRL 自己的装饰器。`DP_COMPUTE_PROTO` 表示 "把 DataProto 按 DP 维度拆分到各 worker，每个 worker 算自己那份，结果合并"

3. **三个 `sharding_manager` 上下文**：
   - `self.rollout_sharding_manager` —— 切到 vLLM 视角
   - `self.ulysses_sharding_manager` —— 切到 Ulysses sequence parallel 视角
   - 默认 —— FSDP 视角

---

# 端到端数据流

把所有看过的代码串成**一次 step 真实执行序列**：

```
ray_trainer.py:fit() 主循环
│
├─ 1. dataloader 拿一个 batch
│    └─ rl_dataset.py:__getitem__ → tokenize + 左 padding
│
├─ 2. ROLLOUT (search 模式)
│    ├─ generation_manager.run_llm_loop(gen_batch, initial_input_ids)
│    │  │
│    │  ├─ for step in range(max_turns):
│    │  │  ├─ actor_rollout_wg.generate_sequences(rollings_active)
│    │  │  │  └─ fsdp_workers.py:generate_sequences
│    │  │  │     ├─ rollout_sharding_manager（切到 vLLM 视角）
│    │  │  │     └─ vllm_rollout.py:generate_sequences
│    │  │  │        └─ self.inference_engine.generate(...)
│    │  │  │
│    │  │  ├─ _postprocess_responses → 截断在 </search>/</answer>
│    │  │  ├─ execute_predictions:
│    │  │  │  ├─ postprocess_predictions: regex 提取 action/content
│    │  │  │  ├─ batch_search(queries) → POST http://127.0.0.1:8000/retrieve
│    │  │  │  │  └─ retrieval_server.py:retrieve_endpoint
│    │  │  │  │     └─ DenseRetriever.batch_search:
│    │  │  │  │        ├─ Encoder.encode(queries) → e5-base-v2
│    │  │  │  │        ├─ faiss.search(query_emb, k=3)
│    │  │  │  │        └─ load_docs(corpus, idxs)
│    │  │  │  └─ 返回 search_results
│    │  │  │
│    │  │  └─ _info_masked_concatenate_with_padding
│    │  │     ★ 关键 ★ 同步维护 (real_tokens, mask_version)
│    │  │
│    │  ├─ 最后一轮：do_search=False，强制出 answer
│    │  └─ _compose_final_output → 暴露 info_mask 给 trainer
│    │
│    └─ actor_rollout_wg.compute_log_prob(final_gen_batch_output)
│       └─ ★ FSDP forward 重算 old_log_prob ★
│
├─ 3. REFERENCE log-prob
│    └─ ref_policy_wg.compute_ref_log_prob(batch)
│
├─ 4. CRITIC values (PPO only)
│    └─ critic_wg.compute_values(batch)
│
├─ 5. REWARD
│    └─ reward_fn(batch) = RewardManager.__call__
│       └─ for each item: qa_em.compute_score_em
│          ├─ extract_solution: regex `<answer>...</answer>` (要 ≥2 个 match)
│          ├─ em_check: normalize + 字符串相等
│          └─ score ∈ {0, 1}
│       └─ ★ reward_tensor[i, valid_response_length - 1] = score ★ (sparse!)
│
├─ 6. KL PENALTY (PPO 风格)
│    └─ apply_kl_penalty(batch, kl_ctrl, kl_penalty='kl')
│       ├─ kld = core_algos.kl_penalty(old_log_probs, ref_log_prob, 'kl')
│       ├─ kld = kld * info_mask  ← retrieved 不算 KL
│       └─ token_level_rewards = token_level_scores - β * kld
│
├─ 7. ADVANTAGE
│    └─ compute_advantage(batch, adv_estimator='gae')
│       └─ core_algos.compute_gae_advantage_return
│
├─ 8. CRITIC UPDATE
│    └─ critic_wg.update_critic(batch)
│
├─ 9. LOSS MASK
│    └─ _create_loss_mask(batch, metrics)
│       ├─ loss_mask = batch['info_mask'][:, -response_length:]
│       └─ batch['loss_mask'] = loss_mask
│
├─ 10. ACTOR UPDATE
│     └─ actor_rollout_wg.update_actor(batch)
│        └─ dp_actor.py:update_policy
│           ├─ for each micro_batch:
│           │  ├─ response_mask = data['loss_mask']  ★ retrieved-token masking 实际生效
│           │  ├─ entropy, log_prob = _forward_micro_batch
│           │  ├─ pg_loss = compute_policy_loss(old_log_prob, log_prob, advantages, response_mask, ...)
│           │  ├─ entropy_loss = masked_mean(entropy, response_mask)
│           │  └─ policy_loss = pg_loss - entropy_coeff * entropy_loss
│           └─ _optimizer_step
│
└─ 11. VALIDATE (每 N steps)
      └─ _validate() → 重新 rollout 一遍 val set，算 EM → log 到 wandb
```

**Search-R1 的"贡献"在这条链上的位置**：步骤 2 整个（替换 rollout）、步骤 6 里 `info_mask` 替换 attention_mask、步骤 9（loss_mask 转换）、步骤 10 里那一行 `response_mask = data['loss_mask']`。

**所有其它步骤都是标准 veRL PPO**。这就是为什么我说 "agentic RL = standard PPO + multi-turn rollout + 一个 mask"。

---

# Search-R1 怎么映射到 veRL 的设计

文件级归属：

| 归属 | 文件 |
| ---- | ---- |
| **Search-R1 直接贡献** | `search_r1/llm_agent/generation.py` (469 行)<br>`search_r1/llm_agent/tensor_helper.py` (74)<br>`search_r1/search/*.py` (各种 retriever 1200+ 行) |
| **Search-R1 给 veRL 的补丁** | `verl/trainer/ppo/ray_trainer.py:_create_loss_mask` (15 行)<br>`verl/workers/actor/dp_actor.py` (`state_masking` 配置 2-3 行)<br>`verl/utils/reward_score/qa_em.py` (139 行，部分继承自 veRL stub) |
| **标准 veRL PPO/GRPO** | `verl/trainer/ppo/core_algos.py` (274)<br>`verl/workers/critic/dp_critic.py` (204)<br>`verl/workers/rollout/vllm_rollout/*.py` (226)<br>`verl/workers/fsdp_workers.py` (1054)<br>`verl/third_party/vllm/*` (~4000) |

Search-R1 总贡献：**~600-700 行论文特有代码，加 ray_trainer.py 的 15 行补丁**。其它全是 veRL。

这是**干净的模块化架构 pattern**。下一篇 agentic-RL 论文 / 项目可以：
- 写自己的 `<their_specific_thing>.py`，参考 `generation.py`
- 给 rollout 输出加 `info_mask`（或等价物）
- 加 `state_masking=true` 一行配置启用 mask 感知 loss
- 其它继承自 veRL

这正是 [[polar|Polar]] 做的（任意 harness，不只 search）—— `prefix_merging` 是 Search-R1 retrieved-token masking 的工程化泛化。

---

# 上手实验 —— 改什么先

按修改难度排序：

### 难度 1：换 reward function

最简单的 ablation。改 `verl/utils/reward_score/qa_em.py`：

```python
# 鼓励多搜索
def compute_score_with_search_bonus(solution_str, ground_truth, format_score=0., score=1.):
    answer = extract_solution(solution_str)
    em = em_check(answer, ground_truth['target']) if answer else 0
    n_searches = len(re.findall(r'<search>', solution_str))
    return em + 0.05 * min(n_searches, 3)

# 抑制过度搜索
def compute_score_with_search_penalty(solution_str, ground_truth):
    em = em_check(...)
    n_searches = len(re.findall(r'<search>', solution_str))
    return em - 0.1 * max(n_searches - 2, 0)
```

观察这些 reward 怎么改变模型的搜索频率和 EM。

### 难度 2：换检索器 / corpus

`retrieval_launch.sh` 改 `--retriever_name` 和 `--index_path`。E5 → BM25 → SPLADE → BGE，看模型怎么适应。

### 难度 3：加新工具

在 `execute_predictions` 加新 action：

```python
elif action == 'calculate':
    result = safe_eval(content)
    next_obs.append(f'\n\n<information>Calculator result: {result}</information>\n\n')
    dones.append(0)
    valid_action.append(1)
    is_search.append(0)
```

`postprocess_predictions` 的 regex 改：`r'<(search|answer|calculate)>(.*?)</\1>'`。Prompt 也要加说明。

GSM8K 上跑看模型能不能学会**何时搜索、何时计算**。

### 难度 4：用 process reward 代替 outcome reward

最难但最有价值。

### 难度 5：把 `state_masking=true` 关掉

这是论文 Table 4 那个消融。`actor_rollout_ref.actor.state_masking=false`。**亲手复现这个对比是真正理解 retrieved-token loss masking 必要性的唯一方式**。

---

# 跟更广 agentic-RL stack 的连接

读完 Search-R1 你会看到 [[prorl-agent|ProRL Agent]] 和 [[polar|Polar]] 在做**同样的事**，只是更工程化：

| 组件 | Search-R1（学术 repo） | [[prorl-agent\|ProRL Agent]] | [[polar\|Polar]] |
| ---- | --------------------- | -------------------------- | --------------- |
| Rollout loop | `generation.py:run_llm_loop`（469 行） | HTTP `POST /process` + AgentHandler ABC | LLM-API proxy + 轨迹重建 |
| Environment | FastAPI search server | rootless Apptainer + AgentHandler | rootless Apptainer + 未修改 harness |
| Mask 机制 | `info_mask` → `loss_mask` | token-in/out wire 协议 | `loss_mask`：sampled=1，canonical interstitial=0 |
| 多轮处理 | 单循环 `for step in range(max_turns)` | INIT/RUN/EVAL 异步 pipeline | INIT/RUN/POSTRUN + READY buffer |
| Trainer 集成 | 直接 in-process（veRL fork） | HTTP，trainer-agnostic | HTTP，trainer-agnostic |

**Search-R1 是这一套范式的最早最干净的实现**。看完它你会发现 Polar 的 `loss_mask` 描述跟 Search-R1 的 `info_mask` 是**同一个东西的不同版本** —— Polar 是把这个概念泛化到任意 harness 任意工具的工程化版本。

---

# 接下来读什么

读完代码后建议：

1. **跑一次复现 Table 2 的训练**（Qwen2.5-3B，单 8×A100/H100 节点，~2 天）
2. **跑 `state_masking=false` 消融**，亲身感受 mask 的重要性
3. **读 [[prorl-agent]] 和 [[polar]] 论文** —— 看相同 pattern 怎么生产化
4. **挑一个难度 3+ 的修改**跑实验。读代码必要但不充分；跑实验才能真正巩固理解
5. **读后续工作**（ReSearch、ToolRL、R1-Searcher、DeepResearcher）看 Search-R1 pattern 怎么扩展

## 相关阅读

- [[search-r1]] —— 论文精读（什么 / 为什么 / 实验）
- [[agentic-rl-foundations]] —— 完整阅读路径的入门 hub
- [[prorl-agent]] —— 服务化 agentic RL 基础设施
- [[polar]] —— 当前 SOTA rollout 基础
- [[nemo-gym]] —— NVIDIA 的环境 catalog 框架
- [[grpo]] —— Search-R1 的另一个 RL 算法
- [[ppo-for-llm]] —— PPO-for-LLM 基础
- [[rl-training-frameworks]] —— veRL / OpenRLHF / TRL 图景
- [[tool-use-rl]] —— 更广的 tool-use RL 家族
