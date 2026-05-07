---
title: "ProRL Agent: Rollout-as-a-Service for Multi-Turn Agentic RL"
category: agentic-rl
tags: [prorl-agent, nvidia, rollout-as-a-service, agentic-rl, infrastructure, openhands, swe-bench, paper-review]
created: 2026-04-13
updated: 2026-05-07
status: mature
paper: arXiv:2603.18815
code: https://github.com/NVIDIA-NeMo/ProRL-Agent-Server
---

# ProRL Agent: Rollout-as-a-Service for Multi-Turn Agentic RL

> [!info] Paper metadata
> - **Paper**: [arXiv:2603.18815](https://arxiv.org/abs/2603.18815) — NVIDIA, March 2026
> - **Code**: [NVIDIA-NeMo/ProRL-Agent-Server](https://github.com/NVIDIA-NeMo/ProRL-Agent-Server) (branch `stable`, Apache-2.0)
> - **Authors**: Hao Zhang, Mingjie Liu, Shaokun Zhang, Songyang Han, Jian Hu, Zhenghui Jin, Yuchi Zhang, Shizhe Diao, Ximing Lu, Binfeng Xu, Zhiding Yu, Jan Kautz, Yi Dong

> [!abstract]+ TL;DR
> ProRL Agent makes **rollout** — running a multi-turn agent in an environment to produce a trajectory — into a standalone HTTP service that any RL trainer can consume. The server (FastAPI parent + multiprocessing child holding three async worker pools) handles container lifecycle, multi-turn agent loops, tool execution, and reward computation; the trainer only sends a `POST /process` request and receives `(token_ids, logprobs, reward)` tuples back. On SWE-Bench Verified the resulting RL training loop lifts Qwen3-4B/8B/14B Pass@1 by **6–8 percentage points** and roughly **doubles** SkyRL-Agent's reported 8B score.

---

## Background: why agentic-RL infrastructure is its own problem

Training an LLM agent with RL involves two workloads with fundamentally different shapes:

|                | Rollout                                                                  | Training                                       |
| -------------- | ------------------------------------------------------------------------ | ---------------------------------------------- |
| Resource       | I/O-bound: sandbox spin-up, multi-turn loops, async tool execution        | GPU-bound: forward/backward, gradient sync     |
| Time scale     | Seconds to minutes per episode (variance dominated by tool I/O)            | ~10s of ms per step                            |
| Failure mode   | Container crashes, network timeouts, tool errors                          | OOM, NaN, NCCL hangs                           |
| Right hardware | Many CPU + sandbox nodes                                                 | Few large 8×GPU nodes                          |

Existing agentic-RL frameworks — SkyRL-Agent, VeRL-Tool, Agent Lightning, rLLM, GEM — keep both workloads inside the **trainer's process**: rollout coroutines, in-memory environments, or an embedded agent loop with offloaded tools. That tight coupling causes three failures: (1) bursty rollouts disrupt training cache locality and starve inference time slots; (2) switching trainers (e.g. veRL → NeMo-RL) forces re-porting agent loops and sandboxes; (3) any improvement to the agent — a new tool, a new memory module — must be pushed through the trainer.

ProRL Agent's response is the obvious move from web-systems land: **separate concerns into independent services with a stable HTTP contract.** That is the same argument that produced microservices in 2014. The non-trivial claim is that this pattern works even for the latency-sensitive RL inner loop.

| Framework         | Decoupled train/rollout | Rootless sandbox | Scaffold-independent |
| ----------------- | ----------------------- | ---------------- | -------------------- |
| SkyRL-Agent       | ✗                       | ✗                | ✓                    |
| VeRL-Tool         | ✗                       | ✗                | ✓                    |
| Agent Lightning   | ✗                       | ✗                | ✗                    |
| rLLM              | ✗                       | ✗                | ✓                    |
| GEM               | ✗                       | ✗                | ✓                    |
| **ProRL Agent**   | **✓**                   | **✓**            | **✓**                |

The "rootless sandbox" column is the deployment-realism contribution that lets the first column actually run on shared HPC clusters where research happens.

---

## The key idea: rollout-as-a-service

> [!quote] The contribution in one sentence
> Treat agentic rollout as an HTTP service with a typed `POST /process` endpoint that produces `(token_ids, logprobs, reward)` tuples consumable by any RL trainer.

Three sub-ideas make this work:

- **Token-in / token-out wire format** — trainer and server share the canonical token sequence; no re-tokenization drift across turns.
- **Three-stage async pipeline** (INIT → RUN → EVAL) with independent worker pools, so jobs at different stages don't block each other.
- **Rootless HPC-compatible sandbox** (Singularity + unprivileged user + per-job loopback IP), so the system actually deploys on the Slurm clusters where research happens.

Remove any one: the system becomes off-policy unstable (no token-in/out), throughput-bound (no pipeline), or undeployable (no rootless sandbox).

---

## How it works

### The three-component architecture

```
┌────────────┐  HTTP  ┌─────────────────┐  exec  ┌──────────────────┐
│ RL Trainer │◄──────►│ ProRL Agent     │◄──────►│ Sandbox Envs     │
│ (verl/NeMo)│ /proc  │ Server (FastAPI)│        │ (Singularity .sif)│
└────────────┘        └─────────────────┘        └──────────────────┘
                              │
                              ▼
                       ┌──────────────┐
                       │ vLLM backends│  (load-balanced via min-heap)
                       └──────────────┘
```

### HTTP API

The wire surface is a handful of Pydantic-validated endpoints (`openhands/nvidia/utils.py`):

```python
class ProcessRequest(BaseModel):
    instance: dict[str, Any]          # task definition (data_source, instance_id, …)
    sampling_params: dict[str, Any]   # model, temperature, top_p, max_tokens, …
    job_id: str | None = None         # client-assigned id (else server hashes one)

class CancelRequest(BaseModel):
    job_id: str

class LLMServerRequest(BaseModel):
    address: str                      # vLLM endpoint URL to register
```

| Endpoint                | Body                | Purpose                                                                            |
| ----------------------- | ------------------- | ---------------------------------------------------------------------------------- |
| `POST /process`         | `ProcessRequest`    | Submit a rollout; blocks until `(token_ids, logprobs, reward, timing)` is ready    |
| `POST /cancel`          | `CancelRequest`     | Abort an in-flight job by id                                                        |
| `POST /add_llm_server`  | `LLMServerRequest`  | Register a vLLM endpoint into the load-balancer min-heap                            |
| `POST /clear_llm_server`| —                   | Flush all backends (used at checkpoint update)                                      |
| `GET /status`           | —                   | Queue depths, server-running flag                                                   |
| `POST /start /stop`     | —                   | Lifecycle control of the child server worker                                        |

### Three-stage async pipeline

`OpenHandsServer.__init__` (`openhands/nvidia/async_server.py`) wires three queues, one per stage:

```python
self.init_queue: queue.Queue[str] = queue.Queue()
self.run_queue:  queue.Queue[str] = queue.Queue()
self.evaluate_queue: queue.Queue[str] = queue.Queue()

self._active_init_jobs: set[str] = set()
self._active_run_jobs:  set[str] = set()
self._active_eval_jobs: set[str] = set()

# Three independent locks, not one — coarse locking would serialize stages.
self._state_lock        = threading.RLock()
self._job_details_lock  = threading.RLock()
self._address_lock      = threading.RLock()

# Min-heap of [in_flight_count, address] entries for vLLM load balancing.
self.weighted_addresses = [[0, addr] for addr in llm_server_addresses]
heapq.heapify(self.weighted_addresses)
```

`start()` then submits three pools of workers, all backed by **one** `ThreadPoolExecutor`:

```python
self._executor = ThreadPoolExecutor(
    max_workers=self.max_init_workers + self.max_run_workers + self.max_eval_workers
)
for i in range(self.max_init_workers):
    self._executor.submit(self._run_worker_in_thread, i, JobType.INIT)
for i in range(self.max_run_workers):
    self._executor.submit(self._run_worker_in_thread, i, JobType.RUN)
for i in range(self.max_eval_workers):
    self._executor.submit(self._run_worker_in_thread, i, JobType.EVAL)
```

Each worker runs a unified `_worker(self, wid, job_type)` coroutine that pops from its assigned queue and dispatches into the registered handler:

```python
with phase_context(job_details.timer, function_type):   # only this counts toward timeout
    if job_type == JobType.INIT:
        runtime, metadata, config = await run_with_timeout_awareness(
            timer, init_coro, job_details
        )
        job_details.runtime = runtime
        self.run_queue.put(job_id)              # advance to RUN stage

    elif job_type == JobType.RUN:
        run_results = await run_with_timeout_awareness(timer, run_coro, job_details)
        job_details.run_results = run_results
        self._cleanup_job_runtime(job_details.runtime, job_id)  # free container before EVAL
        self.evaluate_queue.put(job_id)         # advance to EVAL stage

    elif job_type == JobType.EVAL:
        eval_report = await run_with_timeout_awareness(timer, eval_coro, job_details)
        job_details.eval_results = eval_report.get('report', eval_report)
        job_details.event.set()                  # wake the blocking /process caller
```

> [!note] Two design choices visible only in code
> - **Container is freed at the end of RUN, not EVAL.** EVAL often runs against the produced patch in a separate test sandbox; keeping the rollout container alive through EVAL would waste memory.
> - **Inter-stage time is "others" phase.** `phase_context` is a context manager that switches the timer between active phases; queue waiting between phases is automatically counted as `others` and excluded from the timeout budget. Without this, a worker shortage during a load spike would silently fire false-negative timeouts and corrupt the training signal.

### `/process` — the full job lifecycle

```python
def process(self, instance, sampling_params, job_id=None, timeout=300.0):
    # 1. Pick handler by data_source field (keyed dispatch).
    dataset_type = instance.get('data_source', 'swebench')
    if not is_registered_handler(dataset_type, reasoning=is_reasoning_task):
        raise FunctionNotRegisteredError(...)

    # 2. Build job state with a stage-aware timer.
    job_details = JobDetails(
        job_id=job_id or self.get_unique_id(instance),
        instance=instance, is_reasoning_task=is_reasoning_task,
    )
    job_details.timer = PausableTimer(timeout=timeout); job_details.timer.start()

    # 3. Allocate a vLLM backend via the min-heap (see load balancing).
    job_details.llm_config = self.create_llm_config(sampling_params)
    job_details.event = threading.Event()
    self._job_details[job_id] = job_details

    # 4. Enqueue at INIT; block on the completion Event.
    self.init_queue.put(job_id)
    job_details.event.wait()

    # 5. Aggregate via the handler's final_result + timing breakdown.
    result = get_registered_functions('final_result', dataset_type, ...)(job_details)
    result['timing'] = job_details.timer.get_timing_info()
    return result
```

`PausableTimer.get_timing_info()` returns `{init, run, eval, others, total}` so the trainer can attribute latency.

### AgentHandler: the plugin interface

Task-specific logic plugs in via an abstract base class (`openhands/nvidia/registry.py`):

```python
class AgentHandler(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...                                    # dispatch key

    @abstractmethod
    async def init(self, job_details: JobDetails, sid: str | None = None
                  ) -> tuple[Runtime, EvalMetadata, OpenHandsConfig]: ...

    @abstractmethod
    async def run(self, job_details: JobDetails, sid: str | None = None
                 ) -> dict[str, object]: ...                      # trajectory + artifacts

    @abstractmethod
    async def eval(self, job_details: JobDetails, sid: str | None = None,
                   allow_skip: bool = True, reward: Optional[Reward] = None
                  ) -> dict[str, Any]: ...                        # reward signal

    @abstractmethod
    def init_exception(self, job_details, exc) -> dict[str, Any]: ...
    @abstractmethod
    def run_exception(self,  job_details, exc) -> dict[str, Any]: ...
    @abstractmethod
    def eval_exception(self, job_details, exc) -> dict[str, Any]: ...

    @abstractmethod
    def final_result(self, job_details: JobDetails) -> dict[str, Any]: ...
```

The shared mutable state for one rollout is held in a single dataclass — every handler method takes it and mutates it in place:

```python
@dataclass
class JobDetails:
    job_id: str | None = None
    instance: dict | None = None
    agent_config: dict = field(default_factory=lambda: _DEFAULT_AGENT_CONFIG.copy())
    llm_config: LLMConfig | None = None
    runtime: Runtime | None = None
    metadata: EvalMetadata | None = None
    config: OpenHandsConfig | None = None
    agent: CodeActAgent | None = None        # for intermediate inspection
    controller: AgentController | None = None
    run_results: dict | None = None
    eval_results: dict | None = None
    results: dict | None = None
    event: threading.Event | None = None
    timeout_error: bool = False
    timer: Optional['PausableTimer'] = None
    current_task: Optional[asyncio.Task] = None
    is_reasoning_task: bool = False
```

Registration is a flat dict keyed by handler `.name`, with seven entries per handler (one per abstract method):

```python
def register_agent_handler(handler: AgentHandler, reasoning=False):
    registries['init'][handler.name]            = handler.init
    registries['run'][handler.name]             = handler.run
    registries['eval'][handler.name]            = handler.eval
    registries['init_exception'][handler.name]  = handler.init_exception
    registries['run_exception'][handler.name]   = handler.run_exception
    registries['eval_exception'][handler.name]  = handler.eval_exception
    registries['final_result'][handler.name]    = handler.final_result
    name_mapping[handler.name] = handler.name
```

A separate `_registries_reasoning` registry exists for reasoning-style tasks (math, STEM), so a single deployment can serve both code-agent and reasoning-agent workloads. Repository ships with handlers for SWE-Gym, R2E-Gym, SWE-Bench, math, STEM, and code.

### Token-in / token-out (the part most papers get wrong)

Multi-turn RL has a silent killer: **re-tokenization drift**. At turn $t$ the LLM samples reply IDs using its chat template and spacing rules. If the client logs only the decoded text and rebuilds the full history for turn $t + 1$, tiny format changes — system/tool prefixes, spaces, XML function-call wrappers — re-tokenize the entire conversation differently. Actor and reference no longer share the same token boundaries; per-token logprobs misalign; KL/entropy can spike to NaN; PPO/GRPO updates collapse.

ProRL fixes this by making token IDs the canonical representation throughout training. Each message in the trajectory carries four extra fields the trainer consumes directly (`openhands/nvidia/utils.py:process_messages_from_agent_state`):

```python
new_message['token_ids']          = output_ids                            # what was actually generated
new_message['repetition_penalty'] = ngram_repetition_reward(output_ids).tolist()
new_message['input_ids']          = input_ids                             # context ids at sample time
new_message['logprobs']           = logprobs                              # per-token logprobs at sample time
```

If `output_ids` is missing (e.g. for synthesized observation messages), the system tokenizes them through a custom Qwen3 chat template that **preserves thinking content in the content field** — Qwen3's default template strips `<think>` blocks under some conditions, which would silently lose the reasoning trace:

```python
new_message['token_ids'] = convert_messages_to_tokens(
    [new_message],
    tokenizer,
    chat_template=chat_template,           # qwen3_chat_template, custom variant
    add_generation_prompt=True,
    enable_thinking=enable_thinking,
    tools=tools,
)[0]
```

A small but practical reward-shaping detail: `ngram_repetition_reward(output_ids, ngram_size=64, penalty=-0.001)` applies a per-token penalty at the start position of every repeated 64-gram. This rides along with `token_ids` so the trainer can apply an off-policy-safe penalty against degenerate looping without a separate evaluation pass.

There's also a corner case the code handles explicitly: when a Qwen3 model emits `<think>\n` followed immediately by tool calls but never closes with `</think>`, the HuggingFace tokenizer breaks. The agent post-processes this by compressing the tool-call content back into the `content` string and appending a synthetic `</think>` so the canonical token IDs stay parseable. This is the kind of detail that tells you the authors actually trained models — the same bug shows up in other [[grpo|GRPO]] divergence reports as a reason for instability.

### LLM backend load balancing

The min-heap is two lines plus the lock. From `OpenHandsServer.create_llm_config`:

```python
def create_llm_config(self, sampling_params):
    with self._address_lock:
        if not self.weighted_addresses:
            raise ValueError('No LLM server addresses added')
        address = self.weighted_addresses[0][1]      # peek the lightest-loaded backend
        self.weighted_addresses[0][0] += 1           # increment its in-flight count
        heapq.heapreplace(self.weighted_addresses,   # re-heapify with updated weight
                          self.weighted_addresses[0])
    return LLMConfig(base_url=address, **sampling_params)
```

A whole rollout's call sequence sticks to the same backend (because `LLMConfig` is built once per `/process` and reused throughout the multi-turn loop) so prefix-cache hits stay high (see [[kv-cache-optimization]]). Heavy-loaded backends fall back in priority on the next pop. Simpler than power-of-two-choices, and works because per-call cost variance is dominated by sequence length, not server-side slowdown.

`add_llm_server_address` and `clear_llm_server_addresses` (called by `POST /add_llm_server` and `POST /clear_llm_server`) hot-rotate backends — useful when the trainer publishes a new checkpoint to vLLM and wants to drain the old one.

### Rootless HPC sandbox (SingularityRuntime)

The hard constraint: the system must run as an unprivileged Slurm user — no Docker daemon, no root. Solutions:

- **Singularity Image Files (`.sif`)** — single-file, portable container images, friendly to shared filesystems.
- `--fakeroot` for in-container package install; `--network none` for external isolation.
- **Per-job loopback IP** in `127.x.x.x` via a thread-safe allocator → eliminates port conflicts at scale.
- Each container is a child process in its own session; SIGTERM → SIGKILL escalation for cleanup.
- `SingularityRuntimeBuilder` constructs images from Jinja2 templates with three caching tiers: *Scratch* (full rebuild), *Versioned* (reuse if base image/framework unchanged), *Lock* (reuse if dependency lockfile identical).

The runtime is selected by config (`example.config.toml`):

```toml
[core]
runtime          = "singularity"
run_as_openhands = false             # don't expect an "openhands" user

[sandbox]
run_as_fakeroot       = true          # simulate root inside the .sif
base_container_image  = "ubuntu:24.04"
# isolate_network     = false        # turn on for stricter network sandbox

[llm]
api_key  = "xxx"
base_url = "http://localhost:8000/v1"
model    = "hosted_vllm/Qwen3-1.7B"
```

`utils.py` ships a `kill_all_singularity_jobs(exclude_pids)` helper that uses `pgrep -f apptainer` and `pgrep -f openhands` to clean up orphaned sandbox processes between training runs — rough but effective on a shared cluster.

### Tool latency: where the wall clock actually goes

At high concurrency, tool latency dominates over LLM inference. Three concrete optimizations:

| Tool    | Default approach              | ProRL approach                  | Why                                |
| ------- | ----------------------------- | ------------------------------- | ---------------------------------- |
| Bash    | tmux session routing          | `ptyprocess` direct PTY         | ~6× faster shell round-trip        |
| IPython | Jupyter gateway over network  | Direct in-process kernel API    | No network hop                     |
| IPC     | TCP loopback                  | Unix domain sockets             | Lower latency, no port management  |

These aren't novel research; they're the right engineering. The ablation later shows action time dropping from 0.78 s to 0.42 s when efficient bash is enabled.

### Async DAPO refilling

[[grpo|DAPO]] filters out "Zero-Variance Prompts" (uniform reward → zero gradient). The naive batch-by-batch implementation creates synchronous waste: roll out a batch, filter, roll out another. ProRL replaces this with three primitives:

1. **Continuous replenishment** — refill the queue immediately on depletion.
2. **Early termination** — stop active jobs once enough informative ones complete (`POST /cancel` with the job ids).
3. **Cross-iteration persistence** — unfinished jobs carry into the next training iteration.

This is a genuine systems-level extension to a published RL algorithm, not just deployment plumbing.

### Inside the server process (FastAPI parent + multiprocessing child)

Reading `scripts/start_server.py`: the "server" is actually a parent FastAPI process that forks a child `server_worker` process holding the `OpenHandsServer` instance. They communicate via three `multiprocessing.Queue`s — `request_queue`, `job_result_queue`, `control_response_queue`. The parent process holds an asyncio future map keyed by `job_id`; a background `_response_listener` thread drains the result queue and resolves the futures.

Why two layers:

- **Crash isolation** — a runaway rollout (e.g. container daemon hang, `apptainer` zombie) can be killed without taking down FastAPI.
- **Two-level concurrency** — the child runs both `multiprocessing` (for clean process trees per job; `os.setsid()` on entry, so SIGTERM to the child reliably kills its descendants) and a `ThreadPoolExecutor` sized as `max_init + max_run + max_eval + 30`. The `+30` is generous slack for `process_with_timeout` overhead — pragmatic, not principled.
- **Backpressure is implicit in the queue** — the parent never blocks on the child; if the child is busy, the parent's futures simply take longer to resolve. Cancellation propagates because `cancel_job` flips a flag and sets the per-job `threading.Event`, which any worker checks at the next phase boundary.

`process_with_timeout` is the bridge: the parent runs `asyncio.run(process_with_timeout(...))` in a thread for each job, where each call also gets its own inner `ThreadPoolExecutor` so blocking handler code doesn't starve the worker's event loop:

```python
async def process_with_timeout(server, instance, sampling_params, timeout,
                               thread_pool: ThreadPoolExecutor | None = None,
                               job_id: str | None = None):
    loop = asyncio.get_event_loop()
    future = loop.run_in_executor(
        thread_pool,
        lambda: server.process(instance, sampling_params, job_id, timeout),
    )
    try:
        return await future
    except asyncio.TimeoutError:
        await cleanup_timed_out_job(server, job_id)
        raise JobTimeoutError(...)
```

`cleanup_timed_out_job` flips `timeout_error`, sets the per-job `Event`, removes the job from all three active-set trackers, closes the `Runtime`, and frees the `JobDetails` slot — i.e. enforces the invariant that a timed-out job leaves no resources behind.

---

## Experiments

**Setup.** 32× NVIDIA H100. RL algorithm: DAPO ([[grpo|GRPO]] variant with Zero-Variance-Prompt filtering). Batch 32, mini-batch 8, 8 rollouts per instance, KL = $10^{-4}$, lr = $10^{-6}$. Trainer integration with ProRL framework, veRL, NVIDIA NeMo RL. Models: Qwen3 family.

### Main result — SWE-Bench Verified

293-instance SWE-Gym training subset:

| Model                       | Baseline (Pass@1)  | After RL    | Δ           |
| --------------------------- | -----------------: | ----------: | ----------: |
| Qwen3-4B-Instruct-2507      | 14.8 %             | 21.2 %      | +6.4 pp     |
| Qwen3-8B                    | 9.6 %              | 18.0 %      | +8.4 pp     |
| Qwen3-14B                   | 15.4 %             | 23.6 %      | +8.2 pp     |

> [!important] The 8B headline number
> ProRL Agent's 18.0 % is roughly **2×** SkyRL-Agent-8B-v0's 9.4 % — the largest delta among the three sizes and the result the paper leads with.

### Generality

Three additional agents trained on the same infrastructure:

| Agent                                   | Train data         | Benchmark    | Reward / Pass@1                |
| --------------------------------------- | ------------------ | ------------ | ------------------------------ |
| STEM (web search + tools)               | SCP-116K           | mean reward  | 0.20 → 0.65 in 60 steps        |
| Math (IPython + NumPy/SciPy/SymPy)      | DeepScaleR        | AMC          | 0.40 → 0.90                    |
| Code (str_replace_editor)               | Eurus-2-RL-Data    | Codeforces   | 0.23 → 0.42                    |

### Scalability

Near-linear throughput scaling with rollout-node count for SWE tasks.

### Ablations

Qwen3-14B, 8 H100:

| Config                 | Action time (s) | GPU util | Throughput (inst/s) |
| ---------------------- | --------------: | -------: | ------------------: |
| Full                   | 0.42            | 78 %     | 0.37                |
| − load balancing       | 0.42            | 42 %     | 0.25                |
| − efficient bash       | 0.78            | 68 %     | 0.29                |
| − stale-job cleanup    | 0.42            | 65 %     | 0.30                |

Reading: load balancing and stale-job cleanup recover GPU utilization (failed rollouts → stale KV → wasted inference); efficient bash recovers wall-clock per action.

---

## Strengths and limitations

The standout strengths are the three sub-ideas that make the architecture work — token-in/token-out, three-stage pipelining, and the rootless sandbox — each addresses a real failure mode in prior systems and each is implemented with care visible in the code.

Where the paper is less convincing:

- **HTTP overhead is never quantified.** Decoupling implicitly costs network round-trips. Throughput scaling is shown, but end-to-end *training-step latency* is never compared against a coupled baseline like SkyRL-Agent on identical workloads. For long rollouts the cost is negligible; for short single-turn math problems it might matter.
- **Sandbox story is Singularity-only.** Many groups use Docker, podman, or microVM (Firecracker). The HPC-rootless framing is genuine but not as portable as the paper implies.
- **Only one RL algorithm is validated.** Every experiment uses DAPO. PPO, GRPO, RLOO, REINFORCE++ may exercise different rollout-batch shapes and KV-cache reuse patterns. Cross-algorithm generality is asserted, not shown.
- **293-instance training set is small.** The +6–8 pp Pass@1 lifts on SWE-Bench Verified are real, but the *ceiling* this gives us isn't characterized — does more data continue to scale, or is this near the asymptote of the current reward design?
- **No comparison vs. distributed prefix-cache strategies.** Sticking a rollout to one vLLM backend maximizes its cache hits, but if multiple rollouts share a system prompt, distributing across backends might be better. The trade-off is unanalyzed.
- **Open-sourced, but partially.** The `stable` branch is public, but the paper experiments used internal extensions to ProRL/veRL/NeMo-RL. Reproducing the SWE-Bench numbers from scratch requires assembling several stacks.

> [!warning] The reward-server abstraction is underexplained
> The code references a `reward_server_ip` config and a `Reward` class, but the paper text doesn't fully document the contract or how it interacts with `AgentHandler.eval`. A footnote in `OpenHandsServer.__init__` warns: *"No reward server IP provided. Evaluations would only work for swebench problems."* That is a load-bearing limitation deserving more than a warning.

The authors themselves defer "richer environments and improved cluster-scale robustness" to future work — i.e. the current task suite (SWE / math / STEM / code) is narrow and production-cluster failure modes are not fully studied.

---

## What this means

The system itself is solid, but the more interesting claim is the architectural one: **service-oriented design beats embedded design for agentic RL, even when latency budgets are tight.** That claim generalizes. Expect comparable services to emerge over the next 12 months for **environment-as-a-service** (cf. OpenReward in [[environment-design]]), **reward-as-a-service**, and **trajectory-store-as-a-service**. The eventual shape of an "agentic RL platform" looks more like a control plane connecting interchangeable services than a monolithic trainer.

Two specific lessons worth internalizing whether or not you adopt this exact stack:

- **Token-in / token-out is the only safe wire format for multi-turn RL.** If you build any agentic-RL stack, design this in from day one. Re-tokenization drift is a silent killer.
- **Job-level pipeline parallelism is undervalued.** ProRL's three-stage queue gives near-linear scaling without changing any RL algorithm. Most teams leave this throughput on the table.

---

## Source code & reproduction

Quick start (from the README):

```bash
poetry install --with dev,test,runtime,evaluation
pip install git+https://github.com/SWE-Gym/SWE-Bench-Package.git
pip install git+https://github.com/R2E-Gym/R2E-Gym.git
sudo apt-get install -y apptainer

# 1. start vLLM server with HF model
# 2. pull Singularity sandboxes
python scripts/pull_swe_images.py
# 3. launch eval server
python scripts/start_server.py \
  --port 8006 \
  --max-init-workers 8 --max-run-workers 8 --max-eval-workers 4
# 4. POST /add_llm_server, then /start, then /process
```

Files worth reading next, with the role of each:

| File                                                  | Role                                                                                                                                                              |
| ----------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `openhands/nvidia/registry.py`                        | `AgentHandler` ABC, `JobDetails` dataclass, registration tables.                                                                                                  |
| `openhands/nvidia/async_server.py`                    | `OpenHandsServer`, three-queue pipeline, unified `_worker`, min-heap load balancer.                                                                              |
| `openhands/nvidia/async_server_process.py`            | process-based variant of the server (default; `_Thread` exists for debugging).                                                                                    |
| `openhands/nvidia/utils.py`                           | `ProcessRequest` / `CancelRequest` / `LLMServerRequest` Pydantic models, `process_with_timeout`, `cleanup_timed_out_job`, `process_messages_from_agent_state`, `ngram_repetition_reward`. |
| `openhands/nvidia/timer.py`                           | `PausableTimer`, `phase_context`, `run_with_timeout_awareness`.                                                                                                  |
| `openhands/llm/nvidia/qwen3.py`                       | `convert_messages_to_tokens`, `parse_response_ids`, custom Qwen3 chat template.                                                                                  |
| `openhands/runtime/impl/singularity/singularity_runtime.py` | sandbox lifecycle, loopback IP allocator.                                                                                                                |
| `scripts/start_server.py`                             | FastAPI parent + multiprocessing child wiring.                                                                                                                   |
| `trainer_integration/verl/`                           | how the rollout server plugs into a verl trainer.                                                                                                                |

---

## Related reading

- [[agentic-rl-overview]] — broader landscape of agentic RL frameworks.
- [[environment-design]] — sandbox infrastructure (OpenReward, ARES, Daytona).
- [[rl-training-frameworks]] — veRL, OpenRLHF, TRL — the trainers ProRL talks to.
- [[grpo]] — DAPO is a GRPO variant; see GRPO for algorithm context.
- [[tool-use-rl]] — how tool-using rollouts are trained.
- [[kv-cache-optimization]] — why per-task vLLM affinity matters for prefix caching.
- [[multi-turn-optimization]] — multi-turn KV cache reuse, relevant to LLM backend efficiency.
