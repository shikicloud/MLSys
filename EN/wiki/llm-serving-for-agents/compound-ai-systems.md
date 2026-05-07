---
title: "Compound AI Systems Architecture"
category: llm-serving-for-agents
tags: [compound-ai, dspy, rag, llm-cascades, orchestration, retrieval-augmented, routing, verifier]
created: 2026-04-13
updated: 2026-05-07
status: mature
---

# Compound AI Systems Architecture

> [!abstract]+ TL;DR
> A compound AI system orchestrates multiple components — LLMs, retrievers, tools, specialized models, symbolic engines. Systematized by Matei Zaharia et al. (BAIR, Feb 2024): *"state-of-the-art AI results increasingly come from combining components, not improving single models."* **60 % of enterprise LLM apps use RAG; 30 % employ multi-step chains.** Gartner predicts 40 % of enterprise apps will embed agents by end of 2026. Key patterns: **RAG** (production 2026: semantic chunking, parent-child retrieval, hybrid search + cross-encoder reranking, Self-RAG / Corrective RAG), **LLM cascades** (FrugalGPT matches GPT-4 quality at 98 % cost reduction), **DSPy** (Stanford, "programming not prompting" with auto-optimizers).

```
Monolithic LLM:         Input -> [Single LLM] -> Output
Compound AI System:     Input -> [Router] -> [Retriever] -> [LLM] -> [Verifier] -> Output
```

Key stats: 60% of enterprise LLM apps use RAG, 30% employ multi-step chains, Gartner predicts 40% of enterprise apps will embed agents by end of 2026.

---

## Core Components

| Component | Role | Typical Implementations |
|-----------|------|------------------------|
| **Foundation Models** | Central reasoning engine | GPT-4o, Claude, Llama |
| **Retrievers** | Fetch relevant external knowledge | FAISS, Pinecone, BM25 + reranking |
| **Tools & APIs** | Deterministic operations | Code execution, web search, databases |
| **Specialized Models** | Embedding, classification, scoring | Cross-encoders, embedding models |
| **Verifiers** | Output quality checking | LLM-as-judge, unit tests |
| **Guardrails** | Safety and compliance | Input filtering, output auditing |
| **Orchestration Layer** | Workflow coordination | LangGraph, DSPy, state machines |
| **[[agent-memory|Memory Systems]]** | Short/long-term, episodic | Vector DBs, key-value stores |

---

## Architecture Patterns

### RAG (Retrieval-Augmented Generation)

The canonical compound AI system; 60% of enterprise LLM apps.

```
Basic RAG:
  Query -> [Retriever] -> [Context Augmentation] -> [LLM] -> Response

Production RAG (2026):
  Query -> [Query Understanding]     (intent, rewrite, multi-query)
        -> [Hybrid Retrieval]        (dense + sparse + knowledge graph)
        -> [Post-Processing]         (cross-encoder reranking, filtering)
        -> [Generation]              (prompt template + context + LLM)
        -> [Validation]              (faithfulness check, hallucination detection)
```

**Advanced RAG variants**: Self-RAG (adaptive retrieval -- decides whether to retrieve), Corrective RAG (quality assessment with web search fallback), parent-child retrieval, semantic chunking.

### LLM Cascades

Route through progressively capable models to optimize cost-quality tradeoff.

```
[Router] -> Tier 1: Haiku ($0.25/M, 70% of queries)
         -> Tier 2: Sonnet ($3/M, 25% of queries)
         -> Tier 3: Opus ($15/M, 5% of queries)

FrugalGPT: matches GPT-4 quality at 98% cost reduction
```

### Routing and Orchestration

```
Pattern A: Intent routing       [Query] -> [Classifier] -> [Specialized Pipeline]
Pattern B: Quality routing      [Query] -> [Fast Model] -> [Quality Check] -> OK/Upgrade
Pattern C: Parallel + select    [Query] -> [Model A/B/C parallel] -> [Judge] -> Best
Pattern D: Pipeline             [Query] -> [Summarize] -> [Analyze] -> [Generate] -> [Verify]
```

---

## DSPy: Compound System Optimization

Stanford NLP's framework: **"Programming, not prompting."** Turns prompt engineering into an optimizable programming problem.

```python
import dspy

class FactualQA(dspy.Signature):
    """Answer questions based on context with reasoning."""
    context: str = dspy.InputField()
    question: str = dspy.InputField()
    reasoning: str = dspy.OutputField()
    answer: str = dspy.OutputField()

class RAGSystem(dspy.Module):
    def __init__(self, num_passages=3):
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate = dspy.ChainOfThought(FactualQA)

    def forward(self, question):
        context = self.retrieve(question).passages
        return self.generate(
            context="\n".join(context), question=question
        )

# Auto-optimize (~$2, 20 min)
optimizer = MIPROv2(metric=exact_match, num_threads=4)
optimized = optimizer.compile(RAGSystem(), trainset=trainset)
```

DSPy 2.x adds `dspy.Reasoning` for native reasoning model support.

---

## RAG Pipeline Code Example

```python
class ProductionRAG:
    async def query(self, question, top_k=5):
        # 1. Query rewriting (multi-query generation)
        queries = await self._rewrite_query(question)
        # 2. Hybrid retrieval (dense + sparse)
        dense = await self._dense_retrieve(queries, top_k * 2)
        sparse = await self._sparse_retrieve(question, top_k * 2)
        # 3. Reciprocal Rank Fusion
        merged = self._rrf_fusion(dense, sparse)
        # 4. Cross-encoder reranking
        reranked = await self._rerank(question, merged[:top_k * 2])
        # 5. Generation with context
        answer = await self._generate(question, reranked[:top_k])
        # 6. Faithfulness validation
        validated = await self._validate(question, answer, reranked[:top_k])
        return validated

    def _rrf_fusion(self, dense, sparse, k=60):
        scores = {}
        for rank, doc in enumerate(dense):
            scores[doc["id"]] = scores.get(doc["id"], 0) + 1/(k+rank+1)
        for rank, doc in enumerate(sparse):
            scores[doc["id"]] = scores.get(doc["id"], 0) + 1/(k+rank+1)
        all_docs = {d["id"]: d for d in dense + sparse}
        return [all_docs[id] for id in sorted(scores, key=scores.get, reverse=True)]
```

---

## Evaluation Challenges

Evaluating compound systems is harder than evaluating single models:

- **End-to-end good != every component good** (lucky compensation)
- **Component-level good != end-to-end good** (composition issues)
- **Error attribution**: Was the bad answer caused by retrieval or generation?

Frameworks: RAGAS (RAG evaluation), DSPy Assert (programmatic constraints), DeepEval (LLM-as-judge).

---

## Performance Optimization

```
Latency breakdown (sequential):
  [Query Rewrite] -> [Retrieval] -> [Reranking] -> [Generation] -> [Validation]
     200ms            100ms          150ms          1500ms          800ms  = 2750ms

Optimized (parallelization + caching):
  [Rewrite+Retrieval parallel] -> [Reranking] -> [Streaming Generation]
         250ms                      150ms              1500ms      = 1900ms

Cost optimization strategies:
  1. Model selection: route/classify with small model, generate with medium
  2. Caching: embedding cache (60-80% reduction), semantic cache
  3. Batching: merge embedding requests
  4. Context compression: truncate irrelevant retrievals
```

---

## State of the Art (2025-2026)

- RAG evolved into "context engines" with intelligent retrieval
- Domain-specialized compound stacks (legal, healthcare, finance)
- End-to-end optimization via DSPy
- Blurring line between compound systems and agents
- [[mcp-protocol|MCP]] as integration standard

---

## Challenges

- Design complexity: no one-size-fits-all component selection
- Non-differentiable components prevent gradient optimization
- Error attribution across components
- Latency accumulation from sequential dependencies
- Operational complexity of multi-component monitoring and maintenance

---

## References

- Zaharia et al., "The Shift from Models to Compound AI Systems," BAIR Blog, Feb 2024
- Lewis et al., "RAG for Knowledge-Intensive NLP Tasks," NeurIPS 2020
- Khattab et al., "DSPy: Compiling Declarative LM Calls into Self-Improving Pipelines," ICLR 2024
- Chen et al., "FrugalGPT," 2023
- Asai et al., "Self-RAG," ICLR 2024

---

## Related Pages

- [[ai-agent-overview]] -- Agent architectures that form compound systems
- [[agent-serving-challenges]] -- Serving compound systems
- [[mcp-protocol]] -- Integration standard
- [[function-calling-optimization]] -- Optimizing LLM-tool loops
- [[multi-turn-optimization]] -- Multi-turn optimization
- [[kv-cache-optimization]] -- KV cache management
