---
title: "Compound AI Systems Architecture"
category: llm-serving-for-agents
tags: [compound-ai, dspy, rag, llm-cascades, orchestration, retrieval-augmented, routing, verifier]
created: 2026-04-13
updated: 2026-05-20
status: mature
---

# Compound AI Systems Architecture

> [!abstract]+ TL;DR
> Compound AI systems orchestrate multiple interacting components — LLMs, retrievers, tools, specialized models, symbolic engines. Matei Zaharia et al. (BAIR, 2024/02) articulated the thesis: *"State-of-the-art AI results increasingly come from compound systems, not single-model improvements."* **60% of enterprise LLM apps use RAG; 30% use multi-step chains.** Gartner predicts that by end of 2026, 40% of enterprise apps will embed agents. Key patterns: **RAG** (2026 production: semantic chunking, parent-child retrieval, hybrid search + cross-encoder reranking, Self-RAG / Corrective RAG), **LLM cascades** (FrugalGPT matches GPT-4 quality at 98% cost reduction), **DSPy** (Stanford, "programming, not prompting" + auto-optimizers).

## Overview

A compound AI system is built from multiple interacting components — LLMs, retrievers, tools, specialized models, symbolic engines, and more. The concept was articulated by Matei Zaharia et al. (BAIR, Feb 2024); the core argument: **state-of-the-art AI results increasingly come from compositions of multiple components rather than improvements to a single model**.

### Why Compound > Monolithic?

```
Monolithic LLM:
  Input ──> [single LLM] ──> Output

  Limits:
  - Knowledge cutoff
  - Cannot access private data
  - Hallucination
  - No deterministic computation
  - One model does everything = poor cost/quality

Compound AI system:
  Input ──> [router] ──> [retriever] ──> [LLM] ──> [verifier] ──> Output
              │              │             │            │
              v              v             v            v
        [intent ID]   [vector store]  [reasoning]  [fact check]

  Strengths:
  - Augment LLM with fresh data (RAG)
  - Deterministic components for deterministic tasks
  - Multi-model collaboration with cost optimization
  - Each component optimized/replaced independently
  - Composability and scalability
```

### Industry Data

| Metric | Value |
|--------|-------|
| Share of enterprise LLM apps using RAG | 60% (BAIR 2024) |
| Enterprise apps using multi-step chains | 30% |
| Gartner forecast: enterprise apps embedding agents by end of 2026 | 40% |
| Share of enterprise AI budget spent on orchestration/integration | ~35% |

---

## System Components

### Core Component Inventory

```
┌────────────────────────────────────────────────────────┐
│              Compound AI system components              │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Foundation   │  │  Retriever   │  │ Tools & APIs │  │
│  │ model (LLM)  │  │              │  │              │  │
│  │              │  │              │  │              │  │
│  │  - GPT-4o    │  │  - BM25      │  │  - Code exec │  │
│  │  - Claude    │  │  - Vector    │  │  - Web search│  │
│  │  - Gemini    │  │  - Hybrid    │  │  - Databases │  │
│  │  - OSS models│  │  - Reranker  │  │  - Calculator│  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Specialized  │  │ Orchestration│  │  Memory      │  │
│  │ models       │  │ layer        │  │              │  │
│  │              │  │              │  │              │  │
│  │  - Embedding │  │  - Workflow  │  │  - Short-term│  │
│  │  - Classifier│  │  - Recovery  │  │  - Long-term │  │
│  │  - Scorer    │  │  - Routing   │  │  - Episodic  │  │
│  │  - Reranker  │  │  - State mgr │  │  - Vec store │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐                    │
│  │  Verifier    │  │  Guardrails  │                    │
│  │              │  │              │                    │
│  │              │  │              │                    │
│  │  - Fact check│  │  - Input flt │                    │
│  │  - Code test │  │  - Output rev│                    │
│  │  - Logic vfy │  │  - Safety    │                    │
│  └──────────────┘  └──────────────┘                    │
└────────────────────────────────────────────────────────┘
```

### Component Roles in Detail

| Component | Role | Typical implementation | Key metric |
|-----------|------|------------------------|------------|
| **Foundation model** | Core reasoning engine | GPT-4o, Claude, Llama | Accuracy, latency |
| **Retriever** | Fetch relevant external knowledge | FAISS, Pinecone, Weaviate | Recall@K, latency |
| **Reranker** | Refine ranking of retrieved results | Cross-encoder, Cohere Rerank | NDCG, MRR |
| **Tools** | Execute deterministic operations | Code sandbox, API calls | Reliability, latency |
| **Verifier** | Check output quality | LLM-as-judge, unit tests | Precision, recall |
| **Guardrails** | Safety and compliance | Guardrails AI, NeMo | False-block / escape rate |
| **Router** | Dispatch requests to the right component | Classifier, rule engine | Routing accuracy |
| **Orchestrator** | Coordinate component interaction | LangGraph, DSPy | End-to-end latency |

---

## Architecture Patterns

### 1. RAG (Retrieval-Augmented Generation) — Classic Compound System

RAG is the most widely used compound AI architecture; 60% of enterprise LLM apps are RAG-based.

#### Basic RAG Architecture

```
┌──────────────────────────────────────────────────────┐
│                   Basic RAG architecture              │
│                                                       │
│  ┌──────┐    ┌───────────┐    ┌────────────────┐     │
│  │ User │───>│ Query     │───>│   Retriever    │     │
│  │ query│    │ processing│    │                │     │
│  └──────┘    │ (rewrite) │    └───────┬────────┘     │
│              └───────────┘            │               │
│                                       v               │
│                               ┌───────────────┐      │
│                               │  Vector store │      │
│                               │  (FAISS/      │      │
│                               │   Pinecone)   │      │
│                               └───────┬───────┘      │
│                                       │               │
│                                       v               │
│  ┌──────┐    ┌───────────┐    ┌───────────────┐      │
│  │Resp- │<───│    LLM    │<───│  Context      │      │
│  │ onse │    │ (generator)│    │  augmentation │      │
│  │      │    │           │    │               │      │
│  └──────┘    └───────────┘    └───────────────┘      │
└──────────────────────────────────────────────────────┘
```

#### Production-Grade RAG Architecture (2026)

```
┌───────────────────────────────────────────────────────────┐
│              Production RAG architecture (2026)            │
│                                                            │
│  ┌──────┐    ┌─────────────────────────────────┐          │
│  │ Query│───>│  Query understanding             │          │
│  └──────┘    │  ├─ Intent classification        │          │
│              │  ├─ Query rewrite (HyDE)         │          │
│              │  ├─ Multi-query generation       │          │
│              │  └─ Query routing                │          │
│              └──────────┬──────────────────────┘          │
│                         │                                  │
│                         v                                  │
│              ┌─────────────────────────────────┐          │
│              │  Retrieval                       │          │
│              │  ├─ Dense (vector)               │          │
│              │  ├─ Sparse (BM25)                │          │
│              │  ├─ Hybrid (RRF fusion)          │          │
│              │  └─ Knowledge-graph retrieval    │          │
│              └──────────┬──────────────────────┘          │
│                         │                                  │
│                         v                                  │
│              ┌─────────────────────────────────┐          │
│              │  Post-processing                 │          │
│              │  ├─ Cross-encoder rerank         │          │
│              │  ├─ Relevance filtering          │          │
│              │  ├─ Dedup & diversify            │          │
│              │  └─ Context compression          │          │
│              └──────────┬──────────────────────┘          │
│                         │                                  │
│                         v                                  │
│              ┌─────────────────────────────────┐          │
│              │  Generation                      │          │
│              │  ├─ Prompt template + context    │          │
│              │  ├─ LLM generation               │          │
│              │  └─ Citation tracking            │          │
│              └──────────┬──────────────────────┘          │
│                         │                                  │
│                         v                                  │
│              ┌─────────────────────────────────┐          │
│              │  Validation                      │          │
│              │  ├─ Factual consistency          │          │
│              │  ├─ Hallucination detection      │          │
│              │  └─ Guardrail filtering          │          │
│              └─────────────────────────────────┘          │
└───────────────────────────────────────────────────────────┘
```

#### RAG Evolution: Self-RAG and Corrective RAG

```
Self-RAG (adaptive retrieval):

  Query ──> [retrieve?] ──Yes──> [retrieve] ──> [generate]
                │                                  │
                No                          [self-evaluate]
                │                          /     |       \
                v                      relevant partial irrelevant
          [generate directly]           │       │           │
                                      keep   rewrite     drop & retry

Corrective RAG:

  Query ──> [retrieve] ──> [quality check] ──> high ──> [generate]
                                │
                              low
                                │
                                v
                          [Web search fallback] ──> [regenerate]
```

### 2. LLM Cascades

Route requests through models of increasing capability to optimize the cost-quality trade-off.

```
┌──────────────────────────────────────────────────┐
│              LLM cascade architecture             │
│                                                   │
│  ┌──────┐    ┌─────────────┐                     │
│  │ Query│───>│ Router /     │                     │
│  │      │    │ confidence   │                     │
│  └──────┘    └──────┬──────┘                     │
│                     │                             │
│          ┌──────────┼──────────┐                  │
│          v          v          v                  │
│    ┌─────────┐ ┌─────────┐ ┌─────────┐          │
│    │ Tier 1  │ │ Tier 2  │ │ Tier 3  │          │
│    │ Haiku   │ │ Sonnet  │ │ Opus    │          │
│    │ $0.25/M │ │ $3/M    │ │ $15/M   │          │
│    │ 70% qry │ │ 25% qry │ │ 5% qry  │          │
│    └────┬────┘ └────┬────┘ └────┬────┘          │
│         │           │           │                 │
│         v           v           v                 │
│    ┌─────────────────────────────────┐           │
│    │  Confidence check: good enough? │           │
│    │  yes → return  no → escalate    │           │
│    └─────────────────────────────────┘           │
│                                                   │
│  FrugalGPT: GPT-4 quality at 2% the cost         │
└──────────────────────────────────────────────────┘
```

**FrugalGPT strategy**:
- Try the cheap model first
- Check output confidence / quality
- Escalate to a stronger model if unsatisfactory
- Most easy queries are served by the cheap model

**Cost-savings example**:

```
Assume 1000 queries with this distribution:

  Direct Opus:
  1000 × $0.015 = $15.00

  Cascade:
  700 × Haiku ($0.00025) = $0.175
  250 × Sonnet ($0.003)  = $0.750
   50 × Opus ($0.015)    = $0.750
  Total                  = $1.675

  Savings: 89%
  Quality loss: <2% (only on routing errors)
```

### 3. Routing and Orchestration Patterns

```
Pattern A: intent routing
  ┌──────┐    ┌──────────────┐    ┌──────────┐
  │ Query│───>│ Intent       │───>│ Pipeline │
  └──────┘    │ classifier   │    │ (RAG/    │
              └──────────────┘    │  Agent/  │
                                  │  Direct) │
                                  └──────────┘

Pattern B: quality routing (auto-escalation)
  Query ──> [fast model] ──> [quality check] ──OK──> Response
                                  │
                                not OK
                                  │
                                  v
                            [strong model] ──> Response

Pattern C: parallel execution + best selection
  Query ──┬──> [Model A] ──┐
          ├──> [Model B] ──┼──> [Judge] ──> Best response
          └──> [Model C] ──┘

Pattern D: pipeline orchestration
  Query ──> [Summarize] ──> [Analyze] ──> [Generate] ──> [Verify] ──> Response
```

---

## DSPy: A Compound-System Optimization Framework

### Overview

DSPy (Stanford NLP) is built around **"programming, not prompting"** — it turns prompt engineering into an optimizable programming problem.

### Core Abstractions

```
┌──────────────────────────────────────────────┐
│              DSPy core concepts                │
│                                                │
│  Signatures:                                   │
│    Declarative input/output specs              │
│    "question -> answer"                        │
│    "context, question -> reasoning, answer"    │
│                                                │
│  Modules:                                      │
│    Abstractions over prompting strategies      │
│    - dspy.Predict        (direct prediction)   │
│    - dspy.ChainOfThought (chain of thought)    │
│    - dspy.ReAct          (reason + act)        │
│    - dspy.Reasoning      (native reasoning-model support) │
│                                                │
│  Optimizers:                                   │
│    Auto-optimize prompts/weights               │
│    - MIPROv2 (~$2, 20-min auto-tune)           │
│    - BootstrapFewShot (auto-select demos)      │
│    - GRPO / SFT (weight tuning)                │
└──────────────────────────────────────────────┘
```

### DSPy Code Example

```python
import dspy

# 1. Configure LLM
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# 2. Define a signature
class FactualQA(dspy.Signature):
    """Answer a question grounded in context, with reasoning."""
    context: str = dspy.InputField(desc="Relevant documents")
    question: str = dspy.InputField(desc="User question")
    reasoning: str = dspy.OutputField(desc="Reasoning trace")
    answer: str = dspy.OutputField(desc="Final answer")

# 3. Compose a compound system
class RAGSystem(dspy.Module):
    def __init__(self, num_passages=3):
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate = dspy.ChainOfThought(FactualQA)

    def forward(self, question: str):
        # Retrieve relevant passages
        context = self.retrieve(question).passages
        # Generate the answer
        result = self.generate(
            context="\n".join(context),
            question=question
        )
        return result

# 4. Auto-optimize
from dspy.teleprompt import MIPROv2

# Training data
trainset = [
    dspy.Example(
        question="What is a KV cache?",
        answer="A KV cache stores attention layer key/value pairs..."
    ).with_inputs("question"),
    # ... more examples
]

# Run the optimizer (~$2, 20 min)
optimizer = MIPROv2(metric=exact_match, num_threads=4)
optimized_rag = optimizer.compile(
    RAGSystem(),
    trainset=trainset,
    max_bootstrapped_demos=4,
    max_labeled_demos=4,
)

# 5. Use the optimized system
result = optimized_rag(question="What is prefix caching?")
print(result.answer)
```

### DSPy 2.x New Features

```python
# DSPy 2.x: native reasoning-model support
class ComplexReasoning(dspy.Module):
    def __init__(self):
        # dspy.Reasoning auto-leverages reasoning-model capabilities
        self.reason = dspy.Reasoning(
            "problem -> solution",
            max_thinking_tokens=4096
        )

    def forward(self, problem):
        return self.reason(problem=problem)
```

---

## Complete RAG Pipeline Code

```python
"""Production-grade RAG pipeline."""
from typing import List, Optional
import numpy as np

class ProductionRAG:
    """RAG pipeline with hybrid retrieval, reranking, and validation."""

    def __init__(self, llm, embedder, vector_store, bm25_index):
        self.llm = llm
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_index = bm25_index

    async def query(self, question: str, top_k: int = 5) -> dict:
        """Full RAG query pipeline."""

        # Step 1: query rewrite (optional)
        rewritten_queries = await self._rewrite_query(question)

        # Step 2: hybrid retrieval
        dense_results = await self._dense_retrieve(
            rewritten_queries, top_k=top_k * 2
        )
        sparse_results = await self._sparse_retrieve(
            question, top_k=top_k * 2
        )

        # Step 3: fusion (Reciprocal Rank Fusion)
        merged = self._rrf_fusion(dense_results, sparse_results)

        # Step 4: rerank
        reranked = await self._rerank(question, merged[:top_k * 2])

        # Step 5: context construction
        context = self._build_context(reranked[:top_k])

        # Step 6: generate
        answer = await self._generate(question, context)

        # Step 7: validate
        validated = await self._validate(question, answer, context)

        return {
            "answer": validated["answer"],
            "sources": reranked[:top_k],
            "confidence": validated["confidence"]
        }

    async def _rewrite_query(self, question: str) -> List[str]:
        """Generate multiple query variants for better recall."""
        prompt = f"""Generate 3 search queries from different angles for the following question:
        Question: {question}
        Format: one query per line"""
        result = await self.llm.generate(prompt)
        queries = [question] + result.strip().split("\n")
        return queries

    async def _dense_retrieve(
        self, queries: List[str], top_k: int
    ) -> List[dict]:
        """Vector retrieval."""
        all_results = []
        for q in queries:
            embedding = self.embedder.encode(q)
            results = self.vector_store.search(embedding, top_k=top_k)
            all_results.extend(results)
        # Dedup
        seen = set()
        unique = []
        for r in all_results:
            if r["id"] not in seen:
                seen.add(r["id"])
                unique.append(r)
        return unique

    async def _sparse_retrieve(
        self, query: str, top_k: int
    ) -> List[dict]:
        """BM25 sparse retrieval."""
        return self.bm25_index.search(query, top_k=top_k)

    def _rrf_fusion(
        self,
        dense: List[dict],
        sparse: List[dict],
        k: int = 60
    ) -> List[dict]:
        """Reciprocal Rank Fusion to merge multi-path retrieval."""
        scores = {}
        for rank, doc in enumerate(dense):
            scores[doc["id"]] = scores.get(doc["id"], 0)
            scores[doc["id"]] += 1.0 / (k + rank + 1)
        for rank, doc in enumerate(sparse):
            scores[doc["id"]] = scores.get(doc["id"], 0)
            scores[doc["id"]] += 1.0 / (k + rank + 1)

        # Merge docs and sort by RRF score
        all_docs = {d["id"]: d for d in dense + sparse}
        sorted_ids = sorted(scores, key=scores.get, reverse=True)
        return [all_docs[id] for id in sorted_ids if id in all_docs]

    async def _rerank(
        self, query: str, documents: List[dict]
    ) -> List[dict]:
        """Cross-encoder rerank."""
        pairs = [(query, doc["text"]) for doc in documents]
        scores = self.reranker.predict(pairs)
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = score
        return sorted(documents, key=lambda x: x["rerank_score"],
                      reverse=True)

    def _build_context(self, documents: List[dict]) -> str:
        """Build the generator's input context."""
        context_parts = []
        for i, doc in enumerate(documents):
            context_parts.append(
                f"[Source {i+1}] ({doc.get('title', 'Unknown')})\n"
                f"{doc['text']}\n"
            )
        return "\n---\n".join(context_parts)

    async def _generate(self, question: str, context: str) -> str:
        """Generate the answer grounded in context."""
        prompt = f"""Answer the question using the references below. If the references do not contain the answer, say so explicitly.
Cite source numbers in your answer.

References:
{context}

Question: {question}

Answer:"""
        return await self.llm.generate(prompt)

    async def _validate(
        self, question: str, answer: str, context: str
    ) -> dict:
        """Validate factual consistency of the answer."""
        prompt = f"""Check whether the following answer is consistent with the references.

References: {context}
Question: {question}
Answer: {answer}

Evaluate:
1. Any factual errors? (yes/no)
2. Any hallucinated content? (yes/no)
3. Confidence (0-1):"""

        validation = await self.llm.generate(prompt)
        # Parse the validation result
        confidence = self._parse_confidence(validation)

        return {
            "answer": answer,
            "confidence": confidence,
            "validation": validation
        }
```

---

## Evaluation Challenges

### End-to-End vs. Component Evaluation

```
Evaluation complexity:

  Single LLM:
    Dimensions: 1 (output quality)

  RAG system:
    Dimensions: 5+
    ├── Retrieval quality (Recall@K, MRR, NDCG)
    ├── Rerank quality (NDCG after rerank)
    ├── Context relevance
    ├── Generation quality (Faithfulness, Answer Relevance)
    └── End-to-end (user satisfaction)

  Issues:
  1. Good end-to-end ≠ every component is good (luck may compensate)
  2. Good per-component ≠ good end-to-end (composition can break things)
  3. Error attribution is hard: was the wrong answer due to retrieval or generation?
```

### Evaluation Frameworks

| Framework | Focus | Notes |
|-----------|-------|-------|
| **RAGAS** | RAG system evaluation | Reference-free (Faithfulness, Relevance) |
| **LlamaIndex** | Retrieval + generation | Built-in RAG metrics |
| **DSPy Assert** | Module constraints | Programmatic asserts for optimization |
| **DeepEval** | End-to-end | LLM-as-judge + multi-metric |

```python
# RAGAS evaluation example
from ragas import evaluate
from ragas.metrics import (
    faithfulness,       # Answer's faithfulness to the context
    answer_relevancy,   # Answer relevance to the question
    context_precision,  # Context precision
    context_recall,     # Context recall
)

result = evaluate(
    dataset=eval_dataset,
    metrics=[faithfulness, answer_relevancy,
             context_precision, context_recall],
)
print(result)
# {'faithfulness': 0.85, 'answer_relevancy': 0.92,
#  'context_precision': 0.78, 'context_recall': 0.81}
```

---

## Performance Optimization

### Latency Optimization

```
Compound-system latency decomposition:

  Sequential:
  [rewrite] ──> [retrieve] ──> [rerank] ──> [generate] ──> [validate]
    200ms        100ms          150ms         1500ms         800ms
                                                          = 2750ms

  Optimized (parallel + cache):
  [rewrite + retrieve in parallel] ──> [rerank] ──> [generate(stream)]
          250ms                          150ms         1500ms
                                                     = 1900ms

  Further (speculative execution):
  [retrieve] ──> [rerank + speculative gen in parallel] ──> [validate/fix]
    100ms             1500ms                                  200ms
                                                            = 1800ms

  Tricks:
  1. Parallelize independent steps
  2. Cache retrieval results (similar-query hits)
  3. Stream output to reduce perceived latency
  4. Speculative execution + verify
  5. Component-level caches (embedding cache, rerank cache)
```

### Cost Optimization

```
Cost-optimization tactics:

  1. Model selection
     - Routing/classification: small model (Haiku)
     - Generation: mid-tier model (Sonnet)
     - Verification: strong model only on low confidence

  2. Caching
     - Embedding cache: cut 60-80% of embedding calls
     - Semantic cache: return cached result for similar queries
     - Prompt cache: cache system prompt + templates

  3. Batching
     - Merge embedding requests across queries
     - Batch reranking

  4. Context compression
     - Drop irrelevant retrieved passages
     - Extractive summarization to shrink context length
```

---

## State of the Art (2025–2026)

1. **RAG evolves into the "context engine"**: from naive retrieval to intelligent retrieval (adaptive, corrective, multi-hop)
2. **Domain-specialized compound stacks**: legal, medical, financial each have their own component recipes
3. **End-to-end optimization**: frameworks like DSPy enable full-stack auto-optimization
4. **Compound system = agent**: the boundary between compound systems and agents is blurring
5. **MCP as the integration standard**: [[mcp-protocol]] standardizes inter-component communication

---

## Challenges

1. **Design complexity**: no universally optimal component selection
2. **Non-differentiable components**: retrievers, tools and others block gradient-based optimization
3. **Error attribution**: cross-component errors are hard to root-cause
4. **Latency accumulation**: sequentially dependent components stack latencies
5. **Ops complexity**: multi-component systems are costly to monitor, debug, maintain
6. **Version management**: independent component updates can break overall behavior

---

## References

- Zaharia et al., "The Shift from Models to Compound AI Systems," BAIR Blog, Feb 2024
- Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," NeurIPS 2020
- Khattab et al., "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines," ICLR 2024
- Chen et al., "FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance," 2023
- Asai et al., "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection," ICLR 2024
- Yan et al., "Corrective Retrieval Augmented Generation," 2024
- Es et al., "RAGAS: Automated Evaluation of Retrieval Augmented Generation," 2024
- Gartner, "Predicts 2025: AI Agents Are the Next Frontier," 2024

---

## Related Pages

- [[ai-agent-overview]] -- agent architectures that constitute compound systems
- [[agent-serving-challenges]] -- challenges in serving compound systems
- [[mcp-protocol]] -- component integration standard
- [[function-calling-optimization]] -- optimizing LLM-tool interaction
- [[multi-turn-optimization]] -- multi-turn conversation optimization
- [[kv-cache-optimization]] -- KV cache management
- [[long-context-serving]] -- long-context serving
