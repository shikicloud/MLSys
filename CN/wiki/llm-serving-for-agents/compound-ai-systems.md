---
title: "复合 AI 系统架构"
category: llm-serving-for-agents
tags: [复合ai, dspy, rag, llm级联, 编排, 检索增强, 路由, 验证器]
created: 2026-04-13
updated: 2026-05-07
status: mature
---

# 复合 AI 系统架构

> [!abstract]+ TL;DR
> 复合 AI 系统编排多个交互组件 —— LLM、检索器、工具、专用模型、符号引擎。Matei Zaharia 等（BAIR, 2024/02）系统阐述：*"最先进的 AI 成果越来越多来自多组件组合，而非单模型改进。"* **60 % 企业 LLM 应用使用 RAG，30 % 用多步链。** Gartner 预测 2026 年底 40 % 企业应用嵌入智能体。关键模式：**RAG**（2026 生产版：语义分块、父子检索、混合搜索 + 交叉编码器重排序、Self-RAG / Corrective RAG）、**LLM 级联**（FrugalGPT 以 98 % 成本降低匹配 GPT-4 质量）、**DSPy**（斯坦福，"编程而非提示" + 自动优化器）。

## 概述

复合 AI 系统（Compound AI Systems）是由多个交互组件编排而成的系统 -- LLM、检索器、工具、专用模型、符号引擎等。这一概念由 Matei Zaharia 等人（BAIR, 2024 年 2 月）系统阐述，核心论点是：**最先进的 AI 成果越来越多地来自于多组件系统的组合，而非单一模型的改进**。

### 为什么复合 > 单体？

```
单体 LLM:
  输入 ──> [单一 LLM] ──> 输出
  
  局限:
  - 知识截止日期
  - 无法访问私有数据
  - 幻觉问题
  - 无法执行确定性计算
  - 一个模型做所有事 = 性价比低

复合 AI 系统:
  输入 ──> [路由器] ──> [检索器] ──> [LLM] ──> [验证器] ──> 输出
              │           │           │           │
              v           v           v           v
          [意图识别]  [向量数据库]  [推理引擎]  [事实检查]
  
  优势:
  - 用最新数据增强 LLM (RAG)
  - 确定性组件处理确定性任务
  - 多模型协作，成本优化
  - 每个组件可独立优化/替换
  - 可组合性和可扩展性
```

### 产业数据

| 指标 | 数据 |
|------|------|
| 企业 LLM 应用使用 RAG 的比例 | 60% (BAIR 2024) |
| 使用多步链的企业应用 | 30% |
| Gartner 预测 2026 底嵌入智能体的企业应用 | 40% |
| 企业 AI 预算中用于编排/集成的比例 | ~35% |

---

## 系统组件

### 核心组件清单

```
┌────────────────────────────────────────────────────────┐
│                  复合 AI 系统组件                        │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  基础模型     │  │   检索器      │  │  工具 & API  │  │
│  │  (LLM)       │  │  (Retriever) │  │  (Tools)     │  │
│  │              │  │              │  │              │  │
│  │  - GPT-4o    │  │  - BM25      │  │  - 代码执行   │  │
│  │  - Claude    │  │  - 向量搜索   │  │  - Web 搜索  │  │
│  │  - Gemini    │  │  - 混合搜索   │  │  - 数据库    │  │
│  │  - 开源模型   │  │  - 重排序器   │  │  - 计算器    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  专用模型     │  │  编排层      │  │  记忆系统     │  │
│  │(Specialized) │  │(Orchestration)│  │  (Memory)    │  │
│  │              │  │              │  │              │  │
│  │  - 嵌入模型   │  │  - 工作流引擎 │  │  - 短期记忆   │  │
│  │  - 分类器     │  │  - 错误恢复   │  │  - 长期记忆   │  │
│  │  - 评分模型   │  │  - 路由逻辑   │  │  - 情景记忆   │  │
│  │  - 重排序模型  │  │  - 状态管理   │  │  - 向量存储   │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐                    │
│  │  验证器      │  │  护栏        │                    │
│  │ (Verifier)  │  │ (Guardrails) │                    │
│  │              │  │              │                    │
│  │  - 事实检查   │  │  - 输入过滤   │                    │
│  │  - 代码测试   │  │  - 输出审核   │                    │
│  │  - 逻辑验证   │  │  - 安全边界   │                    │
│  └──────────────┘  └──────────────┘                    │
└────────────────────────────────────────────────────────┘
```

### 组件角色详解

| 组件 | 角色 | 典型实现 | 关键指标 |
|------|------|---------|---------|
| **基础模型** | 核心推理引擎 | GPT-4o, Claude, Llama | 准确率, 延迟 |
| **检索器** | 获取相关外部知识 | FAISS, Pinecone, Weaviate | 召回率@K, 延迟 |
| **重排序器** | 精细化排序检索结果 | Cross-encoder, Cohere Rerank | NDCG, MRR |
| **工具** | 执行确定性操作 | 代码沙箱, API 调用 | 可靠性, 延迟 |
| **验证器** | 检查输出质量 | LLM-as-judge, 单元测试 | 精确率, 召回率 |
| **护栏** | 安全和合规保障 | Guardrails AI, NeMo | 误拦率, 逃逸率 |
| **路由器** | 请求分流到合适组件 | 分类器, 规则引擎 | 路由准确率 |
| **编排器** | 协调组件交互 | LangGraph, DSPy | 端到端延迟 |

---

## 架构模式

### 1. RAG（检索增强生成）-- 经典复合系统

RAG 是最广泛使用的复合 AI 系统架构，60% 的企业 LLM 应用基于 RAG。

#### 基础 RAG 架构

```
┌──────────────────────────────────────────────────────┐
│                   基础 RAG 架构                       │
│                                                       │
│  ┌──────┐    ┌───────────┐    ┌────────────────┐     │
│  │ 用户  │───>│  查询处理  │───>│   检索器        │     │
│  │ 查询  │    │  (可选改写) │    │  (Retriever)   │     │
│  └──────┘    └───────────┘    └───────┬────────┘     │
│                                       │               │
│                                       v               │
│                               ┌───────────────┐      │
│                               │  向量数据库    │      │
│                               │  (FAISS/       │      │
│                               │   Pinecone)    │      │
│                               └───────┬───────┘      │
│                                       │               │
│                                       v               │
│  ┌──────┐    ┌───────────┐    ┌───────────────┐      │
│  │ 响应  │<───│    LLM    │<───│  上下文增强    │      │
│  │      │    │  (生成器)   │    │  (Context      │      │
│  │      │    │           │    │   Augmentation) │      │
│  └──────┘    └───────────┘    └───────────────┘      │
└──────────────────────────────────────────────────────┘
```

#### 生产级 RAG 架构（2026）

```
┌───────────────────────────────────────────────────────────┐
│               生产级 RAG 架构 (2026)                       │
│                                                            │
│  ┌──────┐    ┌─────────────────────────────────┐          │
│  │ 查询  │───>│  查询理解层                     │          │
│  └──────┘    │  ├─ 意图分类                     │          │
│              │  ├─ 查询改写 (HyDE)              │          │
│              │  ├─ 多查询生成                    │          │
│              │  └─ 查询路由                      │          │
│              └──────────┬──────────────────────┘          │
│                         │                                  │
│                         v                                  │
│              ┌─────────────────────────────────┐          │
│              │  检索层                          │          │
│              │  ├─ 稠密检索 (向量)               │          │
│              │  ├─ 稀疏检索 (BM25)              │          │
│              │  ├─ 混合搜索 (RRF 融合)           │          │
│              │  └─ 知识图谱检索                  │          │
│              └──────────┬──────────────────────┘          │
│                         │                                  │
│                         v                                  │
│              ┌─────────────────────────────────┐          │
│              │  后处理层                        │          │
│              │  ├─ 交叉编码器重排序              │          │
│              │  ├─ 相关性过滤                    │          │
│              │  ├─ 去重和多样化                  │          │
│              │  └─ 上下文压缩                    │          │
│              └──────────┬──────────────────────┘          │
│                         │                                  │
│                         v                                  │
│              ┌─────────────────────────────────┐          │
│              │  生成层                          │          │
│              │  ├─ 提示模板 + 检索上下文         │          │
│              │  ├─ LLM 生成                     │          │
│              │  └─ 引用追踪                     │          │
│              └──────────┬──────────────────────┘          │
│                         │                                  │
│                         v                                  │
│              ┌─────────────────────────────────┐          │
│              │  验证层                          │          │
│              │  ├─ 事实一致性检查                │          │
│              │  ├─ 幻觉检测                     │          │
│              │  └─ 护栏过滤                     │          │
│              └─────────────────────────────────┘          │
└───────────────────────────────────────────────────────────┘
```

#### RAG 进化：Self-RAG 和 Corrective RAG

```
Self-RAG (自适应检索):

  查询 ──> [是否需要检索?] ──Yes──> [检索] ──> [生成]
                │                                │
                No                          [自我评估]
                │                           /    |    \
                v                      相关  部分相关  不相关
          [直接生成]                    │      │        │
                                      保留   改写查询  丢弃并重试

Corrective RAG (纠正式检索):

  查询 ──> [检索] ──> [质量评估] ──> 高质量 ──> [生成]
                          │
                       低质量
                          │
                          v
                    [Web 搜索补充] ──> [重新生成]
```

### 2. LLM 级联（LLM Cascades）

通过能力递增的模型路由请求，优化成本-质量权衡。

```
┌──────────────────────────────────────────────────┐
│                LLM 级联架构                       │
│                                                   │
│  ┌──────┐    ┌─────────────┐                     │
│  │ 查询  │───>│  路由器/     │                     │
│  │      │    │  置信度检查   │                     │
│  └──────┘    └──────┬──────┘                     │
│                     │                             │
│          ┌──────────┼──────────┐                  │
│          v          v          v                  │
│    ┌─────────┐ ┌─────────┐ ┌─────────┐          │
│    │ Tier 1  │ │ Tier 2  │ │ Tier 3  │          │
│    │ Haiku   │ │ Sonnet  │ │ Opus    │          │
│    │ $0.25/M │ │ $3/M    │ │ $15/M   │          │
│    │ 70% 查询│ │ 25% 查询│ │ 5% 查询 │          │
│    └────┬────┘ └────┬────┘ └────┬────┘          │
│         │           │           │                 │
│         v           v           v                 │
│    ┌─────────────────────────────────┐           │
│    │  置信度检查: 输出质量够好吗?     │           │
│    │  是 → 返回  否 → 升级到下一级    │           │
│    └─────────────────────────────────┘           │
│                                                   │
│  FrugalGPT: 以 GPT-4 的 2% 成本达到同等质量      │
└──────────────────────────────────────────────────┘
```

**FrugalGPT 策略**：
- 先用便宜模型尝试
- 检查输出的置信度/质量
- 不满意则升级到更强的模型
- 大多数简单查询被便宜模型处理

**成本节省示例**：

```
假设 1000 个查询的分布:

  直接使用 Opus:
  1000 × $0.015 = $15.00

  使用级联:
  700 × Haiku ($0.00025)  = $0.175
  250 × Sonnet ($0.003)   = $0.750
   50 × Opus ($0.015)     = $0.750
  总计                     = $1.675

  节省: 89%
  质量损失: <2% (仅在路由错误时)
```

### 3. 路由与编排模式

```
模式 A: 意图路由
  ┌──────┐    ┌──────────┐    ┌──────────┐
  │ 查询  │───>│ 意图分类器│───>│ 专用管道  │
  └──────┘    └──────────┘    │ (RAG/    │
                              │  Agent/  │
                              │  Direct) │
                              └──────────┘

模式 B: 质量路由 (自动升级)
  查询 ──> [快速模型] ──> [质量检查] ──OK──> 响应
                                │
                              不OK
                                │
                                v
                          [强力模型] ──> 响应

模式 C: 并行执行 + 最优选择
  查询 ──┬──> [模型 A] ──┐
         ├──> [模型 B] ──┼──> [评判器] ──> 最佳响应
         └──> [模型 C] ──┘

模式 D: 管道式编排
  查询 ──> [摘要] ──> [分析] ──> [生成] ──> [校验] ──> 响应
```

---

## DSPy：复合系统优化框架

### 概述

DSPy（Stanford NLP）的核心理念是 **"编程而非提示"（Programming, not prompting）**。它将提示工程转化为可优化的编程问题。

### 核心抽象

```
┌──────────────────────────────────────────────┐
│              DSPy 核心概念                     │
│                                               │
│  Signatures (签名):                           │
│    声明式的输入/输出规范                        │
│    "question -> answer"                       │
│    "context, question -> reasoning, answer"   │
│                                               │
│  Modules (模块):                              │
│    提示策略的抽象                              │
│    - dspy.Predict        (直接预测)            │
│    - dspy.ChainOfThought (思维链)             │
│    - dspy.ReAct          (推理+行动)          │
│    - dspy.Reasoning      (推理模型原生支持)    │
│                                               │
│  Optimizers (优化器):                         │
│    自动优化提示/权重                           │
│    - MIPROv2 (~$2, 20分钟自动优化)            │
│    - BootstrapFewShot (自动选择示例)          │
│    - GRPO / SFT (权重优化)                    │
└──────────────────────────────────────────────┘
```

### DSPy 代码示例

```python
import dspy

# 1. 配置 LLM
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# 2. 定义签名
class FactualQA(dspy.Signature):
    """基于上下文回答问题，并提供推理过程"""
    context: str = dspy.InputField(desc="相关文档")
    question: str = dspy.InputField(desc="用户问题")
    reasoning: str = dspy.OutputField(desc="推理过程")
    answer: str = dspy.OutputField(desc="最终答案")

# 3. 构建复合系统
class RAGSystem(dspy.Module):
    def __init__(self, num_passages=3):
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate = dspy.ChainOfThought(FactualQA)

    def forward(self, question: str):
        # 检索相关文档
        context = self.retrieve(question).passages
        # 生成答案
        result = self.generate(
            context="\n".join(context),
            question=question
        )
        return result

# 4. 自动优化
from dspy.teleprompt import MIPROv2

# 准备训练数据
trainset = [
    dspy.Example(
        question="什么是 KV 缓存?",
        answer="KV 缓存存储注意力层的键值对..."
    ).with_inputs("question"),
    # ... 更多示例
]

# 运行优化器 (~$2, 20分钟)
optimizer = MIPROv2(metric=exact_match, num_threads=4)
optimized_rag = optimizer.compile(
    RAGSystem(),
    trainset=trainset,
    max_bootstrapped_demos=4,
    max_labeled_demos=4,
)

# 5. 使用优化后的系统
result = optimized_rag(question="什么是前缀缓存?")
print(result.answer)
```

### DSPy 2.x 新特性

```python
# DSPy 2.x: 原生推理模型支持
class ComplexReasoning(dspy.Module):
    def __init__(self):
        # dspy.Reasoning 自动利用推理模型的能力
        self.reason = dspy.Reasoning(
            "problem -> solution",
            max_thinking_tokens=4096
        )

    def forward(self, problem):
        return self.reason(problem=problem)
```

---

## RAG 管道完整代码示例

```python
"""生产级 RAG 管道实现"""
from typing import List, Optional
import numpy as np

class ProductionRAG:
    """包含混合检索、重排序和验证的 RAG 管道"""

    def __init__(self, llm, embedder, vector_store, bm25_index):
        self.llm = llm
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_index = bm25_index

    async def query(self, question: str, top_k: int = 5) -> dict:
        """完整的 RAG 查询流程"""

        # Step 1: 查询改写 (可选)
        rewritten_queries = await self._rewrite_query(question)

        # Step 2: 混合检索
        dense_results = await self._dense_retrieve(
            rewritten_queries, top_k=top_k * 2
        )
        sparse_results = await self._sparse_retrieve(
            question, top_k=top_k * 2
        )

        # Step 3: 融合 (Reciprocal Rank Fusion)
        merged = self._rrf_fusion(dense_results, sparse_results)

        # Step 4: 重排序
        reranked = await self._rerank(question, merged[:top_k * 2])

        # Step 5: 上下文构建
        context = self._build_context(reranked[:top_k])

        # Step 6: 生成
        answer = await self._generate(question, context)

        # Step 7: 验证
        validated = await self._validate(question, answer, context)

        return {
            "answer": validated["answer"],
            "sources": reranked[:top_k],
            "confidence": validated["confidence"]
        }

    async def _rewrite_query(self, question: str) -> List[str]:
        """生成多个查询变体以提高召回率"""
        prompt = f"""为以下问题生成3个不同角度的搜索查询:
        问题: {question}
        输出格式: 每行一个查询"""
        result = await self.llm.generate(prompt)
        queries = [question] + result.strip().split("\n")
        return queries

    async def _dense_retrieve(
        self, queries: List[str], top_k: int
    ) -> List[dict]:
        """向量检索"""
        all_results = []
        for q in queries:
            embedding = self.embedder.encode(q)
            results = self.vector_store.search(embedding, top_k=top_k)
            all_results.extend(results)
        # 去重
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
        """BM25 稀疏检索"""
        return self.bm25_index.search(query, top_k=top_k)

    def _rrf_fusion(
        self,
        dense: List[dict],
        sparse: List[dict],
        k: int = 60
    ) -> List[dict]:
        """Reciprocal Rank Fusion 融合多路检索结果"""
        scores = {}
        for rank, doc in enumerate(dense):
            scores[doc["id"]] = scores.get(doc["id"], 0)
            scores[doc["id"]] += 1.0 / (k + rank + 1)
        for rank, doc in enumerate(sparse):
            scores[doc["id"]] = scores.get(doc["id"], 0)
            scores[doc["id"]] += 1.0 / (k + rank + 1)

        # 合并文档并按 RRF 分数排序
        all_docs = {d["id"]: d for d in dense + sparse}
        sorted_ids = sorted(scores, key=scores.get, reverse=True)
        return [all_docs[id] for id in sorted_ids if id in all_docs]

    async def _rerank(
        self, query: str, documents: List[dict]
    ) -> List[dict]:
        """交叉编码器重排序"""
        pairs = [(query, doc["text"]) for doc in documents]
        scores = self.reranker.predict(pairs)
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = score
        return sorted(documents, key=lambda x: x["rerank_score"],
                      reverse=True)

    def _build_context(self, documents: List[dict]) -> str:
        """构建生成器的输入上下文"""
        context_parts = []
        for i, doc in enumerate(documents):
            context_parts.append(
                f"[来源 {i+1}] ({doc.get('title', 'Unknown')})\n"
                f"{doc['text']}\n"
            )
        return "\n---\n".join(context_parts)

    async def _generate(self, question: str, context: str) -> str:
        """基于上下文生成答案"""
        prompt = f"""基于以下参考资料回答问题。如果参考资料不包含答案，请明确说明。
请在回答中引用来源编号。

参考资料:
{context}

问题: {question}

回答:"""
        return await self.llm.generate(prompt)

    async def _validate(
        self, question: str, answer: str, context: str
    ) -> dict:
        """验证答案的事实一致性"""
        prompt = f"""检查以下回答是否与参考资料一致。
        
参考资料: {context}
问题: {question}
回答: {answer}

评估:
1. 是否有事实错误? (是/否)
2. 是否有幻觉内容? (是/否)
3. 置信度 (0-1):"""

        validation = await self.llm.generate(prompt)
        # 解析验证结果
        confidence = self._parse_confidence(validation)

        return {
            "answer": answer,
            "confidence": confidence,
            "validation": validation
        }
```

---

## 评估挑战

### 端到端 vs 组件评估

```
评估复杂度:

  单一 LLM:
    评估维度: 1 (输出质量)

  RAG 系统:
    评估维度: 5+
    ├── 检索质量 (Recall@K, MRR, NDCG)
    ├── 重排序质量 (NDCG after rerank)
    ├── 上下文相关性 (Context Relevance)
    ├── 生成质量 (Faithfulness, Answer Relevance)
    └── 端到端 (用户满意度)

  问题:
  1. 端到端指标好 ≠ 每个组件都好 (可能靠运气补偿)
  2. 组件级指标好 ≠ 端到端好 (组合可能产生问题)
  3. 错误归因困难: 答案错了,是检索还是生成的问题?
```

### 评估框架

| 框架 | 关注点 | 特点 |
|------|--------|------|
| **RAGAS** | RAG 系统评估 | 无参考评估（Faithfulness, Relevance） |
| **LlamaIndex** | 检索+生成 | 内置多种 RAG 评估指标 |
| **DSPy Assert** | 模块约束 | 编程式断言用于优化 |
| **DeepEval** | 端到端 | LLM-as-judge + 多指标 |

```python
# RAGAS 评估示例
from ragas import evaluate
from ragas.metrics import (
    faithfulness,       # 答案对上下文的忠实度
    answer_relevancy,   # 答案与问题的相关性
    context_precision,  # 上下文的精确度
    context_recall,     # 上下文的召回率
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

## 性能优化

### 延迟优化

```
复合系统延迟分解:

  顺序执行:
  [查询改写] ──> [检索] ──> [重排序] ──> [生成] ──> [验证]
     200ms        100ms      150ms       1500ms      800ms
                                                   = 2750ms

  优化后 (并行化 + 缓存):
  [查询改写 + 检索并行] ──> [重排序] ──> [生成(流式)]
          250ms               150ms        1500ms
                                         = 1900ms

  进一步优化 (推测性执行):
  [检索] ──> [重排序 + 投机生成并行] ──> [验证/修正]
   100ms          1500ms                    200ms
                                          = 1800ms

  优化技巧:
  1. 并行化独立步骤
  2. 检索结果缓存 (相似查询命中)
  3. 流式传输减少感知延迟
  4. 推测性执行 + 验证
  5. 组件级缓存 (嵌入缓存, 重排序缓存)
```

### 成本优化

```
成本优化策略:

  1. 模型选择优化
     - 路由/分类: 小模型 (Haiku)
     - 生成: 中等模型 (Sonnet)
     - 验证: 仅在低置信度时用强模型

  2. 缓存策略
     - 嵌入缓存: 减少 60-80% 嵌入调用
     - 语义缓存: 相似查询直接返回缓存结果
     - 提示缓存: 系统提示 + 模板缓存

  3. 批处理
     - 合并多个查询的嵌入请求
     - 批量重排序

  4. 上下文压缩
     - 截断不相关的检索结果
     - 抽取式摘要减少上下文长度
```

---

## 当前最先进水平 (2025-2026)

1. **RAG 进化为"上下文引擎"**：从简单检索到智能检索（自适应、纠正、多跳）
2. **领域专用复合栈**：法律、医疗、金融各有专用组件组合
3. **端到端优化**：DSPy 等框架实现全栈自动优化
4. **复合系统即智能体**：复合系统和智能体的界限模糊化
5. **MCP 作为集成标准**：[[mcp-protocol]] 标准化组件间通信

---

## 挑战

1. **设计复杂度**：没有通用最优的组件选择方案
2. **非可微组件**：检索器、工具等不可微组件阻碍梯度优化
3. **错误归因**：跨组件的错误难以定位根因
4. **延迟累积**：顺序依赖的组件导致延迟叠加
5. **运维复杂度**：多组件系统的监控、调试和维护成本高
6. **版本管理**：组件独立更新可能破坏整体行为

---

## 参考文献

- Zaharia et al., "The Shift from Models to Compound AI Systems," BAIR Blog, Feb 2024
- Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," NeurIPS 2020
- Khattab et al., "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines," ICLR 2024
- Chen et al., "FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance," 2023
- Asai et al., "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection," ICLR 2024
- Yan et al., "Corrective Retrieval Augmented Generation," 2024
- Es et al., "RAGAS: Automated Evaluation of Retrieval Augmented Generation," 2024
- Gartner, "Predicts 2025: AI Agents Are the Next Frontier," 2024

---

## 相关页面

- [[ai-agent-overview]] -- 构成复合系统的智能体架构
- [[agent-serving-challenges]] -- 服务复合系统的挑战
- [[mcp-protocol]] -- 组件集成标准
- [[function-calling-optimization]] -- 优化 LLM 与工具交互
- [[multi-turn-optimization]] -- 多轮对话优化
- [[kv-cache-optimization]] -- KV 缓存管理
- [[long-context-serving]] -- 长上下文服务
