# Shiki's Knowledge Wiki — Schema (CN)

This is the schema file that defines how the LLM should maintain this wiki vault.

## Vault Purpose

个人知识库，涵盖：LLM 推理与服务、强化学习基础设施、机器学习基础设施、机器学习系统、智能体强化学习、AI 智能体、面向 AI 智能体的 LLM 服务。

## Language

本 Vault 使用**中文**维护。对应的英文 Vault 位于 `../EN/`。
更新本 Vault 时，必须同时更新 EN Vault。

## Folder Structure

```
CN/
├── CLAUDE.md          # 本 Schema 文件
├── index.md           # 按类别组织的内容目录
├── log.md             # 仅追加的时间顺序日志
├── sources/           # 原始不可变来源
│   ├── papers/        # 学术论文
│   ├── articles/      # 网络文章、博客
│   ├── notes/         # 个人笔记
│   └── code/          # 代码片段、仓库引用
└── wiki/              # LLM 维护的页面
    ├── llm-inference/       # LLM 推理与服务
    ├── rl-infra/            # 强化学习基础设施
    ├── ml-infra/            # 机器学习基础设施
    ├── ml-sys/              # 机器学习系统
    ├── agentic-rl/          # 智能体强化学习
    ├── ai-agent/            # AI 智能体
    └── llm-serving-for-agents/  # 面向智能体的 LLM 服务
```

## Wiki 页面格式

```markdown
---
title: 页面标题
category: llm-inference | rl-infra | ml-infra | ml-sys | agentic-rl | ai-agent | llm-serving-for-agents
tags: [标签1, 标签2]
created: YYYY-MM-DD
updated: YYYY-MM-DD
status: seed | growing | mature
---

# 页面标题

## 概述
主题简要总结。

## 核心概念
详细解释。

## 相关工作
使用 [[wikilinks]] 链接相关页面。

## 参考文献
- 来源引用及链接。
```

## 操作

### 摄入 (Ingest)
1. 将原始来源放入 `sources/` 对应子文件夹
2. 在 `wiki/` 中创建或更新相关页面
3. 使用 `[[wikilinks]]` 添加交叉引用
4. 更新 `index.md`
5. 在 `log.md` 中追加带时间戳的记录

### 查询 (Query)
1. 搜索 wiki 页面获取相关信息
2. 综合答案并引用 wiki 页面
3. 如果产生有价值的新综合内容，将其归档为新 wiki 页面

### 检查 (Lint)
1. 检查页面间的矛盾
2. 识别需要更新的过时内容
3. 查找孤立页面（无入链）
4. 标记缺失的交叉引用
5. 验证来源引用有效性

## 交叉引用约定
- 使用 Obsidian `[[wikilinks]]` 进行内部链接
- 使用 `[[page#section]]` 进行章节链接
- 在 YAML frontmatter 的 `tags` 字段中标记页面
- 类别对应 wiki 子文件夹
