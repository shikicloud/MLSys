---
title: "模型上下文协议 (MCP)"
category: ai-agent
tags: [mcp, anthropic, 协议, 工具集成, 标准, 智能体互操作, json-rpc, 开放标准]
created: 2026-04-13
updated: 2026-04-14
status: mature
---

# 模型上下文协议 (MCP)

## 概述

模型上下文协议（Model Context Protocol, MCP）是 Anthropic 于 2024 年 11 月推出的开放标准，旨在标准化 AI 系统与外部工具、数据源和服务的集成方式。MCP 被称为 **"AI 智能体的 USB"** -- 一个消除定制集成需求的通用接口。

### 为什么需要 MCP？

在 MCP 之前，每个 AI 应用都需要为每个工具/数据源编写专门的集成代码：

```
MCP 之前 (N x M 集成问题):

  Claude ──── GitHub 集成代码
  Claude ──── Slack 集成代码
  Claude ──── DB 集成代码
  GPT    ──── GitHub 集成代码 (不同)
  GPT    ──── Slack 集成代码 (不同)
  GPT    ──── DB 集成代码 (不同)

  N 个模型 x M 个工具 = N*M 个集成

MCP 之后 (N + M):

  Claude ──┐                ┌── GitHub MCP Server
  GPT    ──┼── MCP 协议 ────┼── Slack MCP Server
  Gemini ──┘                └── DB MCP Server

  N 个模型 + M 个服务器 = N+M 个集成
```

### 核心价值

| 特性 | 说明 |
|------|------|
| **标准化** | 统一的工具集成协议，一次实现到处使用 |
| **解耦** | AI 应用与工具实现完全分离 |
| **生态系统** | 10,000+ 公共 MCP 服务器（2026） |
| **安全** | 标准化的认证和授权框架 |
| **可发现** | 服务器能力自描述，客户端自动发现 |

---

## 架构

### 整体架构

```
┌─────────────────────────────────────────────────────┐
│                    MCP 架构                          │
│                                                      │
│  ┌──────────────────────┐                            │
│  │     MCP Host         │  (如 Claude Desktop,       │
│  │  ┌────────────────┐  │   IDE, AI 应用)            │
│  │  │   MCP Client   │  │                            │
│  │  │  (协议客户端)    │  │                            │
│  │  └───────┬────────┘  │                            │
│  └──────────┼───────────┘                            │
│             │                                        │
│             │  JSON-RPC 2.0                           │
│             │  (stdio / Streamable HTTP)              │
│             │                                        │
│  ┌──────────┼───────────┐                            │
│  │     MCP Server       │  (工具/数据提供者)          │
│  │  ┌───────┴────────┐  │                            │
│  │  │   Protocol     │  │                            │
│  │  │   Handler      │  │                            │
│  │  └───────┬────────┘  │                            │
│  │          │            │                            │
│  │  ┌───┬──┴──┬───┐    │                            │
│  │  │Tools│Resources│Prompts│                        │
│  │  └───┘  └───┘  └───┘    │                        │
│  └──────────────────────────┘                        │
└─────────────────────────────────────────────────────┘
```

### 关键角色

| 角色 | 职责 | 示例 |
|------|------|------|
| **Host** | 发起连接的应用程序 | Claude Desktop, Cursor, VS Code |
| **Client** | 维护与服务器的 1:1 连接 | 嵌入在 Host 中的协议实现 |
| **Server** | 暴露能力给客户端 | GitHub MCP Server, DB MCP Server |

### 传输层

MCP 基于 **JSON-RPC 2.0** 协议，支持两种传输方式：

#### 1. stdio 传输（本地/桌面）

```
┌────────────┐     stdin      ┌─────────────┐
│  MCP Host  │ ──────────────>│  MCP Server │
│  (客户端)   │               │  (子进程)    │
│            │<───────────────│             │
└────────────┘     stdout     └─────────────┘

特点:
- 客户端启动服务器作为子进程
- 通过 stdin/stdout 通信
- 简单、安全、零网络配置
- 适合桌面应用 (Claude Desktop, IDE)
```

#### 2. Streamable HTTP 传输（生产/Web）

```
┌────────────┐    HTTP POST     ┌─────────────┐
│  MCP Host  │ ───────────────> │  MCP Server │
│  (客户端)   │                 │  (远程服务)  │
│            │ <─── SSE ─────── │             │
└────────────┘   (可恢复流)     └─────────────┘

特点:
- 基于 HTTP 的远程通信
- SSE (Server-Sent Events) 实现服务器推送
- 支持可恢复流（断线重连）
- 适合生产环境、云部署
- 2025 年 3 月规范更新引入，替代旧版 HTTP+SSE
```

---

## 核心原语 (Primitives)

MCP 服务器通过三种核心原语暴露能力：

### 1. Tools（工具）

模型可以调用的函数。这是最常用的原语。

```json
// 工具定义示例
{
  "name": "search_files",
  "description": "在代码库中搜索匹配的文件",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "搜索关键词"
      },
      "path": {
        "type": "string",
        "description": "搜索路径",
        "default": "."
      }
    },
    "required": ["query"]
  }
}
```

```json
// 工具调用示例
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "search_files",
    "arguments": {
      "query": "authentication",
      "path": "./src"
    }
  }
}
```

### 2. Resources（资源）

可注入上下文的数据，由应用程序（而非模型）控制加载。

```json
// 资源列表响应
{
  "resources": [
    {
      "uri": "file:///project/README.md",
      "name": "项目说明",
      "mimeType": "text/markdown"
    },
    {
      "uri": "db://users/schema",
      "name": "用户表结构",
      "mimeType": "application/json"
    }
  ]
}
```

**Resources vs Tools 的区别**：
| 维度 | Resources | Tools |
|------|-----------|-------|
| 控制方 | 应用程序/用户 | 模型 (AI) |
| 类比 | GET 请求（读取数据） | POST 请求（执行操作） |
| 副作用 | 无 | 可能有 |
| 使用时机 | 构建上下文 | 执行任务 |

### 3. Prompts（提示模板）

参数化的提示模板，用于标准化交互模式。

```json
// 提示模板定义
{
  "name": "code_review",
  "description": "对代码进行审查",
  "arguments": [
    {
      "name": "code",
      "description": "要审查的代码",
      "required": true
    },
    {
      "name": "language",
      "description": "编程语言",
      "required": false
    }
  ]
}
```

### 4. Sampling（采样 -- 逆向请求）

允许服务器反向请求 LLM 生成文本。这是 MCP 的独特能力。

```
正常流程:   Host -> Client -> Server -> [执行工具]
Sampling:   Server -> Client -> Host -> [LLM 生成] -> 结果返回给 Server

使用场景:
- 服务器需要 AI 帮助做决策
- 智能体中的子智能体调用
- 需要 LLM 判断的复杂工具逻辑
```

---

## 消息流详解

### 完整的工具调用消息流

```
Host (Claude Desktop)           Client              Server (GitHub MCP)
       │                          │                       │
       │  1. initialize           │                       │
       │ ─────────────────────────>                       │
       │                          │  2. initialize        │
       │                          │ ──────────────────────>
       │                          │                       │
       │                          │  3. capabilities      │
       │                          │ <──────────────────────
       │  4. server capabilities  │                       │
       │ <─────────────────────────                       │
       │                          │                       │
       │  5. tools/list           │                       │
       │ ─────────────────────────>                       │
       │                          │  6. tools/list        │
       │                          │ ──────────────────────>
       │                          │                       │
       │                          │  7. tool definitions  │
       │                          │ <──────────────────────
       │  8. available tools      │                       │
       │ <─────────────────────────                       │
       │                          │                       │
    [用户发送消息，LLM 决定调用工具]                       │
       │                          │                       │
       │  9. tools/call           │                       │
       │     {name: "create_pr",  │                       │
       │      args: {...}}        │                       │
       │ ─────────────────────────>                       │
       │                          │  10. tools/call       │
       │                          │ ──────────────────────>
       │                          │                       │
       │                          │  [执行 GitHub API]     │
       │                          │                       │
       │                          │  11. result           │
       │                          │ <──────────────────────
       │  12. tool result         │                       │
       │ <─────────────────────────                       │
       │                          │                       │
    [LLM 处理结果，生成回复]                               │
       │                          │                       │
```

### 生命周期管理

```
初始化阶段:
  Client                    Server
    │── initialize ──────────>│
    │<── capabilities ────────│
    │── initialized ─────────>│  (确认完成)
    │                         │

运行阶段:
    │── tools/call ──────────>│
    │<── result ──────────────│
    │── resources/read ──────>│
    │<── content ─────────────│
    │                         │

关闭阶段:
    │── shutdown ─────────────>│  (优雅关闭)
    │                          │
```

---

## MCP vs 函数调用 (Function Calling)

### 核心区别

| 维度 | MCP | 函数调用 (OpenAI style) |
|------|-----|------------------------|
| **协议标准** | 开放标准，跨模型 | 厂商特定 API |
| **运行位置** | 独立进程/服务 | 应用代码内 |
| **发现机制** | 动态发现服务器能力 | 静态定义在请求中 |
| **状态管理** | 服务器维护自己的状态 | 无状态函数 |
| **生态系统** | 10,000+ 公共服务器 | 每个应用自行实现 |
| **传输** | stdio / HTTP (跨进程) | 进程内函数调用 |
| **复杂度** | 较高（需要服务器进程） | 较低（简单函数） |

### 使用场景对比

```
函数调用适合:
  - 简单、无状态的工具（计算器、格式转换）
  - 单应用内的工具
  - 快速原型开发

MCP 适合:
  - 复杂、有状态的集成（数据库、版本控制）
  - 跨应用共享的工具
  - 企业级生产环境
  - 需要动态发现和热插拔的场景
```

---

## 实现示例

### Python MCP 服务器

```python
"""简单的文件系统 MCP 服务器示例"""
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import os

# 创建服务器实例
server = Server("filesystem-server")

@server.list_tools()
async def list_tools() -> list[Tool]:
    """声明服务器提供的工具"""
    return [
        Tool(
            name="read_file",
            description="读取指定路径的文件内容",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "文件路径"
                    }
                },
                "required": ["path"]
            }
        ),
        Tool(
            name="list_directory",
            description="列出目录内容",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "目录路径"
                    }
                },
                "required": ["path"]
            }
        ),
        Tool(
            name="write_file",
            description="写入内容到文件",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["path", "content"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """处理工具调用"""
    if name == "read_file":
        path = arguments["path"]
        if not os.path.exists(path):
            return [TextContent(type="text", text=f"错误: 文件 {path} 不存在")]
        with open(path, "r") as f:
            content = f.read()
        return [TextContent(type="text", text=content)]

    elif name == "list_directory":
        path = arguments["path"]
        entries = os.listdir(path)
        return [TextContent(type="text", text="\n".join(entries))]

    elif name == "write_file":
        path = arguments["path"]
        content = arguments["content"]
        with open(path, "w") as f:
            f.write(content)
        return [TextContent(type="text", text=f"已写入 {len(content)} 字符到 {path}")]

    raise ValueError(f"未知工具: {name}")

# 启动服务器
async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### TypeScript MCP 服务器

```typescript
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";

const server = new Server({
  name: "weather-server",
  version: "1.0.0"
}, {
  capabilities: { tools: {} }
});

// 声明工具
server.setRequestHandler("tools/list", async () => ({
  tools: [{
    name: "get_weather",
    description: "获取指定城市的天气信息",
    inputSchema: {
      type: "object",
      properties: {
        city: { type: "string", description: "城市名称" }
      },
      required: ["city"]
    }
  }]
}));

// 处理工具调用
server.setRequestHandler("tools/call", async (request) => {
  if (request.params.name === "get_weather") {
    const city = request.params.arguments.city;
    // 实际应用中调用天气 API
    const weather = await fetchWeather(city);
    return {
      content: [{
        type: "text",
        text: `${city}: ${weather.temp}°C, ${weather.condition}`
      }]
    };
  }
  throw new Error(`未知工具: ${request.params.name}`);
});

// 启动
const transport = new StdioServerTransport();
await server.connect(transport);
```

### 客户端配置 (Claude Desktop)

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "python",
      "args": ["path/to/filesystem_server.py"],
      "env": {
        "ALLOWED_PATHS": "/Users/shiki/projects"
      }
    },
    "weather": {
      "command": "node",
      "args": ["path/to/weather_server.js"],
      "env": {
        "WEATHER_API_KEY": "your-key"
      }
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "ghp_xxx"
      }
    }
  }
}
```

---

## MCP 生态系统

### 官方与热门 MCP 服务器

| 类别 | 服务器名称 | 功能 |
|------|-----------|------|
| **文件系统** | @mcp/filesystem | 安全的文件读写操作 |
| **版本控制** | @mcp/git, @mcp/github | Git 操作、PR、Issue 管理 |
| **数据库** | @mcp/postgres, @mcp/sqlite | 数据库查询和管理 |
| **搜索** | @mcp/brave-search | 网页搜索 |
| **通信** | @mcp/slack | Slack 消息和频道管理 |
| **云服务** | @mcp/aws, @mcp/gcp | 云资源管理 |
| **监控** | @mcp/sentry | 错误追踪和监控 |
| **开发** | @mcp/puppeteer | 浏览器自动化 |

### 生态规模（2026 年）

```
MCP 生态增长:

服务器数量
  │
10000│                                     ●
 8000│                               ●
 6000│                          ●
 4000│                    ●
 2000│              ●
 1000│         ●
  100│    ●
   10│●
     └──────────────────────────────────
      2024   2025   2025   2025   2026
      Nov    Mar    Jun    Sep

支持的客户端:
  Claude Desktop, Cursor, VS Code, Windsurf,
  ChatGPT, Gemini, GitHub Copilot, JetBrains, ...
```

---

## 采用时间线

| 时间 | 事件 |
|------|------|
| 2024/11 | Anthropic 推出 MCP，开源规范和 SDK |
| 2025/01 | 早期采用者：Cursor, Windsurf 集成 |
| 2025/03 | OpenAI 宣布全产品线采用 MCP；Streamable HTTP 传输发布 |
| 2025/04 | Google DeepMind 确认 Gemini 支持；800 万+ 下载 |
| 2025/05 | Microsoft 加入指导委员会 |
| 2025/08 | 认证/授权规范（OAuth 2.1）发布 |
| 2025/12 | 捐赠给 **Agentic AI Foundation (AAIF)**，Linux Foundation 托管 |
| 2026 | 10,000+ 公共服务器；成为事实标准 |

---

## MCP + A2A + ACP 协议矩阵

```
┌──────────────────────────────────────────────────┐
│           智能体协议生态                           │
│                                                   │
│  ┌─────────┐    ┌──────────┐    ┌─────────────┐  │
│  │   MCP   │    │   A2A    │    │  ACP (合并)  │  │
│  │Anthropic│    │  Google  │    │    IBM       │  │
│  └────┬────┘    └────┬─────┘    └──────┬──────┘  │
│       │              │                  │         │
│  智能体 ↔ 工具   智能体 ↔ 智能体    (已合并到 A2A)  │
│                                                   │
│  示例:                                            │
│  [用户] -> [主智能体]                              │
│             │                                     │
│             ├── MCP ──> [GitHub Server]           │
│             ├── MCP ──> [Database Server]         │
│             └── A2A ──> [研究智能体]               │
│                          │                        │
│                          └── MCP ──> [搜索Server] │
└──────────────────────────────────────────────────┘
```

| 协议 | 定位 | 发起方 | 标准化机构 |
|------|------|--------|-----------|
| **MCP** | 智能体 ↔ 工具/数据 | Anthropic | AAIF (Linux Foundation) |
| **A2A** | 智能体 ↔ 智能体 | Google | Linux Foundation |
| **ACP** | 智能体 ↔ 智能体 | IBM | 已合并入 A2A |

三者共同构成可扩展、去中心化的智能体 AI 基础设施。

---

## 安全考量

### 威胁模型

```
安全关切:

1. 代码执行风险
   MCP Server 可以执行任意代码
   ──> 沙箱隔离 + 权限控制

2. 数据泄露
   工具可能访问敏感数据
   ──> 最小权限原则 + 数据脱敏

3. 提示注入
   恶意工具返回可能操控 LLM 行为
   ──> 输入验证 + 输出审计

4. 认证问题
   跨服务器的统一认证
   ──> OAuth 2.1 标准 (2025/08 规范)

5. 供应链攻击
   恶意 MCP 服务器
   ──> 服务器签名验证 + 注册表审核
```

### 安全最佳实践

```python
# 安全配置示例
{
    "mcpServers": {
        "filesystem": {
            "command": "python",
            "args": ["fs_server.py"],
            "env": {
                # 限制可访问的路径
                "ALLOWED_PATHS": "/Users/shiki/projects",
                # 只读模式
                "READ_ONLY": "true"
            },
            # 权限声明
            "permissions": {
                "file_read": ["*.py", "*.md"],
                "file_write": [],  # 禁止写入
                "network": false   # 禁止网络访问
            }
        }
    }
}
```

---

## 局限性

1. **复杂度开销**：相比简单函数调用，需要额外的进程管理和通信开销
2. **性能**：JSON-RPC 在高频调用场景下有序列化/反序列化开销
3. **规范演进快**：2025 年经历了 3 次重大修订，早期实现可能需要频繁更新
4. **调试困难**：跨进程通信增加了调试复杂度
5. **安全模型不成熟**：信任机制仍在发展中（OAuth 2.1 规范 2025/08 才发布）
6. **服务器质量参差**：社区贡献的服务器质量和维护水平不一

---

## 参考文献

- Anthropic, "Model Context Protocol Specification," https://modelcontextprotocol.io
- Anthropic, "Introducing the Model Context Protocol," Nov 2024
- MCP GitHub Repository, https://github.com/modelcontextprotocol
- OpenAI, "Adding MCP support to the Agents SDK," Mar 2025
- Google DeepMind, "Gemini MCP Integration," Apr 2025
- Linux Foundation, "Agentic AI Foundation," Dec 2025
- Google, "Agent-to-Agent (A2A) Protocol," Apr 2025
- IBM, "Agent Communication Protocol (ACP)," 2025

---

## 相关页面

- [[tool-use]] -- MCP 标准化的工具使用
- [[agent-frameworks]] -- 集成 MCP 的框架
- [[ai-agent-overview]] -- 智能体架构总览
- [[compound-ai-systems]] -- 基于 MCP 构建的复合系统
- [[environment-design]] -- ORS 扩展 MCP 用于 RL
- [[agent-serving-challenges]] -- 工具调用对服务的影响
