---
title: "Model Context Protocol (MCP)"
category: ai-agent
tags: [mcp, anthropic, protocol, tool-integration, standard, agent-interop, json-rpc, open-standard]
created: 2026-04-13
updated: 2026-05-20
status: mature
---

# Model Context Protocol (MCP)

> [!abstract]+ TL;DR
> An open standard launched by Anthropic in November 2024 to standardize the integration of AI systems with external tools, data sources, and services -- **"USB for AI agents"**. Based on JSON-RPC 2.0, with two transports (stdio for local, Streamable HTTP for production). An MCP server exposes three kinds of capabilities: **Resources** (context to inject), **Tools** (callable functions), and **Prompts** (parameterized templates). Adoption: OpenAI (2025/03), Google DeepMind (2025/04), Microsoft (2025/05); donated to the **Agentic AI Foundation** (Linux Foundation) in 2025/12. 2026: 10,000+ public servers; ChatGPT, Cursor, Gemini, Copilot, and VS Code all support it. Companion protocols: **A2A** (agent-to-agent) and **ACP** -- together they form the interoperability backbone of agentic AI.

## Overview

The Model Context Protocol (MCP) is an open standard launched by Anthropic in November 2024 to standardize how AI systems integrate with external tools, data sources, and services. MCP is called **"USB for AI agents"** -- a universal interface that eliminates the need for bespoke integrations.

### Why MCP?

Before MCP, every AI application needed dedicated integration code for each tool/data source:

```
Before MCP (the N x M integration problem):

  Claude ──── GitHub integration
  Claude ──── Slack integration
  Claude ──── DB integration
  GPT    ──── GitHub integration (different)
  GPT    ──── Slack integration (different)
  GPT    ──── DB integration (different)

  N models x M tools = N*M integrations

After MCP (N + M):

  Claude ──┐                ┌── GitHub MCP Server
  GPT    ──┼── MCP protocol ┼── Slack MCP Server
  Gemini ──┘                └── DB MCP Server

  N models + M servers = N+M integrations
```

### Core value

| Feature | Description |
|---------|-------------|
| **Standardization** | Unified tool-integration protocol -- implement once, use everywhere |
| **Decoupling** | AI applications are fully decoupled from tool implementations |
| **Ecosystem** | 10,000+ public MCP servers (2026) |
| **Security** | Standardized authentication and authorization framework |
| **Discoverable** | Servers self-describe their capabilities, clients auto-discover |

---

## Architecture

### Overall architecture

```
┌─────────────────────────────────────────────────────┐
│                    MCP architecture                  │
│                                                      │
│  ┌──────────────────────┐                            │
│  │     MCP Host         │  (e.g. Claude Desktop,    │
│  │  ┌────────────────┐  │   IDE, AI application)    │
│  │  │   MCP Client   │  │                            │
│  │  │  (protocol     │  │                            │
│  │  │   client)      │  │                            │
│  │  └───────┬────────┘  │                            │
│  └──────────┼───────────┘                            │
│             │                                        │
│             │  JSON-RPC 2.0                          │
│             │  (stdio / Streamable HTTP)             │
│             │                                        │
│  ┌──────────┼───────────┐                            │
│  │     MCP Server       │  (tool / data provider)   │
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

### Key roles

| Role | Responsibility | Examples |
|------|----------------|----------|
| **Host** | The application that initiates a connection | Claude Desktop, Cursor, VS Code |
| **Client** | Maintains a 1:1 connection with a server | Protocol implementation embedded in the Host |
| **Server** | Exposes capabilities to clients | GitHub MCP Server, DB MCP Server |

### Transport layer

MCP is based on **JSON-RPC 2.0** and supports two transports:

#### 1. stdio transport (local/desktop)

```
┌────────────┐     stdin      ┌─────────────┐
│  MCP Host  │ ──────────────>│  MCP Server │
│  (client)  │                │ (subprocess)│
│            │<───────────────│             │
└────────────┘     stdout     └─────────────┘

Properties:
- Client launches the server as a subprocess
- Communicates via stdin/stdout
- Simple, safe, zero network configuration
- Ideal for desktop applications (Claude Desktop, IDE)
```

#### 2. Streamable HTTP transport (production / Web)

```
┌────────────┐    HTTP POST     ┌─────────────┐
│  MCP Host  │ ───────────────> │  MCP Server │
│  (client)  │                  │ (remote     │
│            │ <─── SSE ─────── │   service)  │
└────────────┘  (resumable stream)└──────────┘

Properties:
- HTTP-based remote communication
- SSE (Server-Sent Events) for server push
- Supports resumable streams (reconnect on disconnect)
- Suitable for production, cloud deployment
- Introduced by the March 2025 spec update, replacing the old HTTP+SSE
```

---

## Core primitives

An MCP server exposes capabilities through three core primitives:

### 1. Tools

Functions the model can call. This is the most-used primitive.

```json
// Tool-definition example
{
  "name": "search_files",
  "description": "Search a codebase for matching files",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Search keyword"
      },
      "path": {
        "type": "string",
        "description": "Search path",
        "default": "."
      }
    },
    "required": ["query"]
  }
}
```

```json
// Tool-call example
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

### 2. Resources

Data that can be injected into the context, with loading controlled by the application (not the model).

```json
// Resource-list response
{
  "resources": [
    {
      "uri": "file:///project/README.md",
      "name": "Project README",
      "mimeType": "text/markdown"
    },
    {
      "uri": "db://users/schema",
      "name": "Users table schema",
      "mimeType": "application/json"
    }
  ]
}
```

**Resources vs Tools**:
| Dimension | Resources | Tools |
|-----------|-----------|-------|
| Controlled by | Application / user | Model (AI) |
| Analogy | GET request (read data) | POST request (perform action) |
| Side effects | None | May have |
| When to use | Building context | Executing a task |

### 3. Prompts (prompt templates)

Parameterized prompt templates used to standardize interaction patterns.

```json
// Prompt-template definition
{
  "name": "code_review",
  "description": "Review code",
  "arguments": [
    {
      "name": "code",
      "description": "Code to review",
      "required": true
    },
    {
      "name": "language",
      "description": "Programming language",
      "required": false
    }
  ]
}
```

### 4. Sampling (reverse request)

Lets the server reverse-request text generation from the LLM. This is a unique capability of MCP.

```
Normal flow:  Host -> Client -> Server -> [execute tool]
Sampling:     Server -> Client -> Host -> [LLM generates] -> result returned to Server

Use cases:
- The server needs the AI's help in making a decision
- Sub-agent calls in an agent system
- Complex tool logic that needs LLM judgment
```

---

## Message flow in detail

### Full message flow of a tool call

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
    [User sends a message, LLM decides to call a tool]    │
       │                          │                       │
       │  9. tools/call           │                       │
       │     {name: "create_pr",  │                       │
       │      args: {...}}        │                       │
       │ ─────────────────────────>                       │
       │                          │  10. tools/call       │
       │                          │ ──────────────────────>
       │                          │                       │
       │                          │  [execute GitHub API] │
       │                          │                       │
       │                          │  11. result           │
       │                          │ <──────────────────────
       │  12. tool result         │                       │
       │ <─────────────────────────                       │
       │                          │                       │
    [LLM processes the result and produces a reply]       │
       │                          │                       │
```

### Lifecycle management

```
Initialization phase:
  Client                    Server
    │── initialize ──────────>│
    │<── capabilities ────────│
    │── initialized ─────────>│  (confirm completion)
    │                         │

Run phase:
    │── tools/call ──────────>│
    │<── result ──────────────│
    │── resources/read ──────>│
    │<── content ─────────────│
    │                         │

Shutdown phase:
    │── shutdown ─────────────>│  (graceful shutdown)
    │                          │
```

---

## MCP vs function calling

### Core differences

| Dimension | MCP | Function calling (OpenAI style) |
|-----------|-----|--------------------------------|
| **Protocol standard** | Open standard, cross-model | Vendor-specific API |
| **Runs as** | Independent process / service | Inside application code |
| **Discovery** | Dynamically discover server capabilities | Statically defined in the request |
| **State management** | Server maintains its own state | Stateless functions |
| **Ecosystem** | 10,000+ public servers | Each app implements its own |
| **Transport** | stdio / HTTP (cross-process) | In-process function call |
| **Complexity** | Higher (requires a server process) | Lower (simple function) |

### Use-case comparison

```
Function calling fits:
  - Simple, stateless tools (calculator, format conversion)
  - Tools inside a single application
  - Rapid prototyping

MCP fits:
  - Complex, stateful integrations (database, version control)
  - Tools shared across applications
  - Enterprise-grade production
  - Scenarios that need dynamic discovery and hot-swap
```

---

## Implementation examples

### Python MCP server

```python
"""A simple filesystem MCP-server example"""
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import os

# Create the server instance
server = Server("filesystem-server")

@server.list_tools()
async def list_tools() -> list[Tool]:
    """Declare the tools this server provides"""
    return [
        Tool(
            name="read_file",
            description="Read the contents of a file at the given path",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path"
                    }
                },
                "required": ["path"]
            }
        ),
        Tool(
            name="list_directory",
            description="List the contents of a directory",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path"
                    }
                },
                "required": ["path"]
            }
        ),
        Tool(
            name="write_file",
            description="Write content to a file",
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
    """Handle a tool call"""
    if name == "read_file":
        path = arguments["path"]
        if not os.path.exists(path):
            return [TextContent(type="text", text=f"Error: file {path} does not exist")]
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
        return [TextContent(type="text", text=f"Wrote {len(content)} characters to {path}")]

    raise ValueError(f"Unknown tool: {name}")

# Start the server
async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### TypeScript MCP server

```typescript
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";

const server = new Server({
  name: "weather-server",
  version: "1.0.0"
}, {
  capabilities: { tools: {} }
});

// Declare tools
server.setRequestHandler("tools/list", async () => ({
  tools: [{
    name: "get_weather",
    description: "Get weather for the specified city",
    inputSchema: {
      type: "object",
      properties: {
        city: { type: "string", description: "City name" }
      },
      required: ["city"]
    }
  }]
}));

// Handle tool calls
server.setRequestHandler("tools/call", async (request) => {
  if (request.params.name === "get_weather") {
    const city = request.params.arguments.city;
    // In production this would call a real weather API
    const weather = await fetchWeather(city);
    return {
      content: [{
        type: "text",
        text: `${city}: ${weather.temp}°C, ${weather.condition}`
      }]
    };
  }
  throw new Error(`Unknown tool: ${request.params.name}`);
});

// Start
const transport = new StdioServerTransport();
await server.connect(transport);
```

### Client configuration (Claude Desktop)

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

## MCP ecosystem

### Official and popular MCP servers

| Category | Server name | Function |
|----------|-------------|----------|
| **Filesystem** | @mcp/filesystem | Safe file read/write operations |
| **Version control** | @mcp/git, @mcp/github | Git ops, PR, issue management |
| **Database** | @mcp/postgres, @mcp/sqlite | Database query and management |
| **Search** | @mcp/brave-search | Web search |
| **Communication** | @mcp/slack | Slack messages and channel management |
| **Cloud** | @mcp/aws, @mcp/gcp | Cloud-resource management |
| **Monitoring** | @mcp/sentry | Error tracking and monitoring |
| **Development** | @mcp/puppeteer | Browser automation |

### Ecosystem size (2026)

```
MCP ecosystem growth:

server count
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

Supported clients:
  Claude Desktop, Cursor, VS Code, Windsurf,
  ChatGPT, Gemini, GitHub Copilot, JetBrains, ...
```

---

## Adoption timeline

| Date | Event |
|------|-------|
| 2024/11 | Anthropic launches MCP, open-sources the spec and SDKs |
| 2025/01 | Early adopters: Cursor, Windsurf integrate |
| 2025/03 | OpenAI announces full-product-line adoption of MCP; Streamable HTTP transport released |
| 2025/04 | Google DeepMind confirms Gemini support; 8M+ downloads |
| 2025/05 | Microsoft joins the steering committee |
| 2025/08 | Authentication/authorization spec (OAuth 2.1) released |
| 2025/12 | Donated to the **Agentic AI Foundation (AAIF)**, hosted by the Linux Foundation |
| 2026 | 10,000+ public servers; becomes the de facto standard |

---

## MCP + A2A + ACP protocol matrix

```
┌──────────────────────────────────────────────────┐
│           Agent-protocol ecosystem                │
│                                                   │
│  ┌─────────┐    ┌──────────┐    ┌─────────────┐  │
│  │   MCP   │    │   A2A    │    │  ACP (merged)│  │
│  │Anthropic│    │  Google  │    │    IBM       │  │
│  └────┬────┘    └────┬─────┘    └──────┬──────┘  │
│       │              │                  │         │
│  agent ↔ tool    agent ↔ agent     (merged into A2A) │
│                                                   │
│  Example:                                         │
│  [User] -> [Main agent]                           │
│             │                                     │
│             ├── MCP ──> [GitHub Server]           │
│             ├── MCP ──> [Database Server]         │
│             └── A2A ──> [Research agent]          │
│                          │                        │
│                          └── MCP ──> [Search Server]│
└──────────────────────────────────────────────────┘
```

| Protocol | Position | Initiator | Standards body |
|----------|----------|-----------|----------------|
| **MCP** | agent ↔ tool/data | Anthropic | AAIF (Linux Foundation) |
| **A2A** | agent ↔ agent | Google | Linux Foundation |
| **ACP** | agent ↔ agent | IBM | merged into A2A |

Together they form an extensible, decentralized agentic-AI infrastructure.

---

## Security considerations

### Threat model

```
Security concerns:

1. Code-execution risk
   MCP servers can execute arbitrary code
   ──> sandbox isolation + permission control

2. Data leakage
   Tools may access sensitive data
   ──> least-privilege principle + data masking

3. Prompt injection
   Malicious tool returns may manipulate LLM behavior
   ──> input validation + output auditing

4. Authentication
   Unified authentication across servers
   ──> OAuth 2.1 standard (2025/08 spec)

5. Supply-chain attack
   Malicious MCP servers
   ──> server-signature verification + registry audit
```

### Security best practices

```python
# Example secure configuration
{
    "mcpServers": {
        "filesystem": {
            "command": "python",
            "args": ["fs_server.py"],
            "env": {
                # Restrict accessible paths
                "ALLOWED_PATHS": "/Users/shiki/projects",
                # Read-only mode
                "READ_ONLY": "true"
            },
            # Declared permissions
            "permissions": {
                "file_read": ["*.py", "*.md"],
                "file_write": [],  # forbid writing
                "network": false   # forbid network access
            }
        }
    }
}
```

---

## Limitations

1. **Complexity overhead**: compared to simple function calling, requires additional process management and communication overhead
2. **Performance**: JSON-RPC has serialization/deserialization overhead under high call frequency
3. **Rapidly evolving spec**: 3 major revisions during 2025; early implementations may need frequent updates
4. **Debugging difficulty**: cross-process communication adds debugging complexity
5. **Immature security model**: trust mechanisms are still developing (the OAuth 2.1 spec was only released in 2025/08)
6. **Uneven server quality**: the quality and maintenance of community-contributed servers varies

---

## References

- Anthropic, "Model Context Protocol Specification," https://modelcontextprotocol.io
- Anthropic, "Introducing the Model Context Protocol," Nov 2024
- MCP GitHub Repository, https://github.com/modelcontextprotocol
- OpenAI, "Adding MCP support to the Agents SDK," Mar 2025
- Google DeepMind, "Gemini MCP Integration," Apr 2025
- Linux Foundation, "Agentic AI Foundation," Dec 2025
- Google, "Agent-to-Agent (A2A) Protocol," Apr 2025
- IBM, "Agent Communication Protocol (ACP)," 2025

---

## Related pages

- [[tool-use]] -- tool use standardized by MCP
- [[agent-frameworks]] -- frameworks integrating MCP
- [[ai-agent-overview]] -- overview of agent architectures
- [[compound-ai-systems]] -- compound systems built on MCP
- [[environment-design]] -- ORS extending MCP for RL
- [[agent-serving-challenges]] -- impact of tool calls on serving
