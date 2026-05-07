---
title: "Model Context Protocol (MCP)"
category: ai-agent
tags: [mcp, anthropic, protocol, tool-integration, standard, agent-interop, json-rpc, open-standard]
created: 2026-04-13
updated: 2026-05-07
status: mature
---

# Model Context Protocol (MCP)

> [!abstract]+ TL;DR
> Open standard by Anthropic (Nov 2024) standardizing how AI systems integrate with tools, data sources, and services — **"USB for AI agents."** Built on JSON-RPC 2.0 with two transports (stdio for local, Streamable HTTP for production). MCP servers expose three capability types: **Resources** (injectable context), **Tools** (callable functions), **Prompts** (parameterized templates). Adoption: OpenAI (Mar 2025), Google DeepMind (Apr 2025), Microsoft (May 2025); donated to **Agentic AI Foundation** (Linux Foundation, Dec 2025). 2026: 10,000+ public servers; ChatGPT, Cursor, Gemini, Copilot, VS Code all support. Companion protocols: **A2A** (agent-to-agent), **ACP** — together form the agentic-AI interoperability backbone.

```
Before MCP (N x M):          After MCP (N + M):
  Claude --- GitHub code        Claude ──┐           ┌── GitHub MCP
  Claude --- Slack code         GPT    ──┼── MCP ────┼── Slack MCP
  GPT   --- GitHub code (diff) Gemini ──┘           └── DB MCP
  GPT   --- Slack code (diff)
```

---

## Architecture

```
┌────────────────────────────┐
│  MCP Host (Claude Desktop) │
│  ┌──────────────────────┐  │
│  │     MCP Client       │  │
│  └──────────┬───────────┘  │
└─────────────┼──────────────┘
              │ JSON-RPC 2.0
              │ (stdio / Streamable HTTP)
┌─────────────┼──────────────┐
│  MCP Server                │
│  ┌──────────┴───────────┐  │
│  │   Protocol Handler   │  │
│  └───┬──────┬───────┬───┘  │
│  [Tools] [Resources] [Prompts] │
└────────────────────────────┘
```

**Transports**: stdio (local subprocess, desktop apps) and Streamable HTTP (production/web, resumable SSE streaming; replaced legacy HTTP+SSE in Mar 2025).

---

## Core Primitives

### Tools
Model-invocable functions -- the most commonly used primitive.

```json
{
  "name": "search_files",
  "description": "Search codebase for matching files",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {"type": "string"},
      "path": {"type": "string", "default": "."}
    },
    "required": ["query"]
  }
}
```

### Resources
Injectable context data controlled by the application (not the model). Analogous to GET requests -- read-only, no side effects.

### Prompts
Parameterized templates for standardized interaction patterns.

### Sampling
Reverse capability: server requests LLM generation from the host. Enables sub-agent patterns and complex tool logic requiring AI judgment.

---

## Message Flow

```
Host              Client           Server (GitHub MCP)
 │── initialize ──────────────────────>│
 │<── capabilities ────────────────────│
 │── tools/list ──────────────────────>│
 │<── tool definitions ────────────────│
 │                                      │
 [User sends message, LLM decides tool call]
 │── tools/call {name, args} ─────────>│
 │                          [Execute GitHub API]
 │<── result ──────────────────────────│
 [LLM processes result, generates reply]
```

---

## MCP vs Function Calling

| Dimension | MCP | Function Calling (OpenAI-style) |
|-----------|-----|--------------------------------|
| Standard | Open, cross-model | Vendor-specific API |
| Execution | Separate process/service | In-process function |
| Discovery | Dynamic capability discovery | Static definitions in request |
| State | Server maintains own state | Stateless functions |
| Ecosystem | 10,000+ public servers | Per-app implementation |
| Complexity | Higher (process management) | Lower (simple functions) |

---

## Implementation Example (Python)

```python
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

server = Server("filesystem-server")

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="read_file",
            description="Read file contents",
            inputSchema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "read_file":
        with open(arguments["path"]) as f:
            return [TextContent(type="text", text=f.read())]
    raise ValueError(f"Unknown tool: {name}")

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)
```

---

## Adoption Timeline

| Date | Event |
|------|-------|
| Nov 2024 | Anthropic introduces MCP |
| Mar 2025 | OpenAI adopts across products; Streamable HTTP transport |
| Apr 2025 | Google DeepMind confirms Gemini support; 8M+ downloads |
| May 2025 | Microsoft joins steering committee |
| Aug 2025 | OAuth 2.1 auth specification published |
| Dec 2025 | Donated to **Agentic AI Foundation (AAIF)** under Linux Foundation |
| 2026 | 10,000+ public servers; de facto standard |

---

## MCP + A2A + ACP Protocol Matrix

| Protocol | Scope | Originator | Governance |
|----------|-------|-----------|------------|
| **MCP** | Agent <-> Tools/Data | Anthropic | AAIF (Linux Foundation) |
| **A2A** | Agent <-> Agent | Google | Linux Foundation |
| **ACP** | Agent <-> Agent | IBM | Merged into A2A |

Together these form the backbone of scalable, decentralized agentic AI infrastructure.

---

## Security Considerations

Key concerns:
1. **Code execution risk**: MCP servers can execute arbitrary code -- requires sandboxing + permissions
2. **Data leakage**: Tools may access sensitive data -- minimum privilege + data masking
3. **Prompt injection**: Malicious tool responses may manipulate LLM behavior
4. **Authentication**: OAuth 2.1 standard published Aug 2025
5. **Supply chain**: Malicious MCP servers -- requires signature verification + registry auditing

---

## Limitations

- **Complexity overhead** vs simple function calling
- **JSON-RPC performance** overhead for high-frequency calls
- **Rapid spec evolution** (3 major revisions in 2025)
- **Debugging difficulty** across process boundaries
- **Immature trust model** (OAuth 2.1 spec only published Aug 2025)
- **Server quality variance** in community-contributed servers

---

## References

- Anthropic, "Model Context Protocol Specification," https://modelcontextprotocol.io
- Anthropic, "Introducing the Model Context Protocol," Nov 2024
- MCP GitHub, https://github.com/modelcontextprotocol
- Google, "Agent-to-Agent (A2A) Protocol," Apr 2025
- Linux Foundation, "Agentic AI Foundation," Dec 2025

---

## Related Pages

- [[tool-use]] -- Tool use that MCP standardizes
- [[agent-frameworks]] -- Frameworks integrating MCP
- [[ai-agent-overview]] -- Agent architecture overview
- [[compound-ai-systems]] -- Systems built on MCP integration
- [[environment-design]] -- OpenReward Standard extends MCP for RL
- [[agent-serving-challenges]] -- Impact of tool calls on serving
