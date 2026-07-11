# 🛠️ From-Zero Coding Agent

A **from-scratch implementation** of a coding agent — built to understand every layer of how AI coding assistants work.

## Architecture

```
from-zero/
├── main.py                  # Entry point
├── agent/
│   ├── __init__.py          # Package info
│   ├── config.py            # Configuration (Pydantic models)
│   ├── core.py              # 🔄 Agent Loop (ReAct pattern)
│   ├── tools.py             # 🔧 Tool System (registry + 11 built-in tools)
│   ├── system_prompt.py     # 📝 System Prompt Engineering
│   ├── streaming.py         # 📡 Streaming + Multi-Backend (OpenAI + Anthropic)
│   ├── permissions.py       # 🔒 Permission & Security
│   ├── context.py           # 📊 Context Window Management
│   ├── memory.py            # 🧠 Persistent Memory (file-based + frontmatter)
│   ├── skills.py            # ⚡ Skills/Plugins System
│   ├── plan_mode.py         # 📋 Plan-Before-Execute Mode
│   ├── multi_agent.py       # 👥 Multi-Agent Orchestration
│   ├── mcp_client.py        # 🔌 MCP Protocol Integration
│   ├── autonomy.py          # ⏰ Autonomy & Scheduling
│   └── cli.py               # 💻 Interactive CLI (Rich + prompt_toolkit)
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set your API key
export OPENAI_API_KEY="sk-..."

# Interactive mode
python main.py

# Single prompt
python main.py "Explain this codebase"

# Custom model
python main.py --model deepseek-v3 --api-base https://api.deepseek.com/v1
```

## Phase 1: Core Agent (可用)

| Module | Description |
|--------|-------------|
| `core.py` | ReAct loop: think → act → observe → repeat |
| `tools.py` | 11 built-in tools: read/write/edit file, bash, glob, grep, web_search/fetch, task management |
| `system_prompt.py` | Configurable prompt builder with persona, tools, env, rules |
| `cli.py` | Rich interactive terminal with streaming, slash commands, session persistence |
| `streaming.py` | SSE streaming for OpenAI-compatible + Anthropic backends |
| `permissions.py` | Tool-level allow/deny/ask with path-based rules |
| `context.py` | Token counting, budget tracking, auto-summarization |

## Phase 2: Advanced Capabilities

| Module | Description |
|--------|-------------|
| `memory.py` | File-based memory with YAML frontmatter, MEMORY.md index, keyword recall |
| `skills.py` | Pluggable skill modules with triggers, built-in code-review + simplify |
| `plan_mode.py` | Plan-before-execute: explore → design → present → approve → execute |
| `multi_agent.py` | Sub-agent spawning: parallel, pipeline, map-reduce patterns |
| `mcp_client.py` | MCP stdio transport for external tool servers |

## Phase 3: Autonomy

| Module | Description |
|--------|-------------|
| `autonomy.py` | Cron scheduler, checkpoint/resume, idle detection |

## Built-in Tools

| Tool | Description |
|------|-------------|
| `read_file` | Read file with line numbers |
| `write_file` | Write/create files |
| `edit_file` | Exact string replacement |
| `bash` | Execute shell commands |
| `glob` | File pattern matching |
| `grep` | Regex content search |
| `web_search` | Web search (placeholder) |
| `web_fetch` | URL content extraction |
| `ask_user` | Interactive user query |
| `task_create` | Task tracking |
| `task_update` | Task status updates |

## Slash Commands

| Command | Description |
|---------|-------------|
| `/help` | Show help |
| `/clear` | Clear conversation |
| `/save` | Save session |
| `/load <id>` | Load session |
| `/sessions` | List sessions |
| `/memory list/recall/save/delete` | Memory management |
| `/skills` | List skills |
| `/plan` | Enter plan mode |
| `/tokens` | Token usage |
| `/status` | Agent status |

## Configuration

```yaml
# config.yaml
llm:
  provider: openai
  model: gpt-4o
  api_key: ${OPENAI_API_KEY}
  api_base: https://api.openai.com/v1
  max_tokens: 4096

permissions:
  mode: ask  # ask | auto_allow | strict
  allowed_tools: []
  denied_tools: []

context:
  model_context_limit: 128000
  summarize_at_pct: 0.85

memory:
  enabled: true
  memory_dir: .agent/memory

skills:
  enabled: true
  skills_dir: .agent/skills

mcp:
  enabled: true
  servers:
    - name: filesystem
      command: npx
      args: ["-y", "@modelcontextprotocol/server-filesystem", "."]
```

## Key Design Principles

1. **Readable over clever** — Code is documentation. Every module is self-contained and well-commented.
2. **Pluggable** — Swap backends, add tools, register skills without touching core.
3. **From scratch** — No LangChain, no frameworks. Just httpx, pydantic, and the API specs.
4. **Progressive** — Works with just `OPENAI_API_KEY`, but scales to MCP, multi-agent, autonomy.
