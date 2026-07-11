"""
CLI interface for the coding agent.
Provides a Rich + prompt_toolkit based interactive terminal with:
- Streaming text output
- Syntax-highlighted code blocks
- Slash commands
- Permission prompts
- Session persistence
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

from .config import AgentConfig
from .core import Agent, AgentResult
from .tools import create_builtin_registry
from .memory import MemorySystem, MemoryEntry
from .skills import SkillRegistry
from .plan_mode import PlanMode, Plan
from .permissions import PermissionResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Console output helpers (using Rich if available, else plain text)
# ---------------------------------------------------------------------------

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.live import Live
    from rich.text import Text
    from rich import box

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class ConsoleOutput:
    """Unified console output — uses Rich if available, else plain print."""

    def __init__(self):
        if RICH_AVAILABLE:
            self._console = Console()
        else:
            self._console = None

    def print(self, text: str = "", **kwargs) -> None:
        if self._console:
            self._console.print(text, **kwargs)
        else:
            print(text)

    def print_markdown(self, text: str) -> None:
        if self._console:
            self._console.print(Markdown(text))
        else:
            print(text)

    def print_panel(self, text: str, title: str = "", style: str = "") -> None:
        if self._console:
            self._console.print(Panel(text, title=title))
        else:
            if title:
                print(f"── {title} ──")
            print(text)

    def print_code(self, code: str, language: str = "python") -> None:
        if self._console:
            self._console.print(Syntax(code, language, theme="monokai", line_numbers=True))
        else:
            print(f"```{language}")
            print(code)
            print("```")

    def print_error(self, text: str) -> None:
        if self._console:
            self._console.print(f"[red]❌ {text}[/red]")
        else:
            print(f"ERROR: {text}")

    def print_success(self, text: str) -> None:
        if self._console:
            self._console.print(f"[green]✅ {text}[/green]")
        else:
            print(f"OK: {text}")

    def print_warning(self, text: str) -> None:
        if self._console:
            self._console.print(f"[yellow]⚠️  {text}[/yellow]")
        else:
            print(f"WARN: {text}")

    def print_stream_chunk(self, chunk: str) -> None:
        """Print a streaming text chunk without newline."""
        if self._console:
            self._console.print(chunk, end="")
        else:
            print(chunk, end="", flush=True)

    def input(self, prompt: str = "> ") -> str:
        if RICH_AVAILABLE:
            return Prompt.ask(prompt)
        else:
            return input(prompt)


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------


class SessionManager:
    """Manages conversation session persistence."""

    def __init__(self, workspace: str = "."):
        self.workspace = Path(workspace)
        self.sessions_dir = self.workspace / ".agent" / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def save(self, session_id: str, agent: Agent) -> str:
        """Save the current session."""
        filepath = self.sessions_dir / f"{session_id}.json"
        data = agent.to_dict()
        data["saved_at"] = datetime.now().isoformat()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return str(filepath)

    def load(self, session_id: str) -> dict | None:
        """Load a saved session."""
        filepath = self.sessions_dir / f"{session_id}.json"
        if not filepath.exists():
            return None
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_sessions(self) -> list[dict]:
        """List all saved sessions."""
        sessions = []
        for fpath in sorted(self.sessions_dir.glob("*.json"), reverse=True):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                sessions.append({
                    "session_id": data.get("session_id", fpath.stem),
                    "saved_at": data.get("saved_at", ""),
                    "steps": data.get("steps_count", 0),
                    "workspace": data.get("config_workspace", ""),
                })
            except Exception:
                pass
        return sessions

    def delete(self, session_id: str) -> bool:
        """Delete a saved session."""
        filepath = self.sessions_dir / f"{session_id}.json"
        if filepath.exists():
            filepath.unlink()
            return True
        return False


# ---------------------------------------------------------------------------
# CLI Application
# ---------------------------------------------------------------------------


class AgentCLI:
    """The main CLI application for the coding agent."""

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig.from_env()
        self.out = ConsoleOutput()
        self.session_manager = SessionManager(self.config.workspace_dir)
        self.session_id = datetime.now().strftime("session_%Y%m%d_%H%M%S")

        # Core subsystems
        self.tools = create_builtin_registry(self.config.workspace_dir)
        self.agent = Agent(config=self.config, tool_registry=self.tools)
        self.memory = MemorySystem(self.config.memory, self.config.workspace_dir)
        self.skills = SkillRegistry(self.config.skills, self.config.workspace_dir)
        self.plan_mode = PlanMode()

        # Load skills
        self.skills.load_builtin_skills()

        # Set up callbacks
        self.agent.on_text(self._on_stream_text)
        self.agent.on_approval(self._on_approval_needed)

        # Streaming state
        self._streaming_buffer = ""

    # ---- Public API ----

    async def start(self, initial_prompt: str | None = None) -> None:
        """Start the interactive CLI loop."""
        self._print_banner()

        # Check for MCP servers
        if self.config.mcp.enabled and self.config.mcp.servers:
            self.out.print("[dim]Connecting to MCP servers...[/dim]")
            from .mcp_client import MCPManager
            mcp = MCPManager(self.config.mcp)
            results = await mcp.connect_all()
            for name, ok in results.items():
                if ok:
                    count = mcp.register_in_registry(self.tools)
                    self.out.print_success(f"MCP server '{name}' connected ({count} tools)")
                else:
                    self.out.print_error(f"MCP server '{name}' failed to connect")

        # Check for resume
        checkpoint = None
        if self.config.autonomy.enabled:
            from .autonomy import AutonomyManager
            self.autonomy = AutonomyManager(self.config.autonomy, self.config.workspace_dir)
            checkpoint = self.autonomy.restore_checkpoint()

        if checkpoint:
            self.out.print_warning(f"Found checkpoint from {checkpoint.timestamp}")
            if self._confirm("Resume previous session?"):
                self.agent.from_dict({"messages": checkpoint.messages, "system_prompt": checkpoint.system_prompt})
                self.out.print_success("Session restored")

        # Run initial prompt if provided
        if initial_prompt:
            await self._process_input(initial_prompt)

        # Main interactive loop
        await self._repl()

    async def run_once(self, prompt: str) -> None:
        """Run a single prompt non-interactively and exit."""
        await self._process_input(prompt)

    # ---- Internal: REPL ----

    async def _repl(self) -> None:
        """Read-Eval-Print Loop."""
        while True:
            try:
                user_input = self._get_input()
            except (EOFError, KeyboardInterrupt):
                self.out.print("\nGoodbye!")
                break

            if not user_input:
                continue

            # Handle slash commands
            if user_input.startswith("/"):
                handled = await self._handle_command(user_input)
                if handled is False:  # False = exit
                    break
                continue

            # Process normal input
            await self._process_input(user_input)

            # Auto-checkpoint
            if hasattr(self, 'autonomy') and self.agent._messages:
                self.autonomy.save_checkpoint(
                    session_id=self.session_id,
                    messages=self.agent._messages,
                    system_prompt=self.agent._system_prompt,
                    step_count=len(self.agent._steps),
                )

    def _get_input(self) -> str:
        """Get user input with prompt_toolkit if available."""
        self.out.print("")  # Blank line for spacing

        try:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.styles import Style

            style = Style.from_dict({
                "prompt": "bold green",
            })

            session = PromptSession(style=style)
            return session.prompt([("class:prompt", "▸ ")], multiline=False)
        except ImportError:
            return self.out.input("▸ ")

    async def _process_input(self, user_input: str) -> None:
        """Process a user input through the agent."""
        # Recall relevant memories
        memories = self.memory.recall(user_input)
        if memories:
            memory_context = self.memory.format_for_context(memories)
            self.agent.set_memory_context(memory_context)

        # Check for skill triggers
        matching_skills = self.skills.find_by_trigger(user_input)
        if matching_skills:
            skill_names = ", ".join(f"/{s.meta.name}" for s in matching_skills)
            self.out.print_warning(f"Available skills: {skill_names}")

        # Show thinking indicator
        self.out.print("[dim]Thinking...[/dim]")
        self._streaming_buffer = ""

        try:
            result = await self.agent.run_stream(user_input)
        except Exception as e:
            self.out.print_error(f"Agent error: {e}")
            logger.exception("Agent run failed")
            return

        self.out.print("")  # Newline after streaming

        # Display result
        if result.success:
            self.out.print_markdown(result.content)
        else:
            self.out.print_error(result.content)

        # Show token usage
        self.out.print(f"[dim]{self.agent.usage_report}[/dim]")

        # Memory suggestion (auto-detect important facts)
        await self._maybe_suggest_memory(user_input, result)

    # ---- Internal: Streaming ----

    async def _on_stream_text(self, chunk: str) -> None:
        """Handle streaming text chunks from the agent."""
        self._streaming_buffer += chunk
        self.out.print_stream_chunk(chunk)

    # ---- Internal: Permissions ----

    async def _on_approval_needed(self, tool_name: str, arguments: dict) -> bool:
        """Handle tool approval requests interactively."""
        self.out.print_warning(
            f"Tool '{tool_name}' requires approval:\n"
            f"  {json.dumps(arguments, indent=2, ensure_ascii=False)[:200]}"
        )
        return self._confirm("Allow execution?", default=True)

    # ---- Internal: Slash commands ----

    async def _handle_command(self, cmd: str) -> bool | None:
        """Handle a slash command. Returns False to exit, None to continue."""
        parts = cmd.split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        handlers = {
            "/help": self._cmd_help,
            "/exit": self._cmd_exit,
            "/quit": self._cmd_exit,
            "/clear": self._cmd_clear,
            "/memory": self._cmd_memory,
            "/save": self._cmd_save,
            "/load": self._cmd_load,
            "/sessions": self._cmd_sessions,
            "/skills": self._cmd_skills,
            "/plan": self._cmd_plan,
            "/tokens": self._cmd_tokens,
            "/config": self._cmd_config,
            "/status": self._cmd_status,
        }

        handler = handlers.get(command)
        if handler:
            return await handler(args)
        else:
            self.out.print_warning(f"Unknown command: {command}. Type /help for available commands.")
            return None

    async def _cmd_help(self, args: str) -> None:
        """Show help text."""
        help_text = """
## Commands

### Core
- `/help` — Show this help
- `/exit`, `/quit` — Exit the agent
- `/clear` — Clear conversation history
- `/status` — Show agent status

### Session
- `/save [name]` — Save current session
- `/load <id>` — Load a saved session
- `/sessions` — List saved sessions

### Memory
- `/memory recall <query>` — Recall memories
- `/memory save <name> <content>` — Save a memory
- `/memory list` — List all memories
- `/memory delete <name>` — Delete a memory

### Skills
- `/skills` — List available skills

### Planning
- `/plan` — Enter plan mode for the current task

### Info
- `/tokens` — Show token usage
- `/config` — Show current configuration
"""
        self.out.print_markdown(help_text)

    async def _cmd_exit(self, args: str) -> bool:
        """Exit the agent."""
        # Auto-save session
        if self.agent._messages:
            self.session_manager.save(self.session_id, self.agent)
            self.out.print_success(f"Session saved: {self.session_id}")

        await self.agent.close()
        self.out.print("Goodbye!")
        return False

    async def _cmd_clear(self, args: str) -> None:
        """Clear conversation history."""
        if self._confirm("Clear conversation history?"):
            self.agent.reset()
            self.session_id = datetime.now().strftime("session_%Y%m%d_%H%M%S")
            self.out.print_success("Conversation cleared. New session: " + self.session_id)

    async def _cmd_memory(self, args: str) -> None:
        """Memory management commands."""
        parts = args.split(maxsplit=1)
        sub = parts[0].lower() if parts else "list"
        sub_args = parts[1] if len(parts) > 1 else ""

        if sub == "list":
            self.out.print_markdown(self.memory.get_index_summary())

        elif sub == "recall":
            if not sub_args:
                self.out.print_error("Usage: /memory recall <query>")
                return
            entries = self.memory.recall(sub_args)
            if entries:
                self.out.print_markdown(self.memory.format_for_context(entries))
            else:
                self.out.print("No relevant memories found.")

        elif sub == "save":
            # Parse: /memory save name description | content
            parts2 = sub_args.split("|", maxsplit=1)
            header = parts2[0].strip().split(maxsplit=1)
            name = header[0] if header else ""
            description = header[1] if len(header) > 1 else ""
            content = parts2[1].strip() if len(parts2) > 1 else ""

            if not name or not content:
                self.out.print_error("Usage: /memory save <name> <description> | <content>")
                return

            entry = MemoryEntry(
                name=name.lower().replace(" ", "-"),
                description=description,
                content=content,
                metadata={"type": "user"},
            )
            self.memory.save(entry)
            self.out.print_success(f"Memory saved: {name}")

        elif sub == "delete":
            if not sub_args:
                self.out.print_error("Usage: /memory delete <name>")
                return
            if self.memory.delete(sub_args):
                self.out.print_success(f"Memory deleted: {sub_args}")
            else:
                self.out.print_error(f"Memory not found: {sub_args}")

        else:
            self.out.print_error("Unknown memory subcommand. Use: list, recall, save, delete")

    async def _cmd_save(self, args: str) -> None:
        """Save current session."""
        name = args.strip() or self.session_id
        path = self.session_manager.save(name, self.agent)
        self.out.print_success(f"Session saved to {path}")

    async def _cmd_load(self, args: str) -> None:
        """Load a saved session."""
        if not args.strip():
            self.out.print_error("Usage: /load <session_id>")
            return

        data = self.session_manager.load(args.strip())
        if data:
            self.agent.from_dict(data)
            self.session_id = args.strip()
            self.out.print_success(f"Session loaded: {args.strip()}")
        else:
            self.out.print_error(f"Session not found: {args.strip()}")

    async def _cmd_sessions(self, args: str) -> None:
        """List saved sessions."""
        sessions = self.session_manager.list_sessions()
        if sessions:
            self.out.print_markdown("## Saved Sessions\n")
            for s in sessions[:20]:
                self.out.print(f"- **{s['session_id']}** — {s['saved_at']} ({s['steps']} steps)")
        else:
            self.out.print("No saved sessions.")

    async def _cmd_skills(self, args: str) -> None:
        """List available skills."""
        all_skills = self.skills.list_all()
        if all_skills:
            lines = ["## Available Skills\n"]
            for s in all_skills:
                triggers = ", ".join(s.triggers[:3]) if s.triggers else "none"
                lines.append(f"- **/{s.name}**: {s.description}")
                lines.append(f"  Triggers: {triggers}")
            self.out.print_markdown("\n".join(lines))
        else:
            self.out.print("No skills loaded.")

    async def _cmd_plan(self, args: str) -> None:
        """Enter plan mode."""
        self.out.print_panel(
            "Plan mode: I'll explore the codebase, design a plan, and ask for your approval before executing.",
            title="Plan Mode",
        )

        await self.plan_mode.enter()

        # Set up approval handler
        async def approval_handler(plan: Plan) -> bool:
            self.out.print_markdown(plan.format_for_display())
            return self._confirm("Approve this plan?")

        self.plan_mode.set_approval_handler(approval_handler)

        self.out.print_warning("Plan mode is active. Describe what you want to plan.")
        self.out.print("Use /plan-execute to carry out the plan once approved.")

    async def _cmd_tokens(self, args: str) -> None:
        """Show token usage."""
        self.out.print_markdown(f"## Token Usage\n{self.agent.usage_report}")

    async def _cmd_config(self, args: str) -> None:
        """Show current configuration."""
        import yaml
        config_yaml = yaml.dump(self.config.model_dump(), default_flow_style=False, allow_unicode=True)
        self.out.print_code(config_yaml, "yaml")

    async def _cmd_status(self, args: str) -> None:
        """Show agent status."""
        status_lines = [
            f"## Agent Status",
            f"- State: {self.agent.state.value}",
            f"- Session: {self.session_id}",
            f"- Messages: {len(self.agent._messages)}",
            f"- Steps: {len(self.agent._steps)}",
            f"- Tools: {len(self.tools.tool_names)}",
            f"- Skills: {len(self.skills.list_all())}",
            f"- Memories: {len(self.memory.load_all())}",
            f"- {self.agent.usage_report}",
        ]
        self.out.print_markdown("\n".join(status_lines))

    # ---- Helpers ----

    def _confirm(self, question: str, default: bool = True) -> bool:
        """Ask for user confirmation."""
        suffix = " [Y/n] " if default else " [y/N] "
        try:
            response = input(question + suffix).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False

        if not response:
            return default
        return response in ("y", "yes")

    async def _maybe_suggest_memory(self, user_input: str, result: AgentResult) -> None:
        """Suggest saving important information to memory."""
        # Simple heuristic: if the user is giving explicit instructions/preferences
        instruction_markers = [
            "remember", "always", "never", "i prefer", "i like",
            "my name is", "i work", "i use", "from now on",
        ]
        if any(marker in user_input.lower() for marker in instruction_markers):
            self.out.print_warning("💡 You can save this to memory with: /memory save <name> <description> | <content>")

    def _print_banner(self) -> None:
        """Print the welcome banner."""
        banner = r"""
╔══════════════════════════════════════════╗
║         🛠️  From-Zero Coding Agent       ║
║             v0.1.0 | {model}         ║
╚══════════════════════════════════════════╝
Type /help for commands, or just start chatting.
        """.format(model=self.config.llm.model[:26])
        self.out.print(banner.strip())


# ---------------------------------------------------------------------------
# Entry point helpers
# ---------------------------------------------------------------------------


async def run_cli(config: Optional[AgentConfig] = None, prompt: str | None = None) -> None:
    """Entry point for the CLI application."""
    cli = AgentCLI(config)

    if prompt:
        # Non-interactive: run once and exit
        await cli.run_once(prompt)
    else:
        # Interactive mode
        await cli.start()


def run_cli_sync(config: Optional[AgentConfig] = None, prompt: str | None = None) -> None:
    """Synchronous wrapper for run_cli."""
    asyncio.run(run_cli(config, prompt))
