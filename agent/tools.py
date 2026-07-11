"""
Tool system for the coding agent.
Provides a registry-based tool framework with built-in tools for file ops,
code search, shell execution, and web access.
"""

import os
import re
import json
import asyncio
import subprocess
import fnmatch
from pathlib import Path
from typing import Any, Callable, Optional
from dataclasses import dataclass, field
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------


class ToolParameter(BaseModel):
    """A single parameter in a tool's input schema."""
    name: str
    type: str = "string"
    description: str = ""
    required: bool = False
    default: Any = None
    enum: list[str] | None = None


class ToolDef(BaseModel):
    """Definition of a tool — its name, description, and JSON Schema parameters."""
    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)
    # Optional metadata
    requires_approval: bool = False
    category: str = "general"

    def to_openai_schema(self) -> dict:
        """Convert to OpenAI function-calling schema."""
        properties = {}
        required = []
        for p in self.parameters:
            prop: dict[str, Any] = {"type": p.type, "description": p.description}
            if p.enum:
                prop["enum"] = p.enum
            properties[p.name] = prop
            if p.required:
                required.append(p.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def to_anthropic_schema(self) -> dict:
        """Convert to Anthropic tool-use schema."""
        properties = {}
        required = []
        for p in self.parameters:
            prop: dict[str, Any] = {"type": p.type, "description": p.description}
            if p.enum:
                prop["enum"] = p.enum
            properties[p.name] = prop
            if p.required:
                required.append(p.name)

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }


ToolHandler = Callable[..., Any]


@dataclass
class Tool:
    """A registered tool with its definition and handler."""
    definition: ToolDef
    handler: ToolHandler
    is_async: bool = False

    async def execute(self, **kwargs) -> str:
        """Execute the tool with the given arguments."""
        try:
            if self.is_async:
                result = await self.handler(**kwargs)
            else:
                result = self.handler(**kwargs)
            return str(result) if result is not None else "(no output)"
        except Exception as e:
            return f"Error executing tool '{self.definition.name}': {e}"


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------


class ToolRegistry:
    """Registry for all available tools."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.definition.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_definitions(self) -> list[ToolDef]:
        """List all tool definitions."""
        return [t.definition for t in self._tools.values()]

    def to_openai_tools(self) -> list[dict]:
        """Convert all tools to OpenAI format."""
        return [t.definition.to_openai_schema() for t in self._tools.values()]

    def to_anthropic_tools(self) -> list[dict]:
        """Convert all tools to Anthropic format."""
        return [t.definition.to_anthropic_schema() for t in self._tools.values()]

    async def execute(self, name: str, arguments: dict) -> str:
        """Execute a tool by name."""
        tool = self._tools.get(name)
        if not tool:
            return f"Unknown tool: {name}"
        return await tool.execute(**arguments)

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())


# ---------------------------------------------------------------------------
# Built-in tools
# ---------------------------------------------------------------------------


def _resolve_path(workspace: str, file_path: str) -> Path:
    """Resolve a file path relative to the workspace."""
    p = Path(file_path)
    if p.is_absolute():
        return p
    return (Path(workspace) / p).resolve()


# ---- File tools ----


def tool_read_file(workspace: str, file_path: str, offset: int = 0, limit: int | None = None) -> str:
    """Read a file and return its contents with line numbers."""
    path = _resolve_path(workspace, file_path)
    if not path.exists():
        return f"Error: File not found: {file_path}"
    if path.is_dir():
        return f"Error: Path is a directory: {file_path}"

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        total_lines = len(lines)
        start = max(0, offset)
        end = min(total_lines, offset + limit) if limit else total_lines
        selected = lines[start:end]

        output_lines = []
        for i, line in enumerate(selected, start=start + 1):
            output_lines.append(f"{i:6}\t{line.rstrip()}")

        header = f"File: {file_path} (lines {start+1}-{end} of {total_lines})\n"
        return header + "\n".join(output_lines)
    except Exception as e:
        return f"Error reading file: {e}"


def tool_write_file(workspace: str, file_path: str, content: str) -> str:
    """Write content to a file, creating parent directories if needed."""
    path = _resolve_path(workspace, file_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"File written: {file_path} ({len(content)} bytes)"
    except Exception as e:
        return f"Error writing file: {e}"


def tool_edit_file(workspace: str, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    """Replace text in a file. old_string must match exactly once (or use replace_all)."""
    path = _resolve_path(workspace, file_path)
    if not path.exists():
        return f"Error: File not found: {file_path}"

    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        count = content.count(old_string)
        if count == 0:
            return f"Error: old_string not found in {file_path}"
        if count > 1 and not replace_all:
            return f"Error: old_string found {count} times. Use replace_all=True or make the string more specific."

        new_content = content.replace(old_string, new_string) if replace_all else content.replace(old_string, new_string, 1)

        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)

        occurrences = "all occurrences" if replace_all else "1 occurrence"
        return f"File edited: {file_path} (replaced {occurrences})"
    except Exception as e:
        return f"Error editing file: {e}"


def tool_glob(workspace: str, pattern: str, path: str | None = None) -> str:
    """Find files matching a glob pattern."""
    base = _resolve_path(workspace, path) if path else Path(workspace)
    if not base.exists():
        return f"Error: Directory not found: {base}"

    try:
        matches = sorted(base.rglob(pattern))
        # Filter out common ignore patterns
        ignores = {".git", "__pycache__", "node_modules", ".venv", "venv", ".tox", ".mypy_cache"}
        filtered = [m for m in matches if not any(ig in m.parts for ig in ignores)]

        if not filtered:
            return f"No files matching '{pattern}' found in {base}"
        if len(filtered) > 100:
            return f"Found {len(filtered)} files (showing first 100):\n" + "\n".join(str(m.relative_to(base)) for m in filtered[:100])
        return "\n".join(str(m.relative_to(base)) for m in filtered)
    except Exception as e:
        return f"Error in glob: {e}"


def tool_grep(workspace: str, pattern: str, path: str | None = None,
              glob: str | None = None, output_mode: str = "files_with_matches",
              head_limit: int = 50, ignore_case: bool = False) -> str:
    """Search file contents with regex."""
    base = _resolve_path(workspace, path) if path else Path(workspace)
    if not base.exists():
        return f"Error: Path not found: {base}"

    flags = re.IGNORECASE if ignore_case else 0
    try:
        regex = re.compile(pattern, flags | re.MULTILINE)
    except re.error as e:
        return f"Invalid regex pattern: {e}"

    ignores = {".git", "__pycache__", "node_modules", ".venv", "venv", ".tox", ".mypy_cache", ".png", ".jpg", ".gif", ".bin", ".exe"}
    results = []
    files_searched = 0

    for fpath in base.rglob("*" if not glob else glob):
        if fpath.is_dir():
            continue
        if any(ig in fpath.parts for ig in ignores):
            continue
        if fpath.suffix in {".png", ".jpg", ".gif", ".ico", ".bin", ".exe", ".dll", ".so", ".o", ".pyc", ".pyo"}:
            continue

        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception:
            continue

        files_searched += 1
        if output_mode == "files_with_matches":
            if regex.search(content):
                results.append(str(fpath.relative_to(base)))
        elif output_mode == "count":
            count = len(regex.findall(content))
            if count > 0:
                results.append(f"{fpath.relative_to(base)}: {count}")
        elif output_mode == "content":
            for i, line in enumerate(content.split("\n"), 1):
                m = regex.search(line)
                if m:
                    results.append(f"{fpath.relative_to(base)}:{i}: {line.strip()}")

        if len(results) >= head_limit:
            results.append(f"... (truncated at {head_limit} results, searched {files_searched} files)")
            break

    if not results:
        return f"No matches for '{pattern}' in {base} (searched {files_searched} files)"
    return "\n".join(results)


def tool_bash(command: str, workspace: str = ".", timeout: int = 60) -> str:
    """Execute a shell command in the workspace directory."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr}"
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout}s: {command}"
    except Exception as e:
        return f"Error executing command: {e}"


def tool_web_search(query: str) -> str:
    """Search the web (placeholder — requires API key in production)."""
    return f"[Web search] Query: '{query}'\n(This tool requires a search API key. In production, this would return real search results.)"


def tool_web_fetch(url: str, prompt: str = "") -> str:
    """Fetch a URL and extract its content."""
    import urllib.request
    import urllib.error

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Agent/0.1"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            content = resp.read().decode("utf-8", errors="replace")
        # Simple HTML-to-text (strip tags)
        text = re.sub(r"<script[^>]*>.*?</script>", "", content, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        preview = text[:3000]
        if prompt:
            return f"URL: {url}\nPrompt: {prompt}\n\nContent preview:\n{preview}"
        return f"URL: {url}\n\n{preview}"
    except urllib.error.URLError as e:
        return f"Error fetching URL: {e}"
    except Exception as e:
        return f"Error: {e}"


def tool_ask_user(question: str) -> str:
    """Ask the user a question (for CLI mode, this prompts interactively)."""
    try:
        answer = input(f"\n[Agent asks] {question}\n> ")
        return f"User answered: {answer}" if answer else "(no answer)"
    except (EOFError, KeyboardInterrupt):
        return "(user declined to answer)"


def tool_task_create(subject: str, description: str = "") -> str:
    """Create a task in the task list."""
    return f"Task created: [{subject}] — {description}"


def tool_task_update(task_id: str, status: str = "") -> str:
    """Update a task's status."""
    return f"Task {task_id} updated: status={status}"


# ---------------------------------------------------------------------------
# Register all built-in tools
# ---------------------------------------------------------------------------


def create_builtin_registry(workspace: str = ".") -> ToolRegistry:
    """Create a tool registry with all built-in tools registered."""
    registry = ToolRegistry()

    builtins = [
        Tool(
            definition=ToolDef(
                name="read_file",
                description="Read a file from the filesystem. Returns the file contents with line numbers.",
                parameters=[
                    ToolParameter(name="file_path", type="string", description="Path to the file to read", required=True),
                    ToolParameter(name="offset", type="integer", description="Line number to start reading from", required=False, default=0),
                    ToolParameter(name="limit", type="integer", description="Maximum number of lines to read", required=False),
                ],
                category="file",
            ),
            handler=lambda file_path, offset=0, limit=None, ws=workspace: tool_read_file(ws, file_path, offset, limit),
        ),
        Tool(
            definition=ToolDef(
                name="write_file",
                description="Write content to a file. Creates parent directories if needed. Overwrites existing files.",
                parameters=[
                    ToolParameter(name="file_path", type="string", description="Path to the file to write", required=True),
                    ToolParameter(name="content", type="string", description="Content to write to the file", required=True),
                ],
                requires_approval=True,
                category="file",
            ),
            handler=lambda file_path, content, ws=workspace: tool_write_file(ws, file_path, content),
        ),
        Tool(
            definition=ToolDef(
                name="edit_file",
                description="Replace text in a file by exact string matching.",
                parameters=[
                    ToolParameter(name="file_path", type="string", description="Path to the file to edit", required=True),
                    ToolParameter(name="old_string", type="string", description="The exact text to replace", required=True),
                    ToolParameter(name="new_string", type="string", description="The text to replace it with", required=True),
                    ToolParameter(name="replace_all", type="boolean", description="Replace all occurrences", required=False, default=False),
                ],
                requires_approval=True,
                category="file",
            ),
            handler=lambda file_path, old_string, new_string, replace_all=False, ws=workspace: tool_edit_file(ws, file_path, old_string, new_string, replace_all),
        ),
        Tool(
            definition=ToolDef(
                name="bash",
                description="Execute a shell command in the workspace directory.",
                parameters=[
                    ToolParameter(name="command", type="string", description="The shell command to execute", required=True),
                    ToolParameter(name="timeout", type="integer", description="Timeout in seconds", required=False, default=60),
                ],
                requires_approval=True,
                category="shell",
            ),
            handler=lambda command, timeout=60, ws=workspace: tool_bash(command, ws, timeout),
        ),
        Tool(
            definition=ToolDef(
                name="glob",
                description="Find files matching a glob pattern (e.g., **/*.py, src/**/*.ts).",
                parameters=[
                    ToolParameter(name="pattern", type="string", description="The glob pattern to match", required=True),
                    ToolParameter(name="path", type="string", description="Directory to search in (defaults to workspace)", required=False),
                ],
                category="search",
            ),
            handler=lambda pattern, path=None, ws=workspace: tool_glob(ws, pattern, path),
        ),
        Tool(
            definition=ToolDef(
                name="grep",
                description="Search file contents with a regular expression pattern.",
                parameters=[
                    ToolParameter(name="pattern", type="string", description="The regex pattern to search for", required=True),
                    ToolParameter(name="path", type="string", description="Directory or file to search in", required=False),
                    ToolParameter(name="glob", type="string", description="File filter glob pattern", required=False),
                    ToolParameter(name="output_mode", type="string", description="Output mode", required=False, default="files_with_matches", enum=["content", "files_with_matches", "count"]),
                    ToolParameter(name="head_limit", type="integer", description="Max results to return", required=False, default=50),
                    ToolParameter(name="ignore_case", type="boolean", description="Case-insensitive search", required=False, default=False),
                ],
                category="search",
            ),
            handler=lambda pattern, path=None, glob=None, output_mode="files_with_matches", head_limit=50, ignore_case=False, ws=workspace: tool_grep(ws, pattern, path, glob, output_mode, head_limit, ignore_case),
        ),
        Tool(
            definition=ToolDef(
                name="web_search",
                description="Search the web and return results.",
                parameters=[
                    ToolParameter(name="query", type="string", description="The search query", required=True),
                ],
                category="web",
            ),
            handler=lambda query: tool_web_search(query),
        ),
        Tool(
            definition=ToolDef(
                name="web_fetch",
                description="Fetch a URL and extract its content.",
                parameters=[
                    ToolParameter(name="url", type="string", description="The URL to fetch", required=True),
                    ToolParameter(name="prompt", type="string", description="Optional prompt to run on the content", required=False, default=""),
                ],
                category="web",
            ),
            handler=lambda url, prompt="": tool_web_fetch(url, prompt),
        ),
        Tool(
            definition=ToolDef(
                name="ask_user",
                description="Ask the user a question when you need clarification.",
                parameters=[
                    ToolParameter(name="question", type="string", description="The question to ask", required=True),
                ],
                category="interaction",
            ),
            handler=lambda question: tool_ask_user(question),
        ),
        Tool(
            definition=ToolDef(
                name="task_create",
                description="Create a task in the structured task list for tracking progress.",
                parameters=[
                    ToolParameter(name="subject", type="string", description="Brief task title", required=True),
                    ToolParameter(name="description", type="string", description="What needs to be done", required=False, default=""),
                ],
                category="meta",
            ),
            handler=lambda subject, description="": tool_task_create(subject, description),
        ),
        Tool(
            definition=ToolDef(
                name="task_update",
                description="Update the status of a task.",
                parameters=[
                    ToolParameter(name="task_id", type="string", description="The task ID", required=True),
                    ToolParameter(name="status", type="string", description="New status: pending, in_progress, completed", required=True),
                ],
                category="meta",
            ),
            handler=lambda task_id, status: tool_task_update(task_id, status),
        ),
    ]

    for tool in builtins:
        registry.register(tool)

    return registry
