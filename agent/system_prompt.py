"""
System prompt engineering for the coding agent.
Builds a configurable system prompt with sections for persona, tools, environment,
rules, and memory context.
"""

from datetime import datetime
from typing import Optional
from .config import AgentConfig
from .tools import ToolRegistry


class SystemPromptBuilder:
    """Builds the system prompt from configurable sections."""

    # Base persona — the agent's core identity
    PERSONA = """You are a Coding Agent — an AI assistant specialized in software engineering tasks.
You help users write, edit, debug, and understand code. You have access to tools that
let you read/write files, search code, execute shell commands, and more.

Work carefully and methodically. Read files before editing them. Verify your changes.
When you encounter errors, diagnose them before attempting fixes.
Be thorough — don't skip edge cases or error handling."""

    # Communication guidelines
    COMMUNICATION = """## Communication
- Be concise but complete. Don't repeat yourself.
- Use markdown code blocks for code snippets with language identifiers.
- Reference files as `path/to/file:line_number`.
- When you've completed a task, summarize what was done.
- If you're unsure about something, ask the user."""

    # Tool usage guidelines
    TOOL_GUIDELINES = """## Tool Usage
- Prefer `grep` and `glob` over `bash` for searching code.
- Use `read_file` before `edit_file` — you need to see the exact content.
- Execute `bash` commands with clear descriptions of what they do.
- When editing, match the exact indentation and style of the surrounding code.
- Report outcomes honestly: if something fails, say so."""

    # Environment context template
    ENV_TEMPLATE = """## Environment
- Date: {date}
- Platform: {platform}
- Workspace: {workspace}
- Shell: {shell}"""

    def __init__(self, config: AgentConfig):
        self.config = config
        self._custom_sections: list[str] = []
        self._memory_context: str = ""

    def add_section(self, section: str) -> None:
        """Add a custom section to the system prompt."""
        self._custom_sections.append(section)

    def set_memory_context(self, memories: str) -> None:
        """Set memory recall context for the current turn."""
        self._memory_context = memories

    def build(self, tool_registry: Optional[ToolRegistry] = None) -> str:
        """Build the full system prompt."""
        sections = [self.PERSONA]

        # Environment context
        import platform
        env_text = self.ENV_TEMPLATE.format(
            date=datetime.now().strftime("%Y-%m-%d"),
            platform=platform.platform(),
            workspace=self.config.workspace_dir,
            shell="bash" if platform.system() != "Windows" else "powershell",
        )
        sections.append(env_text)

        # Communication guidelines
        sections.append(self.COMMUNICATION)

        # Tool descriptions
        if tool_registry:
            sections.append(self._format_tools(tool_registry))

        sections.append(self.TOOL_GUIDELINES)

        # Memory context
        if self._memory_context:
            sections.append(f"## Relevant Memories\n{self._memory_context}")

        # Custom sections
        for section in self._custom_sections:
            sections.append(section)

        # Final instructions
        sections.append("""## Instructions
- Analyze the user's request carefully.
- Break complex tasks into smaller steps.
- Use tools to gather information before acting.
- Verify your work before declaring it done.
- If a task requires multiple steps, create a task list to track progress.""")

        return "\n\n".join(sections)

    def _format_tools(self, registry: ToolRegistry) -> str:
        """Format tool definitions for the system prompt."""
        lines = ["## Available Tools"]
        tools = registry.list_definitions()

        # Group by category
        by_category: dict[str, list] = {}
        for t in tools:
            by_category.setdefault(t.category, []).append(t)

        for category, cat_tools in sorted(by_category.items()):
            lines.append(f"\n### {category.title()} Tools")
            for t in cat_tools:
                params_str = ", ".join(
                    f"{p.name}: {p.type}" + (" (required)" if p.required else "")
                    for p in t.parameters
                )
                lines.append(f"- **{t.name}**({params_str}): {t.description}")
                if t.requires_approval:
                    lines.append(f"  ⚠️ Requires user approval")

        return "\n".join(lines)


def build_system_prompt(config: AgentConfig, tool_registry: ToolRegistry,
                        memory_context: str = "", custom_sections: list[str] | None = None) -> str:
    """Convenience function to build a system prompt in one call."""
    builder = SystemPromptBuilder(config)
    if memory_context:
        builder.set_memory_context(memory_context)
    if custom_sections:
        for s in custom_sections:
            builder.add_section(s)
    return builder.build(tool_registry)
