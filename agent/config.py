"""
Configuration management for the coding agent.
Handles loading and merging config from files, env vars, and defaults.
"""

import os
import json
from pathlib import Path
from typing import Optional, Any
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """Configuration for an LLM backend."""
    provider: str = "openai"  # openai, anthropic, deepseek
    model: str = "gpt-4o"
    api_key: str = ""
    api_base: str = "https://api.openai.com/v1"
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 1.0
    # Anthropic-specific
    anthropic_version: str = "2023-06-01"


class PermissionConfig(BaseModel):
    """Permission system configuration."""
    mode: str = "ask"  # ask, auto_allow, strict
    allowed_tools: list[str] = Field(default_factory=list)
    denied_tools: list[str] = Field(default_factory=list)
    allowed_paths: list[str] = Field(default_factory=lambda: ["./"])
    denied_paths: list[str] = Field(default_factory=list)
    ask_once_per_session: bool = True


class ContextConfig(BaseModel):
    """Context window management configuration."""
    max_input_tokens: int = 100000
    reserve_output_tokens: int = 4096
    summarize_at_pct: float = 0.85
    model_context_limit: int = 128000


class MemoryConfig(BaseModel):
    """Memory system configuration."""
    enabled: bool = True
    memory_dir: str = ".agent/memory"
    index_file: str = "MEMORY.md"
    max_recall: int = 5


class SkillsConfig(BaseModel):
    """Skills system configuration."""
    enabled: bool = True
    skills_dir: str = ".agent/skills"
    builtin_skills: list[str] = Field(default_factory=lambda: ["code-review", "simplify"])


class MultiAgentConfig(BaseModel):
    """Multi-agent configuration."""
    enabled: bool = True
    max_parallel: int = 5
    default_model: str = "gpt-4o-mini"
    isolation_mode: str = "inline"  # inline, worktree


class MCPConfig(BaseModel):
    """MCP integration configuration."""
    enabled: bool = True
    servers: list[dict] = Field(default_factory=list)
    # Example: [{"name": "filesystem", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]}]


class AutonomyConfig(BaseModel):
    """Autonomy and scheduling configuration."""
    enabled: bool = True
    checkpoint_dir: str = ".agent/checkpoints"
    checkpoint_interval: int = 60  # seconds
    max_idle_seconds: int = 300


class AgentConfig(BaseModel):
    """Master configuration for the agent."""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    permissions: PermissionConfig = Field(default_factory=PermissionConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    multi_agent: MultiAgentConfig = Field(default_factory=MultiAgentConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    autonomy: AutonomyConfig = Field(default_factory=AutonomyConfig)
    # General
    workspace_dir: str = "."
    max_tool_iterations: int = 50
    debug: bool = False

    @classmethod
    def from_file(cls, path: str) -> "AgentConfig":
        """Load configuration from a YAML or JSON file."""
        path = Path(path)
        if not path.exists():
            return cls()

        with open(path, "r", encoding="utf-8") as f:
            if path.suffix in (".yaml", ".yml"):
                import yaml
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        return cls(**data) if data else cls()

    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Load configuration from environment variables with defaults."""
        return cls(
            llm=LLMConfig(
                provider=os.getenv("AGENT_LLM_PROVIDER", "openai"),
                model=os.getenv("AGENT_LLM_MODEL", "gpt-4o"),
                api_key=os.getenv("OPENAI_API_KEY", os.getenv("ANTHROPIC_API_KEY", "")),
                api_base=os.getenv("AGENT_LLM_API_BASE", "https://api.openai.com/v1"),
            ),
            workspace_dir=os.getenv("AGENT_WORKSPACE", "."),
        )

    def save(self, path: str) -> None:
        """Save configuration to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        import yaml
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, allow_unicode=True)
