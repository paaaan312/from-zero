"""
Skills/plugins system for the coding agent.
Loadable skill modules that extend agent capabilities with specialized knowledge,
workflows, and tool integrations.

Skills are Python modules or YAML configurations in the skills directory.
Each skill defines metadata (name, description, triggers) and a handler.
"""

import os
import sys
import importlib.util
from pathlib import Path
from typing import Any, Callable, Optional
from dataclasses import dataclass, field

from .config import SkillsConfig


@dataclass
class SkillMeta:
    """Metadata for a skill."""
    name: str
    description: str = ""
    version: str = "0.1.0"
    triggers: list[str] = field(default_factory=list)  # Keywords that trigger this skill
    category: str = "general"
    author: str = ""
    requires_approval: bool = False


@dataclass
class Skill:
    """A registered skill with metadata and handler."""
    meta: SkillMeta
    handler: Callable[..., Any]
    is_loaded: bool = True


class SkillRegistry:
    """
    Registry for skills/plugins.

    Skills can be:
    1. Python modules in the skills directory
    2. YAML-defined skills with command handlers
    3. Built-in skills included with the agent
    """

    def __init__(self, config: SkillsConfig, workspace: str = "."):
        self.config = config
        self.workspace = Path(workspace)
        self.skills_dir = self.workspace / config.skills_dir
        self._skills: dict[str, Skill] = {}
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        """Ensure the skills directory exists."""
        self.skills_dir.mkdir(parents=True, exist_ok=True)

    # ---- Registration ----

    def register(self, skill: Skill) -> None:
        """Register a skill."""
        self._skills[skill.meta.name] = skill

    def unregister(self, name: str) -> None:
        """Remove a skill."""
        self._skills.pop(name, None)

    def get(self, name: str) -> Optional[Skill]:
        """Get a skill by name."""
        return self._skills.get(name)

    def list_all(self) -> list[SkillMeta]:
        """List all registered skill metadata."""
        return [s.meta for s in self._skills.values()]

    def find_by_trigger(self, user_input: str) -> list[Skill]:
        """
        Find skills that match the user's input.
        Checks trigger keywords and slash commands (e.g., /code-review).
        """
        input_lower = user_input.lower().strip()
        matched = []

        for skill in self._skills.values():
            # Check for slash command: /skill-name
            if input_lower.startswith(f"/{skill.meta.name}"):
                matched.append(skill)
                continue

            # Check trigger keywords
            for trigger in skill.meta.triggers:
                if trigger.lower() in input_lower:
                    matched.append(skill)
                    break

        return matched

    def invoke(self, name: str, **kwargs) -> Any:
        """Invoke a skill by name."""
        skill = self._skills.get(name)
        if not skill:
            return f"Skill not found: {name}"
        try:
            return skill.handler(**kwargs)
        except Exception as e:
            return f"Skill '{name}' error: {e}"

    # ---- Loading ----

    def load_builtin_skills(self) -> None:
        """Load built-in skills that ship with the agent."""
        # Register built-in skills
        self._register_code_review()
        self._register_simplify()

    def load_from_directory(self) -> int:
        """Load skills from the skills directory. Returns count loaded."""
        count = 0
        if not self.skills_dir.exists():
            return count

        for fpath in self.skills_dir.rglob("*.py"):
            try:
                skill = self._load_python_skill(fpath)
                if skill:
                    self.register(skill)
                    count += 1
            except Exception as e:
                print(f"Warning: Failed to load skill {fpath}: {e}")

        return count

    def _load_python_skill(self, path: Path) -> Optional[Skill]:
        """Load a skill from a Python file."""
        module_name = path.stem
        spec = importlib.util.spec_from_file_location(f"skill_{module_name}", path)
        if not spec or not spec.loader:
            return None

        module = importlib.util.module_from_spec(spec)
        sys.modules[f"skill_{module_name}"] = module
        spec.loader.exec_module(module)

        # Look for skill metadata
        meta_attrs = ["SKILL_META", "skill_meta", "META", "__skill_meta__"]
        meta = None
        for attr in meta_attrs:
            if hasattr(module, attr):
                raw = getattr(module, attr)
                if isinstance(raw, dict):
                    meta = SkillMeta(**raw)
                elif isinstance(raw, SkillMeta):
                    meta = raw
                break

        if not meta:
            meta = SkillMeta(
                name=module_name,
                description=getattr(module, "__doc__", "") or module_name,
            )

        # Look for handler
        handler_attrs = ["handle", "run", "main", "execute"]
        handler = None
        for attr in handler_attrs:
            if hasattr(module, attr):
                handler = getattr(module, attr)
                break

        if not handler:
            return None

        return Skill(meta=meta, handler=handler)

    # ---- Built-in skill definitions ----

    def _register_code_review(self) -> None:
        """Register the built-in code-review skill."""
        def code_review_handler(files: list[str] | None = None, **kwargs) -> str:
            """
            Review code changes for bugs, style issues, and improvements.
            This is a simplified built-in version.
            """
            return (
                "## Code Review\n\n"
                "The code-review skill examines changed files for:\n"
                "- **Correctness bugs**: logic errors, edge cases, null handling\n"
                "- **Simplification**: code that can be simplified or deduplicated\n"
                "- **Efficiency**: performance issues, unnecessary allocations\n"
                "- **Test coverage**: untested code paths\n\n"
                "To use: `/code-review` or mention 'review my code'\n"
                "For a full review, provide specific file paths."
            )

        self.register(Skill(
            meta=SkillMeta(
                name="code-review",
                description="Review code changes for bugs, style, and improvements",
                triggers=["review", "code review", "check my code", "audit"],
                category="quality",
            ),
            handler=code_review_handler,
        ))

    def _register_simplify(self) -> None:
        """Register the built-in simplify skill."""
        def simplify_handler(files: list[str] | None = None, **kwargs) -> str:
            """
            Analyze code and suggest simplifications.
            This is a simplified built-in version.
            """
            return (
                "## Simplify\n\n"
                "The simplify skill analyzes code for:\n"
                "- **Reuse opportunities**: duplicate logic that can be extracted\n"
                "- **Over-abstraction**: patterns that are more complex than needed\n"
                "- **Dead code**: unused variables, functions, imports\n"
                "- **Idiomatic improvements**: more Pythonic ways to express the same logic\n\n"
                "To use: `/simplify` or mention 'simplify this code'"
            )

        self.register(Skill(
            meta=SkillMeta(
                name="simplify",
                description="Find simplification opportunities in code",
                triggers=["simplify", "clean up", "refactor"],
                category="quality",
            ),
            handler=simplify_handler,
        ))

    # ---- Skill discovery for system prompt ----

    def format_for_system_prompt(self) -> str:
        """Format available skills for inclusion in the system prompt."""
        if not self._skills:
            return ""

        lines = ["## Available Skills"]
        for skill in self._skills.values():
            triggers_str = ", ".join(skill.meta.triggers[:3]) if skill.meta.triggers else "no triggers"
            lines.append(f"- **/{skill.meta.name}**: {skill.meta.description}")
            lines.append(f"  Triggers: {triggers_str}")

        return "\n".join(lines)
