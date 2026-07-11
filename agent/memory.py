"""
Persistent memory system for the coding agent.
File-based memory with YAML frontmatter, MEMORY.md index, and relevance-based recall.

Each memory is one .md file in the memory directory with frontmatter:
---
name: short-kebab-slug
description: one-line summary
metadata:
  type: user | feedback | project | reference
---
<the fact>
"""

import os
import re
import yaml
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime

from .config import MemoryConfig


@dataclass
class MemoryEntry:
    """A single memory entry."""
    name: str
    description: str = ""
    content: str = ""
    metadata: dict = field(default_factory=dict)
    file_path: str = ""

    @property
    def mem_type(self) -> str:
        return self.metadata.get("type", "reference")

    @property
    def display_title(self) -> str:
        return self.name.replace("-", " ").title()


class MemorySystem:
    """
    File-based persistent memory for the agent.

    Memory directory structure:
        .agent/memory/
        ├── MEMORY.md          # Index of all memories
        ├── user-preference.md # Individual memory files
        ├── project-context.md
        └── ...
    """

    FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)

    def __init__(self, config: MemoryConfig, workspace: str = "."):
        self.config = config
        self.workspace = Path(workspace)
        self.memory_dir = self.workspace / config.memory_dir
        self.index_path = self.memory_dir / config.index_file
        self._cache: dict[str, MemoryEntry] = {}
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        """Ensure the memory directory exists."""
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        if not self.index_path.exists():
            self.index_path.write_text("# Memory Index\n\n", encoding="utf-8")

    # ---- CRUD ----

    def save(self, entry: MemoryEntry) -> None:
        """Save or update a memory entry."""
        # Check if an entry with this name already exists
        existing = self._find_by_name(entry.name)
        if existing:
            entry.file_path = existing.file_path

        if not entry.file_path:
            # Generate a filename
            safe_name = entry.name.replace("/", "-").replace("\\", "-")
            entry.file_path = str(self.memory_dir / f"{safe_name}.md")

        # Write the memory file
        metadata = entry.metadata.copy()
        metadata.setdefault("type", "reference")
        if "created" not in metadata:
            metadata["created"] = datetime.now().isoformat()
        metadata["updated"] = datetime.now().isoformat()

        frontmatter = yaml.dump(metadata, default_flow_style=False, allow_unicode=True).strip()
        content = f"---\n{frontmatter}\n---\n\n{entry.content}"

        file_path = Path(entry.file_path)
        file_path.write_text(content, encoding="utf-8")

        # Update index
        self._update_index(entry)
        self._cache[entry.name] = entry

    def load(self, name: str) -> Optional[MemoryEntry]:
        """Load a single memory by name."""
        if name in self._cache:
            return self._cache[name]

        entry = self._find_by_name(name)
        if entry:
            return self._read_file(entry.file_path)
        return None

    def load_all(self) -> list[MemoryEntry]:
        """Load all memories."""
        entries = []
        if not self.memory_dir.exists():
            return entries

        for fpath in self.memory_dir.glob("*.md"):
            if fpath.name == self.config.index_file:
                continue
            entry = self._read_file(str(fpath))
            if entry:
                entries.append(entry)
                self._cache[entry.name] = entry
        return entries

    def delete(self, name: str) -> bool:
        """Delete a memory by name."""
        entry = self._find_by_name(name)
        if not entry:
            return False

        file_path = Path(entry.file_path)
        if file_path.exists():
            file_path.unlink()

        self._cache.pop(name, None)
        self._remove_from_index(name)
        return True

    def recall(self, query: str, max_results: int | None = None) -> list[MemoryEntry]:
        """
        Recall memories relevant to a query using simple keyword matching.

        In a production system, this would use embeddings + vector search.
        For our simple agent, we use keyword overlap scoring.
        """
        max_results = max_results or self.config.max_recall
        all_memories = self.load_all()

        if not all_memories:
            return []

        query_lower = query.lower()
        query_words = set(query_lower.split())

        scored = []
        for entry in all_memories:
            score = 0
            text = (entry.name + " " + entry.description + " " + entry.content).lower()

            # Keyword overlap
            for word in query_words:
                if word in text:
                    score += 1

            # Name match bonus
            if entry.name.lower() in query_lower:
                score += 10

            # Description match bonus
            if entry.description.lower() in query_lower:
                score += 5

            # Type boost: feedback > project > user > reference
            type_boost = {"feedback": 3, "project": 2, "user": 1, "reference": 0}
            score += type_boost.get(entry.mem_type, 0)

            if score > 0:
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:max_results]]

    def format_for_context(self, entries: list[MemoryEntry]) -> str:
        """Format memory entries for inclusion in the system prompt."""
        if not entries:
            return ""

        lines = []
        for entry in entries:
            lines.append(f"### {entry.display_title}")
            if entry.description:
                lines.append(f"*{entry.description}*")
            lines.append(entry.content[:500])  # Truncate long memories
            lines.append("")

        return "\n".join(lines)

    def get_index_summary(self) -> str:
        """Get a summary of all memories for display."""
        entries = self.load_all()
        if not entries:
            return "No memories stored."

        by_type: dict[str, list] = {}
        for e in entries:
            by_type.setdefault(e.mem_type, []).append(e)

        lines = [f"## Memories ({len(entries)} total)\n"]
        for mtype, type_entries in sorted(by_type.items()):
            lines.append(f"### {mtype.title()}")
            for e in type_entries:
                lines.append(f"- **{e.display_title}**: {e.description}")
            lines.append("")

        return "\n".join(lines)

    # ---- Internal helpers ----

    def _find_by_name(self, name: str) -> Optional[MemoryEntry]:
        """Find a memory entry by name (checks all .md files)."""
        safe_name = name.replace("/", "-").replace("\\", "-")
        expected_path = self.memory_dir / f"{safe_name}.md"

        if expected_path.exists():
            entry = self._read_file(str(expected_path))
            if entry:
                return entry

        # Fallback: search all files
        if self.memory_dir.exists():
            for fpath in self.memory_dir.glob("*.md"):
                if fpath.name == self.config.index_file:
                    continue
                entry = self._read_file(str(fpath))
                if entry and entry.name == name:
                    return entry

        return None

    def _read_file(self, file_path: str) -> Optional[MemoryEntry]:
        """Read a memory file and parse frontmatter."""
        path = Path(file_path)
        if not path.exists():
            return None

        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            return None

        metadata = {}
        content = text

        # Parse frontmatter
        m = self.FRONTMATTER_RE.match(text)
        if m:
            try:
                metadata = yaml.safe_load(m.group(1)) or {}
            except yaml.YAMLError:
                pass
            content = text[m.end():].strip()

        name = metadata.get("name", path.stem)
        description = metadata.get("description", "")

        return MemoryEntry(
            name=name,
            description=description,
            content=content,
            metadata=metadata,
            file_path=str(path),
        )

    def _update_index(self, entry: MemoryEntry) -> None:
        """Update the MEMORY.md index file."""
        index_content = self.index_path.read_text(encoding="utf-8")

        # Format the index line
        line = f"- [{entry.display_title}]({Path(entry.file_path).name}) — {entry.description}"

        # Check if entry already in index
        marker = f"[{entry.display_title}]"
        if marker in index_content:
            # Replace existing line
            lines = index_content.split("\n")
            new_lines = []
            for l in lines:
                if marker in l:
                    new_lines.append(line)
                else:
                    new_lines.append(l)
            index_content = "\n".join(new_lines)
        else:
            # Append new line
            index_content = index_content.rstrip() + "\n" + line + "\n"

        self.index_path.write_text(index_content, encoding="utf-8")

    def _remove_from_index(self, name: str) -> None:
        """Remove an entry from MEMORY.md index."""
        if not self.index_path.exists():
            return

        index_content = self.index_path.read_text(encoding="utf-8")
        display_title = name.replace("-", " ").title()
        marker = f"[{display_title}]"

        lines = index_content.split("\n")
        new_lines = [l for l in lines if marker not in l]
        self.index_path.write_text("\n".join(new_lines), encoding="utf-8")
