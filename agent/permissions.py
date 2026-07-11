"""
Permission and security system for the coding agent.
Controls which tools can be executed, with allow/deny/ask modes,
path-based restrictions, and ask-once caching.
"""

import fnmatch
import os
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
from .config import PermissionConfig
from .tools import ToolDef


class PermissionResult(Enum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


@dataclass
class PermissionCheck:
    """Result of a permission check."""
    result: PermissionResult
    reason: str = ""
    cached: bool = False


class PermissionManager:
    """Manages tool execution permissions."""

    def __init__(self, config: PermissionConfig, workspace: str = "."):
        self.config = config
        self.workspace = Path(workspace).resolve()
        self._ask_cache: dict[str, PermissionResult] = {}  # tool_name:path -> result
        self._session_approved: set[str] = set()  # tools approved this session

    def check(self, tool: ToolDef, arguments: dict) -> PermissionCheck:
        """Check if a tool execution is permitted."""
        name = tool.name

        # 1. Check deny list first
        for pattern in self.config.denied_tools:
            if fnmatch.fnmatch(name, pattern):
                return PermissionCheck(PermissionResult.DENY, f"'{name}' is on the deny list")

        # 2. Check allow list
        for pattern in self.config.allowed_tools:
            if fnmatch.fnmatch(name, pattern):
                return PermissionCheck(PermissionResult.ALLOW, f"'{name}' is on the allow list")

        # 3. If mode is auto_allow, allow everything not denied
        if self.config.mode == "auto_allow":
            return PermissionCheck(PermissionResult.ALLOW, "auto_allow mode")

        # 4. Check if already approved this session
        if self.config.ask_once_per_session and name in self._session_approved:
            return PermissionCheck(PermissionResult.ALLOW, f"'{name}' approved this session", cached=True)

        # 5. Check path-based permissions for file tools
        if name in ("read_file", "write_file", "edit_file", "glob", "grep"):
            path_check = self._check_path(name, arguments)
            if path_check.result == PermissionResult.DENY:
                return path_check
            if path_check.result == PermissionResult.ALLOW:
                return path_check

        # 6. Check the ask cache
        cache_key = self._cache_key(name, arguments)
        if cache_key in self._ask_cache:
            return PermissionCheck(self._ask_cache[cache_key], "cached answer", cached=True)

        # 7. If the tool requires approval, ask
        if tool.requires_approval or self.config.mode == "strict":
            return PermissionCheck(PermissionResult.ASK, f"'{name}' requires approval")

        # 8. Default: allow
        if self.config.mode != "strict":
            return PermissionCheck(PermissionResult.ALLOW, "default allow")

        return PermissionCheck(PermissionResult.ASK, f"'{name}' requires approval in strict mode")

    def approve(self, tool_name: str, arguments: dict | None = None, remember: bool = False) -> None:
        """Approve a tool execution."""
        if remember or self.config.ask_once_per_session:
            self._session_approved.add(tool_name)
        if arguments:
            self._ask_cache[self._cache_key(tool_name, arguments)] = PermissionResult.ALLOW

    def deny(self, tool_name: str, arguments: dict | None = None) -> None:
        """Deny a tool execution."""
        if arguments:
            self._ask_cache[self._cache_key(tool_name, arguments)] = PermissionResult.DENY

    def _check_path(self, tool_name: str, arguments: dict) -> PermissionCheck:
        """Check path-based permissions."""
        target_path = arguments.get("file_path") or arguments.get("path")

        if not target_path:
            # glob/grep without path are fine (search within workspace)
            return PermissionCheck(PermissionResult.ALLOW, "workspace scoped")

        path = Path(target_path)
        if not path.is_absolute():
            path = (self.workspace / path).resolve()

        # Check if path is in the workspace
        try:
            path.relative_to(self.workspace)
            in_workspace = True
        except ValueError:
            in_workspace = False

        # Read-only tools can read anywhere
        if tool_name in ("read_file", "glob", "grep"):
            if in_workspace:
                return PermissionCheck(PermissionResult.ALLOW, "read in workspace")
            if self.config.mode == "strict":
                return PermissionCheck(PermissionResult.ASK, f"read outside workspace: {target_path}")
            return PermissionCheck(PermissionResult.ALLOW, f"read outside workspace: {target_path}")

        # Write tools: check path restrictions
        denied = False
        for pattern in self.config.denied_paths:
            if fnmatch.fnmatch(str(path), pattern):
                denied = True
                break

        if denied:
            return PermissionCheck(PermissionResult.DENY, f"path denied: {target_path}")

        allowed = False
        for pattern in self.config.allowed_paths:
            if fnmatch.fnmatch(str(path), pattern):
                allowed = True
                break

        if allowed or (in_workspace and not self.config.denied_paths):
            return PermissionCheck(PermissionResult.ALLOW, "path allowed")

        return PermissionCheck(PermissionResult.ASK, f"write outside allowed paths: {target_path}")

    def _cache_key(self, tool_name: str, arguments: dict) -> str:
        """Generate a cache key for ask-once."""
        path = arguments.get("file_path") or arguments.get("path") or ""
        return f"{tool_name}:{path}"

    def format_for_user(self, tool: ToolDef, arguments: dict) -> str:
        """Format a tool call for display to the user."""
        short_args = {}
        for k, v in arguments.items():
            if isinstance(v, str) and len(v) > 100:
                short_args[k] = v[:97] + "..."
            else:
                short_args[k] = v

        import json
        return f"**{tool.name}**\n```json\n{json.dumps(short_args, indent=2, ensure_ascii=False)}\n```"
