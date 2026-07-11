"""
MCP (Model Context Protocol) client integration for the coding agent.

MCP allows the agent to connect to external tool servers via:
- stdio transport (subprocess-based)
- HTTP/SSE transport (network-based)

This module provides discovery of tools, resources, and prompts
from MCP-compatible servers.
"""

import json
import asyncio
import subprocess
import logging
from typing import Any, Optional, AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from .config import MCPConfig
from .tools import Tool, ToolDef, ToolParameter, ToolRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MCP Protocol types (JSON-RPC 2.0 subset)
# ---------------------------------------------------------------------------


@dataclass
class MCPTool:
    """A tool discovered from an MCP server."""
    name: str
    description: str = ""
    input_schema: dict = field(default_factory=dict)
    server_name: str = ""


@dataclass
class MCPServerInfo:
    """Information about a connected MCP server."""
    name: str
    protocol_version: str = ""
    server_info: dict = field(default_factory=dict)
    capabilities: dict = field(default_factory=dict)
    tools: list[MCPTool] = field(default_factory=list)
    connected: bool = False


# ---------------------------------------------------------------------------
# MCP Client (stdio transport)
# ---------------------------------------------------------------------------


class MCPStdioClient:
    """
    MCP client using stdio transport.
    Spawns a subprocess and communicates via JSON-RPC over stdin/stdout.
    """

    def __init__(self, command: str, args: list[str] | None = None,
                 env: dict | None = None, cwd: str | None = None):
        self.command = command
        self.args = args or []
        self.env = env
        self.cwd = cwd
        self._process: Optional[subprocess.Popen] = None
        self._request_id = 0
        self._server_info: dict = {}

    async def connect(self) -> bool:
        """Start the MCP server subprocess and initialize the connection."""
        try:
            self._process = subprocess.Popen(
                [self.command] + self.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self.env,
                cwd=self.cwd,
                text=True,
                encoding="utf-8",
            )

            # Initialize the connection
            result = await self._send_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "from-zero-agent", "version": "0.1.0"},
            })

            if result:
                self._server_info = result
                # Send initialized notification
                await self._send_notification("initialized", {})
                return True

            return False
        except Exception as e:
            logger.error(f"MCP connect error ({self.command}): {e}")
            return False

    async def list_tools(self) -> list[MCPTool]:
        """Discover tools from the MCP server."""
        result = await self._send_request("tools/list", {})
        if not result:
            return []

        tools = []
        for t in result.get("tools", []):
            tools.append(MCPTool(
                name=t.get("name", "unknown"),
                description=t.get("description", ""),
                input_schema=t.get("inputSchema", {}),
            ))
        return tools

    async def call_tool(self, name: str, arguments: dict) -> str:
        """Call a tool on the MCP server."""
        result = await self._send_request("tools/call", {
            "name": name,
            "arguments": arguments,
        })
        if result is None:
            return f"Error calling MCP tool '{name}'"
        if "error" in result:
            return f"MCP tool error: {result['error']}"

        # Extract content from result
        content = result.get("content", [])
        if isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    texts.append(item.get("text", ""))
            return "\n".join(texts)
        return str(content)

    async def list_resources(self) -> list[dict]:
        """List available resources from the MCP server."""
        result = await self._send_request("resources/list", {})
        return result.get("resources", []) if result else []

    async def read_resource(self, uri: str) -> str:
        """Read a resource from the MCP server."""
        result = await self._send_request("resources/read", {"uri": uri})
        if not result:
            return f"Error reading resource: {uri}"
        contents = result.get("contents", [])
        if contents:
            return contents[0].get("text", str(contents[0]))
        return str(result)

    async def disconnect(self) -> None:
        """Shut down the MCP server connection."""
        if self._process and self._process.poll() is None:
            try:
                self._process.stdin.close()
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                self._process.kill()

    async def _send_request(self, method: str, params: dict) -> dict | None:
        """Send a JSON-RPC request and wait for the response."""
        if not self._process or self._process.poll() is not None:
            return None

        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params,
        }

        try:
            request_str = json.dumps(request) + "\n"
            self._process.stdin.write(request_str)
            self._process.stdin.flush()

            # Read response (line-delimited JSON)
            response_line = self._process.stdout.readline()
            if response_line:
                response = json.loads(response_line)
                if "error" in response:
                    logger.error(f"MCP error: {response['error']}")
                    return None
                return response.get("result")
        except Exception as e:
            logger.error(f"MCP request error: {e}")
            return None

        return None

    def _send_notification(self, method: str, params: dict) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        if not self._process or self._process.poll() is not None:
            return

        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }

        try:
            self._process.stdin.write(json.dumps(notification) + "\n")
            self._process.stdin.flush()
        except Exception as e:
            logger.error(f"MCP notification error: {e}")

    @property
    def is_connected(self) -> bool:
        return self._process is not None and self._process.poll() is None


# ---------------------------------------------------------------------------
# MCP Manager
# ---------------------------------------------------------------------------


class MCPManager:
    """
    Manages connections to multiple MCP servers.
    Discovers tools from all servers and registers them in the agent's tool registry.
    """

    def __init__(self, config: MCPConfig):
        self.config = config
        self._clients: dict[str, MCPStdioClient] = {}
        self._servers: dict[str, MCPServerInfo] = {}

    async def connect_all(self) -> dict[str, bool]:
        """Connect to all configured MCP servers."""
        results = {}

        for server_cfg in self.config.servers:
            name = server_cfg.get("name", f"server_{len(self._clients)}")
            command = server_cfg.get("command", "")
            args = server_cfg.get("args", [])
            env = server_cfg.get("env")

            if not command:
                results[name] = False
                continue

            client = MCPStdioClient(command, args, env)
            success = await client.connect()

            if success:
                self._clients[name] = client
                tools = await client.list_tools()
                self._servers[name] = MCPServerInfo(
                    name=name,
                    protocol_version="2024-11-05",
                    server_info=client._server_info,
                    tools=tools,
                    connected=True,
                )

            results[name] = success

        return results

    async def discover_tools(self) -> list[tuple[str, MCPTool]]:
        """Discover tools from all connected servers. Returns (server_name, tool) pairs."""
        all_tools = []
        for name, client in self._clients.items():
            if client.is_connected:
                tools = await client.list_tools()
                for tool in tools:
                    tool.server_name = name
                    all_tools.append((name, tool))
        return all_tools

    def register_in_registry(self, registry: ToolRegistry) -> int:
        """
        Register discovered MCP tools into the agent's tool registry.
        Must be called after discover_tools().
        Returns the number of tools registered.
        """
        count = 0
        for server_name, server_info in self._servers.items():
            if not server_info.connected:
                continue

            for mcp_tool in server_info.tools:
                client = self._clients.get(server_name)
                if not client:
                    continue

                # Convert MCP tool schema to our ToolDef
                schema = mcp_tool.input_schema
                properties = schema.get("properties", {})
                required = schema.get("required", [])

                parameters = []
                for param_name, param_info in properties.items():
                    parameters.append(ToolParameter(
                        name=param_name,
                        type=param_info.get("type", "string"),
                        description=param_info.get("description", ""),
                        required=param_name in required,
                    ))

                tool_def = ToolDef(
                    name=f"mcp_{server_name}_{mcp_tool.name}",
                    description=f"[MCP:{server_name}] {mcp_tool.description}",
                    parameters=parameters,
                    category=f"mcp/{server_name}",
                )

                # Create a handler that calls the MCP tool
                _client = client
                _tool_name = mcp_tool.name

                def make_handler(c, tn):
                    def handler(**kwargs):
                        import asyncio as _asyncio
                        return _asyncio.run(c.call_tool(tn, kwargs))
                    return handler

                tool = Tool(
                    definition=tool_def,
                    handler=make_handler(_client, _tool_name),
                )

                registry.register(tool)
                count += 1

        return count

    async def disconnect_all(self) -> None:
        """Disconnect from all MCP servers."""
        for client in self._clients.values():
            await client.disconnect()
        self._clients.clear()
        self._servers.clear()

    def get_status(self) -> str:
        """Get a human-readable status of all MCP connections."""
        lines = ["## MCP Servers"]
        for name, info in self._servers.items():
            status = "✅ Connected" if info.connected else "❌ Disconnected"
            tool_count = len(info.tools)
            lines.append(f"- **{name}**: {status} ({tool_count} tools)")
            if info.tools:
                for t in info.tools[:5]:
                    lines.append(f"  - `{t.name}`: {t.description[:80]}")
                if len(info.tools) > 5:
                    lines.append(f"  - ... and {len(info.tools) - 5} more")
        if not self._servers:
            lines.append("No MCP servers configured.")
        return "\n".join(lines)
