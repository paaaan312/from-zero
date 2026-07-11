"""
Core Agent Loop — the central ReAct (Reasoning + Acting) loop.
Processes user input, calls the LLM, executes tools, and iterates until completion.
"""

import json
import logging
import asyncio
from typing import Optional, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum

from .config import AgentConfig
from .tools import ToolRegistry, create_builtin_registry
from .system_prompt import build_system_prompt
from .streaming import (
    LLMBackend, LLMResponse, StreamEvent, StreamEventType,
    create_backend,
)
from .permissions import PermissionManager, PermissionResult
from .context import ContextManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------


class AgentState(Enum):
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"         # Executing tools
    WAITING_APPROVAL = "waiting_approval"
    DONE = "done"
    ERROR = "error"


@dataclass
class AgentStep:
    """A single step in the agent's execution trace."""
    step_number: int
    llm_response: Optional[LLMResponse] = None
    tool_calls: list[dict] = field(default_factory=list)  # [{name, arguments, result}]
    state: AgentState = AgentState.IDLE


@dataclass
class AgentResult:
    """Final result from an agent run."""
    content: str
    steps: list[AgentStep] = field(default_factory=list)
    total_tokens: int = 0
    success: bool = True
    error: str = ""


# ---------------------------------------------------------------------------
# Callback types
# ---------------------------------------------------------------------------

# Called when streaming text arrives
TextCallback = Callable[[str], Awaitable[None]]
# Called when a tool is about to be executed (for permission checks)
ToolApprovalCallback = Callable[[str, dict], Awaitable[bool]]
# Called on each step completion
StepCallback = Callable[[AgentStep], Awaitable[None]]


# ---------------------------------------------------------------------------
# The Agent
# ---------------------------------------------------------------------------


class Agent:
    """The core Coding Agent — manages the ReAct loop."""

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        tool_registry: Optional[ToolRegistry] = None,
    ):
        self.config = config or AgentConfig.from_env()
        self.tools = tool_registry or create_builtin_registry(self.config.workspace_dir)
        self.backend = create_backend(self.config.llm)
        self.permissions = PermissionManager(self.config.permissions, self.config.workspace_dir)
        self.context = ContextManager(self.config.context)
        self.state = AgentState.IDLE

        # Conversation history
        self._messages: list[dict] = []
        self._system_prompt: str = ""
        self._steps: list[AgentStep] = []

        # Callbacks
        self._on_text: Optional[TextCallback] = None
        self._on_approval: Optional[ToolApprovalCallback] = None
        self._on_step: Optional[StepCallback] = None

        # Memory context (populated from memory system)
        self._memory_context: str = ""

    # ---- Callback setters ----

    def on_text(self, callback: TextCallback) -> None:
        """Set callback for streaming text deltas."""
        self._on_text = callback

    def on_approval(self, callback: ToolApprovalCallback) -> None:
        """Set callback for tool approval requests."""
        self._on_approval = callback

    def on_step(self, callback: StepCallback) -> None:
        """Set callback for step completion."""
        self._on_step = callback

    # ---- Public API ----

    def set_memory_context(self, memories: str) -> None:
        """Set memory context for the next run."""
        self._memory_context = memories

    def reset(self) -> None:
        """Reset the conversation history."""
        self._messages = []
        self._steps = []
        self.state = AgentState.IDLE
        self.context = ContextManager(self.config.context)

    async def run(self, user_input: str, stream: bool = True) -> AgentResult:
        """
        Run the agent on a user input.

        Args:
            user_input: The user's message/request.
            stream: Whether to use streaming output.

        Returns:
            AgentResult with the final content and execution trace.
        """
        self.state = AgentState.THINKING
        self._steps = []

        # Build initial messages if this is the first turn
        if not self._messages:
            self._init_conversation()

        # Add user message
        self._messages.append({"role": "user", "content": user_input})

        # Main ReAct loop
        iteration = 0
        final_content = ""

        while iteration < self.config.max_tool_iterations:
            iteration += 1
            step = AgentStep(step_number=iteration, state=AgentState.THINKING)

            # Check context limits
            if self.context.should_summarize(self._messages):
                logger.info("Summarizing conversation to manage context window...")
                self._messages = self.context.summarize_conversation(
                    self._messages, self._system_prompt
                )

            # Call the LLM
            tools_schema = self.tools.to_openai_tools()
            response = await self.backend.chat(
                messages=self._messages,
                tools=tools_schema,
                stream=False,
            )

            if not response:
                final_content = "(no response from LLM)"
                self.state = AgentState.ERROR
                break

            step.llm_response = response

            # Track token usage
            self.context.track_usage_from_response({"usage": response.usage})

            # Add assistant message to history
            if response.tool_calls:
                # Format as tool_calls message
                assistant_msg = {
                    "role": "assistant",
                    "content": response.content or None,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["arguments"], ensure_ascii=False),
                            },
                        }
                        for tc in response.tool_calls
                    ],
                }
                self._messages.append(assistant_msg)

                # Execute each tool call
                self.state = AgentState.ACTING
                for tc in response.tool_calls:
                    tool_result = await self._execute_tool(tc["name"], tc["arguments"])
                    step.tool_calls.append({
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                        "result": tool_result,
                    })

                    # Add tool result to messages
                    self._messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": tool_result,
                    })
            else:
                # No tool calls — final response
                final_content = response.content
                self._messages.append({"role": "assistant", "content": response.content})
                self.state = AgentState.DONE
                self._steps.append(step)
                break

            self._steps.append(step)
            if self._on_step:
                await self._on_step(step)

            self.state = AgentState.THINKING

        if iteration >= self.config.max_tool_iterations:
            final_content = "(reached maximum tool iterations)"
            self.state = AgentState.ERROR

        return AgentResult(
            content=final_content,
            steps=self._steps,
            total_tokens=self.context.usage.total_tokens,
            success=self.state == AgentState.DONE,
            error="" if self.state == AgentState.DONE else final_content,
        )

    async def run_stream(self, user_input: str) -> AgentResult:
        """
        Run the agent with streaming output.

        This is a hybrid: uses non-streaming API for the main loop but
        can be extended to use true streaming for the final response.
        """
        # For now, use the non-streaming run with text callback support
        self.state = AgentState.THINKING
        self._steps = []

        if not self._messages:
            self._init_conversation()

        self._messages.append({"role": "user", "content": user_input})

        iteration = 0
        final_content = ""

        while iteration < self.config.max_tool_iterations:
            iteration += 1
            step = AgentStep(step_number=iteration, state=AgentState.THINKING)

            if self.context.should_summarize(self._messages):
                self._messages = self.context.summarize_conversation(
                    self._messages, self._system_prompt
                )

            tools_schema = self.tools.to_openai_tools()

            # Use streaming API
            content_parts = []
            tool_calls_in_progress: dict[int, dict] = {}

            async for event in self.backend.chat_stream(self._messages, tools_schema):
                if event.type == StreamEventType.TEXT_DELTA:
                    content_parts.append(event.content)
                    if self._on_text:
                        await self._on_text(event.content)

                elif event.type == StreamEventType.TOOL_CALL_START:
                    pass  # Tool call being built

                elif event.type == StreamEventType.TOOL_CALL_DELTA:
                    pass  # Arguments accumulating

                elif event.type == StreamEventType.TOOL_CALL_END:
                    # Tool call complete — collect it
                    idx = len(tool_calls_in_progress)
                    tool_calls_in_progress[idx] = {
                        "id": event.tool_id,
                        "name": event.tool_name,
                        "arguments": event.tool_arguments,
                    }

                elif event.type == StreamEventType.DONE:
                    if event.usage:
                        self.context.track_usage_from_response({"usage": event.usage})

                elif event.type == StreamEventType.ERROR:
                    return AgentResult(
                        content=f"Stream error: {event.content}",
                        steps=self._steps,
                        success=False,
                        error=event.content,
                    )

            content = "".join(content_parts)
            parsed_tool_calls = []

            for tc_data in tool_calls_in_progress.values():
                try:
                    args = json.loads(tc_data["arguments"]) if tc_data["arguments"] else {}
                except json.JSONDecodeError:
                    args = {}
                parsed_tool_calls.append({
                    "id": tc_data["id"],
                    "name": tc_data["name"],
                    "arguments": args,
                })

            if parsed_tool_calls:
                # Add assistant message with tool calls
                assistant_msg = {
                    "role": "assistant",
                    "content": content or None,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["arguments"], ensure_ascii=False),
                            },
                        }
                        for tc in parsed_tool_calls
                    ],
                }
                self._messages.append(assistant_msg)

                self.state = AgentState.ACTING
                for tc in parsed_tool_calls:
                    tool_result = await self._execute_tool(tc["name"], tc["arguments"])
                    step.tool_calls.append({
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                        "result": tool_result,
                    })

                    self._messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": tool_result,
                    })
            else:
                # Final text response
                final_content = content
                self._messages.append({"role": "assistant", "content": content})
                self.state = AgentState.DONE
                self._steps.append(step)
                break

            self._steps.append(step)
            if self._on_step:
                await self._on_step(step)
            self.state = AgentState.THINKING

        if iteration >= self.config.max_tool_iterations:
            final_content = "(reached maximum tool iterations)"
            self.state = AgentState.ERROR

        return AgentResult(
            content=final_content,
            steps=self._steps,
            total_tokens=self.context.usage.total_tokens,
            success=self.state == AgentState.DONE,
            error="" if self.state == AgentState.DONE else final_content,
        )

    # ---- Internal methods ----

    def _init_conversation(self) -> None:
        """Initialize the conversation with the system prompt."""
        self._system_prompt = build_system_prompt(
            self.config, self.tools, self._memory_context
        )
        self._messages = [{"role": "system", "content": self._system_prompt}]

    async def _execute_tool(self, name: str, arguments: dict) -> str:
        """Execute a tool with permission checks."""
        tool = self.tools.get(name)
        if not tool:
            return f"Unknown tool: {name}"

        # Permission check
        perm = self.permissions.check(tool.definition, arguments)

        if perm.result == PermissionResult.DENY:
            return f"Permission denied: {perm.reason}"

        if perm.result == PermissionResult.ASK:
            # Need user approval
            if self._on_approval:
                approved = await self._on_approval(name, arguments)
                if not approved:
                    self.permissions.deny(name, arguments)
                    return f"User denied tool execution: {name}"
                self.permissions.approve(name, arguments, remember=True)
            else:
                # In non-interactive mode, deny by default when approval needed
                return f"Tool '{name}' requires user approval but no approval handler is set."

        # Execute the tool
        return await tool.execute(**arguments)

    # ---- Session management ----

    def to_dict(self) -> dict:
        """Serialize agent state for session persistence."""
        return {
            "messages": self._messages,
            "system_prompt": self._system_prompt,
            "steps_count": len(self._steps),
            "config_workspace": self.config.workspace_dir,
        }

    def from_dict(self, data: dict) -> None:
        """Restore agent state from a saved session."""
        self._messages = data.get("messages", [])
        self._system_prompt = data.get("system_prompt", "")
        # Re-initialize the conversation if system prompt exists but no messages
        if self._system_prompt and not self._messages:
            self._messages = [{"role": "system", "content": self._system_prompt}]

    @property
    def messages(self) -> list[dict]:
        """Get the conversation history."""
        return self._messages

    @property
    def usage_report(self) -> str:
        """Get token usage report."""
        return self.context.get_usage_report()

    async def close(self) -> None:
        """Clean up resources."""
        await self.backend.close()
