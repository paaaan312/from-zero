"""
Streaming output and multi-backend LLM support.
Supports OpenAI-compatible (OpenAI, DeepSeek, etc.) and Anthropic backends
with a unified streaming interface.
"""

import json
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional
from dataclasses import dataclass, field
from enum import Enum

import httpx

from .config import LLMConfig
from .tools import ToolRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Streaming event types
# ---------------------------------------------------------------------------


class StreamEventType(Enum):
    TEXT_DELTA = "text_delta"       # Incremental text content
    TOOL_CALL_START = "tool_call_start"  # Tool call beginning
    TOOL_CALL_DELTA = "tool_call_delta"  # Tool call argument chunk
    TOOL_CALL_END = "tool_call_end"     # Tool call complete
    THINKING_DELTA = "thinking_delta"   # Thinking/reasoning content
    DONE = "done"                  # Stream complete
    ERROR = "error"                # Error occurred


@dataclass
class StreamEvent:
    """A single event in the response stream."""
    type: StreamEventType
    content: str = ""
    tool_name: str = ""
    tool_id: str = ""
    tool_arguments: str = ""  # Accumulated JSON arguments
    finish_reason: str = ""
    usage: dict = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Complete (non-streaming) response from an LLM."""
    content: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    finish_reason: str = ""
    usage: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract backend
# ---------------------------------------------------------------------------


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(120.0))

    @abstractmethod
    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        stream: bool = False,
    ) -> LLMResponse:
        """Send a chat completion request (non-streaming)."""
        ...

    @abstractmethod
    async def chat_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Send a chat completion request with streaming."""
        ...

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    def _headers(self) -> dict:
        """Build HTTP headers."""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }


# ---------------------------------------------------------------------------
# OpenAI-compatible backend (OpenAI, DeepSeek, local models, etc.)
# ---------------------------------------------------------------------------


class OpenAICompatibleBackend(LLMBackend):
    """Backend for OpenAI and OpenAI-compatible APIs."""

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        stream: bool = False,
    ) -> LLMResponse:
        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        url = f"{self.config.api_base.rstrip('/')}/chat/completions"

        try:
            resp = await self._client.post(url, headers=self._headers(), json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as e:
            logger.error(f"LLM API error: {e}")
            return LLMResponse(content=f"Error calling LLM: {e}")

        choice = data["choices"][0]
        message = choice["message"]

        tool_calls = []
        if message.get("tool_calls"):
            for tc in message["tool_calls"]:
                tool_calls.append({
                    "id": tc["id"],
                    "name": tc["function"]["name"],
                    "arguments": json.loads(tc["function"]["arguments"]) if tc["function"]["arguments"] else {},
                })

        return LLMResponse(
            content=message.get("content") or "",
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason", ""),
            usage=data.get("usage", {}),
        )

    async def chat_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        url = f"{self.config.api_base.rstrip('/')}/chat/completions"

        try:
            async with self._client.stream("POST", url, headers=self._headers(), json=payload) as resp:
                resp.raise_for_status()

                # Track tool call accumulation
                tool_calls_in_progress: dict[int, dict] = {}
                content_buffer = ""

                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break

                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    delta = data.get("choices", [{}])[0].get("delta", {})
                    finish = data.get("choices", [{}])[0].get("finish_reason")

                    # Text content
                    if delta.get("content"):
                        content_buffer += delta["content"]
                        yield StreamEvent(
                            type=StreamEventType.TEXT_DELTA,
                            content=delta["content"],
                        )

                    # Tool calls
                    if delta.get("tool_calls"):
                        for tc in delta["tool_calls"]:
                            idx = tc.get("index", 0)
                            if idx not in tool_calls_in_progress:
                                tool_calls_in_progress[idx] = {
                                    "id": tc.get("id", ""),
                                    "name": "",
                                    "arguments": "",
                                }
                                yield StreamEvent(
                                    type=StreamEventType.TOOL_CALL_START,
                                    tool_id=tc.get("id", ""),
                                    tool_name="",
                                )

                            entry = tool_calls_in_progress[idx]
                            if tc.get("id"):
                                entry["id"] = tc["id"]
                            if tc.get("function", {}).get("name"):
                                entry["name"] = tc["function"]["name"]
                                yield StreamEvent(
                                    type=StreamEventType.TOOL_CALL_START,
                                    tool_name=entry["name"],
                                    tool_id=entry["id"],
                                )
                            if tc.get("function", {}).get("arguments"):
                                entry["arguments"] += tc["function"]["arguments"]
                                yield StreamEvent(
                                    type=StreamEventType.TOOL_CALL_DELTA,
                                    tool_name=entry["name"],
                                    tool_id=entry["id"],
                                    tool_arguments=entry["arguments"],
                                )

                    # Finish
                    if finish:
                        for entry in tool_calls_in_progress.values():
                            yield StreamEvent(
                                type=StreamEventType.TOOL_CALL_END,
                                tool_name=entry["name"],
                                tool_id=entry["id"],
                                tool_arguments=entry["arguments"],
                            )

                        yield StreamEvent(
                            type=StreamEventType.DONE,
                            finish_reason=finish,
                            usage=data.get("usage", {}),
                        )

        except httpx.HTTPError as e:
            logger.error(f"Stream error: {e}")
            yield StreamEvent(type=StreamEventType.ERROR, content=str(e))


# ---------------------------------------------------------------------------
# Anthropic backend
# ---------------------------------------------------------------------------


class AnthropicBackend(LLMBackend):
    """Backend for Anthropic's Claude API."""

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "x-api-key": self.config.api_key,
            "anthropic-version": self.config.anthropic_version,
        }

    def _convert_messages(self, messages: list[dict]) -> tuple[str | None, list[dict]]:
        """Convert OpenAI-format messages to Anthropic format.
        Returns (system_prompt, messages_list).
        """
        system_prompt = None
        anthropic_messages = []

        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")

            if role == "system":
                system_prompt = content if isinstance(content, str) else str(content)
                continue

            # Map roles
            anthropic_role = "assistant" if role == "assistant" else "user"

            if isinstance(content, str):
                anthropic_messages.append({"role": anthropic_role, "content": content})
            elif isinstance(content, list):
                # Multi-part content
                parts = []
                tool_uses = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "tool_use":
                            tool_uses.append(block)
                        elif block.get("type") == "tool_result":
                            parts.append({
                                "type": "tool_result",
                                "tool_use_id": block.get("tool_use_id", ""),
                                "content": block.get("content", ""),
                            })
                        else:
                            parts.append({"type": "text", "text": str(block)})

                if tool_uses:
                    anthropic_messages.append({"role": "assistant", "content": tool_uses})
                if parts:
                    anthropic_messages.append({"role": "user", "content": parts})

        return system_prompt, anthropic_messages

    def _convert_tools(self, tools: list[dict]) -> list[dict]:
        """Convert OpenAI tool format to Anthropic format."""
        anthropic_tools = []
        for t in tools:
            func = t.get("function", t)
            anthropic_tools.append({
                "name": func["name"],
                "description": func.get("description", ""),
                "input_schema": func.get("parameters", {"type": "object", "properties": {}, "required": []}),
            })
        return anthropic_tools

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        stream: bool = False,
    ) -> LLMResponse:
        system_prompt, anthropic_msgs = self._convert_messages(messages)

        payload = {
            "model": self.config.model,
            "messages": anthropic_msgs,
            "max_tokens": self.config.max_tokens,
        }
        if system_prompt:
            payload["system"] = system_prompt
        if tools:
            payload["tools"] = self._convert_tools(tools)

        url = f"{self.config.api_base.rstrip('/')}/messages"

        try:
            resp = await self._client.post(url, headers=self._headers(), json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as e:
            logger.error(f"Anthropic API error: {e}")
            return LLMResponse(content=f"Error calling Anthropic: {e}")

        # Parse response
        content_text = ""
        tool_calls = []

        for block in data.get("content", []):
            if block["type"] == "text":
                content_text += block["text"]
            elif block["type"] == "tool_use":
                tool_calls.append({
                    "id": block["id"],
                    "name": block["name"],
                    "arguments": block.get("input", {}),
                })

        return LLMResponse(
            content=content_text,
            tool_calls=tool_calls,
            finish_reason=data.get("stop_reason", ""),
            usage=data.get("usage", {}),
        )

    async def chat_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        system_prompt, anthropic_msgs = self._convert_messages(messages)

        payload = {
            "model": self.config.model,
            "messages": anthropic_msgs,
            "max_tokens": self.config.max_tokens,
            "stream": True,
        }
        if system_prompt:
            payload["system"] = system_prompt
        if tools:
            payload["tools"] = self._convert_tools(tools)

        url = f"{self.config.api_base.rstrip('/')}/messages"

        try:
            async with self._client.stream("POST", url, headers=self._headers(), json=payload) as resp:
                resp.raise_for_status()

                current_tool_id = ""
                current_tool_name = ""
                current_tool_input = ""
                content_buffer = ""

                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]

                    try:
                        event = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    evt_type = event.get("type", "")

                    if evt_type == "content_block_delta":
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta":
                            content_buffer += delta.get("text", "")
                            yield StreamEvent(
                                type=StreamEventType.TEXT_DELTA,
                                content=delta.get("text", ""),
                            )
                        elif delta.get("type") == "input_json_delta":
                            current_tool_input += delta.get("partial_json", "")
                            yield StreamEvent(
                                type=StreamEventType.TOOL_CALL_DELTA,
                                tool_name=current_tool_name,
                                tool_id=current_tool_id,
                                tool_arguments=current_tool_input,
                            )
                        elif delta.get("type") == "thinking_delta":
                            yield StreamEvent(
                                type=StreamEventType.THINKING_DELTA,
                                content=delta.get("thinking", ""),
                            )

                    elif evt_type == "content_block_start":
                        block = event.get("content_block", {})
                        if block.get("type") == "tool_use":
                            current_tool_id = block.get("id", "")
                            current_tool_name = block.get("name", "")
                            current_tool_input = ""
                            yield StreamEvent(
                                type=StreamEventType.TOOL_CALL_START,
                                tool_name=current_tool_name,
                                tool_id=current_tool_id,
                            )

                    elif evt_type == "content_block_stop":
                        if current_tool_name:
                            yield StreamEvent(
                                type=StreamEventType.TOOL_CALL_END,
                                tool_name=current_tool_name,
                                tool_id=current_tool_id,
                                tool_arguments=current_tool_input,
                            )
                            current_tool_name = ""
                            current_tool_id = ""
                            current_tool_input = ""

                    elif evt_type == "message_delta":
                        usage = event.get("usage", {})
                        yield StreamEvent(
                            type=StreamEventType.DONE,
                            finish_reason=event.get("delta", {}).get("stop_reason", ""),
                            usage=usage,
                        )

                    elif evt_type == "message_stop":
                        yield StreamEvent(
                            type=StreamEventType.DONE,
                            finish_reason="end_turn",
                        )

                    elif evt_type == "error":
                        yield StreamEvent(
                            type=StreamEventType.ERROR,
                            content=event.get("error", {}).get("message", "Unknown error"),
                        )

        except httpx.HTTPError as e:
            logger.error(f"Anthropic stream error: {e}")
            yield StreamEvent(type=StreamEventType.ERROR, content=str(e))


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------


def create_backend(config: LLMConfig) -> LLMBackend:
    """Create the appropriate backend based on configuration."""
    provider = config.provider.lower()
    if provider == "anthropic":
        return AnthropicBackend(config)
    else:
        # Default: OpenAI-compatible (covers OpenAI, DeepSeek, local models, etc.)
        return OpenAICompatibleBackend(config)
