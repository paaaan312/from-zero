"""
Context window management for the coding agent.
Handles token counting, budget tracking, and automatic summarization
when approaching context limits.
"""

import tiktoken
from typing import Optional
from dataclasses import dataclass, field
from .config import ContextConfig


@dataclass
class TokenUsage:
    """Track token usage for a conversation."""
    system_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def add_input(self, tokens: int) -> None:
        self.input_tokens += tokens
        self.total_tokens += tokens

    def add_output(self, tokens: int) -> None:
        self.output_tokens += tokens
        self.total_tokens += tokens


class ContextManager:
    """Manages the conversation context window, tracking tokens and handling overflow."""

    # Approximate tokens per message overhead (role markers, formatting)
    MESSAGE_OVERHEAD = 4

    def __init__(self, config: ContextConfig):
        self.config = config
        self.usage = TokenUsage()
        self._summaries: list[str] = []
        self._message_count = 0
        try:
            self._encoder = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
        except Exception:
            self._encoder = None

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        if self._encoder:
            try:
                return len(self._encoder.encode(text))
            except Exception:
                pass
        # Fallback: rough estimate (4 chars ≈ 1 token for English)
        return len(text) // 4

    def count_messages(self, messages: list[dict]) -> int:
        """Count total tokens in a list of messages."""
        total = 0
        for msg in messages:
            total += self.MESSAGE_OVERHEAD
            content = msg.get("content", "")
            if isinstance(content, str):
                total += self.count_tokens(content)
            elif isinstance(content, list):
                # Multi-part content (e.g., tool results)
                for block in content:
                    if isinstance(block, dict):
                        total += self.count_tokens(str(block.get("text", "")))
                        total += self.count_tokens(str(block.get("tool_use_id", "")))
            total += self.count_tokens(str(msg.get("role", "")))
        return total

    def estimate_tokens(self, messages: list[dict]) -> int:
        """Estimate total tokens for a messages array."""
        return self.count_messages(messages)

    def remaining_budget(self) -> int:
        """Calculate remaining token budget."""
        used = self.usage.total_tokens
        limit = self.config.model_context_limit
        reserve = self.config.reserve_output_tokens
        return max(0, limit - used - reserve)

    def is_nearing_limit(self, messages: list[dict]) -> bool:
        """Check if we're approaching the context limit."""
        estimated = self.estimate_tokens(messages)
        limit = self.config.model_context_limit
        return estimated > limit * self.config.summarize_at_pct

    def should_summarize(self, messages: list[dict]) -> bool:
        """Determine if summarization is needed."""
        return self.is_nearing_limit(messages)

    def summarize_conversation(self, messages: list[dict], system_prompt: str) -> list[dict]:
        """
        Summarize older messages to make room.
        Keeps system prompt, last N messages, and replaces older ones with a summary.
        """
        if len(messages) <= 4:
            return messages  # Not enough to summarize

        # Keep system message + last 6 messages, summarize the rest
        keep_count = 6
        to_summarize = messages[1:-keep_count] if len(messages) > keep_count + 1 else []
        to_keep = messages[-keep_count:]

        if not to_summarize:
            return messages

        # Build a summary of the conversation history
        summary_parts = []
        for msg in to_summarize:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, str):
                preview = content[:200] + "..." if len(content) > 200 else content
                summary_parts.append(f"[{role}]: {preview}")

        summary = "## Conversation Summary (earlier)\n" + "\n".join(summary_parts)
        self._summaries.append(summary)

        # Reconstruct: system message + summary marker + recent messages
        result = [messages[0]]  # System prompt
        result.append({
            "role": "user",
            "content": f"[Earlier conversation summarized — {len(to_summarize)} messages compressed]\n{summary}",
        })
        result.extend(to_keep)

        return result

    def get_usage_report(self) -> str:
        """Get a human-readable token usage report."""
        limit = self.config.model_context_limit
        pct = (self.usage.total_tokens / limit * 100) if limit > 0 else 0
        return (
            f"Tokens: {self.usage.total_tokens:,}/{limit:,} ({pct:.1f}%) | "
            f"Input: {self.usage.input_tokens:,} | Output: {self.usage.output_tokens:,}"
        )

    def track_usage_from_response(self, response: dict) -> None:
        """Track token usage from an API response."""
        usage = response.get("usage", {})
        if usage:
            self.usage.add_input(usage.get("prompt_tokens", 0))
            self.usage.add_output(usage.get("completion_tokens", 0))
