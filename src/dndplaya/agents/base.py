from __future__ import annotations

import time as _time

import anthropic
from anthropic.types import MessageParam, TextBlock, ToolUseBlock, ThinkingBlock
from dataclasses import dataclass, field

from ..config import Settings


@dataclass
class ToolCall:
    """A tool call from the API response."""

    id: str
    name: str
    arguments: dict


@dataclass
class AgentResponse:
    """Response from a tool-use-enabled API call."""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_content: list = field(default_factory=list)
    stop_reason: str = "end_turn"


@dataclass
class Message:
    role: str  # "user" or "assistant"
    content: str | list  # str for simple messages, list for tool use blocks


class BaseAgent:
    """Base class for all LLM agents (DM, players, critic)."""

    MAX_RETRIES = 3
    RETRY_BASE_DELAY = 1.0  # seconds

    def __init__(
        self,
        name: str,
        system_prompt: str | list,
        settings: Settings,
        tools: list | None = None,
        enable_thinking: bool = False,
        thinking_budget: int = 500,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.settings = settings
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key.get_secret_value())
        self.history: list[Message] = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cache_creation_tokens = 0
        self.total_cache_read_tokens = 0
        self.last_input_tokens = 0  # tokens from the most recent API call
        self.tools = tools
        self.enable_thinking = enable_thinking
        self.thinking_budget = thinking_budget
        self.last_thinking: str | None = None  # most recent thinking block

    def _record_usage(self, response) -> None:
        """Record token usage from an API response, including cache metrics."""
        self.last_input_tokens = response.usage.input_tokens
        self.total_input_tokens += response.usage.input_tokens
        self.total_output_tokens += response.usage.output_tokens
        # Cache metrics — available when prompt caching is active
        cache_creation = getattr(response.usage, "cache_creation_input_tokens", 0) or 0
        cache_read = getattr(response.usage, "cache_read_input_tokens", 0) or 0
        self.total_cache_creation_tokens += cache_creation
        self.total_cache_read_tokens += cache_read

    @staticmethod
    def _add_cache_control(system_prompt: str | list) -> list:
        """Wrap system prompt with cache_control for prompt caching."""
        if isinstance(system_prompt, str):
            return [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}]
        # It's a list — add cache_control to the last block
        result = [dict(block) for block in system_prompt]
        if result:
            result[-1]["cache_control"] = {"type": "ephemeral"}
        return result

    def _make_api_call(self, messages: list[MessageParam], use_tools: bool = False):
        """Make an API call with retry logic. Returns the raw response."""
        kwargs = {
            "model": self.settings.model,
            "max_tokens": self.settings.max_tokens,
            "system": self._add_cache_control(self.system_prompt),
            "messages": messages,
            "timeout": 120.0,
        }
        if use_tools and self.tools:
            kwargs["tools"] = self.tools
        if self.enable_thinking:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget,
            }
            # Extended thinking requires max_tokens > budget_tokens
            kwargs["max_tokens"] = max(kwargs["max_tokens"], self.thinking_budget + 2048)

        last_error: Exception | None = None
        for attempt in range(self.MAX_RETRIES):
            try:
                response = self.client.messages.create(**kwargs)
                return response
            except (
                anthropic.APITimeoutError,
                anthropic.APIConnectionError,
                anthropic.RateLimitError,
                anthropic.InternalServerError,
            ) as e:
                last_error = e
                if attempt < self.MAX_RETRIES - 1:
                    _time.sleep(self.RETRY_BASE_DELAY * (2 ** attempt))
        raise last_error  # type: ignore[misc]

    def send(self, user_message: str) -> str:
        """Send a message and get a text response (no tool use).

        The user message is only committed to history after a successful API call,
        preventing history corruption on transient failures.
        """
        messages: list[MessageParam] = [
            {"role": m.role, "content": m.content}  # type: ignore[typeddict-item]
            for m in self.history
        ]
        self._mark_last_for_caching(messages)
        messages.append({"role": "user", "content": user_message})

        response = self._make_api_call(messages)

        if not response.content:
            raise ValueError(f"Empty response from API for agent '{self.name}'")
        # Find the first non-empty text block
        assistant_text = ""
        for block in response.content:
            if isinstance(block, TextBlock) and block.text.strip():
                assistant_text = block.text
                break
        if not assistant_text:
            raise ValueError(f"No text content in API response for agent '{self.name}'")
        self._record_usage(response)

        # Only commit to history after successful API call
        self.history.append(Message(role="user", content=user_message))
        self.history.append(Message(role="assistant", content=assistant_text))
        return assistant_text

    @staticmethod
    def _mark_last_for_caching(messages: list) -> None:
        """Add cache_control to the last message's content block.

        This marks everything up to (and including) that message as a
        cacheable prefix, so subsequent calls that share the same prefix
        get cache hits instead of reprocessing all prior tokens.

        Creates copies to avoid mutating the original history objects.
        """
        if not messages:
            return
        last = messages[-1]
        content = last["content"]
        if isinstance(content, str):
            last["content"] = [{
                "type": "text",
                "text": content,
                "cache_control": {"type": "ephemeral"},
            }]
        elif isinstance(content, list) and content:
            # Copy the list and last block to avoid mutating history
            new_content = list(content)
            last_block = new_content[-1]
            if isinstance(last_block, dict):
                new_content[-1] = {**last_block, "cache_control": {"type": "ephemeral"}}
            last["content"] = new_content

    def send_with_tools(self, user_message: str) -> AgentResponse:
        """Send a message with tool use enabled. Returns structured response."""
        messages: list[MessageParam] = [
            {"role": m.role, "content": m.content}  # type: ignore[typeddict-item]
            for m in self.history
        ]
        self._mark_last_for_caching(messages)
        messages.append({"role": "user", "content": user_message})

        response = self._make_api_call(messages, use_tools=True)
        return self._process_tool_response(response, user_message)

    def snapshot_history(self) -> int:
        """Save current history length for later rollback."""
        return len(self.history)

    def rollback_history(self, snapshot: int) -> None:
        """Roll back history to a previous snapshot point."""
        self.history = self.history[:snapshot]

    def submit_tool_results(self, tool_results: list[tuple[str, str]]) -> AgentResponse:
        """Submit tool results and get the next response.

        Expects the assistant's tool-use message to already be in history
        (committed by send_with_tools or a prior submit_tool_results).
        """
        result_content = [
            {"type": "tool_result", "tool_use_id": tool_id, "content": result}
            for tool_id, result in tool_results
        ]

        messages: list[MessageParam] = [
            {"role": m.role, "content": m.content}  # type: ignore[typeddict-item]
            for m in self.history
        ]
        self._mark_last_for_caching(messages)
        messages.append({"role": "user", "content": result_content})

        response = self._make_api_call(messages, use_tools=True)
        return self._process_tool_response(response, result_content)

    def _process_tool_response(self, response, user_content) -> AgentResponse:
        """Parse API response into AgentResponse and commit to history.

        Handles edge cases:
        - Empty response (model had nothing to say after tool results)
        - Empty text blocks (filtered out to avoid API rejection on next call)
        - Ensures history always has valid content the API will accept
        """
        self._record_usage(response)

        text_parts = []
        tool_calls = []
        raw_content = []
        self.last_thinking = None

        for block in (response.content or []):
            if isinstance(block, ThinkingBlock):
                self.last_thinking = block.thinking
                # Include the full block dict (with signature) for history round-trip
                raw_content.append({
                    "type": "thinking",
                    "thinking": block.thinking,
                    "signature": block.signature,
                })
            elif isinstance(block, TextBlock):
                # Skip empty text blocks — API rejects them on subsequent calls
                if block.text.strip():
                    text_parts.append(block.text)
                    raw_content.append({"type": "text", "text": block.text})
            elif isinstance(block, ToolUseBlock):
                tool_calls.append(ToolCall(
                    id=block.id, name=block.name, arguments=block.input,
                ))
                raw_content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })

        # Ensure assistant message is never empty (API requires valid content)
        if not raw_content:
            raw_content = [{"type": "text", "text": "(acknowledged)"}]

        # Commit to history
        self.history.append(Message(role="user", content=user_content))
        self.history.append(Message(role="assistant", content=raw_content))

        return AgentResponse(
            text="\n".join(text_parts),
            tool_calls=tool_calls,
            raw_content=raw_content,
            stop_reason=response.stop_reason,
        )

    def get_token_usage(self) -> dict[str, int]:
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "cache_creation_tokens": self.total_cache_creation_tokens,
            "cache_read_tokens": self.total_cache_read_tokens,
        }

    def dump_history(self) -> str:
        """Dump full conversation history as readable text for debugging."""
        lines = [f"=== {self.name} conversation log ===\n"]
        for msg in self.history:
            role = msg.role.upper()
            if isinstance(msg.content, str):
                lines.append(f"--- {role} ---\n{msg.content}\n")
            elif isinstance(msg.content, list):
                parts = []
                for block in msg.content:
                    if isinstance(block, dict):
                        if block.get("type") == "thinking":
                            parts.append(
                                f"[thinking]\n{block.get('thinking', '')}\n[/thinking]\n"
                            )
                        elif block.get("type") == "text":
                            parts.append(block.get("text", ""))
                        elif block.get("type") == "tool_use":
                            parts.append(
                                f"[tool_use: {block.get('name')}({block.get('input', {})})]\n"
                            )
                        elif block.get("type") == "tool_result":
                            content = block.get("content", "")
                            parts.append(f"[tool_result: {content[:200]}]\n")
                        elif block.get("type") == "image":
                            parts.append("[image]\n")
                lines.append(f"--- {role} ---\n{''.join(parts)}\n")
        return "\n".join(lines)

    def set_cached_context(self, context: str) -> None:
        """Replace history with a single cached context exchange.

        The context text is marked with cache_control so that
        system + tools + context form a cached prefix. Subsequent
        send/send_with_tools calls only pay full price for the new message.
        """
        self.history = [
            Message(role="user", content=[{
                "type": "text",
                "text": context,
                "cache_control": {"type": "ephemeral"},
            }]),
            Message(role="assistant", content="Understood. I'm ready to respond in character."),
        ]

    def reset(self) -> None:
        self.history.clear()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cache_creation_tokens = 0
        self.total_cache_read_tokens = 0
        self.last_input_tokens = 0
