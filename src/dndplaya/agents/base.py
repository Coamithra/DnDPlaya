from __future__ import annotations

import time as _time

import anthropic
from anthropic.types import MessageParam, TextBlock, ToolUseBlock
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
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.settings = settings
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key.get_secret_value())
        self.history: list[Message] = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.last_input_tokens = 0  # tokens from the most recent API call
        self.tools = tools

    def _make_api_call(self, messages: list[MessageParam], use_tools: bool = False):
        """Make an API call with retry logic. Returns the raw response."""
        kwargs = {
            "model": self.settings.model,
            "max_tokens": self.settings.max_tokens,
            "system": self.system_prompt,
            "messages": messages,
            "timeout": 60.0,
        }
        if use_tools and self.tools:
            kwargs["tools"] = self.tools

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
        messages.append({"role": "user", "content": user_message})

        response = self._make_api_call(messages)

        if not response.content:
            raise ValueError(f"Empty response from API for agent '{self.name}'")
        first_block = response.content[0]
        if not isinstance(first_block, TextBlock):
            raise TypeError(
                f"Expected TextBlock from API, got {type(first_block).__name__} "
                f"for agent '{self.name}'"
            )
        assistant_text = first_block.text
        self.last_input_tokens = response.usage.input_tokens
        self.total_input_tokens += response.usage.input_tokens
        self.total_output_tokens += response.usage.output_tokens

        # Only commit to history after successful API call
        self.history.append(Message(role="user", content=user_message))
        self.history.append(Message(role="assistant", content=assistant_text))
        return assistant_text

    def send_with_tools(self, user_message: str) -> AgentResponse:
        """Send a message with tool use enabled. Returns structured response."""
        messages: list[MessageParam] = [
            {"role": m.role, "content": m.content}  # type: ignore[typeddict-item]
            for m in self.history
        ]
        messages.append({"role": "user", "content": user_message})

        response = self._make_api_call(messages, use_tools=True)
        return self._process_tool_response(response, user_message)

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
        messages.append({"role": "user", "content": result_content})

        response = self._make_api_call(messages, use_tools=True)
        return self._process_tool_response(response, result_content)

    def _process_tool_response(self, response, user_content) -> AgentResponse:
        """Parse API response into AgentResponse and commit to history."""
        if not response.content:
            raise ValueError(f"Empty response from API for agent '{self.name}'")

        text_parts = []
        tool_calls = []
        raw_content = []

        for block in response.content:
            if isinstance(block, TextBlock):
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

        self.last_input_tokens = response.usage.input_tokens
        self.total_input_tokens += response.usage.input_tokens
        self.total_output_tokens += response.usage.output_tokens

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
        }

    def reset(self) -> None:
        self.history.clear()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.last_input_tokens = 0
