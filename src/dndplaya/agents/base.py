from __future__ import annotations

from dataclasses import dataclass, field

from ..config import Settings
from .provider import LLMResponse, create_provider


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
        self.provider = create_provider(settings)
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

    def _record_usage(self, resp: LLMResponse) -> None:
        """Record token usage from a provider response, including cache metrics."""
        self.last_input_tokens = resp.input_tokens
        self.total_input_tokens += resp.input_tokens
        self.total_output_tokens += resp.output_tokens
        self.total_cache_creation_tokens += resp.cache_creation_tokens
        self.total_cache_read_tokens += resp.cache_read_tokens

    def _make_api_call(self, messages: list[dict], use_tools: bool = False) -> LLMResponse:
        """Delegate to the provider. Returns a normalised LLMResponse."""
        return self.provider.call(
            messages=messages,
            system=self.system_prompt,
            tools=self.tools if use_tools else None,
            max_tokens=self.settings.max_tokens,
            enable_thinking=self.enable_thinking,
            thinking_budget=self.thinking_budget,
        )

    def send(self, user_message: str) -> str:
        """Send a message and get a text response (no tool use).

        The user message is only committed to history after a successful API call,
        preventing history corruption on transient failures.
        """
        messages = [
            {"role": m.role, "content": m.content}
            for m in self.history
        ]
        messages.append({"role": "user", "content": user_message})

        resp = self._make_api_call(messages)

        if not resp.text_parts:
            raise ValueError(f"No text content in API response for agent '{self.name}'")
        assistant_text = resp.text_parts[0]
        self._record_usage(resp)

        # Only commit to history after successful API call
        self.history.append(Message(role="user", content=user_message))
        self.history.append(Message(role="assistant", content=assistant_text))
        return assistant_text

    def send_with_tools(self, user_message: str) -> AgentResponse:
        """Send a message with tool use enabled. Returns structured response."""
        messages = [
            {"role": m.role, "content": m.content}
            for m in self.history
        ]
        messages.append({"role": "user", "content": user_message})

        resp = self._make_api_call(messages, use_tools=True)
        return self._process_response(resp, user_message)

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

        messages = [
            {"role": m.role, "content": m.content}
            for m in self.history
        ]
        messages.append({"role": "user", "content": result_content})

        resp = self._make_api_call(messages, use_tools=True)
        return self._process_response(resp, result_content)

    def _process_response(self, resp: LLMResponse, user_content) -> AgentResponse:
        """Convert LLMResponse into AgentResponse and commit to history."""
        self._record_usage(resp)
        self.last_thinking = resp.thinking

        raw_content = resp.raw_content
        if not raw_content:
            raw_content = [{"type": "text", "text": "(acknowledged)"}]

        # Convert provider ToolCalls to our ToolCall type
        tool_calls = [
            ToolCall(id=tc.id, name=tc.name, arguments=tc.arguments)
            for tc in resp.tool_calls
        ]

        # Commit to history
        self.history.append(Message(role="user", content=user_content))
        self.history.append(Message(role="assistant", content=raw_content))

        return AgentResponse(
            text="\n".join(resp.text_parts),
            tool_calls=tool_calls,
            raw_content=raw_content,
            stop_reason=resp.stop_reason,
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

    # -- Legacy static methods (kept for test compatibility) ---------------

    @staticmethod
    def _add_cache_control(system_prompt: str | list) -> list:
        """Wrap system prompt with cache_control for prompt caching."""
        from .provider import AnthropicProvider
        return AnthropicProvider._add_cache_control(system_prompt)

    @staticmethod
    def _mark_last_for_caching(messages: list) -> None:
        """Add cache_control to the last message's content block."""
        from .provider import AnthropicProvider
        AnthropicProvider._mark_last_for_caching(messages)
