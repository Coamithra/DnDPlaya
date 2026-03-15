from __future__ import annotations

import anthropic
from anthropic.types import MessageParam, TextBlock
from dataclasses import dataclass

from ..config import Settings


@dataclass
class Message:
    role: str  # "user" or "assistant"
    content: str


class BaseAgent:
    """Base class for all LLM agents (DM, players, critic)."""

    def __init__(self, name: str, system_prompt: str, settings: Settings):
        self.name = name
        self.system_prompt = system_prompt
        self.settings = settings
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.history: list[Message] = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def send(self, user_message: str) -> str:
        """Send a message and get a response."""
        self.history.append(Message(role="user", content=user_message))

        messages: list[MessageParam] = [
            {"role": m.role, "content": m.content}  # type: ignore[typeddict-item]
            for m in self.history
        ]

        response = self.client.messages.create(
            model=self.settings.model,
            max_tokens=self.settings.max_tokens,
            system=self.system_prompt,
            messages=messages,
        )

        if not response.content:
            raise ValueError(f"Empty response from API for agent '{self.name}'")
        first_block = response.content[0]
        if not isinstance(first_block, TextBlock):
            raise TypeError(
                f"Expected TextBlock from API, got {type(first_block).__name__} "
                f"for agent '{self.name}'"
            )
        assistant_text = first_block.text
        self.total_input_tokens += response.usage.input_tokens
        self.total_output_tokens += response.usage.output_tokens

        self.history.append(Message(role="assistant", content=assistant_text))
        return assistant_text

    def get_token_usage(self) -> dict[str, int]:
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
        }

    def reset(self) -> None:
        self.history.clear()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
