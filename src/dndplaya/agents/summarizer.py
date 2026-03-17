"""Pre-game module summary generator."""
from __future__ import annotations

from ..config import Settings
from ..prompts import load_prompt
from .base import BaseAgent


def generate_module_summary(module_text: str, settings: Settings) -> str:
    """Generate a pre-game summary of the module using an LLM call.

    Creates a temporary BaseAgent, sends the full module text, and returns
    the summary string. Cost: one Haiku call reading the full module.
    """
    agent = BaseAgent(
        name="Summarizer",
        system_prompt=load_prompt("summarizer_system"),
        settings=settings,
    )
    return agent.send(module_text)
