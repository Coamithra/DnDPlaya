from __future__ import annotations

from ..config import Settings
from ..agents.base import BaseAgent
from ..prompts import load_prompt


def generate_narrative(transcript_text: str, settings: Settings) -> str:
    """Generate a readable narrative from a session transcript."""
    # Build system prompt without .format() on transcript to avoid crashes
    # on curly braces in LLM-generated transcript text.
    system = (
        load_prompt("narrative_system_prefix")
        + transcript_text[-30000:]
        + load_prompt("narrative_system_suffix")
    )

    agent = BaseAgent(
        name="Narrator",
        system_prompt=system,
        settings=settings,
    )

    return agent.send(load_prompt("narrative_user"))
