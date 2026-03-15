from __future__ import annotations

from ..config import Settings
from ..agents.base import BaseAgent


NARRATIVE_SYSTEM_PROMPT = '''You are a skilled writer converting a D&D session transcript into a readable narrative.

## Your Task
Transform the raw session transcript below into an engaging, readable account of the adventure.
This is NOT a raw transcript - it should read like a story that someone can enjoy and use to understand how the dungeon plays.

## Guidelines
- Write in past tense, third person
- Focus on key moments: dramatic combat, clever solutions, important discoveries, tense decisions
- Skip routine actions and mechanical details
- Capture the personality of each character based on their actions
- Include dialogue highlights but don't transcribe everything
- Note when the dungeon design created interesting or frustrating moments
- Keep it concise: aim for ~500-1000 words for a typical session
- Structure with clear scene breaks when moving between rooms

## Session Transcript
{transcript}'''


def generate_narrative(transcript_text: str, settings: Settings) -> str:
    """Generate a readable narrative from a session transcript."""
    agent = BaseAgent(
        name="Narrator",
        system_prompt=NARRATIVE_SYSTEM_PROMPT.format(transcript=transcript_text[-12000:]),
        settings=settings,
    )

    narrative = agent.send(
        "Please write the session narrative based on the transcript above. "
        "Make it engaging and readable."
    )
    return narrative
