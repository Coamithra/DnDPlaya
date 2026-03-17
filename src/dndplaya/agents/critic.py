from __future__ import annotations

from ..config import Settings
from ..prompts import load_prompt
from .base import BaseAgent


class CriticAgent:
    """Generates post-session reviews from each agent's perspective."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def generate_review(
        self,
        agent_name: str,
        role: str,
        perspective: str,
        transcript: str,
        notes: list[str],
    ) -> str:
        """Generate a review from a specific agent's perspective."""
        notes_text = "\n".join(f"- {note}" for note in notes) if notes else "No specific notes recorded."

        full_perspective = f"{perspective}\n\nNotes from during play:\n{notes_text}"

        # Build system prompt without .format() on transcript to avoid crashes
        # on curly braces in LLM-generated transcript text.
        prefix = load_prompt(
            "critic_system_prefix",
            agent_name=agent_name,
            role=role,
            perspective=full_perspective,
        )
        # Use last ~5000 tokens of transcript for review context
        system = prefix + transcript[-20000:] + load_prompt("critic_system_suffix")

        agent = BaseAgent(
            name=f"Critic_{agent_name}",
            system_prompt=system,
            settings=self.settings,
        )

        return agent.send(load_prompt("critic_user"))

    def generate_dm_review(self, transcript: str, runnability_notes: list[str]) -> str:
        """Generate the DM's runnability review."""
        return self.generate_review(
            agent_name="DM",
            role="Dungeon Master",
            perspective=load_prompt("dm_runnability_perspective"),
            transcript=transcript,
            notes=runnability_notes,
        )

    def generate_player_review(
        self,
        player_name: str,
        archetype: str,
        transcript: str,
        engagement_notes: list[str],
    ) -> str:
        """Generate a player's MDA-focused review."""
        from .player import ARCHETYPES

        arch_info = ARCHETYPES.get(archetype, {})
        if not arch_info:
            raise ValueError(f"Unknown archetype: {archetype}. Must be one of {list(ARCHETYPES)}")
        perspective = (
            f"Archetype: {archetype.replace('_', ' ').title()}\n"
            f"MDA Aesthetics: {arch_info.get('aesthetics', 'N/A')}\n"
            f"Focus: {arch_info.get('focus', 'N/A')}\n"
            f"Review through the lens of: {arch_info.get('focus', 'general gameplay')}"
        )

        return self.generate_review(
            agent_name=player_name,
            role=f"Player ({archetype.replace('_', ' ').title()})",
            perspective=perspective,
            transcript=transcript,
            notes=engagement_notes,
        )
