from __future__ import annotations

from ..config import Settings
from .base import BaseAgent

CRITIC_SYSTEM_PROMPT = '''You are a D&D session reviewer. You've just finished playing through a dungeon module and need to write a review.

## Your Perspective
Name: {agent_name}
Role: {role}
{perspective}

## Session Transcript
{transcript}

## Your Task
Write a review of this dungeon module with two sections:

### What I Liked
Specific things that worked well from your perspective. Reference specific rooms, encounters, or moments. Be concrete.

### Take a Look At
Things that could be improved. Be constructive and specific - reference exact moments or design choices. Suggest improvements where possible.

Keep the review focused and actionable. 3-5 bullet points per section.'''


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

        system = CRITIC_SYSTEM_PROMPT.format(
            agent_name=agent_name,
            role=role,
            perspective=full_perspective,
            transcript=transcript[-8000:],  # Last ~2000 tokens of transcript
        )

        agent = BaseAgent(
            name=f"Critic_{agent_name}",
            system_prompt=system,
            settings=self.settings,
        )

        review = agent.send(
            "Please write your review of this dungeon module based on your experience playing through it."
        )
        return review

    def generate_dm_review(self, transcript: str, runnability_notes: list[str]) -> str:
        """Generate the DM's runnability review."""
        return self.generate_review(
            agent_name="DM",
            role="Dungeon Master",
            perspective=(
                "You ran this module as DM. Focus your review on RUNNABILITY:\n"
                "- How clear were the room descriptions?\n"
                "- Was information organized well?\n"
                "- Were transitions between areas smooth?\n"
                "- Did the module support improvisation?\n"
                "- Were encounter tactics clear?\n"
                "- Was the pacing good?"
            ),
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
