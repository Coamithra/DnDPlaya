from __future__ import annotations

from ..config import Settings
from ..mechanics.characters import Character
from ..prompts import load_prompt
from .base import BaseAgent
from .player_tools import PLAYER_TOOLS

# MDA archetype definitions — prompt text loaded from prompts/archetype_*.txt
ARCHETYPES = {
    "roleplayer": {
        "aesthetics": "Fantasy + Narrative",
        "focus": "Immersion, story hooks, NPC interaction, lore, atmosphere",
    },
    "tactician": {
        "aesthetics": "Challenge",
        "focus": "Meaningful tactical choices, puzzles, resource decisions",
    },
    "explorer": {
        "aesthetics": "Discovery",
        "focus": "Secrets, hidden content, investigation rewards, curiosity payoff",
    },
    "free_spirit": {
        "aesthetics": "Expression + Fellowship",
        "focus": "Creative solutions, teamwork, player agency",
    },
}


class PlayerAgent(BaseAgent):
    """Player agent with a character and MDA archetype personality."""

    def __init__(
        self,
        settings: Settings,
        character: Character,
        archetype: str,
        enable_thinking: bool = False,
    ):
        if archetype not in ARCHETYPES:
            raise ValueError(f"Unknown archetype: {archetype}. Must be one of {list(ARCHETYPES)}")

        self.character = character
        self.archetype = archetype
        self.archetype_info = ARCHETYPES[archetype]
        self.engagement_notes: list[str] = []

        # Format skills for the prompt
        if character.skills:
            skill_lines = [
                f"- {skill.replace('_', ' ').title()}: +{bonus}"
                for skill, bonus in sorted(character.skills.items())
                if bonus > 0
            ]
            char_skills = "\n".join(skill_lines) if skill_lines else "No special skill bonuses"
        else:
            char_skills = "No special skill bonuses"

        archetype_prompt = load_prompt(f"archetype_{archetype}")

        system = load_prompt(
            "player_system",
            char_name=character.name,
            char_pronouns=character.pronouns,
            char_class=character.char_class,
            char_level=character.level,
            char_hp=character.max_hp,
            char_ac=character.ac,
            char_skills=char_skills,
            archetype_name=archetype.replace("_", " ").title(),
            aesthetics=self.archetype_info["aesthetics"],
            focus=self.archetype_info["focus"],
            archetype_prompt=archetype_prompt,
        )

        # Players use lower max_tokens — they should be 1-3 sentences
        player_settings = settings.model_copy(update={"max_tokens": min(512, settings.max_tokens)})

        super().__init__(
            name=character.name,
            system_prompt=system,
            settings=player_settings,
            tools=PLAYER_TOOLS,
            enable_thinking=enable_thinking,
            thinking_budget=1024,
        )

    def add_engagement_note(self, note: str) -> None:
        self.engagement_notes.append(note)
