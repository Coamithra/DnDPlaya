from __future__ import annotations

from ..config import Settings
from ..mechanics.characters import Character
from .base import BaseAgent

# MDA archetype definitions
ARCHETYPES = {
    "roleplayer": {
        "aesthetics": "Fantasy + Narrative",
        "focus": "Immersion, story hooks, NPC interaction, lore, atmosphere",
        "prompt_addition": """You are deeply invested in the story and world. You:
- Speak in character with a distinct voice
- Seek out NPCs to talk to and lore to uncover
- React emotionally to dramatic moments
- Value atmosphere and immersion above tactical advantage
- Try to understand motivations of enemies before fighting
- Note when the world feels alive vs when it feels like a game

When something breaks your immersion or when the story feels forced, make a mental note.
When an NPC is memorable or the atmosphere is compelling, note that too.""",
    },
    "tactician": {
        "aesthetics": "Challenge",
        "focus": "Meaningful tactical choices, puzzles, resource decisions",
        "prompt_addition": """You approach the dungeon as a puzzle to be solved optimally. You:
- Analyze encounters for tactical opportunities
- Coordinate with party members on strategy
- Think about resource management (HP, spell slots, abilities)
- Look for environmental advantages in combat
- Appreciate well-designed encounters with multiple valid approaches
- Get frustrated by encounters that feel like simple slugfests with no real choices

Note when encounters offer meaningful tactical decisions vs when they're just damage races.
Note when puzzles are satisfying vs arbitrary.""",
    },
    "explorer": {
        "aesthetics": "Discovery",
        "focus": "Secrets, hidden content, investigation rewards, curiosity payoff",
        "prompt_addition": """You are driven by curiosity and the thrill of discovery. You:
- Search everything - walls, floors, furniture, bodies
- Take the less obvious path when given a choice
- Investigate anything unusual or out of place
- Ask probing questions about the environment
- Value finding secrets, hidden rooms, and lore fragments
- Get excited by rewards for thorough exploration

Note when exploration is rewarded vs when there's nothing to find.
Note when the dungeon has satisfying secrets vs when it feels linear.""",
    },
    "free_spirit": {
        "aesthetics": "Expression + Fellowship",
        "focus": "Creative solutions, teamwork, player agency",
        "prompt_addition": """You value creative freedom and working with your party. You:
- Try unconventional solutions to problems
- Support other party members' ideas and build on them
- Look for ways to use the environment creatively
- Prefer negotiation or trickery over brute force when possible
- Enjoy moments of party banter and camaraderie
- Test the boundaries of what the dungeon allows

Note when the dungeon supports creative solutions vs when it railroads.
Note when teamwork feels organic vs forced.""",
    },
}


PLAYER_SYSTEM_PROMPT = '''You are a D&D player with the following character and play style.

## Your Character
Name: {char_name}
Class: {char_class}
Level: {char_level}
HP: {char_hp}
AC: {char_ac}

## Your Play Style ({archetype_name})
MDA Aesthetics: {aesthetics}
Focus: {focus}

{archetype_prompt}

## How to Play
- Respond to the DM's descriptions with what your character does
- Declare actions clearly: "I attack the goblin", "I search the room", "I try to persuade the guard"
- In combat, state your intended action and target
- Stay in character but be concise - this is a game, keep it moving
- Coordinate with your party when it makes sense for your character
- You can ONLY act on information your character has - no metagaming

## Internal Notes
While playing, silently note moments that engage or frustrate you through your archetype's lens.
These will be used for your review later. Don't mention them in-character.

Respond as your character would. Keep responses to 2-4 sentences in exploration, 1-2 in combat.'''


class PlayerAgent(BaseAgent):
    """Player agent with a character and MDA archetype personality."""

    def __init__(
        self,
        settings: Settings,
        character: Character,
        archetype: str,
    ):
        if archetype not in ARCHETYPES:
            raise ValueError(f"Unknown archetype: {archetype}. Must be one of {list(ARCHETYPES)}")

        self.character = character
        self.archetype = archetype
        self.archetype_info = ARCHETYPES[archetype]
        self.engagement_notes: list[str] = []

        system = PLAYER_SYSTEM_PROMPT.format(
            char_name=character.name,
            char_class=character.char_class,
            char_level=character.level,
            char_hp=character.max_hp,
            char_ac=character.ac,
            archetype_name=archetype.replace("_", " ").title(),
            aesthetics=self.archetype_info["aesthetics"],
            focus=self.archetype_info["focus"],
            archetype_prompt=self.archetype_info["prompt_addition"],
        )

        super().__init__(
            name=character.name,
            system_prompt=system,
            settings=settings,
        )

    def add_engagement_note(self, note: str) -> None:
        self.engagement_notes.append(note)
