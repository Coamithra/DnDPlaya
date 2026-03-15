"""Character sheets with pre-computed D&D Lite stats."""
from __future__ import annotations

import math

from pydantic import BaseModel, Field

# Average damage per round by level (levels 1-11)
CLASS_STATS: dict[str, dict] = {
    "Fighter": {
        "hp_per_level": 10,
        "base_hp": 12,
        "avg_damage": {
            1: 9, 2: 11, 3: 13, 4: 15, 5: 23,
            6: 25, 7: 27, 8: 29, 9: 31, 10: 35, 11: 42,
        },
        "ac": 18,
        "attack_bonus_per_level": 1,
        "base_attack": 5,
    },
    "Rogue": {
        "hp_per_level": 7,
        "base_hp": 10,
        "avg_damage": {
            1: 8, 2: 8, 3: 12, 4: 12, 5: 16,
            6: 16, 7: 20, 8: 20, 9: 24, 10: 24, 11: 28,
        },
        "ac": 15,
        "attack_bonus_per_level": 1,
        "base_attack": 5,
    },
    "Wizard": {
        "hp_per_level": 5,
        "base_hp": 8,
        "avg_damage": {
            1: 7, 2: 7, 3: 11, 4: 11, 5: 17,
            6: 17, 7: 22, 8: 22, 9: 27, 10: 27, 11: 33,
        },
        "ac": 12,
        "attack_bonus_per_level": 0.5,
        "base_attack": 5,
    },
    "Cleric": {
        "hp_per_level": 7,
        "base_hp": 10,
        "avg_damage": {
            1: 6, 2: 6, 3: 8, 4: 8, 5: 14,
            6: 14, 7: 16, 8: 16, 9: 20, 10: 20, 11: 24,
        },
        "ac": 16,
        "attack_bonus_per_level": 1,
        "base_attack": 4,
    },
}


class Character(BaseModel):
    """A player character with pre-computed D&D Lite stats."""

    name: str
    char_class: str
    level: int
    max_hp: int
    current_hp: int
    ac: int
    avg_damage: float
    attack_bonus: int
    spell_slots: dict[int, int] = Field(default_factory=dict)
    abilities: list[str] = Field(default_factory=list)


def compute_spell_slots(char_class: str, level: int) -> dict[int, int]:
    """Compute spell slot allocation for a character.

    Wizard gets level+1 total slots spread across spell levels.
    Cleric gets level total slots spread across spell levels.
    Fighter/Rogue get 0.
    """
    if char_class not in ("Wizard", "Cleric"):
        return {}

    total = (level + 1) if char_class == "Wizard" else level
    if total <= 0:
        return {}

    # Determine the highest spell level available (max 9th, unlocked every 2 levels)
    max_spell_level = min(math.ceil(level / 2), 9)

    # Spread slots across available spell levels, distributing evenly from lowest up
    slots: dict[int, int] = {}
    base_per_level = total // max_spell_level
    remainder = total % max_spell_level

    for spell_level in range(1, max_spell_level + 1):
        count = base_per_level + (1 if spell_level <= remainder else 0)
        if count > 0:
            slots[spell_level] = count

    return slots


def create_character(name: str, char_class: str, level: int) -> Character:
    """Create a character from the CLASS_STATS table."""
    if char_class not in CLASS_STATS:
        raise ValueError(f"Unknown class: {char_class}. Choose from: {list(CLASS_STATS.keys())}")
    if not 1 <= level <= 11:
        raise ValueError(f"Level must be between 1 and 11, got {level}")

    stats = CLASS_STATS[char_class]
    max_hp = stats["base_hp"] + stats["hp_per_level"] * (level - 1)
    avg_damage = stats["avg_damage"][level]
    attack_bonus = stats["base_attack"] + int(stats["attack_bonus_per_level"] * (level - 1))
    spell_slots = compute_spell_slots(char_class, level)

    return Character(
        name=name,
        char_class=char_class,
        level=level,
        max_hp=max_hp,
        current_hp=max_hp,
        ac=stats["ac"],
        avg_damage=avg_damage,
        attack_bonus=attack_bonus,
        spell_slots=spell_slots,
        abilities=[],
    )


def create_default_party(level: int) -> list[Character]:
    """Create the standard four-member party at the given level."""
    return [
        create_character("Thorin", "Fighter", level),
        create_character("Shadow", "Rogue", level),
        create_character("Elara", "Wizard", level),
        create_character("Brother Marcus", "Cleric", level),
    ]
