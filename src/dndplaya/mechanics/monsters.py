"""DMG CR table and monster creation."""
from __future__ import annotations

from pydantic import BaseModel, Field

# Monster statistics by challenge rating, based on the DMG.
CR_TABLE: dict[float, dict[str, int | float]] = {
    0: {"hp": 6, "ac": 12, "attack_bonus": 2, "damage_per_round": 1, "save_dc": 10},
    0.125: {"hp": 15, "ac": 12, "attack_bonus": 3, "damage_per_round": 3, "save_dc": 11},
    0.25: {"hp": 25, "ac": 12, "attack_bonus": 3, "damage_per_round": 5, "save_dc": 11},
    0.5: {"hp": 35, "ac": 13, "attack_bonus": 3, "damage_per_round": 8, "save_dc": 12},
    1: {"hp": 50, "ac": 13, "attack_bonus": 4, "damage_per_round": 12, "save_dc": 12},
    2: {"hp": 65, "ac": 13, "attack_bonus": 5, "damage_per_round": 17, "save_dc": 13},
    3: {"hp": 80, "ac": 13, "attack_bonus": 5, "damage_per_round": 23, "save_dc": 13},
    4: {"hp": 95, "ac": 14, "attack_bonus": 5, "damage_per_round": 28, "save_dc": 14},
    5: {"hp": 110, "ac": 15, "attack_bonus": 6, "damage_per_round": 33, "save_dc": 15},
    6: {"hp": 125, "ac": 15, "attack_bonus": 6, "damage_per_round": 38, "save_dc": 15},
    7: {"hp": 140, "ac": 15, "attack_bonus": 6, "damage_per_round": 43, "save_dc": 15},
    8: {"hp": 155, "ac": 16, "attack_bonus": 7, "damage_per_round": 48, "save_dc": 16},
    9: {"hp": 170, "ac": 16, "attack_bonus": 7, "damage_per_round": 53, "save_dc": 16},
    10: {"hp": 185, "ac": 17, "attack_bonus": 7, "damage_per_round": 58, "save_dc": 17},
}


class Monster(BaseModel):
    """A monster with stats drawn from the CR table."""

    name: str
    cr: float
    max_hp: int
    current_hp: int
    ac: int
    attack_bonus: int
    damage_per_round: float
    save_dc: int
    abilities: list[str] = Field(default_factory=list)


def create_monster(
    name: str,
    cr: float,
    abilities: list[str] | None = None,
) -> Monster:
    """Create a monster from the CR table."""
    if cr not in CR_TABLE:
        raise ValueError(
            f"Unknown CR: {cr}. Available CRs: {sorted(CR_TABLE.keys())}"
        )

    stats = CR_TABLE[cr]
    return Monster(
        name=name,
        cr=cr,
        max_hp=int(stats["hp"]),
        current_hp=int(stats["hp"]),
        ac=int(stats["ac"]),
        attack_bonus=int(stats["attack_bonus"]),
        damage_per_round=stats["damage_per_round"],
        save_dc=int(stats["save_dc"]),
        abilities=abilities or [],
    )
