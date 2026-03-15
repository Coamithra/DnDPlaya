"""Skill checks and saving throws."""
from __future__ import annotations

from dndplaya.mechanics.dice import DiceRoller


def resolve_skill_check(
    dice: DiceRoller,
    modifier: int,
    dc: int,
) -> tuple[bool, int]:
    """Resolve a skill check (d20 + modifier vs DC)."""
    return dice.check(modifier, dc)


def resolve_saving_throw(
    dice: DiceRoller,
    modifier: int,
    dc: int,
) -> tuple[bool, int]:
    """Resolve a saving throw (d20 + modifier vs DC)."""
    return dice.check(modifier, dc)


def resolve_group_check(
    dice: DiceRoller,
    modifiers: list[int],
    dc: int,
) -> tuple[bool, list[tuple[bool, int]]]:
    """Resolve a group check where majority pass means group pass.

    Each member rolls d20 + their modifier vs DC. If at least half succeed,
    the group check passes.
    """
    if not modifiers:
        return (False, [])

    individual_results: list[tuple[bool, int]] = []
    for modifier in modifiers:
        result = dice.check(modifier, dc)
        individual_results.append(result)

    passes = sum(1 for success, _ in individual_results if success)
    group_pass = passes >= len(modifiers) / 2

    return (group_pass, individual_results)
