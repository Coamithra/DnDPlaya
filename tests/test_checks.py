"""Tests for skill checks and saving throws."""
from dndplaya.mechanics.dice import DiceRoller
from dndplaya.mechanics.checks import (
    resolve_skill_check,
    resolve_saving_throw,
    resolve_group_check,
)


def test_skill_check():
    dice = DiceRoller(seed=42)
    success, total = resolve_skill_check(dice, modifier=5, dc=10)
    assert isinstance(success, bool)
    assert isinstance(total, int)
    assert total >= 6  # d20(>=1) + 5


def test_saving_throw():
    dice = DiceRoller(seed=42)
    success, total = resolve_saving_throw(dice, modifier=3, dc=15)
    assert isinstance(success, bool)
    assert isinstance(total, int)


def test_group_check():
    dice = DiceRoller(seed=42)
    modifiers = [5, 3, -1, 2]
    group_pass, individual = resolve_group_check(dice, modifiers, dc=10)
    assert isinstance(group_pass, bool)
    assert len(individual) == 4
    # Count individual passes
    passes = sum(1 for s, _ in individual if s)
    assert group_pass == (passes >= 2)
