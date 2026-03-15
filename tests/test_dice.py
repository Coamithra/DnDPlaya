"""Tests for the dice roller."""
from dndplaya.mechanics.dice import DiceRoller


def test_seeded_determinism():
    """Same seed produces same sequence."""
    a = DiceRoller(seed=42)
    b = DiceRoller(seed=42)
    assert [a.d20() for _ in range(10)] == [b.d20() for _ in range(10)]


def test_d20_range():
    dice = DiceRoller(seed=1)
    rolls = [dice.d20() for _ in range(200)]
    assert min(rolls) >= 1
    assert max(rolls) <= 20


def test_roll_range():
    dice = DiceRoller(seed=1)
    rolls = [dice.roll(6) for _ in range(200)]
    assert min(rolls) >= 1
    assert max(rolls) <= 6


def test_variance_roll_range():
    dice = DiceRoller(seed=1)
    results = [dice.variance_roll(10.0) for _ in range(200)]
    assert all(5.9 <= r <= 14.1 for r in results)


def test_check_pass():
    dice = DiceRoller(seed=42)
    # With high modifier, should pass low DC
    passed, total = dice.check(modifier=10, dc=5)
    assert total >= 11  # d20(>=1) + 10
    assert passed


def test_check_returns_total():
    dice = DiceRoller(seed=42)
    _, total = dice.check(modifier=3, dc=10)
    assert isinstance(total, int)
