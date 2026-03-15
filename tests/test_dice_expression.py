"""Tests for DiceRoller.parse_and_roll — dice expression parsing."""
from __future__ import annotations

from dndplaya.mechanics.dice import DiceRoller


class TestParseAndRoll:
    def test_simple_d6(self):
        roller = DiceRoller(seed=42)
        result = roller.parse_and_roll("1d6")
        assert 1 <= result <= 6

    def test_multiple_dice(self):
        roller = DiceRoller(seed=42)
        result = roller.parse_and_roll("2d6")
        assert 2 <= result <= 12

    def test_dice_with_positive_modifier(self):
        roller = DiceRoller(seed=42)
        result = roller.parse_and_roll("1d6+3")
        assert 4 <= result <= 9

    def test_dice_with_negative_modifier(self):
        roller = DiceRoller(seed=42)
        result = roller.parse_and_roll("1d6-1")
        assert 0 <= result <= 5

    def test_multiple_dice_terms(self):
        roller = DiceRoller(seed=42)
        result = roller.parse_and_roll("1d6+1d4")
        assert 2 <= result <= 10

    def test_complex_expression(self):
        roller = DiceRoller(seed=42)
        result = roller.parse_and_roll("2d6+1d4+3")
        assert 6 <= result <= 19

    def test_d20(self):
        roller = DiceRoller(seed=42)
        result = roller.parse_and_roll("1d20")
        assert 1 <= result <= 20

    def test_bare_d_notation(self):
        """'d6' without leading count should be treated as '1d6'."""
        roller = DiceRoller(seed=42)
        result = roller.parse_and_roll("d6")
        assert 1 <= result <= 6

    def test_plain_number(self):
        roller = DiceRoller(seed=42)
        result = roller.parse_and_roll("5")
        assert result == 5

    def test_negative_result_clamped_to_zero(self):
        """Negative results should be clamped to 0."""
        roller = DiceRoller(seed=42)
        result = roller.parse_and_roll("1d4-100")
        assert result == 0

    def test_empty_expression(self):
        roller = DiceRoller(seed=42)
        result = roller.parse_and_roll("")
        assert result == 0

    def test_spaces_in_expression(self):
        roller = DiceRoller(seed=42)
        result = roller.parse_and_roll("2d6 + 3")
        assert 5 <= result <= 15

    def test_uppercase_ignored(self):
        roller = DiceRoller(seed=42)
        result = roller.parse_and_roll("2D6+3")
        assert 5 <= result <= 15

    def test_deterministic_with_seed(self):
        """Same seed should produce same results."""
        r1 = DiceRoller(seed=99)
        r2 = DiceRoller(seed=99)
        assert r1.parse_and_roll("3d6+2") == r2.parse_and_roll("3d6+2")

    def test_d8(self):
        roller = DiceRoller(seed=42)
        result = roller.parse_and_roll("1d8")
        assert 1 <= result <= 8

    def test_d10(self):
        roller = DiceRoller(seed=42)
        result = roller.parse_and_roll("1d10")
        assert 1 <= result <= 10

    def test_d12(self):
        roller = DiceRoller(seed=42)
        result = roller.parse_and_roll("1d12")
        assert 1 <= result <= 12
