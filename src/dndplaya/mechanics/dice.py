"""Seeded RNG and variance rolls for D&D combat simulation."""
from __future__ import annotations

import random
import re


class DiceRoller:
    """Wraps random.Random with a seed for reproducible dice rolls."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self.seed = seed

    def roll(self, sides: int) -> int:
        """Roll a single die with the given number of sides."""
        return self._rng.randint(1, sides)

    def d20(self) -> int:
        """Roll a d20."""
        return self.roll(20)

    def variance_roll(self, avg: float) -> float:
        """Return avg scaled by a uniform random factor in [0.6, 1.4]."""
        factor = self._rng.uniform(0.6, 1.4)
        return avg * factor

    def check(self, modifier: int, dc: int) -> tuple[bool, int]:
        """Roll d20 + modifier vs DC. Returns (success, total)."""
        roll = self.d20()
        total = roll + modifier
        return (total >= dc, total)

    def parse_and_roll(self, expression: str) -> int:
        """Parse and roll a dice expression like '2d6+3' or '1d8+1d4+2'.

        Supports multiple dice terms and +/- modifiers.
        Returns 0 minimum (negative results clamped to 0).
        """
        expression = expression.strip().lower().replace(" ", "")
        if not expression:
            return 0

        total = 0
        # Split into terms, preserving +/- signs
        terms = re.findall(r'[+-]?[^+-]+', expression)

        for term in terms:
            term = term.strip()
            if not term:
                continue

            dice_match = re.match(r'^([+-]?)(\d*)d(\d+)$', term)
            if dice_match:
                sign = -1 if dice_match.group(1) == '-' else 1
                count = int(dice_match.group(2)) if dice_match.group(2) else 1
                sides = int(dice_match.group(3))
                for _ in range(count):
                    total += sign * self.roll(sides)
            else:
                try:
                    total += int(term)
                except ValueError:
                    pass

        return max(0, total)
