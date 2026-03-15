"""Seeded RNG and variance rolls for D&D combat simulation."""
from __future__ import annotations

import random


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
