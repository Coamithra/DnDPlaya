from __future__ import annotations
from enum import Enum, auto


class Phase(Enum):
    """Game phase state machine."""
    SETUP = auto()
    EXPLORATION = auto()
    COMBAT = auto()
    REST = auto()
    TRANSITION = auto()  # Moving between rooms
    COMPLETED = auto()

    def can_transition_to(self, target: Phase) -> bool:
        """Check if a phase transition is valid."""
        valid = {
            Phase.SETUP: {Phase.EXPLORATION},
            Phase.EXPLORATION: {Phase.COMBAT, Phase.TRANSITION, Phase.REST, Phase.COMPLETED},
            Phase.COMBAT: {Phase.EXPLORATION, Phase.REST, Phase.COMPLETED},
            Phase.REST: {Phase.EXPLORATION, Phase.TRANSITION},
            Phase.TRANSITION: {Phase.EXPLORATION},
            Phase.COMPLETED: set(),
        }
        return target in valid.get(self, set())
