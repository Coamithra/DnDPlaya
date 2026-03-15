"""GameState and GameEvent types for tracking dungeon simulation state."""
from __future__ import annotations

import time
from enum import Enum

from pydantic import BaseModel, Field

from dndplaya.mechanics.characters import Character, compute_spell_slots
from dndplaya.mechanics.monsters import Monster


class EventType(Enum):
    """Types of events that can occur during a dungeon run."""

    ROOM_ENTERED = "room_entered"
    COMBAT_STARTED = "combat_started"
    COMBAT_ENDED = "combat_ended"
    ATTACK = "attack"
    HEAL = "heal"
    SPELL_CAST = "spell_cast"
    CHECK_MADE = "check_made"
    REST_TAKEN = "rest_taken"
    ITEM_FOUND = "item_found"
    NPC_INTERACTION = "npc_interaction"
    PLAYER_ACTION = "player_action"
    DM_NARRATION = "dm_narration"
    PRESSURE_SIGNAL = "pressure_signal"


class GameEvent(BaseModel):
    """A single event in the game log."""

    event_type: EventType
    description: str
    actor: str | None = None
    target: str | None = None
    details: dict = Field(default_factory=dict)
    round_number: int | None = None
    timestamp: float = Field(default_factory=time.time)


class GameState:
    """Tracks the full mutable state of a dungeon simulation."""

    def __init__(self, characters: list[Character]) -> None:
        self.characters: list[Character] = characters
        self.monsters: list[Monster] = []
        self.current_room: str | None = None
        self.events: list[GameEvent] = []
        self.in_combat: bool = False
        self.round_number: int = 0
        self.rooms_visited: list[str] = []

    def add_event(
        self,
        event_or_type: GameEvent | EventType,
        description: str = "",
        *,
        actor: str | None = None,
        target: str | None = None,
        details: dict | None = None,
        round_number: int | None = None,
    ) -> None:
        """Append an event to the log.

        Can be called with a pre-built GameEvent or with individual fields:
            state.add_event(game_event)
            state.add_event(EventType.ATTACK, "hit for 5", actor="Thorin")
        """
        if isinstance(event_or_type, GameEvent):
            self.events.append(event_or_type)
        else:
            self.events.append(GameEvent(
                event_type=event_or_type,
                description=description,
                actor=actor,
                target=target,
                details=details or {},
                round_number=round_number,
            ))

    def get_recent_events(self, n: int) -> list[GameEvent]:
        """Return the last n events."""
        return self.events[-n:]

    def get_character(self, name: str) -> Character | None:
        """Find a character by name (case-insensitive)."""
        for char in self.characters:
            if char.name.lower() == name.lower():
                return char
        return None

    def get_alive_characters(self) -> list[Character]:
        """Return characters with current_hp > 0."""
        return [c for c in self.characters if c.current_hp > 0]

    def get_alive_monsters(self) -> list[Monster]:
        """Return monsters with current_hp > 0."""
        return [m for m in self.monsters if m.current_hp > 0]

    def start_combat(self, monsters: list[Monster]) -> None:
        """Begin a combat encounter with the given monsters."""
        self.monsters = monsters
        self.in_combat = True
        self.round_number = 1
        self.add_event(
            GameEvent(
                event_type=EventType.COMBAT_STARTED,
                description=f"Combat begins with {len(monsters)} monster(s): "
                + ", ".join(m.name for m in monsters),
                details={"monster_names": [m.name for m in monsters]},
                round_number=self.round_number,
            )
        )

    def end_combat(self) -> None:
        """End the current combat encounter."""
        self.in_combat = False
        self.add_event(
            GameEvent(
                event_type=EventType.COMBAT_ENDED,
                description="Combat has ended.",
                round_number=self.round_number,
            )
        )
        self.monsters = []
        self.round_number = 0

    def enter_room(self, room_id: str) -> None:
        """Move the party into a new room."""
        self.current_room = room_id
        if room_id not in self.rooms_visited:
            self.rooms_visited.append(room_id)
        self.add_event(
            GameEvent(
                event_type=EventType.ROOM_ENTERED,
                description=f"Party enters room: {room_id}",
                details={"room_id": room_id},
            )
        )

    def take_short_rest(self) -> None:
        """Short rest: heal 25% max HP and restore 1 spell slot per spell level."""
        for char in self.get_alive_characters():
            heal = int(char.max_hp * 0.25)
            char.current_hp = min(char.max_hp, char.current_hp + heal)

            # Restore 1 slot per spell level (up to the original max)
            original_slots = compute_spell_slots(char.char_class, char.level)
            for spell_level, max_count in original_slots.items():
                current = char.spell_slots.get(spell_level, 0)
                char.spell_slots[spell_level] = min(max_count, current + 1)

        self.add_event(
            GameEvent(
                event_type=EventType.REST_TAKEN,
                description="Party takes a short rest — healed 25% HP, restored some spell slots.",
                details={"rest_type": "short"},
            )
        )

    def take_long_rest(self) -> None:
        """Long rest: full HP and all spell slots restored."""
        for char in self.get_alive_characters():
            char.current_hp = char.max_hp
            original_slots = compute_spell_slots(char.char_class, char.level)
            char.spell_slots = dict(original_slots)

        self.add_event(
            GameEvent(
                event_type=EventType.REST_TAKEN,
                description="Party takes a long rest — fully healed, all spell slots restored.",
                details={"rest_type": "long"},
            )
        )
