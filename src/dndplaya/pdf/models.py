from __future__ import annotations

from pydantic import BaseModel, Field


class MonsterRef(BaseModel):
    """Reference to a monster in an encounter."""
    name: str
    cr: float
    count: int = 1
    notes: str = ""


class Encounter(BaseModel):
    """A combat or social encounter within a room."""
    description: str
    monsters: list[MonsterRef] = Field(default_factory=list)
    trigger: str = ""  # what triggers this encounter
    tactics: str = ""  # how monsters fight
    difficulty: str = ""  # easy/medium/hard/deadly if specified


class Trap(BaseModel):
    """A trap or hazard."""
    description: str
    dc: int = 15
    damage: str = ""
    effect: str = ""


class Treasure(BaseModel):
    """Loot or reward."""
    description: str
    value: str = ""


class Room(BaseModel):
    """A room/area in the dungeon."""
    id: str  # e.g., "room_1", "area_a"
    name: str
    description: str
    read_aloud: str = ""  # boxed text for players
    encounters: list[Encounter] = Field(default_factory=list)
    traps: list[Trap] = Field(default_factory=list)
    treasure: list[Treasure] = Field(default_factory=list)
    connections: list[str] = Field(default_factory=list)  # IDs of connected rooms
    notes: str = ""  # DM notes
    raw_text: str = ""  # original text from PDF


class DungeonModule(BaseModel):
    """A complete parsed dungeon module."""
    title: str
    summary: str = ""
    intended_level: str = ""  # e.g., "3-5"
    rooms: list[Room] = Field(default_factory=list)
    introduction: str = ""
    background: str = ""
    appendices: str = ""
    raw_markdown: str = ""  # full PDF text

    def get_room(self, room_id: str) -> Room | None:
        for room in self.rooms:
            if room.id == room_id:
                return room
        return None

    def get_adjacent_rooms(self, room_id: str) -> list[Room]:
        room = self.get_room(room_id)
        if not room:
            return []
        return [r for r in self.rooms if r.id in room.connections]

    def get_entry_room(self) -> Room | None:
        return self.rooms[0] if self.rooms else None
