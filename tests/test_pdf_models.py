"""Tests for PDF data models."""
from dndplaya.pdf.models import DungeonModule, Room, Encounter, MonsterRef


def test_dungeon_module_get_room():
    rooms = [
        Room(id="room_1", name="Entry", description="A door"),
        Room(id="room_2", name="Hall", description="A hall"),
    ]
    module = DungeonModule(title="Test", rooms=rooms)
    assert module.get_room("room_1") is not None
    assert module.get_room("room_1").name == "Entry"
    assert module.get_room("nonexistent") is None


def test_dungeon_module_adjacent_rooms():
    rooms = [
        Room(id="room_1", name="Entry", description="A door", connections=["room_2"]),
        Room(id="room_2", name="Hall", description="A hall", connections=["room_1", "room_3"]),
        Room(id="room_3", name="Boss", description="A boss", connections=["room_2"]),
    ]
    module = DungeonModule(title="Test", rooms=rooms)
    adjacent = module.get_adjacent_rooms("room_2")
    assert len(adjacent) == 2
    adj_ids = {r.id for r in adjacent}
    assert adj_ids == {"room_1", "room_3"}


def test_dungeon_module_entry_room():
    rooms = [Room(id="room_1", name="Entry", description="Start")]
    module = DungeonModule(title="Test", rooms=rooms)
    assert module.get_entry_room().id == "room_1"


def test_dungeon_module_no_rooms():
    module = DungeonModule(title="Empty")
    assert module.get_entry_room() is None
