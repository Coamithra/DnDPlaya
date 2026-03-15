"""Tests for game state management."""
from dndplaya.mechanics.state import GameState, EventType
from dndplaya.mechanics.characters import create_default_party
from dndplaya.mechanics.monsters import create_monster


def test_initial_state():
    party = create_default_party(3)
    state = GameState(characters=party)
    assert len(state.characters) == 4
    assert state.current_room is None
    assert not state.in_combat
    assert len(state.events) == 0


def test_enter_room():
    party = create_default_party(3)
    state = GameState(characters=party)
    state.enter_room("room_1")
    assert state.current_room == "room_1"
    assert "room_1" in state.rooms_visited
    assert len(state.events) == 1
    assert state.events[0].event_type == EventType.ROOM_ENTERED


def test_combat_lifecycle():
    party = create_default_party(3)
    state = GameState(characters=party)
    monsters = [create_monster("Goblin", 0.25)]

    state.start_combat(monsters)
    assert state.in_combat
    assert len(state.monsters) == 1
    assert state.round_number == 1

    state.end_combat()
    assert not state.in_combat
    assert len(state.monsters) == 0


def test_get_alive():
    party = create_default_party(3)
    state = GameState(characters=party)

    assert len(state.get_alive_characters()) == 4
    party[0].current_hp = 0
    assert len(state.get_alive_characters()) == 3


def test_short_rest():
    party = create_default_party(3)
    state = GameState(characters=party)

    # Wound the fighter
    party[0].current_hp = 10
    state.take_short_rest()

    # Should heal 25% of max_hp
    expected_heal = int(party[0].max_hp * 0.25)
    assert party[0].current_hp == 10 + expected_heal


def test_long_rest():
    party = create_default_party(3)
    state = GameState(characters=party)

    for c in party:
        c.current_hp = 1
    state.take_long_rest()

    for c in party:
        assert c.current_hp == c.max_hp


def test_add_event_with_type():
    party = create_default_party(3)
    state = GameState(characters=party)
    state.add_event(EventType.PLAYER_ACTION, "searches the room", actor="Thorin")
    assert len(state.events) == 1
    assert state.events[0].event_type == EventType.PLAYER_ACTION
    assert state.events[0].actor == "Thorin"


def test_get_character():
    party = create_default_party(3)
    state = GameState(characters=party)
    assert state.get_character("Thorin") is not None
    assert state.get_character("thorin") is not None  # case-insensitive
    assert state.get_character("Nobody") is None
