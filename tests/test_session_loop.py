"""Tests for the new DM-tool-driven session orchestrator."""
from __future__ import annotations

from unittest.mock import MagicMock

from dndplaya.agents.base import AgentResponse, ToolCall
from dndplaya.mechanics.characters import create_default_party
from dndplaya.mechanics.dice import DiceRoller
from dndplaya.mechanics.state import GameState
from dndplaya.orchestrator.session import Session
from dndplaya.orchestrator.transcript import SessionTranscript
from dndplaya.config import Settings
from pydantic import SecretStr


def _make_settings() -> Settings:
    return Settings(anthropic_api_key=SecretStr("test-key-123"), max_tokens=512)


def _agent_response(text="", tool_calls=None, stop_reason="end_turn"):
    """Create an AgentResponse for testing."""
    return AgentResponse(
        text=text,
        tool_calls=tool_calls or [],
        raw_content=[{"type": "text", "text": text}] if text else [],
        stop_reason=stop_reason,
    )


def _tool_call(name, arguments, id="tc_1"):
    return ToolCall(id=id, name=name, arguments=arguments)


class TestSessionToolDispatch:
    """Test individual tool handlers without running the full session loop."""

    def _make_session(self):
        settings = _make_settings()
        party = create_default_party(3)
        session = Session.__new__(Session)
        session.settings = settings
        session.dice = DiceRoller(seed=42)
        session.party = party
        session.state = GameState(characters=party)
        session.transcript = SessionTranscript()
        session.total_turns = 0
        session._terminated = False
        session.dm = MagicMock()
        session.players = []
        session._player_map = {}
        return session

    def test_handle_roll_check_success(self):
        session = self._make_session()
        # Seed 42 first d20 roll is deterministic
        result = session._handle_roll_check({
            "modifier": 5,
            "dc": 10,
            "description": "Thorin attacks goblin",
        })
        assert "Thorin attacks goblin" in result
        assert "SUCCESS" in result or "FAILURE" in result

    def test_handle_roll_dice(self):
        session = self._make_session()
        result = session._handle_roll_dice({
            "expression": "2d6+3",
            "reason": "longsword damage",
        })
        assert "longsword damage" in result
        assert "2d6+3" in result

    def test_handle_apply_damage(self):
        session = self._make_session()
        initial_hp = session.party[0].current_hp
        result = session._handle_apply_damage({
            "character_name": "Thorin",
            "amount": 5,
            "description": "goblin stab",
        })
        assert session.party[0].current_hp == initial_hp - 5
        assert "Thorin" in result
        assert f"{initial_hp - 5}/{session.party[0].max_hp}" in result

    def test_handle_apply_damage_unknown_character(self):
        session = self._make_session()
        result = session._handle_apply_damage({
            "character_name": "Gandalf",
            "amount": 5,
            "description": "fireball",
        })
        assert "not found" in result

    def test_handle_apply_damage_kills_character(self):
        session = self._make_session()
        session.party[0].current_hp = 3
        result = session._handle_apply_damage({
            "character_name": "Thorin",
            "amount": 10,
            "description": "dragon breath",
        })
        assert session.party[0].current_hp == 0
        assert "DOWN" in result

    def test_handle_heal(self):
        session = self._make_session()
        session.party[0].current_hp = 10
        max_hp = session.party[0].max_hp
        result = session._handle_heal({
            "character_name": "Thorin",
            "amount": 5,
            "description": "cure wounds",
        })
        assert session.party[0].current_hp == 15
        assert "healed 5" in result

    def test_handle_heal_caps_at_max(self):
        session = self._make_session()
        session.party[0].current_hp = session.party[0].max_hp - 2
        result = session._handle_heal({
            "character_name": "Thorin",
            "amount": 10,
            "description": "healing word",
        })
        assert session.party[0].current_hp == session.party[0].max_hp
        assert "healed 2" in result

    def test_handle_get_party_status(self):
        session = self._make_session()
        result = session._handle_get_party_status({})
        assert "Thorin" in result
        assert "Shadow" in result
        assert "Elara" in result
        assert "Brother Marcus" in result
        assert "HP" in result
        assert "AC" in result

    def test_handle_enter_room(self):
        session = self._make_session()
        result = session._handle_enter_room({"room_name": "The Dark Cave"})
        assert "The Dark Cave" in result
        assert session.transcript.current_room == "The Dark Cave"
        assert "The Dark Cave" in session.state.rooms_visited

    def test_handle_end_session(self):
        session = self._make_session()
        result = session._handle_end_session({"reason": "Dungeon cleared"})
        assert session._terminated is True
        assert "Dungeon cleared" in result

    def test_handle_request_player_input_unknown_player(self):
        session = self._make_session()
        result = session._handle_request_player_input({
            "player_names": ["Gandalf"],
        })
        assert "Unknown player" in result

    def test_handle_request_player_input_empty(self):
        session = self._make_session()
        result = session._handle_request_player_input({
            "player_names": [],
        })
        assert "No player names" in result

    def test_dispatch_unknown_tool(self):
        session = self._make_session()
        result = session._dispatch_tool("nonexistent_tool", {})
        assert "Unknown tool" in result


class TestSessionTPK:
    """Test TPK detection in the session."""

    def test_party_dead_after_damage(self):
        settings = _make_settings()
        party = create_default_party(3)
        session = Session.__new__(Session)
        session.settings = settings
        session.dice = DiceRoller(seed=42)
        session.party = party
        session.state = GameState(characters=party)
        session.transcript = SessionTranscript()
        session.total_turns = 0
        session._terminated = False

        # Kill all party members
        for char in party:
            char.current_hp = 0

        assert session._is_party_dead() is True

    def test_party_alive(self):
        settings = _make_settings()
        party = create_default_party(3)
        session = Session.__new__(Session)
        session.settings = settings
        session.party = party
        session.state = GameState(characters=party)

        assert session._is_party_dead() is False


class TestProcessToolCalls:
    """Test the tool call processing pipeline."""

    def _make_session(self):
        settings = _make_settings()
        party = create_default_party(3)
        session = Session.__new__(Session)
        session.settings = settings
        session.dice = DiceRoller(seed=42)
        session.party = party
        session.state = GameState(characters=party)
        session.transcript = SessionTranscript()
        session.total_turns = 0
        session._terminated = False
        session.dm = MagicMock()
        session.players = []
        session._player_map = {}
        return session

    def test_process_multiple_tool_calls(self):
        session = self._make_session()
        response = _agent_response(
            text="The battle begins!",
            tool_calls=[
                _tool_call("enter_room", {"room_name": "Cave"}, id="tc_1"),
                _tool_call("get_party_status", {}, id="tc_2"),
            ],
        )
        results = session._process_tool_calls(response)
        assert len(results) == 2
        assert results[0][0] == "tc_1"
        assert "Cave" in results[0][1]
        assert results[1][0] == "tc_2"
        assert "Thorin" in results[1][1]

    def test_narration_recorded_before_tools(self):
        session = self._make_session()
        response = _agent_response(
            text="You enter a dark chamber.",
            tool_calls=[_tool_call("enter_room", {"room_name": "Chamber"})],
        )
        session._process_tool_calls(response)
        narrations = [
            e for e in session.transcript.entries
            if e.entry_type == "narration"
        ]
        assert len(narrations) == 1
        assert narrations[0].content == "You enter a dark chamber."
