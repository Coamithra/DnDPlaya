"""Tests for the DM-tool-driven session orchestrator."""
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
        session.pages = None
        session._last_read_page = None
        session.module_references = []
        session._active_monsters = {}
        session._initiative_order = []
        session._initiative_index = -1
        session._consecutive_no_tool_turns = 0
        session._narration_count = 0
        session._cost_budget = 3.00
        session._cache_check_turn = 10
        session.ui = None
        session._pending_ui_wait = False

        return session

    def test_handle_ask_skill_check_success(self):
        session = self._make_session()
        result = session._handle_ask_skill_check({
            "player": "Thorin",
            "skill": "athletics",
            "difficulty": "easy",
        })
        assert "Thorin" in result
        assert "SUCCESS" in result or "FAILURE" in result
        assert "athletics" in result
        assert "DC 10" in result

    def test_handle_ask_skill_check_with_advantage(self):
        session = self._make_session()
        result = session._handle_ask_skill_check({
            "player": "Thorin",
            "skill": "athletics",
            "difficulty": "medium",
            "has_advantage": True,
        })
        assert "advantage" in result
        assert "Thorin" in result

    def test_handle_ask_skill_check_unknown_character(self):
        session = self._make_session()
        result = session._handle_ask_skill_check({
            "player": "Gandalf",
            "skill": "athletics",
            "difficulty": "easy",
        })
        assert "not found" in result

    def test_handle_ask_skill_check_default_skill_bonus(self):
        session = self._make_session()
        # Thorin (Fighter) doesn't have "arcana" as a skill
        result = session._handle_ask_skill_check({
            "player": "Thorin",
            "skill": "arcana",
            "difficulty": "medium",
        })
        assert "+0 bonus" in result

    def test_handle_ask_skill_check_all_difficulties(self):
        session = self._make_session()
        dcs = {"very_easy": 5, "easy": 10, "medium": 13, "hard": 16, "very_hard": 20, "nearly_impossible": 25}
        for diff, expected_dc in dcs.items():
            session.dice = DiceRoller(seed=42)
            result = session._handle_ask_skill_check({
                "player": "Thorin",
                "skill": "athletics",
                "difficulty": diff,
            })
            assert f"DC {expected_dc}" in result

    def test_handle_attack_monster_hits_pc(self):
        session = self._make_session()
        # Register a goblin first
        from dndplaya.mechanics.monsters import create_monster
        goblin = create_monster("Goblin", 0.25)
        session._active_monsters["goblin"] = goblin

        result = session._handle_attack({
            "attacker": "Goblin",
            "target": "Thorin",
        })
        assert "Goblin" in result
        assert "Thorin" in result
        assert "HIT" in result or "MISS" in result

    def test_handle_attack_monster_not_registered(self):
        session = self._make_session()
        result = session._handle_attack({
            "attacker": "Goblin",
            "target": "Thorin",
        })
        assert "not found" in result
        assert "roll_initiative" in result

    def test_handle_attack_unknown_target(self):
        session = self._make_session()
        from dndplaya.mechanics.monsters import create_monster
        goblin = create_monster("Goblin", 0.25)
        session._active_monsters["goblin"] = goblin

        result = session._handle_attack({
            "attacker": "Goblin",
            "target": "Gandalf",
        })
        assert "not found" in result

    def test_handle_change_hp_damage(self):
        session = self._make_session()
        initial_hp = session.party[0].current_hp
        result = session._handle_change_hp({
            "target": "Thorin",
            "amount": -5,
            "reason": "fire trap",
        })
        assert session.party[0].current_hp == initial_hp - 5
        assert "Thorin" in result
        assert "fire trap" in result

    def test_handle_change_hp_heal(self):
        session = self._make_session()
        session.party[0].current_hp = 10
        result = session._handle_change_hp({
            "target": "Thorin",
            "amount": 5,
            "reason": "healing potion",
        })
        assert session.party[0].current_hp == 15
        assert "healing potion" in result

    def test_handle_change_hp_kill(self):
        session = self._make_session()
        session.party[0].current_hp = 3
        result = session._handle_change_hp({
            "target": "Thorin",
            "amount": -10,
            "reason": "lava",
        })
        assert session.party[0].current_hp == 0
        assert "DOWN" in result

    def test_handle_change_hp_caps_at_max(self):
        session = self._make_session()
        session.party[0].current_hp = session.party[0].max_hp - 2
        result = session._handle_change_hp({
            "target": "Thorin",
            "amount": 100,
            "reason": "divine intervention",
        })
        assert session.party[0].current_hp == session.party[0].max_hp

    def test_handle_change_hp_unknown_character(self):
        session = self._make_session()
        result = session._handle_change_hp({
            "target": "Gandalf",
            "amount": -5,
            "reason": "test",
        })
        assert "not found" in result

    def test_handle_roll_initiative(self):
        session = self._make_session()
        result = session._handle_roll_initiative({
            "monsters": [
                {"name": "Goblin", "cr": 0.25},
                {"name": "Hobgoblin", "cr": 1},
            ],
        })
        assert "Initiative order" in result
        assert "Goblin" in result
        assert "Hobgoblin" in result
        assert "Thorin" in result
        assert session.state.in_combat is True
        assert "goblin" in session._active_monsters
        assert "hobgoblin" in session._active_monsters

    def test_handle_roll_initiative_invalid_cr(self):
        session = self._make_session()
        result = session._handle_roll_initiative({
            "monsters": [{"name": "Dragon", "cr": 99}],
        })
        assert "Error" in result

    def test_handle_roll_initiative_empty(self):
        session = self._make_session()
        result = session._handle_roll_initiative({"monsters": []})
        assert "No monsters" in result

    def test_handle_get_party_status(self):
        session = self._make_session()
        result = session._handle_get_party_status({})
        assert "Thorin" in result
        assert "Shadow" in result
        assert "Elara" in result
        assert "Brother Marcus" in result
        assert "HP" in result
        assert "AC" in result
        assert "Skills:" in result

    def test_handle_end_session(self):
        session = self._make_session()
        result = session._handle_end_session({"reason": "Dungeon cleared"})
        assert session._terminated is True
        assert "Dungeon cleared" in result

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
        session.pages = None
        session._last_read_page = None
        session.module_references = []
        session._active_monsters = {}
        session._initiative_order = []
        session._initiative_index = -1
        session._consecutive_no_tool_turns = 0
        session._narration_count = 0
        session._cost_budget = 3.00
        session._cache_check_turn = 10
        session.ui = None
        session._pending_ui_wait = False

        return session

    def test_process_multiple_tool_calls(self):
        session = self._make_session()
        response = _agent_response(
            text="The battle begins!",
            tool_calls=[
                _tool_call("get_party_status", {}, id="tc_1"),
                _tool_call("get_party_status", {}, id="tc_2"),
            ],
        )
        results = session._process_tool_calls(response)
        assert len(results) == 2
        assert results[0][0] == "tc_1"
        assert "Thorin" in results[0][1]
        assert results[1][0] == "tc_2"
        assert "Thorin" in results[1][1]

    def test_dm_text_recorded_as_internal(self):
        session = self._make_session()
        response = _agent_response(
            text="Let me think about this room...",
            tool_calls=[_tool_call("get_party_status", {})],
        )
        session._process_tool_calls(response)
        system_events = [
            e for e in session.transcript.entries
            if e.entry_type == "system"
        ]
        assert any("DM internal:" in e.content for e in system_events)

    def test_narrate_tool_records_narration(self):
        session = self._make_session()
        result = session._handle_narrate({"text": "You enter a dark chamber."})
        assert result == "Narrated."
        narrations = [
            e for e in session.transcript.entries
            if e.entry_type == "narration"
        ]
        assert len(narrations) == 1
        assert narrations[0].content == "You enter a dark chamber."


class TestModuleReferenceTools:
    """Test the module reference tool handlers."""

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
        session.pages = [
            "# Page 1\nThe entrance to the Goblin Cave.\nA dark tunnel leads north.",
            "# Page 2\nRoom 1: The Guard Room\nTwo goblins stand guard here. They have 7 HP each.",
            "# Page 3\nRoom 2: The Treasure Chamber\nA chest contains 50 gold pieces and a healing potion.",
        ]
        session._last_read_page = None
        session.module_references = []
        session._active_monsters = {}
        session._initiative_order = []
        session._initiative_index = -1
        session._consecutive_no_tool_turns = 0
        session._narration_count = 0
        session._cost_budget = 3.00
        session._cache_check_turn = 10
        session.ui = None
        session._pending_ui_wait = False

        return session

    # --- search_module ---

    def test_search_found(self):
        session = self._make_session()
        result = session._handle_search_module({"query": "goblin"})
        assert "Page 1" in result or "Page 2" in result
        assert "Search results" in result

    def test_search_not_found(self):
        session = self._make_session()
        result = session._handle_search_module({"query": "dragon"})
        assert "No matches" in result

    def test_search_case_insensitive(self):
        session = self._make_session()
        result = session._handle_search_module({"query": "GOBLIN"})
        assert "Search results" in result

    def test_search_no_pages(self):
        session = self._make_session()
        session.pages = None
        result = session._handle_search_module({"query": "goblin"})
        assert "not loaded" in result

    def test_search_empty_query(self):
        session = self._make_session()
        result = session._handle_search_module({"query": ""})
        assert "No search query" in result

    def test_search_logs_reference(self):
        session = self._make_session()
        session._handle_search_module({"query": "goblin"})
        assert len(session.module_references) == 1
        assert session.module_references[0]["tool"] == "search_module"
        assert session.module_references[0]["query"] == "goblin"

    # --- read_page ---

    def test_read_valid_page(self):
        session = self._make_session()
        result = session._handle_read_page({"page_number": 2})
        assert "Guard Room" in result
        assert "Page 2" in result
        assert session._last_read_page == 2

    def test_read_page_out_of_range_high(self):
        session = self._make_session()
        result = session._handle_read_page({"page_number": 10})
        assert "Invalid page number" in result

    def test_read_page_out_of_range_zero(self):
        session = self._make_session()
        result = session._handle_read_page({"page_number": 0})
        assert "Invalid page number" in result

    def test_read_page_no_pages(self):
        session = self._make_session()
        session.pages = None
        result = session._handle_read_page({"page_number": 1})
        assert "not loaded" in result

    def test_read_page_logs_reference(self):
        session = self._make_session()
        session._handle_read_page({"page_number": 2})
        assert len(session.module_references) == 1
        assert session.module_references[0]["tool"] == "read_page"
        assert session.module_references[0]["page"] == 2

    # --- next_page ---

    def test_next_page_from_none(self):
        session = self._make_session()
        result = session._handle_next_page({})
        assert "Page 1" in result
        assert session._last_read_page == 1

    def test_next_page_sequential(self):
        session = self._make_session()
        session._last_read_page = 1
        result = session._handle_next_page({})
        assert "Page 2" in result
        assert session._last_read_page == 2

    def test_next_page_at_end(self):
        session = self._make_session()
        session._last_read_page = 3
        result = session._handle_next_page({})
        assert "last page" in result

    def test_next_page_no_pages(self):
        session = self._make_session()
        session.pages = None
        result = session._handle_next_page({})
        assert "not loaded" in result

    # --- previous_page ---

    def test_previous_page(self):
        session = self._make_session()
        session._last_read_page = 3
        result = session._handle_previous_page({})
        assert "Page 2" in result
        assert session._last_read_page == 2

    def test_previous_page_at_start(self):
        session = self._make_session()
        session._last_read_page = 1
        result = session._handle_previous_page({})
        assert "first page" in result

    def test_previous_page_no_read_yet(self):
        session = self._make_session()
        result = session._handle_previous_page({})
        assert "No page read yet" in result

    def test_previous_page_no_pages(self):
        session = self._make_session()
        session.pages = None
        result = session._handle_previous_page({})
        assert "not loaded" in result

    # --- metrics ---

    def test_multiple_references_tracked(self):
        session = self._make_session()
        session._handle_search_module({"query": "goblin"})
        session._handle_read_page({"page_number": 1})
        session._handle_next_page({})
        assert len(session.module_references) == 3



class TestMonsterRegistration:
    """Test that roll_initiative properly registers monsters."""

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
        session.pages = None
        session._last_read_page = None
        session.module_references = []
        session._active_monsters = {}
        session._initiative_order = []
        session._initiative_index = -1
        session._consecutive_no_tool_turns = 0
        session._narration_count = 0
        session._cost_budget = 3.00
        session._cache_check_turn = 10
        session.ui = None
        session._pending_ui_wait = False

        return session

    def test_monsters_registered_after_initiative(self):
        session = self._make_session()
        session._handle_roll_initiative({
            "monsters": [{"name": "Goblin", "cr": 0.25}],
        })
        assert "goblin" in session._active_monsters
        assert session._active_monsters["goblin"].cr == 0.25

    def test_multiple_monsters_registered(self):
        session = self._make_session()
        session._handle_roll_initiative({
            "monsters": [
                {"name": "Goblin", "cr": 0.25},
                {"name": "Hobgoblin", "cr": 1},
            ],
        })
        assert len(session._active_monsters) == 2

    def test_old_monsters_cleared_on_new_initiative(self):
        session = self._make_session()
        session._handle_roll_initiative({
            "monsters": [{"name": "Goblin", "cr": 0.25}],
        })
        session._handle_roll_initiative({
            "monsters": [{"name": "Orc", "cr": 1}],
        })
        assert "goblin" not in session._active_monsters
        assert "orc" in session._active_monsters

    def test_attack_works_after_initiative(self):
        session = self._make_session()
        session._handle_roll_initiative({
            "monsters": [{"name": "Goblin", "cr": 0.25}],
        })
        result = session._handle_attack({
            "attacker": "Goblin",
            "target": "Thorin",
        })
        assert "HIT" in result or "MISS" in result
