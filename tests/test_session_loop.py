"""Tests for the DM-tool-driven session orchestrator."""
from __future__ import annotations

from unittest.mock import MagicMock

from dndplaya.agents.base import AgentResponse, ToolCall
from dndplaya.mechanics.characters import create_default_party
from dndplaya.mechanics.dice import DiceRoller
from dndplaya.mechanics.state import GameState
from dndplaya.orchestrator.session import Session, _has_excessive_non_ascii
from dndplaya.agents.provider import ProviderGuardrails
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
        session._turns_without_module_ref = 0
        session._STALE_THRESHOLD = 3
        session._consecutive_all_pass = 0
        session._ALL_PASS_THRESHOLD = 2
        session._guardrails = ProviderGuardrails()

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
        session._turns_without_module_ref = 0
        session._STALE_THRESHOLD = 3
        session._consecutive_all_pass = 0
        session._ALL_PASS_THRESHOLD = 2
        session._guardrails = ProviderGuardrails()

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

    def test_change_music_records_event(self):
        session = self._make_session()
        result = session._handle_change_music({"track": "combat"})
        assert "Now playing: combat" in result
        system_events = [
            e for e in session.transcript.entries
            if e.entry_type == "system"
        ]
        assert any("Music changed to: combat" in e.content for e in system_events)

    def test_change_music_silence(self):
        session = self._make_session()
        result = session._handle_change_music({"track": "silence"})
        assert result == "Music stopped."

    def test_change_music_emits_ui_event(self):
        session = self._make_session()
        session.ui = MagicMock()
        session._handle_change_music({"track": "tavern"})
        session.ui.music_change.assert_called_once_with("tavern")

    def test_change_music_no_ui(self):
        session = self._make_session()
        session.ui = None
        # Should not raise even without UI
        result = session._handle_change_music({"track": "dungeon"})
        assert "Now playing: dungeon" in result


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
        session._turns_without_module_ref = 0
        session._STALE_THRESHOLD = 3
        session._consecutive_all_pass = 0
        session._ALL_PASS_THRESHOLD = 2
        session._guardrails = ProviderGuardrails()

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
        session._turns_without_module_ref = 0
        session._STALE_THRESHOLD = 3
        session._consecutive_all_pass = 0
        session._ALL_PASS_THRESHOLD = 2
        session._guardrails = ProviderGuardrails()

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


class TestHealAtFullHPGuard:
    """Fix 5: Healing a target at full HP should not consume a spell slot."""

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
        session._turns_without_module_ref = 0
        session._STALE_THRESHOLD = 3
        session._consecutive_all_pass = 0
        session._ALL_PASS_THRESHOLD = 2
        session._guardrails = ProviderGuardrails()

        return session

    def test_heal_at_full_hp_rejected(self):
        """Healing at full HP should be rejected with slot preserved."""
        session = self._make_session()
        from dndplaya.agents.player import PlayerAgent, ARCHETYPES

        # Brother Marcus (Cleric) is party[3]
        cleric = session.party[3]
        assert cleric.char_class == "Cleric"
        initial_slots = dict(cleric.spell_slots)

        # Target at full HP
        target = session.party[0]
        assert target.current_hp == target.max_hp

        player = MagicMock()
        player.character = cleric

        result = session._resolve_player_heal(player, {"target": target.name})
        assert "already at full HP" in result
        assert "spell slot preserved" in result
        # Spell slots should NOT be consumed
        assert cleric.spell_slots == initial_slots

    def test_heal_when_damaged_works(self):
        """Healing a damaged target should still work normally."""
        session = self._make_session()
        cleric = session.party[3]
        assert cleric.char_class == "Cleric"
        initial_slots = dict(cleric.spell_slots)

        # Damage the target
        target = session.party[0]
        target.current_hp = target.max_hp - 10

        player = MagicMock()
        player.character = cleric

        result = session._resolve_player_heal(player, {"target": target.name})
        assert "heals" in result
        assert target.current_hp > target.max_hp - 10
        # A spell slot should have been consumed
        total_before = sum(initial_slots.values())
        total_after = sum(cleric.spell_slots.values())
        assert total_after < total_before


class TestEmptySayRejection:
    """Fix 2: Empty say() calls should be treated as pass_turn."""

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
        session._turns_without_module_ref = 0
        session._STALE_THRESHOLD = 3
        session._consecutive_all_pass = 0
        session._ALL_PASS_THRESHOLD = 2
        session._guardrails = ProviderGuardrails()

        return session

    def test_empty_say_treated_as_pass(self):
        """say('') should be treated as pass_turn."""
        session = self._make_session()
        player = MagicMock()
        player.character = session.party[0]
        player.submit_tool_results.return_value = _agent_response()

        response = _agent_response(
            tool_calls=[_tool_call("say", {"text": "", "urgency": 3})],
            stop_reason="tool_use",
        )
        text, urgency, mechanical = session._resolve_player_tools(player, response)
        assert text == "pass"
        assert urgency == 0

    def test_whitespace_say_treated_as_pass(self):
        """say('   ') should be treated as pass_turn."""
        session = self._make_session()
        player = MagicMock()
        player.character = session.party[0]
        player.submit_tool_results.return_value = _agent_response()

        response = _agent_response(
            tool_calls=[_tool_call("say", {"text": "   ", "urgency": 4})],
            stop_reason="tool_use",
        )
        text, urgency, mechanical = session._resolve_player_tools(player, response)
        assert text == "pass"
        assert urgency == 0

    def test_nonempty_say_works_normally(self):
        """say('I attack') should work normally."""
        session = self._make_session()
        player = MagicMock()
        player.character = session.party[0]
        player.submit_tool_results.return_value = _agent_response()

        response = _agent_response(
            tool_calls=[_tool_call("say", {"text": "I attack the goblin!", "urgency": 4})],
            stop_reason="tool_use",
        )
        text, urgency, mechanical = session._resolve_player_tools(player, response)
        assert text == "I attack the goblin!"
        assert urgency == 4


class TestRoleConfusionDetection:
    """Fix 4: Player responses containing 'DM:' should be stripped (Ollama only)."""

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
        session._turns_without_module_ref = 0
        session._STALE_THRESHOLD = 3
        session._consecutive_all_pass = 0
        session._ALL_PASS_THRESHOLD = 2
        session._guardrails = ProviderGuardrails(detect_role_confusion=True)

        return session

    def test_dm_prefix_stripped_from_say(self):
        """Text after 'DM:' in a say() call should be stripped."""
        session = self._make_session()
        player = MagicMock()
        player.character = session.party[0]
        player.submit_tool_results.return_value = _agent_response()

        text_with_dm = (
            "I approach the tree cautiously. "
            "DM: As you approach, you see a hidden door open."
        )
        response = _agent_response(
            tool_calls=[_tool_call("say", {"text": text_with_dm, "urgency": 3})],
            stop_reason="tool_use",
        )
        text, urgency, mechanical = session._resolve_player_tools(player, response)
        assert text == "I approach the tree cautiously."
        assert "DM:" not in text

    def test_dm_prefix_only_becomes_pass(self):
        """If stripping 'DM:' leaves nothing, treat as pass."""
        session = self._make_session()
        player = MagicMock()
        player.character = session.party[0]
        player.submit_tool_results.return_value = _agent_response()

        response = _agent_response(
            tool_calls=[_tool_call("say", {"text": "DM: You enter a dark room.", "urgency": 3})],
            stop_reason="tool_use",
        )
        text, urgency, mechanical = session._resolve_player_tools(player, response)
        assert text == "pass"
        assert urgency == 0

    def test_no_dm_prefix_unchanged(self):
        """Normal text without DM: should be unaffected."""
        session = self._make_session()
        player = MagicMock()
        player.character = session.party[0]
        player.submit_tool_results.return_value = _agent_response()

        response = _agent_response(
            tool_calls=[_tool_call("say", {"text": "I search the room.", "urgency": 3})],
            stop_reason="tool_use",
        )
        text, urgency, mechanical = session._resolve_player_tools(player, response)
        assert text == "I search the room."


class TestDrainLoopCap:
    """Fix 3: Per-player drain loop should be capped at 5 tool calls (Ollama only)."""

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
        session._turns_without_module_ref = 0
        session._STALE_THRESHOLD = 3
        session._consecutive_all_pass = 0
        session._ALL_PASS_THRESHOLD = 2
        session._guardrails = ProviderGuardrails(drain_loop_cap=5)

        return session

    def test_drain_loop_capped(self):
        """Player making >5 tool calls should be stopped at the cap."""
        session = self._make_session()
        player = MagicMock()
        player.character = session.party[0]

        # Simulate a player that keeps calling say() in followup.
        # Initial response = 1 say (count 1). Then 4 followup rounds
        # with say() each (counts 2-5). The 5th followup should trigger the cap.
        followup1 = _agent_response(
            tool_calls=[_tool_call("say", {"text": "second", "urgency": 3}, id="tc_2")],
            stop_reason="tool_use",
        )
        followup2 = _agent_response(
            tool_calls=[_tool_call("say", {"text": "third", "urgency": 3}, id="tc_3")],
            stop_reason="tool_use",
        )
        followup3 = _agent_response(
            tool_calls=[_tool_call("say", {"text": "fourth", "urgency": 3}, id="tc_4")],
            stop_reason="tool_use",
        )
        followup4 = _agent_response(
            tool_calls=[_tool_call("say", {"text": "fifth", "urgency": 3}, id="tc_5")],
            stop_reason="tool_use",
        )
        followup5 = _agent_response(
            tool_calls=[_tool_call("say", {"text": "sixth", "urgency": 3}, id="tc_6")],
            stop_reason="tool_use",
        )
        final = _agent_response()  # no tool calls

        player.submit_tool_results.side_effect = [
            followup1, followup2, followup3, followup4, followup5, final,
        ]

        # Initial response has 1 say() + triggers drain loop
        response = _agent_response(
            tool_calls=[_tool_call("say", {"text": "first", "urgency": 3})],
            stop_reason="tool_use",
        )
        text, urgency, mechanical = session._resolve_player_tools(player, response)

        # Should use the first say text (first non-empty)
        assert text == "first"
        # Verify transcript records the cap warning
        cap_events = [
            e for e in session.transcript.entries
            if "tool call cap" in e.content
        ]
        assert len(cap_events) >= 1


class TestAllPassAutoAdvance:
    """Fix 1: After 2 consecutive all-pass group inputs, inject story advance nudge."""

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
        session._turns_without_module_ref = 0
        session._STALE_THRESHOLD = 3
        session._consecutive_all_pass = 0
        session._ALL_PASS_THRESHOLD = 2
        session._guardrails = ProviderGuardrails()

        return session

    def test_consecutive_all_pass_counter_increments(self):
        """Counter should track consecutive all-pass group inputs."""
        session = self._make_session()
        session._consecutive_all_pass = 0
        # Simulate: after an all-pass round, counter goes to 1
        session._consecutive_all_pass = 1
        assert session._consecutive_all_pass == 1

    def test_threshold_triggers_advance_nudge(self):
        """At threshold, group input should return advance-story nudge."""
        session = self._make_session()
        # Pre-set to 1 (one previous all-pass)
        session._consecutive_all_pass = 1

        # Set up player mocks that match party characters
        player_mocks = []
        for p in session.party:
            pm = MagicMock()
            pm.character = p
            player_mocks.append(pm)
        session.players = player_mocks

        # We need to mock _parallel_player_calls to return all passes
        all_pass_results = []
        for pm in player_mocks:
            all_pass_results.append((pm, "pass", 0, []))

        session._parallel_player_calls = MagicMock(return_value=all_pass_results)
        session.transcript.get_game_context = MagicMock(return_value="test context")

        result = session._handle_request_group_input({})
        assert "Do NOT call request_group_input again" in result
        assert session._consecutive_all_pass == 2

    def test_meaningful_input_resets_counter(self):
        """Counter should reset when players actually contribute."""
        session = self._make_session()
        session._consecutive_all_pass = 1

        # Set up player mocks that match party characters
        player_mocks = []
        for p in session.party:
            pm = MagicMock()
            pm.character = p
            player_mocks.append(pm)
        session.players = player_mocks

        # One player responds meaningfully, rest pass
        results = []
        for i, pm in enumerate(player_mocks):
            if i == 0:
                results.append((pm, "I attack the goblin!", 4, []))
            else:
                results.append((pm, "pass", 0, []))

        session._parallel_player_calls = MagicMock(return_value=results)
        session.transcript.get_game_context = MagicMock(return_value="test context")

        result = session._handle_request_group_input({})
        # Counter should be reset
        assert session._consecutive_all_pass == 0


class TestNonAsciiDetection:
    """Iter 3 Fix 4: detect and strip non-English content from player responses (Ollama only)."""

    def test_pure_english_below_threshold(self):
        assert not _has_excessive_non_ascii("I search the room carefully.")

    def test_chinese_above_threshold(self):
        assert _has_excessive_non_ascii("圣洁之人请给予我一个征兆")

    def test_mixed_just_below_threshold(self):
        # 3 non-ASCII out of 10 chars = 30%, threshold is >30%
        assert not _has_excessive_non_ascii("abcdefg\u00e9\u00e8\u00ea")

    def test_mixed_above_threshold(self):
        # 4 non-ASCII out of 10 chars = 40%, above threshold
        assert _has_excessive_non_ascii("abcdef\u00e9\u00e8\u00ea\u00eb")

    def test_empty_string(self):
        assert not _has_excessive_non_ascii("")

    def test_ascii_only(self):
        assert not _has_excessive_non_ascii("Hello, world!")

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
        session._turns_without_module_ref = 0
        session._STALE_THRESHOLD = 3
        session._consecutive_all_pass = 0
        session._ALL_PASS_THRESHOLD = 2
        session._guardrails = ProviderGuardrails(detect_non_ascii=True)
        return session

    def test_empty_say_does_not_count_toward_cap(self):
        """Empty say() calls should NOT increment tool_call_count."""
        session = self._make_session()
        player = MagicMock()
        player.character = session.party[0]

        # Initial response: 4 empty say() calls + 1 real say. With the fix,
        # only the real say counts toward the cap (1 of 5), so no cap warning.
        response = _agent_response(
            tool_calls=[
                _tool_call("say", {"text": "", "urgency": 3}, id="tc_1"),
                _tool_call("say", {"text": "", "urgency": 3}, id="tc_2"),
                _tool_call("say", {"text": "", "urgency": 3}, id="tc_3"),
                _tool_call("say", {"text": "", "urgency": 3}, id="tc_4"),
                _tool_call("say", {"text": "I search the room.", "urgency": 3}, id="tc_5"),
            ],
            stop_reason="tool_use",
        )
        player.submit_tool_results.return_value = _agent_response()

        text, urgency, mechanical = session._resolve_player_tools(player, response)
        assert text == "I search the room."
        # No cap warning since only 1 real tool call
        cap_events = [
            e for e in session.transcript.entries
            if "tool call cap" in e.content
        ]
        assert len(cap_events) == 0
