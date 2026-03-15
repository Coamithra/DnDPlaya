"""Tests for DM tool definitions — validates schema correctness."""
from __future__ import annotations

from dndplaya.agents.dm_tools import DM_TOOLS


EXPECTED_TOOL_NAMES = [
    "roll_check",
    "roll_dice",
    "apply_damage",
    "heal",
    "get_party_status",
    "enter_room",
    "request_player_input",
    "end_session",
]


class TestDMTools:
    def test_all_tools_present(self):
        tool_names = [t["name"] for t in DM_TOOLS]
        assert tool_names == EXPECTED_TOOL_NAMES

    def test_tool_count(self):
        assert len(DM_TOOLS) == 8

    def test_all_tools_have_required_fields(self):
        for tool in DM_TOOLS:
            assert "name" in tool, f"Tool missing 'name': {tool}"
            assert "description" in tool, f"Tool missing 'description': {tool}"
            assert "input_schema" in tool, f"Tool missing 'input_schema': {tool}"

    def test_all_schemas_have_type_object(self):
        for tool in DM_TOOLS:
            schema = tool["input_schema"]
            assert schema["type"] == "object", (
                f"Tool '{tool['name']}' schema type should be 'object'"
            )
            assert "properties" in schema, (
                f"Tool '{tool['name']}' schema missing 'properties'"
            )

    def test_roll_check_schema(self):
        tool = next(t for t in DM_TOOLS if t["name"] == "roll_check")
        props = tool["input_schema"]["properties"]
        required = tool["input_schema"]["required"]

        assert "modifier" in props
        assert "dc" in props
        assert "description" in props
        assert props["modifier"]["type"] == "integer"
        assert props["dc"]["type"] == "integer"
        assert props["description"]["type"] == "string"
        assert set(required) == {"modifier", "dc", "description"}

    def test_roll_dice_schema(self):
        tool = next(t for t in DM_TOOLS if t["name"] == "roll_dice")
        props = tool["input_schema"]["properties"]
        required = tool["input_schema"]["required"]

        assert "expression" in props
        assert "reason" in props
        assert set(required) == {"expression", "reason"}

    def test_apply_damage_schema(self):
        tool = next(t for t in DM_TOOLS if t["name"] == "apply_damage")
        props = tool["input_schema"]["properties"]
        required = tool["input_schema"]["required"]

        assert "character_name" in props
        assert "amount" in props
        assert "description" in props
        assert props["amount"]["type"] == "integer"
        assert set(required) == {"character_name", "amount", "description"}

    def test_heal_schema(self):
        tool = next(t for t in DM_TOOLS if t["name"] == "heal")
        props = tool["input_schema"]["properties"]
        required = tool["input_schema"]["required"]

        assert "character_name" in props
        assert "amount" in props
        assert "description" in props
        assert set(required) == {"character_name", "amount", "description"}

    def test_get_party_status_no_required(self):
        tool = next(t for t in DM_TOOLS if t["name"] == "get_party_status")
        props = tool["input_schema"]["properties"]
        assert props == {}
        assert "required" not in tool["input_schema"]

    def test_enter_room_schema(self):
        tool = next(t for t in DM_TOOLS if t["name"] == "enter_room")
        props = tool["input_schema"]["properties"]
        required = tool["input_schema"]["required"]

        assert "room_name" in props
        assert set(required) == {"room_name"}

    def test_request_player_input_schema(self):
        tool = next(t for t in DM_TOOLS if t["name"] == "request_player_input")
        props = tool["input_schema"]["properties"]
        required = tool["input_schema"]["required"]

        assert "player_names" in props
        assert props["player_names"]["type"] == "array"
        assert props["player_names"]["items"]["type"] == "string"
        assert set(required) == {"player_names"}

    def test_end_session_schema(self):
        tool = next(t for t in DM_TOOLS if t["name"] == "end_session")
        props = tool["input_schema"]["properties"]
        required = tool["input_schema"]["required"]

        assert "reason" in props
        assert set(required) == {"reason"}

    def test_all_descriptions_non_empty(self):
        for tool in DM_TOOLS:
            assert len(tool["description"]) > 10, (
                f"Tool '{tool['name']}' description too short"
            )
