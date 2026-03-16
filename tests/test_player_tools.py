"""Tests for player tool definitions — validates schema correctness."""
from __future__ import annotations

from dndplaya.agents.player_tools import PLAYER_TOOLS


EXPECTED_TOOL_NAMES = [
    "attack",
    "heal",
]


class TestPlayerTools:
    def test_all_tools_present(self):
        tool_names = [t["name"] for t in PLAYER_TOOLS]
        assert tool_names == EXPECTED_TOOL_NAMES

    def test_tool_count(self):
        assert len(PLAYER_TOOLS) == 2

    def test_all_tools_have_required_fields(self):
        for tool in PLAYER_TOOLS:
            assert "name" in tool, f"Tool missing 'name': {tool}"
            assert "description" in tool, f"Tool missing 'description': {tool}"
            assert "input_schema" in tool, f"Tool missing 'input_schema': {tool}"

    def test_all_schemas_have_type_object(self):
        for tool in PLAYER_TOOLS:
            schema = tool["input_schema"]
            assert schema["type"] == "object", (
                f"Tool '{tool['name']}' schema type should be 'object'"
            )
            assert "properties" in schema, (
                f"Tool '{tool['name']}' schema missing 'properties'"
            )

    def test_attack_schema(self):
        tool = next(t for t in PLAYER_TOOLS if t["name"] == "attack")
        props = tool["input_schema"]["properties"]
        required = tool["input_schema"]["required"]

        assert "target" in props
        assert props["target"]["type"] == "string"
        assert set(required) == {"target"}

    def test_heal_schema(self):
        tool = next(t for t in PLAYER_TOOLS if t["name"] == "heal")
        props = tool["input_schema"]["properties"]
        required = tool["input_schema"]["required"]

        assert "target" in props
        assert props["target"]["type"] == "string"
        assert set(required) == {"target"}

    def test_all_descriptions_non_empty(self):
        for tool in PLAYER_TOOLS:
            assert len(tool["description"]) > 10, (
                f"Tool '{tool['name']}' description too short"
            )
