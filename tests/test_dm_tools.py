"""Tests for DM tool definitions — validates schema correctness."""
from __future__ import annotations

from dndplaya.agents.dm_tools import DM_TOOLS, build_music_tool


EXPECTED_TOOL_NAMES = [
    "narrate",
    "review_note",
    "ask_skill_check",
    "attack",
    "change_hp",
    "roll_initiative",
    "next_combat_turn",
    "request_group_input",
    "get_party_status",
    "end_session",
    "search_module",
    "read_page",
    "next_page",
    "previous_page",
]


class TestDMTools:
    def test_all_tools_present(self):
        tool_names = [t["name"] for t in DM_TOOLS]
        assert tool_names == EXPECTED_TOOL_NAMES

    def test_tool_count(self):
        assert len(DM_TOOLS) == 14

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

    def test_ask_skill_check_schema(self):
        tool = next(t for t in DM_TOOLS if t["name"] == "ask_skill_check")
        props = tool["input_schema"]["properties"]
        required = tool["input_schema"]["required"]

        assert "player" in props
        assert "skill" in props
        assert "difficulty" in props
        assert "has_advantage" in props
        assert props["player"]["type"] == "string"
        assert props["skill"]["type"] == "string"
        assert props["difficulty"]["type"] == "string"
        assert "enum" in props["difficulty"]
        assert props["has_advantage"]["type"] == "boolean"
        assert set(required) == {"player", "skill", "difficulty"}

    def test_attack_schema(self):
        tool = next(t for t in DM_TOOLS if t["name"] == "attack")
        props = tool["input_schema"]["properties"]
        required = tool["input_schema"]["required"]

        assert "attacker" in props
        assert "target" in props
        assert props["attacker"]["type"] == "string"
        assert props["target"]["type"] == "string"
        assert set(required) == {"attacker", "target"}

    def test_change_hp_schema(self):
        tool = next(t for t in DM_TOOLS if t["name"] == "change_hp")
        props = tool["input_schema"]["properties"]
        required = tool["input_schema"]["required"]

        assert "target" in props
        assert "amount" in props
        assert "reason" in props
        assert props["amount"]["type"] == "integer"
        assert set(required) == {"target", "amount", "reason"}

    def test_roll_initiative_schema(self):
        tool = next(t for t in DM_TOOLS if t["name"] == "roll_initiative")
        props = tool["input_schema"]["properties"]
        required = tool["input_schema"]["required"]

        assert "monsters" in props
        assert props["monsters"]["type"] == "array"
        items = props["monsters"]["items"]
        assert "name" in items["properties"]
        assert "cr" in items["properties"]
        assert set(items["required"]) == {"name", "cr"}
        assert set(required) == {"monsters"}

    def test_request_group_input_no_required(self):
        tool = next(t for t in DM_TOOLS if t["name"] == "request_group_input")
        props = tool["input_schema"]["properties"]
        assert props == {}
        assert "required" not in tool["input_schema"]

    def test_get_party_status_no_required(self):
        tool = next(t for t in DM_TOOLS if t["name"] == "get_party_status")
        props = tool["input_schema"]["properties"]
        assert props == {}
        assert "required" not in tool["input_schema"]

    def test_end_session_schema(self):
        tool = next(t for t in DM_TOOLS if t["name"] == "end_session")
        props = tool["input_schema"]["properties"]
        required = tool["input_schema"]["required"]

        assert "reason" in props
        assert set(required) == {"reason"}

    def test_search_module_schema(self):
        tool = next(t for t in DM_TOOLS if t["name"] == "search_module")
        props = tool["input_schema"]["properties"]
        required = tool["input_schema"]["required"]

        assert "search_terms" in props
        assert props["search_terms"]["type"] == "string"
        assert "question" in props
        assert props["question"]["type"] == "string"
        assert set(required) == {"search_terms"}

    def test_read_page_schema(self):
        tool = next(t for t in DM_TOOLS if t["name"] == "read_page")
        props = tool["input_schema"]["properties"]
        required = tool["input_schema"]["required"]

        assert "page_number" in props
        assert props["page_number"]["type"] == "integer"
        assert "question" in props
        assert props["question"]["type"] == "string"
        assert set(required) == {"page_number"}

    def test_next_page_no_required(self):
        tool = next(t for t in DM_TOOLS if t["name"] == "next_page")
        props = tool["input_schema"]["properties"]
        assert props == {}
        assert "required" not in tool["input_schema"]

    def test_previous_page_no_required(self):
        tool = next(t for t in DM_TOOLS if t["name"] == "previous_page")
        props = tool["input_schema"]["properties"]
        assert props == {}
        assert "required" not in tool["input_schema"]

    def test_all_descriptions_non_empty(self):
        for tool in DM_TOOLS:
            assert len(tool["description"]) > 10, (
                f"Tool '{tool['name']}' description too short"
            )


class TestBuildMusicTool:
    """Test the dynamic change_music tool builder."""

    def test_basic_schema(self):
        tool = build_music_tool(["combat", "tavern", "dungeon"])
        assert tool["name"] == "change_music"
        assert "description" in tool
        schema = tool["input_schema"]
        assert schema["type"] == "object"
        assert "track" in schema["properties"]
        assert schema["required"] == ["track"]

    def test_enum_includes_tracks_and_silence(self):
        tracks = ["combat", "tavern"]
        tool = build_music_tool(tracks)
        enum = tool["input_schema"]["properties"]["track"]["enum"]
        assert enum == ["combat", "tavern", "silence"]

    def test_empty_tracks_still_has_silence(self):
        tool = build_music_tool([])
        enum = tool["input_schema"]["properties"]["track"]["enum"]
        assert enum == ["silence"]
