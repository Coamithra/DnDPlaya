"""Tests for OllamaProvider — message/tool translation, validation, retry."""
from __future__ import annotations

from dndplaya.agents.provider import (
    OllamaProvider,
    ToolCall,
)


# ── Tool format translation ────────────────────────────────────────

class TestToolTranslation:
    def test_anthropic_to_openai_tool_format(self):
        tools = [
            {
                "name": "attack",
                "description": "Attack a target",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "string"},
                    },
                    "required": ["target"],
                },
            }
        ]
        result = OllamaProvider._translate_tools(tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "attack"
        assert result[0]["function"]["description"] == "Attack a target"
        assert result[0]["function"]["parameters"]["required"] == ["target"]

    def test_multiple_tools(self):
        tools = [
            {"name": "a", "description": "first", "input_schema": {"type": "object", "properties": {}}},
            {"name": "b", "description": "second", "input_schema": {"type": "object", "properties": {}}},
        ]
        result = OllamaProvider._translate_tools(tools)
        assert len(result) == 2
        assert result[0]["function"]["name"] == "a"
        assert result[1]["function"]["name"] == "b"


# ── Message format translation ──────────────────────────────────────

class TestMessageTranslation:
    def test_system_prompt_string(self):
        msgs = OllamaProvider._translate_messages([], "You are a DM.")
        assert msgs[0] == {"role": "system", "content": "You are a DM."}

    def test_system_prompt_list(self):
        system = [
            {"type": "text", "text": "Part 1"},
            {"type": "text", "text": "Part 2", "cache_control": {"type": "ephemeral"}},
        ]
        msgs = OllamaProvider._translate_messages([], system)
        assert msgs[0]["role"] == "system"
        assert "Part 1" in msgs[0]["content"]
        assert "Part 2" in msgs[0]["content"]

    def test_simple_user_message(self):
        messages = [{"role": "user", "content": "Hello"}]
        result = OllamaProvider._translate_messages(messages, "sys")
        assert result[1] == {"role": "user", "content": "Hello"}

    def test_simple_assistant_message(self):
        messages = [{"role": "assistant", "content": "Hi there"}]
        result = OllamaProvider._translate_messages(messages, "sys")
        assert result[1] == {"role": "assistant", "content": "Hi there"}

    def test_user_message_with_cache_control_stripped(self):
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": "cached msg", "cache_control": {"type": "ephemeral"}}],
        }]
        result = OllamaProvider._translate_messages(messages, "sys")
        # Should produce a user message with the text content
        user_msg = result[1]
        assert user_msg["role"] == "user"

    def test_tool_result_messages(self):
        messages = [{
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "Roll: 15"},
                {"type": "tool_result", "tool_use_id": "toolu_2", "content": "Hit!"},
            ],
        }]
        result = OllamaProvider._translate_messages(messages, "sys")
        # Should produce two tool messages
        tool_msgs = [m for m in result if m["role"] == "tool"]
        assert len(tool_msgs) == 2
        assert tool_msgs[0]["tool_call_id"] == "toolu_1"
        assert tool_msgs[1]["tool_call_id"] == "toolu_2"

    def test_assistant_message_with_tool_use(self):
        messages = [{
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me attack."},
                {"type": "tool_use", "id": "toolu_1", "name": "attack", "input": {"target": "goblin"}},
            ],
        }]
        result = OllamaProvider._translate_messages(messages, "sys")
        asst = result[1]
        assert asst["role"] == "assistant"
        assert "Let me attack" in (asst.get("content") or "")
        assert len(asst["tool_calls"]) == 1
        assert asst["tool_calls"][0]["function"]["name"] == "attack"
        # Native Ollama API uses dict arguments (not JSON string)
        assert asst["tool_calls"][0]["function"]["arguments"] == {"target": "goblin"}

    def test_thinking_blocks_stripped(self):
        messages = [{
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "hmm", "signature": "sig"},
                {"type": "text", "text": "I act."},
            ],
        }]
        result = OllamaProvider._translate_messages(messages, "sys")
        asst = result[1]
        assert "thinking" not in str(asst).lower() or "I act" in (asst.get("content") or "")

    def test_image_blocks(self):
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Here is a map"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "abc123"}},
            ],
        }]
        result = OllamaProvider._translate_messages(messages, "sys")
        user_msg = result[1]
        assert user_msg["role"] == "user"
        content = user_msg["content"]
        assert isinstance(content, list)
        assert any(p.get("type") == "image_url" for p in content)
        img_part = [p for p in content if p.get("type") == "image_url"][0]
        assert "data:image/png;base64,abc123" in img_part["image_url"]["url"]


# ── Tool validation ─────────────────────────────────────────────────

class TestToolValidation:
    TOOLS = [
        {
            "name": "attack",
            "description": "Attack a target",
            "input_schema": {
                "type": "object",
                "properties": {
                    "target": {"type": "string"},
                    "weapon": {"type": "string", "enum": ["sword", "bow"]},
                },
                "required": ["target"],
            },
        },
        {
            "name": "heal",
            "description": "Heal a target",
            "input_schema": {
                "type": "object",
                "properties": {"target": {"type": "string"}},
                "required": ["target"],
            },
        },
    ]

    def test_valid_tool_call(self):
        tc = [ToolCall(id="1", name="attack", arguments={"target": "goblin", "weapon": "sword"})]
        errors = OllamaProvider._validate_tool_calls(tc, self.TOOLS)
        assert errors == []

    def test_unknown_tool(self):
        tc = [ToolCall(id="1", name="fireball", arguments={})]
        errors = OllamaProvider._validate_tool_calls(tc, self.TOOLS)
        assert len(errors) == 1
        assert "Unknown tool" in errors[0]
        assert "fireball" in errors[0]

    def test_missing_required_param(self):
        tc = [ToolCall(id="1", name="attack", arguments={"weapon": "sword"})]
        errors = OllamaProvider._validate_tool_calls(tc, self.TOOLS)
        assert len(errors) == 1
        assert "missing required param 'target'" in errors[0]

    def test_invalid_enum_value(self):
        tc = [ToolCall(id="1", name="attack", arguments={"target": "goblin", "weapon": "catapult"})]
        errors = OllamaProvider._validate_tool_calls(tc, self.TOOLS)
        assert len(errors) == 1
        assert "not in enum" in errors[0]

    def test_multiple_errors(self):
        tc = [
            ToolCall(id="1", name="attack", arguments={"weapon": "catapult"}),  # missing + bad enum
        ]
        errors = OllamaProvider._validate_tool_calls(tc, self.TOOLS)
        assert len(errors) == 2  # missing target + bad enum

    def test_valid_minimal(self):
        tc = [ToolCall(id="1", name="attack", arguments={"target": "goblin"})]
        errors = OllamaProvider._validate_tool_calls(tc, self.TOOLS)
        assert errors == []


# ── Provider config ─────────────────────────────────────────────────

# ── Text-based tool call extraction ──────────────────────────────────

class TestTextToolCallExtraction:
    """Test extraction of tool calls written as text by local models."""

    TOOLS = [
        {
            "name": "narrate",
            "description": "Narrate to players",
            "input_schema": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
        {
            "name": "request_group_input",
            "description": "Ask all players what they do",
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
        {
            "name": "ask_skill_check",
            "description": "Ask for a skill check",
            "input_schema": {
                "type": "object",
                "properties": {
                    "player": {"type": "string"},
                    "skill": {"type": "string"},
                    "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
                },
                "required": ["player", "skill", "difficulty"],
            },
        },
        {
            "name": "attack",
            "description": "Monster attacks a player",
            "input_schema": {
                "type": "object",
                "properties": {
                    "attacker": {"type": "string"},
                    "target": {"type": "string"},
                },
                "required": ["attacker", "target"],
            },
        },
    ]

    def test_extract_narrate(self):
        text = 'The DM looks around.\n\nnarrate("You enter a dark cave.")'
        result = OllamaProvider._extract_text_tool_calls(text, self.TOOLS)
        assert len(result) == 1
        assert result[0].name == "narrate"
        assert result[0].arguments["text"] == "You enter a dark cave."

    def test_extract_no_args(self):
        text = "Let me ask everyone.\n\nrequest_group_input()"
        result = OllamaProvider._extract_text_tool_calls(text, self.TOOLS)
        assert len(result) == 1
        assert result[0].name == "request_group_input"

    def test_extract_multiple_calls(self):
        text = (
            'narrate("You see a goblin.")\n\n'
            'request_group_input()'
        )
        result = OllamaProvider._extract_text_tool_calls(text, self.TOOLS)
        assert len(result) == 2
        assert result[0].name == "narrate"
        assert result[1].name == "request_group_input"

    def test_extract_positional_args(self):
        text = 'ask_skill_check("Thorin", "perception", "medium")'
        result = OllamaProvider._extract_text_tool_calls(text, self.TOOLS)
        assert len(result) == 1
        assert result[0].arguments["player"] == "Thorin"
        assert result[0].arguments["skill"] == "perception"
        assert result[0].arguments["difficulty"] == "medium"

    def test_extract_named_args(self):
        text = 'attack(attacker="Goblin", target="Thorin")'
        result = OllamaProvider._extract_text_tool_calls(text, self.TOOLS)
        assert len(result) == 1
        assert result[0].arguments["attacker"] == "Goblin"
        assert result[0].arguments["target"] == "Thorin"

    def test_no_match_for_unknown_tool(self):
        text = 'fireball("everyone")'
        result = OllamaProvider._extract_text_tool_calls(text, self.TOOLS)
        assert result == []

    def test_no_match_in_plain_text(self):
        text = "The party walks into the cave. Nothing happens."
        result = OllamaProvider._extract_text_tool_calls(text, self.TOOLS)
        assert result == []

    def test_extract_generates_unique_ids(self):
        text = 'narrate("a")\nnarrate("b")'
        result = OllamaProvider._extract_text_tool_calls(text, self.TOOLS)
        assert len(result) == 2
        assert result[0].id != result[1].id
        assert result[0].id.startswith("text_")

    def test_extract_empty_string_arg(self):
        text = 'request_group_input("")'
        result = OllamaProvider._extract_text_tool_calls(text, self.TOOLS)
        assert len(result) == 1
        assert result[0].name == "request_group_input"

    def test_extract_json_tool_call(self):
        """Extract tool calls from JSON-in-text (Qwen's common format)."""
        text = 'ystatechange\n{"name": "attack", "arguments": {"attacker": "Goblin", "target": "Thorin"}}\n</tool_call>'
        result = OllamaProvider._extract_text_tool_calls(text, self.TOOLS)
        assert len(result) == 1
        assert result[0].name == "attack"
        assert result[0].arguments["attacker"] == "Goblin"
        assert result[0].arguments["target"] == "Thorin"

    def test_extract_json_say_with_urgency(self):
        """Extract say tool from JSON with urgency (player pattern)."""
        tools = [
            {
                "name": "say",
                "description": "Say something",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "urgency": {"type": "integer"},
                    },
                    "required": ["text", "urgency"],
                },
            },
            {
                "name": "pass_turn",
                "description": "Pass",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
        ]
        text = '.EditorButton\n{"name": "say", "arguments": {"text": "Let us proceed!", "urgency": 3}}\n</tool_call>'
        result = OllamaProvider._extract_text_tool_calls(text, tools)
        assert len(result) == 1
        assert result[0].name == "say"
        assert result[0].arguments["text"] == "Let us proceed!"
        assert result[0].arguments["urgency"] == 3

    def test_extract_json_preferred_over_function_style(self):
        """JSON extraction takes priority over function-call style."""
        text = 'narrate("old style")\n{"name": "narrate", "arguments": {"text": "json style"}}'
        result = OllamaProvider._extract_text_tool_calls(text, self.TOOLS)
        # JSON found, so should return JSON result only
        assert len(result) == 1
        assert result[0].arguments["text"] == "json style"


class TestOllamaProviderConfig:
    def test_basic_config(self):
        """OllamaProvider picks up model and URL from settings."""
        from pydantic import SecretStr
        from dndplaya.config import Settings
        settings = Settings(
            anthropic_api_key=SecretStr(""),
            provider="ollama",
            ollama_model="qwen2.5:14b",
            ollama_url="http://localhost:11434",
        )
        provider = OllamaProvider(settings)
        assert provider.model == "qwen2.5:14b"

    def test_num_ctx_in_guardrails(self):
        """context_window in guardrails matches ollama_num_ctx."""
        from pydantic import SecretStr
        from dndplaya.config import Settings
        settings = Settings(
            anthropic_api_key=SecretStr(""),
            provider="ollama",
            ollama_model="qwen2.5:14b",
            ollama_url="http://localhost:11434",
            ollama_num_ctx=16384,
        )
        provider = OllamaProvider(settings)
        assert provider.guardrails.context_window == 16384
        assert provider.guardrails.compaction_threshold == int(16384 * 0.75)

    def test_default_num_ctx(self):
        """Default num_ctx is 32768."""
        from pydantic import SecretStr
        from dndplaya.config import Settings
        settings = Settings(
            anthropic_api_key=SecretStr(""),
            provider="ollama",
            ollama_model="qwen2.5:14b",
            ollama_url="http://localhost:11434",
        )
        provider = OllamaProvider(settings)
        assert provider._num_ctx == 32768
        assert provider.guardrails.context_window == 32768
