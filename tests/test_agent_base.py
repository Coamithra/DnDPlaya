"""Tests for agents/base.py — BaseAgent with mocked provider."""
from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from dndplaya.agents.base import BaseAgent, ToolCall, AgentResponse
from dndplaya.agents.provider import LLMResponse, ToolCall as ProviderToolCall
from dndplaya.config import Settings


def _make_settings() -> Settings:
    """Create test settings with a dummy API key."""
    from pydantic import SecretStr
    return Settings(anthropic_api_key=SecretStr("test-key-123"), max_tokens=512)


def _mock_response(
    text: str = "Hello!",
    input_tokens: int = 10,
    output_tokens: int = 5,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> LLMResponse:
    """Create a mock LLMResponse with text content."""
    return LLMResponse(
        text_parts=[text] if text else [],
        tool_calls=[],
        raw_content=[{"type": "text", "text": text}] if text else [],
        stop_reason="end_turn",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_tokens=cache_creation_tokens,
        cache_read_tokens=cache_read_tokens,
    )


def _mock_tool_response(
    text: str = "",
    tool_name: str = "roll_check",
    tool_id: str = "toolu_123",
    tool_input: dict | None = None,
    input_tokens: int = 10,
    output_tokens: int = 5,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> LLMResponse:
    """Create a mock LLMResponse with text + tool use."""
    text_parts = [text] if text else []
    raw_content = []
    if text:
        raw_content.append({"type": "text", "text": text})
    tool_args = tool_input or {"modifier": 5, "dc": 15, "description": "test"}
    raw_content.append({
        "type": "tool_use",
        "id": tool_id,
        "name": tool_name,
        "input": tool_args,
    })
    return LLMResponse(
        text_parts=text_parts,
        tool_calls=[ProviderToolCall(id=tool_id, name=tool_name, arguments=tool_args)],
        raw_content=raw_content,
        stop_reason="tool_use",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_tokens=cache_creation_tokens,
        cache_read_tokens=cache_read_tokens,
    )


class TestBaseAgentSend:
    def test_send_returns_text(self):
        settings = _make_settings()
        agent = BaseAgent("Test", "You are a test.", settings)
        with patch.object(agent.provider, "call", return_value=_mock_response("Hi there")):
            result = agent.send("Hello")
        assert result == "Hi there"

    def test_send_tracks_tokens(self):
        settings = _make_settings()
        agent = BaseAgent("Test", "Sys", settings)
        with patch.object(agent.provider, "call", return_value=_mock_response(input_tokens=50, output_tokens=20)):
            agent.send("msg1")
        assert agent.total_input_tokens == 50
        assert agent.total_output_tokens == 20
        assert agent.last_input_tokens == 50

    def test_send_accumulates_history(self):
        settings = _make_settings()
        agent = BaseAgent("Test", "Sys", settings)
        with patch.object(agent.provider, "call", return_value=_mock_response("r1")):
            agent.send("msg1")
        with patch.object(agent.provider, "call", return_value=_mock_response("r2")):
            agent.send("msg2")
        assert len(agent.history) == 4  # 2 user + 2 assistant
        assert agent.history[0].content == "msg1"
        assert agent.history[1].content == "r1"
        assert agent.history[2].content == "msg2"
        assert agent.history[3].content == "r2"

    def test_send_does_not_corrupt_history_on_api_failure(self):
        """If the API call fails, history should not have a dangling user message."""
        settings = _make_settings()
        agent = BaseAgent("Test", "Sys", settings)
        with patch.object(agent.provider, "call", side_effect=RuntimeError("connection failed")):
            with pytest.raises(RuntimeError):
                agent.send("this should fail")
        # History should be empty — the message was NOT committed
        assert len(agent.history) == 0

    def test_send_raises_on_empty_response(self):
        settings = _make_settings()
        agent = BaseAgent("Test", "Sys", settings)
        empty_resp = LLMResponse(text_parts=[], raw_content=[], stop_reason="end_turn")
        with patch.object(agent.provider, "call", return_value=empty_resp):
            with pytest.raises(ValueError, match="No text content"):
                agent.send("hello")

    def test_send_raises_on_no_text_content(self):
        settings = _make_settings()
        agent = BaseAgent("Test", "Sys", settings)
        no_text = LLMResponse(text_parts=[], raw_content=[], stop_reason="end_turn")
        with patch.object(agent.provider, "call", return_value=no_text):
            with pytest.raises(ValueError, match="No text content"):
                agent.send("hello")


class TestBaseAgentReset:
    def test_reset_clears_state(self):
        settings = _make_settings()
        agent = BaseAgent("Test", "Sys", settings)
        with patch.object(agent.provider, "call", return_value=_mock_response()):
            agent.send("msg")
        agent.reset()
        assert len(agent.history) == 0
        assert agent.total_input_tokens == 0
        assert agent.total_output_tokens == 0
        assert agent.last_input_tokens == 0


class TestBaseAgentToolUse:
    def test_send_with_tools_returns_agent_response(self):
        settings = _make_settings()
        tools = [{"name": "test_tool", "description": "test", "input_schema": {"type": "object", "properties": {}}}]
        agent = BaseAgent("Test", "Sys", settings, tools=tools)
        with patch.object(agent.provider, "call", return_value=_mock_tool_response("Narration", "roll_check")):
            result = agent.send_with_tools("Do something")
        assert isinstance(result, AgentResponse)
        assert result.text == "Narration"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "roll_check"
        assert result.stop_reason == "tool_use"

    def test_send_with_tools_commits_history(self):
        settings = _make_settings()
        tools = [{"name": "test_tool", "description": "test", "input_schema": {"type": "object", "properties": {}}}]
        agent = BaseAgent("Test", "Sys", settings, tools=tools)
        with patch.object(agent.provider, "call", return_value=_mock_tool_response("text")):
            agent.send_with_tools("msg")
        assert len(agent.history) == 2
        assert agent.history[0].role == "user"
        assert agent.history[0].content == "msg"
        assert agent.history[1].role == "assistant"
        assert isinstance(agent.history[1].content, list)

    def test_send_with_tools_tracks_tokens(self):
        settings = _make_settings()
        agent = BaseAgent("Test", "Sys", settings, tools=[])
        with patch.object(agent.provider, "call", return_value=_mock_tool_response(input_tokens=100, output_tokens=50)):
            agent.send_with_tools("msg")
        assert agent.total_input_tokens == 100
        assert agent.total_output_tokens == 50

    def test_submit_tool_results(self):
        settings = _make_settings()
        agent = BaseAgent("Test", "Sys", settings, tools=[])
        # First, simulate a send_with_tools that committed tool use to history
        with patch.object(agent.provider, "call", return_value=_mock_tool_response("narration", tool_id="toolu_1")):
            agent.send_with_tools("start")

        # Now submit tool results
        with patch.object(agent.provider, "call", return_value=_mock_response("Next step")):
            result = agent.submit_tool_results([("toolu_1", "Roll result: 15")])
        assert isinstance(result, AgentResponse)
        assert len(agent.history) == 4  # 2 from send_with_tools + 2 from submit

    def test_tool_call_dataclass(self):
        tc = ToolCall(id="toolu_1", name="roll_check", arguments={"dc": 15})
        assert tc.id == "toolu_1"
        assert tc.name == "roll_check"
        assert tc.arguments == {"dc": 15}

    def test_agent_response_defaults(self):
        resp = AgentResponse(text="hello")
        assert resp.text == "hello"
        assert resp.tool_calls == []
        assert resp.raw_content == []
        assert resp.stop_reason == "end_turn"

    def test_send_with_tools_text_only_response(self):
        """When API returns only text (no tools), AgentResponse should have empty tool_calls."""
        settings = _make_settings()
        agent = BaseAgent("Test", "Sys", settings, tools=[])
        with patch.object(agent.provider, "call", return_value=_mock_response("Just text")):
            result = agent.send_with_tools("msg")
        assert result.text == "Just text"
        assert result.tool_calls == []
        assert result.stop_reason == "end_turn"

    def test_send_with_tools_no_tools_configured(self):
        """send_with_tools works even when no tools are set (just acts like send)."""
        settings = _make_settings()
        agent = BaseAgent("Test", "Sys", settings)
        with patch.object(agent.provider, "call", return_value=_mock_response("response")):
            result = agent.send_with_tools("msg")
        assert result.text == "response"


class TestPromptCaching:
    def test_cache_control_string_prompt(self):
        """cache_control is added when system prompt is a string."""
        result = BaseAgent._add_cache_control("Hello system")
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "Hello system"
        assert result[0]["cache_control"] == {"type": "ephemeral"}

    def test_cache_control_list_prompt(self):
        """cache_control is added to the last block when system prompt is a list."""
        prompt = [
            {"type": "text", "text": "First block"},
            {"type": "text", "text": "Second block"},
        ]
        result = BaseAgent._add_cache_control(prompt)
        assert isinstance(result, list)
        assert len(result) == 2
        assert "cache_control" not in result[0]
        assert result[1]["cache_control"] == {"type": "ephemeral"}

    def test_cache_control_empty_list(self):
        """Empty list returns empty list."""
        result = BaseAgent._add_cache_control([])
        assert result == []

    def test_cache_control_in_api_call(self):
        """Verify provider.call receives the system prompt for caching."""
        settings = _make_settings()
        agent = BaseAgent("Test", "You are a test.", settings)
        with patch.object(agent.provider, "call", return_value=_mock_response("Hi")) as mock_call:
            agent.send("Hello")
        call_kwargs = mock_call.call_args[1]
        # The provider receives the raw system prompt — it handles caching internally
        assert call_kwargs["system"] == "You are a test."

    def test_cache_tokens_tracked_on_send(self):
        """Cache creation and read tokens are tracked via send()."""
        settings = _make_settings()
        agent = BaseAgent("Test", "Sys", settings)
        resp = _mock_response(cache_creation_tokens=500, cache_read_tokens=0)
        with patch.object(agent.provider, "call", return_value=resp):
            agent.send("msg1")
        assert agent.total_cache_creation_tokens == 500
        assert agent.total_cache_read_tokens == 0
        # Second call reads from cache
        resp2 = _mock_response(cache_creation_tokens=0, cache_read_tokens=500)
        with patch.object(agent.provider, "call", return_value=resp2):
            agent.send("msg2")
        assert agent.total_cache_creation_tokens == 500
        assert agent.total_cache_read_tokens == 500

    def test_cache_tokens_tracked_on_send_with_tools(self):
        """Cache tokens tracked via send_with_tools()."""
        settings = _make_settings()
        agent = BaseAgent("Test", "Sys", settings, tools=[])
        resp = _mock_tool_response(cache_creation_tokens=200, cache_read_tokens=100)
        with patch.object(agent.provider, "call", return_value=resp):
            agent.send_with_tools("msg")
        assert agent.total_cache_creation_tokens == 200
        assert agent.total_cache_read_tokens == 100

    def test_cache_tokens_in_get_token_usage(self):
        """get_token_usage() includes cache metrics."""
        settings = _make_settings()
        agent = BaseAgent("Test", "Sys", settings)
        resp = _mock_response(cache_creation_tokens=300, cache_read_tokens=150)
        with patch.object(agent.provider, "call", return_value=resp):
            agent.send("msg")
        usage = agent.get_token_usage()
        assert usage["cache_creation_tokens"] == 300
        assert usage["cache_read_tokens"] == 150

    def test_reset_clears_cache_tokens(self):
        """reset() zeroes out cache token counters."""
        settings = _make_settings()
        agent = BaseAgent("Test", "Sys", settings)
        agent.total_cache_creation_tokens = 100
        agent.total_cache_read_tokens = 200
        agent.reset()
        assert agent.total_cache_creation_tokens == 0
        assert agent.total_cache_read_tokens == 0
