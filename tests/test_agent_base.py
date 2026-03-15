"""Tests for agents/base.py — BaseAgent with mocked API."""
from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock
import pytest

from dndplaya.agents.base import BaseAgent, Message
from dndplaya.config import Settings


def _make_settings() -> Settings:
    """Create test settings with a dummy API key."""
    from pydantic import SecretStr
    return Settings(anthropic_api_key=SecretStr("test-key-123"), max_tokens=512)


def _mock_response(text: str = "Hello!", input_tokens: int = 10, output_tokens: int = 5):
    """Create a mock API response."""
    from anthropic.types import TextBlock, Usage
    response = MagicMock()
    response.content = [TextBlock(type="text", text=text)]
    response.usage = Usage(input_tokens=input_tokens, output_tokens=output_tokens)
    return response


class TestBaseAgentSend:
    def test_send_returns_text(self):
        settings = _make_settings()
        agent = BaseAgent("Test", "You are a test.", settings)
        with patch.object(agent.client.messages, "create", return_value=_mock_response("Hi there")):
            result = agent.send("Hello")
        assert result == "Hi there"

    def test_send_tracks_tokens(self):
        settings = _make_settings()
        agent = BaseAgent("Test", "Sys", settings)
        with patch.object(agent.client.messages, "create", return_value=_mock_response(input_tokens=50, output_tokens=20)):
            agent.send("msg1")
        assert agent.total_input_tokens == 50
        assert agent.total_output_tokens == 20
        assert agent.last_input_tokens == 50

    def test_send_accumulates_history(self):
        settings = _make_settings()
        agent = BaseAgent("Test", "Sys", settings)
        with patch.object(agent.client.messages, "create", return_value=_mock_response("r1")):
            agent.send("msg1")
        with patch.object(agent.client.messages, "create", return_value=_mock_response("r2")):
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
        import anthropic
        with patch.object(agent.client.messages, "create", side_effect=anthropic.APIConnectionError(request=MagicMock())):
            with pytest.raises(anthropic.APIConnectionError):
                agent.send("this should fail")
        # History should be empty — the message was NOT committed
        assert len(agent.history) == 0

    def test_send_raises_on_empty_response(self):
        settings = _make_settings()
        agent = BaseAgent("Test", "Sys", settings)
        response = MagicMock()
        response.content = []
        with patch.object(agent.client.messages, "create", return_value=response):
            with pytest.raises(ValueError, match="Empty response"):
                agent.send("hello")

    def test_send_raises_on_non_textblock(self):
        settings = _make_settings()
        agent = BaseAgent("Test", "Sys", settings)
        response = MagicMock()
        block = MagicMock()
        block.__class__.__name__ = "ToolUseBlock"
        type(block).text = PropertyMock(side_effect=AttributeError)
        response.content = [block]
        with patch.object(agent.client.messages, "create", return_value=response):
            with pytest.raises(TypeError, match="Expected TextBlock"):
                agent.send("hello")


class TestBaseAgentReset:
    def test_reset_clears_state(self):
        settings = _make_settings()
        agent = BaseAgent("Test", "Sys", settings)
        with patch.object(agent.client.messages, "create", return_value=_mock_response()):
            agent.send("msg")
        agent.reset()
        assert len(agent.history) == 0
        assert agent.total_input_tokens == 0
        assert agent.total_output_tokens == 0
        assert agent.last_input_tokens == 0
