"""Tests for agents/context.py — history compaction logic."""
from __future__ import annotations

from unittest.mock import MagicMock
from dndplaya.agents.base import Message
from dndplaya.agents.context import compact_history, estimate_tokens, estimate_history_tokens


def _make_agent(history: list[Message], last_input_tokens: int = 0):
    """Create a mock agent with the given history and token counts."""
    agent = MagicMock()
    agent.history = list(history)
    agent.last_input_tokens = last_input_tokens
    agent.total_input_tokens = last_input_tokens * 10  # cumulative should NOT be used
    agent.name = "TestAgent"
    return agent


def test_no_compaction_when_under_budget():
    history = [
        Message(role="user", content="hello"),
        Message(role="assistant", content="hi"),
    ]
    agent = _make_agent(history, last_input_tokens=100)
    compact_history(agent, max_tokens=150000)
    assert len(agent.history) == 2


def test_no_compaction_when_history_too_short():
    """Even if tokens are high, don't compact if <= keep_recent messages."""
    history = [
        Message(role="user", content="hello"),
        Message(role="assistant", content="hi"),
    ] * 5  # 10 messages = keep_recent
    agent = _make_agent(history, last_input_tokens=999999)
    compact_history(agent, max_tokens=1000)
    assert len(agent.history) == 10


def test_compaction_uses_last_input_tokens_not_cumulative():
    """Verify we use last_input_tokens (per-call) not total_input_tokens (cumulative)."""
    history = [
        Message(role="user", content="msg 0"),
        Message(role="assistant", content="reply 0"),
    ]
    # 2 messages, last_input_tokens=100 (under budget) — should NOT compact
    # even though total_input_tokens is 100000 (over budget if we used cumulative)
    agent = _make_agent(history, last_input_tokens=100)
    agent.total_input_tokens = 100000  # this should be ignored
    compact_history(agent, max_tokens=150000)
    assert len(agent.history) == 2


def test_compaction_reduces_history():
    """When over budget, old messages should be compacted."""
    history = []
    for i in range(20):
        history.append(Message(role="user", content=f"message {i}"))
        history.append(Message(role="assistant", content=f"response {i}"))
    # 40 messages total, last_input_tokens exceeds threshold
    agent = _make_agent(history, last_input_tokens=160000)
    compact_history(agent, max_tokens=150000)
    # Should keep 10 recent + 1 summary + possibly 1 synthetic ack
    assert len(agent.history) <= 12
    assert agent.history[0].role == "user"
    assert "[CONVERSATION HISTORY SUMMARY]" in agent.history[0].content


def test_compaction_maintains_alternating_roles():
    """After compaction, messages should alternate user/assistant."""
    history = []
    for i in range(20):
        history.append(Message(role="user", content=f"msg {i}"))
        history.append(Message(role="assistant", content=f"reply {i}"))
    agent = _make_agent(history, last_input_tokens=160000)
    compact_history(agent, max_tokens=150000)
    for i in range(len(agent.history) - 1):
        if agent.history[i].role == agent.history[i + 1].role:
            # Only allowed if it's the synthetic ack
            assert agent.history[i + 1].content == "[Acknowledged history summary.]"


def test_compaction_falls_back_to_estimate():
    """When last_input_tokens is 0, use character-based estimate."""
    history = []
    for i in range(20):
        history.append(Message(role="user", content="x" * 1000))
        history.append(Message(role="assistant", content="y" * 1000))
    # 40 messages * 1000 chars each / 4 chars_per_token = 10000 tokens
    agent = _make_agent(history, last_input_tokens=0)
    compact_history(agent, max_tokens=5000)
    assert len(agent.history) < 40


def test_estimate_tokens():
    assert estimate_tokens("1234") == 1
    assert estimate_tokens("12345678") == 2
    assert estimate_tokens("") == 0


def test_estimate_history_tokens():
    history = [Message(role="user", content="1234"), Message(role="assistant", content="5678")]
    assert estimate_history_tokens(history) == 2
