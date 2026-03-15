from __future__ import annotations

from .base import BaseAgent, Message


# Rough token estimation: ~4 chars per token (fallback only)
CHARS_PER_TOKEN = 4


def _extract_text_from_content(content: str | list) -> str:
    """Extract text from message content (handles both str and list formats)."""
    if isinstance(content, str):
        return content
    text_parts = []
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_result":
                c = block.get("content", "")
                if isinstance(c, str):
                    text_parts.append(c)
    return "\n".join(text_parts)


def estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def estimate_history_tokens(history: list[Message]) -> int:
    return sum(estimate_tokens(_extract_text_from_content(m.content)) for m in history)


def compact_history(agent: BaseAgent, max_tokens: int = 50000) -> None:
    """Compact an agent's history if it exceeds the token budget.

    Uses the most recent API call's input token count as a proxy for
    current context window size, falling back to character-based estimation.

    Strategy: Summarize the oldest messages into a single summary message,
    keeping the most recent messages intact.
    """
    # Use the last API call's input tokens as proxy for current context size.
    # This reflects the actual conversation size, not a cumulative lifetime total.
    if agent.last_input_tokens > 0:
        current_tokens = agent.last_input_tokens
    else:
        current_tokens = estimate_history_tokens(agent.history)

    if current_tokens <= max_tokens:
        return

    # Keep last 10 messages (5 exchanges), summarize the rest
    keep_recent = 10
    if len(agent.history) <= keep_recent:
        return

    old_messages = agent.history[:-keep_recent]
    recent_messages = list(agent.history[-keep_recent:])

    # Create a summary of old messages, extracting text from any format
    summary_parts = []
    for msg in old_messages:
        role_label = "User/System" if msg.role == "user" else agent.name
        content = _extract_text_from_content(msg.content)
        content = content[:300] + "..." if len(content) > 300 else content
        if content.strip():
            summary_parts.append(f"[{role_label}]: {content}")

    summary = "[CONVERSATION HISTORY SUMMARY]\n" + "\n".join(summary_parts)

    # Ensure valid alternating user/assistant message sequence.
    # The summary is injected as a "user" message. If the first recent message
    # is also "user", insert a synthetic assistant acknowledgment between them
    # instead of merging (which would corrupt the game prompt).
    new_history = [Message(role="user", content=summary)]
    if recent_messages and recent_messages[0].role == "user":
        new_history.append(Message(role="assistant", content="[Acknowledged history summary.]"))
    new_history.extend(recent_messages)

    agent.history = new_history
