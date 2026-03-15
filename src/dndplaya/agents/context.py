from __future__ import annotations

from .base import BaseAgent, Message


# Rough token estimation: ~4 chars per token (fallback only)
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def estimate_history_tokens(history: list[Message]) -> int:
    return sum(estimate_tokens(m.content) for m in history)


def compact_history(agent: BaseAgent, max_tokens: int = 50000) -> None:
    """Compact an agent's history if it exceeds the token budget.

    Uses the agent's actual tracked input token usage when available,
    falling back to character-based estimation.

    Strategy: Summarize the oldest messages into a single summary message,
    keeping the most recent messages intact.
    """
    # Prefer actual token count from API responses when available
    if agent.total_input_tokens > 0:
        current_tokens = agent.total_input_tokens
    else:
        current_tokens = estimate_history_tokens(agent.history)

    if current_tokens <= max_tokens:
        return

    # Keep last 6 messages (3 exchanges), summarize the rest
    keep_recent = 6
    if len(agent.history) <= keep_recent:
        return

    old_messages = agent.history[:-keep_recent]
    recent_messages = list(agent.history[-keep_recent:])

    # Create a summary of old messages
    summary_parts = []
    for msg in old_messages:
        role_label = "User/System" if msg.role == "user" else agent.name
        # Truncate individual messages
        content = msg.content[:300] + "..." if len(msg.content) > 300 else msg.content
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
