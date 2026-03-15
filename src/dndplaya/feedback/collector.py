from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NotableMoment:
    """A moment worth noting during play."""
    description: str
    category: str  # "engagement", "frustration", "runnability", "discovery", etc.
    room: str | None = None
    agent: str | None = None


class MomentCollector:
    """Collects notable moments during a session for review generation."""

    def __init__(self):
        self.moments: list[NotableMoment] = []

    def add(self, description: str, category: str, room: str | None = None, agent: str | None = None) -> None:
        self.moments.append(NotableMoment(
            description=description,
            category=category,
            room=room,
            agent=agent,
        ))

    def get_by_agent(self, agent: str) -> list[NotableMoment]:
        return [m for m in self.moments if m.agent == agent]

    def get_by_category(self, category: str) -> list[NotableMoment]:
        return [m for m in self.moments if m.category == category]

    def get_by_room(self, room: str) -> list[NotableMoment]:
        return [m for m in self.moments if m.room == room]

    def to_text(self) -> str:
        lines = []
        for m in self.moments:
            parts = [f"[{m.category}]"]
            if m.agent:
                parts.append(f"({m.agent})")
            if m.room:
                parts.append(f"in {m.room}")
            parts.append(m.description)
            lines.append(" ".join(parts))
        return "\n".join(lines)
