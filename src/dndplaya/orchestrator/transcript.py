from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class TranscriptEntry:
    """A single entry in the session transcript."""
    speaker: str  # "DM", player name, or "System"
    content: str
    entry_type: str  # "narration", "action", "combat", "system"
    round_number: int | None = None
    room: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class SessionTranscript:
    """Records the full session for narrative generation."""

    def __init__(self, log_path: Path | None = None):
        self.entries: list[TranscriptEntry] = []
        self.current_room: str | None = None
        self._log_path = log_path

    def _append(self, entry: TranscriptEntry) -> None:
        """Add entry to list and flush to log file if configured."""
        self.entries.append(entry)
        if self._log_path:
            self._flush(entry)

    def _flush(self, entry: TranscriptEntry) -> None:
        """Append a single entry to the live log file."""
        content = entry.content
        if entry.entry_type == "narration":
            line = f"---\n\n### DM\n\n{content}\n\n"
        elif entry.entry_type == "action":
            line = f"---\n\n### {entry.speaker} ✦\n\n{content}\n\n"
        elif entry.entry_type == "discarded":
            line = f"\n*{entry.speaker} (not selected):* {content}\n\n"
        elif entry.entry_type == "combat":
            line = f"\n> ⚔ {content}\n\n"
        elif content.startswith("MODULE SUMMARY:") or content.startswith("PARTY:"):
            line = f"\n## {content}\n\n"
        elif content.startswith("---"):
            # Group input markers — make them visible
            line = f"\n---\n\n**{content.strip('- ')}**\n\n"
        elif content.startswith("DM internal:"):
            line = f"\n*{content}*\n\n"
        else:
            line = f"\n`{content}`\n\n"
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(line)

    def add_dm_narration(self, content: str, round_number: int | None = None) -> None:
        self._append(TranscriptEntry(
            speaker="DM",
            content=content,
            entry_type="narration",
            round_number=round_number,
            room=self.current_room,
        ))

    def add_player_action(self, player_name: str, content: str, round_number: int | None = None) -> None:
        self._append(TranscriptEntry(
            speaker=player_name,
            content=content,
            entry_type="action",
            round_number=round_number,
            room=self.current_room,
        ))

    def add_combat_result(self, content: str, round_number: int) -> None:
        self._append(TranscriptEntry(
            speaker="System",
            content=content,
            entry_type="combat",
            round_number=round_number,
            room=self.current_room,
        ))

    def add_discarded_response(self, player_name: str, content: str, urgency: int) -> None:
        """Log a player response that wasn't selected (for analysis)."""
        self._append(TranscriptEntry(
            speaker=player_name,
            content=content,
            entry_type="discarded",
            room=self.current_room,
        ))

    def add_system_event(self, content: str) -> None:
        self._append(TranscriptEntry(
            speaker="System",
            content=content,
            entry_type="system",
            room=self.current_room,
        ))

    def set_room(self, room_name: str) -> None:
        self.current_room = room_name

    def to_text(self) -> str:
        """Convert transcript to readable text format."""
        lines = []
        current_room = None

        for entry in self.entries:
            if entry.room and entry.room != current_room:
                current_room = entry.room
                lines.append(f"\n--- {current_room} ---\n")

            if entry.entry_type == "narration":
                lines.append(f"**DM:** {entry.content}\n")
            elif entry.entry_type == "action":
                lines.append(f"**{entry.speaker}:** {entry.content}\n")
            elif entry.entry_type == "discarded":
                lines.append(f"~~**{entry.speaker}:**~~ *(not selected)* {entry.content}\n")
            elif entry.entry_type == "combat":
                lines.append(f"*[Combat: {entry.content}]*\n")
            elif entry.entry_type == "system":
                lines.append(f"*[{entry.content}]*\n")

        return "\n".join(lines)

    def get_recent_dm_narration(self, max_entries: int = 5) -> str:
        """Get recent DM narration + system events for player context."""
        recent = []
        for entry in reversed(self.entries):
            if entry.entry_type in ("narration", "system", "combat"):
                recent.append(entry.content)
            if len(recent) >= max_entries:
                break
        recent.reverse()
        return "\n\n".join(recent) if recent else "The adventure begins."

    def get_summary(self) -> str:
        """Get a brief summary of the session."""
        rooms = set(e.room for e in self.entries if e.room)
        combat_entries = [e for e in self.entries if e.entry_type == "combat"]
        return (
            f"Session: {len(self.entries)} entries, "
            f"{len(rooms)} rooms visited, "
            f"{len(combat_entries)} combat events"
        )
