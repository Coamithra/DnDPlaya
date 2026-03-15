from __future__ import annotations

from ..config import Settings
from ..pdf.models import DungeonModule, Room
from .base import BaseAgent

DM_SYSTEM_PROMPT = '''You are an experienced D&D Dungeon Master running a dungeon module for a party of 4 adventurers.

## Your Role
- Describe rooms vividly using the module's read-aloud text when available
- Present encounters and let players declare actions
- Narrate combat outcomes based on the mechanical results you receive
- Roleplay NPCs and monsters with personality
- Keep the game moving - don't let exploration stall
- Adjudicate player actions fairly based on the module

## Module Information
{module_context}

## Current Room
{current_room}

## Adjacent Rooms
{adjacent_rooms}

## Important Guidelines
- ONLY use information from the module - don't invent rooms or encounters
- When players try creative solutions, adjudicate based on the module's spirit
- Present read-aloud text in quotes when entering new rooms
- When combat is resolved mechanically, narrate the results dramatically
- Note transitions between rooms clearly
- If the module has unclear or missing information, improvise reasonably and make a mental note

## Runnability Critique (Internal)
While running the session, silently note any issues with the module's design:
- Unclear room descriptions or missing information
- Confusing layout or transitions
- Missing monster tactics or encounter guidance
- Pacing issues (too many/few encounters, rest opportunities)
- Information you had to improvise because the module didn't provide it

Keep these notes internal - they'll be used for your review later.
When you narrate, respond naturally as a DM speaking to the players. Keep descriptions focused and atmospheric.'''


class DMAgent(BaseAgent):
    """DM agent that runs the dungeon and critiques runnability."""

    def __init__(self, settings: Settings, module: DungeonModule):
        self.module = module
        self.current_room_id: str | None = None
        self.runnability_notes: list[str] = []

        # Initial system prompt with module overview
        system = self._build_system_prompt()
        super().__init__(name="DM", system_prompt=system, settings=settings)

    def _build_system_prompt(self) -> str:
        module_context = (
            f"Title: {self.module.title}\n"
            f"Background: {self.module.background[:500] if self.module.background else 'Not provided'}\n"
            f"Introduction: {self.module.introduction[:500] if self.module.introduction else 'Not provided'}\n"
            f"Number of rooms: {len(self.module.rooms)}"
        )

        current_room = "No room entered yet."
        adjacent_rooms = "N/A"

        if self.current_room_id:
            room = self.module.get_room(self.current_room_id)
            if room:
                current_room = self._format_room(room)
                adj = self.module.get_adjacent_rooms(self.current_room_id)
                adjacent_rooms = "\n".join(self._format_room_brief(r) for r in adj) or "None"

        return DM_SYSTEM_PROMPT.format(
            module_context=module_context,
            current_room=current_room,
            adjacent_rooms=adjacent_rooms,
        )

    def enter_room(self, room_id: str) -> None:
        """Update context when entering a new room."""
        self.current_room_id = room_id
        self.system_prompt = self._build_system_prompt()

    def add_runnability_note(self, note: str) -> None:
        self.runnability_notes.append(note)

    def _format_room(self, room: Room) -> str:
        parts = [f"**{room.name}** (ID: {room.id})"]
        if room.read_aloud:
            parts.append(f"Read-aloud: \"{room.read_aloud}\"")
        parts.append(f"Description: {room.description[:800]}")
        if room.encounters:
            enc_text = []
            for enc in room.encounters:
                monsters = ", ".join(f"{m.count}x {m.name} (CR {m.cr})" for m in enc.monsters)
                enc_text.append(f"  Encounter: {monsters}")
                if enc.tactics:
                    enc_text.append(f"  Tactics: {enc.tactics}")
            parts.append("Encounters:\n" + "\n".join(enc_text))
        if room.traps:
            parts.append("Traps: " + "; ".join(t.description for t in room.traps))
        if room.treasure:
            parts.append("Treasure: " + "; ".join(t.description for t in room.treasure))
        if room.connections:
            parts.append(f"Exits: {', '.join(room.connections)}")
        return "\n".join(parts)

    def _format_room_brief(self, room: Room) -> str:
        return f"- {room.name} (ID: {room.id}): {room.description[:100]}..."
