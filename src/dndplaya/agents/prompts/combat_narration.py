"""Templates for combat narration prompts sent to the DM agent."""
from __future__ import annotations

COMBAT_START = """[COMBAT BEGINS]
The following enemies are present: {enemies}
Party status: {party_status}

Describe the start of combat and ask the players what they do."""

COMBAT_ROUND_RESULT = """[ROUND {round_number} RESULTS]
{action_results}
{pressure_signals}

Narrate these results dramatically, then ask what the players do next."""

COMBAT_END = """[COMBAT ENDS]
{outcome}
Party status after combat: {party_status}

Narrate the end of combat and describe what the party sees now."""

ROOM_ENTRY = """[ENTERING: {room_name}]
The party enters a new area. Describe it to the players using the read-aloud text.
After the description, let the players react."""

EXPLORATION_PROMPT = """The players are exploring the current room. {player_actions}
Respond to their actions based on the room description and contents.
If there's nothing more to find, suggest moving to an adjacent area."""
