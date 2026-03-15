from __future__ import annotations

from ..agents.player import PlayerAgent
from ..agents.dm import DMAgent
from ..agents.context import compact_history


class TurnManager:
    """Manages turn order and collecting player actions."""

    def __init__(self, dm: DMAgent, players: list[PlayerAgent]):
        self.dm = dm
        self.players = players
        self.round_number = 0

    def get_dm_description(self, prompt: str) -> str:
        """Get a description/narration from the DM."""
        compact_history(self.dm)
        return self.dm.send(prompt)

    def get_player_actions(self, dm_message: str) -> dict[str, str]:
        """Collect actions from all players in response to DM's message.

        Returns dict of player_name -> action_text.
        """
        actions = {}
        for player in self.players:
            compact_history(player)
            action = player.send(dm_message)
            actions[player.name] = action
        return actions

    def next_round(self) -> int:
        self.round_number += 1
        return self.round_number

    def get_token_summary(self) -> dict[str, dict[str, int]]:
        """Get token usage for all agents."""
        summary = {"DM": self.dm.get_token_usage()}
        for player in self.players:
            summary[player.name] = player.get_token_usage()
        return summary
