from __future__ import annotations

from ..config import Settings
from ..agents.critic import CriticAgent
from ..agents.dm import DMAgent
from ..agents.player import PlayerAgent


def generate_all_reviews(
    dm: DMAgent,
    players: list[PlayerAgent],
    transcript_text: str,
    settings: Settings,
) -> dict[str, str]:
    """Generate reviews from all agents (DM + 4 players).

    Returns dict of agent_name -> review_text.
    """
    critic = CriticAgent(settings)
    reviews = {}

    # DM review
    reviews["DM"] = critic.generate_dm_review(
        transcript=transcript_text,
        runnability_notes=dm.runnability_notes,
    )

    # Player reviews
    for player in players:
        reviews[player.name] = critic.generate_player_review(
            player_name=player.name,
            archetype=player.archetype,
            transcript=transcript_text,
            engagement_notes=player.engagement_notes,
        )

    return reviews
