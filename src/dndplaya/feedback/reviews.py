from __future__ import annotations

import logging

from ..config import Settings
from ..agents.critic import CriticAgent
from ..agents.dm import DMAgent
from ..agents.player import PlayerAgent

logger = logging.getLogger(__name__)


def generate_all_reviews(
    dm: DMAgent,
    players: list[PlayerAgent],
    transcript_text: str,
    settings: Settings,
) -> dict[str, str]:
    """Generate reviews from all agents (DM + 4 players).

    Returns dict of agent_name -> review_text.
    Continues generating remaining reviews if one fails.
    """
    critic = CriticAgent(settings)
    reviews = {}

    # DM review
    try:
        reviews["DM"] = critic.generate_dm_review(
            transcript=transcript_text,
            runnability_notes=dm.runnability_notes,
        )
    except Exception as e:
        logger.warning("DM review generation failed: %s", e)
        reviews["DM"] = f"[Review generation failed: {e}]"

    # Player reviews
    for player in players:
        try:
            reviews[player.name] = critic.generate_player_review(
                player_name=player.name,
                archetype=player.archetype,
                transcript=transcript_text,
                engagement_notes=player.engagement_notes,
            )
        except Exception as e:
            logger.warning("Review generation failed for %s: %s", player.name, e)
            reviews[player.name] = f"[Review generation failed: {e}]"

    return reviews
