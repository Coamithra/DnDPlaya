"""Tool definitions for player agents in Anthropic SDK format."""
from __future__ import annotations

PLAYER_TOOLS = [
    {
        "name": "attack",
        "description": (
            "Attack a monster. The orchestrator resolves the attack roll and damage. "
            "Use this when you want to make a melee or ranged attack against an enemy."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "Name of the monster to attack",
                },
            },
            "required": ["target"],
        },
    },
    {
        "name": "heal",
        "description": (
            "Heal a party member. Costs a spell slot. "
            "The orchestrator resolves the healing amount."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "Name of the PC to heal",
                },
            },
            "required": ["target"],
        },
    },
]
