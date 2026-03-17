"""Tool definitions for player agents in Anthropic SDK format."""
from __future__ import annotations

PLAYER_TOOLS = [
    {
        "name": "say",
        "description": (
            "Speak at the table — this is what you say out loud to the DM "
            "and other players. ALL in-character speech and action declarations "
            "MUST go through this tool. Keep it brief: 1-3 sentences."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "What you say or do, e.g. 'I search the altar for traps'",
                },
                "urgency": {
                    "type": "integer",
                    "description": (
                        "How urgently you need to speak, 1-5. "
                        "5=must speak now, 4=important, 3=relevant, "
                        "2=minor, 1=not important"
                    ),
                    "minimum": 1,
                    "maximum": 5,
                },
            },
            "required": ["text", "urgency"],
        },
    },
    {
        "name": "pass_turn",
        "description": (
            "Pass — you have nothing to add right now. "
            "This is always a valid and good choice. Use it liberally."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "review_note",
        "description": (
            "Record a private note for your post-session review. "
            "Not shared with the DM or other players. Use this to note "
            "moments that engaged or frustrated you through your archetype's lens."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Your observation or note",
                },
            },
            "required": ["text"],
        },
    },
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
