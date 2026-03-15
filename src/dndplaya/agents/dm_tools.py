"""Tool definitions for the DM agent in Anthropic SDK format."""
from __future__ import annotations

DM_TOOLS = [
    {
        "name": "roll_check",
        "description": (
            "Roll a d20 + modifier against a DC. "
            "Use for attack rolls, saving throws, and ability checks."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "modifier": {
                    "type": "integer",
                    "description": "The modifier to add to the d20 roll",
                },
                "dc": {
                    "type": "integer",
                    "description": "The difficulty class to beat",
                },
                "description": {
                    "type": "string",
                    "description": "What this check is for (e.g., 'Thorin attacks the goblin')",
                },
            },
            "required": ["modifier", "dc", "description"],
        },
    },
    {
        "name": "roll_dice",
        "description": (
            "Roll a dice expression like '2d6+3'. "
            "Use for damage rolls, random effects, and other variable results."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Dice expression (e.g., '2d6+3', '1d8', '3d6')",
                },
                "reason": {
                    "type": "string",
                    "description": "Why this roll is being made",
                },
            },
            "required": ["expression", "reason"],
        },
    },
    {
        "name": "apply_damage",
        "description": "Apply damage to a player character. Returns their updated HP.",
        "input_schema": {
            "type": "object",
            "properties": {
                "character_name": {
                    "type": "string",
                    "description": "Name of the PC to damage",
                },
                "amount": {
                    "type": "integer",
                    "description": "Amount of damage to deal",
                },
                "description": {
                    "type": "string",
                    "description": "What caused the damage",
                },
            },
            "required": ["character_name", "amount", "description"],
        },
    },
    {
        "name": "heal",
        "description": (
            "Heal a player character. HP is capped at their maximum. "
            "Returns updated HP."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "character_name": {
                    "type": "string",
                    "description": "Name of the PC to heal",
                },
                "amount": {
                    "type": "integer",
                    "description": "Amount of HP to restore",
                },
                "description": {
                    "type": "string",
                    "description": "What caused the healing",
                },
            },
            "required": ["character_name", "amount", "description"],
        },
    },
    {
        "name": "get_party_status",
        "description": (
            "Get current stats for all party members: "
            "HP, AC, attack bonus, spell slots."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "enter_room",
        "description": (
            "Signal that the party is entering a new room/area. "
            "Updates transcript tracking."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "room_name": {
                    "type": "string",
                    "description": "Name of the room being entered",
                },
            },
            "required": ["room_name"],
        },
    },
    {
        "name": "request_player_input",
        "description": (
            "Ask specific players what they want to do. "
            "Call this when you need player decisions — in combat turns, "
            "at choice points, or when addressing the party."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "player_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Names of players to ask (e.g., ['Thorin', 'Shadow'] "
                        "or all four names for group input)"
                    ),
                },
            },
            "required": ["player_names"],
        },
    },
    {
        "name": "end_session",
        "description": (
            "End the adventure session. Call when the dungeon is completed, "
            "the party retreats, or the adventure reaches a natural conclusion."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Why the session is ending",
                },
            },
            "required": ["reason"],
        },
    },
]
