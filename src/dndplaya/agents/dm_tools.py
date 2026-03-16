"""Tool definitions for the DM agent in Anthropic SDK format."""
from __future__ import annotations

DM_TOOLS = [
    {
        "name": "ask_skill_check",
        "description": (
            "Ask a player to make a skill check. The orchestrator rolls the dice "
            "and resolves success/failure based on the character's skill bonus."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "player": {
                    "type": "string",
                    "description": "Name of the PC making the check",
                },
                "skill": {
                    "type": "string",
                    "description": "Skill to check (e.g., 'perception', 'stealth', 'athletics')",
                },
                "difficulty": {
                    "type": "string",
                    "enum": [
                        "very_easy", "easy", "medium",
                        "hard", "very_hard", "nearly_impossible",
                    ],
                    "description": "How hard the check is",
                },
                "has_advantage": {
                    "type": "boolean",
                    "description": "Whether the character has advantage on this check",
                    "default": False,
                },
            },
            "required": ["player", "skill", "difficulty"],
        },
    },
    {
        "name": "attack",
        "description": (
            "A monster attacks a player character. The orchestrator resolves the "
            "attack roll and damage automatically."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "attacker": {
                    "type": "string",
                    "description": "Name of the attacking monster",
                },
                "target": {
                    "type": "string",
                    "description": "Name of the PC being attacked",
                },
            },
            "required": ["attacker", "target"],
        },
    },
    {
        "name": "change_hp",
        "description": (
            "Change a PC's hit points. Use negative values for damage, "
            "positive for healing. Returns updated HP."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "Name of the PC",
                },
                "amount": {
                    "type": "integer",
                    "description": "HP change: negative for damage, positive for healing",
                },
                "reason": {
                    "type": "string",
                    "description": "What caused the HP change (e.g., 'fire trap', 'healing potion')",
                },
            },
            "required": ["target", "amount", "reason"],
        },
    },
    {
        "name": "roll_initiative",
        "description": (
            "Start combat by rolling initiative for all PCs and monsters. "
            "Returns the turn order. Monsters are created from the CR table."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "monsters": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Monster name",
                            },
                            "cr": {
                                "type": "number",
                                "description": "Challenge rating (e.g., 0.25, 1, 3)",
                            },
                        },
                        "required": ["name", "cr"],
                    },
                    "description": "List of monsters entering combat",
                },
            },
            "required": ["monsters"],
        },
    },
    {
        "name": "request_group_input",
        "description": (
            "Ask the entire party what they want to do. "
            "Call this when you need player decisions — at choice points, "
            "during their combat turns, or when addressing the group. "
            "Players respond with actions and may use their own tools (attack/heal)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "get_party_status",
        "description": (
            "Get current stats for all party members: "
            "HP, AC, attack bonus, skills, and spell slots."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
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
    # --- Module Reference Tools ---
    {
        "name": "search_module",
        "description": (
            "Search the module text for a keyword or phrase. "
            "Returns up to 5 matches with page numbers and surrounding context. "
            "Use to find monster stats, room descriptions, trap details, etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search term (e.g., 'goblin', 'trap', 'treasure')",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_page",
        "description": (
            "Read the full text of a specific page from the module. "
            "Pages are 1-indexed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "page_number": {
                    "type": "integer",
                    "description": "Page number to read (1-indexed)",
                },
            },
            "required": ["page_number"],
        },
    },
    {
        "name": "next_page",
        "description": (
            "Read the next page after the last page you read. "
            "If you haven't read any page yet, reads page 1."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "previous_page",
        "description": (
            "Read the previous page before the last page you read. "
            "Requires having read at least one page first."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
]
