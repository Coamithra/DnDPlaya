# DnDPlaya

AI-powered D&D dungeon playtesting tool. Feed it a dungeon module PDF, and a party of LLM agents (1 DM + 4 players) plays through it, producing a session narrative and individual reviews.

## Quick Start

```bash
pip install -e ".[dev]"       # Install dependencies
cp .env.example .env          # Add ANTHROPIC_API_KEY
dndplaya run dungeon.pdf      # Run a playtest
dndplaya parse dungeon.pdf    # Test PDF parsing only
python -m pytest tests/ -v    # Run tests (154 tests)
```

## Architecture

```
PDF → [extractor] → markdown text + map images
                          ↓
                    DM Agent (full module + map in system prompt)
                          ↓
              ┌─── conversation loop ───┐
              │  DM narrates + calls tools │
              │  ↕                         │
              │  Orchestrator resolves tools│
              │  (dice, damage, player input)│
              │  ↕                         │
              │  Players respond when asked │
              └────────────────────────────┘
                          ↓
                Narrative + Reviews (unchanged)
```

The DM agent reads the full module text (+ map images) and drives the adventure organically using 8 tools via the Anthropic tool use API. The orchestrator processes tool calls (dice rolls, damage, healing, player input) and feeds results back. Players respond only when the DM explicitly requests their input.

## Project Layout

- `src/dndplaya/mechanics/` — D&D Lite: seeded dice (incl. expression parser for "2d6+3"), character/monster stats, game state tracking. CombatResolver kept for legacy/`parse` path.
- `src/dndplaya/pdf/` — PDF→Markdown (pymupdf4llm) + image extraction (pymupdf), regex-based chunking into rooms/encounters (used by `parse` command)
- `src/dndplaya/agents/` — Claude API wrapper with retry logic + tool use support (`send_with_tools`, `submit_tool_results`), DM agent (full module in system prompt + 8 tools), 4 MDA archetype player agents, post-session critic
- `src/dndplaya/orchestrator/` — DM conversation loop with tool dispatch, transcript recording, TPK detection
- `src/dndplaya/feedback/` — Narrative generation, per-agent "What I Liked / Take a Look At" reviews with error recovery
- `src/dndplaya/cli.py` — Click CLI: `run`, `parse`, `report` commands with input validation

## DM Tools

The DM agent has 8 tools (defined in `agents/dm_tools.py`):

| Tool | Purpose |
|------|---------|
| `roll_check(modifier, dc, description)` | d20+mod vs DC for attacks, saves, ability checks |
| `roll_dice(expression, reason)` | Parse & roll "2d6+3" etc. for damage, random effects |
| `apply_damage(character_name, amount, description)` | Subtract HP from a PC |
| `heal(character_name, amount, description)` | Add HP to a PC (capped at max) |
| `get_party_status()` | Returns all PC stats |
| `enter_room(room_name)` | Signals room transition for transcript |
| `request_player_input(player_names)` | Collect responses from specific players |
| `end_session(reason)` | Adventure complete, terminates loop |

Key: The DM tracks monster HP itself (in its conversation context). The orchestrator only tracks PC HP.

## Key Design Decisions

- **Model**: `claude-haiku-4-5-20241022` for all agents (cheap, ~$0.50-1.00/run)
- **DM-driven architecture**: DM reads full module + map images, drives adventure organically via tool calls. No BFS room traversal or programmatic combat resolution.
- **Tool use**: BaseAgent supports `send_with_tools()` and `submit_tool_results()` for the Anthropic tool use API. `Message.content` is `str | list` to handle tool use blocks.
- **4 classes**: Fighter, Rogue, Wizard, Cleric with pre-computed stats for levels 1–11
- **Dice expression parser**: `DiceRoller.parse_and_roll("2d6+3")` supports multiple dice terms and +/- modifiers
- **Context management**: History compacted when `last_input_tokens` exceeds budget; handles both text and tool-use message formats
- **PDF parsing**: Image extraction (≥200x200px) for map images sent to DM. Heading-based room detection with regex fallbacks still available via `parse` command.
- **API resilience**: BaseAgent retries transient errors with exponential backoff. History committed only after successful API calls.
- **Prompt injection guards**: Module text wrapped in `<module-text>` tags with explicit instruction to ignore embedded prompts

## Testing

```bash
python -m pytest tests/ -v                           # All 154 tests
python -m pytest tests/test_combat.py -v             # Combat only
python -m pytest tests/test_pdf_chunker.py -v        # PDF parsing only
python -m pytest tests/test_agent_base.py -v         # Agent API layer + tool use (mocked)
python -m pytest tests/test_session_loop.py -v       # Session orchestrator + tool dispatch
python -m pytest tests/test_dm_tools.py -v           # DM tool definitions
python -m pytest tests/test_dice_expression.py -v    # Dice expression parsing
python -m pytest tests/test_context.py -v            # History compaction
python -m pytest tests/test_config.py -v             # Settings/config
```

Tests cover: dice determinism + expression parsing, character/monster creation, combat resolution, pressure signals, game state lifecycle, skill checks, PDF chunking, data models, agent base + tool use (mocked API), session tool dispatch (damage/heal/rolls/rooms/TPK), DM tool schema validation, history compaction (text + tool-use formats), phase state machine, config/settings, and edge cases. No API key needed for tests.

## Dependencies

Runtime: `anthropic`, `pymupdf4llm`, `pydantic`, `python-dotenv`, `click`, `rich`
Dev: `pytest`, `pytest-asyncio`, `ruff`
