# DnDPlaya

AI-powered D&D dungeon playtesting tool. Feed it a dungeon module PDF, and a party of LLM agents (1 DM + 4 players) plays through it, producing a session narrative and individual reviews.

## Quick Start

```bash
pip install -e ".[dev]"       # Install dependencies
cp .env.example .env          # Add ANTHROPIC_API_KEY
dndplaya run dungeon.pdf      # Run a playtest
dndplaya parse dungeon.pdf    # Test PDF parsing only
python -m pytest tests/ -v    # Run tests (48 tests)
```

## Architecture

```
PDF → [extractor] → [chunker] → DungeonModule
                                      ↓
                                [orchestrator/session.py]
                               /         |          \
                        [DM Agent]       |    [Player Agents ×4]
                               \         |          /
                            [mechanics layer]
                         (dice, combat, state)
                                      ↓
                              [feedback layer]
                                      ↓
                        Session Narrative + 5 Reviews
```

## Project Layout

- `src/dndplaya/mechanics/` — D&D Lite combat engine: seeded dice, character/monster stats (DMG CR table), round resolution with pressure signals, game state tracking
- `src/dndplaya/pdf/` — PDF→Markdown (pymupdf4llm), regex-based chunking into rooms/encounters
- `src/dndplaya/agents/` — Claude API wrapper, DM agent (dynamic room context), 4 MDA archetype player agents (Roleplayer/Tactician/Explorer/Free Spirit), post-session critic
- `src/dndplaya/orchestrator/` — Main game loop (BFS room traversal → exploration → combat), turn management, transcript recording
- `src/dndplaya/feedback/` — Narrative generation, per-agent "What I Liked / Take a Look At" reviews
- `src/dndplaya/cli.py` — Click CLI: `run`, `parse`, `report` commands

## Key Design Decisions

- **Model**: `claude-haiku-4-5-20241022` for all agents (cheap, ~$0.50-1.00/run)
- **Combat**: D&D Lite — avg damage × random(0.6, 1.4) per round, no individual attack rolls. `CombatResolver.resolve_attack()` returns damage without modifying state; the session orchestrator applies HP changes
- **Pressure signals**: Mechanics flag low HP (<25%), bloodied enemies (<50%), resource depletion, TPK risk → DM narrates dramatic moments
- **4 classes**: Fighter, Rogue, Wizard, Cleric with pre-computed stats for levels 1–11
- **Monsters**: Looked up by CR from hardcoded DMG table (CR 0–10)
- **Context management**: History compacted when exceeding token budget; DM prompt dynamically rebuilt per room
- **PDF parsing**: Heading-based room detection with regex fallbacks; linear room layout assumed if no cross-references found

## Testing

```bash
python -m pytest tests/ -v                    # All 48 tests
python -m pytest tests/test_combat.py -v      # Combat only
python -m pytest tests/test_pdf_chunker.py -v # PDF parsing only
```

Tests cover: dice determinism, character/monster creation, combat resolution, pressure signals, game state lifecycle, skill checks, PDF chunking, and data models. No API key needed for tests.

## Dependencies

Runtime: `anthropic`, `pymupdf4llm`, `pydantic`, `python-dotenv`, `click`, `rich`, `jinja2`
Dev: `pytest`, `pytest-asyncio`, `ruff`
