# DnDPlaya

AI-powered D&D dungeon playtesting tool. Feed it a dungeon module PDF, and a party of LLM agents (1 DM + 4 players) plays through it, producing a session narrative and individual reviews.

## Quick Start

```bash
pip install -e ".[dev]"       # Install dependencies
cp .env.example .env          # Add ANTHROPIC_API_KEY (not needed for Ollama)
vi dndplaya.ini               # Tweak session settings (provider, model, music, reviews, etc.)
dndplaya run dungeon.pdf      # Run a playtest (console)
dndplaya ui dungeon.pdf       # Run with live web UI (opens browser)
dndplaya parse dungeon.pdf    # Test PDF parsing only
python -m pytest tests/ -v    # Run tests (300 tests)

# Run with local Ollama (free):
ollama pull qwen2.5:14b       # Download model (~9 GB)
dndplaya run dungeon.pdf --provider ollama --ollama-model qwen2.5:14b

# Runbook — automated test session runner for Ollama:
python runbook.py              # Run session with qwen (12 turns default)
python runbook.py --max-turns 20
```

## Configuration

Three-layer config (later wins): `dndplaya.ini` → `.env` → CLI flags.

- **`dndplaya.ini`** — session defaults: provider, model, party level, max turns, UI port, music dir, no_reviews, thinking. See the file for all options.
- **`.env`** — secrets only: `ANTHROPIC_API_KEY` (not needed for Ollama)
- **CLI flags** — override any INI value per-run (e.g. `--level 5`, `--provider ollama`, `--ollama-model qwen2.5:14b`)

## Architecture

```
PDF → [extractor] → markdown text + map images + page list
                          ↓
                    Summarizer (one LLM call → prep sheet)
                          ↓
                    DM Agent (summary in system prompt + page reference tools)
                          ↓
              ┌─── conversation loop ───┐
              │  DM narrates + calls tools │
              │  ↕                         │
              │  Orchestrator resolves tools│
              │  (skill checks, attacks,   │
              │   initiative, group input,  │
              │   module search/read)       │
              │  ↕                         │
              │  Players respond with tools │
              │  (attack, heal) + urgency  │
              └────────────────────────────┘
                          ↓
                Narrative + Reviews + Module Reference Metrics
```

The DM agent receives a pre-game summary (not the full module) and uses page-based reference tools to look up details during play. The **orchestrator is the physics engine** — it resolves all dice mechanics (skill checks, attacks, initiative, HP changes). The DM focuses on narration. Players have their own tools (attack/heal) and use an urgency system to self-select when to speak.

## Project Layout

- `src/dndplaya/mechanics/` — D&D Lite: seeded dice (incl. expression parser for "2d6+3"), character/monster stats with skills + initiative bonuses, game state tracking. CombatResolver kept for legacy/`parse` path.
- `src/dndplaya/pdf/` — PDF→Markdown (pymupdf4llm) + image extraction (pymupdf) + page-aware extraction (`pages.py`), regex-based chunking into rooms/encounters (used by `parse` command)
- `src/dndplaya/agents/provider.py` — LLM provider abstraction: `AnthropicProvider` (prompt caching, thinking, retry) and `OllamaProvider` (OpenAI-compatible API, tool validation + retry). `create_provider(settings)` factory. Each provider declares a `ProviderGuardrails` dataclass controlling defensive checks (drain loop cap, non-ASCII detection, role confusion detection).
- `runbook.py` — Automated Ollama session runner for test-and-improve cycles. Runs `dndplaya run` with qwen, prints output directory. Used with Claude Code agents for analysis and code improvement.
- `src/dndplaya/agents/` — LLM agent layer with provider abstraction + tool use + prompt caching + optional extended thinking, DM agent (summary + 11 tools + optional `change_music`), player agents (2 tools + urgency), pre-game summarizer, 4 MDA archetype player agents, post-session critic
- `src/dndplaya/orchestrator/` — DM conversation loop with tool dispatch, investment/urgency-based group input (parallel player API calls via ThreadPoolExecutor), player response validation (`_validate_player_response` → ok/fixed/needs_retry), player tool resolution (`_resolve_player_tools` — pure game engine), monster tracking, transcript recording, TPK detection, reference metric tracking, live agent log dumping
- `src/dndplaya/feedback/` — Narrative generation, per-agent "What I Liked / Take a Look At" reviews with error recovery
- `src/dndplaya/cli.py` — Click CLI: `run`, `parse`, `report`, `ui` commands; CLI flags override INI settings
- `dndplaya.ini` — INI config file for session/UI/output defaults (loaded by `config.py`)
- `src/dndplaya/ui/` — Web-based live session viewer: aiohttp server + WebSocket + HTML/CSS/JS frontend with thought bubbles, speech bubbles, and typewriter text animation

## DM Tools

The DM agent has 11 core tools (defined in `agents/dm_tools.py`) plus an optional `change_music` tool:

| Tool | Purpose |
|------|---------|
| `ask_skill_check(player, skill, difficulty, has_advantage?)` | Orchestrator rolls d20 + skill bonus vs DC |
| `attack(attacker, target)` | Monster attacks PC — orchestrator resolves hit/damage |
| `change_hp(target, amount, reason)` | Direct HP change (traps, potions, environmental) |
| `roll_initiative(monsters[{name, cr}])` | Creates monsters, rolls initiative, returns turn order |
| `request_group_input()` | Urgency-based multi-round player input collection |
| `get_party_status()` | Returns all PC stats including skills |
| `end_session(reason)` | Adventure complete, terminates loop |
| `search_module(query)` | Search module text, returns page numbers + snippets |
| `read_page(page_number)` | Read full text of a specific page (1-indexed) |
| `next_page()` | Read the page after the last-read page |
| `previous_page()` | Read the page before the last-read page |
| `change_music(track)` | *Optional* — change background music in the web UI (requires music dir in INI or `--music` flag) |

## Player Tools

Players have 2 tools (defined in `agents/player_tools.py`):

| Tool | Purpose |
|------|---------|
| `attack(target)` | Attack a monster — orchestrator resolves hit/damage |
| `heal(target)` | Heal a PC — costs spell slot, orchestrator resolves amount |

Players end every response with `[URGENCY: 1-5]` to self-select turn priority.

## Key Design Decisions

- **LLM provider abstraction**: `BaseAgent` delegates all API calls to a `LLMProvider` (protocol). `AnthropicProvider` handles prompt caching, extended thinking, and Anthropic retry logic. `OllamaProvider` speaks the OpenAI-compatible API, translates Anthropic message/tool formats to OpenAI on every call, and validates tool calls with retry (up to 2 retries on malformed calls, with error feedback to the model). History is stored in Anthropic format internally; `OllamaProvider` translates at call time. `create_provider(settings)` factory selects based on `settings.provider`.
- **Model**: `claude-haiku-4-5-20241022` for all agents (cheap, ~$0.50-1.00/run) or local via Ollama (free, recommended: `qwen2.5:14b`)
- **Orchestrator as physics engine**: All dice mechanics (skill checks, attacks, initiative, HP) resolved by the orchestrator. DM narrates; players act via tools.
- **Character skills**: Per-class skill bonuses computed from 5e proficiency formula. `compute_skills(class, level)` → dict of skill → bonus. Includes saving throw proficiencies.
- **Character pronouns**: Each character has `pronouns` field (e.g. "he/him", "she/her"). Propagated to DM party description and player system prompts. DM instructed to never misgender.
- **Investment/urgency system**: `request_group_input` collects responses from all players in parallel, parses urgency tags, runs follow-up rounds with rising thresholds: `min(round + 1, 5)` — so 1, 2, 3, 4, 5, 5, 5... up to 10 rounds max. Highest urgency speaks first.
- **Parallel player API calls**: All player `send_with_tools` + drain loop (`submit_tool_results`) calls fire simultaneously via `ThreadPoolExecutor`. A `threading.Lock` serializes dice/game-state mutations (attack/heal resolution). ~4x faster than sequential.
- **Anti-repetition**: Each player's prompt includes a short reminder of what they already said in the current group input (extracted from the shared chat). Combined with the followup prompt ("Don't repeat... pass_turn() instead"), this prevents players from restating the same action.
- **Metagaming prevention**: `get_game_context()` filters out DM-private entries (module searches, page reads, DM internal notes, review notes, cache metadata) so players only see narrations, player actions, and combat/skill results.
- **Player tools**: Players can `attack` and `heal` via tool use. Attack results (hit/miss/damage) are bundled into the group response for the DM. Monster HP is NOT modified — the DM tracks it.
- **Prompt caching**: Three-layer strategy: (1) system prompts have `cache_control`, (2) players use `set_cached_context(chat)` where `chat` is a single growing text of DM narrations + selected player actions — discarded responses never enter the chat, (3) `_mark_last_for_caching()` marks the last history message on every API call so the DM's growing conversation is always cached. Cache token tracking in `BaseAgent._record_usage()` feeds cache-aware cost calculation.
- **Players have no persistent history**: Each player call rebuilds from the cached chat via `set_cached_context()`. Only winning responses are appended to the chat. This keeps the prefix stable for cache hits and avoids polluting the model's context with discarded attempts.
- **Provider-aware compaction**: Context compaction threshold is derived from `ProviderGuardrails.context_window` (75% of window). Anthropic: 200K window → compacts at 150K. Ollama: 32K window → compacts at 24K. DM only — players use the cached chat approach instead.
- **Early outs**: Session aborts on: DM stuck (5 consecutive no-tool turns), no narration (15 turns), cost budget exceeded ($3), with cache health warning (0 reads after 10 turns).
- **Player response validation (`_validate_player_response`)**: Three-outcome validation before tool processing: `ok` (clean, proceed), `fixed` (cleaned up in-place, proceed with fixed version), `needs_retry` (unsalvageable, rollback history and resend with correction hint). The retry loop in `_call()` runs up to 3 attempts. Validation checks: (1) non-ASCII language detection (>30% non-ASCII → `needs_retry` with "Respond in English only"), (2) role confusion ("DM:" in say text → `fixed`, strip everything after it), (3) empty say (blank/whitespace → `fixed`, convert to `pass_turn`). After validation, `_resolve_player_tools` is pure game engine — no content checks.
- **ProviderGuardrails (OO)**: `ProviderGuardrails` dataclass on the `LLMProvider` protocol declares what defensive guardrails a provider needs. `AnthropicProvider` → no guardrails. `OllamaProvider` → `drain_loop_cap=5`, `detect_role_confusion=True`, `detect_non_ascii=True`. Session reads `self._guardrails` (from `dm.provider.guardrails`). Provider-specific checks (non-ASCII, role confusion, drain cap) only fire when the provider requests them. Universal checks (empty say, heal guard, stale narration, all-pass advance) always fire.
- **Drain loop cap**: Per-player tool call cap in the drain loop (5 for Ollama, unlimited for Anthropic). Prevents runaway hallucination where the model calls say() endlessly in follow-up rounds. Empty says don't count toward the cap.
- **Session-level guardrails (all providers)**: (1) all-pass auto-advance nudges the DM to advance the story after 2 consecutive all-pass group inputs, (2) stale narration detection nudges the DM to use module reference tools after 3 turns without any search_module/read_page calls, (3) heal-at-full-HP guard preserves spell slots instead of healing for 0 HP.
- **Summarizer anti-hallucination**: The summarizer prompt includes strong anti-fabrication instructions and accepts an optional `pdf_filename` parameter. After generation, `_validate_summary()` checks extracted filename keywords against the summary text. If no keywords match, a warning is prepended. Context-window-aware truncation for small-context models (50% of window).
- **Bootstrap module knowledge**: Instead of one big summarizer call, the session runs 4 focused search+summarize queries at startup (overview, NPCs, hooks, entrance). Each query chains summaries across matched pages — each page builds on the prior summary. Results are compiled into a prep sheet injected into the DM's opening prompt. This replaces the upfront summarizer for better small-context-window compatibility.
- **Search+summarize pipeline**: `search_module(search_terms, question)` does OR-matching on space-separated terms, ranks pages by term frequency (highest relevance last to exploit recency bias in chained summaries), deduplicates ±1 page windows (skips only if zero new pages), and chains summaries so the final result is comprehensive. `read_page(page_number, question)` summarizes with ±1 page context. Side-calls logged to `agent_logs/side_calls.txt`.
- **Room map injection**: Auto-detects `<pdf_stem>.map.txt` files containing text-based dungeon room connections. Injected into the DM's system prompt in `<dungeon-map>` tags for spatial navigation. Essential for non-vision models (Ollama) that can't process map images.
- **DM-driven architecture**: DM receives bootstrap prep sheet + room map, references specific pages during play via search+summarize tools. No BFS room traversal or programmatic combat resolution. DM instructed never to narrate PC actions — only describe world, NPCs, monsters, and outcomes. Map images skipped for non-vision providers (`supports_images` in `ProviderGuardrails`).
- **Page-based module reference**: DM uses `search_module`/`read_page`/`next_page`/`previous_page` to look up details. Module reference frequency is tracked as a metric.
- **Tool use**: BaseAgent supports `send_with_tools()` and `submit_tool_results()` for the Anthropic tool use API. `Message.content` is `str | list` to handle tool use blocks.
- **Extended thinking**: `BaseAgent` supports optional `enable_thinking` flag with configurable `thinking_budget` (min 1024 tokens). Thinking blocks are stored in history (with signature for API round-trip) and logged with `[thinking]...[/thinking]` tags in `dump_history()`. Enabled via `[ui] thinking` in INI or `--thinking` CLI flag for debugging player reasoning.
- **INI config**: `dndplaya.ini` provides session defaults (model, party_level, max_turns, seed), UI settings (port, music dir, no_reviews, thinking), and output dir. Loaded by `config.py` via `configparser`. `.env` is reserved for secrets (API key). CLI flags override INI values. Three-layer precedence: INI → .env → CLI.
- **4 classes**: Fighter, Rogue, Wizard, Cleric with pre-computed stats + skills for levels 1–11
- **Dice expression parser**: `DiceRoller.parse_and_roll("2d6+3")` supports multiple dice terms and +/- modifiers
- **PDF parsing**: Image extraction (≥200x200px) for map images sent to DM. Page-aware extraction for reference tools. Heading-based room detection with regex fallbacks still available via `parse` command.
- **API resilience**: BaseAgent retries transient errors with exponential backoff. History committed only after successful API calls.
- **Prompt injection guards**: Module summary wrapped in `<module-summary>` tags with explicit instruction to ignore embedded prompts
- **Live web UI**: `dndplaya ui` launches an aiohttp server (HTTP+WebSocket). Session runs in a worker thread, emitting events via `UIEmitter` (thread-safe asyncio.Queue). Browser shows the table background image with CSS thought bubbles (bouncing dots, per-character timed to actual LLM latency) during API calls, central speech bubble with markdown-rendered typewriter animation for narration/player dialog, CSS-only speech arrow pointing at speaker, toast notifications for combat/skill results, and "press space to continue" flow. Character positions mapped to image art by class (Fighter=top-left, Cleric=top-right, Wizard=bottom-left, Rogue=bottom-right). Player follow-up rounds prefetch in the background while the user reads (triggered after typewriter completes via `typewriter_done` WebSocket message). F5 reconnect replays `session_start`. The `ui` parameter on `Session.__init__` is optional — `None` preserves the original headless console behavior.
- **Live agent logs**: When `log_path` is set, `_dump_agent_logs()` appends timestamped snapshots of all agent conversation histories after every DM turn and player batch. Preserves full prompt/response history even though players replace their history each call via `set_cached_context`.
- **Background music**: `[ui] music` in INI (or `--music DIR` CLI override) scans a directory for `.mp3` files. Filenames (sans extension) become track names, dynamically added to the DM's `change_music` tool as an enum. The DM chooses tracks based on mood/setting. The orchestrator emits a `music_change` WebSocket event, and the browser plays audio with a 1.5s crossfade. `"silence"` stops playback. Works only in `ui` mode — headless mode logs the event but plays nothing.

## Testing

```bash
python -m pytest tests/ -v                           # All 300 tests
python -m pytest tests/test_characters.py -v         # Character creation + skills
python -m pytest tests/test_combat.py -v             # Combat only
python -m pytest tests/test_pdf_chunker.py -v        # PDF parsing only
python -m pytest tests/test_pages.py -v              # Page-aware PDF extraction
python -m pytest tests/test_summarizer.py -v         # Module summarizer
python -m pytest tests/test_agent_base.py -v         # Agent API layer + tool use + prompt caching (mocked)
python -m pytest tests/test_session_loop.py -v       # Session orchestrator + all tool handlers + urgency/investment
python -m pytest tests/test_dm_tools.py -v           # DM tool definitions
python -m pytest tests/test_player_tools.py -v       # Player tool definitions
python -m pytest tests/test_dice_expression.py -v    # Dice expression parsing
python -m pytest tests/test_context.py -v            # History compaction (150k threshold)
python -m pytest tests/test_config.py -v             # Settings/config
python -m pytest tests/test_provider_ollama.py -v    # Ollama provider: message/tool translation, validation
```

Tests cover: dice determinism + expression parsing, character/monster creation, skill computation (all classes × levels 1/5/11), initiative bonuses, combat resolution, pressure signals, game state lifecycle, skill checks, PDF chunking, page-aware extraction, module summarizer (incl. anti-hallucination validation + keyword extraction), data models, agent base + tool use + prompt caching (mocked provider), Ollama provider message/tool format translation + tool validation + config, session tool dispatch (skill checks/attacks/change_hp/roll_initiative/group input/module search+read/navigation/change_music/TPK), DM + player tool schema validation (including dynamic music tool builder), urgency parsing/stripping, monster registration, history compaction (text + tool-use formats, 150k threshold), config/settings, player response validation (empty say→pass_turn, role confusion DM: stripping, non-ASCII detection), drain loop cap, all-pass auto-advance, heal-at-full-HP guard, summarizer keyword validation, and edge cases. No API key needed for tests.

## TODO (from playtesting sessions)

1. ~~**Cost optimization** — players eat 90% of tokens via follow-up history resend~~ **DONE**: cached chat approach + cache token tracking
2. **Transcript readability** — needs better formatting, module summary at top, explicit group input markers
3. ~~**Random tiebreaker** — same-urgency responses always pick first in list, should randomize~~ **DONE**
4. ~~**Log group input calls** — add explicit "request_group_input called" markers in transcript~~ **DONE**
5. **`consult_map` tool** — Haiku struggles with map images; a text-based map description tool could help
6. ~~**HTML session viewer** — interactive playback~~ **DONE**: live web UI via `dndplaya ui` with thought/speech bubbles
7. **Validate caching live** — run a session and check `token_usage.json` for cache hit rate. If cache reads are 0, investigate breakpoint placement.
8. **Combat overhaul** — combat is messy in live playtesting. Needs investigation: turn order clarity, DM puppeting PCs, attack/heal visibility, monster HP tracking

## Dependencies

Runtime: `anthropic`, `openai`, `pymupdf4llm`, `pydantic`, `python-dotenv`, `click`, `rich`, `aiohttp`
Dev: `pytest`, `pytest-asyncio`, `ruff`
