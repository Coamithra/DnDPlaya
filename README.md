# DnDPlaya

![DnDPlaya Table](resources/uibg.png)

AI-powered D&D dungeon playtesting tool. Feed it a dungeon module PDF, and a party of LLM agents (1 DM + 4 players) plays through it automatically, producing a session narrative and individual reviews.

The DM reads a pre-game summary, references the module by page during play, narrates the adventure, and runs combat via tool calls. Four player agents (Fighter, Rogue, Wizard, Cleric) respond with actions, attacks, and heals using an urgency-based turn system. A live web UI shows the session in real time with thought bubbles, speech bubbles, and background music.

## Setup

### Prerequisites

- Python 3.11+
- An [Anthropic API key](https://console.anthropic.com/)

### Install

```bash
git clone https://github.com/YOUR_USERNAME/DnDPlaya.git
cd DnDPlaya
pip install -e ".[dev]"
```

### Configure

1. **API key** -- create a `.env` file (copy from the example):

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

2. **Settings** -- edit `dndplaya.ini` for session defaults:

```ini
[session]
model = haiku          # haiku, sonnet, or opus
max_tokens = 2048
party_level = 3        # 1-11
max_turns = 100

[ui]
port = 8080
music = ./music        # path to MP3s for background music
no_reviews = true      # skip review_note tools to save tokens
thinking = false       # extended thinking for debugging

[output]
dir =                  # leave empty for default (./output/runs)
```

CLI flags override INI values. `.env` is for secrets only (API key).

## Usage

### Live Web UI (recommended)

```bash
dndplaya ui dungeon.pdf
```

Opens a browser with the D&D table -- thought bubbles appear while agents think, speech bubbles with typewriter animation show narration and player dialog, and toast notifications announce combat results.

### Background Music

Drop `.mp3` files in the `./music` directory (or wherever `[ui] music` points). Filenames become track names that the DM can switch between based on mood:

```
music/
  Combat.mp3
  Combat (1).mp3      # variant -- randomly selected
  Dark Dungeon.mp3
  Tavern.mp3
  Triumph.mp3
```

The DM sees track names like "Combat", "Dark Dungeon", "Tavern" and calls `change_music` to set the mood. Variants (files with `(1)`, `(2)` suffixes) are grouped and randomly picked for variety. Music crossfades over 1.5 seconds.

### Headless Mode

```bash
dndplaya run dungeon.pdf
```

Runs the session in the terminal with no UI. Generates post-session reviews and saves transcripts, token usage, and module reference metrics to the output directory.

### Other Commands

```bash
dndplaya parse dungeon.pdf    # Test PDF parsing only
dndplaya report               # List previous run outputs
```

### CLI Flags

All flags are optional -- they override `dndplaya.ini` values:

| Flag | Description |
|------|-------------|
| `--level N` | Party level 1-11 |
| `--seed N` | Random seed for reproducibility |
| `--max-turns N` | Max DM turns before session ends |
| `--port N` | Web server port (ui only) |
| `--music DIR` | Music directory path (ui only) |
| `--no-reviews` | Disable review_note tools (ui only) |
| `--thinking` | Enable extended thinking (ui only) |

## How It Works

```
PDF --> Summarizer --> DM Agent (summary + page reference tools)
                          |
                   conversation loop
                   DM narrates + calls tools
                          |
                   Orchestrator resolves mechanics
                   (dice, skill checks, combat, HP)
                          |
                   Players respond with tools
                   (attack, heal) + urgency system
                          |
                   Narrative + Reviews + Metrics
```

- **Orchestrator as physics engine**: All dice mechanics resolved by the orchestrator. The DM focuses on narration; players act via tools.
- **Urgency system**: Players self-rate their response urgency 1-5. Higher urgency speaks first. Follow-up rounds use rising thresholds.
- **Parallel API calls**: All player calls fire simultaneously via ThreadPoolExecutor (~4x faster).
- **Prompt caching**: Three-layer strategy keeps costs low (~$0.50-1.00/session with Haiku).
- **Page-based reference**: DM searches and reads module pages during play instead of stuffing the full PDF into context.

## Development

```bash
pip install -e ".[dev]"       # Install with dev dependencies
python -m pytest tests/ -v    # Run all tests (236 tests, no API key needed)
```

## Cost

With `claude-haiku-4-5` (default), a typical session costs $0.50-1.00. Prompt caching reduces this significantly. Use `--no-reviews` (or set in INI) to save tokens during testing.
