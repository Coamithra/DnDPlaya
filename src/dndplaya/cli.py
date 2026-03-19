from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console

from collections import Counter

from .config import Settings, get_output_dir
from .pdf.extractor import extract_pdf_to_markdown, extract_pdf_images
from .pdf.pages import extract_pages
from .pdf.chunker import chunk_markdown
from .mechanics.characters import create_default_party
from .orchestrator.session import Session
from .feedback.reviews import generate_all_reviews
# generate_module_summary and create_provider kept for future use
# when the upfront summarizer is re-enabled alongside bootstrap RAG
from .agents.provider import create_provider  # noqa: F401

console = Console()


def _safe_filename(name: str) -> str:
    """Sanitize a string for use as a filename component."""
    return re.sub(r"[^\w\-]", "_", name.lower())


def _scan_music(music_dir: Path) -> tuple[list[str] | None, dict[str, list[str]] | None]:
    """Scan a directory for MP3s, grouping variants like 'Combat (1).mp3'.

    Returns (track_names, groups) or (None, None) if no tracks found.
    """
    groups: dict[str, list[str]] = {}
    for p in sorted(music_dir.glob("*.mp3")):
        # Strip trailing " (N)" to get the base name
        base = re.sub(r"\s*\(\d+\)$", "", p.stem)
        groups.setdefault(base, []).append(p.name)
    if not groups:
        return None, None
    return sorted(groups.keys()), groups


@click.group()
def cli():
    """DnDPlaya - AI-powered D&D dungeon playtesting tool."""
    pass


@cli.command()
@click.argument("pdf_path", type=click.Path(exists=True))
@click.option("--party", default="default", help="Party preset name (default: default)")
@click.option("--level", default=None, type=click.IntRange(1, 11), help="Party level 1-11 (overrides INI)")
@click.option("--seed", default=None, type=int, help="Random seed for reproducibility")
@click.option("--runs", default=1, type=click.IntRange(1), help="Number of runs")
@click.option("--output", default=None, type=click.Path(), help="Output directory")
@click.option("--max-turns", default=None, type=click.IntRange(1), help="Max DM turns")
@click.option("--provider", default=None, type=click.Choice(["anthropic", "ollama"]), help="LLM provider")
@click.option("--ollama-model", default=None, help="Ollama model name (e.g. qwen2.5:14b)")
def run(pdf_path: str, party: str, level: int | None, seed: int | None, runs: int, output: str | None, max_turns: int | None, provider: str | None, ollama_model: str | None):
    """Run a playtesting session on a dungeon PDF."""
    settings = Settings()
    if provider is not None:
        settings.provider = provider
    if ollama_model is not None:
        settings.ollama_model = ollama_model
    settings.ensure_api_key()

    if level is not None:
        settings.party_level = level
    if seed is not None:
        settings.seed = seed
    if max_turns is not None:
        settings.max_turns = max_turns
    settings.runs = runs
    if output:
        settings.output_dir = Path(output)

    model_display = settings.ollama_model if settings.provider == "ollama" else settings.model
    console.print(f"DnDPlaya | Provider: {settings.provider} | Model: {model_display}")

    # Extract PDF content
    console.print("Extracting PDF...")
    markdown = extract_pdf_to_markdown(pdf_path)
    images = extract_pdf_images(pdf_path)
    pages = extract_pages(pdf_path)
    console.print(f"  {len(markdown):,} chars, {len(pages)} pages, {len(images)} images")

    # Module summary: skip the upfront LLM summarizer call.
    # The session bootstraps module knowledge via targeted RAG queries
    # at startup, which works better with small context windows.
    summary = ""

    # Auto-detect room connections file: {pdf_stem}_room_connections.txt
    pdf_stem = Path(pdf_path).stem
    room_map = ""
    for candidate in [
        Path(pdf_path).parent / f"{pdf_stem}_room_connections.txt",
        Path(f"{pdf_stem}_room_connections.txt"),
    ]:
        if candidate.exists():
            room_map = candidate.read_text(encoding="utf-8")
            console.print(f"  Room map: {candidate}")
            break

    # Run sessions
    for run_num in range(1, runs + 1):
        run_seed = settings.seed + run_num - 1 if settings.seed is not None else None

        if runs > 1:
            console.print(f"\n--- Run {run_num}/{runs} ---")

        # Create output dir upfront so transcript can log live
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = settings.output_dir / f"{timestamp}_run{run_num}"
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "transcript.md"

        # Create party
        char_party = create_default_party(settings.party_level)

        # Run session — DM uses summary + page-based reference tools
        session = Session(
            module_markdown=markdown,
            settings=settings,
            map_images=images,
            party=char_party,
            seed=run_seed,
            pages=pages,
            summary=summary,
            log_path=log_path,
            max_turns=settings.max_turns,
            room_map=room_map,
        )
        result = session.run()

        # Generate reviews
        transcript_text = result.get_transcript_text()
        console.print("Generating reviews...")
        reviews = generate_all_reviews(
            dm=result.dm,
            players=result.players,
            transcript_text=transcript_text,
            settings=settings,
        )

        # Dump per-agent conversation logs
        logs_dir = run_dir / "agent_logs"
        logs_dir.mkdir(exist_ok=True)
        (logs_dir / "dm.txt").write_text(result.dm.dump_history(), encoding="utf-8")
        for player in result.players:
            safe_name = _safe_filename(player.name)
            (logs_dir / f"{safe_name}.txt").write_text(
                player.dump_history(), encoding="utf-8"
            )

        # Save reviews
        for agent_name, review in reviews.items():
            safe_name = _safe_filename(agent_name)
            (run_dir / f"review_{safe_name}.md").write_text(review, encoding="utf-8")

        # Token usage with cache-aware pricing
        usage = result.token_usage
        total_input = sum(u["input_tokens"] for u in usage.values())
        total_output = sum(u["output_tokens"] for u in usage.values())
        total_cache_creation = sum(u.get("cache_creation_tokens", 0) for u in usage.values())
        total_cache_read = sum(u.get("cache_read_tokens", 0) for u in usage.values())

        if settings.provider == "ollama":
            estimated_cost = 0.0
        else:
            # Pricing per million tokens: (input, output, cache_write, cache_read)
            PRICING = {
                "claude-haiku-4-5-20251001": (0.80, 4.00, 1.00, 0.08),
                "claude-sonnet-4-6-20250514": (3.00, 15.00, 3.75, 0.30),
                "claude-opus-4-6-20250514": (15.00, 75.00, 18.75, 1.50),
            }
            input_price, output_price, cache_write_price, cache_read_price = PRICING.get(
                settings.model, (0.80, 4.00, 1.00, 0.08)
            )
            non_cached_input = total_input - total_cache_read
            estimated_cost = (
                non_cached_input * input_price
                + total_output * output_price
                + total_cache_creation * cache_write_price
                + total_cache_read * cache_read_price
            ) / 1_000_000

        usage_data = {
            "per_agent": usage,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_cache_creation_tokens": total_cache_creation,
            "total_cache_read_tokens": total_cache_read,
            "cache_hit_rate": f"{total_cache_read / total_input * 100:.1f}%" if total_input else "0%",
            "model": settings.ollama_model if settings.provider == "ollama" else settings.model,
            "provider": settings.provider,
            "estimated_cost_usd": estimated_cost,
        }
        (run_dir / "token_usage.json").write_text(
            json.dumps(usage_data, indent=2), encoding="utf-8"
        )

        # Module reference metrics
        page_counts = Counter(
            ref["page"] for ref in result.module_references
            if "page" in ref
        )
        ref_data = {
            "total_references": len(result.module_references),
            "references": result.module_references,
            "page_read_counts": dict(
                sorted(page_counts.items(), key=lambda x: -x[1])
            ),
        }
        (run_dir / "module_references.json").write_text(
            json.dumps(ref_data, indent=2), encoding="utf-8"
        )

        # Print summary
        console.print(f"Output: {run_dir}")
        cost_str = "local (free)" if settings.provider == "ollama" else f"${estimated_cost:.4f}"
        cache_pct = f"{total_cache_read / total_input * 100:.0f}" if total_input else "0"
        console.print(
            f"  Tokens: {total_input:,} in + {total_output:,} out | "
            f"Cache: {total_cache_read:,} read / {total_cache_creation:,} write ({cache_pct}% hit) | "
            f"Cost: {cost_str}"
        )
        if page_counts:
            top_pages = ", ".join(
                f"p.{p} ({c}x)" for p, c in
                sorted(page_counts.items(), key=lambda x: -x[1])[:5]
            )
            console.print(f"  Module refs: {len(result.module_references)} | Top pages: {top_pages}")


@cli.command()
@click.argument("pdf_path", type=click.Path(exists=True))
@click.option("--level", default=None, type=click.IntRange(1, 11), help="Party level 1-11")
@click.option("--seed", default=None, type=int, help="Random seed")
@click.option("--max-turns", default=None, type=click.IntRange(1), help="Max DM turns")
@click.option("--port", default=None, type=int, help="Web server port")
@click.option("--thinking", default=None, type=bool, is_flag=True, help="Enable extended thinking")
@click.option("--music", default=None, type=click.Path(exists=True, file_okay=False), help="Music directory")
@click.option("--no-reviews", default=None, type=bool, is_flag=True, help="Disable review_note tools")
@click.option("--provider", default=None, type=click.Choice(["anthropic", "ollama"]), help="LLM provider")
@click.option("--ollama-model", default=None, help="Ollama model name (e.g. qwen2.5:14b)")
def ui(pdf_path: str, level: int | None, seed: int | None, max_turns: int | None, port: int | None, thinking: bool | None, music: str | None, no_reviews: bool | None, provider: str | None, ollama_model: str | None):
    """Run a playtesting session with live web UI."""
    from .ui.server import start_ui

    settings = Settings()
    if provider is not None:
        settings.provider = provider
    if ollama_model is not None:
        settings.ollama_model = ollama_model
    settings.ensure_api_key()

    # CLI overrides
    if level is not None:
        settings.party_level = level
    if seed is not None:
        settings.seed = seed
    if max_turns is not None:
        settings.max_turns = max_turns
    if port is not None:
        settings.port = port
    if thinking is not None:
        settings.thinking = thinking
    if no_reviews is not None:
        settings.no_reviews = no_reviews
    if music is not None:
        settings.music_dir = Path(music)

    model_display = settings.ollama_model if settings.provider == "ollama" else settings.model
    console.print(f"DnDPlaya UI | Provider: {settings.provider} | Model: {model_display}")

    # Extract PDF content
    console.print("Extracting PDF...")
    markdown = extract_pdf_to_markdown(pdf_path)
    images = extract_pdf_images(pdf_path)
    pages = extract_pages(pdf_path)
    console.print(f"  {len(markdown):,} chars, {len(pages)} pages, {len(images)} images")

    # Module summary: skip upfront LLM call, bootstrap via RAG at session start.
    summary = ""

    # Auto-detect room connections file
    pdf_stem = Path(pdf_path).stem
    room_map = ""
    for candidate in [
        Path(pdf_path).parent / f"{pdf_stem}_room_connections.txt",
        Path(f"{pdf_stem}_room_connections.txt"),
    ]:
        if candidate.exists():
            room_map = candidate.read_text(encoding="utf-8")
            console.print(f"  Room map: {candidate}")
            break

    # Create output dir for transcript
    from datetime import datetime as _dt
    timestamp = _dt.now().strftime("%Y%m%d_%H%M%S")
    run_dir = settings.output_dir / f"{timestamp}_ui"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "transcript.md"

    party = create_default_party(settings.party_level)

    # Scan music directory for MP3 tracks
    music_dir = settings.music_dir
    music_tracks: list[str] | None = None
    music_groups: dict[str, list[str]] | None = None
    if music_dir and music_dir.exists():
        music_tracks, music_groups = _scan_music(music_dir)
        if music_tracks and music_groups:
            for name in music_tracks:
                variants = music_groups[name]
                tag = f" ({len(variants)} variants)" if len(variants) > 1 else ""
                console.print(f"  [green]*[/green] {name}{tag}")
        else:
            console.print("  Warning: no .mp3 files found in music directory")
            music_dir = None
    elif music_dir:
        console.print(f"  Warning: music directory not found: {music_dir}")
        music_dir = None

    def session_factory(emitter):
        return Session(
            module_markdown=markdown,
            settings=settings,
            map_images=images,
            party=party,
            seed=settings.seed,
            pages=pages,
            summary=summary,
            log_path=log_path,
            ui=emitter,
            max_turns=settings.max_turns,
            enable_thinking=settings.thinking,
            music_tracks=music_tracks,
            enable_reviews=not settings.no_reviews,
            room_map=room_map,
        )

    console.print(f"Starting UI on http://localhost:{settings.port}")
    start_ui(session_factory, port=settings.port, log_dir=run_dir, music_dir=music_dir, music_groups=music_groups)


@cli.command()
@click.argument("pdf_path", type=click.Path(exists=True))
def parse(pdf_path: str):
    """Test PDF parsing only - shows extracted structure."""
    console.print("[bold]Parsing PDF...[/bold]")

    markdown = extract_pdf_to_markdown(pdf_path)
    module = chunk_markdown(markdown)

    console.print(f"\n[bold]Title:[/bold] {module.title}")
    console.print(f"[bold]Rooms:[/bold] {len(module.rooms)}")

    if module.introduction:
        intro = module.introduction[:300]
        if len(module.introduction) > 300:
            intro += "..."
        console.print(f"\n[bold]Introduction:[/bold]\n{intro}")

    for room in module.rooms:
        console.print(f"\n[bold cyan]{room.name}[/bold cyan] (ID: {room.id})")
        if room.read_aloud:
            ra = room.read_aloud[:150]
            if len(room.read_aloud) > 150:
                ra += "..."
            console.print(f"  [italic]Read-aloud:[/italic] {ra}")
        console.print(f"  Encounters: {len(room.encounters)}")
        for enc in room.encounters:
            for m in enc.monsters:
                console.print(f"    - {m.count}x {m.name} (CR {m.cr})")
        console.print(f"  Traps: {len(room.traps)}")
        console.print(f"  Treasure: {len(room.treasure)}")
        console.print(f"  Connections: {room.connections}")

    console.print(f"\n[bold]Raw markdown length:[/bold] {len(markdown):,} chars")


@cli.command()
@click.option("--output", default=None, type=click.Path(), help="Output directory to list")
def report(output: str | None):
    """List previous run outputs."""
    output_dir = Path(output) if output else get_output_dir()
    if not output_dir.exists():
        console.print("No runs found.")
        return

    runs = sorted(d for d in output_dir.iterdir() if d.is_dir())
    if not runs:
        console.print("No runs found.")
        return

    console.print("[bold]Previous runs:[/bold]")
    for run_dir in runs:
        files = list(run_dir.iterdir())
        console.print(f"  {run_dir.name} ({len(files)} files)")


if __name__ == "__main__":
    cli()
