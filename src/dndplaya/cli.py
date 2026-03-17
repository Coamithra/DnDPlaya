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
from .agents.summarizer import generate_module_summary

console = Console()


def _safe_filename(name: str) -> str:
    """Sanitize a string for use as a filename component."""
    return re.sub(r"[^\w\-]", "_", name.lower())


@click.group()
def cli():
    """DnDPlaya - AI-powered D&D dungeon playtesting tool."""
    pass


@cli.command()
@click.argument("pdf_path", type=click.Path(exists=True))
@click.option("--party", default="default", help="Party preset name (default: default)")
@click.option("--level", default=None, type=click.IntRange(1, 11), help="Party level 1-11 (overrides .env)")
@click.option("--seed", default=None, type=int, help="Random seed for reproducibility")
@click.option("--runs", default=1, type=click.IntRange(1), help="Number of runs")
@click.option("--output", default=None, type=click.Path(), help="Output directory")
@click.option("--max-turns", default=None, type=click.IntRange(1), help="Max DM turns (default: 100)")
def run(pdf_path: str, party: str, level: int | None, seed: int | None, runs: int, output: str | None, max_turns: int | None):
    """Run a playtesting session on a dungeon PDF."""
    settings = Settings()
    settings.ensure_api_key()

    if level is not None:
        settings.party_level = level
    if seed is not None:
        settings.seed = seed
    settings.runs = runs
    if output:
        settings.output_dir = Path(output)

    console.print(f"DnDPlaya | Model: {settings.model}")

    # Extract PDF content
    console.print("Extracting PDF...")
    markdown = extract_pdf_to_markdown(pdf_path)
    images = extract_pdf_images(pdf_path)
    pages = extract_pages(pdf_path)
    console.print(f"  {len(markdown):,} chars, {len(pages)} pages, {len(images)} images")

    # Generate pre-game module summary
    console.print("Generating module summary...")
    summary = generate_module_summary(markdown, settings)
    console.print(f"  {len(summary):,} chars")

    # Run sessions
    for run_num in range(1, runs + 1):
        run_seed = seed + run_num - 1 if seed is not None else None

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
        session_kwargs = dict(
            module_markdown=markdown,
            settings=settings,
            map_images=images,
            party=char_party,
            seed=run_seed,
            pages=pages,
            summary=summary,
            log_path=log_path,
        )
        if max_turns is not None:
            session_kwargs["max_turns"] = max_turns
        session = Session(**session_kwargs)
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

        # Token usage
        PRICING = {
            "claude-haiku-4-5-20251001": (0.80, 4.00),
            "claude-sonnet-4-6-20250514": (3.00, 15.00),
            "claude-opus-4-6-20250514": (15.00, 75.00),
        }
        input_price, output_price = PRICING.get(
            settings.model, (0.80, 4.00)
        )

        usage = result.token_usage
        total_input = sum(u["input_tokens"] for u in usage.values())
        total_output = sum(u["output_tokens"] for u in usage.values())
        estimated_cost = (total_input * input_price + total_output * output_price) / 1_000_000
        usage_data = {
            "per_agent": usage,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "model": settings.model,
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
        console.print(
            f"  Tokens: {total_input:,} in + {total_output:,} out | "
            f"Cost: ${estimated_cost:.4f}"
        )
        if page_counts:
            top_pages = ", ".join(
                f"p.{p} ({c}x)" for p, c in
                sorted(page_counts.items(), key=lambda x: -x[1])[:5]
            )
            console.print(f"  Module refs: {len(result.module_references)} | Top pages: {top_pages}")


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
