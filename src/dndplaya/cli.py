from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from .config import Settings, OUTPUT_DIR
from .pdf.extractor import extract_pdf_to_markdown
from .pdf.chunker import chunk_markdown
from .mechanics.characters import create_default_party
from .orchestrator.session import Session
from .feedback.narrative import generate_narrative
from .feedback.reviews import generate_all_reviews

console = Console()


@click.group()
def cli():
    """DnDPlaya - AI-powered D&D dungeon playtesting tool."""
    pass


@cli.command()
@click.argument("pdf_path", type=click.Path(exists=True))
@click.option("--party", default="default", help="Party preset name (default: default)")
@click.option("--level", default=None, type=int, help="Party level (overrides .env)")
@click.option("--seed", default=None, type=int, help="Random seed for reproducibility")
@click.option("--runs", default=1, type=int, help="Number of runs")
@click.option("--output", default=None, type=click.Path(), help="Output directory")
def run(pdf_path: str, party: str, level: int | None, seed: int | None, runs: int, output: str | None):
    """Run a playtesting session on a dungeon PDF."""
    settings = Settings()
    settings.ensure_api_key()

    if level:
        settings.party_level = level
    if seed is not None:
        settings.seed = seed
    settings.runs = runs
    if output:
        settings.output_dir = Path(output)

    console.print(Panel.fit(
        "[bold]DnDPlaya[/bold] - AI Dungeon Playtester",
        subtitle=f"Model: {settings.model}",
    ))

    # Parse PDF
    console.print("\n[bold]Parsing dungeon PDF...[/bold]")
    markdown = extract_pdf_to_markdown(pdf_path)
    module = chunk_markdown(markdown)

    console.print(f"  Title: {module.title}")
    console.print(f"  Rooms found: {len(module.rooms)}")
    for room in module.rooms:
        enc_count = len(room.encounters)
        console.print(f"    - {room.name} ({enc_count} encounters)")

    # Run sessions
    for run_num in range(1, runs + 1):
        run_seed = seed + run_num - 1 if seed is not None else None

        console.print(f"\n[bold green]{'='*50}[/bold green]")
        console.print(f"[bold green]Run {run_num}/{runs}[/bold green]")
        console.print(f"[bold green]{'='*50}[/bold green]\n")

        # Create party
        char_party = create_default_party(settings.party_level)

        # Run session
        session = Session(
            module=module,
            settings=settings,
            party=char_party,
            seed=run_seed,
        )
        result = session.run()

        # Generate outputs
        console.print("\n[bold]Generating narrative...[/bold]")
        transcript_text = result.get_transcript_text()
        narrative = generate_narrative(transcript_text, settings)

        console.print("[bold]Generating reviews...[/bold]")
        reviews = generate_all_reviews(
            dm=result.dm,
            players=result.players,
            transcript_text=transcript_text,
            settings=settings,
        )

        # Save outputs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = settings.output_dir / f"{timestamp}_run{run_num}"
        run_dir.mkdir(parents=True, exist_ok=True)

        (run_dir / "narrative.md").write_text(narrative, encoding="utf-8")
        (run_dir / "transcript.md").write_text(transcript_text, encoding="utf-8")

        for agent_name, review in reviews.items():
            safe_name = agent_name.lower().replace(" ", "_")
            (run_dir / f"review_{safe_name}.md").write_text(review, encoding="utf-8")

        # Token usage — pricing per million tokens by model
        PRICING = {
            "claude-haiku-4-5-20241022": (0.25, 1.25),
            "claude-sonnet-4-6-20250514": (3.00, 15.00),
            "claude-opus-4-6-20250514": (15.00, 75.00),
        }
        input_price, output_price = PRICING.get(
            settings.model, (0.25, 1.25)  # default to Haiku pricing
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

        # Print summary
        console.print(f"\n[bold]Results saved to:[/bold] {run_dir}")
        console.print(f"  Token usage: {total_input:,} input + {total_output:,} output")
        console.print(f"  Estimated cost: ${usage_data['estimated_cost_usd']:.4f}")

        # Show narrative preview
        console.print("\n[bold]Narrative Preview:[/bold]")
        console.print(Panel(Markdown(narrative[:1000] + "..." if len(narrative) > 1000 else narrative)))

        # Show review summaries
        console.print("\n[bold]Reviews:[/bold]")
        for agent_name, review in reviews.items():
            console.print(Panel(
                Markdown(review[:500] + "..." if len(review) > 500 else review),
                title=f"{agent_name}'s Review",
            ))


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
        console.print(f"\n[bold]Introduction:[/bold]\n{module.introduction[:300]}...")

    for room in module.rooms:
        console.print(f"\n[bold cyan]{room.name}[/bold cyan] (ID: {room.id})")
        if room.read_aloud:
            console.print(f"  [italic]Read-aloud:[/italic] {room.read_aloud[:150]}...")
        console.print(f"  Encounters: {len(room.encounters)}")
        for enc in room.encounters:
            for m in enc.monsters:
                console.print(f"    - {m.count}x {m.name} (CR {m.cr})")
        console.print(f"  Traps: {len(room.traps)}")
        console.print(f"  Treasure: {len(room.treasure)}")
        console.print(f"  Connections: {room.connections}")

    console.print(f"\n[bold]Raw markdown length:[/bold] {len(markdown):,} chars")


@cli.command()
def report():
    """List previous run outputs."""
    if not OUTPUT_DIR.exists():
        console.print("No runs found.")
        return

    runs = sorted(OUTPUT_DIR.iterdir())
    if not runs:
        console.print("No runs found.")
        return

    console.print("[bold]Previous runs:[/bold]")
    for run_dir in runs:
        if run_dir.is_dir():
            files = list(run_dir.iterdir())
            console.print(f"  {run_dir.name} ({len(files)} files)")


if __name__ == "__main__":
    cli()
