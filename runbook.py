#!/usr/bin/env python3
"""DnDPlaya session runner for the test-and-improve runbook.

Runs a dndplaya session with Ollama/qwen and prints the output directory.
Analysis and code improvements are handled by Claude Code agents.

Usage:
    python runbook.py                      # Run with defaults (30 turns)
    python runbook.py --max-turns 20       # Shorter session
    python runbook.py --model qwen2.5:14b  # Specific Ollama model
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

OLLAMA_MODEL = "qwen2.5:14b"
PDF_PATH = "Hidden Grove of the Deep Druids 20220504.pdf"
OUTPUT_BASE = Path("output/runs")
DEFAULT_MAX_TURNS = 12


def find_latest_run() -> Path | None:
    """Return the most-recently-modified run directory."""
    if not OUTPUT_BASE.exists():
        return None
    runs = sorted(OUTPUT_BASE.iterdir(), key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None


def main():
    parser = argparse.ArgumentParser(description="Run a dndplaya session with Ollama")
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument("--model", type=str, default=OLLAMA_MODEL)
    args = parser.parse_args()

    cmd = [
        sys.executable, "-m", "dndplaya", "run", PDF_PATH,
        "--provider", "ollama",
        "--ollama-model", args.model,
        "--max-turns", str(args.max_turns),
    ]
    print(f"Running: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, timeout=3600)
    elapsed = time.time() - t0

    run_dir = find_latest_run()
    print(f"\nDone in {elapsed:.0f}s (exit code {result.returncode})")
    print(f"Output: {run_dir}")


if __name__ == "__main__":
    main()
