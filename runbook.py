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
import os
import subprocess
import sys
import time
from pathlib import Path

OLLAMA_MODEL = "qwen2.5:14b"
OLLAMA_NUM_CTX = 32768
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
    parser.add_argument("--num-ctx", type=int, default=OLLAMA_NUM_CTX,
                        help="Ollama context window in tokens (default: 32768)")
    args = parser.parse_args()

    # Prevent Ollama from multiplying num_ctx by its parallel slot count.
    # On 12GB VRAM, parallel slots + 32k context can cause CPU offload.
    env = {**os.environ, "OLLAMA_NUM_PARALLEL": "1"}

    cmd = [
        sys.executable, "-m", "dndplaya", "run", PDF_PATH,
        "--provider", "ollama",
        "--ollama-model", args.model,
        "--ollama-num-ctx", str(args.num_ctx),
        "--max-turns", str(args.max_turns),
    ]
    print(f"Running: {' '.join(cmd)}")
    print(f"  OLLAMA_NUM_PARALLEL=1, num_ctx={args.num_ctx}")
    t0 = time.time()
    result = subprocess.run(cmd, env=env, timeout=3600)
    elapsed = time.time() - t0

    run_dir = find_latest_run()
    print(f"\nDone in {elapsed:.0f}s (exit code {result.returncode})")
    print(f"Output: {run_dir}")


if __name__ == "__main__":
    main()
