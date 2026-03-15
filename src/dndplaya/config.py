"""Application configuration loaded from .env and CLI args."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output" / "runs"


class Settings(BaseModel):
    """Runtime settings for a playtesting session."""

    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    model: str = os.getenv("DNDPLAYA_MODEL", "claude-haiku-4-5-20241022")
    max_tokens: int = int(os.getenv("DNDPLAYA_MAX_TOKENS", "1024"))
    party_level: int = int(os.getenv("DNDPLAYA_PARTY_LEVEL", "3"))
    seed: int | None = None
    runs: int = 1
    output_dir: Path = OUTPUT_DIR

    def ensure_api_key(self) -> None:
        if not self.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is required. Set it in .env or as an environment variable."
            )
