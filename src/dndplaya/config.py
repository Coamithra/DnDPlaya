"""Application configuration loaded from .env and CLI args."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr

load_dotenv()


def _env_int(key: str, default: int) -> int:
    """Read an integer from an environment variable with a helpful error."""
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        raise ValueError(f"Invalid integer for environment variable {key}: {val!r}")


def get_output_dir() -> Path:
    """Default output directory, relative to cwd for portability."""
    return Path.cwd() / "output" / "runs"


class Settings(BaseModel):
    """Runtime settings for a playtesting session."""

    anthropic_api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(os.getenv("ANTHROPIC_API_KEY", ""))
    )
    model: str = Field(
        default_factory=lambda: os.getenv("DNDPLAYA_MODEL", "claude-haiku-4-5-20241022")
    )
    max_tokens: int = Field(default_factory=lambda: _env_int("DNDPLAYA_MAX_TOKENS", 2048))
    party_level: int = Field(default_factory=lambda: _env_int("DNDPLAYA_PARTY_LEVEL", 3))
    seed: int | None = None
    runs: int = 1
    output_dir: Path = Field(default_factory=get_output_dir)

    def ensure_api_key(self) -> None:
        if not self.anthropic_api_key.get_secret_value():
            raise ValueError(
                "ANTHROPIC_API_KEY is required. Set it in .env or as an environment variable."
            )
