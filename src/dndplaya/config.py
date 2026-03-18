"""Application configuration loaded from dndplaya.ini, .env, and CLI args.

Layering (later wins):
  1. Built-in defaults (in this file)
  2. dndplaya.ini  (session/ui/output settings)
  3. .env          (secrets only: ANTHROPIC_API_KEY)
  4. CLI flags     (override everything)
"""

from __future__ import annotations

import configparser
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr

load_dotenv()

# ── INI loading ──────────────────────────────────────────────────────

_INI_PATH = Path.cwd() / "dndplaya.ini"
_ini = configparser.ConfigParser()
if _INI_PATH.exists():
    _ini.read(_INI_PATH, encoding="utf-8")


def _ini_str(section: str, key: str, default: str) -> str:
    """Read a string from the INI file, falling back to *default*."""
    val = _ini.get(section, key, fallback=default).strip()
    return val if val else default


def _ini_int(section: str, key: str, default: int) -> int:
    """Read an int from the INI file, falling back to *default*."""
    val = _ini.get(section, key, fallback="").strip()
    if not val:
        return default
    try:
        return int(val)
    except ValueError:
        raise ValueError(f"Invalid integer for [{section}] {key} in dndplaya.ini: {val!r}")


def _ini_bool(section: str, key: str, default: bool) -> bool:
    """Read a bool from the INI file, falling back to *default*."""
    val = _ini.get(section, key, fallback="").strip().lower()
    if not val:
        return default
    return val in ("true", "yes", "1")


def _ini_optional_int(section: str, key: str) -> int | None:
    """Read an optional int (returns None if empty)."""
    val = _ini.get(section, key, fallback="").strip()
    if not val:
        return None
    try:
        return int(val)
    except ValueError:
        raise ValueError(f"Invalid integer for [{section}] {key} in dndplaya.ini: {val!r}")


def _ini_optional_path(section: str, key: str) -> Path | None:
    """Read an optional path (returns None if empty)."""
    val = _ini.get(section, key, fallback="").strip()
    return Path(val) if val else None


# ── Helpers ──────────────────────────────────────────────────────────

def get_output_dir() -> Path:
    """Default output directory, relative to cwd for portability."""
    ini_dir = _ini_optional_path("output", "dir")
    return ini_dir if ini_dir else Path.cwd() / "output" / "runs"


# Short aliases → full Anthropic model IDs
MODEL_ALIASES: dict[str, str] = {
    "haiku": "claude-haiku-4-5-20251001",
    "haiku-4.5": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6-20250514",
    "sonnet-4.6": "claude-sonnet-4-6-20250514",
    "opus": "claude-opus-4-6-20250514",
    "opus-4.6": "claude-opus-4-6-20250514",
}


def resolve_model(name: str) -> str:
    """Resolve a model alias to its full ID, or pass through if already full."""
    return MODEL_ALIASES.get(name.lower().strip(), name)


# ── Settings model ───────────────────────────────────────────────────

class Settings(BaseModel):
    """Runtime settings for a playtesting session.

    Defaults come from dndplaya.ini → .env → hardcoded fallbacks.
    CLI flags can override any field after construction.
    """

    anthropic_api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(os.getenv("ANTHROPIC_API_KEY", ""))
    )
    model: str = Field(
        default_factory=lambda: resolve_model(_ini_str("session", "model", "haiku"))
    )
    max_tokens: int = Field(
        default_factory=lambda: _ini_int("session", "max_tokens", 2048)
    )
    party_level: int = Field(
        default_factory=lambda: _ini_int("session", "party_level", 3)
    )
    max_turns: int = Field(
        default_factory=lambda: _ini_int("session", "max_turns", 100)
    )
    seed: int | None = Field(
        default_factory=lambda: _ini_optional_int("session", "seed")
    )
    runs: int = 1
    output_dir: Path = Field(default_factory=get_output_dir)

    # UI settings
    port: int = Field(
        default_factory=lambda: _ini_int("ui", "port", 8080)
    )
    music_dir: Path | None = Field(
        default_factory=lambda: _ini_optional_path("ui", "music")
    )
    no_reviews: bool = Field(
        default_factory=lambda: _ini_bool("ui", "no_reviews", False)
    )
    thinking: bool = Field(
        default_factory=lambda: _ini_bool("ui", "thinking", False)
    )

    def ensure_api_key(self) -> None:
        if not self.anthropic_api_key.get_secret_value():
            raise ValueError(
                "ANTHROPIC_API_KEY is required. Set it in .env or as an environment variable."
            )
