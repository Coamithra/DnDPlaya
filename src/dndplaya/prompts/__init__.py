"""Prompt loader — reads .txt files from this directory and caches them."""
from __future__ import annotations

from pathlib import Path

_PROMPT_DIR = Path(__file__).parent
_cache: dict[str, str] = {}


def load_prompt(name: str, **kwargs: str) -> str:
    """Load a prompt file by name (without .txt) and format with kwargs.

    Examples:
        load_prompt("dm_system", summary="...")
        load_prompt("session_start", party_description="...")
    """
    if name not in _cache:
        path = _PROMPT_DIR / f"{name}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        _cache[name] = path.read_text(encoding="utf-8")

    template = _cache[name]
    if kwargs:
        return template.format(**kwargs)
    return template
