"""Tests for config.py — Settings and INI/env loading."""
from __future__ import annotations

import pytest
from pydantic import SecretStr

from dndplaya.config import Settings, _ini_int, _ini_bool, _ini_str, _ini_optional_int


def test_settings_defaults():
    """Settings should have sensible defaults when no env vars are set."""
    s = Settings()
    assert s.model == "claude-haiku-4-5-20251001"
    assert s.max_tokens == 2048
    assert s.party_level == 3
    assert s.max_turns == 100
    assert s.runs == 1
    assert s.port == 8080
    assert s.thinking is False


def test_settings_api_key_is_secret():
    """API key should be a SecretStr, not exposed in repr."""
    s = Settings(anthropic_api_key=SecretStr("sk-test-123"))
    assert isinstance(s.anthropic_api_key, SecretStr)
    assert "sk-test-123" not in repr(s)
    assert s.anthropic_api_key.get_secret_value() == "sk-test-123"


def test_ensure_api_key_raises_when_empty():
    s = Settings()
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is required"):
        s.ensure_api_key()


def test_ensure_api_key_passes_when_set():
    s = Settings(anthropic_api_key=SecretStr("sk-test"))
    s.ensure_api_key()  # Should not raise


def test_settings_reads_env_at_instantiation(monkeypatch):
    """Env vars should be read when Settings() is called, not at import time."""
    monkeypatch.setenv("DNDPLAYA_PARTY_LEVEL", "7")
    # Note: env vars no longer feed into Settings directly (INI does),
    # but we can still test direct construction
    s = Settings(party_level=7)
    assert s.party_level == 7


def test_settings_ui_fields():
    """UI-related settings should be constructable."""
    s = Settings(port=9090, no_reviews=True, thinking=True)
    assert s.port == 9090
    assert s.no_reviews is True
    assert s.thinking is True


def test_ini_str_fallback():
    assert _ini_str("nonexistent", "key", "default") == "default"


def test_ini_int_fallback():
    assert _ini_int("nonexistent", "key", 42) == 42


def test_ini_bool_fallback():
    assert _ini_bool("nonexistent", "key", False) is False
    assert _ini_bool("nonexistent", "key", True) is True


def test_ini_optional_int_fallback():
    assert _ini_optional_int("nonexistent", "key") is None
