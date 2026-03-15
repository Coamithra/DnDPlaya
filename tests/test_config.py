"""Tests for config.py — Settings and environment variable handling."""
from __future__ import annotations

import pytest
from pydantic import SecretStr

from dndplaya.config import Settings, _env_int


def test_settings_defaults():
    """Settings should have sensible defaults when no env vars are set."""
    s = Settings()
    assert s.model == "claude-haiku-4-5-20241022"
    assert s.max_tokens == 2048
    assert s.party_level == 3
    assert s.seed is None
    assert s.runs == 1


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
    s = Settings()
    assert s.party_level == 7


def test_env_int_valid():
    assert _env_int("NONEXISTENT_KEY", 42) == 42


def test_env_int_invalid(monkeypatch):
    monkeypatch.setenv("BAD_INT", "abc")
    with pytest.raises(ValueError, match="Invalid integer"):
        _env_int("BAD_INT", 0)
