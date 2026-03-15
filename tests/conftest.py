"""Shared test fixtures and environment sanitization."""
from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Ensure tests don't accidentally use a real API key from .env."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("DNDPLAYA_MODEL", raising=False)
    monkeypatch.delenv("DNDPLAYA_MAX_TOKENS", raising=False)
    monkeypatch.delenv("DNDPLAYA_PARTY_LEVEL", raising=False)
