"""Tests for the pre-game module summarizer."""
from __future__ import annotations

from unittest.mock import patch, MagicMock

from pydantic import SecretStr

from dndplaya.agents.summarizer import generate_module_summary
from dndplaya.prompts import load_prompt
from dndplaya.config import Settings


def _make_settings() -> Settings:
    return Settings(anthropic_api_key=SecretStr("test-key-123"), max_tokens=512)


class TestGenerateModuleSummary:
    def test_returns_summary_text(self):
        settings = _make_settings()
        with patch("dndplaya.agents.summarizer.BaseAgent") as MockAgent:
            mock_instance = MagicMock()
            mock_instance.send.return_value = "## Module Summary\nA dungeon adventure."
            MockAgent.return_value = mock_instance

            result = generate_module_summary("Full module text here", settings)

        assert result == "## Module Summary\nA dungeon adventure."
        mock_instance.send.assert_called_once_with("Full module text here")

    def test_agent_created_with_correct_params(self):
        settings = _make_settings()
        with patch("dndplaya.agents.summarizer.BaseAgent") as MockAgent:
            mock_instance = MagicMock()
            mock_instance.send.return_value = "summary"
            MockAgent.return_value = mock_instance

            generate_module_summary("module text", settings)

        MockAgent.assert_called_once_with(
            name="Summarizer",
            system_prompt=load_prompt("summarizer_system"),
            settings=settings,
        )

    def test_summary_prompt_contains_key_sections(self):
        prompt = load_prompt("summarizer_system")
        assert "Title" in prompt
        assert "Level Range" in prompt
        assert "Adventure Overview" in prompt
        assert "Key Locations" in prompt
        assert "Major Encounters" in prompt
        assert "Adventure Flow" in prompt
        assert "Key Warnings" in prompt
