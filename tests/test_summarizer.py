"""Tests for the pre-game module summarizer."""
from __future__ import annotations

from unittest.mock import patch, MagicMock

from pydantic import SecretStr

from dndplaya.agents.summarizer import (
    generate_module_summary,
    _extract_filename_keywords,
    _validate_summary,
)
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
            system_prompt=load_prompt("summarizer_system", filename_note=""),
            settings=settings,
        )

    def test_agent_created_with_filename(self):
        settings = _make_settings()
        with patch("dndplaya.agents.summarizer.BaseAgent") as MockAgent:
            mock_instance = MagicMock()
            mock_instance.send.return_value = "summary"
            MockAgent.return_value = mock_instance

            generate_module_summary("module text", settings, pdf_filename="dungeon.pdf")

        call_kwargs = MockAgent.call_args[1]
        assert "dungeon.pdf" in call_kwargs["system_prompt"]

    def test_summary_prompt_contains_key_sections(self):
        prompt = load_prompt("summarizer_system", filename_note="")
        assert "Title" in prompt
        assert "Level Range" in prompt
        assert "Adventure Overview" in prompt
        assert "Key Locations" in prompt
        assert "Major Encounters" in prompt
        assert "Adventure Flow" in prompt
        assert "Key Warnings" in prompt

    def test_summary_prompt_contains_anti_hallucination(self):
        prompt = load_prompt("summarizer_system", filename_note="")
        assert "Do NOT invent" in prompt or "fabricate" in prompt
        assert "hallucinate" in prompt or "ONLY the content" in prompt

    def test_validation_prepends_warning_on_mismatch(self):
        """When summary doesn't contain any filename keywords, prepend warning."""
        settings = _make_settings()
        with patch("dndplaya.agents.summarizer.BaseAgent") as MockAgent:
            mock_instance = MagicMock()
            # Summary about a completely different adventure
            mock_instance.send.return_value = "Undercrypts of Shadar-Kai"
            MockAgent.return_value = mock_instance

            result = generate_module_summary(
                "module text", settings,
                pdf_filename="Hidden Grove of the Deep Druids.pdf",
            )

        assert result.startswith("[WARNING:")
        assert "Hidden Grove of the Deep Druids.pdf" in result

    def test_validation_no_warning_when_keywords_match(self):
        """When summary contains filename keywords, no warning is prepended."""
        settings = _make_settings()
        with patch("dndplaya.agents.summarizer.BaseAgent") as MockAgent:
            mock_instance = MagicMock()
            mock_instance.send.return_value = (
                "## Hidden Grove of the Deep Druids\n"
                "A druid-themed dungeon adventure."
            )
            MockAgent.return_value = mock_instance

            result = generate_module_summary(
                "module text", settings,
                pdf_filename="Hidden Grove of the Deep Druids.pdf",
            )

        assert not result.startswith("[WARNING:")

    def test_validation_skipped_when_no_filename(self):
        """No validation when pdf_filename is empty."""
        settings = _make_settings()
        with patch("dndplaya.agents.summarizer.BaseAgent") as MockAgent:
            mock_instance = MagicMock()
            mock_instance.send.return_value = "Some summary"
            MockAgent.return_value = mock_instance

            result = generate_module_summary("module text", settings)

        assert result == "Some summary"


class TestFilenameKeywordExtraction:
    def test_basic_extraction(self):
        keywords = _extract_filename_keywords("Hidden Grove of the Deep Druids.pdf")
        assert "hidden" in keywords
        assert "grove" in keywords
        assert "deep" in keywords
        assert "druids" in keywords
        # Stop words excluded
        assert "of" not in keywords
        assert "the" not in keywords

    def test_strips_extension(self):
        keywords = _extract_filename_keywords("dungeon.pdf")
        assert "dungeon" in keywords
        assert "pdf" not in keywords

    def test_splits_on_underscores_and_hyphens(self):
        keywords = _extract_filename_keywords("lost-mine_of_phandelver.pdf")
        assert "lost" in keywords
        assert "mine" in keywords
        assert "phandelver" in keywords

    def test_ignores_numbers(self):
        keywords = _extract_filename_keywords("module 42 dungeon.pdf")
        assert "module" in keywords
        assert "dungeon" in keywords
        assert "42" not in keywords

    def test_empty_filename(self):
        assert _extract_filename_keywords("") == []


class TestSummaryValidation:
    def test_warning_on_mismatch(self):
        result = _validate_summary(
            "Undercrypts of Shadar-Kai", "Hidden Grove of the Deep Druids.pdf"
        )
        assert result.startswith("[WARNING:")

    def test_no_warning_on_match(self):
        result = _validate_summary(
            "The Hidden Grove contains druids", "Hidden Grove.pdf"
        )
        assert not result.startswith("[WARNING:")

    def test_case_insensitive_match(self):
        result = _validate_summary(
            "the hidden grove", "Hidden Grove.pdf"
        )
        assert not result.startswith("[WARNING:")

    def test_no_filename_skips_validation(self):
        result = _validate_summary("anything", "")
        assert result == "anything"

    def test_all_stop_words_filename_skips(self):
        """Filename with only stop words should skip validation."""
        result = _validate_summary("anything", "the of and.pdf")
        assert result == "anything"
