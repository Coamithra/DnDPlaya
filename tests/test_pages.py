"""Tests for page-aware PDF extraction."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from dndplaya.pdf.pages import extract_pages


class TestExtractPages:
    def test_extracts_text_from_chunks(self, tmp_path):
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"fake")
        mock_chunks = [
            {"text": "Page one content"},
            {"text": "Page two content"},
            {"text": "Page three content"},
        ]
        with patch("dndplaya.pdf.pages.pymupdf4llm.to_markdown", return_value=mock_chunks) as mock_md:
            result = extract_pages(str(pdf_file))

        assert result == ["Page one content", "Page two content", "Page three content"]
        mock_md.assert_called_once_with(str(pdf_file), page_chunks=True)

    def test_empty_pdf(self, tmp_path):
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"fake")
        with patch("dndplaya.pdf.pages.pymupdf4llm.to_markdown", return_value=[]):
            result = extract_pages(str(pdf_file))
        assert result == []

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            extract_pages("nonexistent.pdf")

    def test_not_a_pdf(self, tmp_path):
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("not a pdf")
        with pytest.raises(ValueError, match="Not a PDF"):
            extract_pages(str(txt_file))

    def test_single_page(self, tmp_path):
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"fake")
        mock_chunks = [{"text": "Only page"}]
        with patch("dndplaya.pdf.pages.pymupdf4llm.to_markdown", return_value=mock_chunks):
            result = extract_pages(str(pdf_file))
        assert result == ["Only page"]
