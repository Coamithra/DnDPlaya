from __future__ import annotations

import pymupdf4llm
from pathlib import Path


def extract_pdf_to_markdown(pdf_path: str | Path) -> str:
    """Extract a PDF file to markdown text using pymupdf4llm.

    Uses page_chunks=False to get a single markdown string.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if not pdf_path.suffix.lower() == ".pdf":
        raise ValueError(f"Not a PDF file: {pdf_path}")

    md_text = pymupdf4llm.to_markdown(str(pdf_path))
    return md_text


def extract_pdf_pages(pdf_path: str | Path) -> list[dict]:
    """Extract PDF as page-level chunks with metadata.

    Returns list of dicts with 'metadata' and 'text' keys per page.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
    return pages
