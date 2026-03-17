"""Page-aware PDF extraction for module reference tools."""
from __future__ import annotations

from pathlib import Path

import pymupdf4llm


def extract_pages(pdf_path: str | Path) -> list[str]:
    """Extract PDF to a list of markdown strings, one per page (0-indexed).

    Uses pymupdf4llm page_chunks mode. Tools present pages as 1-indexed,
    but this function returns a 0-indexed list.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if not pdf_path.suffix.lower() == ".pdf":
        raise ValueError(f"Not a PDF file: {pdf_path}")

    chunks = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
    return [chunk["text"] for chunk in chunks]
