from __future__ import annotations

import pymupdf4llm
import pymupdf
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


def extract_pdf_images(pdf_path: str | Path) -> list[tuple[bytes, str]]:
    """Extract images from a PDF, filtering by minimum size (200x200 px).

    Returns list of (image_bytes, media_type) tuples.
    Skips small decorative images below the size threshold.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if not pdf_path.suffix.lower() == ".pdf":
        raise ValueError(f"Not a PDF file: {pdf_path}")

    MEDIA_TYPES = {
        "png": "image/png",
        "jpeg": "image/jpeg",
        "jpg": "image/jpeg",
        "gif": "image/gif",
        "webp": "image/webp",
    }

    images: list[tuple[bytes, str]] = []
    doc = pymupdf.open(str(pdf_path))

    seen_xrefs: set[int] = set()
    for page in doc:
        for img_info in page.get_images():
            xref = img_info[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)

            extracted = doc.extract_image(xref)
            if not extracted:
                continue

            width = extracted.get("width", 0)
            height = extracted.get("height", 0)
            if width < 200 or height < 200:
                continue

            image_bytes = extracted["image"]
            ext = extracted.get("ext", "png")
            media_type = MEDIA_TYPES.get(ext, f"image/{ext}")
            images.append((image_bytes, media_type))

    doc.close()
    return images


