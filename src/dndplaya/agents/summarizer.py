"""Pre-game module summary generator."""
from __future__ import annotations

import logging
import re

from ..config import Settings
from ..prompts import load_prompt
from .base import BaseAgent

logger = logging.getLogger(__name__)

# Words too common to be meaningful keywords from a filename
_STOP_WORDS = frozenset(
    "the of and a an in to for by on at is it its with from as or"
    .split()
)


def _extract_filename_keywords(pdf_filename: str) -> list[str]:
    """Extract meaningful keywords from a PDF filename.

    Strips the extension, splits on spaces/underscores/hyphens, removes
    stop words and pure numbers, lowercases everything.
    """
    stem = re.sub(r'\.[^.]+$', '', pdf_filename)  # strip extension
    tokens = re.split(r'[\s_\-]+', stem)
    keywords = [
        t.lower() for t in tokens
        if t and t.lower() not in _STOP_WORDS and not t.isdigit()
    ]
    return keywords


def _validate_summary(summary: str, pdf_filename: str) -> str:
    """Check if the summary references at least one keyword from the filename.

    If none of the filename keywords appear in the summary, prepend a
    warning so the DM (and human reviewer) can see the mismatch.
    """
    if not pdf_filename:
        return summary

    keywords = _extract_filename_keywords(pdf_filename)
    if not keywords:
        return summary

    summary_lower = summary.lower()
    if any(kw in summary_lower for kw in keywords):
        return summary

    # No filename keyword found in summary — likely hallucination
    warning = (
        f"[WARNING: Summary may not match the module. "
        f"PDF: {pdf_filename}]\n\n"
    )
    logger.warning(
        "Summarizer output does not contain any keywords from filename '%s'. "
        "Keywords checked: %s",
        pdf_filename,
        keywords,
    )
    return warning + summary


def generate_module_summary(
    module_text: str,
    settings: Settings,
    pdf_filename: str = "",
    context_window: int = 200_000,
) -> str:
    """Generate a pre-game summary of the module using an LLM call.

    Creates a temporary BaseAgent, sends the full module text, and returns
    the summary string. Cost: one Haiku call reading the full module.

    After generation, a basic sanity check compares filename keywords
    against the summary text.  If no keywords match, a warning is
    prepended to alert downstream consumers.

    Args:
        module_text: Full extracted module text.
        settings: LLM settings.
        pdf_filename: Original PDF filename (used as grounding hint to
            prevent hallucination).
        context_window: Provider's context window in tokens. Used to
            truncate long modules for small-context models.
    """
    if pdf_filename:
        filename_note = (
            f'The source PDF is named "{pdf_filename}". '
            "Your summary MUST match the actual content of this module."
        )
    else:
        filename_note = ""

    agent = BaseAgent(
        name="Summarizer",
        system_prompt=load_prompt("summarizer_system", filename_note=filename_note),
        settings=settings,
    )

    # Truncate if the module text would exceed ~50% of the context window.
    # This leaves room for the system prompt and the response. Estimate
    # tokens as len/4 (rough char-to-token ratio).
    max_input_tokens = context_window // 2
    max_chars = max_input_tokens * 4
    if len(module_text) > max_chars:
        logger.info(
            "Module text (%d chars, ~%d tokens) exceeds 50%% of context "
            "window (%d tokens). Truncating to ~%d chars.",
            len(module_text), len(module_text) // 4,
            context_window, max_chars,
        )
        module_text = (
            module_text[:max_chars]
            + f"\n\n[...TRUNCATED — only first ~{max_chars // 2500} pages shown "
            f"due to {context_window // 1000}k context window...]"
        )

    summary = agent.send(module_text)

    # Post-summary validation: check for hallucination
    summary = _validate_summary(summary, pdf_filename)

    return summary
