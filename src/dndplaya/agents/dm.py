from __future__ import annotations

import base64

from ..config import Settings
from ..prompts import load_prompt
from .base import BaseAgent
from .dm_tools import DM_TOOLS, build_music_tool


class DMAgent(BaseAgent):
    """DM agent that reads a module summary and references pages during play."""

    def __init__(
        self,
        summary: str,
        settings: Settings,
        map_images: list[tuple[bytes, str]] | None = None,
        music_tracks: list[str] | None = None,
        enable_reviews: bool = True,
    ):
        self.runnability_notes: list[str] = []

        system_text = load_prompt("dm_system", summary=summary)

        # Store map images to inject into the first user message
        # (system prompt only supports text blocks).
        # Keep only the largest images (likely maps) and cap total payload.
        MAX_IMAGES = 3
        MAX_TOTAL_BYTES = 3 * 1024 * 1024  # 3 MB
        candidates = sorted(map_images or [], key=lambda x: len(x[0]), reverse=True)
        self._map_images: list[tuple[bytes, str]] = []
        total_bytes = 0
        for img_bytes, media_type in candidates[:MAX_IMAGES]:
            if total_bytes + len(img_bytes) > MAX_TOTAL_BYTES:
                break
            self._map_images.append((img_bytes, media_type))
            total_bytes += len(img_bytes)

        tools = list(DM_TOOLS)
        if not enable_reviews:
            tools = [t for t in tools if t["name"] != "review_note"]
        if music_tracks:
            tools.append(build_music_tool(music_tracks))

        super().__init__(
            name="DM",
            system_prompt=system_text,
            settings=settings,
            tools=tools,
        )

    def send_with_tools(self, user_message: str):
        """Override to inject map images into the first message only."""
        if self._map_images and not self.history:
            # Build a multipart user message: text + images
            content: list = [{"type": "text", "text": user_message}]
            for img_bytes, media_type in self._map_images:
                b64_data = base64.b64encode(img_bytes).decode("utf-8")
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": b64_data,
                    },
                })
            content.append({
                "type": "text",
                "text": "Above are the map images from the module. Use them to understand the dungeon layout.",
            })
            # Call the base class internals directly with the multipart content
            messages: list[dict] = [{"role": "user", "content": content}]
            resp = self._make_api_call(messages, use_tools=True)
            return self._process_response(resp, content)
        return super().send_with_tools(user_message)

    def add_runnability_note(self, note: str) -> None:
        self.runnability_notes.append(note)
