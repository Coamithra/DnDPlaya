"""Thread-safe event bridge between the synchronous session and async WebSocket."""
from __future__ import annotations

import asyncio
import threading


class UIEmitter:
    """Emits UI events from the session thread, consumed by the async WebSocket server.

    The session runs in a worker thread; the aiohttp server runs in the main
    asyncio loop. This class bridges the two via an asyncio.Queue (thread-safe
    put via run_coroutine_threadsafe) and a threading.Event for blocking the
    session thread until the user presses "continue" in the browser.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        self._queue: asyncio.Queue[dict] = asyncio.Queue()
        self._continue = threading.Event()
        self._connected = threading.Event()
        # Last session_start event — replayed on browser reconnect (F5)
        self._last_session_start: dict | None = None
        self._typewriter_done = threading.Event()

    # ------------------------------------------------------------------ #
    # Called from the session thread (synchronous)                        #
    # ------------------------------------------------------------------ #

    def _put(self, event: dict) -> None:
        asyncio.run_coroutine_threadsafe(self._queue.put(event), self._loop)

    def thinking_start(self, who: str) -> None:
        """Show thinking bubble. *who*: ``"dm"`` or an individual player key."""
        self._put({"type": "thinking_start", "who": who})

    def thinking_stop(self, who: str) -> None:
        self._put({"type": "thinking_stop", "who": who})

    def speech(self, who: str, name: str, text: str) -> None:
        """Show speech bubble with typewriter text."""
        self._put({"type": "speech", "who": who, "name": name, "text": text})

    def game_event(self, text: str) -> None:
        """Show a toast notification (combat result, skill check, etc.)."""
        self._put({"type": "game_event", "text": text})

    def session_start(self, dm_name: str, players: list[dict]) -> None:
        event = {"type": "session_start", "dm": dm_name, "players": players}
        self._last_session_start = event
        self._put(event)

    def session_end(self, reason: str) -> None:
        self._put({"type": "session_end", "reason": reason})

    def wait_for_continue(self, timeout: float = 300) -> None:
        """Block until the user presses a key in the browser (or timeout)."""
        self._continue.clear()
        self._put({"type": "waiting"})
        self._continue.wait(timeout=timeout)

    def wait_for_typewriter(self, timeout: float = 60) -> None:
        """Block until the frontend's typewriter animation finishes."""
        self._typewriter_done.clear()
        self._typewriter_done.wait(timeout=timeout)

    def wait_for_connection(self, timeout: float = 120) -> bool:
        """Block until a browser connects via WebSocket."""
        return self._connected.wait(timeout=timeout)

    # ------------------------------------------------------------------ #
    # Called from the asyncio thread                                      #
    # ------------------------------------------------------------------ #

    async def get_event(self) -> dict:
        return await self._queue.get()

    def signal_continue(self) -> None:
        self._continue.set()

    def signal_typewriter_done(self) -> None:
        self._typewriter_done.set()

    def signal_connected(self) -> None:
        self._connected.set()
