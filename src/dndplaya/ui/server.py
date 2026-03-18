"""aiohttp server: serves the UI page and bridges WebSocket to the session."""
from __future__ import annotations

import asyncio
import json
import threading
import webbrowser
from pathlib import Path
from typing import Callable

from aiohttp import web

from .events import UIEmitter

# Class → seat index, matching the background image character art
CLASS_TO_SEAT = {
    "Fighter": 0,   # top-left  (blue armor)
    "Cleric": 1,    # top-right (red cloak)
    "Wizard": 2,    # bottom-left (purple)
    "Rogue": 3,     # bottom-right (blue cloak)
}

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STATIC_DIR = Path(__file__).resolve().parent / "static"
BG_IMAGE = PROJECT_ROOT / "resources" / "uibg.png"


# ------------------------------------------------------------------ #
# HTTP handlers                                                       #
# ------------------------------------------------------------------ #

async def index_handler(request: web.Request) -> web.Response:
    html_path = STATIC_DIR / "index.html"
    return web.Response(
        text=html_path.read_text(encoding="utf-8"),
        content_type="text/html",
    )


async def bg_handler(request: web.Request) -> web.Response:
    if BG_IMAGE.exists():
        return web.Response(body=BG_IMAGE.read_bytes(), content_type="image/png")
    return web.Response(status=404, text="Background image not found")


# ------------------------------------------------------------------ #
# WebSocket handler                                                   #
# ------------------------------------------------------------------ #

async def ws_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    emitter: UIEmitter = request.app["emitter"]

    # Replay session_start if the browser reconnected (F5)
    if emitter._last_session_start is not None:
        await ws.send_json(emitter._last_session_start)

    # Signal connected (first connection unblocks the session thread)
    emitter.signal_connected()

    # Forward events from the emitter queue to the browser
    send_task = asyncio.create_task(_send_events(ws, emitter))

    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                data = json.loads(msg.data)
                if data.get("type") == "continue":
                    emitter.signal_continue()
                elif data.get("type") == "typewriter_done":
                    emitter.signal_typewriter_done()
            elif msg.type in (web.WSMsgType.ERROR, web.WSMsgType.CLOSE):
                break
    finally:
        send_task.cancel()

    return ws


async def _send_events(ws: web.WebSocketResponse, emitter: UIEmitter) -> None:
    try:
        while True:
            event = await emitter.get_event()
            if not ws.closed:
                await ws.send_json(event)
    except asyncio.CancelledError:
        pass


# ------------------------------------------------------------------ #
# Public entry point                                                  #
# ------------------------------------------------------------------ #

def _safe_filename(name: str) -> str:
    import re
    return re.sub(r"[^\w\-]", "_", name.lower())


def start_ui(
    session_factory: Callable[[UIEmitter], object],
    port: int = 8080,
    log_dir: Path | None = None,
) -> None:
    """Start the UI server and run a session.

    *session_factory* receives a ``UIEmitter`` and must return an object with
    a ``run()`` method (i.e. a ``Session``).  It also needs ``.party`` to be
    available so we can send party info to the browser.
    *log_dir* — if provided, agent conversation logs are dumped here after
    the session completes.
    """

    async def _on_startup(app: web.Application) -> None:
        loop = asyncio.get_running_loop()
        emitter = UIEmitter(loop)
        app["emitter"] = emitter

        def _run_session() -> None:
            if not emitter.wait_for_connection():
                print("No browser connected — aborting.")
                return

            session = session_factory(emitter)

            # Build party info for the browser
            players = []
            for i, char in enumerate(session.party):
                seat = CLASS_TO_SEAT.get(char.char_class, i % 4)
                players.append({
                    "name": char.name,
                    "class": char.char_class,
                    "hp": f"{char.current_hp}/{char.max_hp}",
                    "ac": char.ac,
                    "seat": seat,
                })
            emitter.session_start("Dungeon Master", players)

            try:
                session.run()
                emitter.session_end("Session complete!")
            except Exception as e:
                emitter.session_end(f"Error: {e}")
            finally:
                # Dump agent logs
                if log_dir:
                    logs_dir = log_dir / "agent_logs"
                    logs_dir.mkdir(exist_ok=True)
                    logs_dir.joinpath("dm.txt").write_text(
                        session.dm.dump_history(), encoding="utf-8"
                    )
                    for p in session.players:
                        logs_dir.joinpath(f"{_safe_filename(p.name)}.txt").write_text(
                            p.dump_history(), encoding="utf-8"
                        )
                    print(f"Agent logs saved to {logs_dir}")

        thread = threading.Thread(target=_run_session, daemon=True)
        thread.start()

        webbrowser.open(f"http://localhost:{port}")

    app = web.Application()
    app.on_startup.append(_on_startup)
    app.router.add_get("/", index_handler)
    app.router.add_get("/bg.png", bg_handler)
    app.router.add_get("/ws", ws_handler)

    print(f"DnDPlaya UI starting on http://localhost:{port}")
    web.run_app(app, host="localhost", port=port, print=None)
