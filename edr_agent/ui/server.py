"""
ui/server.py — FastAPI + WebSocket control panel server.

Serves the HTML control panel and manages the agent lifecycle.
WebSocket streams CycleEvents to the browser in real time.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from edr_agent.agent import BookingAgent
from edr_agent.config import BookingMode, SeatType, UserConfig, settings, STATION_NAMES
from edr_agent.vlm.schemas import CycleEvent

logger = logging.getLogger(__name__)

app = FastAPI(title="EDR VLM Booking Agent", version="1.0.0")

# Serve screenshot files
SCREENSHOTS_DIR = Path("screenshots")
SCREENSHOTS_DIR.mkdir(exist_ok=True)
app.mount("/screenshots", StaticFiles(directory=str(SCREENSHOTS_DIR)), name="screenshots")

# Serve static UI files
UI_STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(UI_STATIC)), name="static")

# ---------------------------------------------------------------------------
# Global agent state
# ---------------------------------------------------------------------------

_agent: Optional[BookingAgent] = None
_agent_task: Optional[asyncio.Task] = None
_websocket_clients: set[WebSocket] = set()


async def _broadcast(event: CycleEvent) -> None:
    """Send a CycleEvent to all connected WebSocket clients."""
    if not _websocket_clients:
        return

    payload = json.dumps({
        "type": "cycle_event",
        "data": event.model_dump(mode="json"),
    })

    dead = set()
    for ws in _websocket_clients.copy():
        try:
            await ws.send_text(payload)
        except Exception:
            dead.add(ws)

    _websocket_clients.difference_update(dead)


def _sync_broadcast(event: CycleEvent) -> None:
    """Synchronous wrapper for broadcasting from non-async context."""
    loop = asyncio.get_event_loop()
    if loop.is_running():
        asyncio.ensure_future(_broadcast(event))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the control panel HTML."""
    html_path = UI_STATIC / "index.html"
    return FileResponse(str(html_path))


@app.get("/api/stations")
async def get_stations():
    """Return the list of EDR stations."""
    return {"stations": STATION_NAMES}


@app.get("/api/status")
async def get_status():
    """Return current agent status."""
    global _agent
    if _agent and _agent.state:
        return {
            "running": _agent_task is not None and not _agent_task.done(),
            "state": _agent.state.to_summary_dict(),
        }
    return {"running": False, "state": None}


@app.post("/api/start")
async def start_agent(config: dict):
    """Start the booking agent with the provided configuration."""
    global _agent, _agent_task

    if _agent_task and not _agent_task.done():
        return {"error": "Agent is already running"}

    try:
        # Parse config
        travel_date = date.fromisoformat(config["travel_date"])
        preferred_seat = SeatType(config.get("preferred_seat", "Economy"))
        booking_mode = BookingMode(config.get("booking_mode", "NORMAL"))

        # Parse seat ranking for DATE-FIRST mode
        seat_ranking_raw = config.get("seat_ranking", [preferred_seat.value])
        seat_ranking = [SeatType(s) for s in seat_ranking_raw if s]

        user_config = UserConfig(
            origin=config["origin"],
            destination=config["destination"],
            travel_date=travel_date,
            preferred_seat=preferred_seat,
            seat_ranking=seat_ranking,
            booking_mode=booking_mode,
            monitoring_interval_minutes=int(config.get("monitoring_interval_minutes", 10)),
        )

        _agent = BookingAgent()
        _agent.set_event_callback(_sync_broadcast)

        mode = config.get("run_mode", "once")  # "once" or "monitor"

        if mode == "monitor":
            _agent_task = asyncio.create_task(_agent.run_with_monitoring(user_config))
        else:
            _agent_task = asyncio.create_task(_agent.run_once(user_config))

        return {"status": "started", "mode": mode}

    except Exception as e:
        logger.error(f"Failed to start agent: {e}")
        return {"error": str(e)}


@app.post("/api/stop")
async def stop_agent():
    """Stop the running agent."""
    global _agent, _agent_task

    if _agent:
        _agent.request_stop()

    if _agent_task and not _agent_task.done():
        _agent_task.cancel()
        try:
            await asyncio.wait_for(_agent_task, timeout=5.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass

    return {"status": "stopped"}


@app.post("/api/resume")
async def resume_agent():
    """
    Resume the booking agent after a human handoff (e.g. Fayda verification).
    Signals the asyncio.Event that is blocking the agent coroutine.
    """
    global _agent

    if not _agent or not _agent.state:
        return {"error": "No agent is running"}

    if not _agent.state.waiting_for_human or _agent.state.handoff_stage != "FAYDA":
        return {"error": "Agent is not currently paused at a Fayda handoff"}

    _agent.resume_from_handoff()
    logger.info("[SERVER] /api/resume called — Fayda handoff unblocked")
    return {"status": "resumed"}



@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time event streaming to the UI."""
    await websocket.accept()
    _websocket_clients.add(websocket)
    logger.info(f"[WS] Client connected ({len(_websocket_clients)} total)")

    # Send current state immediately
    if _agent and _agent.state:
        await websocket.send_text(json.dumps({
            "type": "state_sync",
            "data": _agent.state.to_summary_dict(),
        }))

    try:
        while True:
            # Keep connection alive; client sends "ping"
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        _websocket_clients.discard(websocket)
        logger.info(f"[WS] Client disconnected ({len(_websocket_clients)} remaining)")
    except Exception as e:
        _websocket_clients.discard(websocket)
        logger.warning(f"[WS] Connection error: {e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info(f"Starting EDR VLM Booking Agent UI on http://{settings.server_host}:{settings.server_port}")
    uvicorn.run(
        "edr_agent.ui.server:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
