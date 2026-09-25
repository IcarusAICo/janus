"""Live Doom HUD. `python -m demos.doom.server --backend typesafe`"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
from pathlib import Path

from starlette.applications import Starlette
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route, WebSocketRoute, Mount
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect

from demos.common.metrics import write_summary
from demos.common.record import artifact_path
from demos.common.systemone import SystemOne
from demos.doom.game import DoomLoop

WEB = Path(__file__).resolve().parent / "web"
loop: DoomLoop | None = None


async def index(request):
    return FileResponse(WEB / "index.html")


async def orders(request):
    body = await request.json()
    loop.set_orders(body.get("orders", ""))
    return JSONResponse({"ok": True})


async def state(request):
    body = await request.json()
    loop.set_override(body.get("override"))
    return JSONResponse({"ok": True})


async def schema(request):
    body = await request.json()
    loop.set_schema(body.get("schema", "json"))
    return JSONResponse({"ok": True})


async def hud_socket(websocket):
    await websocket.accept()
    try:
        while True:
            snap = loop.snapshot()
            payload = {
                "jpeg": base64.b64encode(snap.jpeg).decode("ascii") if snap.jpeg else "",
                "state_text": snap.state_text,
                "standing_orders": snap.standing_orders,
                "schema": snap.schema,
                "kills": snap.kills,
                "health": snap.health,
                "ammo": snap.ammo,
                "latency_ms": snap.latency_ms,
                "answers": snap.answers,
                "buttons": snap.buttons,
                "objective": snap.objective,
                "alive": snap.alive,
                "director_spawned": snap.director_spawned,
                "director_killed": snap.director_killed,
                "minimap": snap.minimap,
            }
            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(1 / 20)
    except WebSocketDisconnect:
        return


def app_factory(session):
    global loop
    loop = session
    return Starlette(routes=[
        Route("/", index),
        Route("/orders", orders, methods=["POST"]),
        Route("/state", state, methods=["POST"]),
        Route("/schema", schema, methods=["POST"]),
        WebSocketRoute("/ws", hud_socket),
        Mount("/static", StaticFiles(directory=WEB), name="static"),
    ])


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Live Doom HUD driven by System One")
    p.add_argument("--backend", choices=("typesafe", "local"), default="typesafe")
    p.add_argument("--base-url", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--map", dest="map_name", default="map01")
    p.add_argument("--scenario", default=None,
                   help="ViZDoom scenario wad, e.g. defend_the_center")
    p.add_argument("--env-file", default="env.sh")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    client = SystemOne(backend=args.backend, base_url=args.base_url, model=args.model,
                       env_file=args.env_file)
    session = DoomLoop(client, map_name=args.map_name, scenario=args.scenario)
    session.start()
    import uvicorn
    try:
        uvicorn.run(app_factory(session), host=args.host, port=args.port, log_level="info")
    finally:
        session.stop()
        if client.records and session.started_at:
            import time
            write_summary(artifact_path("doom-summary.json"), client.records,
                          time.perf_counter() - session.started_at,
                          extra={"kills": session.snapshot().kills})


if __name__ == "__main__":
    main()
