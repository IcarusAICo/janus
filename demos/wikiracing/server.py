"""Live four-way wikirace. `python -m demos.wikiracing.server --challenge baseball-sun`"""
from __future__ import annotations

import argparse
import asyncio
import json
import threading
from pathlib import Path

from starlette.applications import Starlette
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route, WebSocketRoute
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect

from demos.common.metrics import write_summary
from demos.common.record import artifact_path
from demos.common.systemone import SystemOne
from demos.wikiracing.compare import WikiRace, load_challenges
from demos.wikiracing.openai import OpenAISystemOne

WEB = Path(__file__).resolve().parent / "web"
race: WikiRace | None = None
latest = {}


async def index(request):
    return FileResponse(WEB / "index.html")


async def status(request):
    return JSONResponse(latest or (race.snapshot() if race else {}))


async def start(request):
    threading.Thread(target=_run, daemon=True).start()
    return JSONResponse({"ok": True})


async def socket(websocket):
    await websocket.accept()
    try:
        while True:
            await websocket.send_text(json.dumps(latest or {}))
            await asyncio.sleep(0.25)
    except WebSocketDisconnect:
        return


def _run():
    global latest
    latest = race.run(on_update=lambda snap: _store(snap))


def _store(snap):
    global latest
    latest = snap
    return snap


def app_factory(session):
    global race, latest
    race = session
    latest = session.snapshot()
    return Starlette(routes=[
        Route("/", index),
        Route("/status", status),
        Route("/start", start, methods=["POST"]),
        WebSocketRoute("/ws", socket),
        Mount("/static", StaticFiles(directory=WEB), name="static"),
    ])


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--challenge", default="baseball-sun")
    p.add_argument("--backend", choices=("typesafe", "local"), default="typesafe")
    p.add_argument("--base-url", default=None)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8767)
    p.add_argument("--env-file", default="env.sh")
    p.add_argument("--jev-only", action="store_true")
    return p.parse_args(argv)


def build_racers(args):
    jev = SystemOne(backend=args.backend, base_url=args.base_url, env_file=args.env_file, timeout=90)
    if args.backend == "local":
        import demos.wikiracing.race as race_mod
        from demos.bench.run import local_score_batch

        race_mod.SCORE_BATCH = local_score_batch(jev)  # 32/64/128 from the served max_tokens (GET /v1/models limits)
    racers = {"Jev": jev}
    if not args.jev_only:
        racers["Luna"] = OpenAISystemOne("gpt-5.6-luna", env_file=args.env_file)
        racers["Terra"] = OpenAISystemOne("gpt-5.6-terra", env_file=args.env_file)
        racers["Sol"] = OpenAISystemOne("gpt-5.6-sol", env_file=args.env_file)
    return racers


def main(argv=None):
    args = parse_args(argv)
    challenges = load_challenges()
    spec = challenges[args.challenge]
    racers = build_racers(args)
    session = WikiRace(racers, spec["start"], spec["target"], spec.get("max_hops", 25))
    import uvicorn
    try:
        uvicorn.run(app_factory(session), host=args.host, port=args.port, log_level="info")
    finally:
        session.stop()
        extra = {"challenge": args.challenge, "racers": session.snapshot()["racers"]}
        records = []
        for client in racers.values():
            records.extend(getattr(client, "records", []))
        wall = 0.0
        if session.started_at and session.finished_at:
            wall = session.finished_at - session.started_at
        if records:
            write_summary(artifact_path("wikiracing-summary.json"), records, wall or 0.001, extra=extra)


if __name__ == "__main__":
    main()
