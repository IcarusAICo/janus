"""Headless recording of the Doom HUD, including a mid-run standing-order change."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from demos.common.record import artifact_path, record_page

ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=float, default=90)
    p.add_argument("--script", default=str(Path(__file__).parent / "scripts" / "orders_midrun.json"))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--backend", default="typesafe")
    p.add_argument("--base-url", default=None)
    p.add_argument("--output", default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    events = json.loads(Path(args.script).read_text()).get("events", [])
    cmd = [sys.executable, "-m", "demos.doom.server", "--backend", args.backend,
           "--host", args.host, "--port", str(args.port)]
    if args.base_url:
        cmd.extend(["--base-url", args.base_url])
    proc = subprocess.Popen(cmd, cwd=ROOT)
    url = f"http://{args.host}:{args.port}/"
    try:
        for _ in range(50):
            time.sleep(0.2)
            try:
                import urllib.request
                urllib.request.urlopen(url, timeout=1)
                break
            except Exception:
                if proc.poll() is not None:
                    raise RuntimeError("Doom server exited before it became ready")
        actions = []
        for event in events:
            orders = json.dumps(event["orders"])
            actions.append({
                "at_seconds": float(event["at_seconds"]),
                "js": (
                    f"document.getElementById('orders').value = {orders};"
                    "fetch('/orders', {method:'POST', headers:{'content-type':'application/json'}, "
                    f"body: JSON.stringify({{orders: {orders}}})}});"
                ),
            })
        dest = Path(args.output) if args.output else artifact_path("doom.mp4")
        record_page(url, dest, args.seconds, actions=actions)
        print(dest)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
