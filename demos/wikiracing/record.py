"""Record the wikiracing UI. Starts the server, clicks Start race, writes an mp4."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from demos.common.record import artifact_path, record_page

ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--challenge", default="baseball-sun")
    p.add_argument("--seconds", type=float, default=120)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8767)
    p.add_argument("--backend", default="typesafe")
    p.add_argument("--base-url", default=None)
    p.add_argument("--jev-only", action="store_true")
    p.add_argument("--output", default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cmd = [sys.executable, "-m", "demos.wikiracing.server", "--challenge", args.challenge,
           "--backend", args.backend, "--host", args.host, "--port", str(args.port)]
    if args.base_url:
        cmd.extend(["--base-url", args.base_url])
    if args.jev_only:
        cmd.append("--jev-only")
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
                    raise RuntimeError("wikiracing server exited before it became ready")
        dest = Path(args.output) if args.output else artifact_path("wikiracing.mp4")
        record_page(url, dest, args.seconds, actions=[{
            "at_seconds": 1.0,
            "js": "fetch('/start', {method:'POST'})",
        }])
        print(dest)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
