"""Headless page recording. Playwright Chromium for web UIs; Xvfb for real Chrome."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "demos" / "artifacts"


def artifact_path(*parts):
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS.joinpath(*parts)


def xvfb_prefix(display=":99", width=1280, height=800):
    """Wrap a command so it runs under Xvfb (needed for Google Flights)."""
    xvfb = shutil.which("xvfb-run")
    if xvfb is None:
        return []
    return [xvfb, "-a", "-s", f"-screen 0 {width}x{height}x24"]


def remux_to_mp4(src, dest):
    """Re-encode a Playwright webm (or any ffmpeg-readable file) to H.264 mp4."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        shutil.copy2(src, dest.with_suffix(Path(src).suffix))
        return dest.with_suffix(Path(src).suffix)
    subprocess.run(
        [ffmpeg, "-y", "-i", str(src), "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", "-an", str(dest)],
        check=True, capture_output=True,
    )
    return dest


def record_page(url, dest_mp4, seconds, *, width=1280, height=800, actions=None, headed=False):
    """Open `url` in Chromium, run optional timed actions, write an mp4.

    `actions` is a list of `{"at_seconds": float, "js": str}` evaluated in the page.
    """
    from playwright.sync_api import sync_playwright

    dest_mp4 = Path(dest_mp4)
    dest_mp4.parent.mkdir(parents=True, exist_ok=True)
    actions = list(actions or [])
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not headed)
        context = browser.new_context(
            viewport={"width": width, "height": height},
            record_video_dir=str(dest_mp4.parent),
            record_video_size={"width": width, "height": height},
        )
        page = context.new_page()
        page.goto(url, wait_until="domcontentloaded")
        elapsed = 0.0
        step = 0.25
        while elapsed < seconds:
            due = [a for a in actions if a["at_seconds"] <= elapsed]
            for action in due:
                actions.remove(action)
                page.evaluate(action["js"])
            page.wait_for_timeout(int(step * 1000))
            elapsed += step
        video = page.video
        page.close()
        webm = Path(video.path()) if video is not None else None
        context.close()
        browser.close()
        if webm is None or not webm.exists():
            raise RuntimeError("Playwright did not produce a video")
        return remux_to_mp4(webm, dest_mp4)


def env_with_keys(env_file="env.sh"):
    """Copy the process env and overlay keys from env.sh without executing it."""
    from janus.remote import load_env_key
    env = os.environ.copy()
    for name in ("TYPESAFE_API_KEY", "OPENAI_API_KEY", "JANUS_SERVER_TOKEN"):
        if env.get(name):
            continue
        try:
            env[name] = load_env_key(name, env_file)
        except Exception:
            continue
    return env
