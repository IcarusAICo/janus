"""Four-way race. Wikipedia pages are fetched once and shared across models."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import threading
import time

from demos.wikiracing.race import take_hop
from demos.wikiracing.wikipedia import extract_article_links, fetch_article

CHALLENGES = Path(__file__).with_name("challenges.json")


def load_challenges():
    return json.loads(CHALLENGES.read_text())


@dataclass
class RacerView:
    name: str
    title: str
    path: list
    hops: int = 0
    done: bool = False
    failed: str | None = None
    latency_ms: float = 0
    top: list = field(default_factory=list)
    stage: str = ""


class WikiRace:
    def __init__(self, racers, start, target, max_hops=25):
        self.racers = racers  # name -> client
        self.start = start
        self.target = target
        self.max_hops = max_hops
        self.cache = {}
        self.views = {name: RacerView(name=name, title=start, path=[start]) for name in racers}
        self.started_at = None
        self.finished_at = None
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def snapshot(self):
        with self._lock:
            return {
                "start": self.start,
                "target": self.target,
                "elapsed_ms": int(((self.finished_at or time.perf_counter()) - (self.started_at or time.perf_counter())) * 1000),
                "racers": {name: asdict(view) for name, view in self.views.items()},
            }

    def stop(self):
        self._stop.set()

    def _page(self, title):
        if title not in self.cache:
            canonical, html = fetch_article(title)
            links = extract_article_links(html, canonical)
            self.cache[title] = (canonical, html, links)
        return self.cache[title]

    def run(self, on_update=None):
        self.started_at = time.perf_counter()
        target = self.target
        while not self._stop.is_set():
            pending = [name for name, view in self.views.items()
                       if not view.done and view.failed is None and view.hops < self.max_hops]
            if not pending:
                break
            for name in pending:
                view = self.views[name]
                canonical, html, links = self._page(view.title)
                view.title = canonical
                if canonical.replace("_", " ").lower() == target.replace("_", " ").lower():
                    view.done = True
                    continue
                try:
                    picked, hop = take_hop(self.racers[name], view.title, target, view.path, links)
                except Exception as exc:
                    view.failed = str(exc)
                    continue
                view.path.append(picked.title)
                view.title = picked.title
                view.hops += 1
                view.latency_ms = hop.latency_ms
                view.top = hop.top
                view.stage = hop.stage
                if picked.title.replace("_", " ").lower() == target.replace("_", " ").lower():
                    view.done = True
                if on_update:
                    on_update(self.snapshot())
        for view in self.views.values():
            if not view.done and view.failed is None:
                view.failed = "max hops"
        self.finished_at = time.perf_counter()
        if on_update:
            on_update(self.snapshot())
        return self.snapshot()
