"""Time our checkpoint on the same request files and card the competitors were timed on (run_competitor.py):
one request at a time through janus.api (the server's Worker, fast path on CUDA), synchronised wall time after
--warmup untimed requests, the same seeded --limit sample. Prints median/p90/mean ms per request.

    python scripts/competitors/time_janus.py CHECKPOINT CALIBRATION DATA.jsonl --device cuda:1 [--limit 300]
"""
import argparse
import json
import random
import statistics
import tempfile
import time
from pathlib import Path

import torch

from janus.api import Janus

p = argparse.ArgumentParser()
p.add_argument("checkpoint"); p.add_argument("calibration"); p.add_argument("data")
p.add_argument("--device", default="cuda:0"); p.add_argument("--limit", type=int); p.add_argument("--warmup", type=int, default=5)
a = p.parse_args()

rows = [json.loads(l) for l in open(a.data)]
if a.limit and a.limit < len(rows):
    rows = random.Random(17).sample(rows, a.limit)  # ponytail: same seeded sample rule as run_competitor.py --limit
with tempfile.TemporaryDirectory() as d:
    Path(d, "model.pt").symlink_to(Path(a.checkpoint).resolve())
    Path(d, "calibration.json").symlink_to(Path(a.calibration).resolve())
    Path(d, "janus_config.json").write_text(json.dumps({"model_id": "janus"}))
    m = Janus(d, a.device, max_tokens=16384, precapture=True)  # as `janus serve` runs
    body = lambda r: (r["state"], {k: {kk: vv for kk, vv in q.items() if kk != "target"} for k, q in r["questions"].items()})
    for r in rows[:a.warmup]:
        m.predict(*body(r))
    times = []
    for r in rows:
        torch.cuda.synchronize(a.device); t0 = time.perf_counter()
        m.predict(*body(r))
        torch.cuda.synchronize(a.device); times.append((time.perf_counter() - t0) * 1e3)
times.sort()
print(json.dumps({"data": a.data, "checkpoint": a.checkpoint, "device": torch.cuda.get_device_name(a.device), "n": len(times),
                  "median_ms": statistics.median(times), "p90_ms": times[int(.9 * len(times))], "mean_ms": statistics.mean(times)}))
