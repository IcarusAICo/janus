"""Teacher soft labels: run a trained checkpoint over a jsonl of requests and write, per request line, the
teacher's temperature-scaled distribution for every question (`teacher` field). Resumable: lines already in the
output are skipped. Training rows are allowed here on purpose (the overlap guard in `evaluate` is for scores, not
for labels).

    python scripts/teacher_label.py CHECKPOINT CALIBRATION.json IN.jsonl OUT.jsonl [--device cuda:0] [--start N --stop M]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

from janus.data import load_requests
from janus.packing import pack_request
from janus.training import load_checkpoint

p = argparse.ArgumentParser()
p.add_argument("checkpoint"); p.add_argument("calibration"); p.add_argument("data"); p.add_argument("output")
p.add_argument("--device", default="cuda:0"); p.add_argument("--start", type=int, default=0); p.add_argument("--stop", type=int); p.add_argument("--max-tokens", type=int, default=16384)
a = p.parse_args()

# ponytail: one global temperature; the per-family fits in calibration.json go as low as 0.07 and would make the
# teacher sharper than the gold labels it is mixed with.
T = json.loads(Path(a.calibration).read_text())["temperature"]
lines = Path(a.data).read_text().splitlines()
requests = load_requests(a.data)
assert len(lines) == len(requests)
out = Path(a.output)
done = a.start + (sum(1 for _ in out.open()) if out.exists() else 0)
stop = a.stop or len(requests)
model, _ = load_checkpoint(a.checkpoint, a.device, max_tokens=a.max_tokens, batch_states=1)
start = time.time()
with out.open("a") as f, torch.no_grad():
    for i in range(done, stop):
        r = requests[i]
        pack = pack_request(r, model.tokenizer, model.packing_mode, model.config.max_tokens, **model.packing_kwargs)
        logits = model.forward_many([pack])[0]
        row = json.loads(lines[i])
        row["teacher"] = [(z.float().cpu() / T).softmax(-1).tolist() for z in logits]
        assert len(row["teacher"]) == len(row["questions"])
        f.write(json.dumps(row) + "\n")
        if i % 500 == 0:
            f.flush()
            rate = (i - done + 1) / (time.time() - start)
            print(f"{i}/{len(requests)} {rate:.1f} req/s eta {(stop - i) / rate / 3600:.2f} h", file=sys.stderr, flush=True)
print("TEACHER_DONE", file=sys.stderr)
