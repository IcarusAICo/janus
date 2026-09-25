"""Pick the request shapes Worker.warm precaptures (janus/precapture.json) from data.

Every data request runs once through a fast-path Worker (the server's path, prefix cache cleared per request) and its
level graph keys (LevelGraphs._shape: depth, batch and length buckets, ancestor buckets) are recorded. `select` then
takes whole requests' key sets greedily until `--coverage` of the requests need no graph outside the chosen keys, and
each chosen request becomes a shape: its kinds, option keys and per-segment token counts, the text replaced by filler
(janus.server.precapture_bodies). `--verify` loads a second Worker with the new file and counts the requests that
still capture a graph.

    PYTHONPATH=. python scripts/precapture_shapes.py --checkpoint runs/publish/hf/janus-0.8b --device cuda:0 \\
        --data data/distill-v2/dev.jsonl data/public-v1/test.jsonl --verify
"""

import argparse
import collections
import json
import time
from pathlib import Path

import torch


def select(key_sets, coverage=0.998):
    """Indices of representative requests whose keys together leave at most 1 - `coverage` of `key_sets` (one set of
    graph keys per request) needing another key. Greedy on distinct key sets: the one that completes the most requests
    per key it adds.
    ponytail: greedy set cover (not optimal), O(distinct sets^2) per step; fine for a few hundred distinct sets."""
    groups = collections.defaultdict(list)
    for i, keys in enumerate(key_sets):
        groups[frozenset(keys)].append(i)
    chosen, picked, covered = set(), [], 0
    while covered < coverage * len(key_sets):
        gain = lambda s: sum(len(ix) for t, ix in groups.items() if not t <= chosen and t <= chosen | s) / len(s - chosen)
        best = max((s for s in groups if not s <= chosen), key=gain)
        chosen |= best
        picked.append(groups[best][0])
        covered = sum(len(ix) for t, ix in groups.items() if t <= chosen)
    return picked


def segment_lengths(packed):
    return torch.bincount(packed.segment_ids, minlength=len(packed.parents)).tolist()


def shape_of(request, pack):
    """The shape of a Request whose filler body (janus.server.precapture_bodies) packs to the same segment lengths:
    counts are adjusted until they match (tokens merge across a boundary now and then)."""
    from janus.schema import Request
    from janus.server import precapture_bodies
    shape = {"state": 1, "questions": [[q.kind, 1, [o.key for o in q.options] if q.kind == "choice" else
                                        [1] * len(q.options) if q.kind == "score" else None] for q in request.questions]}
    target = segment_lengths(pack(request))
    for _ in range(6):
        got = segment_lengths(pack(Request.from_dict(precapture_bodies([shape])[0])))
        if got == target:
            break
        shape["state"] = max(1, shape["state"] + target[0] - got[0])
        segment = 1
        for q in shape["questions"]:  # tree packing: a block segment, then one leaf per option
            q[1] = max(1, q[1] + target[segment] - got[segment])
            if q[0] == "score":
                q[2] = [max(1, m + target[segment + 1 + j] - got[segment + 1 + j]) for j, m in enumerate(q[2])]
            segment += 1 + (len(q[2]) if q[0] != "noul" else 1)
    return shape, got == target


def rows(paths):
    for path in paths:
        for line in open(path):
            r = json.loads(line)
            yield path, {"state": r["state"], "group_id": r.get("group_id", ""),
                         "questions": {k: {kk: v for kk, v in q.items() if kk != "target"} for k, q in r["questions"].items()}}


def run_all(worker, bodies):
    """Per body: (the graph keys its level passes used, graphs captured while it ran); None when it does not pack."""
    from janus.schema import Request
    from janus.server import null_criteria
    graphs = worker.model.backbone.graphs
    run, seen = graphs.run, set()

    def recording(x, pos, valid, mask, states, parent, keep, depth=0):
        key = graphs._shape(x, mask, states, depth)
        if key is not None:
            seen.add((*key, keep))
        return run(x, pos, valid, mask, states, parent, keep, depth)

    graphs.run = recording
    out = []
    for body in bodies:
        try:
            request = Request.from_dict(null_criteria(body))
            packed = worker.pack(request)
        except ValueError:
            out.append(None)
            continue
        worker.cache.clear()
        seen.clear()
        before = len(graphs.graphs)
        worker._run_many([(packed, worker.temperatures(request), worker.key(request), None)])
        out.append((set(seen), len(graphs.graphs) - before, request))
    graphs.run = run
    return out


def main():
    from janus.server import PRECAPTURE, Worker
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="export dir (model.pt, calibration.json)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--coverage", type=float, default=0.998, help="of all requests; 0.998 is at least 0.99 of each source file here")
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--output", default=str(PRECAPTURE))
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    directory = Path(args.checkpoint)
    data = list(rows(args.data))
    load = lambda precapture: Worker(directory / "model.pt", args.device, directory / "calibration.json", max_tokens=args.max_tokens, precapture=precapture)
    worker = load(False)
    results = run_all(worker, [body for _, body in data])
    ok = [r for r in results if r is not None]
    picked = select([keys for keys, _, _ in ok], args.coverage)
    shapes = []
    for i in picked:
        shape, exact = shape_of(ok[i][2], worker.pack)
        shapes.append(shape)
        if not exact:
            print("shape not exact:", shape)
    keys = set().union(*(ok[i][0] for i in picked))
    print(f"{len(ok)} requests, {len(set().union(*(k for k, _, _ in ok)))} distinct graph keys; {len(picked)} shapes, {len(keys)} keys")
    Path(args.output).write_text("[\n" + ",\n".join(json.dumps(s) for s in shapes) + "\n]\n")
    if args.verify:
        del worker
        torch.cuda.empty_cache()
        start = time.monotonic()
        worker = load(True)
        print(f"load with precapture: {time.monotonic() - start:.1f} s, {len(worker.model.backbone.graphs.graphs)} graphs, "
              f"buffer sets {worker.model.backbone.graphs.bytes / 1e9:.2f} GB, reserved {torch.cuda.memory_reserved(args.device) / 1e9:.2f} GB")
        again = run_all(worker, [body for _, body in data])
        for path in args.data:
            mine = [r for (p, _), r in zip(data, again) if p == path and r is not None]
            print(f"{path}: {sum(r[1] == 0 for r in mine) / len(mine):.4f} of {len(mine)} requests capture nothing")


if __name__ == "__main__":
    main()
