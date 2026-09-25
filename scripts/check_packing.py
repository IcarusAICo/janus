#!/usr/bin/env python
"""Fail before renting a GPU if any row of a mix packs over the config's max_tokens.

`janus train` raises the same error, but only after loading the backbone, which on a rented 35B card is four
minutes and a few dollars. A state+question token count is not a substitute: the packed tree adds a block header
and a leaf per option, so it runs about twice the raw text.

    python scripts/check_packing.py configs/phase4_production/qwen36_a3b_v4.json data/production-v4 [tokenizer]

The optional third argument names a tokenizer to use instead of the config's backbone, for when the backbone itself
is not cached locally (the A3B is only here as the NVFP4 copy). Qwen3.5-4B-Base gives the A3B's pack sizes exactly.

ponytail: re-tokenises the whole mix (about 60 s for 53k rows) instead of caching pack sizes per row. The upgrade,
if it ever matters, is to write the sizes into the mix manifest when the mix is built.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from transformers import AutoTokenizer
from janus.data import load_requests
from janus.packing import pack_request

whole = json.loads(Path(sys.argv[1]).read_text())
config = whole["model"]
name = sys.argv[3] if len(sys.argv) > 3 else config["backbone"]
tok = AutoTokenizer.from_pretrained(name, revision="main" if len(sys.argv) > 3 else config.get("revision", "main"))
kwargs = {k: config[k] for k in ("tree_positions", "tree_block", "score_block") if k in config}
worst, failures = 0, 0
for split in ("train", "dev", "calibration"):
    path = Path(sys.argv[2]) / f"{split}.jsonl"
    if not path.exists():
        continue
    sizes = []
    for request in load_requests(path):
        try:
            sizes.append(pack_request(request, tok, config.get("mode", "tree"), 1 << 20, **kwargs).token_count)
        except ValueError as error:  # a row that cannot be packed at any budget
            print(f"{path}: {error}"); failures += 1
    sizes.sort()
    longest = sizes[-1] if sizes else 0
    # Size group_tokens from THESE numbers, never from a character or raw-text estimate: the packed tree adds a block
    # header and a leaf per option, so it runs about twice the raw state-plus-question text.
    mean = sum(sizes) / len(sizes) if sizes else 0
    rows = max(1, config.get("batch_states", 1))
    print(f"{split:12s} longest {longest:6d} / max_tokens {config['max_tokens']}   mean {mean:6.0f} "
          f"p90 {sizes[int(.9 * len(sizes))] if sizes else 0:6d}   {rows} rows at the mean = {rows * mean:.0f} tokens"
          + (f" against group_tokens {budget}" if (budget := whole.get('group_tokens', 0)) else ""))
    worst = max(worst, longest)
if failures or worst > config["max_tokens"]:
    sys.exit(f"FAIL: {worst} tokens exceeds max_tokens={config['max_tokens']}; raise it or drop the row")
print(f"OK: {worst} <= {config['max_tokens']}")
