"""Evaluation-only conversions of public image multiple-choice sets into Choice requests with image states
(JEV_PUBLIC_IMAGES_V1). Pinned revisions, licences recorded, no training rows; each set gets a test file and a
300-row calibration file disjoint from it (validation or dev rows where the set has them, else a held-out slice)."""

import argparse
import ast
import json
from pathlib import Path
import random
import re

from PIL import Image

from ..data import file_hash, write_json, write_jsonl

DATASET = "JEV_PUBLIC_IMAGES_V1"
MAX_SIDE = 896  # ponytail: photographs and diagrams are shrunk to this long side; enough for a vision encoder's tiles
CAPS = {"scienceqa": 1000, "ai2d": 1000, "mmmu": 900}
CALIBRATION = 300
MARKER = re.compile(r"<image (\d+)>")


def convert_scienceqa(row):
    if row.get("image") is None or row.get("answer") not in range(len(row["choices"])):
        return None
    text = row["question"] + (f"\nContext: {row['hint']}" if row.get("hint") else "")
    return text, [row["image"]], list(row["choices"]), int(row["answer"])


def convert_ai2d(row):
    options = list(row["options"])
    try:
        gold = int(row["answer"])
    except (TypeError, ValueError):
        return None
    if row.get("image") is None or gold not in range(len(options)):
        return None
    return row["question"], [row["image"]], options, gold


def convert_mmmu(row):
    if row.get("question_type") != "multiple-choice":
        return None
    try:
        options = ast.literal_eval(row["options"])
    except (ValueError, SyntaxError):
        return None
    if not isinstance(options, list) or any(not isinstance(o, str) or MARKER.search(o) for o in options):
        return None  # options must be plain text
    letters = [chr(ord("A") + i) for i in range(len(options))]
    if row.get("answer") not in letters:
        return None
    images = [row.get(f"image_{k}") for k in range(1, 8)]
    used = sorted({int(m) for m in MARKER.findall(row["question"])})
    if any(k < 1 or k > 7 or images[k - 1] is None for k in used):
        return None
    if not used:  # no marker: every attached image goes before the text
        used = [k for k in range(1, 8) if images[k - 1] is not None]
    if not used:
        return None
    text = MARKER.sub(lambda m: f"[image:{used.index(int(m.group(1)))}]", row["question"])
    return text, [images[k - 1] for k in used], options, letters.index(row["answer"])


SOURCES = {
    "scienceqa": {"repo": "derek-thomas/ScienceQA", "revision": "f18b0a70359ebfb41f658fd564208d0355b013f4",
                  "files": {"test": "data/test-*.parquet", "calibration": "data/validation-*.parquet"},
                  "converter": convert_scienceqa, "instructions": "Answer the science question about the picture.",
                  "licence": "CC BY-NC-SA 4.0 (original ScienceQA release; the Hub card's metadata says cc-by-sa-4.0)",
                  "licence_source": "https://huggingface.co/datasets/derek-thomas/ScienceQA",
                  "citation": "Lu et al. (2022), Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering"},
    "ai2d": {"repo": "lmms-lab/ai2d", "revision": "c83a9b9692933aff8349157c88a413df9d02c4e5",
             "files": {"test": "data/test-*.parquet"},
             "converter": convert_ai2d, "instructions": "Answer the question about the diagram.",
             "licence": "Not stated on the lmms-lab/ai2d card; AI2 Diagrams is distributed by the Allen Institute for AI for research use",
             "licence_source": "https://huggingface.co/datasets/lmms-lab/ai2d",
             "citation": "Kembhavi et al. (2016), A Diagram Is Worth A Dozen Images"},
    "mmmu": {"repo": "MMMU/MMMU", "revision": "98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68",
             "files": {"test": "*/validation-*.parquet", "calibration": "*/dev-*.parquet"},
             "converter": convert_mmmu, "instructions": "Answer the exam question.",
             "licence": "Apache-2.0", "licence_source": "https://huggingface.co/datasets/MMMU/MMMU",
             "citation": "Yue et al. (2024), MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark"},
}


def load_split(name, ours, cache_dir=".cache/study"):
    from datasets import load_dataset
    spec = SOURCES[name]
    # no_checks: the card declares every split, but only the one file pattern is fetched
    return load_dataset(spec["repo"], data_files={ours: spec["files"][ours]}, split=ours, revision=spec["revision"], cache_dir=cache_dir,
                        verification_mode="no_checks")


def save_image(image, path):
    """Write the image as PNG on a white background with the long side at most MAX_SIDE; returns (width, height)."""
    if "A" in image.getbands():
        flat = Image.new("RGB", image.size, "white")
        flat.paste(image, mask=image.getchannel("A"))
        image = flat
    elif image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, optimize=True)
    return image.size


def convert(name, rows, output, split):
    """Converted Choice rows (image files written under output/images/<name>/) for one source split."""
    spec = SOURCES[name]
    out = []
    for index, row in enumerate(rows):
        converted = spec["converter"](row)
        if converted is None:
            continue
        text, images, options, gold = converted
        stem = f"{name}_{split}_{index}"
        refs, sizes = [], []
        for k, image in enumerate(images):
            path = f"images/{name}/{stem}_{k}.png"
            sizes.append(save_image(image, Path(output) / path))
            refs.append({"path": path})
        keys = [chr(ord("A") + i) for i in range(len(options))]
        out.append({"state": {"text": text, "images": refs},
                    "questions": {f"{name}:answer": {"type": "choice", "instructions": spec["instructions"],
                                                     "criteria": dict(zip(keys, options)),
                                                     "target": [float(i == gold) for i in range(len(options))]}},
                    "group_id": f"{name}:{split}:{index}", "tier": "T1", "family": name, "source": spec["repo"],
                    "image_sizes": sizes})
    return out


def split_rows(test_pool, calibration_pool, cap, seed, name):
    """Calibration takes the other split first, then a held-out slice of the test pool; test is what remains."""
    rng = random.Random(f"{seed}:{name}")
    test_pool, calibration_pool = list(test_pool), list(calibration_pool)
    rng.shuffle(test_pool)
    rng.shuffle(calibration_pool)
    calibration = calibration_pool[:CALIBRATION]
    short = CALIBRATION - len(calibration)
    calibration += test_pool[:short]
    test = test_pool[short:][:cap]
    assert not {r["group_id"] for r in test} & {r["group_id"] for r in calibration}
    return sorted(test, key=lambda r: r["group_id"]), sorted(calibration, key=lambda r: r["group_id"])


def _size_mb(output, rows):
    return round(sum((Path(output) / ref["path"]).stat().st_size for r in rows for ref in r["state"]["images"]) / 1e6, 2)


def prepare_public_images(output, sources=None, seed=17, cache_dir=".cache/study"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"dataset": DATASET, "seed": seed, "caps": CAPS, "calibration_rows": CALIBRATION, "max_side": MAX_SIDE,
                "sources": {}, "skipped": {}}
    for name in sources or list(SOURCES):
        spec = SOURCES[name]
        try:
            pools = {ours: convert(name, load_split(name, ours, cache_dir), output, ours) for ours in spec["files"]}
        except Exception as error:  # download, gating, schema changes
            manifest["skipped"][name] = f"{type(error).__name__}: {error}"[:300]
            continue
        test, calibration = split_rows(pools["test"], pools.get("calibration", []), CAPS[name], seed, name)
        write_jsonl(output / f"test_{name}.jsonl", test)
        write_jsonl(output / f"calibration_{name}.jsonl", calibration)
        kept = {r["group_id"] for r in test + calibration}
        for row in (r for rows in pools.values() for r in rows if r["group_id"] not in kept):
            for ref in row["state"]["images"]:
                (output / ref["path"]).unlink()  # converted but not selected
        manifest["sources"][name] = {"repository": spec["repo"], "revision": spec["revision"], "licence": spec["licence"],
                                     "licence_source": spec["licence_source"], "citation": spec["citation"],
                                     "source_splits": {ours: theirs for ours, theirs in spec["files"].items()},
                                     "converted": {ours: len(rows) for ours, rows in pools.items()},
                                     "counts": {"test": len(test), "calibration": len(calibration)},
                                     "megabytes": {"test": _size_mb(output, test), "calibration": _size_mb(output, calibration)}}
    manifest["files"] = {p.name: file_hash(p) for p in sorted(output.glob("*.jsonl"))}
    write_json(output / "manifest.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default="data/public-images-v1")
    parser.add_argument("--sources", nargs="*", default=None, choices=list(SOURCES))
    parser.add_argument("--cache-dir", default=".cache/study")
    args = parser.parse_args(argv)
    manifest = prepare_public_images(args.output, args.sources, cache_dir=args.cache_dir)
    print(json.dumps({"sources": {k: {"counts": v["counts"], "megabytes": v["megabytes"]} for k, v in manifest["sources"].items()},
                      "skipped": manifest["skipped"]}, indent=2))


if __name__ == "__main__":
    main()
