"""Image-state datasets: Phase 1 renderings, the public image converters and the paired report. No network."""

import json
from pathlib import Path
import random

from PIL import Image
import pytest

from janus.data import write_json, write_jsonl
from janus.synth.render_images import MAX_SIDE, MIN_SIDE, document, prepare_images, render_row
from janus.synth.worlds import TRAIN_NAMES, ordinal_world, posterior_family, posterior_world, relative_menu_world


def _phase1_rows():
    rng = random.Random(3)
    rows = []
    for style in ("json", "prose"):
        for family, make in (("rel", lambda: relative_menu_world(rng, TRAIN_NAMES, style)),
                             ("ord", lambda: ordinal_world(rng, 4, False, style)),
                             ("post", lambda: posterior_world(rng, posterior_family(2), 2, style))):
            _, request = make()
            rows.append({**request.to_dict(), "tier": "T0", "family": family, "style": style})
    rows.append({"state": "Incident 734581 affected 18 users and lasted 11 minutes. Data loss: no. Workaround available: no.",
                 "questions": rows[2]["questions"], "group_id": "ord:paraphrased", "tier": "T0", "family": "ord", "style": "prose"})
    rows.append({"state": "what is going on, i have entered my passcode and its not working " * 6,
                 "questions": {"banking77:intent": {"type": "choice", "instructions": "Which intent?", "criteria": {"o0": "a", "o1": "b"}, "target": [1., 0.]}},
                 "group_id": "banking77:abc", "tier": "T1", "family": "study", "style": "text"})
    return rows


def _check_shape(row, twin, base):
    state = row["state"]
    assert set(state) == {"text", "images"} and isinstance(state["text"], str) and state["text"].strip()
    assert len(state["images"]) == 1 and set(state["images"][0]) == {"path"}
    assert row["questions"] == twin["questions"] and row["group_id"] == twin["group_id"]
    with Image.open(Path(base) / state["images"][0]["path"]) as image:
        assert image.size == (row["width"], row["height"])
    assert MIN_SIDE <= max(row["width"], row["height"]) <= MAX_SIDE


def test_rendering_is_deterministic_and_carries_the_state(tmp_path):
    rows = _phase1_rows()
    first = [render_row(r, tmp_path / "a") for r in rows]
    second = [render_row(r, tmp_path / "b") for r in rows]
    assert first == second
    for a, b in zip(first, second):
        assert (tmp_path / "a" / a["state"]["images"][0]["path"]).read_bytes() == (tmp_path / "b" / b["state"]["images"][0]["path"]).read_bytes()
    for row, twin in zip(first, rows):
        _check_shape(row, twin, tmp_path / "a")
    styles = {r["render_style"] for r in first}
    assert "note" in styles and styles & {"table", "receipt", "list"}
    # The picture holds the menu; the text keeps only the request sentence.
    rel = first[0]
    assert rel["state"]["text"].startswith("The customer wants") and "$" not in rel["state"]["text"]
    _, title, header, cells = document(rows[0])
    assert header == ["Option", "Price", "Rating", "Distance"] and len(cells) == len(rows[0]["questions"]["rel:pick"]["criteria"])
    assert title.startswith("Order ")
    _, _, header, cells = document(rows[1])
    assert header is None and [c[0] for c in cells] == ["Users affected", "Duration", "Data loss", "Workaround available"]
    assert document(rows[-2])[2] is None and document(rows[-2])[3] == [[rows[-2]["state"]]]  # paraphrase: rendered verbatim


def test_prepare_images_writes_paired_manifest(tmp_path):
    rows = _phase1_rows()
    source = tmp_path / "phase1"
    write_jsonl(source / "train.jsonl", rows[:5])
    write_jsonl(source / "dev.jsonl", rows[5:])
    write_json(source / "manifest.json", {"files": {"train.jsonl": "x", "dev.jsonl": "y"}})
    manifest = prepare_images(source, tmp_path / "out", splits=("train", "dev"))
    assert manifest["counts"]["train"] == {"rel": 2, "ord": 2, "post": 1} and manifest["megabytes"]["total"] > 0
    paired = [json.loads(line) for line in (tmp_path / "out" / "paired.jsonl").read_text().splitlines()]
    assert len(paired) == len(rows)
    for split, twins in (("train", rows[:5]), ("dev", rows[5:])):
        out = [json.loads(line) for line in (tmp_path / "out" / f"{split}.jsonl").read_text().splitlines()]
        assert [r["group_id"] for r in out] == [r["group_id"] for r in twins]
        for entry in (p for p in paired if p["split"] == split):
            assert twins[entry["index"]]["group_id"] == entry["group_id"] == out[entry["index"]]["group_id"]
            assert out[entry["index"]]["state"]["images"][0]["path"] == entry["image"]


@pytest.mark.skipif(not Path("data/phase1-image-v1/test.jsonl").exists(), reason="rendered dataset absent")
def test_shipped_dataset_matches_text_twins():
    for split in ("test", "calibration"):
        text = [json.loads(line) for line in Path(f"data/phase1-v1/{split}.jsonl").read_text().splitlines()]
        image = [json.loads(line) for line in Path(f"data/phase1-image-v1/{split}.jsonl").read_text().splitlines()]
        assert len(text) == len(image)
        for row, twin in random.Random(0).sample(list(zip(image, text)), 60):
            _check_shape(row, twin, "data/phase1-image-v1")


# --- public converters ------------------------------------------------------------------------------


def _picture(colour, mode="RGB", size=(1200, 700)):
    return Image.new(mode, size, colour)


def test_public_converters_row_shape(tmp_path):
    from janus.synth.public_images import CALIBRATION, convert, split_rows
    scienceqa = convert("scienceqa", [
        {"image": _picture("red", "RGBA"), "question": "Which is it?", "choices": ["a", "b", "c"], "answer": 2, "hint": "Look."},
        {"image": None, "question": "text only", "choices": ["a", "b"], "answer": 0, "hint": ""}], tmp_path, "test")
    assert len(scienceqa) == 1
    row = scienceqa[0]
    assert row["state"] == {"text": "Which is it?\nContext: Look.", "images": [{"path": "images/scienceqa/scienceqa_test_0_0.png"}]}
    q = row["questions"]["scienceqa:answer"]
    assert q["type"] == "choice" and q["criteria"] == {"A": "a", "B": "b", "C": "c"} and q["target"] == [0., 0., 1.]
    assert row["group_id"] == "scienceqa:test:0" and row["tier"] == "T1"
    with Image.open(tmp_path / row["state"]["images"][0]["path"]) as image:
        assert max(image.size) == 896 and image.mode == "RGB"  # shrunk, alpha flattened
    ai2d = convert("ai2d", [{"image": _picture("blue", "L", (300, 200)), "question": "Q", "options": ["x", "y"], "answer": "1"},
                            {"image": _picture("blue"), "question": "Q", "options": ["x", "y"], "answer": "7"}], tmp_path, "test")
    assert len(ai2d) == 1 and ai2d[0]["questions"]["ai2d:answer"]["target"] == [0., 1.] and ai2d[0]["image_sizes"] == [(300, 200)]
    blank = {f"image_{k}": None for k in range(1, 8)}
    mmmu = convert("mmmu", [
        {**blank, "image_1": _picture("green"), "image_2": _picture("black"), "question": "Compare <image 2> with <image 1>.",
         "options": "['p', 'q', 'r', 's']", "answer": "B", "question_type": "multiple-choice"},
        {**blank, "image_1": _picture("green"), "question": "Open <image 1>", "options": "[]", "answer": "x", "question_type": "open"},
        {**blank, "image_1": _picture("green"), "question": "No marker", "options": "['p', 'q']", "answer": "A", "question_type": "multiple-choice"},
        {**blank, "image_1": _picture("green"), "question": "<image 1>", "options": "['<image 2>', 'q']", "answer": "A", "question_type": "multiple-choice"},
        {**blank, "question": "<image 3> missing", "options": "['p', 'q']", "answer": "A", "question_type": "multiple-choice"}], tmp_path, "calibration")
    assert [r["group_id"] for r in mmmu] == ["mmmu:calibration:0", "mmmu:calibration:2"]
    assert mmmu[0]["state"]["text"] == "Compare [image:1] with [image:0]." and len(mmmu[0]["state"]["images"]) == 2
    assert mmmu[0]["questions"]["mmmu:answer"]["target"] == [0., 1., 0., 0.]
    assert mmmu[1]["state"]["text"] == "No marker" and len(mmmu[1]["state"]["images"]) == 1
    # Calibration takes the other split first, then a held-out slice of the test pool; never the same row twice.
    pool = [{"group_id": f"s:test:{i}"} for i in range(500)]
    other = [{"group_id": f"s:calibration:{i}"} for i in range(100)]
    test, calibration = split_rows(pool, other, 350, 17, "s")
    assert len(calibration) == CALIBRATION and len(test) == 300  # 500 - 200 held out for calibration
    assert len(split_rows(pool, other, 250, 17, "s")[0]) == 250  # the cap binds
    assert sum(r["group_id"].startswith("s:calibration") for r in calibration) == 100
    assert not {r["group_id"] for r in test} & {r["group_id"] for r in calibration}
    assert split_rows(pool, other, 350, 17, "s") == (test, calibration)


def test_paired_report(tmp_path):
    from janus.phase4_images_report import paired, report
    def rows(flip):
        out = []
        for i in range(10):
            family = "rel" if i < 6 else "ord"
            correct = (i % 2 == 0) != (flip and i < 6)
            p = [.9, .1] if correct else [.2, .8]
            out.append({"group_id": f"{family}:{i}", "question_id": "q", "family": family, "target": [1., 0.],
                        "probabilities": p, "nll": 0.1 if correct else 1.6})
        return out
    arm = tmp_path / "arm"
    write_jsonl(arm / "text" / "predictions.jsonl", rows(False))
    write_jsonl(arm / "image" / "predictions.jsonl", rows(True) + [{"group_id": "rel:99", "question_id": "q", "family": "rel",
                                                                    "target": [1., 0.], "probabilities": [1., 0.], "nll": 0.}])
    table = paired(arm / "text" / "predictions.jsonl", arm / "image" / "predictions.jsonl")
    assert table["all"]["count"] == 10  # the unmatched image row is dropped
    assert table["rel"]["text_accuracy"] == pytest.approx(.5) and table["rel"]["image_accuracy"] == pytest.approx(.5)
    assert table["ord"]["image_minus_text_accuracy"] == 0 and table["ord"]["text_nll"] == pytest.approx(.85)
    write_json(arm / "public_ai2d" / "metrics.json", {"raw": {"accuracy": .5, "nll": 1.}, "calibrated": {"nll": .9}})
    text = report(tmp_path, arms=("arm",))
    assert "| rel | 6 | 0.500 | 0.500 | +0.000 |" in text and "| arm | n/a | 0.500 / 1.000 / 0.900 | n/a |" in text
