"""Multilingual set (MASSIVE 51 locales + XNLI 15 languages) on a tiny fixture: no downloads."""

from janus.data import state_hash
from janus.synth.public import build_multilingual

NAMES = [f"intent_{i}" for i in range(25)]
SIZES = {"massive": {"train": 5, "dev": 2, "calibration": 2, "test": 4}, "xnli": {"train": 3, "dev": 1, "calibration": 1, "test": 3}}


def _massive(locale, split, n):
    # "shared" utterances appear in both train and test: they must stay in test only.
    return [{"utt": f"shared {i}" if i < 3 else f"{locale} {split} {i}", "intent": i % 25, "locale": locale} for i in range(n)]


def _xnli(lang, split, n):
    return [{"premise": f"{lang} {split} p{i}", "hypothesis": "h" if i < 2 else f"h{i}", "label": i % 3, "lang": lang} for i in range(n)]


def test_tests_are_disjoint_and_massive_is_twenty_way():
    massive = {loc: {s: _massive(loc, s, 12) for s in ("train", "validation", "test")} for loc in ("en-US", "sw-KE")}
    xnli = {lang: {s: _xnli(lang, s, 8) for s in ("train", "validation", "test")} for lang in ("en", "sw")}
    xnli["en"]["train"][0] = dict(xnli["en"]["test"][0])  # a train pair identical to a test pair
    out = build_multilingual(massive, xnli, NAMES, seed=3, sizes=SIZES)
    test = {state_hash(r["state"]) for name in ("test_massive51", "test_xnli15") for r in out[name]}
    official = {state_hash(r["utt"]) for loc in massive.values() for r in loc["test"]}
    for name in ("train", "dev", "calibration"):
        assert out[name] and not {state_hash(r["state"]) for r in out[name]} & (test | official)
    assert len(out["test_massive51"]) == 8 and len(out["test_xnli15"]) == 6 and len(out["train"]) == 16
    for r in out["test_massive51"]:
        q = r["questions"]["massive:intent"]
        assert list(r["questions"]) == ["massive:intent"] and len(q["criteria"]) == 20 and sum(q["target"]) == 1.
        assert "None of these." not in q["criteria"].values() and len(set(q["criteria"].values())) == 20
        assert r["group_id"].split(":")[1] in massive and r["pool"] == "multilingual"
    for r in out["test_xnli15"]:
        assert len(r["questions"]["xnli:relation"]["criteria"]) == 3 and r["group_id"].split(":")[:2] in (["xnli", "en"], ["xnli", "sw"])
