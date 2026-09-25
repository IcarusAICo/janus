import json


def test_prepare_mix_concatenates_with_provenance_and_disjointness(tmp_path):
    from janus.data import synthetic_requests, write_jsonl
    from janus.synth.mix import prepare_mix
    def dump(root, name, rows):
        write_jsonl(root / name, rows)
    p1, pub, t3 = tmp_path / "p1", tmp_path / "pub", tmp_path / "t3"
    for root in (p1, pub, t3):
        root.mkdir()
    a = [dict(r.to_dict(), tier="T0", family="rel") for r in synthetic_requests(6, seed=1)]
    b = [dict(r.to_dict(), tier="T1", family="massive", source="AmazonScience/massive") for r in synthetic_requests(6, seed=2)]
    c = [dict(r.to_dict(), tier="T3", family="routing", cell="routing_text") for r in synthetic_requests(4, seed=3)]
    d = [dict(r.to_dict(), tier="T4", family="routing", cell="routing_text") for r in synthetic_requests(2, seed=4)]
    dump(p1, "train.jsonl", a[:4]); dump(p1, "dev.jsonl", a[4:5]); dump(p1, "calibration.jsonl", a[5:])
    dump(pub, "train.jsonl", b[:4]); dump(pub, "dev.jsonl", b[4:5]); dump(pub, "calibration.jsonl", b[5:])
    dump(t3, "routing_text.jsonl", c + d)
    (p1 / "manifest.json").write_text("{}"); (pub / "manifest.json").write_text("{}"); (t3 / "manifest.json").write_text("{}")
    manifest = prepare_mix(tmp_path / "mix", phase1=p1, public=pub, t3=t3, evaluation_files=())
    assert manifest["counts"]["train_B"] == 8 and manifest["counts"]["train_C"] == 12
    assert manifest["tiers"]["train_C"] == {"T0": 4, "T1": 4, "T3": 4}
    rows = [json.loads(l) for l in (tmp_path / "mix" / "train_C.jsonl").read_text().splitlines()]
    assert all(r.get("source") for r in rows) and all(r["tier"] != "T4" for r in rows)
