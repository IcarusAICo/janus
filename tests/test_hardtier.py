"""Hard-tier families: every code-generated gold recomputed independently from the world, probability targets exact,
minimal pairs minimal, the JevBench leakage check, and the GPT pipeline on a fake client (no API calls)."""

import calendar
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
import itertools
import json
from pathlib import Path
from math import ceil, comb
import random

import pytest

from janus.schema import Request
from janus.synth.hardtier import (CODE_FAMILIES, Leakage, TRAIN_NAMES, TRAIN_SURNAMES, V2_CELLS, V2_KINDS, _empirical_world, balance_nouls, code_pair, gold_keys, gpt_item,
                                  gpt_spec, jevbench_request, load_jevbench, make_code_row, routing_gold, shingles, word_diff)

JEV = "demos/jevbench/datasets/public/hard.jsonl"


def _gold(row, name=None):
    q = row["questions"][name] if name else next(iter(row["questions"].values()))
    keys = list(q["criteria"]) if isinstance(q["criteria"], dict) else list(range(len(q["criteria"])))
    return keys[max(range(len(q["target"])), key=q["target"].__getitem__)]


def _target(row):
    q = next(iter(row["questions"].values()))
    return dict(zip(q["criteria"], q["target"]))


def _add_months(d, months):
    y, m = divmod(d.month - 1 + months, 12)
    y, m = d.year + y, m + 1
    return date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


def temporal_expected(w):
    if w["kind"] == "deadline":
        deadline = datetime.fromisoformat(w["deadline"]) + timedelta(hours=w["extension_hours"] - w["zone_deadline"])
        received = datetime.fromisoformat(w["received_local"]) - timedelta(hours=w["zone_submitter"])
        late = (received - deadline).total_seconds() / 60
        return "accepted" if late <= 0 else "accepted_with_late_fee" if late <= w["grace_hours"] * 60 else "rejected"
    if w["kind"] == "parcel":
        l, wd, h = w["dims_cm"]
        billable = ceil(max(w["actual_kg"], l * wd * h / w["divisor"]) * 2 - 1e-9) / 2
        return "band_a" if billable <= 2 else "band_b" if billable <= 5 else "band_c" if billable <= 10 else "band_d" if billable <= 20 else "band_e"
    if w["kind"] == "prorate":
        refund = max(round(w["price"] * (w["days"] - w["used"]) / w["days"] - (w["fee"] if w["fee_applies"] else 0), 2), 0.)
        return f"refund_{refund:,.2f}".replace(",", "").replace(".", "_")
    if w["kind"] == "cap":
        start, new = date.fromisoformat(w["year_start"]), date.fromisoformat(w["new_date"])
        total = sum(a for d, a in w["prior"] if start <= date.fromisoformat(d) < new)
        if total >= w["cap"]:
            return "deny_cap_reached"
        over = total + w["amount"] > w["cap"] if w["strict"] else total + w["amount"] >= w["cap"]
        if over and not w["strict"] and w["cap"] - total - 1 <= 0:
            return "deny_cap_reached"
        return "pay_partial" if over else "pay_full"
    end = _add_months(date.fromisoformat(w["start"]), w["months"])
    event = date.fromisoformat(w["event"])
    return "in_term" if event <= end else "in_grace" if w["grace"] and event <= end + timedelta(days=w["grace"]) else "expired"


def probability_expected(w):
    if w["kind"] == "hypergeometric":
        n, k, d = w["lot"], w["defective"], w["draw"]
        p = [comb(k, i) * comb(n - k, d - i) / comb(n, d) for i in range(d + 1)]
        return {"none": p[0], "one": p[1], "two_or_more": sum(p[2:])}
    if w["kind"] == "stages":
        counts = [0., 0., 0.]
        for outcome in itertools.product((0, 1), repeat=len(w["probs"])):
            pm = 1.
            for fail, p in zip(outcome, w["probs"]):
                pm *= p if fail else 1 - p
            counts[min(2, sum(outcome))] += pm
        all_fail = 1.
        for p in w["probs"]:
            all_fail *= p
        fail = 1 - counts[0] if w["series"] else all_fail
        return {"none": counts[0], "one": counts[1], "two_or_more": counts[2], "fail": fail}
    if w["kind"] == "bayes":
        joint = [c * r for c, r in zip(w["counts"], w["rates"])]
        return {k: j / sum(joint) for k, j in zip(w["keys"], joint)}
    if w["kind"] == "empirical":
        rows = [r for r in w["rows"] if r[1] == w["tier"] and (r[2] >= w["band_edge"]) == w["large"]]
        return {o: sum(r[3] == o for r in rows) / len(rows) for o in ("early", "on_time", "late")}
    if w["kind"] == "table":
        counts = w["table"][w["row"]]
        return {k: c / sum(counts) for k, c in zip(w["keys"], counts)}
    if w["kind"] == "plan_version":
        draw = w["draws"][1] if w["status"] == "in_force" else w["draws"][0]
        return {"reject": 1 - comb(w["lot"] - w["defective"], draw) / comb(w["lot"], draw)}
    n, p, a, b = w["n"], w["p"], w["a"], w["b"]
    mass = [comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)]
    return {"low": sum(mass[:a + 1]), "mid": sum(mass[a + 1:b + 1]), "high": sum(mass[b + 1:])}


def multi_hop_expected(w):
    if w["kind"] == "expense":
        c = w["claim"]
        grade = next((g for i, _, g in w["people"] if i == c["id"]), None)
        if grade is None:
            return "reject_unknown_employee"
        eur = round(c["amount"] * w["rates"][c["date"]], 2)
        if not c["receipt"] and eur > w["receipt_threshold"]:
            return "deny_missing_receipt"
        return "pay_capped" if eur > w["limits"][c["category"]][grade] else "pay_full"
    t = w["ticket"]
    asset = next((a for a in w["assets"] if a[0] == t["serial"]), None)
    if asset is None:
        return "unknown_asset"
    plan = w["plans"][asset[3]]
    if date.fromisoformat(t["date"]) > _add_months(date.fromisoformat(asset[2]), plan["months"]):
        return "not_covered_expired"
    if w["sites"][asset[4]] not in plan["regions"]:
        return "not_covered_region"
    if t["accidental"] and not plan["accidental"]:
        return "not_covered_accidental"
    return "covered"


def routing_expected(w):
    t = w["ticket"]
    if t["security_flag"]:
        return "security"
    if t["plan"] == "enterprise" and t["component"] == "billing":
        return "enterprise_accounts"
    return w["owners"].get(t["component"], "triage")


def tradeoff_expected(w):
    risks = ("low", "medium", "high")
    ok = {k: c for k, c in w["candidates"].items() if risks.index(c["risk"]) <= risks.index(w["max_risk"]) and c["cost"] <= w["budget"]}
    best = min(ok, key=lambda k: [ok[k][o] for o in w["order"]])
    return best


EXPECTED = {"temporal_numeric": temporal_expected, "multi_hop": multi_hop_expected, "routing_hard": routing_expected, "tradeoff": tradeoff_expected}


@pytest.mark.parametrize("family", CODE_FAMILIES)
def test_code_gold_recomputes_from_the_world(family):
    from janus.synth import hardtier
    hardtier._FILLER.update({"_expense_world": (700., 60.), "_entitlement_world": (700., 60.)})  # no tokenizer calibration in the test
    seen = set()
    for seed in range(160):
        made = make_code_row(family, random.Random(seed), TRAIN_NAMES, TRAIN_SURNAMES, ("text", "json")[seed % 2], target_tokens=1800)
        if made is None:
            continue
        row, rationale = made
        Request.from_dict(row)  # targets validate as distributions
        w = row["world"]
        if family == "probability":
            expected = probability_expected(w)
            q = next(iter(row["questions"].values()))
            assert abs(sum(q["target"]) - 1) < 1e-9 and max(q["target"]) <= .85 + 1e-9 and max(q["target"]) >= .55 - 1e-9
            if q["type"] == "noul":
                p_yes, name = q["target"][1], next(iter(row["questions"]))
                assert p_yes == pytest.approx({"hypergeometric": lambda: 1 - expected["none"], "stages": lambda: expected["fail"], "empirical": lambda: expected["late"] if name == "probability:late" else 1 - expected["late"],
                                               "plan_version": lambda: expected["reject"]}[w["kind"]](), abs=1e-9)
            else:
                for key, value in _target(row).items():
                    assert value == pytest.approx(expected[key], abs=1e-9), (w["kind"], key)
        else:
            assert _gold(row) == EXPECTED[family](w) == w["gold"], (seed, rationale)
            if family == "temporal_numeric" and len(row["questions"]) == 2:
                second = list(row["questions"])[1]
                truth = {"temporal:on_time": w["gold"] == "accepted", "temporal:fee": w.get("fee_applies"), "temporal:full": w["gold"] == "pay_full",
                         "temporal:in_term": w["gold"] == "in_term"}[second]
                assert _gold(row, second) == ("true" if truth else "false")
        seen.add(row["group_id"])
    assert len(seen) > 60


def test_minimal_pairs_differ_in_few_words_and_in_gold():
    pairs = 0
    for family in ("temporal_numeric", "routing_hard", "tradeoff"):
        for seed in range(80):
            made, twin = code_pair(family, seed, TRAIN_NAMES, TRAIN_SURNAMES, "text")
            if twin is None:
                continue
            pairs += 1
            a, b = made[0], twin[0]
            assert word_diff(a["state"], b["state"]) <= 8 and gold_keys(a) != gold_keys(b)
            assert [list(q["criteria"]) for q in a["questions"].values()] == [list(q["criteria"]) for q in b["questions"].values()]
            assert _gold(b) == EXPECTED[family](b["world"])
    assert pairs > 30
    assert word_diff("a b c d", "a b x d") == 1 and word_diff("a b c", "a b c d e") == 2


@pytest.mark.skipif(not Path(JEV).exists(), reason="needs demos/jevbench (third-party, not in the public release)")
def test_leakage_check_rejects_jevbench_states_and_shingles():
    requests = load_jevbench(JEV)
    assert len(requests) == 111 and all(r.questions[0].target is not None for r in requests)
    leakage = Leakage(requests)
    assert leakage.hit(requests[3].state) == "state"
    words = requests[3].state.split()
    copied = "Our own scenario starts here. " + " ".join(words[40:52]) + " and continues with new material."
    assert leakage.hit(copied).startswith("shingle:")
    assert leakage.hit("Kestrel Mutual - allowance rules: the dental allowance is 1,500.00 per policy year starting 1 April.") is None
    assert len(shingles("one two three", n=2)) == 2
    with open(JEV) as handle:
        row = json.loads(next(handle))
    assert jevbench_request(row).group_id == f"jevbench:{row['id']}"


def test_gpt_item_reviews_and_pairs_with_a_fake_client(tmp_path, monkeypatch):
    from test_synth import FakeOpenAI
    from janus.synth.openai_client import StructuredCompleter
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    spec = gpt_spec("trap_choice", random.Random(5), "train", TRAIN_NAMES, TRAIN_SURNAMES)
    spec["pair"] = True
    state = f"{spec['org']} refund desk. The customer wrote that the box was never opened, but the seal photo shows it split."

    def luna(user):
        if "Write the example" in user:
            return json.dumps({"instructions": "Is the item eligible?", "options": [{"key": "eligible", "description": "Eligible."}, {"key": "not_eligible", "description": "Not eligible."}, {"key": "escalate", "description": "Escalate."}],
                               "gold_key": "not_eligible", "surface_key": "eligible", "state": state, "rationale": "The seal is split."})
        if "\"gold_key\"" in user and "rationale" not in user:  # twin request
            return json.dumps({"edits": [{"find": "split", "replace": "intact"}], "gold_key": "eligible", "rationale": "seal now intact"})
        return json.dumps({"answer_key": "eligible", "confidence": .7})  # blind miss

    def terra(user):
        return json.dumps({"verdict": "accept", "reason": "gold follows the seal rule"})
    luna_client, terra_client = FakeOpenAI(luna), FakeOpenAI(terra)
    rows, failure = gpt_item(spec, StructuredCompleter(cache_dir=tmp_path / "l", client=luna_client),
                             StructuredCompleter(model="gpt-5.6-terra", cache_dir=tmp_path / "t", client=terra_client), [], Leakage([]), None, random.Random(1))
    assert failure is None and len(rows) == 2 and rows[0]["pair_id"] == rows[1]["pair_id"] == rows[0]["group_id"]
    assert rows[0]["gold_key"] == "not_eligible" and rows[1]["gold_key"] == "eligible" and rows[0]["checks"]["blind_correct"] is False
    assert word_diff(rows[0]["state"], rows[1]["state"]) == 1 and len(terra_client.calls) == 2
    blind_inputs = [c["input"] for c in luna_client.calls if '"instructions"' in c["input"] and '"gold_key"' not in c["input"]]
    assert len(blind_inputs) == 2 and all("rationale" not in u and "gold" not in u for u in blind_inputs)
    assert routing_gold({"security_flag": True, "plan": "free", "component": "auth"}, {}) == "security"


# v2 worlds: an independent recomputation of every gold from the world dict

def _key_date(d):
    return f"day_{d.isoformat().replace('-', '_')}"


def _business_days_after(start, n, holidays):
    d, count = start, 0
    while count < n:
        d += timedelta(days=1)
        count += d.weekday() < 5 and d not in holidays
    return d


def v2_expected(w):
    """(main gold key, noul truth or None)."""
    if w["kind"] == "threshold":
        cutoff = date.fromisoformat(w["cutoff"])
        q = sum(l["value"] * l["k"] for l in w["lines"] if date.fromisoformat(l["date"]) >= cutoff and not l["tag"])
        level = sum(q > e if w["strict"] else q >= e for e in w["edges"])
        assert q == w["q"] and level == w["gold_level"] and w["edge"] in w["edges"] and abs(q - w["edge"]) in (0, 1, 100)  # on the edge or one unit past it
        return None, q > w["edge"] if w["strict"] else q >= w["edge"]
    if w["kind"] == "dst_rest":
        hours = [(datetime.fromisoformat(r["start"]).replace(tzinfo=ZoneInfo(r["zones"][1])).timestamp() - datetime.fromisoformat(r["end"]).replace(tzinfo=ZoneInfo(r["zones"][0])).timestamp()) / 3600
                 for r in w["rests"]]
        assert hours == [r["actual"] for r in w["rests"]]
        return ("none", "one", "two", "three")[sum(h >= w["minimum"] for h in hours)], hours[1] >= w["minimum"]
    if w["kind"] == "leap_days":
        last = date.fromisoformat(w["start"]) + timedelta(days=w["span"] - w["inclusive"])
        return _key_date(last), date.fromisoformat(w["event"]) <= last
    if w["kind"] == "isoweek":
        monday = date.fromisocalendar(w["year"], w["week"], 1)
        return _key_date(monday), date.fromisoformat(w["claim"]) <= _add_months(monday, w["months"]) + timedelta(days=w["repair"])
    if w["kind"] == "bizday":
        due = _business_days_after(date.fromisoformat(w["filed"]), w["n"], {date.fromisoformat(h) for h in w["holidays"]})
        return _key_date(due), date.fromisoformat(w["received"]) <= due
    if w["kind"] == "accrual":
        d0, d1, d2, d3 = (date.fromisoformat(d) for d in w["dates"])
        p, b = w["principal"], w["basis"]
        interest = round((p * w["r1"] * (d1 - d0).days + p * w["r2"] * (d2 - d1).days + (p - w["repay"]) * w["r2"] * (d3 - d2).days) / 100 / b, 2)
        return f"amount_{interest:,.2f}".replace(",", "").replace(".", "_"), interest > w["threshold"]
    if w["kind"] == "balance":
        posted, holds, declined = w["opening"], [], None
        for t in w["transactions"]:
            day = date.fromisoformat(t["date"])
            holds = [h for h in holds if (day - date.fromisoformat(h["date"])).days < w["hold_days"]]
            if t["type"] == "hold":
                holds.append(t)
            elif t["type"] in ("refund", "payment"):
                posted -= t["amount"]
            else:
                holds = [h for h in holds if not (t["type"] == "capture" and h["ref"] == t["ref"])]
                if t["amount"] > w["limit"] - posted - sum(h["amount"] for h in holds):
                    declined = t["id"]
                    break
                posted += t["amount"]
        return declined or "none_declined", declined is None
    if w["kind"] == "archive":
        by_id = {r["id"]: r for r in w["records"]}
        r = by_id.get(w["request"]["id"])
        while r is not None and r["status"] in ("superseded", "duplicate"):
            r = by_id.get(r["pointer"])
        if r is None or r["status"] != "active":
            return "deny_no_active_record", None
        row = w["rename"].get(r["code"], r["code"])
        row = w["equivalencies"].get(row, row)
        maximum, required, waived = w["policy"][row]["max"], w["policy"][row]["prereq"], False
        for a in w["amendments"]:
            if a["code"] == row and date.fromisoformat(a["effective"]) <= date.fromisoformat(w["request"]["date"]):
                maximum = a["max"]
        matching = [f for f in w["footnotes"] if f["tag"] == r["tag"] and f["code"] in (None, row)]
        assert len(matching) <= 1
        for f in matching:
            maximum, waived = f["max"], f["waives_prereq"]
        if required and not r["prereq"] and not waived:
            return "deny_prerequisite", None
        return ("approve_full" if w["request"]["quantity"] <= maximum else "approve_reduced"), None
    raise ValueError(w["kind"])


@pytest.mark.parametrize("family", list(V2_KINDS))
def test_v2_code_gold_recomputes_from_the_world(family):
    from janus.synth import hardtier
    hardtier._FILLER["_archive_world"] = (700., 60.)
    seen, kinds, notes = set(), set(), 0
    for seed in range(320):
        made = make_code_row(family, random.Random(seed), TRAIN_NAMES, TRAIN_SURNAMES, ("text", "json")[seed % 2], target_tokens=2000, kinds=V2_KINDS[family])
        if made is None:
            continue
        row, rationale = made
        Request.from_dict(row)
        w = row["world"]
        kinds.add(w["kind"])
        if family == "probability":
            q = next(iter(row["questions"].values()))
            expected = probability_expected(w)
            assert abs(sum(q["target"]) - 1) < 1e-9 and .55 - 1e-9 <= max(q["target"]) <= .85 + 1e-9
            if q["type"] == "noul":
                name = next(iter(row["questions"]))
                assert q["target"][1] == pytest.approx({"probability:late": expected.get("late"), "probability:on_time": 1 - expected.get("late", 0), "probability:reject": expected.get("reject")}[name], abs=1e-9)
            else:
                for key, value in _target(row).items():
                    assert value == pytest.approx(expected[key], abs=1e-9), (w["kind"], key)
        else:
            gold, truth = v2_expected(w)
            questions = list(row["questions"])
            if gold is not None:
                assert str(_gold(row)) == gold == w["gold"], (seed, rationale)
            if w["kind"] == "threshold":
                q = row["questions"][questions[0]]
                keys = list(q["criteria"]) if isinstance(q["criteria"], dict) else [str(i) for i in range(len(q["criteria"]))]
                assert keys[q["target"].index(1.)] == w["gold"]
                assert w["gold"] == str(w["gold_level"]) if q["type"] == "score" else w["gold"].endswith(f"_{w['q']}") if w["gold"].split("_")[0] in ("eur", "users", "hours", "sites") else True
            if truth is not None:
                assert _gold(row, questions[1]) == ("true" if truth else "false") and w["balance"] == truth
        notes += any(s in hardtier._state_text(row["state"]) for s in (" note: ", "Draft decision (", "pre-filled the form"))
        seen.add(row["group_id"])
    assert len(seen) > 100 and kinds == {k.__name__[1:-6] for k in V2_KINDS[family]}, kinds
    assert notes > len(seen) * .4  # the authority note is drawn at 60% where the world has one


def test_v2_empirical_world_survives_the_leakage_filter_and_threshold_pairs_flip_the_edge():
    leakage = Leakage(load_jevbench(JEV))
    rows = [make_code_row("probability", random.Random(seed), TRAIN_NAMES, TRAIN_SURNAMES, "text", kinds=(_empirical_world,)) for seed in range(200)]
    rows = [r for r in rows if r is not None]
    assert len(rows) > 30 and not any(leakage.hit(row["state"]) for row, _ in rows)
    pairs = 0
    for seed in range(120):
        made, twin = code_pair("temporal_numeric", seed, TRAIN_NAMES, TRAIN_SURNAMES, "text", V2_KINDS["temporal_numeric"][:1])
        if twin is None:
            continue
        pairs += 1
        a, b = made[0], twin[0]
        assert word_diff(a["state"], b["state"]) <= 8 and gold_keys(a) != gold_keys(b) and a["world"]["edges"] == b["world"]["edges"]
        assert (a["world"]["q"] == a["world"]["edge"]) != (b["world"]["q"] == b["world"]["edge"])
    assert pairs > 20


def test_balance_nouls_and_v2_cells():
    noul = lambda t, pair=None: {"questions": {"q": {"type": "noul", "target": [1 - t, t]}}, **({"pair_id": pair} if pair else {})}
    rows = [noul(1.), noul(1.), noul(1., "p"), noul(0., "p"), {"questions": {"q": {"type": "choice", "target": [1., 0.]}}}, noul(1.), noul(0.)]
    kept = balance_nouls(rows)
    golds = [r["questions"]["q"]["target"][1] for r in kept if r["questions"]["q"]["type"] == "noul"]
    assert golds.count(1.) == golds.count(0.) == 2 and len(kept) == 5 and not any(r.get("pair_id") for r in kept)
    assert len(balance_nouls(rows, 3)) == 3
    for cell, spec in V2_CELLS.items():
        if "gold_by_kind" in spec:
            assert set(spec["difficulty"]) <= set(spec["gold_by_kind"])
            golds = [spec["gold_by_kind"][d] for d in spec["difficulty"]]
            assert set(golds) == set(spec["gold_by_kind"].values()) and max(golds.count(g) for g in set(golds)) <= 2 * min(golds.count(g) for g in set(golds)), cell  # every gold requested, within 2:1
        made = gpt_spec(cell, random.Random(1), "train", TRAIN_NAMES, TRAIN_SURNAMES, V2_CELLS, turn=3)
        assert made["difficulty"] == spec["difficulty"][3 % len(spec["difficulty"])] and ("polarity" in made) == ("polarity" in spec)


def test_cost_cap_abort_keeps_the_rows_already_accepted(monkeypatch):
    """A cap hit after a family's last item must not drop the rows it paid for (it once emptied hardtier-v3/train)."""
    import janus.synth.hardtier as h
    monkeypatch.setattr(h, "gpt_item", lambda spec, *a: ([{"group_id": f"g{spec['seed']}", "state": f"s{spec['seed']}", "rationale": "r",
                                                            "checks": {"blind_correct": True}}], None))
    monkeypatch.setattr(h, "_finish_row", lambda row, *a: row)

    class Cap:
        def check(self, what):
            raise RuntimeError("cost exceeded")
    with pytest.raises(RuntimeError) as caught:
        h.generate_gpt_family("multi_hop", 3, "train", 1, TRAIN_NAMES, TRAIN_SURNAMES, [], None, None, None, None, Cap(), 2, [], set(), h.V3_CELLS)
    rows, report = caught.value.partial
    assert len(rows) == 3 and report["accepted"] == 3 and report["aborted_after"] == 3
