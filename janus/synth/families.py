"""Phase 4 breadth families. Three code-generated task families (tier T0) and two GPT-5.6 top-up cells (tier T3).

T0: `retrieval` (a query and 4 to 12 passages; pick the one that answers it, rerank two, grade one), `evidence` (a claim
against a record set: supported, contradicted or not addressed) and `record_match` (which of K rendered records is the
same person as a probe record). Gold is computed from a hidden world; prose, JSON and table renderings of the same
world; every Choice offers a `none` option and the gold is `none` at a controlled rate. Test splits use organisation
and family names never seen in train/dev/calibration. `python -m janus.synth.families --output data/families-v1`.

T3: `rubric_grading` and `tool_argument_none`, generated through `generate.run_pilot` unchanged (Luna generates and
checks, Terra second-checks); `python -m janus.synth.families --t3-output data/breadth-v2/t3-topup`.
"""

import argparse
from collections import Counter
from datetime import date, timedelta
import json
from pathlib import Path
import random
import statistics

from ..data import assert_disjoint, file_hash, load_requests, state_hash, write_json, write_jsonl
from ..packing import pack_request
from ..schema import Request
from .worlds import SENDER_POOL, TEST_NAMES, TRAIN_NAMES, world_id

DATASET = "JEV_FAMILIES_V1"
FAMILIES = ("retrieval", "evidence", "record_match")
STYLES = ("prose", "json", "table")
QWEN35 = ("Qwen/Qwen3.5-4B-Base", "1001bb4d826a52d1f399e183466143f4da7b741b")
MAX_TOKENS = 2048
EXCLUDE = ("data/phase1-v1", "data/public-v1", "data/t3-pilot", "data/t3-volume-v1", "data/rank-v1", "data/cardinality-v1")
NONE_RATE = .2

SURNAMES = ("Abara", "Bennett", "Castro", "Dimitrov", "Eriksen", "Fontaine", "Garza", "Hoffmann", "Ibrahim", "Jensen",
            "Kowalski", "Lindqvist", "Moreau", "Nakamura", "Okafor", "Petrov", "Quintero", "Rahman", "Silva", "Tanaka",
            "Ueda", "Varga", "Walsh", "Xu", "Yilmaz", "Zimmer", "Andersen", "Brooks", "Chandra", "Dubois",
            "Ferreira", "Gruber", "Haddad", "Iversen", "Jablonski", "Keller", "Lopez", "Mbeki", "Novak", "Osei")
TRAIN_SURNAMES, TEST_SURNAMES = SURNAMES[:30], SURNAMES[30:]
ORG_KINDS = ("Library", "Clinic", "Depot", "Studio", "Bakery", "Garage", "Gallery", "Pharmacy")
DATE_STYLES = ("iso", "long", "us")


def fmt_date(d, style):
    return {"iso": d.isoformat(), "long": f"{d.day} {d.strftime('%B')} {d.year}", "us": d.strftime("%m/%d/%Y")}[style]


def _text(value):
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)


# ---------------------------------------------------------------------------------------------------------------------
# retrieval: query, passages with a code-known relevance level each (3 answers, 2 near miss, 1 partial, 0 unrelated)

# attribute: (query templates, answer templates (with the value), near-miss templates (entity and topic, no value))
ATTRIBUTES = {
    "opening time": (("What time does {e} open?", "When does {e} open in the morning?", "{e} opening time"),
                     ("{e} opens at {v} every weekday.", "Doors at {e} open at {v}.", "Weekday opening time for {e}: {v}."),
                     ("{e} lists its opening hours on the front door.", "The opening time of {e} changed last spring.")),
    "phone extension": (("What is the phone extension for {e}?", "Which extension reaches {e}?", "{e} extension number"),
                        ("Call extension {v} to reach {e}.", "{e} answers on extension {v}.", "Phone extension for {e}: {v}."),
                        ("{e} can be reached by phone during office hours.", "The extension list for {e} is kept at reception.")),
    "floor": (("Which floor is {e} on?", "On what floor is {e} located?", "{e} floor number"),
              ("{e} is on floor {v}.", "You will find {e} on floor {v} of the building.", "Floor {v} houses {e}."),
              ("{e} moved to a different floor last year.", "Ask at the desk for the floor of {e}.")),
    "manager": (("Who manages {e}?", "Who is the manager of {e}?", "{e} manager"),
                ("{v} manages {e}.", "The manager of {e} is {v}.", "{e} is run by {v}."),
                ("{e} appointed a new manager last year.", "The manager of {e} was not available for comment.")),
    "founding year": (("When was {e} founded?", "In which year did {e} open?", "{e} founding year"),
                      ("{e} was founded in {v}.", "{e} opened its doors in {v}.", "Since {v}, {e} has served the neighbourhood."),
                      ("{e} celebrates its founding every spring.", "Nobody at {e} remembers the exact founding date.")),
    "headcount": (("How many people work at {e}?", "What is the headcount of {e}?", "{e} number of staff"),
                  ("{e} employs {v} people.", "The staff of {e} numbers {v}.", "{v} people work at {e}."),
                  ("{e} is hiring more staff this year.", "The headcount of {e} is reviewed each quarter."))}
RELEVANCE_LEVELS = ["Unrelated: the passage is about a different entity and a different topic",
                    "Partial: the passage is about the entity but a different topic, or about the topic but a different entity",
                    "Near miss: the passage is about the entity and the topic but does not give the answer",
                    "Answers: the passage states the answer to the query"]


def _value(attribute, rng, surnames):
    if attribute == "opening time":
        return f"{rng.randint(6, 11)}:{rng.choice(('00', '30'))}"
    if attribute == "phone extension":
        return str(rng.randint(100, 999))
    if attribute == "floor":
        return str(rng.randint(1, 12))
    if attribute == "manager":
        return f"{rng.choice(SENDER_POOL)} {rng.choice(surnames)}"
    if attribute == "founding year":
        return str(rng.randint(1950, 2020))
    return str(rng.randint(3, 240))


def retrieval_world(rng, names, surnames, style):
    kind = rng.choice(ORG_KINDS)
    name, *other_names = rng.sample(names, 4)
    entity = f"{name} {kind}"
    others = [f"{n} {rng.choice(ORG_KINDS)}" for n in other_names]
    if rng.random() < .5:  # a namesake with another kind is the hardest entity distractor
        others[0] = f"{name} {rng.choice([k for k in ORG_KINDS if k != kind])}"
    attribute = rng.choice(list(ATTRIBUTES))
    other_attributes = [a for a in ATTRIBUTES if a != attribute]
    k = rng.randint(4, 12)
    value = _value(attribute, rng, surnames)
    plan = [(3, entity, attribute, value)] if rng.random() >= NONE_RATE else []
    plan += [(2, entity, attribute, None)] * rng.randint(1, 2)
    for i in range((k - len(plan)) // 2):
        if i % 2 == 0:
            plan.append((1, entity, rng.choice(other_attributes), None))
        else:
            plan.append((1, rng.choice(others), attribute, _value(attribute, rng, surnames)))
    while len(plan) < k:
        plan.append((0, rng.choice(others), rng.choice(other_attributes), None))
    rng.shuffle(plan)

    def sentence(e, a, v):
        _, answers, misses = ATTRIBUTES[a]
        if v is None and a != attribute:  # any other attribute always gets a value; only the query attribute has near misses
            v = _value(a, rng, surnames)
        return rng.choice(answers if v is not None else misses).format(e=e, v=v)
    passages, levels = [], []
    for level, e, a, v in plan:
        text = sentence(e, a, v)
        if rng.random() < .4:  # a second sentence on a third attribute of the passage's own entity never changes the level
            extra = rng.choice([x for x in other_attributes if x != a])
            text = f"{text} {sentence(e, extra, None)}" if rng.random() < .5 else f"{sentence(e, extra, None)} {text}"
        passages.append(text)
        levels.append(level)
    query = rng.choice(ATTRIBUTES[attribute][0]).format(e=entity)
    gold = levels.index(3) if 3 in levels else None
    i, j = rng.choice([(a, b) for a in range(k) for b in range(k) if levels[a] != levels[b]])
    probe = rng.randrange(k)
    world = {"family": "retrieval", "style": style, "entity": entity, "attribute": attribute, "value": value, "query": query,
             "passages": passages, "levels": levels, "gold": gold, "rerank": [i, j], "probe": probe}
    if style == "json":
        state = {"query": query, "results": [{"id": n + 1, "text": t} for n, t in enumerate(passages)]}
    elif style == "table":
        state = f"Query: {query}\n\n| # | passage |\n| --- | --- |\n" + "\n".join(f"| {n + 1} | {t} |" for n, t in enumerate(passages))
    else:
        state = f"Query: {query}\n\nResults:\n" + "\n".join(f"[{n + 1}] {t}" for n, t in enumerate(passages))
    keys = [f"p{n + 1}" for n in range(k)] + ["none"]
    raw = {"state": state, "group_id": world_id("retrieval", world), "questions": {
        "retrieval:pick": {"type": "choice", "instructions": rng.choice((
            "Which passage answers the query?", "Select the passage that answers the query, or none if no passage does.",
            "Which result contains the answer to the query?")),
            "criteria": {**{f"p{n + 1}": t for n, t in enumerate(passages)}, "none": "No passage answers the query."},
            "target": [float(key == ("none" if gold is None else f"p{gold + 1}")) for key in keys]},
        "retrieval:rerank": {"type": "noul", "instructions": f"Is passage {i + 1} more relevant to the query than passage {j + 1}?",
                             "target": [float(levels[i] < levels[j]), float(levels[i] > levels[j])]},
        "retrieval:relevance": {"type": "score", "instructions": f"How relevant is passage {probe + 1} to the query?",
                                "criteria": RELEVANCE_LEVELS, "target": [float(n == levels[probe]) for n in range(4)]}}}
    return world, Request.from_dict(raw)


# ---------------------------------------------------------------------------------------------------------------------
# evidence: a claim against a shipment record set; supported, contradicted, or not addressed

STATUSES = ("delivered", "in transit", "returned", "held at customs")
VERDICTS = {"supported": "Supported: the records establish that the claim is true.",
            "contradicted": "Contradicted: the records establish that the claim is false.",
            "not_addressed": "Not addressed: the records lack the information needed to decide the claim."}
COMPARE_TEXT = {"quantity": "{a} has a larger quantity than {b}.", "weight_kg": "{a} is heavier than {b}.",
                "shipped": "{a} was shipped before {b}.", "delivered": "{a} was delivered after {b}."}
TOTAL_TEXT = {"quantity": "The total quantity across all shipments exceeds {n}.", "weight_kg": "All shipments together weigh more than {n} kg."}


def _records(rng, names):
    n = rng.randint(3, 6)
    ids, quantities = rng.sample(range(1000, 9999), n), rng.sample(range(1, 500), n)
    weights = [w / 10 for w in rng.sample(range(5, 999), n)]
    shipped = [date(2026, 1, 1) + timedelta(days=d) for d in rng.sample(range(300), n)]
    return [{"id": f"SH-{i}", "customer": f"{c} {rng.choice(ORG_KINDS)}", "quantity": q, "weight_kg": w, "shipped": s.isoformat(),
             "delivered": (s + timedelta(days=rng.randint(1, 20))).isoformat(), "status": rng.choice(STATUSES)}
            for i, c, q, w, s in zip(ids, rng.sample(names, n), quantities, weights, shipped)]


def _equals_text(subject, field, value, date_style):
    if field == "quantity":
        return f"{subject} has a quantity of {value}."
    if field == "weight_kg":
        return f"{subject} weighs {value} kg."
    if field == "status":
        return f"{subject} is {value}."
    return f"{subject} was {field} on {fmt_date(date.fromisoformat(value), date_style)}."


def _near_miss(rng, field, value):
    """A value that is close to `value` but differs from it."""
    while True:
        if field == "quantity":
            digits = str(value)
            swapped = int(digits[1] + digits[0] + digits[2:]) if len(digits) > 1 and digits[0] != digits[1] else None
            candidate = rng.choice([value + rng.randint(1, 3), max(1, value - rng.randint(1, 3))] + ([swapped] if swapped else []))
        elif field == "weight_kg":
            candidate = round(value + rng.choice((-1, 1)) * rng.randint(1, 20) / 10, 1)
        elif field == "status":
            candidate = rng.choice(STATUSES)
        else:
            d = date.fromisoformat(value)
            swapped = date(d.year, d.day, d.month) if d.day <= 12 else None
            candidate = rng.choice([d + timedelta(days=rng.choice((-3, -2, -1, 1, 2, 3)))] + ([swapped] if swapped else [])).isoformat()
        if candidate != value and (field != "weight_kg" or candidate > 0):
            return candidate


def evidence_claim(rng, records, names, surnames):
    """(label, claim text, structured claim). Dates in the claim use their own style, not the records'."""
    date_style = rng.choice(DATE_STYLES)
    ref = rng.choice(records)
    by_name = rng.random() < .5
    subject = f"The shipment for {ref['customer']}" if by_name else f"Shipment {ref['id']}"
    label = rng.choice(list(VERDICTS))
    if label == "not_addressed":
        mode = rng.choice(("missing_id", "missing_customer", "absent_field"))
        field = rng.choice(("quantity", "weight_kg", "shipped", "delivered", "status"))
        if mode == "missing_id":
            digits = ref["id"][3:]
            candidate = f"SH-{digits[1]}{digits[0]}{digits[2:]}"
            missing = candidate if candidate not in {r["id"] for r in records} else f"SH-{rng.choice([i for i in range(1000, 9999) if f'SH-{i}' not in {r['id'] for r in records}])}"
            value = ref[field] if field != "status" else rng.choice(STATUSES)
            return label, _equals_text(f"Shipment {missing}", field, value, date_style), {"kind": "equals", "mode": mode, "subject": missing, "field": field, "value": value}
        if mode == "missing_customer":
            customer = f"{rng.choice([n for n in names if n not in {r['customer'].split()[0] for r in records}])} {rng.choice(ORG_KINDS)}"
            value = ref[field] if field != "status" else rng.choice(STATUSES)
            return label, _equals_text(f"The shipment for {customer}", field, value, date_style), {"kind": "equals", "mode": mode, "subject": customer, "field": field, "value": value}
        absent = rng.choice(("carrier", "driver", "invoice", "insured"))
        person = f"{rng.choice(SENDER_POOL)} {rng.choice(surnames)}"
        text = {"carrier": f"{subject} was carried by {rng.choice(names)} Freight.", "driver": f"The driver for {subject[0].lower() + subject[1:]} was {person}.",
                "invoice": f"{subject} has invoice number {rng.randint(10000, 99999)}.", "insured": f"{subject} is insured for {rng.randint(1, 50) * 100} euros."}[absent]
        return label, text, {"kind": "absent_field", "mode": mode, "subject": ref["id"], "field": absent}
    kind = rng.choice(("equals", "compare", "total"))
    if kind == "equals":
        field = rng.choice(("quantity", "weight_kg", "shipped", "delivered", "status"))
        value = ref[field] if label == "supported" else _near_miss(rng, field, ref[field])
        return label, _equals_text(subject, field, value, date_style), {"kind": kind, "subject": ref["id"], "field": field, "value": value}
    if kind == "compare":
        field = rng.choice(list(COMPARE_TEXT))
        other = rng.choice([r for r in records if r is not ref and r[field] != ref[field]] or [None])
        if other is None:  # every other record ties on this field (delivered dates can): fall back to a total claim
            kind = "total"
        else:
            truth = ref[field] > other[field] if field in ("quantity", "weight_kg", "delivered") else ref[field] < other[field]
            a, b = (ref, other) if truth == (label == "supported") else (other, ref)
            subjects = {r["id"]: (f"the shipment for {r['customer']}" if by_name else f"shipment {r['id']}") for r in (a, b)}
            text = COMPARE_TEXT[field].format(a=subjects[a["id"]], b=subjects[b["id"]])
            return label, text[0].upper() + text[1:], {"kind": "compare", "field": field, "a": a["id"], "b": b["id"]}
    field = rng.choice(list(TOTAL_TEXT))
    total = round(sum(r[field] for r in records), 1)
    delta = rng.randint(1, 9) if field == "quantity" else rng.randint(1, 30) / 10
    threshold = round(total - delta if label == "supported" else total + delta, 1)
    threshold = int(threshold) if field == "quantity" else threshold
    return label, TOTAL_TEXT[field].format(n=threshold), {"kind": "total", "field": field, "threshold": threshold}


def evidence_verdict(claim, records):
    """Recompute the label from the structured claim: the test's brute force and the definition of the gold."""
    by_id = {r["id"]: r for r in records}
    if claim["kind"] == "absent_field" or claim.get("mode") in ("missing_id", "missing_customer"):
        return "not_addressed"
    if claim["kind"] == "equals":
        return "supported" if by_id[claim["subject"]][claim["field"]] == claim["value"] else "contradicted"
    if claim["kind"] == "compare":
        a, b, f = by_id[claim["a"]][claim["field"]], by_id[claim["b"]][claim["field"]], claim["field"]
        return "supported" if (a > b if f in ("quantity", "weight_kg", "delivered") else a < b) else "contradicted"
    return "supported" if sum(r[claim["field"]] for r in records) > claim["threshold"] else "contradicted"


def evidence_world(rng, names, surnames, style):
    records = _records(rng, names)
    label, claim_text, claim = evidence_claim(rng, records, names, surnames)
    record_dates = rng.choice(("iso", "long"))
    shown = [{**r, "shipped": fmt_date(date.fromisoformat(r["shipped"]), record_dates),
              "delivered": fmt_date(date.fromisoformat(r["delivered"]), record_dates)} for r in records]
    if style == "json":
        state = {"records": shown, "claim": claim_text}
    elif style == "table":
        state = ("| id | customer | quantity | weight (kg) | shipped | delivered | status |\n| --- | --- | ---: | ---: | --- | --- | --- |\n"
                 + "\n".join(f"| {r['id']} | {r['customer']} | {r['quantity']} | {r['weight_kg']} | {r['shipped']} | {r['delivered']} | {r['status']} |" for r in shown)
                 + f"\n\nClaim: {claim_text}")
    else:
        state = ("Records:\n" + "\n".join(f"Shipment {r['id']} for {r['customer']}: {r['quantity']} units, {r['weight_kg']} kg, shipped {r['shipped']}, "
                                          f"delivered {r['delivered']}, status {r['status']}." for r in shown) + f"\n\nClaim: {claim_text}")
    world = {"family": "evidence", "style": style, "records": records, "record_dates": record_dates, "claim_text": claim_text,
             "claim": claim, "label": label}
    raw = {"state": state, "group_id": world_id("evidence", world), "questions": {
        "evidence:verdict": {"type": "choice", "instructions": rng.choice((
            "Do the records support the claim, contradict it, or not address it?",
            "Check the claim against the records: is it supported, contradicted, or not addressed by them?",
            "What do the records say about the claim?")),
            "criteria": dict(VERDICTS), "target": [float(v == label) for v in VERDICTS]},
        "evidence:supported": {"type": "noul", "instructions": rng.choice(("Is the claim supported by the records?", "Do the records establish that the claim is true?")),
                               "target": [float(label != "supported"), float(label == "supported")]}}}
    return world, Request.from_dict(raw)


# ---------------------------------------------------------------------------------------------------------------------
# record_match: is a rendered record the same person as a probe record

IDENTITY = ("dob", "phone", "email")
STREETS = ("Ferry", "Mill", "Station", "Church", "Park", "Orchard", "Bridge", "Chapel", "Market", "Quarry")
STREET_TYPES = (("Street", "St"), ("Road", "Rd"), ("Avenue", "Ave"), ("Lane", "Ln"))
CITIES = ("Ashford", "Brightwater", "Colton", "Dunmore", "Eastbury", "Fairhaven", "Glenrock", "Halverton")
DOMAINS = ("example.com", "mail.example.org", "post.example.net")
LABELS = {"name": ("Name", "Full name", "Customer"), "dob": ("Date of birth", "DOB", "Born"), "email": ("Email", "E-mail", "Email address"),
          "phone": ("Phone", "Tel", "Telephone"), "address": ("Address", "Street address", "Street"), "city": ("City", "Town", "City/town"),
          "postcode": ("Postcode", "Postal code", "ZIP")}
MATCH_RULE = ("Two records describe the same person when every date of birth, phone number and email address that both records "
              "give agrees once formatting is ignored; if any of these disagrees, they are different people.")


def _entity(rng, surnames):
    first, last = rng.choice(SENDER_POOL), rng.choice(surnames)
    return {"first": first, "last": last, "dob": date(rng.randint(1950, 2005), rng.randint(1, 12), rng.randint(1, 28)).isoformat(),
            "email": f"{first}.{last}{rng.randint(1, 99)}@{rng.choice(DOMAINS)}".lower(),
            "phone": str(rng.randint(2, 9)) + "".join(str(rng.randint(0, 9)) for _ in range(9)),
            "street_no": rng.randint(1, 250), "street": rng.choice(STREETS), "street_type": rng.randrange(len(STREET_TYPES)),
            "city": rng.choice(CITIES), "postcode": str(rng.randint(10000, 99999))}


def _twin(rng, entity, surnames):
    """A different person who shares identifying text with `entity`: same name, a relative at the same address, or the
    same name with one identity field in common. Every identity field not kept is fresh (differs from the entity's)."""
    mode = rng.choice(("namesake", "relative", "near"))
    twin = dict(entity)
    if mode == "relative":
        twin["first"] = rng.choice([n for n in SENDER_POOL if n != entity["first"]])
    keep = {rng.choice(IDENTITY)} if mode == "near" else set()
    for field in set(IDENTITY) - keep:
        while twin[field] == entity[field]:
            twin[field] = (_entity(rng, surnames)[field] if field != "email"
                           else f"{twin['first']}.{twin['last']}{rng.randint(1, 99)}@{rng.choice(DOMAINS)}".lower())
    if mode == "namesake" and rng.random() < .5:
        fresh = _entity(rng, surnames)
        twin.update({k: fresh[k] for k in ("street_no", "street", "street_type", "city", "postcode")})
    return twin, mode


def _view(rng, entity, required):
    """A rendering plan: (field, label, formatted value) for the fields shown; identity fields in `required` are always
    shown, any other field may be missing. Formatting and name typos never touch the identity values."""
    first, last = entity["first"], entity["last"]
    if rng.random() < .25:  # one edit in the surname
        p = rng.randrange(len(last) - 1)
        last = last[:p] + last[p + 1] + last[p] + last[p + 2:] if rng.random() < .5 else last[:p] + last[p + 1:]
    name = rng.choice((f"{first} {last}", f"{last}, {first}", f"{first[0]}. {last}", f"{first} {last}".upper(), f"{first} {last}".lower()))
    d = date.fromisoformat(entity["dob"])
    ph = entity["phone"]
    phone = rng.choice((f"({ph[:3]}) {ph[3:6]}-{ph[6:]}", f"{ph[:3]}-{ph[3:6]}-{ph[6:]}", ph, f"+1 {ph[:3]} {ph[3:6]} {ph[6:]}"))
    long_type, short_type = STREET_TYPES[entity["street_type"]]
    address = rng.choice((f"{entity['street_no']} {entity['street']} {long_type}", f"{entity['street_no']} {entity['street']} {short_type}",
                          f"{entity['street_no']} {entity['street']} {short_type}.".upper()))
    values = {"name": name, "dob": fmt_date(d, rng.choice(DATE_STYLES)), "email": rng.choice((entity["email"], entity["email"].upper())),
              "phone": phone, "address": address, "city": rng.choice((entity["city"], entity["city"].upper())), "postcode": entity["postcode"]}
    shown = [f for f in values if f == "name" or f in required or rng.random() >= .3]
    return [(f, rng.choice(LABELS[f]), values[f]) for f in shown]


def _record_text(view, style):
    if style == "json":
        return json.dumps({label: value for _, label, value in view}, ensure_ascii=False)
    if style == "table":
        return "| field | value |\n| --- | --- |\n" + "\n".join(f"| {label} | {value} |" for _, label, value in view)
    return "\n".join(f"{label}: {value}" for _, label, value in view)


def record_match_world(rng, surnames, style):
    entity = _entity(rng, surnames)
    probe_visible = set(IDENTITY)
    if rng.random() < .3:
        probe_visible.discard(rng.choice(IDENTITY))
    probe = _view(rng, entity, probe_visible)
    k = rng.randint(3, 8)
    has_match = rng.random() >= NONE_RATE
    candidates, kinds = [], []
    gold = rng.randrange(k) if has_match else None
    for n in range(k):
        if n == gold:
            required = set(rng.sample(sorted(probe_visible), 2)) if len(probe_visible) > 2 and rng.random() < .5 else probe_visible
            candidates.append((entity, _view(rng, entity, required)))
            kinds.append("match")
            continue
        other, mode = _twin(rng, entity, surnames) if rng.random() < .6 else (_entity(rng, surnames), "random")
        conflicts = [f for f in probe_visible if other[f] != entity[f]]
        while not conflicts:  # a random person who coincides on every visible identity field: redraw
            other, mode = _entity(rng, surnames), "random"
            conflicts = [f for f in probe_visible if other[f] != entity[f]]
        candidates.append((other, _view(rng, other, {rng.choice(conflicts)})))
        kinds.append(mode)
    probe_noul = gold if gold is not None and rng.random() < .5 else rng.choice([n for n in range(k) if n != gold])
    texts = [_record_text(view, rng.choice(STYLES) if style == "prose" else style) for _, view in candidates]
    probe_text = _record_text(probe, rng.choice(STYLES) if style == "prose" else style)
    world = {"family": "record_match", "style": style, "entity": entity, "probe": probe, "candidates": [c for c, _ in candidates],
             "views": [v for _, v in candidates], "kinds": kinds, "gold": gold, "probe_noul": probe_noul}
    if style == "json":
        state = {"probe": json.loads(probe_text), "candidates": [{"id": n + 1, "record": json.loads(t)} for n, t in enumerate(texts)]}
    else:
        state = f"Probe record:\n{probe_text}\n\nCandidates:\n" + "\n\n".join(f"[{n + 1}]\n{t}" for n, t in enumerate(texts))
    keys = [f"c{n + 1}" for n in range(k)] + ["none"]
    raw = {"state": state, "group_id": world_id("record_match", world), "questions": {
        "record_match:which": {"type": "choice", "instructions": rng.choice((
            f"Which candidate describes the same person as the probe record? {MATCH_RULE}",
            f"Select the candidate record that is the probe's person, or none. {MATCH_RULE}")),
            "criteria": {**{f"c{n + 1}": t for n, t in enumerate(texts)}, "none": "No candidate describes the same person as the probe."},
            "target": [float(key == ("none" if gold is None else f"c{gold + 1}")) for key in keys]},
        "record_match:same": {"type": "noul", "instructions": f"Is candidate {probe_noul + 1} the same person as the probe record? {MATCH_RULE}",
                              "target": [float(probe_noul != gold), float(probe_noul == gold)]}}}
    return world, Request.from_dict(raw)


def same_person(probe_view, candidate_view, probe_entity, candidate_entity):
    """The rule, applied to what the two views show: no visible identity conflict and at least two visible agreements."""
    shown = {f for f, _, _ in probe_view} & {f for f, _, _ in candidate_view} & set(IDENTITY)
    agree = [f for f in shown if probe_entity[f] == candidate_entity[f]]
    return len(agree) == len(shown) and len(agree) >= 2


# ---------------------------------------------------------------------------------------------------------------------

def generate_family(family, count, rng, split, seen, tokenizer, max_tokens, tokens):
    names = TEST_NAMES if split == "test" else TRAIN_NAMES
    surnames = TEST_SURNAMES if split == "test" else TRAIN_SURNAMES
    rows = []
    while len(rows) < count:
        style = STYLES[len(rows) % 3]
        if family == "retrieval":
            world, request = retrieval_world(rng, names, surnames, style)
            kind = "none" if world["gold"] is None else "present"
        elif family == "evidence":
            world, request = evidence_world(rng, names, surnames, style)
            kind = world["label"]
        else:
            world, request = record_match_world(rng, surnames, style)
            kind = "none" if world["gold"] is None else "present"
        keys = {("id", request.group_id), ("state", state_hash(request.state))}
        if keys & seen:
            continue
        seen |= keys
        count_tokens = pack_request(request, tokenizer, "tree", 10 ** 9, score_block="full").token_count
        if count_tokens > max_tokens:
            raise ValueError(f"{family} state packs to {count_tokens} tree tokens, over {max_tokens}")
        tokens.append(count_tokens)
        rows.append({**request.to_dict(), "tier": "T0", "family": family, "style": style, "kind": kind, "world": world})
    return rows


def _existing_keys(directories):
    seen = set()
    for directory in directories:
        for path in sorted(Path(directory).glob("*.jsonl")) if Path(directory).is_dir() else ():
            for request in load_requests(path):
                seen |= {("id", request.group_id), ("state", state_hash(request.state))}
    return seen


def prepare_families(output, tokenizer, seed=17, train=3000, dev=200, calibration=200, test=500, exclude=EXCLUDE,
                     max_tokens=MAX_TOKENS, families=FAMILIES):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    existing = _existing_keys(exclude)
    seen = set(existing)
    sizes = {"train": train, "dev": dev, "calibration": calibration, "test": test}
    splits, tokens = {name: [] for name in sizes}, {name: {} for name in sizes}
    for family in families:
        for split, count in sizes.items():
            rng = random.Random(f"{seed}:{family}:{split}")
            tokens[split][family] = []
            splits[split] += generate_family(family, count, rng, split, seen, tokenizer, max_tokens, tokens[split][family])
    for name, rows in splits.items():
        write_jsonl(output / f"{name}.jsonl", rows)
    check = {name: load_requests(output / f"{name}.jsonl") for name in splits}
    assert_disjoint(check)
    new_keys = {key for requests in check.values() for r in requests for key in (("id", r.group_id), ("state", state_hash(r.state)))}
    if new_keys & existing:
        raise ValueError("Data overlap between the new families and an excluded data set")
    manifest = {"dataset": DATASET, "seed": seed, "tier": "T0", "families": list(families), "max_tokens": max_tokens,
                "tokenizer": getattr(tokenizer, "name_or_path", "bytes"),
                "token_rule": "tree packing, score_block=full (the longest variant), every row at or under max_tokens",
                "counts": {name: dict(Counter(r["family"] for r in rows)) for name, rows in splits.items()},
                "kinds": {name: {f: dict(sorted(Counter(r["kind"] for r in rows if r["family"] == f).items())) for f in families}
                          for name, rows in splits.items()},
                "tokens": {name: {f: {"max": max(v), "mean": round(statistics.mean(v), 1)} for f, v in t.items()} for name, t in tokens.items()},
                "holdouts": {"retrieval": "test organisation names and manager surnames never appear in train/dev/calibration",
                             "evidence": "test customer names and surnames never appear in train/dev/calibration",
                             "record_match": "test surnames never appear in train/dev/calibration"},
                "none_rate": NONE_RATE, "excluded": [str(d) for d in exclude if Path(d).is_dir()],
                "gold_note": "T0: gold computed by code from the hidden world stored under `world`; language is a rendering.",
                "files": {p.name: {"sha256": file_hash(p), "bytes": p.stat().st_size} for p in sorted(output.glob("*.jsonl"))}}
    write_json(output / "manifest.json", manifest)
    return manifest


# ---------------------------------------------------------------------------------------------------------------------
# T3 top-up cells, run through generate.run_pilot with the cells registered for the duration of the run

T3_DATASET = "JEV_T3_BREADTH_TOPUP_V1"
T3_SEED = 19  # 17 is the pilot's, 18 the volume run's
T3_CELLS = {
    "rubric_grading": {"family": "rubric grading", "question_type": "score", "state_format": "text",
                       "difficulty": ["scoping", "borderline", "distractor"], "adversarial": False, "none_option": False,
                       "domain_hints": ["essay paragraphs", "code review comments", "support replies", "commit messages", "product descriptions",
                                        "incident postmortems", "meeting notes", "bug reports", "cover letters", "recipe instructions"],
                       "levels": [3, 4, 5, 6, 7],
                       "brief": "A caller-defined grading rubric for one kind of short artefact, with the ordered levels each given as an object "
                                "with summary and signals, followed by the artefact itself (40 to 120 words); the question asks which level the "
                                "artefact earns under the rubric. The graded dimension must vary across examples (clarity, completeness, tone, "
                                "use of evidence, actionability, adherence to a stated brief) and the level boundaries must be concrete enough "
                                "that a careful reader lands on exactly one level; borderline difficulty places the artefact just above a "
                                "boundary, distractor difficulty adds a strength or weakness on a dimension the rubric does not grade."},
    "tool_argument_none": {"family": "argument selection", "question_type": "choice", "state_format": "json",
                           "difficulty": ["literal", "paraphrase", "near_miss"], "adversarial": False, "none_option": True,
                           "domain_hints": ["banking", "telecom support", "e-commerce returns", "IT helpdesk", "insurance claims", "travel booking",
                                            "healthcare scheduling", "HR requests", "logistics", "SaaS billing", "government services", "utilities"],
                           "cardinalities": [3, 4, 6, 8],
                           "brief": "A JSON state with a tool schema (one parameter with an enum, or a small set of typed candidate values, and a "
                                    "one-line description of what the parameter means) and a natural-language request; the options are the "
                                    "candidate values plus none, and the question asks which value the request states for that parameter. In "
                                    "about half of the examples the request does not state the parameter at all, or mentions a related but "
                                    "different quantity, so that none is the gold; near_miss difficulty mentions a value that resembles a "
                                    "candidate but does not satisfy the parameter's definition. Never infer a value the request does not state."}}


def write_t3_audits(output, cells, seed, audit_size=100):
    """Two audit files per cell: the stratified one (every doubted row first, the volume run's format) and a uniform
    sample of consensus rows, because a cell with more than `audit_size` doubted rows leaves no consensus row in the
    first file (docs/phase2/gap-cells-and-volume.md, severity_rubric). Returns (consensus counts, none rates, audit)."""
    from .volume import STRATA, audit_markdown, stratified_audit_sample
    output = Path(output)
    consensus, none_rate, audit = {}, {}, {}
    for cell in cells:
        with (output / f"{cell}.jsonl").open() as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        agreed = [r for r in rows if r["checks"]["outcome"] == "T3"]
        consensus[cell] = len(agreed)
        golds = [next(k for k, t in zip(q["criteria"], q["target"]) if t == 1.) for r in rows for q in r["questions"].values() if q["type"] == "choice"]
        none_rate[cell] = round(sum(g == "none" for g in golds) / len(golds), 3) if golds else None
        available = {o: sum(1 for r in rows if r["checks"]["outcome"] == o) for o in STRATA}
        sample = stratified_audit_sample(rows, audit_size, random.Random(f"audit:{seed}:{cell}"))
        (output / f"audit-{cell}.md").write_text(audit_markdown(cell, sample, available))
        uniform = random.Random(f"audit2:{seed}:{cell}").sample(agreed, min(audit_size, len(agreed)))
        (output / f"audit2-{cell}-T3.md").write_text(audit_markdown(cell, uniform, available).replace("(volume, stratified)", "(uniform consensus rows)", 1)
                                                      .replace("Stratified: every T4 row, then every T3_second row, then random T3 rows.", "Uniform: a random sample of consensus (T3) rows.", 1))
        audit[cell] = {"size": len(sample), **{o: sum(1 for r in sample if r["checks"]["outcome"] == o) for o in STRATA},
                       "uniform_consensus_file": f"audit2-{cell}-T3.md", "uniform_consensus_size": len(uniform), "available": available}
    return consensus, none_rate, audit


def run_t3_topup(output, per_cell=1200, seed=T3_SEED, luna=None, terra=None, cost_abort_usd=60., workers=16, audit_size=100, cells=None):
    """Generate the top-up cells with the pilot pipeline; write stratified audit files; count consensus rows. A second
    batch (another seed, another directory) tops a cell up when its consensus count falls short."""
    cells = list(cells or T3_CELLS)
    from .generate import CELLS, run_pilot
    from .volume import STRATA, audit_markdown, stratified_audit_sample
    if seed in (17, 18):
        raise ValueError("seeds 17 and 18 are the pilot's and the volume run's")
    added = [name for name in T3_CELLS if name not in CELLS]
    CELLS.update({name: T3_CELLS[name] for name in added})
    try:
        manifest = run_pilot(output, cells=cells, per_cell=per_cell, seed=seed, luna=luna, terra=terra,
                             cost_abort_usd=cost_abort_usd, workers=workers)
    finally:
        for name in added:
            CELLS.pop(name)
    consensus, none_rate, audit = write_t3_audits(output, cells, seed, audit_size)
    manifest.update({"dataset": T3_DATASET, "consensus_rows": consensus, "none_gold_rate": none_rate, "audit": audit,
                     "cell_specs": T3_CELLS,
                     "acceptance_note": "Only rows with outcome T3 (generator gold recovered by the independent first check) enter a mix; "
                                        "T3_second and T4 rows are kept in the files for the audit and excluded by the mix builder."})
    write_json(output / "manifest.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="T0 families output directory")
    parser.add_argument("--t3-output", help="T3 top-up output directory (calls GPT-5.6; reads the key from env.sh)")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--train", type=int, default=3000)
    parser.add_argument("--per-cell", type=int, default=1200)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--cells", help="T3 top-up: comma-separated cell names (default: both)")
    args = parser.parse_args(argv)
    if args.t3_output:
        manifest = run_t3_topup(args.t3_output, per_cell=args.per_cell, seed=args.seed or T3_SEED, workers=args.workers,
                                cells=args.cells.split(",") if args.cells else None)
        print(json.dumps({k: manifest[k] for k in ("cells", "consensus_rows", "none_gold_rate", "cost_usd", "calls", "wall_minutes")}, indent=2))
        return
    if not args.output:
        parser.error("--output or --t3-output is required")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(QWEN35[0], revision=QWEN35[1])
    manifest = prepare_families(args.output, tokenizer, seed=args.seed or 17, train=args.train)
    print(json.dumps({k: manifest[k] for k in ("counts", "kinds", "tokens")}, indent=2))


if __name__ == "__main__":
    main()
