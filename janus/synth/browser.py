"""Browser-action training cell: scripted web tasks recorded through the vendored jev-ultrafast loop.

Every row is the exact /v1/systemone body the loop sends for an observed page (`jev_ultrafast.model.request_body`)
plus one-hot targets: the gold operation, the gold operation's target head, and, for TYPE_TEXT, the click head
pointed at the same field's "Open ..." click. Unused speculative heads are omitted. Distractor rows reuse a recorded
state with a different goal whose gold is another visible control, or DONE when that goal is already satisfied.

Run from the repo root with the browser demo's environment (it needs browser-harness):
    demos/browser/.venv/bin/python -m janus.synth.browser record --output data/browser-v1/rows.jsonl [--tasks name,...]
    demos/browser/.venv/bin/python -m janus.synth.browser split --rows data/browser-v1/rows.jsonl --output data/browser-v1
    demos/browser/.venv/bin/python -m janus.synth.browser explore URL      # print the action table of a live page
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "demos" / "browser"))

FIXTURE = "http://127.0.0.1:8767/fixture.html?scenario=travel"  # demos/browser/jev_ultrafast/static served locally
FLIGHTS = "https://www.google.com/travel/flights?hl=en"
HTTPBIN = "https://httpbin.org/forms/post"
WIKI = "https://en.wikipedia.org/wiki/"


# ---------------------------------------------------------------- task scripts
def flights(name, origin, origin_pick, dest, dest_pick, day, month=None):
    """One-way search; `day` is the calendar day's aria-label, `month` the month heading to page to (None: visible)."""
    goal = (f"Find one-way flights from {origin} to {dest} on {day.split(', ', 1)[1]}, for one adult in economy. "
            "Stop when matching flight options are visible. Do not select or book a flight.")
    home_distractors = [
        ("Open the Hotels section of Google Travel.", "CLICK", {"label": "Hotels", "role": "link"}),
        ("Open the Explore section of Google Travel.", "CLICK", {"label": "Explore", "role": "link"}),
    ]
    steps = [
        {"op": "CLICK", "label": "Change ticket type. Round trip", "distractors": home_distractors},
        {"op": "CLICK", "label": "One way", "role": "option",
         "distractors": [("Switch the ticket type to Round trip.", "DONE", None)]},
        {"op": "TYPE_TEXT", "label": "Where from?", "text": origin,
         "distractors": [("Set the ticket type to One way. Stop once it is set.", "DONE", None),
                         ("Open the Vacation rentals section of Google Travel.", "CLICK", {"label": "Vacation rentals"})]},
        {"op": "CLICK", "label_prefix": origin_pick},
        {"op": "TYPE_TEXT", "label": "Where to?", "text": dest,
         "distractors": [(f"Set the origin airport to {origin}. Stop once it is set.", "DONE", None)]},
        {"op": "CLICK", "label_prefix": dest_pick},
        {"op": "CLICK", "label": "Open Departure",
         "distractors": [(f"Set up a one-way search from {origin} to {dest}, leaving the date empty.", "DONE", None),
                         ("Change the number of passengers.", "CLICK", {"label_prefix": "1 passenger"})]},
    ]
    if month:  # page the calendar forward until the day is offered
        steps.append({"op": "CLICK", "label_prefix": "Next", "until": {"label": day}})
    steps += [
        {"op": "CLICK", "label": day, "distractors": [("Close the date picker without choosing a date.", "CLICK", {"label_prefix": "Reset"})]},
        {"op": "CLICK", "label_prefix": "Done. Search for one-way flights"},
        {"op": "CLICK", "label": "Search", "role": "button", "settle": "Search",
         "distractors": [(f"Pick {day.split(', ', 1)[1]} as the departure date. Stop once it is set.", "DONE", None)]},
        {"op": "WAIT", "until": {"label_contains": "Select flight"}},
        {"op": "DONE", "distractors": [("Sort the results by price.", "CLICK", {"label_prefix": "Sorted by"})]},
    ]
    return {"name": name, "url": FLIGHTS, "goal": goal, "steps": steps}


def hotel(name, city, category, free, place, pick_first_prefix=None):
    filters = " and ".join(x for x in (f"{category} stays" if category else "", "Free cancellation" if free else "") if x)
    goal = f"Use the destination search and filters to find {filters or 'stays'} in {city}, then open {place}."
    steps = [{"op": "TYPE_TEXT", "label": "Destination", "text": city,
              "distractors": [("Open the reading room.", "CLICK", {"label": "Reading room"})]},
             {"op": "CLICK", "label": "Find stays", "role": "button"}]
    if category:
        steps.append({"op": "SELECT", "label": f"Stay category → {category}",
                      "distractors": [(f"Search for stays in {city}. Stop once the search has run.", "DONE", None)]})
    if free:
        steps.append({"op": "CLICK", "label": "Free cancellation", "role": "checkbox"})
    steps += [{"op": "CLICK", "label": f"View {place}",
               "distractors": [(f"Find {filters or 'stays'} in {city} without opening a property.", "DONE", None)]},
              {"op": "DONE", "distractors": [("Go back to the list of all stays.", "CLICK", {"label_prefix": "← All stays"})]}]
    return {"name": name, "url": FIXTURE, "goal": goal, "steps": steps}


def wiki_search(name, query, title, suggestion=None):
    goal = f"Find and open the Wikipedia article about {title}."
    steps = [{"op": "TYPE_TEXT", "label": "Search Wikipedia", "text": query,
              "distractors": [("Log in to Wikipedia.", "CLICK", {"label": "Log in", "role": "link"}),
                              ("Open the Wikipedia donation page.", "CLICK", {"label": "Donate", "role": "link"})]},
             {"op": "CLICK", "label_prefix": suggestion or title, "role": "option", "settle": suggestion or title,
              "distractors": [(f"Type '{query}' into the Wikipedia search box. Stop once it is typed.", "DONE", None)]},
             {"op": "DONE", "distractors": [("Open the talk page of the current article.", "CLICK", {"label": "Talk"})]}]
    return {"name": name, "url": WIKI + "Main_Page", "goal": goal, "steps": steps}


def wiki_link(name, start, link, title):
    goal = f"From the current article, open the Wikipedia article about {title}."
    steps = [{"op": "SCROLL_DOWN", "until": {"label": link, "role": "link"}},
             {"op": "CLICK", "label": link, "role": "link",
              "distractors": [("Open the Wikipedia main page.", "CLICK", {"label_prefix": "Wikipedia", "role": "link"}),
                              (f"Stay on the article about {start.replace('_', ' ')}.", "DONE", None)]},
             {"op": "DONE"}]
    return {"name": name, "url": WIKI + start, "goal": goal, "steps": steps}


def pizza(name, customer, phone, email, size, toppings, note):
    goal = (f"Order a {size.lower()} pizza with {' and '.join(toppings).lower()} for {customer} (phone {phone}, email "
            f"{email}); delivery instructions: {note}. Submit the order and stop when it has been submitted.")
    steps = [{"op": "TYPE_TEXT", "label": "Customer name:", "text": customer,
              "distractors": [(f"Choose the {size.lower()} pizza size only.", "CLICK", {"label": size, "role": "radio"})]},
             {"op": "TYPE_TEXT", "label": "Telephone:", "text": phone},
             {"op": "TYPE_TEXT", "label": "E-mail address:", "text": email,
              "distractors": [(f"Enter {customer} as the customer name and stop.", "DONE", None)]},
             {"op": "CLICK", "label": size, "role": "radio"}]
    steps += [{"op": "CLICK", "label": t, "role": "checkbox"} for t in toppings]
    steps += [{"op": "TYPE_TEXT", "label": "Delivery instructions:", "text": note,
               "distractors": [(f"Fill in the contact details and pick a {size.lower()} pizza with {' and '.join(toppings).lower()}; "
                                "do not submit.", "DONE", None)]},
              {"op": "CLICK", "label": "Submit order", "role": "button"},
              {"op": "WAIT", "until_text": '"custname"'},
              {"op": "DONE"}]
    return {"name": name, "url": HTTPBIN, "goal": goal, "steps": steps}


def tasks():
    t = [flights("flights_zrh_lon_1025", "Zurich", "Zürich, Switzerland", "London", "London, United Kingdom",
                 "Sunday, October 25, 2026")]  # the demo task (JEV_FLIGHT_DATE=2026-10-25): test only
    pairs = [("par_ber", "Paris", "Paris, France", "Berlin", "Berlin, Germany", "Wednesday, October 14, 2026", None),
             ("ams_rom", "Amsterdam", "Amsterdam, Netherlands", "Rome", "Rome, Italy", "Tuesday, November 3, 2026", "November"),
             ("mad_lis", "Madrid", "Madrid, Spain", "Lisbon", "Lisbon, Portugal", "Friday, October 2, 2026", None),
             ("vie_prg", "Vienna", "Vienna, Austria", "Prague", "Prague, Czechia", "Wednesday, November 18, 2026", "November"),
             ("chi_mia", "Chicago", "Chicago, Illinois", "Miami", "Miami, Florida", "Tuesday, October 27, 2026", None),
             ("dub_osl", "Dublin", "Dublin, Ireland", "Oslo", "Oslo, Norway", "Saturday, December 12, 2026", "December"),
             ("yto_yvr", "Toronto", "Toronto, Canada", "Vancouver", "Vancouver, Canada", "Thursday, October 22, 2026", None),
             ("nyc_tyo", "New York", "New York, New York", "Tokyo", "Tokyo, Japan", "Saturday, December 5, 2026", "December"),
             ("cph_bcn", "Copenhagen", "Copenhagen, Denmark", "Barcelona", "Barcelona, Spain", "Monday, October 19, 2026", None),
             ("sfo_sea", "San Francisco", "San Francisco, California", "Seattle", "Seattle, Washington", "Friday, November 6, 2026", "November")]
    t += [flights("flights_" + p[0] + "_" + p[5].split()[1][:3].lower() + p[5].split()[2].rstrip(','), *p[1:]) for p in pairs]
    t += [hotel("hotel_lisbon_design_free", "Lisbon", "Design", True, "Casa Flora"),
          hotel("hotel_lisbon_nature", "Lisbon", "Nature", False, "Serra Lodge"),
          hotel("hotel_copenhagen_free", "Copenhagen", None, True, "The Glasshouse"),
          hotel("hotel_lisbon_any", "Lisbon", None, False, "Casa Flora"),
          hotel("hotel_copenhagen_design", "Copenhagen", "Design", False, "The Glasshouse"),
          hotel("hotel_lisbon_free", "Lisbon", None, True, "Casa Flora")]
    searches = [("Gödel's incompleteness theorems", "Gödel's incompleteness theorems"), ("Photosynthesis", "photosynthesis"),
                ("Ada Lovelace", "Ada Lovelace"), ("Mount Kilimanjaro", "Mount Kilimanjaro"), ("Bayes' theorem", "Bayes' theorem"),
                ("Great Barrier Reef", "the Great Barrier Reef"), ("Haskell", "the Haskell programming language"),
                ("Marie Curie", "Marie Curie"), ("Voyager 1", "the Voyager 1 space probe"), ("Halting problem", "the halting problem"),
                ("Okapi", "the okapi"), ("Treaty of Westphalia", "the Peace of Westphalia")]
    t += [wiki_search("wiki_search_" + re.sub(r"\W+", "", q.split()[0].lower()), q, title, q) for q, title in searches]
    links = [("Berlin", "Spree", "the Spree river"), ("Lisbon", "Tagus", "the Tagus river"), ("Prague", "Vltava", "the Vltava river"),
             ("Tokyo", "Honshu", "Honshu"), ("Oslo", "Norway", "Norway"), ("Amsterdam", "Netherlands", "the Netherlands"),
             ("Photosynthesis", "chlorophyll", "chlorophyll"), ("Zürich", "Limmat", "the Limmat river"),
             ("London", "River Thames", "the River Thames"), ("Python_(programming_language)", "Guido van Rossum", "Guido van Rossum"),
             ("Mars", "Olympus Mons", "Olympus Mons"), ("Coffee", "Ethiopia", "Ethiopia"),
             ("Bicycle", "Draisine", "the draisine"), ("Chess", "Magnus Carlsen", "Magnus Carlsen")]
    t += [wiki_link(f"wiki_link_{s.split('_')[0].lower()}_{l.split()[0].lower()}", s, l, title) for s, l, title in links]
    orders = [("Jane Doe", "555-0100", "jane@example.com", "Large", ["Bacon", "Onion"], "ring twice"),
              ("Omar Haddad", "555-0142", "omar@example.org", "Medium", ["Mushroom"], "leave at the door"),
              ("Lina Berg", "555-0177", "lina@example.net", "Small", ["Extra Cheese", "Bacon"], "call on arrival"),
              ("Kenji Sato", "555-0111", "kenji@example.com", "Large", ["Onion", "Mushroom"], "second floor"),
              ("Ana Costa", "555-0123", "ana@example.org", "Medium", ["Extra Cheese"], "gate code 4471"),
              ("Tom Walsh", "555-0199", "tom@example.net", "Small", ["Bacon"], "no doorbell, knock")]
    t += [pizza("form_pizza_" + o[0].split()[0].lower(), *o) for o in orders]
    return t


# ---------------------------------------------------------------- recording
def matches(action, spec):
    if spec is None:
        return False
    kind = {"CLICK": "click", "TYPE_TEXT": "fill", "SELECT": "select"}.get(spec.get("op", "CLICK"))
    if kind and action["kind"] != kind:
        return False
    if "role" in spec and action.get("role") != spec["role"]:
        return False
    label = action["label"].strip()
    if "label" in spec:
        return label == spec["label"]
    if "label_contains" in spec:
        return spec["label_contains"] in label
    return label.startswith(spec["label_prefix"])


def find(page, spec):
    hits = [a for a in page["actions"] if matches(a, spec)]
    return hits[0] if hits else None


def targets_for(body, targets, op, action):
    """One-hot targets on the loop's questions; None when the gold is not offered on this page."""
    questions = {}
    for name, q in body["questions"].items():
        keys = list(q["criteria"])
        gold = None
        if name == "operation":
            gold = op
        elif action is not None and name == op.lower() + "_target":
            gold = next(k for k, a in targets[op].items() if a["id"] == action["id"])
        elif action is not None and op == "TYPE_TEXT" and name == "click_target":
            opener = next((k for k, a in targets["CLICK"].items() if a["node"] == action["node"]
                           and a["label"] == "Open " + action["label"]), None)
            gold = opener
        if gold is None:
            continue  # an unused speculative head: no target, question omitted from the row
        if gold not in keys:
            return None
        questions[name] = {**q, "target": [1. if k == gold else 0. for k in keys]}
    return questions


def make_row(page, goal, history, op, action, group_id, task):
    from jev_ultrafast.model import request_body
    body, operations, targets, controls = request_body(page, goal, history)
    if op not in operations:
        return None
    questions = targets_for(body, targets, op, action)
    if questions is None:
        return None
    return {"state": body["state"], "questions": questions, "group_id": group_id, "tier": "T1", "family": "browser",
            "source": "browser:" + task, "cell": task.split("_")[0], "goal": goal,
            "gold": {"operation": op, "action": action["id"] if action else None, "label": action["label"] if action else None}}


def settle(browser, spec, seconds=8):
    """Re-observe until the gold control is on the page (dynamic pages: suggestions, calendars, results)."""
    deadline = time.monotonic() + seconds
    while True:
        page = browser.observe(screenshot=False)
        if spec is None or find(page, spec) or time.monotonic() > deadline:
            return page
        time.sleep(0.25)


def record_task(task, log):
    from jev_ultrafast.browser import Browser, StalePage
    rows, history = [], []
    browser = Browser(task["url"])
    try:
        for index, step in enumerate(task["steps"], 1):
            op = step["op"]
            gid = f"browser:{task['name']}:{index}"
            if "until" in step or "until_text" in step:  # gold WAIT/SCROLL_DOWN/CLICK repeated until the awaited control shows
                control = op.lower()
                for attempt in range(30 if op == "WAIT" else 15):
                    page = browser.observe(screenshot=False)
                    ready = (step["until_text"] in page["text"] if "until_text" in step else find(page, step["until"]))
                    if ready and op == "SCROLL_DOWN" and ready["rect"]["y"] < 60:  # under a sticky header: nudge up, unrecorded
                        browser.act(next(a for a in page["actions"] if a["id"] == "scroll_up"), page)
                        continue
                    if ready:
                        break
                    action = (find(page, {k: v for k, v in step.items() if k in ("op", "label", "label_prefix", "role")})
                              if op == "CLICK" else next((a for a in page["actions"] if a["id"] == control), None))
                    if attempt < 3:  # at most three recorded rows per awaited step
                        row = make_row(page, task["goal"], history, op, action, f"{gid}{control[0]}{attempt}", task["name"])
                        if row:
                            rows.append(row)
                    if action is None:
                        raise RuntimeError(f"{task['name']} step {index}: {control} is not offered")
                    try:
                        browser.act(action, page)
                    except StalePage:
                        continue
                    history.append({"action": action["label"], "kind": action["kind"], "text": None, "page_changed": None})
                    new = browser.observe(screenshot=False)
                    history[-1]["page_changed"] = new["fingerprint"] != page["fingerprint"]
                    time.sleep(0.5 if op == "WAIT" else 0.1)
                else:
                    labels = [f"{a['kind']}:{a['label']}" for a in page["actions"]]
                    raise RuntimeError(f"{task['name']} step {index}: awaited content never appeared; text={page['text'][:300]!r} actions={labels}")
                log(f"  {index} {op} x{attempt}")
                continue
            spec = None if op == "DONE" else {k: v for k, v in step.items() if k in ("op", "label", "label_prefix", "label_contains", "role")}
            page = settle(browser, spec if not step.get("settle") else {"op": op, "label_prefix": step["settle"]})
            action = find(page, spec) if spec else None
            if spec and action is None:
                labels = [f"{a['kind']}:{a['label']}" for a in page["actions"]]
                raise RuntimeError(f"{task['name']} step {index}: no {spec} among {labels}")
            row = make_row(page, task["goal"], history, op, action, gid, task["name"])
            if row is None:
                raise RuntimeError(f"{task['name']} step {index}: gold not offered on this page")
            rows.append(row)
            for d, (goal, d_op, d_spec) in enumerate(step.get("distractors", [])):
                d_action = find(page, {**d_spec, "op": d_op}) if d_spec else None
                if d_spec and d_action is None:
                    log(f"  distractor skipped ({task['name']} step {index}): {d_spec} not on page")
                    continue
                d_row = make_row(page, goal, history, d_op, d_action, f"{gid}d{d}", task["name"])
                if d_row:
                    rows.append(d_row)
            if op == "DONE":
                break
            for attempt in range(4):  # the loop's stale-page path: re-observe and choose again
                try:
                    browser.act(action, page, text=step.get("text"))
                    break
                except StalePage:
                    if attempt == 3:
                        raise
                    time.sleep(0.3)
                    page = settle(browser, spec)
                    action = find(page, spec) or action
            history.append({"action": action["label"], "kind": action["kind"], "text": step.get("text"), "page_changed": None})
            new = browser.observe(screenshot=False)
            history[-1]["page_changed"] = new["fingerprint"] != page["fingerprint"]
            log(f"  {index} {op} {action['label'][:50]!r} changed={history[-1]['page_changed']}")
    finally:
        browser.close()
    return rows


def record(output, names=None, log=print):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if output.exists():
        done = {json.loads(l)["source"].split(":", 1)[1] for l in output.read_text().splitlines() if l.strip()}
    for task in tasks():
        if (names and task["name"] not in names) or task["name"] in done:
            continue
        log(f"{task['name']}: {task['goal']}")
        try:
            rows = record_task(task, log)
        except Exception as error:  # a failed task records nothing; nothing is fabricated
            log(f"  FAILED {task['name']}: {error}")
            continue
        with output.open("a") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        log(f"  {len(rows)} rows")


# ---------------------------------------------------------------- splits
def state_hash(state):
    text = json.dumps(state, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest()


def balanced(path, limit, seed=17):
    """Round-robin over group_id prefixes (the families), like janus.training.balanced_subset, on raw rows."""
    groups = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            groups.setdefault(row["group_id"].split(":", 1)[0], []).append(row)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    out = []
    while len(out) < limit and any(groups.values()):
        for name in sorted(groups):
            if groups[name] and len(out) < limit:
                out.append(groups[name].pop())
    return out


def split(rows_path, output, test_task="flights_zrh_lon_1025", seed=17, replay=None, replay_rows=3000, replay_dev=None,
          replay_dev_rows=600, repeat=1):
    """Test: the demo task plus about 10% of the other tasks; dev: another 10%; whole tasks stay together, and a dev
    or test row whose state also occurs in train (the Flights home page is the same for every city pair) is dropped.
    With `replay`, also writes mix_train/mix_dev: the browser train rows (`repeat` times, so a one-epoch budget sees
    them more than once) plus a balanced slice of the old production data."""
    rows = [json.loads(l) for l in Path(rows_path).read_text().splitlines() if l.strip()]
    names = sorted({r["source"].split(":", 1)[1] for r in rows} - {test_task})
    rng = random.Random(seed)
    rng.shuffle(names)
    n = max(1, round(len(names) * .1))
    assign = {test_task: "test", **{k: "test" for k in names[:n]}, **{k: "dev" for k in names[n:2 * n]}}
    splits = {"train": [], "dev": [], "test": []}
    for r in rows:
        splits[assign.get(r["source"].split(":", 1)[1], "train")].append(r)
    seen = {state_hash(r["state"]) for r in splits["train"]}
    dropped = {}
    for name in ("dev", "test"):
        kept = [r for r in splits[name] if state_hash(r["state"]) not in seen]
        dropped[name] = len(splits[name]) - len(kept)
        splits[name] = kept
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        (output / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in part))
    manifest = {"dataset": "browser-v1", "test_task": test_task, "assignment": assign,
                "counts": {k: len(v) for k, v in splits.items()}, "dropped_seen_states": dropped}
    if replay:
        mixes = {"mix_train": splits["train"] * repeat + balanced(replay, replay_rows, seed),
                 "mix_dev": splits["dev"] + balanced(replay_dev, replay_dev_rows, seed)}
        for name, part in mixes.items():
            random.Random(seed).shuffle(part)
            (output / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in part))
        manifest["mix"] = {"replay": replay, "replay_dev": replay_dev, "browser_repeat": repeat,
                           "counts": {k: len(v) for k, v in mixes.items()}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def explore(url):
    from jev_ultrafast.browser import Browser
    browser = Browser(url)
    try:
        time.sleep(1.5)
        page = browser.observe(screenshot=False)
        print(page["title"], page["url"], "text", len(page["text"]))
        for a in page["actions"]:
            print(a["id"], a["kind"], a.get("role"), repr(a["label"]), repr(a.get("value", ""))[:30])
    finally:
        browser.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("record")
    r.add_argument("--output", required=True)
    r.add_argument("--tasks", help="comma-separated task names (default: all not yet recorded)")
    s = sub.add_parser("split")
    s.add_argument("--rows", required=True)
    s.add_argument("--output", required=True)
    s.add_argument("--replay", help="old training file to mix in (balanced slice), e.g. data/production-v3/train.jsonl")
    s.add_argument("--replay-rows", type=int, default=3000)
    s.add_argument("--replay-dev", help="old dev file to mix into mix_dev")
    s.add_argument("--replay-dev-rows", type=int, default=600)
    s.add_argument("--repeat", type=int, default=1, help="browser train rows repeated this many times in mix_train")
    e = sub.add_parser("explore")
    e.add_argument("url")
    args = parser.parse_args(argv)
    if args.command == "record":
        record(args.output, set(args.tasks.split(",")) if args.tasks else None)
    elif args.command == "split":
        print(json.dumps(split(args.rows, args.output, replay=args.replay, replay_rows=args.replay_rows,
                               replay_dev=args.replay_dev, replay_dev_rows=args.replay_dev_rows, repeat=args.repeat), indent=2))
    else:
        explore(args.url)


if __name__ == "__main__":
    main()
