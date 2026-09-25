"""Code-defined hidden worlds (verification tier T0). Language is only a rendering."""

import hashlib
import json
import random

from ..schema import Request

NAMES = ("Atlas", "Borealis", "Cinder", "Dune", "Ember", "Fjord", "Gale", "Harbor", "Iris", "Juniper",
         "Kestrel", "Lumen", "Mesa", "Nimbus", "Orchid", "Pike", "Quill", "Ridge", "Sable", "Tundra",
         "Umber", "Vale", "Willow", "Xenon", "Yarrow", "Zephyr", "Alder", "Basalt", "Cove", "Delta",
         "Echo", "Flint", "Grove", "Heron", "Isle", "Jade", "Knoll", "Lark", "Moss", "Nook")
TRAIN_NAMES, TEST_NAMES = NAMES[:30], NAMES[30:]
REL_KINDS = ("second_cheapest", "median_rating", "closest_price_to_named", "cheapest_above_named_rating")


def world_id(family, world):
    return f"{family}:{hashlib.sha256(json.dumps(world, sort_keys=True).encode()).hexdigest()[:20]}"


def _item_text(item, style):
    if style == "json":
        return json.dumps(item, sort_keys=True)
    if style == "compact":  # cardinality sweep: about 19 Qwen tokens per option line instead of 25
        return f"{item['name']} ${item['price']} r{item['rating']} {item['distance_km']}km"
    if style == "terse":  # K=255 cell: no distance (no request kind reads it), so 255 leaves fit under 8,192 tokens
        return f"{item['name']} ${item['price']} r{item['rating']}"
    return f"{item['name']}: ${item['price']}, rated {item['rating']}, {item['distance_km']} km away"


def relative_menu_world(rng, names, style, k=None, price_pool=range(5, 200)):
    """`k=None` draws the Phase 1 menu sizes; a fixed `k` (2 to len(price_pool)) is the cardinality sweep, where an even `k`
    never asks for the median (it would be ambiguous) and k > 41 rates to two decimals so ratings stay distinct.
    `price_pool` defaults to the Phase 1 pool; the K=255 cell passes a wider one."""
    while True:
        kind = rng.choice(REL_KINDS)
        size = k
        if size is None:
            size = rng.choice((5, 7)) if kind == "median_rating" else rng.choice((4, 5, 6, 8))
        elif kind == "median_rating" and size % 2 == 0:
            continue
        chosen = rng.sample(names, size)
        prices = rng.sample(price_pool, size)
        scale = 10 if size <= 41 else 100
        ratings = [r / scale for r in rng.sample(range(scale, 5 * scale + 1), size)]
        distances = [rng.randint(1, 50) for _ in range(size)]
        items = [{"name": n, "price": p, "rating": r, "distance_km": d} for n, p, r, d in zip(chosen, prices, ratings, distances)]
        reference = rng.choice(items)["name"] if kind in ("closest_price_to_named", "cheapest_above_named_rating") else None
        if kind == "second_cheapest":
            gold = sorted(items, key=lambda i: i["price"])[1]["name"]
            ask = "The customer wants the second cheapest option."
        elif kind == "median_rating":
            gold = sorted(items, key=lambda i: i["rating"])[size // 2]["name"]
            ask = "The customer wants the option whose rating is the median of all the options."
        elif kind == "closest_price_to_named":
            ref = next(i for i in items if i["name"] == reference)
            others = [i for i in items if i["name"] != reference]
            gaps = sorted(abs(i["price"] - ref["price"]) for i in others)
            if len(gaps) > 1 and gaps[0] == gaps[1]:
                continue
            gold = min(others, key=lambda i: abs(i["price"] - ref["price"]))["name"]
            ask = f"The customer wants the option whose price is closest to the option named {reference}."
        else:
            ref = next(i for i in items if i["name"] == reference)
            above = [i for i in items if i["rating"] > ref["rating"]]
            if not above:
                continue
            gold = min(above, key=lambda i: i["price"])["name"]
            ask = f"The customer wants the cheapest option that is rated higher than the option named {reference}."
        break
    world = {"family": "rel", "kind": kind, "style": style, "items": items, "reference": reference, "gold": gold}
    rng.shuffle(items)
    probe = gold if rng.random() < .5 else rng.choice([i["name"] for i in items if i["name"] != gold])
    # Nuisance id only: identical requests must not produce identical state text across splits.
    order_id = rng.randrange(100000, 999999)
    world["order_id"] = order_id
    state = {"order_id": order_id, "customer_request": ask} if style == "json" else f"Order {order_id}. {ask}"
    raw = {"state": state, "group_id": world_id("rel", world), "questions": {
        "rel:pick": {"type": "choice", "instructions": "Which option satisfies the customer's request?",
                     "criteria": {f"o{i}": _item_text(item, style) for i, item in enumerate(items)},
                     "target": [float(item["name"] == gold) for item in items]},
        "rel:is_gold": {"type": "noul", "instructions": f"Is the option named {probe} the correct pick for the customer's request?",
                        "target": [float(probe != gold), float(probe == gold)]}}}
    return world, Request.from_dict(raw)


ORD_LEVELS = {
    5: ["Cosmetic: only one person noticed, nothing was lost, and it was over quickly",
        "Minor: a handful of people were affected and nothing was lost",
        "Moderate: dozens of people were affected, or the problem lasted more than four hours",
        "Major: hundreds of people were affected",
        "Critical: thousands of people were affected, or data was lost"],
    4: ["Minor: at most a handful of people were affected and nothing was lost",
        "Moderate: dozens of people were affected, or the problem lasted more than four hours",
        "Major: hundreds of people were affected",
        "Critical: thousands of people were affected, or data was lost"],
    3: ["Low: at most a handful of people were affected and nothing was lost",
        "Medium: dozens or hundreds of people were affected, or the problem lasted more than four hours",
        "High: thousands of people were affected, or data was lost"]}
ORD_MAPS = {5: {i: i for i in range(5)}, 4: {0: 0, 1: 0, 2: 1, 3: 2, 4: 3}, 3: {0: 0, 1: 0, 2: 1, 3: 1, 4: 2}}


def ordinal_base(users, minutes, data_loss):
    base = 0 if users == 1 else 1 if users < 10 else 2 if users < 100 else 3 if users < 1000 else 4
    if minutes > 240:
        base = max(base, 2)
    return 4 if data_loss else base


def ordinal_world(rng, levels, reversed_order, style):
    users = rng.choice((1, rng.randint(2, 9), rng.randint(10, 99), rng.randint(100, 999), rng.randint(1000, 9999)))
    minutes = rng.choice((rng.randint(1, 240), rng.randint(241, 720)))
    data_loss = rng.random() < .2
    workaround = rng.random() < .5
    base = ordinal_base(users, minutes, data_loss)
    level = ORD_MAPS[levels][base]
    descriptions = list(ORD_LEVELS[levels])
    if reversed_order:
        descriptions = descriptions[::-1]
        level = levels - 1 - level
    incident_id = rng.randrange(100000, 999999)
    world = {"family": "ord", "style": style, "incident_id": incident_id, "users": users, "minutes": minutes,
             "data_loss": data_loss, "workaround": workaround, "base": base, "levels": levels,
             "reversed": reversed_order, "level": level}
    record = {"incident_id": incident_id, "users_affected": users, "duration_minutes": minutes,
              "data_loss": data_loss, "workaround_available": workaround}
    state = record if style == "json" else (f"Incident {incident_id}: {users} users affected. Duration: {minutes} minutes. "
                                            f"Data loss: {'yes' if data_loss else 'no'}. Workaround available: {'yes' if workaround else 'no'}.")
    major_level = ORD_MAPS[levels][3]
    major = ORD_LEVELS[levels][major_level]
    threshold = min(b for b, l in ORD_MAPS[levels].items() if l == major_level)
    raw = {"state": state, "group_id": world_id("ord", world), "questions": {
        "ord:severity": {"type": "score", "instructions": "How severe is this incident?", "criteria": descriptions,
                         "target": [float(i == level) for i in range(levels)]},
        "ord:at_least_major": {"type": "noul", "instructions": f"Is this incident at least as severe as the level described as: {major}?",
                               "target": [float(base < threshold), float(base >= threshold)]}}}
    return world, Request.from_dict(raw)


FEATURES = ("uses_emoji", "formal_greeting", "mentions_deadline", "long_message")
SENDER_POOL = ("Alice", "Bruno", "Chen", "Dara", "Eli", "Farah", "Gus", "Hana", "Ivo", "Jun",
               "Kai", "Lena", "Mira", "Noor", "Omar", "Pia", "Quinn", "Rosa", "Sven", "Tara",
               "Uma", "Vik", "Wren", "Xia", "Yusuf", "Zara", "Ana", "Ben", "Cy", "Dov")


def posterior_family(family_index):
    rng = random.Random(f"posterior-family:{family_index}")
    names = rng.sample(SENDER_POOL, 3)
    theta = [[rng.choice((.1, .3, .7, .9)) for _ in FEATURES] for _ in range(3)]
    return {"names": names, "theta": theta}


def posterior_world(rng, family, family_index, style):
    sender = rng.randrange(3)
    features = {name: rng.random() < family["theta"][sender][f] for f, name in enumerate(FEATURES)}
    likelihood = []
    for k in range(3):
        value = 1.
        for f, name in enumerate(FEATURES):
            p = family["theta"][k][f]
            value *= p if features[name] else 1 - p
        likelihood.append(value)
    posterior = [v / sum(likelihood) for v in likelihood]
    message_id = rng.randrange(100000, 999999)
    world = {"family": "post", "style": style, "family_index": family_index, "message_id": message_id,
             "sender": sender, "features": features, "posterior": posterior}
    phrases = {"uses_emoji": ("uses emoji", "uses no emoji"), "formal_greeting": ("opens with a formal greeting", "opens without a greeting"),
               "mentions_deadline": ("mentions a deadline", "mentions no deadline"), "long_message": ("is long", "is short")}
    state = {"message_id": message_id, "message_features": features} if style == "json" else \
        f"Message {message_id}: the message " + ", ".join(phrases[n][0 if features[n] else 1] for n in FEATURES) + "."
    probe = rng.randrange(3)
    raw = {"state": state, "group_id": f"post:{family_index}:{hashlib.sha256(json.dumps(world, sort_keys=True).encode()).hexdigest()[:20]}",
           "questions": {
               "post:sender": {"type": "choice", "instructions": "Which sender most likely wrote this message?",
                               "criteria": {f"s{k}": family["names"][k] for k in range(3)}, "target": posterior},
               "post:is_named": {"type": "noul", "instructions": f"Was this message sent by {family['names'][probe]}?",
                                 "target": [1 - posterior[probe], posterior[probe]]}}}
    return world, Request.from_dict(raw)
