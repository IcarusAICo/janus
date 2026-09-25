"""T0 world generators for the gap cells the public sets do not cover (spec WP3 section 3.2, gap list).

Four cells: long states with placed evidence, nested JSON records, date and number component selection,
and DOM element action selection. As in `worlds.py`, the gold is computed by code from a hidden world and the
state is a deterministic rendering of it; each generator returns (world, Request, evidence) where `evidence`
is the dict a parser would have to recover from the rendering. Every state carries a nuisance id so identical
worlds never render to identical text across splits."""

from collections import Counter
import datetime as dt
import json
from pathlib import Path
import random

from ..data import assert_disjoint, file_hash, state_hash, write_json, write_jsonl
from ..schema import Request
from .worlds import world_id

DATASET = "JEV_GAPCELLS_V1"
CELL_NAMES = ("long_context", "nested_record", "dates", "dom")

# Measured on 1,920 words of the support-ticket filler below with the Qwen3-0.6B-Base tokenizer
# (revision da87bfb6): 2,801 tokens, 0.685 words per token. Lengths are nominal tokens; `length_words` is exact.
WORDS_PER_TOKEN = .69
LENGTHS = (1000, 4000, 8000, 16000)
LENGTH_WEIGHTS = (4, 3, 2, 1)
DEPTHS = (.1, .5, .9)


def _split(pool, train):
    return tuple(pool[:train]), tuple(pool[train:])


# ----------------------------------------------------------------------------------------------------------
# Cell a: long context with placed evidence
# ----------------------------------------------------------------------------------------------------------

AGENTS = ("Mira", "Tomas", "Priya", "Leo", "Sana", "Iker", "Nadia", "Ravi", "Elin", "Kofi", "Yara", "Oskar",
          "Lin", "Dev", "Marta", "Bo", "Chiara", "Femi")
TRAIN_AGENTS, TEST_AGENTS = _split(AGENTS, 12)
QUEUES = ("Billing", "Tier 2", "Fraud", "Retention", "Onboarding", "Hardware", "Escalations", "Compliance", "Shipping",
          "Refunds", "Accounts", "Security", "Legal", "Enterprise", "Warranty", "Localization", "Payments", "Identity",
          "Partnerships", "Exports")
TRAIN_QUEUES, TEST_QUEUES = _split(QUEUES, 12)
REGIONS = ("Lisbon", "Oslo", "Toronto", "Auckland", "Nairobi", "Denver", "Osaka", "Lima", "Dublin", "Manila", "Cairo",
           "Perth", "Zurich", "Bogota", "Seoul", "Accra", "Vienna", "Quito", "Tallinn", "Hanoi")
TRAIN_REGIONS, TEST_REGIONS = _split(REGIONS, 12)
CUSTOMERS = ("Elena Rossi", "Mark Duval", "Aiko Sato", "Femi Adeyemi", "Lars Berg", "Nina Kowalski", "Omar Haddad",
             "Grace Liu", "Pedro Alves", "Ines Moreau", "Sam Okafor", "Julia Novak")
PRODUCTS = ("Nimbus router", "Atlas desk lamp", "Cinder space heater", "Fjord thermostat", "Lumen smart bulb",
            "Mesa standing desk", "Pike webcam", "Sable headset", "Vale coffee grinder", "Zephyr fan")
ATTRIBUTES = {"queue": {"evidence": "Ticket {tid} was escalated to the {value} queue by agent {agent}.",
                        "question": "To which queue was ticket {tid} escalated?",
                        "probe": "Was ticket {tid} escalated to the {value} queue?",
                        "distractor": "Ticket {other} was escalated to the {value} queue by agent {agent}."},
              "region": {"evidence": "Ticket {tid} was reassigned to the {value} regional office by agent {agent}.",
                         "question": "To which regional office was ticket {tid} reassigned?",
                         "probe": "Was ticket {tid} reassigned to the {value} regional office?",
                         "distractor": "Ticket {other} was reassigned to the {value} regional office by agent {agent}."}}
# Filler never mentions a queue or regional office, so in the `literal` difficulty no candidate value appears
# outside the evidence sentence. Ticket ids in the filler are drawn from 500000-999999; evidence ids from 100000-499999.
FILLER = (
    "Ticket {other} was opened by {customer} about the {product} after a {minutes}-minute outage.",
    "Agent {agent} replied to ticket {other} within {hours} hours and asked for the serial number.",
    "The customer on ticket {other} confirmed the firmware version was {version} and the issue persisted after a reboot.",
    "A replacement {product} was dispatched for ticket {other} and the status was set to pending.",
    "Follow-up on ticket {other}: {customer} reported the new unit works and the ticket was closed after {days} days.",
    "Ticket {other} concerns a duplicate charge of ${amount} on the invoice; agent {agent} documented the payment reference.",
    "{customer} called back on ticket {other} to say the delivery window had been missed twice.",
    "Agent {agent} attached the diagnostic log from the {product} to ticket {other} and requested a second opinion.",
    "Ticket {other} was merged with an older report from the same household about the {product}.",
    "The warranty on the {product} referenced in ticket {other} expires in {days} days according to the purchase record.",
    "Ticket {other}: {customer} asked for written confirmation that the ${amount} credit had been applied.",
    "Agent {agent} left a note on ticket {other} that the customer prefers contact by email after {hours} pm.",
    "Ticket {other} was reopened when the {product} failed again {days} days after the repair.",
    "A courtesy call on ticket {other} lasted {minutes} minutes and ended with the customer satisfied.",
    "The {product} on ticket {other} shipped with the wrong power adapter; a correct one is on its way.",
    "Ticket {other} was placed on hold for {days} days while {customer} travels.",
    "Agent {agent} verified the account on ticket {other} using the last four digits of the payment card.",
    "Ticket {other}: the customer's {product} shows error code E{code} on startup.",
    "{customer} submitted ticket {other} through the mobile app and attached {n} photographs.",
    "Ticket {other} was flagged as a duplicate of an earlier submission and closed by agent {agent}.",
    "The customer on ticket {other} declined the offered replacement and asked for a refund of ${amount} instead.",
    "Ticket {other} records that the {product} arrived {days} days late and the box was damaged.",
    "Agent {agent} scheduled a technician visit for ticket {other} in {days} days.",
    "Ticket {other} was closed automatically after {days} days without a reply from {customer}.",
    "On ticket {other}, {customer} confirmed the address and the order was resent.",
    "Ticket {other}: a firmware update to version {version} resolved the pairing problem with the {product}.",
    "Agent {agent} spent {minutes} minutes on ticket {other} walking the customer through the reset procedure.",
    "Ticket {other} is waiting on the supplier for a spare part for the {product}.",
    "The invoice dispute on ticket {other} was settled with a ${amount} adjustment.",
    "Ticket {other} was rated {n} out of 5 by {customer} in the follow-up survey.",
    "Agent {agent} noted on ticket {other} that the {product} had been purchased second-hand.",
    "Ticket {other}: the customer asked whether the {product} is compatible with the older base station.",
    "A callback for ticket {other} is scheduled in {hours} hours at the customer's request.",
    "Ticket {other} was transferred between two agents before {agent} took ownership.",
    "{customer} reported on ticket {other} that the app shows the {product} as offline every {hours} hours.",
    "Ticket {other}: the shipping label was reprinted and the tracking number shared with the customer.",
    "Agent {agent} confirmed on ticket {other} that no data had been lost during the outage.",
    "The customer on ticket {other} requested a copy of the call recording from {days} days ago.",
    "Ticket {other} was escalated internally for a pricing decision and returned to {agent} the same day.",
    "Ticket {other}: the {product} was found to be running on the wrong regional firmware.")


def _fill(template, rng, agents, **fixed):
    return template.format(other=rng.randrange(500000, 1000000), customer=rng.choice(CUSTOMERS), product=rng.choice(PRODUCTS),
                           agent=rng.choice(agents), minutes=rng.randint(5, 120), hours=rng.randint(1, 12), days=rng.randint(1, 30),
                           amount=rng.randint(5, 900), version=f"{rng.randint(1, 6)}.{rng.randint(0, 9)}.{rng.randint(0, 9)}",
                           code=rng.randint(10, 99), n=rng.randint(1, 5), **fixed)


def long_context_world(rng, vocab, length_tokens, depth, cardinality, difficulty, style):
    """`vocab` has keys agents, queue, region (the split's value pools)."""
    attribute = rng.choice(sorted(ATTRIBUTES))
    forms = ATTRIBUTES[attribute]
    candidates = rng.sample(vocab[attribute], cardinality)
    gold = rng.choice(candidates)
    tid = rng.randrange(100000, 500000)
    agent = rng.choice(vocab["agents"])
    export_id = f"EXP-{rng.randrange(10000, 100000)}"
    evidence_sentence = forms["evidence"].format(tid=tid, value=gold, agent=agent)
    length_words = round(length_tokens * WORDS_PER_TOKEN)
    header = f"Support log export {export_id}."
    sentences, words = [header], len(header.split())
    while words < length_words - len(evidence_sentence.split()):
        sentence = _fill(rng.choice(FILLER), rng, vocab["agents"])
        sentences.append(sentence)
        words += len(sentence.split())
    if difficulty == "distractor":  # other tickets carry every other candidate value, in the same sentence form
        for value in candidates:
            if value != gold:
                index = rng.randrange(1, len(sentences) + 1)
                sentences.insert(index, _fill(forms["distractor"], rng, vocab["agents"], value=value))
    total = sum(len(s.split()) for s in sentences) + len(evidence_sentence.split())
    cumulative, index = len(sentences[0].split()), 1  # the header stays first; evidence goes where the word count reaches depth
    while index < len(sentences) and cumulative + len(sentences[index].split()) <= depth * total:
        cumulative += len(sentences[index].split())
        index += 1
    sentences.insert(index, evidence_sentence)
    paragraphs, start = [], 0
    while start < len(sentences):
        size = rng.randint(3, 6)
        paragraphs.append(" ".join(sentences[start:start + size]))
        start += size
    if style == "json":
        state = {"export_id": export_id, "entries": [{"n": i, "note": p} for i, p in enumerate(paragraphs)]}
    else:
        state = "\n\n".join(paragraphs)
    probe = gold if rng.random() < .5 else rng.choice([c for c in candidates if c != gold])
    order = list(candidates)
    rng.shuffle(order)
    raw = {"state": state, "questions": {
        "lc:pick": {"type": "choice", "instructions": forms["question"].format(tid=tid),
                    "criteria": {f"v{i}": value for i, value in enumerate(order)}, "target": [float(v == gold) for v in order]},
        "lc:is_value": {"type": "noul", "instructions": forms["probe"].format(tid=tid, value=probe),
                        "target": [float(probe != gold), float(probe == gold)]}}}
    request = Request.from_dict(raw)
    before = request.state[:request.state.index(evidence_sentence)]
    depth_actual = len(before.split()) / len(request.state.split())
    world = {"family": "long_context", "style": style, "difficulty": difficulty, "cardinality": cardinality, "export_id": export_id,
             "attribute": attribute, "ticket": tid, "agent": agent, "candidates": candidates, "gold": gold, "probe": probe,
             "length_tokens": length_tokens, "length_words": length_words, "state_words": len(request.state.split()),
             "depth": depth, "depth_actual": round(depth_actual, 4), "evidence_sentence": evidence_sentence}
    request = Request.from_dict({**raw, "group_id": world_id("lc", world)})
    return world, request, {"ticket": tid, "attribute": attribute, "value": gold}


# ----------------------------------------------------------------------------------------------------------
# Cell b: nested JSON records
# ----------------------------------------------------------------------------------------------------------

CITIES = ("Bergen", "Porto", "Graz", "Leeds", "Ghent", "Turku", "Lyon", "Malmo", "Cork", "Basel", "Bilbao", "Utrecht",
          "Aarhus", "Padua", "Nantes", "Leipzig", "Split", "Tartu", "Gdansk", "Brno")
TRAIN_CITIES, TEST_CITIES = _split(CITIES, 12)
STREETS = ("Elm Street", "Harbor Road", "Mill Lane", "Station Way", "Oak Avenue", "Quay Close", "Ridge Drive", "Vine Court")
COUNTRIES = ("Norway", "Portugal", "Austria", "England", "Belgium", "Finland", "France", "Sweden")
TIERS = ("bronze", "silver", "gold", "platinum")
STATUSES = ("pending", "shipped", "delivered", "cancelled", "returned", "on_hold")
FLAGS = ("vip", "fraud_hold", "marketing_opt_in", "past_due")
FLAG_KINDS = {"and": ("Is the customer flagged {a} and also flagged {b}?", lambda x, y: x and y),
              "and_not": ("Is the customer flagged {a} but not flagged {b}?", lambda x, y: x and not y),
              "or": ("Is the customer flagged {a} or flagged {b} (or both)?", lambda x, y: x or y),
              "exactly_one": ("Is exactly one of the flags {a} and {b} set for this customer?", lambda x, y: x != y)}
TRAIN_FLAG_KINDS, TEST_FLAG_KINDS = ("and", "and_not", "or"), ("and", "and_not", "or", "exactly_one")
BUCKETS = {3: (200, 800), 4: (100, 500, 1000), 5: (100, 300, 700, 1500)}
FIELD_KINDS = ("address_city", "order_status")


def bucket_descriptions(levels):
    edges = BUCKETS[levels]
    out = [f"Total under {edges[0]}"]
    out += [f"Total from {lo} to {hi - 1}" for lo, hi in zip(edges, edges[1:])]
    return out + [f"Total {edges[-1]} and above"]


def bucket_of(total, levels):
    return sum(total >= edge for edge in BUCKETS[levels])


def nested_record_world(rng, vocab, orders, cardinality, levels, difficulty, style):
    """`vocab` has keys cities (candidate pool) and flag_kinds."""
    customer_id = f"C-{rng.randrange(100000, 1000000)}"
    candidates = rng.sample(vocab["cities"], cardinality)
    city = rng.choice(candidates)
    others = [c for c in candidates if c != city]
    record = {"customer_id": customer_id, "name": rng.choice(CUSTOMERS), "tier": rng.choice(TIERS),
              "address": {"street": f"{rng.randint(1, 240)} {rng.choice(STREETS)}", "city": city,
                          "postcode": str(rng.randint(1000, 9999)), "country": rng.choice(COUNTRIES)},
              "orders": [], "flags": {flag: rng.random() < .5 for flag in FLAGS}}
    order_ids = rng.sample(range(1000, 10000), orders)
    for oid in order_ids:
        order = {"order_id": f"ORD-{oid}", "total": rng.randint(10, 900), "status": rng.choice(STATUSES), "items": rng.randint(1, 6)}
        if difficulty == "distractor":  # every order ships to a candidate city that is not the billing city
            order["ship_to"] = {"city": rng.choice(others)}
        record["orders"].append(order)
    field_kind = rng.choice(FIELD_KINDS)
    if field_kind == "address_city":
        options, field_gold = list(candidates), city
        instructions = "Which city is the customer's billing address in?"
        named_order = None
    else:
        named_order = rng.choice(record["orders"])
        options, field_gold = list(STATUSES), named_order["status"]
        instructions = f"What is the status of order {named_order['order_id']}?"
    rng.shuffle(options)
    a, b = rng.sample(FLAGS, 2)
    flag_kind = rng.choice(vocab["flag_kinds"])
    flag_text, flag_rule = FLAG_KINDS[flag_kind]
    flag_gold = bool(flag_rule(record["flags"][a], record["flags"][b]))
    total_kind = rng.choice(("literal", "filtered"))
    if total_kind == "literal":
        total = sum(o["total"] for o in record["orders"])
        total_text = "Into which bucket does the sum of the totals of all the customer's orders fall?"
    else:
        total = sum(o["total"] for o in record["orders"] if o["status"] == "delivered")
        total_text = "Into which bucket does the sum of the totals of the customer's delivered orders fall (zero if none is delivered)?"
    level = bucket_of(total, levels)
    if style == "json":
        state = record
    else:
        lines = [f"Customer record {customer_id}. Name: {record['name']}. Tier: {record['tier']}.",
                 f"Billing address: {record['address']['street']}, {city} {record['address']['postcode']}, {record['address']['country']}."]
        for o in record["orders"]:
            ship = f", ships to {o['ship_to']['city']}" if "ship_to" in o else ""
            lines.append(f"Order {o['order_id']}: total {o['total']}, status {o['status']}, {o['items']} items{ship}.")
        lines.append("Flags: " + ", ".join(f"{flag} {'yes' if v else 'no'}" for flag, v in record["flags"].items()) + ".")
        state = "\n".join(lines)
    world = {"family": "nested_record", "style": style, "difficulty": difficulty, "cardinality": len(options), "levels": levels,
             "orders": orders, "customer_id": customer_id, "field_kind": field_kind, "named_order": named_order["order_id"] if named_order else None,
             "field_gold": field_gold, "flag_kind": flag_kind, "flag_a": a, "flag_b": b, "flag_gold": flag_gold,
             "total_kind": total_kind, "total": total, "level": level, "record": record}
    raw = {"state": state, "group_id": world_id("nr", world), "questions": {
        "nr:field": {"type": "choice", "instructions": instructions, "criteria": {f"v{i}": v for i, v in enumerate(options)},
                     "target": [float(v == field_gold) for v in options]},
        "nr:flags": {"type": "noul", "instructions": flag_text.format(a=a, b=b), "target": [float(not flag_gold), float(flag_gold)]},
        "nr:total": {"type": "score", "instructions": total_text, "criteria": bucket_descriptions(levels),
                     "target": [float(i == level) for i in range(levels)]}}}
    evidence = {"address_city": city, "orders": [{k: o[k] for k in ("order_id", "total", "status")} for o in record["orders"]],
                "flags": dict(record["flags"])}
    return world, Request.from_dict(raw), evidence


# ----------------------------------------------------------------------------------------------------------
# Cell c: date and number component selection
# ----------------------------------------------------------------------------------------------------------

MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December")
DATE_FORMATS = ("iso", "long", "us_long")
TRAIN_DATE_FORMATS, TEST_DATE_FORMATS = ("iso", "long"), ("iso", "long", "us_long")
DATE_KINDS = {"earliest_due": "Which of these dates is the earliest due date in the document?",
              "latest_invoice": "Which of these dates is the latest invoice date in the document?",
              "invoice_date_of": "What is the invoice date of invoice {number}?",
              "due_of_largest": "What is the due date of the invoice with the largest amount?",
              "second_earliest_due": "Which of these dates is the second earliest due date in the document?"}
TRAIN_DATE_KINDS = ("earliest_due", "latest_invoice", "invoice_date_of", "due_of_largest")
TEST_DATE_KINDS = TRAIN_DATE_KINDS + ("second_earliest_due",)
VENDORS = ("Harbor Supply", "Meridian Tools", "Northwind Paper", "Quill Office", "Summit Freight", "Tidal Software")


def format_date(day, fmt):
    if fmt == "iso":
        return day.isoformat()
    if fmt == "long":
        return f"{day.day} {MONTHS[day.month - 1]} {day.year}"
    if fmt == "us_long":
        return f"{MONTHS[day.month - 1]} {day.day}, {day.year}"
    raise ValueError(fmt)


def dates_world(rng, vocab, invoices, difficulty, style):
    """`vocab` has keys formats and kinds. Dates and amounts are drawn until every date is distinct and the amounts are distinct."""
    fmt = rng.choice(vocab["formats"])
    kind = rng.choice(vocab["kinds"])
    while True:
        base = dt.date(rng.randint(2022, 2027), rng.randint(1, 12), rng.randint(1, 28))
        rows = []
        for _ in range(invoices):
            issued = base + dt.timedelta(days=rng.randint(0, 120))
            rows.append({"number": f"INV-{rng.randrange(1000, 10000)}", "invoice_date": issued,
                         "due_date": issued + dt.timedelta(days=rng.choice((7, 14, 21, 30, 45, 60))), "amount": rng.randint(50, 9000)})
        extra = []
        if difficulty == "narrative":  # dates that are neither invoice nor due dates appear in the running text
            extra = [("generated", base + dt.timedelta(days=rng.randint(121, 200))), ("reminder", base + dt.timedelta(days=rng.randint(121, 200)))]
        dates = [r["invoice_date"] for r in rows] + [r["due_date"] for r in rows] + [d for _, d in extra]
        numbers = [r["number"] for r in rows]
        if len(set(dates)) == len(dates) and len({r["amount"] for r in rows}) == invoices and len(set(numbers)) == invoices:
            break
    if kind == "earliest_due":
        gold = min(r["due_date"] for r in rows)
    elif kind == "latest_invoice":
        gold = max(r["invoice_date"] for r in rows)
    elif kind == "invoice_date_of":
        named = rng.choice(rows)
        gold = named["invoice_date"]
    elif kind == "due_of_largest":
        gold = max(rows, key=lambda r: r["amount"])["due_date"]
    else:
        gold = sorted(r["due_date"] for r in rows)[1]
    named_number = named["number"] if kind == "invoice_date_of" else None
    total = sum(r["amount"] for r in rows)
    threshold = total + rng.choice((-1, 1)) * rng.randint(1, max(2, total // 5))
    statement_id = f"ST-{rng.randrange(100000, 1000000)}"
    vendor = rng.choice(VENDORS)
    if style == "json":
        state = {"statement_id": statement_id, "vendor": vendor,
                 "invoices": [{"number": r["number"], "invoice_date": format_date(r["invoice_date"], fmt),
                               "due_date": format_date(r["due_date"], fmt), "amount": r["amount"]} for r in rows]}
        for label, day in extra:
            state[f"{label}_on"] = format_date(day, fmt)
    elif difficulty == "narrative":
        lines = [f"Statement {statement_id} from {vendor}" + (f", generated on {format_date(extra[0][1], fmt)}." if extra else ".")]
        for r in rows:
            lines.append(f"Invoice {r['number']} was issued on {format_date(r['invoice_date'], fmt)} for {r['amount']} and is payable by {format_date(r['due_date'], fmt)}.")
        if extra:
            lines.append(f"A payment reminder is scheduled for {format_date(extra[1][1], fmt)}.")
        state = " ".join(lines)
    else:
        lines = [f"Statement {statement_id} from {vendor}."]
        for r in rows:
            lines.append(f"{r['number']}: invoice date {format_date(r['invoice_date'], fmt)}, due date {format_date(r['due_date'], fmt)}, amount {r['amount']}.")
        state = "\n".join(lines)
    options = list(dates)
    rng.shuffle(options)
    world = {"family": "dates", "style": style, "difficulty": difficulty, "cardinality": len(options), "invoices": invoices,
             "format": fmt, "kind": kind, "statement_id": statement_id, "named_number": named_number, "gold": gold.isoformat(),
             "total": total, "threshold": threshold,
             "rows": [{**r, "invoice_date": r["invoice_date"].isoformat(), "due_date": r["due_date"].isoformat()} for r in rows],
             "extra_dates": {label: day.isoformat() for label, day in extra}}
    raw = {"state": state, "group_id": world_id("dt", world), "questions": {
        "dt:pick": {"type": "choice", "instructions": DATE_KINDS[kind].format(number=named_number),
                    "criteria": {f"d{i}": format_date(d, fmt) for i, d in enumerate(options)}, "target": [float(d == gold) for d in options]},
        "dt:total_above": {"type": "noul", "instructions": f"Is the sum of all the invoice amounts in the document above {threshold}?",
                           "target": [float(total <= threshold), float(total > threshold)]}}}
    evidence = {"invoices": world["rows"], "extra_dates": world["extra_dates"]}
    return world, Request.from_dict(raw), evidence


# ----------------------------------------------------------------------------------------------------------
# Cell d: DOM element table action selection
# ----------------------------------------------------------------------------------------------------------

LABELS = ("Add to cart", "Checkout", "Sign in", "Search", "Email", "Password", "Country", "Subscribe", "Continue", "Cancel",
          "Apply coupon", "Track order", "Contact us", "Quantity", "Language", "Currency", "Download invoice", "Submit review",
          "Date of birth", "Shipping method", "Phone number", "Gift message", "Promo code", "Wishlist", "Order history", "Notifications")
TRAIN_LABELS, TEST_LABELS = _split(LABELS, 14)  # the test pool must cover the largest cardinality (12)
SYNONYMS = {"Add to cart": "put the item in the basket", "Checkout": "proceed to payment", "Sign in": "log in", "Search": "look something up",
            "Email": "email address", "Password": "passphrase", "Country": "nation", "Subscribe": "sign up for the newsletter",
            "Continue": "go to the next step", "Cancel": "abandon the form", "Apply coupon": "redeem the discount code",
            "Track order": "follow the shipment", "Contact us": "reach support", "Quantity": "number of units", "Language": "display language",
            "Currency": "billing currency", "Download invoice": "save the bill", "Submit review": "post the rating",
            "Date of birth": "birthday", "Shipping method": "delivery option", "Phone number": "telephone", "Gift message": "card note",
            "Promo code": "voucher", "Wishlist": "saved items", "Order history": "past purchases", "Notifications": "alerts"}
TAGS_BY_OP = {"click": ("button", "a"), "type": ("input", "textarea"), "select": ("select",)}
ROLES = {"button": "button", "a": "link", "input": "textbox", "textarea": "textbox", "select": "combobox", "div": "text", "span": "text",
         "h1": "heading", "img": "image"}
NOISE_TAGS = ("div", "span", "h1", "img")
OPERATIONS = ("click", "type", "select", "scroll", "done")
TYPED_VALUES = ("alex@example.com", "Green Lane 4", "hunter-42", "0400 123 456", "Happy birthday")
SELECT_VALUES = ("Norway", "English", "EUR", "Express", "2")
TAG_NOUN = {"button": "button", "a": "link", "input": "field", "textarea": "field", "select": "dropdown"}
PAGES = ("checkout page of an online shop", "account settings page", "product page", "support portal", "booking form")


def _element_text(e):
    return f"{e['tag']} '{e['text']}' (role {e['role']})"


def dom_world(rng, vocab, cardinality, difficulty, style):
    """`vocab` has key labels. Every element text is distinct except the `distractor` twin, which shares the
    target's text under a different tag."""
    operation = rng.choice(OPERATIONS)
    view_id = f"V-{rng.randrange(100000, 1000000)}"
    labels = rng.sample(vocab["labels"], cardinality)
    elements = []
    target_label = target_tag = value = None
    if operation in TAGS_BY_OP:
        target_tag = rng.choice(TAGS_BY_OP[operation])
        target_label = labels[0]
        elements.append({"tag": target_tag, "text": target_label})
        if difficulty == "distractor":
            twin_tag = rng.choice([t for t in ROLES if t != target_tag and ROLES[t] != ROLES[target_tag]])
            elements.append({"tag": twin_tag, "text": target_label})
    for label in labels[1 if target_label is not None else 0:]:
        if len(elements) >= cardinality:
            break
        elements.append({"tag": rng.choice(tuple(ROLES)), "text": label})
    rng.shuffle(elements)
    for i, e in enumerate(elements):
        e["index"], e["role"] = i, ROLES[e["tag"]]
    if operation == "click":
        noun = TAG_NOUN[target_tag]
        task = (f"Click the '{target_label}' {noun}." if difficulty != "paraphrase"
                else f"Use the {noun} that lets you {SYNONYMS[target_label]}.")
    elif operation == "type":
        value = rng.choice(TYPED_VALUES)
        task = (f"Type '{value}' into the '{target_label}' field." if difficulty != "paraphrase"
                else f"Enter '{value}' in the field for the {SYNONYMS[target_label]}.")
    elif operation == "select":
        value = rng.choice(SELECT_VALUES)
        task = (f"Choose '{value}' in the '{target_label}' dropdown." if difficulty != "paraphrase"
                else f"Pick '{value}' from the dropdown for the {SYNONYMS[target_label]}.")
    elif operation == "scroll":
        task = rng.choice(("Scroll down to reveal the rest of the page.", "The element you need is not visible yet; scroll further down the page."))
    else:
        done_label = rng.choice(labels)
        task = rng.choice((f"The '{done_label}' step has already been completed and nothing else is required; mark the task as done.",
                           "Everything the user asked for is already on the screen; declare the task finished."))
    if target_tag is None:
        gold_index = None
    else:
        gold_index = next(e["index"] for e in elements if e["text"] == target_label and e["tag"] == target_tag)
    page = rng.choice(PAGES)
    if style == "json":
        state = {"view_id": view_id, "page": page, "elements": [{k: e[k] for k in ("index", "tag", "text", "role")} for e in elements], "task": task}
    else:
        lines = [f"Page: {page} (view {view_id}).", "Visible elements:"]
        lines += [f"[{e['index']}] {_element_text(e)}" for e in elements]
        lines.append(f"Task: {task}")
        state = "\n".join(lines)
    criteria = {str(e["index"]): _element_text(e) for e in elements}
    criteria["none"] = "No element: the action does not act on a listed element."
    world = {"family": "dom", "style": style, "difficulty": difficulty, "cardinality": cardinality, "view_id": view_id, "operation": operation,
             "target_label": target_label, "target_tag": target_tag, "value": value, "gold_index": gold_index, "task": task,
             "elements": [{k: e[k] for k in ("index", "tag", "text", "role")} for e in elements]}
    element_gold = "none" if gold_index is None else str(gold_index)
    raw = {"state": state, "group_id": world_id("dom", world), "questions": {
        "dom:element": {"type": "choice", "instructions": "Which visible element should the next action act on?", "criteria": criteria,
                        "target": [float(k == element_gold) for k in criteria]},
        "dom:operation": {"type": "choice", "instructions": "Which operation should the next action perform?",
                          "criteria": {"click": "Click the element", "type": "Type text into the element", "select": "Select a value in the element",
                                       "scroll": "Scroll the page", "done": "Declare the task finished"},
                          "target": [float(op == operation) for op in OPERATIONS]}}}
    evidence = {"elements": world["elements"], "task": {"operation": operation, "target_label": target_label, "target_tag": target_tag, "value": value}}
    return world, Request.from_dict(raw), evidence


# ----------------------------------------------------------------------------------------------------------
# Splits and the dataset
# ----------------------------------------------------------------------------------------------------------

HOLDOUTS = {"long_context": "test uses agent names and queue/region values never in train/dev/calibration (8 of 20 values per attribute)",
            "nested_record": "test uses billing-city candidates never in train/dev/calibration and adds the flag kind exactly_one",
            "dates": "test adds the date format us_long and the kind second_earliest_due",
            "dom": "test uses element labels never in train/dev/calibration (12 of 26), with their own paraphrase synonyms"}


def sample_world(cell, rng, split, style):
    test = split == "test"
    if cell == "long_context":
        vocab = {"agents": TEST_AGENTS if test else TRAIN_AGENTS, "queue": TEST_QUEUES if test else TRAIN_QUEUES,
                 "region": TEST_REGIONS if test else TRAIN_REGIONS}
        length = rng.choices(LENGTHS, weights=LENGTH_WEIGHTS)[0]
        return long_context_world(rng, vocab, length, rng.choice(DEPTHS), rng.randint(4, 8), rng.choice(("literal", "distractor")), style)
    if cell == "nested_record":
        vocab = {"cities": TEST_CITIES if test else TRAIN_CITIES, "flag_kinds": TEST_FLAG_KINDS if test else TRAIN_FLAG_KINDS}
        return nested_record_world(rng, vocab, rng.randint(2, 6), rng.randint(4, 8), rng.choice((3, 4, 5)), rng.choice(("literal", "distractor")), style)
    if cell == "dates":
        vocab = {"formats": TEST_DATE_FORMATS if test else TRAIN_DATE_FORMATS, "kinds": TEST_DATE_KINDS if test else TRAIN_DATE_KINDS}
        return dates_world(rng, vocab, rng.randint(2, 5), rng.choice(("literal", "narrative")), style)
    if cell == "dom":
        vocab = {"labels": TEST_LABELS if test else TRAIN_LABELS}
        return dom_world(rng, vocab, rng.randint(4, 12), rng.choice(("literal", "paraphrase", "distractor")), style)
    raise ValueError(cell)


def generate_rows(cell, count, rng, split, seen=None):
    """Draw `count` fresh worlds; `seen` (group ids and normalised state hashes) is shared across splits so collisions are redrawn."""
    rows, seen = [], set() if seen is None else seen
    while len(rows) < count:
        style = "json" if len(rows) % 2 == 0 else "prose"
        world, request, evidence = sample_world(cell, rng, split, style)
        keys = {("id", request.group_id), ("state", state_hash(request.state))}
        if keys & seen:
            continue
        seen |= keys
        row = {**request.to_dict(), "tier": "T0", "family": cell, "cell": cell, "style": style, "difficulty": world["difficulty"],
               "cardinality": world["cardinality"], "world": world, "evidence": evidence}
        rows.append(row)
    return rows


def prepare_gap_cells(output, seed=17, per_cell=2000, dev=200, calibration=200, test=500, cells=CELL_NAMES):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    sizes = {"train": per_cell, "dev": dev, "calibration": calibration, "test": test}
    splits = {name: [] for name in sizes}
    counts = {name: {} for name in sizes}
    seen = set()
    for cell in cells:
        for split, count in sizes.items():
            rng = random.Random(f"{seed}:{cell}:{split}")
            rows = generate_rows(cell, count, rng, split, seen)
            splits[split].extend(rows)
            counts[split][cell] = len(rows)
    assert_disjoint({name: [Request.from_dict(r) for r in rows] for name, rows in splits.items()})
    for name, rows in splits.items():
        write_jsonl(output / f"{name}.jsonl", rows)
    axes = {}
    for cell in cells:
        rows = [r for rows in splits.values() for r in rows if r["cell"] == cell]
        axes[cell] = {"difficulty": dict(Counter(r["difficulty"] for r in rows)), "style": dict(Counter(r["style"] for r in rows)),
                      "cardinality": dict(sorted(Counter(r["cardinality"] for r in rows).items()))}
        if cell == "long_context":
            axes[cell]["length_tokens"] = dict(sorted(Counter(r["world"]["length_tokens"] for r in rows).items()))
            axes[cell]["depth"] = dict(sorted(Counter(r["world"]["depth"] for r in rows).items()))
            axes[cell]["max_abs_depth_error"] = max(abs(r["world"]["depth_actual"] - r["world"]["depth"]) for r in rows)
    manifest = {"dataset": DATASET, "seed": seed, "tier": "T0", "cells": list(cells), "counts": counts,
                "total": {name: len(rows) for name, rows in splits.items()},
                "holdouts": {cell: HOLDOUTS[cell] for cell in cells}, "axes": axes,
                "length_sampling": {"tokens": list(LENGTHS), "weights": list(LENGTH_WEIGHTS), "words_per_token": WORDS_PER_TOKEN},
                "gold_note": "T0: gold computed by code from the hidden world; the state is a deterministic rendering; no LLM calls.",
                "files": {p.name: {"sha256": file_hash(p), "bytes": p.stat().st_size} for p in sorted(output.glob("*.jsonl"))}}
    write_json(output / "manifest.json", manifest)
    return manifest
