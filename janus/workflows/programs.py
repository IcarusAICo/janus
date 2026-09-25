"""Five deterministic workflows over T0 hidden worlds (spec WP4a).

A workflow asks several questions of one JSON state, runs plain code on the answers, and scores the final
result with a verifier in [0, 1]. Each world is generated together with its per-question gold answers and
the gold outcome, so the same states serve the supervised control. The program sees the state and the
answers only; the verifier sees the world, including what the generator knows."""

from dataclasses import dataclass, field
import hashlib
import json
import random
from pathlib import Path

from ..data import assert_disjoint, file_hash, state_hash, write_json, write_jsonl
from ..schema import Question, Request
from ..synth.worlds import NAMES
from .env import ACTIONS, ACTION_RULES, ACTION_TEXT, Grid, budgeted_grid, distance, optimal_actions, success_within


@dataclass(frozen=True)
class World:
    workflow: str
    state: dict
    gold: dict
    outcome: object
    targets: dict = field(default_factory=dict)
    hidden: dict = field(default_factory=dict)

    def to_dict(self):
        return {"workflow": self.workflow, "state": self.state, "gold": self.gold, "outcome": self.outcome,
                "targets": self.targets, "hidden": self.hidden}

    @classmethod
    def from_dict(cls, raw):
        return cls(raw["workflow"], raw["state"], raw["gold"], raw["outcome"], raw.get("targets", {}), raw.get("hidden", {}))


def world_id(name, state):
    return f"{name}:{hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()[:20]}"


def ordered_credit(levels, a, b):
    """1 for equal ordered outcomes, decreasing linearly with their distance, 0 at the extremes."""
    return 1 - abs(levels.index(a) - levels.index(b)) / (len(levels) - 1)


class Workflow:
    name = ""

    def sample(self, rng):
        raise NotImplementedError

    def raw_questions(self, state):
        raise NotImplementedError

    def program(self, state, answers):
        raise NotImplementedError

    def verify(self, world, outcome):
        raise NotImplementedError

    def questions(self, state):
        return Request.from_dict({"state": state, "group_id": world_id(self.name, state), "questions": self.raw_questions(state)})

    def request(self, world):
        """The questions with their gold targets: one-hot on the gold key unless the world names a distribution."""
        raw = self.raw_questions(world.state)
        for qid, q in raw.items():
            keys = [o.key for o in Question.from_dict(qid, q).options]
            q["target"] = list(world.targets.get(qid) or [float(k == world.gold[qid]) for k in keys])
        return Request.from_dict({"state": world.state, "group_id": world_id(self.name, world.state), "questions": raw})

    def make(self, state, gold, targets=None, hidden=None):
        return World(self.name, state, gold, self.program(state, gold), targets or {}, hidden or {})


def _flag(value):
    return "true" if value else "false"


CATEGORIES = {"software": "engineering", "travel": "sales", "office": "operations", "advertising": "marketing"}
ITEM_WORDS = {"software": ("licence renewal", "cloud credits", "monitoring subscription"),
              "travel": ("flight", "hotel nights", "rail fare"),
              "office": ("standing desks", "printer toner", "copy paper"),
              "advertising": ("banner campaign", "sponsored post", "billboard lease")}
RISK_LEVELS = ("Low: the vendor is verified and the invoice number does not appear among the recent invoice numbers",
               "Medium: the vendor is new and the invoice number does not appear among the recent invoice numbers",
               "High: the vendor is flagged, or the invoice number already appears among the recent invoice numbers")
DECISIONS = ("approve", "hold", "reject")


class InvoiceApproval(Workflow):
    """Choice department + Noul over-limit + Score risk -> approve, hold or reject, charged to a department."""
    name = "invoice"

    def sample(self, rng):
        while True:
            items = [{"description": rng.choice(ITEM_WORDS[c]), "category": c, "cost": rng.randrange(1, 40) * 50}
                     for c in rng.sample(sorted(CATEGORIES), rng.choice((2, 3)))]
            if rng.random() < .3:
                c = items[0]["category"]
                items.append({"description": rng.choice(ITEM_WORDS[c]), "category": c, "cost": rng.randrange(1, 40) * 50})
            totals = {}
            for item in items:
                totals[item["category"]] = totals.get(item["category"], 0) + item["cost"]
            if list(totals.values()).count(max(totals.values())) == 1:
                break
        rng.shuffle(items)
        department = CATEGORIES[max(totals, key=totals.get)]
        limits = {d: rng.choice((1000, 2000, 3000, 5000)) for d in sorted(CATEGORIES.values())}
        status = rng.choice(("verified", "new", "flagged"))
        number = f"INV-{rng.randrange(1000, 9999)}"
        recent = [f"INV-{rng.randrange(1000, 9999)}" for _ in range(3)]
        if rng.random() < .25:
            recent[rng.randrange(3)] = number
        state = {"invoice_id": rng.randrange(100000, 999999), "vendor": {"name": f"{rng.choice(NAMES)} Supply", "status": status},
                 "invoice_number": number, "items": items, "routing": dict(CATEGORIES), "approval_limits": limits,
                 "recent_invoice_numbers": recent}
        total = sum(item["cost"] for item in items)
        risk = 2 if status == "flagged" or number in recent else 1 if status == "new" else 0
        gold = {"invoice:department": department, "invoice:over_limit": _flag(total > limits[department]), "invoice:risk": str(risk)}
        return self.make(state, gold, hidden={"total": total})

    def raw_questions(self, state):
        departments = sorted(set(state["routing"].values()))
        return {"invoice:department": {"type": "choice", "instructions": "Which department's budget should this invoice be charged to? "
                                       "Charge the department that the routing table assigns to the category with the largest total cost on the invoice.",
                                       "criteria": {d: f"The {d} department" for d in departments}},
                "invoice:over_limit": {"type": "noul", "instructions": "Is the invoice total, the sum of all item costs, above the approval limit "
                                       "of the department that should be charged?"},
                "invoice:risk": {"type": "score", "instructions": "How risky is this invoice?", "criteria": list(RISK_LEVELS)}}

    def program(self, state, answers):
        risk = int(answers["invoice:risk"])
        over = answers["invoice:over_limit"] == "true"
        decision = "reject" if risk == 2 else "hold" if over or risk == 1 else "approve"
        return {"decision": decision, "department": answers["invoice:department"]}

    def verify(self, world, outcome):
        return (.5 * ordered_credit(DECISIONS, outcome["decision"], world.outcome["decision"])
                + .5 * float(outcome["department"] == world.outcome["department"]))


TEAMS = {"billing": ("refund", "invoice", "charge"), "technical": ("crash", "login", "bug"),
         "account": ("password", "email", "profile"), "sales": ("upgrade", "quote", "demo")}
FRUSTRATION_LEVELS = ("Calm: no prior contacts about this issue and at most one exclamation mark",
                      "Annoyed: one or two prior contacts, or two or three exclamation marks, and not furious",
                      "Furious: three or more prior contacts, or four or more exclamation marks")
PRIORITIES = ("P0", "P1", "P2", "P3")


class TicketRouting(Workflow):
    """Choice team + Noul urgent + Score frustration -> queue and priority, with escalation."""
    name = "ticket"

    def sample(self, rng):
        team = rng.choice(sorted(TEAMS))
        tags = list(rng.sample(TEAMS[team], 2))
        if rng.random() < .5:
            tags.append(rng.choice(TEAMS[rng.choice([t for t in sorted(TEAMS) if t != team])]))
        rng.shuffle(tags)
        tier = rng.choice(("free", "business", "enterprise"))
        sla, down = rng.randrange(1, 49), rng.random() < .15
        prior, marks = rng.choice((0, 0, 1, 2, 3, 4)), rng.choice((0, 1, 2, 3, 4, 6))
        state = {"ticket_id": rng.randrange(100000, 999999), "customer_tier": tier, "sla_hours_remaining": sla, "service_down": down,
                 "tags": tags, "prior_contacts": prior, "exclamation_marks": marks, "teams": {t: list(v) for t, v in TEAMS.items()}}
        urgent = down or (tier == "enterprise" and sla <= 8)
        frustration = 2 if prior >= 3 or marks >= 4 else 1 if prior >= 1 or marks >= 2 else 0
        gold = {"ticket:team": team, "ticket:urgent": _flag(urgent), "ticket:frustration": str(frustration)}
        return self.make(state, gold)

    def raw_questions(self, state):
        return {"ticket:team": {"type": "choice", "instructions": "Which team should handle this ticket? Each team lists the tags it handles; "
                                "pick the team that handles the most of the ticket's tags.",
                                "criteria": {t: f"The {t} team, handling the tags {', '.join(state['teams'][t])}" for t in sorted(state["teams"])}},
                "ticket:urgent": {"type": "noul", "instructions": "Is this ticket urgent? A ticket is urgent when the service is down, "
                                  "or when the customer is on the enterprise tier and at most 8 SLA hours remain."},
                "ticket:frustration": {"type": "score", "instructions": "How frustrated is the customer?", "criteria": list(FRUSTRATION_LEVELS)}}

    def program(self, state, answers):
        urgent = answers["ticket:urgent"] == "true"
        frustration = int(answers["ticket:frustration"])
        priority = "P0" if urgent and frustration == 2 else "P1" if urgent or frustration == 2 else "P2" if frustration == 1 else "P3"
        return {"queue": "escalation" if priority == "P0" else answers["ticket:team"], "priority": priority}

    def verify(self, world, outcome):
        return (.5 * float(outcome["queue"] == world.outcome["queue"])
                + .5 * ordered_credit(PRIORITIES, outcome["priority"], world.outcome["priority"]))


SIGNALS = ("threat_language", "slur", "targets_person", "sexual_content", "spam_links")
MOD_CATEGORIES = {"violence": "Violence: threat language", "harassment": "Harassment: a slur", "adult": "Adult: sexual content",
                  "spam": "Spam: spam links", "none": "None: no policy category applies"}
SEVERITY_LEVELS = ("None: no policy signal is present",
                   "Low: spam links and nothing worse",
                   "Medium: a slur that does not target a specific person, or sexual content, and nothing worse",
                   "High: threat language, or a slur that targets a specific person")
MOD_ACTIONS = ("allow", "flag", "block")


class Moderation(Workflow):
    """Noul harmful + Score severity + Choice category -> allow, flag or block, gated on severity and strikes."""
    name = "moderation"

    def sample(self, rng):
        signals = sorted(s for s in SIGNALS if rng.random() < (.4 if s == "targets_person" else .3))
        state = {"post_id": rng.randrange(100000, 999999), "author_strikes": rng.randrange(0, 4), "reports": rng.randrange(0, 10),
                 "signals": signals}
        present = set(signals)
        harmful = bool(present & {"threat_language", "slur", "sexual_content", "spam_links"})
        category = ("violence" if "threat_language" in present else "harassment" if "slur" in present
                    else "adult" if "sexual_content" in present else "spam" if "spam_links" in present else "none")
        severity = (3 if "threat_language" in present or {"slur", "targets_person"} <= present
                    else 2 if "slur" in present or "sexual_content" in present else 1 if "spam_links" in present else 0)
        gold = {"moderation:harmful": _flag(harmful), "moderation:severity": str(severity), "moderation:category": category}
        return self.make(state, gold)

    def raw_questions(self, state):
        return {"moderation:harmful": {"type": "noul", "instructions": "Does this post violate policy? It does when any of these signals is present: "
                                       "threat_language, slur, sexual_content, spam_links. The signal targets_person alone is not a violation."},
                "moderation:severity": {"type": "score", "instructions": "How severe is the worst policy signal in this post?",
                                        "criteria": list(SEVERITY_LEVELS)},
                "moderation:category": {"type": "choice", "instructions": "Which policy category applies? When several apply, use the first "
                                        "in this order: violence, harassment, adult, spam.", "criteria": dict(MOD_CATEGORIES)}}

    def program(self, state, answers):
        severity = int(answers["moderation:severity"])
        if answers["moderation:harmful"] != "true":
            action = "allow"
        elif severity >= 2 or (severity >= 1 and state["author_strikes"] >= 2):
            action = "block"
        else:
            action = "flag"
        return {"action": action, "category": answers["moderation:category"] if action != "allow" else "none"}

    def verify(self, world, outcome):
        return (.5 * ordered_credit(MOD_ACTIONS, outcome["action"], world.outcome["action"])
                + .5 * float(outcome["category"] == world.outcome["category"]))


PAGES = ("Pricing", "Docs", "Blog", "Careers", "Support", "Settings")
FORMS = ("login", "profile", "checkout", "search")
FIELDS = ("Email", "Password", "Address", "Phone", "Company")
DROPDOWNS = ("Country", "Language", "Plan", "Currency")
POOLS = {"a": PAGES, "button": FORMS, "input": FIELDS, "select": DROPDOWNS}
OPERATIONS = {"click": "Click the element", "type": "Type text into the element", "select": "Choose an option in the element",
              "hover": "Move the pointer over the element"}
GOAL_OPERATION = {"a": "click", "button": "click", "input": "type", "select": "select"}


def _element(tag, name):
    if tag == "a":
        return {"tag": "a", "text": name, "href": f"/{name.lower()}"}
    if tag == "button":
        return {"tag": "button", "text": name.capitalize(), "form": name}
    return {"tag": tag, "label": name}


class DomAction(Workflow):
    """Choice operation + Choice target among element rows -> an action whose effect on the page is checked."""
    name = "dom"

    def sample(self, rng):
        tag = rng.choice(sorted(POOLS))
        name = rng.choice(POOLS[tag])
        gold_row = _element(tag, name)
        rows = [gold_row]
        for other in rng.sample([n for n in POOLS[tag] if n != name], rng.choice((1, 2))):
            rows.append(_element(tag, other))
        for other_tag in rng.sample([t for t in sorted(POOLS) if t != tag], rng.choice((2, 3))):
            rows.append(_element(other_tag, rng.choice(POOLS[other_tag])))
        rng.shuffle(rows)
        elements = [{"id": f"e{i}", **row} for i, row in enumerate(rows)]
        target = next(e["id"] for e, row in zip(elements, rows) if row is gold_row)
        goal = {"a": f"Open the page titled {name}", "button": f"Submit the {name} form",
                "input": f"Enter the customer's {name.lower()} into the field labelled {name}",
                "select": f"Choose a value in the dropdown labelled {name}"}[tag]
        state = {"url": f"/{rng.choice(PAGES).lower()}", "goal": goal, "elements": elements}
        return self.make(state, {"dom:operation": GOAL_OPERATION[tag], "dom:target": target})

    def raw_questions(self, state):
        return {"dom:operation": {"type": "choice", "instructions": "Which operation achieves the goal on the right element? Links and buttons "
                                  "are clicked, text fields are typed into, dropdowns are selected from.", "criteria": dict(OPERATIONS)},
                "dom:target": {"type": "choice", "instructions": "Which element should the operation act on to achieve the goal?",
                               "criteria": {e["id"]: json.dumps(e, sort_keys=True) for e in state["elements"]}}}

    def program(self, state, answers):
        operation, target = answers["dom:operation"], answers["dom:target"]
        element = next(e for e in state["elements"] if e["id"] == target)
        if operation == "click" and element["tag"] == "a":
            effect = {"navigated": element["href"]}
        elif operation == "click" and element["tag"] == "button":
            effect = {"submitted": element["form"]}
        elif operation == "type" and element["tag"] == "input":
            effect = {"typed_into": element["label"]}
        elif operation == "select" and element["tag"] == "select":
            effect = {"selected_in": element["label"]}
        else:
            effect = {"error": f"{operation} is not valid on <{element['tag']}>"}
        return {"target": target, "effect": effect}

    def verify(self, world, outcome):
        if outcome["effect"] == world.outcome["effect"]:
            return 1.
        return .5 if outcome["target"] == world.outcome["target"] else 0.


class GameControl(Workflow):
    """Choice action + Noul success-within-N for a proposed action -> the move actually taken."""
    name = "game"

    def sample(self, rng):
        # Exact budgets (no slack): success requires progress on every step, so the Noul, the gold
        # outcome, and the progress-requiring verifier agree and no constant policy scores well.
        grid, budget = budgeted_grid(rng, slack=(0,), max_distance=6)
        proposed = rng.choice(ACTIONS)
        state = {**grid.to_state(), "budget": budget, "proposed_action": proposed}
        optimal = optimal_actions(grid)
        gold = {"game:action": optimal[0], "game:proposed_success": _flag(success_within(grid, proposed, budget))}
        targets = {"game:action": [1 / len(optimal) if a in optimal else 0. for a in ACTIONS]}
        return self.make(state, gold, targets)

    def raw_questions(self, state):
        return {"game:action": {"type": "choice", "instructions": "Which action moves the agent one step closer to the goal along a "
                                "shortest path that never enters a hazard? " + ACTION_RULES,
                                "criteria": {a: ACTION_TEXT[a] for a in ACTIONS}},
                "game:proposed_success": {"type": "noul", "instructions": f"If the agent's next action is '{state['proposed_action']}', "
                                          f"can it still enter the goal within {state['budget']} steps in total (counting that action) "
                                          "without ever entering a hazard? " + ACTION_RULES}}

    def program(self, state, answers):
        follow = answers["game:proposed_success"] == "true"
        return {"move": state["proposed_action"] if follow else answers["game:action"]}

    def verify(self, world, outcome):
        grid = Grid.from_state(world.state)
        before = distance(grid)
        after = grid.move(outcome["move"])
        if after.agent in grid.hazards:
            return 0.
        d = distance(after)
        # Full credit only for a safe move that gets closer to the goal and keeps it reachable in budget;
        # a safe move that makes no progress (including "wait") earns 0.25, so a constant policy cannot score well.
        if d is not None and before is not None and d < before and d <= world.state["budget"] - 1:
            return 1.
        return .25


WORKFLOWS = (InvoiceApproval(), TicketRouting(), Moderation(), DomAction(), GameControl())
WORKFLOW_BY_NAME = {w.name: w for w in WORKFLOWS}


def generate_worlds(count, seed, workflows=WORKFLOWS, seen=None):
    """`count` worlds, round-robin over the workflows, unique by group id and by normalised state text."""
    rng = random.Random(f"workflows:{seed}")
    seen = set() if seen is None else seen
    worlds = []
    while len(worlds) < count:
        workflow = workflows[len(worlds) % len(workflows)]
        world = workflow.sample(rng)
        request = workflow.request(world)
        keys = {("id", request.group_id), ("state", state_hash(request.state))}
        if keys & seen:
            continue
        seen |= keys
        worlds.append(world)
    return worlds


def write_worlds(path, worlds):
    rows = [{**WORKFLOW_BY_NAME[w.workflow].request(w).to_dict(), "tier": "T0", "world": w.to_dict()} for w in worlds]
    write_jsonl(path, rows)


def load_worlds(path):
    return [World.from_dict(json.loads(line)["world"]) for line in Path(path).read_text().splitlines() if line.strip()]


def prepare_workflow_data(output, train=2000, heldout=500, seed=17):
    """Disjoint train and held-out world files, readable both as worlds and as labelled requests."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    seen = set()
    splits = {"train": generate_worlds(train, seed, seen=seen), "heldout": generate_worlds(heldout, seed + 1, seen=seen)}
    for name, worlds in splits.items():
        write_worlds(output / f"{name}.jsonl", worlds)
    assert_disjoint({name: [WORKFLOW_BY_NAME[w.workflow].request(w) for w in worlds] for name, worlds in splits.items()})
    manifest = {"dataset": "JEV_WORKFLOWS_V1", "seed": seed, "tier": "T0",
                "counts": {name: len(worlds) for name, worlds in splits.items()},
                "workflows": {name: {w.name: sum(x.workflow == w.name for x in worlds) for w in WORKFLOWS} for name, worlds in splits.items()},
                "files": {p.name: file_hash(p) for p in sorted(output.glob("*.jsonl"))}}
    write_json(output / "manifest.json", manifest)
    return manifest
