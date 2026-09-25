"""Parallel System One questions and the code that maps answers onto buttons."""
from __future__ import annotations

from demos.doom.state import subject_criteria

QUESTIONS = ("firing", "goal", "subject", "movement", "dodge", "turn")
FIRE_CONE_DEG = 15
ENEMY_ROLES = {"enemy"}
ENEMY_KINDS = {
    "zombieman", "shotgun zombie", "imp", "demon", "spectre", "cacodemon",
    "lost soul", "hell knight", "baron", "pinky demon",
}

GOAL_CRITERIA = {
    "kill_enemy": "Attack a visible or remembered hostile",
    "restore_health": "Move toward a medikit, stimpack, or health bonus",
    "stock_ammo": "Collect ammo or shells",
    "upgrade_weapon": "Pick up a better weapon",
    "add_armor": "Collect armor",
    "scout": "Look for what is out of sight",
    "advance": "Advance through the level toward an unexplored edge or room",
    "flee": "Break contact and back off",
}
GOAL_PHRASE = {
    "kill_enemy": "attacking enemies",
    "restore_health": "trying to restore health",
    "stock_ammo": "restocking ammunition",
    "upgrade_weapon": "upgrading weapons",
    "add_armor": "adding armor",
    "scout": "scouting for what's out of sight",
    "advance": "advancing through the level",
    "flee": "fleeing",
}


def _intent_phrase(prior):
    goal = (prior or {}).get("goal") or "act"
    subject = (prior or {}).get("subject")
    head = GOAL_PHRASE.get(goal, "acting")
    if subject and subject not in {"none", "hold"}:
        return f"{head}, focusing on {subject}"
    return head


def questions_for_observation(obs, prior=None):
    phrase = _intent_phrase(prior)
    subjects = subject_criteria(obs)
    look = {key: value for key, value in subjects.items() if key != "none"}
    look["hold"] = "Hold current facing"
    return {
        "firing": {
            "type": "choice",
            "instructions": "Should the player's trigger be held down right now?",
            "criteria": {"fire": "Hold the trigger", "hold_fire": "Do not fire"},
        },
        "goal": {
            "type": "choice",
            "instructions": (
                "Considering 'player', 'enemies', and 'items', what is the player's "
                "highest-priority goal right now?"
            ),
            "criteria": dict(GOAL_CRITERIA),
        },
        "subject": {
            "type": "choice",
            "instructions": "Which actor, item, or landmark should the player focus on?",
            "criteria": subjects,
        },
        "movement": {
            "type": "choice",
            "instructions": (
                f"The player's current top priority is {phrase}. "
                "Given the current situation, how should the player move right now?"
            ),
            "criteria": {
                "close_in": "Close distance to the focus",
                "hold_range": "Keep a fighting distance; back off if in contact",
                "hold_ground": "Do not walk",
                "walk_to": "Walk toward the focus or landmark",
                "back_off": "Move away from the focus",
            },
        },
        "dodge": {
            "type": "choice",
            "instructions": (
                f"The player's current top priority is {phrase}. "
                "What does this exact moment call for?"
            ),
            "criteria": {
                "carry_on": "No emergency dodge",
                "dodge_left": "Sidestep left",
                "dodge_right": "Sidestep right",
                "dodge_back": "Step back",
            },
        },
        "turn": {
            "type": "choice",
            "instructions": (
                "Should the player point right now? Looking and aiming are one act: "
                "the gun goes where the eyes go."
            ),
            "criteria": look,
        },
    }


def _focus(obs, decision):
    chosen = (decision.get("turn") or {}).get("choice")
    if not chosen or chosen == "hold":
        chosen = (decision.get("subject") or {}).get("choice")
    if not chosen or chosen in {"none", "hold"}:
        return None
    for entry in list(obs.visible) + list(obs.remembered) + list(obs.destinations):
        if entry.get("id") == chosen:
            return entry
    return None


def _is_enemy(entry):
    return entry.get("role") in ENEMY_ROLES or entry.get("kind") in ENEMY_KINDS


def buttons_from_decision(obs, decision):
    """Code owns geometry. Jev owns goal, subject, and semantic movement."""
    focus = _focus(obs, decision)
    dodge = (decision.get("dodge") or {}).get("choice") or "carry_on"
    move = (decision.get("movement") or {}).get("choice") or "hold_ground"
    fire = (decision.get("firing") or {}).get("choice") == "fire"
    goal = (decision.get("goal") or {}).get("choice")
    bearing = float(focus["bearing_deg"]) if focus else 0.0
    dist = float(focus["distance"]) if focus and focus.get("distance") is not None else 10_000
    in_cone = bool(focus and _is_enemy(focus) and abs(bearing) <= FIRE_CONE_DEG)

    walk = move in {"close_in", "walk_to"}
    back = move == "back_off" or goal == "flee" or (move == "hold_range" and dist < 64)
    if move == "hold_range" and dist >= 256:
        walk = True
    if dodge == "dodge_back":
        back = True
        walk = False

    buttons = {
        "ATTACK": bool(fire or in_cone),
        "MOVE_FORWARD": walk and dodge == "carry_on" and not back,
        "MOVE_BACKWARD": bool(back),
        "MOVE_LEFT": dodge == "dodge_left",
        "MOVE_RIGHT": dodge == "dodge_right",
        "TURN_LEFT": False,
        "TURN_RIGHT": False,
        "USE": False,
    }
    if focus and dodge == "carry_on":
        if bearing > 8:
            buttons["TURN_LEFT"] = True
        elif bearing < -8:
            buttons["TURN_RIGHT"] = True
    if buttons["MOVE_FORWARD"] and (obs.walls or {}).get("ahead") == "close":
        buttons["USE"] = True
    if buttons["MOVE_LEFT"] and buttons["MOVE_RIGHT"]:
        buttons["MOVE_RIGHT"] = False
    if buttons["TURN_LEFT"] and buttons["TURN_RIGHT"]:
        buttons["TURN_RIGHT"] = False
    if buttons["MOVE_FORWARD"] and buttons["MOVE_BACKWARD"]:
        buttons["MOVE_BACKWARD"] = False
    return buttons
