"""Structured Doom observations. Jev sees this text, never pixels."""
from __future__ import annotations

from dataclasses import dataclass, field
import json


MEASUREMENT = {
    "distance_unit": (
        "Doom map unit; not inches, feet, meters, or miles. "
        "The player's body is 32 units wide."
    ),
    "distance_bands": {
        "contact": "under 64 map units; immediate melee range",
        "close": "64 to under 256; nearby dangerous engagement range",
        "medium": "256 to under 768; ranged engagement and meaningful travel distance",
        "far": "768 or more; distant and substantial travel required",
    },
    "relative_bearing_degrees": {
        "0": "straight ahead",
        "positive": "left",
        "negative": "right",
        "+90": "directly left",
        "-90": "directly right",
        "+/-180": "behind",
    },
    "vertical_aim": (
        "automatic when the target is in view; only horizontal bearing matters then. "
        "A shot stops on the first solid thing in front of the muzzle."
    ),
    "time": {
        "one_game_tick": "1/35 second",
        "control_rate": "decisions update every ~4 ticks (~0.1 seconds); controls persist between decisions",
        "controls": "aiming, movement, and the trigger operate simultaneously",
    },
}

PLAYER_PHYSICS = {
    "max_speed": "~292 map units/second",
    "acceleration": "reaches max speed in ~1.1 seconds from standstill",
}

WEAPONS = {
    "fist": {"attack_type": "melee", "effective_range": "contact", "one_liner": "last resort"},
    "pistol": {"attack_type": "hitscan", "ammo_type": "bullets", "effective_range": "medium",
               "one_liner": "weak and inaccurate"},
    "shotgun": {"attack_type": "hitscan", "ammo_type": "shells", "max_ammo": 50,
                "spread_deg": "±5.6", "effective_range": "under 500",
                "one_liner": "devastating up close, pellets scatter with distance"},
    "super shotgun": {"attack_type": "hitscan", "ammo_type": "shells", "max_ammo": 50,
                      "effective_range": "under 300",
                      "one_liner": "point blank annihilation, slow reload"},
    "chaingun": {"attack_type": "hitscan", "ammo_type": "bullets",
                 "one_liner": "sustained fire, walks the spray"},
    "rocket launcher": {"attack_type": "projectile", "ammo_type": "rockets",
                        "one_liner": "splash kills you too"},
    "plasma rifle": {"attack_type": "projectile", "ammo_type": "cells",
                     "one_liner": "fast bolts, melts at mid range"},
}

ENEMY_TYPES = {
    "zombieman": "slow hitscan; dies in one shotgun blast",
    "shotgun zombie": "hitscan pellets; more dangerous than a zombieman",
    "imp": "fires fireballs; closes if you linger",
    "demon": "melee rusher; keep it off your face",
    "cacodemon": "floats and spits; hold range and dodge the ball",
    "lost soul": "charges; sidestep",
}

KIND_EXTRA = {
    "ammo clip": "10 bullets",
    "box of bullets": "50 bullets",
    "shells": "4 shells",
    "box of shells": "20 shells",
    "stimpack": "10 health",
    "medikit": "25 health",
    "health bonus": "1 health",
    "armor bonus": "1 armor",
    "green armor": "100 armor",
    "blue armor": "200 armor",
    "shotgun": "weapon + 8 shells",
    "super shotgun": "weapon + 8 shells",
    "cell pack": "100 cells",
}


def distance_band(distance):
    if distance < 64:
        return "contact"
    if distance < 256:
        return "close"
    if distance < 768:
        return "medium"
    return "far"


def bearing_phrase(deg):
    angle = abs(float(deg))
    if angle <= 8:
        return "dead ahead"
    if angle >= 165:
        return "behind"
    side = "left" if deg > 0 else "right"
    if 80 <= angle <= 100:
        return f"directly {side}"
    if angle < 45:
        return f"ahead {side}"
    return side


def format_distance(distance):
    return f"{int(round(distance))} ({distance_band(distance)})"


def format_bearing(deg):
    value = float(deg)
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.0f}° ({bearing_phrase(value)})"


def actor(label, kind, distance, bearing_deg, role="enemy", visible=True, extra=""):
    return {
        "id": label,
        "kind": kind,
        "distance": int(distance),
        "bearing_deg": float(bearing_deg),
        "role": role,
        "visible": visible,
        "extra": extra,
        "status": "in view" if visible else "out of view",
        "last_seen": "just now",
    }


@dataclass
class Observation:
    health: int
    armor: int
    ammo: int
    kills: int
    facing_deg: float
    weapon: str = "pistol"
    carrying: list = field(default_factory=list)
    velocity_speed: float = 0
    velocity_dir: float = 0
    walls: dict = field(default_factory=dict)
    visible: list = field(default_factory=list)
    remembered: list = field(default_factory=list)
    destinations: list = field(default_factory=list)
    current_room: str | None = None
    director_spawned: int = 0
    director_killed: int = 0


def _actor_payload(entry, remembered=False):
    extra = entry.get("extra") or KIND_EXTRA.get(entry.get("kind"), "")
    labeled = entry["id"]
    if extra and "(" not in labeled:
        labeled = f"{labeled} ({extra})"
    payload = {
        "labeled": labeled,
        "kind": entry.get("kind"),
        "distance": format_distance(entry["distance"]),
        "bearing": format_bearing(entry["bearing_deg"]),
    }
    if remembered or not entry.get("visible", True):
        payload["last_seen"] = entry.get("last_seen", "just now")
        payload["status"] = entry.get("status", "out of view")
    else:
        payload["range"] = distance_band(entry["distance"])
    return payload


def _yaml_dump(value, indent=0):
    pad = "  " * indent
    if isinstance(value, dict):
        if not value:
            return pad + "{}\n"
        lines = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}{key}:")
                lines.append(_yaml_dump(item, indent + 1).rstrip("\n"))
            else:
                lines.append(f"{pad}{key}: {item}")
        return "\n".join(lines) + "\n"
    if isinstance(value, list):
        if not value:
            return pad + "[]\n"
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                dumped = _yaml_dump(item, indent + 1).rstrip("\n")
                first, *rest = dumped.split("\n")
                lines.append(f"{pad}- {first.lstrip()}")
                lines.extend(rest)
            else:
                lines.append(f"{pad}- {item}")
        return "\n".join(lines) + "\n"
    return f"{pad}{value}\n"


def serialize_state(obs, standing_orders, schema="json", override=None):
    payload = {
        "weapons": WEAPONS,
        "enemy_types": ENEMY_TYPES,
        "player_physics": PLAYER_PHYSICS,
        "measurement_context": MEASUREMENT,
        "player": {
            "health": f"{obs.health}/100",
            "armor": f"{obs.armor}/200",
            "weapon": obs.weapon,
            "ammo": obs.ammo,
            "status": None,
            "carrying": list(obs.carrying),
            "keys": None,
            "switching": None,
        },
        "walls": dict(obs.walls),
        "footing": None,
        "velocity": {
            "speed": f"{int(round(obs.velocity_speed))} map units/second",
            "direction": format_bearing(obs.velocity_dir),
        },
        "visible": [_actor_payload(entry) for entry in obs.visible],
        "remembered": [_actor_payload(entry, remembered=True) for entry in obs.remembered],
        "map": {
            "current_room": obs.current_room,
            "destinations": [{"id": d["id"], "text": d["text"]} for d in obs.destinations],
        },
        "director": {"spawned": obs.director_spawned, "killed": obs.director_killed},
        "standing_orders": standing_orders,
    }
    if override:
        payload.update(override)
    if schema == "yaml":
        return _yaml_dump(payload)
    return json.dumps(payload, ensure_ascii=False)


def subject_criteria(obs):
    criteria = {}
    for entry in list(obs.visible) + list(obs.remembered):
        extra = f", {entry['extra']}" if entry.get("extra") else ""
        where = "in view" if entry.get("visible") else "out of view"
        criteria[entry["id"]] = (
            f"{entry['kind']} {where}, distance {format_distance(entry['distance'])}, "
            f"bearing {format_bearing(entry['bearing_deg'])}{extra}"
        )
    for dest in obs.destinations:
        criteria[dest["id"]] = dest["text"]
    criteria["none"] = "No actor, item, or landmark should be the current focus"
    return criteria
