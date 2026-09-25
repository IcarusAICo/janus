"""ViZDoom session: 35 Hz ticks, async System One decisions, last buttons held."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import threading
import time

from demos.doom.layout import destinations_for, room_at, rooms_from_sectors
from demos.doom.policy import GOAL_PHRASE, buttons_from_decision, questions_for_observation
from demos.doom.state import KIND_EXTRA, Observation, serialize_state
from demos.doom.track import Roster

BUTTON_ORDER = (
    "ATTACK", "MOVE_FORWARD", "MOVE_BACKWARD", "MOVE_LEFT", "MOVE_RIGHT",
    "TURN_LEFT", "TURN_RIGHT", "USE",
)
WEAPON_SLOT = {
    1: "fist", 2: "pistol", 3: "shotgun", 4: "chaingun",
    5: "rocket launcher", 6: "plasma rifle", 7: "bfg",
}
WEAPON_AMMO = {
    2: (2, "bullets"), 3: (3, "shells"), 4: (2, "bullets"),
    5: (4, "rockets"), 6: (5, "cells"), 7: (5, "cells"),
}
CATALOG = {
    "DoomImp": ("imp", "enemy"), "Cacodemon": ("cacodemon", "enemy"),
    "Zombieman": ("zombieman", "enemy"), "ShotgunGuy": ("shotgun zombie", "enemy"),
    "ChaingunGuy": ("chaingunner", "enemy"), "Demon": ("demon", "enemy"),
    "Spectre": ("spectre", "enemy"), "LostSoul": ("lost soul", "enemy"),
    "HellKnight": ("hell knight", "enemy"), "BaronOfHell": ("baron", "enemy"),
    "Medikit": ("medikit", "health"), "Stimpack": ("stimpack", "health"),
    "HealthBonus": ("health bonus", "health"), "Soulsphere": ("soulsphere", "health"),
    "Clip": ("ammo clip", "ammo"), "ClipBox": ("box of bullets", "ammo"),
    "Shell": ("shells", "ammo"), "ShellBox": ("box of shells", "ammo"),
    "Cell": ("cell", "ammo"), "CellPack": ("cell pack", "ammo"),
    "RocketAmmo": ("rocket", "ammo"), "RocketBox": ("box of rockets", "ammo"),
    "GreenArmor": ("green armor", "armor"), "BlueArmor": ("blue armor", "armor"),
    "ArmorBonus": ("armor bonus", "armor"),
    "Shotgun": ("shotgun", "weapon"), "SuperShotgun": ("super shotgun", "weapon"),
    "Chaingun": ("chaingun", "weapon"), "RocketLauncher": ("rocket launcher", "weapon"),
    "PlasmaRifle": ("plasma rifle", "weapon"), "Chainsaw": ("chainsaw", "weapon"),
}


def _bearing(dx, dy, facing_deg):
    angle = math.degrees(math.atan2(dy, dx)) - facing_deg
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return round(angle, 1)


def _distance(dx, dy):
    return int(round(math.hypot(dx, dy)))


def _raw_thing(oid, name, x, y, px, py, facing):
    spec = CATALOG.get(name)
    if spec is None:
        return None
    kind, role = spec
    dx, dy = x - px, y - py
    return {
        "object_id": oid, "kind": kind, "role": role,
        "x": x, "y": y, "distance": _distance(dx, dy),
        "bearing_deg": _bearing(dx, dy, facing),
        "extra": KIND_EXTRA.get(kind, ""),
    }


def observation_from_vizdoom(game, state, roster, world):
    import vizdoom as vzd
    health = int(game.get_game_variable(vzd.GameVariable.HEALTH))
    armor = int(game.get_game_variable(vzd.GameVariable.ARMOR))
    ammo = int(game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO))
    kills = int(game.get_game_variable(vzd.GameVariable.KILLCOUNT))
    facing = float(game.get_game_variable(vzd.GameVariable.ANGLE))
    px = float(game.get_game_variable(vzd.GameVariable.POSITION_X))
    py = float(game.get_game_variable(vzd.GameVariable.POSITION_Y))
    vx = float(game.get_game_variable(vzd.GameVariable.VELOCITY_X))
    vy = float(game.get_game_variable(vzd.GameVariable.VELOCITY_Y))
    slot = int(game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON))
    weapon = WEAPON_SLOT.get(slot, "pistol")
    carrying = []
    for index, name in WEAPON_SLOT.items():
        if index != 1 and not game.get_game_variable(getattr(vzd.GameVariable, f"WEAPON{index}")):
            continue
        note = name
        ammo_slot = WEAPON_AMMO.get(index)
        if ammo_slot:
            count = int(game.get_game_variable(getattr(vzd.GameVariable, f"AMMO{ammo_slot[0]}")))
            note = f"{name} ({count} {ammo_slot[1]})"
        if name == weapon:
            note = f"{note}, in hand" if "(" in note else f"{name} (in hand)"
        carrying.append(note)

    visible_raw, known_raw, enemy_count = [], [], 0
    seen = set()
    for lab in list(getattr(state, "labels", None) or []):
        name = getattr(lab, "object_name", "") or getattr(lab, "object_category", "") or ""
        if name.lower() in {"player", "doomplayer"}:
            continue
        raw = _raw_thing(int(lab.object_id), name, float(lab.object_position_x),
                         float(lab.object_position_y), px, py, facing)
        if raw is None or raw["object_id"] in seen:
            continue
        seen.add(raw["object_id"])
        visible_raw.append(raw)
    for obj in list(getattr(state, "objects", None) or []):
        raw = _raw_thing(int(obj.id), obj.name, float(obj.position_x),
                         float(obj.position_y), px, py, facing)
        if raw is None:
            continue
        if raw["role"] == "enemy":
            enemy_count += 1
        known_raw.append(raw)
    known_raw.sort(key=lambda item: item["distance"])
    tracked = roster.update(visible_raw, known_raw[:24])
    if world["rooms"] is None:
        world["rooms"] = rooms_from_sectors(list(getattr(state, "sectors", None) or []))
    here = room_at(world["rooms"], px, py)
    if here:
        world["visited"].add(here["id"])
    dests = destinations_for(world["rooms"], world["visited"], (px, py), facing)
    if world["spawned"] == 0:
        world["spawned"] = enemy_count
    world["spawned"] = max(world["spawned"], enemy_count + kills)
    speed = math.hypot(vx, vy)
    vel_dir = _bearing(vx, vy, facing) if speed > 1 else 0.0
    return Observation(
        health=health, armor=armor, ammo=ammo, kills=kills,
        facing_deg=round(facing, 1), weapon=weapon, carrying=carrying,
        velocity_speed=speed, velocity_dir=vel_dir,
        walls=_walls_from_depth(getattr(state, "depth_buffer", None)),
        visible=tracked["visible"][:12], remembered=tracked["remembered"][:12],
        destinations=dests, current_room=None if here is None else here["id"],
        director_spawned=world["spawned"], director_killed=kills,
    ), {"rooms": [
        {"id": room["id"], "x": room["cx"], "y": room["cy"],
         "visited": room["id"] in world["visited"]}
        for room in (world["rooms"] or [])
    ], "player": {"x": px, "y": py, "facing": facing},
        "destinations": dests}


def _walls_from_depth(depth):
    if depth is None:
        return {"ahead": "unknown", "left": "unknown", "right": "unknown"}

    def band(arr):
        if arr is None or getattr(arr, "size", 0) == 0:
            return "unknown"
        mid = float(sorted(arr.reshape(-1))[len(arr.reshape(-1)) // 2])
        if mid < 30:
            return "close"
        if mid < 80:
            return "mid"
        return "open"

    height, width = depth.shape[:2]
    col = width // 3
    eye = depth[int(height * 0.38): int(height * 0.48)]
    return {
        "left": band(eye[:, :col]),
        "ahead": band(eye[:, col: 2 * col]),
        "right": band(eye[:, 2 * col :]),
    }


def _make_game(map_name="map01", skill=3, scenario=None):
    import vizdoom as vzd
    from pathlib import Path
    root = Path(vzd.__file__).resolve().parent
    game = vzd.DoomGame()
    wad = root / "freedoom2.wad"
    if wad.exists():
        game.set_doom_game_path(str(wad))
    if scenario:
        extra = root / "scenarios" / f"{scenario}.wad"
        if extra.exists():
            game.set_doom_scenario_path(str(extra))
    game.set_window_visible(False)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X400)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_depth_buffer_enabled(True)
    game.set_labels_buffer_enabled(True)
    game.set_objects_info_enabled(True)
    game.set_sectors_info_enabled(True)
    game.set_render_hud(True)
    game.set_render_weapon(True)
    game.set_render_crosshair(True)
    game.set_mode(vzd.Mode.PLAYER)
    if not scenario:
        game.set_doom_map(map_name)
    game.set_doom_skill(skill)
    for button in (vzd.Button.ATTACK, vzd.Button.MOVE_FORWARD, vzd.Button.MOVE_BACKWARD,
                   vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT, vzd.Button.TURN_LEFT,
                   vzd.Button.TURN_RIGHT, vzd.Button.USE):
        game.add_available_button(button)
    for var in (
        vzd.GameVariable.HEALTH, vzd.GameVariable.ARMOR,
        vzd.GameVariable.SELECTED_WEAPON, vzd.GameVariable.SELECTED_WEAPON_AMMO,
        vzd.GameVariable.KILLCOUNT, vzd.GameVariable.ANGLE,
        vzd.GameVariable.POSITION_X, vzd.GameVariable.POSITION_Y,
        vzd.GameVariable.VELOCITY_X, vzd.GameVariable.VELOCITY_Y,
        vzd.GameVariable.WEAPON1, vzd.GameVariable.WEAPON2, vzd.GameVariable.WEAPON3,
        vzd.GameVariable.WEAPON4, vzd.GameVariable.WEAPON5, vzd.GameVariable.WEAPON6,
        vzd.GameVariable.WEAPON7, vzd.GameVariable.AMMO2, vzd.GameVariable.AMMO3,
        vzd.GameVariable.AMMO4, vzd.GameVariable.AMMO5,
    ):
        game.add_available_game_variable(var)
    game.set_episode_timeout(0)
    game.init()
    return game


def _fresh_world():
    return {"rooms": None, "visited": set(), "spawned": 0}


@dataclass
class HudFrame:
    jpeg: bytes = b""
    state_text: str = ""
    standing_orders: str = ""
    schema: str = "json"
    kills: int = 0
    health: int = 0
    ammo: int = 0
    latency_ms: float = 0
    answers: dict = field(default_factory=dict)
    buttons: dict = field(default_factory=dict)
    objective: str = ""
    alive: bool = True
    director_spawned: int = 0
    director_killed: int = 0
    minimap: dict = field(default_factory=dict)


class DoomLoop:
    def __init__(self, client, standing_orders="Reach the exit alive; kill enemies in the way and collect supplies.",
                 schema="json", map_name="map01", decide_hz=10, scenario=None):
        self.client = client
        self.standing_orders = standing_orders
        self.schema = schema
        self.override = None
        self.raw_state = None
        self.map_name = map_name
        self.scenario = scenario
        self.decide_period = 1.0 / decide_hz
        self._lock = threading.Lock()
        self._game = None
        self._buttons = {name: False for name in BUTTON_ORDER}
        self._hud = HudFrame(standing_orders=standing_orders, schema=schema)
        self._stop = threading.Event()
        self._thread = None
        self.started_at = None
        self._roster = Roster()
        self._world = _fresh_world()
        self._prior = {}

    def start(self):
        self._game = _make_game(map_name=self.map_name, scenario=self.scenario)
        self._game.new_episode()
        self.started_at = time.perf_counter()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        if self._game is not None:
            self._game.close()
            self._game = None

    def snapshot(self):
        with self._lock:
            return self._hud

    def set_orders(self, text):
        with self._lock:
            self.standing_orders = text.strip() or self.standing_orders
            self._hud.standing_orders = self.standing_orders

    def set_schema(self, schema):
        if schema not in {"json", "yaml"}:
            raise ValueError("schema must be json or yaml")
        with self._lock:
            self.schema = schema
            self._hud.schema = schema

    def set_override(self, override):
        with self._lock:
            if override is None:
                self.override = None
                self.raw_state = None
            elif isinstance(override, str):
                self.raw_state = override
                self.override = None
            elif isinstance(override, dict) and "_yaml" in override:
                self.raw_state = override["_yaml"]
                self.override = None
            else:
                self.override = override
                self.raw_state = None

    def _run(self):
        import numpy as np
        from PIL import Image
        last_decide = 0.0
        while not self._stop.is_set():
            tick_start = time.perf_counter()
            if self._game.is_episode_finished():
                self._roster.reset()
                self._world = _fresh_world()
                self._prior = {}
                self._game.new_episode()
            state = self._game.get_state()
            if state is None:
                time.sleep(0.01)
                continue
            obs, minimap = observation_from_vizdoom(self._game, state, self._roster, self._world)
            with self._lock:
                orders = self.standing_orders
                schema = self.schema
                override = self.override
                raw_state = self.raw_state
                prior = dict(self._prior)
            state_text = raw_state if raw_state is not None else serialize_state(
                obs, orders, schema=schema, override=override)
            now = time.perf_counter()
            if now - last_decide >= self.decide_period:
                try:
                    decision = self.client.decide(
                        state_text, questions_for_observation(obs, prior=prior))
                    buttons = buttons_from_decision(obs, decision.answers)
                    with self._lock:
                        self._buttons = buttons
                        self._hud.answers = decision.answers
                        self._hud.latency_ms = decision.record.latency_ms
                        self._hud.buttons = dict(buttons)
                        self._prior = {
                            "goal": (decision.answers.get("goal") or {}).get("choice"),
                            "subject": (decision.answers.get("subject") or {}).get("choice"),
                        }
                    last_decide = now
                except Exception as exc:
                    with self._lock:
                        self._hud.answers = {"error": {"type": "noul", "noul": 0, "choice": str(exc)}}
            action = [int(self._buttons.get(name, False)) for name in BUTTON_ORDER]
            self._game.make_action(action, 1)
            jpeg = b""
            if state.screen_buffer is not None:
                image = Image.fromarray(np.asarray(state.screen_buffer))
                from io import BytesIO
                buf = BytesIO()
                image.save(buf, format="JPEG", quality=70)
                jpeg = buf.getvalue()
            caption = GOAL_PHRASE.get(self._prior.get("goal"), "")
            if self._prior.get("subject") and self._prior.get("subject") not in {"none", "hold"}:
                caption = f"{caption}, focusing on {self._prior['subject']}".strip(", ")
            with self._lock:
                self._hud = HudFrame(
                    jpeg=jpeg,
                    state_text=state_text,
                    standing_orders=orders,
                    schema=schema,
                    kills=obs.kills,
                    health=obs.health,
                    ammo=obs.ammo,
                    latency_ms=self._hud.latency_ms,
                    answers=self._hud.answers,
                    buttons=dict(self._buttons),
                    objective=caption.upper(),
                    alive=obs.health > 0,
                    director_spawned=obs.director_spawned,
                    director_killed=obs.director_killed,
                    minimap=minimap,
                )
            elapsed = time.perf_counter() - tick_start
            time.sleep(max(0.0, (1 / 35) - elapsed))
