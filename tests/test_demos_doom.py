"""Offline Doom policy tests. No ViZDoom, no network."""
from types import SimpleNamespace
import json

from demos.doom.layout import destinations_for, rooms_from_sectors
from demos.doom.policy import QUESTIONS, buttons_from_decision, questions_for_observation
from demos.doom.state import Observation, actor, serialize_state, subject_criteria
from demos.doom.track import Roster


def sample_obs(**overrides):
    data = {
        "health": 40,
        "armor": 0,
        "ammo": 12,
        "kills": 2,
        "facing_deg": 10,
        "weapon": "shotgun",
        "carrying": ["fist", "pistol (50 bullets)", "shotgun (12 shells, in hand)"],
        "velocity_speed": 80,
        "velocity_dir": -4.0,
        "walls": {"ahead": "open", "left": "close", "right": "open"},
        "visible": [actor("cacodemon A", "cacodemon", 180, -12, role="enemy")],
        "remembered": [actor("medikit B", "medikit", 400, 40, role="health", visible=False)],
        "destinations": [{"id": "room-3-edge", "text": "The unexplored edge on the north side of Room 3",
                          "x": 200.0, "y": 800.0, "bearing_deg": 20.0, "distance": 500}],
        "current_room": "Room 3",
        "director_spawned": 11,
        "director_killed": 2,
    }
    data.update(overrides)
    return Observation(**data)


def test_serialize_state_is_situation_report_not_flat_vitals():
    payload = json.loads(serialize_state(sample_obs(), "hunt hostiles", schema="json"))
    assert payload["standing_orders"] == "hunt hostiles"
    assert payload["player"]["health"] == "40/100"
    assert payload["player"]["weapon"] == "shotgun"
    assert payload["visible"][0]["labeled"] == "cacodemon A"
    assert payload["visible"][0]["distance"].endswith("(close)")
    assert "ahead left" in payload["visible"][0]["bearing"] or "ahead right" in payload["visible"][0]["bearing"]
    assert payload["remembered"][0]["status"] == "out of view"
    assert payload["measurement_context"]["distance_bands"]["contact"].startswith("under 64")
    assert "shotgun" in payload["weapons"]
    assert payload["director"] == {"spawned": 11, "killed": 2}
    assert payload["map"]["current_room"] == "Room 3"
    assert payload["map"]["destinations"][0]["id"] == "room-3-edge"


def test_serialize_state_yaml_schema_is_plain_text_not_pixels():
    text = serialize_state(sample_obs(), standing_orders="find medikits", schema="yaml")
    assert "standing_orders: find medikits" in text
    assert "cacodemon A" in text
    assert "\x00" not in text


def test_state_override_replaces_serialized_observation():
    text = serialize_state(sample_obs(), standing_orders="hold", schema="json",
                           override={"player": {"health": "1/100"}})
    payload = json.loads(text)
    assert payload["player"]["health"] == "1/100"
    assert payload["standing_orders"] == "hold"


def test_subject_criteria_include_visible_remembered_destinations_and_none():
    criteria = subject_criteria(sample_obs())
    assert criteria["none"]
    assert "cacodemon A" in criteria
    assert "medikit B" in criteria
    assert "room-3-edge" in criteria
    assert len(criteria) <= 255


def test_questions_are_semantic_not_wasd():
    questions = questions_for_observation(sample_obs(), prior={"goal": "kill_enemy", "subject": "cacodemon A"})
    assert set(questions) == set(QUESTIONS)
    assert questions["firing"]["type"] == "choice"
    assert "fire" in questions["firing"]["criteria"] and "hold_fire" in questions["firing"]["criteria"]
    assert "close_in" in questions["movement"]["criteria"]
    assert "hold_range" in questions["movement"]["criteria"]
    assert "carry_on" in questions["dodge"]["criteria"]
    assert "advance" in questions["goal"]["criteria"]
    assert "cacodemon A" in questions["turn"]["criteria"]
    assert "focusing on cacodemon A" in questions["movement"]["instructions"]


def test_close_in_on_enemy_turns_and_walks_and_can_fire_in_cone():
    decision = {
        "goal": {"choice": "kill_enemy"},
        "subject": {"choice": "cacodemon A"},
        "movement": {"choice": "close_in"},
        "dodge": {"choice": "carry_on"},
        "turn": {"choice": "cacodemon A"},
        "firing": {"choice": "hold_fire"},
    }
    buttons = buttons_from_decision(sample_obs(), decision)
    assert buttons["MOVE_FORWARD"] is True
    assert buttons["TURN_RIGHT"] is True
    assert buttons["ATTACK"] is True


def test_hold_range_backs_off_in_contact():
    obs = sample_obs(visible=[actor("imp A", "imp", 40, 0, role="enemy")])
    decision = {
        "goal": {"choice": "kill_enemy"},
        "subject": {"choice": "imp A"},
        "movement": {"choice": "hold_range"},
        "dodge": {"choice": "carry_on"},
        "turn": {"choice": "imp A"},
        "firing": {"choice": "fire"},
    }
    buttons = buttons_from_decision(obs, decision)
    assert buttons["MOVE_BACKWARD"] is True
    assert buttons["MOVE_FORWARD"] is False
    assert buttons["ATTACK"] is True


def test_walk_to_destination_turns_toward_landmark():
    decision = {
        "goal": {"choice": "advance"},
        "subject": {"choice": "room-3-edge"},
        "movement": {"choice": "walk_to"},
        "dodge": {"choice": "carry_on"},
        "turn": {"choice": "room-3-edge"},
        "firing": {"choice": "hold_fire"},
    }
    buttons = buttons_from_decision(sample_obs(visible=[]), decision)
    assert buttons["MOVE_FORWARD"] is True
    assert buttons["TURN_LEFT"] is True
    assert buttons["ATTACK"] is False


def test_roster_keeps_stable_letters_and_remembers_unseen():
    roster = Roster()
    vis = [{"object_id": 9, "kind": "cacodemon", "role": "enemy", "x": 10.0, "y": 20.0, "distance": 100, "bearing_deg": 5.0}]
    known = vis + [{"object_id": 9, "kind": "cacodemon", "role": "enemy", "x": 12.0, "y": 22.0, "distance": 140, "bearing_deg": 30.0}]
    first = roster.update(vis, known)
    assert first["visible"][0]["id"] == "cacodemon A"
    second = roster.update([], known)
    assert second["visible"] == []
    assert second["remembered"][0]["id"] == "cacodemon A"
    assert second["remembered"][0]["status"] == "out of view"


def test_rooms_cluster_sectors_that_share_open_edges():
    def line(x1, y1, x2, y2, blocking):
        return SimpleNamespace(x1=x1, y1=y1, x2=x2, y2=y2, is_blocking=blocking)

    a = SimpleNamespace(lines=[
        line(0, 0, 100, 0, True), line(100, 0, 100, 100, False),
        line(100, 100, 0, 100, True), line(0, 100, 0, 0, True),
    ])
    b = SimpleNamespace(lines=[
        line(100, 0, 200, 0, True), line(200, 0, 200, 100, True),
        line(200, 100, 100, 100, True), line(100, 100, 100, 0, False),
    ])
    c = SimpleNamespace(lines=[
        line(400, 0, 500, 0, True), line(500, 0, 500, 100, True),
        line(500, 100, 400, 100, True), line(400, 100, 400, 0, True),
    ])
    rooms = rooms_from_sectors([a, b, c])
    assert len(rooms) == 2
    dests = destinations_for(rooms, visited={"Room 1"}, player=(50, 50), facing_deg=0)
    assert dests
    assert all("id" in d and "text" in d for d in dests)


def test_walls_from_depth_reads_eye_level_not_floor_or_weapon():
    import numpy as np
    from demos.doom.game import _walls_from_depth

    depth = np.full((400, 640), 10, dtype=np.uint8)
    depth[150:200, 213:427] = 120
    walls = _walls_from_depth(depth)
    assert walls["ahead"] == "open"
    assert walls["left"] == "close"
    assert walls["right"] == "close"
