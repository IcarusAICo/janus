"""The browser cell's rows are the loop's own request bodies plus one-hot targets on real observed nodes.

Runs in the browser demo's environment, which has the vendored loop's dependencies (skipped elsewhere):
    demos/browser/.venv/bin/python -m pytest tests/test_browser_cell.py
"""

import json

import pytest

pytest.importorskip("browser_harness")
pytest.importorskip("h2")

from janus.schema import Request  # noqa: E402
from janus.synth import browser  # noqa: E402


def page():
    return {"url": "https://example.test/", "title": "Search", "text": "Search", "scroll": {"y": 0}, "actions": [
        {"id": "e1", "kind": "fill", "label": "Where from?", "role": "combobox", "value": "", "node": 10},
        {"id": "e2", "kind": "click", "label": "Open Where from?", "role": "combobox", "value": "", "node": 10},
        {"id": "e3", "kind": "click", "label": "Search", "role": "button", "value": "", "node": 20},
        {"id": "wait", "kind": "wait", "label": "Wait for the page to update"}]}


def test_rows_follow_the_loop_request_shape_and_point_at_real_nodes():
    goal = "Find flights from Zurich."
    typed = browser.make_row(page(), goal, [], "TYPE_TEXT", browser.find(page(), {"op": "TYPE_TEXT", "label": "Where from?"}), "browser:t:1", "t")
    assert set(typed["questions"]) == {"operation", "type_text_target", "click_target"}
    ops = list(typed["questions"]["operation"]["criteria"])
    assert typed["questions"]["operation"]["target"][ops.index("TYPE_TEXT")] == 1.
    assert typed["questions"]["type_text_target"]["target"] == [1.]              # the only editable field
    assert typed["questions"]["click_target"]["target"] == [1., 0.]             # "Open Where from?" on the same node
    assert typed["state"]["elements"][0]["label"] == "Where from?" and typed["family"] == "browser"
    Request.from_dict(json.loads(json.dumps(typed)))                            # a valid janus training request
    clicked = browser.make_row(page(), goal, [], "CLICK", browser.find(page(), {"label": "Search", "role": "button"}), "browser:t:2", "t")
    assert set(clicked["questions"]) == {"operation", "click_target"} and clicked["questions"]["click_target"]["target"] == [0., 1.]
    done = browser.make_row(page(), goal, [], "DONE", None, "browser:t:3", "t")
    assert set(done["questions"]) == {"operation"} and done["questions"]["operation"]["target"][ops.index("DONE")] == 1.
    assert browser.make_row(page(), goal, [], "SCROLL_DOWN", None, "browser:t:4", "t") is None  # not offered on this page
