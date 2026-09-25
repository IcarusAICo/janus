from janus.schema import Request
from janus.server import precapture_bodies
from scripts.precapture_shapes import select


def test_select_covers_requests_with_fewest_keys():
    # 6 requests: three share {a, b}; {a, c} and {a, d} once each; {e, f, g} once.
    sets = [{"a", "b"}, {"a", "b"}, {"a", "b"}, {"a", "c"}, {"a", "d"}, {"e", "f", "g"}]
    assert select(sets, 0.5) == [0]  # {a, b} alone completes half the requests
    picked = select(sets, 0.8)
    assert picked[0] == 0 and len(picked) == 3  # then one of the one-key extensions each, not {e, f, g}
    assert 5 not in picked
    assert set(select(sets, 1.0)) >= {0, 3, 4, 5}


def test_shipped_shapes_are_valid_requests():
    bodies = precapture_bodies()
    assert bodies
    for body in bodies:
        Request.from_dict(body)
