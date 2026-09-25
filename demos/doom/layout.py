"""Room clusters and walk-to landmarks from ViZDoom sectors."""
from __future__ import annotations

from collections import defaultdict
import math


def _key(line):
    a = (round(line.x1), round(line.y1))
    b = (round(line.x2), round(line.y2))
    return (a, b) if a <= b else (b, a)


def _bearing(dx, dy, facing_deg):
    angle = math.degrees(math.atan2(dy, dx)) - facing_deg
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return round(angle, 1)


def _compass(bearing):
    if abs(bearing) <= 22:
        return "north"
    if abs(bearing) >= 158:
        return "south"
    if bearing > 0:
        return "northwest" if bearing < 67 else "west" if bearing < 112 else "southwest"
    return "northeast" if bearing > -67 else "east" if bearing > -112 else "southeast"


def rooms_from_sectors(sectors):
    owners = defaultdict(list)
    for index, sector in enumerate(sectors):
        for line in sector.lines:
            owners[_key(line)].append((index, bool(line.is_blocking)))
    parent = list(range(len(sectors)))

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left, right):
        root_l, root_r = find(left), find(right)
        if root_l != root_r:
            parent[root_r] = root_l

    floors = [getattr(sector, "floor_height", 0) for sector in sectors]
    for occ in owners.values():
        ids = list({index for index, _ in occ})
        if len(ids) >= 2 and any(not blocking for _, blocking in occ):
            first = ids[0]
            for index in ids[1:]:
                if abs(floors[first] - floors[index]) <= 16:
                    union(first, index)

    groups = defaultdict(list)
    for index in range(len(sectors)):
        groups[find(index)].append(index)

    rooms = []
    for members in groups.values():
        points, lines = [], []
        for index in members:
            for line in sectors[index].lines:
                points.extend(((line.x1, line.y1), (line.x2, line.y2)))
                lines.append(line)
        if not points:
            continue
        rooms.append({
            "members": members,
            "cx": sum(x for x, _ in points) / len(points),
            "cy": sum(y for _, y in points) / len(points),
            "lines": lines,
        })
    rooms.sort(key=lambda room: (room["cy"], room["cx"]))
    for number, room in enumerate(rooms, 1):
        room["id"] = f"Room {number}"
    return rooms


def room_at(rooms, x, y):
    if not rooms:
        return None
    return min(rooms, key=lambda room: (room["cx"] - x) ** 2 + (room["cy"] - y) ** 2)


def destinations_for(rooms, visited, player, facing_deg):
    px, py = player
    dests = []
    seen = set()
    for room in rooms:
        dx, dy = room["cx"] - px, room["cy"] - py
        dist = int(round(math.hypot(dx, dy)))
        bearing = _bearing(dx, dy, facing_deg)
        if room["id"] not in visited:
            dest_id = room["id"].lower().replace(" ", "-")
            dests.append({
                "id": dest_id,
                "text": (f"{room['id']} lies to the {_compass(bearing)} — "
                         f"about {dist} map units away"),
                "x": room["cx"], "y": room["cy"],
                "bearing_deg": bearing, "distance": dist,
            })
            seen.add(dest_id)
            continue
        for line in room["lines"]:
            if not line.is_blocking:
                continue
            mx, my = (line.x1 + line.x2) / 2, (line.y1 + line.y2) / 2
            bearing = _bearing(mx - px, my - py, facing_deg)
            dest_id = f"{room['id'].lower().replace(' ', '-')}-edge"
            if dest_id in seen:
                break
            dests.append({
                "id": dest_id,
                "text": f"The unexplored edge on the {_compass(bearing)} side of {room['id']}",
                "x": mx, "y": my,
                "bearing_deg": bearing, "distance": int(round(math.hypot(mx - px, my - py))),
            })
            seen.add(dest_id)
            break
    return dests[:8]
