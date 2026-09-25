"""Stable letter names for ViZDoom object ids."""
from __future__ import annotations


class Roster:
    def __init__(self):
        self._names = {}
        self._counts = {}

    def reset(self):
        self._names.clear()
        self._counts.clear()

    def _name(self, raw):
        oid = raw["object_id"]
        if oid in self._names:
            return self._names[oid]
        kind = raw["kind"]
        index = self._counts.get(kind, 0)
        self._counts[kind] = index + 1
        label = f"{kind} {chr(ord('A') + index)}"
        self._names[oid] = label
        return label

    def update(self, visible_raw, known_raw):
        visible_ids = set()
        visible = []
        for raw in visible_raw:
            label = self._name(raw)
            visible_ids.add(raw["object_id"])
            visible.append({**raw, "id": label, "visible": True,
                            "status": "in view", "last_seen": "just now"})
        remembered = []
        for raw in known_raw:
            oid = raw["object_id"]
            if oid in visible_ids:
                continue
            if oid not in self._names:
                self._name(raw)
            remembered.append({**raw, "id": self._names[oid], "visible": False,
                               "status": "out of view", "last_seen": "just now"})
        return {"visible": visible, "remembered": remembered}
