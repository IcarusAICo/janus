"""Gridworld with JSON state for environment feedback (spec WP4b).

A square grid with walls, hazards and one goal. Moving into a wall or off the grid leaves the agent in
place; entering a hazard ends the episode with return 0; entering the goal ends it with return 1. The
ground truth for "will this action reach the goal within N steps without a hazard" is breadth-first
search over passable, hazard-free cells, so every structured state has unlimited exact T0 outcomes."""

from collections import deque
from dataclasses import dataclass, replace
import hashlib
import json

from ..schema import Request

ACTIONS = ("up", "down", "left", "right", "wait")
MOVES = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1), "wait": (0, 0)}
ACTION_TEXT = {"up": "Move one cell up (row minus one)", "down": "Move one cell down (row plus one)",
               "left": "Move one cell left (column minus one)", "right": "Move one cell right (column plus one)",
               "wait": "Stay in place for one step"}
ACTION_RULES = ("Rows increase downward and columns increase rightward. Moving into a wall or off the grid leaves "
                "the agent in place and still costs a step. Entering a hazard ends the episode in failure.")


@dataclass(frozen=True)
class Grid:
    size: int
    walls: frozenset
    hazards: frozenset
    goal: tuple
    agent: tuple

    def passable(self, cell):
        r, c = cell
        return 0 <= r < self.size and 0 <= c < self.size and cell not in self.walls

    def move(self, action):
        dr, dc = MOVES[action]
        cell = (self.agent[0] + dr, self.agent[1] + dc)
        return replace(self, agent=cell) if self.passable(cell) else self

    @property
    def done(self):
        return self.agent == self.goal or self.agent in self.hazards

    def to_state(self):
        return {"size": self.size, "agent": list(self.agent), "goal": list(self.goal),
                "walls": sorted(list(c) for c in self.walls), "hazards": sorted(list(c) for c in self.hazards)}

    @classmethod
    def from_state(cls, state):
        cell = lambda v: (int(v[0]), int(v[1]))
        return cls(int(state["size"]), frozenset(cell(c) for c in state["walls"]),
                   frozenset(cell(c) for c in state["hazards"]), cell(state["goal"]), cell(state["agent"]))

    def to_json(self):
        return json.dumps(self.to_state(), sort_keys=True)

    @classmethod
    def from_json(cls, text):
        return cls.from_state(json.loads(text))


def bfs_distances(grid, start):
    """Fewest safe steps from `start` to every reachable cell; hazards are never entered, the goal is terminal."""
    if start in grid.hazards or not grid.passable(start):
        return {}
    distances = {start: 0}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        if cell == grid.goal:
            continue
        for action in ACTIONS[:4]:
            dr, dc = MOVES[action]
            nxt = (cell[0] + dr, cell[1] + dc)
            if grid.passable(nxt) and nxt not in grid.hazards and nxt not in distances:
                distances[nxt] = distances[cell] + 1
                queue.append(nxt)
    return distances


def distance(grid, start=None):
    return bfs_distances(grid, grid.agent if start is None else start).get(grid.goal)


def success_within(grid, action, steps):
    """BFS truth: taking `action` now, can the goal be entered within `steps` steps in total without a hazard?"""
    if steps < 1 or grid.done:
        return False
    after = grid.move(action)
    if after.agent in grid.hazards:
        return False
    d = distance(after)
    return d is not None and d <= steps - 1


def brute_force_success(grid, action, steps):
    """Exhaustive enumeration of every continuation: the reference that `success_within` is tested against."""
    if steps < 1 or grid.done:
        return False
    after = grid.move(action)
    if after.agent in grid.hazards:
        return False
    if after.agent == grid.goal:
        return True
    return any(brute_force_success(after, nxt, steps - 1) for nxt in ACTIONS)


def optimal_actions(grid):
    """Actions that move one safe step closer to the goal, in canonical order; empty when none exists."""
    d = distance(grid)
    if d is None or grid.done:
        return ()
    return tuple(a for a in ACTIONS[:4]
                 if grid.move(a).agent != grid.agent and grid.move(a).agent not in grid.hazards
                 and distance(grid.move(a)) == d - 1)


def random_grid(rng, size=10, walls=12, hazards=6, max_distance=8):
    """A solvable grid whose start-to-goal distance lies in [1, max_distance]."""
    while True:
        cells = rng.sample([(r, c) for r in range(size) for c in range(size)], walls + hazards + 2)
        grid = Grid(size, frozenset(cells[:walls]), frozenset(cells[walls:walls + hazards]), cells[-2], cells[-1])
        d = distance(grid)
        if d is not None and 1 <= d <= max_distance:
            return grid


def budgeted_grid(rng, slack=(0, 1, 2), **kwargs):
    """A grid and a step budget equal to its distance plus a small slack."""
    grid = random_grid(rng, **kwargs)
    return grid, distance(grid) + rng.choice(slack)


def episode_state(grid, remaining):
    return {**grid.to_state(), "steps_remaining": remaining}


def episode_request(grid, remaining):
    """One Choice over actions and one Noul per action; targets are T0 truths from breadth-first search."""
    state = episode_state(grid, remaining)
    optimal = optimal_actions(grid)
    action_target = ([1 / len(optimal) if a in optimal else 0. for a in ACTIONS] if optimal
                     else [1 / len(ACTIONS)] * len(ACTIONS))
    questions = {"env:action": {"type": "choice", "instructions": "Which action moves the agent one step closer to the goal "
                                "along a shortest path that never enters a hazard? " + ACTION_RULES,
                                "criteria": {a: ACTION_TEXT[a] for a in ACTIONS}, "target": action_target}}
    for a in ACTIONS:
        ok = success_within(grid, a, remaining)
        questions[f"env:success:{a}"] = {
            "type": "noul", "target": [float(not ok), float(ok)],
            "instructions": f"If the agent's next action is '{a}', can it still enter the goal within {remaining} steps "
                            f"in total (counting that action) without ever entering a hazard? " + ACTION_RULES}
    digest = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()[:20]
    return Request.from_dict({"state": state, "group_id": f"env:{digest}", "questions": questions})


def run_episode(grid, remaining, policy):
    """`policy(grid, remaining) -> action`. The return is 1.0 when the goal is entered within the budget, else 0.0."""
    trajectory = []
    while remaining > 0 and not grid.done:
        action = policy(grid, remaining)
        trajectory.append((grid, remaining, action))
        grid = grid.move(action)
        remaining -= 1
    success = grid.agent == grid.goal
    return {"return": float(success), "success": success, "steps": len(trajectory), "trajectory": trajectory, "final": grid}
