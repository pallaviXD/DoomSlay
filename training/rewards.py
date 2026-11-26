"""Training-only reward shaping. Uses env.privileged_state() (ground truth), which the agent never sees.

All weights live in REWARD_WEIGHTS so they can be tuned in one place and printed into logs.
The shaping mirrors the evaluation categories (EVALUATION.md): exit >> progression (keys, doors)
> exploration > combat, with survival penalties.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

REWARD_WEIGHTS = {
    "exit": 10.0,  # reaching the level exit (largest evaluation category)
    "keycard": 3.0,  # first pickup of each key
    "door_open": 2.0,  # a door sector opened for the first time (doubles as "unlock" for locked doors)
    # Raised 1 -> 4 at 3.9M steps: agents picked up keys but rarely went back to the locked door.
    "locked_door_bonus": 4.0,  # extra on top of door_open when the opened door was locked
    "kill": 1.0,
    # Added at 3.9M steps: dense aiming signal (kills alone are sparse). Mirrors damage_per_hp and is
    # bounded by each enemy's health. Engine DAMAGECOUNT, training only.
    "damage_dealt_per_hp": 0.01,
    "new_cell": 0.05,  # first visit of a 64x64 grid cell (true position, training only)
    "pickup_when_low": 0.1,  # health gained while health < 50, or ammo gained while selected ammo < 10
    "damage_per_hp": -0.01,
    "death": -3.0,
    "step": -0.001,
    "stuck": -1.0,  # no progress for STUCK_TICKS -> penalty and episode cut off
}
CELL_SIZE = 64.0
STUCK_TICKS = 200  # no new cell / kill / key / door within this many tics -> stuck
DOOR_OPEN_GAP = 48.0  # door sector counts as open once ceiling - floor exceeds this


@dataclass
class EpisodeStats:
    """Per-episode metrics, also used by eval.py."""

    reward: float = 0.0
    steps: int = 0
    exit: bool = False
    dead: bool = False
    stuck: bool = False
    kills: int = 0
    keys: int = 0
    doors: int = 0
    locked_doors: int = 0
    cells: int = 0
    damage: float = 0.0
    parts: dict = field(default_factory=dict)  # reward broken down by component

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "parts"}
        d.update({f"r_{k}": v for k, v in self.parts.items()})
        return d


class RewardShaper:
    def __init__(self, weights: dict | None = None, stuck_ticks: int = STUCK_TICKS):
        self.w = dict(REWARD_WEIGHTS, **(weights or {}))
        self.stuck_ticks = stuck_ticks

    def reset(self, priv: dict, door_sectors: dict[int, int]) -> None:
        """door_sectors: env.mapinfo.door_sectors {sector index: lock (0 = ordinary)}."""
        self.door_sectors = dict(door_sectors)
        self.opened: set[int] = set()
        self.visited = {self._cell(priv)}
        self.prev = priv
        self.last_progress_tic = priv["tic"]
        self.stats = EpisodeStats(cells=1)
        self._prev_ammo = None

    def _cell(self, p) -> tuple[int, int]:
        return int(np.floor(p["x"] / CELL_SIZE)), int(np.floor(p["y"] / CELL_SIZE))

    def _add(self, name: str, value: float) -> float:
        self.stats.parts[name] = self.stats.parts.get(name, 0.0) + value
        return value

    def step(self, priv: dict, vitals: np.ndarray) -> tuple[float, bool]:
        """priv: privileged_state() after the step. vitals: the observation's vitals.
        Returns (shaped reward, stuck_cutoff)."""
        w, prev, s = self.w, self.prev, self.stats
        r = self._add("step", w["step"])
        progress = False

        if priv["exit_reached"]:
            r += self._add("exit", w["exit"])
            s.exit = True
        if priv["dead"]:
            r += self._add("death", w["death"])
            s.dead = True

        dk = priv["kills"] - prev["kills"]
        if dk > 0:
            r += self._add("kill", w["kill"] * dk)
            s.kills += dk
            progress = True

        new_keys = float(np.sum(np.maximum(priv["keys"] - prev["keys"], 0)))
        if new_keys > 0:
            r += self._add("keycard", w["keycard"] * new_keys)
            s.keys += int(new_keys)
            progress = True

        # Doors: compare live sector heights (privileged) for door sectors only.
        if self.door_sectors and priv.get("sector_ceiling") is not None:
            ceil, floor = priv["sector_ceiling"], priv["sector_floor"]
            for sec, lock in self.door_sectors.items():
                if sec in self.opened or sec >= len(ceil):
                    continue
                if ceil[sec] - floor[sec] > DOOR_OPEN_GAP:
                    self.opened.add(sec)
                    r += self._add("door_open", w["door_open"])
                    s.doors += 1
                    if lock:
                        r += self._add("locked_door", w["locked_door_bonus"])
                        s.locked_doors += 1
                    progress = True

        cell = self._cell(priv)
        if cell not in self.visited:
            self.visited.add(cell)
            r += self._add("new_cell", w["new_cell"])
            s.cells += 1
            progress = True

        dealt = priv.get("damage_dealt", 0.0) - prev.get("damage_dealt", 0.0)
        if dealt > 0:
            r += self._add("damage_dealt", w["damage_dealt_per_hp"] * dealt)

        dmg = priv["damage_taken"] - prev["damage_taken"]
        if dmg > 0:
            r += self._add("damage", w["damage_per_hp"] * dmg)
            s.damage += dmg

        # Pickups only pay when the resource was actually low (avoids farming item rooms).
        if priv["health"] > prev["health"] and prev["health"] < 50:
            r += self._add("pickup", w["pickup_when_low"])
        # Ammo: compare the summed ammo pools (vitals 4-7); selected-weapon ammo jumps on weapon switch.
        ammo = float(np.sum(vitals[4:8]))
        if self._prev_ammo is not None and ammo > self._prev_ammo and vitals[3] < 10:
            r += self._add("pickup", w["pickup_when_low"])
        self._prev_ammo = ammo

        if progress:
            self.last_progress_tic = priv["tic"]
        stuck = (not priv["finished"]) and priv["tic"] - self.last_progress_tic >= self.stuck_ticks
        if stuck:
            r += self._add("stuck", w["stuck"])
            s.stuck = True

        self.prev = priv
        s.reward += r
        s.steps += 1
        return r, stuck

    _prev_ammo: float | None = None
