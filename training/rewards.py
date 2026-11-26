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
    "kill": 2.0,  # 1 -> 2 at 25M steps: user review, agent ignored monsters and only ran for the exit
    # Added at 3.9M steps: dense aiming signal (kills alone are sparse). Mirrors damage_per_hp and is
    # bounded by each enemy's health. Engine DAMAGECOUNT, training only.
    "damage_dealt_per_hp": 0.02,  # 0.01 -> 0.02 at 25M steps (same reason)
    # Added at 5.0M steps: measured that the policy fired on ~40-50% of steps whether or not a monster
    # was visible (shooting was free). A small cost per ammo unit spent makes misses cost and hits
    # (damage_dealt) pay, so it should learn to fire only at monsters. One pistol bullet = -0.01.
    "ammo_spent": -0.02,  # raised from -0.01 at 13.7M steps (inspection: still firing ~55% regardless of monsters)
    # Added at 5.3M steps: inspection showed USE pressed ~50% of steps with or without a door ahead;
    # spamming USE on a moving door reverses it (CONTRACT §3). Tiny cost per press (rising edge).
    "use_press": -0.003,
    # Added at ~8M steps (user review of gameplay videos: "shoots empty air, no target").
    # Potential-based aim shaping (Ng et al. 1999): phi = aim_potential * max cos(angle to a visible
    # monster); reward gamma*phi(s') - phi(s). Turning a monster towards the crosshair pays, letting it
    # drift away costs the same, so the optimal policy is unchanged; it only makes aiming learnable.
    # Computed from the observation's entity list, in training only; the policy is still pure weights.
    "aim_potential": 0.3,
    "fire_on_target": 0.08,  # ATTACK while a monster is within ~6 deg of the crosshair (0.02 -> 0.08 at 13.7M)
    # Exit efficiency, mirroring the evaluation's "small efficiency bonus for remaining health and time".
    "exit_time_bonus": 3.0,  # x fraction of the episode time left
    "exit_health_bonus": 2.0,  # x health/100 at the exit
    "new_cell": 0.05,  # first visit of a 64x64 grid cell (true position, training only)
    "pickup_when_low": 0.1,  # health gained while health < 50, or ammo gained while selected ammo < 10
    "damage_per_hp": -0.01,
    "death": -3.0,
    "step": -0.001,
    "stuck": -1.0,  # no progress for STUCK_TICKS -> penalty and episode cut off
}
ENEMY_TYPE_IDS = (1, 2, 3)  # doom_env.spaces.ENTITY_TYPES: enemy_melee, enemy_hitscan, enemy_projectile
ON_TARGET_COS = 0.995  # ~5.7 degrees
GAMMA = 0.99  # must match PPO gamma for the shaping to be potential-based


def aim_alignment(ents: dict | None) -> float:
    """max cos(relative angle) over visible monsters, clipped at 0; 0 when none is visible."""
    if ents is None:
        return 0.0
    m = (np.asarray(ents["mask"]) > 0) & np.isin(np.asarray(ents["entity_type"]), ENEMY_TYPE_IDS)
    if not m.any():
        return 0.0
    return float(max(0.0, np.asarray(ents["features"])[m, 2].max()))


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

    def reset(self, priv: dict, door_sectors: dict[int, int], ents: dict | None = None) -> None:
        """door_sectors: env.mapinfo.door_sectors {sector index: lock (0 = ordinary)}.
        ents: the first observation's visible_entities (for aim shaping)."""
        self._align = aim_alignment(ents)
        self.door_sectors = dict(door_sectors)
        self.opened: set[int] = set()
        self.visited = {self._cell(priv)}
        self.prev = priv
        self.last_progress_tic = priv["tic"]
        self.stats = EpisodeStats(cells=1)
        self._prev_ammo = None
        self._prev_use = 0
        self._prev_fire = 0

    def _cell(self, p) -> tuple[int, int]:
        return int(np.floor(p["x"] / CELL_SIZE)), int(np.floor(p["y"] / CELL_SIZE))

    def _add(self, name: str, value: float) -> float:
        self.stats.parts[name] = self.stats.parts.get(name, 0.0) + value
        return value

    def step(self, priv: dict, vitals: np.ndarray, action=None, ents: dict | None = None) -> tuple[float, bool]:
        """priv: privileged_state() after the step. vitals: the observation's vitals.
        action: the 9 buttons pressed (optional; used for the USE-press and fire-on-target terms).
        ents: the new observation's visible_entities (optional; aim shaping).
        Returns (shaped reward, stuck_cutoff)."""
        w, prev, s = self.w, self.prev, self.stats
        r = self._add("step", w["step"])
        if action is not None:
            use = int(action[7])
            if use and not self._prev_use:  # rising edge = an actual press
                r += self._add("use_press", w["use_press"])
            self._prev_use = use
            # The action was chosen on the previous observation, whose alignment is self._align.
            # Fresh trigger press only (organiser note: holding pistol fire spreads shots, only the first
            # shot of a burst is accurate). Rewarding taps teaches tap-firing on target, not spraying.
            fire = int(action[6])
            if fire and not self._prev_fire and self._align > ON_TARGET_COS:
                r += self._add("fire_on_target", w["fire_on_target"])
            self._prev_fire = fire
        if ents is not None:
            align = 0.0 if priv["finished"] else aim_alignment(ents)
            r += self._add("aim", w["aim_potential"] * (GAMMA * align - self._align))
            self._align = align
        progress = False

        if priv["exit_reached"]:
            r += self._add("exit", w["exit"])
            ticks = max(int(priv.get("episode_ticks", 0)), 1)
            if priv.get("episode_ticks"):
                r += self._add("exit_eff", w["exit_time_bonus"] * max(0.0, 1.0 - priv["tic"] / ticks))
            r += self._add("exit_eff", w["exit_health_bonus"] * max(0.0, min(priv["health"], 100.0)) / 100.0)
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
        if self._prev_ammo is not None and ammo < self._prev_ammo:  # ammo pools only shrink by firing
            r += self._add("ammo_spent", w["ammo_spent"] * (self._prev_ammo - ammo))
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
