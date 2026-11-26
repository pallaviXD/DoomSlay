"""Tests for training/mapgen.py (procedural training maps).

The solvability tests drive the player with a scripted controller that uses *privileged* state
(absolute position / angle) and the generator's own grid. This is test code that proves the maps can
be finished; it is never used by the agent.
Run: .\\.venv\\Scripts\\python.exe -m pytest tests/test_mapgen.py -q
"""

from __future__ import annotations

import hashlib
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from doom_env import make_env  # noqa: E402
from doom_env.sensors import RAY_LOCKED  # noqa: E402
from training import mapgen as M  # noqa: E402

NOOP = np.zeros(9, np.int8)


def act(**kw) -> np.ndarray:
    a = np.zeros(9, np.int8)
    for name, i in (("fwd", 0), ("left", 4), ("right", 5), ("use", 7)):
        if kw.get(name):
            a[i] = 1
    return a


# ------------------------------------------------------------------ scripted driver (test only)
class Driver:
    """Walks the player along a BFS cell path using privileged position. Presses USE (rising edge)
    when stuck, which opens doors in front of it."""

    def __init__(self, env, lay: M.Layout):
        self.env, self.lay = env, lay
        self.done = False
        self.info = {"exit_reached": False}
        self.door_rays_seen: list[int] = []  # ray-0 hit type whenever we got stuck (in front of a door)
        self._use_prev = False

    def cell(self, x, y):
        ox, oy = self.lay.offset
        return int((x - ox) // M.CELL), int((y - oy) // M.CELL)

    def open_cell(self, c, blocked):
        g = self.lay.grid
        return 0 <= c[0] < g.shape[0] and 0 <= c[1] < g.shape[1] and g[c] >= 0 and c not in blocked

    def path(self, src, dst, blocked):
        prev = {src: None}
        q = deque([src])
        while q:
            u = q.popleft()
            if u == dst:
                break
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                v = (u[0] + d[0], u[1] + d[1])
                if v not in prev and self.open_cell(v, blocked):
                    prev[v] = u
                    q.append(v)
        if dst not in prev:
            return None
        out = [dst]
        while prev[out[-1]] is not None:
            out.append(prev[out[-1]])
        return out[::-1]

    def clear(self, p, q, blocked, r=20.0):
        n = max(2, int(math.dist(p, q) / 8))
        for i in range(n + 1):
            x = p[0] + (q[0] - p[0]) * i / n
            y = p[1] + (q[1] - p[1]) * i / n
            for dx in (-r, r):
                for dy in (-r, r):
                    if not self.open_cell(self.cell(x + dx, y + dy), blocked):
                        return False
        return True

    def step(self, a):
        obs, r, term, trunc, info = self.env.step(a)
        self.info = info
        self.done = term or trunc
        self.obs = obs
        return obs

    def go(self, target, blocked=frozenset(), max_steps=900, tol=24.0):
        """Drive to cell `target`. Returns True when within `tol` units (or the episode ended)."""
        hist = deque(maxlen=4)
        for _ in range(max_steps):
            if self.done:
                return True
            st = self.env.privileged_state()
            pos = (st["x"], st["y"])
            here = self.cell(*pos)
            goal = self.lay.to_units(target)
            if math.dist(pos, goal) < tol:
                return True
            cells = self.path(here, target, blocked) if self.open_cell(here, blocked) else None
            wp = goal
            if cells:
                pts = [self.lay.to_units(c) for c in cells]
                wp = pts[min(1, len(pts) - 1)]
                for p in reversed(pts):  # string pulling: farthest waypoint with clearance
                    if self.clear(pos, p, blocked):
                        wp = p
                        break
            err = (math.degrees(math.atan2(wp[1] - pos[1], wp[0] - pos[0])) - st["angle"] + 180) % 360 - 180
            stuck = len(hist) == hist.maxlen and math.dist(hist[0], pos) < 3  # pushed forward, didn't move
            if stuck:
                self.door_rays_seen.append(int(self.obs["rays"]["hit_type"][0]) if hasattr(self, "obs") else -1)
                hist.clear()  # USE once (rising edge), wait for the door to rise, then walk again
                self.step(act(use=True))
                for _ in range(14):  # pressing USE on a moving door would reverse it
                    if not self.done:
                        self.step(NOOP)
                continue
            if abs(err) > 25:
                hist.clear()
                self.step(act(left=err > 0, right=err < 0))
            else:
                hist.append(pos)
                self.step(act(fwd=True, left=err > 7, right=err < -7))
        return False

    def face(self, target_units, steps=30):
        for _ in range(steps):
            st = self.env.privileged_state()
            err = (math.degrees(math.atan2(target_units[1] - st["y"], target_units[0] - st["x"])) - st["angle"] + 180) % 360 - 180
            if abs(err) < 8:
                return
            self.step(act(left=err > 0, right=err < 0))


def door_side_cell(lay: M.Layout, conn: M.Conn) -> tuple[int, int]:
    """Open cell in front of `conn`'s door on the side reachable from the start without crossing it."""
    blocked = set(conn.door_cells)
    d = Driver.__new__(Driver)
    d.lay = lay
    reach = set()
    q = deque([lay.start_cell])
    reach.add(lay.start_cell)
    while q:
        u = q.popleft()
        for dd in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            v = (u[0] + dd[0], u[1] + dd[1])
            if v not in reach and d.open_cell(v, blocked):
                reach.add(v)
                q.append(v)
    for c in conn.door_cells:
        for dd in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            v = (c[0] + dd[0], c[1] + dd[1])
            if v in reach:
                return v
    raise AssertionError("door not reachable")


def make(lay_seed, stage, tmp_path, enemies=True):
    path = tmp_path / f"s{stage}_{lay_seed}.wad"
    meta = M.generate(lay_seed, stage, path, enemies=enemies)
    lay = M.build_layout(lay_seed, stage, enemies=enemies)
    env = make_env(str(path), "MAP01", render_buffers=False, episode_ticks=meta["episode_ticks"])
    return meta, lay, env


# ------------------------------------------------------------------ tests
def test_deterministic_per_seed(tmp_path):
    for stage in M.STAGES:
        M.generate(123, stage, tmp_path / "a.wad")
        M.generate(123, stage, tmp_path / "b.wad")
        M.generate(124, stage, tmp_path / "c.wad")
        a, b, c = (hashlib.sha1((tmp_path / f).read_bytes()).hexdigest() for f in ("a.wad", "b.wad", "c.wad"))
        assert a == b and a != c


@pytest.mark.parametrize("stage", M.STAGES)
def test_each_stage_loads(stage, tmp_path):
    meta, lay, env = make(11, stage, tmp_path)
    obs, _ = env.reset()
    assert env.observation_space.contains(obs)
    names = [o.name for o in env._game.get_state().objects]
    assert len(env._game.get_state().sectors) == meta["num_sectors"] == env.mapinfo.num_sectors
    assert env.mapinfo.door_sectors == {s: lock for s, lock in meta["door_sectors"]}
    monsters = {"Zombieman", "ShotgunGuy", "DoomImp", "Demon", "Spectre"}
    assert sum(n in monsters for n in names) == meta["num_enemies"] >= 1
    assert names.count("DoomPlayer") == 1
    for _ in range(20):
        obs, *_ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
    env.close()


@pytest.mark.parametrize("seed", range(0, 40))
def test_stage3_one_locked_door_with_key(seed, tmp_path):
    lay = M.build_layout(seed, 3)
    text, _ = M.to_udmf(lay)
    from doom_env.mapinfo import _from_udmf
    info = _from_udmf(text)
    locked = {s: lk for s, lk in info.door_sectors.items() if lk}
    assert len(locked) == 1
    (sec, lock), = locked.items()
    key_type = {lk: t for t, lk in M.KEYS.values()}[lock]
    assert sum(t["type"] == key_type for t in lay.things) == 1
    assert sum(t["type"] in (5, 6, 13) for t in lay.things) == 1
    # key is reachable from the start without crossing the locked door, the exit is not
    conn = lay.conns[lay.locked_conn]
    d = Driver.__new__(Driver)
    d.lay = lay
    blocked = set(conn.door_cells)
    assert d.path(lay.start_cell, lay.key_cell, blocked) is not None
    assert d.path(lay.start_cell, lay.exit_cells[0], blocked) is None
    assert d.path(lay.start_cell, lay.exit_cells[0], set()) is not None


def test_heldout_seeds_disjoint_from_training():
    assert M.HELDOUT_SEED_BASE >= M.TRAIN_SEED_LIMIT
    assert not M.is_heldout_seed(M.TRAIN_SEED_LIMIT - 1) and M.is_heldout_seed(M.HELDOUT_SEED_BASE)
    with pytest.raises(ValueError):
        M.build_layout(M.HELDOUT_SEED_BASE + 3, 2)
    manifest = M.HELDOUT_DIR / "manifest.json"
    if manifest.exists():
        import json
        for m in json.loads(manifest.read_text()):
            assert M.is_heldout_seed(m["seed"]) and m["seed"] >= M.TRAIN_SEED_LIMIT


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_stage1_exit_reachable(seed, tmp_path):
    meta, lay, env = make(seed, 1, tmp_path, enemies=False)
    env.reset()
    d = Driver(env, lay)
    assert d.go(lay.exit_cells[0], max_steps=500)
    assert d.info["exit_reached"]
    env.close()


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_stage3_locked_door_key_exit(seed, tmp_path):
    """Locked door stays shut without the key, the key can be picked up, then the exit is reachable."""
    meta, lay, env = make(seed, 3, tmp_path, enemies=False)
    obs, _ = env.reset()
    conn = lay.conns[lay.locked_conn]
    door_sec = conn.door_sector
    d = Driver(env, lay)
    front = door_side_cell(lay, conn)
    assert d.go(front, blocked=set(conn.door_cells), max_steps=1500)
    door_c = np.mean([lay.to_units(c) for c in conn.door_cells], axis=0)
    d.face(door_c)
    d.step(NOOP)
    assert int(d.obs["rays"]["hit_type"][0]) == RAY_LOCKED
    for i in range(30):  # press USE repeatedly without the key
        d.step(act(use=i % 2 == 0))
    assert env.privileged_state()["sector_ceiling"][door_sec] == 0
    key_idx = {"blue": 0, "yellow": 1, "red": 2}[lay.key_color]
    assert d.obs["vitals"][8 + key_idx] == 0
    assert d.go(lay.key_cell, blocked=set(conn.door_cells), max_steps=1500, tol=16)
    for _ in range(3):
        d.step(NOOP)
    assert d.obs["vitals"][8 + key_idx] == 1
    assert d.go(lay.exit_cells[0], max_steps=2500)
    assert d.info["exit_reached"]
    env.close()


def _manifest():
    import json
    p = M.HELDOUT_DIR / "manifest.json"
    return json.loads(p.read_text()) if p.exists() else []


@pytest.mark.parametrize("meta", _manifest(), ids=lambda m: m["wad"])
def test_heldout_files_are_current(meta, tmp_path):
    """maps/heldout/*.wad must be exactly what the generator produces for their seeds."""
    out = tmp_path / "x.wad"
    M.generate(meta["seed"], meta["stage"], out, allow_heldout=True)
    assert out.read_bytes() == (M.HELDOUT_DIR / meta["wad"]).read_bytes()


@pytest.mark.parametrize("meta", _manifest(), ids=lambda m: m["wad"])
def test_heldout_solvable(meta, tmp_path):
    """Drive the enemy-free twin of each held-out map (same geometry, key and exit) to the exit."""
    lay = M.build_layout(meta["seed"], meta["stage"], enemies=False, allow_heldout=True)
    assert lay.key_cell == M.build_layout(meta["seed"], meta["stage"], allow_heldout=True).key_cell
    path = tmp_path / "h.wad"
    M.generate(meta["seed"], meta["stage"], path, enemies=False, allow_heldout=True)
    env = make_env(str(path), "MAP01", render_buffers=False, episode_ticks=meta["episode_ticks"])
    env.reset()
    d = Driver(env, lay)
    if lay.key_cell is not None:
        assert d.go(lay.key_cell, blocked=set(lay.conns[lay.locked_conn].door_cells), max_steps=1200, tol=16)
        d.step(NOOP)
    d.go(lay.exit_cells[0], max_steps=1500)
    assert d.info["exit_reached"]
    env.close()
