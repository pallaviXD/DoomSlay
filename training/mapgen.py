"""Procedural training maps for the curriculum (UDMF, namespace "zdoom").

Why a generator: evaluation uses hidden maps in four tiers (combat, exploration, key -> locked door
-> exit, gauntlet). A policy trained on a handful of fixed maps memorises them; thousands of random
layouts force it to learn behaviours (follow corridors, open doors, fetch keys, find the exit) from
the ego-centric observation instead.

Design (all choices aim at "always valid, always solvable" rather than at pretty maps):

* Grid based. The map is an occupancy grid of 64x64-unit cells; every cell is solid or belongs to
  one sector (a room, a corridor piece, a door or the exit pad). Linedefs are emitted only on cell
  edges whose two sides differ, so geometry is closed and consistent by construction (front side on
  the right of v1->v2, Doom convention). Collinear edges between the same two sectors are merged.
  ZDoom builds its own nodes for UDMF, so no node builder is needed.
* Rooms live in a coarse grid of 16x16-cell "slots" (one room per slot, at most one slot row/column
  of rooms per axis). Every room covers the 4x4 central band of its slot, so two rooms in
  neighbouring slots always overlap by >= 4 cells and can be joined by a straight 2-cell-wide
  corridor through the gap between them. Rooms in different slots never touch.
* The room graph is a random spanning tree over neighbouring slots (random growth), plus 0-2 extra
  edges that create loops. Tree leaves are natural dead ends.
* Doors are 1-cell-thick, 2-cell-wide (128 units) sectors inside a corridor, closed
  (ceiling == floor) at map start, `Door_Raise` (special 12) or `Door_LockedRaise` (13, arg3 = ZDoom
  lock number). The exit is a 2x2-cell pad whose boundary lines are walk-over `Exit_Normal` (243).
* Stage 3: the locked door sits on a *bridge* of the room graph on the start -> exit path (so the
  exit cannot be reached around it), and the matching keycard is in a room that is reachable from the
  start without crossing that door, preferably a side branch off the main path.

Seeds: training seeds are ints in [0, TRAIN_SEED_LIMIT). Seeds in
[HELDOUT_SEED_BASE, HELDOUT_SEED_BASE + HELDOUT_SEED_SPAN) are reserved for the held-out
evaluation set in maps/heldout/ and are refused unless `allow_heldout=True`.

CLI:
    python -m training.mapgen --stage 2 --seeds 0-9 --out training/generated_maps/s2
    python -m training.mapgen --heldout            # writes maps/heldout/*.wad + manifest.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from doom_env.wad import Lump, write_wad

CELL = 64  # map units per grid cell
SLOT = 16  # cells per room slot (1024 units)
BAND = (6, 10)  # every room covers slot-relative cells [6, 10) on both axes -> straight corridors
CORRIDOR_WIDTH = 2  # cells (128 units, the door width as well)

TRAIN_SEED_LIMIT = 1_000_000
HELDOUT_SEED_BASE = 9_000_000  # held-out maps use seeds HELDOUT_SEED_BASE + i: never train on them
HELDOUT_SEED_SPAN = 1_000_000
HELDOUT_DIR = Path(__file__).resolve().parent.parent / "maps" / "heldout"

STAGES = (1, 2, 3, 4)
EPISODE_TICKS = {1: 2100, 2: 3150, 3: 4200, 4: 4200}

MONSTERS = {"zombieman": 3004, "shotgunguy": 9, "imp": 3001, "demon": 3002, "spectre": 58}
ITEMS = {"stimpack": 2011, "medikit": 2012, "clip": 2007, "shells": 2008, "ammobox": 2048,
         "greenarmor": 2018, "shotgun": 2001}
# colour -> (keycard thing type, ZDoom lock number as used by Door_LockedRaise arg3)
KEYS = {"red": (13, 1), "blue": (5, 2), "yellow": (6, 3)}
LOCK_TEXTURES = {1: ("DOORRED2", "DOORRED"), 2: ("DOORBLU2", "DOORBLU"), 3: ("DOORYEL2", "DOORYEL")}

# All verified to exist in freedoom2.wad (TEXTURE1/2 and F_START..F_END).
WALL_TEX = ("STARTAN2", "STARTAN3", "STONE2", "STONE3", "BROWN1", "BROWN96", "GRAY1", "GRAY4", "GRAY5",
            "BRICK1", "BROWNGRN", "TEKGREN2", "METAL1", "ICKWALL1", "CEMENT1", "BIGBRIK1")
FLOOR_TEX = ("FLOOR0_1", "FLOOR4_8", "FLOOR5_1", "FLOOR7_1", "FLAT1", "FLAT14", "FLAT5_5", "FLOOR1_1",
             "MFLR8_1", "FLAT3")
CEIL_TEX = ("CEIL1_1", "CEIL3_5", "CEIL5_1", "CEIL3_3", "FLAT20", "TLITE6_4")
DOOR_TEX = ("BIGDOOR2", "BIGDOOR1", "BIGDOOR4")
EXIT_FLOOR = "GATE3"


# ---------------------------------------------------------------------------------- data model
@dataclass
class Sector:
    kind: str  # "room" | "corridor" | "door" | "exit"
    ceil: int
    light: int
    wall: str
    floor_tex: str
    ceil_tex: str
    floor: int = 0
    lock: int = 0  # doors only: 0 ordinary, 1 red, 2 blue, 3 yellow
    door_tex: str = ""
    track_tex: str = "DOORTRAK"


@dataclass
class Room:
    slot: tuple[int, int]
    x0: int
    y0: int
    w: int
    h: int
    sector: int = -1

    @property
    def x1(self) -> int:
        return self.x0 + self.w

    @property
    def y1(self) -> int:
        return self.y0 + self.h

    def cells(self, margin: int = 0) -> list[tuple[int, int]]:
        return [(x, y) for x in range(self.x0 + margin, self.x1 - margin) for y in range(self.y0 + margin, self.y1 - margin)]

    def contains(self, c: tuple[int, int]) -> bool:
        return self.x0 <= c[0] < self.x1 and self.y0 <= c[1] < self.y1


@dataclass
class Conn:
    a: int  # room index on the low side (left / below)
    b: int  # room index on the high side
    axis: str  # "x": rooms side by side, corridor runs along x; "y": stacked, corridor runs along y
    kind: str = "open"  # "open" | "door" | "locked"
    lock: int = 0
    cells: list = field(default_factory=list)
    door_cells: list = field(default_factory=list)
    chain: list = field(default_factory=list)  # sector ids from room a to room b (adjacency whitelist)
    door_sector: int = -1


@dataclass
class Layout:
    seed: int
    stage: int
    grid: np.ndarray  # (W, H) int32, -1 = solid, else sector index
    sectors: list[Sector]
    rooms: list[Room]
    conns: list[Conn]
    things: list[dict]
    start_room: int
    exit_room: int
    start_cell: tuple[int, int]
    start_angle: int
    exit_cells: list[tuple[int, int]]
    exit_sector: int
    key_color: str | None = None
    key_room: int | None = None
    key_cell: tuple[int, int] | None = None
    locked_conn: int | None = None
    offset: tuple[int, int] = (0, 0)  # map-unit translation of the whole grid

    def to_units(self, c: tuple[float, float]) -> tuple[float, float]:
        return ((c[0] + 0.5) * CELL + self.offset[0], (c[1] + 0.5) * CELL + self.offset[1])

    def room_graph(self) -> dict[int, list[int]]:
        g: dict[int, list[int]] = {i: [] for i in range(len(self.rooms))}
        for c in self.conns:
            g[c.a].append(c.b)
            g[c.b].append(c.a)
        return g


class _Retry(Exception):
    """Raised inside one attempt when a random draw violates a constraint; the caller re-draws."""


# ---------------------------------------------------------------------------------- graph utils
def _bfs(n: int, edges: list[tuple[int, int]], src: int, banned: int | None = None) -> dict[int, int]:
    adj: dict[int, list[tuple[int, int]]] = {i: [] for i in range(n)}
    for k, (a, b) in enumerate(edges):
        if k != banned:
            adj[a].append((b, k))
            adj[b].append((a, k))
    dist = {src: 0}
    q = deque([src])
    while q:
        u = q.popleft()
        for v, _ in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def _path_edges(n: int, edges: list[tuple[int, int]], src: int, dst: int) -> tuple[list[int], list[int]]:
    """Shortest path src -> dst: (room list, edge-index list)."""
    adj: dict[int, list[tuple[int, int]]] = {i: [] for i in range(n)}
    for k, (a, b) in enumerate(edges):
        adj[a].append((b, k))
        adj[b].append((a, k))
    prev: dict[int, tuple[int, int] | None] = {src: None}
    q = deque([src])
    while q:
        u = q.popleft()
        for v, k in adj[u]:
            if v not in prev:
                prev[v] = (u, k)
                q.append(v)
    rooms, path = [dst], []
    while prev[rooms[-1]] is not None:
        u, k = prev[rooms[-1]]
        path.append(k)
        rooms.append(u)
    return rooms[::-1], path[::-1]


# ---------------------------------------------------------------------------------- layout
def _place_room(rng: random.Random, slot: tuple[int, int], lo: int, hi: int) -> Room:
    def axis(size: int) -> int:
        # margin >= 1 to the slot border and must cover the central BAND
        return rng.randint(max(1, BAND[1] - size), min(BAND[0], SLOT - 1 - size))

    w, h = rng.randint(lo, hi), rng.randint(lo, hi)
    return Room(slot, slot[0] * SLOT + axis(w), slot[1] * SLOT + axis(h), w, h)


def _room_graph(rng: random.Random, stage: int) -> tuple[tuple[int, int], list[tuple[int, int]], list[tuple[int, int]], int]:
    """Pick slot grid, occupied slots and a spanning tree (+ loops). Returns (dims, slots, edges, n_tree)."""
    if stage == 1:
        n = 1 if rng.random() < 0.35 else 2
        dims = rng.choice([(2, 1), (1, 2)]) if n == 2 else (1, 1)
    else:
        dims = rng.choice([(3, 3), (4, 2), (2, 4)])
        n = rng.randint(4, 8)
    all_slots = [(x, y) for x in range(dims[0]) for y in range(dims[1])]
    slots = [rng.choice(all_slots)]
    edges: list[tuple[int, int]] = []

    def neighbours(s):
        for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            t = (s[0] + d[0], s[1] + d[1])
            if 0 <= t[0] < dims[0] and 0 <= t[1] < dims[1]:
                yield t

    # random growth = random spanning tree over the chosen slots
    while len(slots) < n:
        frontier = [(i, t) for i, s in enumerate(slots) for t in neighbours(s) if t not in slots]
        i, t = rng.choice(frontier)
        slots.append(t)
        edges.append((i, len(slots) - 1))
    n_tree = len(edges)
    if stage >= 2:
        extra = [(i, j) for i, s in enumerate(slots) for j, t in enumerate(slots)
                 if i < j and t in set(neighbours(s)) and (i, j) not in edges and (j, i) not in edges]
        rng.shuffle(extra)
        edges += extra[: rng.randint(0, 2)]
    return dims, slots, edges, n_tree


def _try_build(rng: random.Random, seed: int, stage: int, enemies: bool) -> Layout:
    dims, slots, edges, _ = _room_graph(rng, stage)
    n = len(slots)
    lo, hi = (6, 14)
    rooms = [_place_room(rng, s, lo, hi) for s in slots]

    # --- connections (normalised so a is left/below)
    conns: list[Conn] = []
    for i, j in edges:
        si, sj = slots[i], slots[j]
        if (sj[0], sj[1]) < (si[0], si[1]):
            i, j, si, sj = j, i, sj, si
        conns.append(Conn(i, j, "x" if si[1] == sj[1] else "y"))

    # Sometimes pull two rooms together so the "corridor" is a single cell (a doorway). Room a grows
    # to its slot border, room b keeps a 1-cell margin, so they still never share a wall.
    for c in conns:
        if rng.random() < 0.3:
            A, B = rooms[c.a], rooms[c.b]
            if c.axis == "x":
                border = B.slot[0] * SLOT
                A.w = border - A.x0
                B.w, B.x0 = B.x1 - (border + 1), border + 1
            else:
                border = B.slot[1] * SLOT
                A.h = border - A.y0
                B.h, B.y0 = B.y1 - (border + 1), border + 1

    # --- start / exit (exit = farthest room in graph distance)
    start_room = rng.randrange(n)
    dist = _bfs(n, [(c.a, c.b) for c in conns], start_room)
    if n > 1:
        far = max(dist.values())
        exit_room = rng.choice([r for r, d in dist.items() if d == far])
    else:
        exit_room = start_room

    # --- doors
    locked_conn = None
    key_room = None
    key_color = None
    if stage >= 2:
        for c in conns:
            if rng.random() < 0.4:
                c.kind = "door"
    if stage >= 3:
        cedges = [(c.a, c.b) for c in conns]
        path_rooms, path = _path_edges(n, cedges, start_room, exit_room)
        options = []
        for k in path:
            comp = _bfs(n, cedges, start_room, banned=k)
            if exit_room in comp:
                continue  # not a bridge: a loop would let the agent walk around the locked door
            before = path_rooms[: path.index(k) + 1]
            side = [r for r in comp if r not in path_rooms]
            on_path = [r for r in before if r != start_room]
            if side or on_path:
                options.append((k, side, on_path))
        if not options:
            raise _Retry("no bridge with a key room before it")
        k, side, on_path = rng.choice(options)
        locked_conn = k
        key_color = rng.choice(sorted(KEYS))
        conns[k].kind, conns[k].lock = "locked", KEYS[key_color][1]
        key_room = rng.choice(side) if side else rng.choice(on_path)

    # --- sectors
    sectors: list[Sector] = []
    W, H = dims[0] * SLOT, dims[1] * SLOT
    grid = np.full((W, H), -1, np.int32)
    corridor_wall = rng.choice(WALL_TEX)
    for r in rooms:
        r.sector = len(sectors)
        sectors.append(Sector("room", rng.choice([128, 128, 144, 160, 192]), rng.randint(160, 224), rng.choice(WALL_TEX),
                              rng.choice(FLOOR_TEX), rng.choice(CEIL_TEX)))
        for x, y in r.cells():
            grid[x, y] = r.sector

    def new_sector(kind, ceil, light, **kw):
        sectors.append(Sector(kind, ceil, light, kw.pop("wall", corridor_wall), kw.pop("floor_tex", rng.choice(FLOOR_TEX)),
                              kw.pop("ceil_tex", rng.choice(CEIL_TEX)), **kw))
        return len(sectors) - 1

    for c in conns:
        A, B = rooms[c.a], rooms[c.b]
        if c.axis == "x":
            olo, ohi = max(A.y0, B.y0), min(A.y1, B.y1)
            y = rng.randint(olo, ohi - CORRIDOR_WIDTH)
            cols = [[(x, y + d) for d in range(CORRIDOR_WIDTH)] for x in range(A.x1, B.x0)]
        else:
            olo, ohi = max(A.x0, B.x0), min(A.x1, B.x1)
            x = rng.randint(olo, ohi - CORRIDOR_WIDTH)
            cols = [[(x + d, y) for d in range(CORRIDOR_WIDTH)] for y in range(A.y1, B.y0)]
        if not cols:
            raise _Retry("rooms touch")
        ceil = min(sectors[A.sector].ceil, sectors[B.sector].ceil, rng.choice([96, 112, 128]))
        light = rng.randint(160, 208)
        door_at = rng.randrange(len(cols)) if c.kind != "open" else None
        parts = [cols[:door_at], cols[door_at + 1:]] if door_at is not None else [cols]
        chain = [A.sector]
        for pi, part in enumerate(parts):
            if part:
                sid = new_sector("corridor", ceil, light)
                chain.append(sid)
                for cell in (cc for col in part for cc in col):
                    grid[cell] = sid
                    c.cells.append(cell)
            if door_at is not None and pi == 0:
                lock = c.lock
                top, track = LOCK_TEXTURES[lock] if lock else (rng.choice(DOOR_TEX), "DOORTRAK")
                sid = new_sector("door", 0, light, lock=lock, door_tex=top, track_tex=track, ceil_tex="CEIL1_1", floor_tex="FLAT20")
                c.door_sector = sid
                chain.append(sid)
                c.door_cells = list(cols[door_at])
                for cell in c.door_cells:
                    grid[cell] = sid
                    c.cells.append(cell)
        chain.append(B.sector)
        c.chain = chain

    # --- free cells, start, exit pad
    used: set[tuple[int, int]] = set()

    def entrances(ri: int) -> list[tuple[int, int]]:
        r = rooms[ri]
        out = []
        for c in conns:
            if c.a == ri or c.b == ri:
                for cell in c.cells:
                    for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nb = (cell[0] + d[0], cell[1] + d[1])
                        if r.contains(nb):
                            out.append(nb)
        return out

    S = rooms[start_room]
    if n == 1:
        # single room: start near one end of the long axis, exit at the opposite end
        long_x = S.w >= S.h
        span = S.w if long_x else S.h
        cand = [c for c in S.cells(1) if ((c[0] - S.x0) if long_x else (c[1] - S.y0)) < max(2, span // 3)]
        if rng.random() < 0.5:  # mirror: start at the other end
            cand = [((S.x0 + S.x1 - 1 - c[0]) if long_x else c[0], c[1] if long_x else (S.y0 + S.y1 - 1 - c[1])) for c in cand]
        start_cell = rng.choice(cand)
    else:
        start_cell = rng.choice(S.cells(1))
    used.add(start_cell)

    E = rooms[exit_room]
    refs = entrances(exit_room) + ([start_cell] if exit_room == start_room else [])
    pads = [(x, y) for x in range(E.x0, E.x1 - 1) for y in range(E.y0, E.y1 - 1)
            if all((x + dx, y + dy) != start_cell for dx in (0, 1) for dy in (0, 1))]
    score = {p: min(abs(p[0] + 0.5 - q[0]) + abs(p[1] + 0.5 - q[1]) for q in refs) for p in pads}
    best = max(score.values())
    if best < 4:
        raise _Retry("exit pad too close to the entrance")
    px, py = rng.choice([p for p in pads if score[p] >= 0.85 * best])
    exit_cells = [(px + dx, py + dy) for dx in (0, 1) for dy in (0, 1)]
    exit_sector = new_sector("exit", sectors[E.sector].ceil, min(255, sectors[E.sector].light + 16),
                             wall=sectors[E.sector].wall, floor_tex=EXIT_FLOOR, ceil_tex=sectors[E.sector].ceil_tex)
    for cell in exit_cells:
        grid[cell] = exit_sector
    # keep things off the pad and its ring (walking onto the pad ends the episode)
    used |= {(x + dx, y + dy) for x, y in exit_cells for dx in (-1, 0, 1) for dy in (-1, 0, 1)}

    things: list[dict] = []
    tu = lambda c: ((c[0] + 0.5) * CELL, (c[1] + 0.5) * CELL)  # noqa: E731  (offset added on emit)

    def put(type_, cell, angle=None, jitter=12):
        x, y = tu(cell)
        things.append({"type": type_, "x": x + rng.randint(-jitter, jitter), "y": y + rng.randint(-jitter, jitter),
                       "angle": rng.randrange(0, 360, 45) if angle is None else angle})
        used.add(cell)

    def free(ri: int, margin: int = 1, away: int = 0) -> list[tuple[int, int]]:
        return [c for c in rooms[ri].cells(margin) if c not in used
                and abs(c[0] - start_cell[0]) + abs(c[1] - start_cell[1]) >= away]

    start_angle = rng.randrange(360)
    things.append({"type": 1, "x": tu(start_cell)[0], "y": tu(start_cell)[1], "angle": start_angle})

    key_cell = None
    if key_color is not None:
        cand = free(key_room)
        if not cand:
            raise _Retry("no cell for the key")
        key_cell = rng.choice(cand)
        put(KEYS[key_color][0], key_cell, angle=0, jitter=8)

    # --- enemies: (count range, type weights) per stage
    if stage == 1:
        n_en, weights = rng.randint(1, 4), {"zombieman": 6, "shotgunguy": 2, "imp": 2}
    elif stage in (2, 3):
        n_en, weights = rng.randint(2, 6), {"zombieman": 5, "shotgunguy": 2, "imp": 3, "demon": 1}
    else:
        n_en, weights = rng.randint(6, 12), {"zombieman": 2, "shotgunguy": 3, "imp": 4, "demon": 3, "spectre": 1}
    # stage 1 is about combat, so enemies may share the start room (kept >= 5 cells away); later
    # stages keep the start room quiet so the first seconds are about orientation
    enemy_rooms = list(range(n)) if stage == 1 else ([i for i in range(n) if i != start_room] or [start_room])
    names = list(weights)
    for _ in range(n_en if enemies else 0):
        ri = rng.choice(enemy_rooms)
        cand = free(ri, 1, away=5) or free(ri, 1, away=3)  # not right next to the player start
        if not cand:
            continue
        put(MONSTERS[rng.choices(names, [weights[k] for k in names])[0]], rng.choice(cand))

    # --- items: stage 4 is a gauntlet with fewer pickups
    pool = ["stimpack", "stimpack", "medikit", "clip", "clip", "shells", "ammobox", "greenarmor"]
    n_items = {1: rng.randint(1, 3), 2: rng.randint(n // 2, n + 2), 3: rng.randint(n // 2, n + 2), 4: rng.randint(1, max(2, n // 2))}[stage]
    for _ in range(n_items):
        cand = free(rng.randrange(n), 1)
        if cand:
            put(ITEMS[rng.choice(pool)], rng.choice(cand))
    if stage >= 2 and rng.random() < (0.6 if stage == 4 else 0.35):
        cand = free(rng.randrange(n), 1)
        if cand:
            put(ITEMS["shotgun"], rng.choice(cand))

    lay = Layout(seed, stage, grid, sectors, rooms, conns, things, start_room, exit_room, start_cell, start_angle,
                 exit_cells, exit_sector, key_color, key_room, key_cell, locked_conn,
                 offset=(rng.randint(-8, 8) * CELL, rng.randint(-8, 8) * CELL))
    _validate(lay)
    return lay


def _validate(lay: Layout) -> None:
    """Reject layouts where two sectors touch that should not (would open a hole in a wall) or where
    two open cells meet only at a corner (pinch points)."""
    allowed = set()
    for c in lay.conns:
        for s, t in zip(c.chain, c.chain[1:]):
            allowed |= {(s, t), (t, s)}
    e = lay.rooms[lay.exit_room].sector
    allowed |= {(e, lay.exit_sector), (lay.exit_sector, e)}
    g = lay.grid
    for a, b in ((g[:-1, :], g[1:, :]), (g[:, :-1], g[:, 1:])):
        m = (a != b) & (a >= 0) & (b >= 0)
        for s, t in zip(a[m].tolist(), b[m].tolist()):
            if (s, t) not in allowed:
                raise _Retry(f"unexpected adjacency {s}-{t}")
    o = g >= 0
    pinch = (o[:-1, :-1] & o[1:, 1:] & ~o[1:, :-1] & ~o[:-1, 1:]) | (~o[:-1, :-1] & ~o[1:, 1:] & o[1:, :-1] & o[:-1, 1:])
    if pinch.any():
        raise _Retry("diagonal pinch")


def build_layout(seed: int, stage: int, *, enemies: bool = True, allow_heldout: bool = False) -> Layout:
    """Deterministic layout for (seed, stage). `enemies=False` is for scripted solvability tests."""
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    if is_heldout_seed(seed) and not allow_heldout:
        raise ValueError(f"seed {seed} is reserved for the held-out set (>= {HELDOUT_SEED_BASE})")
    rng = random.Random(int(seed) * 16 + stage)
    for _ in range(200):
        try:
            return _try_build(rng, int(seed), stage, enemies)
        except _Retry:
            continue
    raise RuntimeError(f"could not build a valid layout for seed={seed} stage={stage}")


def is_heldout_seed(seed: int) -> bool:
    return HELDOUT_SEED_BASE <= int(seed) < HELDOUT_SEED_BASE + HELDOUT_SEED_SPAN


# ---------------------------------------------------------------------------------- UDMF output
def _fmt(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, float):
        return f"{v:.1f}"
    return str(v)


def _block(kind: str, props: dict) -> str:
    return kind + " { " + " ".join(f"{k} = {_fmt(v)};" for k, v in props.items()) + " }"


def to_udmf(lay: Layout) -> tuple[str, dict]:
    g = lay.grid
    W, H = g.shape
    secs = lay.sectors

    def at(x, y):
        return int(g[x, y]) if 0 <= x < W and 0 <= y < H else -1

    def orient(lo: int, hi: int) -> tuple[int, int, bool]:
        """(front, back, front_is_low_side). Doors and the exit pad are always the back sector, so
        the door texture / use side faces outward (matches maps/two_rooms.wad)."""
        if lo < 0:
            return hi, -1, False
        if hi < 0:
            return lo, -1, True
        if secs[hi].kind in ("door", "exit"):
            return lo, hi, True
        if secs[lo].kind in ("door", "exit"):
            return hi, lo, False
        return lo, hi, True

    # Collect unit edges keyed by everything that defines the resulting linedef, then merge runs.
    runs: dict[tuple, list[int]] = {}
    for x in range(W + 1):  # vertical edges at x, spanning y..y+1; low side = west
        for y in range(H):
            lo, hi = at(x - 1, y), at(x, y)
            if lo != hi:
                runs.setdefault(("v", x) + orient(lo, hi), []).append(y)
    for y in range(H + 1):  # horizontal edges at y, spanning x..x+1; low side = south
        for x in range(W):
            lo, hi = at(x, y - 1), at(x, y)
            if lo != hi:
                runs.setdefault(("h", y) + orient(lo, hi), []).append(x)

    verts: dict[tuple[int, int], int] = {}
    vtext, stext, ltext = [], [], []
    ox, oy = lay.offset

    def vid(p):
        if p not in verts:
            verts[p] = len(verts)
            vtext.append(_block("vertex", {"x": float(p[0] * CELL + ox), "y": float(p[1] * CELL + oy)}))
        return verts[p]

    def side(sec: int, top="-", mid="-", bottom="-") -> int:
        stext.append(_block("sidedef", {"sector": sec, "texturetop": top, "texturemiddle": mid, "texturebottom": bottom}))
        return len(stext) - 1

    for key in sorted(runs, key=str):
        o, coord, front, back, front_low = key
        pos = sorted(runs[key])
        start = prev = pos[0]
        segs = []
        for p in pos[1:] + [None]:
            if p is not None and p == prev + 1:
                prev = p
                continue
            segs.append((start, prev + 1))
            if p is not None:
                start = prev = p
        for a, b in segs:
            if o == "v":
                p1, p2 = ((coord, a), (coord, b)) if not front_low else ((coord, b), (coord, a))  # right of v1->v2 = front
            else:
                p1, p2 = ((a, coord), (b, coord)) if front_low else ((b, coord), (a, coord))
            line = {"v1": vid(p1), "v2": vid(p2)}
            fs = secs[front]
            if back < 0:
                if fs.kind == "door":  # door jamb
                    line["sidefront"] = side(front, mid=fs.track_tex)
                else:
                    line["sidefront"] = side(front, mid=fs.wall)
                line["blocking"] = True
                line["dontpegbottom"] = True
            else:
                bs = secs[back]
                if bs.kind == "door":
                    line["sidefront"] = side(front, top=bs.door_tex, bottom=fs.wall)
                    line["sideback"] = side(back)
                    line["twosided"] = True
                    if bs.lock:
                        line.update(special=13, arg0=0, arg1=16, arg2=150, arg3=bs.lock, playeruse=True, repeatspecial=True)
                    else:
                        line.update(special=12, arg0=0, arg1=16, arg2=150, playeruse=True, monsteruse=True, repeatspecial=True)
                elif bs.kind == "exit":
                    line["sidefront"] = side(front)
                    line["sideback"] = side(back)
                    line.update(twosided=True, special=243, playercross=True)
                else:  # plain opening (room <-> corridor); upper textures show where ceilings differ
                    line["sidefront"] = side(front, top=fs.wall, bottom=fs.wall)
                    line["sideback"] = side(back, top=bs.wall, bottom=bs.wall)
                    line["twosided"] = True
            ltext.append(_block("linedef", line))

    sectext = [_block("sector", {"texturefloor": s.floor_tex, "textureceiling": s.ceil_tex, "heightfloor": s.floor,
                                 "heightceiling": s.ceil, "lightlevel": s.light}) for s in secs]
    ttext = []
    for t in lay.things:
        props = {"x": float(t["x"] + ox), "y": float(t["y"] + oy), "angle": int(t["angle"]), "type": int(t["type"])}
        props.update({f"skill{i}": True for i in range(1, 6)})
        props["single"] = True
        ttext.append(_block("thing", props))
    text = "\n".join(['namespace = "zdoom";'] + vtext + sectext + stext + ltext + ttext) + "\n"
    return text, {"num_vertices": len(vtext), "num_lines": len(ltext), "num_sides": len(stext), "num_sectors": len(secs)}


# ---------------------------------------------------------------------------------- public API
def metadata(lay: Layout, wad_path: str, stats: dict | None = None) -> dict:
    inv_m = {v: k for k, v in MONSTERS.items()}
    inv_i = {v: k for k, v in ITEMS.items()}
    enemies = Counter(inv_m[t["type"]] for t in lay.things if t["type"] in inv_m)
    items = Counter(inv_i[t["type"]] for t in lay.things if t["type"] in inv_i)
    ox, oy = lay.offset
    ex = np.mean(lay.exit_cells, axis=0)
    dist = _bfs(len(lay.rooms), [(c.a, c.b) for c in lay.conns], lay.start_room)
    room_centers = [[(r.x0 + r.w / 2) * CELL + ox, (r.y0 + r.h / 2) * CELL + oy] for r in lay.rooms]
    doors = [[c.door_sector, c.lock] for c in lay.conns if c.door_sector >= 0]
    return {
        "seed": lay.seed,
        "stage": lay.stage,
        "wad": str(wad_path),
        "map": "MAP01",
        "episode_ticks": EPISODE_TICKS[lay.stage],
        "heldout": is_heldout_seed(lay.seed),
        "num_rooms": len(lay.rooms),
        "num_enemies": sum(enemies.values()),
        "enemies": dict(sorted(enemies.items())),
        "items": dict(sorted(items.items())),
        "key_color": lay.key_color,
        "key_pos": None if lay.key_cell is None else list(lay.to_units(lay.key_cell)),
        "key_room": lay.key_room,
        "lock": 0 if lay.locked_conn is None else lay.conns[lay.locked_conn].lock,
        "locked_door_sector": None if lay.locked_conn is None else lay.conns[lay.locked_conn].door_sector,
        "door_sectors": doors,  # [sector index, lock] (lock 0 = ordinary door)
        "start": list(lay.to_units(lay.start_cell)),
        "start_angle": lay.start_angle,
        "start_room": lay.start_room,
        "exit": list(lay.to_units((float(ex[0]), float(ex[1])))),
        "exit_room": lay.exit_room,
        "exit_graph_distance": dist.get(lay.exit_room, 0),
        "room_centers": room_centers,
        "room_edges": [[c.a, c.b] for c in lay.conns],
        **(stats or {}),
    }


def generate(seed: int, stage: int, out_path: str | Path, *, enemies: bool = True, allow_heldout: bool = False) -> dict:
    """Write a one-map PWAD (MAP01) for (seed, stage) to `out_path`; return JSON-able metadata."""
    lay = build_layout(seed, stage, enemies=enemies, allow_heldout=allow_heldout)
    text, stats = to_udmf(lay)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_wad(out_path, [Lump("MAP01", b""), Lump("TEXTMAP", text.encode("ascii")), Lump("ENDMAP", b"")])
    return metadata(lay, str(out_path), stats)


def generate_set(stage: int, seeds: Iterable[int], out_dir: str | Path, *, allow_heldout: bool = False) -> list[dict]:
    out_dir = Path(out_dir)
    return [generate(s, stage, out_dir / f"s{stage}_{s}.wad", allow_heldout=allow_heldout) for s in seeds]


def generate_heldout(out_dir: str | Path = HELDOUT_DIR, per_stage: int = 2) -> list[dict]:
    """Held-out set: seed HELDOUT_SEED_BASE + i, stage i // per_stage + 1. Writes manifest.json."""
    out_dir = Path(out_dir)
    metas = []
    for i in range(per_stage * len(STAGES)):
        stage = i // per_stage + 1
        seed = HELDOUT_SEED_BASE + i
        m = generate(seed, stage, out_dir / f"heldout_s{stage}_{i}.wad", allow_heldout=True)
        m["wad"] = Path(m["wad"]).name  # relative to the manifest
        metas.append(m)
    (out_dir / "manifest.json").write_text(json.dumps(metas, indent=1))
    return metas


def _parse_seeds(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", type=int, choices=STAGES, default=1)
    ap.add_argument("--seeds", default="0-9", help="e.g. 0-99 or 1,5,7")
    ap.add_argument("--out", default="training/generated_maps")
    ap.add_argument("--heldout", action="store_true", help="write the held-out set to maps/heldout/")
    args = ap.parse_args(argv)
    if args.heldout:
        metas = generate_heldout()
        print(f"wrote {len(metas)} held-out maps to {HELDOUT_DIR}")
        return
    metas = generate_set(args.stage, _parse_seeds(args.seeds), args.out)
    Path(args.out, f"manifest_s{args.stage}.json").write_text(json.dumps(metas, indent=1))
    for m in metas:
        print(f"{m['wad']}: rooms={m['num_rooms']} enemies={m['num_enemies']} key={m['key_color']} sectors={m['num_sectors']} lines={m['num_lines']}")


if __name__ == "__main__":
    main()
