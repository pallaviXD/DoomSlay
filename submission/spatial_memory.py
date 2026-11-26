"""Ego-centric spatial memory: a self-built mini-map from the agent's own observations (numpy only).

Why (track brief: "how the agent maintains spatial memory across rooms"): a GRU alone forgets which
rooms were explored and where the locked door was. This module keeps an explicit occupancy map in a
dead-reckoned frame and hands the policy an ego-centric crop of it every step. It is preprocessing +
memory, not a decision rule: the network decides what to do with it.

Inputs used (all from the contract observation or the agent's own previous action):
- heading change: DOOM's keyboard turn rate is deterministic: per tic +-1.76 deg while the turn key has
  been held < 6 tics, then +-3.52 deg (measured: 7.03, 12.30, 14.06 deg per 4-tic step). We integrate
  our own TURN_LEFT/RIGHT presses.
- translation: vitals vel_forward / vel_right (units per tic, player frame) x 4 tics per decision
  (~5% error measured over 250 units).
- rays (64 x distance + hit type) paint free space, walls, doors and locked doors into the map.
No global coordinates or engine state are used; the map frame is wherever the episode started.

Output: (C=5, 21, 21) float32 crop, 64 units per cell (+-640 units), rotated so "up" = facing direction:
  0 visited (how often the agent stood in the cell, saturating), 1 free (seen passable), 2 wall,
  3 door, 4 locked door.
"""

from __future__ import annotations

import math

import numpy as np

CELL = 64.0
GRID = 160  # allocentric cells per side (10,240 units): larger than any map we expect
CROP = 21  # ego crop cells per side
CHANNELS = 5
NUM_RAYS = 64
RAY_RANGE = 1024.0
SLOW_TURN_DEG = 360.0 * 320 / 65536  # 1.7578 deg per tic
FAST_TURN_DEG = 360.0 * 640 / 65536  # 3.5156 deg per tic
SLOW_TURN_TICS = 6
TICS_PER_STEP = 4

RAY_ANGLES = np.arange(NUM_RAYS) * (2 * math.pi / NUM_RAYS)  # 0 = ahead, counter-clockwise
_SAMPLES = np.arange(0.5, RAY_RANGE / CELL, 1.0) * CELL  # free-space sample distances along each ray
_cr = (np.arange(CROP) - CROP // 2) * CELL
_EGO_F, _EGO_L = np.meshgrid(_cr[::-1], -_cr[::-1], indexing="ij")  # row 0 = far ahead; col 0 = left


class SpatialMemory:
    def __init__(self):
        self.grid = np.zeros((CHANNELS, GRID, GRID), np.float32)
        self.reset()

    def reset(self) -> None:
        self.grid[:] = 0.0
        self.x = self.y = 0.0  # dead-reckoned position (map units), start = origin
        self.h = 0.0  # dead-reckoned heading (radians), start = 0
        self.turnheld = 0

    # ------------------------------------------------------------------ pose
    def _turn(self, buttons) -> None:
        left, right = int(buttons[4]), int(buttons[5])
        d = left - right
        if d == 0:
            self.turnheld = 0
            return
        total = 0.0
        for _ in range(TICS_PER_STEP):
            self.turnheld += 1
            total += SLOW_TURN_DEG if self.turnheld < SLOW_TURN_TICS else FAST_TURN_DEG
        self.h += math.radians(total) * d

    def _cell(self, x, y):
        cx = np.floor(np.asarray(x) / CELL).astype(np.int64) + GRID // 2
        cy = np.floor(np.asarray(y) / CELL).astype(np.int64) + GRID // 2
        return cx, cy

    # ------------------------------------------------------------------ update
    def update(self, obs: dict, buttons=None) -> np.ndarray:
        """Call with every new observation; `buttons` = the 9-button action that led to it (None at reset)."""
        if buttons is not None:
            self._turn(buttons)
            v = obs["vitals"]
            fwd, right = float(v[14]) * TICS_PER_STEP, float(v[15]) * TICS_PER_STEP
            c, s = math.cos(self.h), math.sin(self.h)
            # heading h: forward = (cos h, sin h); right = (sin h, -cos h) (counter-clockwise angles)
            self.x += fwd * c + right * s
            self.y += fwd * s - right * c
        self._paint(obs)
        return self.crop()

    def _paint(self, obs: dict) -> None:
        g = self.grid
        cx, cy = self._cell(self.x, self.y)
        if 0 <= cx < GRID and 0 <= cy < GRID:
            g[0, cx, cy] = min(g[0, cx, cy] + 0.25, 1.0)
        dist = np.asarray(obs["rays"]["distance"], np.float32) * RAY_RANGE
        hit = np.asarray(obs["rays"]["hit_type"])
        ang = self.h + RAY_ANGLES
        ca, sa = np.cos(ang), np.sin(ang)
        # free space: samples strictly before the hit point
        m = _SAMPLES[None, :] < (dist[:, None] - CELL * 0.5)
        px = self.x + ca[:, None] * _SAMPLES[None, :]
        py = self.y + sa[:, None] * _SAMPLES[None, :]
        fx, fy = self._cell(px[m], py[m])
        ok = (fx >= 0) & (fx < GRID) & (fy >= 0) & (fy < GRID)
        g[1, fx[ok], fy[ok]] = 1.0
        # hit cells: wall / door / locked door (enemies and items move or vanish: not mapped)
        for ch, ht in ((2, 1), (3, 2), (4, 3)):
            sel = (hit == ht) & (dist < RAY_RANGE - 1)
            if sel.any():
                hx, hy = self._cell(self.x + ca[sel] * dist[sel], self.y + sa[sel] * dist[sel])
                ok = (hx >= 0) & (hx < GRID) & (hy >= 0) & (hy < GRID)
                g[ch, hx[ok], hy[ok]] = 1.0
        # Doors stay marked after opening: they are useful landmarks ("the locked door was over there").

    # ------------------------------------------------------------------ ego crop
    def crop(self) -> np.ndarray:
        c, s = math.cos(self.h), math.sin(self.h)
        # ego (forward f, left l) -> world: x = x0 + f*cos h - l*sin h ; y = y0 + f*sin h + l*cos h
        wx = self.x + _EGO_F * c - _EGO_L * s
        wy = self.y + _EGO_F * s + _EGO_L * c
        cx, cy = self._cell(wx, wy)
        ok = (cx >= 0) & (cx < GRID) & (cy >= 0) & (cy < GRID)
        out = np.zeros((CHANNELS, CROP, CROP), np.float32)
        out[:, ok] = self.grid[:, cx[ok], cy[ok]]
        return out
