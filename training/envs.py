"""Multiprocessing vector env over the official doom_env wrapper.

- One DoomEnv per worker process (Windows => "spawn": the worker is a top-level function and this
  module does not import torch, so workers start fast and use little RAM).
- Each step returns batched packed observations (the arrays SlayNet uses), shaped rewards, done flags,
  and the finished episodes' stats. Episodes auto-reset inside the worker.
- Creating a DoomEnv costs ~0.5 s but reset only ~2 ms, so a worker replays a map for
  `episodes_per_map` episodes (different seeds) before loading another one from the current map pool.
  The pool is replaced by the curriculum via set_maps().
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

# Before numpy loads: one BLAS thread per worker (OpenBLAS buffers per core x 24 processes exhaust RAM).
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission"))
sys.path.insert(0, str(ROOT))

from slay_obs import OBS_KEYS, pack_obs  # noqa: E402  (numpy-only: workers never import torch)


def _pack(obs) -> tuple:
    p = pack_obs(obs)
    return tuple(p[k] for k in OBS_KEYS)


def _fast_timer() -> None:
    """Windows only: 1 ms timer resolution. Lock-step workers otherwise lose milliseconds per step
    waking up (default 15.6 ms scheduler tick) inside ViZDoom's engine-thread handshake."""
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.winmm.timeBeginPeriod(1)
        except Exception:
            pass


def _worker(conn, wid: int, maps: list[dict], seed: int, episodes_per_map: int, reward_weights, stuck_ticks):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _fast_timer()
    from doom_env import make_env

    from training.rewards import RewardShaper

    rng = np.random.default_rng(seed)
    shaper = RewardShaper(reward_weights, stuck_ticks=stuck_ticks)
    env = None
    cur = None
    eps_on_map = 0

    def load(m):
        nonlocal env, cur, eps_on_map
        if env is not None:
            env.close()
        env = make_env(m["wad"], m.get("map", "MAP01"), episode_ticks=int(m.get("ticks", 2100)), render_buffers=False)
        cur, eps_on_map = m, 0

    def new_episode():
        nonlocal eps_on_map
        if cur is None or eps_on_map >= episodes_per_map or cur not in maps:
            nxt = maps[int(rng.integers(len(maps)))]
            if nxt == cur:
                eps_on_map = 0  # same map drawn again: a cheap reset instead of a 0.5 s engine reload
            else:
                load(nxt)
        eps_on_map += 1
        obs, _ = env.reset(seed=int(rng.integers(2**31)))
        # Per-map stuck window (curriculum sets longer windows on key maps, where backtracking through
        # already-visited rooms is required and earns no new-cell progress).
        shaper.stuck_ticks = int(cur.get("stuck_ticks", stuck_ticks))
        shaper.reset(env.privileged_state(), env.mapinfo.door_sectors)
        return obs

    t_engine = t_shape = t_idle = 0.0
    n_steps = 0
    try:
        while True:
            t0 = time.perf_counter()
            cmd, data = conn.recv()
            t1 = time.perf_counter()
            t_idle += t1 - t0
            if cmd == "step":
                obs, _, term, trunc, info = env.step(data)
                t2 = time.perf_counter()
                priv = env.privileged_state()
                r, stuck = shaper.step(priv, obs["vitals"])
                t_engine += t2 - t1
                t_shape += time.perf_counter() - t2
                n_steps += 1
                done = term or trunc or stuck
                ep = None
                if done:
                    ep = shaper.stats.as_dict()
                    ep.update(stage=cur.get("stage", 0), wad=Path(cur["wad"]).name, timeout=bool(trunc))
                    obs = new_episode()
                conn.send((_pack(obs), r, done, ep))
            elif cmd == "reset":
                conn.send(_pack(new_episode()))
            elif cmd == "stats":  # ms per step: engine, shaping+packing, idle (waiting for the learner)
                k = max(n_steps, 1)
                conn.send({"engine_ms": 1e3 * t_engine / k, "shape_ms": 1e3 * t_shape / k, "idle_ms": 1e3 * t_idle / k})
                t_engine = t_shape = t_idle = 0.0
                n_steps = 0
            elif cmd == "set_maps":
                maps = list(data)
                conn.send(True)
            elif cmd == "close":
                break
    except Exception:  # surface worker crashes to the main process instead of hanging
        conn.send(("__error__", traceback.format_exc()))
    finally:
        if env is not None:
            env.close()
        conn.close()


class VecEnv:
    def __init__(self, num_envs: int, maps: list[dict], seed: int = 0, episodes_per_map: int = 8,
                 reward_weights: dict | None = None, stuck_ticks: int = 200):
        ctx = mp.get_context("spawn")
        self.n = num_envs
        self.conns, self.procs = [], []
        for i in range(num_envs):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(child, i, maps, seed * 1000 + i, episodes_per_map, reward_weights, stuck_ticks),
                            daemon=True)
            p.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(p)
        self.steps = 0
        self._t0 = time.time()

    def _recv(self, c):
        msg = c.recv()
        if isinstance(msg, tuple) and len(msg) == 2 and isinstance(msg[0], str) and msg[0] == "__error__":
            raise RuntimeError("env worker crashed:\n" + msg[1])
        return msg

    @staticmethod
    def _stack(packed: list[tuple]) -> dict[str, np.ndarray]:
        return {k: np.stack([p[i] for p in packed]) for i, k in enumerate(OBS_KEYS)}

    def reset(self) -> dict[str, np.ndarray]:
        for c in self.conns:
            c.send(("reset", None))
        return self._stack([self._recv(c) for c in self.conns])

    def step(self, buttons: np.ndarray):
        """buttons (N,9) {0,1} -> obs dict (N,...), reward (N,), done (N,), list of finished-episode dicts."""
        for c, b in zip(self.conns, buttons):
            c.send(("step", b))
        res = [self._recv(c) for c in self.conns]
        self.steps += self.n
        obs = self._stack([r[0] for r in res])
        rew = np.array([r[1] for r in res], np.float32)
        done = np.array([r[2] for r in res], np.float32)
        eps = [r[3] for r in res if r[3] is not None]
        return obs, rew, done, eps

    # ---- asynchronous halves: lets the learner compute actions for one half while the other steps
    def step_async(self, idx: range, buttons: np.ndarray) -> None:
        for i, b in zip(idx, buttons):
            self.conns[i].send(("step", b))

    def step_wait(self, idx: range):
        res = [self._recv(self.conns[i]) for i in idx]
        self.steps += len(idx)
        obs = self._stack([r[0] for r in res])
        return obs, np.array([r[1] for r in res], np.float32), np.array([r[2] for r in res], np.float32), \
            [r[3] for r in res if r[3] is not None]

    def reset_part(self, idx: range) -> dict[str, np.ndarray]:
        for i in idx:
            self.conns[i].send(("reset", None))
        return self._stack([self._recv(self.conns[i]) for i in idx])

    def set_maps(self, maps: list[dict]) -> None:
        for c in self.conns:
            c.send(("set_maps", maps))
        for c in self.conns:
            self._recv(c)

    def worker_stats(self) -> dict:
        """Mean per-step timing across workers (resets the counters)."""
        for c in self.conns:
            c.send(("stats", None))
        st = [self._recv(c) for c in self.conns]
        return {k: float(np.mean([s[k] for s in st])) for k in st[0]}

    def sps(self) -> float:
        return self.steps / max(time.time() - self._t0, 1e-6)

    def close(self):
        for c in self.conns:
            try:
                c.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
