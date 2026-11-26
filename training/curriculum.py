"""Curriculum over procedurally generated maps.

- Stages 1..4 (see mapgen.py). We train on stage `s` until its rolling exit rate over the last
  `window` episodes reaches `thresholds[s]`, then advance. After the last stage we sample all stages
  mixed (MIX_WEIGHTS), so earlier skills (combat) are not forgotten.
- While on stage s, `replay_frac` of the pool still comes from earlier stages (anti-forgetting).
- The map pool is regenerated with fresh seeds every `regen_episodes` finished episodes, so the agent
  can never memorise a layout: it has to learn behaviours that transfer to unseen maps.
- Training seeds come from a random range below mapgen.HELDOUT_SEED_BASE; held-out maps are never used.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np

from training import mapgen

MIX_WEIGHTS = {1: 0.15, 2: 0.30, 3: 0.35, 4: 0.20}


class Curriculum:
    def __init__(self, out_dir: str | Path, start_stage: int = 1, max_stage: int = 4, pool_size: int = 48,
                 regen_episodes: int = 1500, window: int = 300, thresholds: dict | None = None,
                 replay_frac: float = 0.25, mixed: bool = False, seed: int = 0, fixed_stage: bool = False):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.stage = start_stage
        self.max_stage = max_stage
        self.pool_size = pool_size
        self.regen_episodes = regen_episodes
        self.thresholds = thresholds or {1: 0.6, 2: 0.5, 3: 0.4, 4: 1.1}
        self.replay_frac = replay_frac
        self.mixed = mixed
        self.fixed_stage = fixed_stage  # never auto-advance (single-stage training runs)
        self.rng = np.random.default_rng(seed)
        self.recent = {s: deque(maxlen=window) for s in range(1, 5)}
        self.window = window
        self.eps_since_regen = 0
        self.generation = 0
        self.pool: list[dict] = []

    # ------------------------------------------------------------------ pool
    def _stage_weights(self) -> dict[int, float]:
        if self.mixed:
            w = {s: MIX_WEIGHTS[s] for s in range(1, self.max_stage + 1)}
        elif self.stage == 1 or self.replay_frac <= 0:
            w = {self.stage: 1.0}
        else:
            w = {self.stage: 1.0 - self.replay_frac}
            for s in range(1, self.stage):
                w[s] = self.replay_frac / (self.stage - 1)
        tot = sum(w.values())
        return {s: v / tot for s, v in w.items()}

    def build_pool(self) -> list[dict]:
        weights = self._stage_weights()
        stages = self.rng.choice(list(weights), size=self.pool_size, p=list(weights.values()))
        gen_dir = self.out_dir / f"gen{self.generation:04d}"
        gen_dir.mkdir(parents=True, exist_ok=True)
        pool = []
        for i, st in enumerate(stages):
            seed = int(self.rng.integers(0, mapgen.HELDOUT_SEED_BASE))  # never a held-out seed
            path = gen_dir / f"s{st}_{seed}.wad"
            meta = mapgen.generate(seed, int(st), path)
            pool.append({"wad": str(path), "map": meta.get("map", "MAP01"),
                         "ticks": int(meta.get("episode_ticks", 2100)), "stage": int(st), "seed": seed})
        self.pool = pool
        self.generation += 1
        self.eps_since_regen = 0
        # Keep disk use bounded: only the last 3 generations stay on disk (workers may still use the previous one).
        old = self.out_dir / f"gen{self.generation - 4:04d}"
        if old.is_dir():
            for f in old.iterdir():
                f.unlink(missing_ok=True)
            old.rmdir()
        return pool

    # ------------------------------------------------------------------ progress
    def record(self, episodes: list[dict]) -> bool:
        """Feed finished-episode stats. Returns True if the pool must be rebuilt (stage change / regen)."""
        for ep in episodes:
            s = int(ep.get("stage", 0))
            if s in self.recent:
                self.recent[s].append(1.0 if ep["exit"] else 0.0)
        self.eps_since_regen += len(episodes)
        changed = False
        if not self.mixed and not self.fixed_stage:
            r = self.recent[self.stage]
            if len(r) >= self.window and np.mean(r) >= self.thresholds.get(self.stage, 1.1):
                if self.stage < self.max_stage:
                    self.stage += 1
                else:
                    self.mixed = True
                changed = True
        return changed or self.eps_since_regen >= self.regen_episodes

    def exit_rates(self) -> dict[int, float]:
        return {s: float(np.mean(r)) for s, r in self.recent.items() if len(r)}

    def state(self) -> dict:
        return {"stage": self.stage, "mixed": self.mixed, "generation": self.generation}

    def load_state(self, st: dict) -> None:
        self.stage, self.mixed, self.generation = st["stage"], st["mixed"], st["generation"]


def load_heldout(manifest: str | Path) -> list[dict]:
    """Held-out maps (maps/heldout/manifest.json, written by mapgen --heldout)."""
    manifest = Path(manifest)
    items = json.loads(manifest.read_text())
    out = []
    for m in items:
        wad = Path(m["wad"])
        if not wad.is_absolute():
            wad = manifest.parent / wad.name
        out.append({"wad": str(wad), "map": m.get("map", "MAP01"), "ticks": int(m.get("episode_ticks", 2100)),
                    "stage": int(m.get("stage", 0)), "seed": m.get("seed")})
    return out
