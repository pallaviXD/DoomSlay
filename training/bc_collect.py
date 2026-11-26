"""Behavioural-cloning (BC) stage 1: record teacher demonstrations on generated training maps.

    python -m training.bc_collect --episodes 300 --out training/generated_maps/bc

Why: the organisers suggest "Behavioral Cloning / Imitation Learning -> PPO fine-tuning" for the
key -> locked door -> exit maps, where pure RL exploration rarely stumbles on the right sequence.

TEACHER = training-time scaffolding only. It is the scripted test driver from tests/test_mapgen.py
(imported, not copied): BFS over the generator's grid + privileged x/y/angle + USE when stuck. It is
never used at evaluation and nothing under submission/ imports this file. The *student* (SlayNet)
only ever sees what DoomAgent sees at evaluation; this file records exactly that:

    pack_obs(obs) + egomap from SpatialMemory (reset per episode, update(obs) on the first observation,
    then update(obs, buttons_of_previous_step)) + previous action one-hot (15, zeros at t=0)

and the teacher's 9 buttons converted to the 6 factored head indices (`buttons_to_heads`).

Maps: training seeds only (never the held-out range), stages 2/3/4, a mix of plain, exit-switch,
stairs and switch+stairs maps, ENEMIES ON (real conditions; the teacher ignores them and may die).
Failed episodes are kept but flagged (`ep_success == 0`); bc_train uses successful ones by default.

Shard format (np.savez_compressed, one file per `--shard-episodes` episodes), per step:
    ray_dist f16 (T,64) | ray_type u8 (T,64) | ent_feat f16 (T,32,5) | ent_type u8 (T,32) |
    ent_mask u8 (T,32) | vitals f32 (T,17) | egomap u8 (T,5,21,21) = value * 4 (exact: the map only
    holds multiples of 0.25) | prev_act u8 (T,15) | action i8 (T,6) | buttons i8 (T,9) |
    episode i32 (T,) global episode id | t i32 (T,) step index inside the episode
and per episode: ep_id, ep_len, ep_success, ep_stage, ep_seed, ep_switch, ep_stairs, ep_dead.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "submission"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from slay_obs import HEAD_SIZES, actions_to_buttons, pack_obs  # noqa: E402  (numpy only)
from spatial_memory import SpatialMemory  # noqa: E402

from training import mapgen as M  # noqa: E402

EGOMAP_SCALE = 4.0  # egomap values are in {0, .25, .5, .75, 1}: uint8(value * 4) is lossless
NUM_LOGITS = sum(HEAD_SIZES)
_OFFS = np.cumsum((0,) + HEAD_SIZES[:-1])


def _load_test_driver():
    """Import the scripted driver from tests/test_mapgen.py by file path (tests/ is not a package)."""
    spec = importlib.util.spec_from_file_location("_mapgen_test_driver", ROOT / "tests" / "test_mapgen.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


T = _load_test_driver()


# ----------------------------------------------------------------------------- action encoding
def buttons_to_heads(buttons: np.ndarray) -> np.ndarray:
    """(..., 9) {0,1} buttons -> (..., 6) int64 head indices; inverse of slay_obs.actions_to_buttons.

    Contradictory pairs (MOVE_FORWARD + MOVE_BACKWARD, MOVE_LEFT + MOVE_RIGHT, TURN_LEFT + TURN_RIGHT)
    map to option 0 ("none"): in DOOM the two presses cancel out, so "none" is the action with the same
    effect, and it is the only one the factored heads can express. The scripted teacher never presses
    such pairs anyway; the rule just makes the conversion total."""
    b = np.asarray(buttons).astype(np.int64)

    def pair(pos, neg):  # 1 = first button only, 2 = second only, 0 = neither / both
        return np.where((b[..., pos] == 1) & (b[..., neg] == 0), 1, np.where((b[..., neg] == 1) & (b[..., pos] == 0), 2, 0))

    return np.stack([pair(0, 1), pair(2, 3), pair(4, 5), b[..., 6], b[..., 7], b[..., 8]], -1)


def heads_one_hot(heads: np.ndarray) -> np.ndarray:
    """(6,) head indices -> (15,) uint8 one-hot, the same layout as model.actions_one_hot."""
    out = np.zeros(NUM_LOGITS, np.uint8)
    out[_OFFS + np.asarray(heads)] = 1
    return out


# ----------------------------------------------------------------------------- recording teacher
class RecordingDriver(T.Driver):
    """The test driver, but every env step is intercepted to record (student observation, teacher label).

    The record for step t is taken BEFORE the action is sent, so it holds exactly the inputs DoomAgent
    would have when choosing that action."""

    def __init__(self, env, lay, obs0):
        super().__init__(env, lay)
        self.obs = obs0
        self.mem = SpatialMemory()  # reset in __init__; same call sequence as DoomAgent.act
        self.ego = self.mem.update(obs0)  # first observation of the map: no previous buttons
        self.prev = np.zeros(NUM_LOGITS, np.uint8)  # previous action one-hot, zeros at t=0
        self.rec: dict[str, list] = {k: [] for k in ("ray_dist", "ray_type", "ent_feat", "ent_type", "ent_mask",
                                                     "vitals", "egomap", "prev_act", "action", "buttons")}

    def step(self, a):
        if self.done:  # helpers occasionally call step() after the episode ended: ignore
            return self.obs
        heads = buttons_to_heads(a)
        btn = actions_to_buttons(heads)  # what the student's heads would press for this label
        p = pack_obs(self.obs)
        r = self.rec
        r["ray_dist"].append(p["ray_dist"].astype(np.float16))
        r["ray_type"].append(p["ray_type"].astype(np.uint8))
        r["ent_feat"].append(p["ent_feat"].astype(np.float16))
        r["ent_type"].append(p["ent_type"].astype(np.uint8))
        r["ent_mask"].append(p["ent_mask"].astype(np.uint8))
        r["vitals"].append(p["vitals"])
        r["egomap"].append(np.rint(self.ego * EGOMAP_SCALE).astype(np.uint8))
        r["prev_act"].append(self.prev.copy())
        r["action"].append(heads.astype(np.int8))
        r["buttons"].append(btn.astype(np.int8))
        obs = super().step(btn)
        if not self.done:
            self.ego = self.mem.update(obs, btn)
        self.prev = heads_one_hot(heads)
        return obs

    def arrays(self) -> dict[str, np.ndarray]:
        return {k: np.stack(v) for k, v in self.rec.items()}


def run_teacher(d: RecordingDriver, lay: M.Layout) -> None:
    """Teacher plan = the solvability tests' plan: key (if any, not crossing the locked door) -> exit
    (walk-over exit sector, or walk up to and USE the exit switch)."""
    try:
        if lay.key_cell is not None:
            d.go(lay.key_cell, blocked=set(lay.conns[lay.locked_conn].door_cells), max_steps=1500, tol=16)
            d.step(T.NOOP)
        if lay.switch_exit:
            T.press_switch(d, lay)
        else:
            d.go(lay.exit_cells[0], max_steps=2500)
    except AssertionError:  # press_switch asserts on a failed approach: just an unsuccessful demo
        pass


VARIANTS = ((False, False), (True, False), (False, True), (True, True))  # (switch_exit, stairs)


def episode_spec(i: int, rng: np.random.Generator) -> dict:
    """Round-robin over stages 2/3/4 and the 4 map variants; random TRAINING seed."""
    stage = (2, 3, 4)[i % 3]
    sw, st = VARIANTS[(i // 3) % len(VARIANTS)]
    seed = int(rng.integers(0, M.TRAIN_SEED_LIMIT))
    assert not M.is_heldout_seed(seed) and seed < M.TRAIN_SEED_LIMIT  # never the held-out range
    return {"stage": stage, "seed": seed, "switch_exit": sw, "stairs": st}


def collect_episode(spec: dict, wad: Path, env_seed: int) -> tuple[dict | None, dict]:
    from doom_env import make_env

    flags = {"switch_exit": spec["switch_exit"], "stairs": spec["stairs"]}
    try:
        meta = M.generate(spec["seed"], spec["stage"], wad, enemies=True, **flags)
        lay = M.build_layout(spec["seed"], spec["stage"], enemies=True, **flags)
    except Exception as exc:  # some (seed, flags) combos may not build: skip, never crash a long run
        return None, {"error": repr(exc)}
    env = make_env(str(wad), "MAP01", render_buffers=False, episode_ticks=meta["episode_ticks"])
    try:
        obs, _ = env.reset(seed=env_seed)
        d = RecordingDriver(env, lay, obs)
        run_teacher(d, lay)
        priv = env.privileged_state()
        info = {"success": bool(d.info.get("exit_reached", False)), "dead": bool(priv.get("dead", False)),
                "steps": len(d.rec["action"])}
        return (d.arrays() if info["steps"] else None), info
    finally:
        env.close()


def write_shard(path: Path, eps: list[tuple[int, dict, dict, dict]]) -> None:
    data = {k: np.concatenate([a[k] for _, a, _, _ in eps]) for k in eps[0][1]}
    data["episode"] = np.concatenate([np.full(len(a["action"]), i, np.int32) for i, a, _, _ in eps])
    data["t"] = np.concatenate([np.arange(len(a["action"]), dtype=np.int32) for _, a, _, _ in eps])
    data["ep_id"] = np.array([i for i, *_ in eps], np.int32)
    data["ep_len"] = np.array([len(a["action"]) for _, a, _, _ in eps], np.int32)
    data["ep_success"] = np.array([inf["success"] for *_, inf in eps], np.uint8)
    data["ep_dead"] = np.array([inf["dead"] for *_, inf in eps], np.uint8)
    data["ep_stage"] = np.array([s["stage"] for _, _, s, _ in eps], np.int32)
    data["ep_seed"] = np.array([s["seed"] for _, _, s, _ in eps], np.int64)
    data["ep_switch"] = np.array([s["switch_exit"] for _, _, s, _ in eps], np.uint8)
    data["ep_stairs"] = np.array([s["stairs"] for _, _, s, _ in eps], np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **data)
    tmp.replace(path)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    # Default 300: measured ~100 teacher steps/s single-process (incl. map build + engine load per episode)
    # while a 64-worker PPO run occupies the machine, ~250-300 steps per episode -> roughly 12-15 minutes.
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--out", default=str(ROOT / "training" / "generated_maps" / "bc"))
    ap.add_argument("--seed", type=int, default=0, help="seeds the map-seed RNG (different runs -> different maps)")
    ap.add_argument("--shard-episodes", type=int, default=25)
    ap.add_argument("--start-id", type=int, default=0, help="first global episode id (to append to a dataset)")
    a = ap.parse_args(argv)

    from training.envs import _fast_timer

    _fast_timer()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    wad = out / f"_teacher_{a.seed}.wad"
    rng = np.random.default_rng(a.seed)
    buf, n_ok, n_done, n_steps, n_skip = [], 0, 0, 0, 0
    by_stage: dict[int, list[int]] = {}
    t0 = time.perf_counter()
    for i in range(a.episodes):
        spec = episode_spec(i, rng)
        arrs, info = collect_episode(spec, wad, env_seed=int(rng.integers(2**31)))
        if arrs is None:
            n_skip += 1
            print(f"[bc_collect] skip {spec}: {info}", flush=True)
            continue
        eid = a.start_id + i
        buf.append((eid, arrs, spec, info))
        n_done += 1
        n_ok += info["success"]
        n_steps += info["steps"]
        by_stage.setdefault(spec["stage"], []).append(int(info["success"]))
        dt = time.perf_counter() - t0
        print(f"[bc_collect] ep {i + 1}/{a.episodes} stage {spec['stage']} sw {int(spec['switch_exit'])} "
              f"st {int(spec['stairs'])} steps {info['steps']:4d} exit {int(info['success'])} dead {int(info['dead'])} | "
              f"success {n_ok}/{n_done} | {n_steps / dt:.0f} steps/s | {dt / 60:.1f} min", flush=True)
        if len(buf) >= a.shard_episodes:
            write_shard(out / f"shard_s{a.seed}_{buf[0][0]:05d}.npz", buf)
            buf = []
    if buf:
        write_shard(out / f"shard_s{a.seed}_{buf[0][0]:05d}.npz", buf)
    wad.unlink(missing_ok=True)
    dt = time.perf_counter() - t0
    summary = {"episodes": n_done, "skipped": n_skip, "success": n_ok, "success_rate": n_ok / max(n_done, 1),
               "steps": n_steps, "minutes": dt / 60, "steps_per_s": n_steps / max(dt, 1e-9),
               "success_by_stage": {s: float(np.mean(v)) for s, v in sorted(by_stage.items())}}
    (out / f"collect_summary_s{a.seed}.json").write_text(json.dumps(summary, indent=1))
    print("[bc_collect] done:", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
