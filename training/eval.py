"""Held-out evaluation through the real submission interface (submission/agent.py DoomAgent).

    python -m training.eval --ckpt checkpoints/run/best.pt [--manifest maps/heldout/manifest.json]

Each map runs in its own process (parallel), with fixed seeds. The agent sees only the contract
observation; privileged state is used here only to *measure* (exit, keys, doors, cells, kills, damage).

APPROX_SCORE mirrors the published scoring categories (exact weights are hidden): exit dominates,
then progression, exploration, combat; harder tiers weigh more (tier weight = stage number).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MANIFEST = ROOT / "maps" / "heldout" / "manifest.json"


def approx_score(ep: dict, ticks: int) -> float:
    s = 0.0
    if ep["exit"]:
        s += 50.0 + 5.0 * min(ep.get("final_health", 0) / 100.0, 1.0) + 5.0 * max(0.0, 1.0 - ep["tics"] / ticks)
    s += 8.0 * ep["keys"] + 4.0 * min(ep["doors"], 3) + 4.0 * ep["locked_doors"]
    s += 15.0 * min(ep["cells"] / 40.0, 1.0)
    s += min(1.5 * ep["kills"], 10.0)
    s -= min(0.02 * ep["damage"], 5.0) + (5.0 if ep["dead"] else 0.0)
    return s


def _run_map(args) -> list[dict]:
    ckpt, m, seeds = args
    import torch

    from doom_env import make_env
    from submission.agent import DoomAgent
    from training.rewards import RewardShaper

    env = make_env(m["wad"], m.get("map", "MAP01"), episode_ticks=int(m["ticks"]), render_buffers=True)
    agent = DoomAgent(env.observation_space, env.action_space, "cpu")
    if ckpt:
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)
        agent.net.load_state_dict(sd["model"] if "model" in sd else sd)
    out = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        agent.reset()
        sh = RewardShaper(stuck_ticks=10**9)  # metrics only; never cut evaluation episodes
        sh.reset(env.privileged_state(), env.mapinfo.door_sectors)
        t_agent, steps, invalid = 0.0, 0, False
        while True:
            t0 = time.perf_counter()
            a = agent.act(obs)
            t_agent += time.perf_counter() - t0
            a = np.asarray(a)
            if a.shape != (9,) or not np.all((a == 0) | (a == 1)):
                invalid = True
                break
            obs, _, term, trunc, info = env.step(a)
            priv = env.privileged_state()
            sh.step(priv, obs["vitals"])
            steps += 1
            if term or trunc:
                break
        ep = sh.stats.as_dict()
        ep.update(map=Path(m["wad"]).name, stage=m.get("stage", 0), seed=seed, tics=priv["tic"],
                  final_health=priv["health"], invalid=invalid or agent.failures > 0, act_failures=agent.failures,
                  agent_ms=1e3 * t_agent / max(steps, 1))
        agent.failures = 0
        ep["score"] = approx_score(ep, int(m["ticks"]))
        out.append(ep)
    env.close()
    return out


def evaluate(ckpt: str | None, maps: list[dict], seeds=(11, 22), processes: int | None = None) -> dict:
    jobs = [(ckpt, m, list(seeds)) for m in maps]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes or min(len(jobs), 8)) as pool:
        eps = [e for res in pool.map(_run_map, jobs) for e in res]
    return summarize(eps)


def summarize(eps: list[dict]) -> dict:
    w = np.array([max(e["stage"], 1) for e in eps], float)
    sc = np.array([e["score"] for e in eps])
    summary = {
        "score": float((w * sc).sum() / w.sum()),
        "exit_rate": float(np.mean([e["exit"] for e in eps])),
        "keys": float(np.mean([e["keys"] for e in eps])),
        "doors": float(np.mean([e["doors"] for e in eps])),
        "locked_doors": float(np.mean([e["locked_doors"] for e in eps])),
        "kills": float(np.mean([e["kills"] for e in eps])),
        "cells": float(np.mean([e["cells"] for e in eps])),
        "damage": float(np.mean([e["damage"] for e in eps])),
        "deaths": float(np.mean([e["dead"] for e in eps])),
        "agent_ms": float(np.mean([e["agent_ms"] for e in eps])),
        "invalid": int(sum(e["invalid"] for e in eps)),
    }
    by_stage = {}
    for st in sorted({e["stage"] for e in eps}):
        se = [e for e in eps if e["stage"] == st]
        by_stage[st] = {"exit_rate": float(np.mean([e["exit"] for e in se])), "score": float(np.mean([e["score"] for e in se]))}
    summary["by_stage"] = by_stage
    summary["episodes"] = eps
    return summary


def main():
    from training.curriculum import load_heldout

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="checkpoint .pt (default: untrained/submission weights)")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--seeds", default="11,22")
    ap.add_argument("--json", default=None, help="write full results here")
    a = ap.parse_args()
    maps = load_heldout(a.manifest)
    t = time.time()
    res = evaluate(a.ckpt, maps, seeds=[int(s) for s in a.seeds.split(",")])
    print(f"held-out eval on {len(maps)} maps in {time.time() - t:.1f}s")
    for e in res["episodes"]:
        print(f"  {e['map']:28s} st{e['stage']} seed{e['seed']:>3} exit={int(e['exit'])} keys={e['keys']} doors={e['doors']} "
              f"locked={e['locked_doors']} kills={e['kills']} cells={e['cells']} dmg={e['damage']:.0f} dead={int(e['dead'])} "
              f"score={e['score']:.1f} {e['agent_ms']:.2f}ms/act")
    print(json.dumps({k: v for k, v in res.items() if k != "episodes"}, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
