"""Team Slay training loop (launched via training/train.py): vector env -> recurrent PPO -> tensorboard, checkpoints, held-out eval.

    python -m training.train --run s1 --stage 1 --fixed-stage --total-steps 20000000
    python -m training.train --run cur --stage 1 --resume checkpoints/s1/best.pt   # full curriculum
    python -m training.train --run smoke --smoke                                   # 2-minute smoke test

The best checkpoint is chosen by held-out score (maps/heldout, never trained on), not by recency.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from submission.model import PREV_ACT_DIM, SlayNet, actions_to_buttons  # noqa: E402
from training.curriculum import Curriculum, load_heldout  # noqa: E402
from training.envs import VecEnv, _fast_timer  # noqa: E402
from training.eval import DEFAULT_MANIFEST, evaluate  # noqa: E402
from training.ppo import PPO, PPOConfig, RolloutBuffer, prev_from_actions, to_torch_obs  # noqa: E402
from training.rewards import REWARD_WEIGHTS  # noqa: E402


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--num-envs", type=int, default=24)
    ap.add_argument("--total-steps", type=float, default=20e6)
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--max-stage", type=int, default=4)
    ap.add_argument("--fixed-stage", action="store_true", help="stay on --stage (no auto-advance)")
    ap.add_argument("--mixed", action="store_true", help="sample all stages from the start")
    ap.add_argument("--thresholds", default="0.6,0.5,0.4", help="exit rate to advance from stage 1,2,3")
    ap.add_argument("--pool-size", type=int, default=48)
    ap.add_argument("--regen-episodes", type=int, default=1500)
    ap.add_argument("--resume", default=None, help="checkpoint to initialise weights from")
    ap.add_argument("--resume-opt", action="store_true", help="also restore optimizer/curriculum state")
    ap.add_argument("--eval-every", type=int, default=25, help="updates between held-out evals")
    ap.add_argument("--ckpt-every", type=int, default=25)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-final", type=float, default=1e-4)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--stuck-ticks", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="tiny run on maps/two_rooms.wad")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    return ap.parse_args()


def save(path: Path, net, ppo, extra: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"model": net.state_dict(), "opt": ppo.opt.state_dict(), **extra}, tmp)
    tmp.replace(path)


def main():
    a = parse()
    _fast_timer()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck_dir = ROOT / "checkpoints" / a.run
    ck_dir.mkdir(parents=True, exist_ok=True)
    from torch.utils.tensorboard import SummaryWriter

    tb = SummaryWriter(str(ROOT / "runs" / a.run))

    cfg = PPOConfig(lr=a.lr, lr_final=a.lr_final, ent_coef=a.ent_coef)
    net = SlayNet().to(dev)
    ppo = PPO(net, cfg, dev)
    cur = Curriculum(ROOT / "training" / "generated_maps" / a.run, start_stage=a.stage, max_stage=a.max_stage,
                     pool_size=a.pool_size, regen_episodes=a.regen_episodes, mixed=a.mixed, seed=a.seed,
                     fixed_stage=a.fixed_stage,
                     thresholds={i + 1: float(x) for i, x in enumerate(a.thresholds.split(","))} | {4: 1.1})
    start_update, steps_done, best = 0, 0, -1e9
    if a.resume:
        ck = torch.load(a.resume, map_location=dev, weights_only=False)
        net.load_state_dict(ck["model"])
        if a.resume_opt:
            ppo.opt.load_state_dict(ck["opt"])
            start_update, steps_done = ck.get("update", 0), ck.get("steps", 0)
            if "curriculum" in ck:
                cur.load_state(ck["curriculum"])
            best = ck.get("best_score", best)
        print(f"resumed from {a.resume}")

    if a.smoke:
        pool = [{"wad": str(ROOT / "maps" / "two_rooms.wad"), "map": "MAP01", "ticks": 2100, "stage": 1}]
        a.total_steps = min(a.total_steps, a.num_envs * cfg.rollout_len * 6)
        a.eval_every = a.ckpt_every = 3
    else:
        pool = cur.build_pool()
    heldout = load_heldout(a.manifest) if Path(a.manifest).is_file() else []

    meta = {"args": vars(a), "ppo": cfg.__dict__, "rewards": REWARD_WEIGHTS}
    (ck_dir / "config.json").write_text(json.dumps(meta, indent=1, default=str))
    tb.add_text("config", json.dumps(meta, indent=1, default=str))
    print(json.dumps(meta, default=str))

    venv = VecEnv(a.num_envs, pool, seed=a.seed, stuck_ticks=a.stuck_ticks)
    N, T = a.num_envs, cfg.rollout_len
    # Rollouts run on a CPU copy of the policy (measured: 2.7 ms/step on CPU vs 5.4 ms on the GPU for a
    # 24-env batch, because of per-kernel launch overhead on Windows). PPO updates run on the GPU.
    torch.set_num_threads(2)
    cpu = torch.device("cpu")
    actor = SlayNet()
    actor.load_state_dict(net.state_dict())
    actor.eval()
    # Two pipelined halves: while half A runs in the engines, the policy computes actions for half B.
    halves = [range(0, N // 2), range(N // 2, N)] if N >= 2 else [range(0, N)]
    G = [{"idx": r, "obs": venv.reset_part(r), "h": actor.initial_state(len(r)), "prev": torch.zeros(len(r), PREV_ACT_DIM),
          "starts": torch.ones(len(r)), "buf": RolloutBuffer(T, len(r), net.hidden, cpu), "pending": None} for r in halves]

    def send(g):  # policy forward for one half, then dispatch its actions to the engines
        o = to_torch_obs(g["obs"], cpu)
        act, logp, val, h_new = ppo.act(o, g["prev"], g["h"], net=actor)
        venv.step_async(g["idx"], actions_to_buttons(act.numpy()))
        g["pending"] = (o, act, logp, val, h_new)

    def receive(g, finished):  # collect the half's step results and store the transition
        o, act, logp, val, h_new = g["pending"]
        g["obs"], rew, done, eps = venv.step_wait(g["idx"])
        d = torch.from_numpy(done)
        g["buf"].add(o, g["prev"], g["h"], g["starts"], act, logp, val, torch.from_numpy(rew), d)
        g["h"] = h_new * (1.0 - d).unsqueeze(-1)  # GRU reset on episode end
        g["starts"] = d
        g["prev"] = prev_from_actions(act, d)
        finished += eps

    recent = deque(maxlen=200)
    total_updates = int(a.total_steps // (N * T))
    t_start = time.time()
    update = start_update

    try:
        while update < start_update + total_updates:
            update += 1
            lr = ppo.set_lr((update - start_update) / max(total_updates, 1))
            for g in G:
                g["buf"].clear(net.hidden)
            t_roll = time.time()
            finished = []
            send(G[0])
            for t in range(T):
                for j in range(1, len(G)):
                    send(G[j])
                receive(G[0], finished)
                if t < T - 1:
                    send(G[0])  # overlaps with the other half still stepping
                for j in range(1, len(G)):
                    receive(G[j], finished)
            last_v = torch.cat([ppo.value(to_torch_obs(g["obs"], cpu), g["prev"], g["h"], net=actor) for g in G])
            parts = [g["buf"].stack() for g in G]
            data = {k: ({kk: torch.cat([p[k][kk] for p in parts], 1) for kk in parts[0][k]} if isinstance(parts[0][k], dict)
                        else torch.cat([p[k] for p in parts], 1)) for k in parts[0]}
            roll_s = time.time() - t_roll
            t_upd = time.time()
            stats = ppo.update(data, last_v)
            actor.load_state_dict(net.state_dict())  # GPU learner -> CPU actor
            upd_s = time.time() - t_upd
            steps_done += N * T
            recent.extend(finished)

            # ---- curriculum
            if not a.smoke and cur.record(finished):
                old_stage = cur.stage
                pool = cur.build_pool()
                venv.set_maps(pool)
                print(f"[curriculum] new pool gen={cur.generation} stage={cur.stage} mixed={cur.mixed} (was {old_stage})", flush=True)

            # ---- logging
            sps = N * T / (roll_s + upd_s)
            tb.add_scalar("perf/sps", sps, steps_done)
            tb.add_scalar("perf/rollout_s", roll_s, steps_done)
            tb.add_scalar("perf/update_s", upd_s, steps_done)
            tb.add_scalar("train/lr", lr, steps_done)
            for k, v in stats.items():
                tb.add_scalar(f"loss/{k}", v, steps_done)
            tb.add_scalar("curriculum/stage", cur.stage + (0.5 if cur.mixed else 0), steps_done)
            for s, r in cur.exit_rates().items():
                tb.add_scalar(f"curriculum/exit_rate_s{s}", r, steps_done)
            if recent:
                rs = list(recent)
                for k in ("reward", "exit", "dead", "stuck", "kills", "keys", "doors", "cells", "steps", "damage"):
                    tb.add_scalar(f"episode/{k}", float(np.mean([e[k] for e in rs])), steps_done)
                for k in {k for e in rs for k in e if k.startswith("r_")}:
                    tb.add_scalar(f"reward_parts/{k[2:]}", float(np.mean([e.get(k, 0.0) for e in rs])), steps_done)
            if update % 5 == 0 or update == start_update + 1:
                ws = venv.worker_stats()
                rs = list(recent) or [{}]
                mean = lambda k: float(np.mean([e.get(k, 0) for e in rs]))  # noqa: E731
                print(f"upd {update} steps {steps_done/1e6:.2f}M sps {sps:.0f} (roll {roll_s:.1f}s upd {upd_s:.1f}s; "
                      f"engine {ws['engine_ms']:.2f}ms idle {ws['idle_ms']:.2f}ms) | R {mean('reward'):.2f} exit {mean('exit'):.2f} "
                      f"dead {mean('dead'):.2f} stuck {mean('stuck'):.2f} kills {mean('kills'):.2f} cells {mean('cells'):.1f} "
                      f"len {mean('steps'):.0f} | pg {stats['pg_loss']:.3f} vf {stats['vf_loss']:.3f} ent {stats['entropy']:.2f} "
                      f"kl {stats['approx_kl']:.4f} ev {stats['explained_var']:.2f} | stage {cur.stage}{'+mix' if cur.mixed else ''}",
                      flush=True)
                if not all(np.isfinite(v) for v in stats.values()):
                    raise FloatingPointError(f"non-finite loss stats: {stats}")

            extra = {"update": update, "steps": steps_done, "curriculum": cur.state(), "best_score": best}
            if update % a.ckpt_every == 0:
                save(ck_dir / "latest.pt", net, ppo, extra)
            if heldout and update % a.eval_every == 0:
                save(ck_dir / "eval_tmp.pt", net, ppo, extra)
                t_ev = time.time()
                res = evaluate(str(ck_dir / "eval_tmp.pt"), heldout, seeds=(11,))
                for k in ("score", "exit_rate", "keys", "doors", "kills", "cells", "deaths", "agent_ms"):
                    tb.add_scalar(f"heldout/{k}", res[k], steps_done)
                for st, v in res["by_stage"].items():
                    tb.add_scalar(f"heldout/exit_rate_s{st}", v["exit_rate"], steps_done)
                msg = (f"[heldout] upd {update} score {res['score']:.2f} exit {res['exit_rate']:.2f} keys {res['keys']:.2f} "
                       f"doors {res['doors']:.2f} kills {res['kills']:.2f} cells {res['cells']:.1f} deaths {res['deaths']:.2f} "
                       f"({time.time() - t_ev:.0f}s)")
                if res["score"] > best:
                    best = res["score"]
                    extra["best_score"] = best
                    save(ck_dir / "best.pt", net, ppo, extra)
                    msg += "  ** new best"
                print(msg, flush=True)
                with open(ck_dir / "heldout_log.jsonl", "a") as f:
                    f.write(json.dumps({"update": update, "steps": steps_done, **{k: v for k, v in res.items() if k != "episodes"}},
                                       default=float) + "\n")
    finally:
        save(ck_dir / "latest.pt", net, ppo, {"update": update, "steps": steps_done, "curriculum": cur.state(), "best_score": best})
        venv.close()
        tb.close()
        print(f"done: {steps_done} steps in {(time.time() - t_start)/60:.1f} min, best held-out {best:.2f}")


if __name__ == "__main__":
    main()
