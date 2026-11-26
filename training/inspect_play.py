"""Behaviour inspection: play the real agent on the held-out maps and analyse what it actually does.

    python -m training.inspect_play --ckpt checkpoints/cur1/best.pt --tag upd1700

Writes videos/inspect_<tag>/:
  report.md          plain-language summary + behaviour stats
  <map>_path.png     top-down path (start, end, where keys/doors happened)
  <map>_frames.png   contact sheet of first-person frames (every ~2 s of game time)
  <map>_play.gif     full first-person video (only for --gif-maps)

Logs alone can hide bad behaviour (e.g. firing at walls), so these stats are measured from the
observation the agent saw and the buttons it pressed. Privileged position is used only for drawing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ENEMY_TYPES = (1, 2, 3)


def _r(v):  # NaN-safe rounding (a map may have no monster in view at all)
    return "-" if v != v else round(v)
RAY_DOOR, RAY_LOCKED = 2, 3


def play(env, agent, seed, keep_frames_every=0):
    obs, _ = env.reset(seed=seed)
    agent.reset()
    rec = {k: [] for k in ("fire", "vis", "crosshair", "strafe", "move", "turn", "use", "door_ahead", "x", "y", "keys", "kills", "health")}
    frames = []
    step = 0
    info = {}
    while True:
        e = obs["visible_entities"]
        is_enemy = (e["mask"] > 0) & np.isin(e["entity_type"], ENEMY_TYPES)
        a = np.asarray(agent.act(obs))
        p = env.privileged_state()
        rays = obs["rays"]
        front = np.r_[rays["hit_type"][-2:], rays["hit_type"][:3]]  # rays within ~±28 deg of straight ahead
        front_d = np.r_[rays["distance"][-2:], rays["distance"][:3]] * 1024
        rec["fire"].append(int(a[6]))
        rec["vis"].append(bool(is_enemy.any()))
        rec["crosshair"].append(bool((is_enemy & (e["features"][:, 2] > 0.99)).any()))  # within ~8 deg
        rec["strafe"].append(int(a[2] or a[3]))
        rec["move"].append(int(a[0] or a[1] or a[2] or a[3]))
        rec["turn"].append(int(a[4] or a[5]))
        rec["use"].append(int(a[7]))
        rec["door_ahead"].append(bool(np.any(np.isin(front, (RAY_DOOR, RAY_LOCKED)) & (front_d < 128))))
        rec["x"].append(p["x"]); rec["y"].append(p["y"])
        rec["keys"].append(float(p["keys"].sum())); rec["kills"].append(p["kills"]); rec["health"].append(p["health"])
        if keep_frames_every and step % keep_frames_every == 0:
            frames.append(obs["rgb"].copy())
        obs, _, term, trunc, info = env.step(a)
        step += 1
        if term or trunc:
            break
    p = env.privileged_state()
    rec = {k: np.array(v) for k, v in rec.items()}
    outcome = "EXIT" if info.get("exit_reached") else ("died" if p["dead"] else "out of time")
    return rec, frames, outcome, p, step


def stats(recs):
    cat = {k: np.concatenate([r[k] for r in recs]) for k in recs[0]}
    vis, ch, door = cat["vis"], cat["crosshair"], cat["door_ahead"]
    pct = lambda m, sel: 100.0 * m[sel].mean() if sel.any() else float("nan")  # noqa: E731
    spin = (cat["turn"] == 1) & (cat["move"] == 0)
    return {
        "fire_when_monster_visible_%": pct(cat["fire"], vis),
        "fire_when_no_monster_%": pct(cat["fire"], ~vis),
        "fire_when_monster_in_crosshair_%": pct(cat["fire"], ch),
        "sidestep_when_monster_visible_%": pct(cat["strafe"], vis),
        "sidestep_when_no_monster_%": pct(cat["strafe"], ~vis),
        "use_when_door_ahead_%": pct(cat["use"], door),
        "use_otherwise_%": pct(cat["use"], ~door),
        "spinning_in_place_%": 100.0 * spin.mean(),
        "monster_visible_%_of_time": 100.0 * vis.mean(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--manifest", default=str(ROOT / "maps" / "heldout" / "manifest.json"))
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--gif-maps", default="heldout_s1_1,heldout_s3_4")
    a = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from PIL import Image

    from doom_env import make_env
    from submission.agent import DoomAgent
    from training.curriculum import load_heldout

    out = ROOT / "videos" / f"inspect_{a.tag}"
    out.mkdir(parents=True, exist_ok=True)
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    gif_maps = set(a.gif_maps.split(","))
    rows, recs = [], []
    for m in load_heldout(a.manifest):
        name = Path(m["wad"]).stem
        env = make_env(m["wad"], m["map"], episode_ticks=m["ticks"], render_buffers=True)
        agent = DoomAgent(env.observation_space, env.action_space, "cpu")
        agent.net.load_state_dict(sd["model"] if "model" in sd else sd)
        rec, frames, outcome, p, steps = play(env, agent, a.seed, keep_frames_every=2)
        recs.append(rec)
        # top-down path with events
        fig, ax = plt.subplots(figsize=(5, 5))
        for _, segs in env.sector_polygons():
            for x1, y1, x2, y2 in segs:
                ax.plot([x1, x2], [y1, y2], color="#666", lw=0.5)
        ax.plot(rec["x"], rec["y"], color="#b91c1c", lw=1.2)
        ax.scatter(rec["x"][0], rec["y"][0], c="green", s=35, zorder=3, label="start")
        k = np.flatnonzero(np.diff(rec["keys"]) > 0)
        if len(k):
            ax.scatter(rec["x"][k + 1], rec["y"][k + 1], c="gold", edgecolors="k", s=70, marker="*", zorder=4, label="key picked")
        kk = np.flatnonzero(np.diff(rec["kills"]) > 0)
        if len(kk):
            ax.scatter(rec["x"][kk + 1], rec["y"][kk + 1], c="purple", s=25, marker="^", zorder=4, label="kill")
        ax.scatter(rec["x"][-1], rec["y"][-1], c="black", marker="x", s=45, zorder=3, label=f"end: {outcome}")
        ax.set_aspect("equal"); ax.axis("off"); ax.legend(fontsize=7, loc="lower right")
        ax.set_title(f"{name} (tier {m['stage']})", fontsize=9)
        fig.savefig(out / f"{name}_path.png", dpi=110, bbox_inches="tight"); plt.close(fig)
        # contact sheet: up to 24 frames spread over the episode
        sel = np.linspace(0, len(frames) - 1, min(24, len(frames))).astype(int)
        cols = 6
        sheet = Image.new("RGB", (cols * 160, ((len(sel) + cols - 1) // cols) * 120))
        for i, j in enumerate(sel):
            sheet.paste(Image.fromarray(frames[j]), ((i % cols) * 160, (i // cols) * 120))
        sheet.save(out / f"{name}_frames.png")
        if name in gif_maps:
            ims = [Image.fromarray(f).resize((320, 240), Image.NEAREST) for f in frames]
            ims[0].save(out / f"{name}_play.gif", save_all=True, append_images=ims[1:], duration=int(1000 / 35 * 8), loop=0)
        s = stats([rec])
        rows.append({"map": name, "tier": m["stage"], "outcome": outcome, "seconds": round(steps * 4 / 35),
                     "kills": int(p["kills"]), "keys": int(p["keys"].sum()), "health_end": int(p["health"]),
                     "fire_vis": _r(s["fire_when_monster_visible_%"]), "fire_novis": _r(s["fire_when_no_monster_%"])})
        env.close()
        print(rows[-1], flush=True)

    S = stats(recs)
    lines = [f"# Behaviour inspection: {a.tag}", "", f"checkpoint: `{a.ckpt}`, seed {a.seed}", "",
             "| map | tier | outcome | game s | kills | keys | health at end | fire% monster visible | fire% none |",
             "|---|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r['map']} | {r['tier']} | {r['outcome']} | {r['seconds']} | {r['kills']} | {r['keys']} | {r['health_end']} | "
              f"{r['fire_vis']} | {r['fire_novis']} |" for r in rows]
    lines += ["", "## Behaviour (all maps)", ""] + [f"- {k.replace('_', ' ')}: **{v:.0f}**" for k, v in S.items()]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "stats.json").write_text(json.dumps({"maps": rows, "behaviour": S}, indent=1))
    print(json.dumps(S, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
