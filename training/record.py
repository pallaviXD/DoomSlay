"""Watch the agent: record a first-person GIF and a top-down map with the walked path.

    python -m training.record --wad maps/heldout/heldout_s2_2.wad --ckpt checkpoints/cur1/best.pt
    -> videos/<map>_play.gif, videos/<map>_path.png

The agent sees only the contract observation; the screen buffer and the true position are used here
only to *show* what happened (visualisation, never fed to the policy).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wad", required=True)
    ap.add_argument("--map", default="MAP01")
    ap.add_argument("--ckpt", default=None, help="checkpoint (default: submission/weights.pt)")
    ap.add_argument("--ticks", type=int, default=4200)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--every", type=int, default=2, help="keep every n-th frame in the GIF")
    ap.add_argument("--out", default=str(ROOT / "videos"))
    a = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from PIL import Image

    from doom_env import make_env
    from submission.agent import DoomAgent

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    name = Path(a.wad).stem
    env = make_env(a.wad, a.map, episode_ticks=a.ticks, render_buffers=True)
    agent = DoomAgent(env.observation_space, env.action_space, "cpu")
    if a.ckpt:
        sd = torch.load(a.ckpt, map_location="cpu", weights_only=True)
        agent.load_weights(sd)
    obs, _ = env.reset(seed=a.seed)
    agent.reset()
    frames, path = [], []
    step = 0
    info = {}
    while True:
        p = env.privileged_state()
        path.append((p["x"], p["y"]))
        if step % a.every == 0:
            frames.append(Image.fromarray(obs["rgb"]).resize((320, 240), Image.NEAREST))
        obs, _, term, trunc, info = env.step(agent.act(obs))
        step += 1
        if term or trunc:
            break
    p = env.privileged_state()
    result = "EXIT REACHED" if info.get("exit_reached") else ("died" if p["dead"] else "time ran out")
    print(f"{name}: {result} after {step} steps ({step * 4 / 35:.0f} s game time), kills {p['kills']}, "
          f"health {p['health']:.0f}, keys {int(p['keys'].sum())}")

    gif = out / f"{name}_play.gif"
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=int(1000 / 35 * 4 * a.every), loop=0)

    fig, ax = plt.subplots(figsize=(6, 6))
    for _, segs in env.sector_polygons():
        for x1, y1, x2, y2 in segs:
            ax.plot([x1, x2], [y1, y2], color="#555", lw=0.6)
    xy = np.array(path)
    ax.plot(xy[:, 0], xy[:, 1], color="#b91c1c", lw=1.4, label="agent path")
    ax.scatter(*xy[0], c="green", s=40, zorder=3, label="start")
    ax.scatter(*xy[-1], c="black", marker="x", s=50, zorder=3, label=f"end ({result})")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(f"{name}: top-down view (for humans only; the agent never sees this)", fontsize=9)
    png = out / f"{name}_path.png"
    fig.savefig(png, dpi=130, bbox_inches="tight")
    env.close()
    print(f"saved {gif} ({gif.stat().st_size / 1e6:.1f} MB) and {png}")


if __name__ == "__main__":
    main()
