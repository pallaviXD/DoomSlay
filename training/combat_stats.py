"""Combat diagnostics from real play: which weapon is held, ammo, weapon switches, fire timing, kills.

    python -m training.combat_stats --ckpt checkpoints/mem1/best_upd3000.pt

Weapon slots (vitals[2]): 1 = fist/chainsaw, 2 = pistol, 3 = shotgun, ... Ammo = vitals[3] for the held weapon.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seeds", default="11,22")
    a = ap.parse_args()
    import torch

    from doom_env import make_env
    from submission.agent import DoomAgent
    from training.curriculum import load_heldout

    sd = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    weapon, switches, fire_fist, fire_gun, fire_novis, fire_vis, steps, kills, vis_steps = Counter(), 0, 0, 0, 0, 0, 0, 0, 0
    empty_ammo_steps = 0
    for m in load_heldout(ROOT / "maps" / "heldout" / "manifest.json"):
        env = make_env(m["wad"], m["map"], episode_ticks=m["ticks"], render_buffers=False)
        agent = DoomAgent(env.observation_space, env.action_space, "cpu")
        agent.load_weights(sd)
        for seed in [int(s) for s in a.seeds.split(",")]:
            obs, _ = env.reset(seed=seed)
            agent.reset()
            while True:
                v = obs["vitals"]
                e = obs["visible_entities"]
                vis = bool(((e["mask"] > 0) & np.isin(e["entity_type"], (1, 2, 3))).any())
                act = np.asarray(agent.act(obs))
                w = int(v[2])
                weapon[w] += 1
                steps += 1
                vis_steps += vis
                empty_ammo_steps += int(v[3] <= 0 and w != 1)
                switches += int(act[8])
                if act[6]:
                    fire_fist += int(w <= 1)
                    fire_gun += int(w > 1)
                    fire_vis += int(vis)
                    fire_novis += int(not vis)
                obs, _, t, tr, _ = env.step(act)
                if t or tr:
                    kills += env.privileged_state()["kills"]
                    break
        env.close()
    tot_fire = max(fire_fist + fire_gun, 1)
    print("weapon held (% of steps):", {k: round(100 * c / steps, 1) for k, c in sorted(weapon.items())})
    print(f"next-weapon presses: {switches} ({100 * switches / steps:.1f}% of steps)")
    print(f"shots with fist: {100 * fire_fist / tot_fire:.0f}% | with a gun: {100 * fire_gun / tot_fire:.0f}%")
    print(f"shots while a monster is visible: {100 * fire_vis / tot_fire:.0f}% | at nothing: {100 * fire_novis / tot_fire:.0f}%")
    print(f"monster visible {100 * vis_steps / steps:.0f}% of the time | gun held with 0 ammo {100 * empty_ammo_steps / steps:.1f}% of steps")
    print(f"kills: {kills} in {16} games")


if __name__ == "__main__":
    main()
