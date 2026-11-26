"""Watch the agent play live in a window (real game speed).

    python -m training.watch                                   # all 8 held-out maps, submission weights
    python -m training.watch --wad maps/heldout/heldout_s4_7.wad --ckpt checkpoints/cur1/best_upd2200.pt

Keys: SPACE = pause, N = next map, ESC = quit. Display only; the agent sees just its normal observation.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wad", default=None, help="one map (default: every held-out map)")
    ap.add_argument("--ckpt", default=None, help="checkpoint (default: submission/weights.pt)")
    ap.add_argument("--speed", type=float, default=1.0, help="1.0 = real time, 2.0 = double speed")
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()

    import numpy as np
    import pygame
    import torch

    from doom_env import make_env
    from submission.agent import DoomAgent
    from training.curriculum import load_heldout

    if a.wad:
        maps = [{"wad": a.wad, "map": "MAP01", "ticks": 4200, "stage": "?"}]
    else:
        maps = load_heldout(ROOT / "maps" / "heldout" / "manifest.json")
    pygame.init()
    screen = pygame.display.set_mode((640, 520))
    font = pygame.font.SysFont("consolas", 18)
    step_dt = 4 / 35 / a.speed  # one decision = 4 game tics

    for m in maps:
        env = make_env(m["wad"], m.get("map", "MAP01"), episode_ticks=m["ticks"], render_buffers=True)
        agent = DoomAgent(env.observation_space, env.action_space, "cpu")
        if a.ckpt:
            sd = torch.load(a.ckpt, map_location="cpu", weights_only=True)
            agent.net.load_state_dict(sd["model"] if "model" in sd else sd)
        obs, _ = env.reset(seed=a.seed)
        agent.reset()
        name, step, paused, skip, result = Path(m["wad"]).stem, 0, False, False, ""
        while True:
            t0 = time.time()
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                    pygame.quit()
                    return
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_SPACE:
                    paused = not paused
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_n:
                    skip = True
            if skip:
                break
            if not paused and not result:
                obs, _, term, trunc, info = env.step(agent.act(obs))
                step += 1
                if term or trunc:
                    p = env.privileged_state()
                    result = "EXIT REACHED!" if info.get("exit_reached") else ("DIED" if p["dead"] else "TIME UP")
                    end_time = time.time()
            p = env.privileged_state()
            frame = pygame.surfarray.make_surface(np.transpose(obs["rgb"], (1, 0, 2)))
            screen.fill((15, 15, 15))
            screen.blit(pygame.transform.scale(frame, (640, 480)), (0, 0))
            hud = f"{name} (tier {m['stage']})  {step * 4 / 35:5.1f}s  health {p['health']:.0f}  kills {p['kills']}  keys {int(p['keys'].sum())}"
            screen.blit(font.render(hud, True, (230, 230, 230)), (8, 492))
            if result:
                screen.blit(font.render(result + "   (N = next map)", True, (255, 220, 60)), (8, 8))
            pygame.display.flip()
            if result and time.time() - end_time > 3:
                break
            time.sleep(max(0.0, step_dt - (time.time() - t0)))
        env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
