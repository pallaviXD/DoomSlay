"""One MP4 of the AI playing every map, with the real DOOM HUD (status bar, face, ammo, keys, messages).

    python -m training.make_video --out videos/final_ai/all_maps.mp4

How: (1) the agent plays each map through the official doom_env wrapper (it sees only the contract
observation) while ViZDoom records a demo (.lmp); (2) each demo is replayed in a separate game with
the HUD on at 640x480, capturing every tic (35 fps), with a title card, a live score panel and a result
card; (3) everything is encoded into one H.264 MP4. Display only: nothing here affects the agent.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FPS = 35
W, H = 640, 480
TIER = {1: "Tier 1 - Combat", 2: "Tier 2 - Multi-room exploration", 3: "Tier 3 - Keycard, locked door, exit",
        4: "Tier 4 - Gauntlet", 0: "Hand-made test map", 5: "Real Freedoom level (all monsters)"}


def font(size):
    from PIL import ImageFont

    for f in ("C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf"):
        if os.path.exists(f):
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def card(lines, sub=None, bg=(10, 10, 14)):
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    y = H // 2 - 20 * len(lines) - (20 if sub else 0)
    for i, (txt, size, col) in enumerate(lines):
        f = font(size)
        w = d.textlength(txt, font=f)
        d.text(((W - w) / 2, y), txt, font=f, fill=col)
        y += size + 14
    if sub:
        f = font(16)
        for s in sub:
            w = d.textlength(s, font=f)
            d.text(((W - w) / 2, y + 6), s, font=f, fill=(170, 170, 180))
            y += 24
    return np.asarray(img)


def overlay(frame, title, kills, secs, idx, total):
    from PIL import Image, ImageDraw

    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle([0, 0, W, 26], fill=(0, 0, 0, 150))
    f = font(15)
    d.text((8, 4), f"{idx}/{total}  {title}", font=f, fill=(255, 220, 90))
    right = f"Kills {kills}   Time {int(secs) // 60}:{int(secs) % 60:02d}"
    d.text((W - d.textlength(right, font=f) - 8, 4), right, font=f, fill=(255, 255, 255))
    return np.asarray(img)


def result_frame(frame, result, color):
    from PIL import Image, ImageDraw

    img = Image.fromarray((frame * 0.45).astype(np.uint8))
    d = ImageDraw.Draw(img)
    for i, (txt, size, col) in enumerate(result):
        f = font(size)
        d.text(((W - d.textlength(txt, font=f)) / 2, H // 2 - 40 + i * (size + 12)), txt, font=f, fill=col or color)
    return np.asarray(img)


def play_and_record(m, lmp, seed):
    from doom_env import make_env
    from submission.agent import DoomAgent

    env = make_env(m["wad"], m["map"], episode_ticks=m["ticks"], render_buffers=False)
    agent = DoomAgent(env.observation_space, env.action_space, "cpu")
    obs, _ = env.reset(seed=seed, options={"record_path": str(lmp)})
    agent.reset()
    info, steps = {}, 0
    while True:
        obs, _, term, trunc, info = env.step(agent.act(obs))
        steps += 1
        if term or trunc:
            break
    p = env.privileged_state()
    res = {"exit": bool(info.get("exit_reached")), "dead": p["dead"], "kills": p["kills"], "health": p["health"],
           "keys": int(p["keys"].sum()), "secs": steps * 4 / 35}  # the engine clock resets when a level ends
    env.close()
    return res


def replay(m, lmp, writer, title, idx, total):
    import vizdoom as vd

    from doom_env.spaces import BUTTONS

    g = vd.DoomGame()
    g.set_doom_game_path(os.path.join(os.path.dirname(vd.__file__), "freedoom2.wad"))
    g.set_doom_scenario_path(str(Path(m["wad"]).resolve()))
    g.set_doom_map(m["map"])
    g.set_doom_skill(3)
    g.set_mode(vd.Mode.PLAYER)
    g.set_window_visible(False)
    g.set_sound_enabled(False)
    g.set_screen_resolution(vd.ScreenResolution.RES_640X480)
    g.set_screen_format(vd.ScreenFormat.RGB24)
    for setter in ("set_render_hud", "set_render_crosshair", "set_render_weapon", "set_render_messages",
                   "set_render_decals", "set_render_particles", "set_render_corpses", "set_render_effects_sprites"):
        if hasattr(g, setter):
            getattr(g, setter)(True)
    g.set_available_buttons([getattr(vd.Button, b) for b in BUTTONS])
    g.set_available_game_variables([vd.GameVariable.KILLCOUNT])
    g.set_episode_timeout(m["ticks"] + 10)
    g.init()
    g.replay_episode(str(lmp))
    last, n = None, 0
    while not g.is_episode_finished():
        s = g.get_state()
        if s is not None:
            last = np.ascontiguousarray(s.screen_buffer)
            writer.append_data(overlay(last, title, int(s.game_variables[0]), n / FPS, idx, total))
        g.advance_action()
        n += 1
    kills = int(g.get_game_variable(vd.GameVariable.KILLCOUNT))
    g.close()
    return last, kills


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "videos" / "final_ai" / "all_maps.mp4"))
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--only", default=None, help="comma list of map stems (default: all)")
    a = ap.parse_args()
    import imageio.v2 as imageio

    from training.curriculum import load_heldout

    maps = load_heldout(ROOT / "maps" / "heldout" / "manifest.json")
    maps.append({"wad": str(ROOT / "maps" / "two_rooms.wad"), "map": "MAP01", "ticks": 2100, "stage": 0})
    maps += load_heldout(ROOT / "maps" / "heldout_real" / "manifest.json")
    if a.only:
        keep = set(a.only.split(","))
        maps = [m for m in maps if Path(m["wad"]).stem in keep]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="slay_demo_"))
    writer = imageio.get_writer(str(out), fps=FPS, codec="libx264", quality=7, macro_block_size=16,
                                pixelformat="yuv420p")
    for _ in range(int(FPS * 4)):
        writer.append_data(card([("TEAM SLAY", 46, (220, 40, 40)), ("A neural network playing DOOM", 24, (235, 235, 235))],
                                [f"{len(maps)} maps it never trained on - recurrent PPO agent with spatial memory",
                                 "Every action comes from the trained network (no scripts, no aimbot)"]))
    summary = []
    for i, m in enumerate(maps, 1):
        name = Path(m["wad"]).stem
        title = f"{name} - {TIER.get(int(m.get('stage', 0)), '')}"
        lmp = tmp / f"{name}.lmp"
        res = play_and_record(m, lmp, a.seed)
        for _ in range(int(FPS * 2.5)):
            writer.append_data(card([(f"Map {i} of {len(maps)}", 22, (200, 200, 210)), (name, 40, (255, 220, 90)),
                                     (TIER.get(int(m.get("stage", 0)), ""), 22, (235, 235, 235))]))
        last, replay_kills = replay(m, lmp, writer, title, i, len(maps))
        if replay_kills != res["kills"]:  # the video must show exactly what the agent did
            print(f"WARNING {name}: replay kills {replay_kills} != live game kills {res['kills']}", flush=True)
        if res["exit"]:
            verdict, col = f"EXIT REACHED in {res['secs']:.0f} s", (90, 230, 110)
        elif res["dead"]:
            verdict, col = f"DIED after {res['secs']:.0f} s", (240, 80, 70)
        else:
            verdict, col = "TIME RAN OUT", (240, 190, 60)
        stats = f"kills {res['kills']}   health {max(res['health'], 0):.0f}   keys {res['keys']}"
        if last is None:
            last = np.zeros((H, W, 3), np.uint8)
        for _ in range(int(FPS * 2.5)):
            writer.append_data(result_frame(last, [(verdict, 34, col), (stats, 20, (255, 255, 255))], col))
        summary.append((name, verdict, stats))
        print(f"{i}/{len(maps)} {name}: {verdict} | {stats}", flush=True)
    lines = [("RESULTS", 34, (220, 40, 40))]
    sub = [f"{n}: {v} ({s})" for n, v, s in summary]
    for _ in range(int(FPS * 8)):
        writer.append_data(card(lines, sub))
    writer.close()
    (out.with_suffix(".json")).write_text(json.dumps(summary, indent=1))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
