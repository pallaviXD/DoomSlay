"""Play a map yourself (human player) with the same engine, IWAD, skill and maps as the agent.

    python -m training.play_human                    # random new stage-3 map (key + locked door + exit)
    python -m training.play_human --stage 1          # 1 small combat arena ... --stage 4 gauntlet
    python -m training.play_human --wad maps/heldout/heldout_s4_7.wad

Controls: W/S forward/back, A/D strafe, mouse or arrow keys to turn, left click or Ctrl to shoot,
E or Space to open doors (USE), Q next weapon, Esc menu. Time limit = the map's episode tick cap.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BINDS = [
    "w +forward", "s +back", "a +moveleft", "d +moveright", "uparrow +forward", "downarrow +back",
    "leftarrow +left", "rightarrow +right", "mouse1 +attack", "ctrl +attack", "e +use", "space +use",
    "q weapnext",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=3, choices=(1, 2, 3, 4))
    ap.add_argument("--wad", default=None)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()

    import vizdoom as vd

    from training import mapgen

    if a.wad:
        wad, ticks, desc = a.wad, 4200, Path(a.wad).name
    else:
        seed = a.seed if a.seed is not None else random.randrange(mapgen.TRAIN_SEED_LIMIT)
        wad = str(ROOT / "training" / "generated_maps" / "human" / f"s{a.stage}_{seed}.wad")
        meta = mapgen.generate(seed, a.stage, wad)
        ticks = meta["episode_ticks"]
        desc = f"stage {a.stage}, seed {seed}: {meta['num_rooms']} rooms, {meta['num_enemies']} monsters" + (
            f", {meta['key_color']} key + locked door" if meta.get("key_color") else "")
    print(f"Map: {desc}. Time limit {ticks / 35:.0f} s. Find the exit pad (it ends the level).")

    g = vd.DoomGame()
    g.set_doom_game_path(os.path.join(os.path.dirname(vd.__file__), "freedoom2.wad"))
    g.set_doom_scenario_path(str(Path(wad).resolve()))
    g.set_doom_map("MAP01")
    g.set_doom_skill(3)  # same skill as evaluation
    g.set_mode(vd.Mode.SPECTATOR)  # a human plays; the engine just records
    g.set_screen_resolution(vd.ScreenResolution.RES_800X600)
    g.set_window_visible(True)
    g.set_render_hud(True)
    g.set_render_crosshair(True)
    g.set_render_messages(True)
    g.set_episode_timeout(ticks)
    g.set_available_game_variables([vd.GameVariable.KILLCOUNT, vd.GameVariable.HEALTH, vd.GameVariable.ITEMCOUNT])
    # ViZDoom passes through only the listed buttons, even in spectator mode: the same 9 as the agent
    # plus mouse turning (TURN_LEFT_RIGHT_DELTA) so a human can aim with the mouse.
    B = vd.Button
    g.set_available_buttons([B.MOVE_FORWARD, B.MOVE_BACKWARD, B.MOVE_LEFT, B.MOVE_RIGHT, B.TURN_LEFT, B.TURN_RIGHT,
                             B.ATTACK, B.USE, B.SELECT_NEXT_WEAPON, B.TURN_LEFT_RIGHT_DELTA])
    g.add_game_args("+freelook 0 " + " ".join(f"+bind {b}" for b in BINDS))
    g.init()
    g.new_episode()
    while not g.is_episode_finished():
        g.advance_action()
    kills, health, items = (g.get_game_variable(v) for v in (vd.GameVariable.KILLCOUNT, vd.GameVariable.HEALTH,
                                                             vd.GameVariable.ITEMCOUNT))
    if g.is_player_dead():
        result = "You died"
    elif g.is_episode_timeout_reached():
        result = "Time ran out"
    else:
        result = "EXIT REACHED!"
    print(f"{result} after {g.get_episode_time() / 35:.0f} s | kills {kills:.0f} | health {health:.0f} | items {items:.0f}")
    g.close()


if __name__ == "__main__":
    main()
