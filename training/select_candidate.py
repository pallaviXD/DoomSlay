"""Candidate selection: a checkpoint must beat the current submission on BOTH
  (a) the generated held-out set (8 maps x 12 seeds, tier-weighted approx score), and
  (b) the hand-made template map maps/two_rooms.wad (8 seeds, exit count),
because a version that scored best on (a) failed (b) 0/8 (it never took the corridor to the exit room):
generated maps alone do not guarantee transfer to hand-made layouts.

    python -m training.select_candidate checkpoints/mem2/cand_x.pt --baseline checkpoints/mem2/cand_l3_4240.pt
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.curriculum import load_heldout  # noqa: E402
from training.eval import DEFAULT_MANIFEST, evaluate  # noqa: E402

SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 111, 122, 133]
TWO_ROOMS = [{"wad": str(ROOT / "maps" / "two_rooms.wad"), "map": "MAP01", "ticks": 2100, "stage": 2}]


def assess(ckpt: str) -> dict:
    two = evaluate(ckpt, TWO_ROOMS, seeds=range(8), processes=2)
    two_exits = sum(e["exit"] for e in two["episodes"])
    out = {"two_rooms_exits": two_exits}
    if two_exits < 4:  # fail fast: not worth 12-seed held-out time
        return out
    held = evaluate(ckpt, load_heldout(DEFAULT_MANIFEST), seeds=SEEDS, processes=2)
    out.update(score=held["score"], exit_rate=held["exit_rate"], kills=held["kills"], deaths=held["deaths"],
               by_tier={k: round(v["exit_rate"], 2) for k, v in held["by_stage"].items()})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--baseline_two_rooms", type=int, default=6, help="baseline exits on two_rooms (of 8)")
    ap.add_argument("--baseline_score", type=float, default=54.2, help="baseline 12-seed held-out score")
    a = ap.parse_args()
    r = assess(a.ckpt)
    ok = r["two_rooms_exits"] >= a.baseline_two_rooms and r.get("score", -1) > a.baseline_score
    print(json.dumps({"ckpt": Path(a.ckpt).name, **r, "PASS": ok}, default=float))

