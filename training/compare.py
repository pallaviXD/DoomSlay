"""Compare candidate checkpoints fairly: every held-out map x several seeds, averaged.

    python -m training.compare checkpoints/cur1/best_upd1600.pt checkpoints/cur1/best_upd1800.pt --seeds 11,22,33

A single-seed held-out eval is noisy (one map exit = ~8 points), so the final submission is chosen
with this multi-seed comparison.
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

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--seeds", default="11,22,33")
    ap.add_argument("--out", default=str(ROOT / "docs" / "compare.json"))
    a = ap.parse_args()
    maps = load_heldout(DEFAULT_MANIFEST)
    seeds = [int(s) for s in a.seeds.split(",")]
    results = {}
    for ck in a.ckpts:
        r = evaluate(ck, maps, seeds=seeds)
        exits = sum(e["exit"] for e in r["episodes"])
        results[Path(ck).stem] = {k: v for k, v in r.items() if k != "episodes"} | {"exits": f"{exits}/{len(r['episodes'])}"}
        print(f"{Path(ck).stem:24s} score {r['score']:.1f} | exits {exits}/{len(r['episodes'])} | keys {r['keys']:.2f} "
              f"doors {r['doors']:.2f} kills {r['kills']:.2f} deaths {r['deaths']:.2f} | by tier "
              + " ".join(f"t{s}:{v['exit_rate']:.2f}" for s, v in r["by_stage"].items()), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=1, default=float))
