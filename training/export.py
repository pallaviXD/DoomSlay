"""Export a training checkpoint to submission/weights.pt (model weights only, no optimizer state).

    python -m training.export checkpoints/cur1/best.pt

Written atomically (tmp file + replace) so a concurrently running eval never reads a partial file.
"""

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "submission" / "weights.pt"


def main(src: str) -> None:
    ck = torch.load(src, map_location="cpu", weights_only=False)
    meta = {k: ck[k] for k in ("update", "steps", "best_score", "curriculum") if k in ck}
    payload = {"model": ck["model"], "sample": True, "meta": json.dumps({"source": str(src), **meta}, default=str)}
    tmp = OUT.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(OUT)
    print(f"exported {src} -> {OUT} ({OUT.stat().st_size / 1e6:.1f} MB) {meta}")


if __name__ == "__main__":
    main(sys.argv[1])
