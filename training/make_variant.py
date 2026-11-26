"""Build inference variants of trained checkpoints (no retraining), for the expert-panel experiments.

    python -m training.make_variant ensemble A.pt B.pt --out V.pt      # average per-head log-probs of A and B
    python -m training.make_variant temp A.pt --temp 0.7,0.7,1,1,1,1 --out V.pt
    python -m training.make_variant swa A.pt B.pt --out V.pt           # average the weights of A and B

Variants keep A's checkpoint fields (mask_level, update, ...) and are loaded by DoomAgent unchanged.
"""

import argparse

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["ensemble", "temp", "swa"])
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--temp", default=None, help="6 per-head temperatures")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    base = torch.load(a.ckpts[0], map_location="cpu", weights_only=False)
    out = {k: v for k, v in base.items() if k != "opt"}
    if a.kind == "ensemble":
        out["ensemble"] = [torch.load(p, map_location="cpu", weights_only=False)["model"] for p in a.ckpts[1:]]
    elif a.kind == "swa":
        sds = [torch.load(p, map_location="cpu", weights_only=False)["model"] for p in a.ckpts]
        out["model"] = {k: sum(sd[k].float() for sd in sds) / len(sds) for k in sds[0]}
    if a.temp:
        out["temperature"] = [float(x) for x in a.temp.split(",")]
    torch.save(out, a.out)
    print("wrote", a.out, "| ensemble members:", len(out.get("ensemble", [])), "| temperature:", out.get("temperature"))


if __name__ == "__main__":
    main()
