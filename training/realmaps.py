"""Real hand-made training maps from freedoom2.wad (bundled with ViZDoom).

Why: our generated maps are flat grids with walk-over exits. The hidden evaluation maps are
hand-made: stairs, lifts, odd shapes, exit *switches*. Freedoom's 32 levels have all of these.

    python -m training.realmaps            # writes training/generated_maps/real/*.wad + maps/heldout_real/

- Each level is copied into its own PWAD as MAP01 (Doom binary format, nodes included).
- Training copies: a fraction of monsters is kept (THINGS filtered) so a beginner agent can survive
  long enough to learn navigation; `keep` grows with the curriculum. (Map edits are allowed in training.)
- HELDOUT_REAL levels are never trained on; they get all monsters at skill 3 (as at evaluation) and
  form a judge-like test set in maps/heldout_real/ (manifest.json).
"""

from __future__ import annotations

import json
import os
import random
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from doom_env.wad import MAP_LUMPS_DOOM, Lump, read_wad, write_wad  # noqa: E402

HELDOUT_REAL = ("MAP02", "MAP08", "MAP15", "MAP31")  # never trained on
EXCLUDE = ("MAP30",)  # boss shooter level (Icon-of-Sin style), not useful
TRAIN_REAL = tuple(f"MAP{i:02d}" for i in range(1, 33) if f"MAP{i:02d}" not in HELDOUT_REAL + EXCLUDE)
REAL_TICKS = 4200  # the evaluation cap maximum
REAL_DIR = ROOT / "training" / "generated_maps" / "real"
HELDOUT_REAL_DIR = ROOT / "maps" / "heldout_real"

# DoomEd numbers of monsters (Doom 2 / Freedoom).
MONSTERS = {3004, 9, 65, 3001, 3002, 58, 3006, 3005, 69, 3003, 64, 66, 67, 68, 71, 16, 7, 84, 72}


def iwad_path() -> str:
    import vizdoom as vd

    return os.path.join(os.path.dirname(vd.__file__), "freedoom2.wad")


def map_lumps(lumps: list[Lump], name: str) -> list[Lump]:
    i = next(k for k, l in enumerate(lumps) if l.name == name)
    out = []
    for l in lumps[i + 1 :]:
        if l.name not in MAP_LUMPS_DOOM:
            break
        out.append(l)
    return out


def thin_things(data: bytes, keep: float, rng: random.Random) -> tuple[bytes, int, int]:
    """Keep each monster with probability `keep` (other things untouched). Returns (data, kept, total)."""
    out, kept, total = bytearray(), 0, 0
    for off in range(0, len(data), 10):
        rec = data[off : off + 10]
        _x, _y, _a, typ, _flags = struct.unpack("<hhHHH", rec)
        if typ in MONSTERS:
            total += 1
            if rng.random() >= keep:
                continue
            kept += 1
        out += rec
    return bytes(out), kept, total


def write_map(lumps_all, name: str, out: Path, keep: float = 1.0, seed: int = 0) -> dict:
    rng = random.Random(seed)
    ml = map_lumps(lumps_all, name)
    kept = total = 0
    new = []
    for l in ml:
        if l.name == "THINGS" and keep < 1.0:
            data, kept, total = thin_things(l.data, keep, rng)
            new.append(Lump("THINGS", data))
        else:
            new.append(l)
    if keep >= 1.0:
        _, kept, total = thin_things(next(l.data for l in ml if l.name == "THINGS"), 1.0, rng)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_wad(out, [Lump("MAP01", b"")] + new)
    return {"wad": str(out), "map": "MAP01", "source": name, "monsters_kept": kept, "monsters_total": total,
            "episode_ticks": REAL_TICKS}


def build_training(keep: float = 0.15, variants: int = 2) -> list[dict]:
    """Training copies of the non-held-out levels, `variants` random monster subsets each."""
    lumps = read_wad(iwad_path())
    metas = []
    for name in TRAIN_REAL:
        for v in range(variants):
            metas.append(write_map(lumps, name, REAL_DIR / f"{name}_k{int(keep * 100)}_v{v}.wad", keep, seed=int(name[3:]) * 100 + v))
    return metas


def build_heldout() -> list[dict]:
    lumps = read_wad(iwad_path())
    metas = []
    for name in HELDOUT_REAL:
        m = write_map(lumps, name, HELDOUT_REAL_DIR / f"real_{name}.wad", 1.0)
        m["wad"] = Path(m["wad"]).name
        m["stage"] = 5
        metas.append(m)
    (HELDOUT_REAL_DIR / "manifest.json").write_text(json.dumps(metas, indent=1))
    return metas


if __name__ == "__main__":
    h = build_heldout()
    print("held-out real:", [(m["source"], m["monsters_total"]) for m in h])
    t = build_training()
    print(f"training real maps: {len(t)}", [(m["source"], m["monsters_kept"], m["monsters_total"]) for m in t[:6]], "...")
