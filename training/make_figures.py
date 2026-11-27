"""Figures for README.md and brief.pdf (written to docs/media/).

    python -m training.make_figures

Reads the held-out logs (checkpoints/*/heldout_log.jsonl, if present) and the fair-test results in
docs/results/*.json; everything else is hard-coded from those results.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "media"
RES = ROOT / "docs" / "results"
RED, INK, GREY, GREEN, BLUE, AMBER = "#c0392b", "#1f2937", "#eef0f3", "#27ae60", "#2563eb", "#e67e22"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", OUT / name)


def architecture():
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 44)
    ax.axis("off")

    def box(x, y, w, h, title, sub, fc=GREY, ec=INK):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.5", fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=10.5, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=8.3, color="#4b5563")

    def arrow(x1, y1, x2, y2, c=INK):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, lw=1.3, color=c))

    rows = [(34, "64 distance rays", "walls, doors, monsters, items"), (24, "Visible things", "up to 32 monsters / items"),
            (14, "Player status", "health, ammo, keys, speed"), (4, "Self-drawn map", "where it has been and what it saw")]
    for y, t, s in rows:
        box(1, y, 24, 8, t, s, fc="#fdecea" if "map" in t else GREY, ec=RED if "map" in t else INK)
        arrow(26, y + 4, 33, 22)
    box(34, 15.5, 17, 13, "Encoders", "small neural layers\nfor each input")
    arrow(52, 22, 57, 22)
    box(58, 15.5, 15, 13, "Memory (GRU)", "remembers the\nlast few seconds", fc="#fdecea", ec=RED)
    arrow(74, 22, 79, 30)
    arrow(74, 22, 79, 13)
    box(80, 25, 19, 11, "Buttons", "move, turn, shoot,\nopen doors")
    box(80, 7, 19, 11, "Value", "how good is this\nsituation (training)")
    ax.text(50, 42.5, "SlayNet: one small neural network (1.0 M weights, about 2 ms per decision)",
            ha="center", fontsize=11, color=RED, fontweight="bold")
    save(fig, "architecture.png")


def load_logs():
    rows = {}
    for r in ("cur1", "aim1", "mem1", "mem2", "mem3"):
        p = ROOT / "checkpoints" / r / "heldout_log.jsonl"
        if p.is_file():
            for line in p.read_text().splitlines():
                if line.strip():
                    d = json.loads(line)
                    rows[d["steps"]] = d
    return [rows[k] for k in sorted(rows)]


def training_curve():
    log = load_logs()
    if not log:
        print("no held-out logs found; skipping training curve")
        return
    steps = [r["steps"] / 1e6 for r in log]
    ex = [100 * r["exit_rate"] for r in log]
    k = 6
    sm = [sum(ex[max(0, i - k + 1): i + 1]) / len(ex[max(0, i - k + 1): i + 1]) for i in range(len(ex))]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(steps, ex, color=RED, alpha=0.25, lw=1)
    ax.plot(steps, sm, color=RED, lw=2.4, label="levels finished on unseen test maps (rolling average)")
    for x, label, y in [(5.5, "curriculum done", 94), (10.4, "spatial memory\n+ real levels", 94),
                        (24.5, "aim + weapon\nmasks", 94), (27.2, "4-agent review\nplan", 22)]:
        ax.axvline(x, color=GREY, lw=1, zorder=0)
        ax.text(x + 0.25, y, label, fontsize=8, color="#4b5563", va="top")
    ax.set_ylim(0, 100)
    ax.set_xlabel("practice (millions of game steps)")
    ax.set_ylabel("% of test levels finished")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    save(fig, "training_curve.png")


def results_by_tier():
    tiers = ["Combat", "Multi-room", "Keycard", "Gauntlet"]
    before = [100, 50, 33, 17]  # update 1800 (evening baseline, 6 seeds)
    final = [100, 96, 50, 17]  # update 4800 (12 seeds)
    fin = RES / "results_final.json"
    if fin.is_file():
        bs = json.loads(fin.read_text())["final"]["by_stage"]
        final = [round(100 * bs[str(i)]["exit_rate"]) for i in range(1, 5)]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    x = range(len(tiers))
    ax.bar([i - 0.2 for i in x], before, width=0.4, color="#cbd5e1", label="early version")
    ax.bar([i + 0.2 for i in x], final, width=0.4, color=RED, label="final version")
    for i, v in enumerate(final):
        ax.text(i + 0.2, v + 2, f"{v}%", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(list(x), tiers)
    ax.set_ylim(0, 112)
    ax.set_ylabel("% levels finished")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_title("Unseen test levels, by type", fontsize=11)
    save(fig, "results_by_tier.png")


def behaviour_fix():
    labels = ["Shots fired at empty air", "Time holding the fist"]
    before = [88, 44]
    after = [0, 5]
    fig, ax = plt.subplots(figsize=(7.5, 2.6))
    y = range(len(labels))
    ax.barh([i + 0.2 for i in y], before, height=0.4, color="#cbd5e1", label="before the fix")
    ax.barh([i - 0.2 for i in y], after, height=0.4, color=GREEN, label="after (final version)")
    for i, (b, a) in enumerate(zip(before, after)):
        ax.text(b + 1, i + 0.2, f"{b}%", va="center", fontsize=9)
        ax.text(a + 1, i - 0.2, f"{a}%", va="center", fontsize=9, fontweight="bold")
    ax.set_yticks(list(y), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title("Fixing 'shooting and punching the air'  (kills per level: 0.9 before, 3.3 after)", fontsize=10.5)
    save(fig, "behaviour_fix.png")


def route_maps():
    src = ROOT / "videos" / "final_ai"
    for name in ("heldout_s3_5", "heldout_s2_3", "heldout_s4_6"):
        p = src / f"{name}_path.png"
        if p.is_file():
            shutil.copy(p, OUT / f"route_{name}.png")
            print("copied", p.name)


if __name__ == "__main__":
    architecture()
    training_curve()
    results_by_tier()
    behaviour_fix()
    route_maps()
