"""Build docs/brief.pdf (exactly 2 pages) from the held-out log and final eval results.

    uv run --no-project --with reportlab --with matplotlib python training/make_brief.py \
        --log checkpoints/cur1/heldout_log.jsonl --final docs/results_final.json

Figures (architecture diagram, held-out curve) are drawn with matplotlib; layout is reportlab.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
INK, ACCENT, SOFT = "#1f2937", "#b91c1c", "#f3f4f6"


# ----------------------------------------------------------------------------- figures
def fig_bytes(fig) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def architecture_figure() -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(7.6, 2.9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")

    def box(x, y, w, h, text, fc=SOFT, ec=INK, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.2, color=INK,
                fontweight="bold" if bold else "normal")

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8, lw=0.9, color=INK))

    rows = [
        (31, "rays\n64 x [dist, near, 6 hit types]", "circular Conv1d x3\n-> 256"),
        (18, "visible entities\n<=32 x [5 feats + type emb]", "entity MLP + masked\nattention pool -> 193"),
        (5, "vitals (26, normalised)\n+ previous action (15)", "MLP -> 64"),
    ]
    for y, inp, enc in rows:
        box(1, y, 25, 7.5, inp)
        box(31, y, 21, 7.5, enc)
        arrow(26.6, y + 3.75, 30.4, y + 3.75)
        arrow(52.6, y + 3.75, 57.4, 21.5)
    box(58, 17, 11, 9, "fuse\nLinear 256", bold=True)
    box(73, 17, 10, 9, "GRU 256\n(memory)", fc="#fee2e2", ec=ACCENT, bold=True)
    arrow(69.6, 21.5, 72.4, 21.5)
    ax.add_patch(FancyArrowPatch((81, 26.6), (75, 26.6), connectionstyle="arc3,rad=1.6", arrowstyle="-|>",
                                 mutation_scale=7, lw=0.8, color=ACCENT))
    box(87, 24, 12.5, 10, "actor\n6 categorical\nheads")
    box(87, 8, 12.5, 7, "critic\nV(s)")
    arrow(83.6, 23, 86.4, 28)
    arrow(83.6, 20, 86.4, 12)
    ax.text(50, 39, "SlayNet (0.9 M params, ~2 ms / act on 1 CPU thread)", ha="center", fontsize=8, color=ACCENT,
            fontweight="bold")
    return fig_bytes(fig)


def curve_figure(log: list[dict]) -> io.BytesIO:
    fig, ax1 = plt.subplots(figsize=(7.6, 2.3))
    steps = [r["steps"] / 1e6 for r in log]
    ax1.plot(steps, [r["score"] for r in log], color=ACCENT, lw=1.6, label="held-out approx score")
    ax1.set_xlabel("environment steps (millions)", fontsize=7)
    ax1.set_ylabel("approx score", fontsize=7, color=ACCENT)
    ax1.tick_params(labelsize=6.5)
    ax2 = ax1.twinx()
    ax2.plot(steps, [r["exit_rate"] for r in log], color=INK, lw=1, ls="--", label="exit rate")
    ax2.plot(steps, [r["keys"] for r in log], color="#2563eb", lw=1, ls=":", label="keys / map")
    ax2.set_ylabel("rate", fontsize=7)
    ax2.tick_params(labelsize=6.5)
    for s in ("top",):
        ax1.spines[s].set_visible(False)
        ax2.spines[s].set_visible(False)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=6.5, loc="upper left", frameon=False)
    ax1.set_title("Held-out maps (never trained on) during training", fontsize=8)
    return fig_bytes(fig)


# ----------------------------------------------------------------------------- document
def build(log_path: Path, final_path: Path | None, out: Path) -> None:
    log = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
    final = json.loads(final_path.read_text()) if final_path and final_path.is_file() else None
    best = max(log, key=lambda r: r["score"])

    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["Normal"], fontSize=8.4, leading=10.6, spaceAfter=3)
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=15, leading=18, spaceAfter=2, textColor=colors.HexColor(INK))
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=10, leading=12, spaceBefore=5, spaceAfter=2,
                        textColor=colors.HexColor(ACCENT))
    sub = ParagraphStyle("s", parent=body, alignment=1, textColor=colors.HexColor("#4b5563"))
    P = lambda t, s=body: Paragraph(t, s)  # noqa: E731

    st = []
    st += [P("Team Slay: a recurrent PPO agent that plays unseen DOOM maps", h1),
           P("Yugaantar 2026 · DevForge Track 1 \"Can It Play DOOM?\" · github.com/pallaviXD/DoomSlay", sub)]
    st += [P("USP", h2), P(
        "A compact (0.9 M parameter) policy that sees the world like a bat, not a camera: 64 ego-centric rays, "
        "a variable-size set of visible entities and vitals. It is trained from scratch with <b>our own recurrent "
        "PPO</b> on <b>an endless stream of procedurally generated maps</b> (combat, multi-room exploration, "
        "keycard → locked door → exit, gauntlet), so it can only score by learning <i>transferable behaviour</i>, "
        "never by memorising layouts. Every action comes from network weights; there are no rules, planners or aimbots.")]
    st += [P("Architecture", h2), Image(architecture_figure(), width=180 * mm, height=68 * mm)]
    st += [P("Variable-entity handling", h2), P(
        "Between 0 and 32 entities are visible. Each one (distance, sin/cos of relative angle, relative velocity, "
        "learned type embedding) goes through a shared MLP. Two learned attention queries pool the set with a "
        "<b>masked softmax</b>; padding slots get −1e9 logits and zero weight, and a masked max-pool keeps a single "
        "dangerous enemy from being averaged away. The output is permutation-invariant, ignores padding contents, and "
        "is exactly zero when nothing is visible. All three properties are unit-tested (tests/test_slay.py).")]
    st += [P("Spatial memory design", h2), P(
        "The agent observes no position or map, so a <b>GRU(256)</b> carries memory across steps: which corridors "
        "were explored, which door was locked, whether a key is held (keys are also in vitals). Rays use a "
        "<b>circular</b> 1-D convolution: the 360° ring wraps, and weights shared over heading make a doorway look "
        "the same at any angle. The previous action is an input, so the policy can learn the button-down behaviour "
        "of USE. In training the GRU is reset at every episode boundary, and PPO replays 32-step sequences from the "
        "stored hidden state (truncated BPTT). <i>Stretch goal not done:</i> an egocentric grid memory "
        "shifted by dead-reckoned motion.")]

    st.append(PageBreak())
    st += [P("Training and exploration pipeline", h2), P(
        "<b>Maps:</b> a grid-based UDMF generator (training/mapgen.py) builds a spanning-tree room graph with extra "
        "loops, 2-cell corridors, doors, enemies, health and ammo, and an exit pad. In stage 3/4 the locked door sits on "
        "a bridge of the room graph and its keycard is in a side branch. A scripted solvability check reached the exit on "
        "100/100 maps. <b>Curriculum:</b> stage 1 (1–2 rooms, weak enemies) → 2 (4–8 rooms, exit in the farthest "
        "room) → 3 (key/lock) → mixed, advancing on rolling exit rate, with 25% replay of earlier stages and a "
        "fresh map pool every 1,500 episodes. <b>Reward</b> (training only, uses privileged position): exit +10, "
        "keycard +3, door +2 (+1 locked), kill +1, new 64-unit cell +0.05, pickup when low +0.1, damage −0.01/HP, "
        "death −3, step −0.001, and stuck (no progress for 200 tics) −1 with episode cut-off. <b>PPO:</b> GAE(0.99, "
        "0.95), clip 0.2 with clipped value loss, 4 epochs, 128-step rollouts × 24 envs, entropy 0.01, grad clip 0.5, "
        "lr 3e-4 → 1e-4. 24 spawn workers; a CPU actor does rollouts and the GPU learns, with two env halves "
        "pipelined. The best checkpoint is chosen by <b>held-out</b> score, not by recency.")]
    st += [P("Held-out results", h2), Image(curve_figure(log), width=180 * mm, height=55 * mm)]
    rows = [["checkpoint", "approx score", "exit rate", "keys/map", "doors/map", "cells", "deaths"],
            ["untrained", "8.6", "0.00", "0.00", "0.00", "42", "0.62"]]
    pick = [r for r in log if r["update"] in (100, 300)] + [best]
    for r in pick:
        rows.append([f"update {r['update']} ({r['steps'] / 1e6:.1f}M)", f"{r['score']:.1f}", f"{r['exit_rate']:.2f}",
                     f"{r['keys']:.2f}", f"{r['doors']:.2f}", f"{r['cells']:.0f}", f"{r['deaths']:.2f}"])
    if final:
        rows.append(["final, 3 seeds", f"{final['score']:.1f}", f"{final['exit_rate']:.2f}", f"{final['keys']:.2f}",
                     f"{final['doors']:.2f}", f"{final['cells']:.0f}", f"{final['deaths']:.2f}"])
    t = Table(rows, hAlign="LEFT", colWidths=[42 * mm] + [22 * mm] * 6)
    t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 7.5), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(SOFT)),
                           ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor(INK)), ("TOPPADDING", (0, 0), (-1, -1), 1.5),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]))
    st += [t, Spacer(1, 3), P(
        "Held-out = 8 generated maps (2 per tier) from a reserved seed range, played through the real DoomAgent "
        "interface. The approximate score uses our own guess at the hidden weights (exit 50 + efficiency, key 8, door 4, "
        "exploration ≤ 15, kills ≤ 10, death −5), weighted by tier. Per-step latency is about 2 ms against a 40 ms budget.")]
    st += [P("Limitations", h2), P(
        "Training maps are flat and grid-aligned (no stairs, lifts, switches or skull keys, one lock), so real "
        "hand-made maps are out of distribution. Memory lives only in the GRU, so long backtracking in tier 4 is the "
        "weakest skill. Pixels are not used. The held-out set is small (8 maps), so its score is noisy (about ±5 between "
        "evaluations). Training ran on one laptop (about 1.2k steps/s) for a few hours.")]

    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=14 * mm, rightMargin=14 * mm, topMargin=11 * mm,
                            bottomMargin=10 * mm, title="Team Slay brief", author="Team Slay")
    doc.build(st)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(ROOT / "checkpoints" / "cur1" / "heldout_log.jsonl"))
    ap.add_argument("--final", default=str(ROOT / "docs" / "results_final.json"))
    ap.add_argument("--out", default=str(ROOT / "brief.pdf"))
    a = ap.parse_args()
    build(Path(a.log), Path(a.final), Path(a.out))
    print("wrote", a.out)
