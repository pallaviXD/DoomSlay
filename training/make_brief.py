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
    ax.set_ylim(0, 42)
    ax.axis("off")

    def box(x, y, w, h, text, fc=SOFT, ec=INK, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.2, color=INK,
                fontweight="bold" if bold else "normal")

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8, lw=0.9, color=INK))

    rows = [
        (32, "rays\n64 x [dist, near, 6 hit types]", "circular Conv1d x3\n-> 256"),
        (22, "visible entities\n<=32 x [5 feats + type emb]", "entity MLP + masked\nattention pool -> 193"),
        (12, "vitals (26, normalised)\n+ previous action (15)", "MLP -> 64"),
        (2, "spatial memory\n(self-built map, 5x21x21)", "CNN x2 -> 64"),
    ]
    for y, inp, enc in rows:
        box(1, y, 25, 7.5, inp, fc="#fee2e2" if y == 2 else SOFT, ec=ACCENT if y == 2 else INK, bold=y == 2)
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
    ax.text(50, 40.5, "SlayNet (1.0 M params, ~2 ms / act on 1 CPU thread)", ha="center", fontsize=8, color=ACCENT,
            fontweight="bold")
    return fig_bytes(fig)


def curve_figure(log: list[dict]) -> io.BytesIO:
    fig, ax1 = plt.subplots(figsize=(7.6, 2.3))
    steps = [r["steps"] / 1e6 for r in log]
    scores = [r["score"] for r in log]
    ax1.plot(steps, scores, color=ACCENT, lw=0.8, alpha=0.35)
    k = 5  # rolling mean over 5 evaluations: single held-out evals swing by about +-8 points
    smooth = [sum(scores[max(0, i - k + 1): i + 1]) / len(scores[max(0, i - k + 1): i + 1]) for i in range(len(scores))]
    ax1.plot(steps, smooth, color=ACCENT, lw=1.8, label="held-out approx score (rolling mean of 5)")
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
def load_logs(paths: list[Path]) -> list[dict]:
    """Held-out logs of consecutive runs (cur1 -> aim1 -> mem1), merged and ordered by environment steps."""
    rows = {}
    for p in paths:
        if p.is_file():
            for line in p.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    rows[r["steps"]] = r  # a resumed run may repeat a step count: keep the latest
    return [rows[k] for k in sorted(rows)]


def load_fair(spec: list[tuple[str, list[tuple[Path, str]]]]) -> list[tuple[str, dict]]:
    """Average multi-seed comparison results (training/compare.py outputs) per version."""
    out = []
    for label, parts in spec:
        rs = []
        for path, key in parts:
            if path.is_file():
                d = json.loads(path.read_text())
                if key in d:
                    rs.append(d[key])
        if rs:
            avg = {k: sum(r[k] for r in rs) / len(rs) for k in ("score", "exit_rate", "keys", "doors", "deaths")}
            avg["by_stage"] = rs[-1].get("by_stage", {})
            out.append((label, avg))
    return out


def build(log_paths: list[Path], fair_spec, out: Path) -> None:
    log = load_logs(log_paths)
    fair = load_fair(fair_spec)
    globals()["fair_rows"] = fair
    last = fair[-1][1] if fair else None
    tier_text = ""
    if last and last.get("by_stage"):
        names = {"1": "combat", "2": "multi-room", "3": "keycard", "4": "gauntlet"}
        tier_text = "Final version, levels finished per tier: " + ", ".join(
            f"{names.get(str(k), k)} {100 * v['exit_rate']:.0f}%" for k, v in sorted(last["by_stage"].items())) + "."
    globals()["tier_text"] = tier_text

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
        "A compact (1.0 M parameter) policy that sees the world like a bat, not a camera: 64 ego-centric rays, "
        "a variable-size set of visible entities, vitals, and a <b>map it draws itself</b> from what it has seen "
        "(dead-reckoned spatial memory). It is trained from scratch with <b>our own recurrent PPO</b> on an endless "
        "stream of <b>procedurally generated maps</b> (combat, exploration, keycard → locked door → exit, gauntlet; "
        "with exit switches and stairs) mixed with <b>real hand-made Freedoom levels</b>, so it can only score by "
        "learning <i>transferable behaviour</i>, never by memorising layouts. Every action comes from network "
        "weights; there are no rules, planners or aimbots.")]
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
        "stored hidden state (truncated BPTT). On top of the GRU, an <b>ego-centric occupancy grid</b> "
        "(submission/spatial_memory.py) draws a 160×160 occupancy map (64 units per cell) in a dead-reckoned frame: "
        "heading from the agent's own TURN presses (DOOM's keyboard turn rate is deterministic: 1.76°/tic for the "
        "first 5 tics, then 3.52°/tic), translation from the vitals velocity, and the 64 rays paint free space, walls, "
        "doors and locked doors. Each step a 5×21×21 crop, rotated so 'up' is the facing direction, goes through a "
        "small CNN into the fusion layer. It answers 'have I been here?' and 'where was that locked door?' without "
        "any global coordinates. It was added to a trained network by <b>warm start</b>: new fusion columns start "
        "at zero, so behaviour is unchanged at first (unit-tested exactly), and PPO then learns to use the map.")]

    st.append(PageBreak())
    st += [P("Training and exploration pipeline", h2), P(
        "<b>Maps:</b> a grid-based UDMF generator (training/mapgen.py) builds a spanning-tree room graph with extra "
        "loops, corridors, doors, enemies, health and ammo; in stage 3/4 the locked door sits on a bridge of the room "
        "graph with its keycard in a side branch. Training-only options add <b>exit switches</b> (press USE) and "
        "<b>stairs</b> between rooms of different height, and 4–8-monster <b>combat arenas</b>. A scripted check "
        "solved 100/100 maps. <b>Real levels:</b> 28 Freedoom levels (monsters thinned for training) give hand-made "
        "geometry: lifts, odd shapes, real exits; 4 others are a held-out real test. <b>Curriculum:</b> stage 1 → 2 → "
        "3 → 4 on rolling exit rate; the final mix is 30% real levels, 25% combat arenas, the rest generated stages "
        "with 25% replay; map pools regenerate every 1,500 episodes. <b>Reward</b> (training only): exit +10 (+ time "
        "and health bonus), keycard +3, door +2 (+4 locked), kill +1, damage dealt +0.01/HP, ammo spent −0.02, "
        "<b>potential-based aim shaping</b> (0.3·cos to the nearest visible monster, policy-invariant), a "
        "<b>count-based intrinsic curiosity</b> bonus (+0.05 per new 64-unit cell), damage taken −0.01/HP, death −3, "
        "stuck −1 with cut-off. <b>Action masking</b> from observation fields: fire only with a visible monster and "
        "tap-fire (holding pistol fire spreads shots), USE only within reach. <b>Recurrent PPO "
        "(ours):</b> GAE(0.99, 0.95), clip 0.2, clipped value loss, 4 epochs, 128-step rollouts × 64 envs, 32-step "
        "truncated BPTT, entropy 0.01, lr 3e-4 → 1e-4; a CPU actor runs rollouts and the GPU learns. Checkpoints "
        "are chosen by <b>held-out</b> score, confirmed by a 6-seed test and by watching replays.")]
    st += [P("Held-out results", h2), Image(curve_figure(log), width=180 * mm, height=50 * mm)]
    rows = [["version (6 seeds x 8 held-out maps)", "approx score", "levels finished", "keys/map", "doors/map", "deaths/map"],
            ["untrained network", "8.6", "0%", "0.00", "0.00", "0.62"]]
    for label, r in fair_rows:
        rows.append([label, f"{r['score']:.1f}", f"{100 * r['exit_rate']:.0f}%", f"{r['keys']:.2f}",
                     f"{r['doors']:.2f}", f"{r['deaths']:.2f}"])
    t = Table(rows, hAlign="LEFT", colWidths=[62 * mm] + [22 * mm] * 5)
    t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 7.5), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(SOFT)),
                           ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor(INK)), ("TOPPADDING", (0, 0), (-1, -1), 1.5),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]))
    st += [t, Spacer(1, 3), P(
        "Held-out = 8 generated maps (2 per tier) from a reserved seed range, played through the real DoomAgent "
        "interface, 6 seeds each. The approximate score uses our own guess at the hidden weights (exit 50 + efficiency, "
        "key 8, door 4, exploration ≤ 15, kills ≤ 10, death −5), weighted by tier. " + tier_text +
        " Per-step latency is about 2 ms against the 40 ms budget. We also inspect every candidate by replay "
        "(videos, top-down paths, behaviour statistics such as 'fires when a monster is visible' vs 'not visible').")]
    st += [P("Limitations", h2), P(
        "Aiming is the weakest skill, so the agent still dies on crowded tier-4 maps. Next steps: behavioural cloning "
        "from our scripted keycard solver followed by PPO fine-tuning, and RND curiosity. Full-size Freedoom levels "
        "with all monsters are still unsolved (0/8 exits on the "
        "real held-out set), though they are far larger than a 2–4k-tic evaluation map. Dead reckoning drifts over "
        "long episodes (the map stays locally consistent). Pixels are not used. Training ran on one laptop "
        "(~1.3k steps/s, about " + f"{log[-1]['steps'] / 1e6:.0f}" + " M steps in total).")]

    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=14 * mm, rightMargin=14 * mm, topMargin=11 * mm,
                            bottomMargin=10 * mm, title="Team Slay brief", author="Team Slay")
    doc.build(st)


FAIR_SPEC = [  # (label, [(comparison json, checkpoint key), ...]) averaged; seeds 11-66 in total
    ("baseline curriculum, update 1800 (5.5 M)", [(ROOT / "docs" / "compare.json", "best_upd1800"),
                                                    (ROOT / "docs" / "compare2.json", "best_upd1800")]),
    ("+ hit/ammo/door rewards, update 2050 (6.3 M)", [(ROOT / "docs" / "compare.json", "snap_1822"),
                                                        (ROOT / "docs" / "compare2.json", "snap_1822")]),
    ("+ spatial memory, real levels, update 2800", [(ROOT / "docs" / "compare_mem2800.json", "best_upd2800")]),
    ("+ stronger aim shaping, update 3000 (14.4 M)", [(ROOT / "docs" / "compare_mem3000.json", "best_upd3000")]),
    ("final submission", [(ROOT / "docs" / "results_final.json", "final")]),
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "brief.pdf"))
    a = ap.parse_args()
    logs = [ROOT / "checkpoints" / r / "heldout_log.jsonl" for r in ("cur1", "aim1", "mem1")]
    build(logs, FAIR_SPEC, Path(a.out))
    print("wrote", a.out)
