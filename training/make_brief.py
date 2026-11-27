"""Build brief.pdf: a 2-page, plain-language project brief for the judges.

    uv run --no-project --python 3.13 --with reportlab --with pypdf python training/make_brief.py --out brief.pdf

Uses the figures in docs/media/ (made by `python -m training.make_figures` and the gameplay video)
and the final numbers in docs/results/results_final.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "docs" / "media"
RED, INK, SOFT, MUTED = colors.HexColor("#b91c1c"), colors.HexColor("#1f2937"), colors.HexColor("#f3f4f6"), colors.HexColor("#6b7280")
W = A4[0] - 24 * mm  # usable width

H1 = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=20, leading=23, textColor=RED)
SUB = ParagraphStyle("sub", fontName="Helvetica", fontSize=9.5, leading=12, textColor=MUTED)
H2 = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=11.5, leading=14, textColor=RED, spaceBefore=5, spaceAfter=2)
BODY = ParagraphStyle("body", fontName="Helvetica", fontSize=8.8, leading=11.4, textColor=INK)
BUL = ParagraphStyle("bul", parent=BODY, leftIndent=9, bulletIndent=0)
CAP = ParagraphStyle("cap", parent=BODY, fontSize=7.4, leading=9, textColor=MUTED, alignment=1)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8.2, leading=10)
CELLB = ParagraphStyle("cellb", parent=CELL, fontName="Helvetica-Bold")


def img(name, width):
    p = MEDIA / name
    from reportlab.lib.utils import ImageReader

    iw, ih = ImageReader(str(p)).getSize()
    return Image(str(p), width=width, height=width * ih / iw)


def bullets(items):
    return [Paragraph(t, BUL, bulletText="•") for t in items]


def table(rows, widths, header=True):
    data = [[Paragraph(str(c), CELLB if (header and i == 0) or j == 0 else CELL) for j, c in enumerate(r)]
            for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths)
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d1d5db")),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
             ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), SOFT))
    t.setStyle(TableStyle(style))
    return t


def side(left, right, lw):
    t = Table([[left, right]], colWidths=[lw, W - lw])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                           ("RIGHTPADDING", (0, 0), (0, -1), 8), ("RIGHTPADDING", (1, 0), (1, -1), 0)]))
    return t


def page_one(f):
    s = [Paragraph("DoomSlay: an AI that taught itself to play DOOM", H1),
         Paragraph("Team Slay · Yugaantar 2026 DevForge · Track 1 “Can It Play DOOM?”", SUB),
         Spacer(1, 5)]
    shots = Table([[img("shot_gauntlet.png", W / 3 - 4), img("shot_demons.png", W / 3 - 4), img("shot_keycard.png", W / 3 - 4)],
                   [Paragraph("Fighting a group with the shotgun", CAP), Paragraph("Shooting two charging demons", CAP),
                    Paragraph("Keycard level, heading for the exit", CAP)]], colWidths=[W / 3] * 3)
    shots.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                               ("TOPPADDING", (0, 0), (-1, -1), 1), ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))
    s += [shots, Paragraph("What we built", H2),
          Paragraph("An AI player for DOOM that finds its way through levels it has <b>never seen before</b>: it fights "
                    "monsters, picks up keycards, opens locked doors and reaches the exit. Nobody programmed these moves. "
                    "The AI taught itself by playing thousands of practice levels and getting points for good results "
                    "(this is called <b>reinforcement learning</b>). Every button press comes from the trained neural "
                    f"network: no scripts, no pathfinding, no aimbot. It finishes <b>{f['exit']}%</b> of unseen test levels "
                    "and every run on the hand-made template map.", BODY),
          Paragraph("How the AI sees, thinks and remembers", H2),
          img("architecture.png", W * 0.92),
          Spacer(1, 2)]
    s += bullets([
        "<b>Sees:</b> no screen pixels and no map. It gets 64 distance “feelers” around it (like a bat), a list "
        "of up to 32 visible monsters and items, and its health, ammo and keys.",
        "<b>Thinks:</b> one small network (SlayNet, 1.0 million weights) turns this into 6 choices every step: move, "
        "sidestep, turn, shoot, use and switch weapon. A memory unit (GRU) remembers the last few seconds.",
        "<b>Remembers places:</b> the AI draws its own map. It tracks its position from its own speed and turns, marks "
        "where it walked and where it saw walls, doors and locked doors, and looks at that map every step. This answers "
        "“have I been here?” and “where was that locked door?”.",
        "<b>Button rules (allowed by the contract):</b> shooting is allowed only when a monster is near the crosshair, "
        "weapon switching only when stuck on the fist, and the use key only near a door. The network still makes every choice.",
    ])
    s += [Paragraph("How it learned", H2)]
    s += bullets([
        "<b>Practice levels:</b> our own map generator creates endless random levels in 4 types (combat, multi-room, "
        "keycard + locked door, gauntlet) with doors, keys, exit switches and stairs. 27 real Freedoom levels add "
        "hand-made shapes. 8 generated and 4 real levels are kept aside <b>only for testing</b>.",
        "<b>Step by step:</b> easy levels first, harder ones once it improves (curriculum). New maps every 1,500 games so "
        "it cannot memorise them.",
        "<b>Points:</b> big reward for the exit (more when fast and healthy), plus keycards, doors, kills and exploring "
        "new areas; small penalties for damage, dying, wasted ammo and getting stuck.",
        "<b>Method:</b> our own recurrent PPO written from scratch in PyTorch, with 64 games running in parallel on one laptop.",
    ])
    return s


def page_two(f):
    res_rows = [["Level type", "Finished"], ["Combat", f"{f['s'][0]}%"], ["Multi-room", f"{f['s'][1]}%"],
                ["Keycard, locked door, exit", f"{f['s'][2]}%"], ["Gauntlet (many monsters)", f"{f['s'][3]}%"],
                ["All unseen levels (96 games)", f"<b>{f['exit']}%</b>"], ["Hand-made template map", "<b>16 / 16</b>"],
                ["Kills per level", f"{f['kills']}"], ["Deaths per level", f"{f['deaths']}"],
                ["Shots at empty air", "88% → 0%"], ["Time per decision", "~2 ms (limit 40 ms)"]]
    s = [Paragraph("Results on levels it never practised on", H2),
         side(table(res_rows, [W * 0.27, W * 0.17]), img("results_by_tier.png", W * 0.54), W * 0.45),
         Spacer(1, 3), img("training_curve.png", W * 0.88),
         Paragraph("Levels finished on the unseen test maps while it practised. We kept a new version only if it beat the "
                   "old one on the unseen levels <b>and</b> on the hand-made map, so a lucky score could not fool us.", CAP),
         Paragraph("Tech stack", H2)]
    tech = [["Part", "What we used"],
            ["Game", "ViZDoom 1.3.3 with Freedoom 2, through the official doom_env wrapper (Gymnasium API)"],
            ["AI and learning", "Python 3.13, PyTorch 2.15, our own recurrent PPO; GRU memory, attention, small CNNs"],
            ["Practice maps", "Own UDMF map generator + real Freedoom levels"],
            ["Speed", "64 games in parallel (multiprocessing), CPU plays, GPU learns"],
            ["Tools", "NumPy, TensorBoard, Matplotlib, ImageIO + FFmpeg (videos), ReportLab (this PDF), uv, Git LFS"],
            ["Hardware", "One laptop: Intel i7-14700HX, RTX 5060 Laptop GPU, 16 GB RAM"]]
    s += [table(tech, [W * 0.18, W * 0.82])]
    rules = [["Rule", "How we follow it"],
             ["Actions from trained weights", "Every button comes from the network; masks only remove pointless options"],
             ["No pathfinding or aimbot", "No A*, no hand-written turning or aiming; it learned to aim by itself"],
             ["Own RL code", "PPO written from scratch (training/ppo.py), not Stable-Baselines3"],
             ["Limits", "1 CPU thread, no GPU needed, about 2 ms per decision, 4 MB weights in Git LFS"],
             ["Only allowed inputs", "Only the observation it is given; its map is built from those and its own moves"]]
    s += [Paragraph("Following the rules", H2), table(rules, [W * 0.27, W * 0.73]),
          Paragraph("Honest limits and next steps", H2)]
    s += bullets([
        "Gauntlet levels (crowds of monsters plus a long route in 2 minutes) are the hardest, and keycard levels "
        "sometimes run out of time after the key. Big real campaign levels are not solved yet.",
        "Tried and dropped: copying a scripted player first (fewer exits), mixing two versions, less random choices.",
        "Next: longer training, a curiosity reward for exploring, and learning from a teacher that moves like the agent.",
    ])
    s += [Spacer(1, 3), Paragraph("Code, videos and full details: README.md in the repository.", CAP)]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", default=str(ROOT / "docs" / "results" / "results_final.json"))
    ap.add_argument("--out", default=str(ROOT / "brief.pdf"))
    a = ap.parse_args()
    r = json.loads(Path(a.final).read_text())["final"]
    f = {"exit": round(100 * r["exit_rate"]), "kills": f"{r['kills']:.2f}", "deaths": f"{r['deaths']:.2f}",
         "s": [round(100 * r["by_stage"][str(i)]["exit_rate"]) for i in range(1, 5)]}
    doc = SimpleDocTemplate(a.out, pagesize=A4, leftMargin=12 * mm, rightMargin=12 * mm, topMargin=10 * mm,
                            bottomMargin=9 * mm, title="DoomSlay brief", author="Team Slay")
    doc.build(page_one(f) + [PageBreak()] + page_two(f))
    from pypdf import PdfReader

    n = len(PdfReader(a.out).pages)
    print(f"wrote {a.out}: {n} pages")
    if n != 2:
        raise SystemExit("brief must be exactly 2 pages")


if __name__ == "__main__":
    main()
