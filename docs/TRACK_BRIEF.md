# Track 1 brief: "Can It Play DOOM?" (Yugaantar DevForge), condensed from the official PDF

## Mission
Train an agent that plays DOOM (ViZDoom, Freedoom 2 IWAD, skill 3) **zero-shot on unseen maps**:
survive ambushes, navigate multi-room layouts, handle doors and keycards, reach the exit. No dataset.
Ego-centric spatial data is recommended over raw pixels (rays + hit types, depth, vitals, visible entities, low-res RGB).

## Core objectives
- **Generalise** across maps and unseen layouts (no overfitting to one scenario).
- **Partial observability:** no god-view; only the field of view and local sensors.
- **Long-horizon progression:** find the keycard in a branch, remember the locked door, backtrack to the exit.
- **Tactical combat & survival:** health, ammo, momentum, dodging projectiles, prioritising threats.

## Deliverables (judged areas)
- **Sensory & state representation:** encoders/preprocessing on the fixed observation dict (no global coordinates).
- **Model architecture & memory:** custom topology; *how the agent keeps spatial memory across rooms* and
  *handles variable numbers of enemies*.
- **Training & exploration pipeline:** own RL, reward shaping, exploration mechanisms that avoid getting stuck.
- Public GitHub repo from the template, ≤ 1 GB incl. LFS, weights in LFS; `submission/agent.py` DoomAgent
  (`__init__` loads once, `reset` clears memory, `act` every 4 ticks); no Dockerfile; optional pinned
  `submission/requirements.txt`; `python validate.py` must pass; **2-page PDF brief** (USP, neural
  architecture, spatial memory design, training/exploration pipeline).
- **Innovation is heavily rewarded:** bespoke neural modules, custom spatial memory, intrinsic curiosity,
  hierarchical control. Default SB3 PPO scores poorly.

## Evaluation & scoring
- **Hidden map suite**, same unseen maps for all, 4 tiers of increasing difficulty *and weight*:
  T1 combat & survival, T2 multi-room exploration, T3 keycard → locked door → exit, T4 full gauntlet.
  Fixed tick cap per map (~2,100–4,200) and fixed seed.
- **Non-binary per-map points:** exit (largest, + small bonus for remaining health and time);
  progression (keycards, doors opened, locked doors unlocked); exploration (unique areas, tracked
  privately by the scorer); combat (kills, minus small penalties for damage taken and dying early).
  Leaderboard = tier-weighted sum over all maps.
- Hardware: RTX 3090 sandbox; per agent 2 GB VRAM, 8 GB RAM, 1 CPU thread, no network, repo read-only.
- Timeouts: init 60 s, reset 10 s, first act 10 s, other act 5 s, ~40 ms average per decision per map.
  Crash / timeout / invalid action → player stands still for the rest of the map (points kept, penalties apply).
- **Advancement:** automated ranking → **Top 10** → code verification + **desk review with live demo and
  technical Q&A** → **Top 2** → Round 3 final judging, where innovative custom architectures are rewarded.

## Ground rules
- Decisions from trained neural parameters; hard-coded if/else trees, A*/Dijkstra, atan2 aimbots discouraged.
- Engine/maps/game variables may be modified **for training only**; evaluation = unmodified engine, vanilla rules.
- Repo created during the hackathon; libraries OK, core architecture and training built during the event;
  training code stays in the repo (judges read it).
- Fair play: only the given observations; no access to engine/maps/network/host; don't exhaust shared resources.

## Organiser hints (message to Track 1 teams, 2026-11-27)
Research topics they named, and where we stand:
- Core RL: PPO, actor-critic, recurrent PPO for memory → **done** (training/ppo.py, GRU policy).
- Exploration & rewards: reward shaping, curriculum learning, intrinsic curiosity (RND / count-based) →
  shaping + curriculum **done**; **count-based** novelty bonus (new 64-unit cell) done; RND not implemented.
- Hard maps (keycards/doors): behavioural cloning / imitation → PPO fine-tuning → **not done** (possible next step:
  record the scripted solver from tests/test_mapgen.py as a teacher on stage-3/4 maps, BC-pretrain, then PPO).
- Inputs & actions: action masking, ego-centric occupancy grids → **both done** (model.apply_action_mask,
  submission/spatial_memory.py).
- Training speed & generalisation: vectorised envs, procedural WAD maps / Freedoom → **done** (64 envs,
  training/mapgen.py, training/realmaps.py).
Earlier organiser checklist: real weights via LFS, env/ + validate.py synced to the template, 2 GB VRAM / 8 GB RAM /
1 CPU, inference self-contained in submission/, 6 hidden maps across 4 tiers; quirks: USE fires on a fresh press,
killing all enemies doesn't end a map, holding pistol fire spreads shots.
