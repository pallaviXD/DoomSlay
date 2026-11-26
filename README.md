# Team Slay — DoomSlay

Yugaantar 2026 · DevForge — Track 1 *"Can It Play DOOM?"*

A recurrent PPO agent (our own implementation) that plays unseen ViZDoom maps zero-shot from the
ego-centric observation only: 64 raycasts, visible entities, vitals, plus a map it draws itself
(spatial memory). It is trained on thousands of procedurally generated maps and real Freedoom levels
through a four-stage curriculum: combat → exploration → keycard/locked door → gauntlet.

## Setup

Windows 11, Linux or macOS (Apple Silicon), Python 3.13, [uv](https://docs.astral.sh/uv/):

```bash
uv sync                          # vizdoom 1.3.3, gymnasium, doom_env (editable), torch 2.15.0, tensorboard
uv run python validate.py        # official submission check
uv run pytest -q                 # wrapper, map generator and agent/PPO/reward tests
```

`uv sync` installs the `train` dependency group by default. On Windows, torch comes from the PyTorch
cu132 index (CUDA 13.2, which RTX 50xx needs); on Linux it comes from PyPI. The submission itself needs
no extra packages (`submission/requirements.txt` is empty): torch and numpy are in the official image,
and inference only imports files inside `submission/`.

> Windows note: Smart App Control blocks numpy/torch DLLs that lack reputation; it had to be turned off.

## Run the agent

```python
from doom_env import make_env
from submission.agent import DoomAgent

env = make_env("maps/heldout/heldout_s3_4.wad", episode_ticks=4200)
agent = DoomAgent(env.observation_space, env.action_space, "cpu")
obs, _ = env.reset(seed=0); agent.reset()
done = False
while not done:
    obs, r, term, trunc, info = env.step(agent.act(obs)); done = term or trunc
print(info)
```

## Train

```bash
uv run python -m training.realmaps                   # real Freedoom levels -> training copies + maps/heldout_real/
uv run python -m training.train --run smoke --smoke  # 1-minute pipeline check
uv run python -m training.train --run mem1 --stage 1 --num-envs 64 --total-steps 40e6 \
    --combat-frac 0.25 --real-frac 0.3 --variety-frac 0.5 --masks
uv run tensorboard --logdir runs                     # curves
```

Checkpoints go to `checkpoints/<run>/` (`latest.pt`, and `best.pt` = best *held-out* score), and
logs go to `runs/<run>/`. All of these are git-ignored. `--resume <ckpt> --resume-opt` continues a run;
older checkpoints warm-start into newer architectures (`submission/model.py: load_compatible`).

## Evaluate and watch

```bash
uv run python -m training.compare checkpoints/mem1/best.pt --seeds 11,22,33,44,55,66   # fair 6-seed test
uv run python -m training.compare <ckpt> --manifest maps/heldout_real/manifest.json    # real Freedoom levels
uv run python -m training.inspect_play --ckpt <ckpt> --tag check   # videos, paths, behaviour stats
uv run python -m training.watch --ckpt <ckpt>                      # watch it play live (Space/N/Esc)
uv run python -m training.play_human --stage 3                     # play a generated map yourself
uv run python -m training.export <ckpt>                            # -> submission/weights.pt
```

## Architecture

```
rays 64 x [dist, near, 6 hit-type one-hot] ----- circular Conv1d x3 -------------> 256
visible entities <=32 x [5 feats, type emb] ---- per-entity MLP + masked attn pool -> 193
vitals (26 normalised) + previous action (15) -- MLP ----------------------------> 64
spatial memory: self-built map, 5 x 21 x 21 ---- CNN x2 -------------------------> 64
          concat -> Linear 256 -> GRUCell 256 -> actor (6 categorical heads, action masks) / critic
```

| Component | Why |
|---|---|
| **Circular 1-D conv over rays** | The 64 rays form a 360° ring, so circular padding makes "behind" wrap correctly, and weights shared over heading make a doorway look the same at any angle. |
| **Masked attention pooling over entities** | 0 to 32 visible entities: a per-entity MLP, 2 learned attention queries and a masked max-pool. Permutation-invariant, ignores padding, zero when nothing is visible (unit-tested). |
| **Spatial memory** ([submission/spatial_memory.py](submission/spatial_memory.py)) | The agent draws its own occupancy map in a dead-reckoned frame. Heading comes from its own TURN presses (DOOM's keyboard turn rate is deterministic: 1.76°/tic for the first 5 tics, then 3.52°/tic) and translation from the vitals velocity; the rays paint free space, walls, doors and locked doors. A 5×21×21 crop, rotated so "up" is the facing direction, goes through a small CNN. No global coordinates are used. It was added by **warm start**: the new fusion weights start at zero, so a trained policy is unchanged at first (unit-tested exactly). |
| **GRU memory (256)** | Short-term memory across steps; reset in `reset()` and at every episode boundary in training. |
| **Factored action heads** | The 9 buttons become 6 categoricals (move, strafe, turn, attack, use, next weapon): no contradictory presses, and the previous action is an input so the policy can learn `USE`'s button-down behaviour. |
| **Action masks** (from observation fields, allowed by the rules) | ATTACK only when a monster is visible and not two steps in a row (holding pistol fire spreads shots); USE only when something is within reach ahead. The network still makes every decision; masks only remove pointless options. They are stored per checkpoint (`masks`) because they are part of how a policy was trained. The submitted checkpoint was trained without masks, so it runs without them. |
| **1.0 M parameters, CPU inference** | About 2–4 ms per `act()` on 1 thread, well inside the 40 ms budget; no VRAM used. |

## Training pipeline

- **Own recurrent PPO** ([training/ppo.py](training/ppo.py)): GAE (γ 0.99, λ 0.95), clip 0.2 with clipped value
  loss, 4 epochs, 128-step rollouts × 64 envs cut into 32-step truncated-BPTT sequences from the stored GRU state,
  entropy 0.01, grad-norm clip 0.5, learning rate 3e-4 → 1e-4.
- **Environments** ([training/envs.py](training/envs.py)): 64 spawn-based workers (the best measured throughput on a
  20-core laptop with real levels); each builds the same spatial memory as the agent. A CPU actor runs rollouts
  in two pipelined halves and the GPU learns.
- **Procedural maps** ([training/mapgen.py](training/mapgen.py)): UDMF grid maps from a spanning-tree room graph
  with loops, corridors, doors, enemies, items and an exit.
  - Stage 1: combat in 1–2 rooms.
  - Stage 2: exploration in 4–8 rooms, with the exit in the farthest room.
  - Stage 3: keycard → locked door (on the only route) → exit.
  - Stage 4: a gauntlet.
  - Training-only options: **exit switches**, **stairs** between rooms of different height, and **combat arenas**
    with 4–8 monsters. A scripted driver (test code only) solves every generated variant.
- **Real levels** ([training/realmaps.py](training/realmaps.py)): 28 Freedoom levels with thinned monsters for
  training. Four others (MAP02/08/15/31, all monsters) form a held-out real test set (`maps/heldout_real/`).
- **Curriculum** ([training/curriculum.py](training/curriculum.py)): stage 1 → 4 on rolling exit rate, then a mix
  of 30% real levels, 25% combat arenas and generated stages with 25% replay. Map pools are regenerated every
  1,500 episodes.
- **Reward shaping** ([training/rewards.py](training/rewards.py)), training only:

  | Event | Reward |
  |---|---|
  | Exit | +10, plus up to +3 for time left and +2 for health left |
  | Keycard | +3 |
  | Door opened | +2 (+4 more if it was locked) |
  | Kill | +1 |
  | Damage dealt | +0.01 per HP |
  | Aim (potential-based, policy-invariant) | 0.3 × change in cos(angle to the nearest visible monster) |
  | Fresh trigger press with a monster in the crosshair | +0.08 |
  | Ammo spent | −0.02 per unit |
  | New 64-unit cell (count-based exploration) | +0.05 |
  | Health or ammo pickup when low | +0.1 |
  | Damage taken | −0.01 per HP |
  | Death | −3 |
  | USE press | −0.003 |
  | Each step | −0.001 |
  | Stuck (no progress for 200 tics, 400 on key maps, 600 on real levels) | −1, episode ends |

- **Model selection**: held-out evaluation through the real `DoomAgent` every few minutes. Candidates are
  confirmed with a 6-seed test ([training/compare.py](training/compare.py)) and by watching replays
  ([training/inspect_play.py](training/inspect_play.py)): videos, top-down paths, and behaviour statistics such as
  "fires when a monster is visible vs not".

## Results

Held-out test: 8 generated maps (2 per tier) from a reserved seed range, never trained on, played through
the real `DoomAgent` interface ([training/compare.py](training/compare.py)). The approximate score uses our
own guess at the hidden weights, weighted by tier. Raw results are in `docs/compare*.json` and
`docs/results_final.json`.

| Version | Seeds | Approx score | Levels finished | Deaths / map |
|---|---|---|---|---|
| Untrained network | 1 | 8.6 | 0% | 0.62 |
| Baseline curriculum, update 1800 (5.5 M steps) | 6 | 44.0 | 50% | 0.42 |
| + hit / ammo / door rewards, update 2050 | 6 | 47.6 | 48% | 0.23 |
| + spatial memory + real levels, update 2800 | 6 | 46.4 | 54% | 0.29 |
| + masks (trained with them), update 4160 | 12 | 48.4 | 60% | 0.23 |
| **Final submission: + aim shaping, update 3000 (14.4 M steps)** | **12** | **52.3** | **56%** | **0.22** |

Final submission per tier (levels finished): **combat 100%, multi-room 62%, keycard → locked door 46%,
gauntlet 17%**. Average decision time 1.9 ms (budget 40 ms); `validate.py` passes.

The final checkpoint was chosen by a 12-seed comparison of the four best candidates (all within ~4 points):
it is the only one that reliably finishes tier-4 (gauntlet) maps, which carry the most weight. The mask-trained
version finishes slightly more easy and medium maps and wastes no shots in replays, but scored lower overall.
On 4 real Freedoom levels with all monsters (much larger than an evaluation map), no version reaches the
exit yet; the score there rose from 11.3 to 14.4 after adding real levels to training.

Training: about 25 M environment steps on one laptop (RTX 5060 Laptop GPU, i7-14700HX), roughly 10 hours
including all experiments.

## Known limitations

- **Aiming** is the weakest skill; crowded tier-4 maps still kill the agent often.
- Full-size Freedoom levels with all monsters are not solved (they are much larger than a 2–4k-tic evaluation map).
- Dead reckoning drifts over long episodes (the map stays locally consistent); lifts and teleporters are not modelled.
- Pixels (RGB/depth) are not used.
- The approximate score in `eval.py` uses guessed weights; the real scoring weights are hidden.

## Repository

```
submission/   agent.py (DoomAgent), model.py (SlayNet), spatial_memory.py, slay_obs.py, weights.pt (LFS)
training/     train.py/trainer.py, ppo.py, envs.py, rewards.py, curriculum.py, mapgen.py, realmaps.py,
              eval.py, compare.py, inspect_play.py, watch.py, record.py, play_human.py, make_brief.py
maps/         heldout/ (8 generated held-out maps), heldout_real/ (4 real Freedoom levels), two_rooms.wad
tests/        test_env.py (wrapper), test_mapgen.py, test_slay.py (model, memory, masks, PPO, rewards)
docs/         CONTRACT.md (official), TRACK_BRIEF.md, SPEC.md, compare*.json (fair-test results)
brief.pdf     2-page project brief
```
