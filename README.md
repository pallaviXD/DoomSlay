# Team Slay — DoomSlay

Yugaantar 2026 · DevForge — Track 1 *"Can It Play DOOM?"*

A recurrent PPO agent (our own implementation) that plays unseen ViZDoom maps zero-shot from the
ego-centric observation only: 64 raycasts, visible entities, vitals. It is trained on thousands of
procedurally generated maps through a four-stage curriculum: combat → exploration → keycard/locked
door → gauntlet.

## Setup

Windows 11, Linux or macOS (Apple Silicon), Python 3.13, [uv](https://docs.astral.sh/uv/):

```bash
uv sync                          # vizdoom 1.3.3, gymnasium, doom_env (editable), torch 2.15.0, tensorboard
uv run python validate.py        # official submission check
uv run pytest -q                 # wrapper, map generator and agent/PPO/reward tests
```

`uv sync` installs the `train` dependency group by default. On Windows, torch comes from the PyTorch
cu132 index (CUDA 13.2, which RTX 50xx needs); on Linux it comes from PyPI. The submission itself needs no extra packages
(`submission/requirements.txt` is empty): torch and numpy are in the official image.

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
uv run python -m training.train --run smoke --smoke                       # 1-minute pipeline check
uv run python -m training.train --run cur1 --stage 1 --total-steps 30e6   # full curriculum
uv run tensorboard --logdir runs                                          # curves
```

Checkpoints go to `checkpoints/<run>/` (`latest.pt`, and `best.pt` = best *held-out* score), and
logs go to `runs/<run>/`. All of these are git-ignored.

## Evaluate

```bash
uv run python -m training.eval --ckpt checkpoints/cur1/best.pt     # held-out maps, real DoomAgent
uv run python -m training.mapgen --stage 3 --seeds 0-9 --out training/generated_maps/s3   # make maps
uv run python validate.py --map maps/heldout/heldout_s3_4.wad:MAP01
```

## Architecture

```
rays 64 x [dist, near, 6 hit-type one-hot] -- circular Conv1d x3 --------------> 256
visible entities <=32 x [5 feats, type emb] -- per-entity MLP, masked attn pool -> 129+64
vitals (26 normalised) + previous action (15) -- MLP ---------------------------> 64
          concat -> Linear 256 -> GRUCell 256 -> actor (6 categorical heads) / critic
```

| Component | Why |
|---|---|
| **Circular 1-D conv over rays** | The 64 rays form a 360° ring, so circular padding makes "behind" wrap correctly. The conv weights are shared over heading, so a doorway looks the same at any angle, which helps on unseen layouts. |
| **Masked attention pooling over entities** | Between 0 and 32 entities are visible. A per-entity MLP plus 2 learned attention queries and a masked max-pool is permutation-invariant, ignores padding, and gives zeros when nothing is visible (all unit-tested). |
| **GRU memory (256)** | No map or position is observed, so the GRU is the agent's memory of where it has been, which doors it has seen, and whether it holds a key. It is reset in `reset()` and at every episode boundary in training. |
| **Factored action heads** | The 9 buttons become 6 categoricals (move none/fwd/back, strafe, turn, attack, use, next weapon). This rules out contradictory presses, keeps log-probs a simple sum, and the previous action is an input so the policy can learn `USE`'s button-down behaviour. |
| **0.9 M parameters, CPU inference** | About 1.5 ms per `act()` on 1 thread, well inside the 40 ms budget, and it never touches shared VRAM. |

## Training pipeline

- **Own recurrent PPO** ([training/ppo.py](training/ppo.py)):
  - GAE with γ 0.99, λ 0.95; clip 0.2 with clipped value loss; 4 epochs.
  - Rollouts of 128 steps × 24 envs, cut into 32-step truncated-BPTT sequences that start from the stored GRU state.
  - Entropy 0.01, grad-norm clip 0.5, learning rate decaying linearly from 3e-4 to 1e-4.
- **Environments:** 24 spawn-based env processes ([training/envs.py](training/envs.py)). Two halves are pipelined so the policy computes actions for one half while the other half steps.
- **CPU actor, GPU learner:** the policy runs rollouts on the CPU and the PPO update runs on the GPU. On Windows, small GPU batches lose more time to kernel launch overhead than they save.
- **Procedural maps** ([training/mapgen.py](training/mapgen.py)): UDMF grid maps built from a spanning-tree room graph with extra loops, corridors, doors, enemies, health and ammo items, and an exit.
  - Stage 1: 1–2 rooms, weak enemies.
  - Stage 2: 4–8 rooms, exit in the farthest room.
  - Stage 3: the locked door sits on the only route to the exit, and its keycard is in a side branch.
  - Stage 4: a gauntlet with more enemies and fewer items.
  - A scripted driver (test-only code) reached the exit on all 100 maps in its sweep.
- **Curriculum** ([training/curriculum.py](training/curriculum.py)):
  - The agent advances a stage when its exit rate passes 60%, 50%, then 40%; after the last stage all stages are mixed.
  - 25% of each pool replays earlier stages.
  - The pool is regenerated with fresh seeds every 1,500 episodes, so nothing can be memorised.
- **Reward shaping** ([training/rewards.py](training/rewards.py)): uses the privileged position, training only.

  | Event | Reward |
  |---|---|
  | Exit | +10 |
  | Keycard | +3 |
  | Door opened | +2 (+1 if locked) |
  | Kill | +1 |
  | New 64-unit cell | +0.05 |
  | Health or ammo pickup when low | +0.1 |
  | Damage | −0.01 per HP |
  | Death | −3 |
  | Each step | −0.001 |
  | Stuck 200 tics without progress | −1, episode ends |

- **Held-out evaluation** ([training/eval.py](training/eval.py)): runs the real `DoomAgent` on 8 maps from a reserved seed range (`maps/heldout/`) every 100 updates. The best checkpoint is kept by held-out score, not by recency.

## Results

_To be filled at the end of training: held-out exit rate per tier, approximate score, curves._

## Known limitations

- The generated maps are flat and grid-aligned: no stairs, lifts, switches or skull keys, and only one key/lock pair.
- There is no explicit metric map. Memory lives only in the GRU, so very long backtracking (tier 4) is the weakest point.
- The policy does not use pixels (RGB or depth), so it cannot read wall textures that hint at an exit.
- The approximate score in `eval.py` uses guessed weights; the real scoring weights are hidden.

## Repository

```
submission/   agent.py (DoomAgent), model.py (SlayNet), slay_obs.py (numpy packing), weights.pt (LFS)
training/     train.py/trainer.py, ppo.py, envs.py, mapgen.py, curriculum.py, rewards.py, eval.py
maps/heldout/ held-out evaluation maps (never trained on)
tests/        test_env.py (wrapper), test_mapgen.py, test_slay.py (model / attention / PPO / rewards)
docs/         CONTRACT.md (official), SPEC.md (our condensed spec), RULES.md, EVALUATION.md
```
