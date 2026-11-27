# Training pipeline

Everything used to train, test and present the agent. Nothing here is imported by `submission/`
at evaluation time. See the main [README](../README.md) for the overview and results.

## Files

| File | What it does |
|---|---|
| `train.py` | Light launcher (keeps torch out of the worker processes), starts `trainer.py` |
| `trainer.py` | Main loop: 64 parallel games, CPU actor + GPU learner, curriculum, held-out eval, checkpoints |
| `ppo.py` | Our recurrent PPO from scratch: GAE, clipped loss, truncated BPTT over GRU sequences |
| `envs.py` | Worker processes running the official `doom_env` wrapper, with reward shaping on top |
| `rewards.py` | Shaped rewards (exit, keys, doors, kills, exploring, aim, penalties) using training-only info |
| `curriculum.py` | Stage progression and the map pool (generated + real levels, refreshed every 1,500 games) |
| `mapgen.py` | UDMF map generator: rooms, corridors, doors, keycards, locked doors, exit switches, stairs, arenas |
| `realmaps.py` | Copies 27 real Freedoom levels for training and keeps 4 aside for testing |
| `eval.py`, `compare.py` | Fair repeated tests on the held-out maps (several seeds) |
| `inspect_play.py` | Replays + behaviour stats (empty-air shots, fist time, loops) |
| `export.py` | Writes `submission/weights.pt` from a checkpoint |
| `watch.py`, `record.py`, `make_video.py` | Watch the agent live, record clips, build the full-HUD MP4 |
| `play_human.py` | Play a generated level yourself |

## Reproduce

```bash
uv sync
uv run python -m training.realmaps
uv run python -m training.train --run demo --stage 1 --num-envs 64 --mask-level 3
uv run python -m training.export checkpoints/demo/best.pt
```

Checkpoints, logs and generated maps go to `checkpoints/`, `runs/` and `training/generated_maps/`
(git-ignored). Map, engine and reward changes here are for **training only**; evaluation uses the
vanilla engine through the official wrapper.
