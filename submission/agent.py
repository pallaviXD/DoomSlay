"""Team Slay evaluation entry point. The scorer instantiates DoomAgent from this file.

Contract: docs/CONTRACT.md §6. Every action comes from the trained network in model.py
(recurrent PPO policy); this file only converts observations to tensors and keeps the GRU state.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch

SUBMISSION_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = SUBMISSION_DIR / "weights.pt"


if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))  # model.py imports its sibling slay_obs.py


def _load_model_module():
    # Load submission/model.py under a unique name so it never collides with other "model" modules.
    spec = importlib.util.spec_from_file_location("slay_model", SUBMISSION_DIR / "model.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["slay_model"] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load_model_module()
from spatial_memory import SpatialMemory  # noqa: E402  (sibling file, numpy only)

SAFE_ACTION = np.zeros(9, dtype=np.int8)  # stand still: always a valid action


class DoomAgent:
    def __init__(self, observation_space, action_space, device: str):
        torch.set_num_threads(1)  # evaluation gives us exactly one CPU thread
        self.observation_space = observation_space
        self.action_space = action_space
        # The net is ~0.9 M params: one CPU step takes ~1-2 ms, cheaper than GPU launch + transfer
        # overhead per single-observation step, and it never touches the shared 2 GB VRAM budget.
        self.device = torch.device("cpu")
        self.net = M.SlayNet()
        self.sample = True  # stochastic policy (as trained); fixed seed keeps runs reproducible
        if WEIGHTS_PATH.is_file():
            ckpt = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
            self.load_weights(ckpt)
            self.sample = bool(ckpt.get("sample", True)) if isinstance(ckpt, dict) else True
        else:
            torch.manual_seed(0)  # untrained fallback (Phase 1 baseline); still a neural policy
        self.net.eval()
        self.gen = torch.Generator().manual_seed(1234)
        self.memory = SpatialMemory()  # self-built mini-map (see spatial_memory.py)
        self.failures = 0  # act() exceptions (eval.py reports this; must stay 0)
        self.reset()

    def load_weights(self, ckpt) -> None:
        """Accepts a checkpoint dict or a raw state dict; older nets without spatial memory warm-start."""
        M.load_compatible(self.net, ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)
        # Policies trained with action masks must also run with them (flag saved by trainer/export).
        if isinstance(ckpt, dict) and "model" in ckpt:
            self.net.use_masks = int(ckpt.get("mask_level", 2 if ckpt.get("masks", False) else 0))
        else:
            self.net.use_masks = 0

    def reset(self) -> None:
        """Before every map: clear GRU memory, the spatial-memory map and the previous-action input."""
        self.h = self.net.initial_state(1)
        self.prev = torch.zeros(1, M.PREV_ACT_DIM)
        self.memory.reset()
        self.prev_buttons = None  # None -> the next observation is the first of the map

    @torch.no_grad()
    def act(self, observation: dict) -> np.ndarray:
        try:
            p = M.pack_obs(observation)
            p["egomap"] = self.memory.update(observation, self.prev_buttons)
            o = {k: torch.from_numpy(v).unsqueeze(0) for k, v in p.items()}
            logits, _, self.h = self.net.step(o, self.prev, self.h)
            a = M.sample_actions(logits, greedy=not self.sample, generator=self.gen)
            self.prev = M.actions_one_hot(a)
            buttons = M.actions_to_buttons(a.numpy())[0]
            self.prev_buttons = buttons
            return buttons
        except Exception as exc:  # never crash the episode: stand still for this step
            self.failures += 1
            if self.failures <= 3:  # don't flood the scorer logs
                print(f"[DoomAgent] act() failed: {exc!r}", file=sys.stderr)
            self.prev_buttons = SAFE_ACTION
            return SAFE_ACTION.copy()
