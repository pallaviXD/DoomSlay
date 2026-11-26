"""Numpy-only observation packing and action encoding, shared by the agent, the model and the
training env workers. Kept free of torch so env worker processes start fast and stay small."""

from __future__ import annotations

import numpy as np

# "egomap" is not in the raw observation: the spatial-memory crop added by spatial_memory.SpatialMemory.
OBS_KEYS = ("ray_dist", "ray_type", "ent_feat", "ent_type", "ent_mask", "vitals", "egomap")

# Factored action heads: sizes of each categorical.
HEAD_SIZES = (3, 3, 3, 2, 2, 2)  # move, strafe, turn, attack, use, next_weapon

# Head option -> button index (None = press nothing). Button order = doom_env.spaces.BUTTONS.
HEAD_BUTTONS = (
    (None, 0, 1),  # move: none, MOVE_FORWARD, MOVE_BACKWARD
    (None, 2, 3),  # strafe: none, MOVE_LEFT, MOVE_RIGHT
    (None, 4, 5),  # turn: none, TURN_LEFT, TURN_RIGHT
    (None, 6),  # ATTACK
    (None, 7),  # USE
    (None, 8),  # SELECT_NEXT_WEAPON
)


def pack_obs(obs: dict) -> dict[str, np.ndarray]:
    """Contract observation dict -> flat dict of the arrays the network uses (pixels are not used)."""
    rays, ents = obs["rays"], obs["visible_entities"]
    return {
        "ray_dist": np.asarray(rays["distance"], np.float32),
        "ray_type": np.asarray(rays["hit_type"], np.int64),
        "ent_feat": np.asarray(ents["features"], np.float32),
        "ent_type": np.asarray(ents["entity_type"], np.int64),
        "ent_mask": np.asarray(ents["mask"], np.float32),
        "vitals": np.asarray(obs["vitals"], np.float32),
    }


def actions_to_buttons(head_actions: np.ndarray) -> np.ndarray:
    """(..., 6) int head choices -> (..., 9) {0,1} buttons. A fixed encoding, not a decision rule."""
    head_actions = np.asarray(head_actions)
    out = np.zeros(head_actions.shape[:-1] + (9,), np.int8)
    for h, table in enumerate(HEAD_BUTTONS):
        for opt, btn in enumerate(table):
            if btn is not None:
                out[..., btn] |= (head_actions[..., h] == opt).astype(np.int8)
    return out
