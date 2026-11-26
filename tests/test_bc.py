"""Tests for the behavioural-cloning tools (training/bc_collect.py, training/bc_train.py).

Run: .\\.venv\\Scripts\\python.exe -m pytest tests/test_bc.py -q      (< 60 s)
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from submission.model import HEAD_SIZES, SlayNet, actions_one_hot, actions_to_buttons  # noqa: E402
from training import bc_collect as C  # noqa: E402
from training import bc_train as BT  # noqa: E402

torch.set_num_threads(2)  # the machine may be busy with RL workers: stay modest
CKPT = ROOT / "checkpoints" / "mem1" / "cand_mask4160.pt"


def test_heads_buttons_round_trip():
    combos = np.array(list(itertools.product(*[range(n) for n in HEAD_SIZES])))  # all 216 head choices
    assert (C.buttons_to_heads(actions_to_buttons(combos)) == combos).all()
    for h in combos[::17]:  # one-hot layout matches the model's previous-action encoding
        assert (C.heads_one_hot(h) == actions_one_hot(torch.as_tensor(h)).numpy()).all()
    both = np.zeros(9, np.int8)
    both[[0, 1, 2, 3, 4, 5, 7]] = 1  # fwd+back, left+right, turn L+R cancel in DOOM -> "none"; USE kept
    assert C.buttons_to_heads(both).tolist() == [0, 0, 0, 0, 1, 0]


@pytest.fixture(scope="module")
def shard(tmp_path_factory):
    out = tmp_path_factory.mktemp("bc")
    spec = {"stage": 2, "seed": 4242, "switch_exit": False, "stairs": False}
    arrs, info = C.collect_episode(spec, out / "t.wad", env_seed=1)
    assert arrs is not None and info["steps"] > 10
    C.write_shard(out / "shard_test_00000.npz", [(0, arrs, spec, info), (1, arrs, spec, info)])
    return out, info


def test_shard_shapes_and_consistency(shard):
    out, info = shard
    z = np.load(out / "shard_test_00000.npz")
    n = info["steps"]
    assert z["ray_dist"].shape == (2 * n, 64) and z["ent_feat"].shape == (2 * n, 32, 5)
    assert z["egomap"].shape == (2 * n, 5, 21, 21) and z["egomap"].max() <= C.EGOMAP_SCALE
    assert z["prev_act"].shape == (2 * n, 15) and z["action"].shape == (2 * n, 6)
    assert (z["ep_len"] == n).all() and z["t"][0] == 0 and z["t"][n] == 0
    # previous action = one-hot of the label one step earlier, zeros at the first step
    assert (z["prev_act"][0] == 0).all()
    for t in range(1, n):
        assert (z["prev_act"][t] == C.heads_one_hot(z["action"][t - 1])).all()
    assert (actions_to_buttons(z["action"]) == z["buttons"]).all()
    # what forward_sequence expects: (T,B,...) dicts, prev (T,B,15), starts (T,B)
    D = BT.load_dataset(out, include_failed=True)
    idx, valid, starts = next(BT.chunks(D, np.array([0, 1]), 32))
    o, prev, act = BT.to_obs(D, idx, "cpu")
    assert o["egomap"].shape == (32, 2, 5, 21, 21) and prev.shape == (32, 2, 15) and act.shape == (32, 2, 6)
    net = SlayNet()
    logits, v = net.forward_sequence(o, prev, net.initial_state(2), torch.from_numpy(starts))
    assert logits.shape == (32, 2, 15) and v.shape == (32, 2) and torch.isfinite(logits).all()


def test_bc_reduces_nav_loss_keeps_attack(shard):
    out, _ = shard
    torch.manual_seed(0)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False) if CKPT.is_file() else {"model": SlayNet().state_dict()}
    net, ref, _ = BT.make_nets(ck, "cpu")
    D = BT.load_dataset(out, include_failed=True)
    eps = np.arange(len(D["ep_len"]))
    before = BT.run_epoch(net, ref, D, eps, "cpu")
    assert before["kl_keep"] < 1e-6  # student == reference at the start
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=1e-4, eps=1e-5)
    crit0 = [p.clone() for p in net.critic.parameters()]
    BT.run_epoch(net, ref, D, eps, "cpu", opt=opt, rng=np.random.default_rng(0), max_updates=12)
    after = BT.run_epoch(net, ref, D, eps, "cpu")
    print(f"nav {before['nav_loss']:.3f} -> {after['nav_loss']:.3f}, kl_keep {after['kl_keep']:.4f}")
    assert after["nav_loss"] < 0.9 * before["nav_loss"]
    assert after["kl_keep"] < 0.05
    assert all(torch.equal(a, b) for a, b in zip(crit0, net.critic.parameters()))  # critic head frozen
