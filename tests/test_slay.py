"""Team Slay unit tests: model shapes, masked attention pooling, PPO loss/GAE, reward shaping.
Run: .venv\\Scripts\\python.exe -m pytest tests/test_slay.py -q"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from submission.model import (  # noqa: E402
    HEAD_SIZES, NUM_LOGITS, PREV_ACT_DIM, EntityEncoder, SlayNet, actions_one_hot, actions_to_buttons,
    log_prob_entropy, sample_actions,
)
from training.ppo import PPOConfig, compute_gae, ppo_loss  # noqa: E402
from training.rewards import REWARD_WEIGHTS, RewardShaper  # noqa: E402


def fake_obs(n, seed=0, lead=()):
    g = np.random.default_rng(seed)
    s = lead + (n,)
    return {
        "ray_dist": torch.tensor(g.random(s + (64,)), dtype=torch.float32),
        "ray_type": torch.tensor(g.integers(0, 6, s + (64,)), dtype=torch.long),
        "ent_feat": torch.tensor(g.random(s + (32, 5)), dtype=torch.float32),
        "ent_type": torch.tensor(g.integers(0, 12, s + (32,)), dtype=torch.long),
        "ent_mask": torch.tensor(g.random(s + (32,)) > 0.7, dtype=torch.float32),
        "vitals": torch.tensor(g.random(s + (17,)) * 100, dtype=torch.float32),
    }


# ----------------------------------------------------------------------------- model
def test_step_shapes():
    net = SlayNet()
    o = fake_obs(5)
    logits, v, h = net.step(o, torch.zeros(5, PREV_ACT_DIM), net.initial_state(5))
    assert logits.shape == (5, NUM_LOGITS) and v.shape == (5,) and h.shape == (5, 256)


def test_sequence_matches_stepwise_and_resets():
    torch.manual_seed(0)
    net = SlayNet()
    T, B = 6, 3
    o = fake_obs(B, lead=(T,))
    prev = torch.zeros(T, B, PREV_ACT_DIM)
    starts = torch.zeros(T, B)
    starts[3, 1] = 1.0  # env 1 starts a new episode at t=3
    h0 = torch.randn(B, 256) * 0.1
    lg_seq, v_seq = net.forward_sequence(o, prev, h0, starts)
    h = h0
    for t in range(T):
        h = h * (1 - starts[t]).unsqueeze(-1)
        lg, v, h = net.step({k: x[t] for k, x in o.items()}, prev[t], h)
        assert torch.allclose(lg, lg_seq[t], atol=1e-5) and torch.allclose(v, v_seq[t], atol=1e-5)
    # After the reset, env 1's output must not depend on h0.
    lg2, _ = net.forward_sequence(o, prev, h0 * 0 + 5.0, starts)
    assert torch.allclose(lg_seq[3:, 1], lg2[3:, 1], atol=1e-5)
    assert not torch.allclose(lg_seq[:3, 1], lg2[:3, 1], atol=1e-5)


def test_actions_to_buttons_and_heads():
    a = np.array([[1, 2, 1, 1, 0, 1], [0, 0, 0, 0, 1, 0]])
    b = actions_to_buttons(a)
    assert b.tolist() == [[1, 0, 0, 1, 1, 0, 1, 0, 1], [0, 0, 0, 0, 0, 0, 0, 1, 0]]
    logits = torch.randn(1000, NUM_LOGITS)
    s = sample_actions(logits)
    for h, n in enumerate(HEAD_SIZES):  # padded option never sampled
        assert int(s[:, h].max()) < n
    assert actions_one_hot(s).shape == (1000, NUM_LOGITS)


def test_log_prob_entropy_matches_reference():
    logits = torch.randn(7, NUM_LOGITS, requires_grad=True)
    acts = sample_actions(logits.detach())
    lp, ent = log_prob_entropy(logits, acts)
    ref_lp, ref_ent = 0, 0
    for h, lg in enumerate(torch.split(logits, HEAD_SIZES, -1)):
        d = torch.distributions.Categorical(logits=lg)
        ref_lp = ref_lp + d.log_prob(acts[:, h])
        ref_ent = ref_ent + d.entropy()
    assert torch.allclose(lp, ref_lp, atol=1e-5) and torch.allclose(ent, ref_ent, atol=1e-5)
    (lp.sum() + ent.sum()).backward()
    assert torch.isfinite(logits.grad).all()


# ----------------------------------------------------------------------------- masked attention
def test_entity_encoder_ignores_padding_and_order():
    torch.manual_seed(0)
    enc = EntityEncoder()
    o = fake_obs(1)
    mask = torch.zeros(1, 32)
    mask[0, :4] = 1
    base = enc(o["ent_feat"], o["ent_type"], mask)
    # 1) garbage in padded slots does not change the output
    f2, t2 = o["ent_feat"].clone(), o["ent_type"].clone()
    f2[0, 4:] = 123.0
    t2[0, 4:] = 7
    assert torch.allclose(base, enc(f2, t2, mask), atol=1e-5)
    # 2) permutation invariant over real entities
    perm = torch.tensor([2, 0, 3, 1] + list(range(4, 32)))
    assert torch.allclose(base, enc(o["ent_feat"][:, perm], o["ent_type"][:, perm], mask[:, perm]), atol=1e-5)
    # 3) no entities -> finite output (pooled parts are zero)
    empty = enc(o["ent_feat"], o["ent_type"], torch.zeros(1, 32))
    assert torch.isfinite(empty).all() and float(empty.abs().sum()) == 0.0


# ----------------------------------------------------------------------------- PPO
def test_gae_simple():
    r = torch.tensor([[1.0], [1.0], [1.0]])
    v = torch.zeros(3, 1)
    d = torch.tensor([[0.0], [1.0], [0.0]])  # episode ends after t=1
    adv, ret = compute_gae(r, v, d, torch.tensor([10.0]), gamma=0.5, lam=1.0)
    assert torch.allclose(adv[:, 0], torch.tensor([1.5, 1.0, 6.0]))  # t2 bootstraps from last value
    assert torch.allclose(ret, adv)


def test_ppo_loss_finite_and_decreases():
    torch.manual_seed(0)
    net = SlayNet()
    cfg = PPOConfig()
    L, B = 8, 4
    o = fake_obs(B, lead=(L,))
    prev = torch.zeros(L, B, PREV_ACT_DIM)
    starts = torch.zeros(L, B)
    starts[0] = 1
    h0 = net.initial_state(B)
    with torch.no_grad():
        logits, values = net.forward_sequence(o, prev, h0, starts)
        acts = sample_actions(logits)
        logp, _ = log_prob_entropy(logits, acts)
    adv = torch.randn(L, B)
    mb = {"obs": o, "prev_act": prev, "h0": h0, "starts": starts, "actions": acts, "logp": logp,
          "values": values, "adv": adv, "returns": values + adv}
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss0, st = ppo_loss(net, mb, cfg)
    assert np.isfinite(loss0.item()) and abs(st["approx_kl"]) < 1e-6  # same policy -> ratio 1
    for _ in range(20):
        loss, _ = ppo_loss(net, mb, cfg)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < loss0.item()


# ----------------------------------------------------------------------------- rewards
def priv(**kw):
    p = {"tic": 0, "x": 0.0, "y": 0.0, "health": 100.0, "kills": 0, "damage_taken": 0.0, "keys": np.zeros(6, np.float32),
         "dead": False, "exit_reached": False, "finished": False, "sector_floor": np.zeros(3), "sector_ceiling": np.zeros(3)}
    p.update(kw)
    return p


def vit(sel_ammo=50, ammo=(50, 0, 0, 0)):
    v = np.zeros(17, np.float32)
    v[3] = sel_ammo
    v[4:8] = ammo
    return v


def test_reward_components():
    w = REWARD_WEIGHTS
    sh = RewardShaper()
    sh.reset(priv(), {2: 1})
    r, stuck = sh.step(priv(tic=4, x=100.0), vit())  # new cell
    assert r == pytest.approx(w["step"] + w["new_cell"]) and not stuck
    r, _ = sh.step(priv(tic=8, x=100.0, kills=1), vit())
    assert r == pytest.approx(w["step"] + w["kill"])
    r, _ = sh.step(priv(tic=12, x=100.0, kills=1, damage_taken=10.0), vit())
    assert r == pytest.approx(w["step"] + 10 * w["damage_per_hp"])
    keys = np.zeros(6, np.float32)
    keys[2] = 1
    r, _ = sh.step(priv(tic=16, x=100.0, kills=1, damage_taken=10.0, keys=keys), vit())
    assert r == pytest.approx(w["step"] + w["keycard"])
    r, _ = sh.step(priv(tic=20, x=100.0, kills=1, damage_taken=10.0, keys=keys,
                        sector_ceiling=np.array([0, 0, 128.0])), vit())  # locked door sector 2 opened
    assert r == pytest.approx(w["step"] + w["door_open"] + w["locked_door_bonus"])
    r, _ = sh.step(priv(tic=24, x=100.0, kills=1, damage_taken=10.0, keys=keys, exit_reached=True, finished=True,
                        sector_ceiling=np.array([0, 0, 128.0])), vit())
    assert r == pytest.approx(w["step"] + w["exit"])
    s = sh.stats
    assert (s.kills, s.keys, s.doors, s.locked_doors, s.exit, s.cells) == (1, 1, 1, 1, True, 2)


def test_weapon_switch_is_not_a_pickup_and_stuck_cutoff():
    sh = RewardShaper(stuck_ticks=200)
    sh.reset(priv(), {})
    sh.step(priv(tic=4), vit(sel_ammo=5, ammo=(5, 20, 0, 0)))
    r, _ = sh.step(priv(tic=8), vit(sel_ammo=20, ammo=(5, 20, 0, 0)))  # switched to shotgun: no pickup
    assert r == pytest.approx(REWARD_WEIGHTS["step"])
    r, _ = sh.step(priv(tic=12), vit(sel_ammo=5, ammo=(15, 20, 0, 0)))  # real clip pickup while low
    assert r == pytest.approx(REWARD_WEIGHTS["step"] + REWARD_WEIGHTS["pickup_when_low"])
    r, stuck = sh.step(priv(tic=200), vit())
    assert stuck and r == pytest.approx(REWARD_WEIGHTS["step"] + REWARD_WEIGHTS["stuck"])


def test_damage_dealt_reward():
    sh = RewardShaper()
    sh.reset(priv(damage_dealt=0.0), {})
    r, _ = sh.step(priv(tic=4, damage_dealt=20.0), vit())
    assert r == pytest.approx(REWARD_WEIGHTS["step"] + 20 * REWARD_WEIGHTS["damage_dealt_per_hp"])
    r, _ = sh.step(priv(tic=8, damage_dealt=20.0), vit())  # no new damage -> no reward
    assert r == pytest.approx(REWARD_WEIGHTS["step"])


def test_ammo_spent_cost():
    sh = RewardShaper()
    sh.reset(priv(), {})
    sh.step(priv(tic=4), vit(sel_ammo=50, ammo=(50, 0, 0, 0)))
    r, _ = sh.step(priv(tic=8), vit(sel_ammo=48, ammo=(48, 0, 0, 0)))  # two bullets fired, no hit
    assert r == pytest.approx(REWARD_WEIGHTS["step"] + 2 * REWARD_WEIGHTS["ammo_spent"])
