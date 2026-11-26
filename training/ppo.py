"""Team Slay recurrent PPO (written from scratch; no Stable-Baselines3).

Rollout: T=128 steps x N envs. Each step stores the GRU state *before* the step, so training can
restart BPTT from the exact hidden state the policy had (the "stored state" strategy) at the start of
every 32-step sequence chunk. Inside a chunk, the hidden state is zeroed wherever an episode starts.

Loss = -clipped surrogate + 0.5 * value loss (clipped) - 0.01 * entropy, Adam, grad-norm clip 0.5,
GAE(gamma=0.99, lambda=0.95), 4 epochs, advantages normalised per minibatch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from submission.model import OBS_KEYS, SlayNet, actions_one_hot, log_prob_entropy, sample_actions


@dataclass
class PPOConfig:
    rollout_len: int = 128
    seq_len: int = 32
    epochs: int = 4
    minibatches: int = 4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    lr: float = 3e-4
    lr_final: float = 1e-4


def compute_gae(rewards, values, dones, last_value, gamma, lam):
    """rewards/values/dones: (T,N); dones[t]=1 if the episode ended after step t. Returns adv, ret."""
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    gae = torch.zeros_like(last_value)
    for t in reversed(range(T)):
        nxt = last_value if t == T - 1 else values[t + 1]
        nonterm = 1.0 - dones[t]
        delta = rewards[t] + gamma * nxt * nonterm - values[t]
        gae = delta + gamma * lam * nonterm * gae
        adv[t] = gae
    return adv, adv + values


def ppo_loss(net: SlayNet, mb: dict, cfg: PPOConfig):
    """mb tensors shaped (L, B, ...) (L = seq_len). Returns (loss, stats dict)."""
    logits, values = net.forward_sequence(mb["obs"], mb["prev_act"], mb["h0"], mb["starts"])
    logp, ent = log_prob_entropy(logits, mb["actions"])
    adv = mb["adv"]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    ratio = torch.exp(logp - mb["logp"])
    pg = -torch.min(ratio * adv, torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv).mean()
    # Clipped value loss (PPO2 style) keeps the critic from jumping on rare big rewards (exit +10).
    v_clip = mb["values"] + torch.clamp(values - mb["values"], -cfg.clip, cfg.clip)
    vf = 0.5 * torch.max((values - mb["returns"]) ** 2, (v_clip - mb["returns"]) ** 2).mean()
    entropy = ent.mean()
    loss = pg + cfg.vf_coef * vf - cfg.ent_coef * entropy
    with torch.no_grad():
        approx_kl = ((ratio - 1) - torch.log(ratio)).mean()
        clipfrac = ((ratio - 1).abs() > cfg.clip).float().mean()
    return loss, {"pg_loss": pg.item(), "vf_loss": vf.item(), "entropy": entropy.item(),
                  "approx_kl": approx_kl.item(), "clipfrac": clipfrac.item()}


class RolloutBuffer:
    def __init__(self, T: int, N: int, hidden: int, device):
        self.T, self.N, self.device = T, N, device
        self.obs: dict[str, list] = {k: [] for k in OBS_KEYS}
        self.clear(hidden)

    def clear(self, hidden):
        self.obs = {k: [] for k in OBS_KEYS}
        self.prev_act, self.h, self.starts = [], [], []
        self.actions, self.logp, self.values, self.rewards, self.dones = [], [], [], [], []

    def add(self, obs_t, prev_act, h, starts, actions, logp, values, rewards, dones):
        for k in OBS_KEYS:
            self.obs[k].append(obs_t[k])
        self.prev_act.append(prev_act)
        self.h.append(h)
        self.starts.append(starts)
        self.actions.append(actions)
        self.logp.append(logp)
        self.values.append(values)
        self.rewards.append(rewards)
        self.dones.append(dones)

    def stack(self) -> dict:
        s = lambda xs: torch.stack(xs)  # noqa: E731
        return {"obs": {k: s(v) for k, v in self.obs.items()}, "prev_act": s(self.prev_act), "h": s(self.h),
                "starts": s(self.starts), "actions": s(self.actions), "logp": s(self.logp),
                "values": s(self.values), "rewards": s(self.rewards), "dones": s(self.dones)}


def to_torch_obs(obs: dict[str, np.ndarray], device) -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(v).to(device, non_blocking=True) for k, v in obs.items()}


class PPO:
    def __init__(self, net: SlayNet, cfg: PPOConfig, device):
        self.net, self.cfg, self.device = net, cfg, device
        self.opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5)

    def set_lr(self, frac_done: float) -> float:
        lr = self.cfg.lr + (self.cfg.lr_final - self.cfg.lr) * min(max(frac_done, 0.0), 1.0)
        for g in self.opt.param_groups:
            g["lr"] = lr
        return lr

    @torch.no_grad()
    def act(self, obs_t, prev_act, h, net: SlayNet | None = None):
        """Rollout step. `net` = CPU actor copy (Windows GPU launch overhead makes tiny batches slower on GPU)."""
        logits, value, h_new = (net or self.net).step(obs_t, prev_act, h)
        actions = sample_actions(logits)
        logp, _ = log_prob_entropy(logits, actions)
        return actions, logp, value, h_new

    @torch.no_grad()
    def value(self, obs_t, prev_act, h, net: SlayNet | None = None):
        return (net or self.net).step(obs_t, prev_act, h)[1]

    def update(self, data: dict, last_value: torch.Tensor) -> dict:
        cfg = self.cfg
        dev = self.device  # rollout data may live on CPU; train on the learner device
        data = {k: ({kk: vv.to(dev) for kk, vv in v.items()} if isinstance(v, dict) else v.to(dev)) for k, v in data.items()}
        last_value = last_value.to(dev)
        adv, ret = compute_gae(data["rewards"], data["values"], data["dones"], last_value, cfg.gamma, cfg.lam)
        T, N = data["rewards"].shape
        L = cfg.seq_len
        assert T % L == 0, "rollout_len must be a multiple of seq_len"
        # Cut the (T,N) rollout into T/L x N sequences of length L; each keeps its stored start state.
        def chunks(x):  # (T,N,...) -> (L, T/L*N, ...)
            x = x.reshape(T // L, L, N, *x.shape[2:]).transpose(0, 1)
            return x.reshape(L, (T // L) * N, *x.shape[3:])

        seq = {
            "obs": {k: chunks(v) for k, v in data["obs"].items()},
            "prev_act": chunks(data["prev_act"]), "starts": chunks(data["starts"]),
            "actions": chunks(data["actions"]), "logp": chunks(data["logp"]),
            "values": chunks(data["values"]), "adv": chunks(adv), "returns": chunks(ret),
            "h0": chunks(data["h"])[0],  # GRU state before the first step of each chunk
        }
        S = seq["starts"].shape[1]
        mb_size = max(S // cfg.minibatches, 1)
        stats_acc: dict[str, list] = {}
        for _ in range(cfg.epochs):
            perm = torch.randperm(S, device=self.device)
            for i in range(0, S, mb_size):
                idx = perm[i : i + mb_size]
                mb = {k: (v[:, idx] if k != "h0" else v[idx]) for k, v in seq.items() if k != "obs"}
                mb["obs"] = {k: v[:, idx] for k, v in seq["obs"].items()}
                loss, st = ppo_loss(self.net, mb, cfg)
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                st["grad_norm"] = nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm, foreach=True).item()
                self.opt.step()
                for k, v in st.items():
                    stats_acc.setdefault(k, []).append(v)
        out = {k: float(np.mean(v)) for k, v in stats_acc.items()}
        var_y = ret.var()
        out["explained_var"] = float(1 - (ret - data["values"]).var() / (var_y + 1e-8))
        return out


def prev_from_actions(actions: torch.Tensor, starts_next: torch.Tensor) -> torch.Tensor:
    """Previous-action input for the next step: one-hot of the actions, zeroed for new episodes."""
    return actions_one_hot(actions) * (1.0 - starts_next).unsqueeze(-1)
