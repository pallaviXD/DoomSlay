"""Team Slay policy network: shared by training (training/ppo.py) and evaluation (submission/agent.py).

Design (kept small for the 40 ms / 1 CPU thread budget; ~0.9 M parameters):

    rays (64 x [dist, near, 6 hit-type one-hot])  --circular Conv1d x3-->  256
    visible entities (<=32 x [5 feats + type emb]) --per-entity MLP + masked attention pool--> 128
    vitals (17, normalised) + previous action      --MLP-->                    64
                         concat --> Linear(448->256) --> GRU(256) --> actor heads / critic

Why these choices (Q&A notes):
- Rays wrap around 360 deg, so the conv uses circular padding: a wall straight behind the player is
  "next to" rays 31 and 33. Convs give translation equivariance over heading, which generalises
  across unseen layouts better than a flat MLP.
- The number of visible entities varies (0..32). A per-entity MLP followed by attention pooling with
  learned queries is permutation-invariant and handles any count; the mask removes padding slots.
  We also keep a masked max-pool so a single dangerous enemy is never "averaged away".
- Partial observability (no map, no position) is handled by the GRU: it is the agent's memory of
  where it has been, which doors it saw and whether it holds a key.
- The 9 MultiBinary buttons are factored into 6 categorical heads with mutually exclusive options
  (move: none/fwd/back, strafe: none/left/right, turn: none/left/right, attack, use, next-weapon).
  This removes contradictory presses (fwd+back) and keeps the joint log-prob a simple sum.
  The mapping from head choices to buttons is a fixed encoding, not a decision rule.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # imported as a package module (training) or as a top-level file (agent adds submission/ to sys.path)
    from .slay_obs import HEAD_SIZES, OBS_KEYS, actions_to_buttons, pack_obs  # noqa: F401
except ImportError:
    from slay_obs import HEAD_SIZES, OBS_KEYS, actions_to_buttons, pack_obs  # noqa: F401

NUM_RAYS = 64
NUM_HIT_TYPES = 6
MAX_ENTITIES = 32
ENT_FEATS = 5
NUM_ENTITY_TYPES = 12
NUM_VITALS = 17

NUM_LOGITS = sum(HEAD_SIZES)
PREV_ACT_DIM = NUM_LOGITS  # one-hot of the previous factored action

# ----------------------------------------------------------------------------- action masking
# "Action masking from observation fields" (allowed by docs/RULES.md; organisers prefer RL / RL-heuristic
# hybrids). Masks only REMOVE clearly pointless options; the network still chooses every action:
#  - ATTACK only when a monster is visible (entity list is already field-of-view + line-of-sight filtered),
#    and not two steps in a row: holding pistol fire spreads shots, so the policy must tap-fire.
#  - USE only when something is within use reach straight ahead (door, switch, wall: DOOM's use range is
#    64 units); pressing in open space does nothing, and re-pressing a moving door reverses it.
_HEAD_OFFSETS = [sum(HEAD_SIZES[:i]) for i in range(len(HEAD_SIZES))]
FIRE_IDX = _HEAD_OFFSETS[3] + 1  # logit of "ATTACK pressed"
USE_IDX = _HEAD_OFFSETS[4] + 1  # logit of "USE pressed"
NEXT_IDX = _HEAD_OFFSETS[5] + 1  # logit of "SELECT_NEXT_WEAPON pressed"
# Mask levels (stored per checkpoint): 0 none; 1 weapon only; 2 weapon + fire + use.
# Weapon mask (found in replays: policies pressed NEXT_WEAPON on ~50-60% of steps and held the FIST ~45% of
# the time, "punching air"): NEXT_WEAPON only while holding the fist/chainsaw (slot <= 1) with ammo left, so
# the agent can get back to a gun; DOOM still auto-switches on weapon pickups and when ammo runs out.
ENEMY_TYPE_IDS = (1, 2, 3)  # doom_env.spaces.ENTITY_TYPES: enemy_melee, enemy_hitscan, enemy_projectile
USE_REACH = 80.0 / 1024.0  # ray distance units (64 use range + margin)
_FRONT_RAYS = [-2, -1, 0, 1, 2]  # rays within about +-11 degrees of straight ahead


def apply_action_mask(logits: torch.Tensor, o: dict[str, torch.Tensor], prev_act: torch.Tensor,
                      level: int = 2) -> torch.Tensor:
    """logits (N,15); o tensors with leading N; prev_act (N,15) one-hot of the previous action."""
    neg = torch.full_like(logits[:, 0], -1e9)
    v = o["vitals"]
    next_ok = (v[:, 2] <= 1) & (v[:, 4:8].sum(-1) > 0)
    logits = logits.clone()
    logits[:, NEXT_IDX] = torch.where(next_ok, logits[:, NEXT_IDX], neg)
    if level < 2:
        return logits
    et = o["ent_type"].long()
    enemy = ((o["ent_mask"] > 0.5) & (et >= ENEMY_TYPE_IDS[0]) & (et <= ENEMY_TYPE_IDS[-1])).any(-1)
    fire_ok = enemy & (prev_act[:, FIRE_IDX] < 0.5)
    front_d = o["ray_dist"][:, _FRONT_RAYS]
    front_t = o["ray_type"][:, _FRONT_RAYS]
    use_ok = ((front_d < USE_REACH) & (front_t > 0)).any(-1)
    logits[:, FIRE_IDX] = torch.where(fire_ok, logits[:, FIRE_IDX], neg)
    logits[:, USE_IDX] = torch.where(use_ok, logits[:, USE_IDX], neg)
    return logits


def actions_one_hot(head_actions: torch.Tensor) -> torch.Tensor:
    """(..., 6) long -> (..., NUM_LOGITS) float one-hot (used as previous-action input)."""
    return torch.cat([F.one_hot(head_actions[..., h], n).float() for h, n in enumerate(HEAD_SIZES)], -1)


# ----------------------------------------------------------------------------- encoders
class RayEncoder(nn.Module):
    """Circular 1D conv over the 64 rays (360 deg ring)."""

    def __init__(self, out_dim: int = 256):
        super().__init__()
        c_in = 2 + NUM_HIT_TYPES
        self.convs = nn.Sequential(
            nn.Conv1d(c_in, 32, 5, padding=2, padding_mode="circular"), nn.ReLU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2, padding_mode="circular"), nn.ReLU(),  # 64 -> 32
            nn.Conv1d(64, 64, 5, stride=2, padding=2, padding_mode="circular"), nn.ReLU(),  # 32 -> 16
        )
        self.fc = nn.Sequential(nn.Linear(64 * 16, out_dim), nn.ReLU())

    def forward(self, dist: torch.Tensor, hit: torch.Tensor) -> torch.Tensor:
        # dist (N,64) in [0,1]; "near" emphasises close obstacles (collision / doorways).
        near = torch.exp(-dist * 8.0)
        x = torch.cat([dist.unsqueeze(1), near.unsqueeze(1), F.one_hot(hit.long(), NUM_HIT_TYPES).float().transpose(1, 2)], 1)
        return self.fc(self.convs(x).flatten(1))


class EntityEncoder(nn.Module):
    """Per-entity MLP + masked multi-query attention pooling + masked max pool."""

    def __init__(self, hidden: int = 64, queries: int = 2):
        super().__init__()
        self.type_emb = nn.Embedding(NUM_ENTITY_TYPES, 16)
        self.mlp = nn.Sequential(nn.Linear(ENT_FEATS + 1 + 16, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.score = nn.Linear(hidden, queries)  # one attention logit per learned query
        self.out_dim = hidden * queries + hidden + 1

    def forward(self, feat: torch.Tensor, etype: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # feat (N,32,5), etype (N,32), mask (N,32) float 0/1
        f = feat.clone()
        f[..., 3:5] = torch.clamp(f[..., 3:5], -30.0, 30.0) / 10.0  # relative velocities
        near = torch.exp(-f[..., :1] * 4.0)
        h = self.mlp(torch.cat([f, near, self.type_emb(etype.long())], -1))  # (N,32,H)
        valid = mask > 0.5
        logits = self.score(h).masked_fill(~valid.unsqueeze(-1), -1e9)  # (N,32,Q)
        w = torch.softmax(logits, dim=1) * valid.unsqueeze(-1).float()  # all-masked rows -> zeros
        pooled = torch.einsum("nkq,nkh->nqh", w, h).flatten(1)  # (N,Q*H)
        maxp = h.masked_fill(~valid.unsqueeze(-1), 0.0).amax(dim=1)  # h >= 0 after ReLU, so 0 = empty
        count = mask.sum(1, keepdim=True) / 8.0
        return torch.cat([pooled, maxp, count], -1)


def normalize_vitals(v: torch.Tensor) -> torch.Tensor:
    """(N,17) raw vitals -> (N,VITALS_FEAT=26) roughly unit-scale features."""
    health = v[:, 0:1] / 100.0
    armor = v[:, 1:2] / 100.0
    weapon = F.one_hot(v[:, 2].long().clamp(0, 7), 8).float()
    sel_ammo = torch.log1p(v[:, 3:4].clamp(min=0)) / 5.0
    ammo = torch.log1p(v[:, 4:8].clamp(min=0)) / 5.0
    keys = v[:, 8:14]
    vel = v[:, 14:17] / 16.0
    low_health = (v[:, 0:1] < 40).float()
    no_ammo = (v[:, 3:4] <= 0).float()
    return torch.cat([health, armor, weapon, sel_ammo, ammo, keys, vel, low_health, no_ammo], -1)


VITALS_FEAT = 26  # 1 health + 1 armor + 8 weapon one-hot + 1 sel ammo + 4 ammo + 6 keys + 3 vel + 2 flags
MAP_FEAT = 64
MAP_CHANNELS, MAP_CROP = 5, 21  # submission/spatial_memory.py: visited, free, wall, door, locked door


class MapEncoder(nn.Module):
    """Tiny CNN over the ego-centric spatial-memory crop (5 x 21 x 21, 64 units per cell, up = facing).
    Lets the policy see where it has been and where doors / locked doors were, beyond the current view."""

    def __init__(self, out_dim: int = MAP_FEAT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(MAP_CHANNELS, 16, 3, stride=2, padding=1), nn.ReLU(),  # 21 -> 11
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),  # 11 -> 6
            nn.Flatten(), nn.Linear(32 * 6 * 6, out_dim), nn.ReLU(),
        )

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        return self.net(m)


def load_compatible(net: "SlayNet", sd: dict) -> list[str]:
    """Load a state dict from an older SlayNet (no spatial memory) into the current one.

    Warm start: the fusion layer gained MAP_FEAT input columns; old columns are copied and the new ones
    are zero, so the loaded network computes exactly what it computed before, and training then learns
    how much to use the map. Returns the names of parameters that were newly initialised."""
    own = net.state_dict()
    fresh = []
    for k, v in own.items():
        if k not in sd:
            fresh.append(k)
            continue
        src = sd[k]
        if src.shape == v.shape:
            own[k] = src
        elif k == "fuse.0.weight" and src.shape[0] == v.shape[0] and src.shape[1] < v.shape[1]:
            w = torch.zeros_like(v)
            w[:, : src.shape[1]] = src
            own[k] = w
            fresh.append(k + "[map columns]")
        else:
            raise ValueError(f"incompatible parameter {k}: {tuple(src.shape)} vs {tuple(v.shape)}")
    net.load_state_dict(own)
    return fresh


# ----------------------------------------------------------------------------- policy
class SlayNet(nn.Module):
    def __init__(self, hidden: int = 256, use_masks: int = 0):
        super().__init__()
        self.hidden = hidden
        # Action masks are part of how a policy was trained: stored with the weights (checkpoint "masks").
        self.use_masks = int(use_masks)  # mask level 0/1/2 (see apply_action_mask)
        self.rays = RayEncoder(256)
        self.ents = EntityEncoder()
        self.vit = nn.Sequential(nn.Linear(VITALS_FEAT + PREV_ACT_DIM, 64), nn.ReLU())
        self.map = MapEncoder(MAP_FEAT)
        self.fuse = nn.Sequential(nn.Linear(256 + self.ents.out_dim + 64 + MAP_FEAT, hidden), nn.ReLU())
        self.gru = nn.GRUCell(hidden, hidden)
        self.actor = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(), nn.Linear(128, NUM_LOGITS))
        self.critic = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(), nn.Linear(128, 1))
        # Small init on the policy output -> near-uniform initial policy (good exploration start).
        nn.init.orthogonal_(self.actor[-1].weight, 0.01)
        nn.init.zeros_(self.actor[-1].bias)

    def initial_state(self, n: int, device=None) -> torch.Tensor:
        return torch.zeros(n, self.hidden, device=device)

    def encode(self, o: dict[str, torch.Tensor], prev_act: torch.Tensor) -> torch.Tensor:
        """o: tensors with a leading batch dim N. prev_act: (N, PREV_ACT_DIM)."""
        r = self.rays(o["ray_dist"], o["ray_type"])
        e = self.ents(o["ent_feat"], o["ent_type"], o["ent_mask"])
        v = self.vit(torch.cat([normalize_vitals(o["vitals"]), prev_act], -1))
        if "egomap" in o:
            m = self.map(o["egomap"])
        else:  # observation packed without spatial memory (old tools): neutral input
            m = torch.zeros(r.shape[0], MAP_FEAT, device=r.device, dtype=r.dtype)
        return self.fuse(torch.cat([r, e, v, m], -1))

    def step(self, o, prev_act, h):
        """Single decision step. Returns (logits (N,15), value (N,), new h)."""
        h = self.gru(self.encode(o, prev_act), h)
        logits = self.actor(h)
        if self.use_masks:
            logits = apply_action_mask(logits, o, prev_act, self.use_masks)
        return logits, self.critic(h).squeeze(-1), h

    def forward_sequence(self, o, prev_act, h0, starts):
        """Training: o tensors shaped (T,B,...), starts (T,B) = 1 where a new episode begins at t.
        GRU state is zeroed at episode starts (truncated BPTT over T steps)."""
        T, B = starts.shape
        flat = {k: v.reshape(T * B, *v.shape[2:]) for k, v in o.items()}
        x = self.encode(flat, prev_act.reshape(T * B, -1)).reshape(T, B, -1)  # encoders batched over T*B
        h = h0
        hs = []
        for t in range(T):
            h = h * (1.0 - starts[t]).unsqueeze(-1)
            h = self.gru(x[t], h)
            hs.append(h)
        hs = torch.stack(hs)  # (T,B,H)
        logits = self.actor(hs)
        if self.use_masks:
            logits = apply_action_mask(logits.reshape(T * B, -1), flat, prev_act.reshape(T * B, -1),
                                       self.use_masks).reshape(T, B, -1)
        return logits, self.critic(hs).squeeze(-1)


# ----------------------------------------------------------------------------- distributions
def split_logits(logits: torch.Tensor):
    return torch.split(logits, HEAD_SIZES, dim=-1)


# All 6 heads are evaluated in one vectorised pass: the flat logits (…,15) are gathered into a padded
# (…,6,3) tensor; 2-option heads get a -1e9 (never sampled) third option. Fewer tiny ops = faster CPU rollouts/agent.
_PAD_IDX = []
_off = 0
for _n in HEAD_SIZES:
    _PAD_IDX.append([_off + i if i < _n else NUM_LOGITS for i in range(3)])
    _off += _n
PAD_IDX = torch.tensor(_PAD_IDX)  # (6,3); index NUM_LOGITS = padding column
PAD_MASK = PAD_IDX == NUM_LOGITS  # (6,3) True where padded


def padded_logits(logits: torch.Tensor) -> torch.Tensor:
    # Large finite negative, not -inf: 0 * -inf = NaN would poison gradients of the entropy term.
    pad = torch.full(logits.shape[:-1] + (1,), -1e9, dtype=logits.dtype, device=logits.device)
    return torch.cat([logits, pad], -1)[..., PAD_IDX.to(logits.device)]  # (...,6,3)


def sample_actions(logits: torch.Tensor, greedy: bool = False, generator: torch.Generator | None = None) -> torch.Tensor:
    """(…,15) logits -> (…,6) head choices. Stochastic sampling uses the Gumbel-max trick."""
    pl = padded_logits(logits)
    if not greedy:
        u = torch.rand(pl.shape, generator=generator, device=pl.device).clamp_(1e-9, 1.0 - 1e-9)
        pl = pl - torch.log(-torch.log(u))
    return pl.argmax(-1)


def log_prob_entropy(logits: torch.Tensor, actions: torch.Tensor):
    """Joint log-prob (sum over independent heads) and joint entropy."""
    logp = F.log_softmax(padded_logits(logits), -1)  # (...,6,3), -inf at padding
    lp = logp.gather(-1, actions.unsqueeze(-1)).squeeze(-1).sum(-1)
    plogp = torch.where(PAD_MASK.to(logp.device), torch.zeros_like(logp), logp.exp() * logp)
    return lp, -plogp.sum((-1, -2))
