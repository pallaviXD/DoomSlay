# SPEC — Team Slay working spec (derived from docs/CONTRACT.md 1.0.0-draft + env/doom_env source)

If anything here disagrees with `docs/CONTRACT.md` or `env/doom_env/`, those win.

## Interface
`submission/agent.py` → `class DoomAgent`:
- `__init__(observation_space, action_space, device)`: once per run, ≤ 60 s. Load weights relative to
  `Path(__file__).parent`. `device` is `"cuda"` or `"cpu"`.
- `reset()`: before every map, ≤ 10 s. Clear GRU state and all memory.
- `act(obs) -> np.ndarray` shape `(9,)`, values in {0,1}. First act per map ≤ 10 s, others ≤ 5 s,
  **total agent compute per map ≤ 40 ms × decision steps**. Invalid action ⇒ agent stops (stands still).
- Repo read-only, scratch only in `/tmp`, no network, 1 CPU thread, 2 GB VRAM (torch memory fraction set
  right after `__init__`), 8 GB RAM.

## Observation (`gymnasium.spaces.Dict`)
| key | shape / dtype | notes |
|---|---|---|
| `rgb` | (120,160,3) uint8 | HUD hidden |
| `depth` | (120,160) uint8 | |
| `vitals` | (17,) float32 | 0 health(0–200), 1 armor(0–200), 2 selected_weapon(0–7), 3 selected_weapon_ammo, 4–7 ammo bullets/shells/rockets/cells, 8–13 keys blue/yellow/red card, blue/yellow/red skull (0/1, sticky), 14–16 vel forward/right/up (units/tic, ego frame) |
| `rays.distance` | (64,) float32 in [0,1] | dist/1024, 1.0 = nothing hit. Ray 0 = ahead, CCW, ray 16 = 90° left, 32 = behind |
| `rays.hit_type` | (64,) int MultiDiscrete(6) | 0 none, 1 wall, 2 door, 3 locked door, 4 enemy, 5 item |
| `visible_entities.features` | (32,5) float32 | [dist/1024, sin(rel_angle), cos(rel_angle), rel_vel_fwd, rel_vel_right]; only within 90° FOV and line of sight, ≤ 4096 units, nearest first, zero-padded |
| `visible_entities.entity_type` | (32,) int MultiDiscrete(12) | 0 pad, 1 enemy_melee, 2 enemy_hitscan, 3 enemy_projectile, 4 projectile, 5 health, 6 armor, 7 ammo, 8 weapon, 9 key, 10 barrel, 11 other |
| `visible_entities.mask` | (32,) MultiBinary | 1 = real entity |

No absolute position or heading is ever observed. Rays stop at walls, steps > 24, openings < 56,
alive monsters, pickups; closed doors report door/locked_door; open doors are transparent.

## Action
`MultiBinary(9)`: 0 FORWARD, 1 BACKWARD, 2 STRAFE_LEFT, 3 STRAFE_RIGHT, 4 TURN_LEFT, 5 TURN_RIGHT,
6 ATTACK, 7 USE, 8 SELECT_NEXT_WEAPON. Buttons held for all 4 tics.
- `USE` and `SELECT_NEXT_WEAPON` trigger on the **rising edge** only → the policy must release
  (output 0) for ≥ 1 step to press again.
- `USE` on a moving door reverses it → press once, wait 1–2 s.

## Timing
- Frame skip 4 tics per `act()` (35 tics = 1 s ⇒ ~8.75 decisions / s game time).
- Episode cap 2,100–4,200 tics (525–1,050 decisions). `terminated` = exit or death, `truncated` = cap.

## Game
IWAD `freedoom2.wad`, skill 3, ViZDoom 1.3.3 vanilla. Maps may be Doom/Boom/Hexen/UDMF format; the
provided ones are UDMF `namespace="zdoom"`:
- door: linedef `special=12` (Door_Raise, arg0 tag 0 = manual, arg1 speed, arg2 delay), `playeruse`, `repeatspecial`
- locked door: `special=13` (Door_LockedRaise, arg3 = lock: 1 red, 2 blue, 3 yellow) (per `mapinfo.HEXEN_DOORS`)
- exit: `special=243` (Exit_Normal) with `playercross` (walk-over) or `playeruse` (switch)
- things: 1 player start, 3004 zombieman, 9 shotgun guy, 3001 imp, 3002 demon, 5/6/13 blue/yellow/red card,
  2011 stimpack, 2012 medikit, 2007 clip, 2008 shells, 2048 ammo box

## Scoring (hidden weights)
Per map: exit reached (largest, + efficiency bonus for health/time) > progression (keycards, doors,
unlocks) > exploration (unique areas, sector coverage) > combat (kills, small penalties for damage/early death).
Tiers 1 combat, 2 exploration, 3 key → locked door → exit, 4 gauntlet; harder tiers weigh more.

## Training-only API
`env.privileged_state()` → x, y, z, angle, health, kills, items, damage_taken, keys, dead, timeout,
exit_reached, sector_floor/ceiling, door_lock. Used only for reward shaping and metrics.
`make_env(wad, map, episode_ticks=..., render_buffers=False)` skips pixel copies (~900 steps/s/env measured).
