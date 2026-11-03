# User notes (stable decisions)

Keep this file short: only "sticky" decisions that shape the implementation.

## Goal

Build a **No-Limit Texas Hold'em** engine in Python that can be used for:

- CLI play / simulation
- self-play training for ML agents

## Stack

- **Python**: 3.11+
- **Core engine**: no ML deps required to run the CLI/session
- **ML**: PyTorch

## Rules/behavior (as implemented today)

- **Players**: 2–9 in the ML environment (nine seat slots, zero-filled)
- **Antes**: supported; posted before blinds each hand
- **Blinds**: fixed or escalating via `BlindSchedule`
- **Rake**: configurable percent + optional cap; applied per pot (main + side pots)
- **Run-it-twice**: supported when enabled, heads-up, both all-in, and streets remain
- **Min-raise**: `max(big_blind, last_raise_size)` as the raise increment
- **All-in re-open**: re-opens only if it adds at least a full min-raise increment

Known deviation: a player facing a short all-in is still offered a raise.
`reopens_betting()` is correct but `legal_actions()` does not consult it.

## Configuration

`GameConfig` (`poker/config/game_config.py`) has exactly these fields:

| Field | Meaning |
|---|---|
| `num_players` | 2–10 |
| `starting_stack` | Chips per player at hand start |
| `small_blind`, `big_blind` | Blinds |
| `ante` | Flat per-player ante, posted before blinds |
| `rake_percent`, `rake_cap` | Rake percent and optional per-hand cap |
| `blind_schedule` | A `BlindSchedule` instance |
| `run_it_twice` | Heads-up all-in only |
| `tournament_mode` | `TournamentMode.SNG` / `MTT` / `None` for cash |
| `final_table_size` | MTT final-table notification size |

`BlindSchedule` (`poker/config/blind_schedule.py`) has:

| Field | Meaning |
|---|---|
| `levels` | Sequence of `BlindLevel(small_blind, big_blind, ante)` |
| `hands_per_level` | Hands before escalation |
| `fixed` | `True` = cash-game mode, ignore escalation |

```python
from poker.config.blind_schedule import BlindLevel, BlindSchedule
from poker.config.game_config import GameConfig

config = GameConfig(
    num_players=6,
    starting_stack=1000,
    small_blind=25,
    big_blind=50,
    ante=0,
    rake_percent=0.0,
    rake_cap=None,
    blind_schedule=BlindSchedule(
        levels=[BlindLevel(25, 50, 0)],
        hands_per_level=10**9,
        fixed=True,
    ),
    run_it_twice=False,
)
```

## Bots + learning surface

- Humans and bots share the same `Bot` protocol (`act()` + `observe_result()`).
- `observe_result()` is the hook for learning bots; NFSP trains through
  `poker.ml.nfsp_loop.consume_hand`, not through that hook.

## ML environment

- `PokerEnv.step()` is wired to the engine and returns the full Gymnasium tuple.
- Two observation encoders: v1 (98 floats, legacy) and v2 (318 floats, hero-relative).
- Current checkpoints: NFSP `models/nfsp_v2/nfsp_v2.pt`, SD-CFR
  `models/sdcfr_prod/sdcfr_ckpt_500.pt`.

