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

## Bots + learning surface

- Humans and bots share the same `Bot` protocol (`act()` + `observe_result()`).
- `observe_result()` is the hook for learning bots; NFSP trains through
  `poker.ml.nfsp_loop.consume_hand`, not through that hook.

## ML environment

- `PokerEnv.step()` is wired to the engine and returns the full Gymnasium tuple.
- Two observation encoders: v1 (98 floats, legacy) and v2 (318 floats, hero-relative).
- Current checkpoints: NFSP `models/nfsp_v2/nfsp_v2.pt`, SD-CFR
  `models/sdcfr_prod/sdcfr_ckpt_500.pt`.

