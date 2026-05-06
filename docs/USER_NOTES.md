## User notes (stable decisions)

Keep this file short: only “sticky” decisions that shape the implementation.

## Goal

Build a **No-Limit Texas Hold’em** engine in Python that can be used for:

- CLI play / simulation
- future **self-play** training for ML agents

## Stack

- **Python**: 3.11+
- **Core engine**: no ML deps required to run the CLI/session
- **ML (future training)**: PyTorch is acceptable when we start training loops

## Rules/behavior (as implemented today)

- **Players**: 2–10 (CLI and engine validate this)
- **Antes**: supported (`--ante`); posted before blinds each hand
- **Blinds**: fixed or escalating via `BlindSchedule`
- **Rake**: configurable percent + optional cap; applied to contested pots (main + side pots)
- **Run-it-twice**: supported when enabled, heads-up, both all-in, and streets remain
- **Min-raise**: based on `max(big_blind, last_raise_size)` for the raise increment
- **All-in re-open**: only re-opens betting if it adds at least a full min-raise increment

## Bots + learning surface

- Humans and bots share the same `Bot` protocol (`act()` + `observe_result()`).
- `observe_result()` is the intended hook for learning bots, but training loops are not yet built.

## ML environment (current constraint)

- `poker.ml.env.PokerEnv.step()` is not wired up yet; `reset()`/obs/mask utilities exist.

