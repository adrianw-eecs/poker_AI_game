# Project Status

This repo is a **No-Limit Texas Hold'em** engine in Python with a working ML training pipeline.

---

## What Works Today

### Core Engine
- Multi-hand **session runner** with CLI and text UI for human play
- Deterministic, testable core logic — 128 tests, ~85% coverage on core engine
- **Betting model**: per-street rounds with correct half-raise / all-in re-open logic
- **Side pots + rake + odd chip** distribution
- **Run-it-twice** (heads-up all-in, configurable)
- **Rebuy system** (resets eliminated players between hands)
- Performance: ~1.4 ms/hand (8 players), 40–60% faster than original via:
  - Diagnostics disabled by default (`SessionConfig.enable_diagnostics = False`)
  - Lazy showdown evaluation (pre-convert community cards once, fast path for single winner)
  - Event I/O batching in `GameLogger` (30+ events → 1–2 disk writes per hand)

### ML / RL Environment
- `PokerEnv` (Gymnasium-compatible) with `step()` fully wired to the engine
- Observation: `(142,)` float32 vector — cards, stacks, positions, action history
- Action space: Discrete(7) — fold, check/call, 5 raise buckets
- Action masking: `get_action_mask()` returns `(7,)` legal-action mask
- Reward: `10× (stack_change / starting_stack)` at hand end; 0 during hand
- NFSP and SD-CFR model implementations verified correct

### Training Pipeline
- `scripts/train_nfsp_v3.py` — sequential training, ~13 min for 250K episodes
- `scripts/train_nfsp_parallel.py` — 4-worker parallel training, ~4 min for 250K episodes (3–4× speedup)
- `scripts/validate_training.py` — validate any checkpoint against multiple opponents
- Reward design fixed (v3): pure game rewards, no artificial anti-fold bonuses

---

## Run the CLI Game

```bash
pip install -e ".[dev]"
python -m poker.main --help
python -m poker.main -n 2 -b human random
python -m poker.main -n 3 -b flop_bot random random
```

## Run Tests

```bash
# Smoke suite only (fast, ~30 seconds)
pytest -m smoke

# Full suite (~128 tests, < 2 minutes)
pytest
```

---

## Code Map

| Module | What lives here |
|--------|----------------|
| `poker.main` | CLI session runner |
| `poker.engine` | Hand/session orchestration, betting, validation, showdown |
| `poker.state` | Immutable `GameState` + pot/player state helpers |
| `poker.evaluation` | Hand evaluator + equity helpers |
| `poker.bots` | Built-in bots + bot protocol |
| `poker.logging` | JSONL event logging + replay helpers |
| `poker.ml` | Gymnasium env, observation encoder, action space, NFSP/SD-CFR models |

---

## Rules Implemented

- **Min-raise**: `current_bet_to_call + max(big_blind, last_raise_size)`
- **All-in re-open**: Only re-opens if all-in adds at least a full min-raise increment
- **Antes**: Posted before blinds each hand
- **Blinds**: Fixed or escalating via `BlindSchedule`
- **Rake**: Configurable percent + optional cap; applied per pot (main + side pots)
- **Side pots**: Created whenever a player is all-in for less than the current bet
- **Odd chip**: Goes to first eligible seat left of the dealer button

---

## Critical Bug History

### Betting Round Infinite Loop (Fixed)
`_round_is_closed()` in `betting_round.py` didn't handle side pot players correctly.
A player all-in for less than `max_commitment` kept the round open forever.

**Fix**: Check `total_available = stack + committed_this_street` before returning False.

### Hand Numbering Cycling (Fixed)
Secondary effect of the betting round infinite loop — fixed by fixing root cause.

### Multiple Actions in Single Round (Fixed)
Player could act multiple times when all others were all-in or folded.
**Fix**: Single-active-player check in betting round prevents cycling back.

### ML Reward Misalignment (Fixed)
Anti-folding bonuses (`-0.01 FOLD, +0.005 PLAY`) were mathematically insufficient,
causing models to learn 100% fold. Fixed in v3: pure game rewards at 10× scale.

---

## Known Limitations

- **Persistence depth**: Session JSON saver is summary-only; use JSONL event log for replay
- **No global seed flag** in CLI (RNG is seedable programmatically via `poker.rng.RNG`)
- **Batch `replace()` optimization** reverted for stability (betting round control flow is complex)

---

## Next Steps

1. **Multi-generation population training** — run Gen 1+ with `--generation N`
2. **Tournament mode** — track elimination order, prize pools, SNG/MTT formats
3. **Timing instrumentation** — `TimingEvent` logging for bottleneck identification
4. **Equity display** — real-time equity for human play UI
