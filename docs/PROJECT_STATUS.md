# Project Status

No-Limit Texas Hold'em engine in Python with a working ML training pipeline.

---

## Current Models (verified 2026-11-03)

Only two checkpoints are kept. Everything else was deleted.

| Role | Path | Obs | Architecture | Step / iter |
|---|---|---|---|---|
| NFSP (current) | `models/nfsp_v2/nfsp_v2.pt` | v2, 318-wide | `PokerNetV2` (card embeddings + dueling head) | step 87,020 |
| SD-CFR (current) | `models/sdcfr_prod/sdcfr_ckpt_500.pt` | v1, 98-wide | `PokerNetwork` (residual MLP) | iteration 499 |

### On the "v3" naming

There is **no NFSP v3 checkpoint**. The old `nfsp_v3.pt` / `nfsp_v3_improved.pt` files were deleted:
they dated from 2026-05-14, used the retired 155-wide observation, carried no `model_meta`,
and predated the current architecture. "v3" in the training docs refers to a **reward-design
iteration** (dropping the artificial anti-fold bonus), not a newer model.

`nfsp_v2` is newer than those files on every axis that matters — mtime, observation
version, and architecture. It is the model to use.

### Why SD-CFR stays on the legacy encoder

`SDCFRModel` is hard-wired to `_OBS_DIM = 98` and `BetScheme.LEGACY`, and
`SDCFRBot` builds its observation with `build_observation` (v1) and its mask with
the legacy bucket scheme. Migrating SD-CFR to obs v2 is the main prerequisite
for the hybrid design in [`HYBRID_PLAN.md`](HYBRID_PLAN.md).

---

## What Works

### Core engine
- Multi-hand session runner with CLI and text UI for human play
- Deterministic, testable core logic — 235 tests passing
- Betting model: per-street rounds with correct half-raise / all-in re-open logic
- Side pots + rake + odd-chip distribution
- Run-it-twice (heads-up all-in, configurable)
- Rebuy system between hands

### ML environment
- `PokerEnv` (Gymnasium-compatible), `step()` fully wired to the engine
- Observation v1: 98 floats; observation v2: 318 floats, hero-relative, with
  card ids in their own block and 9 seat slots
- Action space: `Discrete(7)` — fold, check/call, 5 raise buckets
- `REWARD_SCALE = 0.5`; reward is `REWARD_SCALE * stack_change / starting_stack`

---

## Run the CLI Game

```bash
pip install -e ".[dev]"

python -m poker.main -n 6 --hands 20 --seed 1 \
  --bots human nfsp:models/nfsp_v2/nfsp_v2.pt \
         sdcfr:models/sdcfr_prod/sdcfr_ckpt_500.pt random random flop
```

## Evaluate a Table

```bash
python scripts/validate_training.py --hands 200 --seed 1 \
  --bots nfsp:models/nfsp_v2/nfsp_v2.pt sdcfr:models/sdcfr_prod/sdcfr_ckpt_500.pt random flop
```

## Run Tests

```bash
pytest -m smoke   # fast suite, ~4 s
pytest            # full suite, 235 tests, ~10 s
```

---

## Code Map

| Module | Contents |
|---|---|
| `poker.main` | CLI session runner |
| `poker.engine` | Hand/session orchestration, betting, validation, showdown, pure rules |
| `poker.state` | Immutable `GameState` + pot/player state helpers |
| `poker.evaluation` | Hand evaluator + equity helpers |
| `poker.bots` | Built-in bots + bot protocol |
| `poker.logging` | JSONL event logging + replay helpers |
| `poker.ml` | Gymnasium env, observation encoders, action space, NFSP/SD-CFR models |

### Scripts

| Script | Purpose |
|---|---|
| `train_nfsp_selfplay.py` | NFSP v1 training (98-wide, one learning seat) |
| `train_nfsp_v2.py` | NFSP v2 training (318-wide, pot-fraction, multi-seat) |
| `train_sdcfr_v2.py` | SD-CFR training on obs v2 (produces the hybrid teacher) |
| `train_hybrid.py` | NFSP + SD-CFR hybrid: distillation driver |
| `train_sdcfr_selfplay.py` | SD-CFR training on the legacy encoder |
| `train_sdcfr.py`, `train_sdcfr_extended.py` | Earlier SD-CFR runs |
| `validate_training.py` | Single-model sweep, or `--bots` for a full scored table |
| `profile_training.py` | Timing profiler for the training loop |
| `play_interactive.py` | Interactive play harness |
| `build_preflop_equity_table.py` | Regenerates `poker/evaluation/data/preflop_equity.npy` |
| `validate_game.py`, `validate_poker_logic.py`, `verify_game_logic.py` | Engine rule validators |

> `preflop_equity.npy` is a generated artifact but is committed because
> observation v2 depends on it. If it is missing, `_preflop_equity()` silently
> returns `0.0` instead of failing, so a lost table degrades the v2 encoder
> without an error. Rebuild with `python scripts/build_preflop_equity_table.py`.

---

## Rules Implemented

- **Min-raise**: `current_bet_to_call + max(big_blind, last_raise_size)`
- **All-in re-open**: re-opens only when the all-in adds at least a full min-raise increment
- **Antes** posted before blinds; blinds fixed or escalating via `BlindSchedule`
- **Rake** percent + optional cap, applied per pot
- **Side pots** created whenever a player is all-in for less than the current bet
- **Odd chip** goes to the first eligible seat left of the dealer button

### Short all-in

A short all-in that does not complete a full raise does not re-open betting.
`legal_actions()` consults `may_raise()` / `reopens_betting()`: a player who
has already acted may only call or fold, and an all-in above the current bet
is withheld. A player who has not acted yet may still raise. The first wager
of a street is not treated as an incomplete raise.

---

## Known Limitations

- The `nn` seat in `poker.main` is reserved and not implemented.
- Session JSON is summary-only; use the text transcript or JSONL log to replay a hand.
- Tournament semantics are absent: no ICM, no payout structure, no hand-for-hand,
  and blinds escalate by hand count rather than by clock.
- Antes are flat per-player, not big-blind-ante.
- Run-it-twice averages the two boards (`//2`) rather than splitting the pot,
  which is a different game from what live rooms do.

---

## Next Steps

1. Migrate SD-CFR to observation v2 + `BetScheme.POT_FRACTION`.
2. Implement SD-CFR-as-teacher distillation into NFSP (see [`HYBRID_PLAN.md`](HYBRID_PLAN.md)).
3. Retrain NFSP on the v2 encoder across 2–9 players.
