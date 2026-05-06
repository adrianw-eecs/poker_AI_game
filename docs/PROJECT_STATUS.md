# Project status (current)

This repo is a **No-Limit Texas Hold’em** engine in Python with:

- A working multi-hand **session runner** (CLI)
- A text UI for human play (same bot interface as AIs)
- Deterministic, testable core logic (broad unit/integration coverage in `tests/`)
- A partially implemented **Gymnasium environment** (`poker.ml`) intended for RL/self-play

## Run the CLI game

From the repo root:

```bash
pip install -e ".[dev]"
python -m poker.main --help
python -m poker.main -n 2 -b human random
```

Notes:
- The CLI entrypoint is `src/poker/main.py` (module `poker.main`).
- Seats are configured with `--bots`, e.g. `-b human flop_bot random`.

## What gets written to disk

- **Event log (optional)**: `--log <path>` writes **JSONL** events via `poker.logging.logger.GameLogger`.
- **Invalid bot actions log**: always created under `games/AI_games/` as a timestamped `game_*.txt`
  (written by `poker.engine.action_handler.ActionHandler`).
- **Session summary JSON**: `games/AI_games/game_*.json` via `poker.persistence.save_game_session`.
  - This is currently a **summary only** (final stacks, players, hand_count). It does *not* include full per-hand history yet.

## Rules implemented (high level)

- **Blinds + ante**: posted at hand start (`hand_engine._post_antes`, `_post_blinds`)
- **Betting model**: per-street betting rounds (`poker.engine.betting_round.BettingRound`)
- **Legal actions**: fold/check/call/raise/all-in (`poker.engine.action_validator.legal_actions`)
- **Side pots + rake + odd chip**: handled by `poker.state.pot_manager` and `poker.engine.showdown`
- **Run-it-twice**: supported when `config.run_it_twice=True`, exactly 2 players remain, both all-in, and streets remain

Important current details:
- **Min-raise**: computed as `current_bet_to_call + max(big_blind, last_raise_size)` (see `action_validator.py`).
- **All-in “re-open” logic**: betting round re-opens only when the all-in adds at least a full min-raise increment.
  (There is no separate “50% rule” threshold implemented yet.)

## Known gaps / limitations

- **ML env stepping**: `poker.ml.env.PokerEnv.step()` is intentionally `NotImplementedError` today.
  Observations and legal-action masking exist, but gameplay is not yet wired to the engine.
- **Persistence depth**: the JSON session saver currently can’t reconstruct per-hand detail from a single final `GameState`.
- **Reproducibility knobs**: RNG is seedable (`poker.rng.RNG`), but the CLI currently does not expose a global seed flag.

## Code map (where things live)

- `poker.main`: CLI session runner
- `poker.engine`: hand/session orchestration, betting, validation
- `poker.state`: immutable `GameState` + pot/player state helpers
- `poker.evaluation`: hand evaluator + equity helpers
- `poker.bots`: built-in bots + bot protocol
- `poker.logging`: JSONL event logging + replay helpers
- `poker.ml`: observation + action-mask utilities and a stub Gymnasium env

