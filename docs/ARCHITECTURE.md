# Architecture

Project layout, current design, and proposals for refactoring and optimisation.

---

## 1. Package layout

```
poker/
├── domain/        # Card, Rank, Suit, Deck, Action, HandRank — no deps
├── state/         # Immutable GameState, PlayerState, Pot, pot_manager
├── engine/        # rules, betting_round, hand_engine, showdown, dealer,
│                  #   action_validator, action_handler, session
├── evaluation/    # Hand evaluator, equity helpers, data/preflop_equity.npy
├── bots/          # random, flop, call, human, nfsp, sdcfr, linear/tree/deep
├── ml/            # env, observations (v1 + v2), action_space, models/, cfr/
├── logging/       # events, logger, replay, transcript, session_text_logger
├── config/        # GameConfig, BlindSchedule
├── interface/     # text_ui
├── persistence/   # game_saver
├── stats/          # aggregator, game_analyzer
└── main.py        # CLI entry point
```

Dependency direction is clean and mostly one-way:
`domain ← state ← engine ← {bots, ml, interface, logging}`.
`engine/rules.py` was extracted specifically so `ml/env.py` and `engine` share
the same betting rules and cannot drift apart.

---

## 2. Engine: observations and proposals

### 2.1 `legal_actions` is rebuilt several times per decision

Verified call sites in the decision path:

```
ml/env.py:493            legal_actions(self.state, seat)     # mask build
ml/action_space.py:223   legal_actions(state, seat)          # _mask_pot_fraction
ml/action_space.py:340   legal_actions(state, seat)          # action_index_to_action
ml/action_space.py:424   legal_actions(state, seat)          # action_to_action_index
```

`get_action_mask()` → `build_action_mask()` → `_mask_pot_fraction()` →
`legal_actions()`, and then `step_multi()` → `action_index_to_action()` →
`legal_actions()` again. For each bucket under `POT_FRACTION` there is an
additional `pot_fraction_raise_to()` → `capped_raise_bounds()` →
`min_raise_to()` chain plus a `validate()` call.

**Proposal.** Compute the legal action set **once** per decision and pass it
down. Either memoise on `GameState` (it is a frozen dataclass, so `id(state)` is
a safe key) or thread a `LegalActions` object through the mask → action-index
conversion. This is pure allocation removal on the hottest path in training and
should be the first optimisation attempted.

### 2.2 Frozen dataclass `replace()` churn

`GameState` is frozen, and `rules.apply_action` rebuilds the player tuple plus
both history lists on every action. `rebuild_pots` then reconstructs the whole
pot list from scratch. In a training loop that runs millions of hands this
dominates allocation cost.

**Proposal.** Keep immutability at the API boundary but make `committed_this_hand`
accounting incremental. `build_pots` currently iterates every unique commitment
level for every call; it could be cached on the state and invalidated only when a
commitment changes. Do not remove immutability — the determinism and replay
guarantees depend on it.

### 2.3 `pot_manager.build_pots` allocation

`build_pots` allocates a `frozenset` per level per call. With 9 players and
several side pots this is repeated on every `rebuild_pots`, which is called after
*every* action (`betting_round.py`) and again at end of round.

**Proposal.** Same cache as 2.2, or pre-size the structures. Low risk, measurable
gain.

### 2.4 Missing short-all-in raise suppression

Documented in `ENGINE.md` §Known Rule Deviation. `reopens_betting()` computes the
correct answer but `legal_actions()` does not consume it.

**Proposal.** Thread a `betting_reopened: bool` into `legal_actions()` and drop
`Action.raise_to(...)` when the last aggressive action was a short all-in. This is
a correctness fix, not an optimisation, and it changes the strategy space the
models learn — retrain after applying it.

### 2.5 Showdown evaluator

`evaluate()` enumerates all C(7,5) = 21 combinations and calls `_evaluate_hand`
on each. Correct, but 21× redundant work per showdown.

**Proposal.** Add a direct 7-card evaluator (count ranks, detect flush suits,
check straight windows) as a fast path, keeping the combination version as the
reference implementation for tests. Verify equivalence against the current
implementation over random 7-card sets before switching.

---

## 3. ML: observations and proposals

### 3.1 Two observation encoders is the main source of coupling

`v1` (98 floats) and `v2` (318 floats) both exist, and the model has to branch on
`obs_version` in `select_action`, `_masked_q`, `_update_q`, `load`, and the v2
trainers. The fold-return loss only runs under v2. This branching is the largest
ongoing maintenance cost in `ml/`.

**Proposal.** Retire v1 for *training* and keep it only as a frozen inference
shim for old checkpoints. Concretely: make `build_observation_v2` the single
encoder, and give `SDCFRModel` an `obs_version` parameter so the last v1
consumer can migrate. This removes every `if self.obs_version == 2` branch in
`nfsp_model.py`.

**Blocker.** SD-CFR must be retrained on v2 first — its weights are tied to the
98-d encoder. See `HYBRID_PLAN.md` §3.

### 3.2 Per-decision forward-pass cost

In `select_action`, a decision costs one policy forward (average mode) or one Q
forward (best-response mode). Under batch size 1 over a 700K-parameter network the
GPU is idle most of the time.

**Proposal.** The highest-leverage speedup is **batched inference across parallel
hands**. Run N independent hands concurrently in the env and batch their
observations into one forward pass. This turns ~1 decision/forward into
~N decisions/forward. It is a much larger change than it sounds — `consume_hand`
assumes one hand at a time — but it is the only change that will move throughput
by an order of magnitude rather than a few percent.

A cheaper intermediate: collect decisions for a hand, then batch the *training*
updates more aggressively (`updates_per_train` already exists). This improves GPU
utilisation without restructuring the env.

### 3.3 float16 buffer storage

v2 buffers already store float16 and upcast on sample, which halves buffer
memory. Worth keeping. Note `sample()` always returns float32, so the conversion
cost is paid per batch — acceptable, but if profiling shows it, converting once
at sample time into a preallocated array would help.

### 3.4 `_step` counts decisions, not episodes

`train_step()` increments `_step` on every call, and the trainers call it per
*decision*. `_q_min` (the warm-up gate) and the `inv_sqrt` epsilon schedule both
read `_step`. Consequences:

- The warm-up gate opens after `max(batch_size, q_capacity // 100)` **decisions**,
  not episodes. With `q_capacity=1_000_000` that is 10,000 decisions — a small
  fraction of one episode's worth of data from a 9-handed table.
- Epsilon decays on the decision clock, so with `train_every=64` the schedule is
  effectively 64× slower than a per-episode reading would suggest.

**Proposal.** Separate the two counters: `self._step` for epsilon (decisions) and
`self._episodes` for the warm-up gate. Or, more simply, document the current
semantics precisely and tune `epsilon_decay_steps` against decisions. Do this
deliberately — it changes all training curves, so it must be a versioned change.

### 3.5 Duplicated BB/100 conversion

`poker/ml/metrics.py` owns `reward_to_bb100`, but `train_nfsp_selfplay.py` and
`validate_training.py` each carry their own formatting/interval helpers. They
have stayed consistent so far, but this is exactly the shape of a silent metric
divergence between training and evaluation.

**Proposal.** Make `metrics.py` the single source and delete the local copies.

---

## 4. Logging and evaluation

`logging/events.py` is ~30 KB of event dataclasses, and `GameLogger` batches
writes. Training runs mostly use `NullLogger`, so the cost is avoided in the hot
path — good.

**Proposal.** Nothing urgent. If profiling ever shows logging cost, gate event
*construction* (not just the write) behind `NullLogger.is_null`, since building
the dataclass is not free.

---

## 5. Test suite

235 tests, 36 marked smoke. The suite runs in ~10 s, which is fast enough that
the smoke/full split is barely necessary — but keep it, since it lets a
pre-commit hook stay under a second.

**Proposal.** Add a property test (Hypothesis is already a dev dependency) that
asserts chip conservation across randomly generated states. The manual audit found
zero leaks across ~1,200 hands, but a property test would catch future
regressions deterministically rather than by sampling.

---

## 6. Priority order

Ranked by value ÷ risk:

| # | Change | Effort | Risk | Payoff |
|---|---|---|---|---|
| 1 | Single `legal_actions` per decision | S | Low | Hot-path allocation |
| 2 | Consolidate metrics into `metrics.py` | S | Low | Correctness |
| 3 | Property test for chip conservation | S | Low | Regression safety |
| 4 | Fast 7-card evaluator | M | Low | Showdown throughput |
| 5 | Short-all-in raise suppression | M | Medium | Rule correctness |
| 6 | Retire obs v1 for training | L | Medium | Removes branching |
| 7 | Batched multi-hand inference | L | High | Order-of-magnitude throughput |

Do 1–3 immediately; they are cheap and safe. Items 6–7 are the real wins but
should follow the hybrid work, not precede it.
