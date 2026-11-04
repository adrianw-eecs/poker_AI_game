# NFSP: Observation Gaps and Training Improvements

> **Status (2026-11-03): items 1–7 are implemented.** See §6 for what shipped,
> how to enable or disable each, and the two measurement questions that remain
> open. The analysis below is kept as the rationale.

An audit of what the v2 encoder captures, what it misses, and what would most
reduce training time or improve final performance. Ordered by expected impact.

---

## 1. What the encoder already does well

Worth stating first, because several "obvious" additions are already present:

- **Made-hand strength** — 9-way one-hot via `_made_hand_index`, plus draw flags
  (flush draw, OESD, gutshot, combo draw).
- **Preflop equity** — a 169-class × 8-opponent table, so preflop hand strength is
  measured rather than inferred.
- **Board texture** — paired, trips, three/four/five-flush, monotone, two-tone.
- **Pot odds** — `call_chips / pot`, `call_chips / (pot + call_chips)`, SPR,
  effective stack, min-raise size relative to pot.
- **Position** — dealer/SB/BB flags per seat, hero's relative seat to dealer.
- **Per-street action summary** — raise/call/fold/check counts, last aggressor,
  largest bet relative to pot, actions taken.
- **Recent action history** — last 6 actions with seat, type, size, street.

That is a genuinely strong state representation, and it is why the model reached a
positive win rate against Random/Flop bot opponents. The gaps below are
incremental relative to this base.

---

## 2. Gaps that likely matter

### 2.1 No "players still to act behind me" feature

Verified: the only count-like feature is `active / MAX_SEATS`
(`observation_v2.py:433`), which is *total* live players. There is no signal for
how many players have yet to act after the hero on the current street.

This is a standard and material preflop/postflop input. Facing a bet with four
players behind is a different decision from facing the same bet with one. The
model currently has to infer it from seat positions and action history, which is
indirect and, with only 6 action slots, gets truncated in long hands.

**Proposal.** Add two scalars:
- `players_to_act_after_hero / MAX_SEATS`
- `aggressor_is_hero` and `hero_is_last_to_act` (boolean flags)

Cheap to compute, replaces information currently lost to truncation.

### 2.2 Action history truncation at 6 slots

`ACTION_SLOTS = 6`, but a 9-handed preflop with a raise and callers routinely
exceeds six actions. The tail is kept (`indexed[-ACTION_SLOTS:]`), so early
actions silently vanish. The per-street *summary* partially compensates, but the
specific sequence does not.

**Proposal.** Increase to 10–12 slots, or add a compact per-street
"raise count, last raise size" pair so the information survives truncation
regardless of hand length. Prefer the latter — it is bounded and does not grow
the input for the long-handed case.

### 2.3 No opponent-tendency features

The model sees a single opponent's *stack and last action*, but nothing about how
that opponent has been playing. Against a fixed opponent pool (Random, Flop,
Call, frozen SD-CFR generations) this is a real loss — opponent-specific
adaptation is unavailable, so the model must learn an average over all opponents.

This is exactly the gap the adaptive-NFSP preprint in `NFSP_Notes.md` §7 targets,
and where it reports ~80% of its gain came from.

**Proposal.** Per-seat rolling aggregates over the last N decisions:
- VPIP (voluntarily put chips in pot)
- PFR (preflop raise frequency)
- aggression factor (raises / calls)
- fold-to-bet frequency

These are cheap to maintain in the env and add 4 × seats features. They are the
single change most likely to convert "break-even against SD-CFR" into a positive
number, because they let the model exploit rather than merely equilibrate.

### 2.4 No explicit hand-strength percentile on later streets

`_hero_block` has preflop equity but no postflop equivalent. The model gets
`_made_hand_index` (a *category* one-hot), which is coarse — top pair on a
four-flush board is not the same as top pair on a dry board.

**Proposal.** Add a Monte Carlo equity estimate sampled at decision time on
postflop streets, or a cheaper proxy: number of opponent hole-card combinations
that beat the hero's current made hand given the board. Budget permitting, a
small fixed-trial MC estimate (e.g. 50 trials) against a random opponent hand
range is the standard approach.

Note this is the most expensive item here per-decision. Measure before committing.

### 2.5 No time/stack-depth context

Stack depth in big blinds changes strategy fundamentally (15bb vs 200bb). The
encoder has `log1p(stack/bb)` per seat, which partially covers it, but no explicit
effective-SPR class or "short-stack mode" flag.

**Proposal.** Already largely covered by `feats[8]` (log SPR) and `feats[9]`
(saturated effective stack / pot). Low priority; the existing features are close.

---

## 3. Training-time improvements

### 3.1 Warm-up gate counts decisions, not episodes

Verified: `train_step()` increments `_step` per call, and trainers call it per
*decision*. `_q_min = max(batch_size, q_capacity // 100)` therefore measures
decisions. With `q_capacity = 1_000_000` the gate opens after 10,000 decisions —
possibly mid-way through the first few hands, before the buffer holds a diverse
sample.

**Impact.** Early Q updates are trained on a highly correlated, few-hand sample.
That is a plausible contributor to the BR-proxy instability observed in
`eval_history.csv`.

**Proposal.** Gate on distinct **episodes/hands**, not decisions. Add
`self._hands_completed` incremented at hand end (the `HandResult` already knows
when a hand finishes).

### 3.2 Epsilon schedule is on the decision clock

Same root cause. `epsilon_decay_steps = 300_000` with `train_every = 64` means
epsilon reaches its floor after ~300K decisions, which at ~4 decisions/hand/seat
is ~75K hands. Document this, or convert the schedule to hands.

### 3.3 Batch size vs. update frequency

`batch_size = 2048` with `updates_per_train = 2` and `train_every = 64`. Only two
gradient steps back-propagate over a 2048-sample batch per 64 decisions — roughly
32 decisions per gradient sample. The replay ratio is low, so the network sees
each transition rarely.

**Proposal.** Either raise `updates_per_train` (3–4) or lower `train_every`. This
is a pure hyperparameter change and the cheapest lever on sample efficiency.
Watch for divergence — the `br_proxy` column is the check.

### 3.4 Reservoir does not discriminate by seat

With a shared multi-seat model, all seats write to one reservoir. Seat 0 and
seat 5 have different positional priors, so the supervised target mixes them.
The v2 observation is hero-relative, which mostly mitigates this (position is
normalised relative to hero), but not entirely.

**Proposal.** Either accept it (defensible given hero-relative encoding) or
maintain per-seat reservoirs and sample proportionally. Measure the difference
before adding complexity — the hero-relative design probably makes this moot.

### 3.5 Target network sync cadence

`target_sync_every = 300` **Q updates** (hard sync), not decisions. At
`updates_per_train = 2`, that is every 150 train events. Reasonable. No change
recommended, but note the sync trigger reads `self._q_updates`, not `_step`, which
is correct — worth keeping that distinction explicit in the code.

### 3.6 No gradient noise / LR schedule for the policy head

The policy net uses a fixed Adam LR. NFSP's supervised stage benefits from a
decaying LR as the reservoir target stabilises.

**Proposal.** Add a cosine or step decay for `policy_lr` after a warm-up fraction
of training. Low risk, small but reliable gain.

---

## 4. Recommended order

| Priority | Change | Expected effect |
|---|---|---|
| 1 | Warm-up gate on hands, not decisions | Fixes corrupted early Q targets |
| 2 | Higher replay ratio (`updates_per_train` 3–4) | Faster sample efficiency, cheap |
| 3 | `players_to_act_after_hero` + last-to-act flags | Better preflop/postflop decisions |
| 4 | Opponent tendency features (VPIP/PFR/AF/fold-to-bet) | Enables exploitation; largest accuracy win |
| 5 | Raise action slots or add bounded per-street summary | Stops silent history loss |
| 6 | Postflop equity estimate | Accuracy, costs per-decision time |
| 7 | Policy LR decay | Small, safe gain |

Items 1–2 are configuration-plus-small-code changes that should be tried before
any architecture work. Item 4 is the one with the highest ceiling but the largest
implementation surface.

---

## 5. Note on model capacity

`PokerNetV2` is ~700K parameters with a 318-dim input. That is not obviously
undersized, and the audit found no evidence of underfitting (policy loss converges
to near zero on the reservoir task). Before adding capacity, verify the loss curves
in `eval_history.csv`; if `policy_loss` plateaus above ~1.5 while `q_loss`
continues to fall, that indicates capacity or representation limits worth
addressing. Otherwise, invest in features and replay ratio instead.

---

## 6. What shipped (2026-11-03)

All seven changes are implemented and on by default. Each has an off switch so
the plateau hypothesis can still be tested by ablation.

| # | Change | Where | Off switch |
|---|---|---|---|
| 1 | Exponential averaging in the average-policy reservoir | `ReservoirBuffer(min_prob=)` | `--reservoir-min-prob 0` |
| 2 | Shared trunk (with the PCGrad caveat below) | `NFSPModel(shared_trunk=)` | omit `--shared-trunk` |
| 3 | Rolling opponent tendencies (VPIP/PFR/AF/fold-to-bet) | `poker/ml/tendencies.py` | `--obs-version 2` |
| 4 | n-step returns | `NFSPModel(n_step=)` | `--n-step 1` |
| 5 | Smaller batch, more updates | trainer flags | `--batch-size 2048 --updates-per-train 1` |
| 6 | C51 distributional value head | `NFSPModel(value_distribution="c51")` | `--value-distribution none` |
| 7 | Attention over occupied seats | `poker/ml/models/nfsp_networks_v3.py` | `--obs-version 2` |

### Observation v3

354 floats: the v2 vector (318) followed by 36 tendency floats (4 per seat).
The v2 prefix is byte-identical, verified by test, so v3 is a strict extension.
Tendencies are smoothed toward a Beta prior and computed from *completed hands*
only, so a feature never depends on the hand in progress.

v3 is the default for `train_hybrid.py`. **v2 checkpoints cannot load into v3** �
they have a different input width and a different network class.

### Distributional value (C51)

The Q network keeps its scalar dueling head and gains a parallel `dist_head`
emitting `7 x 51` logits. `forward()` returns the expectation, so every existing
call site (argmax, gather, TD targets, tests reading `q_network(x)`) is unchanged.
Support defaults to `[-10, 10]`; with `REWARD_SCALE = 0.5` and returns in
stack-change units, a busted hand is about -5, so the support has headroom.

### n-step returns

`store_transition` buffers up to `n_step` transitions and emits discounted sums.
A partial window is flushed by `end_hand()` (called by all trainers) or by
`flush_pending()`. Callers that store transitions without ever reporting hand
end must call `flush_pending()` before expecting the buffer to be full.

### Two things NOT done

1. **PCGrad is not implemented.** `shared_trunk` still shares one representation
   between the Q and policy losses with no gradient surgery, so the two
   objectives can still interfere. `docs/HYBRID_PLAN.md` calls this out as the
   principled version of sharing; it remains open.
2. **No ablation has been run.** The seven changes were verified to work
   individually but not measured against each other. The plateau question �1
   posed is still unanswered � see below.

### The measurement that should come next

Everything here is implemented, and none of it is proven to help. Run these
against the same seed and compare `eval_history.csv`:

```bash
# control: reproduce the old behaviour
python scripts/train_hybrid.py --teacher models/sdcfr_v2/sdcfr_v2.pt \
  --obs-version 2 --value-distribution none --n-step 1 \
  --reservoir-min-prob 0 --batch-size 2048 --updates-per-train 1 \
  --episodes 60000 --model-name abl_control

# one change at a time, then all together
```

The question �1 raised � learning dynamics (1, 4, 6) versus representation
(3, 7) � is answerable with four runs. Guessing wrong means rebuilding the wrong
thing, so measure before investing further.
