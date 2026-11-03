# Poker ML Environment

A No-Limit Texas Hold'em engine in Python designed as a simulation environment for training machine learning agents via self-play.

---

## Quick Start

```bash
pip install -e ".[dev]"
python -m poker.main -n 6 --hands 20 --seed 1 --bots human nfsp:models/nfsp_v2/nfsp_v2.pt sdcfr:models/sdcfr_prod/sdcfr_ckpt_500.pt random random flop
```

Current checkpoints (verified 2026-11-03):

- NFSP: `models/nfsp_v2/nfsp_v2.pt` — obs v2, 318-wide, pot-fraction sizing
- SD-CFR: `models/sdcfr_prod/sdcfr_ckpt_500.pt` — obs v1, 98-wide, legacy sizing

These are the only two checkpoints kept. Older runs (`nfsp_prod`, `multihand`,
`nfsp_multiplayer`, `NFSP_reworked`, `nfsp_v3*`, and the root-level `nfsp_4player*` /
`sdcfr_ckpt_*` files) were removed on 2026-11-03. Note there is **no v3
checkpoint**: the deleted `nfsp_v3*.pt` files used the retired 155-wide observation
and predated the current architecture.

Score a full table, including whether bets match the cards:

```bash
python scripts/validate_training.py --hands 200 --seed 1 --bots nfsp:models/nfsp_v2/nfsp_v2.pt sdcfr:models/sdcfr_prod/sdcfr_ckpt_500.pt random flop
```

---

## Configuration

All game parameters live in `config/game_config.py`:

```python
GameConfig(
    num_players     = 6,        # 2–6
    starting_stack  = 1000,
    small_blind     = 5,
    big_blind       = 10,
    ante            = 10,       # posted upfront by every player each hand
    rake_percent    = 5.0,      # % taken from each pot before distribution
    rake_cap        = None,     # optional max rake per hand in chips
    run_it_twice    = False,    # heads-up all-in only: deal remaining streets twice, split pot by run-out
    blind_schedule  = [         # escalating blinds by default; set fixed_blinds=True for cash game
        (5, 10, 0),             # (small_blind, big_blind, ante) per level
        (10, 20, 0),
        (15, 30, 5),
        # ...
    ],
    level_duration  = 20,       # hands per blind level
    fixed_blinds    = False,    # True = cash game mode (ignore blind_schedule)
)
```

---

## Rules & Assumptions

### Variant
No-Limit Texas Hold'em.

### Antes
Every active player posts an ante at the start of each hand before the blinds are dealt. The ante goes directly into the pot and does not count toward any player's bet for that street.

### Blinds
Standard rotating small blind and big blind. Both are posted after the ante. The big blind is the minimum opening bet for the pre-flop street.

### Raise Sizing Rules

These rules are based on standard TDA (Tournament Directors Association) guidelines and widely accepted No-Limit Hold'em convention:

**Minimum raise:**
The raise increment must be at least equal to the largest previous bet or raise in the same betting round.

> Example: BB = 10. Player A raises to 30 (a raise of 20). Player B's minimum re-raise is another 20, making the total 50.

**Pre-flop opening raise:**
Before any raise, the minimum opening raise is equal to the big blind.

**The half-raise (all-in) rule:**

If a player goes all-in for **less than a full minimum raise**, it is a "short
all-in" and does **not** re-open the betting. A player who has already acted in
that round may only call or fold; they may not re-raise. A player who has not yet
acted still has all options.

> **Known deviation.** `reopens_betting()` computes this correctly, but
> `legal_actions()` does not yet consult it when building the raise option, so the
> engine still offers a raise to a player facing a short all-in. The engine is
> therefore slightly more permissive than real Hold'em in this one spot. Fixing it
> is tracked in `docs/PROJECT_STATUS.md`.

**Maximum raise:**
In No-Limit, a player may raise any amount up to their entire remaining stack (all-in).

**Cap:**
There is no cap on the number of raises per street in No-Limit Hold'em (caps are a Limit Hold'em concept).

**Summary table:**

| All-in raise amount | Re-opens betting? | Engine behaviour |
|---|---|---|
| ≥ full minimum raise | Yes, for all remaining players | Correct |
| < full minimum raise | No — prior actors may only call/fold | **Not enforced for the raise option** |

### Pot Management & Rake

- Rake is taken as a configurable percentage of each pot (main pot and each side pot independently) before distribution.
- An optional rake cap limits the maximum rake taken per hand.
- Side pots are created whenever a player is all-in for less than the current street bet.
- On a chopped pot (tied hands), the odd chip goes to the first eligible seat left of the dealer button.

### Showdown
- Players must show cards if called to showdown. A player who bet or raised last on the final street shows first.
- A player can muck a losing hand without showing, but only after the winning hand is revealed.
- Best 5-card hand from any combination of 2 hole cards + 5 community cards wins.

---

## ML Environment

`poker.ml.env.PokerEnv` is the Gymnasium environment. Two observation encoders exist:
v1 is 98 floats (nine seat slots zero-filled, so 2–9 players share one input size),
and v2 is 318 floats (hero-relative, card ids in their own block, pot-fraction
friendly). Actions are seven discrete choices: fold, check/call, and five raise
buckets. See `docs/ML.md` and `docs/TRAINING.md`.

---

## Project Structure

```
poker/
├── core/           # Pure game logic — no UI or ML dependencies
├── config/         # GameConfig dataclass
├── interface/      # Text CLI for human play
├── ml/             # Gym env, observation encoder, action space
├── bots/           # RandomBot, HumanPlayer, and base class
├── logging/        # JSON game logger and replay tool
└── main.py
```

## Documentation

```
docs/
├── README.md          # Index of all docs
├── PROJECT_STATUS.md  # Current state, how to run, known issues
├── ENGINE.md          # Architecture deep dive (data structures, betting, showdown)
├── ML.md              # ML environment: PokerEnv, observations, models
├── TRAINING.md        # Training and validation guide: NFSP, SD-CFR
├── HYBRID_PLAN.md     # SD-CFR-as-teacher design for a unified model
├── NFSP_Notes.md      # NFSP paper/reference-implementation notes
├── SD-CFR_Notes.md    # SD-CFR notes
├── BOTS.md            # Available bots, how to add one
└── USER_NOTES.md      # Stable design decisions
```

**Start here:**
- [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md) — what works, how to run it
- [docs/TRAINING.md](docs/TRAINING.md) — ML training (NFSP/SD-CFR) guide
- [docs/HYBRID_PLAN.md](docs/HYBRID_PLAN.md) — combined NFSP + SD-CFR design
- [docs/ENGINE.md](docs/ENGINE.md) — engine architecture and internals
