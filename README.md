# Poker ML Environment

A No-Limit Texas Hold'em engine in Python designed as a simulation environment for training machine learning agents via self-play.

---

## Quick Start

```bash
pip install -r requirements.txt
python main.py
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
If a player goes all-in for an amount that is **less than a full minimum raise** but **at least 50% of the minimum raise**, the bet is called a "short all-in." A short all-in does **not** re-open the betting for players who have already acted — they may only call or fold. A player who has not yet acted, or who made the last full raise, retains the right to re-raise.

If a player goes all-in for an amount that is **less than 50% of the minimum raise**, it is treated purely as a call-size all-in and does not re-open betting for anyone.

**Maximum raise:**
In No-Limit, a player may raise any amount up to their entire remaining stack (all-in).

**Cap:**
There is no cap on the number of raises per street in No-Limit Hold'em (caps are a Limit Hold'em concept).

**Summary table:**

| All-in raise amount | Re-opens betting? |
|---|---|
| ≥ full minimum raise | Yes, for all remaining players |
| ≥ 50% but < full min raise | No — only players who haven't yet acted |
| < 50% of minimum raise | No — treated as a call |

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

The `ml/poker_env.py` module exposes a Gym-compatible interface:

```python
from ml.poker_env import PokerEnv
from config.game_config import GameConfig
from bots.random_bot import RandomBot

env = PokerEnv(config=GameConfig(), bots=[RandomBot(), RandomBot()])
obs = env.reset()
obs, reward, done, info = env.step(action)
```

### Observation Space
Fixed-length numpy array encoding:
- Hole cards (2 cards, one-hot encoded)
- Community cards (5 cards, one-hot encoded, padded until revealed)
- Current pot (normalized to starting stack)
- Each player's stack (normalized)
- Seat position relative to dealer
- Amount to call (normalized)
- Current street (0=pre-flop, 1=flop, 2=turn, 3=river)
- Betting history for current street (last N actions encoded)
- **Per-player history: last 10 hands** — for each player, encodes outcome (won/lost), chips won/lost, and simplified action sequence (VPIP, aggression) over their last 10 hands

### Action Space
- `0` — Fold
- `1` — Check
- `2` — Call
- `3` — Raise (paired with a continuous value: raise fraction of pot, e.g. 0.5 = half-pot)
- `4` — All-in

### Reward
Change in chip stack at end of hand, normalized to `starting_stack`. Zero during a hand; delivered at `done=True`.

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
├── TRAINING.md        # Full training guide: NFSP, SD-CFR, parallel training
├── BOTS.md            # Available bots, how to add one
└── USER_NOTES.md      # Stable design decisions
```

**Start here:**
- [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md) — what works, how to run it
- [docs/TRAINING.md](docs/TRAINING.md) — ML training (NFSP/SD-CFR) guide
- [docs/ENGINE.md](docs/ENGINE.md) — engine architecture and internals
