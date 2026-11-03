# ML / RL Environment

`poker.ml` provides a Gymnasium-compatible RL interface plus the NFSP and SD-CFR models.

---

## PokerEnv (`poker.ml.env.PokerEnv`)

- `reset()` — initial observation + info dict
- `step(action)` — wired to the engine; returns `(obs, reward, done, truncated, info)`
- `reset_multi()` / `step_multi()` — multi-seat variants returning `(seat, obs, mask, rewards, done)`
- `get_action_mask()` — `(7,)` legal-action mask
- `REWARD_SCALE = 0.5`; terminal reward is `REWARD_SCALE * stack_change / starting_stack`

---

## Observation encoders

Two encoders coexist. **They are not interchangeable** — a network built for one
cannot consume the other.

### v1 — `build_observation` → 98 floats

Legacy flat vector: hole cards, community cards, pot, per-player stacks, seat
positions, amount to call, street, betting history. Used by SD-CFR, all scripted
bots, and NFSP v1 training.

`adapt_observation(obs, target_dim)` can widen a 98-dim vector to the retired
155-dim layout for old checkpoints. This exists only for backwards compatibility.

### v2 — `build_observation_v2` → 318 floats

Hero-relative vector. Byte offsets are published in `OBS_SPEC_V2` (`ObsSpecV2`)
rather than hard-coded:

| Block | Width | Contents |
|---|---|---|
| `card_ids` | 7 | Hole + board card ids (52 = empty) |
| `derived` | 20 | Suitedness, pair, gaps, board texture, draw flags |
| `hero` | 16 | Made-hand one-hot, draws, preflop equity, position signals |
| `seats` | 9 × 15 | Per-seat status, stack, commitments, last action |
| `street_hist` | 4 × 10 | Per-street raise/call/fold/check counts, last aggressor |
| `action_hist` | 6 × 14 | Last 6 actions as seat/type/size/street |
| `scalars` | 16 | Pot, call size, SPR, effective stack, min-raise, positions |

The card-id block is deliberately kept out of the `[0, 1]` clamp so
`PokerNetV2._embed` can cast it to embedding indices.

---

## Action space

`Discrete(7)`: `0` fold, `1` check/call, `2–6` raise buckets.

Two bucket schemes (`poker.ml.action_space.BetScheme`):

- `LEGACY` — min-raise / all-in quantisation. Used by SD-CFR and all old models.
- `POT_FRACTION` — bucket 2 is the min raise (or a 2.5bb open), buckets 3–5 are
  50% / 100% / 200% of the pot after a call, bucket 6 is all-in. Used by NFSP v2.

---

## Models

### NFSP — `poker.ml.models.nfsp_model.NFSPModel`

Two networks, two memories:

- **Q-network** — Double DQN on `q_buffer` (M_RL, circular). Best response is
  ε-greedy over legal actions.
- **Average-policy network** — cross-entropy on `policy_buffer` (M_SL, reservoir).
  **M_SL receives only best-response actions**, written by `select_action` when the
  episode is in best-response mode. `store_transition` writes M_RL only.

Anticipatory dynamics pick one mode per episode: best response with probability
η, otherwise the average policy. Evaluation always samples from the average policy.

Architecture is chosen by `obs_version`:

| | v1 | v2 |
|---|---|---|
| Network | `PokerNetwork` (residual MLP) | `PokerNetV2` (card embeddings, dueling head) |
| Width | 155 (adapted) or 98 | 318 |
| `gamma` default | 0.99 | 1.0 |
| Q loss default | MSE | Huber |
| Bet scheme default | `LEGACY` | `POT_FRACTION` |
| Buffer dtype | float32 | float16 (sampled as float32) |

Two implementation details worth knowing:

- The **fold-return auxiliary loss** (targeting `−0.5 × committed fraction` for
  action 0) only runs under obs v2, because it reads the v2 commitment feature. It
  is skipped entirely under v1, where the target would be a constant zero.
- `load()` sizes buffers to the checkpoint's stored row counts. Loading a snapshot
  larger than the constructor's capacity is safe.

### SD-CFR — `poker.ml.models.sdcfr_model.SDCFRModel`

Single Deep CFR. An advantage network is trained on regret targets from
external-sampling traversals; `get_strategy()` applies regret matching to its
output. Hard-wired to `_OBS_DIM = 98` and `BetScheme.LEGACY`.

`train_iteration()` currently re-initialises the advantage network every iteration.
That is tolerable for a standalone SD-CFR run but must become warm-starting before
any shared-trunk hybrid work — see [`HYBRID_PLAN.md`](HYBRID_PLAN.md) §4.

### Replay buffers — `poker.ml.buffers`

All thread-safe.

| Buffer | Used by | Notes |
|---|---|---|
| `CircularBuffer` | NFSP Q memory | FIFO; stores `next_masks` and `current_masks` |
| `ReservoirBuffer` | NFSP policy memory | `max(width/seen, min_prob)` reservoir sampling |
| `WeightedCircularBuffer` | SD-CFR regret memory | FIFO insert, weight-biased sample |
| `WeightedReservoirBuffer` | legacy only | Kept for old tests; internally inconsistent |

---

## Usage

```python
from poker.config.game_config import GameConfig
from poker.ml.env import PokerEnv

env = PokerEnv(config=GameConfig(num_players=2))
obs, info = env.reset()
mask = env.get_action_mask()          # (7,) bool

obs, reward, done, truncated, info = env.step(1)   # 1 = check/call
```

For a multi-seat model, use `reset_multi` / `step_multi` or the
`poker.ml.nfsp_loop.consume_hand` helper, which writes one transition per learner
decision and assigns each seat the full-hand return at termination.

---

## Training

See [`TRAINING.md`](TRAINING.md) for commands, reward design, and troubleshooting.
See [`HYBRID_PLAN.md`](HYBRID_PLAN.md) for the combined-model design.
