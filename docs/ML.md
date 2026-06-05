# ML / RL Environment

The `poker.ml` package provides a **Gymnasium-compatible** RL interface for training poker agents.

---

## Current State

### PokerEnv (`poker.ml.env.PokerEnv`)

- `reset()` — returns initial observation + info dict
- `step(action)` — fully wired to the engine; returns `(obs, reward, done, truncated, info)`
- `get_action_mask()` — returns `(7,)` legal-action boolean mask
- Reward: `10 × (stack_change / starting_stack)` at hand end, 0 during hand

### Observation (`poker.ml.observation`)

`build_observation(state, seat) → np.ndarray` returns a `(142,)` float32 vector:
- Hole cards (one-hot encoded)
- Community cards (one-hot, padded until revealed)
- Current pot (normalized to starting stack)
- Each player's stack (normalized)
- Seat position relative to dealer
- Amount to call (normalized)
- Current street (0=pre-flop, 1=flop, 2=turn, 3=river)
- Betting history for current street

### Action Space

`PokerEnv.action_space = Discrete(7)`:

| Index | Action |
|-------|--------|
| 0 | Fold |
| 1 | Check/Call (prefers call if legal) |
| 2–6 | Raise buckets (quantized raise-to amounts) |

---

## Models

### NFSP (`poker.ml.models.nfsp_model.NFSPModel`)

Neural Fictitious Self-Play — two networks:
- **Q-network** (DQN): learns best-response policy via TD learning
- **Policy network**: behavioral cloning from best-response actions

Mixing: 15% best-response, 85% average policy (`eta = 0.15`).
Architecture: deep residual network, 4 blocks, ~350K parameters.

### SD-CFR (`poker.ml.models.sdcfr_model.SDCFRModel`)

Single Deep Counterfactual Regret Minimization:
- **Advantage network** learns counterfactual regret values
- Regret matching converges to approximate Nash equilibrium

### Replay Buffers (`poker.ml.buffers`)

All buffers are thread-safe (support parallel training):
- `CircularBuffer` — Q-network transitions (FIFO, 1M capacity default)
- `ReservoirBuffer` — Policy network samples (uniform, 5M capacity default)
- `WeightedReservoirBuffer` — Reserved for future use

---

## Usage Example

```python
from poker.ml.env import PokerEnv
from poker.config.game_config import GameConfig

env = PokerEnv(config=GameConfig(num_players=2))
obs, info = env.reset()
mask = env.get_action_mask()   # (7,) boolean array

action = 1  # check/call
obs, reward, done, truncated, info = env.step(action)
```

---

## Training

See [`docs/TRAINING.md`](TRAINING.md) for the full training guide, including:
- Sequential and parallel training scripts
- Reward design rationale (why v3 uses pure game rewards)
- Expected training curves and performance targets
- Troubleshooting common failures (100% fold, high q_loss, etc.)
