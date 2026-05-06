# ML / RL environment

The `poker.ml` package is the start of a **Gymnasium-compatible** RL interface.

## What exists today

- `poker.ml.env.PokerEnv`
  - `reset()` returns an initial observation + info
  - `get_action_mask()` returns a `(7,)` legal-action mask
  - `step()` is currently **not implemented**
- `poker.ml.observation.build_observation(state, seat) -> np.ndarray`
  - Returns a fixed `(142,) float32` vector (card indices + normalized stacks/positions + basic street/action features).
- `poker.ml.action_space`
  - Discrete action space of size **7** with masking utilities and conversion helpers.

### Discrete actions (current)

`PokerEnv.action_space = Discrete(7)`:

- `0`: fold
- `1`: check/call (prefers call if legal)
- `2..6`: raise buckets (quantized selection among currently legal raise-to amounts)

## What’s intentionally missing

`PokerEnv.step()` raises `NotImplementedError` because it isn’t yet wired to the engine/session loop.
Until that happens, this repo supports:

- building observations from `GameState`
- computing legal-action masks from `GameState`
- running full games via the CLI/session engine (outside Gym stepping)

## Roadmap (next concrete steps)

- **Wire `PokerEnv.step()` to the engine**
  - Use `poker.engine.action_validator.legal_actions` + `poker.ml.action_space.action_index_to_action`
  - Apply the action through the existing hand/session machinery (one action per env step)
  - Return reward at hand end (stack delta), and keep intermediate reward at 0
- **Episode definition**
  - Decide: “one hand = episode” vs “tournament until 1 left = episode”
  - Current `PokerEnv` docstring assumes “tournament until 1 left”
- **Self-play hooks**
  - Provide a training loop that uses `NullLogger` and swaps in learning bots that update in `observe_result()`
- **Reproducibility**
  - Add an explicit seed flow (env seed + bot seeds + deck RNG seed) so runs are repeatable end-to-end

