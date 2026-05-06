# Bots

Bots are “players” that implement the `poker.bots.base.Bot` protocol:

- `name: str`
- `act(state: GameState, legal: list[Action]) -> Action`
- `observe_result(final_state: GameState, reward: float) -> None`

The engine always calls bots with `state.view_for(seat)` so **opponent hole cards are hidden**.

## Included bots

- `poker.bots.human_bot.HumanBot`
  - Renders via `poker.interface.text_ui.render()` and prompts for input.
- `poker.bots.random_bot.RandomBot`
  - Chooses a legal action via weighted randomness.
  - Accepts an optional `seed` for reproducible action selection.
- `poker.bots.flop_bot.FlopBot`
  - Preflop: check/call to see a flop.
  - Flop+: raises with any detected pair (simple rank-duplicate check), otherwise folds.

## Using bots from the CLI

Examples:

```bash
python -m poker.main -n 2 -b human random
python -m poker.main -n 3 -b flop_bot random random
```

`--bots` takes one token per seat, in seat order. Unspecified seats default to `random`.

## Adding a new bot

Create a new file under `src/poker/bots/` that provides the protocol surface above.
For quick iteration, copy `random_bot.py` and replace `act()`.

If you want it selectable from the CLI, add it to the mapping in `poker.main` where bots are constructed.

