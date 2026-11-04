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

`--bots` takes one token per seat. Count must match `--num-players` (2–9). Model seats need a checkpoint. The same path is loaded once if several seats name it.

```bash
python -m poker.main -n 6 --hands 20 --seed 1 --bots human nfsp:models/nfsp_v2/nfsp_v2.pt sdcfr:models/sdcfr_prod/sdcfr_ckpt_500.pt random random flop
```

Tokens: `human`, `random`, `flop`, `nfsp:PATH`, `sdcfr:PATH`, `nn:PATH` (not implemented yet).

Current checkpoints: NFSP `models/nfsp_v2/nfsp_v2.pt` (obs v2, 318-wide), SD-CFR `models/sdcfr_prod/sdcfr_ckpt_500.pt` (obs v1, 98-wide). These are the only two checkpoints kept in the repo.

## Adding a new bot

Create a file under `poker/bots/` with `name`, `act`, and `observe_result`. Copy `random_bot.py` and replace `act()`. Register the token in `poker.main` where seats are built.

