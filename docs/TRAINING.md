# ML Training Guide

How to train and evaluate the NFSP and SD-CFR models.

---

## Current Checkpoints (verified 2026-11-03)

| Role | Path | Obs | Encoder |
|---|---|---|---|
| NFSP | `models/nfsp_v2/nfsp_v2.pt` | v2, 318-wide | `PokerNetV2` |
| SD-CFR | `models/sdcfr_prod/sdcfr_ckpt_500.pt` | v1, 98-wide | `PokerNetwork` |

There is no v3 checkpoint. The old `nfsp_v3*.pt` files were deleted — they used the
retired 155-wide observation and predate the current architecture. "v3" refers to the
reward-design iteration described below, not a model version.

---

## Quick Start

```bash
# Continue NFSP v2 training from the current checkpoint
python scripts/train_nfsp_v2.py --checkpoint models/nfsp_v2/nfsp_v2.pt

# Train SD-CFR
python scripts/train_sdcfr_selfplay.py

# Score a mixed table
python scripts/validate_training.py --hands 200 --seed 1 \
  --bots nfsp:models/nfsp_v2/nfsp_v2.pt \
         sdcfr:models/sdcfr_prod/sdcfr_ckpt_500.pt random flop
```

NFSP v1 (98-wide, one learning seat) is still available via
`scripts/train_nfsp_selfplay.py`. SD-CFR seats stay on the legacy 98-d encoder and
`BetScheme.LEGACY` — migrating them to v2 is the prerequisite for the hybrid work
in [`HYBRID_PLAN.md`](HYBRID_PLAN.md).

```bash
python scripts/profile_training.py --v2 --episodes 100
```

---

## Long-Form NFSP Training

`train_nfsp_v2.py` is the current trainer. Defaults are already tuned for a long
run, so the minimal long-form invocation is small.

### Resume a long run (recommended)

Resuming from the current checkpoint preserves buffers, optimizer state, epsilon
position, and self-play schedule:

```bash
python scripts/train_nfsp_v2.py \
  --checkpoint models/nfsp_v2/nfsp_v2.pt \
  --sdcfr-opponent models/sdcfr_prod/sdcfr_ckpt_500.pt \
  --episodes 200000 \
  --eval-every 5000 \
  --checkpoint-every 10000 \
  --model-name nfsp_v2
```

### Fresh long run into a new folder

```bash
python scripts/train_nfsp_v2.py \
  --sdcfr-opponent models/sdcfr_prod/sdcfr_ckpt_500.pt \
  --episodes 500000 \
  --eval-every 10000 \
  --checkpoint-every 25000 \
  --min-players 2 --max-players 9 \
  --batch-size 2048 \
  --train-every 64 \
  --updates-per-train 2 \
  --eta 0.1 \
  --random-eval-hands 100 \
  --stop-min-episode 50000 \
  --stop-patience 15 \
  --model-name nfsp_long
```

### Unattended overnight run

Higher patience, fewer evaluations, and no early stop surprise. Validation is the
slowest part of the loop, so evaluating less often is what buys wall-clock time:

```bash
python scripts/train_nfsp_v2.py \
  --checkpoint models/nfsp_v2/nfsp_v2.pt \
  --sdcfr-opponent models/sdcfr_prod/sdcfr_ckpt_500.pt \
  --episodes 1000000 \
  --eval-every 25000 \
  --checkpoint-every 50000 \
  --stop-patience 40 \
  --stop-min-episode 100000 \
  --random-eval-hands 50 \
  --sdcfr-eval-only \
  --model-name nfsp_overnight
```

### Parameter reference

| Flag | Default | Notes |
|---|---|---|
| `--episodes` | 100000 | Episodes, not hands. ~10 hands/episode |
| `--eval-every` | 5000 | Validation is the main non-training cost; raise it for long runs |
| `--checkpoint-every` | 10000 | Writes `nfsp_gen_N.pt`; these are ~150 MB each |
| `--batch-size` | 2048 | Gradient batch |
| `--train-every` | 64 | Decisions between training steps |
| `--updates-per-train` | 1 | Raise to 2–4 to increase replay ratio |
| `--eta` | 0.1 | Best-response mix; 0.1 is the paper-faithful value |
| `--min-players` / `--max-players` | 2 / 9 | 6-player is weighted 2× |
| `--stop-min-episode` | 20000 | Earliest stall stop |
| `--stop-patience` | 10 | Evals without improvement before stopping |
| `--random-eval-hands` | 100 | Hands per player count for Random+Flop eval |
| `--sdcfr-eval-only` | off | Skips Random+Flop eval, roughly halves eval time |
| `--model-name` | `nfsp_v2` | Output folder `models/<name>/` |
| `--no-validation` | off | Disables evaluation entirely (throughput testing only) |

### Outputs

For `--model-name nfsp_v2`:

- `models/nfsp_v2/nfsp_v2.pt` — final / best checkpoint
- `models/nfsp_v2/nfsp_best.pt` — best SD-CFR lower bound so far
- `models/nfsp_v2/nfsp_gen_N.pt` — periodic generations
- `models/nfsp_v2/eval_history.csv` — per-eval metrics

### Monitoring

```bash
# Live loss / eval lines
Get-Content models/nfsp_v2/training_v2.txt -Wait -Tail 20

# Best BB/100 trend
Get-Content models/nfsp_v2/eval_history.csv | Select-Object -First 1
Get-Content models/nfsp_v2/eval_history.csv | Select-Object -Last 10
```

The honest health check is `br_proxy_bb100` in `eval_history.csv`. A rising mean
with a wildly oscillating proxy means the best response is not converging, even if
the average policy is improving.

---

## Reward Design

### The problem (v1/v2)

Early runs added an artificial anti-folding bonus:

```python
if action == FOLD:
    reward -= 0.01
else:
    reward += 0.005
```

This mathematically incentivised always folding. Playing a −0.05 EV hand gives
`−0.05 + 0.005 = −0.045`, while folding gives `−0.01`, so folding is always
optimal. Models converged to a 100% fold rate.

### The fix (v3)

Remove the bonuses and use pure game rewards scaled by `REWARD_SCALE`:

```python
# poker/ml/env.py
REWARD_SCALE = 0.5
reward = REWARD_SCALE * (stack_change / starting_stack)
```

Larger scales were tested and rejected: 50× exploded Q-loss past 600, while 1× gave
too weak a signal to learn from.

---

## Training Phases

### Phase 1 — quick test

```bash
python scripts/train_nfsp_v2.py --episodes 200 --eval-every 100 --no-validation
python scripts/train_sdcfr.py --cfr-iterations 200 --traversals-per-iteration 500
```

### Phase 2 — medium

```bash
python scripts/train_nfsp_v2.py --episodes 20000 --eval-every 2000
python scripts/train_sdcfr.py --cfr-iterations 2000
```

### Phase 3 — full

```bash
python scripts/train_nfsp_v2.py --episodes 100000 --eval-every 5000
python scripts/train_sdcfr_extended.py --cfr-iterations 10000 --traversals-per-iteration 1000
```

---

## Monitoring

Progress prints every `--eval-every` episodes:

```
Ep 50000 | hands=... | random=+341.8 | sp=0%
  [EVAL] SD-CFR: -209.1 BB/100 (lower -370.1)
  [RAMP] self-play 0% -> 10%
```

### Key metrics

| Metric | Target | Warning sign |
|---|---|---|
| `q_loss` | < 5.0 | > 10 sustained |
| `policy_loss` | 1.0–1.5 | > 2.0 (diverging) |
| `sdcfr_bb100` | trending up | falling while `q_loss` falls |
| `br_proxy_bb100` | **stable** | swinging ±400 (best response not converging) |

The `br_proxy_bb100` column is the honest check on whether the best response is
working. A rising mean with a wildly oscillating proxy means nothing has converged.

---

## Troubleshooting

### FOLD% stuck near 100% after 50K episodes

1. Confirm artificial bonuses are absent from `poker/ml/env.py`.
2. Confirm `REWARD_SCALE == 0.5`.
3. Check the fold-return loss is only active under obs v2 (it reads the
   committed-fraction feature, which v1 does not have).

### q_loss stuck above 10

1. Lower the Q learning rate: `--q-lr 0.0002`.
2. Train longer — the warm-up gate is
   `_q_min = max(batch_size, q_capacity // 100)`, so a large `q_capacity` delays
   the first update substantially.
3. Confirm GPU use: look for `[GPU]` in the first line of output.

### `ValueError: could not broadcast input array` on load

Fixed. `NFSPModel.load()` now sizes buffers to the checkpoint rather than assuming
the constructor's capacities. If it reappears, the checkpoint buffer is larger than
the configured capacity and `_ensure_buffer_capacity` did not run.

### Loading a checkpoint into the wrong encoder

`load()` validates metadata against weights and raises with a specific message
("Checkpoint metadata obs_version=... does not match ... weights"). Do not
hand-edit `model_meta` to force a load — the shapes will not match.

---

## Validation

```bash
# One model across player counts
python scripts/validate_training.py models/nfsp_v2/nfsp_v2.pt --hands 100

# A full scored table (every seat, Sense% checks)
python scripts/validate_training.py --hands 200 --seed 1 \
  --bots nfsp:models/nfsp_v2/nfsp_v2.pt sdcfr:models/sdcfr_prod/sdcfr_ckpt_500.pt random flop
```

`Sense%` flags decisions that look wrong: folding a strong hand, or making a large
bet with a weak one. A healthy model sits above ~90%.

### Engine validators

```bash
python scripts/validate_poker_logic.py
python scripts/validate_game.py
python scripts/verify_game_logic.py
```

Rebuild the v2 preflop equity table (needed by `build_observation_v2`):

```bash
python scripts/build_preflop_equity_table.py
```

---

## File Reference

| File | Purpose |
|---|---|
| `scripts/train_nfsp_v2.py` | Current NFSP training (v2 encoder, pot-fraction, multi-seat) |
| `scripts/train_nfsp_selfplay.py` | NFSP v1 training (98-wide, one learning seat) |
| `scripts/train_sdcfr_selfplay.py` | Current SD-CFR training |
| `scripts/train_sdcfr.py`, `train_sdcfr_extended.py` | Earlier SD-CFR runs |
| `scripts/validate_training.py` | Single-model sweep or scored table |
| `scripts/profile_training.py` | Training-loop profiler |
| `poker/ml/env.py` | RL environment, `REWARD_SCALE` |
| `poker/ml/models/nfsp_model.py` | NFSP |
| `poker/ml/models/sdcfr_model.py` | SD-CFR |
| `poker/ml/nfsp_loop.py` | One-hand transition writer for shared multi-seat models |
| `docs/HYBRID_PLAN.md` | SD-CFR-as-teacher distillation design |
