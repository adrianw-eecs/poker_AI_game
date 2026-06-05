# ML Training Guide

Comprehensive guide for training NFSP and SD-CFR poker AI models.

---

## Quick Start

```bash
# 4-player self-play training (most advanced, recommended)
python scripts/train_nfsp_selfplay.py --episodes 250000

# 2-player production training with full logging
python scripts/train_nfsp_production.py --episodes 250000

# SD-CFR extended training (~8 hours for 10K iterations)
python scripts/train_sdcfr_extended.py --cfr-iterations 10000

# SD-CFR standard training
python scripts/train_sdcfr.py --cfr-iterations 5000

# Validate a checkpoint or final model
python scripts/validate_training.py models/nfsp_4player.pt --hands 200
```

---

## Models Overview

### NFSP (Neural Fictitious Self-Play)
- Two-network architecture: Q-network (DQN) + Policy network (behavioral cloning)
- Mixes best-response (15%) with average policy (85%)
- **Training script**: `scripts/train_nfsp_v3.py` (sequential) or `scripts/train_nfsp_parallel.py` (parallel)
- **Architecture**: Deep residual network, 4 blocks, 350K parameters

### SD-CFR (Single Deep Counterfactual Regret Minimization)
- Advantage network learns counterfactual regret
- Reservoir buffer collects 1M+ samples before full convergence
- **Training script**: `scripts/train_sdcfr.py`
- **Architecture**: `src/poker/ml/models/sdcfr_model.py`

---

## Reward Design (v3 — Current)

### The Problem (v1/v2)

Early training attempts used an artificial anti-folding bonus:

```python
# v1/v2 (broken):
if action == FOLD:
    reward -= 0.01
else:
    reward += 0.005
```

This mathematically incentivized always folding:
- Playing a -0.05 EV hand: `-0.05 + 0.005 = -0.045`
- Folding: `-0.01`
- Decision: **Fold is always optimal**

Result: Models converged to 100% fold rate.

### The Fix (v3 — Pure Game Rewards)

Remove artificial bonuses. Use 10x-scaled pure game rewards:

```python
# v3 (working):
reward = 10.0 * (stack_change / starting_stack)
# No artificial FOLD penalty or PLAY bonus
```

Why 10x scaling? Tested empirically:
- 5x: Q-loss too high, network struggles
- **10x: Optimal, clean convergence**
- 50x: Gradient explosion (Q-loss explodes to 600+)

The game structure creates natural incentives:
- Strong hands → win pot → large positive reward → learn to play
- Weak hands → lose chips → large negative reward → learn to fold
- Natural equilibrium: ~30% fold, balanced strategy

### Diverse Opponent Training

Train against a rotating mix of opponents for diverse learning:
- **RandomBot**: Teaches exploiting chaotic play
- **FlopBot**: Teaches hand-strength awareness
- **CallBot** (`src/poker/bots/call_bot.py`): Teaches value betting vs calling stations

---

## Training Phases

### Phase 1: Quick Test (~30 min)

Validate implementation before committing to full training.

```bash
# NFSP quick test (5K episodes)
python scripts/train_nfsp_selfplay.py --episodes 5000 --eval-every 500

# SD-CFR quick test (200 iterations)
python scripts/train_sdcfr.py --cfr-iterations 200 --traversals-per-iteration 500
```

### Phase 2: Medium Training (2–6 hours)

```bash
# NFSP: 100K episodes (~2 hours on RTX 3060)
python scripts/train_nfsp_selfplay.py --episodes 100000 --eval-every 5000

# SD-CFR: 2000 iterations (~4 hours on RTX 3060)
python scripts/train_sdcfr.py --cfr-iterations 2000
```

### Phase 3: Full Training (8–20 hours)

```bash
# NFSP: 250K episodes (self-play, recommended)
python scripts/train_nfsp_selfplay.py --episodes 250000 --eval-every 10000

# NFSP: 250K episodes (2-player production with logging)
python scripts/train_nfsp_production.py --episodes 250000

# SD-CFR: 10K iterations (~8 hours on RTX 3060)
python scripts/train_sdcfr_extended.py --cfr-iterations 10000 --traversals-per-iteration 1000
```

---

## Expected Training Curves

### NFSP v3 (250K episodes)

| Episode | Q-Loss | Policy-Loss | Eval Reward | FOLD% |
|---------|--------|-------------|-------------|-------|
| 0       | 20.0   | 1.5         | -2.0        | 100%  |
| 10K     | 8.5    | 1.2         | -1.2        | 85%   |
| 50K     | 4.8    | 1.1         | -0.3        | 45%   |
| 100K    | 3.2    | 1.0         | +0.05       | 35%   |
| 250K    | 1.8    | 0.95        | +0.22       | 28%   |

### SD-CFR (10K iterations)

| Iteration | Loss | Buffer Size | Status    |
|-----------|------|-------------|-----------|
| 0         | -1.0 | 0           | Collecting |
| 500       | 28.5 | 50K         | Training   |
| 2000      | 12.1 | 200K        | Improving  |
| 10000     | 3.2  | 1M+         | Final      |

### Performance Targets

| Model  | Quick (10K) | Medium (100K) | Full (250K+) |
|--------|-------------|---------------|--------------|
| NFSP   | -20% to 0%  | -5% to +5%    | +15–25% EV   |
| SD-CFR | -80% to -50%| -30% to -10%  | +10–20% EV   |

---

## Parallel Training Architecture

`scripts/train_nfsp_parallel.py` achieves 3–4x wall-clock speedup via multiprocessing:

```
Worker 0 ─┐
Worker 1 ─┤─→ Transition Queue ─→ Main Process ─→ Buffers ─→ Training
Worker 2 ─┤                                           ↑
Worker 3 ─┘                              Weight sync every 1000 eps
```

**Key design decisions:**
- Queue-based (not shared-buffer) to avoid pickle issues with threading.Lock
- Main process is sole writer to buffers — no locking needed during training
- Weight sync every 1000 episodes (~0.4% of training, acceptable staleness)
- Workers block on put() if queue depth exceeds 10,000 (backpressure)

**Performance:**

| Metric            | Sequential (v3) | Parallel       | Speedup |
|-------------------|-----------------|----------------|---------|
| 250K episode time | ~13 min         | ~3.5–4 min     | 3–3.7×  |
| CPU utilization   | 1 core          | 4 cores        | 4×      |
| Memory usage      | ~2 GB           | ~8 GB          | 4×      |
| GPU utilization   | 3–7%            | Same           | No change|

**Thread-safe buffers** (`src/poker/ml/buffers.py`): All three buffer classes (CircularBuffer, ReservoirBuffer, WeightedReservoirBuffer) have `threading.Lock()` protecting all public methods.

---

## Monitoring During Training

### Automatic (logged every `--eval-every` episodes)

```
Ep  50000/250000 | q_loss=4.5810 | policy_loss=1.1234 | eval=-0.2105 | elapsed=1842s | ETA=1658m
```

### Manual Checkpoint Validation

```bash
python scripts/validate_training.py models/nfsp_v3_ckpt_050000.pt --hands 100
```

Expected output:
```
Performance vs Opponents:
  RandomBot    :  -0.0524 reward/hand  ~ NEUTRAL
  CallBot      :   0.1234 reward/hand  ✓ POSITIVE
  FlopBot      :  -0.0891 reward/hand  ~ NEUTRAL

Action Distribution:
  FOLD:  45%   CHECK/CALL: 28%   RAISE: 22%   ALL_IN: 5%
```

### Key Metrics

| Metric       | Target (ep 100K) | Warning sign           |
|--------------|------------------|------------------------|
| q_loss       | < 5.0            | > 10 at any point      |
| policy_loss  | 1.0–1.5          | > 2.0 (diverging)      |
| eval_reward  | > -0.3           | < -1.0 at ep 100K      |
| FOLD%        | 30–40%           | > 80% after ep 50K     |

---

## Troubleshooting

### FOLD% stuck at 100% after 50K episodes

1. Verify running v3: `grep "NO ARTIFICIAL BONUS" scripts/train_nfsp_v3.py`
2. Check `env.py` has no anti-fold bonus code
3. Confirm reward scaling is `10.0` (not `1.0` or `50.0`)

### q_loss stuck above 10

1. Reduce learning rate: add `--lr 0.0002`
2. Train longer (more exploration needed)
3. Check GPU is being used: look for `[GPU]` in output

### eval_reward very negative (< -1.5) past episode 50K

- Expected at ep 0–10K (normal)
- Should reach > -0.3 by ep 50K
- If still below -1.0 at ep 100K: reward signal is corrupted

### GPU not detected

```bash
python -c "import torch; print(torch.cuda.is_available())"
# If False: check nvidia-smi and CUDA installation
```

### Out of memory (GPU)

- Reduce `--num-players` to 2
- Reduce batch size in script (requires code change)
- Fall back to CPU (10–50x slower)

---

## Multi-Generation Training (Optional)

Train subsequent generations against previous ones for population-based self-play:

```bash
# Gen 0 (baseline)
python scripts/train_nfsp_selfplay.py --episodes 250000

# Continue training from checkpoint (loads previous model as frozen opponent)
python scripts/train_nfsp_selfplay.py --episodes 250000 --checkpoint models/nfsp_4player.pt
```

---

## File Reference

| File | Purpose |
|------|---------|
| `scripts/train_nfsp_selfplay.py` | NFSP 4-player self-play training (recommended) |
| `scripts/train_nfsp_production.py` | NFSP 2-player production training with full logging |
| `scripts/train_sdcfr_extended.py` | SD-CFR extended training with checkpointing |
| `scripts/train_sdcfr.py` | SD-CFR standard training |
| `scripts/validate_training.py` | Validate NFSP/SD-CFR checkpoints anytime |
| `scripts/test_sdcfr_s1.py` | SD-CFR model evaluation |
| `scripts/test_4player_game.py` | 4-player game harness for environment validation |
| `src/poker/ml/env.py` | RL environment (10× reward scaling) |
| `src/poker/ml/models/nfsp_model.py` | NFSP algorithm |
| `src/poker/ml/models/sdcfr_model.py` | SD-CFR algorithm |
| `src/poker/ml/buffers.py` | Thread-safe replay buffers |
| `src/poker/bots/call_bot.py` | CallBot (calling station opponent) |
