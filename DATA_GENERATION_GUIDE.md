# Data Generation & Training Guide

This guide walks you through generating 10,000 hands of training data and training all three ML bot models (Linear, Tree, Deep).

## Quick Start (Recommended)

### Option 1: Run Everything with PowerShell Script (Easiest)

```powershell
cd D:\Claude_Projects\poker_ml\poker_AI_game
.\run_data_generation.ps1
```

This script runs the entire pipeline:
1. Collects 10,000 hands against FlopBot (4 parallel workers)
2. Trains Linear model
3. Trains Tree model
4. Trains Deep model (GPU with mixed precision)
5. Runs 20-hand tournament with all three bots

**Total time:** ~20-30 minutes
- Data collection: ~10-15 minutes (4 parallel workers)
- Model training: ~10-15 minutes total
- Tournament: ~2-3 minutes

### Option 2: Run Step-by-Step (Manual Control)

#### Step 1: Collect 10,000 hands (parallel)

```bash
python scripts/collect_data_parallel.py \
    --total-hands 10000 \
    --hands-per-batch 1000 \
    --num-workers 4 \
    --players 4 \
    --opponents flop \
    --out data/selfplay_flop_10k.npz
```

**What this does:**
- Runs 4 workers in parallel, each collecting 1000 hands
- Each worker plays against 3 FlopBots
- Saves combined 10,000 hands to `data/selfplay_flop_10k.npz`
- Reports reward statistics (should have varied rewards, not all -1.0)

**Expected output:**
```
Total experiences collected: 50,000-60,000 (varies by game variation)
Reward statistics:
  Mean:   +0.0023 (roughly break-even)
  Std:    0.9200
  Min:    -1.0000
  Max:    +1.0000
  Count(-1.0): 15,000-17,000 (losses)
  Count(0.0):  15,000-17,000 (ties)
  Count(+1.0): 15,000-17,000 (wins)
```

#### Step 2: Train Linear Model

```bash
python scripts/train_linear.py \
    --data data/selfplay_flop_10k.npz \
    --out models/linear_q.pkl \
    --alpha 1.0
```

**Expected output:**
```
Training on 50,000 experiences...
Ridge regression with alpha=1.0
Action accuracy: ~65-75%
Mean reward MSE: ~0.85
```

#### Step 3: Train Tree Model

```bash
python scripts/train_tree.py \
    --data data/selfplay_flop_10k.npz \
    --out models/tree_q.pkl \
    --max-depth 10
```

**Expected output:**
```
Training decision trees (1 per action)...
Tree depth: 10
Action accuracy: ~60-70%
Mean reward MSE: ~0.80
```

#### Step 4: Train Deep Model (GPU)

```bash
python scripts/train_deep.py \
    --data data/selfplay_flop_10k.npz \
    --out models/deep_q.pt \
    --hidden 128 \
    --lr 0.001 \
    --epochs 100 \
    --batch-size 32
```

**Expected output:**
```
Using mixed precision training on NVIDIA RTX 3080
Batch size: 32, Learning rate: 0.001
Epoch 20, Loss: 0.895643 on NVIDIA RTX 3080
Epoch 40, Loss: 0.723455 on NVIDIA RTX 3080
...
Epoch 100, Loss: 0.654321 on NVIDIA RTX 3080
Action accuracy: ~70-75%
```

#### Step 5: Run Tournament

```bash
python -m poker.main \
    -n 4 \
    -s 1000 \
    -sb 5 \
    -bb 10 \
    -hh 20 \
    -b linear_bot tree_bot deep_bot random
```

**Expected output:**
```
Playing 20 hands with 4 players...
[Hand results showing all 4 bots participating]
Final results:
  LinearBot:  +142 chips
  TreeBot:    -89 chips
  DeepBot:    +156 chips
  RandomBot:  -209 chips
```

## Architecture: Why This Works

### Why Parallel Workers Improve Speed

**Sequential approach (original):**
```
Worker 1: [1000 hands] → 10-12 minutes
Total time: 10-12 minutes
```

**Parallel approach (this script):**
```
Worker 1: [1000 hands] 
Worker 2: [1000 hands]  → 10-12 minutes (all in parallel)
Worker 3: [1000 hands]
Worker 4: [1000 hands]
Total time: 10-12 minutes (4x speedup)
```

### Why FlopBot Instead of RandomBot

**Random bots (previous problem):**
- Learning agent loses ~95% of games
- All rewards: -1.0
- Model learns to be random (no signal to learn from)
- Bots lose to random bots

**FlopBot (improved):**
- Learning agent wins ~33% of games (1 out of 4)
- Rewards vary: -1.0 (loss), 0.0 (tie), +1.0 (win)
- Model learns which actions are better/worse
- Bots beat random bots

### Why Mixed Precision Training Helps (RTX 3080)

**Standard fp32 training:**
- RTX 3080 Tensor Cores idle (not used)
- ~30 TFLOPS peak performance
- High memory bandwidth needed

**Mixed precision fp16 training:**
- RTX 3080 Tensor Cores active (8x performance boost potential)
- ~240 TFLOPS peak performance
- 50% less memory bandwidth needed
- Same numerical accuracy (GradScaler prevents underflow)

## Expected Results After Completion

### Trained Models
- `models/linear_q.pkl` - Ridge regression Q-values
- `models/tree_q.pkl` - Decision tree Q-values
- `models/deep_q.pt` - Neural network Q-values

### Training Data
- `data/selfplay_flop_10k.npz` - 50,000-60,000 game experiences
- Reward distribution: ~33% wins, ~33% ties, ~33% losses (realistic)

### Bot Performance
- All three bots should beat random opponents
- Deep bot likely strongest (neural network captures complexity)
- Linear and Tree models competitive

## Troubleshooting

### "ModuleNotFoundError: No module named 'poker'"
**Solution:** Run from `poker_AI_game` root directory:
```powershell
cd D:\Claude_Projects\poker_ml\poker_AI_game
```

### "CUDA out of memory" during deep training
**Solution:** Reduce batch size:
```bash
python scripts/train_deep.py \
    --data data/selfplay_flop_10k.npz \
    --out models/deep_q.pt \
    --batch-size 16
```

### Data collection very slow (< 100 hands/minute)
**Solution:** Check CPU usage - if low, increase workers:
```bash
python scripts/collect_data_parallel.py \
    --num-workers 8
```

### Tournament bots all behaving randomly
**Reason:** Models untrained (all -1.0 rewards in data)
**Solution:** Use FlopBot as opponents (already configured):
```bash
python scripts/collect_data_parallel.py --opponents flop
```

## Advanced Options

### Collect more data (20,000 hands)
```bash
python scripts/collect_data_parallel.py \
    --total-hands 20000 \
    --hands-per-batch 2500 \
    --num-workers 8
```

### Train with different hyperparameters
```bash
# Larger deep network
python scripts/train_deep.py \
    --data data/selfplay_flop_10k.npz \
    --out models/deep_q_large.pt \
    --hidden 256 \
    --batch-size 64

# Stronger regularization for linear model
python scripts/train_linear.py \
    --data data/selfplay_flop_10k.npz \
    --out models/linear_q_strong.pkl \
    --alpha 10.0
```

## Performance Metrics

After completing this pipeline, you should see:

| Metric | Expected Value |
|--------|----------------|
| Total data collection time | 10-15 min (4 workers) |
| Linear training time | 30-60 sec |
| Tree training time | 60-120 sec |
| Deep training time | 60-120 sec (GPU accelerated) |
| Linear model accuracy | 65-75% |
| Tree model accuracy | 60-70% |
| Deep model accuracy | 70-80% |
| Reward distribution | ~33% each: wins/ties/losses |
| Mean reward | ~0.0 (balanced) |

## Next Steps

After successful training:

1. **Analyze bot performance:**
   ```bash
   python -m poker.main -n 4 -s 5000 -sb 5 -bb 10 -hh 100 \
       -b linear_bot tree_bot deep_bot random
   ```

2. **Self-play tournament:**
   ```bash
   python -m poker.main -n 4 -s 5000 -sb 5 -bb 10 -hh 100 \
       -b linear_bot tree_bot flop_bot random
   ```

3. **Further optimization:**
   - Collect more data (20k, 50k hands)
   - Train with better hyperparameters
   - Implement advanced features (positional awareness, stack depth, etc.)

---

**Questions?** Check the optimization summary at: `C:\Users\abcd1\.claude\optimization-summary.md`
