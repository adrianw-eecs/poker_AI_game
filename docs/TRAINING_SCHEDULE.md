# Training Schedule & Validation Guide

## Quick Test Phase (15 minutes each)

Start with these quick tests to validate the implementation before committing to full training.

### Phase 1a: Quick NFSP Test

```bash
python scripts/quick_train_nfsp.py --num-players 2 --episodes 10000 --eval-every 2000
```

**Expected Output**:
```
✅ GPU Detected: NVIDIA GeForce RTX 3060
   Memory: 12.0GB

======================================================================
Quick NFSP Training Test: 10,000 episodes
======================================================================
Players: 2, Eval every: 2,000 episodes
Expected time: ~15 minutes on RTX 3060+

Ep      0/  10000 | q_loss=n/a | policy_loss=n/a | eval=0.0300 | elapsed=   0.0s | ETA=15.2m
Ep   2000/  10000 | q_loss=0.1234 | policy_loss=0.5678 | eval=0.0455 | elapsed= 123.4s | ETA=12.8m
Ep   4000/  10000 | q_loss=0.0987 | policy_loss=0.4321 | eval=0.0789 | elapsed= 246.8s | ETA=11.2m
Ep   6000/  10000 | q_loss=0.0654 | policy_loss=0.3456 | eval=0.1234 | elapsed= 370.2s | ETA= 9.6m
Ep   8000/  10000 | q_loss=0.0432 | policy_loss=0.2789 | eval=0.1567 | elapsed= 493.6s | ETA= 8.1m
Ep  10000/  10000 | q_loss=0.0321 | policy_loss=0.2345 | eval=0.1890 | elapsed= 617.0s | ETA= 0.0m

======================================================================
Quick training complete!
Total time: 617.0s (10.3m)
Avg time per episode: 0.0617s
Saved model to models/nfsp_quick_test.pt
======================================================================
```

**Validation Checklist**:
- ✅ GPU is detected and being used
- ✅ Q-loss is decreasing (0.12 → 0.03)
- ✅ Policy-loss is decreasing (0.57 → 0.23)
- ✅ Eval reward is improving (0.03 → 0.19)
- ✅ No CUDA out-of-memory errors
- ✅ Training completes in ~10 minutes

### Phase 1b: Quick SD-CFR Test

```bash
python scripts/quick_train_sdcfr.py --num-players 2 --cfr-iterations 200 --traversals-per-iteration 500
```

**Expected Output**:
```
✅ GPU Detected: NVIDIA GeForce RTX 3060
   Memory: 12.0GB

======================================================================
Quick SD-CFR Training Test: 200 iterations
======================================================================
Players: 2, Traversals per iter: 500
Expected time: ~15 minutes on RTX 3060+

Iter    0/ 200 | {'iteration': 0, 'loss': -1.0, 'buffer_size': 0} | elapsed=  0.0s | avg_iter=0.00s | ETA=15.0m
Iter   50/ 200 | {'iteration': 50, 'loss': 0.8234, 'buffer_size': 12500} | elapsed= 456.7s | avg_iter=9.13s | ETA=12.1m
Iter  100/ 200 | {'iteration': 100, 'loss': 0.5678, 'buffer_size': 25000} | elapsed= 913.4s | avg_iter=9.13s | ETA= 8.2m
Iter  150/ 200 | {'iteration': 150, 'loss': 0.3456, 'buffer_size': 37500} | elapsed=1370.1s | avg_iter=9.13s | ETA= 4.1m
Iter  200/ 200 | {'iteration': 200, 'loss': 0.2345, 'buffer_size': 50000} | elapsed=1826.8s | avg_iter=9.13s | ETA= 0.0m

======================================================================
Quick training complete!
Total time: 1826.8s (30.4m)
Avg time per iteration: 9.13s
Saved final model to models/sdcfr_quick_test.pt
======================================================================
```

**Validation Checklist**:
- ✅ GPU is detected and being used
- ✅ Loss is decreasing (0.82 → 0.23)
- ✅ Buffer is filling correctly
- ✅ No CUDA out-of-memory errors
- ✅ ~9s per iteration is expected
- ✅ Training completes in ~30 minutes

---

## Medium Training Phase (2-4 hours each)

Once quick tests pass, scale up to medium training to get meaningful performance improvements.

### Phase 2a: Medium NFSP Training

```bash
python scripts/train_nfsp.py --num-players 2 --episodes 100000 --eval-every 10000 --generation 0
```

**Expected time**: ~2 hours on RTX 3060+

**Output milestones**:
- 10K episodes: ~12 minutes
- 50K episodes: ~60 minutes
- 100K episodes: ~120 minutes

**Scaling formula**: Time ≈ episodes / 833 (minutes)

### Phase 2b: Medium SD-CFR Training

```bash
python scripts/quick_train_sdcfr.py --num-players 2 --cfr-iterations 2000 --traversals-per-iteration 500
```

**Expected time**: ~4-5 hours on RTX 3060+

**Output milestones**:
- 500 iterations: ~75 minutes
- 1000 iterations: ~150 minutes
- 2000 iterations: ~300 minutes

**Scaling formula**: Time ≈ iterations / 6.7 (minutes)

---

## Full Training Phase (18-24 hours each)

Run full training once medium tests show good convergence.

### Phase 3a: Full NFSP Training (Gen 0)

```bash
python scripts/train_nfsp.py --num-players 2 --episodes 500000 --eval-every 10000 --generation 0
```

**Expected time**: ~20 hours on RTX 3060+

**Monitoring**:
```bash
# In another terminal, watch GPU usage
nvidia-smi
nvidia-smi dmon  # More detailed stats
```

**Checkpoints**: Every 50K episodes (saved as `nfsp_gen0_ckpt_50000.pt`, etc.)

### Phase 3b: Full SD-CFR Training

```bash
python scripts/train_sdcfr.py --num-players 2 --cfr-iterations 10000 --traversals-per-iteration 1000
```

**Expected time**: ~18 hours on RTX 3060+

**Checkpoints**: Every 500 iterations (saved as `sdcfr_ckpt_500.pt`, etc.)

---

## Multi-Generation Training (Optional)

After Gen 0 completes, train subsequent generations with population-based self-play:

```bash
# Gen 1: Train against Gen 0 + others
python scripts/train_nfsp.py --num-players 2 --episodes 500000 --eval-every 10000 --generation 1

# Gen 2: Train against Gen 0, Gen 1 + others
python scripts/train_nfsp.py --num-players 2 --episodes 500000 --eval-every 10000 --generation 2

# Continue to Gen 10 for best performance
```

**Total time for 3 generations**: ~60 hours wall-clock (can parallelize on multiple GPUs)

---

## Testing Quick Improvements

After each training phase, test the model:

```bash
# Test NFSP Gen 0
python scripts/test_nfsp_s1.py

# Test SD-CFR
python scripts/test_sdcfr_s1.py

# Compare vs baseline
# Expected: NFSP improves from -31% → 0-10%+, SD-CFR from -98% → -50% to 0%+
```

---

## Recommended Schedule

| Phase | Task | Duration | When | Checkpoint |
|-------|------|----------|------|------------|
| 1a | Quick NFSP | 15m | Day 0, 9am | `nfsp_quick_test.pt` |
| 1b | Quick SD-CFR | 15m | Day 0, 9:30am | `sdcfr_quick_test.pt` |
| Review | Analyze results | 30m | Day 0, 10am | N/A |
| 2a | Medium NFSP (100K) | 2h | Day 0, 11am | `nfsp.pt` (partial) |
| 2b | Medium SD-CFR (2K) | 4h | Day 0, 1pm | `sdcfr.pt` (partial) |
| Test | Benchmark both | 30m | Day 0, 5:30pm | Metrics |
| 3a | Full NFSP Gen 0 | 20h | Day 1, 6pm–Day 2, 2pm | `nfsp_gen_0.pt` |
| 3b | Full SD-CFR | 18h | Day 1, 6pm–Day 2, 12am | `sdcfr.pt` |
| Test | Final benchmark | 1h | Day 2, 2pm | Final metrics |

**Total time**: ~50 hours of compute (can overlap phases with multiple GPUs)

---

## Performance Targets by Phase

### NFSP Expected Win Rates

| Phase | Episodes | Expected | Notes |
|-------|----------|----------|-------|
| Quick | 10K | -20% to 0% | Initial learning |
| Medium | 100K | -5% to +5% | Good convergence |
| Full | 500K | +10% to +20% | Near optimal |
| Multi-Gen | 2.5M | +20% to +35% | Population benefits |

### SD-CFR Expected Win Rates

| Phase | Iterations | Expected | Notes |
|-------|------------|----------|-------|
| Quick | 200 | -80% to -50% | Initial buffer fill |
| Medium | 2K | -30% to -10% | Advantage network learns |
| Full | 10K | +5% to +20% | Regret minimization |

---

## Troubleshooting

### GPU Not Detected
```bash
python -c "import torch; print(torch.cuda.is_available())"
# If False, check: nvidia-smi, CUDA installation
```

### Out of Memory
- Reduce `--batch-size` in script (not exposed yet, would need code change)
- Reduce `--num-players` to 2 (smallest configuration)
- Use older GPU or CPU (will be much slower)

### Slow Training (< 5 ep/sec in NFSP)
- Check GPU usage: `nvidia-smi` should show GPU%>80%
- If CPU-only: training will be 10-50x slower (expected)
- Consider reducing iterations for validation only

### Model Not Improving
- For NFSP: check that Q-loss and policy-loss are decreasing
- For SD-CFR: check that loss is decreasing over iterations
- Try with `--seed` for reproducibility
- Verify opponent bots are working (check evaluation reward)

---

## Commands Summary

```bash
# Quick tests (30 min total)
python scripts/quick_train_nfsp.py --num-players 2 --episodes 10000
python scripts/quick_train_sdcfr.py --num-players 2 --cfr-iterations 200

# Medium tests (6 hours total)
python scripts/train_nfsp.py --num-players 2 --episodes 100000 --generation 0
python scripts/train_sdcfr.py --num-players 2 --cfr-iterations 2000

# Full training (40 hours total, can parallelize)
python scripts/train_nfsp.py --num-players 2 --episodes 500000 --generation 0
python scripts/train_sdcfr.py --num-players 2 --cfr-iterations 10000

# Test results
python scripts/test_nfsp_s1.py
python scripts/test_sdcfr_s1.py

# Monitor during training
watch -n 1 nvidia-smi
```

---

## Success Criteria

✅ **All Phases Complete**:
- [ ] Quick NFSP finishes in <15 min with improving losses
- [ ] Quick SD-CFR finishes in <15 min with improving loss
- [ ] Medium NFSP shows +5% win rate improvement over quick
- [ ] Medium SD-CFR shows +20% win rate improvement over quick
- [ ] Full NFSP Gen 0 shows +15% win rate vs baseline (-31%)
- [ ] Full SD-CFR shows +10% win rate vs baseline (-98%)
- [ ] Multi-Gen training shows cumulative improvements

Expected final performance:
- **NFSP**: -31% → **+20-35%** (46-66 pp improvement)
- **SD-CFR**: -98% → **+10-25%** (108-123 pp improvement)
