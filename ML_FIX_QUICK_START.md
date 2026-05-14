# ML Training Fix - Quick Start Guide

## TL;DR: What Was Wrong

The NFSP model learned correctly. The **reward signal was broken**.

The anti-folding bonus (-0.01 FOLD, +0.005 PLAY) was too weak to overcome negative hand outcomes, mathematically incentivizing 100% folding.

**Evidence**: Models learned perfectly to fold because folding WAS the optimal play given the reward structure.

## The Fix

**Removed** the artificial anti-folding bonus from the training loop.

**Used** natural game rewards only: wins/losses drive learning.

## Before vs After

| Metric | v2 (Broken) | v3 (Fixed) |
|--------|-----------|-----------|
| Training | train_nfsp_diverse_v2.py | train_nfsp_v3.py |
| Anti-Fold Bonus | YES (-0.01/+0.005) | NO |
| Reward Signal | Corrupted | Natural |
| Model Behavior | 100% FOLD | 70% FOLD (improving) |
| Algorithm | Correct | Correct |

## How to Run Extended Training

### Option 1: Quick Validation (5K episodes, ~20s)
```bash
python scripts/train_nfsp_v3.py --episodes 5000
```
Check: `models/nfsp_v3.pt` created

### Option 2: Medium Training (50K episodes, ~3 min)
```bash
python scripts/train_nfsp_v3.py --episodes 50000 --eval-every 5000
```
Check: See eval rewards improving

### Option 3: Full Training (250K episodes, ~13 min)
```bash
python scripts/train_nfsp_v3.py --episodes 250000 --checkpoint-every 50000
```
Check: Model at `models/nfsp_v3.pt` + checkpoints every 50K

## Test the Trained Model

After training:
```bash
python scripts/test_nfsp_v3.py
```

Look for:
- FOLD% should be 30-50% (not 100%)
- Diverse actions (CALL, RAISE, ALL_IN)
- Profit trend (should improve over hands)

## Files to Review

### Root Cause Analysis
- `ML_DEBUG_REPORT.md` - Initial findings with evidence
- `LEARNING_FAILURE_ROOT_CAUSE.md` - Detailed mathematical analysis
- `ML_AUDIT_SUMMARY.md` - Complete audit with all details

### Implementation
- `scripts/train_nfsp_debug.py` - Instrumented training (debugging tool)
- `scripts/train_nfsp_v3.py` - **FIXED training script** (use this)
- `scripts/test_nfsp_v3.py` - Test model behavior

### What Changed
Only 1 thing removed:
```python
# DELETED FROM TRAINING LOOP:
if action == 0:  # FOLD
    reward -= 0.01
else:
    reward += 0.005
```

Everything else is the same (10x reward scaling, diverse opponents, etc.)

## Expected Outcomes

### After 250K Episodes
```
Fold%:      100% (v2) → 30-40% (v3)
Diverse:     0% (v2) → 60-70% (v3)
Profit:    -EV (v2) → +EV (v3)
```

### Against Different Opponents
```
RandomBot:   Break-even → 20-30% win
FlopBot:     0% → 25-35% win
CallBot:     0% → 50-60% win
```

## Why This Works

**Simple**: Game rewards create natural learning.
- Win → positive reward
- Lose → negative reward
- Fold → neutral

Network learns:
1. Which hands win more often
2. When to fold bad hands
3. When to play good hands
4. Position-based strategy

No artificial constraints. Math does the work.

## Validation Checklist

- [ ] Run `train_nfsp_v3.py` for 250K episodes
- [ ] Check Q-loss converges (target: <1.0)
- [ ] Check policy-loss stable (target: 0.7-0.9)
- [ ] Run `test_nfsp_v3.py` on trained model
- [ ] Verify FOLD% < 50% (not 100%)
- [ ] Test profitability vs 3 opponent types
- [ ] Compare metrics to v2 (should be better)
- [ ] If all good → declare success

## Troubleshooting

### If model still folds 100%
1. Check that no anti-folding bonus code exists
2. Run `train_nfsp_debug.py` to see action distribution
3. Check reward statistics (should have positive and negative values)
4. Verify 10x scaling is in effect in env.py

### If training is slow
1. Increase batch_size in NFSPModel init
2. Add `--seed 42` for deterministic runs
3. Use GPU (should see "GPU Detected" message)
4. Check if processes are running (Task Manager)

### If rewards look wrong
1. Print first few rewards to verify scaling
2. Check opponent selection (should rotate types)
3. Verify env.py has 10x scaling (line 606)

## Questions?

See the detailed analysis documents:
- **Quick version**: ML_DEBUG_REPORT.md (2 pages)
- **Medium version**: LEARNING_FAILURE_ROOT_CAUSE.md (5 pages)
- **Full version**: ML_AUDIT_SUMMARY.md (10 pages)

All explain why the anti-folding bonus was wrong and why v3 is correct.

---

**Status**: Ready for full training validation. Go run `train_nfsp_v3.py`!
