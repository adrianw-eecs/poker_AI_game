# NFSP Training Improvements - Summary

## Problem Identified
Model trained for 250K episodes was folding 100% of hands, achieving break-even performance instead of learning real poker strategy.

**Root Causes**:
1. Single opponent type (RandomBot only) - no diverse strategies to learn from
2. Reward scale too small (0.001 unit differences) - model couldn't differentiate actions
3. Intrinsic rewards too weak (0.01 max) - during-hand progress didn't matter
4. FOLD reward = 0, CALL = small loss → FOLD was optimal

---

## Improvements Implemented

### 1. **New CallBot (Opponent Diversity)**
- **File**: `src/poker/bots/call_bot.py`
- **Strategy**: Always calls, never raises or folds voluntarily
- **Purpose**: Forces model to learn value betting (exploit weak playing)
- **Simplicity**: ~15 lines of code (just returns CALL action)

### 2. **Diverse Opponent Training Script**
- **File**: `scripts/train_nfsp_diverse.py`
- **Opponents**: RandomBot + FlopBot + CallBot (rotating every 33K episodes)
- **Evaluation**: Tests against all three opponent types
- **Expected Impact**: 
  - RandomBot: Teaches exploiting chaos
  - FlopBot: Teaches hand strength awareness
  - CallBot: Teaches value betting aggression

### 3. **Reward System Fixes**

#### Fix 1: Scale Extrinsic Reward 5x
```python
# OLD: reward = normalized_change / starting_stack
# NEW: reward = 5.0 * normalized_change

# Example (1000 chip stack):
# FOLD:            0.00 (was 0.0000)
# CALL bad hand:  -0.05 (was -0.0020) ← NOW VISIBLY WORSE
# WIN small pot:   0.25 (was 0.0100) ← NOW VISIBLY BETTER
```

**Impact**: Reward differences now large enough for model to distinguish actions.

#### Fix 2: Boost Intrinsic Reward 5x
```python
# OLD: intrinsic_reward = 0.01 * equity_delta
# NEW: intrinsic_reward = 0.05 * equity_delta

# Example:
# Equity improves +10%: +0.005 (was +0.001) ← More incentive
```

**Impact**: During-hand progress now meaningful relative to end-of-hand reward.

---

## Expected Outcomes

### Before Fixes
```
Opponent     | Win%  | Fold%  | Notes
-------------|-------|--------|------------------
RandomBot    | 0%    | 100%   | Breaks even (FOLD = safest)
FlopBot      | 0%    | 100%   | Breaks even
CallBot      | 0%    | 100%   | (untested, but folding)
```

### After Fixes + Diverse Training
```
Opponent     | Win%  | Fold%  | Expected Behavior
-------------|-------|--------|------------------
RandomBot    | 30%+  | 40%    | Learns to exploit
FlopBot      | 20%+  | 45%    | Adapts to strength
CallBot      | 50%+  | 20%    | Value bets weakly
```

---

## Training Commands

### Run Diverse Training (Recommended)
```bash
python scripts/train_nfsp_diverse.py \
    --num-players 2 \
    --episodes 250000 \
    --eval-every 10000 \
    --checkpoint-every 50000
```

**Expected time**: ~20 minutes on RTX 3080

### Test the New Model
```bash
python scripts/test_models_debug.py
# (will show action breakdown instead of just-folding)
```

---

## Key Files Changed/Created

**New Files**:
- `src/poker/bots/call_bot.py` - Simple calling station bot
- `scripts/train_nfsp_diverse.py` - Training with 3 opponent types
- `REWARD_SYSTEM_ANALYSIS.md` - Detailed reward system analysis

**Modified Files**:
- `src/poker/ml/env.py` - Applied reward scaling fixes (5x boost)

---

## Why These Fixes Work

1. **Diverse Opponents**: Each opponent type reveals different weaknesses
   - RandomBot → learn to exploit randomness
   - FlopBot → learn hand evaluation  
   - CallBot → learn aggression/value betting

2. **Reward Scaling**: Makes model decisions consequential
   - Before: FOLD vs CALL difference = 0.001 (lost in noise)
   - After: FOLD vs CALL difference = 0.05 (clear signal)

3. **Intrinsic Boosting**: Makes progress during hand matter
   - Before: Improving equity = +0.001 signal (drowned out)
   - After: Improving equity = +0.005 signal (noticeable)

---

## Validation Plan

1. Train NFSP with diverse opponents (250K episodes)
2. Test with `test_models_debug.py` - should see:
   - Action mix (not 100% FOLD)
   - Fold%, Call%, Raise% breakdown
   - Actual pots won/lost
3. Run full test suite (15 hands × 3 runs × 4 scenarios)
4. Compare against previous model (0 wins → should be positive)

---

## Next Steps If Still Failing

If model still folds too much after these fixes:
1. Further increase reward scale (10x instead of 5x)
2. Add explicit anti-folding penalty (−0.01 per fold)
3. Use curriculum learning (start with CallBot only, add complexity)
4. Reduce epsilon decay (longer exploration phase)
