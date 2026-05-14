# Reward System Analysis - Why Model Folds Everything

## Critical Issues Found

### 1. **Folding Reward = 0 (or tiny loss)**
```python
# Line 602 in env.py
normalized_change = float(stack_change) / self.starting_stack
reward = normalized_change

# Example: starting_stack = 1000
# FOLD: stack_change ≈ 0 → reward = 0.0000 ✓ SAFEST
# CALL bad hand: stack_change ≈ -2 → reward = -0.0020 ✗ Worse
# RAISE: stack_change ≈ -5 → reward = -0.0050 ✗ Even worse
```

**Problem**: FOLD has the best reward (zero loss), so model learns to always fold.

---

### 2. **Reward Scale Too Small**
```python
# Main reward at hand end
reward = normalized_change / starting_stack

# 1000 chip stack:
# Win $100 → reward = 0.10 (rare)
# Lose $1   → reward = -0.001 (common when folding poorly)
# Break even → reward = 0.0000 (when folding)

# Decision: FOLD (0.0) vs CALL (−0.001)
# Difference: 0.001 → TOO SMALL to overcome noise
```

**Result**: Model uses FOLD as default safe strategy.

---

### 3. **Intrinsic Reward Barely Matters**
```python
# During hand (line 582)
intrinsic_reward = 0.01 * equity_delta
# equity_delta in [-1, 1]
# intrinsic_reward in [-0.01, 0.01]

# End of hand dominates:
# Intrinsic: max +0.01 across entire hand
# Extrinsic: −0.001 to +0.001 per decision
# Extrinsic wins in importance
```

---

### 4. **No Penalty for Folding**
```python
# When you fold:
stack_change = 0 (blind already paid in reset)
reward = 0.0

# When you play poorly:
stack_change = -5 to -20
reward = -0.005 to -0.020

# Model learns: "Folding is safe, playing is risky"
```

---

## Root Cause: Reward Misalignment

The reward function rewards **minimizing losses** rather than **maximizing wins**.

```
CURRENT (broken):
├─ Fold → reward = 0 (safe)
├─ Call bad hand → reward = -0.005 (bad)
└─ Call good hand → reward = +0.05 (rare)

Model finds: "Always fold = 0 reward = best"

DESIRED:
├─ Fold bad hand → reward = 0.01 (good decision)
├─ Call bad hand → reward = -0.02 (bad decision)  
└─ Call good hand → reward = +0.10 (great decision)
```

---

## Proposed Fixes

### Fix 1: Scale Extrinsic Reward (5-10x boost)
```python
# Instead of:
reward = normalized_change / starting_stack

# Use:
reward = 5.0 * normalized_change  # Amplify stack changes
```

**Effect**: 
- FOLD = 0.0
- CALL bad hand = -0.01 (now matters!)
- WIN pot = +0.50 (strong signal)

### Fix 2: Add Anti-Folding Incentive
```python
# Penalize excessive folding
if action == FOLD:
    reward -= 0.005  # Small cost to folding
else:
    reward += 0.002  # Small bonus for playing
```

**Effect**: Encourages playing hands, not just folding.

### Fix 3: Improve Intrinsic Reward Scale
```python
# Current:
intrinsic_reward = 0.01 * equity_delta

# Better:
intrinsic_reward = 0.05 * equity_delta  # 5x boost
```

**Effect**: Makes progress during hand matter more.

### Fix 4: Blind Normalization
```python
# Normalize rewards by the cost of entry (blinds)
# High stakes = higher reward scale needed
# Low stakes = smaller reward scale ok

cost_of_entry = small_blind + big_blind
reward *= cost_of_entry / big_blind
```

---

## Recommended Implementation Order

1. **Apply Fix 1 first** (easiest, biggest impact)
2. **Apply Fix 3** (boost during-hand signals)
3. **Test with diverse opponents** (verify no folding)
4. **Apply Fix 2 if needed** (anti-folding incentive)

---

## Expected Impact on "Always Fold" Problem

| Fix | Impact |
|-----|--------|
| Fix 1 (5x scale) | FOLD = 0 vs CALL = -0.01 → difference now matters ✓ |
| Fix 3 (5x intrinsic) | Hand progress signals become relevant ✓ |
| Diverse opponents | Forces model to learn playing, not folding ✓ |

With these changes + diverse opponents, model should learn actual poker strategy instead of degenerate FOLD.
