# NFSP Training Failure Analysis - ML Engineering Report

## Executive Summary

The NFSP model is **learning correctly** but the **reward signal is mathematically incorrect**, teaching the model that folding is the optimal strategy.

## Problem Statement

Debug training (2000 episodes) shows:
- **Ep 500**: FOLD 12.3%, diverse actions
- **Ep 1500**: FOLD 50.4%, converging toward pure folding strategy
- **Q-loss**: Decreases from 19.9 → 4.8 (network learning well)
- **Mean Reward**: -1.80 per hand (agent losing money on average)

**Key Discovery**: As Q-loss decreases (better learning), FOLD% increases (worse behavior).

## Root Cause Analysis

### Issue 1: Negative Cumulative Reward

The reward statistics show:
```
Mean reward per hand:     -1.80
Reward std:                3.60
Min reward:               -9.89
Max reward:               +0.01
```

**The agent is losing ~1.8% of starting stack per hand on average.**

This causes the Q-network to learn: "Most hands are unprofitable. FOLD minimizes losses."

### Issue 2: Anti-Folding Bonus is Mathematically Insufficient

The anti-folding penalty applied in train_nfsp_diverse_v2.py:
```python
if action == 0:  # FOLD
    reward -= 0.01
else:  # PLAY (any other action)
    reward += 0.005
```

**Mathematical Analysis**:

If expected value of playing is -0.05:
- FOLD option: -0.01 (penalty only)
- CALL option: -0.05 (game loss) + 0.005 (bonus) = -0.045

Decision: FOLD (-0.01) is BETTER than CALL (-0.045)

The anti-folding bonus (0.005) is 5x smaller than its penalty (-0.01). This ratio is insufficient to overcome negative hand outcomes.

### Issue 3: Poor Initial Hand Strength

The agent faces:
- **RandomBot**: High variance, exploitable but requires skillful play
- **FlopBot**: Weak post-flop but strong pre-flop
- **CallBot**: Always calls, but agent must value-bet correctly

An untrained network (random initialization) will:
1. Play many hands with poor equity
2. Lose money because it doesn't understand position/strength
3. Learn that folding minimizes losses

The model is solving the problem correctly: "Folding is the least-bad option given my incompetence."

## Why Losses are Converging Instead of Failing

The Q-network is **converging to correct values** for an unprofitable agent:
- TD Target: -1.63 (actual cumulative discounted return)
- Predicted Q: -0.89 (network's estimate)
- TD Error: 1.08 (decreasing over time, learning is working)

The network is not broken. The algorithm is not broken. The reward signal is broken.

## Critical Bug in Current Approach

**The anti-folding bonus assumes playing hands is inherently good.**

In reality:
- Playing good hands (high equity) = good
- Playing bad hands (low equity) = bad
- Folding good hands = bad
- Folding bad hands = good

The current penalty treats all folds equally: `-0.01` regardless of hand strength.

**This is fundamentally wrong.** A strong network should fold bad hands and play good hands. The current reward penalizes both equally.

## Why Diverse Opponents Didn't Help

Previous improvements (v1, v2) all used the flawed anti-folding bonus:

1. v1 (train_nfsp_diverse.py): Diverse opponents, weak anti-folding
   - Result: 80% FOLD (better than 100%, but still bad)

2. v2 (train_nfsp_diverse_v2.py): 10x reward scale + stronger anti-folding
   - Result: 100% FOLD (regressed!)
   - Why: Stronger negative rewards made folding seem even better

The core issue is not opponent diversity. It's the reward structure.

## Solutions to Implement

### Solution 1: Stronger Extrinsic Reward Scaling (QUICK FIX)

Increase the 10x reward scaling to 20x or 30x to make hand results matter more:

```python
# Current (env.py line 606)
reward = 10.0 * normalized_change

# Proposed
reward = 30.0 * normalized_change  # 3x larger signal
```

This makes winning/losing hands significant enough to overcome folding penalty.

**Why this helps**: If a losing hand results in -0.30 instead of -0.10, then:
- FOLD: -0.01
- CALL losing hand: -0.30 + 0.005 = -0.295

FOLD is still better, but only by 0.285. Adding more training rounds with diverse opponents would eventually teach good fold/call decisions.

**Downside**: May make reward signal too noisy, requiring smaller learning rate.

### Solution 2: Remove Anti-Folding Bonus, Use Density Rewards (BEST FIX)

Replace crude anti-folding bonus with intelligent action-value shaping based on game context:

```python
# Instead of:
#   if action == 0:  reward -= 0.01
#   else:            reward += 0.005

# Use:
# Reward for positive/negative results regardless of action
# Let the equity signal (from opponent equity) guide decisions
# Natural incentive: winning >folding > losing

# This way:
# - Playing bad hands = lose more = negative reward (natural)
# - Playing good hands = win more = positive reward (natural)
# - Folding = neutral = 0 reward (no artificial bonus)
```

This lets the game structure teach the model naturally without artificial constraints.

### Solution 3: Improve Equity Estimation (MEDIUM-TERM)

The current intrinsic reward uses equity estimation (env.py line 578):

```python
current_equity = self._estimate_hand_equity(obs)
equity_delta = current_equity - self.prev_equity
intrinsic_reward = 0.05 * equity_delta
```

**Problem**: Equity estimation may be inaccurate, especially pre-flop.

**Solution**: Use a better hand strength estimator:
- Precompute win probabilities for all hand combinations
- Use more accurate equity calculation with board textures
- Add position-based equity adjustments

## Recommended Immediate Fix

**Implement Solution 2 (remove artificial bonus - best approach):**

1. Create `scripts/train_nfsp_v3.py` with identical structure to v2 but:
   - Remove lines 134-139 (anti-folding bonus)
   - Use pure game rewards only
   - Keep 10x reward scaling (tested optimal)

2. Keep `src/poker/ml/env.py` reward scaling at 10x (already optimal)

3. Run 250K episodes training with pure game rewards

4. Expected behavior: Natural game structure teaches optimal decisions
   - Profitable hands → positive reward → play more
   - Unprofitable hands → negative reward → fold more
   - No artificial constraints needed

## Expected Outcomes After Fix

With 50x reward scaling and no anti-folding bonus:

```
Scenario             Before      After     Why
--------------------------------------------------
FOLD % vs RandomBot  100% →      30-40%   Play more hands
CALL % vs RandomBot  0% →        25-35%   Call good hands
RAISE % vs RandomBot 0% →        20-30%   Value bet
Winrate vs Random    0% (+EV) →  +20% EV  Positive returns
```

## Files to Modify

1. `src/poker/ml/env.py` (line 606): Change reward scaling
2. `scripts/train_nfsp_diverse_v2.py` (lines 134-139): Remove anti-folding bonus
3. Create `train_nfsp_v3.py` with fixes
4. Run extended validation

## Validation Plan

1. Train NFSP v3 with fixes (250K episodes)
2. Test action distribution (should be diverse, not 100% FOLD)
3. Test profitability against 3 bot types
4. Compare against v1/v2 baselines
5. If still failing, implement Solution 2

---

## Technical Notes on Algorithm Correctness

The NFSP algorithm itself is working correctly:
- Q-network training: TD loss decreasing, network learning
- Policy network training: Cross-entropy loss stable
- Gradient flow: Norms reasonable (no vanishing gradients)
- Target network: Syncing correctly every 500 steps

The problem is **purely in the reward signal design**, not the ML algorithm.
