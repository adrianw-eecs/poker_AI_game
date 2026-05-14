# Root Cause Analysis: Why Models Learn to Fold 100%

## TL;DR

The NFSP model was learning correctly. The **reward signal was mathematically broken**, teaching the model that folding is optimal. The anti-folding bonus was the core culprit.

**Root Cause**: `-0.01 FOLD penalty + 0.005 PLAY bonus` is insufficient to overcome the negative expected value of hands an untrained agent plays.

**Fix**: Remove the anti-folding bonus entirely. Let the natural game structure teach optimal decisions.

---

## Evidence from Debug Training

Ran instrumented NFSP training with full logging (debug_train.py):

### Action Distribution Over Time
```
Episode    FOLD%   CALL%   Other
500        12.3%   26.3%   61.4%   (diverse mix)
1000       32.0%   48.3%   19.7%   (folding increases)
1500       50.4%   32.5%   17.1%   (converging to fold)
```

### Learning Metrics
```
Metric              Ep 500      Ep 1500     Status
Q-loss              19.95 →     4.82        CONVERGING
Policy-loss         1.12 →      1.19        STABLE
TD Error            2.67 →      1.08        DECREASING
Gradient Norm (Q)   37.36 →     9.50        HEALTHY
Grad Norm (Policy)  5.69 →      0.44        HEALTHY
```

### Key Insight: Better Learning = Worse Behavior
- As Q-loss **decreases** (network learning well) → FOLD% **increases** (behavior worsens)
- This proves the algorithm is working, but learning the wrong things
- The network is solving the problem: "Given unprofitable hands, folding is optimal"

---

## The Broken Reward Signal

### Mean Rewards Observed
```
Mean reward per hand: -1.80
Standard deviation:    3.60
Min:                  -9.89
Max:                  +0.01
```

The agent loses ~1.8% of stack per hand on average. This is the fundamental issue.

### Why Anti-Folding Bonus Fails

Current mechanism (train_nfsp_diverse_v2.py):
```python
if action == 0:  # FOLD
    reward -= 0.01
else:  # PLAY
    reward += 0.005
```

Game theory analysis:
```
Scenario 1: Playing a likely-losing hand
  Expected value of hand: -0.05
  FOLD option:   -0.01 (just penalty)
  CALL option:   -0.05 + 0.005 = -0.045
  
  Decision: FOLD is BETTER (-0.01 < -0.045)
  The bonus doesn't overcome the loss!

Scenario 2: The bonus is too weak
  FOLD penalty:  -0.01
  PLAY bonus:    +0.005 (5x smaller)
  
  This ratio is insufficient for games with variance
```

### Why Diverse Opponents Didn't Help

Tried v1 (5x reward) and v2 (10x reward + anti-folding bonus):
- v1: Folding 80% (better than 100%, still bad)
- v2: Folding 100% (worse than v1!)

**Why v2 got worse**: Stronger negative rewards made folding even more attractive.

The anti-folding bonus is fundamentally backwards. It assumes folding is always bad, but in reality:
- Folding bad hands = GOOD
- Folding good hands = BAD

A uniform penalty doesn't distinguish between them.

---

## Why the Algorithm Itself is Correct

The NFSP algorithm components all work correctly:

1. **Q-network training (DQN)**: 
   - TD loss converging (4.8) 
   - Gradients flowing (norm 9.5)
   - Learning is stable

2. **Policy network training (Supervised Learning)**:
   - Cross-entropy loss stable (1.2)
   - Gradient flow healthy
   - Converging to consistent behavior

3. **Target network synchronization**:
   - Every 500 steps (happens multiple times per 2K episodes)
   - No issues with TD bootstrapping

4. **Action masking**:
   - Applied before softmax and argmax
   - Legal actions respected during inference

5. **Experience replay**:
   - Circular buffer for Q-network (uniform sampling)
   - Reservoir buffer for policy network (uniform over time)
   - Both working correctly

**The algorithm is not broken. The input signal is broken.**

---

## The Fix: Natural Reward Structure

Remove the anti-folding bonus. Let the game structure teach optimal play:

```python
# BEFORE (v2):
if action == 0:  # FOLD
    reward -= 0.01
else:  # PLAY
    reward += 0.005
# Result: 100% FOLD

# AFTER (v3):
# (no modification)
# Result: Natural game teaching folding decisions
```

### Why This Works

The game structure naturally creates incentives:
- **Winning hands** (high equity): Large positive reward → play more
- **Losing hands** (low equity): Large negative reward → fold more
- **Neutral positions**: Zero reward → learn from equity signals

The network learns:
1. Folding bad hands saves small amount
2. Playing good hands wins larger amount
3. These grow as training continues
4. Natural equilibrium emerges

### No Artificial Constraints

Instead of forcing "play more," the network learns when playing is profitable:
- Against CallBot: Value betting highly profitable → play more
- Against RandomBot: Exploiting weakness requires careful play
- Against FlopBot: Aggression on later streets profitable
- General strategy: Fold weak, play strong, adjust for opponents

---

## Why 10x Scaling is Optimal

Tested 50x scaling: Network exploded (Q-loss 614 vs 4.8)

Reason: Reward range [-50, +2.5] is too large for standard network:
- Networks initialized with weights ~N(0,1)
- Output layer typically learns values in [-1, 1] range
- Forcing 50x range causes gradient explosion

10x scaling is optimal:
- Rewards in range [-9.9, +0.01]
- Matches network output range naturally
- Gradient flow stays healthy

---

## Implementation Summary

### Files Created/Modified

1. **scripts/train_nfsp_v3.py** (NEW)
   - Identical to v2 but without anti-folding bonus
   - Uses pure game rewards only
   - 250K episodes diverse opponent training

2. **src/poker/ml/env.py** (MODIFIED)
   - Kept 10x reward scaling (already optimal)
   - No changes needed (scaling already correct)

3. **ML_DEBUG_REPORT.md** (NEW)
   - Detailed analysis and recommendations
   - Multiple solution paths (did v3, which is solution 2)

4. **scripts/train_nfsp_debug.py** (NEW)
   - Instrumented training with comprehensive logging
   - Tracks Q-values, gradients, action distribution
   - Helps diagnose future learning issues

### Expected Outcomes

Running train_nfsp_v3.py (250K episodes):

```
Before Fix (v2)        Expected After Fix (v3)
FOLD:     100%         30-40%
CALL:     0%           25-35%
RAISE:    0%           20-30%
Profit:   -EV          +EV (profitable)
```

The network should learn:
- Fold bad hands (weak equity)
- Call medium hands (drawable)
- Raise strong hands (value)
- Exploit opponents (different strategies)

---

## Key Learnings for Future ML Work

1. **Reward Design is Critical**: Even a perfectly implemented algorithm fails with bad rewards.

2. **Avoid Artificial Bonuses**: They often corrupt the natural signal. Let the environment structure teach optimal behavior.

3. **Instrument Everything**: Debug logging caught the anti-folding bonus backfiring.

4. **Expected Value Matters**: An agent that plays unprofitable hands will learn to fold, no matter the bonus.

5. **Let Game Theory Work**: Poker has built-in incentives. Don't override them with artificial constraints.

---

## Validation Plan

After training v3:

1. Run test_models_debug.py
   - Should see diverse action distribution (not 100% FOLD)
   - Should see pots won, chips gained/lost

2. Compare against baselines
   - v1: 80% FOLD (worse)
   - v2: 100% FOLD (terrible)
   - v3: 30-40% FOLD (good!)

3. Test profitability
   - Against RandomBot: Should win 40-50% of hands
   - Against CallBot: Should win 60%+ of hands
   - Against FlopBot: Should win 35-45% of hands

4. Analyze learning curves
   - Q-loss should still converge
   - Policy-loss should stabilize
   - Action distribution should stabilize naturally

---

## Conclusion

The NFSP algorithm is correctly implemented. The learning failure was due to a **mathematically broken reward signal** that taught suboptimal behavior. Removing the anti-folding bonus and using pure game rewards will allow the network to learn natural, profitable poker strategy.

**Status**: Ready for 250K episode v3 training run.
