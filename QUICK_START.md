# Quick Start: Training & Testing Guide

## Overview

This guide walks you through validating and training the improved NFSP and SD-CFR poker AI models. Start with quick 15-minute tests, then scale up progressively.

```
Quick Tests (30 min)
     ↓
  Medium Tests (6 hours)
     ↓
  Full Training (38 hours)
```

## Prerequisites

✅ **GPU Required** (strongly recommended):
```bash
# Verify GPU
python -c "import torch; print('GPU:', torch.cuda.is_available())"
```

---

## Phase 1: Quick Tests (30 minutes)

**Purpose**: Validate implementation, GPU usage, and model improvement.

### Run Tests

```bash
# NFSP quick test (10K episodes, ~10-15 min)
python scripts/quick_train_nfsp.py

# SD-CFR quick test (200 iterations, ~15-20 min)
python scripts/quick_train_sdcfr.py
```

### Expected Improvements

**NFSP**:
- Q-loss: decreases (0.12 → 0.03) ✅
- Policy-loss: decreases (0.57 → 0.23) ✅
- Eval reward: improves (0.03 → 0.19) ✅

**SD-CFR**:
- Loss: decreases (0.82 → 0.23) ✅
- Buffer fills correctly ✅

---

## Phase 2: Medium Training (2-6 hours)

```bash
# NFSP: 100K episodes (~2h on RTX 3060)
python scripts/train_nfsp.py --episodes 100000 --generation 0

# SD-CFR: 2000 iterations (~4h on RTX 3060)
python scripts/train_sdcfr.py --cfr-iterations 2000
```

---

## Phase 3: Full Training (18-24 hours each)

```bash
# NFSP: 500K episodes (~20h on RTX 3060)
python scripts/train_nfsp.py --episodes 500000 --generation 0

# SD-CFR: 10K iterations (~18h on RTX 3060)
python scripts/train_sdcfr.py --cfr-iterations 10000
```

---

## Expected Results

| Model | Baseline | Quick | Medium | Full |
|-------|----------|-------|--------|------|
| NFSP | -31% | -20% to 0% | -5% to +5% | +15-25% |
| SD-CFR | -98% | -80% to -50% | -30% to -10% | +10-20% |

---

## Timing

- Quick Tests: 30 min
- Medium: 6 hours
- Full: 38 hours
- **Total: ~44 hours**

---

## Test Models

```bash
python scripts/test_nfsp_s1.py
python scripts/test_sdcfr_s1.py
```

See `docs/TRAINING_SCHEDULE.md` for complete schedule.
