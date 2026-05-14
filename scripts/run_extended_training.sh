#!/bin/bash
# Extended training runner - validates models with 500K NFSP + 10K SD-CFR iterations

set -e

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPTS_DIR")"

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║        POKER AI - EXTENDED TRAINING SUITE                         ║"
echo "║              (Expected time: ~4 hours total)                      ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Run NFSP extended training
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Phase 1: Extended NFSP Training (500K episodes, ~20 minutes)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
NFSP_START=$(date +%s)
python "$SCRIPTS_DIR/train_nfsp_extended.py" \
    --num-players 2 \
    --episodes 500000 \
    --eval-every 10000 \
    --checkpoint-every 50000 \
    --save-path "models/nfsp_extended.pt"
NFSP_END=$(date +%s)
NFSP_TIME=$((NFSP_END - NFSP_START))
echo "[NFSP] Extended training completed in ${NFSP_TIME}s"
echo ""

# Run SD-CFR extended training
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Phase 2: Extended SD-CFR Training (10K iterations, ~3-4 hours)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
SDCFR_START=$(date +%s)
python "$SCRIPTS_DIR/train_sdcfr_extended.py" \
    --num-players 2 \
    --cfr-iterations 10000 \
    --traversals-per-iteration 1000 \
    --checkpoint-every 500 \
    --save-path "models/sdcfr_extended.pt"
SDCFR_END=$(date +%s)
SDCFR_TIME=$((SDCFR_END - SDCFR_START))
echo "[SD-CFR] Extended training completed in ${SDCFR_TIME}s"
echo ""

# Summary
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                  EXTENDED TRAINING SUMMARY                        ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Results:"
echo "   [NFSP]   500K episodes:  ${NFSP_TIME}s ($((NFSP_TIME / 60))m)"
echo "   [SD-CFR] 10K iterations: ${SDCFR_TIME}s ($((SDCFR_TIME / 60))m)"
echo ""
TOTAL_TIME=$((NFSP_TIME + SDCFR_TIME))
echo "Total time: ${TOTAL_TIME}s ($((TOTAL_TIME / 60))m = $((TOTAL_TIME / 3600))h $((TOTAL_TIME % 3600 / 60))m)"
echo ""

echo "Models saved:"
echo "   • models/nfsp_extended.pt"
echo "   • models/sdcfr_extended.pt"
echo ""

echo "Next steps:"
echo "   1. Test models with:"
echo "      python scripts/test_models.py --nfsp-model models/nfsp_extended.pt --sdcfr-model models/sdcfr_extended.pt --hands 100"
echo "   2. Review performance metrics"
echo "   3. If good results, proceed to full training (500K NFSP + 10K SD-CFR)"
echo ""
