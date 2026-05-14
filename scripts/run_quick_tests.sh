#!/bin/bash
# Master quick test runner - validates both NFSP and SD-CFR in ~30 minutes

set -e  # Exit on error

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPTS_DIR")"

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║         POKER AI - QUICK VALIDATION TEST SUITE                    ║"
echo "║                    (Expected time: ~30 min)                       ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Check GPU
echo "🔍 Checking GPU availability..."
python -c "
import torch
if torch.cuda.is_available():
    print(f'✅ GPU Available: {torch.cuda.get_device_name()}')
    print(f'   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
else:
    print('⚠️  No GPU detected - training will be slow')
    print('   CPU-only training: expect 5-10x longer times')
"
echo ""

# Run NFSP quick test
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Phase 1a: Quick NFSP Training (10K episodes, ~10-15 min)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
NFSP_START=$(date +%s)
python "$SCRIPTS_DIR/quick_train_nfsp.py" \
    --num-players 2 \
    --episodes 10000 \
    --eval-every 2000 \
    --save-path "models/nfsp_quick_test.pt" \
    --seed 42
NFSP_END=$(date +%s)
NFSP_TIME=$((NFSP_END - NFSP_START))
echo "✅ NFSP quick test completed in ${NFSP_TIME}s"
echo ""

# Run SD-CFR quick test
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Phase 1b: Quick SD-CFR Training (200 iterations, ~15-20 min)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
SDCFR_START=$(date +%s)
python "$SCRIPTS_DIR/quick_train_sdcfr.py" \
    --num-players 2 \
    --cfr-iterations 200 \
    --traversals-per-iteration 500 \
    --save-path "models/sdcfr_quick_test.pt" \
    --seed 42
SDCFR_END=$(date +%s)
SDCFR_TIME=$((SDCFR_END - SDCFR_START))
echo "✅ SD-CFR quick test completed in ${SDCFR_TIME}s"
echo ""

# Summary
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                    QUICK TEST SUMMARY                             ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "📈 Results:"
echo "   ✅ NFSP (10K episodes):  ${NFSP_TIME}s ($((NFSP_TIME / 60))m)"
echo "   ✅ SD-CFR (200 iters):   ${SDCFR_TIME}s ($((SDCFR_TIME / 60))m)"
echo ""
TOTAL_TIME=$((NFSP_TIME + SDCFR_TIME))
echo "⏱️  Total time: ${TOTAL_TIME}s ($((TOTAL_TIME / 60))m $((TOTAL_TIME % 60))s)"
echo ""

echo "📁 Models saved:"
echo "   • models/nfsp_quick_test.pt"
echo "   • models/sdcfr_quick_test.pt"
echo ""

echo "🎯 Next steps:"
echo "   1. Review the output above for:"
echo "      • Decreasing Q-loss and policy-loss (NFSP)"
echo "      • Decreasing loss (SD-CFR)"
echo "      • Improving eval rewards"
echo "   2. If all looks good, scale up to medium training:"
echo "      python scripts/train_nfsp.py --episodes 100000 --generation 0"
echo "      python scripts/train_sdcfr.py --cfr-iterations 2000"
echo "   3. After medium tests, run full training:"
echo "      python scripts/train_nfsp.py --episodes 500000 --generation 0"
echo "      python scripts/train_sdcfr.py --cfr-iterations 10000"
echo ""
echo "📖 For full schedule: see docs/TRAINING_SCHEDULE.md"
echo ""
