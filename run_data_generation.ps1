# PowerShell script to generate 10,000 hands of training data
# This script runs all data collection and training steps

$ErrorActionPreference = "Stop"

Write-Host "================================" -ForegroundColor Cyan
Write-Host "10K HANDS DATA GENERATION PIPELINE" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Check if we're in the right directory
if (-not (Test-Path "scripts\collect_data_parallel.py")) {
    Write-Host "ERROR: Must run from poker_AI_game root directory" -ForegroundColor Red
    exit 1
}

# Create data directory
if (-not (Test-Path "data")) {
    New-Item -ItemType Directory -Path "data" | Out-Null
    Write-Host "Created data directory" -ForegroundColor Green
}

# Step 1: Collect 10,000 hands using 4 parallel workers
Write-Host ""
Write-Host "STEP 1: Collecting 10,000 hands against FlopBot..." -ForegroundColor Yellow
Write-Host "This will use 4 parallel workers (1000 hands each)" -ForegroundColor Yellow
Write-Host "ETA: 10-15 minutes" -ForegroundColor Yellow
Write-Host ""

python scripts/collect_data_parallel.py `
    --total-hands 10000 `
    --hands-per-batch 1000 `
    --num-workers 4 `
    --players 4 `
    --opponents flop `
    --out data/selfplay_flop_10k.npz

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Data collection failed" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "✓ Data collection complete!" -ForegroundColor Green
Write-Host ""

# Step 2: Train Linear Model
Write-Host "STEP 2: Training Linear Regression Model..." -ForegroundColor Yellow
Write-Host "ETA: 1-2 minutes" -ForegroundColor Yellow
Write-Host ""

python scripts/train_linear.py `
    --data data/selfplay_flop_10k.npz `
    --out models/linear_q.pkl `
    --alpha 1.0

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Linear training failed" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "✓ Linear model training complete!" -ForegroundColor Green
Write-Host ""

# Step 3: Train Tree Model
Write-Host "STEP 3: Training Decision Tree Model..." -ForegroundColor Yellow
Write-Host "ETA: 1-2 minutes" -ForegroundColor Yellow
Write-Host ""

python scripts/train_tree.py `
    --data data/selfplay_flop_10k.npz `
    --out models/tree_q.pkl `
    --max-depth 10

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Tree training failed" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "✓ Tree model training complete!" -ForegroundColor Green
Write-Host ""

# Step 4: Train Deep Model (uses GPU with mixed precision)
Write-Host "STEP 4: Training Deep Neural Network (GPU with mixed precision)..." -ForegroundColor Yellow
Write-Host "ETA: 2-3 minutes (RTX 3080 will run mixed precision fp16)" -ForegroundColor Yellow
Write-Host ""

python scripts/train_deep.py `
    --data data/selfplay_flop_10k.npz `
    --out models/deep_q.pt `
    --hidden 128 `
    --lr 0.001 `
    --epochs 100 `
    --batch-size 32

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Deep training failed" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "✓ Deep model training complete!" -ForegroundColor Green
Write-Host ""

# Step 5: Run final tournament with all three bots
Write-Host "STEP 5: Running 20-hand tournament (Linear + Tree + Deep + Random)..." -ForegroundColor Yellow
Write-Host "ETA: 1-2 minutes" -ForegroundColor Yellow
Write-Host ""

python -m poker.main `
    -n 4 `
    -s 1000 `
    -sb 5 `
    -bb 10 `
    -hh 20 `
    -b linear_bot tree_bot deep_bot random

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Tournament failed" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "================================" -ForegroundColor Green
Write-Host "ALL STEPS COMPLETE!" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green
Write-Host ""
Write-Host "Summary:" -ForegroundColor Green
Write-Host "  ✓ 10,000 hands collected against FlopBot (varied rewards)" -ForegroundColor Green
Write-Host "  ✓ Linear model trained (Ridge regression)" -ForegroundColor Green
Write-Host "  ✓ Tree model trained (Decision tree)" -ForegroundColor Green
Write-Host "  ✓ Deep model trained (Neural network with mixed precision)" -ForegroundColor Green
Write-Host "  ✓ Tournament completed (all 3 bots vs random)" -ForegroundColor Green
Write-Host ""
Write-Host "Models saved to:" -ForegroundColor Green
Write-Host "  - models/linear_q.pkl" -ForegroundColor Green
Write-Host "  - models/tree_q.pkl" -ForegroundColor Green
Write-Host "  - models/deep_q.pt" -ForegroundColor Green
Write-Host ""
Write-Host "Data saved to:" -ForegroundColor Green
Write-Host "  - data/selfplay_flop_10k.npz" -ForegroundColor Green
Write-Host ""
