#!/usr/bin/env bash
# setup_pod.sh — RunPod ortam kurulumu (R3 — feature selection + prod config)
# Kullanım: RunPod terminalinde çalıştır
#   bash setup_pod.sh [GITHUB_REPO_URL]
#
# Ön koşul: Parquet dosyaları SCP ile /workspace/ altına atılmış olmalı
#   scp -P PORT features_historical.parquet features_forecast.parquet root@IP:/workspace/
set -euo pipefail

REPO_URL="${1:-https://github.com/Farukakdmir/regional-energy-forecast.git}"
WORKSPACE="/workspace"
PROJECT_DIR="$WORKSPACE/regional-energy-forecast"

echo "============================================"
echo "  RunPod R3 Setup — RTX PRO 6000 (96GB)"
echo "============================================"
echo ""

# --- Step 1: Clone repo ---
echo "[1/6] Cloning repository..."
if [ -d "$PROJECT_DIR" ]; then
    echo "  Project directory exists. Pulling latest..."
    cd "$PROJECT_DIR"
    git pull
else
    cd "$WORKSPACE"
    git clone "$REPO_URL"
    cd "$PROJECT_DIR"
fi
echo "  Done."
echo ""

# --- Step 2: Copy parquet files ---
echo "[2/6] Copying parquet data files..."
mkdir -p "$PROJECT_DIR/data/processed"
if [ -f "$WORKSPACE/features_historical.parquet" ]; then
    cp "$WORKSPACE/features_historical.parquet" "$PROJECT_DIR/data/processed/"
    cp "$WORKSPACE/features_forecast.parquet" "$PROJECT_DIR/data/processed/"
    echo "  Parquet files copied."
else
    echo "  [ERROR] Parquet files not found in /workspace/"
    echo "  Transfer first:"
    echo "    scp -P PORT features_historical.parquet features_forecast.parquet root@IP:/workspace/"
    exit 1
fi
echo ""

# --- Step 3: Create .env ---
echo "[3/6] Setting up .env..."
cat > "$PROJECT_DIR/.env" << 'ENVEOF'
APP_ENV=development
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
ENVEOF
echo "  Minimal .env created."
echo ""

# --- Step 4: Install dependencies ---
echo "[4/6] Installing dependencies..."
if ! command -v uv &> /dev/null; then
    echo "  Installing uv..."
    pip install uv
fi
echo "  Running uv sync..."
cd "$PROJECT_DIR"
uv sync --all-extras
echo "  Dependencies installed."
echo ""

# --- Step 5: CUDA verification ---
echo "[5/6] Verifying CUDA + GPU..."
uv run python -c "
import torch
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f'  CUDA: OK')
    print(f'  GPU:  {gpu_name}')
    print(f'  VRAM: {vram:.1f} GB')
    print(f'  bf16: {torch.cuda.is_bf16_supported()}')
else:
    print('  [ERROR] CUDA not available!')
    exit(1)
"
echo ""

# --- Step 6: Verify dataset ---
echo "[6/6] Verifying dataset..."
uv run python -c "
import pandas as pd
h = pd.read_parquet('data/processed/features_historical.parquet')
f = pd.read_parquet('data/processed/features_forecast.parquet')
print(f'  Historical: {h.shape[0]:,} rows x {h.shape[1]} features')
print(f'  Forecast:   {f.shape[0]} rows x {f.shape[1]} features')
print(f'  Date range: {h.index.min()} to {h.index.max()}')
"
echo ""

echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
echo "Config summary:"
grep 'n_trials:' configs/models/hyperparameters.yaml | head -3
grep 'n_splits:' configs/models/hyperparameters.yaml
echo ""
echo "Next: Start training"
echo "  bash scripts/runpod/run_training.sh"
echo ""
