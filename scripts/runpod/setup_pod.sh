#!/usr/bin/env bash
# setup_pod.sh — Set up RunPod environment after SSH connection
# Usage: bash setup_pod.sh [GITHUB_REPO_URL]
# Run inside the pod after SCP transfer of runpod_data.tar.gz
set -euo pipefail

REPO_URL="${1:-https://github.com/YOUR_USERNAME/regional-energy-forecast.git}"
WORKSPACE="/workspace"
PROJECT_DIR="$WORKSPACE/regional-energy-forecast"

echo "============================================"
echo "  RunPod Environment Setup"
echo "============================================"
echo ""

# --- Step 1: Clone repo ---
echo "[1/7] Cloning repository..."
if [ -d "$PROJECT_DIR" ]; then
    echo "  Project directory already exists. Pulling latest..."
    cd "$PROJECT_DIR"
    git pull
else
    cd "$WORKSPACE"
    git clone "$REPO_URL"
    cd "$PROJECT_DIR"
fi
echo "  Done."
echo ""

# --- Step 2: Extract data ---
echo "[2/7] Extracting data files..."
if [ -f "$WORKSPACE/runpod_data.tar.gz" ]; then
    tar -xzf "$WORKSPACE/runpod_data.tar.gz"
    echo "  Data extracted."
else
    echo "  [WARNING] runpod_data.tar.gz not found in $WORKSPACE"
    echo "  Transfer it first: scp -P <PORT> -i ~/.ssh/id_ed25519 runpod_data.tar.gz root@<POD_IP>:/workspace/"
    exit 1
fi
echo ""

# --- Step 3: Copy .env ---
echo "[3/7] Setting up .env..."
if [ -f "$WORKSPACE/.env" ]; then
    cp "$WORKSPACE/.env" "$PROJECT_DIR/.env"
    echo "  .env copied."
else
    echo "  [INFO] No .env found. Creating minimal .env..."
    cat > "$PROJECT_DIR/.env" << 'ENVEOF'
APP_ENV=development
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
ENVEOF
    echo "  Minimal .env created (no EPIAS credentials)."
fi
echo ""

# --- Step 4: Install dependencies ---
echo "[4/7] Installing dependencies..."
# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "  Installing uv..."
    pip install uv
fi
echo "  Running uv sync..."
uv sync --all-extras
echo "  Dependencies installed."
echo ""

# --- Step 5: CUDA verification ---
echo "[5/7] Verifying CUDA..."
uv run python -c "
import torch
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_mem / (1024**3)
    print(f'  CUDA: OK')
    print(f'  GPU:  {gpu_name}')
    print(f'  VRAM: {vram:.1f} GB')
else:
    print('  [ERROR] CUDA not available!')
    print('  Check your RunPod template has GPU support.')
"
echo ""

# --- Step 6: CatBoost GPU verification ---
echo "[6/7] Verifying CatBoost GPU support..."
uv run python -c "
try:
    from catboost import CatBoostRegressor
    import numpy as np
    # Quick GPU test with dummy data
    m = CatBoostRegressor(task_type='GPU', iterations=5, verbose=0)
    X = np.random.rand(100, 5)
    y = np.random.rand(100)
    m.fit(X, y)
    print('  CatBoost GPU: OK')
except Exception as e:
    print(f'  [WARNING] CatBoost GPU issue: {e}')
    print('  Falling back to CPU may be needed.')
"
echo ""

# --- Step 7: Prepare dataset ---
echo "[7/7] Preparing dataset (feature engineering)..."
if [ ! -f "$PROJECT_DIR/scripts/prepare_dataset.py" ]; then
    echo "  [ERROR] prepare_dataset.py not found at $PROJECT_DIR/scripts/"
    echo "  Git clone may have failed. Check step 1."
    exit 1
fi
uv run python "$PROJECT_DIR/scripts/prepare_dataset.py" --skip-epias -v

# Verify output
echo ""
echo "Verifying dataset..."
uv run python -c "
import pandas as pd
h = pd.read_parquet('$PROJECT_DIR/data/processed/features_historical.parquet')
f = pd.read_parquet('$PROJECT_DIR/data/processed/features_forecast.parquet')
print(f'  Historical: {h.shape[0]:,} rows x {h.shape[1]} features')
print(f'  Forecast:   {f.shape[0]} rows x {f.shape[1]} features')
print(f'  Date range: {h.index.min()} to {h.index.max()}')
"

echo ""
echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Apply production config for this GPU:"
echo "     bash scripts/runpod/apply_prod_config.sh"
echo ""
echo "  2. Start R2 training (CB + Prophet parallel, then TFT + Ensemble):"
echo "     bash scripts/runpod/run_training_r2.sh"
echo ""
