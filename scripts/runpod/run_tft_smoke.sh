#!/usr/bin/env bash
# run_tft_smoke.sh — NeuralForecast TFT smoke test on RunPod GPU
# Run from project root: bash scripts/runpod/run_tft_smoke.sh
#
# Config: n_trials=1, n_splits=2, max_steps=200, windows_batch_size=1024
# Expected: ~5-10 min on A100, ~10-20 min on RTX 4000+
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "============================================"
echo "  NeuralForecast TFT Smoke Test"
echo "============================================"
echo ""

# Show config
echo "Config:"
echo "  max_steps:          $(grep 'max_steps:' configs/models/tft.yaml | head -1 | awk '{print $2}')"
echo "  windows_batch_size: $(grep 'windows_batch_size:' configs/models/tft.yaml | head -1 | awk '{print $2}')"
echo "  accelerator:        $(grep 'accelerator:' configs/models/tft.yaml | head -1 | awk '{print $NF}' | tr -d '"')"
echo "  precision:          $(grep 'precision:' configs/models/tft.yaml | head -1 | awk '{print $NF}' | tr -d '"')"
echo "  n_trials:           $(grep -A1 '^tft:' configs/models/hyperparameters.yaml | grep 'n_trials' | awk '{print $2}')"
echo "  n_splits:           $(grep 'n_splits:' configs/models/hyperparameters.yaml | awk '{print $2}')"
echo ""

# CUDA check
echo "GPU info:"
uv run python -c "
import torch
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_mem / (1024**3)
    print(f'  GPU:  {gpu_name}')
    print(f'  VRAM: {vram:.1f} GB')
    print(f'  CUDA: {torch.version.cuda}')
else:
    print('  [ERROR] CUDA not available!')
    exit(1)
"
echo ""

START=$(date +%s)

echo "Starting TFT training..."
echo ""

uv run python -m energy_forecast.training.run \
    --model tft \
    --no-mlflow \
    2>&1 | tee logs/tft_nf_smoke.log

ELAPSED=$(( $(date +%s) - START ))

echo ""
echo "============================================"
echo "  TFT Smoke Test Complete"
echo "  Time: $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s"
echo "============================================"
echo ""

# Show results summary
echo "Results from log:"
grep -E "(Val MAPE|Test MAPE|MAPE|it/s|steps)" logs/tft_nf_smoke.log | tail -10
echo ""
echo "Full log: logs/tft_nf_smoke.log"
