#!/usr/bin/env bash
# run_training.sh — R3 Full training: CatBoost → TFT → TSMixerx → Ensemble
# Kullanim: RunPod terminalinde calistir
#   bash scripts/runpod/run_training.sh
#
# tmux kullanir — SSH kopsa bile training devam eder
#   tmux attach -t training     → baglan
#   Ctrl+B, D                   → ayir (training devam eder)
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

# GPU memory fragmentation fix
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SESSION_NAME="training"

echo "============================================"
echo "  R3 Production Training"
echo "  CB(30t) → TFT(15t) → TSMixerx → Ensemble"
echo "  CV: 12-fold TSCV"
echo "============================================"
echo ""

# Show config
echo "Config:"
echo "  CatBoost:  30 trials, 12-fold, 229 features, 10000 iter (early stop)"
echo "  TFT:       15 trials, 12-fold, hidden=128, max_steps=10000, bf16"
echo "  TSMixerx:  n_block=2, ff_dim=128, max_steps=3000"
echo "  Ensemble:  weighted average (SLSQP)"
echo ""

# Check tmux
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "[WARNING] tmux session '$SESSION_NAME' already exists."
    echo "  Attach: tmux attach -t $SESSION_NAME"
    echo "  Kill:   tmux kill-session -t $SESSION_NAME"
    exit 1
fi

# --- Main script: CatBoost → TFT → TSMixerx → Ensemble ---
MAIN_SCRIPT=$(mktemp /tmp/main_XXXXXX.sh)
cat > "$MAIN_SCRIPT" << MAINEOF
#!/usr/bin/env bash
set -euo pipefail
cd "$PROJECT_ROOT"

TOTAL_START=\$(date +%s)

echo ""
echo "========================================"
echo "  [1/4] CatBoost Training (CPU)"
echo "  30 trials x 12-fold, 229 features"
echo "========================================"
echo ""
START=\$(date +%s)

uv run python -m energy_forecast.training.run \
    --model catboost \
    --no-mlflow \
    2>&1 | tee catboost_training.log

CB_TIME=\$(( \$(date +%s) - START ))
echo ""
echo "[DONE] CatBoost: \$(( CB_TIME / 60 ))m \$(( CB_TIME % 60 ))s"
echo ""

echo "========================================"
echo "  [2/4] TFT Training (GPU)"
echo "  15 trials x 12-fold, hidden=128, bf16"
echo "========================================"
echo ""
START=\$(date +%s)

uv run python -m energy_forecast.training.run \
    --model tft \
    --no-mlflow \
    2>&1 | tee tft_training.log

TFT_TIME=\$(( \$(date +%s) - START ))
echo ""
echo "[DONE] TFT: \$(( TFT_TIME / 60 ))m \$(( TFT_TIME % 60 ))s"
echo ""

echo "========================================"
echo "  [3/4] TSMixerx Training (GPU)"
echo "  n_block=2, ff_dim=128, max_steps=3000"
echo "========================================"
echo ""
START=\$(date +%s)

uv run python -m energy_forecast.training.run \
    --model tsmixerx \
    --no-mlflow \
    2>&1 | tee tsmixerx_training.log

TSMIX_TIME=\$(( \$(date +%s) - START ))
echo ""
echo "[DONE] TSMixerx: \$(( TSMIX_TIME / 60 ))m \$(( TSMIX_TIME % 60 ))s"
echo ""

echo "========================================"
echo "  [4/4] Ensemble Training"
echo "  Weighted average (SLSQP)"
echo "========================================"
echo ""
START=\$(date +%s)

uv run python -m energy_forecast.training.run \
    --model ensemble \
    --no-mlflow \
    2>&1 | tee ensemble_training.log

E_TIME=\$(( \$(date +%s) - START ))

TOTAL_TIME=\$(( \$(date +%s) - TOTAL_START ))

echo ""
echo "============================================"
echo "  R3 TRAINING COMPLETE!"
echo "============================================"
echo ""
echo "  CatBoost:  \$(( CB_TIME / 60 ))m \$(( CB_TIME % 60 ))s"
echo "  TFT:       \$(( TFT_TIME / 60 ))m \$(( TFT_TIME % 60 ))s"
echo "  TSMixerx:  \$(( TSMIX_TIME / 60 ))m \$(( TSMIX_TIME % 60 ))s"
echo "  Ensemble:  \$(( E_TIME / 60 ))m \$(( E_TIME % 60 ))s"
echo "  ----------------------------"
echo "  Wall clock: \$(( TOTAL_TIME / 60 ))m \$(( TOTAL_TIME % 60 ))s"
echo ""

# Pack results
echo "Packing results..."
bash scripts/runpod/pack_results.sh

echo ""
echo "Models in: models/ and final_models/"
echo "Download via Jupyter file browser or:"
echo "  scp -P PORT root@IP:/workspace/trained_models.tar.gz ~/Desktop/"
echo ""
echo "REMINDER: Stop the pod!"
MAINEOF
chmod +x "$MAIN_SCRIPT"

# --- Launch tmux ---
echo "Starting training in tmux session: $SESSION_NAME"
echo ""

tmux new-session -d -s "$SESSION_NAME" -n "main" "bash $MAIN_SCRIPT"

echo "Training started!"
echo ""
echo "  Window 0 (main): CatBoost → TFT → TSMixerx → Ensemble"
echo ""
echo "Commands:"
echo "  tmux attach -t $SESSION_NAME    → baglan"
echo "  Ctrl+B, D                       → ayir (devam eder)"
echo ""
echo "SSH kopsa bile training devam eder."
