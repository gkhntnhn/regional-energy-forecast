#!/usr/bin/env bash
# run_training_r2.sh — R2 Training: CatBoost → TFT → TSMixerx → Ensemble
# Run from project root: bash scripts/runpod/run_training_r2.sh
#
# Monitoring:
#   tmux attach -t r2-training
#   Ctrl+B, D → Detach (training continues)
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

SESSION_NAME="r2-training"

echo "============================================"
echo "  R2 Production Training Runner"
echo "  Strategy: CB → TFT → TSMixerx → Ensemble"
echo "============================================"
echo ""

# Show current config
echo "Current R2 config:"
echo "  CatBoost trials: $(grep 'n_trials:' configs/models/hyperparameters.yaml | head -1 | awk '{print $2}')"
echo "  TFT trials:      $(grep 'n_trials:' configs/models/hyperparameters.yaml | sed -n '2p' | awk '{print $2}')"
echo "  CV splits:       $(grep 'n_splits:' configs/models/hyperparameters.yaml | awk '{print $2}')"
echo "  CatBoost mode:   $(grep 'task_type:' configs/models/catboost.yaml | awk '{print $2}')"
echo "  TFT epochs:      $(grep 'max_epochs:' configs/models/tft.yaml | awk '{print $2}')"
echo ""

# Check if tmux session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "[WARNING] tmux session '$SESSION_NAME' already exists."
    echo "  Attach: tmux attach -t $SESSION_NAME"
    echo "  Kill:   tmux kill-session -t $SESSION_NAME"
    exit 1
fi

# --- Main script: CatBoost → TFT → TSMixerx → Ensemble ---
MAIN_SCRIPT=$(mktemp /tmp/r2_main_XXXXXX.sh)
cat > "$MAIN_SCRIPT" << MAINEOF
#!/usr/bin/env bash
set -euo pipefail
cd "$PROJECT_ROOT"

TOTAL_START=\$(date +%s)

echo ""
echo "========================================"
echo "  [1/4] CatBoost Training"
echo "  R2: 50 trials, 12-fold, 153 features"
echo "========================================"
echo ""
START=\$(date +%s)

uv run python -m energy_forecast.training.run \
    --model catboost \
    --no-mlflow \
    2>&1 | tee catboost_training.log

CB_TIME=\$(( \$(date +%s) - START ))
echo ""
echo "[DONE] CatBoost completed in \$(( CB_TIME / 60 ))m \$(( CB_TIME % 60 ))s"
echo ""

echo "========================================"
echo "  [2/4] TFT Training (GPU, n_jobs=1)"
echo "  R2: 20 trials, 6 optuna_splits, 50 epochs"
echo "========================================"
echo ""
START=\$(date +%s)

uv run python -m energy_forecast.training.run \
    --model tft \
    --no-mlflow \
    2>&1 | tee tft_training.log

TFT_TIME=\$(( \$(date +%s) - START ))
echo ""
echo "[DONE] TFT completed in \$(( TFT_TIME / 60 ))m \$(( TFT_TIME % 60 ))s"
echo ""

echo "========================================"
echo "  [3/4] TSMixerx Training (GPU)"
echo "========================================"
echo ""
START=\$(date +%s)

uv run python -m energy_forecast.training.run \
    --model tsmixerx \
    --no-mlflow \
    2>&1 | tee tsmixerx_training.log

TSMIX_TIME=\$(( \$(date +%s) - START ))
echo ""
echo "[DONE] TSMixerx completed in \$(( TSMIX_TIME / 60 ))m \$(( TSMIX_TIME % 60 ))s"
echo ""

echo "========================================"
echo "  [4/4] Ensemble Training"
echo "========================================"
echo ""
START=\$(date +%s)

uv run python -m energy_forecast.training.run \
    --model ensemble \
    --no-mlflow \
    2>&1 | tee ensemble_training.log

E_TIME=\$(( \$(date +%s) - START ))
echo ""
echo "[DONE] Ensemble completed in \$(( E_TIME / 60 ))m \$(( E_TIME % 60 ))s"
echo ""

TOTAL_TIME=\$(( \$(date +%s) - TOTAL_START ))
echo "============================================"
echo "  R2 TRAINING COMPLETE!"
echo "============================================"
echo ""
echo "  CatBoost:  \$(( CB_TIME / 60 ))m \$(( CB_TIME % 60 ))s"
echo "  TFT:       \$(( TFT_TIME / 60 ))m \$(( TFT_TIME % 60 ))s"
echo "  TSMixerx:  \$(( TSMIX_TIME / 60 ))m \$(( TSMIX_TIME % 60 ))s"
echo "  Ensemble:  \$(( E_TIME / 60 ))m \$(( E_TIME % 60 ))s"
echo "  ----------------------------"
echo "  Wall clock: \$(( TOTAL_TIME / 60 ))m \$(( TOTAL_TIME % 60 ))s"
echo ""
echo "Next: bash scripts/runpod/pack_results.sh"
echo ""
echo "REMINDER: Stop the pod after downloading results!"
MAINEOF
chmod +x "$MAIN_SCRIPT"

# --- Launch tmux ---
echo "Starting R2 training in tmux session: $SESSION_NAME"
echo ""

tmux new-session -d -s "$SESSION_NAME" -n "main" "bash $MAIN_SCRIPT"

echo "R2 training started in tmux."
echo ""
echo "Layout:"
echo "  Window 0 (main): CatBoost → TFT → TSMixerx → Ensemble"
echo ""
echo "Commands:"
echo "  Attach:              tmux attach -t $SESSION_NAME"
echo "  Detach (safe):       Ctrl+B, D"
echo "  Check if running:    tmux ls"
echo ""
echo "You can safely disconnect SSH — training continues in tmux."
