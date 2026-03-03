#!/usr/bin/env bash
# run_training.sh — Run all model trainings sequentially inside tmux
# Run from project root: bash scripts/runpod/run_training.sh
# This script launches tmux and runs all 4 trainings in sequence.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

SESSION_NAME="hpo-training"

echo "============================================"
echo "  Production HPO Training Runner"
echo "============================================"
echo ""

# Show current config
echo "Current HPO config:"
echo "  CatBoost trials: $(grep 'n_trials:' configs/models/hyperparameters.yaml | head -1 | awk '{print $2}')"
echo "  Prophet trials:  $(grep 'n_trials:' configs/models/hyperparameters.yaml | sed -n '2p' | awk '{print $2}')"
echo "  TFT trials:      $(grep 'n_trials:' configs/models/hyperparameters.yaml | sed -n '3p' | awk '{print $2}')"
echo "  CV splits:       $(grep 'n_splits:' configs/models/hyperparameters.yaml | awk '{print $2}')"
echo "  CatBoost GPU:    $(grep 'task_type:' configs/models/catboost.yaml | awk '{print $2}')"
echo "  TFT epochs:      $(grep 'max_epochs:' configs/models/tft.yaml | awk '{print $2}')"
echo ""

# Check if tmux session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "[WARNING] tmux session '$SESSION_NAME' already exists."
    echo "  Attach: tmux attach -t $SESSION_NAME"
    echo "  Kill:   tmux kill-session -t $SESSION_NAME"
    exit 1
fi

# Create training script that runs inside tmux
# Use unquoted heredoc so $PROJECT_ROOT is expanded at write-time.
# All other $ expressions are escaped to defer to runtime.
TRAIN_SCRIPT=$(mktemp /tmp/train_all_XXXXXX.sh)
cat > "$TRAIN_SCRIPT" << TRAINEOF
#!/usr/bin/env bash
set -euo pipefail

cd "$PROJECT_ROOT"

echo ""
echo "========================================"
echo "  [1/4] CatBoost Training (GPU)"
echo "  Expected: ~30-60 min"
echo "========================================"
echo ""
START=\$(date +%s)

uv run python -m energy_forecast.training.run \\
    --model catboost \\
    --no-mlflow \\
    2>&1 | tee catboost_training.log

CB_TIME=\$(( \$(date +%s) - START ))
echo ""
echo "[DONE] CatBoost completed in \$(( CB_TIME / 60 ))m \$(( CB_TIME % 60 ))s"
echo ""

echo "========================================"
echo "  [2/4] Prophet Training (CPU)"
echo "  Expected: ~45-90 min"
echo "========================================"
echo ""
START=\$(date +%s)

uv run python -m energy_forecast.training.run \\
    --model prophet \\
    --no-mlflow \\
    2>&1 | tee prophet_training.log

P_TIME=\$(( \$(date +%s) - START ))
echo ""
echo "[DONE] Prophet completed in \$(( P_TIME / 60 ))m \$(( P_TIME % 60 ))s"
echo ""

echo "========================================"
echo "  [3/4] TFT Training (GPU)"
echo "  Expected: ~60-120 min"
echo "========================================"
echo ""
START=\$(date +%s)

uv run python -m energy_forecast.training.run \\
    --model tft \\
    --no-mlflow \\
    2>&1 | tee tft_training.log

TFT_TIME=\$(( \$(date +%s) - START ))
echo ""
echo "[DONE] TFT completed in \$(( TFT_TIME / 60 ))m \$(( TFT_TIME % 60 ))s"
echo ""

echo "========================================"
echo "  [4/4] Ensemble Training (CPU)"
echo "  Expected: ~10-15 min"
echo "========================================"
echo ""
START=\$(date +%s)

uv run python -m energy_forecast.training.run \\
    --model ensemble \\
    --no-mlflow \\
    2>&1 | tee ensemble_training.log

E_TIME=\$(( \$(date +%s) - START ))
echo ""
echo "[DONE] Ensemble completed in \$(( E_TIME / 60 ))m \$(( E_TIME % 60 ))s"
echo ""

TOTAL_TIME=\$(( CB_TIME + P_TIME + TFT_TIME + E_TIME ))
echo "============================================"
echo "  ALL TRAINING COMPLETE!"
echo "============================================"
echo ""
echo "  CatBoost: \$(( CB_TIME / 60 ))m \$(( CB_TIME % 60 ))s"
echo "  Prophet:  \$(( P_TIME / 60 ))m \$(( P_TIME % 60 ))s"
echo "  TFT:      \$(( TFT_TIME / 60 ))m \$(( TFT_TIME % 60 ))s"
echo "  Ensemble: \$(( E_TIME / 60 ))m \$(( E_TIME % 60 ))s"
echo "  ----------------------------"
echo "  Total:    \$(( TOTAL_TIME / 60 ))m \$(( TOTAL_TIME % 60 ))s"
echo ""
echo "Next: bash scripts/runpod/pack_results.sh"
echo ""
echo "REMINDER: Stop the pod after downloading results!"
TRAINEOF

chmod +x "$TRAIN_SCRIPT"

# Launch in tmux
echo "Starting training in tmux session: $SESSION_NAME"
echo ""
tmux new-session -d -s "$SESSION_NAME" "bash $TRAIN_SCRIPT"

echo "Training started in background tmux session."
echo ""
echo "Commands:"
echo "  Attach (watch live):  tmux attach -t $SESSION_NAME"
echo "  Detach (back to SSH): Ctrl+B then D"
echo "  Check if running:     tmux ls"
echo ""
echo "You can safely disconnect SSH — training continues in tmux."
echo "Reconnect and attach to check progress."
