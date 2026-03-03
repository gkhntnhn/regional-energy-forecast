#!/usr/bin/env bash
# run_training_r2.sh — R2 Training: CatBoost + Prophet parallel, then TFT, then Ensemble
# Run from project root: bash scripts/runpod/run_training_r2.sh
#
# Strategy:
#   Window 0 (main):    CatBoost → wait for Prophet → TFT → Ensemble
#   Window 1 (prophet): Prophet (parallel with CatBoost)
#
# Monitoring:
#   tmux attach -t r2-training
#   Ctrl+B, 0 → CatBoost/TFT window
#   Ctrl+B, 1 → Prophet window
#   Ctrl+B, D → Detach (training continues)
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

SESSION_NAME="r2-training"
SENTINEL_DIR="/tmp/r2_training_$$"

echo "============================================"
echo "  R2 Production Training Runner"
echo "  Strategy: CB + Prophet parallel → TFT → Ensemble"
echo "============================================"
echo ""

# Show current config
echo "Current R2 config:"
echo "  CatBoost trials: $(grep 'n_trials:' configs/models/hyperparameters.yaml | head -1 | awk '{print $2}')"
echo "  Prophet trials:  $(grep 'n_trials:' configs/models/hyperparameters.yaml | sed -n '2p' | awk '{print $2}')"
echo "  TFT trials:      $(grep 'n_trials:' configs/models/hyperparameters.yaml | sed -n '3p' | awk '{print $2}')"
echo "  CV splits:       $(grep 'n_splits:' configs/models/hyperparameters.yaml | awk '{print $2}')"
echo "  CatBoost mode:   $(grep 'task_type:' configs/models/catboost.yaml | awk '{print $2}')"
echo "  TFT epochs:      $(grep 'max_epochs:' configs/models/tft.yaml | awk '{print $2}')"
echo "  Features:        153 (R2 pruned)"
echo ""

# Check if tmux session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "[WARNING] tmux session '$SESSION_NAME' already exists."
    echo "  Attach: tmux attach -t $SESSION_NAME"
    echo "  Kill:   tmux kill-session -t $SESSION_NAME"
    exit 1
fi

# Create sentinel directory
mkdir -p "$SENTINEL_DIR"

# --- Main window script: CatBoost → wait Prophet → TFT → Ensemble ---
MAIN_SCRIPT=$(mktemp /tmp/r2_main_XXXXXX.sh)
cat > "$MAIN_SCRIPT" << MAINEOF
#!/usr/bin/env bash
set -euo pipefail
cd "$PROJECT_ROOT"

SENTINEL_DIR="$SENTINEL_DIR"
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
touch "\$SENTINEL_DIR/catboost_done"
echo ""

# Wait for Prophet to finish
if [ ! -f "\$SENTINEL_DIR/prophet_done" ]; then
    echo "[WAITING] Prophet still running... (check window 1: Ctrl+B, 1)"
    while [ ! -f "\$SENTINEL_DIR/prophet_done" ]; do
        sleep 10
    done
    echo "[OK] Prophet finished. Continuing to TFT."
fi
echo ""

echo "========================================"
echo "  [3/4] TFT Training (GPU, n_jobs=1)"
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

# Read Prophet time from sentinel
P_TIME=0
if [ -f "\$SENTINEL_DIR/prophet_time" ]; then
    P_TIME=\$(cat "\$SENTINEL_DIR/prophet_time")
fi

TOTAL_TIME=\$(( \$(date +%s) - TOTAL_START ))
echo "============================================"
echo "  R2 TRAINING COMPLETE!"
echo "============================================"
echo ""
echo "  CatBoost: \$(( CB_TIME / 60 ))m \$(( CB_TIME % 60 ))s"
echo "  Prophet:  \$(( P_TIME / 60 ))m \$(( P_TIME % 60 ))s (parallel)"
echo "  TFT:      \$(( TFT_TIME / 60 ))m \$(( TFT_TIME % 60 ))s"
echo "  Ensemble: \$(( E_TIME / 60 ))m \$(( E_TIME % 60 ))s"
echo "  ----------------------------"
echo "  Wall clock: \$(( TOTAL_TIME / 60 ))m \$(( TOTAL_TIME % 60 ))s"
echo ""
echo "Next: bash scripts/runpod/pack_results.sh"
echo ""
echo "REMINDER: Stop the pod after downloading results!"

# Cleanup sentinels
rm -rf "\$SENTINEL_DIR"
MAINEOF
chmod +x "$MAIN_SCRIPT"

# --- Prophet window script ---
PROPHET_SCRIPT=$(mktemp /tmp/r2_prophet_XXXXXX.sh)
cat > "$PROPHET_SCRIPT" << PROPEOF
#!/usr/bin/env bash
set -euo pipefail
cd "$PROJECT_ROOT"

SENTINEL_DIR="$SENTINEL_DIR"

echo ""
echo "========================================"
echo "  [2/4] Prophet Training (CPU, parallel)"
echo "  R2: 30 trials, 12-fold, 14 regressors"
echo "========================================"
echo ""
START=\$(date +%s)

uv run python -m energy_forecast.training.run \
    --model prophet \
    --no-mlflow \
    2>&1 | tee prophet_training.log

P_TIME=\$(( \$(date +%s) - START ))
echo ""
echo "[DONE] Prophet completed in \$(( P_TIME / 60 ))m \$(( P_TIME % 60 ))s"
echo "\$P_TIME" > "\$SENTINEL_DIR/prophet_time"
touch "\$SENTINEL_DIR/prophet_done"
echo ""
echo "CatBoost window will detect this and proceed to TFT."
echo "Switch to window 0: Ctrl+B, 0"
PROPEOF
chmod +x "$PROPHET_SCRIPT"

# --- Launch tmux with 2 windows ---
echo "Starting R2 training in tmux session: $SESSION_NAME"
echo ""

# Create session with main window (CatBoost → TFT → Ensemble)
tmux new-session -d -s "$SESSION_NAME" -n "catboost" "bash $MAIN_SCRIPT"

# Add Prophet window
tmux new-window -t "$SESSION_NAME" -n "prophet" "bash $PROPHET_SCRIPT"

# Select first window
tmux select-window -t "$SESSION_NAME:0"

echo "R2 training started in tmux with parallel strategy."
echo ""
echo "Layout:"
echo "  Window 0 (catboost): CatBoost → waits for Prophet → TFT → Ensemble"
echo "  Window 1 (prophet):  Prophet (runs in parallel)"
echo ""
echo "Commands:"
echo "  Attach:              tmux attach -t $SESSION_NAME"
echo "  Switch windows:      Ctrl+B, 0 (catboost) / Ctrl+B, 1 (prophet)"
echo "  Detach (safe):       Ctrl+B, D"
echo "  Check if running:    tmux ls"
echo ""
echo "You can safely disconnect SSH — training continues in tmux."
