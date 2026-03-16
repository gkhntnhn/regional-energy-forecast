#!/usr/bin/env bash
# run_training.sh — R3 Full training: CB + Prophet parallel → TFT → Ensemble
# Kullanım: RunPod terminalinde çalıştır
#   bash scripts/runpod/run_training.sh
#
# tmux kullanır — SSH kopsa bile training devam eder
#   tmux attach -t training     → bağlan
#   Ctrl+B, 0                   → CatBoost/TFT/Ensemble window
#   Ctrl+B, 1                   → Prophet window
#   Ctrl+B, D                   → ayır (training devam eder)
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

SESSION_NAME="training"
SENTINEL_DIR="/tmp/training_$$"

echo "============================================"
echo "  R3 Production Training"
echo "  CB(30t) + Prophet(30t) parallel → TFT(15t) → Ensemble"
echo "  CV: 12-fold TSCV"
echo "============================================"
echo ""

# Show config
echo "Config:"
echo "  CatBoost:  30 trials, 12-fold, 229 features, 10000 iter (early stop)"
echo "  Prophet:   30 trials, 12-fold, 14 regressors, daily Fourier=10"
echo "  TFT:       15 trials, 12-fold, hidden=128, max_steps=10000, bf16"
echo "  Ensemble:  stacking meta-learner (depth=2)"
echo ""

# Check tmux
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "[WARNING] tmux session '$SESSION_NAME' already exists."
    echo "  Attach: tmux attach -t $SESSION_NAME"
    echo "  Kill:   tmux kill-session -t $SESSION_NAME"
    exit 1
fi

mkdir -p "$SENTINEL_DIR"

# --- Main window: CatBoost → wait Prophet → TFT → Ensemble ---
MAIN_SCRIPT=$(mktemp /tmp/main_XXXXXX.sh)
cat > "$MAIN_SCRIPT" << MAINEOF
#!/usr/bin/env bash
set -euo pipefail
cd "$PROJECT_ROOT"

SENTINEL_DIR="$SENTINEL_DIR"
TOTAL_START=\$(date +%s)

echo ""
echo "========================================"
echo "  [1/4] CatBoost Training (CPU)"
echo "  30 trials × 12-fold, 229 features"
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
touch "\$SENTINEL_DIR/catboost_done"

# Wait for Prophet
if [ ! -f "\$SENTINEL_DIR/prophet_done" ]; then
    echo "[WAITING] Prophet still running... (Ctrl+B, 1 to check)"
    while [ ! -f "\$SENTINEL_DIR/prophet_done" ]; do
        sleep 10
    done
fi
echo "[OK] Prophet done. Starting TFT."
echo ""

echo "========================================"
echo "  [3/4] TFT Training (GPU)"
echo "  15 trials × 12-fold, hidden=128, bf16"
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
echo "  [4/4] Ensemble Training"
echo "  Stacking meta-learner (depth=2)"
echo "========================================"
echo ""
START=\$(date +%s)

uv run python -m energy_forecast.training.run \
    --model ensemble \
    --no-mlflow \
    2>&1 | tee ensemble_training.log

E_TIME=\$(( \$(date +%s) - START ))

P_TIME=0
if [ -f "\$SENTINEL_DIR/prophet_time" ]; then
    P_TIME=\$(cat "\$SENTINEL_DIR/prophet_time")
fi

TOTAL_TIME=\$(( \$(date +%s) - TOTAL_START ))

echo ""
echo "============================================"
echo "  R3 TRAINING COMPLETE!"
echo "============================================"
echo ""
echo "  CatBoost: \$(( CB_TIME / 60 ))m \$(( CB_TIME % 60 ))s"
echo "  Prophet:  \$(( P_TIME / 60 ))m \$(( P_TIME % 60 ))s (parallel)"
echo "  TFT:      \$(( TFT_TIME / 60 ))m \$(( TFT_TIME % 60 ))s"
echo "  Ensemble: \$(( E_TIME / 60 ))m \$(( E_TIME % 60 ))s"
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

rm -rf "\$SENTINEL_DIR"
MAINEOF
chmod +x "$MAIN_SCRIPT"

# --- Prophet window ---
PROPHET_SCRIPT=$(mktemp /tmp/prophet_XXXXXX.sh)
cat > "$PROPHET_SCRIPT" << PROPEOF
#!/usr/bin/env bash
set -euo pipefail
cd "$PROJECT_ROOT"

SENTINEL_DIR="$SENTINEL_DIR"

echo ""
echo "========================================"
echo "  [2/4] Prophet Training (CPU, parallel)"
echo "  30 trials × 12-fold, 14 regressors"
echo "========================================"
echo ""
START=\$(date +%s)

uv run python -m energy_forecast.training.run \
    --model prophet \
    --no-mlflow \
    2>&1 | tee prophet_training.log

P_TIME=\$(( \$(date +%s) - START ))
echo ""
echo "[DONE] Prophet: \$(( P_TIME / 60 ))m \$(( P_TIME % 60 ))s"
echo "\$P_TIME" > "\$SENTINEL_DIR/prophet_time"
touch "\$SENTINEL_DIR/prophet_done"
echo ""
echo "Window 0 will detect this and start TFT."
echo "Switch: Ctrl+B, 0"
PROPEOF
chmod +x "$PROPHET_SCRIPT"

# --- Launch tmux ---
echo "Starting training in tmux session: $SESSION_NAME"
echo ""

tmux new-session -d -s "$SESSION_NAME" -n "main" "bash $MAIN_SCRIPT"
tmux new-window -t "$SESSION_NAME" -n "prophet" "bash $PROPHET_SCRIPT"
tmux select-window -t "$SESSION_NAME:0"

echo "Training started!"
echo ""
echo "  Window 0 (main):    CatBoost → TFT → Ensemble"
echo "  Window 1 (prophet): Prophet (parallel)"
echo ""
echo "Commands:"
echo "  tmux attach -t $SESSION_NAME    → bağlan"
echo "  Ctrl+B, 0/1                     → window değiştir"
echo "  Ctrl+B, D                       → ayır (devam eder)"
echo ""
echo "SSH kopsa bile training devam eder."
