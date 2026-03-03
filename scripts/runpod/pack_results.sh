#!/usr/bin/env bash
# pack_results.sh — Package training results on RunPod pod for download
# Run from project root on the pod: bash scripts/runpod/pack_results.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== Packaging Training Results ==="
echo ""

# --- Trained models ---
echo "Packaging trained models..."
if [ -d "models" ] || [ -d "final_models" ]; then
    tar -czf /workspace/trained_models.tar.gz \
        models/ \
        final_models/ \
        2>/dev/null || true
    echo "[OK] /workspace/trained_models.tar.gz ($(du -h /workspace/trained_models.tar.gz | cut -f1))"
else
    echo "[WARNING] No models/ or final_models/ directories found."
fi

# --- Training logs ---
echo ""
echo "Packaging training logs..."
LOG_COUNT=$(ls -1 *_training.log 2>/dev/null | wc -l)
if [ "$LOG_COUNT" -gt 0 ]; then
    tar -czf /workspace/training_logs.tar.gz *_training.log
    echo "[OK] /workspace/training_logs.tar.gz ($LOG_COUNT log files)"
else
    echo "[INFO] No training log files found."
fi

# --- Production configs (with optimized values) ---
echo ""
echo "Packaging production configs..."
tar -czf /workspace/prod_configs.tar.gz \
    configs/models/hyperparameters.yaml \
    configs/models/catboost.yaml \
    configs/models/tft.yaml \
    configs/models/ensemble.yaml
echo "[OK] /workspace/prod_configs.tar.gz"

# --- Optuna studies (if SQLite) ---
echo ""
if [ -d "models/optuna_studies" ] && ls models/optuna_studies/*.db >/dev/null 2>&1; then
    echo "Packaging Optuna study databases..."
    tar -czf /workspace/optuna_studies.tar.gz models/optuna_studies/
    echo "[OK] /workspace/optuna_studies.tar.gz"
else
    echo "[INFO] No Optuna SQLite studies found."
fi

# --- Summary ---
echo ""
echo "============================================"
echo "  Files ready for download in /workspace/"
echo "============================================"
ls -lh /workspace/*.tar.gz 2>/dev/null
echo ""
echo "Download from local machine:"
echo "  scp -P <PORT> -i ~/.ssh/id_ed25519 root@<POD_IP>:/workspace/*.tar.gz ~/Desktop/"
echo ""
echo "IMPORTANT: Stop the pod after downloading to stop billing!"
