#!/usr/bin/env bash
# pack_results.sh — Package training results for download
# Otomatik olarak run_training.sh sonunda çağrılır
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

# --- Optuna studies (if SQLite) ---
echo ""
if [ -d "models/optuna_studies" ] && ls models/optuna_studies/*.db >/dev/null 2>&1; then
    echo "Packaging Optuna study databases..."
    tar -czf /workspace/optuna_studies.tar.gz models/optuna_studies/
    echo "[OK] /workspace/optuna_studies.tar.gz"
else
    echo "[INFO] No Optuna SQLite studies found."
fi

# --- All-in-one package ---
echo ""
echo "Creating all-in-one package..."
tar -czf /workspace/r3_results.tar.gz \
    models/ \
    final_models/ \
    *_training.log \
    configs/models/ \
    2>/dev/null || true
echo "[OK] /workspace/r3_results.tar.gz ($(du -h /workspace/r3_results.tar.gz | cut -f1))"

# --- Summary ---
echo ""
echo "============================================"
echo "  Files ready in /workspace/"
echo "============================================"
ls -lh /workspace/*.tar.gz 2>/dev/null
echo ""
echo "Download options:"
echo "  1. Jupyter file browser: /workspace/*.tar.gz"
echo "  2. Local terminal:"
echo "     scp -P PORT -i ~/.ssh/id_ed25519 root@IP:/workspace/r3_results.tar.gz ~/Desktop/runpod/"
echo ""
echo "IMPORTANT: Stop the pod after downloading!"
