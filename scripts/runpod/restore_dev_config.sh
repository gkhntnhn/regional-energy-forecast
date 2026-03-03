#!/usr/bin/env bash
# restore_dev_config.sh — Restore dev/smoke-test config values
# Run from project root: bash scripts/runpod/restore_dev_config.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

BACKUP_DIR="configs/models/.dev_backup"

echo "=== Restoring Dev Config ==="

if [ -d "$BACKUP_DIR" ]; then
    cp "$BACKUP_DIR/hyperparameters.yaml" configs/models/hyperparameters.yaml
    cp "$BACKUP_DIR/catboost.yaml" configs/models/catboost.yaml
    cp "$BACKUP_DIR/tft.yaml" configs/models/tft.yaml
    echo "[OK] Restored from backup ($BACKUP_DIR/)"
    echo ""
    echo "Tip: You can also use 'git checkout -- configs/models/' to restore from git."
else
    echo "[INFO] No backup found. Using git restore instead..."
    if git checkout -- configs/models/hyperparameters.yaml configs/models/catboost.yaml configs/models/tft.yaml; then
        echo "[OK] Restored from git."
    else
        echo "[ERROR] Git restore failed. No backup or git history available."
        echo "  You may need to manually edit the config files."
        exit 1
    fi
fi

echo ""
echo "Current config values:"
grep "n_trials:" configs/models/hyperparameters.yaml | head -3
grep "n_splits:" configs/models/hyperparameters.yaml
grep "task_type:" configs/models/catboost.yaml
grep "max_epochs:" configs/models/tft.yaml
grep "n_jobs:" configs/models/tft.yaml
grep "accelerator:" configs/models/tft.yaml
