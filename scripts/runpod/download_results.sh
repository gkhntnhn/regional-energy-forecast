#!/usr/bin/env bash
# download_results.sh — Download training results from RunPod to local machine
# Run locally: bash scripts/runpod/download_results.sh <POD_IP> <SSH_PORT>
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: bash scripts/runpod/download_results.sh <POD_IP> <SSH_PORT>"
    echo ""
    echo "Example: bash scripts/runpod/download_results.sh 194.68.245.99 22288"
    echo ""
    echo "Find POD_IP and PORT in RunPod dashboard → Connect → SSH"
    exit 1
fi

POD_IP="$1"
SSH_PORT="$2"
SSH_KEY="${3:-$HOME/.ssh/id_ed25519}"
DOWNLOAD_DIR="${4:-$HOME/Desktop/runpod_results}"

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

echo "=== RunPod Results Downloader ==="
echo "  Pod:      $POD_IP:$SSH_PORT"
echo "  SSH Key:  $SSH_KEY"
echo "  Download: $DOWNLOAD_DIR"
echo ""

mkdir -p "$DOWNLOAD_DIR"

# Download all tar.gz files from /workspace/
echo "[1/4] Downloading trained models..."
scp -P "$SSH_PORT" -i "$SSH_KEY" \
    "root@${POD_IP}:/workspace/trained_models.tar.gz" \
    "$DOWNLOAD_DIR/" 2>/dev/null && echo "  OK" || echo "  [SKIP] Not found"

echo "[2/4] Downloading training logs..."
scp -P "$SSH_PORT" -i "$SSH_KEY" \
    "root@${POD_IP}:/workspace/training_logs.tar.gz" \
    "$DOWNLOAD_DIR/" 2>/dev/null && echo "  OK" || echo "  [SKIP] Not found"

echo "[3/4] Downloading production configs..."
scp -P "$SSH_PORT" -i "$SSH_KEY" \
    "root@${POD_IP}:/workspace/prod_configs.tar.gz" \
    "$DOWNLOAD_DIR/" 2>/dev/null && echo "  OK" || echo "  [SKIP] Not found"

echo "[4/4] Downloading Optuna studies..."
scp -P "$SSH_PORT" -i "$SSH_KEY" \
    "root@${POD_IP}:/workspace/optuna_studies.tar.gz" \
    "$DOWNLOAD_DIR/" 2>/dev/null && echo "  OK" || echo "  [SKIP] Not found"

echo ""
echo "Downloaded files:"
ls -lh "$DOWNLOAD_DIR/"*.tar.gz 2>/dev/null || echo "  No files downloaded."

echo ""
echo "=== Installing Results ==="
echo ""
read -p "Extract models to project? ($PROJECT_ROOT) [y/N] " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cd "$PROJECT_ROOT"

    if [ -f "$DOWNLOAD_DIR/trained_models.tar.gz" ]; then
        echo "Extracting models..."
        tar -xzf "$DOWNLOAD_DIR/trained_models.tar.gz"
        echo "[OK] Models extracted."
    fi

    if [ -f "$DOWNLOAD_DIR/prod_configs.tar.gz" ]; then
        echo "Extracting production configs..."
        tar -xzf "$DOWNLOAD_DIR/prod_configs.tar.gz"
        echo "[OK] Configs extracted."
    fi

    if [ -f "$DOWNLOAD_DIR/training_logs.tar.gz" ]; then
        echo "Extracting training logs..."
        tar -xzf "$DOWNLOAD_DIR/training_logs.tar.gz"
        echo "[OK] Logs extracted."
    fi

    echo ""
    echo "Verification:"
    echo "  Run: make test"
    echo "  Run: make serve"
else
    echo ""
    echo "Manual extraction:"
    echo "  cd $PROJECT_ROOT"
    echo "  tar -xzf $DOWNLOAD_DIR/trained_models.tar.gz"
    echo "  tar -xzf $DOWNLOAD_DIR/prod_configs.tar.gz"
fi

echo ""
echo "REMINDER: Stop the RunPod pod to stop billing!"
