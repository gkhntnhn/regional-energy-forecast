#!/usr/bin/env bash
# pack_data.sh — Package local data files for RunPod transfer
# Run from project root: bash scripts/runpod/pack_data.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

OUTPUT="runpod_data.tar.gz"

echo "=== RunPod Data Packer ==="
echo "Project root: $PROJECT_ROOT"
echo ""

# Verify required files exist
REQUIRED_FILES=(
    "data/raw/Consumption_Input_Format.xlsx"
    "data/static/turkish_holidays.parquet"
    "data/external/weather_cache.sqlite"
)

MISSING=0
for f in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$f" ]; then
        echo "[ERROR] Missing: $f"
        MISSING=1
    else
        echo "[OK]    $f ($(du -h "$f" | cut -f1))"
    fi
done

# Check EPIAS cache (at least some parquets)
EPIAS_COUNT=$(find data/external/epias -name "*.parquet" 2>/dev/null | wc -l)
if [ "$EPIAS_COUNT" -eq 0 ]; then
    echo "[ERROR] No EPIAS parquet files found in data/external/epias/"
    MISSING=1
else
    echo "[OK]    data/external/epias/ ($EPIAS_COUNT parquet files)"
fi

# Check profile coefficients
PROFILE_COUNT=$(find data/external/profile -name "*.parquet" 2>/dev/null | wc -l)
if [ "$PROFILE_COUNT" -eq 0 ]; then
    echo "[ERROR] No profile parquet files found in data/external/profile/"
    MISSING=1
else
    echo "[OK]    data/external/profile/ ($PROFILE_COUNT parquet files)"
fi

if [ "$MISSING" -eq 1 ]; then
    echo ""
    echo "[ABORT] Missing required files. Fix above errors first."
    echo ""
    echo "Hints:"
    echo "  - Weather cache missing? Run: uv run python scripts/prepare_dataset.py --skip-epias -v"
    echo "  - EPIAS cache missing?   Run: uv run python scripts/backfill_epias.py --start-year 2020"
    echo "  - Holidays missing?      Run: uv run python scripts/generate_holidays.py"
    exit 1
fi

echo ""
echo "Packaging data files..."

tar -czf "$OUTPUT" \
    data/raw/Consumption_Input_Format.xlsx \
    data/external/epias/ \
    data/external/profile/ \
    data/external/weather_cache.sqlite \
    data/static/turkish_holidays.parquet

SIZE=$(du -h "$OUTPUT" | cut -f1)
echo ""
echo "=== Done ==="
echo "Output: $OUTPUT ($SIZE)"
echo ""
echo "Next step: Transfer to RunPod pod via SCP:"
echo "  scp -P <PORT> -i ~/.ssh/id_ed25519 $OUTPUT root@<POD_IP>:/workspace/"
