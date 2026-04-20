"""Promote R12 multi-seed checkpoints to final_models/ for serving.

Copies ``models/tsmixerx_multi_seed/seed_*/`` contents to
``final_models/tsmixerx/seed_*/`` and writes a top-level ``metadata.json``
summary for reporting / observability.

Post-promotion, the serving stack auto-detects the multi-seed structure
(see ``EnsembleForecaster.load_models``) and loads all 5 seeds on startup.

Usage::

    uv run python scripts/promote_multi_seed.py
    uv run python scripts/promote_multi_seed.py --dry-run
    uv run python scripts/promote_multi_seed.py \
        --source models/tsmixerx_multi_seed --target final_models/tsmixerx
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Ensure local src/ is importable
sys.path.insert(0, str(Path("src").resolve()))

from loguru import logger

DEFAULT_SOURCE = Path("models/tsmixerx_multi_seed")
DEFAULT_TARGET = Path("final_models/tsmixerx")

# R12 FAZ 6 reference metrics (from debug/r12_research/06_multi_seed_results.json)
R12_REFERENCE = {
    "seeds": [42, 123, 456, 789, 2026],
    "per_seed_val_mape": [1.7479, 1.7695, 1.7058, 1.7427, 1.8573],
    "per_seed_test_mape": [1.7670, 1.7695, 1.8005, 1.7895, 1.8780],
    "naive_avg_val_mape": 1.7646,
    "naive_avg_test_mape": 1.8009,
    "ensemble_val_mape": 1.6140,
    "ensemble_test_mape": 1.6494,
    "std_val_mape": 0.0506,
    "std_test_mape": 0.0405,
    # From debug/r12_research/07_preanalysis_data.json bootstrap results
    "bootstrap_jensen_gain_ci95": [0.1467, 0.1575],
    "bootstrap_p_value_gt_zero": 0.0,
    "bootstrap_n_samples": 10000,
    "r12_commit": "0e9fcb4",
}


def _hash_file(path: Path) -> str:
    """SHA256 of file contents for integrity verification."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _copy_seed_dir(src_dir: Path, dst_dir: Path, *, dry_run: bool) -> int:
    """Copy contents of one seed_* directory, return number of files copied."""
    if dry_run:
        files = list(src_dir.iterdir())
        logger.info("[dry-run] Would copy {} files: {} -> {}", len(files), src_dir, dst_dir)
        return len(files)

    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for item in src_dir.iterdir():
        if item.is_file():
            # follow_symlinks=False blocks attacker-planted symlinks in src
            shutil.copy2(item, dst_dir / item.name, follow_symlinks=False)
            count += 1
    return count


def _augment_seed_metadata(seed_dst: Path, *, dry_run: bool) -> bool:
    """Upgrade legacy `ckpt_hashes`-only metadata to include `.pkl` hashes.

    Security P0-1: pickle files loaded via ``NeuralForecast.load()`` are RCE
    vectors. Pre-R12 checkpoints only hashed ``*.ckpt``. This post-copy
    augmentation computes SHA256 for all pkl + ckpt on destination and writes
    a new ``artifact_hashes`` field, making ``TSMixerxForecaster.from_checkpoint``
    verify the full artifact set.

    Returns True if metadata was updated, False if already in new format or
    dry-run mode.
    """
    meta_path = seed_dst / "metadata.json"
    if not meta_path.exists():
        logger.warning("No metadata.json in {}, skipping augmentation", seed_dst)
        return False

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    # Compute fresh artifact hashes over destination files
    new_hashes: dict[str, str] = {}
    for ext in ("*.ckpt", "*.pkl"):
        for artifact in seed_dst.glob(ext):
            new_hashes[artifact.name] = _hash_file(artifact)

    if not new_hashes:
        logger.warning("No artifact files in {}, skipping", seed_dst)
        return False

    # Preserve any existing ckpt_hashes for backward-compat readers
    existing_ckpt_hashes = metadata.get("ckpt_hashes", {})
    metadata["artifact_hashes"] = new_hashes
    metadata["ckpt_hashes"] = {
        k: v for k, v in new_hashes.items() if k.endswith(".ckpt")
    } or existing_ckpt_hashes

    if dry_run:
        logger.info(
            "[dry-run] Would augment {}: artifact_hashes with {} files",
            meta_path,
            len(new_hashes),
        )
        return False

    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info(
        "Augmented metadata: {} (artifact_hashes +{} files)",
        meta_path,
        len(new_hashes),
    )
    return True


def _write_top_metadata(target: Path, seed_names: list[str], *, dry_run: bool) -> None:
    """Write top-level metadata.json for observability / tooling."""
    metadata = {
        "ensemble_type": "multi_seed_jensen",
        "promoted_at": datetime.now().isoformat(timespec="seconds"),
        "source_dir": str(DEFAULT_SOURCE),
        "seed_dirs": seed_names,
        "reference_metrics": R12_REFERENCE,
    }
    path = target / "metadata.json"
    if dry_run:
        logger.info("[dry-run] Would write: {}", path)
        return
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Wrote top-level metadata: {}", path)


def _verify_promoted(src: Path, dst: Path) -> tuple[int, list[str]]:
    """Compare file hashes between source and destination.

    metadata.json files are intentionally rewritten during ``_augment_seed_metadata``
    (P0-1 pkl hash injection), so they are verified semantically (JSON-parseable
    + has ``artifact_hashes``) rather than byte-wise.

    Returns:
        (n_verified, mismatches).
    """
    mismatches: list[str] = []
    n_verified = 0
    for src_file in src.rglob("*"):
        if not src_file.is_file():
            continue
        relative = src_file.relative_to(src)
        dst_file = dst / relative
        if not dst_file.exists():
            mismatches.append(f"missing: {relative}")
            continue

        # metadata.json is augmented on destination — semantic check only.
        if src_file.name == "metadata.json":
            try:
                dst_meta = json.loads(dst_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                mismatches.append(f"metadata.json corrupt: {relative} ({e})")
                continue
            if "artifact_hashes" not in dst_meta:
                mismatches.append(
                    f"metadata.json missing artifact_hashes: {relative} (P0-1 migration failed)"
                )
                continue
            n_verified += 1
            continue

        if _hash_file(src_file) != _hash_file(dst_file):
            mismatches.append(f"hash mismatch: {relative}")
            continue
        n_verified += 1
    return n_verified, mismatches


def main() -> int:
    parser = argparse.ArgumentParser(description="Promote R12 multi-seed checkpoints")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--dry-run", action="store_true", help="Show actions without copying")
    parser.add_argument(
        "--skip-verify", action="store_true", help="Skip post-copy hash verification"
    )
    parser.add_argument(
        "--allow-unsafe-target",
        action="store_true",
        help="Allow --target outside final_models/ (non-standard deployments)",
    )
    args = parser.parse_args()

    source: Path = args.source
    target: Path = args.target

    logger.info("=== R12 Multi-Seed Promotion ===")
    logger.info("Source: {}", source)
    logger.info("Target: {}", target)
    logger.info("Dry-run: {}", args.dry_run)

    # Path safety: refuse writes outside the project-local final_models/ tree
    # unless operator explicitly opts out (operator-footgun protection, audit P1-3).
    allowed_root = Path("final_models").resolve()
    try:
        target_resolved = target.resolve()
        target_resolved.relative_to(allowed_root)
    except (ValueError, OSError):
        if not args.allow_unsafe_target:
            logger.error(
                "Refusing to write outside final_models/: {} (use --allow-unsafe-target "
                "to override for non-standard deployments)",
                target_resolved,
            )
            return 3

    if not source.exists():
        logger.error("Source directory does not exist: {}", source)
        return 1

    seed_dirs = sorted(source.glob("seed_*"))
    if not seed_dirs:
        logger.error("No seed_*/ subdirectories found in {}", source)
        return 1

    logger.info("Found {} seed directories: {}", len(seed_dirs), [d.name for d in seed_dirs])

    if target.exists() and any(target.iterdir()) and not args.dry_run:
        logger.warning(
            "Target {} is not empty — existing seed files will be overwritten.", target
        )

    # Copy each seed dir + augment metadata with pkl hashes (P0-1 migration)
    total_files = 0
    seed_names: list[str] = []
    for src_seed in seed_dirs:
        dst_seed = target / src_seed.name
        n = _copy_seed_dir(src_seed, dst_seed, dry_run=args.dry_run)
        total_files += n
        seed_names.append(src_seed.name)
        logger.info("Promoted {} ({} files)", src_seed.name, n)
        _augment_seed_metadata(dst_seed, dry_run=args.dry_run)

    # Write top-level metadata
    _write_top_metadata(target, seed_names, dry_run=args.dry_run)

    # Verify
    if args.dry_run or args.skip_verify:
        logger.info("Verification skipped")
    else:
        logger.info("Verifying SHA256 hashes...")
        n_verified, mismatches = _verify_promoted(source, target)
        if mismatches:
            logger.error("Verification failed: {} mismatches", len(mismatches))
            for m in mismatches:
                logger.error("  {}", m)
            return 2
        logger.info("Verification passed: {} files hash-matched", n_verified)

    logger.info(
        "Promotion complete: {} seeds, {} files total -> {}",
        len(seed_dirs),
        total_files,
        target,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
