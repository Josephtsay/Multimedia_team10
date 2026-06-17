"""filter_polluted_hagrid.py

Filter "hanging idle hand" samples out of the 5 *target* gesture categories
inside hagridv2_512_hand_preprocess_landmark/, using ONLY the
`tips_below_wrist` heuristic.

Why only this one rule (and a separate script from filter_polluted_samples.py):
  filter_polluted_samples.py's own single-rule analysis on the baseline dataset
  showed that of the three candidate heuristics, `tips_below_wrist` is the only
  one that is both selective (~1.7% flag rate) and evenly distributed across
  gesture classes -- `palm_angle` almost never fires (0.02%) and `tip_spread`
  fires on ~80% of ALL samples regardless of gesture (no discriminative power,
  would gut the dataset). So here we apply `tips_below_wrist` alone.

This is a standalone script per request -- filter_polluted_samples.py is left
untouched. The dataset layout here also differs (flat, landmark-only, no
crop_image pairing, and a much larger HaGRIDv2 category set):

    <root>/<category>/<uuid>.npy      # shape (21,2) float32, crop-relative xy

Only the 5 *target* categories are scanned (fist, like, ok, one, palm).
The ~29 other HaGRIDv2 categories are intentionally left alone -- they are all
folded into the N/A class for training regardless of "pollution" (an idle hand
inside an N/A-bound sample is not a labeling problem).

Usage:
    # preview only
    python filter_polluted_hagrid.py --dry-run

    # quarantine flagged samples into <root>/_polluted/<category>/
    python filter_polluted_hagrid.py --quarantine

    # custom root / quarantine folder
    python filter_polluted_hagrid.py --root <path> --quarantine-dir _polluted --quarantine
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

import numpy as np

WRIST = 0
FINGERTIPS = [4, 8, 12, 16, 20]

TARGET_CATEGORIES = ["fist", "like", "ok", "one", "palm"]

BASE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.join(BASE, "hagridv2_512_hand_preprocess_landmark")

log = logging.getLogger("filter_polluted_hagrid")


def load_xy_landmarks(npy_path: Path) -> np.ndarray:
    lm = np.load(npy_path)
    lm = np.asarray(lm, dtype=np.float32)
    if lm.ndim != 2 or lm.shape[0] < 21 or lm.shape[1] < 2:
        raise ValueError(f"unexpected landmark shape {lm.shape}")
    return lm[:21, :2]


def is_hanging_hand(lm: np.ndarray) -> bool:
    """tips_below_wrist: all 5 fingertips sit below the wrist in image-y
    (MediaPipe image coords: y increases downward, so 'below' means larger y)."""
    wrist_y = lm[WRIST, 1]
    tip_ys = lm[FINGERTIPS, 1]
    return bool(np.all(tip_ys > wrist_y))


def quarantine_sample(root: Path, quarantine_dir: str, category: str, npy_path: Path) -> None:
    qdir = root / quarantine_dir / category
    qdir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(npy_path), str(qdir / npy_path.name))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Filter hanging-idle-hand samples from HaGRIDv2 target categories "
                    "(tips_below_wrist rule only)")
    p.add_argument("--root", default=DEFAULT_ROOT,
                   help=f"Dataset root (default: {DEFAULT_ROOT})")
    p.add_argument("--categories", nargs="+", default=TARGET_CATEGORIES,
                   help=f"Category folders to scan (default: the 5 targets {TARGET_CATEGORIES})")
    p.add_argument("--quarantine", action="store_true",
                   help="Move flagged .npy files into <root>/<quarantine-dir>/<category>/")
    p.add_argument("--quarantine-dir", default="_polluted",
                   help="Quarantine subfolder name (default: _polluted)")
    p.add_argument("--dry-run", action="store_true",
                   help="Preview only -- never moves files, even if --quarantine is set")
    p.add_argument("--log-file", default="polluted_hagrid.log",
                   help="Path to write the detailed flag log (default: polluted_hagrid.log)")
    return p.parse_args()


def setup_logging(log_file: str) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def main() -> None:
    args = parse_args()
    setup_logging(args.log_file)

    root = Path(args.root).resolve()
    if not root.is_dir():
        sys.exit(f"Root not found: {root}")

    move_files = args.quarantine and not args.dry_run
    mode = "DRY-RUN (preview only)" if args.dry_run else ("QUARANTINE" if args.quarantine else "LOG-ONLY")

    log.info(f"Root           : {root}")
    log.info(f"Categories     : {args.categories}")
    log.info(f"Rule           : tips_below_wrist (all 5 fingertips below wrist in image-y)")
    log.info(f"Mode           : {mode}")
    log.info(f"Quarantine to  : <root>/{args.quarantine_dir}/<category>/")
    log.info("-" * 72)

    checked: dict[str, int] = {}
    flagged: dict[str, int] = {}

    for category in args.categories:
        cat_dir = root / category
        if not cat_dir.is_dir():
            log.warning(f"[SKIP] category dir not found: {cat_dir}")
            continue

        for npy_path in sorted(cat_dir.glob("*.npy")):
            checked[category] = checked.get(category, 0) + 1
            try:
                lm = load_xy_landmarks(npy_path)
            except Exception as e:
                log.warning(f"[SKIP] {category}/{npy_path.stem}: {e}")
                continue

            if not is_hanging_hand(lm):
                continue

            flagged[category] = flagged.get(category, 0) + 1
            log.info(f"[POLLUTED] {category}/{npy_path.stem}")

            if move_files:
                quarantine_sample(root, args.quarantine_dir, category, npy_path)

    log.info("-" * 72)
    log.info("Summary (checked / flagged / rate):")
    total_checked = total_flagged = 0
    for cat in args.categories:
        c = checked.get(cat, 0)
        f = flagged.get(cat, 0)
        total_checked += c
        total_flagged += f
        log.info(f"  {cat:<8} checked={c:>6}  flagged={f:>5}  ({f / max(c, 1) * 100:5.2f}%)")
    log.info(f"  {'TOTAL':<8} checked={total_checked:>6}  flagged={total_flagged:>5}  "
             f"({total_flagged / max(total_checked, 1) * 100:5.2f}%)")

    if args.dry_run:
        log.info("\n[dry-run] no files were moved. Re-run with --quarantine to move flagged samples.")
    elif args.quarantine:
        log.info(f"\nFlagged samples moved into: <root>/{args.quarantine_dir}/<category>/")
    else:
        log.info("\n[log-only] files left in place. Re-run with --quarantine to move flagged samples.")


if __name__ == "__main__":
    main()
