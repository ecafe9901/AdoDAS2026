#!/usr/bin/env python3
"""Calibrate LLM-extracted DASS scores against training labels.

Fits per-item linear regression: true_score = slope * llm_score + intercept.
Saves calibrated features and coefficients for test-set application.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

log = logging.getLogger("calibrate_llm")

DASS_ITEMS = [f"d{i:02d}" for i in range(1, 22)]


def calibrate(llm_dir: Path, manifest_path: Path, output_dir: Path,
              split: str = "train") -> dict:
    """Calibrate LLM features against true labels. Returns coefficients."""
    df = pd.read_csv(manifest_path)
    pid_to_labels = {}
    for pid, group in df.groupby("anon_pid"):
        pid_to_labels[str(pid)] = group[DASS_ITEMS].iloc[0].values.astype(float)

    # Load LLM features + match to labels
    llm_all, true_all = [], []
    for f in (llm_dir / split).glob("*.npy"):
        pid = f.stem.split("_")[4]
        if pid not in pid_to_labels:
            continue
        data = np.load(f)
        llm_all.append(data[:21])
        true_all.append(pid_to_labels[pid])

    llm_all = np.array(llm_all)
    true_all = np.array(true_all)
    log.info(f"Matched {len(llm_all)} participants for calibration")

    # Per-item linear regression
    coefs = {}
    calibrated = np.zeros_like(llm_all)
    for i in range(21):
        X = llm_all[:, i].reshape(-1, 1)
        y = true_all[:, i]
        lr = LinearRegression().fit(X, y)
        coefs[DASS_ITEMS[i]] = {"slope": float(lr.coef_[0]),
                                  "intercept": float(lr.intercept_)}
        calibrated[:, i] = lr.predict(X)
        # Clamp to valid range
        calibrated[:, i] = np.clip(calibrated[:, i], 0, 3)

    # Save calibrated features
    out_dir = output_dir / split
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = 0
    for f in sorted((llm_dir / split).glob("*.npy")):
        pid = f.stem.split("_")[4]
        if pid not in pid_to_labels:
            continue
        data = np.load(f)
        data[:21] = calibrated[idx]
        np.save(out_dir / f.name, data)
        idx += 1
    log.info(f"Saved {idx} calibrated features to {out_dir}")

    # Save coefficients for test-set calibration
    np.savez(output_dir / "calibration_coefs.npz", **{
        f"{item}_slope": coefs[item]["slope"] for item in DASS_ITEMS
    } | {
        f"{item}_intercept": coefs[item]["intercept"] for item in DASS_ITEMS
    })

    # Per-item stats
    before_pcc = np.corrcoef(llm_all.flatten(), true_all.flatten())[0, 1]
    after_pcc = np.corrcoef(calibrated.flatten(), true_all.flatten())[0, 1]
    log.info(f"PCC: {before_pcc:.3f} → {after_pcc:.3f}")
    log.info("Per-item calibration:")
    for item in DASS_ITEMS[:8]:
        c = coefs[item]
        log.info(f"  {item}: slope={c['slope']:.3f}, intercept={c['intercept']:.3f}")
    log.info("  ...")
    return coefs


def apply_calibration(llm_dir: Path, coefs_path: Path, output_dir: Path,
                      split: str):
    """Apply saved calibration coefficients to new LLM features (e.g., test set)."""
    coef_data = np.load(coefs_path)
    slopes = np.array([coef_data[f"{item}_slope"] for item in DASS_ITEMS])
    intercepts = np.array([coef_data[f"{item}_intercept"] for item in DASS_ITEMS])

    in_dir = llm_dir / split
    out_dir = output_dir / split
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for f in in_dir.glob("*.npy"):
        data = np.load(f)
        dass = data[:21]
        calibrated = np.clip(slopes * dass + intercepts, 0, 3)
        data[:21] = calibrated
        np.save(out_dir / f.name, data)
        count += 1
    log.info(f"Applied calibration to {count} {split} features")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_dir", required=True,
                        help="Directory with raw LLM .npy features")
    parser.add_argument("--manifest", required=True,
                        help="Training manifest CSV with labels")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for calibrated features")
    parser.add_argument("--apply_to", default=None,
                        help="Apply existing calibration to another split (test/val)")
    parser.add_argument("--coefs", default=None,
                        help="Path to calibration_coefs.npz for --apply_to")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    if args.apply_to:
        apply_calibration(Path(args.llm_dir), Path(args.coefs),
                          Path(args.output_dir), args.apply_to)
    else:
        calibrate(Path(args.llm_dir), Path(args.manifest), Path(args.output_dir))


if __name__ == "__main__":
    main()
