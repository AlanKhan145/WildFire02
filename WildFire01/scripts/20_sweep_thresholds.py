"""Sweep thresholds for model selection."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import argparse

import numpy as np
import tensorflow as tf

from wildfire.eval.threshold_sweep import sweep_thresholds_stream
from wildfire.utils.io import read_json, read_yaml
from wildfire.utils.logging import get_logger

LOGGER = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--artifacts_dir", default="artifacts")
    parser.add_argument("--thr_min", type=float, default=0.1)
    parser.add_argument("--thr_max", type=float, default=0.9)
    parser.add_argument("--thr_step", type=float, default=0.05)
    args = parser.parse_args()

    base_cfg = read_yaml(Path(args.config))
    data_dir = Path(args.data_dir) if args.data_dir else None

    train_pat = base_cfg["data"]["train_pat"]
    val_pat = base_cfg["data"]["val_pat"]
    test_pat = base_cfg["data"]["test_pat"]
    train_pat = str(data_dir / train_pat) if data_dir else train_pat
    val_pat = str(data_dir / val_pat) if data_dir else val_pat
    test_pat = str(data_dir / test_pat) if data_dir else test_pat

    stats_path = Path(args.artifacts_dir) / "data" / base_cfg["data"]["stats_json"]
    stats = read_json(stats_path)

    model = tf.keras.models.load_model(Path(args.artifacts_dir) / "gnn" / "model_final.keras", compile=False)
    k_patch = read_json(Path(args.artifacts_dir) / "gnn" / "best_cfg.json")["k_patch"]

    thresholds = np.arange(args.thr_min, args.thr_max + 1e-6, args.thr_step).tolist()
    out_csv = Path(args.artifacts_dir) / "reports" / "threshold_summary.csv"
    plot_dir = Path(args.artifacts_dir) / "reports" / "plots"

    best_thr, _ = sweep_thresholds_stream(
        train_pat,
        val_pat,
        test_pat,
        model,
        int(k_patch),
        thresholds,
        base_cfg["train"]["max_tiles_eval"],
        base_cfg["train"]["batch_patches"],
        stats,
        out_csv,
        plot_dir,
    )
    LOGGER.info("Best threshold: %s", best_thr)


if __name__ == "__main__":
    main()
