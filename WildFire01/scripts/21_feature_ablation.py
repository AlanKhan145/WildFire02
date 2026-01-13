"""Run feature ablation on validation tiles."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import argparse

import tensorflow as tf

from wildfire.eval.ablation import run_feature_ablation
from wildfire.utils.io import read_json, read_yaml
from wildfire.utils.logging import get_logger

LOGGER = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--artifacts_dir", default="artifacts")
    parser.add_argument("--thr_prob", type=float, default=None)
    parser.add_argument("--max_tiles", type=int, default=20)
    args = parser.parse_args()

    base_cfg = read_yaml(Path(args.config))
    data_dir = Path(args.data_dir) if args.data_dir else None

    val_pat = base_cfg["data"]["val_pat"]
    val_pat = str(data_dir / val_pat) if data_dir else val_pat

    stats_path = Path(args.artifacts_dir) / "data" / base_cfg["data"]["stats_json"]
    stats = read_json(stats_path)

    model = tf.keras.models.load_model(Path(args.artifacts_dir) / "gnn" / "model_final.keras", compile=False)
    k_patch = read_json(Path(args.artifacts_dir) / "gnn" / "best_cfg.json")["k_patch"]
    thr_prob = args.thr_prob if args.thr_prob is not None else base_cfg["train"]["thr_prob"]

    out_csv = Path(args.artifacts_dir) / "reports" / "feature_ablation.csv"
    plot_path = Path(args.artifacts_dir) / "reports" / "plots" / "feature_ablation.png"

    run_feature_ablation(
        val_pat,
        model,
        int(k_patch),
        thr_prob,
        args.max_tiles,
        base_cfg["train"]["batch_patches"],
        stats,
        out_csv,
        plot_path,
    )
    LOGGER.info("Saved feature ablation to %s", out_csv)


if __name__ == "__main__":
    main()
