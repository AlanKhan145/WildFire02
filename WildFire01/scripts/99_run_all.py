"""Run full pipeline."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import subprocess


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--artifacts_dir", default="artifacts")
    args = parser.parse_args()

    base_cmd = ["python"]
    data_arg = ["--data_dir", args.data_dir] if args.data_dir else []

    run(base_cmd + ["scripts/00_download_data.py", "--config", args.config, "--artifacts_dir", args.artifacts_dir])
    run(base_cmd + ["scripts/01_count_records.py", "--config", args.config] + data_arg)
    run(base_cmd + ["scripts/02_compute_train_stats.py", "--config", args.config, "--artifacts_dir", args.artifacts_dir] + data_arg)
    run(base_cmd + ["scripts/10_hpo_gnn.py", "--config", args.config, "--artifacts_dir", args.artifacts_dir] + data_arg)
    run(base_cmd + ["scripts/11_train_gnn_final.py", "--config", args.config, "--artifacts_dir", args.artifacts_dir] + data_arg)
    run(base_cmd + ["scripts/20_sweep_thresholds.py", "--config", args.config, "--artifacts_dir", args.artifacts_dir] + data_arg)
    run(base_cmd + ["scripts/21_feature_ablation.py", "--config", args.config, "--artifacts_dir", args.artifacts_dir] + data_arg)
    run(base_cmd + ["scripts/30_vis_largest_fire_tile.py", "--config", args.config, "--artifacts_dir", args.artifacts_dir] + data_arg)
    run(base_cmd + ["scripts/31_vis_best_worst_tiles.py", "--config", args.config, "--artifacts_dir", args.artifacts_dir] + data_arg)
    run(base_cmd + ["scripts/32_vis_multi_resolution.py", "--config", args.config, "--artifacts_dir", args.artifacts_dir] + data_arg)


if __name__ == "__main__":
    main()
