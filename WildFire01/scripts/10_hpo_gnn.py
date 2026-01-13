"""Run HPO for PatchGNN."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import itertools
from typing import Dict, List

import pandas as pd

from wildfire.data.constants import C_IN
from wildfire.train.hpo import successive_halving_hpo
from wildfire.utils.io import read_json, read_yaml, write_json
from wildfire.utils.logging import get_logger

LOGGER = get_logger(__name__)


def _grid_from_search(search_space: Dict[str, List[float]]) -> List[Dict[str, float]]:
    keys = list(search_space.keys())
    values = [search_space[k] for k in keys]
    configs = []
    for combo in itertools.product(*values):
        cfg = {k: combo[i] for i, k in enumerate(keys)}
        configs.append(cfg)
    return configs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--hpo_config", default="configs/gnn_hpo.yaml")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--artifacts_dir", default="artifacts")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base_cfg = read_yaml(Path(args.config))
    hpo_cfg = read_yaml(Path(args.hpo_config))
    data_dir = Path(args.data_dir) if args.data_dir else None

    train_pat = base_cfg["data"]["train_pat"]
    val_pat = base_cfg["data"]["val_pat"]
    train_pat = str(data_dir / train_pat) if data_dir else train_pat
    val_pat = str(data_dir / val_pat) if data_dir else val_pat

    stats_path = Path(args.artifacts_dir) / "data" / base_cfg["data"]["stats_json"]
    stats = read_json(stats_path)

    search_space = hpo_cfg["search_space"]
    configs = _grid_from_search(search_space)
    for cfg in configs:
        cfg.update({
            "c_in": C_IN,
            "thr_prob": hpo_cfg["train"].get("thr_prob", base_cfg["train"]["thr_prob"]),
            "steps_per_epoch": hpo_cfg["train"].get("steps_per_epoch", base_cfg["train"]["steps_per_epoch"]),
        })

    best_cfg, df_hpo = successive_halving_hpo(
        configs,
        train_pat,
        val_pat,
        stats,
        batch_size=base_cfg["train"]["batch_size"],
        n_pos=base_cfg["train"]["n_pos"],
        n_neg=base_cfg["train"]["n_neg"],
        max_tiles_eval=base_cfg["train"]["max_tiles_eval"],
        batch_patches=base_cfg["train"]["batch_patches"],
        rung_epochs=base_cfg["hpo"]["rung_epochs"],
        monitor=base_cfg["hpo"]["monitor"],
    )

    out_dir = Path(args.artifacts_dir) / "gnn"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_hpo.to_csv(out_dir / "df_hpo.csv", index=False)
    write_json(out_dir / "best_cfg.json", best_cfg)
    LOGGER.info("Saved HPO results to %s", out_dir)


if __name__ == "__main__":
    main()
