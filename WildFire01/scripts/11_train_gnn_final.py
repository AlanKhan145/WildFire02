"""Train final PatchGNN model."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import argparse

import tensorflow as tf

from wildfire.data.constants import C_IN
from wildfire.data.datasets import tile_dataset
from wildfire.data.patch_sampling import make_train_patch_dataset
from wildfire.eval.streaming_metrics import full_eval_metrics_stream
from wildfire.models.patch_gnn import build_patch_gnn
from wildfire.train.callbacks import StreamSplitMetrics
from wildfire.train.trainer import compile_patch_gnn
from wildfire.utils.io import read_json, read_yaml, write_json
from wildfire.utils.logging import get_logger

LOGGER = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--final_config", default="configs/gnn_final.yaml")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--artifacts_dir", default="artifacts")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base_cfg = read_yaml(Path(args.config))
    final_cfg = read_yaml(Path(args.final_config))
    data_dir = Path(args.data_dir) if args.data_dir else None

    train_pat = base_cfg["data"]["train_pat"]
    val_pat = base_cfg["data"]["val_pat"]
    train_pat = str(data_dir / train_pat) if data_dir else train_pat
    val_pat = str(data_dir / val_pat) if data_dir else val_pat

    stats_path = Path(args.artifacts_dir) / "data" / base_cfg["data"]["stats_json"]
    stats = read_json(stats_path)

    best_cfg_path = Path(args.artifacts_dir) / "gnn" / "best_cfg.json"
    best_cfg = read_json(best_cfg_path)
    best_cfg["c_in"] = C_IN

    model = build_patch_gnn(
        k=int(best_cfg["k_patch"]),
        c_in=best_cfg["c_in"],
        d_hidden=int(best_cfg["d_hidden"]),
        n_layers=int(best_cfg["n_layers"]),
        dropout=float(best_cfg["dropout"]),
    )
    compile_patch_gnn(model, lr=float(best_cfg["lr"]), pos_weight=float(best_cfg["pos_weight"]), use_auc=False)

    train_tiles = tile_dataset(train_pat, stats, shuffle=True, repeat=True)
    train_patches = make_train_patch_dataset(
        train_tiles,
        int(best_cfg["k_patch"]),
        base_cfg["train"]["n_pos"],
        base_cfg["train"]["n_neg"],
        base_cfg["train"]["batch_size"],
    )

    stream_cb = StreamSplitMetrics(
        train_pat,
        val_pat,
        int(best_cfg["k_patch"]),
        final_cfg["train"].get("thr_prob", base_cfg["train"]["thr_prob"]),
        base_cfg["train"]["max_tiles_eval"],
        base_cfg["train"]["batch_patches"],
        stats,
    )

    history = model.fit(
        train_patches,
        epochs=final_cfg["train"]["epochs"],
        steps_per_epoch=final_cfg["train"]["steps_per_epoch"],
        callbacks=[stream_cb],
        verbose=2,
    )

    out_dir = Path(args.artifacts_dir) / "gnn"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / "model_final.keras")

    final_metrics = full_eval_metrics_stream(
        val_pat,
        model,
        int(best_cfg["k_patch"]),
        final_cfg["train"].get("thr_prob", base_cfg["train"]["thr_prob"]),
        base_cfg["train"]["max_tiles_eval"],
        base_cfg["train"]["batch_patches"],
        stats,
    )
    write_json(out_dir / "final_row.json", final_metrics)
    LOGGER.info("Saved final model and metrics to %s", out_dir)


if __name__ == "__main__":
    main()
