"""Feature ablation evaluation."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from wildfire.data.constants import INPUT_FEATURES
from wildfire.data.datasets import tile_dataset
from wildfire.data.patch_sampling import valid_core_mask
from wildfire.eval.streaming_metrics import infer_core_logit_map_valid


def zero_channel_tile(x_tile: tf.Tensor, idx: int) -> tf.Tensor:
    """Zero out channel idx on tile."""
    mask = tf.one_hot(idx, tf.shape(x_tile)[-1], dtype=x_tile.dtype)
    mask = 1.0 - mask
    return x_tile * mask


def _eval_channel(
    file_pattern: str,
    model: tf.keras.Model,
    k: int,
    thr_prob: float,
    max_tiles: int,
    batch_patches: int,
    stats: Dict[str, Dict[str, float]],
    x_transform: Callable[[tf.Tensor], tf.Tensor],
) -> Dict[str, float]:
    conf = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    dataset = tile_dataset(file_pattern, stats, shuffle=False, repeat=False)
    count = 0
    for sample in dataset:
        x_tile = x_transform(sample["x"])
        prev_raw = sample["prev_raw"]
        y_raw = sample["y_raw"]
        ndvi_raw = sample["ndvi_raw"]
        valid = valid_core_mask(prev_raw, y_raw, ndvi_raw, k).numpy()
        logits = infer_core_logit_map_valid(x_tile, model, k, batch_patches).numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))
        r = k // 2
        y_core = y_raw.numpy()[r:-r, r:-r]
        y_valid = (y_core[valid] > 0).astype(np.int32)
        pred_valid = (probs[valid] >= thr_prob).astype(np.int32)
        conf["tp"] += int(np.sum((y_valid == 1) & (pred_valid == 1)))
        conf["fp"] += int(np.sum((y_valid == 0) & (pred_valid == 1)))
        conf["fn"] += int(np.sum((y_valid == 1) & (pred_valid == 0)))
        conf["tn"] += int(np.sum((y_valid == 0) & (pred_valid == 0)))
        count += 1
        if max_tiles is not None and count >= max_tiles:
            break

    tp = conf["tp"]
    fp = conf["fp"]
    fn = conf["fn"]
    denom_iou = tp + fp + fn
    denom_f1 = 2 * tp + fp + fn
    iou = tp / denom_iou if denom_iou > 0 else 0.0
    f1 = 2 * tp / denom_f1 if denom_f1 > 0 else 0.0
    return {"iou": iou, "f1": f1}


def run_feature_ablation(
    val_pat: str,
    model: tf.keras.Model,
    k: int,
    thr_prob: float,
    max_tiles: int,
    batch_patches: int,
    stats: Dict[str, Dict[str, float]],
    out_csv: Path,
    plot_path: Path,
) -> pd.DataFrame:
    """Run feature ablation by zeroing each channel."""
    rows = []
    baseline = _eval_channel(
        val_pat,
        model,
        k,
        thr_prob,
        max_tiles,
        batch_patches,
        stats,
        x_transform=lambda x: x,
    )

    for idx, name in enumerate(INPUT_FEATURES):
        metrics = _eval_channel(
            val_pat,
            model,
            k,
            thr_prob,
            max_tiles,
            batch_patches,
            stats,
            x_transform=partial(zero_channel_tile, idx=idx),
        )
        rows.append({
            "feature": name,
            "iou": metrics["iou"],
            "f1": metrics["f1"],
            "iou_drop": baseline["iou"] - metrics["iou"],
            "f1_drop": baseline["f1"] - metrics["f1"],
        })

    df = pd.DataFrame(rows).sort_values("iou_drop", ascending=False)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.bar(df["feature"], df["iou_drop"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("IoU Drop")
    plt.title("Feature Ablation (IoU Drop)")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    return df
