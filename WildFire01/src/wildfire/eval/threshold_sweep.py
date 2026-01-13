"""Threshold sweep utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from wildfire.data.datasets import tile_dataset
from wildfire.data.patch_sampling import valid_core_mask
from wildfire.eval.streaming_metrics import infer_core_logit_map_valid


def _confusion_from_stream(
    file_pattern: str,
    model: tf.keras.Model,
    k: int,
    thr_prob: float,
    max_tiles: int,
    batch_patches: int,
    stats: Dict[str, Dict[str, float]],
) -> Dict[str, int]:
    conf = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    dataset = tile_dataset(file_pattern, stats, shuffle=False, repeat=False)
    count = 0
    for sample in dataset:
        x_tile = sample["x"]
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
    return conf


def _metrics_from_conf(conf: Dict[str, int]) -> Dict[str, float]:
    tp = conf["tp"]
    fp = conf["fp"]
    fn = conf["fn"]
    tn = conf["tn"]
    denom_iou = tp + fp + fn
    denom_f1 = 2 * tp + fp + fn
    iou = tp / denom_iou if denom_iou > 0 else 0.0
    f1 = 2 * tp / denom_f1 if denom_f1 > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    total = tp + tn + fp + fn
    p0 = acc
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (total * total) if total > 0 else 0.0
    kappa = (p0 - pe) / (1 - pe) if (1 - pe) > 0 else 0.0
    return {
        "iou": iou,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "acc": acc,
        "kappa": kappa,
    }


def sweep_thresholds_stream(
    train_pat: str,
    val_pat: str,
    test_pat: str,
    model: tf.keras.Model,
    k: int,
    thresholds: List[float],
    max_tiles: int,
    batch_patches: int,
    stats: Dict[str, Dict[str, float]],
    out_csv: Path,
    plot_dir: Path,
) -> Tuple[float, pd.DataFrame]:
    """Sweep thresholds and pick best by val IoU (tie-break F1)."""
    rows = []
    for thr in thresholds:
        train_conf = _confusion_from_stream(train_pat, model, k, thr, max_tiles, batch_patches, stats)
        val_conf = _confusion_from_stream(val_pat, model, k, thr, max_tiles, batch_patches, stats)
        train_metrics = _metrics_from_conf(train_conf)
        val_metrics = _metrics_from_conf(val_conf)
        rows.append({
            "thr": thr,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("thr")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    best = df.sort_values(["val_iou", "val_f1"], ascending=False).iloc[0]
    best_thr = float(best["thr"])

    plot_dir.mkdir(parents=True, exist_ok=True)
    for metric in ["acc", "iou", "f1", "precision", "recall", "kappa"]:
        plt.figure(figsize=(6, 4))
        plt.plot(df["thr"], df[f"train_{metric}"], label="train")
        plt.plot(df["thr"], df[f"val_{metric}"], label="val")
        plt.xlabel("Threshold")
        plt.ylabel(metric.upper())
        plt.title(f"{metric.upper()} vs Threshold")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"threshold_{metric}.png")
        plt.close()

    test_conf = _confusion_from_stream(test_pat, model, k, best_thr, max_tiles, batch_patches, stats)
    test_metrics = _metrics_from_conf(test_conf)
    test_row = {"thr": best_thr, **{f"test_{k}": v for k, v in test_metrics.items()}}
    df_summary = pd.DataFrame([test_row])
    df_summary.to_csv(out_csv.parent / "threshold_test_summary.csv", index=False)

    return best_thr, df
