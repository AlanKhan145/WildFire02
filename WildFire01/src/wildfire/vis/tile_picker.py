"""Tile selection utilities."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import tensorflow as tf

from wildfire.data.datasets import tile_dataset
from wildfire.data.patch_sampling import valid_core_mask
from wildfire.eval.streaming_metrics import infer_core_logit_map_valid


def pick_largest_fire_tile(
    file_pattern: str,
    stats: Dict[str, Dict[str, float]],
) -> Dict[str, tf.Tensor]:
    """Pick tile with largest fire area."""
    dataset = tile_dataset(file_pattern, stats, shuffle=False, repeat=False)
    best_sample = None
    best_count = -1
    for sample in dataset:
        y_raw = sample["y_raw"].numpy()
        count = int(np.sum(y_raw > 0))
        if count > best_count:
            best_count = count
            best_sample = sample
    return best_sample


def _iou_core(
    sample: Dict[str, tf.Tensor],
    model: tf.keras.Model,
    k: int,
    thr_prob: float,
    batch_patches: int,
) -> float:
    x_tile = sample["x"]
    prev_raw = sample["prev_raw"]
    y_raw = sample["y_raw"]
    ndvi_raw = sample["ndvi_raw"]
    valid = valid_core_mask(prev_raw, y_raw, ndvi_raw, k).numpy()
    logits = infer_core_logit_map_valid(x_tile, model, k, batch_patches).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    r = k // 2
    y_core = (y_raw.numpy()[r:-r, r:-r] > 0).astype(np.int32)
    pred = (probs >= thr_prob).astype(np.int32)
    y_valid = y_core[valid]
    pred_valid = pred[valid]
    tp = np.sum((y_valid == 1) & (pred_valid == 1))
    fp = np.sum((y_valid == 0) & (pred_valid == 1))
    fn = np.sum((y_valid == 1) & (pred_valid == 0))
    denom = tp + fp + fn
    return float(tp / denom) if denom > 0 else 0.0


def pick_best_worst_tiles(
    file_pattern: str,
    stats: Dict[str, Dict[str, float]],
    model: tf.keras.Model,
    k: int,
    thr_prob: float,
    max_tiles: int,
    batch_patches: int,
) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
    """Pick best and worst tiles by IoU."""
    dataset = tile_dataset(file_pattern, stats, shuffle=False, repeat=False)
    best = None
    worst = None
    best_iou = -1.0
    worst_iou = 2.0
    count = 0
    for sample in dataset:
        iou = _iou_core(sample, model, k, thr_prob, batch_patches)
        if iou > best_iou:
            best_iou = iou
            best = sample
        if iou < worst_iou:
            worst_iou = iou
            worst = sample
        count += 1
        if max_tiles is not None and count >= max_tiles:
            break
    return best, worst
