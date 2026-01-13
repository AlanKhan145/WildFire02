"""Streaming evaluation utilities."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import tensorflow as tf

from wildfire.data.datasets import tile_dataset
from wildfire.data.patch_sampling import valid_core_mask


def infer_core_logit_map_valid(
    x_tile: tf.Tensor,
    model: tf.keras.Model,
    k: int,
    batch_patches: int,
) -> tf.Tensor:
    """Infer logits for core positions using VALID padding."""
    patches = tf.image.extract_patches(
        x_tile[tf.newaxis, ...],
        sizes=[1, k, k, 1],
        strides=[1, 1, 1, 1],
        rates=[1, 1, 1, 1],
        padding="VALID",
    )
    h = tf.shape(patches)[1]
    w = tf.shape(patches)[2]
    c = tf.shape(x_tile)[-1]
    patches = tf.reshape(patches, (-1, k, k, c))
    logits = model.predict(patches, batch_size=batch_patches, verbose=0)
    logits = tf.reshape(logits, (h, w))
    return logits


def infer_full_map_logit_same(
    x_tile: tf.Tensor,
    model: tf.keras.Model,
    k: int,
    batch_patches: int,
) -> tf.Tensor:
    """Infer logits for full map using SAME padding."""
    patches = tf.image.extract_patches(
        x_tile[tf.newaxis, ...],
        sizes=[1, k, k, 1],
        strides=[1, 1, 1, 1],
        rates=[1, 1, 1, 1],
        padding="SAME",
    )
    h = tf.shape(patches)[1]
    w = tf.shape(patches)[2]
    c = tf.shape(x_tile)[-1]
    patches = tf.reshape(patches, (-1, k, k, c))
    logits = model.predict(patches, batch_size=batch_patches, verbose=0)
    logits = tf.reshape(logits, (h, w))
    return logits


def _update_confusion(conf: Dict[str, int], y_true: np.ndarray, y_pred: np.ndarray) -> None:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    conf["tp"] += tp
    conf["fp"] += fp
    conf["fn"] += fn
    conf["tn"] += tn


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
    return {
        "iou": iou,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "acc": acc,
    }


def full_eval_metrics_stream(
    file_pattern: str,
    model: tf.keras.Model,
    k: int,
    thr_prob: float,
    max_tiles: int,
    batch_patches: int,
    stats: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """Evaluate streaming metrics on tiles."""
    dataset = tile_dataset(file_pattern, stats, shuffle=False, repeat=False)
    conf = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
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
        y_valid = y_core[valid]
        pred_valid = (probs[valid] >= thr_prob).astype(np.int32)
        y_valid = (y_valid > 0).astype(np.int32)
        _update_confusion(conf, y_valid, pred_valid)
        count += 1
        if max_tiles is not None and count >= max_tiles:
            break
    return _metrics_from_conf(conf)
