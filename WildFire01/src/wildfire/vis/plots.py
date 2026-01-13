"""Plotting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from wildfire.eval.streaming_metrics import infer_core_logit_map_valid, infer_full_map_logit_same
from wildfire.vis.colormaps import fire_mask_cmap


def plot_tile_predictions(
    sample: Dict[str, tf.Tensor],
    model: tf.keras.Model,
    k: int,
    thr_prob: float,
    batch_patches: int,
    out_path: Path,
) -> None:
    """Plot GT vs prob vs prediction mask for a tile."""
    x_tile = sample["x"]
    y_raw = sample["y_raw"].numpy()
    logits = infer_full_map_logit_same(x_tile, model, k, batch_patches).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    pred = (probs >= thr_prob).astype(np.int32)

    cmap = fire_mask_cmap()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(y_raw, cmap=cmap, vmin=-1, vmax=1)
    axes[0].set_title("Ground Truth")
    axes[1].imshow(probs, cmap="viridis")
    axes[1].set_title("Probability")
    axes[2].imshow(pred, cmap=cmap, vmin=0, vmax=1)
    axes[2].set_title("Prediction")
    for ax in axes:
        ax.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def downsample_with_missing(arr: np.ndarray, factor: int) -> np.ndarray:
    """Downsample preserving missing values (-1)."""
    if factor == 1:
        return arr
    h, w = arr.shape
    new_h = h // factor
    new_w = w // factor
    arr = arr[: new_h * factor, : new_w * factor]
    arr = arr.reshape(new_h, factor, new_w, factor)
    missing = np.any(arr < 0, axis=(1, 3))
    mean = np.mean(arr, axis=(1, 3))
    mean[missing] = -1
    return mean


def plot_multi_resolution(
    sample: Dict[str, tf.Tensor],
    model: tf.keras.Model,
    k: int,
    thr_prob: float,
    batch_patches: int,
    factors: List[int],
    out_path: Path,
) -> None:
    """Plot multi-resolution outputs for a tile."""
    x_tile = sample["x"]
    y_raw = sample["y_raw"].numpy()
    logits = infer_full_map_logit_same(x_tile, model, k, batch_patches).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    pred = (probs >= thr_prob).astype(np.int32)

    cmap = fire_mask_cmap()
    n = len(factors)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    for idx, factor in enumerate(factors):
        y_ds = downsample_with_missing(y_raw, factor)
        p_ds = downsample_with_missing(probs, factor)
        pred_ds = downsample_with_missing(pred, factor)
        axes[idx, 0].imshow(y_ds, cmap=cmap, vmin=-1, vmax=1)
        axes[idx, 0].set_title(f"GT x{factor}")
        axes[idx, 1].imshow(p_ds, cmap="viridis")
        axes[idx, 1].set_title(f"Prob x{factor}")
        axes[idx, 2].imshow(pred_ds, cmap=cmap, vmin=0, vmax=1)
        axes[idx, 2].set_title(f"Pred x{factor}")
        for ax in axes[idx]:
            ax.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
