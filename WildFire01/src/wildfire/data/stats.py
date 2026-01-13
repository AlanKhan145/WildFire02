"""Training statistics utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import tensorflow as tf

from wildfire.data.constants import ENV11
from wildfire.utils.logging import get_logger

LOGGER = get_logger(__name__)


def compute_clip_bounds_percentile(values: np.ndarray, lower: float = 1.0, upper: float = 99.0) -> Tuple[float, float]:
    """Compute percentile-based clip bounds."""
    lower_val = float(np.percentile(values, lower))
    upper_val = float(np.percentile(values, upper))
    return lower_val, upper_val


def compute_mean_std_after_clip(values: np.ndarray, clip_bounds: Tuple[float, float]) -> Tuple[float, float]:
    """Compute mean/std after clipping values."""
    clipped = np.clip(values, clip_bounds[0], clip_bounds[1])
    mean = float(np.mean(clipped))
    std = float(np.std(clipped))
    return mean, std


def compute_train_stats_cached(
    dataset: tf.data.Dataset,
    save_json: Path,
    max_tiles: int | None = None,
) -> Dict[str, Dict[str, float]]:
    """Compute clip/mean/std for ENV11 channels and cache to JSON."""
    if save_json.exists():
        LOGGER.info("Loading cached stats from %s", save_json)
        with save_json.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    values_by_key = {key: [] for key in ENV11}
    count = 0
    for batch in dataset:
        x_env = batch["x_env"].numpy()
        for idx, key in enumerate(ENV11):
            values_by_key[key].append(x_env[..., idx].reshape(-1))
        count += 1
        if max_tiles is not None and count >= max_tiles:
            break

    stats: Dict[str, Dict[str, float]] = {}
    for key, chunks in values_by_key.items():
        values = np.concatenate(chunks, axis=0)
        clip_bounds = compute_clip_bounds_percentile(values)
        mean, std = compute_mean_std_after_clip(values, clip_bounds)
        stats[key] = {
            "clip_min": clip_bounds[0],
            "clip_max": clip_bounds[1],
            "mean": mean,
            "std": std,
        }

    save_json.parent.mkdir(parents=True, exist_ok=True)
    with save_json.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    LOGGER.info("Saved stats to %s", save_json)
    return stats


def clip_norm_tf(x: tf.Tensor, stats: Dict[str, Dict[str, float]], key: str) -> tf.Tensor:
    """Clip and z-score normalize tensor for key."""
    cfg = stats[key]
    x = tf.clip_by_value(x, cfg["clip_min"], cfg["clip_max"])
    return (x - cfg["mean"]) / (cfg["std"] + 1e-6)
