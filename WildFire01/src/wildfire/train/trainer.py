"""Training utilities."""

from __future__ import annotations

from typing import Dict, Tuple

import tensorflow as tf
from tensorflow.keras import callbacks

from wildfire.data.datasets import tile_dataset
from wildfire.data.patch_sampling import make_train_patch_dataset
from wildfire.models.losses import WeightedBCEFromLogits
from wildfire.models.patch_gnn import build_patch_gnn
from wildfire.train.callbacks import StreamSplitMetrics


def compile_patch_gnn(
    model: tf.keras.Model,
    lr: float,
    pos_weight: float,
    use_auc: bool = False,
) -> None:
    """Compile PatchGNN with weighted BCE and metrics."""
    loss = WeightedBCEFromLogits(pos_weight=pos_weight)
    metrics = []
    if use_auc:
        metrics.append(tf.keras.metrics.AUC(curve="ROC", from_logits=True, name="auc_roc"))
        metrics.append(tf.keras.metrics.AUC(curve="PR", from_logits=True, name="auc_pr"))
    else:
        metrics.append(tf.keras.metrics.BinaryAccuracy(threshold=0.0, name="acc"))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=loss,
        metrics=metrics,
    )


def run_train_trial(
    cfg: Dict[str, float],
    budget: int,
    train_pat: str,
    val_pat: str,
    stats: Dict[str, Dict[str, float]],
    batch_size: int,
    n_pos: int,
    n_neg: int,
    max_tiles_eval: int,
    batch_patches: int,
    monitor: str = "val_f1",
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """Train a model with streaming metrics."""
    model = build_patch_gnn(
        k=cfg["k_patch"],
        c_in=cfg["c_in"],
        d_hidden=cfg["d_hidden"],
        n_layers=cfg["n_layers"],
        dropout=cfg["dropout"],
    )
    compile_patch_gnn(model, lr=cfg["lr"], pos_weight=cfg["pos_weight"], use_auc=True)

    train_tiles = tile_dataset(train_pat, stats, shuffle=True, repeat=True)
    train_patches = make_train_patch_dataset(train_tiles, cfg["k_patch"], n_pos, n_neg, batch_size)

    stream_cb = StreamSplitMetrics(
        train_pat,
        val_pat,
        cfg["k_patch"],
        cfg["thr_prob"],
        max_tiles_eval,
        batch_patches,
        stats,
    )
    early_stop = callbacks.EarlyStopping(monitor=monitor, patience=5, mode="max", restore_best_weights=True)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor=monitor, patience=3, factor=0.5, mode="max")

    history = model.fit(
        train_patches,
        steps_per_epoch=cfg["steps_per_epoch"],
        epochs=budget,
        callbacks=[stream_cb, early_stop, reduce_lr],
        verbose=2,
    )
    return model, history
