"""Training callbacks."""

from __future__ import annotations

from typing import Dict

import tensorflow as tf

from wildfire.eval.streaming_metrics import full_eval_metrics_stream


class StreamSplitMetrics(tf.keras.callbacks.Callback):
    """Streaming metrics on train/val splits."""

    def __init__(
        self,
        train_pat: str,
        val_pat: str,
        k: int,
        thr_prob: float,
        max_tiles: int,
        batch_patches: int,
        stats: Dict[str, Dict[str, float]],
    ) -> None:
        super().__init__()
        self.train_pat = train_pat
        self.val_pat = val_pat
        self.k = k
        self.thr_prob = thr_prob
        self.max_tiles = max_tiles
        self.batch_patches = batch_patches
        self.stats = stats

    def on_epoch_end(self, epoch: int, logs: Dict[str, float] | None = None) -> None:
        logs = logs or {}
        train_metrics = full_eval_metrics_stream(
            self.train_pat,
            self.model,
            self.k,
            self.thr_prob,
            self.max_tiles,
            self.batch_patches,
            self.stats,
        )
        val_metrics = full_eval_metrics_stream(
            self.val_pat,
            self.model,
            self.k,
            self.thr_prob,
            self.max_tiles,
            self.batch_patches,
            self.stats,
        )
        logs.update({
            "tr_iou": train_metrics["iou"],
            "tr_f1": train_metrics["f1"],
            "tr_precision": train_metrics["precision"],
            "tr_recall": train_metrics["recall"],
            "tr_acc_stream": train_metrics["acc"],
            "val_iou": val_metrics["iou"],
            "val_f1": val_metrics["f1"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_acc_stream": val_metrics["acc"],
        })
        for key, value in logs.items():
            self.model.history.history.setdefault(key, []).append(value)
