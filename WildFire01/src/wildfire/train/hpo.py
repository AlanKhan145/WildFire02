"""Hyperparameter optimization utilities."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from wildfire.train.trainer import run_train_trial


@dataclass
class TrialResult:
    cfg: Dict[str, float]
    score: float
    val_iou: float
    val_f1: float
    val_acc: float


def _score_trial(val_iou: float, val_f1: float, val_acc: float) -> float:
    score = val_iou * (0.85 + 0.15 * val_f1)
    if val_acc < 0.5:
        score *= 0.5
    return score


def safe_train_trial(
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
    monitor: str,
) -> TrialResult:
    """Run a training trial with OOM safety."""
    steps = cfg["steps_per_epoch"]
    budget = max(budget, 4)
    while steps >= 1:
        cfg["steps_per_epoch"] = steps
        try:
            model, history = run_train_trial(
                cfg,
                budget,
                train_pat,
                val_pat,
                stats,
                batch_size,
                n_pos,
                n_neg,
                max_tiles_eval,
                batch_patches,
                monitor,
            )
            hist = history.history
            val_iou = float(hist.get("val_iou", [0.0])[-1])
            val_f1 = float(hist.get("val_f1", [0.0])[-1])
            val_acc = float(hist.get("val_acc_stream", [0.0])[-1])
            score = _score_trial(val_iou, val_f1, val_acc)
            return TrialResult(cfg=cfg.copy(), score=score, val_iou=val_iou, val_f1=val_f1, val_acc=val_acc)
        except tf.errors.ResourceExhaustedError:
            steps = steps // 2
            tf.keras.backend.clear_session()
    return TrialResult(cfg=cfg.copy(), score=0.0, val_iou=0.0, val_f1=0.0, val_acc=0.0)


def successive_halving_hpo(
    configs: List[Dict[str, float]],
    train_pat: str,
    val_pat: str,
    stats: Dict[str, Dict[str, float]],
    batch_size: int,
    n_pos: int,
    n_neg: int,
    max_tiles_eval: int,
    batch_patches: int,
    rung_epochs: List[int],
    monitor: str = "val_f1",
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Run successive halving HPO."""
    rung_epochs = [max(int(e), 4) for e in rung_epochs]
    survivors = configs
    results_rows = []

    for rung, budget in enumerate(rung_epochs):
        trial_results = []
        for cfg in survivors:
            result = safe_train_trial(
                cfg,
                budget,
                train_pat,
                val_pat,
                stats,
                batch_size,
                n_pos,
                n_neg,
                max_tiles_eval,
                batch_patches,
                monitor,
            )
            trial_results.append(result)
            results_rows.append({
                "rung": rung,
                "budget": budget,
                **result.cfg,
                "score": result.score,
                "val_iou": result.val_iou,
                "val_f1": result.val_f1,
                "val_acc": result.val_acc,
            })
            tf.keras.backend.clear_session()
            gc.collect()

        trial_results.sort(key=lambda r: r.score, reverse=True)
        keep = max(1, len(trial_results) // 2)
        survivors = [r.cfg for r in trial_results[:keep]]

    df = pd.DataFrame(results_rows)
    best = df.sort_values("score", ascending=False).iloc[0]
    best_cfg = {key: best[key] for key in ["k_patch", "n_layers", "d_hidden", "dropout", "lr", "pos_weight", "c_in", "thr_prob", "steps_per_epoch"]}
    return best_cfg, df
