"""Loss functions."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import losses


class WeightedBCEFromLogits(losses.Loss):
    """Weighted BCE from logits using tf.nn.weighted_cross_entropy_with_logits."""

    def __init__(self, pos_weight: float, name: str = "weighted_bce"):
        super().__init__(name=name)
        self.pos_weight = pos_weight

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        loss = tf.nn.weighted_cross_entropy_with_logits(
            labels=y_true, logits=y_pred, pos_weight=self.pos_weight
        )
        return tf.reduce_mean(loss)
