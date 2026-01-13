"""GNN layers for patch model."""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def moore8_adj_unweighted(k: int) -> tf.Tensor:
    """Create unweighted adjacency for kxk grid with 8-neighborhood."""
    n = k * k
    adj = tf.zeros((n, n), dtype=tf.float32)
    coords = [(i, j) for i in range(k) for j in range(k)]
    idx_map = {coords[i]: i for i in range(n)}
    for i in range(k):
        for j in range(k):
            node = idx_map[(i, j)]
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < k and 0 <= nj < k:
                        neigh = idx_map[(ni, nj)]
                        adj = tf.tensor_scatter_nd_update(adj, [[node, neigh]], [1.0])
    return adj


def normalize_adj(adj: tf.Tensor) -> tf.Tensor:
    """Symmetric adjacency normalization."""
    adj = adj + tf.eye(tf.shape(adj)[0])
    deg = tf.reduce_sum(adj, axis=-1)
    deg_inv_sqrt = tf.pow(deg, -0.5)
    deg_inv_sqrt = tf.where(tf.math.is_finite(deg_inv_sqrt), deg_inv_sqrt, tf.zeros_like(deg_inv_sqrt))
    norm = adj * deg_inv_sqrt[:, tf.newaxis] * deg_inv_sqrt[tf.newaxis, :]
    return norm


class GraphConv(layers.Layer):
    """Simple graph convolution."""

    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dense = layers.Dense(units)

    def call(self, x: tf.Tensor, adj: tf.Tensor) -> tf.Tensor:
        support = self.dense(x)
        return tf.linalg.matmul(adj, support)


class GNNBlock(layers.Layer):
    """Graph convolution block with residual and dropout."""

    def __init__(self, units: int, dropout: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.conv = GraphConv(units)
        self.act = layers.Activation("relu")
        self.dropout = layers.Dropout(dropout)
        self.proj = layers.Dense(units)

    def call(self, x: tf.Tensor, adj: tf.Tensor, training: bool = False) -> tf.Tensor:
        h = self.conv(x, adj)
        h = self.act(h)
        h = self.dropout(h, training=training)
        if x.shape[-1] != h.shape[-1]:
            x = self.proj(x)
        return x + h
