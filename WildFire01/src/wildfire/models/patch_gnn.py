"""Patch-wise GNN model."""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from wildfire.models.gnn_layers import GNNBlock, moore8_adj_unweighted, normalize_adj


def build_patch_gnn(
    k: int,
    c_in: int,
    d_hidden: int,
    n_layers: int,
    dropout: float,
) -> keras.Model:
    """Build PatchGNN model."""
    n_nodes = k * k
    inputs = keras.Input(shape=(k, k, c_in), name="patch")
    x = layers.Reshape((n_nodes, c_in))(inputs)
    x = layers.Dense(d_hidden, activation="relu")(x)

    pos_ids = tf.range(n_nodes, dtype=tf.int32)
    pos_embed = layers.Embedding(input_dim=n_nodes, output_dim=d_hidden)(pos_ids)
    pos_embed = layers.Lambda(lambda t: tf.expand_dims(t, 0))(pos_embed)
    x = layers.Add()([x, pos_embed])

    adj = moore8_adj_unweighted(k)
    adj = normalize_adj(adj)

    for _ in range(n_layers):
        x = GNNBlock(d_hidden, dropout=dropout)(x, adj)

    center_idx = n_nodes // 2
    center = layers.Lambda(lambda t: t[:, center_idx, :])(x)
    global_mean = layers.Lambda(lambda t: tf.reduce_mean(t, axis=1))(x)
    readout = layers.Concatenate()([center, global_mean])
    outputs = layers.Dense(1, name="logit")(readout)
    outputs = layers.Reshape(())(outputs)

    model = keras.Model(inputs=inputs, outputs=outputs, name="PatchGNN")
    return model
