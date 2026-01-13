"""TFRecord parsing utilities."""

from __future__ import annotations

from typing import Dict, Tuple

import tensorflow as tf

from wildfire.data.constants import DATA_SIZE, ENV11, PREV_KEY, NEXT_KEY
from wildfire.data.stats import clip_norm_tf

FEATURE_SPEC = {
    **{name: tf.io.FixedLenFeature([DATA_SIZE * DATA_SIZE], tf.float32) for name in ENV11},
    PREV_KEY: tf.io.FixedLenFeature([DATA_SIZE * DATA_SIZE], tf.float32),
    NEXT_KEY: tf.io.FixedLenFeature([DATA_SIZE * DATA_SIZE], tf.float32),
}


def _reshape_feature(x: tf.Tensor) -> tf.Tensor:
    return tf.reshape(x, (DATA_SIZE, DATA_SIZE))


def parse_to_tile(example_proto: tf.Tensor, stats: Dict[str, Dict[str, float]]) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Parse TFRecord to tile tensors."""
    parsed = tf.io.parse_single_example(example_proto, FEATURE_SPEC)
    x_env_raw = [_reshape_feature(parsed[name]) for name in ENV11]
    x_env = [clip_norm_tf(x, stats, key=name) for x, name in zip(x_env_raw, ENV11)]
    x_env = tf.stack(x_env, axis=-1)

    prev_raw = _reshape_feature(parsed[PREV_KEY])
    prev_bin = tf.cast(prev_raw > 0.0, tf.float32)

    ndvi_raw = _reshape_feature(parsed["NDVI"])
    y_raw = _reshape_feature(parsed[NEXT_KEY])

    x = tf.concat([x_env, tf.expand_dims(prev_bin, axis=-1)], axis=-1)
    return x, prev_raw, ndvi_raw, y_raw


def parse_to_dict(example_proto: tf.Tensor, stats: Dict[str, Dict[str, float]]) -> Dict[str, tf.Tensor]:
    """Parse TFRecord into dict for dataset pipelines."""
    x, prev_raw, ndvi_raw, y_raw = parse_to_tile(example_proto, stats)
    x_env = x[..., :-1]
    return {
        "x": x,
        "x_env": x_env,
        "prev_raw": prev_raw,
        "ndvi_raw": ndvi_raw,
        "y_raw": y_raw,
    }
