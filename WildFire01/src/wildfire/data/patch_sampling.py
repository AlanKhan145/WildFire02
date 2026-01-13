"""Patch sampling utilities."""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf

USE_FUEL_MASK = False
CROP_TRAIN = 32


def random_crop_triplet(
    x: tf.Tensor,
    prev_raw: tf.Tensor,
    y_raw: tf.Tensor,
    ndvi_raw: tf.Tensor,
    crop: int = CROP_TRAIN,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Randomly crop x/prev/y/ndvi to a square crop."""
    combined = tf.concat(
        [x, prev_raw[..., tf.newaxis], y_raw[..., tf.newaxis], ndvi_raw[..., tf.newaxis]], axis=-1
    )
    combined = tf.image.random_crop(combined, size=[crop, crop, combined.shape[-1]])
    x_crop = combined[..., :-3]
    prev_crop = combined[..., -3]
    y_crop = combined[..., -2]
    ndvi_crop = combined[..., -1]
    return x_crop, prev_crop, y_crop, ndvi_crop


def valid_core_mask(
    prev_raw: tf.Tensor,
    y_raw: tf.Tensor,
    ndvi_raw: tf.Tensor,
    k: int,
    use_fuel_mask: bool = USE_FUEL_MASK,
) -> tf.Tensor:
    """Compute valid core mask for patch centers."""
    valid = tf.logical_and(y_raw >= 0, prev_raw >= 0)
    if use_fuel_mask:
        valid = tf.logical_and(valid, ndvi_raw >= 0)
    r = k // 2
    if r == 0:
        return valid
    core = valid[r:-r, r:-r]
    return core


def _extract_patches(x: tf.Tensor, k: int) -> tf.Tensor:
    patches = tf.image.extract_patches(
        x[tf.newaxis, ...],
        sizes=[1, k, k, 1],
        strides=[1, 1, 1, 1],
        rates=[1, 1, 1, 1],
        padding="VALID",
    )
    h = tf.shape(patches)[1]
    w = tf.shape(patches)[2]
    c = tf.shape(x)[-1]
    patches = tf.reshape(patches, (h, w, k, k, c))
    return patches


def _sample_indices(num: tf.Tensor, count: int) -> tf.Tensor:
    num = tf.cast(num, tf.int32)
    count = tf.cast(count, tf.int32)
    def sample():
        idx = tf.random.uniform([count], 0, num, dtype=tf.int32)
        return idx
    return tf.cond(num > 0, sample, lambda: tf.zeros([count], dtype=tf.int32))


def sample_patches_from_tile(
    x: tf.Tensor,
    prev_raw: tf.Tensor,
    y_raw: tf.Tensor,
    ndvi_raw: tf.Tensor,
    k: int,
    n_pos: int,
    n_neg: int,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Sample positive/negative patches from a tile with replacement."""
    patches = _extract_patches(x, k)
    r = k // 2
    y_core = y_raw[r:-r, r:-r]
    prev_core = prev_raw[r:-r, r:-r]
    ndvi_core = ndvi_raw[r:-r, r:-r]
    valid = valid_core_mask(prev_raw, y_raw, ndvi_raw, k)

    pos_mask = tf.logical_and(valid, y_core > 0)
    neg_mask = tf.logical_and(valid, y_core <= 0)

    pos_idx = tf.where(pos_mask)
    neg_idx = tf.where(neg_mask)

    pos_sel = _sample_indices(tf.shape(pos_idx)[0], n_pos)
    neg_sel = _sample_indices(tf.shape(neg_idx)[0], n_neg)

    pos_points = tf.gather(pos_idx, pos_sel)
    neg_points = tf.gather(neg_idx, neg_sel)
    points = tf.concat([pos_points, neg_points], axis=0)

    patch_list = tf.gather_nd(patches, points)
    labels = tf.concat([
        tf.ones([n_pos], dtype=tf.float32),
        tf.zeros([n_neg], dtype=tf.float32),
    ], axis=0)
    return patch_list, labels


def make_train_patch_dataset(
    tile_ds: tf.data.Dataset,
    k: int,
    n_pos: int,
    n_neg: int,
    batch_size: int,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """Create a patch dataset for training."""
    def map_crop(sample: dict) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        return random_crop_triplet(sample["x"], sample["prev_raw"], sample["y_raw"], sample["ndvi_raw"], CROP_TRAIN)

    def map_patches(x: tf.Tensor, prev_raw: tf.Tensor, y_raw: tf.Tensor, ndvi_raw: tf.Tensor):
        patches, labels = sample_patches_from_tile(x, prev_raw, y_raw, ndvi_raw, k, n_pos, n_neg)
        return tf.data.Dataset.from_tensor_slices((patches, labels))

    dataset = tile_ds.map(map_crop, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.interleave(map_patches, cycle_length=4, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(1024)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
