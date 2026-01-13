"""Dataset builders."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import tensorflow as tf

from wildfire.data.tfrecord import parse_to_dict


def tile_dataset(
    file_pattern: str,
    stats: Dict[str, Dict[str, float]],
    shuffle: bool = False,
    seed: Optional[int] = None,
    repeat: bool = False,
) -> tf.data.Dataset:
    """Create a dataset of parsed tiles from TFRecords."""
    files = tf.data.Dataset.list_files(file_pattern, shuffle=shuffle, seed=seed)
    dataset = files.interleave(
        tf.data.TFRecordDataset,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not shuffle,
    )
    if shuffle:
        dataset = dataset.shuffle(512, seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.map(lambda x: parse_to_dict(x, stats), num_parallel_calls=tf.data.AUTOTUNE)
    if repeat:
        dataset = dataset.repeat()
    return dataset


def count_records(file_pattern: str) -> int:
    """Count number of records matching file pattern."""
    files = tf.io.gfile.glob(file_pattern)
    count = 0
    for filename in files:
        for _ in tf.data.TFRecordDataset(filename):
            count += 1
    return count


def save_dataset_paths(base_dir: Path, file_pattern: str) -> None:
    """Save list of TFRecord paths (utility)."""
    paths = tf.io.gfile.glob(file_pattern)
    path_file = base_dir / "file_list.txt"
    path_file.parent.mkdir(parents=True, exist_ok=True)
    with path_file.open("w", encoding="utf-8") as handle:
        for path in paths:
            handle.write(f"{path}\n")
