"""Compute training statistics for normalization."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import argparse

import tensorflow as tf

from wildfire.data.constants import DATA_SIZE, ENV11
from wildfire.data.stats import compute_train_stats_cached
from wildfire.utils.io import read_yaml
from wildfire.utils.logging import get_logger

LOGGER = get_logger(__name__)


def _raw_env_dataset(file_pattern: str) -> tf.data.Dataset:
    feature_spec = {name: tf.io.FixedLenFeature([DATA_SIZE * DATA_SIZE], tf.float32) for name in ENV11}

    def parse(example_proto: tf.Tensor):
        parsed = tf.io.parse_single_example(example_proto, feature_spec)
        x_env = [tf.reshape(parsed[name], (DATA_SIZE, DATA_SIZE)) for name in ENV11]
        x_env = tf.stack(x_env, axis=-1)
        return {"x_env": x_env}

    files = tf.data.Dataset.list_files(file_pattern, shuffle=False)
    return files.interleave(tf.data.TFRecordDataset, num_parallel_calls=tf.data.AUTOTUNE).map(
        parse, num_parallel_calls=tf.data.AUTOTUNE
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--artifacts_dir", default="artifacts")
    parser.add_argument("--max_tiles", type=int, default=None)
    args = parser.parse_args()

    cfg = read_yaml(Path(args.config))
    data_dir = Path(args.data_dir) if args.data_dir else None
    train_pat = cfg["data"]["train_pat"]
    file_pattern = str(data_dir / train_pat) if data_dir else train_pat

    dataset = _raw_env_dataset(file_pattern)
    stats_path = Path(args.artifacts_dir) / "data" / cfg["data"]["stats_json"]
    compute_train_stats_cached(dataset, stats_path, max_tiles=args.max_tiles)


if __name__ == "__main__":
    main()
