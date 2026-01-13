"""Count TFRecord records."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import argparse

from wildfire.data.datasets import count_records
from wildfire.utils.io import read_yaml
from wildfire.utils.logging import get_logger

LOGGER = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--data_dir", default=None)
    args = parser.parse_args()

    cfg = read_yaml(Path(args.config))
    data_dir = Path(args.data_dir) if args.data_dir else None
    patterns = cfg["data"]
    for split in ["train_pat", "val_pat", "test_pat"]:
        pat = patterns[split]
        file_pattern = str(data_dir / pat) if data_dir else pat
        count = count_records(file_pattern)
        LOGGER.info("%s: %s records", split, count)


if __name__ == "__main__":
    main()
