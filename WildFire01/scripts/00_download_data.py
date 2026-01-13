"""Download Kaggle dataset."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import argparse

import kagglehub

from wildfire.utils.logging import get_logger
from wildfire.utils.io import read_yaml

LOGGER = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--artifacts_dir", default="artifacts")
    args = parser.parse_args()

    cfg = read_yaml(Path(args.config))
    dataset = cfg["data"]["dataset"]
    out_path = kagglehub.dataset_download(dataset)
    LOGGER.info("Downloaded dataset to %s", out_path)

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "data" / "dataset_path.txt").write_text(str(out_path), encoding="utf-8")


if __name__ == "__main__":
    main()
