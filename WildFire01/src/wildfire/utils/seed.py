"""Seed helpers."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import tensorflow as tf


def set_global_seed(seed: Optional[int] = None) -> None:
    """Set random seeds for python, numpy, and tensorflow."""
    if seed is None:
        return
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
