"""Colormaps for visualization."""

from __future__ import annotations

import matplotlib.colors as mcolors


def fire_mask_cmap() -> mcolors.ListedColormap:
    """Colormap for fire masks with missing values."""
    colors = ["black", "silver", "orangered"]
    return mcolors.ListedColormap(colors)
