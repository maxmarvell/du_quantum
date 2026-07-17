from __future__ import annotations

from importlib.resources import as_file, files
from pathlib import Path
from typing import Sequence

WONG = {"blue": "#0072B2", "black": "#000000", "pink": "#CC79A7",
        "vermilion": "#D55E00", "orange": "#E69F00", "green": "#009E73",
        "sky": "#56B4E9", "yellow": "#F0E442"}


def use_style() -> None:
    import matplotlib.pyplot as plt

    with as_file(files("du") / "wong.mplstyle") as style_path:
        plt.style.use(style_path)


def save_figure(
    fig,
    path: Path | str,
    formats: Sequence[str] = ("pdf", "svg", "png"),
    dpi: int = 300,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(path.with_suffix(f".{ext}"), bbox_inches="tight", dpi=dpi)


def plot_band(ax, x, mean, std, color, label, *, marker=".", capsize=3,
              alpha=0.10, linewidth=0.8, zorder=2, markersize=6, connect=False):
    import numpy as np

    std = np.asarray(std)
    if std.ndim == 2 and std.shape[0] == 2:
        low, high = std[0], std[1]
    else:
        low = high = std
    mean = np.asarray(mean)
    mask = ~(np.isnan(mean) | np.isnan(low) | np.isnan(high))
    x_m, mean_m, low_m, high_m = np.asarray(x)[mask], mean[mask], low[mask], high[mask]
    ax.errorbar(
        x_m, mean_m, [low_m, high_m],
        capsize=capsize, marker=marker, mfc="none", color=color,
        linestyle="-" if connect else "none",
        label=label, linewidth=linewidth, markersize=markersize, zorder=zorder,
    )
    ax.fill_between(
        x_m, mean_m - low_m, mean_m + high_m,
        color=color, alpha=alpha, linewidth=0, zorder=zorder - 1,
    )
