# src/plots/plot_IRAS.py
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

from models import DatasetModel  

def plot_iras(
    ds: DatasetModel,
    *,
    title: str = "IRAS: Transfer Functions vs ActiWiz (msr)",
    figsize: Tuple[float, float] = (6, 4.5),
    savepath: Optional[str] = None,
    show: bool = True,
    frac_x: float = 0.35,  # label x-position in log range
    frac_y: float = 0.08,  # label y-position in log range
) -> plt.Axes:
    """
    Plot IRAS (y) vs true_IRAS (x) on log–log axes with a 1:1 line + TF/AW label.
    """
    df = ds.df.copy()

    # Ratio for label (allow NaN where denom invalid)
    valid_den = np.isfinite(df["true_IRAS"]) & (df["true_IRAS"] > 0)
    df["IRAS_ratio"] = np.where(valid_den, df["IRAS"] / df["true_IRAS"], np.nan)
    label_value = float(np.nanmean(df["IRAS_ratio"].to_numpy()))

    # Keep only positive, finite pairs for log scales
    m = (
        np.isfinite(df["true_IRAS"]) & np.isfinite(df["IRAS"]) &
        (df["true_IRAS"] > 0) & (df["IRAS"] > 0)
    )
    dfp = df[m].copy()
    if dfp.empty:
        raise ValueError("No positive finite (IRAS, true_IRAS) pairs to plot on log–log axes.")

    # Anchors for label (interpolate in log10 space)
    rx_min, rx_max = float(dfp["true_IRAS"].min()), float(dfp["true_IRAS"].max())
    ry_min, ry_max = float(dfp["IRAS"].min()), float(dfp["IRAS"].max())

    def interp_log(a, b, t):
        return 10 ** (np.log10(a) + t * (np.log10(b) - np.log10(a)))

    x_anchor = interp_log(rx_min, rx_max, frac_x)
    y_anchor = interp_log(ry_min, ry_max, frac_y)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(dfp["true_IRAS"], dfp["IRAS"], s=18)

    xline = np.array([rx_min, rx_max])
    ax.plot(xline, xline, linestyle="--", linewidth=1.2)  # 1:1 line

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("ActiWiz IRAS (true_IRAS)")
    ax.set_ylabel("Transfer Functions IRAS (IRAS)")
    ax.set_title(title)

    ax.annotate(
        f"TF/AW = {label_value:.2f}",
        xy=(x_anchor, y_anchor),
        textcoords="offset points", xytext=(0, 0),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, ec="none"),
    )
    ax.grid(True, which="both", linewidth=0.4, alpha=0.35)

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200)
    if show:
        plt.show()
    return ax
