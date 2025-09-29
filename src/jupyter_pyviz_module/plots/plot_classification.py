from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Tuple

__all__ = ["plot_classification"]


def plot_classification(
    csv_path: str,
    *,
    # columns
    X_axis: str = "Co-60_eq",
    Y_axis: str = "DR_10cm_uSv_h",
    COL_LL: str = "item_LL",
    COL_MASS_KG: str = "mass_kg",
    COL_MATERIAL: str = "material",
    material_item_value: str = "item",
    # figure & styling
    title: str = "Classification scatterplot",
    figsize: Tuple[float, float] = (9, 7),
    dpi: int = 120,
    x_ref: Optional[float] = 0.04,
    y_ref: Optional[float] = 0.01,
    xlim: Tuple[float, float] = (1e-3, 1e-1),
    ylim: Tuple[float, float] = (1e-3, 1e-1),
) -> Tuple[plt.Figure, plt.Axes]:

    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    if COL_MATERIAL not in df.columns:
        raise KeyError(f"Column '{COL_MATERIAL}' not found in CSV.")
    dfi = df.loc[df[COL_MATERIAL].astype(str).str.lower().eq(material_item_value.lower())].copy()
    if dfi.empty:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.text(0.5, 0.5, "No rows with material == 'item'", ha="center", va="center")
        ax.axis("off")
        return fig, ax

    needed = [X_axis, Y_axis, COL_LL, COL_MASS_KG]
    for c in needed:
        if c not in dfi.columns:
            raise KeyError(f"Column '{c}' not found in CSV.")
        dfi[c] = pd.to_numeric(dfi[c], errors="coerce")
    dfi = dfi.dropna(subset=needed)
    if dfi.empty:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.text(0.5, 0.5, "No valid numeric data for item rows", ha="center", va="center")
        ax.axis("off")
        return fig, ax

    # classification
    dr = dfi[Y_axis].to_numpy()
    ll = dfi[COL_LL].to_numpy()
    dfi["class"] = np.where((dr < 0.1) & (ll < 1.0), 0, 1)

    # mass encodings
    mass = dfi[COL_MASS_KG].to_numpy()
    mmin, mmax = np.nanpercentile(mass, [5, 95]) if mass.size else (0.0, 1.0)
    if not np.isfinite(mmin): mmin = 0.0
    if not np.isfinite(mmax): mmax = 1.0
    span = (mmax - mmin) if mmax > mmin else 1.0
    sizes  = 10 + 80 * (np.clip(mass, mmin, mmax) - mmin) / span
    alphas = 0.20 + 0.80 * (np.clip(mass, mmin, mmax) - mmin) / span

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # plot class 0, then class 1 
    for cls, z, lw in [(0, 2, 0.55), (1, 3, 0.65)]:
        m = (dfi["class"].values == cls)
        if not np.any(m):
            continue  # nothing of this class; skip

        color = "#2ca02c" if cls == 0 else "#d62728"
        x = dfi.loc[m, X_axis].to_numpy()
        y = dfi.loc[m, Y_axis].to_numpy()
        s = sizes[m]

        sc = ax.scatter(
            x, y,
            s=s,
            c=color,               
            alpha=1.0,             
            edgecolor="black",
            linewidths=lw,
            zorder=z,
        )
        # set per-point alpha on facecolors 
        fc = sc.get_facecolors()
        if len(fc) == s.size:      # expected: one facecolor per point
            fc[:, 3] = alphas[m]
            sc.set_facecolors(fc)

    # axes
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    xticks = [0.001, 0.003, 0.01, 0.03, 0.1]
    yticks = [0.001, 0.003, 0.01, 0.03, 0.1]
    ax.set_xticks(xticks); ax.set_yticks(yticks)
    fmt = lambda v: ("%.3f" % v).rstrip("0").rstrip(".")
    ax.set_xticklabels([fmt(v) for v in xticks])
    ax.set_yticklabels([fmt(v) for v in yticks])
    ax.minorticks_off()

    ax.set_xlabel("Co-60 equivalent activity [Bq/g]")
    ax.set_ylabel("Dose rate [µSv/h]")
    ax.set_title(title, pad=10, fontsize=14, weight="bold")
    ax.grid(True, which="both", axis="both", alpha=0.15)

    # guides & labels at fixed reference positions 
    if x_ref is not None:
        ax.axvline(x_ref, ls="--", lw=1.2, color="dodgerblue")
        ax.text(0.041, 0.0011, f"{x_ref:g} Bq/g", color="dodgerblue")
    if y_ref is not None:
        ax.axhline(y_ref, ls="--", lw=1.2, color="dodgerblue")
        ax.text(0.0012, 0.0105, "0.01 uSv/h", color="dodgerblue")

    # legends
    h0 = plt.Line2D([], [], marker="o", linestyle="None",
                    markerfacecolor="#2ca02c", markeredgecolor="black",
                    markersize=8, label="0")
    h1 = plt.Line2D([], [], marker="o", linestyle="None",
                    markerfacecolor="#d62728", markeredgecolor="black",
                    markersize=8, label="1")
    leg1 = ax.legend(
        handles=[h0, h1], title="Classification:",
        loc="upper left", bbox_to_anchor=(0.02, 0.98),
        frameon=True, fancybox=False, edgecolor="black", borderpad=0.6
    )
    ax.add_artist(leg1)

    def m2s(mass_val: float) -> float:
        return 10 + 80 * (np.clip(mass_val, mmin, mmax) - mmin) / span

    m_handles = [
        plt.scatter([], [], s=m2s(250), facecolors="none", edgecolors="black"),
        plt.scatter([], [], s=m2s(500), facecolors="none", edgecolors="black"),
        plt.scatter([], [], s=m2s(750), facecolors="none", edgecolors="black"),
    ]
    ax.legend(
        m_handles, ["250", "500", "750"], title="Mass [kg]:",
        loc="upper left", bbox_to_anchor=(0.40, 0.98),
        frameon=True, fancybox=False, edgecolor="black", borderpad=0.6
    )

    fig.tight_layout()
    return fig, ax
