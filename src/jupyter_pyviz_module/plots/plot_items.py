# src/plots/items.py
from __future__ import annotations
from typing import Optional, Tuple
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# --- helpers  ---
def _fd_bins_log(x: pd.Series) -> np.ndarray:
    """Freedman–Diaconis bins in log10 space; returns log-spaced edges."""
    x = pd.to_numeric(x, errors="coerce")
    x = x[np.isfinite(x) & (x > 0)]
    if x.empty or x.min() == x.max():
        lo = x.min() if not x.empty else 1e-3
        hi = x.max() if not x.empty else 1.0
        return np.logspace(np.log10(lo), np.log10(hi), 10)
    lx = np.log10(x)
    q75, q25 = np.percentile(lx, [75, 25])
    iqr = q75 - q25
    n = len(lx)
    if iqr <= 0 or n < 2:
        return np.logspace(lx.min(), lx.max(), 20)
    h = 2 * iqr * (n ** (-1/3))
    span = lx.max() - lx.min()
    k = max(5, int(math.ceil(span / max(h, 1e-6))))
    return np.logspace(lx.min(), lx.max(), k + 1)

def _order_by_median(cat: pd.Series, values: pd.Series, ascending=True):
    med = (
        pd.DataFrame({"cat": cat, "val": values})
        .dropna().groupby("cat")["val"].median()
        .sort_values(ascending=ascending)
    )
    return pd.Categorical(cat, categories=med.index.tolist(), ordered=True)

# --- main plot ---
def plot_items(
    filename: str,
    *,
    figsize: Tuple[float, float] = (14, 10),
    show: bool = True,
    savepath: Optional[str] = None,
):
    """
    Plot (almost) all item characteristics .
    Expected columns:
      material, position, DR_10cm_uSv_h, item_DR_10cm_uSv_h,
      item_LL, item_IRAS, total_activity_Bq_g
    Returns the Matplotlib Figure.
    """
    df = pd.read_csv(filename)

    needed = [
        "material","position","DR_10cm_uSv_h","item_DR_10cm_uSv_h",
        "item_LL","item_IRAS","total_activity_Bq_g"
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for items plot: {missing}")

    # numeric coercions
    for c in ["DR_10cm_uSv_h","item_DR_10cm_uSv_h","item_LL","item_IRAS","total_activity_Bq_g"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    m_item = (df["material"] == "item")

    # classification 0/1
    cls = np.where(
        (df["item_DR_10cm_uSv_h"] < 0.1) & (df["item_LL"] < 1.0),
        0, 1
    )
    df["item_classification"] = cls

    # layout: 3 rows x 3 cols (plot01/plot00/plot02)
    fig, axes = plt.subplots(3, 3, figsize=figsize)
    (axA1, axA2, axA3), (axB1, axB2, axB3), (axC1, axC2, axC3) = axes

    # Row A: Classification, LL, item_IRAS
    counts = pd.Series(df["item_classification"]).value_counts().reindex([0,1]).fillna(0).astype(int)
    labels = ["Conventional", "Radioactive"]
    axA1.bar(labels, counts.values, width=0.6)
    total = counts.sum() if counts.sum() > 0 else 1
    for i, v in enumerate(counts.values):
        pct = 100.0 * v / total
        axA1.text(i, v, f"{pct:.1f}%", va="bottom", ha="center")
    axA1.set_title("Classification"); axA1.set_ylabel("Count")

    x = df.loc[m_item, "item_LL"]
    bins = _fd_bins_log(x); axA2.hist(x[np.isfinite(x) & (x > 0)], bins=bins, histtype="step")
    axA2.set_xscale("log"); axA2.set_xlabel("LL"); axA2.set_ylabel("Number of items")
    axA2.axvline(1.0, linestyle="--", linewidth=0.8)

    x = df.loc[m_item, "item_IRAS"]
    bins = _fd_bins_log(x); axA3.hist(x[np.isfinite(x) & (x > 0)], bins=bins, histtype="step")
    axA3.set_xscale("log"); axA3.set_xlabel("IRAS"); axA3.set_ylabel("Number of items")

    # Row B: item dose @10 cm hists + box by position/material
    x = df.loc[m_item, "item_DR_10cm_uSv_h"]
    bins = _fd_bins_log(x); axB1.hist(x[np.isfinite(x) & (x > 0)], bins=bins, histtype="step")
    axB1.set_xscale("log"); axB1.set_xlabel("Dose rate @ 10 cm [µSv/h]"); axB1.set_ylabel("Number of items")

    sub = df.loc[m_item, ["position","item_DR_10cm_uSv_h"]].dropna()
    if not sub.empty:
        sub = sub.assign(position=_order_by_median(sub["position"], sub["item_DR_10cm_uSv_h"], ascending=True))
        pos = list(sub["position"].cat.categories)
        data = [sub.loc[sub["position"] == p, "item_DR_10cm_uSv_h"].values for p in pos]
        axB2.boxplot(data, vert=False, labels=pos, showfliers=False)
        axB2.set_xscale("log")
    axB2.set_xlabel("Dose rate @ 10 cm [µSv/h]"); axB2.set_ylabel("Irradiation position")

    sub = df[["material","DR_10cm_uSv_h"]].dropna()
    if not sub.empty:
        sub = sub.assign(material=_order_by_median(sub["material"], sub["DR_10cm_uSv_h"], ascending=True))
        mats = list(sub["material"].cat.categories)
        data = [sub.loc[sub["material"] == m, "DR_10cm_uSv_h"].values for m in mats]
        axB3.boxplot(data, vert=False, labels=mats, showfliers=False)
        axB3.set_xscale("log")
    axB3.set_xlabel("Dose rate @ 10 cm [µSv/h]"); axB3.set_ylabel("Item component")

    # Row C: total activity hists + box by position/material
    x = df.loc[m_item, "total_activity_Bq_g"]
    bins = _fd_bins_log(x); axC1.hist(x[np.isfinite(x) & (x > 0)], bins=bins, histtype="step")
    axC1.set_xscale("log"); axC1.set_xlabel("Total activity [Bq/g]"); axC1.set_ylabel("Number of items")

    sub = df.loc[m_item, ["position","total_activity_Bq_g"]].dropna()
    if not sub.empty:
        sub = sub.assign(position=_order_by_median(sub["position"], sub["total_activity_Bq_g"], ascending=True))
        pos = list(sub["position"].cat.categories)
        data = [sub.loc[sub["position"] == p, "total_activity_Bq_g"].values for p in pos]
        axC2.boxplot(data, vert=False, labels=pos, showfliers=False)
        axC2.set_xscale("log")
    axC2.set_xlabel("Total activity [Bq/g]"); axC2.set_ylabel("Irradiation position")

    sub = df[["material","total_activity_Bq_g"]].dropna()
    if not sub.empty:
        sub = sub.assign(material=_order_by_median(sub["material"], sub["total_activity_Bq_g"], ascending=True))
        mats = list(sub["material"].cat.categories)
        data = [sub.loc[sub["material"] == m, "total_activity_Bq_g"].values for m in mats]
        axC3.boxplot(data, vert=False, labels=mats, showfliers=False)
        axC3.set_xscale("log")
    axC3.set_xlabel("Total activity [Bq/g]"); axC3.set_ylabel("Item component")

    fig.tight_layout()
    if savepath: fig.savefig(savepath, dpi=200)
    if show: plt.show()
    return fig
