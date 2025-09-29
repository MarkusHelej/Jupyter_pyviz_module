# src/plots/plot_logistic_regression.py
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Sequence

from sklearn.linear_model import LogisticRegression

__all__ = ["plot_logistic_regression"]

def _logit(p: float) -> float:
    p = float(p)
    if not (0 < p < 1):
        raise ValueError("probability must be in (0, 1)")
    return np.log(p / (1.0 - p))

def plot_logistic_regression(
    csv_path: str,
    *,
    # column names (rename here)
    COL_X: str = "Co-60_eq",           
    COL_DR: str = "DR_10cm_uSv_h",     
    COL_LL: str = "item_LL",           
    COL_MATERIAL: str = "material",    
    material_item_value: str = "item",

    # plot cosmetics
    title: str = "Logistic regression",
    figsize: Tuple[float, float] = (8, 6),
    dpi: int = 120,

    # rules & axes
    fit_on_log10: bool = True,        
    dr_exclude_threshold: float = 0.01, # keep rows with DR < this
    class_rule_is: Tuple[float, float] = (0.0, 1.0),  
    prob_threshold: float = 0.05,      # horizontal dashed line & threshold 
    x_limits: Tuple[float, float] = (1e-3, 3e-1),
    x_ticks: Optional[Sequence[float]] = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1),
    jitter_height: float = 0.02,
) -> Tuple[plt.Figure, plt.Axes, Optional[float]]:


    # ---------- load & filter ----------
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    if COL_MATERIAL not in df.columns:
        raise KeyError(f"Column '{COL_MATERIAL}' not found in CSV.")

    # ITEM rows only
    dfi = df.loc[
        df[COL_MATERIAL].astype(str).str.lower().eq(material_item_value.lower())
    ].copy()

    # coerce to numeric and drop NA
    for c in (COL_X, COL_DR, COL_LL):
        if c not in dfi.columns:
            raise KeyError(f"Column '{c}' not found in CSV.")
        dfi[c] = pd.to_numeric(dfi[c], errors="coerce")
    dfi = dfi.dropna(subset=[COL_X, COL_DR, COL_LL])

    # apply DR filter
    dfi = dfi.loc[dfi[COL_DR] < dr_exclude_threshold].copy()

    if dfi.empty:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.text(0.5, 0.5,
                "No ITEM rows after DR filter",
                ha="center", va="center")
        ax.axis("off")
        return fig, ax, None

    # ---------- labels & predictor ----------
    # classification: 0 if LL < 1, else 1
    y = (dfi[COL_LL] >= 1.0).astype(int).to_numpy()

    X_raw = dfi[COL_X].to_numpy()
    # avoid non-positive values before log10 transform
    X_raw = np.where(X_raw <= 0, np.nan, X_raw)
    mask = np.isfinite(X_raw) & np.isfinite(y)
    X_raw = X_raw[mask]
    y = y[mask]

    if X_raw.size < 3 or len(np.unique(y)) < 2:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.text(0.5, 0.5,
                "Not enough variation to fit logistic regression",
                ha="center", va="center")
        ax.axis("off")
        return fig, ax, None

    X_feat = np.log10(X_raw).reshape(-1, 1) if fit_on_log10 else X_raw.reshape(-1, 1)

    # ---------- fit logistic ----------
    
    lr = LogisticRegression(solver="lbfgs", max_iter=1000, class_weight="balanced")
    lr.fit(X_feat, y)

    # ---------- compute x at target probability ----------
    beta0 = float(lr.intercept_[0])
    beta1 = float(lr.coef_[0, 0])
    z_star = _logit(prob_threshold)
    
    x_thr = 10 ** ((z_star - beta0) / beta1) if fit_on_log10 else (z_star - beta0) / beta1
    x_thr = float(x_thr) if np.isfinite(x_thr) else None

    # ---------- prediction curve ----------
    x_grid = np.linspace(x_limits[0], x_limits[1], 500)
    x_grid_feat = (np.log10(x_grid).reshape(-1, 1)
                   if fit_on_log10 else x_grid.reshape(-1, 1))
    p_grid = lr.predict_proba(x_grid_feat)[:, 1]

    # ---------- plot ----------
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # background 0/1 guidelines
    ax.axhline(class_rule_is[0], color="black", lw=0.6)
    ax.axhline(class_rule_is[1], color="black", lw=0.6)

    # jittered true labels
    rng = np.random.default_rng(42)
    y_jit = y + rng.uniform(-jitter_height, jitter_height, size=y.shape)
    ax.scatter(X_raw, y_jit, s=8, color="navy", alpha=0.5)

    # logistic curve
    ax.plot(x_grid, p_grid, color="darkorange", lw=2)
    ax.text(x_limits[0]*1.6, 0.5, "Logistic fit", color="darkorange")

    # dashed guides
    ax.axhline(prob_threshold, color="blue", ls="--", lw=1)
    ax.text(x_limits[1]*0.82, prob_threshold + 0.03, f"P = {prob_threshold:0.03f}",
            color="blue")

    if (x_thr is not None) and (x_limits[0] < x_thr < x_limits[1]):
        ax.axvline(x_thr, color="blue", ls="--", lw=1)
        ax.text(x_thr * 1.05, class_rule_is[0] - 0.05,
                f"{x_thr:0.003f} Bq/g", color="blue")

    # axes 
    ax.set_xlim(*x_limits)
    ax.set_xlabel("Co-60 equivalent activity [Bq/g]")
    ax.set_ylabel("Probability/Classification")
    ax.set_title(title)

    if x_ticks is not None:
        ax.set_xticks(list(x_ticks))
        ax.set_xticklabels([f"{t:g}" for t in x_ticks])

    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, axis="both", alpha=0.15)

    fig.tight_layout()
    return fig, ax, x_thr
