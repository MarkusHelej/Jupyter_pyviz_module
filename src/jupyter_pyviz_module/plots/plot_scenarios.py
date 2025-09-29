from __future__ import annotations
from typing import Optional, Tuple, Iterable, List

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import joypy  

__all__ = ["plot_scenarios"]

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _fd_bins_linear(x: np.ndarray, min_bins: int = 20) -> np.ndarray:
    """Freedman–Diaconis bin edges on a linear axis (fallback-safe)."""
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.linspace(0.0, 1.0, min_bins)
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    if iqr <= 0:
        return np.linspace(float(x.min()), float(x.max()), min_bins)
    h = 2 * iqr * (x.size ** (-1 / 3))
    if h <= 0:
        return np.linspace(float(x.min()), float(x.max()), min_bins)
    k = max(min_bins, int(math.ceil((x.max() - x.min()) / h)))
    return np.linspace(float(x.min()), float(x.max()), k + 1)


def _kde_or_hist_01(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """KDE (SciPy if available) or histogram curve on `grid`; normalized to max=1."""
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.zeros_like(grid)
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(x)
        y = kde(grid)
    except Exception:
        hist, edges = np.histogram(x, bins=_fd_bins_linear(x), density=True)
        centers = 0.5 * (edges[1:] + edges[:-1])
        if centers.size == 0:
            return np.zeros_like(grid)
        y = np.interp(grid, centers, hist, left=0.0, right=0.0)
    m = float(np.max(y)) if np.size(y) else 0.0
    return y / (m if m else 1.0)


def _barh_with_percent(ax: plt.Axes, series: pd.Series, title: str) -> None:
    """Horizontal bars with right-aligned % labels."""
    s = series.dropna().astype(str).value_counts().sort_values(ascending=True)
    total = float(s.sum()) or 1.0
    ax.barh(s.index, s.values, color="#26828e")
    for y, v in enumerate(s.values):
        pct = 100.0 * v / total
        ax.text(v, y, f"{pct:.1f}%", va="center", ha="left", fontsize=9)
    ax.set_title(title)
    ax.set_xlabel("Count")
    ax.grid(True, axis="x", alpha=0.15, linewidth=0.6)

def _style_ridge_panel(ax: plt.Axes, *, xleft: float | None = None, show_ylabel: bool = False) -> None:
    """Make the ridgeline panel look like ggridges: no frame, light guides."""
    # Hide the box
    for sp in ax.spines.values():
        sp.set_visible(False)
    # Keep only bottom ticks; y ticks/labels
    ax.tick_params(axis="both", which="both", length=3)
    ax.tick_params(axis="y", left=False, labelleft=show_ylabel)
    ax.tick_params(axis="x", top=False, right=False)
    # Subtle vertical grid lines only
    ax.grid(True, axis="x", alpha=0.12, linewidth=0.6)
    ax.grid(False, axis="y")
    
    if xleft is not None:
        xmin, xmax = ax.get_xlim()
        ax.set_xlim(left=xleft, right=xmax)



def _ridgeline_axes(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    value_col: str,
    group_col: str = "machine",
    groups_order: Optional[Iterable[str]] = None,
    log_x: bool = False,
    palette: str = "viridis",
    ridge_height: float = 0.90,   # shrink so ridges overlap & fit
) -> None:
    data = df[[group_col, value_col]].copy()
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    data = data.dropna()
    if data.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return

    # --- ordering: top → bottom ---
    unique_groups = data[group_col].unique().tolist()
    if groups_order:
        order = [g for g in groups_order if g in unique_groups]
    else:
        prefer = ["LHC", "SPS", "PS", "PSB", "Lin4", "Lin2"]
        order = [g for g in prefer if g in unique_groups] or sorted(unique_groups)

    if not order:
        ax.text(0.5, 0.5, "No groups", ha="center", va="center")
        ax.set_axis_off()
        return

    n = len(order)
    y_pos = np.arange(n)[::-1]  # top→bottom


    
    prefer = ["LHC", "SPS", "PS", "PSB", "Lin4", "Lin2"]
    if groups_order is not None:
        prefer = [g for g in groups_order if g in prefer] + [g for g in prefer if g not in groups_order]
    order = [g for g in prefer if g in data[group_col].unique()]
    if not order:
        order = sorted(data[group_col].unique())

    n = len(order)
    
    y_pos = np.arange(n)        # 0,1,2,... (bottom→top)
    y_pos = y_pos[::-1]         # reverse so first label plots at the top

    # x-grid
    x_all = pd.to_numeric(data[value_col], errors="coerce").to_numpy()
    x_all = x_all[np.isfinite(x_all)]

    if log_x:
        pos = x_all[x_all > 0]
        if pos.size == 0:
            ax.text(0.5, 0.5, "No positive values for log scale", ha="center", va="center")
            ax.set_axis_off(); 
            return

        lo = np.log10(pos.min())
        hi = np.log10(pos.max())

        
        pad_hi = 1.0  
        target_hi = max(hi + pad_hi, 15.0)

        x_grid = np.logspace(lo, target_hi, 500)
        kde_grid = np.log10(x_grid)

    else:
        
        xmin = x_all.min()
        xmax = x_all.max()
        span = xmax - xmin if xmax > xmin else 1.0
        xmin_expanded = xmin - span * 0.5   
        xmax_expanded = xmax + span * 0.5
        x_grid = np.linspace(xmin_expanded, xmax_expanded, 400)
        kde_grid = x_grid



    cmap = plt.get_cmap(palette, n)
    for i, g in enumerate(order):
        x = pd.to_numeric(data.loc[data[group_col].eq(g), value_col], errors="coerce").to_numpy()
        x = x[np.isfinite(x)]
        if log_x:
            x = x[x > 0]
            if x.size < 2: 
                continue
            xk = np.log10(x)
        else:
            if x.size < 2:
                continue
            xk = x

        y = _kde_or_hist_01(xk, kde_grid) * ridge_height  # scale height
        base = y_pos[i]
        color = cmap(i)
        ax.fill_between(x_grid, base, base + y, color=color, alpha=0.6, linewidth=0)
        ax.plot(x_grid, base + y, color=color, linewidth=0.9)

    if log_x:
        ax.set_xscale("log")

    # ticks/labels (top→bottom)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(order)

    
    if value_col == "beam_p_s" and log_x:
        cur_left, cur_right = ax.get_xlim()
        ax.set_xlim(left=max(cur_left, 1e-12), right=1e15)  
        ax.margins(x=0.01)


    
    ax.set_ylim(-0.3, (n - 1) + ridge_height + 0.25)
    ax.margins(x=0.02, y=0)  # tiny x padding; y handled by set_ylim
    ax.grid(True, axis="x", alpha=0.15, linewidth=0.6)


# ---------------------------------------------------------------------
#   fig_1 = plot00 / plot02 / plot01
#   fig_2 = plot1  / plot2
# ---------------------------------------------------------------------

def plot_scenarios(
    filename: str,
    *,
    figsize_1: Tuple[float, float] = (12, 10),
    figsize_2: Tuple[float, float] = (12, 6),
    show: bool = True,
    savepath_1: Optional[str] = None,
    savepath_2: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Figure]:

    df = pd.read_csv(filename)

    # Machine labels mapping 
    if "machine" in df.columns:
        mapping = {
            "7000GeV": "LHC", "50MeV": "Lin2", "400GeVc": "SPS",
            "160MeV": "Lin4", "14GeVc": "PS", "1400MeV": "PSB",
        }
        df["machine"] = df["machine"].map(mapping).fillna(df["machine"])

    
    df_items = df[df["material"].eq("item")] if "material" in df.columns else df

    # -------------------------------------------------------------------------
    # FIGURE 1: plot_1 = plot00/plot02/plot01  (stacked rows)
    # -------------------------------------------------------------------------

    fig_1 = plt.figure(figsize=figsize_1, constrained_layout=False)
    gs = GridSpec(3, 3, figure=fig_1,
                height_ratios=[0.9, 1.1, 1.4],  # room for ridges
                hspace=0.35, wspace=0.35)

    # ---------------- plot00 (top row) ----------------
    ax00_1 = fig_1.add_subplot(gs[0, 0])  # Machine
    ax00_2 = fig_1.add_subplot(gs[0, 1])  # Irradiation position
    ax00_3 = fig_1.add_subplot(gs[0, 2])  # empty (patchwork parity)

    # Machine bars 
    if "machine" in df.columns:
        counts = df["machine"].astype(str).value_counts()
        # Draw as horizontal bars with % labels and a capped x-range like coord_cartesian
        xmax = max(1.0, 0.75 * len(df))  
        s = counts.sort_values(ascending=True)
        total = float(counts.sum()) or 1.0
        ax00_1.barh(s.index, s.values, color="#26828e", height=0.6)
        for y, v in enumerate(s.values):
            pct = 100.0 * v / total
            ax00_1.text(min(v + xmax * 0.02, xmax * 0.98), y, f"{pct:.1f}%",
                        va="center", ha="left", fontsize=9,
                        bbox=dict(fc="white", ec="none", alpha=0.9, pad=0.1))
        ax00_1.set_xlim(0, xmax)
        ax00_1.set_title("Machine")
        # theme_no_x_axis
        ax00_1.set_xticks([]); ax00_1.set_xlabel("")
        ax00_1.grid(True, axis="x", alpha=0.12, linewidth=0.6)
    else:
        ax00_1.axis("off"); ax00_1.text(0.5, 0.5, "No 'machine'", ha="center")

    # Position bars (fct_infreq + fct_rev)
    if "position" in df.columns:
        pos_counts = df["position"].astype(str).value_counts()  
        s = pos_counts.sort_values(ascending=True)              # reverse for barh bottom→top
        total = float(pos_counts.sum()) or 1.0
        xmax = max(1.0, 0.75 * len(df))
        ax00_2.barh(s.index, s.values, color="#26828e", height=0.6)
        for y, v in enumerate(s.values):
            pct = 100.0 * v / total
            ax00_2.text(min(v + xmax * 0.02, xmax * 0.98), y, f"{pct:.1f}%",
                        va="center", ha="left", fontsize=9,
                        bbox=dict(fc="white", ec="none", alpha=0.9, pad=0.1))
        ax00_2.set_xlim(0, xmax)
        ax00_2.set_title("Irradiation position")
        # theme_no_x_axis
        ax00_2.set_xticks([]); ax00_2.set_xlabel("")
        ax00_2.grid(True, axis="x", alpha=0.12, linewidth=0.6)
    else:
        ax00_2.axis("off"); ax00_2.text(0.5, 0.5, "No 'position'", ha="center")

    ax00_3.axis("off")  

    # ---------------- plot02 (middle row) ----------------
    ax02_1 = fig_1.add_subplot(gs[1, 0])  # Mass
    ax02_2 = fig_1.add_subplot(gs[1, 1])  # Volume
    ax02_3 = fig_1.add_subplot(gs[1, 2])  # Density

    def _step_hist(ax: plt.Axes, arr: pd.Series, xlabel: str) -> None:
        x = pd.to_numeric(arr, errors="coerce").to_numpy()
        x = x[np.isfinite(x)]
        if x.size == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_xlabel(xlabel); ax.set_ylabel("Number of items"); return
        # Freedman–Diaconis bins, step-outline (geom_step look)
        edges = _fd_bins_linear(x)
        ax.hist(x, bins=edges, histtype="step", edgecolor="black", linewidth=0.8)
        ax.set_xlabel(xlabel); ax.set_ylabel("Number of items")
        ax.grid(True, axis="x", alpha=0.15, linewidth=0.6)

    # Mass [kg]
    _step_hist(ax02_1, df_items.get("mass_kg", pd.Series(dtype=float)), "Mass [kg]")

    # Volume [m3] from cm3
    vol_m3 = pd.to_numeric(df_items.get("volume_cm3", pd.Series(dtype=float)), errors="coerce") / 1_000_000.0
    _step_hist(ax02_2, vol_m3, "Volume [m3]")

    # Density [g/cm3]
    _step_hist(ax02_3, df_items.get("density_g_cm3", pd.Series(dtype=float)), "Density [g/cm3]")

    # ---------------- plot01 (bottom row) ----------------
    # Create the three axes first
    ax01_1 = fig_1.add_subplot(gs[2, 0])  # Beam losses (log)
    ax01_2 = fig_1.add_subplot(gs[2, 1])  # Irradiation time
    ax01_3 = fig_1.add_subplot(gs[2, 2])  # Waiting time

    # Bottom row: ridgelines (Beam losses, Irradiation time, Waiting time)
    order = ["LHC", "SPS", "PS", "PSB", "Lin4", "Lin2"]  # top → bottom

    _ridgeline_axes(ax01_1, df_items, value_col="beam_p_s",
                    group_col="machine", log_x=True, groups_order=order)
    ax01_1.set_ylabel("Machine")
    ax01_1.set_xlabel("Beam losses [p/s]")

    # extend right edge of log scale to 1e15
    ax01_1.set_xlim(right=1e13)

    _style_ridge_panel(ax01_1, xleft=None, show_ylabel=True)  # keep y labels only on the first panel


    _ridgeline_axes(ax01_2, df_items, value_col="irradiation_y",
                    group_col="machine", log_x=False, groups_order=order)
    ax01_2.set_xlabel("Irradiation time [y]")
    _style_ridge_panel(ax01_2, xleft=0.0, show_ylabel=False)  # start at 0, no y labels

    _ridgeline_axes(ax01_3, df_items, value_col="waiting_y",
                    group_col="machine", log_x=False, groups_order=order)
    ax01_3.set_xlabel("Waiting time [y]")
    _style_ridge_panel(ax01_3, xleft=-5.0, show_ylabel=False)  # start at 0, no y labels




    fig_1.tight_layout()

    # -------------------------------------------------------------------------
    # FIGURE 2: stacked layout
    # -------------------------------------------------------------------------
    fig_2 = plt.figure(figsize=(12, 10))  # taller canvas
    gs2 = fig_2.add_gridspec(
        nrows=3, ncols=6,
        height_ratios=[2.2, 1.0, 1.0],  # top big panel + two facet rows
        hspace=0.6, wspace=0.6
    )

    # ----- Top wide panel (density) spans full width -----
    axL = fig_2.add_subplot(gs2[0, :])

    # Pick only the main material percentage columns (0..1 range).

    main_material_order = ["steel", "copper", "aluminum", "lead", "plastic", "wood", "asbestos"]

    exclude_cols = {
        "thickness_cm", "Ag", "volume_cm3", "mass_kg", "density_g_cm3",
        "beam_p_s", "irradiation_y", "waiting_y", "machine", "position", "material"
    }

    candidates = [
        c for c in df_items.columns
        if c not in exclude_cols and not c.endswith("_group") and c in main_material_order
    ]

    main_cols = []
    for c in candidates:
        s = pd.to_numeric(df_items[c], errors="coerce").to_numpy()
        s = s[np.isfinite(s)]
        if s.size >= 2 and s.min() >= 0.0 and s.max() <= 1.0:
            main_cols.append(c)

    if main_cols:
        # keep the given material order, but only those present
        ordered_cols = [m for m in main_material_order if m in main_cols]
        cmap = plt.get_cmap("viridis", len(ordered_cols))
        xgrid = np.linspace(0.0, 1.0, 400)

        for i, c in enumerate(ordered_cols):
            x = pd.to_numeric(df_items[c], errors="coerce").to_numpy()
            x = x[np.isfinite(x)]
            y = _kde_or_hist_01(x, xgrid)
            axL.fill_between(xgrid * 100.0, y, step="mid", alpha=0.35, color=cmap(i))
            axL.plot(xgrid * 100.0, y, linewidth=1.4, color=cmap(i), label=c)

        axL.set_xlim(0, 110)
        axL.set_xlabel("Percentage [%]")
        axL.set_ylabel("Percentage distribution [a.u.]")
        axL.set_title("Material")
        axL.grid(True, axis="both", alpha=0.15, linewidth=0.6)
        axL.legend(title="Material", fontsize=9, title_fontsize=10,
                bbox_to_anchor=(1.02, 1.0), loc="upper left", frameon=False)
    else:
        axL.text(0.5, 0.5, "No main material percentage columns in [0,1] found",
                ha="center", va="center")
        axL.axis("off")

    
    # ----- Bottom facets from *_group columns  -----
    type_cols = [c for c in df_items.columns if c.endswith("_group")]
    parts = []
    for c in type_cols:
        mat = c[:-6]  # strip '_group'
        vc = (df_items[c].dropna().astype(str).value_counts(normalize=True) * 100.0)
        if not vc.empty:
            parts.append(pd.DataFrame({"Material": mat, "Type": vc.index, "Percent": vc.values}))

    table = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    # Title above the facet grid
    fig_2.text(0.5, 0.50, "Material Types", ha="center", va="bottom", fontsize=12)

    if not table.empty:
        # preferred order first, then any others
        mats_all = table["Material"].unique().tolist()
        mats_ordered = [m for m in main_material_order if m in mats_all] + \
                    [m for m in sorted(mats_all) if m not in main_material_order]

        n = len(mats_ordered)
        ncols = 3
        nrows = int(math.ceil(n / ncols))

        
        gs_facets = gs2[1:, :].subgridspec(nrows, ncols, hspace=0.55, wspace=0.55)

    for i, m in enumerate(mats_ordered):
        r, c = divmod(i, ncols)
        axF = fig_2.add_subplot(gs_facets[r, c])
        sub = table[table["Material"].eq(m)].sort_values("Percent")
        if sub.empty:
            axF.axis("off"); continue

        axF.barh(sub["Type"], sub["Percent"], color="0.60", height=0.6)
        axF.set_xlim(0, 100)
        axF.grid(True, axis="x", alpha=0.15, linewidth=0.6)
        axF.set_title(m, fontsize=10, pad=2)
        for spine in ("top", "right"):
            axF.spines[spine].set_visible(False)

        
        # set facet xlabel, just global one
        axF.set_xlabel("")




    # single bottom x-label for the whole figure 
    fig_2.text(0.5, 0.04, "Percentage [%]", ha="center", va="bottom", fontsize=11)




    # Save / show
    if savepath_1: fig_1.savefig(savepath_1, dpi=200, bbox_inches="tight")
    if savepath_2: fig_2.savefig(savepath_2, dpi=200, bbox_inches="tight")
    if show:
        plt.show()

    return fig_1, fig_2
