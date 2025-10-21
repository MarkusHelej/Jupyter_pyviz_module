
# Run this in your Anaconda Prompt the first time you use it

### 1. Clone the repository
git clone https://github.com/MarkusHelej/Jupyter_pyviz_module.git

### 2. Create and activate a Conda environment
cd Jupyter_pyviz_module # where lives in your directory
conda create -n pyviz python=3.11 -y
conda activate pyviz

### 3. Install JupyterLab
pip install jupyterlab

### 4. Launch JupyterLab
jupyter lab

### 5. After first use, run the pyviz module from Anaconda Prompt
conda activate pyviz
jupyter lab


## Usage

This project is designed to be used directly from Jupyter notebooks without requiring installation as a Python package.

Keep the source code in the `src/` folder and make it available inside notebooks by updating `sys.path`.  
This ensures that any changes you make in `src/` are picked up immediately without reinstalling anything.

### Setup

Open `analysis.ipynb` in JupyterLab.

1. Run the first cell to install required dependencies (uncomment if needed).
2. Run the second cell to load the module from `src/`.

After that you can run any of the plotting cells.

# Jupyter_pyviz_module with Anaconda

## Work flow

*(Optional) bootstrap cell* — one-time pip install … in case a machine is missing numpy/pandas/matplotlib/joypy/scikit-learn.

*Import cell* — makes your local src/ code importable and pulls in the plotting functions.

*Plot cells* — each cell loads data (or a validated model of that data) and draws one specific, pre-canned visualization.

Everything is designed so you don’t install a package; you run the code straight from the src/ folder. That keeps iteration transparent.

## Repository layout

```Jupyter_pyviz_module/
├─ data                    #CSV inputs
├─ notebook/
│  └─ analysis.ipynb               # the Jupyter notebook
└─ src/
   └─ jupyter_pyviz_module/
      ├─ models.py                 # DatasetModel and helpers
      └─ plots/
         ├─ plot_IRAS.py
         ├─ plot_items.py
         ├─ plot_scenarios.py
         ├─ plot_classification.py
         └─ plot_logistic_regression.py
```

There are init.py build in the structure as an preparation for future rebuild to an more robust package. 

**The import cell does two path insertions:**

```import sys, os
sys.path.insert(0, os.path.abspath("../src"))
sys.path.insert(0, os.path.abspath("../src/jupyter_pyviz_module"))
```

1. The first line allows from jupyter_pyviz_module import … (package-style imports if ever build as installable package).

2. The second makes module-level imports like from models import DatasetModel and from plots.plot_IRAS import plot_iras resolve without errors.

Order matters: insert(0, …) puts those directories at the front of Python’s search path, so your local code is found before anything else.

**If you edit code in src/ while the notebook is running, re-run the import cell (or enable %load_ext autoreload / %autoreload 2) to pick up changes.**

The import block (what it brings in):

```from models import DatasetModel
from plots.plot_IRAS import plot_iras
from plots.plot_items import plot_items
from plots.plot_scenarios import plot_scenarios
from plots.plot_classification import plot_classification
from plots.plot_logistic_regression import plot_logistic_regression
```

DatasetModel: a light wrapper around a pandas DataFrame that also records a kind (e.g., msr, itm, scn) inferred from the filename and validates required columns.

plot_ functions*: one focused plot per file. They take either a DatasetModel (for IRAS) or a CSV path (for the others), do checks and cosmetics, and return a Matplotlib Figure (and sometimes Axes) so you can .show() or .savefig() as you like.

Data expectations (what columns each plot needs)

DatasetModel.from_csv(path) infers the kind from the filename and validates columns:

Measurement (_msr)
Required: ["IRAS", "true_IRAS"]
Used by: plot_iras(ds, …) (which expects a DatasetModel with kind "msr")

Items (_itm)
Required: ["item_IRAS", "item_DR_10cm_uSv_h"] (plus others the plot may read)
Used by: plot_items("…_itm.csv") and plot_scenarios("…_itm.csv")

Scenarios (_scn)
Required: ["machine", "position", "beam_p_s"] (example)
Used by: scenario-type plots (your notebook calls plot_scenarios with the items CSV for now)

Classification & logistic regression (Cleric items CSV) expect at least:

- Co-60_eq (x),

- DR_10cm_uSv_h (y),

- item_LL (limit flag),

- material (filter by "item"),

- mass_kg (used for marker alpha/size in classification).

If a CSV is missing columns, you’ll get a clear ValueError listing what’s missing.

What each plot cell does:

**1) IRAS vs ActiWiz**

```ds = DatasetModel.from_csv("../data/reproduce_sherpa_msr.csv")
plot_iras(ds, title="IRAS vs ActiWiz (msr)")
```

Loads and validates the measurement file as a DatasetModel(kind="msr").

Filters to finite, positive pairs (log axes).

Draws scatter of (true_IRAS, IRAS), a 1:1 reference line, and a TF/AW label (ratio summary).

Returns an Axes (so you can tweak labels or save afterward).

**2) Items distributions**
```
plot_items("../data/reproduce_sherpa_mass_itm.csv")

```
Reads the items CSV directly (no DatasetModel needed).

Computes and shows distributions across item characteristics (e.g., mass / volume / density).

Typical kwargs include figsize, show, and savepath (if you want a PNG written automatically).

**3) Scenarios**
```
fig1, fig2 = plot_scenarios("../data/reproduce_sherpa_mass_itm.csv")
fig1.show(); fig2.show()
```

Produces two complementary figures from the same dataset (e.g., by scenario/material breakdowns).

Returns two Matplotlib figures so you can display, save, or further customize each.

**4) Classification scatterplot**
```
fig, ax = plot_classification(
    "../data/Cleric_items_13_May.csv",
    x_axis="Co-60_eq",
    y_axis="DR_10cm_uSv_h",
    COL_LL="item_LL",
    COL_MASS_KG="mass_kg",
    COL_MATERIAL="material",
    material_item_value="item",
    title="Classification scatterplot",
    x_ref=0.04, y_ref=0.01,          # guide lines
    xlim=(0.001, 0.12), ylim=(0.001, 0.2)
)
fig.show()
```

Filters to material == "item" (so you’re truly looking at items).


Optional guide lines (x_ref, y_ref) help you eyeball thresholds.

Fully customizable limits & labels.

**5) Logistic regression (LL probability model)**
```
fig, ax, x_thr = plot_logistic_regression(
    "../data/Cleric_items_13_May.csv",
    COL_X="Co-60_eq",
    COL_DR="DR_10cm_uSv_h",
    COL_LL="item_LL",
    COL_MATERIAL="material",
    material_item_value="item",
    title="Logistic regression",
    dr_exclude_threshold=0.01,  # drop low-dose rows
    prob_threshold=0.05,        # draw P=0.05 line
    x_limits=(1e-3, 3e-1),
    x_ticks=(1e-3, 3e-2, 1e-1, 3e-1),
    jitter_height=0.02,
)
fig.show()
```

Filters and prepares data (excludes very low DR if you choose).

Fits a logistic model (probability of LL vs Co-60_eq), typically on log axes.

Draws the probability curve and a threshold marker at P == prob_threshold (returned as x_thr).

Jitter helps reveal overlapping points.


## Typical use
1. In Anaconda prompt:
    - Activate env → ```conda activate pyviz```

    - Launch → ```jupyter lab```

*This launches Jupyter in your browser*

2. Open notebook/analysis.ipynb

    - Run the import cell (so src/ is on sys.path)

    - Run any plot cell you need

    Tweak parameters (titles, limits, thresholds) as needed

    Save figures (either with savepath args or fig.savefig())

If you change code in src/, just re-run the import cell (or enable %autoreload 2 at the top) so the notebook uses the latest version.

3. Troubleshooting 

    - “No module named models / plots”
You’re likely missing the two sys.path.insert lines or ran the notebook from a different working directory. Keep the notebook under notebook/ and the src/ folder one level up (as your current layout).

    - “Missing required columns …”
The CSV doesn’t match the expected schema (see “Data expectations”). Open the file, check column names (case sensitive), and fix upstream or update the plot call’s column arguments.

    -    Plots don’t reflect new code
Re-run the import cell (or enable %load_ext autoreload + %autoreload 2 once at the top).

    - Wrong kernel
Make sure the selected kernel is the pyviz environment you created.

