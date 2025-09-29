
# Run this in your Anaconda Prompt

git clone https://github.com/MarkusHelej/Jupyter_pyviz_module.git
cd Jupyter_pyviz_module # where lives in your directory
conda env create -f environment.yml
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
