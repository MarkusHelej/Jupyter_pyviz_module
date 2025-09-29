
# Run this in your Anaconda Prompt

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

## Usage

This project is designed to be used directly from Jupyter notebooks without requiring installation as a Python package.

Keep the source code in the `src/` folder and make it available inside notebooks by updating `sys.path`.  
This ensures that any changes you make in `src/` are picked up immediately without reinstalling anything.

### Setup

Open `analysis.ipynb` in JupyterLab.

1. Run the first cell to install required dependencies (uncomment if needed).
2. Run the second cell to load the module from `src/`.

After that you can run any of the plotting cells.
