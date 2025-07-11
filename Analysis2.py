# %% [markdown]
# Capillary Bridge Force Analysis Notebook

This notebook orchestrates the full analysis by calling your existing scripts via `%run`.

# %% [markdown]
## 1. Setup & Imports

Ensure your working directory is the project root.

# %%
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path.cwd()))

# Load config (must be in project root)
import config

# %% [markdown]
## 2. Define Substrate Lines

Run your original substrate‐line definition script to collect clicks and save interpolated lines. It will use `config.image_dir` and `config.lines_file`.

# %%
%run -i define_substrate_lines.py

# %% [markdown]
## 3. Run Frame‐by‐Frame Analysis & Plot

Invoke your multiprocessing analysis script to generate the Excel workbook and force‑vs‑separation plot. It reads from `config.mask_dir`, `config.lines_file`, and writes to `config.excel_output`.

# %%
%run -i plot_forces_multiprocessing.py

# %% [markdown]
## 4. Display Final Plot

# %%
from IPython.display import Image
plot_path = Path(config.excel_output).with_suffix('.png')
Image(str(plot_path))

# %% [markdown]
### Next Steps

- Inspect the Excel file at `
  `{config.excel_output}` for raw data and results.
- Share this notebook with collaborators; it calls your scripts directly.
