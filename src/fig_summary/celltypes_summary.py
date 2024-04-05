# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Celltype summary
#
# generate the Summary Group pdf and svg files.
#
# This notebook is mostly for debugging, use the snakemake rule for parallel execution.

# %%
# %load_ext autoreload

from pathlib import Path
import sys

from dotenv import load_dotenv, find_dotenv
load_dotenv()
PROJECT_ROOT = Path(
    find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath('src')))
print(f"Project root directory: {PROJECT_ROOT}")

from utils import olc_client

c = olc_client.connect(verbose=True)


# %%
# %autoreload 2
from neurontype_summary import cli

# %% [markdown]
# # Usage
#
# `neurontype_summary` is a command line tool that supports 3 commands: `count`, `get` and `plot`.
#
# You can access a full list of commands from Jupyter via `cli([], standalone_mode=False)` and from the command line via `python src/fig_summary/neurontype_summary.py`
#
# `count` returns the total number of groups
#
# `get X` returns the list of items in that group, with X being a number >=0 and <= `count -1`.
#
# `plot X` generates a plot for group X. This can take a long time.
#
# Having this as a command line tool allows us, to run all the different groups in parallel, for example via `snakemake`. 
#
# If you want to see what is in one of the groups, for example group 0, you can run this:
#
# ```python
# cli(["get", "0"], standalone_mode=False)
# ```

# %%
per_page = 24

n_groups = cli(["count", f"--per-page={per_page}"], standalone_mode=False)

for group_num in range(0, n_groups):
    cli(["plot", f"--per-page={per_page}", f"{group_num}"], standalone_mode=False)

# %%
# cli(["plot", "--per-page=24", f"16"], standalone_mode=False)
