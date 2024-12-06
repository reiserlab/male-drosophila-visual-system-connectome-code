# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: default
#     language: python
#     name: python3
# ---

# %%
from pathlib import Path
import sys
from dotenv import load_dotenv, find_dotenv
import pandas as pd

load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath("src")))

from utils.completeness_plots import generate_completeness_plots

from utils import olc_client

c = olc_client.connect(verbose=True)

# %% [markdown]
# Generate hexagonal heatmap plots and scatter plots of the synapses / connection completeness for one of the three main optic lobe neuropils.  

# %%
# Expected run time to generate all 3 pickle files: ~17 minutes.

completeness_plots = []

for roi_str in ["ME(R)", "LO(R)", "LOP(R)"]:
    completeness_plots.extend(generate_completeness_plots(roi_str=roi_str))

# %%
for plot in completeness_plots:
    display(plot)

# %%
