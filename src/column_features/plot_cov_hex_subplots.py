# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
"""
This cell does the initial project setup.
If you start a new script or notebook, make sure to copy & paste this part.

A script with this code uses the location of the `.env` file as the anchor for
the whole project (= PROJECT_ROOT). Afterwards, code inside the `src` directory
are available for import.
"""

from pathlib import Path
import sys

from dotenv import load_dotenv, find_dotenv

load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath("src")))

from utils import olc_client
c = olc_client.connect(verbose=True)

from utils.trim_helper import TrimHelper
from utils.column_plotting_functions import plot_per_col_subplot

# %% [markdown]
# Create a subplot (2 x 3) of the number of unique cells of the TmY4 and Dm3c subtypes within the medulla, lobula and lobula plate as hexagonal 'eyemap' heatplots. 

# %% metadata={}
# set formatting parameters
style = {
    "font_type": "arial"
  , "markerlinecolor": "rgba(0,0,0,0)"
  , "linecolor": "black"
}

sizing = {
    "fig_width": 48  # units = mm, max 180
  , "fig_height": 28  # units = mm, max 170
  , "fig_margin": 0
  , "fsize_ticks_pt": 5
  , "fsize_title_pt": 5
  , "markersize": 1.3175
  , "ticklen": 1.445
  , "tickwidth": 0.425
  , "axislinewidth": 0.51
  , "markerlinewidth": 0.0425
  , "cbar_thickness": 3
  , "cbar_len": 0.8
}  

# %% metadata={}
instance = 'Dm3c_R'
trim_helper_dm3c = TrimHelper(instance)
trim_df_dm3 = trim_helper_dm3c.trim_df

# %% metadata={}
instance = 'TmY4_R'
trim_helper_tmy4 = TrimHelper(instance)
trim_df_tmy4 = trim_helper_tmy4.trim_df

# %% metadata={}
plot_specs = {
    "filename": 'TmY4_Dm3c_subplot_old'
  , "cmax_cells": 10
  , "cmax_syn": 0
  , "cbar_title": '# cells/<br>column'
  , "export_type": "pdf"
  , "cbar_title_x": 1.17
  , "cbar_title_y": 0.06
}

plot_per_col_subplot(trim_df_dm3, trim_df_tmy4, style, sizing, plot_specs)
