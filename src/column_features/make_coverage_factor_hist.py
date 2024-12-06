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
"""
This cell does the initial project setup.
If you start a new script or notebook, make sure to copy & paste this part.

A script with this code uses the location of the `.env` file as the anchor for
the whole project (= PROJECT_ROOT). Afterwards, code inside the `src` directory
are available for import.
"""
import sys
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath('src')))

from utils import olc_client
c = olc_client.connect(verbose=True)

from utils.coverage_metric_functions import plot_coverage_metric_histogram

# %% metadata={}
style = {
    'font_type': 'arial'
  , 'markerlinecolor': 'black'
  , 'linecolor': 'black',
}

# %% metadata={}
sizing = {
    'fig_width': 40
  , 'fig_height': 40
  , 'fig_margin': 0
  , 'fsize_ticks_pt': 5
  , 'fsize_title_pt': 6
  , 'markersize': 2
  , 'ticklen': 2
  , 'tickwidth': 0.7
  , 'axislinewidth': 0.5
  , 'markerlinewidth': 0.07
}

# %% metadata={}
plot_specs = {
    'log_x': 'linear'
  , 'log_y': 'log'
  , 'range_x': [0, 8]
  , 'range_y': [-1, 3]
  , 'save_path': PROJECT_ROOT / 'results' / 'cov_compl'
  , 'tickvals_y': [0.1, 1, 10, 100]
  , 'ticktext_y': ['0', '1', '10', '100']
  , 'tickvals_x': [0, 2, 4, 6, 8]
  , 'x_bin_start': 0.25
  , 'x_bin_end': 15
  , 'x_bin_width': 0.25
  , 'export_type': 'pdf'
}

# %% metadata={}
fig = plot_coverage_metric_histogram(
    style
  , sizing
  , plot_specs
  , metric='coverage_factor_trim'
)
fig.show()
