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

from utils.scatterplot_functions import make_scatterplot_with_star_cells
from utils.column_features_helper_functions import find_neuropil_hex_coords
from utils.scatter_plot_config import ScatterConfig

# %% [markdown]
# ### Scatterplots from Fig 5e,f & Extended Data Fig. 10

# %% metadata={}
roi_str = 'ME(R)'
cfg = ScatterConfig(roi_str=roi_str)
_, graph_lims = find_neuropil_hex_coords(roi_str=roi_str)

# %% metadata={}
# population size vs cell size - color by coverage factor

plot_specs = cfg.plot_specs
plot_specs2 = {
    'log_x': True
  , 'log_y': True
  , 'range_x': [-0.3, 3.3]
  , 'range_y': [-0.1, 3]
  , 'cbar_title_x':  1.33
  , 'cbar_title_y': -0.23
}
plot_specs.update(plot_specs2)

make_scatterplot_with_star_cells(
    xval='population_size'
  , yval='cell_size_cols'
  , roi_str=roi_str
  , style=cfg.style
  , sizing=cfg.sizing
  , plot_specs=plot_specs
  , star_neurons=cfg.star_neurons
)

# %% metadata={}
# population columns innervated vs population area covered - color by coverage factor

plot_specs = cfg.plot_specs
plot_specs2 = {
    'log_x': False
  , 'log_y': False
  , 'range_x': [0, graph_lims * 1.05]
  , 'range_y': [0, 1000]
  , 'cbar_title_x': 1.33
  , 'cbar_title_y': -0.23
}
plot_specs.update(plot_specs2)

make_scatterplot_with_star_cells(
    xval='cols_covered_pop'
  , yval='area_covered_pop'
  , roi_str=roi_str
  , style=cfg.style
  , sizing=cfg.sizing
  , plot_specs=plot_specs
  , star_neurons=cfg.star_neurons
)

# %% metadata={}
roi_str = 'LO(R)'
cfg = ScatterConfig(roi_str=roi_str)

# %% metadata={}
# population size vs cell size - color by coverage factor

plot_specs = cfg.plot_specs
plot_specs2 = {
    'log_x': True
  , 'log_y': True
  , 'range_x': [-0.3, 3.3]
  , 'range_y': [-0.1, 3]
  , 'cbar_title_x': 1.28
  , 'cbar_title_y': -0.23
}
plot_specs.update(plot_specs2)

make_scatterplot_with_star_cells(
    xval='population_size'
  , yval='cell_size_cols'
  , roi_str=roi_str
  , style=cfg.style
  , sizing=cfg.sizing
  , plot_specs=plot_specs
  , star_neurons=cfg.star_neurons
)

# %% metadata={}
# population columns innervated vs population area covered - color by coverage factor

plot_specs = cfg.plot_specs
plot_specs2 = {
    'log_x': False
  , 'log_y': False
  , 'range_x': [-10, graph_lims * 1.05]
  , 'range_y': [0, graph_lims * 1.05]
  , 'cbar_title_x': 1.28
  , 'cbar_title_y': -0.23
}
plot_specs.update(plot_specs2)

make_scatterplot_with_star_cells(
    xval='cols_covered_pop'
  , yval='area_covered_pop'
  , roi_str=roi_str
  , style=cfg.style
  , sizing=cfg.sizing
  , plot_specs=plot_specs
  , star_neurons=cfg.star_neurons
)

# %% metadata={}
roi_str = 'LOP(R)'
cfg = ScatterConfig(roi_str=roi_str)

# %% metadata={}
# population size vs cell size - color by coverage factor

plot_specs = cfg.plot_specs
plot_specs2 = {
    'log_x': True
  , 'log_y': True
  , 'range_x': [-0.3, 3.3]  # log range: 10^0=1, 10^3=1000
  , 'range_y': [-0.1, 3]
  , 'cbar_title_x': 1.28
  , 'cbar_title_y': -0.23
}
plot_specs.update(plot_specs2)

make_scatterplot_with_star_cells(
    xval='population_size'
  , yval='cell_size_cols'
  , roi_str=roi_str
  , style=cfg.style
  , sizing=cfg.sizing
  , plot_specs=plot_specs
  , star_neurons=cfg.star_neurons
)

# %% metadata={}
# population columns innervated vs population area covered - color by coverage factor

plot_specs = cfg.plot_specs
plot_specs2 = {
    'log_x': False
  , 'log_y': False
  , 'range_x': [-10, graph_lims * 1.05]
  , 'range_y': [0, graph_lims * 1.05]
  , 'cbar_title_x': 1.28
  , 'cbar_title_y': -0.23
}
plot_specs.update(plot_specs2)

make_scatterplot_with_star_cells(
    xval='cols_covered_pop' 
  , yval='area_covered_pop'
  , roi_str=roi_str
  , style=cfg.style
  , sizing=cfg.sizing
  , plot_specs=plot_specs
  , star_neurons=cfg.star_neurons
)
