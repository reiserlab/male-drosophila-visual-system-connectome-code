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
load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath('src')))

from utils.ROI_plots import plot_pin_assignment\
  , find_max_pin_length\
  , find_max_pin_deviation\
  , find_max_pin_volume\
  , plot_pin_length_subplot\
  , plot_pin_deviation_subplot\
  , plot_pin_volume_subplot\
  , plot_synapses_per_depth

from utils.hex_plot_config import HexPlotConfig

from utils import olc_client
c = olc_client.connect(verbose=True)

# %% [markdown]
# Plot hexagonal 'eyemap' heatplots of quantifiable features of the columns in the Medulla (ME), Lobula (LO) and Lobula Plate (LOP). 
#
# The plots are saved as PDFs in the folder `PROJECT_ROOT/results/eyemap`.

# %%
cfg = HexPlotConfig()

# %%
# set formatting parameters
sizing = {
    'fig_width': 95
  , 'fig_height': 25
  , 'fig_margin': 0
  , 'fsize_ticks_pt': 5
  , 'fsize_title_pt': 6
  , 'markersize': 2
  , 'ticklen': 1.7
  , 'tickwidth': 0.5
  , 'axislinewidth': 0.6
  , 'markerlinewidth': 0.2
  , 'cbar_len': 1
  , 'cbar_thickness': 7
}

plot_specs = {
    'export_type':'pdf'
}

# %% [markdown]
# Plot the length of each column.

# %%
plot_specs['cmax'] = find_max_pin_length()
fig = plot_pin_length_subplot(cfg.style, sizing, plot_specs)
fig.show()

# %% [markdown]
# Plot the deviation of the column from a straight line.

# %%
plot_specs['cmax'] = find_max_pin_deviation()
fig = plot_pin_deviation_subplot(cfg.style, sizing, plot_specs)
fig.show()

# %% [markdown]
# Plot the volume of each column.

# %%
plot_specs['cmax'] = find_max_pin_volume()
fig = plot_pin_volume_subplot(cfg.style, sizing, plot_specs)
fig.show()

# %% [markdown]
# Plot the number of `pre` and `post` synapses for all cells from all cell types that innervate the right optic lobe across the depths of the columns, within each main optic lobe region (ME, LO, LOP).

# %%
# set formatting parameters
sizing = {
    'fig_width': 95
  , 'fig_height': 70
  , 'fig_margin': 0
  , 'fsize_ticks_pt': 6
  , 'fsize_title_pt': 7
  , 'markersize': 2
  , 'ticklen': 2
  , 'tickwidth': 1
  , 'axislinewidth': 0.6
  , 'markerlinewidth': 1.1
}

plot_specs = {
    'export_type':'pdf'
}

# %%
fig = plot_synapses_per_depth(
    cfg.style
  , sizing
  , plot_specs
)
fig.show()

# %% [markdown]
# Create plots showing synapse assignments to columns (for specific cell-types depending on the neuropil).

# %%
plot_pin_assignment()
