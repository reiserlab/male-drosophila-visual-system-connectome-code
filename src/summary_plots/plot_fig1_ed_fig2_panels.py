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

from utils.ol_types import OLTypes
from utils.overall_summary_queries import \
    add_color_group\
  , make_ncell_nconn_nsyn_data\
  , make_ncell_nconn_data\
  , make_connectivity_sufficiency_data
from utils.overall_summary_plotting_functions import \
    plot_ncells_nsyn_linked\
  , plot_summary_scatterplots\
  , make_circles_celltype_groups\
  , make_neuropil_celltype_groups_panel\
  , make_connectivity_sufficiency_scatter

c = olc_client.connect(verbose=True)


# %%
ol = OLTypes()
types = ol.get_neuron_list(side='both')
cell_instances = types['instance']

# %% [markdown]
# # Counting cells and connections
#
# The following plot counts the number of cells and connections for the top N cell types.
# Specifically, the plot consists of
#
# - Number of cells per cell type
# - Number of pre and post synapses per cell type
# - Cumulative fraction of number of cells
# - Cumulative fraction of number of synapses
#
# This plot is used in the manuscript Fig. 1d

# %%
ncell_nconn_nsyn_df = make_ncell_nconn_nsyn_data(cell_instances)\
    .reset_index()

# set formatting parameters
style = {
    'export_type': 'svg'
  , 'font_type': 'arial'
  , 'markerlinecolor': 'black'
  , 'linecolor': 'black'
  , 'opacity': 0.7
}

sizing = {
    'fig_width': 300
  , 'fig_height': 100
  , 'fig_margin': 0
  , 'fsize_ticks_pt': 5
  , 'fsize_title_pt': 7
  , 'markersize': 5
  , 'ticklen': 3.5
  , 'tickwidth': 0.1
  , 'axislinewidth': 1
  , 'markerlinewidth': 1
}

plot_specs = {
    'grouping': 1
  , 'n_celltypes_to_plot': 160
  , 'tickvals_y1': [0, 2, 4]
  , 'ticktext_y1': [1, 100, 10000]
  , 'range_y2': [-2250000, 750000]
  , 'tickvals_y2': [-2250000, -1500000, -750000, 0, 750000]
  , 'range_y4': [0,1]
  , 'tickvals_y4': [0, 0.25, 0.5, 0.75, 1]
  , 'plot_name': 'ncells_nconn_linked_plot'
  , 'save_path': PROJECT_ROOT / 'results' / 'summary_plots'
  , 'export_type': 'svg'
}

df = add_color_group(ncell_nconn_nsyn_df)

# if you want to only consider the 4 groups without 'other':
df = df[df['main_group'] != 'other']

fig = plot_ncells_nsyn_linked(
    df=df
  , xval='type'
  , yval1='n_cells'
  , yval2='upstream'
  , yval3='downstream'
  , yval4='cum_cell'
  , yval5='cum_syn'
  , style=style
  , sizing=sizing
  , plot_specs=plot_specs
)

fig.show()

# %% [markdown]
# # Counting types, cells, and connections
#
# Bubble chart where the size of the circle represents the relative size within
# a characteristic such as cell type count, cell count, upstream count, or downstream count
# within one of the four cell type groups (and "other")
#
# This plot is used in Fig. 1e

# %%
style = {
    'export_type': 'svg'
  , 'font_type': 'arial'
  , 'markerlinecolor': 'black'
  , 'linecolor': 'black'
  , 'fillcolor': 'white'
}

sizing = {
    'fig_width': 100
  , 'fig_height': 400
}

# %%
ref_circle_areas=[1000, 1000, 1000, 1000]
plot_specs = {
    'save_path': PROJECT_ROOT / 'results' / 'summary_plots'
  , 'export_type': 'svg'
  , 'plot_name': 'celltype_groups_circles'
}

fig = make_circles_celltype_groups(
    types=types
  , ref_circle_areas=ref_circle_areas
  , plot_specs=plot_specs
  , style=style
  , sizing=sizing
)

fig.show()

# %% [markdown]
# Bubble chart representing cell type count, cell count, number of input connections,
# and number of output connections and a pie chart of the main contributing
# cell type groups organized by neuropil.
#
# This is used in Fig. 1f

# %%
plot_specs = {
    'save_path': PROJECT_ROOT / 'results' / 'summary_plots'
  , 'export_type': 'svg'
  , 'plot_name': 'neuropil_groups_piecharts'
}

ref_circle_areas=[1000, 1000, 1000, 1000]
threshold = 0.02

fig = make_neuropil_celltype_groups_panel(
    types=types
  , threshold=threshold
  , ref_circle_areas=ref_circle_areas
  , plot_specs=plot_specs
  , style=style
  , sizing=sizing
)


fig.show()

# %% [markdown]
# # Connections
#
# Number of input cells vs output cells per cell type, colored by cell type group.
#
# This is used in Fig. 1g

# %%
summary_df = make_ncell_nconn_data(cell_instances)

#set formatting parameters
style = {
    'export_type':'svg'
  , 'font_type': 'arial'
  , 'markerlinecolor':'black'
  , 'linecolor':'black'
  , 'opacity':0.95
}

sizing = {
    'fig_width': 50
  , 'fig_height': 50
  , 'fig_margin': 0
  , 'fsize_ticks_pt': 5
  , 'fsize_title_pt': 5
  , 'markersize': 3.5
  , 'ticklen': 2
  , 'tickwidth': 0.7
  , 'axislinewidth': 0.65
  , 'markerlinewidth': 0.07
}

plot_specs = {
    'log_x': True
  , 'log_y': True
  , 'range_x': [-0.301, 4.31]
  , 'range_y': [-0.301, 4.31]
  , 'tickvals_x': [1, 10, 100, 1000, 10000, 20000]
  , 'tickvals_y': [1, 10, 100, 1000, 10000, 20000]
  , 'xlabel': 'number of connected output cells'
  , 'ylabel': 'number of connected input cells'
  , 'save_path': PROJECT_ROOT / 'results' / 'summary_plots'
  , 'export_type': 'svg'
}

df = add_color_group(summary_df)

fig = plot_summary_scatterplots(
    df=df
  , xval='n_post_mean_conn_cells'
  , yval='n_pre_mean_conn_cells'
  , star_neurons=['Mi1', 'TmY5a', 'LC17', 'LoVC16', 'R1-R6', 'CT1', 'Am1', 'L1']
  , style=style
  , sizing=sizing
  , plot_specs=plot_specs
)

fig.show()

# %% [markdown]
# # number of connected cells (or types) over type size
#
# This is used in Extended Data (ED) Figure 2 Panel a (ED 2a)

# %%
summary_df = make_ncell_nconn_data(cell_instances)

# set formatting parameters
plot_specs = {
    'log_x': True
  , 'log_y': True
  , 'range_x': [-0.301, 4.0]
  , 'range_y': [-0.301, 4.31]
  , 'tickvals_x': [1, 10, 100, 1000, 10000, 20000]
  , 'tickvals_y': [1, 10, 100, 1000, 10000, 20000]
  , 'xlabel': 'number of cells per type'
  , 'ylabel': 'number of connected cells'
  , 'save_path': PROJECT_ROOT / 'results' / 'summary_plots'
  , 'export_type': 'svg'
}

df = add_color_group(summary_df)

fig = plot_summary_scatterplots(
    df=df
  , xval='n_cells'
  , yval='n_mean_conn_cells'
  , star_neurons=['Mi1', 'TmY5a', 'LC17', 'LoVC16', 'R1-R6', 'CT1', 'Am1', 'L1']
  , style=style
  , sizing=sizing
  , plot_specs=plot_specs
)

fig.show()

# %% [markdown]
# # number of connected types vs number of cells per type
#
# This is used in ED 2b

# %%
summary_df = make_ncell_nconn_data(cell_instances)

# set formatting parameters
plot_specs = {
    'log_x': True
  , 'log_y': False
  , 'range_x': [-0.301, 4.0]
  , 'range_y': [-20, 400]
  , 'tickvals_x': [1, 10, 100, 1000, 10000]
  , 'tickvals_y': [0, 200, 400]
  , 'xlabel': 'number of cells per type'
  , 'ylabel': 'number of connected types'
  , 'save_path': PROJECT_ROOT / 'results' / 'summary_plots'
  , 'export_type': 'svg'
}

df = add_color_group(summary_df)

fig = plot_summary_scatterplots(
    df=df
  , xval='n_cells'
  , yval='n_mean_conn_types'
  , star_neurons=['Mi1', 'TmY5a', 'LC17', 'LoVC16', 'R1-R6', 'CT1', 'Am1', 'L1']
  , style=style
  , sizing=sizing
  , plot_specs=plot_specs
)

fig.show()

# %% [markdown]
# # Proportion of unique combination of connections across all types
#
# This is used in Fig. 2i

# %%
unique_combinations_df = make_connectivity_sufficiency_data(
    n_top_connections=5
)


# set formatting parameters
style = {
    'export_type':'svg'
  , 'font_type': 'arial'
  , 'markerlinecolor':'black'
  , 'linecolor':'black'
  , 'opacity':0.95
}

sizing = {
    'fig_width': 50
  , 'fig_height': 100
  , 'fig_margin': 0
  , 'fsize_ticks_pt': 5
  , 'fsize_title_pt': 5
  , 'markersize': 3.5
  , 'ticklen': 2
  , 'tickwidth': 0.7
  , 'axislinewidth': 0.65
  , 'markerlinewidth': 0.07
}

# set formatting parameters
plot_specs = {
    'log_x':False
  , 'log_y':False
  , 'range_x':[0, 5]
  , 'range_y':[0, 1]
  , 'tickvals_x': [1, 2, 3, 4, 5]
  , 'tickvals_y': [0, 1]
  , 'xlabel': 'number of top connections'
  , 'ylabel': 'fraction of unique combinations'
  , 'save_path': PROJECT_ROOT / 'results' / 'summary_plots'
  , 'export_type': 'svg'
  , 'plot_name': 'connectivity_sufficiency'
}

fig = make_connectivity_sufficiency_scatter(
    unique_combinations_df
  , xval='n_connections'
  , yval1='frac_unique_combinations'
  , yval2='frac_unique_combinations_pre'
  , yval3='frac_unique_combinations_post'
  , style=style
  , sizing=sizing
  , plot_specs=plot_specs
)

fig.show()


