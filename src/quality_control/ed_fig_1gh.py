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
import sys
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from IPython.display import display

from neuprint import NeuronCriteria as NC, merge_neuron_properties, NotNull
from neuprint.queries import fetch_adjacencies, fetch_all_rois

from dotenv import load_dotenv, find_dotenv

load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath('src')))
print(f"Project root directory: {PROJECT_ROOT}")

from utils import olc_client
from utils.column_plotting_functions import plot_per_col_simple

# %%
c = olc_client.connect(verbose=True)

data_dir = PROJECT_ROOT / "results" / "quality_control_figure"
cache_dir = PROJECT_ROOT / "cache" / "quality_control_figure"

# %% [markdown]
# # List of Medulla Column ROIs
#
# Generate a list of medulla columns and identify edge columns.
#
# Edge columns here defined as columns missing one or more
# neighbors in a hexagonal grid.

# %%
ME_columns = [x for x in fetch_all_rois() if x.startswith('ME_R_col_')]

column_df = pd.DataFrame({'column': ME_columns})

edge_columns = []
hex1_column = []
hex2_column = []
for column in ME_columns:
    hex1_id, hex2_id = list(map(int, re.findall(r'\d+', column)))
    hex1_column.append(hex1_id)
    hex2_column.append(hex2_id)

    adjacent_columns = []
    for hex1_id_new, hex2_id_new in [
        (hex1_id + 1, hex2_id), (hex1_id - 1, hex2_id)
      , (hex1_id, hex2_id + 1), (hex1_id, hex2_id - 1)
      , (hex1_id + 1, hex2_id + 1), (hex1_id - 1, hex2_id - 1)
    ]:
        new_col_name = f"ME_R_col_{hex1_id_new:02d}_{hex2_id_new:02d}"
        if new_col_name in ME_columns:
            adjacent_columns.append(new_col_name)
    if len(adjacent_columns) < 6:
        edge_columns.append(column)

column_df['hex1_id'] = hex1_column
column_df['hex2_id'] = hex2_column

print(f"Number of Edge columns: {len(edge_columns)}")

display(f"Edge columns: {edge_columns}")

display(column_df)

# %% [markdown]
# # Get downstream synaptic partners
#
# Find all downstream partenrs of all named bodies in ME(R).
# The downstream partners include unnamed bodies.

# %%
neu_fn = cache_dir / "neurons_me_r.pickle"
syn_fn = cache_dir / "synapses_me_r.pickle"

if neu_fn.is_file() and syn_fn.is_file():
    ## load dataframes with connection data (faster than getting these from neuprint
    ## and soon no further changes will be expected for the right optic lobe for now)
    neu = pd.read_pickle(neu_fn)
    syn = pd.read_pickle(syn_fn)
else:
    criteria = NC(type=NotNull, rois=['ME(R)'])
    neu, syn = fetch_adjacencies(criteria, None, include_nonprimary=True) # targets
    ## save dataframes with connection data (reload is faster than getting these from neuprint)
    cache_dir.mkdir(exist_ok=True)
    neu.to_pickle(neu_fn)
    syn.to_pickle(syn_fn)

# %% [markdown]
# ## Group downstream connections
#
# Group connections by presynaptic type and ROI, select ME(R) column ROIs.
#
# Create synthetic groups "R7" and "R8" by grouping all subtyptes together.

# %%
syn_grouped = syn[syn.roi.isin(ME_columns)]\
    .groupby(by=['bodyId_pre', 'roi'], as_index=False)\
    .sum()\
    .merge(neu[['bodyId', 'type']], left_on='bodyId_pre', right_on='bodyId')\
    .drop('bodyId', axis='columns')

syn_table = syn_grouped\
    .groupby(by=['roi', 'type'], as_index=False)\
    .sum(numeric_only=True)\
    .pivot(index='roi', columns='type',values='weight')\
    .fillna(0)\
    .assign(
        R7 = lambda row: row['R7d'] + row['R7p'] + row['R7y'] + row['R7_unclear']
      , R8 = lambda row: row['R8d'] + row['R8p'] + row['R8y'] + row['R8_unclear']
    )\
    .reset_index()


# %% [markdown]
# ## Split group into two
#
# Get group of non-edge columns, split that in two based on R7 output connection

# %%
syn_table_R7_sorted = syn_table[~(syn_table.roi.isin(edge_columns))]\
    .sort_values(by='R7')[['roi', 'R7']]

low_R7_columns_count = len(syn_table_R7_sorted) // 2

low_R7_columns = syn_table_R7_sorted['roi']\
    .head(low_R7_columns_count)\
    .tolist()
other_columns = syn_table_R7_sorted['roi']\
    .tail(len(syn_table_R7_sorted) - low_R7_columns_count)\
    .tolist()

# table with average downstream conections of each cell type in the two groups
two_groups = pd.concat(
    [
        syn_table.set_index('roi').loc[other_columns, :].mean()
      , syn_table.set_index('roi').loc[low_R7_columns, :].mean()
    ]
  , axis=1
)

# %% [markdown]
# ## Compare output synapses
#
# Generate a scatterplot that compares the total output synapses
# per type in the two groups defined above

# %%
# Formatting for plot:
style = {
    'export_type': 'html'
  , 'font_type': 'arial'
  , 'markerlinecolor': 'black'
  , 'linecolor': 'black'
}

sizing = {
    'fig_width': 240    # units = mm
  , 'fig_height': 175   # units = mm
  , 'fig_margin': 0
  , 'fsize_ticks_pt': 18
  , 'fsize_title_pt': 20
  , 'markersize': 9.5
  , 'ticklen': 10
  , 'tickwidth': 3
  , 'axislinewidth': 2.85
  , 'markerlinewidth': 0.35
  , 'cbar_thickness': 8
  , 'cbar_length': 0.95
  , 'cbar_tick_length': 4
  , 'cbar_tick_width': 2
  , 'cbar_title_x': 1.09
  , 'cbar_title_y': -0.2
}

pixelsperinch = 96 # 72 for svg
fsize_ticks_px = sizing['fsize_ticks_pt'] * (1/72) * pixelsperinch

# %% [markdown]
# ## Compare low and high R7 Synapses
#
# select cell types for plotting, criteria used here:
#
# - only cell types with an average of at least 30 synapses per column in at
#     least 600 columns are included
#
# The purpose is to exclude cell types that are only present in a minority of
# columns. For example, TmY19a does not meet current criteria.

# %%
columns_with_synapses = (syn_table.set_index('roi') > 30).sum()
cell_types_with_synapses_in_gt_600_columns = columns_with_synapses[columns_with_synapses>600]\
    .index\
    .tolist()
cell_types_with_synapses_in_gt_600_columns = [
    x for x in cell_types_with_synapses_in_gt_600_columns if (not x in ['index','level_0'])
]

two_groups_rel=two_groups.copy()

two_groups_rel = two_groups_rel.reset_index()

two_groups_rel['group'] = two_groups_rel['type']\
    .apply(lambda x: 1 if x in ['R7','R8','Dm9','Dm14', 'Tm20', 'Mi1'] else 0 )

two_groups_rel = two_groups_rel[
    two_groups_rel.type.isin(cell_types_with_synapses_in_gt_600_columns)
]

two_groups_rel_group1 = two_groups_rel[two_groups_rel.group==0]
two_groups_rel_group2 = two_groups_rel[two_groups_rel.group==1]


# %% [markdown]
# Generate a scatterplot with comparing the total output synapses per type
# in the 'low R7 synapses' and 'high R7 synapses' groups (continued)

# %%
fig = go.Figure()

def _add_name_annotation(fig:go.Figure, type_str:str, rel_gr:pd.DataFrame) -> None:
    """
    Helper function to add several annotations to a figure

    Parameters
    ----------
    fig : go.Figure
        Figure that receives new annotations
    type_str : str
        the neuron type that receives the annotation
    rel_gr : pd.DataFrame
        dataframe that contains the location
    """
    xanchor = 'left'
    match type_str:
        case 'Tm20' | 'Mi1' | 'Dm14':
            xanchor = 'right'
    fig.add_annotation(
        x=np.log10(rel_gr.set_index('type').loc[type_str, 0])
      , y=np.log10(rel_gr.set_index('type').loc[type_str, 1])
      , text=type_str
      , showarrow=True
      , xanchor=xanchor
    )

for val in [1, 1.25, 0.75]:
    fig.add_shape(
        type='line'
      , x0=20
      , y0=20 * val
      , x1=2300
      , y1=2300 * val
      , layer="below"
      , line={'color': 'black', 'width': 0.5, 'dash': 'solid'}
    )

# Group 1 = all other cell types
fig.add_trace(
    go.Scatter(
        x=two_groups_rel_group1[0]
      , y=two_groups_rel_group1[1]
      , mode='markers'
      , marker={
            'size': 15
          , 'color': 'lightgrey'
          , 'line': {
                'width': 0.3
              , 'color': 'black'
            }
          }
      , text=two_groups_rel_group1['type']
      , showlegend = False
    )
)

# Group 2 = types to highlight
fig.add_trace(
    go.Scatter(
        x=two_groups_rel_group2[0]
      , y=two_groups_rel_group2[1]
      , mode='markers'
      , marker={
            'size': 20
          , 'color': 'darkgrey'
          , 'line': {
                'width': 1.5
              , 'color': 'black'
            }
        }
      , text=two_groups_rel_group2['type']
      , showlegend = False
    )
)

for annotate in ['Dm9', 'R7', 'R8', 'Tm20', 'Mi1', 'Dm14']:
    _add_name_annotation(fig, annotate, two_groups_rel_group2)

fig.update_layout(
    yaxis={
        'tickmode': 'array'
      , 'tickvals': [1, 5, 10, 50, 100, 250, 500, 1000]
    }
  , xaxis={
        'tickmode': 'array'
      , 'tickvals': [1, 5, 10, 50, 100, 250, 500, 1000]
    }
  , width=700
  , height=700
  , autosize=False
  , showlegend=False
  , paper_bgcolor='rgba(255,255,255,1)'
  , plot_bgcolor='rgba(255,255,255,1)'
)


fig.update_xaxes(
    range=(np.log10(21), np.log10(2300))
  , autorange=False
  , ticks='outside'
  , tickcolor='black'
  , ticklen=sizing["ticklen"]
  , tickwidth=sizing["tickwidth"]
  , tickfont={
        'size': fsize_ticks_px
      , 'family': style['font_type']
      , 'color': 'black'
    }
  , showgrid=False
  , showline=False
  , linewidth=sizing['axislinewidth']
  , linecolor='black'
  , type="log"
  , title="Average synapses in the top 392 non-edge ME(R) "\
        "columns (ranked by R7 synapses per column)"
)

fig.update_yaxes(
    range=(np.log10(21), np.log10(2300))
  , autorange=False
  , ticks='outside'
  , tickcolor='black'
  , ticklen=sizing["ticklen"]
  , tickwidth=sizing["tickwidth"]
  , tickfont={
        'size': fsize_ticks_px
      , 'family': style['font_type']
      , 'color': 'black'
    }
  , showgrid=False
  , showline=True
  , linewidth=sizing['axislinewidth']
  , linecolor='black'
  , scaleanchor="x"
  , scaleratio=1
  , anchor="free"
  , side="left"
  , type="log"
  , title="Average synapses in the bottom 391 non-edge ME(R) "\
        "columns (ranked by R7 synapses per column)"
)

fig.show()

 # saving the plot (used in ED Fig1g) as .html version (interactive) and pdf
output_dir = data_dir / 'plots'
output_dir.mkdir(exist_ok=True)
fig.write_html(output_dir / "ED_Fig1g.html")
fig.write_image(output_dir / "ED_Fig1g.pdf", width=700, height=700)

# %% [markdown]
# ## Add groups to dataframe
#
# Add column types (low R7 synapses, high R7 synapses, edge)
# to column dataframe (used for the spatial plot in ED Fig. 1g)

# %%
column_df = column_df.set_index('column')

def assign_column_type(column_col):
    """
    Helper function to add color intensity to groups
    """
    column_type = 99            ## Dark
    if column_col in low_R7_columns:
        column_type = 10        ## Light
    if column_col in other_columns:
        column_type=30          ## Medium
    return column_type

column_df['column_type'] = column_df\
    .index\
    .tolist()
column_df['column_type'] = column_df['column_type']\
    .apply(assign_column_type)
column_df['low_R7_column'] = column_df\
    .index.tolist()
column_df['low_R7_column'] = column_df['low_R7_column']\
    .apply(lambda x: 1 if x in low_R7_columns else 0)


display(column_df)


# %% [markdown]
# ## R8 synapse counts
#
# Synapse from R8 and Mi4 to Tm20 by column for plotting
# (used for the spatial plots in ED Fig. 1h)

# %%
cell_type_pre = ['Mi4']
cell_type_post = ['Tm20']
neu, syn = fetch_adjacencies(
    NC(type=cell_type_pre + ['R8p','R8y','R8_unclear','R8d'])
  , NC(type=cell_type_post)
  , include_nonprimary=True
)
table_selected_connections = merge_neuron_properties(neu, syn)
table_selected_connections = table_selected_connections[
    table_selected_connections.roi.isin(ME_columns)
]
table_selected_connections_by_column = table_selected_connections\
    .groupby(by=['type_pre', 'roi'], as_index=False)\
    .sum(numeric_only=True)\
    .pivot(index='roi', columns='type_pre', values='weight')\
    .fillna(0)
table_selected_connections_by_column['R8'] = (
    table_selected_connections_by_column['R8p']
  + table_selected_connections_by_column['R8y']
  + table_selected_connections_by_column['R8d']
  + table_selected_connections_by_column['R8_unclear']
)
table_selected_connections_by_column=table_selected_connections_by_column[['Mi4', 'R8']]
table_selected_connections_by_column.columns =['Mi4_Tm20', 'R8_Tm20']

# table that includes the data used for the plots

table_for_plotting = pd.concat([column_df, table_selected_connections_by_column], axis=1).fillna(0)
table_for_plotting = pd.concat([table_for_plotting, syn_table.set_index('roi')[['R8']]], axis=1)
display(table_for_plotting)


# %% [markdown]
# # Spatial plots
#
# These spatial plots will show  R8 output synapses, R8→Tm20 synapses,
# Mi4→Tm20 synapses and column types (R7 low, R7 high, edge).
# These figures are use in Extended Data Fig. 1g,h

# %%
# plot parameters that are the same for all plot below

style = {
    "font_type": "arial"
  , "markerlinecolor": "rgba(0,0,0,0)" # transparent
  , "linecolor": "black"
}

sizing = {
    "fig_width": 235    # units = mm
  , "fig_height": 215   # units = mm
  , "fig_margin": 0
  , "fsize_ticks_pt": 20
  , "fsize_title_pt": 20
  , "markersize": 17
  , "ticklen": 10
  , "tickwidth": 3
  , "axislinewidth": 3
  , "markerlinewidth": 0.9
  , "cbar_thickness": 18
  , "cbar_len": 0.75
}

# %% [markdown]
# ## R8 output connections
#
# This plot shows all R8 output connections per column.

# %%
plot_specs = {
    "filename": 'ED_Fig_1h_column_map_R8_all_output_connections'
  , "export_type": 'pdf'
  , "column_to_plot": 'R8'
  , "save_path": output_dir
}

fig = plot_per_col_simple(
    df=table_for_plotting
  , roi_str='ME(R)'
  , style=style
  , sizing=sizing
  , plot_specs=plot_specs
  , save_fig=True
)
fig.show()

# %% [markdown]
# ## R8 to Tm20
#
# This plot shows the R8 output connections to Tm20 (per column).

# %%
plot_specs = {
    "filename": 'ED_Fig_1h_column_map_R8_to_Tm20_connections'
  , "export_type": 'pdf'
  , "column_to_plot": 'R8_Tm20'
  , "save_path": output_dir
}

fig = plot_per_col_simple(
    df=table_for_plotting
  , roi_str='ME(R)'
  , style=style
  , sizing=sizing
  , plot_specs=plot_specs
  , save_fig=True
)
fig.show()

# %% [markdown]
# ## Mi4 to Tm20
#
# This plot shows the Mi4 output connections to Tm20 (per column.) This is helpful to
# compare with the R8 to Tm20 plot.

# %%
plot_specs = {
    "filename": 'ED_Fig_1h_column_map_Mi4_to_Tm20_connections'
  , "export_type": 'pdf'
  , "column_to_plot": 'Mi4_Tm20'
  , "save_path": output_dir
}

fig = plot_per_col_simple(
    df=table_for_plotting
  , roi_str='ME(R)'
  , style=style
  , sizing=sizing
  , plot_specs=plot_specs
  , save_fig=True
)
fig.show()

# %% [markdown]
# ## Groups of columns
#
# This plot shows three type of columns:
#
# 1. bright color: low R7 synapses
# 2. medium color: high R7 synapses
# 3. dark color: edge
#
# Edge cells are excluded for most plots.

# %%
plot_specs = {
    "filename": 'ED_Fig_1g_column_map_column_groups_used'
  , "export_type": 'pdf'
  , "column_to_plot": 'column_type'
  , "save_path": output_dir
}

fig = plot_per_col_simple(
    df=table_for_plotting
  , roi_str='ME(R)'
  , style=style
  , sizing=sizing
  , plot_specs=plot_specs
  , save_fig=True
)
fig.show()

# %%
