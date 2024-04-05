# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: ol-connectome
#     language: python
#     name: ol-connectome
# ---

# %% [markdown]
# ### Plot the No. of cells in each medulla column for several cell types

# %%
from pathlib import Path
import sys

from dotenv import load_dotenv, find_dotenv
load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath('src')))
print(f"Project root directory: {PROJECT_ROOT}")
import numpy as np
import plotly.graph_objects as go
import pandas as pd

# load some helper functions
from utils.hex_hex import \
    hex_to_bids\
  , get_hex_df

# %%
# directory to save results
result_dir = PROJECT_ROOT / 'results' / 'fig_ME_col_occupancy'
result_dir.mkdir(parents=True, exist_ok=True)

# %%
# color palette
p_col = [
    'rgb(229,255,204)', 'rgb(204,255,153)', 'rgb(178,255,102)'
  , 'rgb(204,255,204)', 'rgb(153,255,153)', 'rgb(204,255,229)'
  , 'rgb(153,255,204)', 'rgb(204,255,255)', 'rgb(  0, 51,102)'
  , 'rgb(153,255,255)', 'rgb( 51,255,255)', 'rgb(  0,204,204)'
  , 'rgb(153,204,255)', 'rgb(153,255,255)', 'rgb(204,255,255)'
  , 'rgb(  0,  0,  0)']
HEX_COLOR = 'rgb(240,240,240)'
HEX_LINE_COLOR = 'rgb(170,170,170)'
EDGE_COLOR = 'rgb(0,0,0)'

MARKER_SYMBOL = 2
HEX_MARKER_SIZE = 22 #20
HEX_LINE_WIDTH = 0.4

CELL_LINE_WIDTH = 0.2

# %%
# coordinates within the hex column for different cell types
COORDINATE_FACTOR = 2.333 # number derived based on a bunch of heuristics
add_crds = np.multiply(
    COORDINATE_FACTOR
  , [   (0,1.5), (-0.5,1), (0.5,1), (-1,0.5), (0,0.5), (1,0.5), (-1.5,0)
      , (-0.5,0), (0.5,0), (1.5,0), (-1,-0.5), (0,-0.5), (1,-0.5), (-0.5,-1)
      , (0.5,-1), (0,-1.5)])

# %%
# assign within-column coordinates "ct_add_crds" and color "ct_cols" to cell types

hex_df = get_hex_df()
all_ct = hex_df.columns
ct_add_crds = {}
ct_cols = {}
for key, val in zip(all_ct[2:], add_crds):
    ct_add_crds[key] = val

for key, col in zip(all_ct[2:], p_col[:len(all_ct[2:])]):
    ct_cols[key] = col


# %%
# make a df with original index in hex_df, cell type,
#   x and y coordinates, color and size
plot_df = pd.DataFrame({
        'orig_idx':0
      , 'cell_type':'0'
      , 'x_crds':0, 'y_crds':0
      , 'col':p_col[0]
      , 'size':0}
  , index=[0])

MUL_FAC = 5 # multiply the hex coordinates by this factor to scale up the figure
CELL_SIZE = 5

# iterate over all hexes
for df_idx, temp_row in hex_df.iterrows():
    temp_h1 =temp_row['hex1_id']
    temp_h2 =temp_row['hex2_id']
    row_ct_dict = hex_to_bids((temp_h1, temp_h2)) # get cell types at this hex

    # iterate over cell types at this hex coord and add them if there is at
    #   least one cell of that cell type present
    for ct_k, ct_bid in row_ct_dict.items():
        if len(ct_bid) >= 1:
            ctk_tot_crds = np.multiply(
                    [temp_h1 - temp_h2, temp_h1 + temp_h2]
                  , MUL_FAC)\
              + ct_add_crds[ct_k]

            plot_df.loc[len(plot_df.index)] = [
                df_idx
              , ct_k
              , ctk_tot_crds[0], ctk_tot_crds[1]
              , ct_cols[ct_k]
              , CELL_SIZE]

# %%
tot_max = np.multiply(
    [hex_df['hex1_id'].max() + hex_df['hex2_id'].max()]
  , MUL_FAC) # max x and y coordinates
tot_min = np.multiply(
    [hex_df['hex1_id'].min() - hex_df['hex2_id'].max()]
  , MUL_FAC) # min x and y coordinates

# plot columns as disks
fig = go.Figure(data=go.Scatter(
        x=(hex_df['hex1_id'] - hex_df['hex2_id']).multiply(MUL_FAC)
      , y=(hex_df['hex1_id'] + hex_df['hex2_id']).multiply(MUL_FAC)
      , mode='markers'
      , marker={
            'size':HEX_MARKER_SIZE
          , 'symbol': MARKER_SYMBOL
          , 'color': HEX_COLOR
          , 'line': {'width':HEX_LINE_WIDTH, 'color':EDGE_COLOR}
            }
      , showlegend=False
))

# plot cell types as disks
fig.add_trace(
    go.Scatter(
        x=plot_df['x_crds']
      , y=plot_df['y_crds']
      , mode='markers'
      , marker={
            'symbol': MARKER_SYMBOL
          , 'size': plot_df['size']
          , 'color': plot_df['col']
          , 'line': {'width':CELL_LINE_WIDTH, 'color':plot_df['col']}
            }
      , showlegend=False
))

# legend
for ct_name, ct_col in ct_cols.items():
    fig.add_trace(
      go.Scatter(
            x=[ct_add_crds[ct_name][0] * 1.5*MUL_FAC] + tot_min/2
          , y=[ct_add_crds[ct_name][1] * 3*MUL_FAC] + tot_max
          , mode='markers+text'
          , name=ct_name
          , text = ct_name
          , textposition = 'top center'
          , marker={
                'size':CELL_SIZE
              , 'color':ct_col
              , 'line': {'width':0.5, 'color':EDGE_COLOR}
            }
          , showlegend=False
    ))

fig.update_layout(
    title='Column configuration'
  , width= 800
  , height= 1600
  , paper_bgcolor='rgba(0,0,0,0)'
  , plot_bgcolor='rgba(0,0,0,0)'
)

# set aspect ratio to 1
fig.update_xaxes(scaleanchor="y", scaleratio=1)
fig.update_yaxes(scaleanchor="x", scaleratio=1)

# remove grid and tick labels
fig.update_xaxes(showgrid=False, showticklabels=False)
fig.update_yaxes(showgrid=False, showticklabels=False)

fig.show()

# %%
# save figure
# fig.write_image(result_dir / 'column_celltypes_v2.pdf')
