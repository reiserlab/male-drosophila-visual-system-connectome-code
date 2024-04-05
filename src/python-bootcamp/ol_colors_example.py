# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3.11.5 ('ol-connectome')
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## notebook explaining how to use the ol_color.py 

# %%
# %load_ext autoreload
"""
This cell does the initial project setup.
"""
from pathlib import Path
import sys
import pandas as pd

import plotly.graph_objects as go

from dotenv import load_dotenv, find_dotenv
load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath('src')))
print(f"Project root directory: {PROJECT_ROOT}")

from neuprint import fetch_neurons, NeuronCriteria as NC

from utils import olc_client
c = olc_client.connect(verbose=True)

# %% [markdown]
# ### accessing colors from `ol_color.py`

# %%
# import this to get our optic lobe color palette
from utils.ol_color import OL_COLOR 

# %%
"""
'OL_COLOR.OL_TYPES' retrieves the color palette corresponding to the 5 groupings of cell-types ('OL_intrinsic', 'OL_connecting','VPN','VCN','other (central)')
Similarly, if you wanted to color by neuropil instead of groups, you would use 'OL_COLOR.OL_NEUROPIL'. Depending on whether you need a hex or rgb, that can be
retrieved by simply saying 'OL_COLOR.OL_TYPES.rgb' or 'OL_COLOR.OL_TYPES.hex'

Plotly uses the rgb format.
"""
print(f"colors in hex :{OL_COLOR.OL_TYPES.hex}")
print(f"3rd color in hex: {OL_COLOR.OL_TYPES.hex[2]}")

print(f"colors in plotly format : {OL_COLOR.OL_TYPES.rgb}")
print(f"3rd color in plotly format: {OL_COLOR.OL_TYPES.rgb[2]}")

print(f"3rd color only as rgb: {OL_COLOR.OL_TYPES.rgb[2][1]}")

print(f"3rd color only as rgba: {OL_COLOR.OL_TYPES.rgba[2]}")

# %%
"""
To also look at the color palette, simply use 'OL_COLOR.OL_TYPES.cmap'

(quick preview, yet still not fixing the problem from https://github.com/reiserlab/optic-lobe-connectome/pull/273#issuecomment-1846128791)
"""
OL_COLOR.OL_TYPES.cmap

# %%
"""
    To also look at the dictionary of color assignments, simply use 'OL_COLOR.OL_TYPES.map'
"""
print(f"Dictionary of colors: {OL_COLOR.OL_TYPES.map}")
print(f"Color with name 'intrinsic': {OL_COLOR.OL_TYPES.map['intrinsic']}")

# %% [markdown]
# ### example code to plot number of cells per cell-type (from the new master list) grouped by OL cell-type groupings

# %%
data_dir = PROJECT_ROOT / 'params' 
ol_df = pd.read_excel(data_dir / 'Primary_cell_type_table.xlsx')
# subset of dataframe only belonging to 5 groups
df = ol_df.loc[((ol_df['main_groups'].eq('OL_intrinsic')) | (ol_df['main_groups'].eq('OL_connecting')) | (ol_df['main_groups'].eq('VPN')) | (ol_df['main_groups'].eq('VCN')) | (ol_df['main_groups'].eq('other (central)')))]

ol_types = df['type'].values 

# %%
# fetch neurons belong to the cell-types
neurons_df,roi_counts_df = fetch_neurons(NC(type=ol_types))

# %%
# getting number of cells per cell-type
ncells_df = neurons_df.groupby('type')['bodyId'].nunique().reset_index(name='n_cells')
ncells_sorted_df = ncells_df.sort_values(by='n_cells',ascending=False)
ncells_sorted_df.columns = ['type','n_cells']
ncells_sorted_df = ncells_sorted_df.reset_index(drop=True)

# %%
# merging to get the group information for every cell-type
ncells_sorted_grouped_df = pd.merge(ncells_sorted_df,df[['type','main_groups']],left_on='type',right_on='type')
# getting all the groups
main_groups = ncells_sorted_grouped_df['main_groups'].unique()
main_groups = main_groups.tolist()


# %% [markdown]
# ### Adding colors to the dataframe to color by group (in this case, color by OL cell-type groupings)

# %%
# function to add the color column to the dataframe
def add_color_group(df:pd.DataFrame, main_groups:list, colors:list): 

    for index, row in df.iterrows():
        group = row['main_groups']
        if group in main_groups[0]: 
            grp = 1
            col = colors[0]
        elif group in main_groups[1]:
            grp = 2
            col = colors[1]
        elif group in main_groups[2]:
            grp = 3
            col = colors[2]
        elif group in main_groups[3]:
            grp = 4
            col = colors[3]
        elif group in main_groups[4]:
            grp = 5
            col = colors[4]
        else:
            grp = 0
            col = colors[5]

        row['color'] = col
        df.loc[index, 'color']= col
        row['group'] = grp
        df.loc[index, 'group']= grp

    df['color'].astype(dtype='object') 
    df['group'].astype(dtype='object') 

    return df



# %%
ncells_sorted_grouped_df

# %%
# inserting the color column into your dataframe
df_colored = add_color_group(
    ncells_sorted_grouped_df
  , main_groups
  , OL_COLOR.OL_TYPES.hex
)
df_colored

# %% [markdown]
# # plotting
#
# To plot all the data points in one color from `ol_color.py`

# %%
layout = go.Layout(
    paper_bgcolor='rgba(255,255,255,1)'
  , plot_bgcolor='rgba(255,255,255,1)'
)

fig = go.Figure(layout = layout)

fig.add_trace(
    go.Scatter(
        x = df_colored['type']
      , y = df_colored['n_cells']
      , hovertext = df_colored['type']
      , hoverinfo = 'text'
    #   , opacity = 0.3
      , mode='markers'
      , marker={
            'size':10
          , 'color': OL_COLOR.OL_TYPES.rgb[1][1]
          , 'line': {
                'width':1
              , 'color': OL_COLOR.OL_TYPES.rgb[1][1]
            }
        }
    )
)
fig.show()

# %% [markdown]
# To plot all the data points in colors grouped by OL cell-type groupings (colors from `ol_color.py`)

# %%
fig2 = go.Figure(layout = layout)

fig2.add_trace(
    go.Scatter(
        x = df_colored['type']
      , y = df_colored['n_cells']
      , hovertext = df_colored['type']
      , hoverinfo = 'text'
      # , opacity = 0.3
      , mode='markers'
      , marker={'size':10
          ,'color': df_colored['group']
          , 'colorscale': OL_COLOR.OL_TYPES.rgb
          ,'line': {
                'width':1
              , 'color': df_colored['group']
              , 'colorscale': OL_COLOR.OL_TYPES.rgb
            }
        }
    )
)
fig2.show()

# %% [markdown]
# Same plot, but in a ligher color scheme:

# %%
fig2['data'][0]['marker']['colorscale'] = OL_COLOR.OL_LIGHT_TYPES.rgb
fig2.show()

# %% [markdown]
# … or in VPN colors :-)

# %%
fig2['data'][0]['marker']['colorscale'] = OL_COLOR.OL_VPN_SEQ.rgb
fig2.show()

# %%
