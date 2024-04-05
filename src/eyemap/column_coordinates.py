# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: ol-connectome
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
print(f"Project root directory: {PROJECT_ROOT}")

# %%
import numpy as np
import plotly.graph_objects as go

# %%
# function to load hex grid and equatorial columns.
from utils.hex_hex import get_hex_df

from utils.eyemap_equator import \
    get_columns_with_8pr\
  , get_columns_with_7pr\
  , get_columns_with_6pr

# %%
# load hex dataframe
hex_df = get_hex_df() 

# Load dataframes for 6, 7, and 8 Photo Receptors (PRs)
result_dir = PROJECT_ROOT / 'results' / 'eyemap'

pr8 = get_columns_with_8pr(result_dir)
pr7 = get_columns_with_7pr(result_dir)
pr6 = get_columns_with_6pr(result_dir)

# %%
# plotting settings (cf. "equator_r7r8.ipynb")

COLOR_BACKGROUND = 'rgb(200,200,200)'
COLOR_LINE = 'rgb(0,0,0)'

COLOR_NO_R7R8 = 'rgb(200,200,200)' # Missing R7/R8
COLOR_R7R8 = 'rgb(170,212,106)' # R7 or R8
COLOR_PALE = 'rgb(115,194,251)' # Pale
COLOR_YELLOW = 'rgb(248,213,104)'
COLOR_DRA = 'rgb(190,126,89)' # DRA
COLOR_8PR = 'rgb(141, 160, 203)' # 8 Photo Receptors (PR)
COLOR_7PR = 'rgb(252, 141, 98)' # 7 PR
COLOR_6PR = 'rgb(102, 194, 165)' # 6 PR

DOTSIZE = 15
SYMBOL_NUMBER = 15

# %%
MUL_FAC = 1
tot_max = np.multiply([hex_df['hex1_id'].max() + hex_df['hex2_id'].max()],  MUL_FAC)
tot_min = np.multiply([hex_df['hex1_id'].min() - hex_df['hex2_id'].max()],  MUL_FAC)

fig = go.Figure(
    data=go.Scatter(
        x=(hex_df['hex1_id'] - hex_df['hex2_id'])
      , y=(hex_df['hex1_id'] + hex_df['hex2_id'])
      , hovertemplate = 'q,p  = Hex1, Hex2: %{text}'
      , text = list(zip(hex_df['hex1_id'].tolist(), hex_df['hex2_id'].tolist()))
      , mode='markers'
      , marker_symbol = SYMBOL_NUMBER
      , marker={
            'size':DOTSIZE
          , 'color': COLOR_BACKGROUND
          , 'line': {'width':0.5, 'color':COLOR_BACKGROUND}
        }
      , showlegend=False
    )
)

# 7\8 PRs
fig.add_trace(
    go.Scatter(
        x = pr8['p']
      , y = pr8['q']
      , mode='markers'
      , marker_symbol = SYMBOL_NUMBER
      , marker={
            'size':DOTSIZE-5
          , 'color': COLOR_8PR
          , 'line': {'width':0.5, 'color':COLOR_8PR}
        }
      , name="8 PR"
    )
)

fig.add_trace(
    go.Scatter(
        x = pr7['p']
      , y = pr7['q']
      , mode='markers'
      , marker_symbol = SYMBOL_NUMBER
      , marker={
            'size':DOTSIZE-5
          , 'color': COLOR_7PR
          , 'line': {'width':0.5, 'color':COLOR_7PR}
        }
      , name="7 PR"
    )
)

# equator
eqx = hex_df['hex1_id'] - hex_df['hex2_id']
eqy = hex_df['hex1_id'] + hex_df['hex2_id']
fig.add_trace(
    go.Scatter(
      #   x = pr8[(pr8.hex1_id + pr8.hex2_id == 36) | (pr8.hex1_id + pr8.hex2_id == 37)]['p']
      # , y = pr8[(pr8.hex1_id + pr8.hex2_id == 36) | (pr8.hex1_id + pr8.hex2_id == 37)]['q']
        x = eqx[(eqy == 36) | (eqy == 37)]
      , y = eqy[(eqy == 36) | (eqy == 37)]
      , mode='markers'
      , marker_symbol = SYMBOL_NUMBER
      , marker={
            'size':DOTSIZE-10
          , 'color': COLOR_8PR
          , 'line': {'width':0.5, 'color':"red"}
        }
      , name="eq"
    )
)

# origin
fig.add_trace(
    go.Scatter(
        x = [18-19]
      , y = [18+19]
      , mode='markers'
      , marker_symbol = SYMBOL_NUMBER
      , marker={
            'size':DOTSIZE
          , 'color': 'black'
          , 'line': {'width':0.5, 'color':COLOR_7PR}
        }
      , name="origin" # [Hex1, Hex2] = [18, 19]
    )
)

# add p, q arrows
axes = {
    'q': {'x': [18-19, 17], 'y': [18+19, 55]}
  , 'p': {'x': [-1, -18], 'y': [37, 54]}
  , 'v': {'x': [-1, -1], 'y': [37, 75]}
}

list_of_all_arrows = []
for axk, axv in axes.items():
    # https://plotly.com/python/reference/layout/annotations/
    arrow = go.layout.Annotation({
        'arrowside': "start"
      , 'xref': "x", 'yref': "y"
      , 'x': axv['x'][0], 'y': axv['y'][0]
      , 'showarrow': True
      , 'axref': "x", 'ayref': 'y'
      , 'ax': axv['x'][1], 'ay': axv['y'][1]
      , 'text': axk
      , 'font': {'size': 20}
      , 'arrowhead': 3
      , 'arrowwidth': 1.5
      , 'arrowcolor': 'rgb(255,51,0)'
    })
    list_of_all_arrows.append(arrow)

# # add more arrows 
axes = [
    {'label': '17 v-rows', 'x': [-3, -21], 'y': [45, 45]}
  , {'label': '17 v-rows', 'x': [-1, 17], 'y': [45, 45]}
  , {'label': '16 h-rows', 'x': [-2, -2], 'y': [38, 7]}
  , {'label': '16 h-rows', 'x': [-2, -2], 'y': [40, 71]}
]

list_of_all_arrows2 = []
# for x0,y0,x1,y1,text in zip(x_start, y_start, x_end, y_end, axes):
for axv in axes:
    
    arrow = go.layout.Annotation({
        'arrowside': "end+start"
      , 'xref': "x", 'yref': "y"
      , 'x': axv['x'][0], 'y': axv['y'][0]
      , 'showarrow': True
      , 'axref': "x", 'ayref': "y"
      , 'ax': axv['x'][1], 'ay': axv['y'][1]
      , 'text': axv['label']
      , 'font': {'size': 10}
      , 'arrowhead': 3
      , 'arrowwidth': 1.5
      , 'arrowcolor': 'rgb(0,0,0)'
    })
    list_of_all_arrows2.append(arrow)

fig.update_layout(annotations= list_of_all_arrows + list_of_all_arrows2)


# add some annotations
fig.add_annotation(
    x=-20, y=74
  , text="""
        hex1 → q<br>
        hex2 → p<br>
        origin:<br>
        [hex1, hex2] = [18, 19]<br>
        [p, q] = [0, 0]
    """
  , align="left"
  , showarrow=False
  , font={'size':12}
  , xref="x", yref="y"
)


fig.update_layout(
    title=''
  , yaxis_range=[tot_min , tot_max + tot_max/10]
  , xaxis_range=[tot_min, tot_max + tot_max/10]
  , width=750
  , height=750
  , paper_bgcolor='rgba(0,0,0,0)'
  , plot_bgcolor='rgba(0,0,0,0)'
  , xaxis={'zeroline': False, 'showticklabels': False, 'showgrid': False}
  , yaxis={'zeroline': False, 'showticklabels': False, 'showgrid': False}
)
     
fig.update_scenes(aspectmode="data")

fig.show()

# SAVE
fig.write_image(PROJECT_ROOT / "docs" / "assets" / "column_coord.png")
fig.write_image(PROJECT_ROOT / "docs" / "assets" / "column_coord.pdf")

# %% [markdown]
# ### Choice of origin:
# - Apart from the single column in the rightmost vertical row, there are the same number of vertical rows to the left and right of the vertical row with hexid (x, x+2), eg. (18,20).
# - Apart from the single column in the top horizontal row, there are the same number of horizontal rows above (19, 21) and below (18, 20). 
# - Given the equatorial columns, we'll define the origin as (18, 19), that is, shifting slightly to the front and ventral side from the center given by the 2 obeservations above. 
