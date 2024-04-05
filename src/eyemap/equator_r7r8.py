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

# %%
##
# Original author: @PavithraaSeenivasan

from pathlib import Path
import sys

from dotenv import load_dotenv, find_dotenv
load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath('src')))
print(f"Project root directory: {PROJECT_ROOT}")
import numpy as np
import plotly.graph_objects as go

#from cmap import Colormap

# load some helper functions
from utils.hex_hex import get_hex_df

from utils.eyemap_equator import \
    get_columns_without_r7r8\
  , get_columns_with_r7r8\
  , get_columns_with_pale\
  , get_columns_with_yellow\
  , get_columns_with_dra\
  , get_columns_with_8pr\
  , get_columns_with_7pr\
  , get_columns_with_6pr

data_dir = PROJECT_ROOT / 'results' / 'eyemap'
result_dir = data_dir

# %%
# Load dataframes of R7R8 subtype data
no_r7r8 = get_columns_without_r7r8(result_dir)
r7r8 = get_columns_with_r7r8(result_dir)
pale = get_columns_with_pale(result_dir)
yellow = get_columns_with_yellow(result_dir)
dra = get_columns_with_dra(result_dir)

# Load dataframes for 6, 7, and 8 Photo Receptors (PRs)
pr8 = get_columns_with_8pr(result_dir)
pr7 = get_columns_with_7pr(result_dir)
pr6 = get_columns_with_6pr(result_dir)

# %%
# Color and marker information

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
hex_df = get_hex_df()

# %%
# Map with missing R7,R8; R78; pale; yellow; DRA

MUL_FAC = 1
tot_max = np.multiply([hex_df['hex1_id'].max() + hex_df['hex2_id'].max()],  MUL_FAC)
tot_min = np.multiply([hex_df['hex1_id'].min() - hex_df['hex2_id'].max()],  MUL_FAC)

fig = go.Figure(data=go.Scatter(
            x=(hex_df['hex1_id'] - hex_df['hex2_id']),
            y=(hex_df['hex1_id'] + hex_df['hex2_id']),
            mode='markers',
            marker_symbol = SYMBOL_NUMBER,
            marker={
                'size':DOTSIZE,
                'color': COLOR_BACKGROUND,
                'line': {'width':0.5, 'color':COLOR_BACKGROUND}
                },
            showlegend=False
    )
)

# Missing R7, R8
fig.add_trace(go.Scatter(
            x = no_r7r8['p'],
            y = no_r7r8['q'],
            mode='markers',
            marker_symbol = SYMBOL_NUMBER,
            marker={
                'size':DOTSIZE,
                'color': COLOR_NO_R7R8,
                'line': {'width':0.5, 'color':COLOR_NO_R7R8}
                },
            name="No R7,R8"
    )
)

# R78
fig.add_trace(go.Scatter(
            x = r7r8['p'],
            y = r7r8['q'],
            mode='markers',
            marker_symbol = SYMBOL_NUMBER,
            marker={
                'size':DOTSIZE,
                'color': COLOR_R7R8,
                'line': {'width':0.5, 'color':COLOR_R7R8}
                },
            name="R78"
    )
)

# Pale
fig.add_trace(go.Scatter(
            x = pale['p'],
            y = pale['q'],
            mode='markers',
            marker_symbol = SYMBOL_NUMBER,
            marker={
                'size':DOTSIZE,
                'color': COLOR_PALE,
                'line': {'width':0.5, 'color':COLOR_PALE}
                },
            name="Pale"
    )
)

# Yellow
fig.add_trace(go.Scatter(
            x = yellow['p'],
            y = yellow['q'],
            mode='markers',
            marker_symbol = SYMBOL_NUMBER,
            marker={
                'size':DOTSIZE,
                'color': COLOR_YELLOW,
                'line': {'width':0.5, 'color':COLOR_YELLOW}
                },
            name="Yellow"
    )
)

# DRA
fig.add_trace(go.Scatter(
            x = dra['p'],
            y = dra['q'],
            mode='markers',
            marker_symbol = SYMBOL_NUMBER,
            marker={
                'size':DOTSIZE,
                'color': COLOR_DRA,
                'line': {'width':0.5, 'color':COLOR_DRA}
                },
            name="DRA"
    )
)

fig.update_layout(title='',
                    yaxis_range=[tot_min , tot_max + tot_max/10],
                    xaxis_range=[tot_min, tot_max + tot_max/10],
                    width=750,
                    height=750,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )

fig.update_xaxes(showgrid=False, showticklabels=False)
fig.update_yaxes(showgrid=False, showticklabels=False)

fig.update_scenes(aspectmode="data")

fig.show()

# %%
fig2 = go.Figure(data=go.Scatter(
            x=(hex_df['hex1_id'] - hex_df['hex2_id']),
            y=(hex_df['hex1_id'] + hex_df['hex2_id']),
            mode='markers',
            marker_symbol = SYMBOL_NUMBER,
            marker={
                'size':DOTSIZE,
                'color': COLOR_BACKGROUND,
                'line': {'width':0.5, 'color':COLOR_BACKGROUND}
                },
            showlegend=False
    )
)

fig2.add_trace(go.Scatter(
            x = pr8['p'],
            y = pr8['q'],
            mode='markers',
            marker_symbol = SYMBOL_NUMBER,
            marker={
                'size':DOTSIZE,
                'color': COLOR_8PR,
                'line': {'width':0.5, 'color':COLOR_8PR}
                },
            name="8 PR"
    )
)

fig2.add_trace(go.Scatter(
            x = pr7['p'],
            y = pr7['q'],
            mode='markers',
            marker_symbol = SYMBOL_NUMBER,
            marker={
                'size':DOTSIZE,
                'color': COLOR_7PR,
                'line': {'width':0.5, 'color':COLOR_7PR}
                },
            name="7 PR"
    )
)

fig2.add_trace(go.Scatter(
            x = pr6['p'],
            y = pr6['q'],
            mode='markers',
            marker_symbol = SYMBOL_NUMBER,
            marker={
                'size':DOTSIZE,
                'color': COLOR_6PR,
                'line': {'width':0.5, 'color':COLOR_6PR}
                },
            name="6 PR"
    )
)

fig2.update_layout(title='',
                    yaxis_range=[tot_min , tot_max + tot_max/10],
                    xaxis_range=[tot_min, tot_max + tot_max/10],
                    width=750,
                    height=750,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )

fig2.update_xaxes(showgrid=False, showticklabels=False)
fig2.update_yaxes(showgrid=False, showticklabels=False)

fig2.update_scenes(aspectmode="data")

fig2.show()

# %%
## Generate output
fig.write_image(result_dir / 'R7R8_ME_eyemap.pdf')
fig2.write_image(result_dir / 'equator_ME_eyemap.pdf')
