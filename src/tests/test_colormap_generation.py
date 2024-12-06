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
import pandas as pd

load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath("src")))

import numpy as np
from utils.column_plotting_functions import ticks_from_max
import plotly.graph_objects as go

# %% [markdown]
# Notebook for testing the colormap that is generated using the function `utils.column_plotting_functions.ticks_from_max`. 
#
# After setting the value of 'maxval' below, it will generate a plot with dummy data with the colorbar for that 'maxval'. 

# %%
maxval = 0

# %%
# Generate fake data
data = np.arange(1, maxval + 1)

# Generate the colormap and the appropriate tick values
tickvals, tickvals_text, cmap = ticks_from_max(maxval)

# Initiate plot
fig = go.Figure()

fig.update_layout(
    autosize=False
  , height=200
  , width=200
  , margin={"l": 0, "r": 0, "b": 0, "t": 0, "pad": 0}
  , paper_bgcolor="rgba(255,255,255,255)"
  , plot_bgcolor="rgba(255,255,255,255)"
)

fig.add_trace(
    go.Scatter(
        x=data
      , y=data
      , mode="markers"
      , marker_symbol=0
      , marker={
            "cmin": 0
          , "cmax": maxval
          , "size": 3
          , "color": cmap
          , "line": {
                "width": 0.5
              , "color": 'black'
            }
          , "colorbar": {
                "orientation": "v"
              , "thickness": 20
              , "len": 3
              , "tickmode": "array"
              , "tickvals": tickvals
              , "ticktext": tickvals_text
              , "ticklen": 5
              , "tickwidth": 1
              , "tickcolor": 'black'
            }
          , "colorscale": cmap
        }
      , showlegend=False
    )
)

# Display the figure
fig.show()

