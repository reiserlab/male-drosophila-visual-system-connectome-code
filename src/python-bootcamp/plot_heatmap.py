# %%
# %load_ext autoreload
# %autoreload 2

# %%
"""
This cell does the initial project setup.
If you start a new script or notebook, make sure to copy & paste this part.

A script with this code uses the location of the `.env` file as the anchor for
the whole project (= PROJECT_ROOT). Afterwards, code inside the `src` directory
are available for import.
"""
from pathlib import Path
import sys
from dotenv import load_dotenv, find_dotenv
import pandas as pd
import numpy as np

load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath('src')))
print(f"Project root directory: {PROJECT_ROOT}")

from utils.plotting_functions import plot_heatmap
# # %%
# from utils import olc_client
# c = olc_client.connect(verbose=True)

# %%
# make up some data
heatmap = pd.DataFrame(
    np.random.randint(0, 100, size=(10, 10))
  , columns=list('ABCDEFGHIJ')
)
heatmap.index = list('ABCDEFGHIJ')

# %%
# annotation, show values greater than 0.5
anno = heatmap\
    .where(heatmap > 50)\
    .replace({np.nan: ''})


# %% plot binned heatmap with pre-defined anno, optinally provide bins, bvals is automatically computed
# binned heatmap. 
# Two unresolved problems:
# 1.to have aspect ratio = 1, the axis ticks are positioned away from the axis.
# 2.the color bar is missing the lowest tick, and the rest ticks are shifted. 

# optionally, one can normalize the heatmap first
# heatmap = heatmap / heatmap.values.max()

fig = plot_heatmap(
    heatmap=heatmap
  , anno=anno
  , binned=True
  , anno_text_size=6
  , show_colorbar=True
  , equal_aspect_ratio=True
  , manual_margin=True
)

# further change the layout
fig.update_layout(title='binned heatmap', width=350, height=350)

fig.show()

# %% plot continuous heatmap, bins sets the color scale range with bins[-1] being the max
# continuous heatmap

fig = plot_heatmap(
    heatmap=heatmap
  , anno=anno
  , binned=False
  , bins=[0, 25, 50, 75, 100]
  , anno_text_size=6
  , show_colorbar=True
  , equal_aspect_ratio=True
  , manual_margin=True
)
# fig.update_layout(margin={'l':80, 'r':80, 'pad':0})

fig.show()

# %% save plot
# fig.write_image(Path(result_dir, 'heatmap_name' + '.svg'))

# or
# import plotly.io as pio
# pio.write_image(fig, file= Path(result_dir, 'heatmap_name.svg')   )
