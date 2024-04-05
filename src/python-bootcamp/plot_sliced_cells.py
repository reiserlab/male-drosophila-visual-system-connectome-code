# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Cell slicing
#
# This is a demo file for how to create galleries of cells.

# %%
# %load_ext autoreload
"""
This cell does the initial project setup.
"""
from pathlib import Path
import sys
from dotenv import load_dotenv, find_dotenv
load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath('src')))
print(f"Project root directory: {PROJECT_ROOT}")

# %%
""" Imports related to data loading """
from neuprint import NeuronCriteria as NC

import navis
import navis.interfaces.neuprint as neu

from utils import olc_client
c = olc_client.connect(verbose=True)

# %%
# %autoreload 2
"""
imports related cell selection and plotting

if you modify any of the libraries, you will only need to rerun this cell, `autoreload` will
reload your newest version of the code.

"""
from utils.hex_hex import hex_to_bids
from utils.plotter import group_plotter, show_figure, save_figure
from utils.ng_view import NG_View
from utils.helper import slugify
from utils.neuron_bag import NeuronBag

# %% [markdown]
# ## Plot L1 and Mi1 from the center
#
# The next cell plots L1 (black) and Mi1 (magenta) from the location 18,18.
#
# The plot will be dynamic, you can rotate it.

# %%
# define which column you want cells from
hex_1_2_column = (18,18)

# pull the body ids of columnar cells from the previously defined column
ids_1818 = hex_to_bids(
    hex_1_2_column
  , n_types=['L1', 'Mi1']   # only get L1 and Mi1
  , return_type='list'     # get the body ids as a simple list
)


# %%
lc6_bag = NeuronBag(cell_type='LC6')
ids_lc6 = lc6_bag.get_body_ids()

# %%
lc4_bag = NeuronBag(cell_type='LC4')
lc4_bag.sort_by_distance_to_hex('LO(R)', 18, 18)
ids_lc4 = lc4_bag.get_body_ids()

# %%
## See src/utils/plotter.py#group_plotter for all available options

fig1 = group_plotter(
    ids_1818                                    # list of body ids

  , colors=[(0.,0.,0.,1.), (1., 0., 1., 1.)]    # list of colors (optional, if none given
                                                #   a colorful pallette will be chose, first cell
                                                #   in red)

  , shadow_rois=['ME(R)', 'LO(R)', 'LOP(R)']    # This defines, for which ROIs you want to see
                                                #   the backdrop.
  
  , prune_roi='slice'                           # say that you want your neuron pruned into a slice

  , plot_synapses=False                         # The standard behavior of group_plotter is to 
                                                #   plot the synapses, but that might get a bit
                                                #   crowded for this type of plots.

  , plot_mesh=True                              # The standard behavior of group_plotter only shows
                                                #   skeletons, no meshes.

  , view=NG_View.GALLERY1                       # This defines the viewing direction. Gallery1 is
                                                #   the one Art used in his initial prototype.
)

show_figure(
    fig1
  , width=1000, height=1000                     # Define the size of your plot (in px). 1000×10000 might be
                                                #   a good size for preview and making decisions.
  , showlegend=False                            # Disable the legend
)

## Expected runtime: about 15s

# %% [markdown]
# If you like what you see, you can save it.

# %%
save_figure(
    fig1
  , name="L1-Mi1_ME-LO-LOP_18x18"                   # This will save the file to 
                                                    #   results/cell_gallery/L1-Mi1_ME-LO-LOP_18x18.png
                                                    #   If that file already exists, it will add a timestamp
                                                    #   to the filename.
  
  , width=1000, height=1000                         # same parameters as for `show_figure`
  , showlegend=False
)

## Expected runtime: about 3s

# %% [markdown]
# ## Example automation
#
#

# %% magic_args="false --no-raise-error" language="script"
# # Remove the line above if you want to run the 2hr example.
#
# for c_t in ['L1', 'L2', 'L3', 'Mi1', 'Mi4', 'Mi9', 'C2', 'C3', 'Tm1', 'Tm2', 'Tm4', 'Tm9', 'Tm20', 'T1']:
#     for hex1 in range(15, 22):
#         for hex2 in range(15, 22):
#             body_id = hex_to_bids( (hex1, hex2), n_types=[c_t], return_type='list')
#             fig2 = group_plotter(body_id
#               , colors=[(0.,0.,0.,1.)]
#               , shadow_rois=['ME(R)', 'LO(R)', 'LOP(R)']
#               , prune_roi='slice'
#               , plot_synapses=False
#               , view=NG_View.GALLERY1)
#             filename = slugify(f"{c_t}_{body_id}_{hex1}x{hex2}", to_lower=False)
#             save_figure(fig2, name=filename, showlegend=False)
#
# ## Expected runtime: >2hr

# %%
