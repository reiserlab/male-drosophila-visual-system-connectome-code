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

# %%
# %load_ext autoreload
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
load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath("src")))
print(f"Project root directory: {PROJECT_ROOT}")

from utils import olc_client
c = olc_client.connect(verbose=True)


# %% [markdown]
# # Neuron Bag
#
# Neuron Bag is a collection of neurons. Currently they can be found by providing a cell type (as shown below).

# %%
# %autoreload 2
from utils.neuron_bag import NeuronBag
from utils.ng_view import NG_View

a_bag = NeuronBag(cell_type='LC6')

a_bag.get_body_ids()

# %% [markdown]
# So if you wanted to plot the first 3 LC6, you could do this:

# %%
from utils.plotter import group_plotter, save_figure

group_plotter(
    a_bag.get_body_ids(3)
  , shadow_rois=['ME(R)'])

# %% [markdown]
# or just copy&paste the output

# %%
group_plotter(
    [35321, 35534, 35598, 35825]
  , shadow_rois=['ME(R)']
  , plot_roi='ME(R)'
)

# %%
another_bag = NeuronBag(cell_type='LC4')

print(f"Is 'another bag' (of length {another_bag.size}) sorted? {another_bag.is_sorted}")


# %%
another_bag.get_body_ids(another_bag.size)

# %%
another_bag.sort_by_distance_to_hex(
   neuropil="ME(R)", hex1_id=18, hex2_id=18
)

another_bag.get_body_ids(another_bag.size)

# %%
print(f"Is 'another bag' (of length {another_bag.size}) sorted? {another_bag.is_sorted}")

# Get top 10 LC4 closest to 18/18
print(f"LC4 top 10: {another_bag.get_body_ids()}")



# %%
# save plots of top 3 LC4 closest to ME 18/18

for body_id in another_bag.get_body_ids(3):

    fig = group_plotter(
        [body_id]  
      , shadow_rois=['ME(R)', 'LO(R)', 'LOP(R)']
      , view=NG_View.GALLERY1
    )

    save_figure(fig, f"LC4_{body_id}")


# %%
