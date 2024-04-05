# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
from pathlib import Path
import sys
from dotenv import load_dotenv, find_dotenv
load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath('src')))
print(f"Project root directory: {PROJECT_ROOT}")

# data import & manipulation
import pandas as pd

# %%
# %autoreload 2
import cmap
# Optic Lobe libraries
from utils import olc_client
c = olc_client.connect(verbose=True)

from utils.plotter import plot_cns, save_figure
from utils.helper import slugify
from utils.neuron_bag import NeuronBag

# %% [markdown]
# ## Gallery for grouped VPN with cns background
#
# For each subgroup, pick one cell from each type and plot the frontal view with cns outline as the background

# %%
# result path
result_dir = PROJECT_ROOT / 'results' / 'cell_gallery'
result_dir.mkdir(parents=True, exist_ok=True)

param_dir = PROJECT_ROOT / 'params'


# %%
# load csv file
file_path = Path(param_dir, "Medulla_VPN_groups_092023_1.csv")
df = pd.read_csv(file_path)

# %%
# unique Subgroups,
print(df['Subgroup'].unique())

# %%
# loop all the subgroups
# however, the grouping should be checked and optimized

result_dir_skel = result_dir / 'skeletons'
result_dir_skel.mkdir(parents=True, exist_ok=True)

for i in df['Subgroup'].unique()[0:1]:
    # get the df['type'] whose "Subgroup" == i
    types = df[df['Subgroup'] == i]['type'].values # get the types in this subgroup
    n_types = len(types) # get the number of types
    print(n_types)

    # get one cell from each type in one subgroup
    bid = []
    for t in types:
        bag = NeuronBag(cell_type=t)
        # sort the ids by distance to the center of the medulla
        bag.sort_by_distance_to_hex("ME(R)", 18, 18)
        bid.append(bag.first_item) # get the first
    print(bid)

    fig = plot_cns(
        bodyid=bid,
        celltype=types,
        show_skeletons=True, #choose skeleton or mesh
        # show_meshes=True,
        show_outline=True, # the first time to load the outline will take a while
        zoom=4,
        palette='tab20'
        )
    fn = slugify(f"subgroup_{i}", to_lower=False)
    save_figure(fig
      , name=fn
      , width=1000
      , height=600
      , path=result_dir_skel
      , showlegend=False
    ) # adjust size and path

# %%
# # DEBUG

# # get the df['type'] whose "Subgroup" == 2
# types = df[df['Subgroup'] == 2]['type'].values # get the types in this subgroup
# n_types = len(types) # get the number of types
# print(n_types)

# # get one cell from each type in one subgroup
# bid = []
# for t in types:
#     bag = NeuronBag(cell_type=t)
#     # bag.sort_by_distance_to_hex(18,18) # sort the ids by distance to the center of the medulla
#     bid.append(bag.get_body_ids()[0]) # get the first
# print(bid)

# # choose a view and define the plotter
# gp = GalleryPlotter(
#       body_id=bid
#     , view= Gallery_View_cns.GALLERY_cns
# ) # use GALLERY_cns for the CNS view
