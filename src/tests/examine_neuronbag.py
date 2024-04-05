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
# # Helper script to see how the NeuronBag and OLTypes work
#
#

# %%
# %load_ext autoreload
from pathlib import Path
import sys
from dotenv import load_dotenv, find_dotenv
load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath('src')))
print(f"Project root directory: {PROJECT_ROOT}")
from utils import olc_client
c = olc_client.connect()

# %% [markdown]
# # Get the star for a cell type
#
# For the example C2, this should currently be 78453

# %%
# %autoreload 2

from utils.ol_types import OLTypes
olt = OLTypes()

all = olt.get_neuron_list(side='both')
display(all.sample(frac=1).head(5))

r_dom = olt.get_neuron_list()
display(f"The dataset has {len(all)} named instances, {len(r_dom)} of which have dominant features in the right hemisphere.")

# %% [markdown]
# `OLTypes` is an interface to the whole dataset and allows you to retrieve all neurons from a group:

# %%
ol_all = olt.get_neuron_list(primary_classification='OL', side='both')

ol_combined = ol_intrinsic = olt.get_neuron_list(primary_classification=['OL_intrinsic', 'OL_connecting'])

assert len(ol_all)==len(ol_combined), "something went wrong with the OLTypes"

ol_intrinsic = olt.get_neuron_list(primary_classification='OL_intrinsic', side='both')
vpn_all = olt.get_neuron_list(primary_classification='OL_connecting', side='both')

non_ol = olt.get_neuron_list(primary_classification='non-OL', side='both')

vcn_vpn_other = olt.get_neuron_list(primary_classification=['VCN', 'VPN', 'other'], side='both')

assert len(non_ol)==len(vcn_vpn_other), "something went wrong inside OLTypes"

display(f"random 5/{len(ol_all)} OL neuron types")
display(ol_all.sample(frac=1).head(5))
display(f"random 5/{len(ol_intrinsic)} OL intrinsic neuron types")
display(ol_intrinsic.sample(frac=1).head(5))
display(f"random 5/{len(vpn_all)} VPN neuron types")
display(vpn_all.sample(frac=1).head(5))

display(f"random 5/{len(non_ol)} non-OL neuron types")
display(non_ol.sample(frac=1).head(5))

# %%
olt.get_neuron_list(primary_classification=['OL', 'non-OL'], side='both')

# %%
olt.get_star('C2')

# %% [markdown]
# # Create a C2 NeuronBag
#
# … and list all the body IDs

# %%
from utils.neuron_bag import NeuronBag
c2bag = NeuronBag('C2')
c2bag.get_body_ids()

# %% [markdown]
# # Sort by distance to star
#
# Without having to specify what the star is, this function orders the NeuronBag by their distance to the star.

# %%
c2bag.sort_by_distance_to_star()
c2bag.get_body_ids()

# %% [markdown]
# # Get Star from the bag
#
# After sorting, the star should be the first item in the NeuronBag.

# %%
c2bag.first_item

# %%
