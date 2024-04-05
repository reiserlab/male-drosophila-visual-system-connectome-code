# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: 'Python 3.11.5 (''.venv'': venv)'
#     language: python
#     name: python3
# ---

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
import numpy as np
from neuprint import NeuronCriteria as NC
from neuprint import SynapseCriteria as SC
from neuprint import fetch_neurons


import pandas as pd
import navis
import navis.interfaces.neuprint as neu

from utils import olc_client
c = olc_client.connect(verbose=True)

from utils.neuron_bag import NeuronBag
from utils.ol_neuron import OLNeuron

# %% [markdown]
# ## This notebook shows how to get column and layer rois for the star neuron

# %%
# example celltype
celltype = 'Dm15'

# %% [markdown]
# ### Using neuronbag to generate a bag of bodyIds belonging to 'celltype'

# %%
a_bag = NeuronBag(cell_type=celltype)

# %% [markdown]
# ### Sorting the bag of bodyIds by distance from the (hex1,hex2) column in the specified 'neuropil'

# %%
a_bag.sort_by_distance_to_hex('ME(R)',18,18)

# %% [markdown]
# ### Getting the sorted bodyIds

# %%
a_bag.get_body_ids(a_bag.size)

# %% [markdown]
# ### Defining the star neuron as the first item in the list of sorted bodyIds – the closest bodyId to the central (18,18) column in the ME(R) (in this case)

# %%

neuron = OLNeuron(a_bag.first_item)

star_neuron = neuron.get_body_id()

# %%
star_neuron

# %% [markdown]
# ### Getting the fraction of synapses of the star neuron in each column/layer roi

# %%
neuron_df, roi_df = fetch_neurons(NC(bodyId=star_neuron))
roi_df['syncount'] = roi_df['pre'] + roi_df['post']
roi_df['fraction_syn']= roi_df['syncount']/roi_df['syncount'].sum()

# %%
roi_df

# %% [markdown]
# ### Getting the column and layer rois that are innervated with a threshold fraction of synapses

# %%
column_rois, layer_rois = neuron.innervated_rois(column_threshold=0.02, layer_threshold=0.01)

# %%
column_rois

# %%
layer_rois
