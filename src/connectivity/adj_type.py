# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: ol-c-kernel
#     language: python
#     name: ol-c-kernel
# ---

# %% [markdown]
# ## Get adjacency matrix by type
#
# 1. query all cell types in the optic lobe, 
# 2. pick some cell types (eg. >=500 instances)
# 3. query all-to-all connectivity (i.e., adjacency matrix) within a neuropil
# 4. plot the adj matrix and compute some stats
#

# %% [markdown]
# ### init setup

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
load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath('src')))
print(f"Project root directory: {PROJECT_ROOT}")

# %%
import neuprint
print(neuprint.__version__)

# %% Import libraries
from neuprint import fetch_neurons, fetch_synapses,  fetch_adjacencies, connection_table_to_matrix, merge_neuron_properties
from neuprint import NeuronCriteria as NC, SynapseCriteria as SC

# This library wasn't installed before, you might need to rerun library installation
import navis
import navis.interfaces.neuprint as navnp


# %%
from utils import olc_client
c = olc_client.connect(verbose=True)

# %%
import plotly.io as pio

import matplotlib.pyplot as plt

import pandas as pd
pd.options.display.float_format = '{:.2f}'.format

import numpy as np

# %%
from neuprint.queries import fetch_all_rois, fetch_roi_hierarchy

# # Show the ROI hierarchy, with primary ROIs marked with '*'
# print(fetch_roi_hierarchy(include_subprimary=True, mark_primary=True, format='text'))

# show primary ROIs
print(fetch_all_rois())

# %% [markdown]
# ### get all cell types in OL, and some histograms

# %%
from queries.completeness import fetch_ol_types, fetch_ol_types_and_instances, fetch_ol_complete
ol_type = fetch_ol_types(client=c)
print(ol_type)
# ol_type_inst = fetch_ol_types_and_instances(client=c)
# ol_comp = fetch_ol_complete(client=c) #long runtime

# %%
ol_type['count'].hist(bins=np.linspace(10,1200,100))

# %%
_=plt.hist(ol_type['count'].values, cumulative=-1, bins=100)

# %% [markdown]
# ### pick some cell types, here >= 500 cells

# %%
type_col = ol_type['type'][ol_type['count']>= 500]
cell_types = list(type_col.values)
print(cell_types)

# %%
# NOT run, fectch cells
# ctype = ['T4a']
# neu_df, roi_df = fetch_neurons(NC(type=ctype))

# %% [markdown]
# ### get connectivity and construct adj matrix for plotting

# %%
# syn_rois = ['ME(R)', 'LO(R)', 'LOP(R)']
syn_rois = ['LO(R)']
neuron_types_rois_df, conn_types_rois_df = fetch_adjacencies(NC(type=cell_types), NC(type=cell_types), rois=syn_rois, batch_size=1000)

# neuron_types_rois_df, conn_types_rois_df = fetch_adjacencies(NC(type=cell_types), NC(type=cell_types))


# %%
# # save and load csv
# base_dir = PROJECT_ROOT / 'results' / 'connectivity'

# neuron_types_rois_df.to_csv(base_dir / 'neuron_types_rois_df.csv')
# conn_types_rois_df.to_csv(base_dir / 'conn_types_rois_df.csv')

# neuron_types_rois_df= pd.read_csv(base_dir / 'neuron_types_rois_df.csv')
# conn_types_rois_df= pd.read_csv(base_dir / 'conn_types_rois_df.csv')

# %%
conn_df = merge_neuron_properties(neuron_types_rois_df, conn_types_rois_df, 'type')
conn_matrix = connection_table_to_matrix(conn_df,'type')
conn_matrix = conn_matrix.rename_axis('type_pre', axis=0).rename_axis('type_post', axis=1)
conn_matrix = conn_matrix.loc[sorted(conn_matrix.index), sorted(conn_matrix.columns)]

# %%
# plot adjacency matrix
pd.set_option('display.max_columns', 100)
conn_matrix

# %%
# who's in this matirx 
ol_type[ol_type['type'].isin( list(conn_matrix.index) )]
