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
# # InstanceSummary
#
# The class InstanceSummary provides an interface to the information we show in the Cell Type Catalog summary plots.
#
# The class might be useful for other applications and here we explain how to access the data.

# %%
# %load_ext autoreload
# %autoreload 2
from pathlib import Path
import sys
from dotenv import load_dotenv, find_dotenv
load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath('src')))
print(f"Project root directory: {PROJECT_ROOT}")

# %%
# %autoreload 2
from neuprint import NeuronCriteria as NC

import navis
import navis.interfaces.neuprint as neu

from utils import olc_client

c = olc_client.connect(verbose=True)

# %% [markdown]
# `InstanceSummary` is initialized with the name of an instance. It doesn't do any checking, so make sure the instance exists.
#

# %%
# %autoreload 2

from utils.instance_summary import InstanceSummary

insum = InstanceSummary('Mi1_R', connection_cutoff=None, per_cell_cutoff=1.0)


# %% [markdown]
# Basic descriptions of the instance are available, including a cell count:

# %%

print(f"Type: {insum.type_name}")
print(f"Instance: {insum.instance_name}")

print(f"Number of cells: {insum.count}")


print(f"is this a bilateral type: {insum.is_bilateral}")

print(f"or part of the r-dominant set: {insum.is_r_dominant}")

# %% [markdown]
# The consensus neuro transmitter (see methods section) is available, also in an abbreviated spelling.

# %%

print(f"Consensus neurotransmitter: {insum.consensus_neurotransmitter}")
print(f"Consensus NT: {insum.consensus_nt}")

# %% [markdown]
# Get the top 5 connecting upstream and downstream instance names:

# %%


print(f"Top 5 Upstream: {insum.top5_upstream}")

print(f"Top 5 Downstream: {insum.top5_downstream}")


# %% [markdown]
# Get all synapses and their depth

# %%

print(f"columns: {insum.columns}")


# %% [markdown]
# Retrieve the synapses and their depth for each cell.

# %%

print(f"synapses: {insum.synapses}")



# %% [markdown]
# Get the column innervation

# %%

print(f"Innervation: {insum.innervation}")


# %% [markdown]
# Simple examples for innervation plots

# %%
# fig = go.Figure()
import plotly.graph_objects as go
import scipy


for roi in ['ME(R)', 'LO(R)', 'LOP(R)']:

    inn = insum.innervation[insum.innervation['roi']==roi]
    fig = go.Figure(
        data =go.Scatter(
            x=inn['depth']
          , y=scipy.signal.savgol_filter(inn['col_innervation'], 5, 1, mode='constant')
        #   , line={'shape':'spline',  'smoothing': 1.3}
         
        )
        
      
    )
    fig.update_layout(title=roi)
    fig.show()

# %%
