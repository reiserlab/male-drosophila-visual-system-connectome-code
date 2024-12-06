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
"""
This cell does the initial project setup.
If you start a new script or notebook, make sure to copy & paste this part.

A script with this code uses the location of the `.env` file as the anchor for
the whole project (= PROJECT_ROOT). Afterwards, code inside the `src` directory
are available for import.
"""
import sys
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath('src')))

from neuprint import fetch_neurons, NeuronCriteria as NC

from utils import olc_client
from utils.completion_metrics import get_upstream_downstream_connections, get_completion_metrics
c = olc_client.connect(verbose=True)



# %% [markdown]
# # Supporting notebook
#
# This notebook extracts data to explain the discrepancy between the number of
# upstream and downstream connections in Fig. 1e,f

# %% [markdown]
# check the number of upstream and downstream connections within the accessory medulla
# if I include ALL the objects within the neuropil (using segments, a superset of neurons)

# %%
segments, segment_syndist = fetch_neurons(NC(label='Segment', rois='AME(R)'))
print(segment_syndist.query('roi == "AME(R)"')[['upstream', 'downstream']].sum())

# %%
roi_str = ['LA(R)', 'ME(R)', 'LO(R)', 'LOP(R)', 'AME(R)']

# %% [markdown]
# Calculating the total number of upstream and downstream connections within
# the optic lobe neuropils ('LA(R)', 'ME(R)', 'LO(R)', 'LOP(R)', 'AME(R)') for all the segments

# %%
result = get_upstream_downstream_connections(roi_str)

# %% [markdown]
# get completion metrics

# %%
result_neurons, result_segments = get_completion_metrics(roi_str)
