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

# %% [markdown]
# # Query and save neuron and neuron transmitters prediction data
#
# Save data to a cache dir, overwriting if exist alread

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
from IPython.display import display
load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath('src')))
print(f"Project root directory: {PROJECT_ROOT}")

# %%
from utils.neurotransmitter import get_special_neuron_list
from utils.neurotransmitter import get_nt_for_bid
from utils import olc_client
c = olc_client.connect(verbose=True)

# %% [markdown]
# ### Get neuron info

# %%
neuron_df = get_special_neuron_list()

# %%
tally = neuron_df\
    .groupby('main_groups')\
    .agg({
        'bodyId': 'count'
      , 'downstream': 'sum'
      , 'upstream': 'sum'
      , 'pre': 'sum'
      , 'post': 'sum'
      , 'instance': 'nunique'
      , 'type': 'nunique'
    })
display(tally)

# %%
display(tally.sum(axis=0))

# %% [markdown]
# ### Get all pre-synapses with nt

# %%
# query in batches
syn = get_nt_for_bid(neuron_df)

# %%
display(syn['nt'].value_counts())

# %%
