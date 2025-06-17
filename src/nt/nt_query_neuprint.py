# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
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
from IPython.display import display
from madvisc.utils.neurotransmitter import get_special_neuron_list
from madvisc.utils.neurotransmitter import get_nt_for_bid
from madvisc.utils import olc_client
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
