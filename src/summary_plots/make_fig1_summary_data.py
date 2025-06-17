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

# %%
from IPython.display import display

from madvisc.utils import olc_client
from madvisc.utils.ol_types import OLTypes

from madvisc.utils.overall_summary_queries import \
    make_ncell_nconn_nsyn_data\
  , make_ncell_nconn_data\
  , make_connectivity_sufficiency_data

c = olc_client.connect(verbose=True)

# %%
ol = OLTypes()
types = ol.get_neuron_list(side='both')
cell_instances = types['instance']

# %% [markdown]
# ## Generate a dataframe with the following summary quantities
#
# - Number of cells per cell type
# - Number of pre and post synapses per cell type
# - Number of upstream and downstream connections per cell type
#

# %%
ncell_nconn_nsyn_df = make_ncell_nconn_nsyn_data(cell_instances)
display(ncell_nconn_nsyn_df)

# %% [markdown]
# ## Generate a dataframe with the following summary quantities
#
# - Number of connected cells per cell type
# - Number of connected cell types per cell type
# - Number of connected input cells vs number of connected output cells
#

# %%
summary_df = make_ncell_nconn_data(cell_instances)
display(summary_df)

# %% [markdown]
# ## Generate a dataframe with the following summary quantities
#
# - fraction of unique pre cell type combinations as a function of number of
#   top connections considered
# - fraction of unique post cell type combinations as a function of number of
#   top connections considered
# - fraction of unique all cell type combinations as a function of number of
#   top connections considered
#

# %%
unique_combinations_df = make_connectivity_sufficiency_data(
    n_top_connections=5
)
display(unique_combinations_df)
