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
from pathlib import Path
import sys
from IPython.display import display

import navis.interfaces.neuprint as navnp

from dotenv import load_dotenv, find_dotenv
load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath('src')))
print(f"Project root directory: {PROJECT_ROOT}")

from utils.plotter import group_plotter, show_figure
from utils import olc_client
c = olc_client.connect()

# load some helper functions
from utils.hex_hex import \
    hex_to_bids\
  , bid_to_hex\
  , get_hex_df\
  , get_incomplete_hex\
  , get_overfull_hex

# %% [markdown]
# # Conversions between hex and body IDs
#
# The `hex_hex.py` file has some helper functions to access the data frame provided by @kitlongden.
# Since the underlying data frame might change over time, please use the functions to access the data
# frame, don't load the file directly.
#
# ## Find hex coordinates for body ID
#
# `bid_to_hex()` finds the Medulla hex coordinates for a single body ID.

# %%
hex_for_nonexisting_body_id = bid_to_hex(23)
display(f"If the cell with body_id doesn't exist, the function returns "
        f"'{hex_for_nonexisting_body_id}'")

hex_for_body_id = bid_to_hex(26973)
display(f"If the cell exists within the columnar structure, the function returns "
    f"the (hex1, hex2) coordinates as a tuple: {hex_for_body_id}")

bid_to_hex(54865)


# %% [markdown]
# ## Find bodyIDs for a hex
#
# `hex_to_bids()` gets all the cells that are assigned to a specific column. Define the column using
# a tuple. By default the function returns the body ids for all 12 cell types, but `n_types` can
# provide a list of cell types that the function should return.
#
# The function returns a dictionary with the cell types as keys and the body ids as a list of values.
# In most cases each key will have a list with a single item.
#
# The dictionary will not contain keys for cell types that are note represented in the column.

# %%
hex_to_bids((30,17), n_types=['L1', 'Mi1'])


# %% [markdown]
# …or just get a flat list of body ids

# %%
hex_to_bids((30,17), n_types=['L1', 'Mi1'], return_type='list')


# %% [markdown]
# ## Get the whole DataFrame
#
# This function gives you the whole DataFrame in the raw format. You will need to ask @kitlongden about
# the exact definition, what duplicate hex IDs and `NaN` means, and why the data frame has more rows
# than we have columns in the eye. So use at your own risk.

# %%
get_hex_df()

# %% [markdown]
# ## List of columns that are missing cell types
#
# This function returns a list of hex ID tuples where at least one of the 12 cell types is not
# present.

# %%
get_incomplete_hex()

# %% [markdown]
# ## List columns with duplicates
#
# This function returns a list of tuples with hex IDs for columns where more at least one cell type
# is present more than once.

# %%
get_overfull_hex()

# %%
hex_to_bids((34,30), return_type='list')


# %% [markdown]
# # Test some weird observation

# %%
# This used to have 2 different columns
bid_to_hex(104527)

# %% [markdown]
# ## combine functions
#
# You can combine these functions, for example iterate over the list of columns with more than one
# cell type and then get the list of body IDs for that column. You can, for example, see that
# hex (6,9) has two Mi9 cell with body IDs 138465 and 138015.

# %%
for of in get_overfull_hex():
    display(f"hex {of} has the following 'type': [body_ids] {hex_to_bids(of)}")

# %% [markdown]
# A list of columns that is missing at least one of the 12 cell types and count how many different
# types there are.

# %%
for of in get_incomplete_hex():
    display(f"hex {of} has {len(hex_to_bids(of))} different neuron types")

# %%
me_r = navnp.fetch_roi('ME_R_col_34_30')

col_34_30 = group_plotter(
      body_ids=hex_to_bids((34,30), return_type='list')
    , camera_distance=1.4
    #, ignore_cache=True
  )
show_figure(col_34_30)

# %%
