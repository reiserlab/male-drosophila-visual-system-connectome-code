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
# # Create file for DB upload
#
# This script converts the data from `hex_hex` to a format that is better suited for the database import.
#
# `hex_hex` contains the manual assignment of columnar neurons to their base column.
#
# The result is stored in `results/exchange/ME_assigned_columns.csv`

# %%
from pathlib import Path
from dotenv import find_dotenv
from madvisc.utils.hex_hex import get_hex_df

from madvisc.utils import olc_client
c = olc_client.connect(verbose=True)

# %%
all_columns = get_hex_df()

long_col = all_columns\
    .melt(id_vars=['hex1_id', 'hex2_id'], var_name='neuron_type', value_name='bodyId')\
    .dropna()

upload_format = long_col[~long_col['bodyId'].duplicated()]\
    .set_index(['bodyId', 'neuron_type'])\
    .sort_values(by=['hex1_id', 'hex2_id'])\
    .rename({'hex1_id': 'assigned_hex1', 'hex2_id': 'assigned_hex2'}, axis=1)

# %%
data_path = Path(find_dotenv()).parent / 'results' / 'exchange'
data_path.mkdir(exist_ok=True)
upload_format.to_csv(data_path / 'ME_assigned_columns.csv')

# %%
