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
# # Check neuron types in hex_hex
#
# This script checks that all body IDs from the `hex_hex` data set have the same neuron type assigned in the neuprint database.
#
# This used to be an issue in the past, this automatic check verifies correctness.

# %%
from pathlib import Path
import sys
from dotenv import load_dotenv, find_dotenv
load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath("src")))
print(f"Project root directory: {PROJECT_ROOT}")

from utils import olc_client
c = olc_client.connect(verbose=True)

# %%
from utils.hex_hex import get_hex_df
from utils.ol_neuron import OLNeuron
import pandas as pd

# %%
all_columns = get_hex_df()

# %%
# TODO(@loeschef): Speed up, avoid for loops

for idx, row in all_columns.iterrows():
    hx1 = row['hex1_id']
    hx2 = row['hex2_id']
    for n_name, n_bid in row.items():
        if n_name in ['hex1_id', 'hex2_id']:
            continue
        if n_bid is pd.NA:
            continue
        neuron = OLNeuron(n_bid)
        assert neuron.get_type() == n_name, f"Type for {n_bid} is {neuron.get_type()}, while it is documented as {n_name}"


# %%
