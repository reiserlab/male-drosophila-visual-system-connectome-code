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
from madvisc.utils.ROI_calculus import create_ol_layer_boundaries

from madvisc.utils import olc_client
c = olc_client.connect(verbose=True)

# %% [markdown]
# # Create Layer boundaries file
#
# Create or replace `cache/eyemap/[ME|LO|LOP]_layer_bdry.csv` files.
#
# This script is also called by `snakemake layer_boundaries` and, if the files don't exist, but the `ROI_calculus.load_layer_thre()` function which uses the files.

# %%
create_ol_layer_boundaries()

# expected runtime: 1 min 20 secs (ME), 10 secs (LO), 40 secs (LOP)

# %%
