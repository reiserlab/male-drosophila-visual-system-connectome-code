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
from utils import olc_client
c = olc_client.connect(verbose=True)

# %%
from utils.align_mi1_t4 import create_alignment

# %%
# Creates file `results/eyemap/mi1_t4_alignment.xlsx`
create_alignment()

# Expected run time: 20 min

# %%
