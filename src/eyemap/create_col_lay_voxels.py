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
from utils.ROI_voxels import voxelize_col_and_lay

# %%
voxelize_col_and_lay()

#expected runtime for all neuropils, col and lay: 10 min

# %%
