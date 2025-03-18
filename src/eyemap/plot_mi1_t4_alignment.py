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
from utils.ROI_plots import plot_mi1_t4_alignment
from utils import olc_client
c = olc_client.connect(verbose=True)

# %%
# This creates the file `results/eyemap/Alignment_mi1_t4.pdf`
plot_mi1_t4_alignment()

# Expected run time: 20m

# %%
