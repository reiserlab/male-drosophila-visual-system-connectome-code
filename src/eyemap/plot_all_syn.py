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

# %% metadata={}
from madvisc.utils import olc_client
from madvisc.utils.ROI_plots import plot_all_syn

c = olc_client.connect(verbose=True)

# %% [markdown]
# Plot hexagonal 'eyemap' heatplot of the sum of all the `post` synapses from all cells of all cell types within the right optic lobe.
#
# Plot is saved as a PDF in the folder `PROJECT_ROOT/results/cov_compl`.

# %% metadata={}
side = 'both'
syn_type = 'post'
fig = plot_all_syn(side=side, syn_type=syn_type)
fig.show()

# %%
