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
# # Show one Neuron
#
# Questions to: Frank
#
# Additional libraries and an update to your configuration are required to run notebooks that access the [neuprint](https://neuprint.janelia.org) database and plot outputs using [navis](https://navis.readthedocs.io/). Refer to the [starter guide](../../docs/python-getting-started.md) on how to [update your configuration](../../docs/python-getting-started.md#project-configuration) and [install libraries](../../docs/python-getting-started.md#dependency-management).

# %% Import libraries
from neuprint import NeuronCriteria as NC

import navis
import navis.interfaces.neuprint as neu

from madvisc.utils import olc_client


# %% connect to the client and set up search criteria
c = olc_client.connect(verbose=True)

example_neuron_criteria = NC(bodyId=37117)  # This is just a randomly selected L1

# Alternatively you can search for all neurons of a type etc.
# example_neuron_criteria = NC(type="L3")

# %% get a ROI
me_r = neu.fetch_roi("ME(R)")

# %% Pull the skeleton for one (or many) neurons
example_skel = neu.fetch_skeletons(example_neuron_criteria)

# If you wanted to pull more information about a neuron:
# neuron_df, roi_df = neu.fetch_neurons(example_neuron_criteria)

# %% Plot the ROI and the cell(s)
fig = navis.plot3d([example_skel, me_r])

# %%
