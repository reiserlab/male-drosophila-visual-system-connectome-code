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
from madvisc.utils import olc_client
from madvisc.utils.ol_types import OLTypes
from madvisc.utils.overall_summary_queries import add_color_group
from madvisc.utils.overall_summary_table_plotting_functions import \
    plot_group_summary_table\
  , plot_neuropil_group_table\
  , plot_neuropil_group_celltype_table\
  , plot_neuropil_group_cell_table

c = olc_client.connect(verbose=True)


# %%
## Formatting parameters
style = {
    'export_type':'svg'
  , 'font_type': 'arial'
  , 'markerlinecolor':'black'
  , 'linecolor':'black'
  , 'fillcolor':'white'
}

sizing = {
    'fig_width':600 # units = mm, max 180
  , 'fig_height':500 # units = mm, max 170'markersize':5,
}


# %%
ol = OLTypes()
types = ol.get_neuron_list(side='both')
cell_instances = types['instance']

# %% [markdown]
# ### Table 1: cell type groups
#
# # Cell type groups
#
# This is the data used in Fig. 1e

# %%
fig, celltype_groups_df = plot_group_summary_table(
    neuron_list=types
  , style=style
  , sizing=sizing
)

fig.show()

# %% [markdown]
# # Neuron inventory by brain region
#
# This data is used in Fig. 1f and represents the size of the bubble charts.

# %%
df = add_color_group(types).reset_index()

fig, _ = plot_neuropil_group_table(
    df=df
  , threshold=0.02
  , style=style
  , sizing=sizing
)

fig.show()

# %% [markdown]
# The data represents the pie charts in Fig. 1f

# %%
df = add_color_group(types).reset_index()

fig, _ = plot_neuropil_group_celltype_table(
    df=df
  , threshold=0.02
  , style=style
  , sizing=sizing
)

fig.show()

# %% [markdown]
# # cell count per cell type group and neuropil. 
#
# This is not directly used in the main figures of the manuscript.

# %%
df = add_color_group(types).reset_index()

fig, _ = plot_neuropil_group_cell_table(
    df=df
  , threshold=0.02
  , style=style
  , sizing=sizing
)

fig.show()
