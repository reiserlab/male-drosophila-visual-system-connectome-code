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
# # Plot 3D
#
# Questions to: Art
#
# For installation see also see [show_one_neuron.py](show_one_neuron.py).
#

# %%
"""
This cell does the initial project setup.
If you start a new script or notebook, make sure to copy & paste this part.

A script with this code uses the location of the `.env` file as the anchor for
the whole project (= PROJECT_ROOT). Afterwards, code inside the `src` directory
are available for import.
"""
from pathlib import Path
import sys
from dotenv import load_dotenv, find_dotenv
load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath('src')))
print(f"Project root directory: {PROJECT_ROOT}")


# %%
""" Imports related to data loading and wrangling """

import pickle

from neuprint import NeuronCriteria as NC

# use shorthand suggested in 
# https://navis.readthedocs.io/en/latest/source/tutorials/neuprint.html
import navis
import navis.interfaces.neuprint as neu

from utils.plotter import get_skeletons, get_meshes
from utils import olc_client
c = olc_client.connect(verbose=True)


# %%
""" Imports related to plotting """
# plotly backend
import plotly.express as px
# use shorthand suggested in https://plotly.com/python/graph-objects/
import plotly.graph_objects as go

# K3D backend
import k3d


# %%
""" 
load medulla columns as a reference

you'll need to get this file first "results/eyemap/Mi1_to_T4_hex.pickle"
"""

with open(Path(PROJECT_ROOT, 'results', 'eyemap', 'Mi1_to_T4_hex.pickle'), 'rb') as f:
    xyzpq = pickle.load(f) # [p,q] are reversed wrt eyemap paper, ie [q,p]

xyzpq.rename(columns={'hex1_id':'p', 'hex2_id':'q'}, inplace=True)
xyzpq.reset_index(inplace=True, drop=True)

# %%
""" load some neurons """

# Let's load all Dm4 cells, first get meta data
neu_df, roi_df = neu.fetch_neurons(NC(type="Pm4"))

# use a helper to load skeletons and meshes
neu_ske = get_skeletons(neu_df['bodyId'][0:3].to_list())
neu_msh = get_meshes(neu_df['bodyId'][0:3].to_list())


# %%
""" load a neuropil mesh """
ME_R = neu.fetch_roi('ME(R)')
LO_R = neu.fetch_roi('LO(R)')
LOP_R = neu.fetch_roi('LOP(R)')


# %%
""" plotting option 1, plotly """

fig_n = navis.plot3d(
    neu_ske,
    soma=False,
    color='black', linewidth=2,
    inline=False, backend='plotly')

fig_col = px.scatter_3d(xyzpq,
    x='x', y='y', z='z',
    title=('med col'),
    hover_name='bodyId',
    hover_data=['p', 'q'])

fig_col.update_traces(marker_size = 6, marker={"color":"gray"}, opacity=0.2)

fig_mesh = navis.plot3d(
    [ME_R, LO_R, LOP_R]
    , color=['yellow','yellow','grey']
    , alpha=0.2
    , inline=False
    , backend='plotly')

fig = go.Figure(data= fig_col.data + fig_n.data + fig_mesh.data)

fig.update_layout(autosize=False, width=900, height=600)
fig.update_layout(margin={"l":0, "r":0, "b":0, "t":0})

fig.show()

# %%
""" plot option 2, k3d """
# This options seems to be broken on different machines.
# This is deprecated for out code basis, we decided to use 
# plotly (either directly or through navis) everywhere

fig = k3d.plot(grid_visible=False)
fig += k3d.points(
    positions= xyzpq[['x','y','z']]
  , point_size=200
  , shader='3d'
  , color=0x3f6bc5
)

# FL: Adding the labels in one go instead of loop (faster)
position = xyzpq.loc[:,['x','y','z']]
label = xyzpq.apply(lambda row:  f"[{row.p:.0f},{row.q:.0f}]", axis=1)
textpq = k3d.text(text=label.tolist(), position=position, size= 0.5, label_box=False)
fig += textpq


# FL: Adding skeletons by plotting them in navis and then extracting the k3d objects
skel_plot = navis.plot3d(
    neu_ske
  , soma=False
  , color='black', linewidth=2
  , inline=False, backend='k3d'
)

for k3d_o in skel_plot.objects:
    fig += k3d_o

# FL: Adding neuropil by plotting it in navis and then extracting the k3d objects
np_plots = navis.plot3d(ME_R, color='yellow', inline=False, backend='k3d')

for k3d_o in np_plots.objects:
    fig += k3d_o

fig.display()

# %%
""" plot option 3, pure navis """

ME_R.color=(255, 255, 0, 0.2)
navis.plot3d([xyzpq, neu_ske, ME_R, LO_R, LOP_R], color = 'k', scatter_kws={"color":"b"})

# %%
