# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: ol-connectome
#     language: python
#     name: ol-connectome
# ---

# %% [markdown]
# # Generate cross section view to make a gallery of all neurons in OL
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
""" Imports related to data loading """
from neuprint import NeuronCriteria as NC

import navis
import navis.interfaces.neuprint as neu

from utils import olc_client
c = olc_client.connect(verbose=True)

# %%
""" Imports related to plotting """
# plotly backend
import plotly.express as px
# use shorthand suggested in https://plotly.com/python/graph-objects/
import plotly.graph_objects as go


# %%
""" imports related to data analysis """
import pandas as pd
import numpy as np

import pickle

# %%
result_dir = PROJECT_ROOT / 'results' / 'plot_cells' / 'cell_gallery'
result_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ### Load neurons  
# find column, load cell of given type

# %%
from utils.plotter import get_mesh, get_skeletons, get_skeleton, get_meshes

# %% [markdown]
# ## Plot the cross section of a neuron for given plane(s)

# %%
# testing neurons

# (massive neuron) OA-AL2i1, 10072
# (medium) TmY16
# (small) TmY5a

# id = 10072
# neu_ske = get_skeleton(id)
# neu_msh = get_mesh(id)

# %%
from utils.hex_hex import hex_to_bids

# %%
# exmaple L1 and Mi1 neurons, [18,18] is a good column for the current slicing planes
ids = hex_to_bids((18,18), n_types=['L1', 'Mi1'], return_type='list')

neu_ske = get_skeletons(ids)
neu_msh = get_meshes(ids)

# %% [markdown]
# @Frank, what's the best way to do find the cells of given type in the desired column ?

# %%
# find the cells of given type in the desired column
# @Frank, what's the best way to do this ?

# some types to plot first
types = ["TmY3", "TmY4", "TmY5a", "TmY9a", "TmY10", "TmY13", "TmY14", "TmY15", "TmY16", "TmY17", "TmY18", "TmY19", "TmY20", "TmY21"]
# append text to all string in a list
types = [x + '_R' for x in types]

neu_df, roi_df = neu.fetch_neurons(NC(instance=types))

# %% [markdown]
# ### design slicing planes, need to choose a normal and a translation vector for each plane

# %%
neu_df[neu_df['inputRois'].apply(lambda x: True if 'ME_R_col_1818' in x else False)]

# %%
from utils.geometry import plane_square

vn = np.array([1, 2, 0]) # normal vector
vn = vn / np.linalg.norm(vn) # normalize
mf = 3e4 # multiplication factor for the plane size
vt1 = np.array([20e3, 40e3, 35e3]) * 0.85 # translation vector
vt2 = np.array([22e3, 44e3, 35e3]) * 0.95 # translation vector

# planes
pl_rot1 = plane_square(vn, vt1, mf)
pl_rot2 = plane_square(vn, vt2, mf)

# %% [markdown]
# ### slice skeletons/meshes

# %%
# slicing neuron skeletons

# which nodes fall within the planes
d1 = vt1 @ vn
d2 = vt2 @ vn
ind_in = neu_ske[0].nodes[['x','y','z']].apply(lambda row: row @ vn > d1 and row @ vn < d2, axis=1)

# get the cross section of a neuron
ske_in = navis.subset_neuron(neu_ske[0], ind_in.values, inplace=False)

# %%
# slicing neuron meshes

# which nodes fall within the planes
d1 = vt1 @ vn
d2 = vt2 @ vn

# get the cross section of a neuron
ind_ske_in = neu_ske[0].nodes[['x','y','z']].apply(lambda row: row @ vn > d1 and row @ vn < d2, axis=1)
ske_in = navis.subset_neuron(neu_ske[0], ind_ske_in.values, inplace=False)

msh_v = pd.DataFrame(neu_msh[0].vertices[0], columns=['x', 'y', 'z'])
ind_msh_in = msh_v.apply(lambda row: row @ vn > d1 and row @ vn < d2, axis=1)
msh_in = navis.subset_neuron(neu_msh[0], ind_msh_in.values, inplace=False)

# %%
# load layer meshes and slice them

ME_R_layer = [None] * 10
ME_R_layer_sec = [None] * 10
# selected layers
for i in [1,3,5,7,9]:
    exec(f"ME_R_layer[{i-1}] = neu.fetch_roi('ME_R_layer_{i}')") # load layer meshes with a constructed variable name
    ME_R_layer_sec[i-1] = ME_R_layer[i-1].section(vn, vt1)
    ME_R_layer_sec[i-1] = pd.DataFrame(ME_R_layer_sec[i-1].vertices, columns=['x','y','z'])  # convert TrackedArray to dataframe

LO_R_layer = [None] * 7
LO_R_layer_sec = [None] * 7
# selected layers
for i in [1,3,5,7]:
    exec(f"LO_R_layer[{i-1}] = neu.fetch_roi('LO_R_layer_{i}')") # load layer meshes with a constructed variable name
    LO_R_layer_sec[i-1] = LO_R_layer[i-1].section(vn, vt1)
    LO_R_layer_sec[i-1] = pd.DataFrame(LO_R_layer_sec[i-1].vertices, columns=['x','y','z'])  # convert TrackedArray to dataframe

LOP_R_layer = [None] * 4
LOP_R_layer_sec = [None] * 4
# selected layers
for i in [1,3]:
    exec(f"LOP_R_layer[{i-1}] = neu.fetch_roi('LOP_R_layer_{i}')") # load layer meshes with a constructed variable name
    LOP_R_layer_sec[i-1] = LOP_R_layer[i-1].section(vn, vt1)
    LOP_R_layer_sec[i-1] = pd.DataFrame(LOP_R_layer_sec[i-1].vertices, columns=['x','y','z'])  # convert TrackedArray to dataframe


# %%
# make alpha meshes for layers
from utils.plotter import alpha_plane

ME_R_layer_m = [None] * 10
ME_R_layer_bd = [None] * 10
for i in [1,3,5,7,9]:
    pts = ME_R_layer_sec[i-1].values
    ME_R_layer_m[i-1], ME_R_layer_bd[i-1] = alpha_plane(pts, vn, alpha=0.0004)


LO_R_layer_m = [None] * 7
LO_R_layer_bd = [None] * 7
for i in [1,3,5,7]:
    pts = LO_R_layer_sec[i-1].values
    LO_R_layer_m[i-1], LO_R_layer_bd[i-1] = alpha_plane(pts, vn, alpha=0.0004)


LOP_R_layer_m = [None] * 4
LOP_R_layer_bd = [None] * 4
for i in [1,3]:
    pts = LOP_R_layer_sec[i-1].values
    LOP_R_layer_m[i-1], LOP_R_layer_bd[i-1] = alpha_plane(pts, vn, alpha=0.0004)



# %%
# load a neuropil mesh
ME_R = neu.fetch_roi('ME(R)')
LO_R = neu.fetch_roi('LO(R)')
LOP_R = neu.fetch_roi('LOP(R)')

# %%
# slicing neuropil meshes

ME_R_sec = ME_R.section(vn, vt1)
# convert TrackedArray to dataframe
ME_R_sec = pd.DataFrame(ME_R_sec.vertices, columns=['x','y','z'])

LO_R_sec = LO_R.section(vn, vt1)
# convert TrackedArray to dataframe
LO_R_sec = pd.DataFrame(LO_R_sec.vertices, columns=['x','y','z'])

LOP_R_sec = LOP_R.section(vn, vt1)
# convert TrackedArray to dataframe
LOP_R_sec = pd.DataFrame(LOP_R_sec.vertices, columns=['x','y','z'])

# %%
# alpha meshes for neuropils
ME_R_m, ME_R_bd = alpha_plane(ME_R_sec.values, vn, alpha=0.0004) 
LO_R_m, LO_R_bd = alpha_plane(LO_R_sec.values, vn, alpha=0.0004)
LOP_R_m, LOP_R_bd = alpha_plane(LOP_R_sec.values, vn, alpha=0.0004)

# %% [markdown]
# ### Testing plot to select slicing planes, and final plot

# %%
# planes
data_pl1 = {
    'type': 'mesh3d',
    'x': pl_rot1['x'],
    'y': pl_rot1['y'],
    'z': pl_rot1['z'],
    'delaunayaxis':'x',
    'color': 'red',
    'opacity': 0.5,
}

data_pl2 = {
    'type': 'mesh3d',
    'x': pl_rot2['x'],
    'y': pl_rot2['y'],
    'z': pl_rot2['z'],
    'delaunayaxis':'x',
    'color': 'magenta',
    'opacity': 0.5,
}

# whole skeleton
fig_n_ske = navis.plot3d(
    neu_ske,
    soma=False,
    color='blue', linewidth=2,
    inline=False, backend='plotly')

# sliced skeleton and mesh
fig_n_ske_in = navis.plot3d(
    ske_in,
    soma=False,
    color='black', linewidth=2,
    inline=False, backend='plotly')

fig_n_msh_in = navis.plot3d(
    msh_in,
    soma=False,
    color='black', linewidth=2,
    inline=False, backend='plotly')

# whole neuropils
# fig_mesh = navis.plot3d([ME_R_sec, ME_R, LO_R, LOP_R], color=['gray', 'yellow','yellow','yellow'], alpha=0.4, inline=False, backend='plotly')
# fig_mesh = navis.plot3d([LOP_R_layer[0]], color=['yellow'], alpha=0.4, inline=False, backend='plotly')

# sliced neuropils
fig_mesh_slice = navis.plot3d([ME_R_layer_m, LO_R_layer_m, LOP_R_layer_m], 
                              color=['gray']*10 + ['gray']*7 + ['gray']*4, 
                              alpha=0.4, inline=False, backend='plotly')


fig_outline_ME = go.Figure(data=go.Scatter3d(x=ME_R_bd[:,0], y=ME_R_bd[:,1], z=ME_R_bd[:,2],
                                  mode='lines', line=dict(color='gray', width=3)
                                  ))

fig_outline_LO = go.Figure(data=go.Scatter3d(x=LO_R_bd[:,0], y=LO_R_bd[:,1], z=LO_R_bd[:,2],
                                  mode='lines', line=dict(color='gray', width=3)
                                  ))
fig_outline_LOP = go.Figure(data=go.Scatter3d(x=LOP_R_bd[:,0], y=LOP_R_bd[:,1], z=LOP_R_bd[:,2],
                                  mode='lines', line=dict(color='gray', width=3)
                                  ))



# %%
# testing plot, choose slicing planes
fig = go.Figure(
    fig_n_msh_in.data
    # fig_col.data #medulla column
    # + fig_mesh.data #whole meshes
    # + surf.data 
    # + fig_n_ske.data
    # + go.Figure(data=[data_pl1,data_pl2]).data #slice planes +
    # + go.Figure(data=[data_neupil2, data_neupil3]).data
    + fig_mesh_slice.data
    + fig_outline_ME.data + fig_outline_LO.data + fig_outline_LOP.data
)

# # final plot
# fig = go.Figure(
#     # fig_n_ske_in.data
#     fig_n_msh_in.data
#     # +fig_n_ske.data
#     # + fig_mesh.data #whole meshes
#     + go.Figure(data=[data_neupil1, data_neupil2, data_neupil3, data_neupil4, data_neupil5]).data
# )



# fig.update_layout(autosize=False, width=900, height=600)
# fig.update_layout(margin={"l":0, "r":0, "b":0, "t":0})

camera_distance = 70000
fig.update_scenes(
    camera={
        # "up": {"x":-5, "y":3, "z":-5}
        "up": {"x":-4, "y":3, "z":-4}
        , "eye": {
            "x":vn[0]*camera_distance
            , "y":vn[1]*camera_distance
            , "z":vn[2]*camera_distance}
        , "center": {"x":0, "y":0, "z":0}
        , "projection": {"type": "orthographic"}}
)

fig.update_layout(
        margin={'l':0, 'r':0, 'b':0, 't':0}
      , showlegend=True
      , scene = {
            "aspectmode": "auto",
            # "aspectratio": {'x': 1, 'y': 1, 'z':1},
            "xaxis" : {
                "showgrid": False
              , "showbackground": False
              , "visible": False}
          , "yaxis" : {
                "showgrid": False
              , "showbackground": False
              , "visible": False}
          , "zaxis" : {
                "showgrid":False
              , "showbackground":False
              , "visible":False}
    })

fig.show()

# %%
from utils.plotter import show_figure
from utils.neuroglancer_plotter import image_saver
from PIL import Image
import io

#my_img = show_figure(fig, width=3000, height=3000, static=True, showlegend=False )

fig.update_layout(showlegend=False)
img_bytes = fig.to_image(
    format="png"
  , width=3000, height=3000
  , scale=1)


img = Image.open(io.BytesIO(img_bytes))
image_saver(img, "test", Path("."))

# %%
img = Image.open(io.BytesIO(img_bytes))

# %% [markdown]
# todo @Franck
#
# 1/ find the desired cell of given type
# 2/ left-right flip in the final plot -- lobula plate best be on the right side
# 3/ save
