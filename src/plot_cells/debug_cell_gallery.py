# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: default
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Alternative neuron plotter
#
# This notebook demonstrates another way of plotting neurons in front of a sliced neuropil.
# For the paper we ended up using a blender based pipeline since it produced better output.
# Here we share an alternative approach that is faster and works directly in the notebook.

# %%
# %load_ext autoreload
"""
This cell does the initial project setup.
If you start a new script or notebook, make sure to copy & paste this part.

A script with this code uses the location of the `.env` file as the anchor for
the whole project (= PROJECT_ROOT). Afterwards, code inside the `src` directory
are available for import.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

import plotly.graph_objects as go
from IPython.display import display

import navis
import navis.interfaces.neuprint as neu

from dotenv import load_dotenv, find_dotenv
load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath('src')))
print(f"Project root directory: {PROJECT_ROOT}")

from utils.hex_hex import hex_to_bids
from utils.plotter import get_meshes, show_figure, alpha_plane, group_plotter
from utils.ng_view import NG_View
from utils.geometry import plane_square

from utils import olc_client

c = olc_client.connect(verbose=True)

# %%
ids = hex_to_bids((18,18), n_types=['L1', 'Mi1'], return_type='list')

# %%
ids = [ids[1]]

# %%
display(ids)

# %%
neu_msh = get_meshes(ids)

# %%
vn = np.array([1, 2, 0]) # normal vector
vn = vn / np.linalg.norm(vn) # normalize
mf = 3e4 # multiplication factor for the plane size
vt1 = np.array([20e3, 40e3, 35e3]) * 0.85 # translation vector
vt2 = np.array([22e3, 44e3, 35e3]) * 0.95 # translation vector

# planes
pl_rot1 = plane_square(vn, vt1, mf)
pl_rot2 = plane_square(vn, vt2, mf)

# %%
# slicing neuron meshes

# which nodes fall within the planes
d1 = vt1 @ vn
d2 = vt2 @ vn

# get the cross section of a neuron

msh_v = pd.DataFrame(neu_msh[0].vertices[0], columns=['x', 'y', 'z'])
ind_msh_in = msh_v.apply(lambda row: row @ vn > d1 and row @ vn < d2, axis=1)
msh_in = navis.subset_neuron(neu_msh[0], ind_msh_in.values, inplace=False)

# %%
# load layer meshes and slice them

ME_R_layer = [None] * 10
ME_R_layer_sec = [None] * 10
# selected layers
for i in [1,3,5,7,9]:
    # load layer meshes with a constructed variable name
    ME_R_layer[i-1] = neu.fetch_roi(f'ME_R_layer_{i:02d}')
    ME_R_layer_sec[i-1] = ME_R_layer[i-1].section(vn, vt1)
    # convert TrackedArray to dataframe
    ME_R_layer_sec[i-1] = pd.DataFrame(ME_R_layer_sec[i-1].vertices, columns=['x','y','z'])

LO_R_layer = [None] * 7
LO_R_layer_sec = [None] * 7
# selected layers
for i in [1,3,5,7]:
    # load layer meshes with a constructed variable name
    LO_R_layer[i-1] = neu.fetch_roi(f'LO_R_layer_{i}')
    LO_R_layer_sec[i-1] = LO_R_layer[i-1].section(vn, vt1)
    # convert TrackedArray to dataframe
    LO_R_layer_sec[i-1] = pd.DataFrame(LO_R_layer_sec[i-1].vertices, columns=['x','y','z'])

LOP_R_layer = [None] * 4
LOP_R_layer_sec = [None] * 4
# selected layers
for i in [1,3]:
    # load layer meshes with a constructed variable name
    LOP_R_layer[i-1] = neu.fetch_roi(f'LOP_R_layer_{i}')
    LOP_R_layer_sec[i-1] = LOP_R_layer[i-1].section(vn, vt1)
    # convert TrackedArray to dataframe
    LOP_R_layer_sec[i-1] = pd.DataFrame(LOP_R_layer_sec[i-1].vertices, columns=['x','y','z'])

# %%
# make alpha meshes for layers

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

fig_n_msh_in = navis.plot3d(
    msh_in,
    soma=False,
    color='black', linewidth=2,
    inline=False, backend='plotly')

# # whole neuropils
# fig_mesh = navis.plot3d(
#     [ME_R_sec, ME_R, LO_R, LOP_R]
#   , color=['gray', 'yellow','yellow','yellow']
#   , alpha=0.4, inline=False, backend='plotly'
# )
# fig_mesh = navis.plot3d(
#     [LOP_R_layer[0]]
#   , color=['yellow']
#   , alpha=0.4, inline=False, backend='plotly'
# )

# sliced neuropils
fig_mesh_slice = navis.plot3d(
    [ME_R_layer_m, LO_R_layer_m, LOP_R_layer_m]
  , color=['gray']*10 + ['gray']*7 + ['gray']*4
  , alpha=0.4, inline=False, backend='plotly')


fig_outline_ME = go.Figure(
    data=go.Scatter3d(
        x=ME_R_bd[:,0], y=ME_R_bd[:,1], z=ME_R_bd[:,2]
      , mode='lines', line={'color': 'gray', 'width': 3}
    )
)

fig_outline_LO = go.Figure(
    data=go.Scatter3d(
        x=LO_R_bd[:,0], y=LO_R_bd[:,1], z=LO_R_bd[:,2]
      , mode='lines', line={'color': 'gray', 'width': 3}
    )
)
fig_outline_LOP = go.Figure(
    data=go.Scatter3d(
        x=LOP_R_bd[:,0], y=LOP_R_bd[:,1], z=LOP_R_bd[:,2]
      , mode='lines', line={'color': 'gray', 'width': 3}
    )
)

# %%
# testing plot, choose slicing planes
fig = go.Figure(
    fig_n_msh_in.data
    + fig_mesh_slice.data
    + fig_outline_ME.data + fig_outline_LO.data + fig_outline_LOP.data
)

# # final plot
# fig = go.Figure(
#     # fig_n_ske_in.data
#     fig_n_msh_in.data
#     # +fig_n_ske.data
#     # + fig_mesh.data #whole meshes
#     + go.Figure(data=[
#           data_neupil1, data_neupil2, data_neupil3, data_neupil4, data_neupil5
#           ]).data
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
display(vn[0]*camera_distance, vn[1]*camera_distance, vn[2]*camera_distance)

# %%
fig2 = group_plotter(
    [56564]
  , colors=[(0.0,0.0,0.0,1.0)]
  , shadow_rois=['ME(R)', 'LO(R)', 'LOP(R)']
  , plot_synapses=False
  , view=NG_View.GALLERY1)
fig2.show()

# %%
show_figure(fig2, static=True, showlegend=False)
