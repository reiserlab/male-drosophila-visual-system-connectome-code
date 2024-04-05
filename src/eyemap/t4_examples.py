# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: ol-connectome
#     language: python
#     name: python3
# ---

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
import pickle
import logging
from dotenv import load_dotenv, find_dotenv
load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath('src')))
print(f"Project root directory: {PROJECT_ROOT}")

# data import & manipulation
import numpy as np
from neuprint import NeuronCriteria as NC
from neuprint import fetch_simple_connections, fetch_synapse_connections

# plotting
import plotly
import plotly.graph_objects as go
import navis
import navis.interfaces.neuprint as neu
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

# Optic Lobe libraries
from utils.plotter import get_skeleton, get_mesh
from utils import olc_client

# %%
## Setup
logging.getLogger('navis').setLevel(logging.WARN)

c = olc_client.connect(verbose=True)

data_dir = PROJECT_ROOT / 'results' / 'eyemap'
result_dir = data_dir / 'T4_example_plot'
result_dir.mkdir(parents=True, exist_ok=True)

# %%
# load medulla column map

with open( data_dir / 'ME_pincushion_df.pickle', 'rb') as f:
    xyzpq = pickle.load(f) # [p,q] are reversed wrt eyemap paper, ie [q,p]

# column xyz
col_xyz = xyzpq[['x_top','y_top','z_top']].values

# %%
# example T4b
T4_BID = 114698

fetch_simple_connections(NC(type=['Mi1','Tm3','Mi4','Mi9']), NC(bodyId=T4_BID))

# %%
# get post-synapse xyz
syn_mi1 = fetch_synapse_connections(NC(type= 'Mi1'), NC(bodyId=T4_BID))
syn_tm3 = fetch_synapse_connections(NC(type= 'Tm3'), NC(bodyId=T4_BID))
syn_mi4 = fetch_synapse_connections(NC(type= 'Mi4'), NC(bodyId=T4_BID))
syn_mi9 = fetch_synapse_connections(NC(type= 'Mi9'), NC(bodyId=T4_BID))

# %%
# get T4 skel
ske_t4 = get_skeleton(T4_BID)  # FL: use function with cache etc…
ske_t4[0].n_trees

# %%
# ME portion
me_r = neu.fetch_roi('ME(R)')
ske_t4_me = ske_t4.prune_by_volume(me_r)
com_xyz = ske_t4_me.nodes[['x','y','z']].mean().values

# %%
# strahler
n = ske_t4_me[0]
_ = navis.strahler_index(n)

# %%
# find nearest med columns
ii = np.argsort(np.sum((col_xyz - com_xyz)**2, axis=1))[:9]

# %%
fig = navis.plot3d([ske_t4, ske_t4_me, me_r, col_xyz[ii,:]])

# %%
# rotate for 2D plot
# svd/pca
xyz_recenter = col_xyz[ii,:].mean(0)[np.newaxis,:] #center of pca
col_xyz_centered = col_xyz[ii,:] - xyz_recenter
u, s, v = np.linalg.svd(col_xyz_centered.T)
med_nb_pc = col_xyz_centered @ u
nodes_pc = (n.nodes[['x','y','z']].values - xyz_recenter) @ u

n_pc = n.copy()
n_pc.nodes[['x','y','z']] = nodes_pc

syn_mi1_pc = (syn_mi1[['x_post','y_post','z_post']].values - xyz_recenter) @ u
syn_tm3_pc = (syn_tm3[['x_post','y_post','z_post']].values - xyz_recenter) @ u
syn_mi4_pc = (syn_mi4[['x_post','y_post','z_post']].values - xyz_recenter) @ u
syn_mi9_pc = (syn_mi9[['x_post','y_post','z_post']].values - xyz_recenter) @ u

# %%
# get mesh
t4_mesh = get_mesh(T4_BID)
# in med
t4_mesh_me = navis.in_volume(t4_mesh, me_r)

# %%
# plot whole neuron mesh

fig = navis.plot3d(t4_mesh, color=(166, 97, 26, 0.8), inline=False)

fig.update_layout(
    width=1200, height=900
  , template='simple_white'
  , paper_bgcolor='#FFFFFF'
  , autosize=False
  # , template='none'
  , scene={
        'camera': {
            'up': {'x': 0.29, 'y': 2.3, 'z': -0.09}
          , 'center': {'x': 0, 'y': 0, 'z': 0}
          , 'eye': {'x': -u[0,2]*1.7, 'y': -u[1,2]*1.7, 'z': -u[2,2]*1.7}
        }
      , 'xaxis': {'visible': False}
      , 'yaxis': {'visible': False}
      , 'zaxis': {'visible': False}
  }
)

fig.show()

# plotly.offline.plot(fig, filename=str(result_dir / f"T4b_{T4_BID}_color.html")) # save interactive
# fig.write_image(result_dir / f"T4b_{T4_BID}.png", width=2000, height=2000) # save static


# %%
# plot, add synapses

COL_SIZE = 240/3
SYN_SIZE = 15/3

fig = navis.plot3d(t4_mesh_me, color=(200, 200, 200, 0.6), inline=False)

xyz = col_xyz[ii,:] + u[:,2]*5e2

fig.add_trace(
    go.Scatter3d(
        x=xyz[:,0], y=xyz[:,1], z=xyz[:,2]
      , name="Columns"
      , mode="markers"
      , marker={
          'symbol': "circle"
        , 'size': COL_SIZE
        , 'color': 'pink'
        , 'opacity': 0.2}
))

fig_data = {
    'Mi1': {'color': "#9F2735", 'df': syn_mi1}
  , 'Tm3': {'color': "#E49F3D", 'df': syn_tm3}
  , 'Mi4': {'color': "#3577AB", 'df': syn_mi4}
  , 'Mi9': {'color': "#48AA9A", 'df': syn_mi9}
}

for name, options in fig_data.items():
    xyz = options['df'][['x_post','y_post','z_post']].values
    fig.add_trace(
        go.Scatter3d(
            x=xyz[:,0], y=xyz[:,1], z=xyz[:,2]
          , name=name
          , mode='markers'
          , marker={'color': options['color'], 'size': SYN_SIZE}
        )
    )

fig.update_layout(
    width=1200, height=900
  , template='simple_white'
  , paper_bgcolor="#FFFFFF"
  , autosize=False
  , scene={
        'xaxis': {'visible': False}
      , 'yaxis': {'visible': False}
      , 'zaxis': {'visible': False}
      , 'camera': {
            'up': {'x': 0.29, 'y': 2.3, 'z': -0.09}
          , 'center': {'x': 0, 'y': 0, 'z': 0}
          , 'eye': {'x': -0.74*1.7, 'y': 0.05*1.7, 'z': -0.66*1.7}
        }
    }
)

fig.show()

# plotly.offline.plot(fig, filename=str(result_dir / f"T4b_{T4_BID}_syn.html"))
# fig.write_image(result_dir / f"T4b_{T4_BID}_syn.png", width=2000, height=2000)


# %%
# play with the camera, eg., can choose 2 points to set the "up" value

fig.update_scenes(
    camera = {
        'up': {'x': 0.29, 'y': 2.3, 'z': -0.09}
      , 'center': {'x': 0, 'y': 0, 'z': 0}
      , 'eye': {'x': -0.74, 'y': 0.05, 'z': -0.66}
    }
)
fig.show()

# %%
# PLOT, skeleton with synapses

fig = navis.plot3d(n_pc, color_by='strahler_index', palette='viridis', inline=False)

fig_data = {
    'Column': {'color': "gray", 'size': 10, 'alpha': 0.1, 'df': med_nb_pc}
  , 'Mi1': {'color': "#9F2735", 'size': 3 , 'alpha': 0.8, 'df': syn_mi1_pc}
  , 'Tm3': {'color': "#E49F3D", 'size': 3 , 'alpha': 0.8, 'df': syn_tm3_pc}
  , 'Mi4': {'color': "#3577AB", 'size': 3 , 'alpha': 0.8, 'df': syn_mi4_pc}
  , 'Mi9': {'color': "#48AA9A", 'size': 3 , 'alpha': 0.8, 'df': syn_mi9_pc}
}

for name, options in fig_data.items():
    fig.add_trace(
        go.Scatter3d(
            x=options['df'][:,0], y=options['df'][:,1], z=options['df'][:,2]
          , name=name
          , mode='markers'
          , marker={'color': options['color'], 'size': options['size']}
        )
    )

fig.update_layout(
    width=1200, height=600
  , paper_bgcolor="#FFFFFF"
  , autosize=False
  , scene_aspectmode='data'
)

fig.show()

# %%
# save 2D pdf

pp = PdfPages(Path(result_dir, 'T4b.pdf'))

fig, ax = navis.plot2d(
    n_pc
  , method='3d'
  , color= 'black'
  , alpha =0.5
  # , color_by='strahler_index',  # color based on Strahler index column
  # , shade_by='strahler_index',  # shade (alpha) based on Strahler index column
  # , palette='cool',             # accepts any matplotlib palette
  , linewidth= 3
)

for name, options in fig_data.items():
    ax.plot(
        options['df'][:,0], options['df'][:,1]
      , color=options['color']
      , marker='o'
      , ls=''
      , alpha=options['alpha']
      , markersize=options['size']*5)

ax.elev = ax.azim = -90
# ax.dist = 6 3 this controls zoom
pp.savefig()
pp.close()



# %%
images = []
for i in range(0, 360, 5):
    # Change rotation
    ax.azim = i
    # Save each incremental rotation as frame
    frame_fn = result_dir / f"frame_{i:03d}.png"
    fig.savefig(frame_fn, dpi=200, transparent=True)
    images.append(Image.open(frame_fn).convert("RGB"))

images[0].save(
   result_dir / "T4b.gif"
 , save_all = True
 , append_images = images[1:]
 , optimize = True
 , duration = 30
 , loop=0)


# %%
