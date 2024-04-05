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

# %% Project setup
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
# plotting setup
import plotly.graph_objects as go

# %%
import numpy as np
from cmath import pi

from utils.geometry import cart2sph, sph2cart
from utils.geometry import sph2Mercator

# %% [markdown]
# ### Transformation between Cartesian <-> spherical coordinates 

# %%
# define a few points in [theta, phi] in degree
pts = np.array([[45,0], [90,0], [45,45], [90,45], [135,-135], [135,-45]])
# convert to spherical coordinate in [r=1, theta, phi] in radian
rtp = np.insert(pts/180*pi, 0, np.repeat(1, pts.shape[0]), axis=1)

# convert to Cartesian
xyz = sph2cart(rtp)
# convert to spherical
rtp = cart2sph(xyz)

# %% [markdown]
# ### Mollweide projections
# https://mathworld.wolfram.com/MollweideProjection.html

# %%
from utils.geometry import sph2Mollweide

# define a few points in [theta, phi] in degree
pts = np.array([[45,0], [90,0], [45,45], [90,45], [135,-135], [135,-45]])
# convert to spherical coordinate in [r=1, theta, phi] in radian
rtp = np.insert(pts/180*pi, 0, np.repeat(1, pts.shape[0]), axis=1)

# Mollweide projection, only use theta and phi
xy = sph2Mollweide(rtp[:,1:3])

# Plot
# define Mollweide guidelines
ww = np.stack((np.linspace(0,180,19), np.repeat(-180,19)), axis=1) # west \ left boundary
w = np.stack((np.linspace(180,0,19), np.repeat(-90,19)), axis=1)
m = np.stack((np.linspace(0,180,19), np.repeat(0,19)), axis=1) # central meridian
e = np.stack((np.linspace(180,0,19), np.repeat(90,19)), axis=1)
ee = np.stack((np.linspace(0,180,19), np.repeat(180,19)), axis=1) # east \ right boundary
pts = np.vstack((ww,w,m,e,ee))
rtp = np.insert(pts/180*pi, 0, np.repeat(1, pts.shape[0]), axis=1) # add unit radius if to make 3d plot
meridians_xy = sph2Mollweide(rtp[:,1:3])

pts = np.stack((np.repeat(45,37), np.linspace(-180,180,37)), axis=1) # 37 points on 45 degree north latitude
rtp = np.insert(pts/180*pi, 0, np.repeat(1, pts.shape[0]), axis=1)
n45_xy = sph2Mollweide(rtp[:,1:3])
pts = np.stack((np.repeat(90,37), np.linspace(-180,180,37)), axis=1) # 37 points on equator
rtp = np.insert(pts/180*pi, 0, np.repeat(1, pts.shape[0]), axis=1)
eq_xy = sph2Mollweide(rtp[:,1:3])
pts = np.stack((np.repeat(135,37), np.linspace(-180,180,37)), axis=1) # 37 points on 45 degree south latitude
rtp = np.insert(pts/180*pi, 0, np.repeat(1, pts.shape[0]), axis=1)
s45_xy = sph2Mollweide(rtp[:,1:3])


# plotly plot
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=meridians_xy[:,0], y=meridians_xy[:,1]
      , mode='lines', name=''
      , line={'color':'black', 'width':1}
    )
)
fig.add_trace(
    go.Scatter(
        x=eq_xy[:,0], y=eq_xy[:,1]
      , mode='lines', name=''
      , line={'color':'black', 'width':1}
    )
)
fig.add_trace(
    go.Scatter(
        x=s45_xy[:,0], y = s45_xy[:,1]
      , mode='lines', name=''
      , line={'color': 'black', 'width': 1}
    )
)
fig.add_trace(
    go.Scatter(
        x=n45_xy[:,0], y = n45_xy[:,1]
      , mode='lines', name=''
      , line={'color': 'black', 'width': 1}
    )
)

fig.add_trace(
    go.Scatter(
        x=xy[:,0], y=xy[:,1]
      , mode='markers', name=''
      , marker={'color': 'blue', 'size': 10}
    )
)

fig.update_layout(showlegend=False)
fig.update_xaxes(title_text='azimuth')
fig.update_yaxes(title_text='elevation')

fig.show()


# %% [markdown]
# ### Mercator projections

# %%


# define a few points in [theta, phi] in degree
pts = np.array([[45,0], [90,0], [45,45], [90,45], [135,-135], [135,-45]])
# convert to spherical coordinate in [r=1, theta, phi] in radian
rtp = np.insert(pts/180*pi, 0, np.repeat(1, pts.shape[0]), axis=1)
xy = sph2Mercator(rtp[:,1:3])


# plotly plot
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=xy[:,0], y=xy[:,1]
      , mode='markers', name=''
      , marker={'color':'blue', 'size':10}
    )
)

# set x lim
fig.update_xaxes(range=[-np.pi, np.pi])
fig.update_xaxes(
    tickvals= np.arange(-180, 180, step=45)/180*np.pi
  , ticktext= np.arange(-180, 180, step=45) 
)
fig.update_xaxes(title_text='azimuth')
# set y lim
fig.update_yaxes(
    range=[
        np.log(np.tan(np.pi/4 -75/180*np.pi/2))
      , np.log(np.tan(np.pi/4 +75/180*np.pi/2))
    ]
)
fig.update_yaxes(
    tickvals=np.log(np.tan(np.pi/4 - np.arange(-75, 75, step=15)/180*np.pi/2))
  , ticktext=-np.arange(-75, 75, step=15) )
fig.update_yaxes(title_text='elevation')

fig.update_layout(showlegend=False)
fig.show()
