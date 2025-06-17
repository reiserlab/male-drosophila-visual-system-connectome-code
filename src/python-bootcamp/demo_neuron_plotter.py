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
# # Demo Group Plotter
#
# Early plotter (`group_plotter`) that quickly shows morphology and location of a set of neurons. 
# Uses similar interface as the `neuroglancer_plotter`. Not used in the paper.

# %%
from madvisc.utils.plotter import group_plotter, show_figure
from madvisc.utils.neuroglancer_plotter import group_plotter as ng_group_plotter
from madvisc.utils import olc_client

c = olc_client.connect(verbose=True)

# %%
# Plot 4 neurons in 2 different colors

grp1 = group_plotter(
      body_ids=[17871, 20135, 27788, 31492]
    , colors=[(1,0,0,.5), (0,1,0,.5)]
    , camera_distance=1.4
    #, ignore_cache=True
  )

show_figure(grp1)

# %%
grp2 = group_plotter(
      body_ids=[46214, None, 55962, 61400]
    , colors=[(1,0,0,.5), (0,1,0,.5)]
    , plot_roi="ME(R)"
    , prune_roi="ME(R)"
    , camera_distance=1.8
  )

show_figure(
    grp2
  , width=500, height=300
  , static=True
  , showlegend=False)

# %%
img, lnk = ng_group_plotter(
    body_ids=[17871, 20135, 27788, 31492]
  #, colors=[(1,0,0,.5), (0,1,0,.5)]
  , camera_distance=0.8
)
display(img)
# #12000?

# %%
img, lnk  = ng_group_plotter(
          body_ids=[65399, 74994, 75510]
)
display(img)

# %%

ng_group_plotter(
    body_ids=[33418,32198,31940,34549,34233,30811,39519,36926,33584,28480,27552,27318,27532,39197]
)
