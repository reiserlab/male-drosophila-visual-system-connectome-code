# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
from pathlib import Path
from IPython.display import display
import sys
from dotenv import load_dotenv, find_dotenv
load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath('src')))
print(f"Project root directory: {PROJECT_ROOT}")

from utils import olc_client
from utils.layer_tools import \
    fetch_neuron_pairs\
  , hexify_med_lob\
  , plot_med_lob\
  , merge_and_color

c = olc_client.connect(verbose=True)

# %%
#Mi1 connections in Medulla layer 1,3,5,9 and 10

#Medulla layer 1: L1 & Mi1  -note "_ME1" suffix
l1_Mi1_post, l1_Mi1_post_conns = fetch_neuron_pairs('L1', 'Mi1', coord_suffix="_ME1")

#Medulla layer 3: L3 & Mi1  -note "_ME3" suffix
l3_Mi1_post, l3_Mi1_post_conns = fetch_neuron_pairs('L3', 'Mi1'
    , group_neuron="bodyId_post", coord_suffix="_ME3")

#Medulla layer 5: L5 & Mi1  -note "_M_5" suffix
l5_Mi1_post, l5_Mi1_post_conns = fetch_neuron_pairs('L5', 'Mi1'
    , group_neuron="bodyId_post", coord_suffix="_ME5")

#Medulla layer 9: Mi1 & Pm2a  -note "_M_9" suffix
Mi1_Pm2a, Mi1_Pm2a_conns = fetch_neuron_pairs('Mi1', 'Pm2a'
    , group_neuron="bodyId_pre", coord_suffix="_ME9")

#Medulla layer 10: Mi1 & T4  -note "_M_10" suffix
Mi1_T4, Mi1_T4_conns = fetch_neuron_pairs('Mi1', 'T4[abcd]'
    , group_neuron="bodyId_pre", coord_suffix="_ME10")

# %%
Mi1_for_M_layers = merge_and_color(
    [l1_Mi1_post, l3_Mi1_post, l5_Mi1_post, Mi1_Pm2a, Mi1_T4]
  , color_by_suffix='_ME1')


# %%
#add hex coordinates to data frames

Mi1_for_M_layers_hex = hexify_med_lob(Mi1_for_M_layers)

# %%
display(Mi1_for_M_layers_hex)

# %%
#Plot layers colored by "regions" in Medulla layer 1

plot_med_lob(
    Mi1_for_M_layers
    , color_column='regions'
    , figure_title="Connect ME Layers 1,3,5,9 and 10. Colored by regions in ME1"
).show()

# %%
#Plot layers colored by Mi1 Hex coordinates "Hex2"

plot_med_lob(
    Mi1_for_M_layers_hex
    , color_column='mod_for_color'
    , figure_title="Connect ME Layers 1,3,5,9 and 10. Colored by Hex1 in ME1"
).show()

# %%
#Plot layers colored by Mi1 Hex coordinates "Hex2"

plot_med_lob(
    Mi1_for_M_layers_hex
    , color_column='mod_for_color2'
    , figure_title="Connect ME Layers 1,3,5,9 and 10. Colored by Hex2 in ME1"
).show()

# %%
