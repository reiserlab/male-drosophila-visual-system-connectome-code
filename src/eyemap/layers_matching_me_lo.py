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
from pathlib import Path
import sys
from dotenv import load_dotenv, find_dotenv
load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath('src')))
print(f"Project root directory: {PROJECT_ROOT}")

import pandas as pd

from utils import olc_client
from utils.layer_tools import \
    fetch_neuron_pairs\
    , merge_and_color, hexify_med_lob\
    , plot_med_lob, plot_layers

c = olc_client.connect(verbose=True)

# %%
# Lobula Layer 1
#
#     med: L2-pre Tm1-post    lob:Tm1-pre T5-post

l2_tm1_post, l2_tm1_post_conns = fetch_neuron_pairs('L2', 'Tm1')

Tm1_T5, Tm1_T5_conns = fetch_neuron_pairs('Tm1', 'T5[abcd]'
  , group_neuron="bodyId_pre", coord_suffix="_lob")

# Lobula Layer 2
#
#     med: L2-pre Tm2-post    lob:Tm2-pre LC4-post

l2_tm2_post, l2_tm2_post_conns = fetch_neuron_pairs('L2', 'Tm2')

tm2_LC4, tm2_LC4_conns = fetch_neuron_pairs('Tm2', 'LC4'
  , group_neuron="bodyId_pre", coord_suffix="_lob")

# Lobula Layer 3
#     med: Mi1-pre T3-post    lob:T3-pre LC11-post

Mi1_T3_post, Mi1_T3_post_conns = fetch_neuron_pairs('Mi1', 'T3')

T3_LC11, T3_LC11_conns = fetch_neuron_pairs('T3', 'LC11'
  , group_neuron="bodyId_pre", coord_suffix="_lob")

# Lobula Layer 4
#
#     med: L2-pre Tm4-post
#     lob:Tm4-pre LC4-post
#     ~also~ 
#     med: L5-pre Tm6/14-post
#     lob:Tm6/14-pre LC11-post

l2_tm4_post, l2_tm4_post_conns = fetch_neuron_pairs('L2', 'Tm4')

tm4_LC4, tm4_LC4_conns = fetch_neuron_pairs('Tm4', 'LC4'
  , group_neuron="bodyId_pre", coord_suffix="_lob")

L5_tm614_post, L5_tm614__post_conns = fetch_neuron_pairs('L5', 'Tm6/14')

tm614_LC11, tm614_LC11_conns = fetch_neuron_pairs('Tm6/14', 'LC11'
  , group_neuron="bodyId_pre", coord_suffix="_lob")


# Lobula Layer 5B
#
#     med: L3-pre Tm20-post     lob:Tm20-pre LC16-post
#     ~also~                    lob:Tm20-pre LPLC2-post

l3_tm20_post, l3_tm20_post_conns = fetch_neuron_pairs('L3', 'Tm20')

tm20_LC16, tm20_LC16_conns = fetch_neuron_pairs('Tm20', 'LC16'
  , group_neuron="bodyId_pre", coord_suffix="_lob")

tm20_LPLC2, tm20_LPLC2_conns = fetch_neuron_pairs('Tm20', 'LPLC2'
  , group_neuron="bodyId_pre", coord_suffix="_lob")

# %%
## Other Lobula Layer 5B pairs

tm20_LPLC1, tm20_LPLC1_conns = fetch_neuron_pairs('Tm20', 'LPLC1'
  , group_neuron="bodyId_pre", coord_suffix="_lob")

tm20_Tm5Y, tm20_Tm5Y_conns = fetch_neuron_pairs('Tm20', 'Tm5Y'
  , group_neuron="bodyId_pre", coord_suffix="_lob")

tm20_LC10c, tm20_LC10c_conns = fetch_neuron_pairs('Tm20', 'LC10c'
  , group_neuron="bodyId_pre", coord_suffix="_lob")

tm20_Li20, tm20_Li20_conns = fetch_neuron_pairs('Tm20', 'Li20'
  , group_neuron="bodyId_pre", coord_suffix="_lob")

tm20_Li22, tm20_Li22_conns = fetch_neuron_pairs('Tm20', 'Li22'
  , group_neuron="bodyId_pre", coord_suffix="_lob")


# %%
# Lobula layer 5B: mean of LC16, LPLC2, LPLC1, LC10c and Li20 together
#     also added Tm5Y and Li22

tm20_x_conns = pd.concat([
    tm20_LC16_conns, tm20_LPLC1_conns, tm20_LPLC2_conns
  , tm20_LC10c_conns, tm20_Li20_conns, tm20_Li22_conns
])

tm20_x = tm20_x_conns\
    .assign(
        x_syn = lambda r: (r.x_pre + r.x_post)/2
      , y_syn = lambda r: (r.y_pre + r.y_post)/2
      , z_syn = lambda r: (r.z_pre + r.z_post)/2)\
    .groupby(by='bodyId_pre')[['x_syn','y_syn','z_syn']]\
    .agg('mean')\
    .reset_index()

tm20_x = tm20_x\
    .rename(columns = {
        "bodyId_pre":"bodyId"
      , "x_syn":"x_lob", "y_syn":"y_lob", "z_syn":"z_lob"
    })

# %%
# Merge medulla and lobula data sets by

# Lobula coordinates are x_lob,y_lob,z_lob

## Lo1
Tm1_T5_for_Lo1 = merge_and_color([l2_tm1_post, Tm1_T5])

## Lo2
Tm2_LC4_for_Lo2 = merge_and_color([l2_tm2_post, tm2_LC4])

## Lo3
T3_LC11_for_Lo3 = merge_and_color([Mi1_T3_post, T3_LC11])

## Lo4
Tm4_LC4_for_Lo4 = merge_and_color([l2_tm4_post, tm4_LC4])
Tm614_LC11_for_Lo4 = merge_and_color([L5_tm614_post, tm614_LC11])

## Lo5B
Tm20_LPLC2_for_Lo5 = merge_and_color([l3_tm20_post, tm20_LPLC2])
Tm20_LPLC1_for_Lo5 = merge_and_color([l3_tm20_post, tm20_LPLC1])
Tm20_LC16_for_Lo5 = merge_and_color([l3_tm20_post, tm20_LC16])
Tm20_x_for_Lo5 = merge_and_color([l3_tm20_post, tm20_x])

# %%
#add hex coordinates to data frames

## Lo1
Tm1_T5_for_Lo1_Hex = hexify_med_lob(Tm1_T5_for_Lo1)

## Lo2
Tm2_LC4_for_Lo2_Hex = hexify_med_lob(Tm2_LC4_for_Lo2)

## Lo5B
Tm20_LC16_for_Lo5_Hex = hexify_med_lob(Tm20_LC16_for_Lo5)

Tm20_LPLC2_for_Lo5_Hex = hexify_med_lob(Tm20_LPLC2_for_Lo5)

Tm20_LPLC1_for_Lo5_Hex = hexify_med_lob(Tm20_LPLC1_for_Lo5)

Tm20_x_for_Lo5_Hex = hexify_med_lob(Tm20_x_for_Lo5)

# %%
# LOBULA LAYER 1

plot_med_lob(
    Tm1_T5_for_Lo1
  , color_column='regions'
  , figure_title="LOBULA LAYER 1    Medulla: L2-Tm1    Lobula: Tm1-T5"
).show()

# %%
# LOBULA LAYER 2

plot_med_lob(
    Tm2_LC4_for_Lo2
  , color_column='regions'
  , figure_title="LOBULA LAYER 2    Medulla: L2-Tm2     Lobula: Tm2-LC4"
).show()

# %%
# LOBULA LAYER 3

plot_med_lob(
    T3_LC11_for_Lo3
  , color_column='regions'
  , figure_title="LOBULA LAYER 3   Medulla: Mi1-T3 Lobula: T3-LC11"
).show()

# %%
# LOBULA LAYER 4

plot_med_lob(
    Tm4_LC4_for_Lo4
  , color_column='regions'
  , figure_title="LOBULA LAYER 4    Medulla: L2-Tm4       Lobula: Tm4-LC4"
).show()

plot_med_lob(
    Tm614_LC11_for_Lo4
  , color_column='regions'
  , figure_title="LOBULA LAYER 4    Medulla: L5-Tm6/14    Lobula: Tm6/14-LC11"
).show()

# %%
# LOBULA LAYER 5B

plot_med_lob(
    Tm20_LC16_for_Lo5
  , color_column='regions'
  , figure_title="LOBULA LAYER 5B    Medulla: L3-Tm20    Lobula: Tm20-LC16"
).show()

plot_med_lob(
    Tm20_LPLC2_for_Lo5
  , color_column='regions'
  , figure_title="LOBULA LAYER 5B    Medulla: L3-Tm20    Lobula: Tm20-LPLC2"
).show()

plot_med_lob(
    Tm20_LPLC1_for_Lo5
  , color_column='regions'
  , figure_title="LOBULA LAYER 5B    Medulla: L3-Tm20    Lobula: Tm20-LPLC1"
).show()

plot_med_lob(
    Tm20_x_for_Lo5
  , color_column='regions'
  , figure_title="LOBULA LAYER 5B    Medulla: L3-Tm20    Lobula: Tm20-X"
).show()

# %%
# Plot Lobula Layer "Rainbow"

plot_layers({
    'Lo1:Tm1-T5': Tm1_T5, 'Lo2:tm2-LC4': tm2_LC4, 'Lo3:T3-LC11': T3_LC11
  , 'Lo4:Tm4-LC4': tm4_LC4, 'Lo4:Tm6/14-LC4': tm614_LC11
  , 'Lo5B:Tm20-LC16': tm20_LC16, 'Lo5B:Tm20-LPLC1': tm20_LPLC1
  , 'Lo5B:Tm20-LPLC2': tm20_LPLC2
})



# %%
# Lobula Layer 1

plot_med_lob(
    Tm1_T5_for_Lo1_Hex
  , color_column='mod_for_color'
  , figure_title="Lobula Layer1  Med: L2-Tm1 Lob: Tm1-T5 Stripes:Hex1"
).show()

plot_med_lob(
    Tm1_T5_for_Lo1_Hex
  , color_column='mod_for_color'
  , figure_title="Lobula Layer1  Med: L2-Tm1 Lob: Tm1-T5 Stripes:Hex1"
  , show_medulla=False
).show()

plot_med_lob(
    Tm1_T5_for_Lo1_Hex
  , color_column='mod_for_color2'
  , figure_title="Lobula Layer1  Med: L2-Tm1 Lob: Tm1-T5 Stripes:Hex2"
  , show_medulla=False
).show()

# %%
# Lobula Layer 2

plot_med_lob(
    Tm2_LC4_for_Lo2_Hex
  , color_column='mod_for_color'
  , figure_title="Lobula Layer 2  Med: L2-Tm2 Lob: Tm2-LC4 Stripes:Hex1"
  , show_medulla=False
).show()

plot_med_lob(
    Tm2_LC4_for_Lo2_Hex
  , color_column='mod_for_color2'
  , figure_title="Lobula Layer 2  Med: L2-Tm2 Lob: Tm2-LC4 Stripes:Hex2"
  , show_medulla=False
).show()

# %%
# Lobula Layer 5B

plot_med_lob(
    Tm20_LC16_for_Lo5_Hex
  , color_column='mod_for_color'
  , figure_title="Lobula Layer 5B  Med: L3-Tm20 Lob: Tm20-LC16 Stripes:Hex1"
).show()

plot_med_lob(
    Tm20_LC16_for_Lo5_Hex
  , color_column='mod_for_color'
  , figure_title="Lobula Layer 5B  Med: L3-Tm20 Lob: Tm20-LC16 Stripes:Hex1"
  , show_medulla=False
).show()

plot_med_lob(
    Tm20_LC16_for_Lo5_Hex
  , color_column='mod_for_color2'
  , figure_title="Lobula Layer 5B  Med: L3-Tm20 Lob: Tm20-LC16 Stripes:Hex2"
  , show_medulla=False
).show()

# %%
# Lobula Layer 5B

plot_med_lob(
    Tm20_LPLC2_for_Lo5_Hex
  , color_column='mod_for_color'
  , figure_title="Lobula Layer 5B  Med: L3-Tm20 Lob: Tm20-LPLC2 Stripes:Hex1"
  , show_medulla=False
).show()

plot_med_lob(
    Tm20_LPLC2_for_Lo5_Hex
  , color_column='mod_for_color2'
  , figure_title="Lobula Layer 5B  Med: L3-Tm20 Lob: Tm20-LPLC2 Stripes:Hex2"
  , show_medulla=False
).show()

# %%
# Lobula Layer 5B

plot_med_lob(
    Tm20_LPLC1_for_Lo5_Hex
  , color_column='mod_for_color'
  , figure_title="Lobula Layer 5B  Med: L3-Tm20 Lob: Tm20-x Stripes:Hex1"
  , show_medulla=False
).show()

plot_med_lob(
    Tm20_LPLC1_for_Lo5_Hex
  , color_column='mod_for_color2'
  , figure_title="Lobula Layer 5B  Med: L3-Tm20 Lob: Tm20-x Stripes:Hex2"
  , show_medulla=False
).show()

# %%
# Lobula Layer 5B

plot_med_lob(
    Tm20_x_for_Lo5_Hex
  , color_column='mod_for_color'
  , figure_title="Lobula Layer 5B  Med: L3-Tm20 Lob: Tm20-x Stripes:Hex1"
  , show_medulla=False
).show()

plot_med_lob(
    Tm20_x_for_Lo5_Hex
  , color_column='mod_for_color2'
  , figure_title="Lobula Layer 5B  Med: L3-Tm20 Lob: Tm20-x Stripes:Hex2"
  , show_medulla=False
).show()

# %%
