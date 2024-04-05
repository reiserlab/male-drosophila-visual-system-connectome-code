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
from pathlib import Path
import sys

import pandas as pd

from dotenv import load_dotenv, find_dotenv
load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath('src')))
print(f"Project root directory: {PROJECT_ROOT}")
data_path = Path(PROJECT_ROOT, 'results', 'eyemap')

from utils.layer_tools import\
    fetch_neuron_pairs\
  , hexify_med_lob\
  , merge_and_color\
  , get_com_and_hex

from utils import olc_client
c = olc_client.connect(verbose=True)

# %% [markdown]
# ## Medulla: ME2, 3, 9, 10

# %%
layers = {'ME2':['L2', 'Tm1'], 'ME10': ['Mi1', 'T4[abcd]']}

for layer, neurons in layers.items():
    hexgrid = get_com_and_hex(neurons[0], neurons[1], pickle_path=data_path)
    layer_fn = data_path / f"{layer}_hex.pickle"
    hexgrid.to_pickle(layer_fn)

# %% [markdown]
# ## Lobula: LO1, 5B

# %%
#Lobula Layer 1
l2_tm1, _ = fetch_neuron_pairs('L2', 'Tm1')
tm1_t5, _ = fetch_neuron_pairs('Tm1', 'T5[abcd]'
     , group_neuron="bodyId_pre", coord_suffix="_lob")

tm1_t5_for_lo1 = merge_and_color([l2_tm1, tm1_t5])

lo1_hex = hexify_med_lob(
    data_path / "Tm1.pickle", tm1_t5_for_lo1)

lo1_hex = lo1_hex[['bodyId','x_lob','y_lob','z_lob','hex1_id','hex2_id']]\
    .rename(columns={'x_lob':'x','y_lob':'y','z_lob':'z'})

lo1_hex.to_pickle( data_path / "LO1_hex.pickle")

# %%
#Lobula Layer 5B
l3_tm20, _ = fetch_neuron_pairs('L3', 'Tm20')
tm20_lc16, _ = fetch_neuron_pairs('Tm20', 'LC16'
     , group_neuron="bodyId_pre", coord_suffix="_lob")

# %%
tm20_lc16_for_lo5 = merge_and_color([l3_tm20, tm20_lc16])

lo5_hex = hexify_med_lob(
    data_path / 'L3_to_Tm20.pickle', tm20_lc16_for_lo5)

lo5_hex = lo5_hex\
    [['bodyId', 'x_lob', 'y_lob', 'z_lob', 'hex1_id', 'hex2_id']]\
    .rename(columns={'x_lob':'x', 'y_lob':'y', 'z_lob':'z'})

lo5_hex.to_pickle( data_path / "LO5_hex.pickle")

# %% [markdown]
# ## Lobula Plate LOP1, 4

# %%
mi1_t4_df = pd.read_pickle(data_path / 'Mi1_T4_align.pickle')
mi1_df = pd.read_pickle(data_path / 'Mi1.pickle')

# %%
t4_hex_df = mi1_t4_df[mi1_t4_df['valid_group']==1]\
    [['mi1_bid','t4a_bid','t4b_bid','t4c_bid','t4d_bid']]\
    .dropna(subset=['t4a_bid', 't4d_bid'])\
    .astype({
        'mi1_bid':'int'
      , 't4a_bid': 'int'
      , 't4d_bid': 'int'})\
    .merge(
        mi1_df[['bodyId_pre','hex1_id','hex2_id']]
      , left_on='mi1_bid', right_on='bodyId_pre')

cache_path = PROJECT_ROOT / "cache" / "layers"
cache_path.mkdir(parents=True, exist_ok=True)

t4_hex_df\
  .rename(columns={'t4a_bid':'bodyId'})\
  .set_index('bodyId')[['hex1_id', 'hex2_id']]\
  .to_pickle(cache_path / "T4a.pickle")

t4_hex_df\
  .rename(columns={'t4d_bid':'bodyId'})\
  .set_index('bodyId')[['hex1_id', 'hex2_id']]\
  .to_pickle(cache_path / "T4d.pickle")

# %%
t4a_hex = get_com_and_hex("T4a", None, cache_path, src_roi="LOP(R)")
t4a_hex.to_pickle( data_path / "LOP1_hex.pickle")
# Remove cached file
(cache_path / "T4a.pickle").unlink()


# %%
t4d_hex = get_com_and_hex("T4d", None, cache_path, src_roi="LOP(R)")
t4d_hex.to_pickle( data_path / "LOP4_hex.pickle")
# Remove cached file
(cache_path / "T4d.pickle").unlink()
