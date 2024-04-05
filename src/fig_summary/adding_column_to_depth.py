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

# %%
from pathlib import Path
import sys

from dotenv import load_dotenv, find_dotenv
load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath('src')))
print(f"Project root directory: {PROJECT_ROOT}")
import numpy as np

import plotly.graph_objects as go
import pandas as pd

# %%
# load some helper functions
from utils.ROI_calculus import find_depth, find_hex_ids
from utils.celltype_conn_by_roi import CelltypeConnByRoi
from utils import olc_client


# %%
c = olc_client.connect(verbose=True)

# %%
mi1_me = CelltypeConnByRoi('Mi1', 'ME(R)')

# %%
syn_inp = mi1_me.get_input_synapses()
syn_out = mi1_me.get_output_synapses()

just_inp_syn = syn_inp[['x_post', 'y_post', 'z_post']]\
    .rename(columns={'x_post':'x', 'y_post':'y', 'z_post':'z'})
just_out_syn = syn_out[['x_pre', 'y_pre', 'z_pre']]\
    .rename(columns={'x_pre':'x', 'y_pre':'y', 'z_pre':'z'})

inp_depth = find_depth(just_inp_syn)
out_depth = find_depth(just_out_syn)

inp_cols = find_hex_ids(just_inp_syn)
out_cols = find_hex_ids(just_out_syn)

# %%
syn_inp_w_coldep = syn_inp.join(inp_cols).join(inp_depth)\
    .rename(columns={'bodyId_post':'bodyId', 'bodyId_pre':'bodyId_conn'})
syn_out_w_coldep = syn_out.join(out_cols).join(out_depth)\
    .rename(columns={'bodyId_pre':'bodyId', 'bodyId_post':'bodyId_conn'})
syn_inp_w_coldep['M_layers'] = 0
syn_out_w_coldep['M_layers'] = 0

# %%
syn_out_w_coldep

# %%
me_layer_bdry = [-0.01, 0.09,0.19,0.34,0.39,0.46,0.61,0.72,0.80,0.97, 1.01]

for dep_rng_ind in range(len(me_layer_bdry)-1):
    dep_low = me_layer_bdry[dep_rng_ind]
    dep_high = me_layer_bdry[dep_rng_ind+1]
    dep_inp_bool = (syn_inp_w_coldep['depth'] >= dep_low) & (syn_inp_w_coldep['depth'] < dep_high)
    dep_out_bool = (syn_out_w_coldep['depth'] >= dep_low) & (syn_out_w_coldep['depth'] < dep_high)
    syn_inp_w_coldep.loc[dep_inp_bool, 'M_layers'] = dep_rng_ind +1
    syn_out_w_coldep.loc[dep_out_bool, 'M_layers'] = dep_rng_ind +1

# %%
syn_out_w_coldep['M_layers'].hist(bins=np.linspace(0.5, 10.5, 11))

# %%
syn_out_w_coldep['syn_type'] = 'output'
syn_inp_w_coldep['syn_type'] = 'input'

# %%

inp_out_w_coldep = pd.concat([syn_inp_w_coldep[['bodyId', 'bodyId_conn', 'syn_type', 'depth', 'col_id', 'M_layers']], syn_out_w_coldep[['bodyId', 'bodyId_conn', 'syn_type', 'depth', 'col_id', 'M_layers']]])

# %%
inp_out_w_coldep

# %%
inp_out_gpby = inp_out_w_coldep\
    .groupby(['bodyId', 'col_id'], as_index=False)['M_layers']\
    .count()\
    .rename(columns={'M_layers':'syn_per_col'})
tmp_gpby = inp_out_w_coldep\
    .groupby(['bodyId'], as_index=False)['M_layers']\
    .count()\
    .rename(columns={'M_layers':'tot_syn'})
inp_out_gpby = inp_out_gpby.merge(tmp_gpby, on='bodyId')
inp_out_gpby['col_frac_from_tot'] = inp_out_gpby['syn_per_col']\
    .div(inp_out_gpby['tot_syn'])
inp_out_gpby\
    .sort_values(['bodyId', 'col_frac_from_tot'], ascending=False, inplace=True)
inp_out_gpby['col_frac_cs'] = inp_out_gpby\
    .groupby(['bodyId'])['col_frac_from_tot']\
    .cumsum()
inp_out_gpby['rank_frac_cs'] = inp_out_gpby\
    .groupby(['bodyId'])['col_frac_cs']\
    .rank(method='first')
tmp_ser = inp_out_gpby[inp_out_gpby['col_frac_cs']\
    .ge(0.8)]\
    .groupby(['bodyId'])['rank_frac_cs']\
    .min()\
    .rename('min_rank')
inp_out_gpby = inp_out_gpby.merge(tmp_ser, on='bodyId')


# %%
rel_bid_col_df = inp_out_gpby[
    inp_out_gpby['rank_frac_cs']<= inp_out_gpby['min_rank']]\
    [['bodyId', 'col_id']]

# %%
fin_df = rel_bid_col_df\
    .merge(inp_out_w_coldep, on=['bodyId', 'col_id'], how='left')\
    .dropna()

# %%
count_per_layer = fin_df['M_layers']\
    .value_counts()
cum_frac_layer = count_per_layer\
    .cumsum()\
    .div(count_per_layer.sum())

# %%
cum_frac_layer[cum_frac_layer.ge(0.9)].index[0]

# %%
cum_frac_layer.index.get_loc(5)

# %%
cum_frac_layer.index

# %%
cum_frac_layer.index[4:].values

# %%
count_col_df = fin_df\
    .groupby(['bodyId', 'M_layers'], as_index=False)['col_id']\
    .nunique()

# %%
tmp_ind = count_col_df[
    count_col_df['M_layers'].isin(cum_frac_layer.index[4:].values)]\
    .index

# %%
count_col_df.loc[tmp_ind, 'col_id'] = 0

# %%
count_col_df

# %%
fig = go.Figure(go.Box(x=count_col_df['M_layers'], y=count_col_df['col_id']))
fig.show()

# %%
by_layer_col_count_df = count_col_df\
    .reset_index()\
    .groupby(['bodyId', 'M_layers'])['col_id']\
    .aggregate('first')\
    .unstack()

# %%
by_layer_col_count_df

# %%
