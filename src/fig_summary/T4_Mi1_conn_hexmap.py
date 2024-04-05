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
# %load_ext autoreload

from pathlib import Path
import sys

from dotenv import load_dotenv, find_dotenv
load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath('src')))
print(f"Project root directory: {PROJECT_ROOT}")
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


# %%

# %autoreload 2
# load some helper functions
from utils.celltype_conn_by_roi import CelltypeConnByRoi
from utils import olc_client

# %%

c = olc_client.connect(verbose=True)

# %%
# directory to save results
result_dir = PROJECT_ROOT / 'results' / 'fig_summary'
result_dir.mkdir(parents=True, exist_ok=True)

# %%

mi1_obj = CelltypeConnByRoi('Mi1', 'ME(R)')


# %%
t4_names = ['T4a', 'T4b', 'T4c', 'T4d']
mi1_out = mi1_obj.get_output_neurons_w_stats()
mi1_out_red = mi1_out[mi1_out['type_post'].isin(t4_names)]

# %%
mi1_out_red.shape

# %%
mi1_t4_conn = mi1_out_red.groupby(['bodyId_pre', 'type_post'])['syn_count'].sum().reset_index()

# %%
sort_count = mi1_t4_conn['syn_count'].sort_values()
px.scatter(x=range(sort_count.shape[0]), y=sort_count)

# %%
mi1_t4_conn['tot_t4_syn'] = mi1_t4_conn.groupby('bodyId_pre')['syn_count'].transform('sum')
mi1_t4_conn['min_t4_syn'] = mi1_t4_conn.groupby('bodyId_pre')['syn_count'].transform('min')

# %%
px.scatter(x=mi1_t4_conn['tot_t4_syn'], y=mi1_t4_conn['min_t4_syn'])

# %%
t4a_obj = CelltypeConnByRoi('T4a', 'ME(R)')
t4b_obj = CelltypeConnByRoi('T4b', 'ME(R)')
t4c_obj = CelltypeConnByRoi('T4c', 'ME(R)')
t4d_obj = CelltypeConnByRoi('T4d', 'ME(R)')

# %%
t4a_inp = t4a_obj.get_input_neurons_w_stats()
t4b_inp = t4b_obj.get_input_neurons_w_stats()
t4c_inp = t4c_obj.get_input_neurons_w_stats()
t4d_inp = t4d_obj.get_input_neurons_w_stats()

t4a_inp_red = t4a_inp[t4a_inp['type_pre'] == 'Mi1']
t4b_inp_red = t4b_inp[t4b_inp['type_pre'] == 'Mi1']
t4c_inp_red = t4c_inp[t4c_inp['type_pre'] == 'Mi1']
t4d_inp_red = t4d_inp[t4d_inp['type_pre'] == 'Mi1']

all_t4_inp = pd.concat([t4a_inp_red,t4b_inp_red,t4c_inp_red,t4d_inp_red]).reset_index(drop=True)

# %%
all_t4_inp = all_t4_inp.sort_values('syn_count').reset_index()
px.scatter(all_t4_inp, all_t4_inp.index , y='syn_count', color='type_post')

# %%
all_t4_inp.groupby('bodyId_pre')['bodyId_post'].nunique()

# %%
# px.histogram(all_t4_inp['syn_count'])
px.histogram(np.log2(all_t4_inp['syn_count']))

# %% [markdown]
# ### seems like the cutoff between 2 different connections (weak and strong) is around 16 synapses

# %%
strong_t4_inp = all_t4_inp[all_t4_inp['syn_count'] > 16]

# %%
strong_t4_inp.groupby('bodyId_pre')['type_post'].nunique().lt(4).sum()

# %%
mi1_t4_all = all_t4_inp.groupby(['bodyId_pre', 'type_post'])['syn_count'].sum().reset_index().rename(columns={'bodyId_pre':'mi1_bid'})
mi1_t4_strong = strong_t4_inp.groupby(['bodyId_pre', 'type_post'])['syn_count'].sum().reset_index().rename(columns={'bodyId_pre':'mi1_bid'})

# %%
fig = go.Figure()
fig.add_trace(go.Histogram(x=mi1_t4_all['syn_count']))
fig.add_trace(go.Histogram(x=mi1_t4_strong['syn_count']))

fig.update_layout(barmode='overlay')
fig.update_traces(opacity=0.75)
fig.show()


# %%
px.histogram(mi1_t4_strong[mi1_t4_strong['type_post']=='T4c'], x='syn_count')


# %%
mi1_com = mi1_out_red.groupby('bodyId_pre')[['x_post', 'y_post', 'z_post']].mean()

# %%
mi1_t4_strong
