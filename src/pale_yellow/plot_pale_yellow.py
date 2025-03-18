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

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from neuprint import NeuronCriteria as NC
from neuprint import fetch_adjacencies, merge_neuron_properties, fetch_neurons
from queries.completeness import fetch_ol_types
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import find_dotenv
from utils import olc_client

PROJECT_ROOT = Path(find_dotenv()).parent
print(f"Project root directory: {PROJECT_ROOT}")
store_path = Path(PROJECT_ROOT, 'results', 'pale_yellow')
c = olc_client.connect(verbose=True)

# %%
# PANEL A: EYEMAP
# Load column identities from table created by A Nern that includes aMe12 annotations for pale columns
table_fn = PROJECT_ROOT / 'params' / 'Pale-Yellow_column-assignment.xlsx'
coltype_df = pd.read_excel(table_fn)
r78_hid = []
for col_type in ['pale', 'yellow', 'DRA', 'unclear']:
    r78_hid.append(coltype_df.loc[coltype_df['column_type']==col_type][['hex1_id','hex2_id']])

# %%
# PANEL B: CONNECTIONS ALONG PALE AND YELLOW PATHWAYS

# Panel B1-2: R7, R8

# Get connections between OL cell types and R7p, R7y, R8p, R8y
ct_df = fetch_ol_types()
neurons_df,connections_df = fetch_adjacencies(
    NC(type=['R7p', 'R7y', 'R8p', 'R8y', 'R7d', 'R8d'])
  , NC(type=ct_df['type'])
)
R78_pre_df = merge_neuron_properties(neurons_df, connections_df, 'type')
R78_pre_df = R78_pre_df\
    .groupby(['type_pre','type_post'])['weight']\
    .sum()\
    .to_frame()\
    .reset_index()

# R7/8 pale fraction is 28%
n_R7p = ct_df.loc[ct_df['type']=='R7p']['count'].iloc[0]
n_R7y = ct_df.loc[ct_df['type']=='R7y']['count'].iloc[0]
pale_fraction_R7 = n_R7p/(n_R7p + n_R7y) # 0.28
n_R8p = ct_df.loc[ct_df['type']=='R8p']['count'].iloc[0]
n_R8y = ct_df.loc[ct_df['type']=='R8y']['count'].iloc[0]
pale_fraction_R8 = n_R8p/(n_R8p + n_R8y) # 0.28
pale_fraction = (pale_fraction_R7 + pale_fraction_R8)/2

# Names of cell types with R7-8 presynapses
R78_pre_names_df = pd.DataFrame(R78_pre_df['type_post'].unique())
R78_pre_names_df.columns = ['type']

# Pick out connection weights with pale yellow R7-8
pre_R7p_df = R78_pre_df\
    .loc[R78_pre_df['type_pre']=='R7p'][['type_post', 'weight']]\
    .rename(columns={"type_post": "type", "weight": "pre_R7p"})
pre_R7y_df = R78_pre_df\
    .loc[R78_pre_df['type_pre']=='R7y'][['type_post', 'weight']]\
    .rename(columns={"type_post": "type", "weight": "pre_R7y"})
pre_R8p_df = R78_pre_df\
    .loc[R78_pre_df['type_pre']=='R8p'][['type_post', 'weight']]\
    .rename(columns={"type_post": "type", "weight": "pre_R8p"})
pre_R8y_df = R78_pre_df\
    .loc[R78_pre_df['type_pre']=='R8y'][['type_post', 'weight']]\
    .rename(columns={"type_post": "type", "weight": "pre_R8y"})
pre_R7d_df = R78_pre_df\
    .loc[R78_pre_df['type_pre']=='R7d'][['type_post', 'weight']]\
    .rename(columns={"type_post": "type", "weight": "pre_R7d"})
pre_R8d_df = R78_pre_df\
    .loc[R78_pre_df['type_pre']=='R8d'][['type_post', 'weight']]\
    .rename(columns={"type_post": "type", "weight": "pre_R8d"})

# Recombine into a conveniently formatted dataframe
PY_df = R78_pre_names_df\
    .merge(pre_R7p_df, how='outer', left_on='type', right_on='type')\
    .sort_values(by=['pre_R7p'], ascending=False)\
    .fillna(0)
PY_df = PY_df\
    .merge(pre_R7y_df, how='outer', left_on='type', right_on='type')\
    .sort_values(by=['pre_R7y'], ascending=False)\
    .fillna(0)
PY_df = PY_df\
    .merge(pre_R8p_df, how='outer', left_on='type', right_on='type')\
    .sort_values(by=['pre_R8p'], ascending=False)\
    .fillna(0)
PY_df = PY_df\
    .merge(pre_R8y_df, how='outer', left_on='type', right_on='type')\
    .sort_values(by=['pre_R8y'], ascending=False)\
    .fillna(0)
PY_df = PY_df\
    .merge(pre_R7d_df, how='outer', left_on='type', right_on='type')\
    .sort_values(by=['pre_R7d'], ascending=False)\
    .fillna(0)
PY_df = PY_df\
    .merge(pre_R8d_df, how='outer', left_on='type', right_on='type')\
    .sort_values(by=['pre_R8d'], ascending=False)\
    .fillna(0)

## The above could use the pivot function, eg:
# r78_pre_piv = R78_pre_df.pivot(index='type_post', columns="type_pre", values="weight")
# r78_pre_piv.columns = ['pre_' + col for col in r78_pre_piv.columns]
# PY_df = r78_pre_piv.merge(
#         R78_pre_df[['type_post']].drop_duplicates()
#       , on='type_post'
#       , how='left')\
#     .fillna(0)
#     .rename(columns={'type_post': 'type'})

# Add fractions of pale yellow R7-8
PY_df['Fpr7p']  = PY_df['pre_R7p'] / (PY_df['pre_R7p'] + PY_df['pre_R7y'])
PY_df['Fpr7y']  = PY_df['pre_R7y'] / (PY_df['pre_R7p'] + PY_df['pre_R7y'])
PY_df['Fpr8p']  = PY_df['pre_R8p'] / (PY_df['pre_R8p'] + PY_df['pre_R8y'])
PY_df['Fpr8y']  = PY_df['pre_R8y'] / (PY_df['pre_R8p'] + PY_df['pre_R8y'])
PY_df['Fpr7pd'] = PY_df['pre_R7p'] / (PY_df['pre_R7p'] + PY_df['pre_R7y'] + PY_df['pre_R7d'])
PY_df['Fpr7yd'] = PY_df['pre_R7y'] / (PY_df['pre_R7p'] + PY_df['pre_R7y'] + PY_df['pre_R7d'])
PY_df['Fpr8pd'] = PY_df['pre_R8p'] / (PY_df['pre_R8p'] + PY_df['pre_R8y'] + PY_df['pre_R8d'])
PY_df['Fpr8yd'] = PY_df['pre_R8y'] / (PY_df['pre_R8p'] + PY_df['pre_R8y'] + PY_df['pre_R8d'])

# Add cell numbers
PY_df = PY_df.merge(ct_df, how='inner', left_on='type', right_on='type')

# Tidy up
PY_df = PY_df[[
    'type', 'count'
  , 'Fpr7p', 'Fpr7y', 'Fpr8p', 'Fpr8y'
  , 'pre_R7p', 'pre_R7y', 'pre_R8p', 'pre_R8y', 'pre_R7d', 'Fpr7pd', 'pre_R8d'
  , 'Fpr8pd']]
PY_df = PY_df.round(decimals=2)

# Cell with presynaptic R7
pre_R7_pcell_thresh = 3
pre_R7_tot_thresh = 30     # Gives 21 cells total
pre_R7_df = PY_df[[
    'type', 'count'
  , 'Fpr7p', 'Fpr7y'
  , 'pre_R7p', 'pre_R7y', 'pre_R7d'
  , 'Fpr7pd'
  , 'pre_R8p', 'pre_R8y', 'pre_R8d'
  , 'Fpr8pd']]\
  .copy()
pre_R7_df['pre_R7'] = pre_R7_df['pre_R7p'] + pre_R7_df['pre_R7y']
pre_R7_df = pre_R7_df.loc[
    ((pre_R7_df['pre_R7']) / pre_R7_df['count'] >= pre_R7_pcell_thresh) \
  & (pre_R7_df['pre_R7'] >= pre_R7_tot_thresh)
]
pre_R7_df = pre_R7_df\
    .sort_values(by='type', ascending=False)\
    .reset_index(drop=True)
pre_R7_df = pre_R7_df\
    .sort_values(by='Fpr7p', ascending=False)\
    .reset_index(drop=True)

# Cell with presynaptic R8
pre_R8_pcell_thresh = 3
pre_R8_tot_thresh = 30     # Gives 23 cells total
pre_R8_df = PY_df[['type', 'count', 'Fpr8p', 'Fpr8y', 'pre_R8p', 'pre_R8y']].copy()
pre_R8_df['pre_R8'] = pre_R8_df['pre_R8p']+pre_R8_df['pre_R8y']
pre_R8_df = pre_R8_df.loc[
    ((pre_R8_df['pre_R8']) / pre_R8_df['count'] >= pre_R8_pcell_thresh)\
  & (pre_R8_df['pre_R8'] >= pre_R8_tot_thresh)
]
pre_R8_df = pre_R8_df\
    .sort_values(by='Fpr8p', ascending=False)\
    .reset_index(drop=True)

# Cells with large differences from pale_fraction
R7_pale_fraction_thr = 0.17 # First 5
R7_yellow_fraction_thr = 0.17 # Last 5
R7_pale_data = pre_R7_df.loc[(pre_R7_df['Fpr7p']-pale_fraction) > R7_pale_fraction_thr]
R7_yellow_data = pre_R7_df.loc[(pre_R7_df['Fpr7p']-pale_fraction) < -R7_yellow_fraction_thr]

R8_pale_fraction_thr = 0.19 # First 4
R8_yellow_fraction_thr = 0.17 # Last 4
R8_pale_data = pre_R8_df.loc[(pre_R8_df['Fpr8p']-pale_fraction) > R8_pale_fraction_thr]
R8_yellow_data = pre_R8_df.loc[(pre_R8_df['Fpr8p']-pale_fraction) < -R8_yellow_fraction_thr]

# R7_pale_fraction_thr
# R7_pale_data
# R7_yellow_data
# R8_pale_data
R8_yellow_data

# %%
# Panel B3: Tm5a/b

# Tm5b fraction is 46%
n_Tm5a = ct_df.loc[ct_df['type']=='Tm5a']['count'].iloc[0]
n_Tm5b = ct_df.loc[ct_df['type']=='Tm5b']['count'].iloc[0]
pale_fraction_Tm5 = n_Tm5b/(n_Tm5a + n_Tm5b)

# Connections between OL cell types and Tm5a, Tm5b
neurons_df,connections_df = fetch_adjacencies(
    NC(type=['Tm5a','Tm5b'])
  , NC(type=ct_df['type'])
)
Tm5_pre_df = merge_neuron_properties(neurons_df, connections_df, 'type')
Tm5_pre_df = Tm5_pre_df\
    .groupby(['type_pre', 'type_post'])['weight']\
    .sum()\
    .to_frame()\
    .reset_index()

# Names of cell types with Tm5 presynapses
Tm5_pre_names_df = pd.DataFrame(Tm5_pre_df['type_post'].unique())
Tm5_pre_names_df.columns = ['type']

# Connection weights with Tm5a/b
pre_Tm5a_df = Tm5_pre_df\
    .loc[Tm5_pre_df['type_pre']=='Tm5a'][['type_post', 'weight']]\
    .rename(columns={"type_post": "type", "weight": "pre_Tm5a"})
pre_Tm5b_df = Tm5_pre_df\
    .loc[Tm5_pre_df['type_pre']=='Tm5b'][['type_post', 'weight']]\
    .rename(columns={"type_post": "type", "weight": "pre_Tm5b"})

# Recombine into a conveniently formatted dataframe
Tm5_PY_df = Tm5_pre_names_df\
    .merge(pre_Tm5a_df, how='outer', left_on='type', right_on='type')\
    .sort_values(by=['pre_Tm5a'], ascending=False)\
    .fillna(0)
Tm5_PY_df = Tm5_PY_df\
    .merge(pre_Tm5b_df, how='outer', left_on='type', right_on='type')\
    .sort_values(by=['pre_Tm5b'], ascending=False)\
    .fillna(0)

# Add fractions of Tm5a vs Tm5b
Tm5_PY_df['Fpr5a'] = Tm5_PY_df['pre_Tm5a'] / (Tm5_PY_df['pre_Tm5a'] + Tm5_PY_df['pre_Tm5b'])
Tm5_PY_df['Fpr5b'] = Tm5_PY_df['pre_Tm5b'] / (Tm5_PY_df['pre_Tm5a'] + Tm5_PY_df['pre_Tm5b'])

# Add cell numbers
Tm5_PY_df = Tm5_PY_df.merge(ct_df, how='inner', left_on='type', right_on='type')

# Tidy up
Tm5_PY_df = Tm5_PY_df[['type', 'count', 'Fpr5a', 'Fpr5b', 'pre_Tm5a', 'pre_Tm5b']]
Tm5_PY_df = Tm5_PY_df.round(decimals=2)

# Extract data to plot

# Cell with presynaptic Tm5b
pre_Tm5_pcell_thresh = 3
pre_Tm5_tot_thresh = 1000     # Gives 46 cells total
pre_Tm5_df = Tm5_PY_df[['type', 'count', 'Fpr5a', 'Fpr5b', 'pre_Tm5a', 'pre_Tm5b']].copy()
pre_Tm5_df['pre_Tm5'] = pre_Tm5_df['pre_Tm5a'] + pre_Tm5_df['pre_Tm5b']
pre_Tm5_df = pre_Tm5_df\
    .loc[
        ((pre_Tm5_df['pre_Tm5']) / pre_Tm5_df['count'] >= pre_Tm5_pcell_thresh) \
      & (pre_Tm5_df['pre_Tm5'] >= pre_Tm5_tot_thresh)
    ]
pre_Tm5_df = pre_Tm5_df\
    .sort_values(by='type', ascending=False)\
    .reset_index(drop=True)
pre_Tm5_df = pre_Tm5_df\
    .sort_values(by='Fpr5b', ascending=False)\
    .reset_index(drop=True)

# Cells with large differences from pale_fraction
Tm5b_pale_fraction_thr = 0.5 # First 5
Tm5a_pale_fraction_thr = 0.43 # Last 5
Tm5_pale_data = pre_Tm5_df\
    .loc[(pre_Tm5_df['Fpr5b'] - pale_fraction_Tm5) > Tm5b_pale_fraction_thr]
Tm5_yellow_data = pre_Tm5_df\
    .loc[(pre_Tm5_df['Fpr5b'] - pale_fraction_Tm5) < -Tm5a_pale_fraction_thr]

# Tm5_pale_data
# Tm5_yellow_data

# %%
# Panel B4: Dm8

# Dm8b fraction is 48%
n_Dm8a = ct_df.loc[ct_df['type']=='Dm8a']['count'].iloc[0]
n_Dm8b = ct_df.loc[ct_df['type']=='Dm8b']['count'].iloc[0]
pale_fraction_Dm8 = n_Dm8b / (n_Dm8a + n_Dm8b)

# Connections between OL cell types and Dm8a, Dm8b
neurons_df,connections_df = fetch_adjacencies(
    NC(type=['Dm8a','Dm8b'])
  , NC(type=ct_df['type'])
)
Dm8_pre_df = merge_neuron_properties(neurons_df, connections_df, 'type')
Dm8_pre_df = Dm8_pre_df\
    .groupby(['type_pre', 'type_post'])['weight']\
    .sum()\
    .to_frame()\
    .reset_index()

# Names of cell types with R7-8 presynapses
Dm8_pre_names_df = pd.DataFrame(Dm8_pre_df['type_post'].unique())
Dm8_pre_names_df.columns = ['type']

# Connection weights with Dm8a/b
pre_Dm8a_df = Dm8_pre_df\
    .loc[Dm8_pre_df['type_pre']=='Dm8a'][['type_post', 'weight']]\
    .rename(columns={"type_post": "type", "weight": "pre_Dm8a"})
pre_Dm8b_df = Dm8_pre_df\
    .loc[Dm8_pre_df['type_pre']=='Dm8b'][['type_post', 'weight']]\
    .rename(columns={"type_post": "type", "weight": "pre_Dm8b"})

# Recombine into a conveniently formatted dataframe
Dm8_PY_df = Dm8_pre_names_df\
    .merge(pre_Dm8a_df, how='outer', left_on='type', right_on='type')\
    .sort_values(by=['pre_Dm8a'], ascending=False)\
    .fillna(0)
Dm8_PY_df = Dm8_PY_df\
    .merge(pre_Dm8b_df, how='outer', left_on='type', right_on='type')\
    .sort_values(by=['pre_Dm8b'], ascending=False)\
    .fillna(0)

# Add fractions of Dm8a vs Dm8b
Dm8_PY_df['Fpr8a'] = Dm8_PY_df['pre_Dm8a'] / (Dm8_PY_df['pre_Dm8a'] + Dm8_PY_df['pre_Dm8b'])
Dm8_PY_df['Fpr8b'] = Dm8_PY_df['pre_Dm8b'] / (Dm8_PY_df['pre_Dm8a'] + Dm8_PY_df['pre_Dm8b'])

# Add cell numbers
Dm8_PY_df = Dm8_PY_df.merge(ct_df, how='inner', left_on='type', right_on='type')

# Tidy up
Dm8_PY_df = Dm8_PY_df[['type', 'count', 'Fpr8a', 'Fpr8b', 'pre_Dm8a', 'pre_Dm8b']]
Dm8_PY_df = Dm8_PY_df.round(decimals=6)

# Extract data to plot

# Cell with presynaptic Dm8a
pre_Dm8_pcell_thresh = 3
pre_Dm8_tot_thresh = 300     # Gives 41 cells total
pre_Dm8_df = Dm8_PY_df[['type', 'count', 'Fpr8a', 'Fpr8b', 'pre_Dm8a', 'pre_Dm8b']].copy()
pre_Dm8_df['pre_Dm8'] = pre_Dm8_df['pre_Dm8a'] + pre_Dm8_df['pre_Dm8b']
pre_Dm8_df = pre_Dm8_df.loc[
    ((pre_Dm8_df['pre_Dm8']) / pre_Dm8_df['count'] >= pre_Dm8_pcell_thresh) \
  & (pre_Dm8_df['pre_Dm8'] >= pre_Dm8_tot_thresh)
]
pre_Dm8_df = pre_Dm8_df\
    .sort_values(by='type', ascending=False)\
    .reset_index(drop=True)
pre_Dm8_df = pre_Dm8_df\
    .sort_values(by='Fpr8b', ascending=False)\
    .reset_index(drop=True)

# Cells with large differences from pale_fraction
Dm8b_pale_fraction_thr = 0.4   # First 4
Dm8a_pale_fraction_thr = 0.448 # Last 4
Dm8_pale_data = pre_Dm8_df\
    .loc[(pre_Dm8_df['Fpr8b'] - pale_fraction_Dm8) > Dm8b_pale_fraction_thr]
Dm8_yellow_data = pre_Dm8_df\
    .loc[(pre_Dm8_df['Fpr8b'] - pale_fraction_Dm8) < -Dm8a_pale_fraction_thr]

# Dm8_pale_data
Dm8_yellow_data

# %%
# PANEL C: ASSIGNMENT OF Tm5a/b, Dm8a/b, R7p/y, R8p/y

# Get R7py synapses with Tm5a, Tm5b, Dm8a, Dm8b
neurons_df,connections_df = fetch_adjacencies(
    NC(type='R7p')
  , NC(type=['Dm8a', 'Dm8b', 'Tm5a', 'Tm5b'])
)
R7p_df = merge_neuron_properties(neurons_df, connections_df, 'type')
R7pW = R7p_df\
    .groupby(['bodyId_pre', 'type_post'])['weight']\
    .sum().to_frame().reset_index().copy()

neurons_df, connections_df = fetch_adjacencies(
    NC(type='R7y')
  , NC(type=['Dm8a', 'Dm8b', 'Tm5a', 'Tm5b'])
)
R7y_df = merge_neuron_properties(neurons_df, connections_df, 'type')
R7yW = R7y_df\
    .groupby(['bodyId_pre', 'type_post'])['weight']\
    .sum().to_frame().reset_index().copy()

neurons_df, connections_df = fetch_adjacencies(
    NC(type='R7_unclear')
  , NC(type=['Dm8a', 'Dm8b', 'Tm5a', 'Tm5b'])
)
R7u_df = merge_neuron_properties(neurons_df, connections_df, 'type')
R7uW = R7u_df\
    .groupby(['bodyId_pre', 'type_post'])['weight']\
    .sum().to_frame().reset_index().copy()

# Get R8py synapses with R7py
neurons_df, connections_df = fetch_adjacencies(
    NC(type='R8p')
  , NC(type=['R7p','R7y'])
)
R8p_df = merge_neuron_properties(neurons_df, connections_df, 'type')
R8pW = R8p_df\
    .groupby(['bodyId_pre', 'type_post'])['weight']\
    .sum().to_frame().reset_index().copy()

neurons_df, connections_df = fetch_adjacencies(
    NC(type='R8y'), NC(type=['R7p','R7y'])
)
R8y_df = merge_neuron_properties(neurons_df, connections_df, 'type')
R8yW = R8y_df\
    .groupby(['bodyId_pre', 'type_post'])['weight']\
    .sum().to_frame().reset_index().copy()

# Get Dm8 synapses with Tm5
neurons_df, connections_df = fetch_adjacencies(
    NC(type='Dm8a')
  , NC(type=['Tm5a','Tm5b'])
)
Dm8a_df = merge_neuron_properties(neurons_df, connections_df, 'type')
Dm8aW = Dm8a_df\
    .groupby(['bodyId_pre', 'type_post'])['weight']\
    .sum().to_frame().reset_index().copy()

neurons_df, connections_df = fetch_adjacencies(
    NC(type='Dm8b')
  , NC(type=['Tm5a','Tm5b'])
)
Dm8b_df = merge_neuron_properties(neurons_df, connections_df, 'type')
Dm8bW = Dm8b_df\
    .groupby(['bodyId_pre', 'type_post'])['weight']\
    .sum().to_frame().reset_index().copy()

# Data to plot: Dm8 -> Tm5a, Tm5b
Dm8b_Tm5a = Dm8bW.loc[Dm8bW['type_post']=='Tm5a'][['bodyId_pre', 'weight']]
Dm8b_Tm5b = Dm8bW.loc[Dm8bW['type_post']=='Tm5b'][['bodyId_pre', 'weight']]
Dm8b_Tm5  = Dm8b_Tm5a.merge(Dm8b_Tm5b, how='outer', on='bodyId_pre')
Dm8b_Tm5  = Dm8b_Tm5.rename(
    columns={"bodyId_pre": "Dm8b", "weight_x": "Tm5a", "weight_y": "Tm5b"}
)
Dm8b_Tm5  = Dm8b_Tm5.fillna(0)
Dm8b_Tm5  = Dm8b_Tm5.sort_values('Tm5a', ascending=False)

Dm8a_Tm5a = Dm8aW.loc[Dm8aW['type_post']=='Tm5a'][['bodyId_pre', 'weight']]
Dm8a_Tm5b = Dm8aW.loc[Dm8aW['type_post']=='Tm5b'][['bodyId_pre', 'weight']]
Dm8a_Tm5  = Dm8a_Tm5a.merge(Dm8a_Tm5b, how='outer', on='bodyId_pre')
Dm8a_Tm5  = Dm8a_Tm5.rename(
    columns={"bodyId_pre": "Dm8a", "weight_x": "Tm5a", "weight_y": "Tm5b"}
)
Dm8a_Tm5  = Dm8a_Tm5.fillna(0)
Dm8a_Tm5  = Dm8a_Tm5.sort_values('Tm5b', ascending=False)

# Data to plot: R7 -> Tm5a, Tm5b
R7p_Tm5a = R7pW.loc[R7pW['type_post']=='Tm5a'][['bodyId_pre', 'weight']]
R7p_Tm5b = R7pW.loc[R7pW['type_post']=='Tm5b'][['bodyId_pre', 'weight']]
R7p_Tm5  = R7p_Tm5a.merge(R7p_Tm5b, how='outer', on='bodyId_pre')
R7p_Tm5  = R7p_Tm5.rename(
    columns={"bodyId_pre": "R7p", "weight_x": "Tm5a", "weight_y": "Tm5b"}
)
R7p_Tm5  = R7p_Tm5.fillna(0)
R7p_Tm5  = R7p_Tm5.sort_values('Tm5a', ascending=False)

R7u_Tm5a = R7uW.loc[R7uW['type_post']=='Tm5a'][['bodyId_pre', 'weight']]
R7u_Tm5b = R7uW.loc[R7uW['type_post']=='Tm5b'][['bodyId_pre', 'weight']]
R7u_Tm5  = R7u_Tm5a.merge(R7u_Tm5b, how='outer', on='bodyId_pre')
R7u_Tm5  = R7u_Tm5.rename(
    columns={"bodyId_pre": "R7u", "weight_x": "Tm5a", "weight_y": "Tm5b"}
)
R7u_Tm5  = R7u_Tm5.fillna(0)
R7u_Tm5  = R7u_Tm5.sort_values('Tm5a', ascending=False)

R7y_Tm5a = R7yW.loc[R7yW['type_post']=='Tm5a'][['bodyId_pre','weight']]
R7y_Tm5b = R7yW.loc[R7yW['type_post']=='Tm5b'][['bodyId_pre','weight']]
R7y_Tm5  = R7y_Tm5a.merge(R7y_Tm5b, how='outer', on='bodyId_pre')
R7y_Tm5  = R7y_Tm5.rename(
    columns={"bodyId_pre": "R7y", "weight_x": "Tm5a", "weight_y": "Tm5b"}
)
R7y_Tm5  = R7y_Tm5.fillna(0)
R7y_Tm5  = R7y_Tm5.sort_values('Tm5b', ascending=False)

# Data to plot: R7 -> Dm8
R7p_Dm8a = R7pW.loc[R7pW['type_post']=='Dm8a'][['bodyId_pre','weight']]
R7p_Dm8b = R7pW.loc[R7pW['type_post']=='Dm8b'][['bodyId_pre','weight']]
R7p_Dm8  = R7p_Dm8a.merge(R7p_Dm8b, how='outer', on='bodyId_pre')
R7p_Dm8  = R7p_Dm8.rename(
    columns={"bodyId_pre": "R7p", "weight_x": "Dm8a", "weight_y": "Dm8b"}
)
R7p_Dm8  = R7p_Dm8.fillna(0)
R7p_Dm8  = R7p_Dm8.sort_values('Dm8a', ascending=False)

R7u_Dm8a = R7uW.loc[R7uW['type_post']=='Dm8a'][['bodyId_pre','weight']]
R7u_Dm8b = R7uW.loc[R7uW['type_post']=='Dm8b'][['bodyId_pre','weight']]
R7u_Dm8  = R7u_Dm8a.merge(R7u_Dm8b, how='outer', on='bodyId_pre')
R7u_Dm8  = R7u_Dm8.rename(
    columns={"bodyId_pre": "R7u", "weight_x": "Dm8a", "weight_y": "Dm8b"}
)
R7u_Dm8  = R7u_Dm8.fillna(0)
R7u_Dm8  = R7u_Dm8.sort_values('Dm8a', ascending=False)

R7y_Dm8a = R7yW.loc[R7yW['type_post']=='Dm8a'][['bodyId_pre','weight']]
R7y_Dm8b = R7yW.loc[R7yW['type_post']=='Dm8b'][['bodyId_pre','weight']]
R7y_Dm8  = R7y_Dm8a.merge(R7y_Dm8b, how='outer', on='bodyId_pre')
R7y_Dm8  = R7y_Dm8.rename(
    columns={"bodyId_pre": "R7y", "weight_x": "Dm8a", "weight_y": "Dm8b"}
)
R7y_Dm8  = R7y_Dm8.fillna(0)
R7y_Dm8  = R7y_Dm8.sort_values('Dm8b', ascending=False)

# Data to plot: R7 -> Dm8a and Tm5a, Dm8b and Tm5b
R7p_Dm8Tm5 = R7p_Dm8.merge(R7p_Tm5, how='outer', on='R7p')
R7p_Dm8Tm5 = R7p_Dm8Tm5.fillna(0)

R7u_Dm8Tm5 = R7u_Dm8.merge(R7u_Tm5, how='outer', on='R7u')
R7u_Dm8Tm5 = R7u_Dm8Tm5.fillna(0)

R7y_Dm8Tm5 = R7y_Dm8.merge(R7y_Tm5, how='outer', on='R7y')
R7y_Dm8Tm5 = R7y_Dm8Tm5.fillna(0)

# Data to plot: R8 -> R7p, R7y
R8p_R7p = R8pW.loc[R8pW['type_post']=='R7p'][['bodyId_pre','weight']]
R8p_R7y = R8pW.loc[R8pW['type_post']=='R7y'][['bodyId_pre','weight']]
R8p_R7  = R8p_R7p.merge(R8p_R7y, how='outer', on='bodyId_pre')
R8p_R7  = R8p_R7.rename(
    columns={"bodyId_pre": "R8p", "weight_x": "R7p", "weight_y": "R7y"}
)
R8p_R7  = R8p_R7.fillna(0)
R8p_R7  = R8p_R7.sort_values('R7y', ascending=False)

R8y_R7p = R8yW.loc[R8yW['type_post']=='R7p'][['bodyId_pre','weight']]
R8y_R7y = R8yW.loc[R8yW['type_post']=='R7y'][['bodyId_pre','weight']]
R8y_R7  = R8y_R7p.merge(R8y_R7y, how='outer', on='bodyId_pre')
R8y_R7  = R8y_R7.rename(
    columns={"bodyId_pre": "R8y", "weight_x": "R7p", "weight_y": "R7y"}
)
R8y_R7  = R8y_R7.fillna(0)
R8y_R7  = R8y_R7.sort_values('R7p', ascending=False)

# Get a connections between OL cell types and Tm5a, Tm5b
ct_df = fetch_ol_types()
neurons_df,connections_df = fetch_adjacencies(
    NC(type=['Tm5a','Tm5b'])
  , NC(type=ct_df['type'])
)
pre_df = merge_neuron_properties(neurons_df, connections_df, 'type')

# Make sure we start with all Tm5a, Tm5b
all_Tm5 = pre_df\
    .groupby(['bodyId_pre','type_pre'])['weight']\
    .sum().to_frame().reset_index()[['bodyId_pre','type_pre']]
all_Tm5 = all_Tm5.rename(columns={"type_pre": "type"})

# Add up synapses to ...LC6
LC6W = pre_df.loc[pre_df['type_post']=='LC6'][['bodyId_pre','weight']].reset_index(drop=True)
LC6W = LC6W.groupby('bodyId_pre').sum().reset_index()
LC6W = LC6W.rename(columns={"weight": "LC6"})

LC17W = pre_df.loc[pre_df['type_post']=='LC17'][['bodyId_pre','weight']].reset_index(drop=True)
LC17W = LC17W.groupby('bodyId_pre').sum().reset_index()
LC17W = LC17W.rename(columns={"weight": "LC17"})

LT58W = pre_df.loc[pre_df['type_post']=='LT58'][['bodyId_pre','weight']].reset_index(drop=True)
LT58W = LT58W.groupby('bodyId_pre').sum().reset_index()
LT58W = LT58W.rename(columns={"weight": "LT58"})

LoVP2W = pre_df.loc[pre_df['type_post']=='LoVP2'][['bodyId_pre','weight']].reset_index(drop=True)
LoVP2W = LoVP2W.groupby('bodyId_pre').sum().reset_index()
LoVP2W = LoVP2W.rename(columns={"weight": "LoVP2"})

Tm5ab_df = all_Tm5.merge(LC6W, how='outer', on='bodyId_pre')
Tm5ab_df = Tm5ab_df.merge(LC17W, how='outer', on='bodyId_pre')
Tm5ab_df = Tm5ab_df.merge(LT58W, how='outer', on='bodyId_pre')
Tm5ab_df = Tm5ab_df.merge(LoVP2W, how='outer', on='bodyId_pre')

Tm5ab_df = Tm5ab_df.fillna(0)

# Data to plot:
Tm5a_df = Tm5ab_df.loc[Tm5ab_df['type']=='Tm5a']
Tm5b_df = Tm5ab_df.loc[Tm5ab_df['type']=='Tm5b']



# %%
AN_df = coltype_df.loc[(coltype_df['aMe12_column']==1) & (coltype_df['R7']>0)][['R7']]

# %%
# Panel F: aMe12 validation

# Dataframe of body Ids of cells with aMe12 processes in the column
AN_df = coltype_df.loc[(coltype_df['aMe12_column']==1) & (coltype_df['R7']>0)][['R7']]
AN_df = AN_df.rename(columns={"R7": "bodyId"})

# bodyIds of R7p/y
neurons_df, roi_counts_df = fetch_neurons(NC(type='R7p'))
R7p_bid = neurons_df[['bodyId','type']]
neurons_df, roi_counts_df = fetch_neurons(NC(type='R7y'))
R7y_bid = neurons_df[['bodyId','type']]

# Numbers of R7p/y in columns with aMe12 processes
N_R7p_in_aMe12_col = len(R7p_bid.merge(AN_df, how='inner', on='bodyId'))
N_R7y_in_aMe12_col = len(R7y_bid.merge(AN_df, how='inner', on='bodyId'))

# Synaptic connections between R8p and aMe12
neurons_df,connections_df = fetch_adjacencies(NC(type='R8p'),NC(type='aMe12'))
R8p_aMe12_df = merge_neuron_properties(neurons_df, connections_df, 'type')

# Synaptic connections between R8y and aMe12
neurons_df,connections_df = fetch_adjacencies(NC(type='R8y'),NC(type='aMe12'))
R8y_aMe12_df = merge_neuron_properties(neurons_df, connections_df, 'type')

R7p_in_aMe12 = int(
    np.round(100 * N_R7p_in_aMe12_col / (N_R7p_in_aMe12_col + N_R7y_in_aMe12_col), decimals=0))
R7y_in_aMe12 = int(
    np.round(100 * N_R7y_in_aMe12_col / (N_R7p_in_aMe12_col + N_R7y_in_aMe12_col), decimals=0))


# %%
# Plot Figure

# 96 pixels per inch
# Full width 7"
pixperinch = 96
full_w = 7
full_h = 6.5

# Plot colours. Colour definition in dict() not working, not sure why
pale_col = 'rgba(148, 56, 131, 1)', #943883
yellow_col = 'rgba(254, 199, 43, 1)', #FEC72B
DRA_col = 'rgba(0, 0, 0, 1)',
Unclear_col = 'rgba(140, 140, 140, 1)',
None_col = 'rgba(204, 204, 204, 1)',
gray_line_col = 'rgba(128, 128, 128, 1)',
text_label_col = 'rgba(0, 0, 0, 1)',

# Style parameters
mks = 4.5  # marker size for panel A
lw = 0.25  # line width for equality line in panel C
mks2 = 2   # marker size for panel C
xso = 0    # xaxis standoff in panel C
yso = 0    # yaxis standoff in panel C
tl = 2     # tick length in panel C


fig = make_subplots(
    rows=5, cols=5
  , specs=[     
        [{"rowspan": 2, "colspan": 2, "type": "scatter"}, None, {"colspan":2,"type": "bar"}, None, {}]
      , [None, None, {"colspan": 2,"type": "bar"}, None, {}]
      , [{}, {"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}]
      , [{}, {}, {"rowspan": 2}, {}, {}]
      , [{}, {}, None, {}, {"type": "histogram"}]]
)

fig.update_layout(
    font={'family': "Arial", 'size': 6}
  , paper_bgcolor='rgba(255,255,255,1)'
  , plot_bgcolor='rgba(255,255,255,1)'
  , width=full_w*pixperinch
  , height=full_h*pixperinch
  , showlegend=False
)

# PANEL A: EYEMAP
for r78col, colr in zip(r78_hid, ['rgba(148, 56, 131, 1)', 'rgba(254, 199, 43, 1)', 'rgba(0, 0, 0, 1)', 'rgba(140, 140, 140, 1)']):
    fig.add_trace(
        go.Scatter(
            x=r78col['hex2_id'] - r78col['hex1_id']
          , y=r78col['hex1_id'] + r78col['hex2_id']
          , mode="markers"
          , marker_symbol=15
          , marker={
                'color': colr
              , 'size': mks
              , 'line': {
                    'color': 'rgba(0,0,0,0)'
                  , 'width': 2
                }
            }
        )
      , row=1, col=1)

# Panel A: Legend
xleg = 16 
yleg = 21 
dyleg = 3
dxleg = 2.5
fig.add_trace(
    go.Scatter(x=[xleg], y=[yleg]
      , mode="markers", marker_symbol=15
      , marker={
            'color':'rgba(148, 56, 131, 1)'
          , 'size': mks
          , 'line': {'color': 'rgba(0,0,0,0)', 'width': 2}
        }
    )
  , row=1,col=1)
fig.add_annotation(
    x=xleg + dxleg, y=yleg
  , text="Pale    "
  , showarrow=False
  , font={
        'family': "Arial"
      , 'size': 5
      , 'color': 'rgba(0, 0, 0, 1)'
    }
  , row=1, col=1),
fig.add_trace(
    go.Scatter(
        x=[xleg], y=[yleg - dyleg]
      , mode="markers", marker_symbol=15
      , marker={
            'color': 'rgba(254, 199, 43, 1)'
          , 'size': mks
          , 'line': {
                'color': 'rgba(0,0,0,0)'
              , 'width': 2
            }
        }
    )
  , row=1, col=1)
fig.add_annotation(
    x=xleg + dxleg, y=yleg - dyleg
  , text="Yellow  "
  , showarrow=False
  , font={
        'family': "Arial"
      , 'size': 5
      , 'color': 'rgba(0, 0, 0, 1)'
    }
  , row=1, col=1)
fig.add_trace(
    go.Scatter(
        x=[xleg], y=[yleg - 2 * dyleg]
      , mode="markers", marker_symbol=15
      , marker={
            'color': 'rgba(0, 0, 0, 1)'
          , 'size': mks
          , 'line': {
                'color': 'rgba(0,0,0,0)'
              , 'width': 2
            }
        }
    )
  , row=1, col=1)
fig.add_annotation(
    x=xleg + dxleg, y=yleg - 2 * dyleg
  , text="DRA     "
  , showarrow=False
  , font={
        'family': "Arial"
      , 'size': 5
      , 'color': 'rgba(0, 0, 0, 1)'
    }
  , row=1, col=1)
fig.add_trace(
    go.Scatter(
        x=[xleg], y=[yleg - 3 * dyleg]
      , mode="markers"
      , marker_symbol=15
      , marker={
            'color': 'rgba(140, 140, 140, 1)'
          , 'size': mks
          , 'line': {
                'color': 'rgba(0,0,0,0)'
              , 'width': 2
            }
        }
    )
  , row=1, col=1)
fig.add_annotation(
    x=xleg + dxleg, y=yleg - 3 * dyleg
  , text="Unclear "
  , showarrow=False
  , font={
        'family': "Arial"
      , 'size': 5
      , 'color': 'rgba(0, 0, 0, 1)'
    }
  , row=1, col=1),
# Layout
fig.update_layout(
    xaxis={'visible': False, 'range': [-17, 25]}
  , yaxis={'visible': False, 'range': [5, 73]}
)
fig.add_annotation(
    xref="x domain", yref="y domain"
  , x=-0.075, y=1.05
  , text="<b>a<b>"
  , showarrow=False
  , font={
        'family': "Arial"
      , 'size': 8
      , 'color': 'rgba(0, 0, 0, 1)'
    }
  , row=1, col=1)

# PANEL B: Pale/yellow pathway connectivity for R7, R8, Tm5, Dm8 

# Panel B1: R7
fig.add_trace(
    go.Bar(
        x=R7_pale_data['type']
      , y=100 * (R7_pale_data['Fpr7p'] - pale_fraction)
      , marker={
            'color': 'rgba(148, 56, 131, 1)'
          , 'line': {
                'color': 'rgba(140, 140, 140, 1)'
            }
        }
      , base=pale_fraction * 100
    )
  , row=1, col=3)
fig.add_trace(
    go.Bar(
        x=R7_yellow_data['type']
      , y=100 * (R7_yellow_data['Fpr7p'] - pale_fraction)
      , marker={
            'color': 'rgba(254, 199, 43, 1)'
          , 'line': {
                'color': 'rgba(140, 140, 140, 1)'
            }
        }
      , base=pale_fraction * 100
    )
  , row=1, col=3)
fig.add_annotation(
    x=0.9, y=10
  , text="R7 outputs"
  , showarrow=False
  , font={
        'family': "Arial"
      , 'size': 6
      , 'color': 'rgba(0, 0, 0, 1)'
    }
  , row=1, col=3)
fig.add_annotation(
    x=1.0, y=20
  , text="Pale-selective"
  , showarrow=False
  , font={
        'family': "Arial"
      , 'size': 6
      , 'color': 'rgba(148, 56, 131, 1)'
    }
  , row=1, col=3)
fig.add_annotation(
    x=5.8, y=32
  , text="Yellow-selective"
  , showarrow=False
  , font={
        'family': "Arial"
      , 'size': 6
      , 'color': 'rgba(254, 199, 43, 1)'
    }
  , row=1, col=3)
fig.add_annotation(
    x=8.0, y=32
  , text="#R7p/#R7py (%)"
  , showarrow=False
  , font={
        'family': "Arial"
      , 'size': 5
      , 'color': 'rgba(0, 0, 0, 1)'
    }
  , row=1, col=3)
fig.add_trace(
    go.Scatter(
        x=R7_yellow_data['type']
      , y=0 * R7_yellow_data['Fpr7p'] + pale_fraction * 100
      , mode='lines'
      , line={
            'color': 'rgba(0,0,0,1)'
          , 'width': lw
        }
    )
  , row=1, col=3)
# Layout
fig.update_layout(
    xaxis2={
        'tickmode': 'linear'
      , 'tickangle': 30
      , 'tickfont': {'size': 5}
      , 'tickwidth': 0.25
      , 'showline': True, 'linewidth': 0.25, 'linecolor': 'black'
      , 'ticklen': tl, 'ticks': 'outside'
    }
  , yaxis2={
        'tickmode': 'linear'
      , 'tickfont': {'size': 5}
      , 'tickwidth': 0.25
      , 'showline': True, 'linewidth': 0.25, 'linecolor': 'black'
      , 'range': [0, 100], 'dtick': 10
      , 'ticklen': tl, 'ticks': 'outside'
      , 'title': {
            'text': '% Pale <br>#R7p syn / #R7py syn (%)'
          , 'standoff': yso
          , 'font': {
                'family': "Arial"
              , 'size': 5
              , 'color': 'rgba(0, 0, 0, 1)'
            }
        }
    }
)
fig.add_annotation(
    xref="x domain", yref="y domain"
  , x=-0.15, y=1.1
  , text="<b>b<b>"
  , showarrow=False
  , font={
        'family': "Arial"
      , 'size': 8
      , 'color': 'rgba(0, 0, 0, 1)'
    }
  , row=1, col=3)

# Panel B2: R8
fig.add_trace(
    go.Bar(
        x=R8_pale_data['type']
      , y=100 * (R8_pale_data['Fpr8p'] - pale_fraction)
      , marker={
            'color': 'rgba(148, 56, 131, 1)'
          , 'line': {'color': 'rgba(140, 140, 140, 1)'}
        }
      , base=pale_fraction * 100
    )
  , row=1, col=5)
fig.add_trace(
    go.Bar(
        x=R8_yellow_data['type']
      , y=100 * (R8_yellow_data['Fpr8p'] - pale_fraction)
      , marker={
            'color': 'rgba(254, 199, 43, 1)'
          , 'line': {'color': 'rgba(140, 140, 140, 1)'}
        }
      , base=pale_fraction * 100
    )
  , row=1, col=5)
fig.add_annotation(
    x=1.2, y=10
  , text="R8 outputs"
  , showarrow=False
  , font={
        'family': "Arial"
      , 'size': 6
      , 'color': 'rgba(0, 0, 0, 1)'
    }
  , row=1, col=5)
fig.add_annotation(
    x=5.5, y=32
  , text="#R8p/#R8py (%)"
  , showarrow=False
  , font={
        'family': "Arial"
      , 'size': 5
      , 'color': 'rgba(0, 0, 0, 1)'
    }
  , row=1, col=5)
fig.add_trace(
    go.Scatter(
        x=R8_yellow_data['type']
      , y=0 * R8_yellow_data['Fpr8p'] + pale_fraction * 100
      , mode='lines'
      , line={
            'color': 'rgba(0,0,0,1)'
          , 'width': lw
        }
    )
  , row=1, col=5)
fig.update_layout(
    xaxis3={
        'tickmode': 'linear', 'tickangle': 30, 'tickfont': {'size': 5}
      , 'tickwidth': 0.25
      , 'showline': True, 'linewidth': 0.25, 'linecolor': 'black'
      , 'ticklen': tl, 'ticks': 'outside'
    }
  , yaxis3={
        'tickmode': 'linear', 'tickfont': {'size': 5}, 'tickwidth': 0.25
      , 'showline': True, 'linewidth': 0.25, 'linecolor': 'black'
      , 'range': [0, 100]
      , 'dtick': 10, 'ticklen': tl, 'ticks': 'outside'
      , 'title': {
            'text': '#R8p syn / #R8py syn (%)'
          , 'standoff': yso
          , 'font': {
                'family': "Arial"
              , 'size': 5
              , 'color': 'rgba(0, 0, 0, 1)'
            }
        }
    }
)

# Panel B3: Tm5
fig.add_trace(
    go.Bar(
        x=Tm5_pale_data['type']
      , y=100 * (Tm5_pale_data['Fpr5b'] - pale_fraction_Tm5)
      , marker={
            'color': 'rgba(148, 56, 131, 1)'
          , 'line': {'color': 'rgba(140, 140, 140, 1)'}
        }
      , base=pale_fraction_Tm5 * 100
    )
  , row=2, col=3)
fig.add_trace(
    go.Bar(
        x=Tm5_yellow_data['type']
      , y=100 * (Tm5_yellow_data['Fpr5b'] - pale_fraction_Tm5)
      , marker={
            'color': 'rgba(254, 199, 43, 1)'
          , 'line': {
                'color': 'rgba(140, 140, 140, 1)'
            }
        }
      , base=pale_fraction_Tm5 * 100
    )
  , row=2, col=3)
fig.add_annotation(
    x=1.3, y=10
  , text="Tm5 outputs"
  , showarrow=False
  , font={
        'family': "Arial"
      , 'size': 6
      , 'color': 'rgba(0, 0, 0, 1)'
    }
  , row=2, col=3)
fig.add_annotation(
    x=8, y=53
  , text="#Tm5b/#Tm5ab (%)"
  , showarrow=False
  , font={
        'family': "Arial"
      , 'size': 5
      , 'color': 'rgba(0, 0, 0, 1)'
    }
  , row=2, col=3)
fig.add_trace(
    go.Scatter(
        x=Tm5_yellow_data['type']
      , y=0 * Tm5_yellow_data['Fpr5b'] + pale_fraction_Tm5 * 100
      , mode='lines'
      , line={
            'color': 'rgba(0,0,0,1)'
          , 'width': lw
        }
    )
  , row=2, col=3)
fig.update_layout(
    xaxis4={
        'tickmode': 'linear', 'tickangle': 30
      , 'tickfont': {'size': 5}, 'tickwidth': 0.25
      , 'showline': True, 'linewidth': 0.25, 'linecolor': 'black'
      , 'ticklen': tl, 'ticks': 'outside'
    }
  , yaxis4={
        'tickmode': 'linear', 'tickfont': {'size': 5}, 'tickwidth': 0.25
      , 'showline': True, 'linewidth': 0.25, 'linecolor': 'black'
      , 'range': [0, 100]
      , 'dtick': 10, 'ticklen': tl, 'ticks': 'outside'
      , 'title': {
            'text': '#Tm5b syn / #Tm5ab syn (%)'
          , 'standoff': yso
          , 'font': {
                'family': "Arial"
              , 'size': 5
              , 'color': 'rgba(0, 0, 0, 1)'
            }
        }
    }
)

# Panel B4: Dm8
fig.add_trace(
    go.Bar(
        x=Dm8_pale_data['type']
      , y=100 * (Dm8_pale_data['Fpr8b'] - pale_fraction_Dm8)
      , marker=dict(
            color='rgba(148, 56, 131, 1)'
          , line=dict(
                color='rgba(140, 140, 140, 1)'
            )
        )
      , base=pale_fraction_Dm8*100
    )
  , row=2, col=5)
fig.add_trace(
    go.Bar(
        x=Dm8_yellow_data['type']
      , y=100 * (Dm8_yellow_data['Fpr8b'] - pale_fraction_Dm8)
      , marker=dict(
            color='rgba(254, 199, 43, 1)'
          , line=dict(
                color='rgba(140, 140, 140, 1)'
            )
        )
      , base=pale_fraction_Dm8 * 100
    )
  , row=2, col=5)
fig.add_annotation(
    x = 1.4, y = 10
  , text="Dm8 outputs"
  , showarrow=False
  , font=dict(
        family="Arial"
      , size=6
      , color='rgba(0, 0, 0, 1)'
    )
  , row=2, col=5)
fig.add_annotation(
    x=5.5, y=60
  , text="#Dm8b/<br>#Dm8ab (%)"
  , showarrow=False
  , font=dict(
        family="Arial"
      , size=5
      , color='rgba(0, 0, 0, 1)'
    )
  , row=2, col=5)
fig.add_trace(
    go.Scatter(
        x=Dm8_yellow_data['type']
      , y=0 * Dm8_yellow_data['Fpr8b'] + pale_fraction_Dm8 * 100
      , mode='lines'
      , line={'color': 'rgba(0,0,0,1)', 'width': lw}
    )
  , row=2, col=5)
fig.update_layout(
    xaxis5=dict(
        tickmode='linear'
      , tickangle=30
      , tickfont={'size': 5}
      , tickwidth=0.25
      , showline=True
      , linewidth=0.25
      , linecolor='black'
      , ticklen=tl
      , ticks= 'outside'
    )
  , yaxis5=dict(
        tickmode='linear'
      , tickfont={'size': 5}
      , tickwidth=0.25
      , showline=True
      , linewidth=0.25
      , linecolor='black'
      , range=[0, 100]
      , dtick=10
      , ticklen=tl
      , ticks='outside'
      , title=dict(
            text='#Dm8b syn / #Dm8ab syn (%))'
          , standoff=yso
          , font=dict(
                family="Arial"
              , size=5
              , color='rgba(0, 0, 0, 1)'
            )
        )
    )
)

# Panel C: scatterplots demonstrating p/y classification

# Panel C0: Algorithm
fig.add_annotation(
    xref="x domain"
  , yref="y domain"
  , x=-0.15
  , y = 1.15
  , text="<b>c<b>"
  , showarrow=False
  , font=dict(
        family="Arial"
      , size=8
      , color='rgba(0, 0, 0, 1)'
    )
  , row=3, col=1)
fig.add_annotation(
    x = 0, y = 100
  , text= ("<b>pale/yellow identification: </b><br>1) Tm5a/b identified by connectivity<br>"
      "    and morphology<br><b>2)</b> Dm8a/b identified by connectivity<br><b>3)</b> "
      "R7p/y identified by connectivity to<br>    Dm8a & Tm5a and Dm8b & Tm5b<br>    "
      "and anatomical markers in columns<br>    (Tm5a and aMe12); some R7 not<br>    "
      "assigned (R7_unclear, see Methods)<br><b>4)</b> R8p/y identified by R7 connectivity,"
      "<br>    and anatomical markers in columns")
  , showarrow=False
  , font=dict(
        family="Arial"
      , size=6
      , color='rgba(0, 0, 0, 1)'
    )
  , align='left'
  , row=3, col=1)
fig.update_layout(
    xaxis6=dict(visible=False)
  , yaxis6=dict(visible=False)
)

# Panel C1: Tm5
fig.add_trace(
    go.Scatter(
        x=Tm5a_df['LC6'] + Tm5a_df['LC17']
      , y=Tm5a_df['LT58'] + Tm5a_df['LoVP2']
      , mode='markers'
      , name="Tm5a"
      , marker=dict(
            size=mks2
          , color='rgba(254, 199, 43, 1)'
          , line={'color': 'rgba(0,0,0,0)'}
        )
    )
  , row=3, col=2)
fig.add_trace(
    go.Scatter(
        x=Tm5b_df['LC6'] + Tm5b_df['LC17']
      , y=Tm5b_df['LT58'] + Tm5b_df['LoVP2']
      , mode='markers'
      , name="Tm5b"
      , marker=dict(
            size=mks2
          , color='rgba(148, 56, 131, 1)'
          , line={'color': 'rgba(0,0,0,0)'}
        )
    )
  , row=3, col=2)
fig.add_annotation(
    x=20, y=75
  , text="Tm5a"
  , showarrow=False
  , font=dict(
        family="Arial"
      , size=6
      , color='rgba(0, 0, 0, 1)'
    )
  , row=3, col=2)
fig.add_annotation(
    x=75, y=15
  , text="Tm5b"
  , showarrow=False
  , font=dict(
        family="Arial"
      , size=6
      , color='rgba(0, 0, 0, 1)'
    )
  , row=3, col=2)
# Layout
fig.update_layout(
    xaxis7=dict(
        showline=True
      , linecolor='black'
      , dtick=20
      , tick0=0
      , tickwidth=0.25
      , range=[-5, 90]
      , ticklen=tl
      , ticks='outside'
      , linewidth=0.25
      , tickfont={'size': 5}
      , title=dict(
            text="#Syn to LC6 LC17"
          , standoff=xso
          , font=dict(size=6)
        )
    )
  , yaxis7=dict(
        showline=True
      , linecolor='black'
      , dtick=20
      , tick0=0
      , tickwidth=0.25
      , range=[-5, 90]
      , ticklen=tl
      , ticks='outside'
      , linewidth=0.25
      , tickfont=dict(size=5)
      , title=dict(
            text="#Syn to LT58 LoVP2"
          , standoff=yso
          , font=dict(size=6)
        )
    )
)


# Panel C2: Dm8
fig.add_trace(
    go.Scatter(
        x=Dm8a_Tm5['Tm5b']
      , y=Dm8a_Tm5['Tm5a']
      , mode='markers'
      , name="Dm8a"
      , marker=dict(
            size=mks2
          , color='rgba(254, 199, 43, 1)'
          , line={'color': 'rgba(0,0,0,0)'}
        )
    )
  , row=3, col=3)
fig.add_trace(
    go.Scatter(
        y=Dm8b_Tm5['Tm5a']
      , x=Dm8b_Tm5['Tm5b']
      , mode='markers'
      , name="Dm8b"
      , marker=dict(
            size=mks2
          , color='rgba(148, 56, 131, 1)'
          , line=dict(color='rgba(0,0,0,0)')
        )
    )
  , row=3, col=3)
fig.add_annotation(
    x=20 * 7 / 9
  , y=75 * 7 / 9
  , text="Dm8a"
  , showarrow=False
  , font=dict(
        family="Arial"
      , size=6
      , color='rgba(0, 0, 0, 1)'
    )
  , row=3, col=3)
fig.add_annotation(
    x=75 * 7 / 9
  , y=15 * 7 / 9
  , text="Dm8b"
  , showarrow=False
  , font=dict(
        family="Arial"
      , size=6
      , color='rgba(0, 0, 0, 1)'
    )
  , row=3, col=3)
fig.update_layout(
    xaxis8=dict(
        showline=True
      , linecolor='black'
      , dtick=20
      , tick0=0
      , tickwidth=0.25
      , range=[-5 * 7 / 9, 70]
      , ticklen=tl
      , ticks='outside'
      , linewidth=0.25
      , tickfont=dict(size=5)
      , title=dict(
            text="#Syn to Tm5b"
          , standoff=xso
          , font=dict(size=6)
        )
    )
  , yaxis8=dict(
        showline=True
      , linecolor='black'
      , dtick=20
      , tick0=0
      , tickwidth=0.25
      , range=[-5 * 7 / 9, 70]
      , ticklen=tl
      , ticks='outside'
      , linewidth = 0.25
      , tickfont=dict(size=5)
      , title=dict(
            text="#Syn to Tm5a"
          , standoff=yso
          , font=dict(size=6)
        )
    )
)

# Panel C3: R7
fig.add_trace(
    go.Scatter(
        y=R7u_Dm8Tm5['Dm8a'] + R7u_Dm8Tm5['Tm5a']
      , x=R7u_Dm8['Dm8b'] + R7u_Tm5['Tm5b']
      , mode='markers'
      , name="R7u"
      , marker=dict(
            size=mks2
          , color='rgba(180, 180, 180, 1)'
          , line=dict(color='rgba(0,0,0,0)')
        )
    )
  , row=3, col=4)
fig.add_trace(
    go.Scatter(
        y=R7y_Dm8Tm5['Dm8a'] + R7y_Dm8Tm5['Tm5a']
      , x=R7y_Dm8['Dm8b'] + R7y_Tm5['Tm5b']
      , mode='markers'
      , name="R7y"
      , marker=dict(
            size=mks2
          , color='rgba(254, 199, 43, 1)'
          , line=dict(color='rgba(0,0,0,0)')
        )
    )
  , row=3, col=4)
fig.add_trace(
    go.Scatter(
        y=R7p_Dm8Tm5['Dm8a'] + R7p_Dm8Tm5['Tm5a']
      , x=R7p_Dm8['Dm8b'] + R7p_Tm5['Tm5b']
      , mode='markers'
      , name="R7p"
      , marker=dict(
            size=mks2
          , color='rgba(148, 56, 131, 1)'
          , line=dict(color='rgba(0,0,0,0)')
        )
    )
  , row=3, col=4)

xleg = 60
yleg = 120
dxleg = 30
dyleg = 10
mks = 3
fig.add_trace(
    go.Scatter(
        x=[xleg], y=[yleg]
      , mode="markers"
      , marker=dict(
            color='rgba(148, 56, 131, 1)'
          , size=mks
          , line=dict(color='rgba(0,0,0,0)', width=2)
        )
    )
  , row=3, col=4)
fig.add_annotation(
    x=xleg+dxleg, y=yleg
  , text="R7p            "
  , showarrow=False
  , font=dict(
        family="Arial"
      , size=5
      , color='rgba(0, 0, 0, 1)'
    )
  , row=3, col=4)
fig.add_trace(
    go.Scatter(
        x=[xleg], y=[yleg-dyleg]
      , mode="markers"
      , marker_symbol=15
      , marker=dict(
            color='rgba(254, 199, 43, 1)'
          , size=mks
          , line=dict(color='rgba(0,0,0,0)', width=2)
        )
    )
  , row=3, col=4)
fig.add_annotation(
    x=xleg + dxleg, y=yleg - dyleg
  , text="R7y            "
  , showarrow=False
  , font=dict(
        family="Arial"
      , size=5
      , color='rgba(0, 0, 0, 1)'
    )
  , row=3, col=4)
fig.add_trace(
    go.Scatter(
        x=[xleg], y=[yleg-2*dyleg]
      , mode="markers"
      , marker=dict(
            color='rgba(0, 0, 0, 1)'
          , size=mks
          , line=dict(color='rgba(0,0,0,0)', width=2)
        )
    )
  , row=3, col=4)
fig.add_annotation(
    x=xleg + dxleg, y=yleg - 2 * dyleg
  , text="R7_unclear"
  , showarrow=False
  , font=dict(
        family="Arial"
      , size=5
      , color='rgba(0, 0, 0, 1)'
    )
  , row=3, col=4)

fig.update_layout(
    xaxis9=dict(
        showline=True
      , linecolor='black'
      , dtick=30
      , tick0=0
      , tickwidth=0.25
      , range=[-5*14/9, 140]
      , ticklen=tl
      , ticks='outside'
      , linewidth=0.25
      , tickfont=dict(size=5)
      , title=dict(
            text="#Syn to Dm8b Tm5b"
          , standoff=xso
          , font=dict(size=6)
        )
    )
  , yaxis9=dict(
        showline=True
      , linecolor='black'
      , dtick=30
      , tick0=0
      , tickwidth=0.25
      , range=[-5*14/9, 140]
      , ticklen=tl
      , ticks='outside'
      , linewidth=0.25
      , tickfont=dict(size=5)
      , title=dict(
            text="#Syn to Dm8a Tm5a"
          , standoff=yso
          , font=dict(size=6)
        )
    )
)

# Panel C4: R8
fig.add_trace(
    go.Scatter(
        x=R8y_R7['R7p']
      , y=R8y_R7['R7y']
      , mode='markers'
      , name="R8y"
      , marker=dict(
            size=mks2
          , color='rgba(254, 199, 43, 1)'
          , line=dict(color='rgba(0,0,0,0)')
        )
    )
  , row=3, col=5)
fig.add_trace(
    go.Scatter(
        x=R8p_R7['R7p']
      , y=R8p_R7['R7y']
      , mode='markers'
      , name="R8p"
      , marker=dict(
            size=mks2
          , color='rgba(148, 56, 131, 1)'
          , line=dict(color='rgba(0,0,0,0)')
        )
    )
  , row=3, col=5)
fig.add_annotation(
    x = 20 * 6 / 9
  , y = 75 * 6 / 9
  , text="R8y"
  , showarrow=False
  , font=dict(
        family="Arial"
      , size=6
      , color='rgba(0, 0, 0, 1)'
    )
  , row=3, col=5)
fig.add_annotation(
    x = 75 * 6.5 / 9
  , y = 15 * 6 / 9
  , text="R8p"
  , showarrow=False
  , font=dict(
        family="Arial"
      , size=6
      , color='rgba(0, 0, 0, 1)'
    )
  , row=3, col=5)
fig.update_layout(
    xaxis10=dict(
        showline=True
      , linecolor='black'
      , dtick=10
      , tick0=0
      , tickwidth=0.25
      , range=[-5 * 6 / 9, 60]
      , ticklen=tl
      , ticks='outside'
      , linewidth=0.25
      , tickfont=dict(size=5)
      , title=dict(
            text="#Syn to R7p"
          , standoff=xso
          , font=dict(size=6)
        )
    )
  , yaxis10=dict(
        showline=True
      , linecolor='black'
      , dtick=10
      , tick0=0
      , tickwidth=0.25
      , range=[-5 * 6 / 9, 60]
      , ticklen=tl
      , ticks='outside'
      , linewidth=0.25
      , tickfont=dict(size=5)
      , title=dict(
            text="#Syn to R7y"
          , standoff=yso
          , font=dict(size=6)
        )
    )
)

# PANEL D: CLUSTERING

# Panel D1: Spatial clustering plot

#Tm5a 307 0
#Tm5b 0 262

# Panel D2: Clustering data
xc1 = 0
xc2 = 10
xc3 = 20
dxl = 3
yr0 = 140
yr1 = 100
yr2 = 80
yr3 = 60
yr4 = 40
dyr = 7
tfs = 6 # table fontsize
dtr = 4 # D table row
dtc = 3 # D table column
dtlw = lw # 0.25 D table line width

# Tm5a Tm5b
fig.add_annotation(
    x=xc2, y=yr0+dyr
  , text="<b>Clustering: all cells</b>"
  , align='left'
  , showarrow=False
  , font=dict(
        family="Arial"
      , size=tfs
      , color='rgba(0, 0, 0, 1)'
    )
  , row=4, col=3)
fig.add_annotation(
    x=xc1, y=yr0
  , text="Tm5a"
  , align='left'
  , showarrow=False
  , font=dict(
        family="Arial"
      , size=tfs
      , color='rgba(0, 0, 0, 1)'
    )
  , row=4, col=3)
fig.add_annotation(
    x=xc2, y=yr0
  , text="307"
  , align='left'
  , showarrow=False
  , font=dict(
        family="Arial"
      , size=tfs
      , color='rgba(0, 0, 0, 1)'
    )
  , row=4, col=3)
fig.add_annotation(
    x=xc3, y=yr0
  , text="0"
  , align='left'
  , showarrow=False
  , font=dict(
        family="Arial"
      , size=tfs
      , color='rgba(0, 0, 0, 1)'
    )
  , row=4, col=3)
fig.add_annotation(
    x=xc1, y=yr0-dyr
  , text="Tm5b"
  , align='left'
  , showarrow=False
  , font=dict(
        family="Arial"
      , size=tfs
      , color='rgba(0, 0, 0, 1)'
    )
  , row=4, col=3)
fig.add_annotation(
    x=xc2, y=yr0-dyr
  , text="0"
  , align='left'
  , showarrow=False
  , font=dict(
        family="Arial"
      , size=tfs
      , color='rgba(0, 0, 0, 1)'
    )
  , row=4, col=3)
fig.add_annotation(
    x=xc3, y=yr0-dyr
  , text="262"
  , align='left'
  , showarrow=False
  , font=dict(
        family="Arial"
      , size=tfs
      , color='rgba(0, 0, 0, 1)'
    )
  , row=4, col=3)
fig.add_trace(
    go.Scatter(
        x=[xc1-dxl, xc3+dxl]
      , y=[yr0-dyr/2, yr0-dyr/2]
      , mode='lines'
      , line=dict(
            color='black'
          , width=dtlw
        )
    )
  , row=dtr, col=dtc)
fig.add_trace(
    go.Scatter(
        x=[(xc1+xc2)/2, (xc1+xc2)/2]
      , y=[yr0-dyr*3/2, yr0+dyr/2]
      , mode='lines'
      , line=dict(
            color='black'
          , width=dtlw
        )
    )
  , row=dtr, col=dtc)
fig.add_trace(
    go.Scatter(
        x=[(xc2+xc3)/2, (xc2+xc3)/2]
      , y=[yr0-dyr*3/2, yr0+dyr/2]
      , mode='lines'
      , line=dict(
            color='black'
          , width=dtlw
        )
    )
  , row=dtr, col=dtc)

# Tm5a Tm5b
fig.add_annotation(
    x=xc2, y=yr1+dyr*1.5
  , text="<b>Clustering: without<br> Tm5, Dm8, R7, R8</b>"
  , align='left'
  , showarrow=False
  , font=dict(
        family="Arial"
      , size=tfs
      , color='rgba(0, 0, 0, 1)'
    )
  , row=4, col=3)
fig.add_annotation(
    x=xc1, y=yr1
  , text="Tm5a"
  , align='left'
  , showarrow=False
  , font=dict(
        family="Arial"
      , size=tfs
      , color='rgba(0, 0, 0, 1)'
    )
  , row=4, col=3)
fig.add_annotation(
    x=xc2, y=yr1
  , text="307"
  , align='left'
  , showarrow=False
  , font=dict(
        family="Arial"
      , size=tfs
      , color='rgba(0, 0, 0, 1)'
    )
  , row=4, col=3)
fig.add_annotation(
    x=xc3, y=yr1
  , text="0"
  , align='left'
  , showarrow=False
  , font=dict(
        family="Arial"
      , size=tfs
      , color='rgba(0, 0, 0, 1)'
    )
  , row=4, col=3)
fig.add_annotation(
    x=xc1, y=yr1-dyr
  , text="Tm5b"
  , align='left'
  , showarrow=False
  , font=dict(
        family="Arial"
      , size=tfs
      , color='rgba(0, 0, 0, 1)'
    )
  , row=4, col=3)
fig.add_annotation(
    x=xc2, y=yr1-dyr
  , text="0"
  , align='left'
  , showarrow=False
  , font=dict(
        family="Arial"
      , size=tfs
      , color='rgba(0, 0, 0, 1)'
    )
  , row=4, col=3)
fig.add_annotation(
    x=xc3, y=yr1-dyr
  , text="262"
  , align='left'
  , showarrow=False
  , font=dict(
        family="Arial"
      , size=tfs
      , color='rgba(0, 0, 0, 1)'
    )
  , row=4, col=3)
fig.add_trace(
    go.Scatter(
        x=[xc1-dxl, xc3+dxl]
      , y=[yr1-dyr/2, yr1-dyr/2]
      , mode='lines'
      , line=dict(
            color='black'
          , width=dtlw
        )
    )
  , row=dtr, col=dtc)
fig.add_trace(
    go.Scatter(
        x=[(xc1+xc2)/2, (xc1+xc2)/2]
      , y=[yr1-dyr*3/2, yr1+dyr/2]
      , mode='lines'
      , line=dict(color='black', width=dtlw)
    )
  , row=dtr, col=dtc)
fig.add_trace(
    go.Scatter(
        x=[(xc2+xc3)/2, (xc2+xc3)/2]
      , y=[yr1-dyr*3/2, yr1+dyr/2]
      , mode='lines'
      , line=dict(color='black', width=dtlw)
    )
  , row=dtr, col=dtc)

# Tm5b Tm29
fig.add_annotation(
    x=xc1, y=yr2
  , text="Tm5a"
  , align='left'
  , showarrow=False
  , font=dict(
        family="Arial"
      , size=tfs
      , color='rgba(0, 0, 0, 1)'
    )
  , row=4, col=3)
fig.add_annotation(
    x=xc2, y=yr2
  , text="307"
  , align='left'
  , showarrow=False
  , font=dict(
        family="Arial"
      , size=tfs
      , color='rgba(0, 0, 0, 1)'
    )
  , row=4, col=3)
fig.add_annotation(
    x=xc3, y=yr2
  , text="0"
  , align='left'
  , showarrow=False
  , font=dict(
        family="Arial"
      , size=tfs
      , color='rgba(0, 0, 0, 1)'
    )
  , row=4, col=3)
fig.add_annotation(
    x=xc1, y=yr2-dyr
  , text="Tm29"
  , align='left'
  , showarrow=False
  , font=dict(
        family="Arial"
      , size=tfs
      , color='rgba(0, 0, 0, 1)'
    )
  , row=4, col=3)
fig.add_annotation(
    x=xc2, y=yr2-dyr
  , text="0"
  , align='left'
  , showarrow=False
  , font=dict(
        family="Arial"
      , size=tfs
      , color='rgba(0, 0, 0, 1)'
    )
  , row=4, col=3)
fig.add_annotation(
    x=xc3, y=yr2-dyr
  , text="276"
  , align='left'
  , showarrow=False
  , font=dict(
        family="Arial"
      , size=tfs
      , color='rgba(0, 0, 0, 1)'
    )
  , row=4, col=3)
fig.add_trace(
    go.Scatter(
        x=[xc1-dxl, xc3+dxl]
      , y=[yr2-dyr/2, yr2-dyr/2]
      , mode='lines'
      , line=dict(color='black', width=dtlw)
    )
  , row=dtr, col=dtc)
fig.add_trace(
    go.Scatter(
        x=[(xc1+xc2)/2, (xc1+xc2)/2]
      , y=[yr2-dyr*3/2, yr2+dyr/2]
      , mode='lines'
      , line=dict(color='black', width=dtlw)
    )
  , row=dtr, col=dtc)
fig.add_trace(
    go.Scatter(
        x=[(xc2+xc3)/2, (xc2+xc3)/2]
      , y=[yr2-dyr*3/2, yr2+dyr/2]
      , mode='lines'
      , line=dict(color='black', width=dtlw)
    )
  , row=dtr, col=dtc)

# Tm5b Tm29
fig.add_annotation(
    x=xc1, y=yr3
  , text="Tm5b"
  , align='left'
  , showarrow=False
  , font=dict(
        family="Arial"
      , size=tfs
      , color='rgba(0, 0, 0, 1)'
    )
  , row=4, col=3)
fig.add_annotation(
    x=xc2, y=yr3
  , text="262"
  , align='left'
  , showarrow=False
  , font=dict(
        family="Arial"
      , size=tfs
      , color='rgba(0, 0, 0, 1)'
    )
  , row=4, col=3)
fig.add_annotation(
    x=xc3, y=yr3
  , text="0"
  , align='left'
  , showarrow=False
  , font=dict(
        family="Arial"
      , size=tfs
      , color='rgba(0, 0, 0, 1)'
    )
  , row=4, col=3)
fig.add_annotation(
    x=xc1, y=yr3-dyr
  , text="Tm29"
  , align='left'
  , showarrow=False
  , font=dict(
        family="Arial"
      , size=tfs
      , color='rgba(0, 0, 0, 1)'
    )
  , row=4, col=3)
fig.add_annotation(
    x=xc2, y=yr3-dyr
  , text="0"
  , align='left'
  , showarrow=False
  , font=dict(
        family="Arial"
      , size=tfs
      , color='rgba(0, 0, 0, 1)'
    )
  , row=4, col=3)
fig.add_annotation(
    x=xc3, y=yr3-dyr
  , text="276"
  , align='left'
  , showarrow=False
  , font=dict(
        family="Arial"
      , size=tfs
      , color='rgba(0, 0, 0, 1)'
    )
  , row=4, col=3)
fig.add_trace(
    go.Scatter(
        x=[xc1-dxl, xc3+dxl]
      , y=[yr3-dyr/2, yr3-dyr/2]
      , mode='lines'
      , line=dict(color='black', width=dtlw)
    )
  , row=dtr, col=dtc)
fig.add_trace(
    go.Scatter(
        x=[(xc1+xc2)/2, (xc1+xc2)/2]
      , y=[yr3-dyr*3/2, yr3+dyr/2]
      , mode='lines'
      , line=dict(color='black', width=dtlw)
    )
  , row=dtr, col=dtc)
fig.add_trace(
    go.Scatter(
        x=[(xc2+xc3)/2, (xc2+xc3)/2]
      , y=[yr3-dyr*3/2, yr3+dyr/2]
      , mode='lines'
      , line=dict(color='black', width=dtlw)
    )
  , row=dtr, col=dtc)

# Dm8a Dm8b
fig.add_annotation(
    x=xc1, y=yr4
  , text="Dm8a"
  , align='left'
  , showarrow=False
  , font=dict(
        family="Arial"
      , size=tfs
      , color='rgba(0, 0, 0, 1)'
    )
  , row=4, col=3)
fig.add_annotation(
    x=xc2, y=yr4
  , text="285"
  , align='left'
  , showarrow=False
  , font=dict(
        family="Arial"
      , size=tfs
      , color='rgba(0, 0, 0, 1)'
    )
  , row=4, col=3)
fig.add_annotation(
    x=xc3, y=yr4
  , text="0"
  , align='left'
  , showarrow=False
  , font=dict(
        family="Arial"
      , size=tfs
      , color='rgba(0, 0, 0, 1)'
    )
  , row=4, col=3)
fig.add_annotation(
    x=xc1, y=yr4-dyr
  , text="Dm8b"
  , align='left'
  , showarrow=False
  , font=dict(
        family="Arial"
      , size=tfs
      , color='rgba(0, 0, 0, 1)'
    )
  , row=4, col=3)
fig.add_annotation(
    x=xc2, y=yr4-dyr
  , text="0"
  , align='left'
  , showarrow=False
  , font=dict(
        family="Arial"
      , size=tfs
      , color='rgba(0, 0, 0, 1)'
    )
  , row=4, col=3)
fig.add_annotation(
    x=xc3, y=yr4-dyr
  , text="265"
  , align='left'
  , showarrow=False
  , font=dict(
        family="Arial"
      , size=tfs
      , color='rgba(0, 0, 0, 1)'
    )
  , row=4, col=3)
fig.add_trace(
    go.Scatter(
        x=[xc1-dxl, xc3+dxl]
      , y=[yr4-dyr/2, yr4-dyr/2]
      , mode='lines'
      , line=dict(color='black', width=dtlw)
    )
  , row=dtr, col=dtc)
fig.add_trace(
    go.Scatter(
        x=[(xc1+xc2)/2, (xc1+xc2)/2]
      , y=[yr4-dyr*3/2, yr4+dyr/2]
      , mode='lines'
      , line=dict(
            color='black'
          , width=dtlw
        )
    )
  , row=dtr, col=dtc)
fig.add_trace(
    go.Scatter(
        x=[(xc2+xc3)/2, (xc2+xc3)/2]
      , y=[yr4-dyr*3/2, yr4+dyr/2]
      , mode='lines'
      , line=dict(
            color='black'
          , width=dtlw
        )
    )
  , row=dtr, col=dtc)
# Layout
fig.update_layout(
    xaxis13=dict(visible=False, range=[-5, 25])
  , yaxis13=dict(visible=False, range=[25, 160]))

# Panel F1: Numbers of R7p/y in columns with aMe12 processes
fig.add_annotation(
    x=(xc1+xc2)/2
  , y=yr0+dyr
  , text="<b>Cells in aMe12 columns</b>"
  , align='left'
  , showarrow=False
  , font=dict(
        family="Arial"
      , size=tfs
      , color='rgba(0, 0, 0, 1)'
    )
  , row=4, col=5)
fig.add_annotation(
    x=xc1, y=yr0
  , text="R7p (%)"
  , align='left'
  , showarrow=False
  , font=dict(
        family="Arial"
      , size=tfs
      , color='rgba(0, 0, 0, 1)'
    )
  , row=4, col=5)
fig.add_annotation(
    x=xc2, y=yr0
  , text="R7y (%)"
  , align='left'
  , showarrow=False
  , font=dict(
        family="Arial"
      , size=tfs
      , color='rgba(0, 0, 0, 1)'
    )
  , row=4, col=5)
fig.add_annotation(
    x=xc1, y=yr0-dyr
  , text=str(R7p_in_aMe12)
  , align='left'
  , showarrow=False
  , font=dict(
        family="Arial"
      , size=tfs
      , color='rgba(0, 0, 0, 1)'
    )
  , row=4, col=5)
fig.add_annotation(
    x=xc2, y=yr0-dyr
  , text=str(R7y_in_aMe12)
  , align='left'
  , showarrow=False
  , font=dict(
        family="Arial"
      , size=tfs
      , color='rgba(0, 0, 0, 1)'
    )
  , row=4, col=5)
fig.add_trace(
    go.Scatter(
        x=[xc1-dxl*1.5, xc2+dxl*1.5]
      , y=[yr0-dyr/2, yr0-dyr/2]
      , mode='lines'
      , line=dict(color='black', width=dtlw)
    )
  , row=4, col=5)
fig.add_trace(
    go.Scatter(
        x=[(xc1+xc2)/2, (xc1+xc2)/2]
      , y=[yr0-dyr*3/2, yr0+dyr/2]
      , mode='lines'
      , line=dict(color='black', width=dtlw)
    )
  , row=4, col=5)
# Layout
fig.update_layout(
    xaxis15=dict(visible=False, range=[-10, 20])
  , yaxis15=dict(visible=False, range=[110, 160]))
fig.add_annotation(
    xref="x domain"
  , yref="y domain"
  , x=-0.2, y=1.00
  , text="<b>f<b>"
  , showarrow=False
  , font=dict(
        family="Arial"
      , size=8
      , color='rgba(0, 0, 0, 1)'
    )
  , row=4, col=5)

# Panel F2: R8p/y synapses with aMe12
fig.add_trace(
    go.Histogram(
        x=R8p_aMe12_df['weight']
      , xbins=dict(start=0.5, end=24.5, size=1)
      , marker=dict(
            color='rgba(148, 56, 131, 1)'
          , line=dict(color='rgba(148, 56, 131, 1)')
        )
    )
  , row=5, col=5)
fig.add_trace(
    go.Histogram(
        x=R8y_aMe12_df['weight']
      , xbins=dict(start=0.5, end=18.5, size=1)
      , marker=dict(
            color='rgba(254, 199, 43, 1)'
          , line=dict(color='rgba(254, 199, 43, 1)')
        )
    )
  , row=5, col=5)
fig.add_annotation(
    x=19, y=14
  , text="R8p"
  , showarrow=False
  , font=dict(
        family="Arial"
      , size=7
      , color='rgba(148, 56, 131, 1)'
    )
  , row=5, col=5)
fig.add_annotation(
    x=19, y=12
  , text="R8y"
  , showarrow=False
  , font=dict(
        family="Arial"
      , size=7
      , color='rgba(254, 199, 43, 1)'
    )
  , row=5, col=5)
fig.update_layout(
    xaxis19=dict(
        title=dict(
            text="#connections to aMe12"
          , standoff=xso
          , font=dict(size=6)
        )
      , linewidth=0.25
      , showline=True
      , linecolor='black'
      , dtick=3
      , tick0=1
      , tickwidth=0.25
      , range=[0, 24.5]
      , ticklen=tl
      , ticks='outside'
      , tickfont=dict(size=5)
    )
  , yaxis19=dict(
        title=dict(
            text="#cells"
          , standoff=yso
          , font=dict(size=6)
        )
      , linewidth=0.25
      , showline=True
      , linecolor='black'
      , dtick=5
      , tick0=0
      , tickwidth=0.25
      , range=[-0.5, 10]
      , ticklen=tl
      , ticks='outside'
      , tickfont=dict(size=5)
    )
  , barmode='overlay')

fig.show()

out_file = PROJECT_ROOT / 'results' / 'pale_yellow' / 'Pale_yellow_figure.pdf'
out_file.parent.mkdir(exist_ok=True, parents=True)

fig.write_image(out_file)

# %%
