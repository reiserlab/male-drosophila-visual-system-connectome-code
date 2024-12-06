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

# %%
"""
This cell does the initial project setup.
If you start a new script or notebook, make sure to copy & paste this part.

A script with this code uses the location of the `.env` file as the anchor for
the whole project (= PROJECT_ROOT). Afterwards, code inside the `src` directory
are available for import.
"""
import sys
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv, find_dotenv

load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath('src')))

from utils import olc_client
from utils.ol_types import OLTypes
from utils.ol_color import OL_COLOR
from utils.overall_summary_queries import get_conn_ol_df
from utils.chiasm_connectivity_plotting_function import plot_chiasm_connectivity
c = olc_client.connect(verbose=True)

# %% [markdown]
# Getting all the cell instances from OLTypes

# %%
ol = OLTypes()
types = ol.get_neuron_list(side='both')
cell_instances = types['instance']

# %% [markdown]
# Loading the optic lobe connectivity dataframe

# %%
conn_ol_df = get_conn_ol_df()

# %% [markdown]
# Getting the connectivity within non primary regions

# %%
prepost_notprimary_df_sorted = conn_ol_df[conn_ol_df['roi']=='NotPrimary']\
    .groupby(['type_pre', 'type_post'])\
    .agg({'weight': 'sum'})\
    .sort_values(by='weight', ascending=False)\
    .reset_index()\
    .rename(columns={'weight': 'NotPrimary_weight'})

# %% [markdown]
# Getting the connectivity of the same cell pairs within all regions

# %%
prepost_allroi_df_sorted = conn_ol_df\
    .groupby(['type_pre', 'type_post'])\
    .agg({'weight': 'sum'})\
    .sort_values(by='weight', ascending=False)\
    .reset_index()\
    .rename(columns={'weight': 'AllRoi_weight'})

# %%
merge_df = prepost_notprimary_df_sorted\
    .merge(prepost_allroi_df_sorted)\
    .assign(
        cell_pairs = lambda row: row['type_pre'] + "-" + row['type_post']
      , chiasm_frac = lambda row: row['NotPrimary_weight'] / row['AllRoi_weight']
    )

# %% [markdown]
# Removing connections <200 and connections with chiasm fraction <0.05

# %%
non_primary_synapses_df = merge_df\
    .query('NotPrimary_weight >= 200')\
    .query('chiasm_frac >= 0.05')\
    .reset_index(drop=True)

# %%
save_path = PROJECT_ROOT / "results" / "summary_plots"

save_path.mkdir(parents=True, exist_ok=True)
non_primary_synapses_df.to_csv(save_path / "non_primary_synapses.csv")

# %% [markdown]
# adding main groups to dataframe

# %%
type_groups = types\
    .loc[:, ['type', 'main_groups']]\
    .drop_duplicates(subset='type')

main_groups_all_chiasm_df = non_primary_synapses_df\
    .merge(type_groups, left_on='type_pre', right_on='type')\
    .merge(type_groups, left_on='type_post', right_on='type')\
    .drop(columns=['type_x', 'type_y'])\
    .rename(columns={
        'main_groups_x': 'main_groups_pre'
      , 'main_groups_y': 'main_groups_post'
    })\
    .sort_values(by='NotPrimary_weight', ascending=False)

# %% [markdown]
# adding colors to the dataframe

# %%
colors = pd.DataFrame(
    data={
        'groups': OL_COLOR.OL_TYPES.map.keys()
      , 'color': OL_COLOR.OL_TYPES.map.values()
    }
)
colors['groups'] = ['OL_intrinsic', 'OL_connecting', 'VPN', 'VCN', 'other']
colors.columns = ['main_group', 'color']

# %%
# First merge for 'main_groups_pre' to extract 'color_pre'
color_pre_chiasm_df = pd\
    .merge(
        main_groups_all_chiasm_df
      , colors
      , how='left'
      , left_on='main_groups_pre'
      , right_on='main_group')\
    .rename(columns={'color': 'color_pre'})\
    .drop(['main_group'], axis=1)\
    .reset_index(drop=True)

# Second merge for 'main_groups_post' to extract 'color_post'
color_all_chiasm_df = pd\
    .merge(
        color_pre_chiasm_df
      , colors
      , how='left'
      , left_on='main_groups_post'
      , right_on='main_group')\
    .rename(columns={'color': 'color_post'})\
    .drop(['main_group'], axis=1)\
    .reset_index(drop=True)\
    .assign(chiasm_frac = lambda row: row['chiasm_frac'].round(2))

color_all_chiasm_df.to_excel(save_path / "chiasm_connectivity.xlsx")

# %% [markdown]
# plot # non primary connections for these cell pairs

# %%
# set formatting parameters
style = {
    'export_type': 'svg'
  , 'font_type': 'arial'
  , 'markerlinecolor': 'black'
  , 'linecolor': 'black'
  , 'opacity': 0.7
}

sizing = {
    'fig_width': 300
  , 'fig_height': 100
  , 'fig_margin': 0
  , 'fsize_ticks_pt': 3
  , 'fsize_title_pt': 7
  , 'markersize': 2.5
  , 'ticklen': 3.5
  , 'tickwidth': 0.1
  , 'axislinewidth': 1
  , 'markerlinewidth': 0.5
}

plot_specs = {
    'range_y': [0, 8000]
  , 'tickvals_y': [0, 4000, 8000]
  , 'save_path': PROJECT_ROOT / 'results' / 'summary_plots'
  , 'export_type': 'svg'
  , 'plot_name': 'chiasm_connectivity'
}

color_all_chiasm_df['chiasm_frac'] = color_all_chiasm_df['chiasm_frac'].round(2)

fig = plot_chiasm_connectivity(
    df=color_all_chiasm_df
  , xval='cell_pairs'
  , yval1='NotPrimary_weight'
  , yval2='chiasm_frac'
  , style=style
  , sizing=sizing
  , plot_specs=plot_specs
)

fig.show()

# %%
