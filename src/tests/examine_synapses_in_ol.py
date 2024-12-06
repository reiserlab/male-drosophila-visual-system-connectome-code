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
# %load_ext autoreload
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
sys.path.append(str(PROJECT_ROOT.joinpath("src")))
print(f"Project root directory: {PROJECT_ROOT}")

from utils import olc_client
c = olc_client.connect(verbose=True)


# %%
from utils.ol_types import OLTypes
olt = OLTypes()

# %%
import pandas as pd
from neuprint import fetch_custom
import warnings
warnings.filterwarnings("error")

def get_con_weights(classification):
    """ 
    One way to calculate the connections is looking at actual neuron-to-neuron connections.
    """

    neuron_list = olt.get_neuron_list(primary_classification=classification, side='both')
    df = pd.DataFrame()
    for name, test in neuron_list.iterrows():
        cql = f"""
            MATCH (n:Neuron)-[e:ConnectsTo]->(m:Neuron)
            WHERE n.instance='{test['instance']}'
            WITH apoc.convert.fromJsonMap(e.roiInfo) AS eri
              , n
              , m
              , e.weight AS wgt
            WITH 
                COALESCE(eri['ME(R)']['post'], 0) as me_wgt
              , COALESCE(eri['LA(R)']['post'], 0) as la_wgt
              , COALESCE(eri['LO(R)']['post'], 0) as lo_wgt
              , COALESCE(eri['LOP(R)']['post'], 0) as lop_wgt
              , COALESCE(eri['AME(R)']['post'], 0) as ame_wgt
              , COALESCE(eri['OL(R)']['post'], 0) as ol_wgt
              , wgt
              , n
              , m
              , eri
            RETURN 
                n.bodyId AS bodyId
              , n.instance AS instance
              , '{classification}' as class
              , sum(wgt) as total_weight
              , sum(ol_wgt) AS OL_weight
              , sum(la_wgt) AS LA_weight
              , sum(me_wgt) AS ME_weight
              , sum(lop_wgt) AS LOP_weight
              , sum(lo_wgt) AS LO_weight
              , sum(ame_wgt) AS AME_weight 
              , toFloat(sum(la_wgt) + sum(me_wgt) + sum(lop_wgt) + sum(lo_wgt) + sum(ame_wgt))/sum(wgt) as five_weight
              """
        named_df = fetch_custom(cql)
        try:
            df = pd.concat([df, named_df])
        except FutureWarning:
            print(f"no results for {test['instance']}")
    return df


def get_neu_weights(neuron_list, area):
    """
    Get the connections from the neurons themselves.

    Parameters
    ----------
    neuron_list : pd.DataFrame
        must contain a column 'instance', which is used in the query
    area : str
        name of the classification

    Returns
    ------
    df : pd.DataFrame
        synaptic weights per bodyId
    """
    df = pd.DataFrame()
    for name, test in neuron_list.iterrows():
        cql = f"""
            MATCH (n:Neuron)
            WHERE n.instance='{test['instance']}'
            WITH apoc.convert.fromJsonMap(n.roiInfo) AS nri, n
            WITH 
                COALESCE(nri['ME(R)']['post'], 0) as me_post_wgt
              , COALESCE(nri['LA(R)']['post'], 0) as la_post_wgt
              , COALESCE(nri['LO(R)']['post'], 0) as lo_post_wgt
              , COALESCE(nri['LOP(R)']['post'], 0) as lop_post_wgt
              , COALESCE(nri['AME(R)']['post'], 0) as ame_post_wgt
              , COALESCE(nri['OL(R)']['post'], 0) as ol_post_wgt
              , COALESCE(n.post, 0) as post_wgt
              , COALESCE(nri['ME(R)']['pre'], 0) as me_wgt
              , COALESCE(nri['LA(R)']['pre'], 0) as la_wgt
              , COALESCE(nri['LO(R)']['pre'], 0) as lo_wgt
              , COALESCE(nri['LOP(R)']['pre'], 0) as lop_wgt
              , COALESCE(nri['AME(R)']['pre'], 0) as ame_wgt
              , COALESCE(nri['OL(R)']['pre'], 0) as ol_wgt
              , COALESCE(n.pre, 0) as pre_wgt
              , COALESCE(nri['ME(R)']['downstream'], 0) as me_down_wgt
              , COALESCE(nri['LA(R)']['downstream'], 0) as la_down_wgt
              , COALESCE(nri['LO(R)']['downstream'], 0) as lo_down_wgt
              , COALESCE(nri['LOP(R)']['downstream'], 0) as lop_down_wgt
              , COALESCE(nri['AME(R)']['downstream'], 0) as ame_down_wgt
              , COALESCE(nri['OL(R)']['downstream'], 0) as ol_down_wgt
              , COALESCE(n.downstream, 0) as down_wgt
              , n
            RETURN 
                n.bodyId AS bodyId
              , n.instance AS instance
              , '{area}' as class
              , sum(la_post_wgt) as la_post_weight
              , sum(me_post_wgt) as me_post_weight
              , sum(lop_post_wgt) as lop_post_weight
              , sum(lo_post_wgt) as lo_post_weight
              , sum(ame_post_wgt) as ame_post_weight
              , sum(ol_post_wgt) as ol_post_weight
              , sum(post_wgt) as post_weight
              , sum(la_wgt) as la_weight
              , sum(me_wgt) as me_weight
              , sum(lop_wgt) as lop_weight
              , sum(lo_wgt) as lo_weight
              , sum(ame_wgt) as ame_weight
              , sum(ol_wgt) as ol_weight
              , sum(pre_wgt) as pre_weight
              , sum(la_down_wgt) as la_down_weight
              , sum(me_down_wgt) as me_down_weight
              , sum(lop_down_wgt) as lop_down_weight
              , sum(lo_down_wgt) as lo_down_weight
              , sum(ame_down_wgt) as ame_down_weight
              , sum(ol_down_wgt) as ol_down_weight
              , sum(down_wgt) as down_weight
              """
        named_df = fetch_custom(cql)
        try:
            df = pd.concat([df, named_df])
        except FutureWarning:
            print(f"no results for {test['instance']}")
    return df


def summarize(df, name):
    """
    Aggregate the connections per bodyId to connections per cell instance and saves it to a file.

    Parameters
    ----------
    df : pd.DataFrame
        data frame to be aggregated
    name : str
        Excel file name to be created inside `results/test`
    """
    out_fn = PROJECT_ROOT / 'results' / 'test' / name
    out_fn.parent.mkdir(parents=True, exist_ok=True)
    df\
        .assign(
            post_5_perc = lambda row: (row['la_post_weight'] + row['me_post_weight'] + row['lop_post_weight'] + row['lo_post_weight'] + row['ame_post_weight']) / row['post_weight']
          , post_5_ol_perc = lambda row: row['ol_post_weight'] / row['post_weight']
          , post_ol_perc = lambda row: (row['la_post_weight'] + row['me_post_weight'] + row['lop_post_weight'] + row['lo_post_weight'] + row['ame_post_weight']) / row['ol_post_weight']
          , pre_5_perc = lambda row: (row['la_weight'] + row['me_weight'] + row['lop_weight'] + row['lo_weight'] + row['ame_weight']) / row['pre_weight']
          , pre_5_ol_perc = lambda row: row['ol_weight'] / row['pre_weight']
          , pre_ol_perc = lambda row: (row['la_weight'] + row['me_weight'] + row['lop_weight'] + row['lo_weight'] + row['ame_weight']) / row['ol_weight']
          , down_5_perc = lambda row: (row['la_down_weight'] + row['me_down_weight'] + row['lop_down_weight'] + row['lo_down_weight'] + row['ame_down_weight']) / row['down_weight']
          , down_5_ol_perc = lambda row: row['ol_down_weight'] / row['down_weight']
          , down_ol_perc = lambda row: (row['la_down_weight'] + row['me_down_weight'] + row['lop_down_weight'] + row['lo_down_weight'] + row['ame_down_weight']) / row['ol_down_weight']
          , post_5_tresh = lambda row: round((row['post_5_perc'])-.1+.5)
          , pre_5_tresh = lambda row: round((row['pre_5_perc'])-.1+.5)
          , down_5_tresh = lambda row: round((row['down_5_perc'])-.1+.5)
        )\
        .groupby('instance')\
        .agg({
            'bodyId': 'size'
          , 'class': 'first'
          , 'post_5_perc': 'mean'
          , 'post_5_tresh': 'sum'
          , 'pre_5_perc': 'mean'
          , 'pre_5_tresh': 'sum'
          , 'down_5_perc': 'mean'
          , 'down_5_tresh': 'sum'
        })\
        .rename(columns={
            'bodyId': 'num. cells of type'
          , 'post_5_perc': '% input connections across all cells of type in the 5 OL neuropils'
          , 'post_5_tresh': '# cells of type above 10% input connections in the 5 OL neuropils'
          , 'down_5_perc': '% output connections across all cells of type in the 5 OL neuropils'
          , 'down_5_tresh': '# cells of type above 10% output connections in the 5 OL neuropils'
          , 'pre_5_perc': 'Bonus: % presynaptic sites across all cells of type in the 5 OL neuropils'
          , 'pre_5_tresh': 'Bonus: # cells of type above 10% presynaptic sites in the 5 OL neuropils'
        })\
        .to_excel(out_fn)


# %%
df_neu = pd.DataFrame()
for area in ['VPN', 'VCN', 'other']:
    neuron_list = olt.get_neuron_list(primary_classification=area, side='both')
    df_neu = pd.concat([df_neu, get_neu_weights(neuron_list, area)])
summarize(df_neu, 'issue_649_vpn-vcn-other.xlsx')

# %% [markdown]
# ## Observed unused neurons
#
# There are a number of neurons that are in neuPrint, but not used in the analysis. Here we 

# %%
neuron_list = olt.get_neuron_list(side='both')

cql = """
    MATCH (n:Neuron) 
    where not n.type is null
    RETURN distinct n.type as type, n.instance as instance
"""
all_n = fetch_custom(cql)
unused_list = all_n[~all_n['type'].isin(neuron_list['type'])]

# %%
df_unused = get_neu_weights(unused_list, 'unused')

summarize(df_unused, 'issue_649_unused.xlsx')

# %% [markdown]
# # Single cell type analysis
#
# The following cell helps looking at the synapse counts relevant for VPN or VCN cells. They show the percentage of total synapses for both hemispheres.

# %%
neuron_type = 'PVLP046'
neuron_class = 'VPN'


match neuron_class:
    case 'VPN':
        dir = 'pre'
    case 'VCN':
        dir = 'post'
    case _:
        dir = 'pre'

cql = f"""
    MATCH (n:Neuron)
    WHERE n.type='{neuron_type}'
    WITH apoc.convert.fromJsonMap(n.roiInfo) AS nri, n
    WITH 
        COALESCE(nri['ME(R)']['{dir}'], 0) as me_wgt
      , COALESCE(nri['LA(R)']['{dir}'], 0) as la_wgt
      , COALESCE(nri['LO(R)']['{dir}'], 0) as lo_wgt
      , COALESCE(nri['LOP(R)']['{dir}'], 0) as lop_wgt
      , COALESCE(nri['AME(R)']['{dir}'], 0) as ame_wgt
      , COALESCE(nri['OL(R)']['{dir}'], 0) as ol_wgt
      , COALESCE(nri['ME(L)']['{dir}'], 0) as mel_wgt
      , COALESCE(nri['LA(L)']['{dir}'], 0) as lal_wgt
      , COALESCE(nri['LO(L)']['{dir}'], 0) as lol_wgt
      , COALESCE(nri['LOP(L)']['{dir}'], 0) as lopl_wgt
      , COALESCE(nri['AME(L)']['{dir}'], 0) as amel_wgt
      , COALESCE(nri['OL(L)']['{dir}'], 0) as oll_wgt
      , COALESCE(n.{dir}, 0) as wgt
      , n
    RETURN 
        n.bodyId AS bodyId
      , n.instance AS instance
      , '{neuron_class}' as class
      , sum(la_wgt) as la_weight
      , sum(me_wgt) as me_weight
      , sum(lop_wgt) as lop_weight
      , sum(lo_wgt) as lo_weight
      , sum(ame_wgt) as ame_weight
      , sum(ol_wgt) as ol_weight
      , sum(lal_wgt) as lal_weight
      , sum(mel_wgt) as mel_weight
      , sum(lopl_wgt) as lopl_weight
      , sum(lol_wgt) as lol_weight
      , sum(amel_wgt) as amel_weight
      , sum(oll_wgt) as oll_weight
      , sum(wgt) as prel_weight
      , toFloat(sum(la_wgt) + sum(me_wgt) + sum(lop_wgt) + sum(lo_wgt) + sum(ame_wgt) + sum(lal_wgt) + sum(mel_wgt) + sum(lopl_wgt) + sum(lol_wgt) + sum(amel_wgt)) / sum(wgt) as perc
      , toFloat(sum(la_wgt) + sum(me_wgt) + sum(lop_wgt) + sum(lo_wgt) + sum(ame_wgt)) / sum(wgt) as perc_r
      , toFloat(sum(lal_wgt) + sum(mel_wgt) + sum(lopl_wgt) + sum(lol_wgt) + sum(amel_wgt)) / sum(wgt) as perc_l
      , toFloat(sum(ol_wgt)) / sum(wgt) as olr_perc
      , toFloat(sum(ol_wgt) + sum(oll_wgt)) / sum(wgt) as ol_perc
"""

df_single = fetch_custom(cql)
df_single

# %%
