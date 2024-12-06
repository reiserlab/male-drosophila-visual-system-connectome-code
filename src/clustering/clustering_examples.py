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
import sys
from pathlib import Path

import pandas as pd

from IPython.display import display

from dotenv import load_dotenv, find_dotenv

from neuprint import NeuronCriteria as NC, merge_neuron_properties
from neuprint.queries import fetch_neurons, fetch_adjacencies


load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent

sys.path.append(str(PROJECT_ROOT.joinpath('src')))
print(f"Project root directory: {PROJECT_ROOT}")

from utils import olc_client
from utils.clustering_functions import cluster_neurons, generate_clustering_data


# %%
c = olc_client.connect(verbose=True)

data_dir = PROJECT_ROOT / 'results' / 'clustering' / 'clustering_results_for_figures'
cache_dir = PROJECT_ROOT / 'cache' / 'clustering'

data_dir.mkdir(parents=True, exist_ok=True)

# %%
bid_type, exclude_from_clustering, fragment_type_dict = generate_clustering_data()

# %%
display(fragment_type_dict)

# %%
## list of bodyIds to cluster (here: based on region and synapse numbers and whether the
#  body has type or instance name)

criteria = NC(rois=['ME(R)', 'LO(R)', 'LOP(R)','AME(R)','LA(R)'], roi_req='any')
neurons_all, _ = fetch_neurons(criteria)
## the >100 threshold is ok for most OL cells except a few near the edges
neurons_all = neurons_all[neurons_all.synweight > 100]

neuron_selection = list(set(neurons_all.bodyId.tolist( ) + list(bid_type.keys())))
display(f"Number of selected neurons: {len(neuron_selection)}")

# %%
## get up- and downstream synaptic partners of all bodies in neuron_selection

cache_target_fn = cache_dir / "ROL_targets_df_neuprint_only_102023_v11.pickle"

if cache_target_fn.is_file():
    ## load dataframes with connection data (faster than getting these from neuprint
    ## and soon no further changes will be expected for the right optic lobe for now)
    conn_df_targets = pd.read_pickle(cache_target_fn)
else:
    criteria = NC(bodyId=neuron_selection)
    neuron_df1, conn_df1 = fetch_adjacencies(criteria, None, include_nonprimary=False) # targets
    conn_df_targets = merge_neuron_properties(neuron_df1, conn_df1)
    del neuron_df1, conn_df1
    ## save dataframes with connection data (reload is faster than getting these from neuprint)
    cache_dir.mkdir(exist_ok=True)
    conn_df_targets.to_pickle(cache_target_fn)

# %%
cache_input_fn  = cache_dir / "ROL_inputs_df_neuprint_only_102023_v11.pickle"

if cache_input_fn.is_file():
    ## load dataframes with connection data (faster than getting these from neuprint
    ## and soon no further changes will be expected for the right optic lobe for now)
    conn_df_inputs = pd.read_pickle(cache_input_fn)
else:
    criteria = NC(bodyId=neuron_selection)
    neuron_df2, conn_df2 = fetch_adjacencies(None, criteria, include_nonprimary=False) # inputs
    conn_df_inputs = merge_neuron_properties(neuron_df2, conn_df2)
    del neuron_df2, conn_df2
    ## save dataframes with connection data (reload is faster than getting these from neuprint)
    cache_dir.mkdir(exist_ok=True)
    conn_df_inputs.to_pickle(cache_input_fn)


# %% [markdown]
# ## clustering examples
#
# ### example 1
#
# clustering a subset of neurons
#
# to run this for the full optic lobe, set `cell_list=neuron_selection` (see above)
#     and number_of_clusters to e.g. 600
#
# example: set of cell types with one cell per column shown in Figure 2

# %%
type_selection = [
    'L1', 'L2', 'L3', 'L5'
  , 'Mi1', 'Mi4', 'Mi9'
  , 'C2', 'C3', 'T1'
  , 'Tm1', 'Tm2', 'Tm4', 'Tm9', 'Tm20'
]

cells_per_cluster_by_type = cluster_neurons(
    type_selection=type_selection
  , bid_type=bid_type
  , exclude_from_clustering=exclude_from_clustering
  , exclude=False
  , fragment_type_dict=fragment_type_dict
  , input_df=conn_df_inputs
  , output_df=conn_df_targets
  , number_of_clusters=len(type_selection) # one cluster per type in this case
)

cells_per_cluster_by_type.to_csv(data_dir / 'clustering_Fig2d.csv')

display(cells_per_cluster_by_type)

# %% [markdown]
# ### example 2
#
# clustering example: cell types with synapses only in ME(R) (Dm,Cm,Pm and Mi cells)
#     and at least 10 instances (cells) per type

# %%
type_selection = (
    list({
        cell_type for cell_type in bid_type.values() \
            if cell_type.startswith(('Dm', 'Cm', 'Pm', 'Mi'))
    })
)

# cell types with at least 10 instances
type_selection = [
    cell_type for cell_type in type_selection \
        if len(
            [bodyId for bodyId in bid_type.keys() if bid_type[bodyId] == cell_type]
        )>=10]

# exclude named fragments
type_selection = [cell_type for cell_type in type_selection if not 'fragment' in cell_type]

cells_per_cluster_by_type = cluster_neurons(
    type_selection=type_selection
  , bid_type=bid_type
  , exclude_from_clustering=exclude_from_clustering
  , exclude=False
  , fragment_type_dict=fragment_type_dict
  , input_df=conn_df_inputs
  , output_df=conn_df_targets
  , number_of_clusters=80
)

cells_per_cluster_by_type.to_csv(data_dir / 'clustering_ED_Fig3.csv')

display(cells_per_cluster_by_type)

# %% [markdown]
# ### example 3
#
# clustering example: clustering cells without using the connections to selected cell types
#
# Examples: Pairs of Tm5a/Tm5b/Tm29 and Dm8a/Dm8b without using synapses with R7 and R8 types,
#     Dm8/Dm8b, Tm5a/Tm5b and Tm29

# %%
type_selections = [
    ['Tm5a', 'Tm5b'], ['Tm5a', 'Tm5b']
  , ['Tm5a', 'Tm29'], ['Tm29', 'Tm5b']
  , ['Dm8a', 'Dm8b']
]
exclude_R7_R8_Tm5ab_Dm8ab = ['No', 'Yes', 'Yes', 'Yes', 'Yes']


combined_results = pd.DataFrame()

for type_selection, exclude in zip(type_selections, exclude_R7_R8_Tm5ab_Dm8ab):
    cells_per_cluster_by_type = cluster_neurons(
        type_selection=type_selection
      , bid_type=bid_type
      , exclude_from_clustering=exclude_from_clustering
      , exclude=exclude
      , fragment_type_dict=fragment_type_dict
      , input_df=conn_df_inputs
      , output_df=conn_df_targets
      , number_of_clusters=2
    )
    combined_results = pd.concat([combined_results, cells_per_cluster_by_type])

combined_results.to_csv(data_dir / 'clustering_ED_Fig5.csv')

display(combined_results)
