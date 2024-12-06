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

from neuprint import NeuronCriteria as NC, merge_neuron_properties
from neuprint.queries import fetch_neurons, fetch_adjacencies

from dotenv import load_dotenv, find_dotenv
load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath('src')))
print(f"Project root directory: {PROJECT_ROOT}")

from utils.clustering_functions import \
    set_pca_for_projections \
  , generate_clustering_data

from utils.clustering_plotting_functions import \
    make_spatialmap_two_clusters_fig \
  , make_spatialmap_three_clusters_fig

from utils import olc_client

# %%
c = olc_client.connect(verbose=True)

data_dir = PROJECT_ROOT / "results" / "clustering"
cache_dir = PROJECT_ROOT / "cache" / "clustering"

# %%
[bid_type,exclude_from_clustering, fragment_type_dict] = generate_clustering_data()

# %%
## list of bodyIds to cluster (here: based on region and synapse numbers and whether the
#  body has type or instance name)

criteria = NC(rois=['ME(R)', 'LO(R)', 'LOP(R)','AME(R)','LA(R)'], roi_req='any')
neurons_all, _ = fetch_neurons(criteria)
## the >100 threshold is ok for most OL cells except a few near the edges
neurons_all = neurons_all[neurons_all.synweight > 100]

neuron_selection= list(set(neurons_all.bodyId.tolist( ) + list(bid_type.keys())))
len(neuron_selection)

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

# %%
## provides coordinate systems for plotting neuron postions in the main OL regions

pca_medulla = set_pca_for_projections(
    cell_type_pre=['Mi1', 'Tm1']
  , cell_type_post='Pm2a'
)

pca_lobula = set_pca_for_projections(
    cell_type_pre=['Tm5a', 'Tm5b-1']
  , cell_type_post=['LC6', 'LC10c', 'LC16']
  , neuropile_region='LO(R)'
)

pca_lobula_plate = set_pca_for_projections(
    cell_type_pre=['T4b', 'T5b']
  , cell_type_post='H2'
  , neuropile_region='LOP(R)'
)

# %% [markdown]
# x and y axis limits: for plotting purposes

# %%
xrange_medulla_L1 = [-13000, 12300]
yrange_medulla_L1 = [-19474.62704644016, 20873.851667095125]

xrange_lobula = [-6190.723084580102, 7880.732504892222]
yrange_lobula = [-10823.824799486023, 13200]

xrange_lobula_plate = [-7111.968436474855, 9258.252129121123]
yrange_lobula_plate = [-12574.663667345072, 15279.919872641707]

# For Pm2 alone, reduced XY limits
xrange_medulla_L1_Pm2 = [xrange_medulla_L1[0] + 2500, xrange_medulla_L1[1] - 2500]
yrange_medulla_L1_Pm2 = [yrange_medulla_L1[0] + 3000, yrange_medulla_L1[1] - 3000]

# %% [markdown]
# size and plotting parameters

# %%
# for the desired size: 90x80 mm
# set formatting parameters
style = {
    'export_type': 'svg'
  , 'font_type': 'arial'
  , 'markerlinecolor_spatialmap': 'black'
  , 'markerlinecolor': 'black'
  , 'linecolor': 'black'
  , 'opacity_spatialmap': 0.7
  , 'opacity': 0.6
  , 'jitter_extent': 0.8
  , 'jitter_extent_3C': 0.8 # jitter extent three clusters
  , 'x_centers': [1, 5]
  , 'x_deviation': 1.8
  , 'x_centers_three_clusters': [1, 5, 9]
  , 'x_deviation_three_clusters': 1.8
}

sizing = {
    'fig_width': 90
  , 'fig_height': 80
  , 'fig_margin': 0
  , 'fsize_ticks_pt': 5
  , 'fsize_title_pt': 7
  , 'markersize': 2
  , 'ticklen': 2
  , 'tickwidth': 0.7
  , 'axislinewidth': 0.65
  , 'markerlinewidth': 0.2
  , 'markerlinewidth_spatialmap': 0.02
}

# %% [markdown]
# All the following distribution plots, along with the spatial maps, show the number of synapses
# for the two/three clusters of a given cell type with some selected inputs and outputs. These
# input and output celltypes were manually chosen to display a diverse set of morphological
# and connectivity differences that underlie cell-typing

# %% [markdown]
# # Figure 2 main:

# %% [markdown]
# ## Pm2a/Pm2b
#
# Used in manuscript's Fig. 2g,h

# %%
main_conn_celltypes = ['TmY16-IN', 'TmY5a-IN', 'Pm10-OUT', 'Mi1-OUT']
plot_specs = {
    'range_x' : xrange_medulla_L1_Pm2
  , 'range_y' : yrange_medulla_L1_Pm2
  , 'save_path': PROJECT_ROOT / 'results' / 'clustering'
}

# Order: reversed
# for cases where the cluster order returned by the
# function is different from the required plotting order

fig = make_spatialmap_two_clusters_fig(
    rois=['ME(R)']
  , pca_roi=pca_medulla
  , type_names=['Pm2b', 'Pm2a']
  , type_name='Pm2'
  , input_df=conn_df_inputs
  , output_df=conn_df_targets
  , bid_type=bid_type
  , exclude_from_clustering=exclude_from_clustering
  , fragment_type_dict=fragment_type_dict
  , main_conn_celltypes=main_conn_celltypes
  , marker_size_sf=1
  , order='reversed'
  , style=style
  , sizing=sizing
  , plot_specs=plot_specs
)

fig.show()

# %% [markdown]
# ## Mi4, Mi9
#
# Used in manuscript's Fig. 2c

# %%
# desired_connection_indices_Mi4_Mi9 = [0, -2, 2, -1]
main_conn_celltypes = ['L5-IN', 'L3-IN', 'Mi9-OUT', 'Mi10-OUT']
plot_specs = {
    'range_x': xrange_medulla_L1
  , 'range_y': yrange_medulla_L1
  , 'save_path': PROJECT_ROOT / 'results' / 'clustering'
}

fig = make_spatialmap_two_clusters_fig(
    rois=['ME(R)']
  , pca_roi=pca_medulla
  , type_names=['Mi4', 'Mi9']
  , type_name='Mi4_Mi9'
  , input_df=conn_df_inputs
  , output_df=conn_df_targets
  , bid_type=bid_type
  , exclude_from_clustering=exclude_from_clustering
  , fragment_type_dict=fragment_type_dict
  , main_conn_celltypes=main_conn_celltypes
  , marker_size_sf=1
  , order='straight'
  , style=style
  , sizing=sizing
  , plot_specs=plot_specs
)

fig.show()

# %% [markdown]
# # Extended data (ED) figure 4
#
# ## ED 4a
#
# ### TmY9a/TmY9b

# %%
main_conn_celltypes = ['Dm3b-IN', 'Dm3a-IN', 'Y13-OUT', 'TmY9b-OUT']
plot_specs = {
    'range_x': xrange_medulla_L1
  , 'range_y': yrange_medulla_L1
  , 'save_path': PROJECT_ROOT / 'results' / 'clustering'
}

fig = make_spatialmap_two_clusters_fig(
    rois=['ME(R)']
  , pca_roi=pca_medulla
  , type_names=['TmY9a', 'TmY9b']
  , type_name='TmY9'
  , input_df=conn_df_inputs
  , output_df=conn_df_targets
  , bid_type=bid_type
  , exclude_from_clustering=exclude_from_clustering
  , fragment_type_dict=fragment_type_dict
  , main_conn_celltypes=main_conn_celltypes
  , marker_size_sf=1
  , order='straight'
  , style=style
  , sizing=sizing
  , plot_specs=plot_specs
)

fig.show()

# %% [markdown]
# ### Dm3a/b/c

# %%

main_conn_celltypes = ['TmY9b-IN', 'TmY4-IN', 'TmY9b-OUT', 'TmY9a-OUT']
plot_specs = {
    'range_x': xrange_medulla_L1
  , 'range_y': yrange_medulla_L1
  , 'save_path': PROJECT_ROOT / 'results' / 'clustering'
}

fig = make_spatialmap_three_clusters_fig(
    rois=['ME(R)']
  , pca_roi=pca_medulla
  , type_names=['Dm3a', 'Dm3c', 'Dm3b']
  , type_name='Dm3'
  , input_df=conn_df_inputs
  , output_df=conn_df_targets
  , bid_type=bid_type
  , exclude_from_clustering=exclude_from_clustering
  , fragment_type_dict=fragment_type_dict
  , main_conn_celltypes=main_conn_celltypes
  , marker_size_sf=0.7
  , style=style
  , sizing=sizing
  , plot_specs=plot_specs
)

fig.show()

# %% [markdown]
# ## ## ED 4b
#
# ### Cm2/Cm4

# %%

main_conn_celltypes = ['Dm8b-IN', 'Mi15-IN', 'Cm17-OUT', 'Cm6-OUT']
plot_specs = {
    'range_x': xrange_medulla_L1
  , 'range_y': yrange_medulla_L1
  , 'save_path': PROJECT_ROOT / 'results' / 'clustering'
}

fig = make_spatialmap_two_clusters_fig(
    rois=['ME(R)']
  , pca_roi=pca_medulla
  , type_names=['Cm4', 'Cm2']
  , type_name='Cm'
  , input_df=conn_df_inputs
  , output_df=conn_df_targets
  , bid_type=bid_type
  , exclude_from_clustering=exclude_from_clustering
  , fragment_type_dict=fragment_type_dict
  , main_conn_celltypes=main_conn_celltypes
  , marker_size_sf=1
  , order='reversed'
  , style=style
  , sizing=sizing
  , plot_specs=plot_specs
)

fig.show()

# %% [markdown]
# ### LC14a-1

# %%

main_conn_celltypes = ['LC9-IN', 'TmY5a-IN', 'MeLo13-OUT', 'LC17-OUT']
plot_specs = {
    'range_x': xrange_lobula
  , 'range_y': yrange_lobula
  , 'save_path': PROJECT_ROOT / 'results' / 'clustering'
}

fig = make_spatialmap_two_clusters_fig(
    rois=['LO(R)']
  , pca_roi=pca_lobula
  , type_names=['LC14a-1_R', 'LC14a-1_L']
  , type_name='LC14a-1'
  , input_df=conn_df_inputs
  , output_df=conn_df_targets
  , bid_type=bid_type
  , exclude_from_clustering=exclude_from_clustering
  , fragment_type_dict=fragment_type_dict
  , main_conn_celltypes=main_conn_celltypes
  , marker_size_sf=1
  , order='straight'
  , style=style
  , sizing=sizing
  , plot_specs=plot_specs
)

fig.show()

# %% [markdown]
# ## ED 4c
#
# ### Mi15

# %%

main_conn_celltypes = ['Mi1-IN', 'Cm35-IN', 'Cm5-OUT', 'MeTu3b-OUT']
plot_specs = {
    'range_x': xrange_medulla_L1
  , 'range_y': yrange_medulla_L1
  , 'save_path': PROJECT_ROOT / 'results' / 'clustering'
}

fig = make_spatialmap_two_clusters_fig(
    rois=['ME(R)']
  , pca_roi=pca_medulla
  , type_names=['Mi15']
  , type_name='Mi15'
  , input_df=conn_df_inputs
  , output_df=conn_df_targets
  , bid_type=bid_type
  , exclude_from_clustering=exclude_from_clustering
  , fragment_type_dict=fragment_type_dict
  , main_conn_celltypes=main_conn_celltypes
  , marker_size_sf=0.8
  , order='straight'
  , style=style
  , sizing=sizing
  , plot_specs=plot_specs
)

fig.show()

# %% [markdown]
# ### Dm2</i></b>

# %%

main_conn_celltypes = ['Cm3-IN', 'Tm5c-IN', 'Cm3-OUT', 'Cm9-OUT']
plot_specs = {
    'range_x': xrange_medulla_L1
  , 'range_y': yrange_medulla_L1
  , 'save_path': PROJECT_ROOT / 'results' / 'clustering'
}

fig = make_spatialmap_two_clusters_fig(
    rois=['ME(R)']
  , pca_roi=pca_medulla
  , type_names=['Dm2']
  , type_name='Dm2'
  , input_df=conn_df_inputs
  , output_df=conn_df_targets
  , bid_type=bid_type
  , exclude_from_clustering=exclude_from_clustering
  , fragment_type_dict=fragment_type_dict
  , main_conn_celltypes=main_conn_celltypes
  , marker_size_sf=0.8
  , order='straight'
  , style=style
  , sizing=sizing
  , plot_specs=plot_specs
)

fig.show()

# %% [markdown]
# ## ED 4d
#
# ### L5

# %%

main_conn_celltypes = ['L1-IN', 'C2-IN', 'Tm3-OUT', 'C3-OUT']
plot_specs = {
    'range_x': xrange_medulla_L1
  , 'range_y': yrange_medulla_L1
  , 'save_path': PROJECT_ROOT / 'results' / 'clustering'
}

fig = make_spatialmap_two_clusters_fig(
    rois=['ME(R)']
  , pca_roi=pca_medulla
  , type_names=['L5']
  , type_name='L5'
  , input_df=conn_df_inputs
  , output_df=conn_df_targets
  , bid_type=bid_type
  , exclude_from_clustering=exclude_from_clustering
  , fragment_type_dict=fragment_type_dict
  , main_conn_celltypes=main_conn_celltypes
  , marker_size_sf=0.8
  , order='straight'
  , style=style
  , sizing=sizing
  , plot_specs=plot_specs
)

fig.show()

# %% [markdown]
# ### LC9

# %%
main_conn_celltypes = ['Tm5Y-IN', 'LC18-IN', 'LC14a-1_R-OUT', 'LC10a-OUT']
plot_specs = {
    'range_x': xrange_lobula
  , 'range_y': yrange_lobula
  , 'save_path': PROJECT_ROOT / 'results' / 'clustering'
}

fig = make_spatialmap_two_clusters_fig(
    rois=['LO(R)']
  , pca_roi=pca_lobula
  , type_names=['LC9']
  , type_name='LC9'
  , input_df=conn_df_inputs
  , output_df=conn_df_targets
  , bid_type=bid_type
  , exclude_from_clustering=exclude_from_clustering
  , fragment_type_dict=fragment_type_dict
  , main_conn_celltypes=main_conn_celltypes
  , marker_size_sf=1
  , order='straight'
  , style=style
  , sizing=sizing
  , plot_specs=plot_specs
)

fig.show()

# %% [markdown]
# ## ED 4e
#
# ### L1

# %%

main_conn_celltypes = ['R1-R6-IN', 'Mi1-IN', 'L2-OUT', 'Tm3-OUT']
plot_specs = {
    'range_x': xrange_medulla_L1
  , 'range_y': yrange_medulla_L1
  , 'save_path': PROJECT_ROOT / 'results' / 'clustering'
}

fig = make_spatialmap_two_clusters_fig(
    rois=['ME(R)']
  , pca_roi=pca_medulla
  , type_names=['L1']
  , type_name='L1'
  , input_df=conn_df_inputs
  , output_df=conn_df_targets
  , bid_type=bid_type
  , exclude_from_clustering=exclude_from_clustering
  , fragment_type_dict=fragment_type_dict
  , main_conn_celltypes=main_conn_celltypes
  , marker_size_sf=0.8
  , order='straight'
  , style=style
  , sizing=sizing
  , plot_specs=plot_specs
)

fig.show()

# %% [markdown]
# ### LC12

# %%

main_conn_celltypes = ['Tm4-IN', 'T2-IN', 'LC12-OUT', 'LoVC16-OUT']
plot_specs = {
    'range_x': xrange_lobula
  , 'range_y': yrange_lobula
  , 'save_path': PROJECT_ROOT / 'results' / 'clustering'
}

fig = make_spatialmap_two_clusters_fig(
    rois=['LO(R)']
  , pca_roi=pca_lobula
  , type_names=['LC12']
  , type_name='LC12'
  , input_df=conn_df_inputs
  , output_df=conn_df_targets
  , bid_type=bid_type
  , exclude_from_clustering=exclude_from_clustering
  , fragment_type_dict=fragment_type_dict
  , main_conn_celltypes=main_conn_celltypes
  , marker_size_sf=1
  , order='straight'
  , style=style
  , sizing=sizing
  , plot_specs=plot_specs
)

fig.show()

# %% [markdown]
# # Extended data figure 5
#
# ## ED 5d: 
#
# ### Tm5a, Tm5b

# %%

main_conn_celltypes = ['Dm8a-IN', 'Dm8b-IN', 'LT58-OUT', 'Tm29-OUT']
plot_specs = {
    'range_x': xrange_medulla_L1
  , 'range_y': yrange_medulla_L1
  , 'save_path': PROJECT_ROOT / 'results' / 'clustering'
}
sizing = {
    'fig_width':120 #53.5, # 60 # units = mm, max 180
  , 'fig_height':120 # units = mm, max 170
  , 'fig_margin':0
  , 'fsize_ticks_pt':5
  , 'fsize_title_pt':7
  , 'markersize':4
  , 'ticklen':2
  , 'tickwidth':0.7
  , 'axislinewidth':0.65
  , 'markerlinewidth':0.02
}

fig = make_spatialmap_two_clusters_fig(
    rois=['ME(R)']
  , pca_roi=pca_medulla
  , type_names=['Tm5a', 'Tm5b']
  , type_name='Tm5'
  , input_df=conn_df_inputs
  , output_df=conn_df_targets
  , bid_type=bid_type
  , exclude_from_clustering=exclude_from_clustering
  , fragment_type_dict=fragment_type_dict
  , main_conn_celltypes=main_conn_celltypes
  , marker_size_sf=0.8
  , order='straight'
  , style=style
  , sizing=sizing
  , plot_specs=plot_specs
)

fig.show()
