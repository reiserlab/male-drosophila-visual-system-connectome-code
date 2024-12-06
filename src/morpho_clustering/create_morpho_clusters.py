# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: ol-c-kernel
#     language: python
#     name: ol-c-kernel
# ---

# %%
from pathlib import Path
import sys
from dotenv import load_dotenv, find_dotenv
load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath('src')))

from utils import olc_client
c = olc_client.connect(verbose=True)

# %%
import pandas as pd
from utils.morpho_clustering_functions import create_morpho_data, cluster_morpho_data, create_morpho_confusion_matrix, find_clustering_scores
from utils.plotting_functions import plot_confusion_matrix_w_colors, plot_morpho_feature_vectors

# %%
#number of clusters
n_clusters = 80

#only look at synapses in rois_list
rois = ['ME(R)']

#only take cells from these cell types
types = [
    'Cm22', 'Cm10', 'Dm8b', 'Cm21', 'Cm11c', 'Mi10', 'Cm3', 'Mi2', 'Pm2b'
  , 'Dm14', 'Cm9', 'Pm10', 'Cm11a', 'Cm16', 'Mi14', 'Mi15', 'Dm12', 'Mi17'
  , 'Pm6', 'Dm13', 'Cm17', 'Dm19', 'Cm19', 'Dm9', 'Dm6', 'Pm9', 'Mi18', 'Cm15'
  , 'Cm5', 'Cm6', 'Pm4', 'Cm14', 'Mi16', 'Mi4', 'Dm3a', 'Cm2', 'Dm3b', 'Dm4'
  , 'Pm7', 'Pm1', 'Cm8', 'Mi1', 'Pm5', 'Dm3c', 'Dm20', 'Dm16', 'Dm1', 'Dm11'
  , 'Dm8a', 'Cm12', 'Pm8', 'Cm7', 'Mi9', 'Cm1', 'Cm13', 'Cm11b', 'Cm20', 'Cm4'
  , 'Dm18', 'Dm-DRA2', 'Dm15', 'Pm3', 'Pm2a', 'Dm-DRA1', 'Cm18', 'Dm10', 'Mi13', 'Dm2'
]

# %% [markdown]
# ### cluster morphology feature vectors

# %%
create_morpho_data(types, rois)

# expected runtime: 18 mins

# %%
cluster_morpho_data(n_clusters)

hom, com = find_clustering_scores(n_clusters)
print( f'Homogeneity score: {hom:.2f}')
print( f'Completeness score: {com:.2f}')

# %% [markdown]
# ### plot confusion matrix

# %%
create_morpho_confusion_matrix(n_clusters)

# %%
#load confusion matrix and plot
data_path = Path(find_dotenv()).parent / 'cache' / 'morpho_clustering'
data = pd.read_csv(data_path / f'morpho_confusion_mat_{n_clusters}clu.csv')

plot_confusion_matrix_w_colors(data.values[:,1:], data.values[:,0])

# %% [markdown]
# ### plot feature vectors of some cells

# %%
bodyId_list = [56564, 22177, 20423, 13869, 12540]

plot_morpho_feature_vectors(bodyId_list, rois[0], n_clusters)

# %%
