from collections import Counter

import numpy as np
import pandas as pd
from sklearn import decomposition

from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import pdist
import fastcluster 

from neuprint import fetch_synapse_connections, fetch_synapses\
  , NeuronCriteria as NC, SynapseCriteria as SC

## functions related to clustering and display of clustering results

def remove_brackets(x):
    """
    FIXME: we need some documentation here.
    """

    x= str(x)
    while x.startswith('('):
        x = x[1:len(x)]
    while x.endswith(')'):
         x = x[0:len(x)-1]
    return x


def make_in_and_output_df(input_df, target_df,bodyId_type,
                          named_types_for_clustering ='all',types_to_exclude = None,
                         fragment_type_dict = None, bodyIds_to_use = None):
    """
    FIXME: we need some documentation here.
    """
    connection_df_inputs = input_df.copy()
    connection_df_targets = target_df.copy()
    
    if bodyIds_to_use != None:
        connection_df_inputs = (connection_df_inputs[
                connection_df_inputs ['bodyId_post'].isin(bodyIds_to_use)])
        connection_df_targets =(connection_df_targets[
                connection_df_targets ['bodyId_pre'].isin(bodyIds_to_use)])
    
    if named_types_for_clustering =='all':
        named_types_for_clustering =list(set(list(bodyId_type.values())))
    
    if types_to_exclude != None:
        named_types_for_clustering=[x for x in named_types_for_clustering if x not in types_to_exclude]
    
    annotate = set_annotations(bodyId_type)
                          
    if fragment_type_dict != None:                      
        rename_fragments = set_annotations(fragment_type_dict)
    else:
        rename_fragments  = lambda x:x    
                          
    if connection_df_inputs.any != None:
        connection_df_inputs['type'] = connection_df_inputs['bodyId_pre'].apply(annotate)
        connection_df_inputs['type'] = connection_df_inputs['type'].apply(rename_fragments)
        

        grouped_df_inputs = connection_df_inputs.groupby(by=['bodyId_post','type'],as_index=False).sum(numeric_only=True)
        grouped_df_inputs = grouped_df_inputs.drop('bodyId_pre',axis=1)

        grouped_df_inputs=grouped_df_inputs[grouped_df_inputs['type'].isin([ x for x in grouped_df_inputs.type if x in named_types_for_clustering])]
        connectivity_table_inputs = grouped_df_inputs.pivot(index='bodyId_post',columns='type',values='weight').fillna(0)
        
        connectivity_table_inputs.columns = [x+'-IN' for x in connectivity_table_inputs.columns]

    if connection_df_targets.any != None:   
        connection_df_targets['type'] = connection_df_targets['bodyId_post'].apply(annotate)
        connection_df_targets['type'] = connection_df_targets['type'].apply(rename_fragments)

        grouped_df_targets = connection_df_targets.groupby(by=['bodyId_pre','type'],as_index=False).sum(numeric_only=True)
        grouped_df_targets = grouped_df_targets.drop('bodyId_post',axis=1)

        grouped_df_targets=grouped_df_targets[grouped_df_targets['type'].isin([ x for x in grouped_df_targets.type if x in named_types_for_clustering])]
        connectivity_table_targets = grouped_df_targets.pivot(index='bodyId_pre',columns='type',values='weight').fillna(0)

        connectivity_table_targets.columns = [x+'-OUT' for x in connectivity_table_targets.columns]
        
    
    connectivity_table = pd.merge(connectivity_table_inputs, connectivity_table_targets, how="outer",left_index=True, right_index=True)
    connectivity_table=connectivity_table.fillna(0)
    connectivity_table = connectivity_table.dropna()
    connectivity_table = connectivity_table.dropna(axis =1)

    return connectivity_table


def get_row_linkage (df_for_clustering,metric='cosine',linkage_method = 'ward',optimal_ordering=False):
    """
    FIXME: we need some documentation here.
    """
    condensed_dist_mat = pdist(df_for_clustering, metric =metric)
    row_linkage = fastcluster.linkage(condensed_dist_mat, method=linkage_method)

    return row_linkage
    

def cluster_dict_from_linkage (row_linkage,df_for_clustering,t=500,criterion = 'maxclust'):
    """
    FIXME: we need some documentation here.
    """
    cluster_assignments = fcluster(row_linkage, t=t, criterion= criterion)
    cluster_numbers = list(set(cluster_assignments))
    cluster_IDs = list(zip(cluster_assignments,list(df_for_clustering.index)))
    cluster_dict = { y:[x[1] for x in cluster_IDs  if x[0]== y] for y in cluster_numbers } # value: list IDs

    return cluster_dict

def set_annotations(annotation_dict):
    """
    FIXME: we need some documentation here.
    """
    def annotate(old_name):
        if old_name in annotation_dict.keys():
            name = annotation_dict[old_name]
        else:
            name = str(old_name)
        return name
    return annotate



def make_count_table (cluster_dict):
    """
    FIXME: we need some documentation here.
    """
    cluster_dict_counts = {}
    for cluster in cluster_dict.keys():
        names = ([x if str(x).isnumeric()==False 
             else "new" for x in cluster_dict[cluster]])
        counter_dict_temp = {cluster:dict(Counter(names))}
        cluster_dict_counts ={**cluster_dict_counts,**counter_dict_temp}
    counts_df = pd.DataFrame(cluster_dict_counts).fillna(0).astype(int)
    return counts_df

def cluster_with_type_names(clusters_bodyIds,bodyId_type):
    
    """
    FIXME: we need some documentation here.
    """
    clusters_cell_types={}
    for n in range(1, len(clusters_bodyIds)+1):
        clusters_cell_types[n] =([bodyId_type[bodyId] if bodyId in bodyId_type.keys() else bodyId for bodyId  in clusters_bodyIds[n]])
    return clusters_cell_types

def format_list(bodyId_list):
    """
    FIXME: we need some documentation here.
    """
    formatted = "".join([str(bodyId)+"," for bodyId in bodyId_list])
    formatted=formatted[0:len(formatted)-1]
    return formatted



## functions for visualization of cell type splits

## set coordinate system via PCA
def set_pca_for_projections (
    cell_type_pre='L1',
    cell_type_post='Mi1',neuropile_region = 'ME(R)'):
    """
    FIXME: we need some documentation here.
    """
    source_criteria =NC(type=cell_type_pre)
    target_criteria = NC(type=cell_type_post)
    synapse_criteria = SC(rois=[neuropile_region], primary_only=True)
    synapses = fetch_synapse_connections (source_criteria,target_criteria,synapse_criteria)

    X = synapses[['x_pre','y_pre','z_pre']] 
    X.columns = ['x','y','z']
    
    pca = decomposition.PCA(n_components=3)
    pca.fit(X)
    return pca

def get_combined_synapses_with_stdev (criteria, synapse_criteria =None, rois=['ME(R)'],pca=[],bodyIds_to_exclude=[]):
    
    synapses = fetch_synapses(criteria,synapse_criteria)
    
    X = synapses[['x','y','z']] 

    X = pca.transform(X)

    synapses['X']=X[:,0]
    synapses['Y']=X[:,1]
    synapses['Z']=X[:,2]
    
    
    combined_synapses = synapses.groupby('bodyId',as_index=False).mean(numeric_only=True)
    combined_synapses ['stdX'] = synapses.groupby('bodyId',as_index=False).std(numeric_only=True)['X']
    combined_synapses ['stdY'] = synapses.groupby('bodyId',as_index=False).std(numeric_only=True)['Y']
    combined_synapses['weight'] = synapses.groupby('bodyId',as_index=False).count()['x']
    
    combined_synapses=combined_synapses[~(combined_synapses['bodyId'].isin(bodyIds_to_exclude))]
    

    combined_synapses_stdev = combined_synapses['weight'].std(numeric_only=True)
    combined_synapses_mean = combined_synapses['weight'].mean(numeric_only=True)
    combined_synapses['z_score'] = ((combined_synapses['weight']-combined_synapses_mean)/combined_synapses_stdev)
    
    
    return combined_synapses


def remove_R_or_L(name):
    if '_R' in name:
        name=name.split('_R')[0]
    if '_L' in name:
        name=name.split('_L')[0]
    return name