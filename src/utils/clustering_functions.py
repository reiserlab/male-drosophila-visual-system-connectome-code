from collections import Counter

import numpy as np
import pandas as pd
from sklearn import decomposition

from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import pdist
import fastcluster

from neuprint import fetch_synapse_connections, fetch_synapses\
  , NeuronCriteria as NC, SynapseCriteria as SC, fetch_neurons, NotNull

def make_in_and_output_df(
    input_df
  , target_df
  , bid_type
  , named_types_for_clustering='all'
  , types_to_exclude=None
  , fragment_type_dict=None
  , bids_to_use=None
  , combine_r7_r8=False
) -> pd.DataFrame:
    """
    This function generates a combined input-output connectivity table for neurons, based on
    synaptic connections. It can process specific body IDs, handle fragment types, and optionally
    combine R7 and R8 types into generalized categories.

    Parameters
    ----------
    input_df : pd.DataFrame
        A DataFrame representing synaptic inputs, with 'bodyId_pre', 'bodyId_post', and 'weight'
        columns.
    target_df : pd.DataFrame
        A DataFrame representing synaptic outputs, with 'bodyId_pre', 'bodyId_post', and 'weight'
        columns.
    bid_type : dict
        A dictionary where keys are body IDs and values are corresponding neuron types or
        instances.
    named_types_for_clustering : str or list, optional, default='all'
        A list of specific neuron types to include in clustering, or 'all' to include all neuron
        types.
    types_to_exclude : list, optional
        A list of neuron types to exclude from clustering.
    fragment_type_dict : dict, optional
        A dictionary mapping fragment names to full neuron type names, used for combining fragment
        types with their corresponding types.
    bids_to_use : list, optional
        A list of specific body IDs to include in the analysis. If not provided, all body IDs are
        used.
    combine_r7_r8 : bool, default=False
        If True, combines specific R7 and R8 neuron subtypes like 'R7y' and 'R7p' into general
        'R7' and 'R8' categories, and excludes certain unclear types.

    Returns
    -------
    connectivity_table : pd.DataFrame
        A DataFrame containing the combined input-output connectivity data. The rows represent
        body IDs, and the columns represent the synaptic input and output types. The table
        includes synaptic weights and fills missing values with 0.
    """
    connection_df_inputs = input_df.copy()
    connection_df_targets = target_df.copy()

    if combine_r7_r8:
        types_to_exclude = [
            cell_type for cell_type in types_to_exclude \
                if not (cell_type in ['R7_unclear', 'R8_unclear'])
        ]
        fragment_type_dict = {
            **fragment_type_dict
          , **{
                'R7_unclear': 'R7'
              , 'R7p': 'R7'
              , 'R7y': 'R7'
              , 'R8_unclear': 'R8'
              , 'R8p': 'R8'
              , 'R8y': 'R8'
            }
        }

    if bids_to_use is not None:
        connection_df_inputs = (connection_df_inputs[
                connection_df_inputs ['bodyId_post'].isin(bids_to_use)])
        connection_df_targets =(connection_df_targets[
                connection_df_targets ['bodyId_pre'].isin(bids_to_use)])

    if named_types_for_clustering == 'all':
        named_types_for_clustering = list(set(list(bid_type.values())))
        if combine_r7_r8:
            named_types_for_clustering = named_types_for_clustering + ['R7', 'R8']

    if types_to_exclude is not None:
        named_types_for_clustering = [
            x for x in named_types_for_clustering if not (x in types_to_exclude)
        ]

    annotate = set_annotations(bid_type)

    if fragment_type_dict is not None:
        rename_fragments = set_annotations(fragment_type_dict)
    else:
        rename_fragments = lambda x:x

    if connection_df_inputs.any is not None:
        connection_df_inputs['type'] = connection_df_inputs['bodyId_pre']\
            .apply(annotate)
        connection_df_inputs['type'] = connection_df_inputs['type']\
            .apply(rename_fragments)

        grouped_df_inputs = connection_df_inputs\
            .groupby(by=['bodyId_post', 'type'], as_index=False)\
            .sum(numeric_only=True)
        grouped_df_inputs = grouped_df_inputs.drop('bodyId_pre',axis=1)

        grouped_df_inputs=grouped_df_inputs[grouped_df_inputs['type']\
            .isin([ x for x in grouped_df_inputs.type if x in named_types_for_clustering])]
        connectivity_table_inputs = grouped_df_inputs\
            .pivot(index='bodyId_post', columns='type', values='weight')\
            .fillna(0)

        connectivity_table_inputs.columns = [x+'-IN' for x in connectivity_table_inputs.columns]

    if connection_df_targets.any is not None:
        connection_df_targets['type'] = connection_df_targets['bodyId_post'].apply(annotate)
        connection_df_targets['type'] = connection_df_targets['type'].apply(rename_fragments)

        grouped_df_targets = connection_df_targets\
            .groupby(by=['bodyId_pre', 'type'], as_index=False)\
            .sum(numeric_only=True)
        grouped_df_targets = grouped_df_targets\
            .drop('bodyId_post',axis=1)

        grouped_df_targets = grouped_df_targets[grouped_df_targets['type']\
            .isin([ x for x in grouped_df_targets.type if x in named_types_for_clustering])]
        connectivity_table_targets = grouped_df_targets\
            .pivot(index='bodyId_pre', columns='type', values='weight')\
            .fillna(0)

        connectivity_table_targets.columns = [
            x + '-OUT' for x in connectivity_table_targets.columns
        ]

    connectivity_table = pd\
        .merge(
            connectivity_table_inputs
          , connectivity_table_targets
          , how="outer"
          , left_index=True
          , right_index=True)\
        .fillna(0)\
        .dropna()\
        .dropna(axis=1)

    return connectivity_table


def get_row_linkage(
    df_for_clustering
  , metric='cosine'
  , linkage_method='ward'
) -> np.ndarray:
    """
    This function computes the hierarchical clustering linkage matrix for the rows of a given
    DataFrame ("df_for_clustering" parameter). It uses a specified distance metric and linkage
    method for clustering.

    Parameters
    ----------
    df_for_clustering : pd.DataFrame
        The DataFrame containing the data for which the row-wise hierarchical clustering will be
        computed.
    metric : str, optional, default='cosine'
        The distance metric to use for computing pairwise distances between rows. Examples include
        'euclidean', 'cosine', etc.
    linkage_method : str, optional, default='ward'
        The linkage algorithm to use for hierarchical clustering. Common methods include 'single',
        'complete', 'average', 'ward', etc.

    Returns
    -------
    row_linkage : ndarray
        The hierarchical clustering linkage matrix for the rows of the DataFrame.
    """
    condensed_dist_mat = pdist(df_for_clustering, metric=metric)
    row_linkage = fastcluster.linkage(condensed_dist_mat, method=linkage_method)

    return row_linkage


def cluster_dict_from_linkage(
    row_linkage
  , df_for_clustering
  , threshold=500
  , criterion='maxclust'
) -> dict:
    """
    This function generates a dictionary that maps cluster numbers to lists of row IDs based on
    hierarchical clustering results. It uses a linkage matrix ("row_linkage") and the rows of the
    DataFrame ("df_for_clustering") to assign clusters.

    Parameters
    ----------
    row_linkage : ndarray
        The hierarchical clustering linkage matrix, obtained from the function `get_row_linkage`.
    df_for_clustering : pd.DataFrame
        The DataFrame containing the data that was clustered. The index of this DataFrame will
        be used as the row IDs.
    threshold : float, default=500
        The threshold for forming flat clusters. This value determines the maximum distance
        between points within the same cluster.
    criterion : str, default='maxclust'
        The criterion to use in forming flat clusters. Common criteria include 'maxclust',
        'distance', etc.

    Returns
    -------
    cluster_dict : dict
        A dictionary where keys are cluster numbers and values are lists of row IDs (from the
        DataFrame) that belong to each cluster.
    """
    cluster_assignments = fcluster(row_linkage, t=threshold, criterion=criterion)
    cluster_numbers = list(set(cluster_assignments))
    cluster_ids = list(zip(
        cluster_assignments
      , list(df_for_clustering.index)
    ))
    cluster_dict = {
        y: [ x[1] for x in cluster_ids  if x[0]==y ] \
            for y in cluster_numbers
    } # value: list IDs

    return cluster_dict


def set_annotations(annotation_dict):
    """
    This function creates and returns an inner function that renames items based on a given
    annotation dictionary ("annotation_dict").

    Parameters
    ----------
    annotation_dict : dict
        A dictionary where the keys are original names and the values are the corresponding new
        names (annotations).

    Returns
    -------
    annotate : function
        A function that takes an input (old_name) and returns the corresponding new name from
        the annotation dictionary if it exists. If the name does not exist in the dictionary,
        it returns the original name as a string.

    Inner Function Parameters
    -------------------------
    old_name : str
        The original name that will be annotated or returned as is if no annotation exists.

    Returns
    -------
    name : str
        The new name (annotation) if available, otherwise the original name as a string.
    """
    def annotate(old_name):
        if old_name in annotation_dict.keys():
            name = annotation_dict[old_name]
        else:
            name = str(old_name)
        return name
    return annotate


def make_count_table(cluster_dict) -> pd.DataFrame:
    """
    This function creates a count table (DataFrame) based on the occurrences of unique names
    within each cluster in the provided cluster dictionary ("cluster_dict"). It converts any
    numeric values in the dictionary to the label "new" and counts the frequency of each unique
    name.

    Parameters
    ----------
    cluster_dict : dict
        A dictionary where the keys are cluster numbers and the values are lists of names or IDs
        associated with each cluster.

    Returns
    -------
    counts_df : pd.DataFrame
        A DataFrame where the columns represent clusters and the rows represent unique names or
        IDs. The cells contain the count of each name/ID within the corresponding cluster. If a
        name does not appear in a cluster, its count is 0.
    """
    cluster_dict_counts = {}
    for cluster in cluster_dict.keys():
        names = ([x if not str(x).isnumeric()
             else "new" for x in cluster_dict[cluster]])
        counter_dict_temp = {cluster:dict(Counter(names))}
        cluster_dict_counts = {**cluster_dict_counts,**counter_dict_temp}
    counts_df = pd.DataFrame(cluster_dict_counts).fillna(0).astype(int)
    return counts_df


def cluster_with_type_names(
    clusters_bids
  , bid_type
) -> dict:
    """
    This function assigns descriptive type names to body IDs in each cluster, based on a provided
    mapping ("bid_type"). It returns a dictionary where the cluster number is the key and the
    value is a list of corresponding type names or body IDs.

    Parameters
    ----------
    clusters_bids : dict
        A dictionary where the keys are cluster numbers, and the values are lists of body IDs
        belonging to each cluster.

    bid_type : dict
        A dictionary mapping body IDs to their respective type names. If a body ID is not found
        in this dictionary, the body ID itself is returned.

    Returns
    -------
    clusters_cell_types : dict
        A dictionary where the keys are cluster numbers, and the values are lists of type names
        or body IDs for each cluster. If a body ID does not have a corresponding type name in the
        "bid_type" dictionary, the body ID is kept in the list.
    """
    clusters_cell_types={}
    for n in range(1, len(clusters_bids)+1):
        clusters_cell_types[n] = ([
            bid_type[bid] \
                if bid in bid_type.keys()\
                else bid \
                    for bid in clusters_bids[n]])
    return clusters_cell_types


def set_pca_for_projections (
    cell_type_pre='L1'
  , cell_type_post='Mi1'
  , neuropile_region='ME(R)'
) -> decomposition.PCA:
    """
    This function performs Principal Component Analysis (PCA) on the synaptic connection
    coordinates between two specified cell types in a given neuropile region. It returns a fitted
    PCA model based on the 3D coordinates of the synapses.

    Parameters
    ----------
    cell_type_pre : str, default='L1'
        The type of the presynaptic cells. Used to define the source criteria for fetching
        synapse data.
    cell_type_post : str, default='Mi1'
        The type of the postsynaptic cells. Used to define the target criteria for fetching
        synapse data.
    neuropile_region : str, default='ME(R)'
        The region of the neuropile where the synapses are located. This is used as part of the
        criteria for selecting synapses.

    Returns
    -------
    pca : sklearn.decomposition.PCA
        A PCA model fitted on the 3D synapse coordinates, which can be used to project synapse
        data into principal components.
    """
    source_criteria = NC(type=cell_type_pre)
    target_criteria = NC(type=cell_type_post)
    synapse_criteria = SC(rois=[neuropile_region], primary_only=True)
    synapses = fetch_synapse_connections(source_criteria, target_criteria, synapse_criteria)

    syn_pos = synapses[['x_pre', 'y_pre', 'z_pre']]
    syn_pos.columns = ['x', 'y', 'z']

    pca = decomposition.PCA(n_components=3)
    pca.fit(syn_pos)
    return pca


def get_combined_synapses_with_stdev(
    criteria
  , pca
  , synapse_criteria=None
  , bids_to_exclude=None
) -> pd.DataFrame:
    """
    This function retrieves synapse data based on the provided criteria, applies PCA
    transformation to 3D coordinates, and returns a DataFrame of synapses with their mean
    coordinates, standard deviations, synapse weights, and z-scores.

    Parameters
    ----------
    criteria : NC object
        The criteria used to fetch synapse data, typically specifying conditions such as cell
        types or body IDs.
    pca : sklearn.decomposition.PCA
        A fitted PCA model used to transform the 3D coordinates of the synapses. If not provided,
        no PCA transformation is applied.
    synapse_criteria : SC object, default=None
        Additional criteria to filter synapses (e.g., region of interest or specific connection
        types).
    bids_to_exclude : list, default=None
        A list of body IDs to be excluded from the resulting combined synapse data.

    Returns
    -------
    combined_synapses : pd.DataFrame
        A DataFrame containing the combined synapse data for each body ID, with columns for the
        mean PCA-transformed coordinates (X, Y, Z), standard deviations (stdX, stdY), the number
        of synapses per body ID (weight), and the z-score of the synapse count for each body ID.
    """
    if bids_to_exclude is None:
        bids_to_exclude = []

    synapses = fetch_synapses(criteria,synapse_criteria)

    com = synapses[['x','y','z']]

    com = pca.transform(com)

    synapses['X']=com[:,0]
    synapses['Y']=com[:,1]
    synapses['Z']=com[:,2]

    combined_synapses = synapses\
        .groupby('bodyId', as_index=False)\
        .mean(numeric_only=True)
    combined_synapses ['stdX'] = synapses\
        .groupby('bodyId', as_index=False)\
        .std(numeric_only=True)['X']
    combined_synapses ['stdY'] = synapses\
        .groupby('bodyId', as_index=False)\
        .std(numeric_only=True)['Y']
    combined_synapses['weight'] = synapses\
        .groupby('bodyId', as_index=False)\
        .count()['x']

    combined_synapses=combined_synapses[~(combined_synapses['bodyId'].isin(bids_to_exclude))]

    combined_synapses_stdev = combined_synapses['weight'].std(numeric_only=True)
    combined_synapses_mean = combined_synapses['weight'].mean(numeric_only=True)
    combined_synapses['z_score'] = (
        (combined_synapses['weight'] - combined_synapses_mean) / combined_synapses_stdev
    )

    return combined_synapses


def generate_clustering_data(
    verbose:bool=True
) -> tuple[dict, list, dict]:
    """
    This function generates the necessary data for clustering neurons based on their body IDs,
    types, and instances. It creates several dictionaries and lists used in clustering, including
    mappings of body IDs to types, types to exclude from clustering, and fragments to their
    corresponding full types.

    Parameters
    ----------
    verbose : bool, default=True
        print some statistics about the cells included and excluded

    Returns
    -------
    bid_type : dict
        A dictionary where keys are body IDs and values are the corresponding cell types or
        instances.

    exclude_from_clustering : list
        A list of neuron types or instances to be excluded from the clustering process. Typically
        includes types with 'unclear' in their names.

    fragment_type_dict : dict
        A dictionary where keys are fragment names, and values are the full cell type names
        associated with the fragments.
    """
    # make a directionary of bodyIds->cell types
    # neuprint only version (i.e. no additional external annotations

    # named optic lobe neurons from neuprint
    criteria = NC(type=NotNull, rois=['ME(R)', 'LO(R)', 'LOP(R)','AME(R)','LA(R)'], roi_req='any')
    named_neurons, _ = fetch_neurons(criteria)

    #  dictionary bodyId:type
    bid_type = dict(zip(list(named_neurons.bodyId.astype(int)),list(named_neurons.type)))

    # list of  named cell types with arbors in both OL (i.e.a left and right instance in the
    # dataset). purpose is to treat L and R instances a different types for clustering purposes
    named_neurons['LR'] = named_neurons.instance
    named_neurons['LR'] = named_neurons['LR']\
        .apply(lambda x: x[-2:])  # assumes instances ending with '_L'or '_R'
    bilateral_count = named_neurons\
        .groupby(
            by=['type','LR']
          , as_index=False)\
        .count()\
        .groupby(by='type')\
        .count()
    bilateral_cell_types = bilateral_count[bilateral_count['LR']==2].index.tolist()


    # dictionary of bodyId:instance for instances of bilateral types
    bilateral_cells_bid_instance = dict(
        zip(
            named_neurons[named_neurons['type'].isin(bilateral_cell_types)].bodyId.tolist()
          , named_neurons[named_neurons['type'].isin(bilateral_cell_types)].instance.tolist()
        )
    )

    # bodies with named instances without a type name
    # (fragments have an instance but not a type name)

    criteria = NC(
        instance=NotNull
      , rois=['ME(R)', 'LO(R)', 'LOP(R)','AME(R)','LA(R)']
      , roi_req='any'
    )
    named_instances, _ = fetch_neurons(criteria)
    named_instances= (
        named_instances[(
            (named_instances['type']!= named_instances['type'])\
          | (named_instances['type']=='')
        )]
    )   #. type either None or empty string

    named_instances.instance= named_instances.instance.apply(remove_hemisphere)

    #   dictionary bodyId:instance for EM bodies without a type
    bid_instance = dict(
        zip(
            list(named_instances.bodyId.astype(int))
          , list(named_instances.instance)
        )
    )

    # combine dictionaries:
    # replace entries for bilateral cells with bilateral_cells_bid_instance
    bid_type = {**bid_type, **bilateral_cells_bid_instance}
    ## add instances
    bid_type = {**bid_instance, **bid_type}  # don't overwrite what is already in dictionary

    # remove None types (should not be needed anymore)
    bid_type = {k: v for k, v in bid_type.items() if v is not None}

    ## types names in 'exclude from clustering' are not used for clustering
    ## in this version all cells with 'unclear' in name are not used for clustering
    exclude_from_clustering = [x for x in list(set(bid_type.values())) if 'unclear' in x]

    # list of all unique names in use
    type_and_instance_names = set(list (bid_type.values()))

    # dictionay of names to be changed when generating connectivity table for clustering
    # here used to identify 'fragments' to be combined with tge corresponding types

    names_with_fragment = [x for x in type_and_instance_names if 'fragment' in x]
    fragment_type_dict = dict(
        zip(
            names_with_fragment
          , [x.split('_fragment')[0] for x in names_with_fragment]
        )
    )
    fragment_type_dict = {bodyId: name for bodyId, name in fragment_type_dict.items()}

    # summary
    names_with_unclear = [x for x in type_and_instance_names if 'unclear' in x]
    names_with_fragment = [x for x in type_and_instance_names if 'fragment' in x]
    if verbose:
        print ('total cells', len(bid_type))
        print ('total instance only names', len(bid_instance))
        print ('total instance types', len(list(set(list (bid_instance.values())))))
        print('all names in use', len(type_and_instance_names))
        print('fragment names', len(names_with_fragment))
        print('"unclear" names', len(names_with_unclear))
        print('bilateral_types', len(bilateral_cell_types))
        print('cell types', (len(type_and_instance_names)-len(names_with_fragment)
                            -len(names_with_unclear)-len(bilateral_cell_types)))

    return bid_type, exclude_from_clustering, fragment_type_dict


def cluster_neurons(
    type_selection:list[str]
  , bid_type:dict
  , exclude_from_clustering:list[str]
  , fragment_type_dict:dict
  , input_df:pd.DataFrame
  , output_df:pd.DataFrame
  , number_of_clusters:int
):
    """
    Plot spatial map and a distribution of #connections for the clusters

    Parameters
    ----------
    type_select : list[str]
        list of cell types that need to be clustered
    bid_type: dict
        dictionary containing the mapping between bodyIds and types.
    exclude_from_clustering: list
        all the cell types with an 'unclear' in its name
    fragment_type_dict: dict
        dictionary that maps the cell type fragment to the cell type
    input_df: pd.dataFrame
        connectivity dataframe that contains all the inputs to all the cell types
    output_df: pd.dataFrame
        connectivity dataframe that contains all the output to all the cell types
    number_of_clusters : int
        expected number of clusters to return

    Returns
    -------
    cells_per_cluster_by_type : pd.DsataFrame
        a table of number of celltypes in the different clusters
    """
    cell_list = [
      bodyId for bodyId in bid_type.keys()\
          if  bid_type[bodyId] in type_selection
    ]

    connection_table = (make_in_and_output_df(
        input_df
      , output_df
      , bid_type
      , types_to_exclude=exclude_from_clustering
      , fragment_type_dict=fragment_type_dict
      , bids_to_use=cell_list
    ))

    row_linkage = get_row_linkage(
        connection_table
      , metric='cosine'
      , linkage_method='ward'
    )

    clusters_bids = cluster_dict_from_linkage(
        row_linkage
      , connection_table
      , threshold=number_of_clusters
      , criterion='maxclust'
    )

    clusters_cell_types = cluster_with_type_names(
        clusters_bids
      , bid_type
    )

    # table with results
    cells_per_cluster_by_type = make_count_table(clusters_cell_types)

    return cells_per_cluster_by_type


def remove_hemisphere(name):
    """
    This function removes any '_R' or '_L' suffix from a given string ("name"), typically used to
    denote the laterality (right or left) of a neuron or body ID.

    Parameters
    ----------
    name : str
        The input string which may contain a '_R' or '_L' suffix indicating the right or left
        instance.

    Returns
    -------
    name : str
        The input string with '_R' or '_L' removed, if present. If neither is found, the original
        string is returned unchanged.
    """
    if '_R' in name:
        name = name.split('_R')[0]
    if '_L' in name:
        name = name.split('_L')[0]
    return name


def find_keys_by_value(
    dictionary
  , target_value
) -> list:
    """
    This function finds and returns a list of keys from a dictionary where the corresponding values
    contain the specified target value ("target_value").

    Parameters
    ----------
    dictionary : dict
        A dictionary where the values are iterable (e.g., lists or sets), and the function will
        search for the target value within these iterables.
    target_value : any
        The value to search for within the dictionary's values. If found, the corresponding key(s)
        will be returned.

    Returns
    -------
    keys : list
        A list of keys from the dictionary for which the target value is found within the
        corresponding values.
    """
    keys = [key for key, values in dictionary.items() if target_value in values]
    return keys


def add_two_colors_to_df(
    df:pd.DataFrame
  , type_names:list
  , cluster_bid:dict
  , colors:list
) -> pd.DataFrame:
    """
    This function adds color, cluster number, and label information to a DataFrame based on body
    IDs and cluster assignments. It updates the DataFrame with two distinct color labels
    corresponding to the clusters.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing a column 'bodyId', which represents the body IDs for each entry.
    type_names : list
        A list containing two type names to be used as labels for the two clusters. The first item
        corresponds to cluster 1, and the second to cluster 2.
    cluster_bid : dict
        A dictionary where the keys represent cluster numbers and the values are lists of body IDs
        belonging to each cluster.
    colors : list
        A list containing two colors (as strings or color codes). The first color corresponds to
        cluster 1, and the second to cluster 2.

    Returns
    -------
    df : pd.DataFrame
        The original DataFrame, now updated with three new columns:
        color : str
            The color associated with the cluster the body ID belongs to.
        c_number : int
            The cluster number (1 or 2) assigned to the body ID.
        label : str
            The type name corresponding to the cluster.
    """
    for i in range(df.shape[0]):
        bid = df['bodyId'][i]
        cluster_number = find_keys_by_value(cluster_bid, bid)
        if cluster_number[0]==1 :
            df.loc[i,'color'] = colors[0]
            df.loc[i,'c_number']  = cluster_number[0]
            df.loc[i,'label']  = type_names[0]
        elif cluster_number[0]==2 :
            df.loc[i,'color']  =  colors[1]
            df.loc[i,'c_number'] = cluster_number[0]
            df.loc[i,'label']  = type_names[1]

    return df


def add_three_colors_to_df(
    df:pd.DataFrame
  , type_names:list
  , cluster_bid:dict
  , colors:list
) -> pd.DataFrame:
    """
    This function adds color, cluster number, and label information to a DataFrame based on body
    IDs and cluster assignments. It updates the DataFrame with three distinct color labels
    corresponding to the clusters.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing a column 'bodyId', which represents the body IDs for each entry.
    type_names : list
        A list containing three type names to be used as labels for the three clusters. The first
        item corresponds to cluster 1, the second to cluster 2, and the third to cluster 3.
    cluster_bid : dict
        A dictionary where the keys represent cluster numbers, and the values are lists of body IDs
        belonging to each cluster.
    colors : list
        A list containing three colors (as strings or color codes). The first color corresponds to
        cluster 1, the second to cluster 2, and the third to cluster 3.

    Returns
    -------
    df : pd.DataFrame
        The original DataFrame, now updated with three new columns:
        - 'color': The color associated with the cluster the body ID belongs to.
        - 'c_number': The cluster number (1, 2, or 3) assigned to the body ID.
        - 'label': The type name corresponding to the cluster.
    """
    for i in range(df.shape[0]):
        bid = df['bodyId'][i]
        cluster_number = find_keys_by_value(cluster_bid, bid)
        if cluster_number[0]==1:
            df.loc[i,'color'] = colors[0]
            df.loc[i,'c_number']  = cluster_number[0]
            df.loc[i,'label']  = type_names[0]
        elif cluster_number[0]==2:
            df.loc[i,'color']  =  colors[1]
            df.loc[i,'c_number'] = cluster_number[0]
            df.loc[i,'label']  = type_names[1]
        elif cluster_number[0]==3:
            df.loc[i,'color']  =  colors[2]
            df.loc[i,'c_number'] = cluster_number[0]
            df.loc[i,'label']  = type_names[2]

    return df


def compare_two_clusters(
    cluster1
  , cluster2
  , cluster_dict
  , df_for_clustering
) -> pd.DataFrame:
    """
    This function compares two clusters by calculating the mean of the weight values for each
    cluster. It returns a DataFrame with the mean values of the specified clusters side by side
    for comparison.

    Parameters
    ----------
    cluster1 : int or str
        The identifier of the first cluster to compare.

    cluster2 : int or str
        The identifier of the second cluster to compare.

    cluster_dict : dict
        A dictionary where the keys are cluster identifiers and the values are lists of body IDs
        or indices belonging to each cluster.

    df_for_clustering : pd.DataFrame
        The DataFrame containing the data used for clustering. The rows correspond to the items
        (body IDs or indices) in the clusters, and the columns represent the features used for
        clustering.

    Returns
    -------
    comparison_df : pd.DataFrame
        A DataFrame with two columns, one for each cluster, showing the mean values of the features
        for both clusters. The rows represent the feature names, and the columns are labeled with
        the identifiers of `cluster1` and `cluster2`.
    """
    comparison_df = (
        pd.concat([
            df_for_clustering.loc[cluster_dict[cluster1],:].mean()
          , df_for_clustering.loc[cluster_dict[cluster2],:].mean()
        ]
      , axis=1
    ))
    comparison_df.columns = [cluster1, cluster2]

    return comparison_df


def compare_three_clusters(
    cluster1
  , cluster2
  , cluster3
  , cluster_dict
  , df_for_clustering
) -> pd.DataFrame:
    """
    This function compares three clusters by calculating the mean of the weight values for each
    cluster. It returns a DataFrame with the mean values of the specified clusters side by side
    for comparison.

    Parameters
    ----------
    cluster1 : int or str
        The identifier of the first cluster to compare.

    cluster2 : int or str
        The identifier of the second cluster to compare.

    cluster3 : int or str
        The identifier of the third cluster to compare.

    cluster_dict : dict
        A dictionary where the keys are cluster identifiers, and the values are lists of body IDs
        or indices belonging to each cluster.

    df_for_clustering : pd.DataFrame
        The DataFrame containing the data used for clustering. The rows correspond to the items
        (body IDs or indices) in the clusters, and the columns represent the features used for
        clustering.

    Returns
    -------
    comparison_df : pd.DataFrame
        A DataFrame with three columns, one for each cluster, showing the mean values of the
        features for all three clusters. The rows represent the feature names, and the columns are
        labeled with the identifiers of `cluster1`, `cluster2`, and `cluster3`.
    """
    comparison_df = (
        pd.concat([
            df_for_clustering.loc[cluster_dict[cluster1],:].mean()
          , df_for_clustering.loc[cluster_dict[cluster2],:].mean()
          , df_for_clustering.loc[cluster_dict[cluster3],:].mean()
        ]
      , axis=1
    ))
    comparison_df.columns = [cluster1, cluster2, cluster3]

    return comparison_df
