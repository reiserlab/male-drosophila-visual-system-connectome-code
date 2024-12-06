import csv

from pathlib import Path
from dotenv import find_dotenv
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from sklearn.metrics.cluster import homogeneity_score, completeness_score

from neuprint import NeuronCriteria as NC, SynapseCriteria as SC, fetch_neurons, fetch_synapses
from utils.ROI_calculus import \
    load_depth_bins, find_depth, find_per_columnbin_spanned_no_cols


def find_clustering_scores(
    n_clusters:int
) -> None:
    """
    Homogeneity and completeness of morphology clustering scores.

    Parameters
    ----------
    n_clusters : int
        number of clusters

    Returns
    -------
    hom : float
        homogeneity score
    com : float
        completeness score
    """
    #load morphology data
    data_path = Path(find_dotenv()).parent / 'cache' / 'morpho_clustering'
    data_path.mkdir(parents=True, exist_ok=True)

    types = np.squeeze(
        pd.read_csv(data_path / 'morpho_types.csv', header=None).values
    )
    cluster_ids = np.squeeze(
        pd.read_csv(data_path / f'morpho_clusters_{n_clusters}clu.csv', header=None).values
    )

    #convert cell types to ints
    type_names = np.unique(types)
    n_types = len(type_names)
    type_ints = range(n_types)
    types_to_ints = {type_names[i]: type_ints[i] for i in range(len(type_names))}
    ints = np.array([types_to_ints[item] for item in types])

    hom = homogeneity_score(ints, cluster_ids-1)
    com = completeness_score(ints, cluster_ids-1)

    return hom, com


def create_morpho_confusion_matrix(
    n_clusters:int
) -> None:
    """
    Make confusion matrix corresponding to morphology clustering with n_clusters clusters.
    Save as dataframe in 'morpho_confusion_mat_'+str(n_clusters)+'clu.csv' in the results folder
    morpho_clustering.

    Parameters
    ----------
    n_clusters : int
        number of clusters

    Returns
    -------
    None
    """

    #load morphology data
    data_path = Path(find_dotenv()).parent / 'cache' / 'morpho_clustering'
    data_path.mkdir(parents=True, exist_ok=True)

    types = np.squeeze(pd.read_csv( data_path / 'morpho_types.csv', header=None).values)
    cluster_ids = np.squeeze(
        pd.read_csv(data_path / f'morpho_clusters_{n_clusters}clu.csv', header=None).values
    )

    #convert cell types to ints
    type_names = np.unique(types)
    n_types = len(type_names)
    type_ints = range(n_types)
    types_to_ints = {type_names[i]: type_ints[i] for i in range(len(type_names))}
    ints = np.array([types_to_ints[item] for item in types])

    #create confusion matrix
    hist, _, _ = np.histogram2d(
        ints
      , cluster_ids
      , bins=[
            np.linspace(-.5, n_types - .5, n_types + 1)
          , np.linspace(.5, n_clusters + .5, n_clusters + 1)
        ]
    )

    #store as dataframe
    confusion_df = pd.DataFrame(
        hist.astype(int)
      , index=type_names
      , columns=np.array(range(n_clusters)) + 1
    )
    confusion_df.to_csv( data_path / f'morpho_confusion_mat_{n_clusters}clu.csv' )


def cluster_morpho_data(
    n_clusters:int
) -> None:
    """
    Cluster morphology clustering data into n_clusters clusters.
    Assigned cluster identities are saved as 'morpho_clusters_'+str(n_clusters)+'clu.csv' in the
    cache folder morpho_clustering. Rows in that file correspond to the rows in the clustering
    data.

    Parameters
    ----------
    n_clusters : int
        number of clusters

    Returns
    -------
    None
    """
    # load morphology data
    data_path = Path(find_dotenv()).parent / 'cache' / 'morpho_clustering'
    data_path.mkdir(parents=True, exist_ok=True)

    features0 = pd.read_csv(data_path / 'morpho_features.csv', header=None).values

    # number of depth bins in feature vector
    n_bins = int(features0.shape[1] / 4)
    # normalize synapse to size distributions
    features = features0.copy()
    ind_pre = np.where( features0[:,:n_bins].sum(1) )[0]
    features[ind_pre,:n_bins] = (
            (features0[:, n_bins:2*n_bins]).mean(1) / features0[ind_pre, :n_bins].mean(1)
        )[:, np.newaxis] * features0[ind_pre, :n_bins]
    ind_post = np.where( features0[:,2*n_bins:3*n_bins].sum(1) )[0]
    features[ind_post,2*n_bins:3*n_bins] = (
            features0[:, 3 * n_bins:4 * n_bins].mean(1) \
          / features0[ind_post, 2 * n_bins:3 * n_bins].mean(1)
        )[:, np.newaxis] * features0[ind_post, 2 * n_bins:3 * n_bins]

    # hierarchical clustering
    hierarch_cluster = hierarchy.linkage(features, method="ward", metric='euclidean')

    # flat cut of dendogram at fixed number of clusters
    cluster_ids = hierarchy.fcluster(hierarch_cluster, t=n_clusters, criterion='maxclust')

    # save cluster assignment
    np.savetxt( data_path / f'morpho_clusters_{n_clusters}clu.csv', cluster_ids)


def create_morpho_data(
    types:list[str]
  , rois:list[str]
  , min_syn=5
  , samp=2
) -> None:
    """
    Create morphology clustering data as 3 csv files (all start with 'morpho_', in cache folder
    morpho_clustering), each with equal numbers of rows.
    The rows in these files correspond to:
    1) bodyIds
    2) cell type
    3) feature vector

    Parameters
    ----------
    types : list[str]
        specifies the cell types from which to fetch all neurons
    rois : list[str]
        specifies rois with synapses; can only be a list including 'ME(R)', 'LO(R)', 'LOP(R)'
    min_syn : int, default=5
        minimum number of synapses of a neuron in one ROI to be included
    samp : int, default=2
        sub-sampling factor for depth bins

    Returns
    -------
    None
    """
    assert np.isin(rois,['ME(R)','LO(R)','LOP(R)']).all(), \
        f"ROI list can only contain 'ME(R)', 'LO(R)', 'LOP(R)', but is actually '{rois}'"

    data_path = Path(find_dotenv()).parent / 'cache' / 'morpho_clustering'
    data_path.mkdir(parents=True, exist_ok=True)
    out_files = {
        'bid':  data_path / 'morpho_bodyIds.csv'
      , 'typ':  data_path / 'morpho_types.csv'
      , 'feat': data_path / 'morpho_features.csv'
    }

    for out_fn in  out_files.values():
        out_fn.unlink(missing_ok=True)

    #find total number of depth bins across all rois
    n_tot_pins_samp = 0
    for roi_str in rois:
        bin_edges, bin_centers = load_depth_bins(roi_str=roi_str, samp=samp)
        n_tot_pins_samp += bin_centers.shape[0]

    #store data one cell type at a time
    for target_type in types:
        neuron_df, _ = fetch_neurons(
            NC(
                type=target_type
              , inputRois=rois
              , outputRois=rois
              , min_roi_inputs=min_syn
              , min_roi_outputs=min_syn
              , roi_req ='any'
            )
        )
        nid_all = neuron_df['bodyId'].unique()
        n_neurons_all = nid_all.shape[0]

        #initialize dataframes
        syn_dist_pre_df = pd.DataFrame(np.zeros((n_neurons_all, n_tot_pins_samp)), index=nid_all)
        size_dist_pre_df = pd.DataFrame(np.zeros((n_neurons_all, n_tot_pins_samp)), index=nid_all)
        syn_dist_post_df = pd.DataFrame(np.zeros((n_neurons_all, n_tot_pins_samp)), index=nid_all)
        size_dist_post_df = pd.DataFrame(np.zeros((n_neurons_all, n_tot_pins_samp)), index=nid_all)
        ct_tot_pins_samp = 0
        for roi_str in rois:

            syn_df = fetch_synapses(NC(type=target_type), SC(rois=roi_str, confidence=.9))
            bin_edges, bin_centers = load_depth_bins(roi_str=roi_str, samp=samp)
            n_pins_samp = bin_centers.shape[0]

            if syn_df.shape[0]<min_syn:
                ct_tot_pins_samp += n_pins_samp
                continue

            nid = syn_df['bodyId'].unique()
            n_neurons = nid.shape[0]

            #find depth bins
            depth_df = find_depth(syn_df, roi_str=roi_str, samp=samp)
            syn_df['depth'] = depth_df['depth'].values

            #split pre and post synapses
            syn_df_pre = syn_df[ syn_df['type']=='pre' ]
            syn_df_post = syn_df[ syn_df['type']=='post' ]

            #find synapse distributions per depth
            syn_dist_pre = np.zeros((n_neurons, n_pins_samp))
            syn_dist_post = np.zeros((n_neurons, n_pins_samp))
            for n in range(n_neurons):
                tmp_pre = syn_df_pre[ syn_df_pre['bodyId']==nid[n] ]
                if tmp_pre.shape[0]>0:
                    count0_pre, _ = np.histogram( tmp_pre['depth'].values, bins=bin_edges )
                    syn_dist_pre[n] = count0_pre
                tmp_post = syn_df_post[ syn_df_post['bodyId']==nid[n] ]
                if tmp_post.shape[0]>0:
                    count0_post, _ = np.histogram( tmp_post['depth'].values, bins=bin_edges )
                    syn_dist_post[n] = count0_post

            #find size distributions per depth
            if syn_df_pre.shape[0]<min_syn*n_neurons:
                size_dist_pre = np.zeros((n_neurons, n_pins_samp))
            else:
                size_df_pre, _, _ = find_per_columnbin_spanned_no_cols(
                    syn_df_pre
                  , roi_str=roi_str
                  , samp=samp
                )
                size_dist_pre = np.zeros((n_neurons, n_pins_samp))
                for n in range(n_neurons):
                    tmp_pre = size_df_pre[ size_df_pre['bodyId']==nid[n] ]
                    if tmp_pre.shape[0]>0:
                        for i in range(n_pins_samp):
                            if (tmp_pre['bin']==i).sum()>0:
                                size_dist_pre[n,i] = tmp_pre[tmp_pre['bin']==i]['size'].values[0]
            if syn_df_post.shape[0]<min_syn*n_neurons:
                size_dist_post = np.zeros((n_neurons, n_pins_samp))
            else:
                size_df_post, _, _ = find_per_columnbin_spanned_no_cols(
                    syn_df_post
                  , roi_str=roi_str
                  , samp=samp
                )
                size_dist_post = np.zeros((n_neurons, n_pins_samp))
                for n in range(n_neurons):
                    tmp_post = size_df_post[ size_df_post['bodyId']==nid[n] ]
                    if tmp_post.shape[0]>0:
                        for i in range(n_pins_samp):
                            if (tmp_post['bin']==i).sum()>0:
                                size_dist_post[n,i] = tmp_post[tmp_post['bin']==i]['size']\
                                    .values[0]

            #store in dataframe
            syn_dist_pre_df.loc[
                nid, ct_tot_pins_samp:(ct_tot_pins_samp + n_pins_samp - 1)] = syn_dist_pre
            syn_dist_post_df.loc[
                nid, ct_tot_pins_samp:(ct_tot_pins_samp + n_pins_samp - 1)] = syn_dist_post
            size_dist_pre_df.loc[
                nid, ct_tot_pins_samp:(ct_tot_pins_samp + n_pins_samp - 1)] = size_dist_pre
            size_dist_post_df.loc[
                nid, ct_tot_pins_samp:(ct_tot_pins_samp + n_pins_samp - 1)] = size_dist_post
            ct_tot_pins_samp += n_pins_samp

        # remove bodyIds that are full of 0s (for all depth bins of all rois)
        ind_nonzero = (~(syn_dist_pre_df==0).all(axis=1)) & (~(syn_dist_post_df==0).all(axis=1))
        syn_dist_pre_df = syn_dist_pre_df.loc[ind_nonzero]
        syn_dist_post_df = syn_dist_post_df.loc[ind_nonzero]
        size_dist_pre_df = size_dist_pre_df.loc[ind_nonzero]
        size_dist_post_df = size_dist_post_df.loc[ind_nonzero]

        if ind_nonzero.shape[0]==0:
            continue

        #store bodyId
        with open(out_files['bid'], mode='a', newline='', encoding='utf-8') as my_csv:
            csv_writer=csv.writer(my_csv)
            csv_writer.writerows(map(lambda x: [x], syn_dist_pre_df.index.values))
        #store cell type name
        with open(out_files['typ'], mode='a', newline='', encoding='utf-8') as my_csv2:
            csv_writer2=csv.writer(my_csv2)
            csv_writer2.writerows(map(lambda x: [x], [target_type] * nid_all.shape[0]))
        #store feature vector
        syn_features = np.concatenate(
            (
                syn_dist_pre_df.values
              , size_dist_pre_df.values
              , syn_dist_post_df.values
              , size_dist_post_df.values
            )
          , axis=1)
        with open(out_files['feat'], mode='a', newline='', encoding='utf-8') as my_csv3:
            csv_writer3 = csv.writer(my_csv3)
            csv_writer3.writerows(syn_features)
