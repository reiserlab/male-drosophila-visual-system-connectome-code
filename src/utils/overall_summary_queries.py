from pathlib import Path
from dotenv import find_dotenv

import numpy as np
import pandas as pd

from neuprint import \
    fetch_neurons, merge_neuron_properties, fetch_adjacencies, NeuronCriteria as NC

from utils.ol_types import OLTypes
from utils.ol_color import OL_COLOR


def add_color_group(
    df:pd.DataFrame
) -> pd.DataFrame:
    """
    This function adds color based on cell type groups from ol_color

    Parameters
    ----------
    df : pd.DataFrame
         dataframe to which color and/or group needs to be added as a column

    Returns
    -------
    df : pd.DataFrame
        dataframe with color and/or group added
    """
    colors = OL_COLOR.OL_TYPES.hex

    types = OLTypes()
    gp1 = types.get_neuron_list(primary_classification='OL_intrinsic')['type'].to_list()
    gp2 = types.get_neuron_list(primary_classification='OL_connecting')['type'].to_list()
    gp3 = types.get_neuron_list(primary_classification='VPN')['type'].to_list()
    gp4 = types.get_neuron_list(primary_classification='VCN')['type'].to_list()
    gp5 = types.get_neuron_list(primary_classification='other')['type'].to_list()

    for index, row in df.iterrows():
        type_n = row['type']

        if type_n in gp1:
            df.loc[index, 'group'] = 1
            df.loc[index, 'color'] = colors[0]
            df.loc[index, 'main_group'] = 'OL_intrinsic'
        elif type_n in gp2:
            df.loc[index, 'group'] = 2
            df.loc[index, 'color'] = colors[1]
            df.loc[index, 'main_group'] = 'OL_connecting'
        elif type_n in gp3:
            df.loc[index, 'group'] = 3
            df.loc[index, 'color'] = colors[2]
            df.loc[index, 'main_group'] = 'VPN'
        elif type_n in gp4:
            df.loc[index, 'group'] = 4
            df.loc[index, 'color'] = colors[3]
            df.loc[index, 'main_group'] = 'VCN'
        elif type_n in gp5:
            df.loc[index, 'group'] = 5
            df.loc[index, 'color'] = colors[4]
            df.loc[index, 'main_group'] = 'other'

    df['color'].astype(dtype='object')
    df['group'].astype(dtype='object')

    return df


def make_ncell_nconn_nsyn_data(
    cell_instances:pd.Series
) -> pd.DataFrame:
    """
    This function generates a dataframe with the number of cells, "pre", "post",
    "upstream" and "downstream" connections for every cell type

    Parameters
    ----------
    cell_instances : pd.Series
        series containing a list of cell instances for which the query has to be made

    Returns
    -------
    df : pd.DataFrame
        dataframe with #cells, #pre synapses and #post synapses for all the cell types
    """
    cache_dir = Path(find_dotenv()).parent / "cache" / "summary_plots"
    cache_fn = cache_dir / "ncell_nconn_nsyn_data.pickle"

    if cache_fn.is_file():
        ncell_nconn_nsyn_df = pd.read_pickle(cache_fn)
    else:
        # ncells vs cell-type
        neurons_df, _ = fetch_neurons(NC(instance=cell_instances))
        ncell_nconn_nsyn_df = neurons_df\
            .groupby('type')[['bodyId', 'upstream', 'downstream', 'pre', 'post']]\
            .agg({
                'bodyId': 'size'
              , 'upstream': 'sum', 'downstream': 'sum'
              , 'pre': 'sum', 'post': 'sum'
            })\
            .rename(columns={'bodyId': 'n_cells', 'pre': 'n_pre_syn', 'post': 'n_post_syn'})\
            .assign(
                updown = lambda row: row.upstream + row.downstream
              , n_syn = lambda row: row.n_pre_syn + row.n_post_syn
            )
        cache_fn.parent.mkdir(parents=True, exist_ok=True)
        ncell_nconn_nsyn_df.to_pickle(cache_fn)

    return ncell_nconn_nsyn_df


def make_ncell_nconn_data(
    cell_instances:pd.Series
    ) -> pd.DataFrame:
    """
    This function generates a dataframe with
    (a) mean number of connected cells for every cell type
    (b) mean number of connected cell types for every cell type
    (c) number of connected input and output cells for every cell type.

    Parameters
    ----------
    cell_instances : pd.Series
        series containing a list of cell instances for which the query has to be made

    Returns
    -------
    summary_df : pd.DataFrame
        dataframe with the following information for all the cell types
        (a) number of connected cells for all cell types
        (b) number of connected cell types for all cell types
        (c) number of connected input cells and number of connected output cells for all cell types
    """
    cache_dir = Path(find_dotenv()).parent / "cache" / "summary_plots"
    cache_fn = cache_dir / "summary_data.pickle"

    if cache_fn.is_file():
        summary_df = pd.read_pickle(cache_fn)
    else:
        # number of cells per cell type
        neurons_df, _ = fetch_neurons(NC(instance=cell_instances))

        ncells_df = neurons_df\
            .groupby('type')['bodyId']\
            .nunique()\
            .reset_index(name='n_cells')
        ncells_sorted_df = ncells_df\
            .sort_values(by='n_cells', ascending=False)
        ncells_sorted_df.columns = ['type', 'n_cells']
        ncells_sorted_df = ncells_sorted_df.reset_index(drop=True)

        # fetching connections
        conn_ol_df = get_conn_ol_df()

        # prepare the dataframe
        conn_ol_df = conn_ol_df\
            .groupby(['bodyId_pre','bodyId_post','type_pre','type_post'])['weight']\
            .sum()\
            .to_frame()\
            .reset_index()

        # prepare for the nconn cells and nconn types analysis
        conn_ol_df_reversed = conn_ol_df.copy()
        conn_ol_df_reversed.columns = [
            'bodyId_post'
          , 'bodyId_pre'
          , 'type_post'
          , 'type_pre'
          , 'weight'
        ]
        concat_df = pd.concat([conn_ol_df, conn_ol_df_reversed])
        concat_df = concat_df\
            .groupby(['bodyId_pre','bodyId_post','type_pre','type_post'])['weight']\
            .sum()\
            .to_frame()\
            .reset_index()

        concat_df = concat_df[concat_df['weight']>1]\
            .reset_index()

        # nconn cells vs cell-type (mean)
        post_conn_cells_df = concat_df\
            .groupby(['type_pre','bodyId_pre'])['bodyId_post']\
            .unique()\
            .reset_index(name='n_post_conn_cells')
        post_conn_cells_df.columns = ['type', 'bodyId', 'n_post_conn_cells']
        pre_conn_cells_df = concat_df\
            .groupby(['type_post', 'bodyId_post'])['bodyId_pre']\
            .unique()\
            .reset_index(name='n_pre_conn_cells')
        pre_conn_cells_df.columns = ['type','bodyId','n_pre_conn_cells']

        n_totalmean_conn_cells_df = pd.merge(post_conn_cells_df, pre_conn_cells_df, how='outer')
        n_totalmean_conn_cells_df = n_totalmean_conn_cells_df.replace(np.nan, None)
        columns_to_concat_cells = ['n_post_conn_cells','n_pre_conn_cells']
        n_totalmean_conn_cells_df['n_total_conn_cells'] = n_totalmean_conn_cells_df\
            .apply(lambda row: np.concatenate(
                [row[col] for col in columns_to_concat_cells if row[col] is not None]
                )
            , axis=1)
        n_totalmean_conn_cells_df['n_total_conn_cells'] = \
            n_totalmean_conn_cells_df['n_total_conn_cells']\
            .apply(lambda x: [val for val in x if not pd.isna(val)])
        n_totalmean_conn_cells_df['UniqueCount'] = n_totalmean_conn_cells_df['n_total_conn_cells']\
            .apply(lambda x: len(set(x)))
        n_mean_conn_cells_df = n_totalmean_conn_cells_df\
            .groupby('type')['UniqueCount']\
            .mean()\
            .reset_index(name='n_mean_conn_cells')

        ncells_n_mean_conn_cells_df = ncells_sorted_df.merge(n_mean_conn_cells_df)

        # nconn types vs cell-type (mean)
        n_post_conn_types_df = concat_df\
            .groupby(['type_pre','bodyId_pre'])['type_post']\
            .unique()\
            .reset_index(name='n_post_conn_types')
        n_post_conn_types_df.columns = ['type','bodyId','n_post_conn_types']
        n_pre_conn_types_df = concat_df\
            .groupby(['type_post','bodyId_post'])['type_pre']\
            .unique()\
            .reset_index(name='n_pre_conn_types')
        n_pre_conn_types_df.columns = ['type','bodyId','n_pre_conn_types']

        n_totalmean_conn_types_df = pd.merge(
            n_post_conn_types_df
          , n_pre_conn_types_df
          , how='outer'
        )
        n_totalmean_conn_types_df = n_totalmean_conn_types_df.replace(np.nan, None)
        columns_to_concat_types = ['n_post_conn_types','n_pre_conn_types']
        n_totalmean_conn_types_df['n_total_conn_types'] = n_totalmean_conn_types_df\
            .apply(lambda row: np.concatenate(
                [row[col] for col in columns_to_concat_types if row[col] is not None]
                )
            , axis=1)
        n_totalmean_conn_types_df['n_total_conn_types'] = \
            n_totalmean_conn_types_df['n_total_conn_types']\
            .apply(lambda x: [val for val in x if not pd.isna(val)])
        n_totalmean_conn_types_df['UniqueCount'] = n_totalmean_conn_types_df['n_total_conn_types']\
            .apply(lambda x: len(set(x)))
        n_mean_conn_types_df = n_totalmean_conn_types_df\
            .groupby('type')['UniqueCount']\
            .mean()\
            .reset_index(name='n_mean_conn_types')

        med_nconncells_nconntypes_mean_df = ncells_n_mean_conn_cells_df.merge(n_mean_conn_types_df)

        # nconn input cells vs nconn output cells (mean)
        conn_ol_df_io_temp = conn_ol_df[conn_ol_df['weight']>1]
        n_post_conn_cells_df = conn_ol_df_io_temp\
            .groupby(['type_pre', 'bodyId_pre'])['bodyId_post']\
            .nunique()\
            .reset_index(name='n_post_conn_cells')
        n_post_conn_cells_df.columns = ['type', 'bodyId', 'n_post_conn_cells']
        n_pre_conn_cells_df = conn_ol_df_io_temp\
            .groupby(['type_post','bodyId_post'])['bodyId_pre']\
            .nunique()\
            .reset_index(name='n_pre_conn_cells')
        n_pre_conn_cells_df.columns = ['type', 'bodyId', 'n_pre_conn_cells']

        n_total_conn_cells_df = pd.merge(n_post_conn_cells_df, n_pre_conn_cells_df, how='outer')
        n_post_mean_conn_cells_df = n_total_conn_cells_df\
            .groupby('type')['n_post_conn_cells']\
            .mean()\
            .reset_index(name='n_post_mean_conn_cells')
        n_pre_mean_conn_cells_df = n_total_conn_cells_df\
            .groupby('type')['n_pre_conn_cells']\
            .mean()\
            .reset_index(name='n_pre_mean_conn_cells')
        n_all_mean_conn_cells_df = n_post_mean_conn_cells_df.merge(n_pre_mean_conn_cells_df)

        summary_df = med_nconncells_nconntypes_mean_df.merge(n_all_mean_conn_cells_df)

        cache_fn.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_pickle(cache_fn)
    return summary_df


def make_connectivity_sufficiency_data(
    n_top_connections:int
    ) -> pd.DataFrame:
    """
    This function computes percentage of unique combinations of connections as a function of
    number of top connections considered

    Parameters
    ----------
    n_top_connections : int
        number of top connections to consider to compute the percentage of unique
        connectivity combinations

    Returns
    -------
    df : pd.DataFrame
        dataframe with fraction of unique combinations of connections for different numbers of
        top connections considered
    """
    cache_dir = Path(find_dotenv()).parent / "cache" / "summary_plots"
    cache_fn = cache_dir / "unique_combinations_data.pickle"

    if cache_fn.is_file():
        unique_combinations_df = pd.read_pickle(cache_fn)
    else:
        # fetching connections
        conn_ol_df = get_conn_ol_df()

        # Get the top connecting cell-type for each cell-type
        df_pre = conn_ol_df\
            .groupby(['type_pre', 'type_post'])\
            .agg({'weight': 'sum'})\
            .reset_index()\
            .rename(columns={
                'type_pre': 'type'
              , 'type_post': 'connecting_type'
            })

        df_post = conn_ol_df\
            .groupby(['type_post', 'type_pre'])\
            .agg({'weight': 'sum'})\
            .reset_index()\
            .rename(columns={
                'type_post': 'type'
              , 'type_pre': 'connecting_type'
            })

        df = pd.concat([df_pre, df_post])

        # getting the top n connections for each cell-type
        frac_unique_combinations_pre = []
        frac_unique_combinations_post = []
        frac_unique_combinations = []
        n_conn = np.arange(1, n_top_connections + 1)

        for i1 in n_conn:
            frac_unique_combinations_pre.append(calc_unique_connections(df_pre, i1))
            frac_unique_combinations_post.append(calc_unique_connections(df_post, i1))
            frac_unique_combinations.append(calc_unique_connections(df, i1))

        # preparing dataframes
        data = {
            'n_connections': n_conn
          , 'frac_unique_combinations_pre': frac_unique_combinations_pre
          , 'frac_unique_combinations_post': frac_unique_combinations_post
          , 'frac_unique_combinations': frac_unique_combinations
        }
        unique_combinations_df = pd.DataFrame(data)
        cache_fn.parent.mkdir(parents=True, exist_ok=True)
        unique_combinations_df.to_pickle(cache_fn)

    return unique_combinations_df


def calc_unique_connections(
    df:pd.DataFrame
  , length:int
    ) -> float:
    """
    Calculate the unique connections.

    Parameters
    ----------
    df : pd.DataFrame
        type : str
            source cell type
        connecting_type : str
            target cell type
        weight : int
            connecting strength
    length : int
        number of connectings to consider

    Returns
    -------
    perc : float
        percentage of how many of the top

    """
    assert 0 < length <= 10, f"only tested for top 10, not for top {length}"
    df_tmp = df\
        .sort_values(['type', 'weight'], ascending=False)\
        .groupby('type')\
        .head(length)\
        .groupby('type')['connecting_type']\
        .apply(','.join)\
        .reset_index()
    return df_tmp['connecting_type'].nunique() / df_tmp.shape[0]


def get_conn_ol_df() -> pd.DataFrame:
    """
    Retrieve the connection dataframe. If a cache exists, read it from the cache.

    Returns
    -------
    conn_ol_df : pd.DataFrame
        dataframe with connectivity information from and to all instances within "cell_instances"
    """
    cache_dir =  Path(find_dotenv()).parent / "cache" / "summary_plots"
    cache_fn = cache_dir / "conn_ol_df.pickle"

    olt = OLTypes()
    types = olt.get_neuron_list(side='both')
    cell_instances = types['instance']

    if cache_fn.is_file():
        conn_ol_df = pd.read_pickle(cache_fn)
    else:
        neurons_ol_df, connections_ol_df = fetch_adjacencies(
            NC(instance=cell_instances)
          , NC(instance=cell_instances)
        )
        conn_ol_df = merge_neuron_properties(neurons_ol_df, connections_ol_df, 'type')
        cache_fn.parent.mkdir(parents=True, exist_ok=True)
        conn_ol_df.to_pickle(cache_fn)

    return conn_ol_df
