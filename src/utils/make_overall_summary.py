from pathlib import Path
import pandas as pd

from dotenv import find_dotenv

from neuprint import fetch_custom

def get_neuropil_df(
    df:pd.DataFrame
  , threshold:float=0.02
):
    """
    Generate the neuropil table based on brain region.

    Parameters
    ----------
    df : pd.DataFrame
        data with all the cell instances and their corresponding cell type groups, color, etc.
    threshold : float
        threshold fraction of synapses/cells that needs to be crossed within a brain region
        to be assigned to that brain region

    Returns
    -------
    neuropil_df : pd.DataFrame with –
        n_celltypes : int
            number of cell instances whose population level synapses surpass the `threshold`
        n_cells : int
            number of cells who have more than `threshold` of their synapses in the brain region
        n_synapses : int
            number of synapses per brain region
        number of cell types, number of cells, and number of synapsesdata grouped by brain
            region – number of cell types, cells and synapses grouped by brain region (LA(R),
            ME(R), LO(R), LOP(R), AME(R))

    """
    assert 0<= threshold <= 1, f"Threshold must be between 0 and 1, not {threshold}"

    cql_synapses=f"""
        UNWIND ['LA(R)', 'ME(R)', 'LO(R)', 'LOP(R)', 'AME(R)'] as roi
        MATCH (n:Neuron)
        WHERE n.instance in {df['instance'].to_list()}
        WITH apoc.convert.fromJsonMap(n.roiInfo) as nri, roi
        WITH
            coalesce(nri[roi].upstream, 0) as syn_upstream
          , coalesce(nri[roi].downstream, 0) as syn_downstream
          , roi
        RETURN
            distinct roi
          , sum(syn_upstream) as n_upstream
          , sum(syn_downstream) as n_downstream
    """
    synapse_df = fetch_custom(cql_synapses).set_index('roi')

    cql_cells=f"""
        UNWIND ['LA(R)', 'ME(R)', 'LO(R)', 'LOP(R)', 'AME(R)'] as roi
        MATCH (n:Neuron)
        WHERE n.instance in {df['instance'].to_list()} and n[roi]
        WITH
            distinct n
          , apoc.convert.fromJsonMap(n.roiInfo) as nri
          , coalesce(n.pre, 0) + coalesce(n.post,0) as syn_total
          , roi
          , n.bodyId as bid
        with
            coalesce(nri[roi].pre, 0) as syn_pre
          , coalesce(nri[roi].post, 0) as syn_post
          , syn_total
          , roi
          , bid
        WITH
            CASE
                WHEN syn_total> 0 THEN toFloat(syn_pre + syn_post) / syn_total
                ELSE 0
            END AS syn_frac
          , roi
          , bid

        WHERE syn_frac >= {threshold}
        RETURN distinct roi, count(bid) as n_cells

    """
    cell_df = fetch_custom(cql_cells).set_index('roi')

    cql_cell_types=f"""
        UNWIND ['LA(R)', 'ME(R)', 'LO(R)', 'LOP(R)', 'AME(R)'] as roi
        MATCH (n:Neuron)
        WHERE n.instance in {df['instance'].to_list()} and n[roi]
        WITH
            distinct n
          , apoc.convert.fromJsonMap(n.roiInfo) as nri
          , coalesce(n.pre, 0) + coalesce(n.post,0) as syn_total
          , roi
        with
            coalesce(nri[roi].pre, 0) as syn_pre
          , coalesce(nri[roi].post, 0) as syn_post
          , syn_total
          , roi
          , n
        WITH
            distinct n.type as type
          , roi
          , sum(syn_pre) as syn_pre
          , sum(syn_post) as syn_post
          , sum(syn_total) as syn_total

        WITH
            CASE
                WHEN syn_total> 0 THEN toFloat(syn_pre + syn_post)/ syn_total
                ELSE 0
            END AS syn_frac
          , roi
          , type
        WHERE syn_frac >= {threshold}
        RETURN
            distinct roi
          , count(type) as n_celltypes
    """

    celltype_df = fetch_custom(cql_cell_types).set_index('roi')

    df_tmp = cell_df\
        .join(synapse_df, how='right')

    neuropil_df = celltype_df\
        .join(df_tmp, how='right')\
        .fillna(0)\
        .astype(int)

    return neuropil_df


def get_neuropil_groups_celltypes_df(
    df:pd.DataFrame
  , threshold:float=0.02
):
    """
    Generate the neuropil table for main groups aggregated at cell instance level.

    Parameters
    ----------
    df : pd.DataFrame
        data with all the cell instances and their corresponding cell type groups, color, etc.
    threshold : float, default=0.02
        threshold fraction of synapses/cells that needs to be crossed within a brain region
        to be assigned to that brain region

    Returns
    -------
    ct_df : pd.DataFrame
        The DataFrame shows the main groups from `df` as columns, along with the count of
        cell instances having synapses in the visual system brain region. On the instance
        population level more than `threshold` synapses must be within the region to count.
    """
    ct_df = pd.DataFrame()
    for grp_name, grp in df.groupby(by='main_groups'):
        cql_cell_types=f"""
            UNWIND ['LA(R)', 'ME(R)', 'LO(R)', 'LOP(R)', 'AME(R)'] as roi
            MATCH (n:Neuron)
            WHERE n.instance in {grp['instance'].to_list()} and n[roi]
            WITH
                distinct n
              , apoc.convert.fromJsonMap(n.roiInfo) as nri
              , coalesce(n.pre, 0) + coalesce(n.post,0) as syn_total
              , roi
            with
                coalesce(nri[roi].pre, 0) as syn_pre
              , coalesce(nri[roi].post, 0) as syn_post
              , syn_total
              , roi
              , n
            WITH
                distinct n.type as type
              , roi
              , sum(syn_pre) as syn_pre
              , sum(syn_post) as syn_post
              , sum(syn_total) as syn_total

            WITH
              CASE
                    WHEN syn_total> 0 THEN toFloat(syn_pre + syn_post)/ syn_total
                    ELSE 0
                END AS syn_frac
              , roi
              , type
            WHERE syn_frac >= {threshold}
            RETURN
                distinct roi
              , count(type) as n_celltypes
        """
        celltype_df = fetch_custom(cql_cell_types)\
            .set_index('roi')\
            .rename(columns={'n_celltypes': grp_name})
        ct_df = pd.concat([ct_df, celltype_df], axis=1)

    ct_df = ct_df.fillna(0).astype(int)

    return ct_df



def get_neuropil_groups_cells_df(
    df:pd.DataFrame
  , threshold:float=0.02
):
    """
    Generate the neuropil table for main groups aggregated at cell level.

    Parameters
    ----------
    df : pd.DataFrame
        data with all the cell instances and their corresponding cell type groups, color, etc.
    threshold : float, default=0.02
        threshold fraction of synapses/cells that needs to be crossed within a brain region
        to be assigned to that brain region

    Returns
    -------
    t_df : pd.DataFrame
        The DataFrame contains the main groups from df as columns and the number of individual
        cells of that group in each brain region that have more than `threshold` of their synapses
        in that region.
    """
    t_df = pd.DataFrame()
    for grp_name, grp in df.groupby(by='main_groups'):
        cql_cells=f"""
            UNWIND ['LA(R)', 'ME(R)', 'LO(R)', 'LOP(R)', 'AME(R)'] as roi
            MATCH (n:Neuron)
            WHERE n.instance in {grp['instance'].to_list()} and n[roi]
            WITH
                distinct n
              , apoc.convert.fromJsonMap(n.roiInfo) as nri
              , coalesce(n.pre, 0) + coalesce(n.post,0) as syn_total
              , roi
              , n.bodyId as bid
            with
                coalesce(nri[roi].pre, 0) as syn_pre
              , coalesce(nri[roi].post, 0) as syn_post
              , syn_total
              , roi
              , bid
            WITH
                CASE
                    WHEN syn_total> 0 THEN toFloat(syn_pre + syn_post)/ syn_total
                    ELSE 0
                END AS syn_frac
              , roi
              , bid

            WHERE syn_frac >= {threshold}
            RETURN distinct roi, count(bid) as n_cells
        """
        cells_df = fetch_custom(cql_cells)\
            .set_index('roi')\
            .rename(columns={'n_cells': grp_name})
        t_df = pd.concat([t_df, cells_df], axis=1)

    t_df = t_df.fillna(0).astype(int)

    return t_df


def get_celltypes_groups_df(
    neuron_list:pd.DataFrame
  , correct_counts:bool=False
) -> pd.DataFrame:
    """
    Get counts for the main groups.

    Parameters
    ----------
    neuron_list : pd.DataFrame
        list of neurons. 'instance' and 'main_groups' are needed for this function to work.
    correct_counts : bool
        Load manual corrections from `params/Corrections.xlsx` if true, otherwise use raw data
        from neuprint.

    Returns
    -------
    df : pd.DataFrame
        main_group : str
            name of the main group
        n_celltypes : int
            number of unique cell instances from `neuron_list` that belong to the `main_group`
        n_cells : int
            number of unique cells matching the instances from `neuron_list` that belong to the
            `main_group`
        n_synapses : int
            number of unique synapses of the cells contributing to `n_cells`
    """
    assert 'main_groups' in neuron_list.keys(),\
        "neuron list must contain column 'main_groups'. Try OLTypes to get the list."

    ct_df = pd.DataFrame()
    project_root = Path(find_dotenv()).parent
    corr = pd.read_excel(project_root / 'params' / 'Corrections.xlsx',  sheet_name='cells')
    corr = corr.drop(columns='Info').set_index('instance')
    for grp_name, grp in neuron_list.groupby(by='main_groups'):
        n_cells_correction = 0
        if correct_counts:
            n_cells_correction = grp\
                .set_index('instance')\
                .join(corr, how='inner')['diff_n_cells']\
                .sum()
        cql_cells=f"""
            MATCH (n:Neuron)
            WHERE n.instance in {grp['instance'].to_list()}
            RETURN
                '{grp_name}' as main_group
              , count(distinct n.instance) as n_cellinstances
              , count(distinct n.type) as n_celltypes
              , count(distinct n.bodyId) + coalesce({n_cells_correction}, 0) as n_cells
              , sum(n.upstream) as n_upstream
              , sum(n.downstream) as n_downstream
        """
        cells_df = fetch_custom(cql_cells)
        ct_df = pd.concat([ct_df, cells_df], axis=0)

    ct_df = ct_df.set_index('main_group').fillna(0).astype(int)
    return ct_df


def get_threshold_celltypes(
    df:pd.DataFrame
  , threshold:float=0.05
  , plot_specs:dict=None
) -> pd.DataFrame:
    """
    Generate the neuropil table based on brain region.

    Parameters
    ----------
    df : pd.DataFrame
        data with all the cell instances and their corresponding cell type groups, color, etc.
    threshold : float
        threshold fraction of synapses/cells that needs to be crossed within a brain region
        to be assigned to that brain region
    plot_spec : dict
        Must contain the key 'save_path'

    Returns
    -------
    neuropil_df : pd.DataFrame
        celltypes : str
            list of cell types whose population level synapses surpass the `threshold`
    """
    assert 0 <= threshold <= 1, f"Threshold must be between 0 and 1, not {threshold}"
    assert plot_specs is not None, "Please defined plot specs"
    assert 'save_path' in plot_specs.keys()

    cql_cell_types=f"""
        UNWIND ['LA(R)', 'ME(R)', 'LO(R)', 'LOP(R)', 'AME(R)'] as roi
        MATCH (n:Neuron)
        WHERE n.instance in {df['instance'].to_list()} and n[roi]
        WITH
            distinct n
          , apoc.convert.fromJsonMap(n.roiInfo) as nri
          , coalesce(n.pre, 0) + coalesce(n.post,0) as syn_total
          , roi
        with
            coalesce(nri[roi].pre, 0) as syn_pre
          , coalesce(nri[roi].post, 0) as syn_post
          , syn_total
          , roi
          , n
        WITH
            distinct n.type as type
          , roi
          , sum(syn_pre) as syn_pre
          , sum(syn_post) as syn_post
          , sum(syn_total) as syn_total

        WITH
          CASE
                WHEN syn_total> 0 THEN toFloat(syn_pre + syn_post)/ syn_total
                ELSE 0
            END AS syn_frac
          , roi
          , type
        WHERE syn_frac >= {threshold}
        RETURN
            distinct roi
          , count(type) as n_celltypes
          , type as type
    """

    celltype_df = fetch_custom(cql_cell_types).set_index('roi')

    return celltype_df
