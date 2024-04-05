""" Queries for spatial coverage and complete notebooks """

import pandas as pd
from neuprint import Client
from neuprint.client import inject_client
from queries.completeness import fetch_ol_types_and_instances

@inject_client
def fetch_cells_synapses_per_col(
    *
    , cell_type:str=None
    , cell_instance:str=None
    , roi_str:str | list[str]=None
    , side:str='R-dominant'
    , syn_type:str='all'
    , client:Client=None
) -> pd.DataFrame:
    """
    For a given cell type and region (roi) return a dataframe where each row contains information
    about one roi-column that contains synapses from that cell type. Return the number of cells,
    the number of synapses and the bodyIds of the cells that innervate that roi-column sorted by
    the number of cells in the column.

    Parameters
    ----------
    cell_type : str
        cell type of interest
    cell_instance : str
        instance of neuron type of interest
    roi_str : str | list[str]
        optic lobe region of interest
    side : str, default = 'R-dominant'
        options include, 'R', 'L', 'R-dominant' or 'both'.
        'R' means all neurons that have their cellbody on the right side, 'L' means that their
        cellbody is on the left side, 'R-dominant' chooses the neurons that have their
        'dominant features' in the right hemisphere, and 'both' means to get both sides
        (if available).
        For most analysis that works on one side, the 'R-dominant' is probably the best choice.
        There will be a 'L-dominant' once the other side is proof-read. 'both' returns the types
        that are present on either side and counts their total. If you know what you are doing and
        there is a reasonto diverge, you can choose 'R' or 'L'.
    syn_type : str, default = 'all'
        synapse type to use. Possible options are 'pre', 'post' or 'all'
    client : neuprint.Client
        Client used for the connection. If no explicit client is provided, then the `defaultclient`
        is used.

    Returns
    -------
    df : pd.DataFrame
        column : str
            column id. In the style '(hex1_id)_(hex2_id)' i.e. '22_28'
        roi : str 
            region roi in which the synapse is found
        n_cells : int
            number of unique cells assigned to that column
        n_syn : int
            number of synapses assigned to that column
        cell_body_ids : list
            list of the bodyIds of the cells assigned to that column.

    """
    assert cell_type is None or cell_instance is None, "Can only use type or instance, not both."
    
    if cell_type:
        assert side in ["R", "L", "R-dominant", "both"],\
            f"Unsupported side '{side}' for type, only 'R', 'L', 'R-dominant' or 'both' are allowed."

    assert syn_type in ["pre", "post", "all"],\
        f"Unsupported syn_type '{syn_type}', only 'pre', 'post' or 'all'  are allowed"

    str_type_instance = ""
    str_side = ""
    str_syn = ""

    m_to_s = "(m:Neuron)-[:Contains]->(mss:SynapseSet)-[:Contains]->(ms:Synapse)"
    if syn_type=='post':
        str_syn = f"AND EXISTS {{{m_to_s}-[:SynapsesTo]->(ns)}}"
    elif syn_type == 'pre':
        str_syn = f"AND EXISTS {{{m_to_s}<-[:SynapsesTo]-(ns)}}"

    if isinstance(roi_str, str):
        roi_str = [roi_str]

    if cell_type:
        str_type_instance = f"AND n.type='{cell_type}'"
        if side in ['R', 'L']:
            str_side = f"AND n.instance ENDS WITH '_{side}'"

        if side in ['R-dominant', 'L-dominant']:
            types = fetch_ol_types_and_instances(side='R-dominant')
            side_to_choose = types[types['type']==cell_type]['instance'].to_list()[0][-1]
            str_side = f"AND n.instance ENDS WITH '_{side_to_choose}'"
    
    if cell_instance:
        str_type_instance = f"AND n.instance='{cell_instance}'"

    cql = f"""
        UNWIND {roi_str} as roi
        MATCH
            (n:Neuron)-[:Contains]->(nss:SynapseSet)-[:Contains]->(ns:Synapse)
        WHERE (
            n.`LA(R)`=True
            OR n.`ME(R)`=True
            OR n.`LO(R)`=True
            OR n.`LOP(R)`=True
            OR n.`AME(R)`=True
        )
        {str_type_instance}
        AND ns[roi] IS NOT NULL
        AND (exists(ns.olHex1) AND ns.olHex1 IS NOT NULL)
        AND (exists(ns.olHex2) AND ns.olHex2 IS NOT NULL)
        AND (n.instance IS NOT NULL AND n.instance <> ''
        {str_side})
        {str_syn}
        WITH DISTINCT n,ns, toString(ns.olHex1)+"_"+toString(ns.olHex2) as column, roi
        RETURN
            DISTINCT column
              , roi
              , count(distinct n.bodyId) as n_cells
              , count(distinct ns) as n_syn
              , collect(distinct n.bodyId) as cell_body_ids
        ORDER BY n_cells DESC
    """
    df = client.fetch_custom(cql)
    return df


@inject_client
def fetch_syn_per_col_for_instance(
    *
  , neuron_instance:str
  , roi_str:str="ME(R)"
  , syn_type:str='all'
  , client:Client=None
) -> pd.DataFrame:

    df = __fetch_syn_per_col(
        neuron_constraint = f"n.instance='{neuron_instance}'"
      , roi_str=roi_str
      , syn_type=syn_type
      , client=client
    )
    return df


@inject_client
def fetch_syn_per_col_for_bid(
    *
  , bodyId:int
  , roi_str:str="ME(R)"
  , syn_type:str='all'
  , client:Client=None
) -> pd.DataFrame:
    """
    For a given bodyId and region (roi) return a dataframe where each row contains information
    about one roi-column that contains synapses from that bodyId. Return a dataframe of the number
    of synapses ('synapse_count') and percentage of all synapses ('synapse_perc')
    from all innervated roi-columns for a particular cell.

    Parameters
    ----------
    bodyId : int
        neuron's bodyId
    roi : str, default='ME(R)'
        neuropil roi
    syn_type : str, default='all'
        synapse type to use. Possible options are 'pre', 'post' or 'all'
    client : neuprint.Client, default=None
        Client used for the connection. If no explicit client is provided, then the `defaultclient`
        is used.

    Returns
    -------
    df : pd.DataFrame
        column : str
            column id. In the style '(hex1_id)_(hex2_id)' i.e. '22_28'
        roi : str
            region roi in which the synapse is found
        bodyId : int
            bodyId of the neuron of interest
        synapse_count : int
            number of synapses assigned to that column
        synapse_perc : list
            proportion of all synapses of that cell in that column.

    """
    df = __fetch_syn_per_col(
        neuron_constraint = f"n.bodyId={bodyId}"
      , roi_str=roi_str
      , syn_type=syn_type
      , client=client
    )
    return df


@inject_client
def __fetch_syn_per_col(
    *
  , neuron_constraint:str
  , roi_str:str="ME(R)"
  , syn_type:str='all'
  , client:Client=None
) -> pd.DataFrame:
    """ 
    syn_type : str, default='all'
        synapse type to use. Possible options are 'pre', 'post' or 'all'
    """

    str_syn = ""

    assert syn_type in ["pre", "post", "all"],\
        f"Unsupported syn_type '{syn_type}', only 'pre', 'post' or 'all'  are allowed"
    
    assert roi_str in ["ME(R)", "LO(R)", "LOP(R)"],\
        f"Unsupported roi_str '{roi_str}', only 'ME(R)', 'LO(R)' or 'LOP(R)'  are allowed"

    m_to_s = "(m:Neuron)-[:Contains]->(mss:SynapseSet)-[:Contains]->(ms:Synapse)"
    if syn_type=='post':
        str_syn = f"AND EXISTS {{{m_to_s}-[:SynapsesTo]->(ns)}}"
    elif syn_type == 'pre':
        str_syn = f"AND EXISTS {{{m_to_s}<-[:SynapsesTo]-(ns)}}"

    cql = f"""
    MATCH (n:Neuron)-[:Contains]->(nss:SynapseSet)-[:Contains]->(ns:Synapse)
    WHERE {neuron_constraint}
        AND ns['{roi_str}'] IS NOT NULL
        AND (exists(ns.olHex1) and ns.olHex1 IS NOT NULL)
        AND (exists(ns.olHex2) and ns.olHex2 IS NOT NULL)
        {str_syn}
    WITH n, ns, toString(ns.olHex1)+'_'+toString(ns.olHex2) AS col
    WITH {{bid: n.bodyId, col: col, syn: count(distinct ns)}} as tmp_res, n.bodyId as tmpbid, count(distinct ns) as syn_count
    WITH tmpbid, collect(tmp_res) as agg_res, sum(syn_count) as total_syn_count
    UNWIND agg_res as per_col
    RETURN
        per_col.col as column
      , '{roi_str}' as roi
      , per_col.bid as bodyId
      , per_col.syn as synapse_count
      , toFloat(per_col.syn)/total_syn_count as synapse_perc
    ORDER BY bodyId, synapse_count DESC
    """
    df_neuron = client.fetch_custom(cql)
    return df_neuron