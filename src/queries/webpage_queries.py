import warnings

import pandas as pd

from neuprint import Client
from neuprint.client import inject_client

@inject_client
def fetch_nt_per_cell_type(
    *
  , cell_type: str
  , client:Client=None
) -> pd.DataFrame:
    """
    Get cell type level neurotransmitter prediction.

    Parameters
    ----------
    cell_type : str
        cell type of interest
    client : neuprint.Client
        Client used for the connection. If no explicit client is provided, then the `defaultclient`
        is used.

    Returns
    -------
    nt_pred : str
        cell type level neurotransmitter prediction
    """

    cql = f"""
        MATCH
            (n:Neuron)-[:Contains]->(nss:SynapseSet)-[:Contains]->(ns:Synapse)
        WHERE n.type='{cell_type}'
        WITH DISTINCT n
        RETURN DISTINCT n.type AS type
        , n.celltypePredictedNt AS NT

    """
    df = client.fetch_custom(cql)

    if len(df)>1:
        # more than one neurotransmitter assigned to that neuron type
        uni_nt = df['NT'].unique().tolist()
        # if both None and unclear are the options then take unclear
        if None in uni_nt and 'unclear' in uni_nt:
            df['nt_pred'] = 'unclear'
        # if None + other - take the other option
        elif None in uni_nt and 'unclear' not in uni_nt:
            df.dropna(subset=['NT'], inplace=True)
            df = df.reset_index().drop(columns='index')
            df['nt_pred'] = df['NT']
        # if unclear + other - take the other option
        elif 'unclear' in uni_nt and None not in uni_nt:
            df = df.drop(df[df['NT'] == 'unclear'].index)
            df['nt_pred'] = df['NT']
    else:
        df['nt_pred'] = df['NT'][0]

    nt_pred = df['nt_pred'][0]

    return nt_pred


@inject_client
def consensus_nt_for_instance(
    *
  , instance:str
  , client:Client=None
) -> str:
    """
    Get NT prediction for instance with the highest count.

    Parameters
    ----------
    instance : str
        name of the instance
    client : Client
        neuprint client. Uses default client when None

    Returns
    -------
    consensus_nt : str
        returns most likely neurotransmitter prediction.
    """
    cql = f"""
        MATCH (n:Neuron)
        WHERE n.instance='{instance}'
        RETURN n.consensusNt as consensusNT, count(n.bodyId) as cell_count
        ORDER BY cell_count
    """

    named_df = client.fetch_custom(cql)

    ret = None

    if len(named_df)>=1:
        ret = named_df['consensusNT'].values[0]
    if ret=='unclear':
        warnings.warn(f"Consensus NT for {instance} is 'unclear'.")
    if ret is None:
        warnings.warn(f"Can't find a Consensus NT for {instance}")
    return ret


@inject_client
def get_io_table(
    *
  , instance:str
  , direction:str
  , connection_cutoff:int=1
  , per_cell_cutoff:int=None
  , rois:list[str]=None
  , neuron_in_roi:bool=True
  , synapse_in_roi:bool=False
  , client:Client=None
) -> pd.DataFrame:
    """
    Get the values for the input or output table on the website.

    Parameters
    ----------
    instance : str
        name of the instance.
    direction : {'input', 'output'}
        get input or output table
    connection_cutoff : int, default=1
        remove any connecting neurons below that count. By default, instances that
        only have a single connection are removed from the table.
    rois : list, default=['ME(R)', 'LO(R)', 'LOP(R)', 'LA(R)', 'AME(R)']
        brain regions to consider for the connections.
    neuron_in_roi : bool, default=True
        Neurons have to be assigned to the specified brain regions.
    synapse_in_roi : bool, default=False
        Synapses have to lie within the specified brain regions. This is not the case by default.
    client : Client
        neuprint client. Uses default client if None.

    Returns
    -------
    io_table : DataFrame
        instance : str
            name of the target instance
        consensusNT : str
            consensus neurotransmitter prediction for target instance
        total_connections : int
            number of connections between specified instance and target instance
        cell_connections
            Median of cell connections per cell of instance
        percentage
            percentage of total input/output
        perc_cum
            cummulative percentage of input/output
    """
    assert direction in ['input', 'output'],\
        f"Only 'input' and 'output' allowed, not '{direction}'"

    assert connection_cutoff is None or per_cell_cutoff is None,\
        "you can only cut by total connections (connection_cutoff) or per cell, not both."
    if rois is None:
        rois = ['ME(R)', 'LO(R)', 'LOP(R)', 'LA(R)', 'AME(R)']

    if direction=='input':
        match_dir = "(n:Neuron)<-[e:ConnectsTo]-(m:Neuron)"
    elif direction=='output':
        match_dir = "(n:Neuron)-[e:ConnectsTo]->(m:Neuron)"

    roi_cyp = ""
    roi_cyp_con = ""

    if neuron_in_roi and rois:
        roi_cyp = " OR ".join([f"n.`{r_n}`" for r_n in rois])
        roi_cyp = f" AND ({roi_cyp}) "

    if synapse_in_roi and rois:
        roi_cyp_con = " OR ".join(
            [f"apoc.convert.fromJsonMap(e.roiInfo)['{r_n}'].post>1" for r_n in rois]
        )
        roi_cyp_con = f" AND ({roi_cyp_con}) "

    if connection_cutoff:
        cutoff = f"WHERE total_connections >= {connection_cutoff}"
    elif per_cell_cutoff:
        cutoff = f"WHERE cell_connections >= {per_cell_cutoff}"


    cql = f"""
        MATCH {match_dir}
        WHERE n.instance='{instance}'
            AND (NOT m.type IS NULL AND NOT m.instance IS NULL)
            {roi_cyp}
            {roi_cyp_con}
        WITH
            count(distinct n.bodyId) as cell_count
          , sum(e.weight) as allsyn
        MATCH
            {match_dir}
        WHERE n.instance='{instance}'
            AND (NOT m.type IS NULL AND NOT m.instance IS NULL)
            {roi_cyp}
            {roi_cyp_con}
        WITH
            m.type as type
          , m.instance as instance
          , sum(e.weight) as total_connections
          , tofloat(sum(e.weight)) / cell_count as cell_connections
          , tofloat(sum(e.weight)) / allsyn as percentage
          , collect(DISTINCT m.consensusNt) as consensusNT
        ORDER BY total_connections DESC, tolower(m.type) ASC
        {cutoff}
        RETURN
            instance
          , consensusNT
          , total_connections
          , cell_connections
          , percentage
    """
    named_df = client.fetch_custom(cql)
    named_df['consensusNT'] = named_df['consensusNT']\
        .apply(lambda col: sorted(col, key=str.lower))\
        .str[0] # alphabetical order will have "unclear" as the last item and
        # therefore prefer other NT if there is more than one.
    named_df['perc_cum'] = named_df['percentage'].cumsum()
    return named_df


@inject_client
def get_layer_synapses(
    *
  , instance:str
  , roi_str:str
  , client:Client=None
) -> pd.DataFrame:
    """
    Find instance's synapses for layers in ROI.

    Parameters
    ----------
    instance : str
        name of the instance
    roi_str : str
        one of the 3 layered brain regions (ME(R), LO(R), LOP(R)).
    client : Client
        neuprint client. Uses default of None
    """
    roi_layers = {
        'ME(R)': [f'M{idx}' for idx in range(1,11)]
      , 'LO(R)': ['LO1', 'LO2', 'LO3', 'LO4', 'LO5a', 'LO5b', 'LO6']
      , 'LOP(R)': [f'LOP{idx}' for idx in range(1,5)]
    }

    assert roi_str in ['ME(R)', 'LO(R)', 'LOP(R)'],\
        f"only ME, LO, and LOP have layers. "\
        f"You are looking for {roi_str}. Maybe use get_roi_synapses?"

    layer_list = []
    for idx, _ in enumerate(roi_layers[roi_str], start=1):
        if roi_str=='ME(R)':
            layer_list.append(f"ns.`{roi_str[:-3]}_{roi_str[-2]}_layer_{idx:02d}`")
        else:
            layer_list.append(f"ns.`{roi_str[:-3]}_{roi_str[-2]}_layer_{idx}`")
    layer_str = " OR ".join(layer_list)

    template = pd.DataFrame({
        'syn_type': ['pre']*len(roi_layers[roi_str])+['post']*len(roi_layers[roi_str])
      , 'named_layer': roi_layers[roi_str]*2
    })
    cql = f"""
        MATCH (n:Neuron)
        WHERE n.instance='{instance}'
        WITH count(DISTINCT n.bodyId) as bid_n
        MATCH (n:Neuron)-[:Contains]->(:SynapseSet)-[:Contains]->(ns:Synapse)
        WHERE
            n.instance='{instance}'
            AND ns.`{roi_str}`
            AND NOT ns.olLayer IS NULL
            AND ({layer_str})
        WITH
            n.bodyId as bid
          , ns.type as syn_type
          , ns.olLayer as layer
          , count(distinct ns) as syn_count
          , bid_n
        RETURN
            syn_type
          , layer
          , toFloat(sum(syn_count))/bid_n as mean_count
        ORDER BY syn_type DESC, layer ASC
    """
    # Median: , percentileDisc(syn_count, 0.5) as median_count

    named_df = client.fetch_custom(cql)
    layer_names = pd.DataFrame({'named_layer': roi_layers[roi_str]})
    layer_names.index.names = ['layer']
    layer_names.index += 1
    named_df = named_df.merge(layer_names, on='layer', how='left')
    named_df = template\
        .merge(
            named_df
          , on=['syn_type', 'named_layer']
          , how='left')\
        .reset_index(drop=True)
    return named_df


@inject_client
def get_roi_synapses(
    *
  , instance:str
  , roi_str:str
  , client:Client=None
) -> pd.DataFrame:
    """
    Find synapses inside an ROI.

    Parameters
    ----------
    instance : str
        name of the instance
    roi_str : str
        name of the ROI to find synapses in. Besides the ones in the database, the function
        supports the virtual ROI 'non-OL', which uses all primary ROIs except the known central
        brain ones.
    client : Client
        neuprint client. Uses the default client if not provided.
    """
    roi_list_cyp = f"WITH ['{roi_str}'] as roi_list"
    if roi_str == 'non-OL':
        roi_list_cyp = """
            MATCH (m:Meta)
            UNWIND m.primaryRois as rois
            WITH rois
            WHERE NOT rois IN [
                'AB(L)', 'AB(R)', 'ME(R)', 'LO(R)', 'LOP(R)', 'AME(R)', 'ME(L)', 'LO(L)', 'LOP(L)'
              , 'AME(L)', 'LA(R)', 'LA(L)', 'vnc-shell'
            ]
            WITH COLLECT(rois) as roi_list
        """

    cql = f"""
        {roi_list_cyp}
        MATCH (n:Neuron)
        WHERE n.instance='{instance}'
        WITH count(distinct n.bodyId) AS bid_n, roi_list
        MATCH (n:Neuron)
        WHERE n.instance='{instance}'
        WITH
            apoc.convert.fromJsonMap(n.roiInfo) as roiinfo
          , roi_list
          , n
          , bid_n
        UNWIND(roi_list) as roi
        UNWIND(['pre', 'post']) as syn_type
        RETURN
            syn_type
          , toFloat(sum(roiinfo[roi][syn_type])) / bid_n as mean_count
        ORDER BY syn_type DESC
    """
    # Median (wrong), percentileDisc(roiinfo[roi][syn_type], 0.5) as median_count

    named_df = client.fetch_custom(cql)
    return named_df
