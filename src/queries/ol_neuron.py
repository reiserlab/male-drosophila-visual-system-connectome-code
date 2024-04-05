import pandas as pd

from neuprint import Client
from neuprint.client import inject_client


@inject_client
def fetch_ol_rois_distance(
    *
  , body_id:int
  , client:Client=None
) -> pd.DataFrame|list[str]:
    """
    Get the column and layer ROIs in ME(R), LO(R), and LOP(R) closest to centroid of all 
    synaptic sites.

    Parameters
    ----------
    body_id : int
        Body ID of a neuron
    
    Returns
    -------
    named_df : pd.DataFrame
        syn_keys : str
            name of column and layer ROIs
    """

    # It is faster to iterate through several thresholds than having a single (large)
    # one that captures all synapses…
    for dist_thresh in [100, 500, 1000, 5000]:
        cql = f"""
        WITH ['ME(R)', 'LO(R)', 'LOP(R)'] as rois
        UNWIND rois AS roi
        MATCH (n:Neuron)-[:Contains]->(nss:SynapseSet)-[:Contains]->(ns:Synapse)
        WHERE n.bodyId={body_id} AND ns[roi]
        WITH DISTINCT n, ns, roi
        WITH n, 
            point({{x:avg(ns.location.x), y:avg(ns.location.y), z:avg(ns.location.z)}}) as com, 
            roi
        MATCH (ns:Synapse) 
        WHERE ns[roi] AND point.distance(com, ns.location) < {dist_thresh}
        WITH point.distance(com, ns.location) AS dist, roi, ns
        ORDER BY dist
        UNWIND keys(ns) AS syn_keys
        WITH ns, roi, syn_keys, left(syn_keys, 5) in ['ME_R_', 'LO_R_', 'LOP_R'] AS is_in_OLR
        WHERE is_in_OLR
        WITH roi, ns, collect(syn_keys) as syn_key_list
        WITH roi, head(collect(syn_key_list)) as syn_key_list
        UNWIND(syn_key_list) as syn_keys
        RETURN syn_keys
        """
        named_df = client.fetch_custom(cql)
        if named_df.size:
            break
    return named_df

@inject_client
def fetch_ol_rois_synapses(
    *
  , body_id:int
  , client:Client=None
) -> pd.DataFrame:

    cql = f"""
    MATCH (n:Neuron)-[:Contains]->(nss:SynapseSet)-[:Contains]->(ns:Synapse)
    WHERE n.bodyId={body_id}
    WITH DISTINCT ns,
    CASE 
        WHEN ns.`ME(R)` THEN 'ME(R)'
        WHEN ns.`LO(R)` THEN 'LO(R)'
        WHEN ns.`LOP(R)` THEN 'LOP(R)'
        ELSE NULL
    END AS roi
    RETURN roi as ROI, ns.olHex1 as hex1_id, ns.olHex2 as hex2_id , count(*) as synapse_count
    ORDER BY synapse_count DESC
    """
    named_df = client.fetch_custom(cql)
    return named_df


@inject_client
def fetch_ol_rois_assigned(
    *
  , body_id:int
  , client:Client=None
) -> pd.DataFrame:

    cql = f"""
    MATCH (n:Neuron)
    WHERE 
        n.bodyId={body_id} 
        AND n.assignedOlHex1 IS NOT NULL
        AND n.assignedOlHex2 IS NOT NULL
    RETURN
        'ME(R)' AS ROI
      , n.assignedOlHex1 as hex1_id
      , n.assignedOlHex2 as hex2_id
    """
    named_df = client.fetch_custom(cql)
    return named_df