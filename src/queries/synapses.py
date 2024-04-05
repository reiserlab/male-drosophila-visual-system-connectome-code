""" Queries for completeness notebooks """
import pandas as pd

from neuprint import Client
from neuprint.client import inject_client


@inject_client
def fetch_avg_synapses(
    *
  , body_ids:list[int]
  , client: Client = None
) -> pd.DataFrame:
    """
    Return the mean location of the synapses per body_id

    Parameter
    ---------
    body_ids : list[int]
        List of valid body IDs
    
    Returns
    -------
    synapse_df : pd.DataFrame
        bodyId : int
            neuron body ID
        x : float
            synapse center of mass X coordinate
        y : float
            synapse center of mass Y coordinate
        z : float
            synapse center of mass Z coordinate
    """

    cql = f"""
            WITH {list(body_ids)} as bodyIds
            MATCH (n: Neuron)
            WHERE n.bodyId in bodyIds
            MATCH
                (n)-[:Contains]->(ss:SynapseSet),
                (ss)-[:Contains]->(s:Synapse)
            WITH DISTINCT n, s
            RETURN
                n.bodyId as bodyId,
                avg(s.location.x) as x,
                avg(s.location.y) as y,
                avg(s.location.z) as z
            ORDER BY bodyId
    """

    synapse_df = client.fetch_custom(cql)
    return synapse_df