""" Queries for completeness notebooks """
import pandas as pd

from neuprint import Client
from neuprint.client import inject_client
from textwrap import dedent


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


@inject_client
def fetch_nt_synapses(
    *
  , body_ids:list[int]
  , client:Client=None
) -> pd.DataFrame:
    """
    Fetches synaptic data for the given body IDs.

    Parameters
    ----------
    client : neuprint.Client
        The client object used to fetch data.
    body_ids : list[int] 
        A list of body IDs for which synaptic data is to be fetched.
    
    Returns
    -------
    nt_synapses : pd.DataFrame
        A pandas DataFrame containing the fetched synaptic data with the following columns:
        bodyId : int 
            The body ID of the neuron.
        type : str, {'pre', 'post'} 
            The type of the synapse.
        x : float
            The x-coordinate of the synapse location.
        y : float
            The y-coordinate of the synapse location.
        z : float
            The z-coordinate of the synapse location.
        ACh : float
            The probability of acetylcholine neurotransmitter.
        Glu : float
            The probability of glutamate neurotransmitter.
        GABA : float
            The probability of GABA neurotransmitter.
        His : float
            The probability of histamine neurotransmitter.
        Dop : float
            The probability of dopamine neurotransmitter.
        OA : float
            The probability of octopamine neurotransmitter.
        5HT : float
            The probability of serotonin neurotransmitter.
    """
    # Function implementation...
    cql = dedent(f"""
        WITH {list(body_ids)} as bodyIds
        MATCH
            (n:Neuron)-[:Contains]->(ss:SynapseSet),
            (ss)-[:Contains]->(s:Synapse)
        WHERE
            n.bodyId in bodyIds
            AND s.type = 'pre'
        WITH DISTINCT n, s
        RETURN 
            n.bodyId as bodyId,
            s.type as type,
            s.location.x as x,
            s.location.y as y,
            s.location.z as z,
            s.ntAcetylcholineProb as ACh,
            s.ntGlutamateProb as Glu,
            s.ntGabaProb as GABA,
            s.ntHistamineProb as His,
            s.ntDopamineProb as Dop,
            s.ntOctopamineProb as OA,
            s.ntSerotoninProb as Ser
        ORDER BY bodyId
    """)
    nt_synapse_df = client.fetch_custom(cql)
    nt_synapse_df = nt_synapse_df.rename(columns={'Ser': '5HT'})
    return nt_synapse_df


@inject_client
def fetch_nt_win_synapses(
    *
  , body_ids:list[int]
  , client:Client=None
) -> pd.DataFrame:
    """
    Fetch the name of the transmitter with the highest probability for each synapse.

    This is similar to `fetch_nt_synapses()`, but instead of providing the probability
    for each of the seven neurotransmitters as a numeric value, the function returns
    the name of the most likely neurotransmitter.

    Parameters
    ----------
    client : neuprint.Client
        The client object used to fetch data.
    body_ids : list[int] 
        A list of body IDs for which synaptic data is to be fetched.
    
    Returns
    -------
    nt_synapses : pd.DataFrame
        A pandas DataFrame containing the fetched synaptic data with the following columns:
        bodyId : int 
            The body ID of the neuron.
        type : str, {'pre', 'post'} 
            The type of the synapse.
        x : float
            The x-coordinate of the synapse location.
        y : float
            The y-coordinate of the synapse location.
        z : float
            The z-coordinate of the synapse location.
        nt : str, {'ACh', 'Glu', 'GABA'. 'His', 'Dop', 'OA', '5HT'}
            one of the 7 abbreviations for neurotransmitter
    """
    cql = dedent(f"""
        MATCH
            (n:Neuron)-[:Contains]->(ss:SynapseSet)-[:Contains]->(s:Synapse)
        WHERE
            n.bodyId IN {list(body_ids)}
            AND s.type = 'pre'
        WITH DISTINCT n, s
        UNWIND [
            s.ntAcetylcholineProb, s.ntGlutamateProb, s.ntGabaProb, 
            s.ntHistamineProb, s.ntDopamineProb, s.ntOctopamineProb, 
            s.ntSerotoninProb] AS nt_vals
        WITH n, s, max(nt_vals) AS maxcol
        RETURN
            n.bodyId AS bodyId,
            s.type AS type,
            s.location.x AS x,
            s.location.y AS y,
            s.location.z AS z,
            CASE 
            	WHEN maxcol=s.ntAcetylcholineProb then 'ACh'
                WHEN maxcol=s.ntGlutamateProb then 'Glu'
                WHEN maxcol=s.ntGabaProb then 'GABA'
                WHEN maxcol=s.ntHistamineProb then 'His'
                WHEN maxcol=s.ntDopamineProb then 'Dop'
                WHEN maxcol=s.ntOctopamineProb then 'OA'
                WHEN maxcol=s.ntSerotoninProb then '5HT'
                ELSE 'unknown'
            END AS nt,
            REDUCE(layer_roi = '', key IN keys(s) | CASE WHEN key CONTAINS '_layer_' OR key CONTAINS 'AME(R)' THEN key ELSE layer_roi END) AS layer_roi
        ORDER BY bodyId
    """)
    nt_synapse_df = client.fetch_custom(cql)
    return nt_synapse_df
