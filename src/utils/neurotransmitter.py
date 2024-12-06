from itertools import batched
from pathlib import Path
import hashlib

import pandas as pd
from dotenv import find_dotenv
from neuprint import fetch_neurons, NeuronCriteria as NC

from utils.ol_types import OLTypes

from queries.synapses import fetch_nt_win_synapses


def get_nt_for_bid(
    neuron_df:pd.DataFrame
  , ignore_unknown:bool=True
) -> pd.DataFrame:
    """

    Parameters
    ----------
    neuron_df : pd.DataFrame
        data frame with column 'bodyId'.
    ignore_unknown : bool
        ignores synapses that don't have a prediction. If false, these
        synapses are returned with `nt` set to 'unknown'

    Returns
    -------
    syn : pd.DataFrame
        DataFrame with information about Synapses for bodyId
        bodyId : int
            bodyId of neuron, generate from list in `neuron_df`
        type : str, {'pre', 'post'}
            type of synapse
        x : float
            x-position of synapse
        y : float
            y-position of synapse
        z : float
            z-position of synapse
        nt : str
            abbreviation for neuro transmitter prediction for synapse
    """

    assert isinstance(neuron_df, pd.DataFrame), "neuron_df list needs to be a DataFrame"
    assert 'bodyId' in neuron_df.columns, "neuron_df has to have column 'bodyId'"

    bid_list = list(set(neuron_df['bodyId'].to_list()))
    bid_list.sort()
    short_code = hashlib.sha1(",".join(map(str, bid_list)).encode("UTF-8")).hexdigest()[:10]

    syn_fn = Path(find_dotenv()).parent / 'cache' / 'nt' / f'syn_nt_{short_code}.pickle'
    
    if syn_fn.is_file():
        syn = pd.read_pickle(syn_fn)
    else:
        syn = pd.DataFrame()
        for batch_bodies in batched(bid_list, 10):
            batch_df = fetch_nt_win_synapses(body_ids=batch_bodies)
            syn = pd.concat([syn, batch_df], ignore_index=True)
        
        syn_fn.parent.mkdir(parents=True, exist_ok=True)
        syn.to_pickle(syn_fn)
    
    if ignore_unknown:
        syn = syn[syn['nt']!='unknown']
    return syn


def get_special_neuron_list() -> pd.DataFrame:
    """
    Retrieves a DataFrame of special neurons, either from a cached file or by fetching and processing data.
    The function first checks if a cached DataFrame file exists. If it does, the DataFrame is loaded from the file.
    If the file does not exist, it fetches the neuron list, processes the data, and saves it to the cache file for future use.

    Returns
    -------
    df : pd.DataFrame
        A DataFrame containing information about special neurons, including their bodyId, instance, type,
        pre, post, downstream, upstream, consensusNt, predictedNt, celltypePredictedNt, and main_groups.
    """

    neuron_fn = Path(find_dotenv()).parent / 'cache' / 'nt' / 'neuron_df.pickle'

    return_df = pd.DataFrame()

    if neuron_fn.is_file():
        return_df = pd.read_pickle(neuron_fn)
    else:
        olt = OLTypes()
        oltypes = olt.get_neuron_list(
            primary_classification=['OL', 'non-OL']
          , side='both'
        )
        neuron_df, _ = fetch_neurons(NC(instance=oltypes['instance'].tolist()))
        return_df = pd.merge(
            neuron_df[[
                'bodyId','instance', 'type'
              , 'pre', 'post', 'downstream', 'upstream'
              , 'consensusNt', 'predictedNt', 'celltypePredictedNt']]
          , oltypes[['instance', 'main_groups']]
          , left_on='instance'
          , right_on='instance'
          , how = 'left'
        )
        neuron_fn.parent.mkdir(parents=True, exist_ok=True)
        return_df.to_pickle(neuron_fn)
    return return_df
