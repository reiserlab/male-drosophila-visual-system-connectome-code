"""
Functions to define edge IDs.

Functions extracted from @hoellerj's Edge_hex_ids.ipynb.
"""

import os
import pandas as pd
import numpy as np

from utils.ROI_calculus import _get_data_path

def create_edge_ids(
    roi_str = 'ME(R)'
) -> None:
    """
    Find edge coordinates of pins and store in, e.g. ME_hex_ids_edge.csv

    Parameters
    ----------
    syn_df : pd.DataFrame
        dataframe with 'bodyId', 'x', 'y', 'z' columns
    roi_str : str
        neuprint ROI, can only be ME(R), LO(R), LOP(R)
    samp : int
        sub-sampling factor for depth bins

    Returns
    -------
    syn_col_df : pd.DataFrame
        'hex1_id' : int
            defines column
        'hex2_id' : int
            defines column
        'col_count' : int
            number of synapses in that column (across all depth bins)
    """

    data_path = _get_data_path(reason='cache')
    
    pincushion_f = os.path.join(data_path,roi_str[:-3]+'_col_center_pins.pickle')
    with open(pincushion_f, mode='rb') as pc_fh:
        pin_cushion_df = pd.read_pickle(pc_fh)
    pin_cushion_df = pin_cushion_df.dropna()
    
    hex_edge_df = fl_get_edge_ids(pin_cushion_df)

    hex_edge_f = data_path / f"{roi_str[:-3]}_hex_ids_edge.csv"
    hex_edge_df.to_csv(hex_edge_f, index=True, header=False)


def fl_get_edge_ids(
    df:pd.DataFrame
) -> pd.DataFrame:
    """
    Find the hex ids of the edges

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that has `['hex1_id', 'hex2_id']` columns.
    
    Returns
    -------
    hex_edge_df : pd.DataFrame
        DataFrame with `['hex1_id', 'hex2_id']` of the edges 
        index is the subset of indeces of df that are on the edge
    """
    df = df\
        .drop_duplicates(['hex1_id', 'hex2_id'])[['hex1_id', 'hex2_id']]\
        .applymap(int)
    df['neighbors'] = df.apply(
        lambda r: np.where(
            (np.abs(df['hex1_id']-r.hex1_id)==1)
           &(np.abs(df['hex2_id']-r.hex2_id)==1))[0]
      , axis=1)
    df['is_edge'] = df.apply(lambda r: len(r['neighbors'])<4, axis=1)
    hex_edge_df = df[df['is_edge']][['hex1_id', 'hex2_id']]
        
    return hex_edge_df

