from pathlib import Path
from dotenv import find_dotenv
from typing import Union

import pandas as pd

from utils.ol_neuron import OLNeuron

#FIXME(@floesche): integrate results from align_mi1_t4.py

def all_hex(
) -> pd.DataFrame:
    """
    Return dataframe with all 'hex1_id', 'hex2_id' coordinates
        
    Parameters
    ----------
    None

    Returns
    -------
    rtn : pd.DataFrame
        the columns are 'hex1_id' and 'hex2_id' and their values give all (hex1, hex2) in the ME,
        which is used as a reference in all neuropils
    """
    data_dir = Path(find_dotenv()).parent / "params"
    me_hex_fn = data_dir / "ME_columnar-cells_location.xlsx"
    me_df = pd.read_excel(me_hex_fn).convert_dtypes()
    hex_df = me_df[['hex1_id','hex2_id']].drop_duplicates()

    return hex_df.reset_index(drop=True)


def hex_to_bids(
    hex_ids:tuple[int,int]
  , n_types:list[str]\
        =['L1', 'L2', 'L3', 'L5', 'Mi1', 'Mi4', 'Mi9', 'C2', 'C3', 'Tm1', 'Tm2', 'Tm4', 'Tm9', 'Tm20', 'T1']
  , return_type:str = 'dict'
  , neuropil:str='ME'
) -> dict[str,list[int]]:
    """
    Get body ids for the cells that were assigned to a specific column.

    Parameters
    ----------
    hex_ids : tuple[int,int]
        tuple consisting of (hex1_id, hex2_id) that defines the location of one column.
    n_types : list[str], default=[list of all columnar cell types]
        the cell types to return body ids for
    return_type : str, default='dict'
        allows either ['dict', 'list'] as the return, so that the function either returns a dictionary or a list.
    neuropil : str, default='ME'
        meant for future extension of the function. At this point we only have Medulla columns.

    Returns
    -------
    rtn : dict[str,list[int]]
        the keys represent cell types, the values are lists of body ids for this cell type
          within (hex1, hex2)
    """
    assert neuropil == 'ME', "only Medulla is supported so far"
    assert return_type in ['dict', 'list'], "return_type must be 'dict' or 'list'"
    rtn = {}
    data_dir = Path(find_dotenv()).parent / "params"
    me_hex_fn = data_dir / "ME_columnar-cells_location.xlsx"
    me_df = pd.read_excel(me_hex_fn).convert_dtypes()
    assert me_df['hex1_id'].min()<=hex_ids[0]<=me_df['hex1_id'].max()\
        , "hex1_id out of range"
    assert me_df['hex2_id'].min()<=hex_ids[1]<=me_df['hex2_id'].max()\
        , "hex2_id out of range"
    for n_type in n_types:
        if n_type not in me_df.columns:
            continue
        me_cl = me_df[(me_df['hex1_id']==hex_ids[0]) & (me_df['hex2_id']==hex_ids[1])]\
            .loc[:, n_type]\
            .dropna()\
            .astype('Int64')\
            .unique()\
            .tolist()
        if me_cl:
            rtn[n_type] = me_cl
    if return_type=='list':
        rtn = sum([*rtn.values()], [])
    return rtn


def bid_to_hex(bid:int, neuropil='ME(R)') -> Union[tuple[int,int], list[tuple[int,int]]]:
    """
    Find hex coordinates for body ID

    Parameters
    ----------
    bid : int
        body id for neuron

    Returns
    -------
    rtn : tuple[int,int] | list[tuple[int,int]]
        If the body id is in a Medulla column, the function returns the (hex1_id, hex2_id). In the
          rare case that a neuron is assigned to more than one column, it returns a list of tuples.
    """
    assert neuropil in ['ME(R)', 'LO(R)', 'LOP(R)'],\
        f"only  ME(R)', 'LO(R)' and 'LOP(R)' are supported so far, not {neuropil}"

    rtn = None
    oln = OLNeuron(bid)
    np_df = oln.get_hex_id(roi_str=neuropil)
    for _, row in np_df.iterrows():
        if rtn:
            rtn = [rtn, (row['hex1_id'], row['hex2_id'])]
        else:
            rtn = (row['hex1_id'], row['hex2_id'])
    return rtn


def get_hex_df(neuropil:str='ME') -> pd.DataFrame:
    """
    Get access to the full data frame

    This gives you the data frame without needing to know where exactly the file is stored or if
      the columns are directly pulled from neuPrint (future extension). Generally using this
      function is more advisable than using the current pickle file, but less advisable than using
      specialized functions to pull the information you need. If a function doesn't work as
      expected or if you are missing something, get in contact with @floesche or @kitlongden instead.

    Parameter
    ---------
    neuropil : str, default='ME'
        define the neuropil for which you want the data frame.
          (inactive, placeholder for future extension)

    Returns
    -------
    me_df : pd.DataFrame
        data frame with columns ['hex1_id', 'hex2_id'] and one additional column for each cell type
          that is assigned.

    """
    assert neuropil == 'ME', "only Medulla is supported so far"
    data_dir = Path(find_dotenv()).parent / "params"
    me_hex_fn = data_dir / "ME_columnar-cells_location.xlsx"
    me_df = pd.read_excel(me_hex_fn).convert_dtypes()
    me_df = me_df\
        .sort_values(['hex1_id', 'hex2_id'])
    return me_df


def get_incomplete_hex(neuropil:str='ME') -> list[tuple[int,int]]: # TODO: add treshold of how many are missing?
    """
    Get a list of hex ids that are missing at least one cell type

    Parameter
    ---------
    neuropil : str, default='ME'
        define the neuropil for which you want the data frame.
          (inactive, placeholder for future extension)

    Return
    ------
    rtn : list[tuple[int,int]]
        list of (hex1_id, hex2_id)
    """
    assert neuropil == 'ME', "only Medulla is supported so far"
    rtn = []
    data_dir = Path(find_dotenv()).parent / "params"
    me_hex_fn = data_dir / "ME_columnar-cells_location.xlsx"
    me_df = pd.read_excel(me_hex_fn).convert_dtypes()
    lst = me_df[me_df.isna().any(axis='columns')]\
        .loc[:,['hex1_id', 'hex2_id']]\
        .astype('Int64')\
        .drop_duplicates(subset=['hex1_id', 'hex2_id'])\
        .sort_values(['hex1_id', 'hex2_id'])
    for _, row in lst.iterrows():
        rtn.append((row['hex1_id'], row['hex2_id']))
    return rtn


def get_overfull_hex(neuropil:str='ME') -> list[tuple[int,int]]: # threshold for overfull cell types
    """
    Get a list of hex ids that have more than one cell of any type.

   Parameter
    ---------
    neuropil : str, default='ME'
        define the neuropil for which you want the data frame.
          (inactive, placeholder for future extension)

    Return
    ------
    rtn : list[tuple[int,int]]
        list of (hex1_id, hex2_id)
    """
    assert neuropil == 'ME', "only Medulla is supported so far"
    rtn = []
    data_dir = Path(find_dotenv()).parent / "params"
    me_hex_fn = data_dir / "ME_columnar-cells_location.xlsx"
    me_df = pd.read_excel(me_hex_fn).convert_dtypes()
    lst = me_df[me_df.duplicated(['hex1_id', 'hex2_id'])]\
        .loc[:,['hex1_id', 'hex2_id']]\
        .astype('int64')\
        .drop_duplicates(subset=['hex1_id', 'hex2_id'])\
        .sort_values(['hex1_id', 'hex2_id'])
    for _, row in lst.iterrows():
        rtn.append((row['hex1_id'], row['hex2_id']))
    return rtn
