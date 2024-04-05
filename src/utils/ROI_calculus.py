from pathlib import Path
import os

import numpy as np
import pandas as pd
from scipy import spatial
import kneed
import navis

from dotenv import find_dotenv
from neuprint import NeuronCriteria as NC, SynapseCriteria as SC, fetch_synapses
from utils.hex_hex import all_hex



def load_layer_thre(
    roi_str:str='ME(R)'
) -> np.ndarray:
    """
    Load layer tresholds

    Parameters
    ----------
    roi_str : str, default='ME(R)'
        neuprint ROI, can only be ME(R), LO(R), LOP(R)

    Returns
    -------
    output : np.ndarray
        boundaries values for depth to separate layers
    """

    assert roi_str in ['ME(R)', 'LO(R)', 'LOP(R)'],\
            f"ROI must be one of 'ME(R)', 'LO(R)', 'LOP(R)', but is actually '{roi_str}'"

    data_path = _get_data_path(reason='cache')

    roi_fn = data_path / f"{roi_str[:-3]}_layer_bdry.csv"
    if not roi_fn.is_file():
        create_ol_layer_boundaries([roi_str])

    depth_bdry = pd.read_csv(
        roi_fn
      , header=None
    )
    
    return np.squeeze(depth_bdry.values)


def find_mesh_layers(
    xyz_df
  , roi_str='ME(R)'
  , samp=1
) -> pd.DataFrame:
    """
    For a dataframe of 3D points, find which layer the points lie in.
    The difference to find_layers is only at the boundaries between the layers.
    This function uses smooth layer meshes to determine where a point close to a boundary belongs.

    Parameters
    ----------
    xyz_df : pd.DataFrame
        DataFrame with 'x', 'y', 'z' columns
    roi_str : str, default='ME(R)'
        neuprint ROI, can only be ME(R), LO(R), LOP(R)
    samp : int, default=1
        sub-sampling factor for depth bins

    Returns
    -------
    layer_df : pd.DataFrame
        'layer' : int
            layer numbers (starting from 1 at the top) that the corresponding points xyz_df lie in
    """
    
    assert roi_str in ['ME(R)', 'LO(R)', 'LOP(R)'],\
            f"ROI must be one of 'ME(R)', 'LO(R)', 'LOP(R)', but is actually '{roi_str}'"
    
    data_path = _get_data_path(reason='cache')
    layer_df = find_layers(xyz_df, roi_str=roi_str, samp=samp)

    layer_f = data_path / f"{roi_str[:-3]}_layer_1_L.obj"
    if not layer_f.is_file():
        make_large_mesh(roi_str=roi_str)

    depth_bdry = load_layer_thre(roi_str=roi_str)
    for i in range(len(depth_bdry)-1):
        layer_f = data_path / f"{roi_str[:-3]}_layer_{str(i+1)}_L.obj"
        layer_i = navis.Volume.from_file(layer_f)
        layer_df[navis.in_volume(xyz_df, layer_i)] = i+1
    
    return layer_df
    

def find_layers(
    xyz_df
  , roi_str='ME(R)'
  , samp=1
) -> pd.DataFrame:
    """
    For a dataframe of 3D points, find which layer the points lie in

    Parameters
    ----------
    xyz_df : pd.DataFrame
        DataFrame with 'x', 'y', 'z' columns
    roi_str : str, default='ME(R)'
        neuprint ROI, can only be ME(R), LO(R), LOP(R)
    samp : int, default=1
        sub-sampling factor for depth bins

    Returns
    -------
    layer_df : pd.DataFrame
        'layer' : int
            layer numbers (starting from 1 at the top) that the corresponding points xyz_df lie in
    """

    assert roi_str in ['ME(R)', 'LO(R)', 'LOP(R)'],\
            f"ROI must be one of 'ME(R)', 'LO(R)', 'LOP(R)', but is actually '{roi_str}'"

    #load layer tresholds
    depth_bdry = load_layer_thre(roi_str=roi_str)

    #find fine depth
    depth_df = find_depth(xyz_df, roi_str=roi_str, samp=samp)

    layer_ass = np.empty(xyz_df.shape[0])
    layer_ass[:] = np.nan
    for i in range(len(depth_bdry)-1):
        layer_ass[
            (depth_df['depth'] >  depth_bdry[i])\
          & (depth_df['depth'] <= depth_bdry[i + 1])
        ] = i + 1
    layer_df = pd.DataFrame(layer_ass, columns=['layer'])

    return layer_df


def find_depth(
    xyz_df
  , roi_str='ME(R)'
  , samp=2
) -> pd.DataFrame:
    """
    For a dataframe of 3D points, find depth (between 0 and 1 where 0 is at the top and 1 at the
    bottom) and depth bin (an integer from 0 to N-1 where N is the number of depth bins).

    Parameters
    ----------
    xyz_df : pd.DataFrame
        DataFrame with 'x', 'y', 'z' columns
    roi_str : str
        neuprint ROI, can only be ME(R), LO(R), LOP(R)
    samp : int
        sub-sampling factor for depth bins

    Returns
    -------
    depth_values_df : pd.DataFrame
        'depth' : float
            normalized depths that corresponding points in `xyz_df` lie in
        'bin : int
            depth bins that corresponding points in `xyz_df` lie in
    """

    assert roi_str in ['ME(R)', 'LO(R)', 'LOP(R)'],\
            f"ROI must be one of 'ME(R)', 'LO(R)', 'LOP(R)', but is actually '{roi_str}'"

    _, n_bins, pins = load_pins(roi_str=roi_str)

    #fast way to find minimal distance between points in "pins" and "xyz_df"
    tree = spatial.KDTree(pins)
    _, minid = tree.query(xyz_df[['x','y','z']].values)

    #find synapses within depth range
    depth_bins = np.mod(minid, n_bins)

    #subsample
    n_bins = int(np.floor((n_bins-1)/samp))+1
    depth_bins = np.asarray(np.floor(depth_bins / samp), dtype='int')

    #store in dataframe
    depth_values_df = pd.DataFrame\
        .from_dict({
            'depth': (n_bins-1-depth_bins)/(n_bins-1)
          , 'bin': n_bins-1-depth_bins
        })

    return depth_values_df


def load_depth_bins(
    roi_str:str='ME(R)'
  , samp:int=2
):
    """
    Load edge and center bins for depth

    Parameters
    ----------
    roi_str : str
        neuprint ROI, can only be ME(R), LO(R), LOP(R)
    samp : int
        sub-sampling factor for depth bins

    Returns
    -------
    bin_edges : np.ndarray
        bin edges for depth
    bin_centers : np.ndarray
        bin centers for depth
    """

    assert roi_str in ['ME(R)', 'LO(R)', 'LOP(R)'],\
            f"ROI must be one of 'ME(R)', 'LO(R)', 'LOP(R)', but is actually '{roi_str}'"

    #load depth pins
    _, n_bins, _ = load_pins(roi_str=roi_str)
    n_bins_samp = int(np.floor((n_bins - 1) / samp)) + 1
    #binning in depth
    bin_edges = np.linspace(
        0 - 1 / (n_bins_samp - 1) / 2
      , 1 + 1 / (n_bins_samp - 1) / 2
      , n_bins_samp + 1
    )
    #centers of bins = depth bins
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

    return bin_edges, bin_centers


def load_pins(
    roi_str:str='ME(R)'
  , suffix:str=''
):
    """
    Load columns/pins

    Parameters
    ----------
    roi_str : str, default='ME(R)'
        neuprint ROI, can only be ME(R), LO(R), LOP(R)
    suffix: str, default=''
        allows to load files named {roi_str[:-3]}_col_center_pins{suffix}.pickle

    Returns
    -------
    col_ids : np.ndarray
        integers that are in 1-1 correspondence with hex ids,
        the correspondence is given by the rank of the ascending
            ordering of all (hex1_id, hex2_id) in the ME
    n_bins : int
        number of depth bins (same for all pins)
    pins : np.ndarray
        xyz positions of pin nodes
        size is (col_ids.shape[0]*n_bins)x3
        e.g. the first n_bins rows of pins make up the pin with col_id[0]
    """

    assert roi_str in ['ME(R)', 'LO(R)', 'LOP(R)'],\
            f"ROI must be one of 'ME(R)', 'LO(R)', 'LOP(R)', but is actually '{roi_str}'"

    data_path = _get_data_path(reason='cache')
    col_df = pd.read_pickle(
        data_path / f"{roi_str[:-3]}_col_center_pins{suffix}.pickle"
    )

    #remove columns that were not created
    col_df = col_df.dropna()
    col_ids = col_df.index.values

    #number of depth bins
    n_bins = int((col_df.shape[1]-3)/3)
    #get xyz positions of columns nodes
    pins = col_df.iloc[:, 3:].values.reshape((-1, 3))

    return col_ids, n_bins, pins


def find_neuron_hex_ids(
    syn_df
  , roi_str='ME(R)'
  , method='majority'
):
    """
    Assign a single hex coordinate to a neuron, either based on where the majority of synapses lie
        or based on the center of mass (COM).
        This assumes all synapses lie in the the ROI given by roi_str

    Parameters
    ----------
    syn_df : pd.DataFrame
        DataFrame with 'bodyId', 'x', 'y', 'z' columns
    roi_str : str
        neuprint ROI, can only be ME(R), LO(R), LOP(R)
    method : str
        either 'majority' or 'COM'

    Returns
    -------
    target_df : pd.DataFrame
        'bodyId' : int
            body ID of neuron
        'col_id' : int
            column descriptor
        'hex1_id' : int
            column descriptor
        'hex2_id' : int
            column descriptor
    """

    assert roi_str in ['ME(R)', 'LO(R)', 'LOP(R)'],\
            f"ROI must be one of 'ME(R)', 'LO(R)', 'LOP(R)', but is actually '{roi_str}'"

    data_path = _get_data_path(reason='cache')

    if method=='majority':
        hex_df = find_hex_ids(syn_df, roi_str=roi_str)
        syn_df['col_id'] = hex_df['col_id'].values
        target_df = pd.DataFrame(
            syn_df\
                .groupby('bodyId')[['col_id']]\
                .apply(lambda x:x.mode())\
                .droplevel(1)
        )
    elif method=='COM':
        target_df = pd.DataFrame(syn_df.groupby('bodyId')[['x','y','z']].mean())
        hex_df = find_hex_ids(target_df, roi_str=roi_str)
        target_df['col_id'] = hex_df['col_id'].values

    #load all hex ids
    col_df = all_hex()
    col_df.index.name = 'col_id'

    #attach hex1_id, hex2_id
    target_df.sort_values('col_id', inplace=True)
    target_df.reset_index(inplace=True)
    target_df = target_df.merge(col_df, 'left', on='col_id')
    return target_df


def find_hex_ids(
    xyz_df
  , roi_str='ME(R)'
) -> pd.DataFrame:
    """
    Assign 3D points to columns.
        This assumes all synapses lie in the the ROI given by roi_str

    Parameters
    ----------
    xyz_df : pd.DataFrame
        DataFrame with 'x', 'y', 'z' columns
    roi_str : str, default='ME(R)'
        neuprint ROI, can only be ME(R), LO(R), LOP(R)

    Returns
    -------
    result_df : pd.DataFrame
        'col_id' : int
            column descriptor
        'hex1_id' : int
            column descriptor
        'hex2_id' : int
            column descriptor
    """

    assert roi_str in ['ME(R)', 'LO(R)', 'LOP(R)'],\
            f"ROI must be one of 'ME(R)', 'LO(R)', 'LOP(R)', but is actually '{roi_str}'"

    data_path = _get_data_path(reason='cache')
    col_ids, n_bins, pins = load_pins(roi_str=roi_str)

    #fast way to find minimal distance between points in "pins" and "xyz_df"
    tree = spatial.KDTree(pins)
    _, minid = tree.query(xyz_df[['x','y','z']].values)

    #load all hex ids
    col_df = all_hex()
    col_df.index.name = 'col_id'

    #get column index and convert to hex ids
    result_df = pd.DataFrame(col_ids[np.floor(minid / n_bins).astype(int)], columns=['col_id'])
    result_df = result_df.merge(col_df, 'left', on='col_id')

    return result_df


def find_straight_hex_ids(
    xyz_df
  , roi_str='ME(R)'
  , suffix=''
) -> pd.DataFrame:
    """
    Assign 3D points to straight columns.
        This assumes all synapses lie in the the ROI given by roi_str

    Parameters
    ----------
    xyz_df : pd.DataFrame
        DataFrame with 'x', 'y', 'z' columns
    roi_str : str, default='ME(R)'
        neuprint ROI, can only be ME(R), LO(R), LOP(R)
    suffix: str, default=''
        allows to load files named {roi_str[:-3]}_col_center_pins{suffix}.pickle

    Returns
    -------
    result_df : pd.DataFrame
        'col_id' : int
            column descriptor
        'hex1_id' : int
            column descriptor
        'hex2_id' : int
            column descriptor
    """

    assert roi_str in ['ME(R)', 'LO(R)', 'LOP(R)'],\
            f"ROI must be one of 'ME(R)', 'LO(R)', 'LOP(R)', but is actually '{roi_str}'"

    data_path = _get_data_path(reason='cache')
    col_ids, n_bins, pins = load_pins(roi_str=roi_str, suffix=suffix)
    pin_interp = np.linspace(0,1,n_bins)
    for j in range(col_ids.shape[0]):
        pins[j*n_bins:(j+1)*n_bins] = (1-pin_interp[:,np.newaxis])*pins[j*n_bins][np.newaxis,:] +\
                                    pin_interp[:,np.newaxis]*pins[(j+1)*n_bins-1][np.newaxis,:]

    #fast way to find minimal distance between points in "pins" and "xyz_df"
    tree = spatial.KDTree(pins)
    _, minid = tree.query(xyz_df[['x','y','z']].values)

    #load all hex ids
    col_df = all_hex()
    col_df.index.name = 'col_id'

    #get column index and convert to hex ids
    result_df = pd.DataFrame(col_ids[np.floor(minid / n_bins).astype(int)], columns=['col_id'])
    result_df = result_df.merge(col_df, 'left', on='col_id')

    return result_df


def _get_data_path(
    reason:str='data'
) -> Path:
    """
    Get data path

    Move from global variable to function.

    Parameter
    ---------
    reason : str, default='data'
        what the path will be used for. Currently supports 'data', 'cache', 'params'

    Returns
    -------
    data_path : str
        data path used throughout the functions of this file.
    """
    assert reason in ['data', 'cache', 'params'], f"path requested for unknonw {reason}"

    if reason == 'data':
        data_path = Path(find_dotenv()).parent / 'results' / 'eyemap'
    elif reason == 'cache':
        data_path = Path(find_dotenv()).parent / 'cache' / 'eyemap'
    elif reason == 'params':
        data_path = Path(find_dotenv()).parent / 'params'

    data_path.mkdir(parents=True, exist_ok=True)
    return data_path

from utils.ROI_layers import create_ol_layer_boundaries, make_large_mesh

def find_col_names(
    hex_df:pd.DataFrame
):
    """
    Define 'col_name' as hex1_id*100 + hex2_id

    Parameters
    ----------
    hex_df : pd.DataFrame
        'hex1_id' : int
            column descriptor
        'hex2_id' : int
            column descriptor

    Returns
    -------
    col_df : pd.DataFrame
        'col_name' : int
            hex1_id*100 + hex2_id
    """

    hex_df.reset_index(inplace=True, drop=True)
    col_names = 100 * hex_df['hex1_id'].values + hex_df['hex2_id'].values
    #store in dataframe
    col_df = pd.DataFrame.from_dict({'col_name': col_names})

    return col_df