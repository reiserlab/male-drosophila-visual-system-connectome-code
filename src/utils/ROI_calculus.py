import numpy as np
import pandas as pd
from scipy import spatial
import navis
import warnings

from utils.hex_hex import all_hex
from utils.helper import get_data_path
from queries.coverage_queries import fetch_pin_points

def find_per_columnbin_spanned_no_cols(
    syn_df
  , roi_str='ME(R)'
  , samp=2
):
    """
    For each depth and neuron, count number of columns that synapses lie in.
        Option to trim synapses.

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
    size_df : pd.DataFrame
        'bodyId' : int
            body ID of neuron
        'bin' : int
            depth bin of synapse after trimming
        'size' : int
            number of columns for remaining synapses
    rank_thre : int
        max. rank of columns to keep for each neuron, set to -1 if trim=False
    cumsum_thre : float
        cumulative fraction of synapses that is reached at that rank for the median,
            set to -1 if trim=False
    """

    assert roi_str in ['ME(R)', 'LO(R)', 'LOP(R)'],\
            f"ROI must be one of 'ME(R)', 'LO(R)', 'LOP(R)', but is actually '{roi_str}'"

    hex_df = find_hex_ids(syn_df, roi_str=roi_str)
    depth_df = find_depth(syn_df, roi_str=roi_str, samp=samp)
    syn_hex_df = pd.concat([
            syn_df.reset_index(drop=True)
          , hex_df[['hex1_id','hex2_id']]
          , depth_df]
      , axis=1)
    #dummies
    rank_thre = -1
    cumsum_thre = -1
    syn_hex_df['hex1_id'] = syn_hex_df['hex1_id'].astype(int)
    syn_hex_df['hex2_id'] = syn_hex_df['hex2_id'].astype(int)
    #size per bin is the number of columns for remaining synapses
    size_df = syn_hex_df\
        .groupby(['bodyId','bin'])\
        .apply(count_hex_loc)\
        .reset_index()

    return size_df, rank_thre, cumsum_thre


def count_hex_loc(group) -> pd.Series:
    """
    Helper function used in DataFrameGroupBy.apply to count columns

    Parameters
    ----------
    group : pd.DataFrame
        'hex1_id' : int
            defines column
        'hex2_id' : int
            defines column

    Returns
    -------
    output: pd.Series
        'size': int
            count how many different tuples (hex1_id, hex2_id) exist
    """
    _, idcs = np.unique(group[['hex1_id','hex2_id']].values, axis=0, return_index=True)
    count_dict = {}
    count_dict['size'] = idcs.shape[0]
    return pd.Series(count_dict)


def load_layer_thre(
    roi_str:str='ME(R)'
) -> np.ndarray:
    """
    Load layer thresholds

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

    data_path = get_data_path(reason='cache')

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

    data_path = get_data_path(reason='cache')
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

    _, _, n_bins, pins = load_pins(roi_str=roi_str)

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
    _, _, n_bins, _ = load_pins(roi_str=roi_str)
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
  , ignore_cache:bool=False
  , suffix:str=''
) -> tuple[np.ndarray, pd.DataFrame, int, np.ndarray]:
    """
    Load columns/pins

    Parameters
    ----------
    roi_str : str, default='ME(R)'
        neuprint ROI, can only be ME(R), LO(R), LOP(R)
    ignore_cache : bool, default=False
        When enabled, load the pin points from neuprint and overwrite the currently cached file.
    suffix: str, default=''
        allows to load files named {roi_str[:-3]}_col_center_pins{suffix}.pickle

    Returns
    -------
    col_ids : np.ndarray
        integers that are in 1-1 correspondence with hex ids,
        the correspondence is given by the rank of the ascending
            ordering of all (hex1_id, hex2_id) in the ME
    hex_ids : pd.DataFrame
        Data frame containing the 'hex1_id' and 'hex2_id' values of
         each of the columns in the ROI (roi_str).
    n_bins : int
        number of depth bins (same for all pins)
    pins : np.ndarray
        xyz positions of pin nodes
        size is (col_ids.shape[0]*n_bins)x3
        e.g. the first n_bins rows of pins make up the pin with col_id[0]
    """

    assert roi_str in ['ME(R)', 'LO(R)', 'LOP(R)'],\
            f"ROI must be one of 'ME(R)', 'LO(R)', 'LOP(R)', but is actually '{roi_str}'"

    if ignore_cache:
        # fetch pin points from the database for the given roi
        col_df = fetch_pin_points(roi_str=roi_str)
        # remove columns that were not created
        col_df = col_df.dropna()
        # data frame of column ids
        col_ids = col_df[['hex1_id', 'hex2_id']].drop_duplicates().reset_index().index.values
        # data frame of hex ids
        hex_ids = col_df[['hex1_id', 'hex2_id']].drop_duplicates().reset_index(drop='true')
        # number of depth bins
        n_bins = int(col_df.groupby(['hex1_id', 'hex2_id'])['x'].nunique().values.max())
        # get xyz positions of columns nodes
        pins = col_df[['x', 'y', 'z']].to_numpy()
    else:
        data_path = get_data_path(reason='cache')
        cache_file = data_path / f"{roi_str[:-3]}_col_center_pins{suffix}.pickle"
        if not cache_file.is_file():
            warnings.warn("WARNING. Strongly recommended to NOT run this function. "
                          "The required file does not exist, but should exist within the cache. "
                          f"Generating the file {cache_file} from scratch. "
                          "This will take a long time (>36 hours)."
            )
            # Create column pins from scratch. NOT RECOMMENDED.
            create_column_pins()
        # Read in data from pickle files in cache
        col_df = pd.read_pickle(cache_file)
        # remove columns that were not created
        col_df = col_df.dropna()
        col_ids = col_df.index.values
        # data frame of hex ids
        hex_ids = col_df[['hex1_id', 'hex2_id']].reset_index(drop=True)
        # number of depth bins
        n_bins = int((col_df.shape[1]-3)/3)
        # get xyz positions of columns nodes
        pins = col_df.iloc[:, 3:].values.reshape((-1, 3))

    return col_ids, hex_ids, n_bins, pins


def find_neuron_hex_ids(
    syn_df
  , roi_str='ME(R)'
  , method='majority'
) -> pd.DataFrame:
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
    assert method in ['majority', 'COM'],\
        f"method must be 'majority' or 'COM', not {method}"

    if method=='majority':
        hex_df = find_hex_ids(syn_df, roi_str=roi_str)
        syn_df['col_id'] = hex_df['col_id'].values
        target_df = pd.DataFrame(
            syn_df\
                .groupby('bodyId')[['col_id']]\
                .apply(lambda x:x.mode())\
                .droplevel(1)
        )
    else:
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

    col_ids, _, n_bins, pins = load_pins(roi_str=roi_str)

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

    col_ids, _, n_bins, pins = load_pins(roi_str=roi_str, ignore_cache=False, suffix=suffix)
    pin_interp = np.linspace(0,1,n_bins)
    for j in range(col_ids.shape[0]):
        pins[j*n_bins:(j+1)*n_bins] = \
            (1 - pin_interp[:, np.newaxis]) * pins[j * n_bins][np.newaxis, :]\
          + pin_interp[:, np.newaxis] * pins[(j + 1) * n_bins - 1][np.newaxis, :]

    #fast way to find minimal distance between points in "pins" and "xyz_df"
    tree = spatial.KDTree(pins)
    _, minid = tree.query(xyz_df[['x', 'y', 'z']].values)

    #load all hex ids
    col_df = all_hex()
    col_df.index.name = 'col_id'

    #get column index and convert to hex ids
    result_df = pd.DataFrame(col_ids[np.floor(minid / n_bins).astype(int)], columns=['col_id'])
    result_df = result_df.merge(col_df, 'left', on='col_id')

    return result_df


def find_col_names(
    hex_df:pd.DataFrame
) -> pd.DataFrame:
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

def create_column_pins():
    """
    Function to run the code that generates the column pins from scratch. Not recommended.
    These files should already exist within the `cache/eyemap` folder. 

    Expected runtimes (on 4 cores @3.7GHz):
    - ME(R): 20hrs
    - ME(R) old: 28hrs
    - LO(R): 7hrs
    - LOP(R): 40 min

    """
    #specify which neuropils to make pins in, and how to anchor the pins to the neuropil ROI
    roi_pins_dict_list = [
        {'roi': 'LOP(R)', 'anchor_method': 'combined', 'n_anchor_bottom': 0, 'n_anchor_top': 0}
      , {'roi': 'LO(R)', 'anchor_method': 'combined', 'n_anchor_bottom': 0, 'n_anchor_top': 37}
      , {'roi': 'ME(R)', 'anchor_method': 'separate', 'n_anchor_bottom': 800, 'n_anchor_top': 800}
      , {'roi': 'ME(R)', 'anchor_method': 'combined', 'n_anchor_bottom': 0, 'n_anchor_top': 0}
    ]

    # max number of columns - from database
    _, hex_ids, _, _ = load_pins(roi_str='ME(R)', ignore_cache=True)
    col_max_count = hex_ids.shape[0]

    for roi_pins_dict in roi_pins_dict_list:
        roi_str = roi_pins_dict['roi']

        #create columns: gives some output, i.e., if created columns are straight
        create_center_column_pins(
            roi_str=roi_str
          , anchor_method=roi_pins_dict['anchor_method']
          , n_anchor_bottom=roi_pins_dict['n_anchor_bottom']
          , n_anchor_top=roi_pins_dict['n_anchor_top']
          , verbose=True
        )

        #count number of initially created columns
        col_ids, _, _, _ = load_pins(roi_str=roi_str)
        col_count = col_ids.shape[0]
        print(f"Number of initial {roi_str[:-3]} columns: {col_ids.shape[0]}")

        #smoothen and fill-in columns
        ctr_smooth = 0
        while col_count < col_max_count:
            smooth_center_columns_w_median(roi_str=roi_str)
            col_ids, _, _, _ = load_pins(roi_str=roi_str)
            ctr_smooth += 1
            if col_ids.shape[0] == col_count:
                break
            col_count = col_ids.shape[0]

        print(f"Number of smoothing steps: {ctr_smooth}")
        print(f"Number of final {roi_str[:-3]} columns: {col_ids.shape[0]}")


# Needs to be at the end because of circular import
from utils.ROI_layers import create_ol_layer_boundaries, make_large_mesh
from utils.ROI_columns import create_center_column_pins, smooth_center_columns_w_median

