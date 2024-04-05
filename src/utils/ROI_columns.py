import numpy as np
import pandas as pd

import networkx as nx
import alphashape
import trimesh

from sklearn.metrics.pairwise import euclidean_distances
from scipy import interpolate

import navis.interfaces.neuprint as neu
from neuprint import NeuronCriteria as NC, SynapseCriteria as SC, fetch_synapses

from utils.hex_hex import get_hex_df
from utils.ROI_calculus import _get_data_path
from utils.align_mi1_t4 import create_alignment


def find_holes(pts_all, pts):
    """
    Find the indices of pts_all with holes. A "hole" is a point in pts_all that is not in pts.

    We only consider 3 types of holes:

        1. Holes (x,y) in which the 4 points (x+-1,y), (x,y+-1) are not holes,
        2. Holes (x,y) in which the 2 points (x+-1,y) are not holes but
            at least one of (x,y+-1) is,
        3. Holes (x,y) in which the 2 points (x,y+-1) are not holes but
            at least one of (x+-1,y) is.

    Parameters
    ----------
    pts_all : np.ndarray
        list of 2D integer lattice points
    pts : np.ndarray
        array of 2D integer lattice points that is a sublattice of pts_all

    Returns
    -------
    ids_vh : list[int]
        list of integers that contains the indices of type 1 holes
    ids_h : list[int]
        list of integers that contains the indices of type 2 holes
    ids_v : list[int]
        list of integers that contains the indices of type 3 holes
    """

    #find edge points
    hull = alphashape.alphashape(pts_all, 1)
    hull_pts = np.asarray(hull.exterior.coords)
    edge_ids = np.zeros(hull_pts.shape[0],dtype=int)
    for i in range(hull_pts.shape[0]):
        edge_ids[i] = np.argmin( np.linalg.norm(hull_pts[i][np.newaxis,:]-pts_all, axis=1) )

    #find holes that are not on boundary. dinstinguish if 4 neighbors, 2 horizontal or 2 vertical
    pts_all_df = pd.DataFrame(pts_all, columns=['x','y'])
    pts_df = pd.DataFrame(pts, columns=['x','y'])
    pts_df['sub'] = 1
    pts_all_df = pts_all_df.merge(pts_df, 'left', on=['x','y'])
    sub_ids = pts_all_df[pts_all_df['sub']==1].index.values

    missing_ids = list(set(pts_all_df.index.values)-set(sub_ids))
    pts_all_df['h_neighbors'] = pts_all_df.apply(
        lambda r: list(set(np.where(
            (np.abs(pts_all_df['x'] - r.x) == 1)
          & (np.abs(pts_all_df['y'] - r.y) == 0))[0]) - set(missing_ids))
      , axis=1)
    pts_all_df['v_neighbors'] = pts_all_df.apply(
        lambda r: list(set(np.where(
            (np.abs(pts_all_df['x'] - r.x) == 0)
          & (np.abs(pts_all_df['y'] - r.y) == 1))[0]) - set(missing_ids))
      , axis=1)
    pts_all_df['no_v_neighbors'] = pts_all_df.apply(
        lambda r: len(r['v_neighbors'])
      , axis=1
    )

    pts_all_df['no_h_neighbors'] = pts_all_df.apply(
        lambda r: len(r['h_neighbors'])
      , axis=1
    )

    ids_vh = pts_all_df.iloc[missing_ids][
            (pts_all_df.iloc[missing_ids]['no_v_neighbors'] == 2) &\
            (pts_all_df.iloc[missing_ids]['no_h_neighbors'] == 2)]\
        .index\
        .values
    ids_h = pts_all_df.iloc[missing_ids][
            (pts_all_df.iloc[missing_ids]['no_v_neighbors'] <  2) &\
            (pts_all_df.iloc[missing_ids]['no_h_neighbors'] == 2)]\
        .index\
        .values
    ids_v = pts_all_df.iloc[missing_ids][
            (pts_all_df.iloc[missing_ids]['no_v_neighbors'] == 2) &\
            (pts_all_df.iloc[missing_ids]['no_h_neighbors'] <  2)]\
        .index\
        .values

    return ids_vh, ids_h, ids_v


def smooth_center_columns_w_median(
    roi_str='ME(R)'
  , r_neighb=2
) -> None:
    """
    Override column pins by a median column (computed over a local neighborhood of radius r_neighb)
    Missing columns, which form a hole of types 1-3, are filled in.

    This function changes the pickle file roi_str[:-3]+'_col_center_pins.pickle'

    Parameters
    ----------
    roi_str : str, default='ME(R)'
        specifying roi, can only be ME(R), LO(R), LOP(R))
    r_neighb : int, default=2
        number of neighbors in either hex1_id or hex2_id direction
    """

    assert roi_str in ['ME(R)', 'LO(R)', 'LOP(R)'],\
            f"ROI must be one of 'ME(R)', 'LO(R)', 'LOP(R)', but is actually '{roi_str}'"

    data_path = _get_data_path(reason='cache')

    #load manually assigned hex ids
    column_df = load_hexed_body_ids(roi_str=roi_str)
    hex_df = column_df[['hex1_id','hex2_id']]\
        .drop_duplicates()\
        .sort_values(['hex1_id','hex2_id'])\
        .astype('float')\
        .reset_index(drop=True)
    n_col = hex_df.shape[0]

    #load created column centers
    col_all_df = pd.read_pickle(data_path / f'{roi_str[:-3]}_col_center_pins.pickle')

    col_ids0 = col_all_df[pd.isna(col_all_df.iloc[:,3])==0].index.values
    n_pins = int((col_all_df.shape[1]-3)/3)

    ids_vh, ids_h, ids_v = find_holes(hex_df.values, hex_df.values[col_ids0])
    col_ids = np.concatenate([col_ids0,ids_v,ids_h,ids_vh])
    col_ids = np.sort(col_ids)
    n_col = col_ids.shape[0]

    #define local neighborhood coordinates in hex ids for standard columns and for holes
    x_v, y_v = np.meshgrid(range(-r_neighb,r_neighb+1),range(-r_neighb,r_neighb+1))
    xyv_std = np.concatenate((x_v.reshape((-1,1)), y_v.reshape((-1,1))),axis=1)
    xyv_vh = np.array([[-1,0],[0,0],[1,0],[0,-1],[0,1]])
    xyv_v = np.array([[0,-1],[0,1]])
    xyv_h = np.array([[-1,0],[1,0]])

    col_names = np.array( 100 * hex_df['hex1_id'].values[col_ids]\
        + hex_df['hex2_id'].values[col_ids] )
    col_neighb_names_std = col_names[:, np.newaxis]\
        + 100 * xyv_std[:, 0][np.newaxis, :]\
        + xyv_std[:, 1][np.newaxis, :]
    col_neighb_names_vh = col_names[:, np.newaxis]\
        + 100 * xyv_vh[:,0][np.newaxis, :]\
        + xyv_vh[:, 1][np.newaxis, :]
    col_neighb_names_v = col_names[:, np.newaxis]\
        + 100 * xyv_v[:,0][np.newaxis, :]\
        + xyv_v[:, 1][np.newaxis, :]
    col_neighb_names_h = col_names[:, np.newaxis]\
        + 100 * xyv_h[:,0][np.newaxis, :]\
        + xyv_h[:, 1][np.newaxis, :]

    #go through all columns and find the median over appropriate neighbors
    pins_depth = col_all_df.iloc[col_ids].iloc[:,3:].values.reshape((n_col,-1,3))
    median_local = np.empty((n_col,n_pins,3))
    median_local[:] = np.nan
    for i in range(n_col):
        #collect all indices of existing neighboring columns
        idx = []
        col_neighb_names = np.array([])
        if pd.isna(pins_depth[i,0,0])==0:
            col_neighb_names = col_neighb_names_std[i]
        else:
            if np.isin(col_ids[i],ids_v):
                col_neighb_names = col_neighb_names_v[i]
            elif np.isin(col_ids[i],ids_h):
                col_neighb_names = col_neighb_names_h[i]
            elif np.isin(col_ids[i],ids_vh):
                col_neighb_names = col_neighb_names_vh[i]
        for j in range(col_neighb_names.shape[0]):
            id1 = np.where( col_neighb_names[j]==col_names )[0]
            if id1.shape[0]>0:
                if pd.isna(pins_depth[id1[0],0,0])==0:
                    idx.append( id1[0] )
        if len(idx)>0:
            # for standard columns, compute median relative to bottom
            #     then shift by bottom of standard column
            if pd.isna(pins_depth[i,0,0])==0:
                median_local[i] = np.median(
                        pins_depth[idx] - pins_depth[idx,-1][:,np.newaxis,:]
                      , axis=0)\
                  + pins_depth[i,-1][np.newaxis,np.newaxis,:]
            # for missing column, compute median across neighbors
            else:
                median_local[i] = np.median(pins_depth[idx], axis=0)

    #Define dataframes to store pins
    col_df3 = pd.DataFrame(
        np.concatenate(
            (col_ids[:, np.newaxis], median_local.reshape((n_col,-1)))
          , axis=1)
    )
    col_df3 = col_df3.rename(columns={0: 'col_id'})
    col_df3['col_id'] = col_df3['col_id'].astype(int)
    col_df3 = hex_df.reset_index(names='col_id').merge(col_df3, 'left', on='col_id')

    #ordering of columns should be hex1_id, hex2_id, n_syn
    col_list = list(col_df3)
    col_list[0], col_list[1], col_list[2] = col_list[1], col_list[2], col_list[0]
    col_df3 = col_df3.loc[:,col_list]
    col_df3.iloc[:,2] = col_all_df.iloc[:,2]
    col_df3.columns = col_all_df.columns
    col_df3['hex1_id'] = col_df3['hex1_id'].astype('Int64')
    col_df3['hex2_id'] = col_df3['hex2_id'].astype('Int64')

    col_df3.to_pickle(data_path / f"{roi_str[:-3]}_col_center_pins.pickle")


def create_center_column_pins(
    anchor_method:str
  , n_anchor_bottom:int
  , n_anchor_top:int
  , roi_str='ME(R)'
  , verbose=False
) -> None:
    """
    A pickle file is created (named `roi_str[:-3]+_col_center_pins.pickle` or
    `ME_col_center_pins_old.pickle`) of a Dataframe with columns `hex1_id`, `hex2_id` (not
    duplicated), `n_syn` and then the flattened xyz positions of the corresponding pin.

    Parameters
    ----------
    anchor_method : str
       can only be 'combined' or 'separate'
           'combined' uses PC 1 from all points specified by pc_from
           'separate' uses PC 1 separately for points (specified by pc_from) at the bottom and the
           top
    n_anchor_bottom : int
        if 0 then the bottom anchor is the intersection of PC 1 (as defined per anchor_method)
        with the neuropil ROI
        if >0 & anchor_method='combined', n_anchor_bottom specifies the number of bottom synapses,
        the median of which we use to place the bottom anchor point (along PC 1)
                anchor_method='separate', n_anchor_bottom specifies the number of bottom synapses
                that are used to define a separate PC 1
    n_anchor_top : int
        analogue of n_anchor_bottom but for top instead of bottom
    roi_str : str
        specifying roi, can only be ME(R), LO(R), LOP(R))
    verbose : bool
        print pin creation information

    Returns
    -------
    None
    """
    assert roi_str in ['ME(R)', 'LO(R)', 'LOP(R)'],\
        f"ROI must be one of 'ME(R)', 'LO(R)', 'LOP(R)', but is actually '{roi_str}'"
    assert anchor_method in ['combined', 'separate'],\
        f"pc_from must be one of 'combined', 'separate', but is actually '{anchor_method}'"
    if anchor_method=='separate':
        assert (n_anchor_bottom>0)&(n_anchor_top>0),\
            f"n_anchor_bottom and n_anchor_top should both be bigger than 0, but are actually '{n_anchor_bottom}' and '{n_anchor_top}'"

    data_path = _get_data_path(reason='cache')

    n_neighbors, n_segments, n_pins\
        , thre_std, valid_cols, pc_coord, pc_sign, pc_from\
        = load_roi_pin_params(roi_str=roi_str)

    #find all manually assigned hex ids
    column_df = load_hexed_body_ids(roi_str=roi_str)
    all_hex_ids = (
        column_df[['hex1_id','hex2_id']]\
            .drop_duplicates()\
            .sort_values(['hex1_id','hex2_id']))\
        .values
    n_col = all_hex_ids.shape[0]

    #Initialize array to store number of synapses per column
    n_syn = np.zeros((n_col,1))
    #Initialize array to store xyz positions of depth bins, using spline interpolation
    xyz_pins = np.empty((n_col,3*n_pins))
    xyz_pins[:] = np.nan

    ctr_id = 0
    ctr_syn = 0
    ctr_top = 0
    ctr_bottom = 0
    ctr_straight = 0
    #loop through columns
    for idx in range(n_col):
        # find all body_ids of valid_cols in column idx
        body_ids=[]
        column_hex_df = column_df[
            (column_df['hex1_id']==all_hex_ids[idx,0])\
          & (column_df['hex2_id']==all_hex_ids[idx,1])
        ]
        for j in range(column_hex_df.shape[0]):
            for i in valid_cols:
                if pd.isna(column_hex_df.iloc[j].loc[i])==0:
                    body_ids.append(column_hex_df.iloc[j].loc[i])
        body_ids = list(np.unique(body_ids).astype(int))
        if len(body_ids)==0:
            if verbose:
                print(f"Pin {idx} not created: not enough assigned neurons")
            ctr_id += 1
            continue

        syn_df = fetch_synapses(NC(bodyId=body_ids), SC(rois=roi_str))

        syn_trim_df = trim_syn_by_pc(
            syn_df
          , pc_coord
          , pc_sign
          , pc_from
          , thre_std=thre_std
        )
        if syn_trim_df.shape[0]<n_neighbors:
            if verbose:
                print(f"Pin {idx} not created: not enough trimmed synapses")
            ctr_syn += 1
            continue

        n_syn[idx] = syn_trim_df.shape[0]

        #find anchor points of pins (first and last points), and check if not too far from neighboring synapses
        anchor_bottom, anchor_top = find_anchors(
            syn_trim_df
          , pc_coord
          , pc_sign
          , pc_from
          , anchor_method
          , n_anchor_bottom
          , n_anchor_top
          , roi_str=roi_str
        )
        proj_rad = np.linalg.norm( anchor_top-anchor_bottom )/n_segments
        ordered_pts = syn_trim_df[['x','y','z']].values
        n_bdry = int(ordered_pts.shape[0]*0.05)
        if np.linalg.norm(anchor_top - ordered_pts[-n_bdry:].mean(0)) > proj_rad:
            if verbose:
                print(f"Pin {idx} not created: top anchor point too far from top synapses")
            ctr_top += 1
            continue
        elif np.linalg.norm(anchor_bottom - ordered_pts[:n_bdry].mean(0)) > proj_rad:
            if verbose:
                print(f"Pin {idx} not created: bottom anchor point too far from bottom synapses")
            ctr_bottom += 1
            continue

        #attach anchor points to ordered points, smoothen, and reattach anchor points
        ordered_pts = np.concatenate((anchor_bottom[np.newaxis,:], ordered_pts, anchor_top[np.newaxis,:]),axis=0)
        pts_smooth = find_neighbor_avg(ordered_pts, n_neighbors=n_neighbors)
        pts_smooth = np.concatenate((anchor_bottom[np.newaxis,:], pts_smooth, anchor_top[np.newaxis,:]),axis=0)

        #piecewise linear approximation of pts_smooth at length scale proj_rad (b_straight=1 if straight line; b_straight=0 otherwise)
        shortest_path_line, b_straight = find_shortest_path(pts_smooth, proj_rad)
        ctr_straight += b_straight

        pchip_uniform = find_uniform_interpolation(shortest_path_line, n_pins, mode='PCHIP')

        xyz_pins[idx] = np.squeeze(pchip_uniform.reshape((-1,1)))

    ctr_all = (~np.isnan(xyz_pins[:,0])).sum()
    if verbose:
        print(f"Created {ctr_all} pins in {roi_str}, of which {ctr_straight} are straight.")
        print(f"Missing pins because not enough neurons ({ctr_id}), not enough trimmed synapses ({ctr_syn}), or because the top ({ctr_top}) or bottom ({ctr_bottom}) anchor points are too far from neighboring synapses.")

    col_df = pd.DataFrame( np.concatenate((all_hex_ids, n_syn, xyz_pins), axis=1) )
    col_df = col_df.rename(columns={0: 'hex1_id', 1: 'hex2_id', 2: 'n_syn'})

    if roi_str=='ME(R)':
        if anchor_method=='separate':
            col_df.to_pickle(data_path / f'ME_col_center_pins.pickle')
        else:
            col_df.to_pickle(data_path / f'ME_col_center_pins_old.pickle')
    else:
        col_df.to_pickle(data_path / f'{roi_str[:-3]}_col_center_pins.pickle')


def load_hexed_body_ids(
    roi_str='ME(R)'
) -> pd.DataFrame:
    """
    Load dataframe with hex_ids and assigned body_ids

    Parameters
    ----------
    roi_str : str, default='ME(R)'
        specifying ROI, can only be ME(R), LO(R), or LOP(R)

    Returns
    -------
    column_df : pd.DataFrame
        rows are hex1_id, hex2_id (potentially duplicated) and columns are body_ids of
        assigned cell-types
    """

    assert roi_str in ['ME(R)', 'LO(R)', 'LOP(R)'],\
            f"ROI must be one of 'ME(R)', 'LO(R)', 'LOP(R)', but is actually '{roi_str}'"

    #load Kit's manual column assignment
    column_df = get_hex_df(neuropil='ME')

    #Mi1-T4 assignment based on synapse fractions
    if roi_str=='LOP(R)':
        data_path = _get_data_path(reason='data')
        alignment_file = data_path / 'mi1_t4_alignment.xlsx'
        if not alignment_file.is_file():
            create_alignment()
        column_df2 = pd.read_excel(alignment_file).convert_dtypes()
        column_df2 = column_df2[column_df2['valid_group']==1]
        column_df = column_df\
            .merge(
                column_df2.rename(
                    columns={
                        'mi1_bid': 'Mi1'
                      , 't4a_bid': 'T4a'
                      , 't4b_bid': 'T4b'
                      , 't4c_bid': 'T4c'
                      , 't4d_bid': 'T4d'}
                )
              , how='left'
              , on='Mi1')
        column_df = column_df.loc[:,['hex1_id','hex2_id','Mi1','T4a','T4b','T4c','T4d']]
        column_df.drop_duplicates(inplace=True)
    return column_df


def load_roi_pin_params(roi_str='ME(R)'):
    """
    Load parameters for pin creation.

    All parameters were chosen heuristically.

    Parameters
    ----------
    roi_str : str, default='ME(R)'
        specifying ROI, can only be ME(R), LO(R), LOP(R)

    Returns
    -------
    n_neighbors : int
        Number of neighbors to average over
    n_segments : int
        Number of segments for shortest path; should be bigger than 1
    n_pins : int
        Number of depth bins
    thre_std : float
        factor of lateral standard deviation to trim points (if too far)
    valid_cols : list[str]
        cell-types to include in column creation
    pc_coord : int
        specifies which coordinate, i.e., x=0, y=1, z=2, points along depth
    pc_sign : int
        specifies if a vector pointing along pc_coord goes from top to bottom (+1) or bottom to top (-1)
    pc_from : str
        Specifies if PCA is taken from synapses (pc_from='syn') or from the mean synapse position, for each bodyId (pc_from='COM')
    """
    assert roi_str in ['ME(R)', 'LO(R)', 'LOP(R)'],\
            f"ROI must be one of 'ME(R)', 'LO(R)', 'LOP(R)', but is actually '{roi_str}'"

    data_path = _get_data_path(reason='params')
    params_fn = data_path / "pin_creation_parameters.xlsx"
    params_df = pd.read_excel(params_fn).convert_dtypes()
    params_df = params_df[ params_df['neuropil']==roi_str[:-3] ]

    n_neighbors = int( params_df[ params_df['parameter']=='n_neighbors' ]['value'].values[0] )
    n_segments = int( params_df[ params_df['parameter']=='n_segments' ]['value'].values[0] )
    n_pins = int( params_df[ params_df['parameter']=='n_pins' ]['value'].values[0] )
    thre_std = float( params_df[ params_df['parameter']=='thre_std' ]['value'].values[0] )
    valid_cols = params_df[ params_df['parameter']=='valid_cols' ]['value'].values[0].split(", ")
    pc_coord = int( params_df[ params_df['parameter']=='pc_coord' ]['value'].values[0] )
    pc_sign = int( params_df[ params_df['parameter']=='pc_sign' ]['value'].values[0] )
    pc_from = str( params_df[ params_df['parameter']=='pc_from' ]['value'].values[0] )

    return n_neighbors\
        , n_segments\
        , n_pins\
        , thre_std\
        , valid_cols\
        , pc_coord\
        , pc_sign\
        , pc_from


def trim_syn_by_pc(
    syn_df:pd.DataFrame
  , pc_coord:int
  , pc_sign:int
  , pc_from='syn'
  , thre_std:float=1.0
):
    """
    Trim synapses to lie close to their PC1

    Parameters
    ----------
    syn_df : pd.DataFrame
        with 'x', 'y', 'z' columns
    pc_coord : int
        specifies which coordinate, i.e., x=0, y=1, z=2, points along depth
    pc_sign : int
        specifies if a vector pointing along pc_coord goes from bottom to top (+1) or top to bottom (-1)
    pc_from : str
        Specifies if PCA is taken from synapses (pc_from='syn') or from the mean synapse position, for each bodyId (pc_from='COM')
    thre_std : float, default=1.0
        max threshold of lateral std

    Returns
    -------
    syn_trim_df : pd.DataFrame
        with 'x', 'y', 'z' columns, after trimming, sorted bottom to top
    """
    assert pc_from in ['syn', 'COM'],\
        f"pc_from must be one of 'syn', 'COM', but is actually '{pc_from}'"

    #points to take PCA from
    if pc_from=='COM':
        pc_pts = syn_df.groupby('bodyId')[['x','y','z']].mean().values
    else:
        pc_pts = syn_df[['x','y','z']].values
    unitary, _, _ = np.linalg.svd(pc_pts.T-pc_pts.T.mean(1)[:,np.newaxis])

    #fix sign of PC1 along one direction (differs between neuropils): positive should be bottom to top
    if np.sign(unitary[pc_coord,0])==pc_sign:
        unitary[:,0] = -unitary[:,0]

    #compute distance in PC2-PC3 plane
    lateral = np.sqrt(
        ((syn_df[['x','y','z']].values-pc_pts.mean(0)[np.newaxis,:])@unitary[:,1])**2
      + ((syn_df[['x','y','z']].values-pc_pts.mean(0)[np.newaxis,:])@unitary[:,2])**2
    )
    #only take points within thre_std std in PC 2-3 plane
    lateral_max = thre_std*np.sqrt((lateral**2).mean())
    syn_trim_df = syn_df[lateral<lateral_max]

    #project onto PC1, and then sort to get bottom to top order
    proj = (syn_trim_df[['x','y','z']].values-pc_pts.mean(0)[np.newaxis,:])@unitary[:,0]
    isort = np.argsort(proj)
    syn_trim_df = syn_trim_df.iloc[isort]

    #redo PCA on trimmed points and resort
    if pc_from=='COM':
        pc_pts = syn_trim_df.groupby('bodyId')[['x','y','z']].mean().values
    else:
        pc_pts = syn_trim_df[['x','y','z']].values
    unitary, _, _ = np.linalg.svd(pc_pts.T-pc_pts.T.mean(1)[:,np.newaxis])
    if np.sign(unitary[pc_coord,0])==pc_sign:
        unitary[:,0] = -unitary[:,0]
    proj = (syn_trim_df[['x','y','z']].values-pc_pts.mean(0)[np.newaxis,:])@unitary[:,0]
    isort = np.argsort(proj)

    return syn_trim_df.iloc[isort]


def find_anchors(
    syn_df:pd.DataFrame
  , pc_coord:int
  , pc_sign:int
  , pc_from:str
  , anchor_method:str
  , n_anchor_bottom:int
  , n_anchor_top:int
  , roi_str='ME(R)'
):
    """
    Find anchor points, i.e., first and last pin points

    Parameters
    ----------
    syn_df : pd.DataFrame
        with 'x', 'y', 'z' columns
    pc_coord : int
        specifies which coordinate, i.e., x=0, y=1, z=2, points along depth
    pc_sign : int
        specifies if a vector pointing along pc_coord goes from bottom to top (+1) or top to bottom (-1)
    pc_from : str
        specifies is PCA is taken from synapses (pc_from='syn') or from the mean synapse position, for each bodyId (pc_from='COM')
    anchor_method : str
       can only be 'combined' or 'separate'
           'combined' uses PC 1 from all points specified by pc_from
           'separate' uses PC 1 separately for points (specified by pc_from) at the bottom and the top
    n_anchor_bottom : int
        if 0 then the bottom anchor is the intersection of PC 1 (as defined per anchor_method) with the neuropil ROI
        if >0 & anchor_method='combined', n_anchor_bottom specifies the number of bottom synapses, the median of which we use
                    to place the bottom anchor point (along PC 1)
                anchor_method='separate', n_anchor_bottom specifies the number of bottom synapses that are used to define a separate PC 1
    n_anchor_top : int
        analogue of n_anchor_bottom but for top instead of bottom
    roi_str : str, default='ME(R)'
        specifying ROI, can only be ME(R), LO(R), LOP(R)

    Returns
    -------
    pc1_bottom : np.ndarray (float) of length 3
        3D point where PC1 intersects neuprint ROI at the bottom
    pc1_top : np.ndarray (float) of length 3
        3D point where PC1 intersects neuprint ROI at the top
    """
    assert pc_from in ['syn', 'COM'],\
        f"pc_from must be one of 'syn', 'COM', but is actually '{pc_from}'"
    assert anchor_method in ['combined', 'separate'],\
        f"pc_from must be one of 'combined', 'separate', but is actually '{anchor_method}'"
    if anchor_method=='separate':
        assert (n_anchor_bottom>0)&(n_anchor_top>0),\
            f"n_anchor_bottom and n_anchor_top should both be bigger than 0, but are actually '{n_anchor_bottom}' and '{n_anchor_top}'"
    assert roi_str in ['ME(R)', 'LO(R)', 'LOP(R)'],\
        f"ROI must be one of 'ME(R)', 'LO(R)', 'LOP(R)', but is actually '{roi_str}'"

    #PCA on points specified by pc_from
    if pc_from=='COM':
        pc_pts = syn_df.groupby('bodyId')[['x','y','z']].mean().values
    else:
        pc_pts = syn_df[['x','y','z']].values
    unitary, _, _ = np.linalg.svd(pc_pts.T-pc_pts.T.mean(1)[:,np.newaxis])
    if np.sign(unitary[pc_coord,0])==pc_sign:
        unitary[:,0] = -unitary[:,0]

    #sort to get bottom to top order
    proj = (syn_df[['x','y','z']].values-pc_pts.mean(0)[np.newaxis,:])@unitary[:,0]
    isort = np.argsort(proj)
    ordered_pts = syn_df[['x','y','z']].values[isort]

    #load neuropil ROI, used to determine anchor points
    roi_vol = neu.fetch_roi(roi_str)

    if anchor_method=='combined':
        #find intersection of PC1 with neuprint roi
        pc1_line = 10 * (proj.max()-proj.min())\
            * np.linspace(-1,1,1000)[:,np.newaxis]\
            * unitary[:,0]\
            + pc_pts.T.mean(1)
        signed_dists = trimesh.proximity.signed_distance(roi_vol,pc1_line)
        pc1_line = pc1_line[signed_dists>0]
        anchor_bottom = pc1_line[0]
        anchor_top = pc1_line[-1]

        #override pc1_bottom with median bottom synapses projected onto PC 1
        if n_anchor_bottom>0:
            proj_bottom = (np.median(ordered_pts[:n_anchor_bottom],axis=0)-pc_pts.mean(0))@unitary[:,0]
            anchor_bottom = proj_bottom*unitary[:,0]+pc_pts.mean(0).T

        #override pc1_top with median top synapses projected onto PC 1
        if n_anchor_top>0:
            proj_top = (np.median(ordered_pts[-n_anchor_top:],axis=0)-pc_pts.mean(0))@unitary[:,0]
            anchor_top = proj_top*unitary[:,0]+pc_pts.mean(0).T

    if anchor_method=='separate':
        #normalize proj to -1 and 1
        proj = (proj - (proj.max()+proj.min())/2)/(proj.max()-proj.min())*2

        #points to use for bottom PCA
         #the upper threshold 0.1 was chosen manually
         #it is >0 because in ME and LO there are gaps in the synapses for the more bottom layers
        pc_pts_bottom = pc_pts[proj<.1]
        if pc_pts_bottom.shape[0]<n_anchor_bottom:
            pc_pts_bottom = pc_pts[:n_anchor_bottom]

        #points to use for top PCA
         #the lower threshold 0.2 was chosen manually
         #it was chosen to be bigger than the upper threshold used for bottom
        pc_pts_top = pc_pts[proj>.2]
        if pc_pts_top.shape[0]<n_anchor_top:
            pc_pts_top = pc_pts[-n_anchor_top:]

        #compute bottom PCA
        unitary_bottom, _, _ = np.linalg.svd(pc_pts_bottom.T-pc_pts_bottom.T.mean(1)[:,np.newaxis])
        if np.sign(unitary_bottom[pc_coord,0])==pc_sign:
            unitary_bottom[:,0] = -unitary_bottom[:,0]
        proj_bottom = (ordered_pts-pc_pts_bottom.mean(0)[np.newaxis,:])@unitary_bottom[:,0]

        #compute top PCA
        unitary_top, _, _ = np.linalg.svd(pc_pts_top.T-pc_pts_top.T.mean(1)[:,np.newaxis])
        if np.sign(unitary_top[pc_coord,0])==pc_sign:
            unitary_top[:,0] = -unitary_top[:,0]
        proj_top = (ordered_pts-pc_pts_top.mean(0)[np.newaxis,:])@unitary_top[:,0]

        #find intersection of bottom PC1 with neuprint roi
        pc1_bottom_line = 10*(-proj_bottom.min())*np.linspace(-1,0,1000)[:,np.newaxis]\
            * unitary_bottom[:,0] + pc_pts_bottom.T.mean(1)
        signed_dists_bottom = trimesh.proximity.signed_distance(roi_vol,pc1_bottom_line)
        anchor_bottom = pc1_bottom_line[signed_dists_bottom>0][0]

        #find intersection of top PC1 with neuprint roi
        pc1_top_line = 10*proj_top.max()*np.linspace(0,1,1000)[:,np.newaxis]\
            * unitary_top[:,0] + pc_pts_top.T.mean(1)
        signed_dists_top = trimesh.proximity.signed_distance(roi_vol,pc1_top_line)
        anchor_top = pc1_top_line[signed_dists_top>0][-1]

    return anchor_bottom, anchor_top


def find_neighbor_avg(
    pts_sorted
  , n_neighbors=100
):
    """
    average over n_neighbors (neighbors are defined by the ordering of pts_sorted)

    Parameter
    ---------
    pts_sorted : np.ndarray (float)
        array with `N` number of 3D points ordered by their projection onto PC1
    n_neighbors : int, default=100
        number of neighbors to smooth over

    Returns
    -------
    pts_smooth : np.ndarray (float)
        array with `N` number of 3d points: `pts_sorted` smoothened over <= `n_neighbors`
    """

    pts_smooth = np.zeros((pts_sorted.shape[0]-n_neighbors+1,3))
    for j in range(3):
        pts_smooth[:,j] = np.convolve(
            pts_sorted[:,j]
          , np.ones(n_neighbors)/n_neighbors
          , mode='valid')
    return pts_smooth


def find_shortest_path(pts_smooth, proj_rad):
    """
    compute shortest path from pts_smooth[0] to pts_smooth[-1]

    Parameters
    ----------
    pts_smooth : list
        (array of size Nx3)
    proj_rad : float
        max distance to "hop"

    Returns
    -------
    shortest_path_line : list
        Kx3 list of K points that are the edge points for the shortest path
    b_straight : bool
        0 if column created is bent; 1 if it is straight
    """

    b_straight = True
    if proj_rad>=np.linalg.norm(pts_smooth[0]-pts_smooth[-1]):
        shortest_path_line = np.zeros((2,3))
        shortest_path_line[0] = pts_smooth[0]
        shortest_path_line[1] = pts_smooth[-1]
    else:
        try:
            b_straight = False
            distances = euclidean_distances(pts_smooth)
            distances[distances>proj_rad]=0
            graph = nx.from_numpy_array(distances, create_using=nx.DiGraph)
            path = nx.shortest_path(
                graph
              , source=0
              , target=distances.shape[0]-1
              , weight='weight'
            )
            shortest_path_line = pts_smooth[path]
        except nx.NetworkXNoPath:
            b_straight = True
            shortest_path_line = np.zeros((2,3))
            shortest_path_line[0] = pts_smooth[0]
            shortest_path_line[1] = pts_smooth[-1]

    return shortest_path_line, b_straight


def find_shortest_path_dist(shortest_path_line):

    """
    compute distances of neighboring line segments in path

    Parameters
    ----------
    shortest_path_line : np.ndarray (float)
        array with `K` number of 3D points that lie on a curve.

    Returns
    -------
    shortest_dists : np.ndarray (float)
        distance between neighboring points in `shortest_path_line` (first entry is 0)
    """

    shortest_dists = np.zeros(shortest_path_line.shape[0])
    for i in range(shortest_dists.shape[0]-1):
        shortest_dists[i+1] = np.linalg.norm(shortest_path_line[i+1]-shortest_path_line[i])

    return shortest_dists


def find_uniform_interpolation(
    shortest_path_line
  , n_pts:float
  , mode:str='PCHIP'
) -> np.ndarray[list[float]]:
    """
    interpolate shortest_path_line by n_pts uniform points (uniform in the sense of path length)

    Parameters
    ----------
    shortest_path_line : np.ndarray (float)
        array with `K` number of 3D points that lie on a curve
    n_pts : float
        number of interpolation points
    mode : str
        can only be 'cubic', 'linear', or 'PCHIP'

    Returns
    -------
    interp_uniform : np.ndarray[list[float]]
        interpolation of shortest_path_line, array with `n_pts` 3D points
    """

    assert mode in ['cubic', 'linear', 'PCHIP'],\
            f"Mode must be one of 'cubic', 'linear', 'PCHIP', but is actually '{mode}'"

    #find indivudal path lengths and their cumulative sum
    shortest_dists = find_shortest_path_dist(shortest_path_line)
    shortest_dists_cumsum = np.cumsum(shortest_dists)
    #uniformly sample path length using n_pts number of points
    path_uniform = np.linspace(0,shortest_dists_cumsum[-1],n_pts)
    #fit interpolation for each coordinate
    if mode=='cubic':
        f_x = interpolate.interp1d(shortest_dists_cumsum, shortest_path_line[:,0], kind='cubic')
        f_y = interpolate.interp1d(shortest_dists_cumsum, shortest_path_line[:,1], kind='cubic')
        f_z = interpolate.interp1d(shortest_dists_cumsum, shortest_path_line[:,2], kind='cubic')
    elif mode=='linear':
        f_x = interpolate.interp1d(shortest_dists_cumsum, shortest_path_line[:,0], kind='linear')
        f_y = interpolate.interp1d(shortest_dists_cumsum, shortest_path_line[:,1], kind='linear')
        f_z = interpolate.interp1d(shortest_dists_cumsum, shortest_path_line[:,2], kind='linear')
    elif mode=='PCHIP':
        f_x = interpolate.PchipInterpolator(shortest_dists_cumsum, shortest_path_line[:,0])
        f_y = interpolate.PchipInterpolator(shortest_dists_cumsum, shortest_path_line[:,1])
        f_z = interpolate.PchipInterpolator(shortest_dists_cumsum, shortest_path_line[:,2])
    #do interpolation for each coordinate
    x_pred = f_x(path_uniform)
    y_pred = f_y(path_uniform)
    z_pred = f_z(path_uniform)
    interp_uniform = np.vstack([x_pred, y_pred, z_pred]).T

    return interp_uniform
