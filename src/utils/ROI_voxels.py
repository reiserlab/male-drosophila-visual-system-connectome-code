import gzip
import os

import numpy as np
import pandas as pd

import requests

from utils.ROI_calculus import find_hex_ids, find_col_names, find_mesh_layers, _get_data_path


def region_boxes(vol):
    """
    [Code from Stuart Berg]
    Determines the bounding boxes of all label regions (segments) in a label volume.

    Since the result is indexed by segment ID, this function is only
    suitable for volumes in which the maximum label ID is relatively low.
    For instance, if the volume contains labels [1,2,3, int(1e9)],
    then the result will have length 1e9.
    
    If the input array contains non-consecutive label IDs,
    (i.e., some labels are not present), then the results in those
    entries will be intentionally nonsensical: the 'min' will
    be GREATER than the 'max'.

    Parameters
    ----------
    vol : np.ndarray
        voxel volume with consecutive integer labels (starting from 0)

    Returns
    -------
    boxes : ndarray
        array of shape (N, 2, 3) where N is the number of unique label values in vol (including label 0).
        The min box coordinate for label i is given in entry [i, 0, :] and the max in entry [i, 1, :].
    """
    
    z_vol, y_vol, x_vol = vol.shape
    grid = np.ogrid[:z_vol, :y_vol, :x_vol]
    
    # Initialize box min (and max) coords with extreme max (and min)
    # values so that any encountered coordinate overrides the initial value.
    boxes = np.zeros((vol.max()+1, 2, 3), dtype=int)
    boxes[:, 0, :] = vol.shape
    
    for axis in (0,1,2):
        np.minimum.at(boxes[:, 0, axis], vol, grid[axis])
        np.maximum.at(boxes[:, 1, axis], vol, grid[axis])
        
    boxes[:, 1, :] += 1
    return boxes


def fetch_brain_voxels(
) -> None:
    """
    [Modified code from Stuart Berg]
    Download optic lobe voxels from google cloud, store as gzip, and create bounding box file roi-bounding-boxes-nm.csv

    Parameters
    ----------
    None
    
    Returns
    -------
    None
    """
    
    data_path = _get_data_path(reason='cache')
    brain_url = os.environ['BRAIN_VOLUME_URL']
    brain_f = brain_url.split('/')[-1]    
    brain_file_npy = data_path / brain_f
    brain_file_gz = data_path / (brain_f+'.gz')

    #download and store as gzip
    r = requests.get(brain_url, allow_redirects=True)
    with open(brain_file_npy, 'wb') as f:
        f.write(r.content)    
    with open(brain_file_npy, 'rb') as f_in, gzip.open(brain_file_gz, 'wb') as f_out:
        f_out.writelines(f_in)
    os.remove(brain_file_npy)
    
    # load voxels and compute the bounding boxes for each compartment and convert to nm units
    with gzip.GzipFile(brain_file_gz, "rb") as f: 
        vol_nm = np.load(f)
    boxes = region_boxes(vol_nm)
    boxes_nm = boxes * 512
    z0, y0, x0, z1, y1, x1 = boxes_nm[1:].reshape((-1, 6)).T

    # ROIs are numbered in alphabetical (ascii) order.
    rois = [
        "AL(L)", "AL(R)", "AME(L)", "AME(R)", "AMMC(L)", "AMMC(R)",
        "AOTU(L)", "AOTU(R)", "ATL(L)", "ATL(R)", "AVLP(L)", "AVLP(R)",
        "BU(L)", "BU(R)", "CA(L)", "CA(R)", "CAN(L)", "CAN(R)",
        "CRE(L)", "CRE(R)", "EB", "EPA(L)", "EPA(R)", "FB",
        "FLA(L)", "FLA(R)", "GA(L)", "GA(R)", "GNG", "GOR(L)", "GOR(R)",
        "IB", "ICL(L)", "ICL(R)", "IPS(L)", "IPS(R)", "LA(L)", "LA(R)",
        "LAL(L)", "LAL(R)", "LH(L)", "LH(R)", "LO(L)", "LO(R)",
        "LOP(L)", "LOP(R)", "ME(L)", "ME(R)", "NO", "PB", "PED(L)", "PED(R)",
        "PLP(L)", "PLP(R)", "PRW", "PVLP(L)", "PVLP(R)", "ROB(L)", "ROB(R)",
        "RUB(L)", "RUB(R)", "SAD", "SCL(L)", "SCL(R)", "SIP(L)", "SIP(R)",
        "SLP(L)", "SLP(R)", "SMP(L)", "SMP(R)", "SPS(L)", "SPS(R)",
        "VES(L)", "VES(R)", "WED(L)", "WED(R)", "a'L(L)", "a'L(R)",
        "aL(L)", "aL(R)", "b'L(L)", "b'L(R)", "bL(L)", "bL(R)",
        "gL(L)", "gL(R)"
    ]
    bbox_df = pd.DataFrame({
        'roi': rois,
        'x0_nm': x0, 'y0_nm': y0, 'z0_nm': z0,
        'x1_nm': x1, 'y1_nm': y1, 'z1_nm': z1
    }).set_index('roi')
    bbox_df.to_csv(data_path / "roi-bounding-boxes-nm.csv", index=True, header=True)


def voxelize_col_and_lay(
    rois:list[str]=None
  , columns=True
  , layers=True
) -> None:
    """
    Create npy and gzip with neuroglancer voxel volume of column and/or layer rois for all optic lobe neuropils (currently ME/LO/LOP).

    The files (named, e.g. 'ME_ZYX_columns' and 'ME_ZYX_layers') save a 3D array (zyx notation) with integers
      indicating the columns and layers (format for columns: hex1_id*100+hex2_id, and layers: 1 to number of layers)

    Parameters
    ----------
    rois : list[str], default=None
        If `rois` is None, it uses ['ME(R)', 'LO(R)', 'LOP(R)'].
    columns : bool
        create column voxels
    layers : bool
        create layer voxels

    Returns
    -------
    None
    """

    #voxelize at 512nm resolution
    samp = 2**6   

    #load brain voxels and bounding box
    data_path = _get_data_path(reason='cache')
    brain_url = os.environ['BRAIN_VOLUME_URL']
    brain_f = brain_url.split('/')[-1]    
    brain_file_gz = data_path / (brain_f+'.gz')
    bb_fn = data_path / "roi-bounding-boxes-nm.csv"
    
    if (not brain_file_gz.is_file()) or (not bb_fn.is_file()):
        fetch_brain_voxels()        
    with gzip.GzipFile(brain_file_gz, "rb") as fh_roi:
        vol_512nm = np.load(fh_roi)
    bbox_df = pd.read_csv( data_path/bb_fn ).set_index('roi')
    
    if rois is None:
        rois = ['ME(R)', 'LO(R)', 'LOP(R)']        
    for roi_str in rois:
        assert roi_str in ['ME(R)', 'LO(R)', 'LOP(R)'],\
                f"ROI must be a list with 'ME(R)', 'LO(R)', 'LOP(R)', but is actually '{roi_str}'"

        #create cube for roi_str
        roi_box_zyx_nm = bbox_df.loc[roi_str, ['z0_nm', 'y0_nm', 'x0_nm', 'z1_nm', 'y1_nm', 'x1_nm']].values
        z, y, x, Z, Y, X = roi_box_zyx_nm // (8*samp)
        roi_vol_512nm = vol_512nm[:Z, :Y, :X]
        zv, yv, xv = np.meshgrid(np.arange(0, Z*samp, samp),\
                                 np.arange(0, Y*samp, samp), \
                                 np.arange(0, X*samp, samp), indexing='ij')
        roi_xyz = np.concatenate((xv.reshape((-1,1)), yv.reshape((-1,1)), zv.reshape((-1,1))),axis=1)
        roi_df = pd.DataFrame( roi_xyz, columns=['x','y','z'] )

        #Find neuprint index corresponding to roi_str
        if roi_str=='ME(R)':
            neuprint_id=48
        elif roi_str=='LO(R)':
            neuprint_id=44
        elif roi_str=='LOP(R)':
            neuprint_id=46
    
        #chop cube to roi_str
        ind_roi = np.where( roi_vol_512nm.reshape((-1,1))==neuprint_id )[0]    
        roi_chop_df = roi_df.iloc[ind_roi]

        #assign layer to voxels
        #initialize voxel assignments
        if columns:
            zyx_col = np.zeros(roi_vol_512nm.shape,dtype=np.uint16).reshape((-1,1))
            zyx_col = np.squeeze(zyx_col)
            hex_df = find_hex_ids(roi_chop_df, roi_str=roi_str)
            col_df = find_col_names(hex_df)
            zyx_col[ind_roi] = col_df['col_name'].values 
            zyx_col = zyx_col.reshape((xv.shape[0],xv.shape[1],xv.shape[2]))
            column_gz_f = data_path / f"{roi_str[:-3]}_ZYX_columns.npy.gz"
            with gzip.GzipFile(column_gz_f, 'wb') as gz_fh:
                np.save(gz_fh, zyx_col)
    
        if layers:
            zyx_lay = np.zeros(roi_vol_512nm.shape,dtype=np.uint16).reshape((-1,1))
            zyx_lay = np.squeeze(zyx_lay)
            layer_df = find_mesh_layers(roi_chop_df, roi_str=roi_str)
            zyx_lay[ind_roi] = layer_df['layer'].values.astype(int)            
            zyx_lay = zyx_lay.reshape((xv.shape[0],xv.shape[1],xv.shape[2]))
            layer_gz_f = data_path / f"{roi_str[:-3]}_ZYX_layers.npy.gz"
            with gzip.GzipFile(layer_gz_f, 'wb') as gz_fh:
                np.save(gz_fh, zyx_lay)
            
