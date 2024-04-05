"""
Set of tools developed by Ed for matching layers in Medulla and Lobula.
"""
from pathlib import Path
import re

import plotly.graph_objects as go
import pandas as pd
from cmap import Colormap

from neuprint import NeuronCriteria as NC, SynapseCriteria as SC, fetch_synapse_connections

from utils.hex_hex import bid_to_hex

def fetch_neuron_pairs(
    cell_type_1:str, cell_type_2:str
  , group_neuron:str="bodyId_post", coord_suffix:str="_med"
) -> (pd.DataFrame, pd.DataFrame):
    """
    Fetch synapses for pairs of neurons

    Parameters
    ----------
    cell_type_1 : str
        name of a cell type that forms synapses with cell_type_2
    cell_type_2 : str
        name of a cell type that forms synapses with cell_type_1
    group_neuron : str, default = 'bodyId_post'
        name of the column that is used for grouping. For Medulla and if we are
        interested in the downstream connection, this is 'bodyId_post' (the default),
        for Lobula where the upstream connection is relevant, this should be 'bodyId_pre'.
    coord_suffix : str, default = '_med'
        suffix added to the x/y/z coordinates. For Medulla this is '_med' (the default)
        resulting in ['x_med', 'y_med', 'z_med'], for lobula it should be '_lob'.

    Returns
    -------
    ct1_ct2 : pandas.DataFrame
        Neuropil dataset with primary bodyId and coordinates.
    conns : pandas.DataFrame
        the results from neuprint.fetch_synapse_connections
    """
    conns = fetch_synapse_connections(
          NC(type=cell_type_1, regex=True)
        , NC(type=cell_type_2, regex=True))
    ct1_ct2 = conns\
        .assign(
            x_syn = lambda r: (r.x_pre + r.x_post)/2,
            y_syn = lambda r: (r.y_pre + r.y_post)/2,
            z_syn = lambda r: (r.z_pre + r.z_post)/2)\
        .groupby(by=[group_neuron])[['x_syn', 'y_syn', 'z_syn']]\
        .agg('mean')
    ct1_ct2.reset_index(inplace=True)
    ct1_ct2.rename(
        columns={
            group_neuron:"bodyId",
            "x_syn":f"x{coord_suffix}", 'y_syn':f"y{coord_suffix}", 'z_syn':f"z{coord_suffix}"}
        , inplace=True)
    return ct1_ct2, conns


def merge_and_color(
    data_frames:list[pd.DataFrame]
  , color_by_suffix:str='_med') -> pd.DataFrame:
    """
    merge data sets on `bodyId` column so that mean x,y,z for medulla pair and lobula pair are in
      rows by bodyID. This is a consolidation of the previous functions
      `med_lob()` and `M_layers()`

    Parameters
    ----------
    data_frames : list[pd.DataFrame]
        a list of (at least one) data frame with ['x_med', 'y_med', 'z_med'] of synapses between
        populations of neurons
    color_by_suffix : str, default = '_med'
        coordinate suffix to use for coloring

    Returns
    -------
    med_lob_pairs : pd.DataFrame
        merge between the two data sets. It also contains additional columns that will be used 
        for coloring.
    """
    assert isinstance(data_frames[0], pd.DataFrame), "Need at least one data frame to merge"
    res_df = data_frames[0]
    for tmp_df in data_frames[1:]:
        res_df = res_df.merge(tmp_df, on='bodyId')
    res_df.loc[:,'y_10000'] = res_df.loc[:,f'y{color_by_suffix}']>10000
    res_df.loc[:,'y_20000'] = res_df.loc[:,f'y{color_by_suffix}']>20000
    res_df.loc[:,'y_30000'] = res_df.loc[:,f'y{color_by_suffix}']>30000
    res_df.loc[:,'y_40000'] = res_df.loc[:,f'y{color_by_suffix}']>40000
    res_df.loc[:,'y_50000'] = res_df.loc[:,f'y{color_by_suffix}']>50000
    res_df.loc[:,'x_16000'] = res_df.loc[:,f'x{color_by_suffix}']>16000
    res_df.loc[:,'regions'] = res_df[
        ['y_10000','y_20000', 'y_30000','y_40000','y_50000']]\
        .sum(axis=1)+(5*res_df[['x_16000']]\
        .sum(axis=1))
    return res_df

def hexify_med_lob(ml_df:pd.DataFrame) -> pd.DataFrame:
    """
    add hex coordinates to data frames
     
    adds two extra columns ['hex1_id', 'hex2_id'] for hex coordinates 
    and ['mod_for_color', 'mod_for_color'] for colors used in plotting 
    to a medulla/lobula data set

    Parameters
    ----------
    pickle_name : str
        file name for a pickle file in `results/eyemap` containing hex IDs for bodyIds in 
          `merge_and_color()`
    ml_df : pandas.DataFrame
        Medulla / Lobula data frame

    Returns
    -------
    med_lob_hex : pandas.DataFrame
        medulla / lobula data frame with additional columns
    """
    med_lob_hex = ml_df.copy()
    med_lob_hex.loc[:,'HEXIDS'] = med_lob_hex["bodyId"].apply(bid_to_hex)
    for i, _ in med_lob_hex.iterrows():
        med_lob_hex.loc[i,'hex1_id'], med_lob_hex.loc[i,'hex2_id'] = med_lob_hex.loc[i,'HEXIDS']
    del med_lob_hex['HEXIDS']
    med_lob_hex['mod_for_color'] = med_lob_hex.loc[:,'hex1_id'] % 3
    med_lob_hex['mod_for_color2'] = med_lob_hex.loc[:,'hex2_id'] % 3
    return med_lob_hex


def plot_med_lob(
    ml_df:pd.DataFrame, color_column:str
  , figure_title:str
  , show_medulla:bool=True, show_lobula:bool=True)\
    -> go.Figure:
    """
    Return plot of medulla and lobula for further manipulation or plotting.

    Parameters
    ----------
    ml_df : pandas.DataFrame
        data frame with coordinates for medulla ['x_med', 'y_med', 'z_med']
        and / or lobula ['x_lob', 'y_lob', 'z_lob'] as well as a column for
        a shared color between medulla and lobula markers.
    color_column : str
        name of the column to signify color
    figure_title : str
        the name of the figure
    show_medulla : bool, default = True
        Specify if data from the medulla should be plotted
    show_lobula : bool, default = True
        Specify if data from the lobula should be plotted
    
    Returns
    -------
    fig : graph_object.Figure
        Plotly figure containing the specified layers.
    """
    fig = go.Figure()
    if show_medulla:
        roi_suffix = re.compile(r"^(x_((ME\d*)|(med)))$")
        for suffix_name in filter(roi_suffix.match, ml_df.columns):
            suffix = roi_suffix.search(suffix_name)[2]
            fig.add_trace(
                go.Scatter3d(
                    name=suffix,
                    x=ml_df[f'x_{suffix}'], y=ml_df[f'y_{suffix}'], z=ml_df[f'z_{suffix}'],
                    hovertext=ml_df['bodyId'], hovertemplate="bodyId: %{hovertext}",
                    mode="markers",
                    marker = {"color":ml_df[color_column], "size":6}))
    if show_lobula:
        roi_suffix = re.compile(r"^(x_((LO\d*)|(lob)))$")
        for suffix_name in filter(roi_suffix.match, ml_df.columns):
            suffix = roi_suffix.search(suffix_name)[2]
            fig.add_trace(
                go.Scatter3d(
                    name=suffix,
                    x=ml_df[f'x_{suffix}'], y=ml_df[f'y_{suffix}'], z=ml_df[f'z_{suffix}'],
                    hovertext=ml_df['bodyId'], hovertemplate="bodyId: %{hovertext}",
                    mode="markers",
                    marker = {"color":ml_df[color_column], "size":6}))

    fig.update_layout(
        title=f"{figure_title}  n={len(ml_df)}", title_font_size=24,
        width=1200, height=800)
    return fig

def plot_layers(
    layers:dict
  , colormap_name:str="crameri:hawaii") -> go.Figure:
    """
    Plot multiple layers

    This function takes a dictionary with lobyla layer names and a data frame describing
    that layer and plots it using a discrete color scheme. It returns a figure for further
    manipulation and plotting.

    Parameters
    ----------
    layers : dict
        Dictionary of the layers to be plotted. The key doubles as a label for
        the plot, the value is a pandas.DataFrame with lobula coordinates, which
        means having columns ['x_lob', 'y_lob', 'z_lob'].
    colormap_name : str, default="crameri:hawaii"
        Name of the cmap colormap used for the plot. Will stretch across the whole range of colors.

    Returns
    -------
    fig : graph_object.Figure
        Plotly figure containing the specified layers
    """
    fig = go.Figure()
    cmp = Colormap(colormap_name)
    colors = cmp.to_altair(len(layers))
    for idx, (layer_name, layer) in enumerate(layers.items()):
        fig.add_trace(go.Scatter3d(
            name=f"{layer_name} ({len(layer)})",
            x=layer['x_lob'], y=layer['y_lob'], z=layer['z_lob'],
            hovertext=layer['bodyId'], hovertemplate="bodyId: %{hovertext}",
            mode="markers", marker={"color":colors[idx], "size":3}))
    fig.update_layout(title="LOBULA LAYERS", width=1200, height=800)
    return fig


def get_com_and_hex(
    src_type:str
  , target_type:str
  , pickle_path:Path
  , src_roi:str="ME(R)"
  , synapse_threshold:float=0.8
) -> pd.DataFrame:
    """
Assign hex ids for a layer definition.

This is code extracted from hoellerj's `from_connection_to_grids`

Parameters
----------
src_type : str
    type of the source neurons for a connection
target_type : str
    type of the target neurons for a connection
pickle_path : pathlib.Path
    path where a pickle file with the name `src_type.pickle` is located (e.g. T4a.pickle).
src_roi : str, default="ME(R)"
    ROI, a search criteria for the synapses
synapse_threshold : float, default=0.8
    the minimal confidence for the synapse

Returns
-------
merge_df : pd.DataFrame
    contains the bodyId of the src_type, x, y, z coordinates of the center of mass for their 
      synapse, and the hex1_id and hex2_id
    """
    col_fn = pickle_path / f"{src_type}.pickle"
    col_df = pd.read_pickle(col_fn)\
        .loc[:, ['hex1_id', 'hex2_id']]\
        .astype({'hex1_id':'Int64', 'hex2_id':'Int64'}) # make it nullable for missing data
    conns = fetch_synapse_connections(
          NC(type=src_type, regex=True)
        , NC(type=target_type, regex=True)
        , SC(rois=src_roi, confidence=synapse_threshold))
    com_df = conns\
        .assign(
            x_syn = lambda r: (r.x_pre + r.x_post)/2,
            y_syn = lambda r: (r.y_pre + r.y_post)/2,
            z_syn = lambda r: (r.z_pre + r.z_post)/2)\
        .groupby(by=['bodyId_pre'])[['x_syn', 'y_syn', 'z_syn']]\
        .agg('mean')
    #com_df.reset_index(inplace=True)
    merge_df = com_df.join(col_df, how='left')
    merge_df = merge_df\
        .reset_index()\
        .rename(columns={
            'bodyId_pre': 'bodyId'
          , 'x_syn':'x', 'y_syn':'y', 'z_syn':'z'})\
        .sort_values(['hex1_id', 'hex2_id'], ignore_index=True)
    return merge_df
