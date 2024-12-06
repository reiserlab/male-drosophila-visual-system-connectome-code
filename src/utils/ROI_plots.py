from pathlib import Path
import gzip

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from cmap import Colormap
from dotenv import find_dotenv

from neuprint import NeuronCriteria as NC, SynapseCriteria as SC, fetch_synapses
from utils.ROI_calculus import \
    load_pins, find_hex_ids, find_straight_hex_ids\
  , load_depth_bins, load_layer_thre
from utils.helper import get_data_path
from utils.ROI_columns import load_hexed_body_ids
from utils.ROI_voxels import voxelize_col_and_lay
from utils.ol_color import OL_COLOR
from utils.column_features_functions import hex_from_col, find_cmax_across_all_neuropils
from utils.column_plotting_functions import plot_per_col
from utils.column_features_helper_functions import find_neuropil_hex_coords
from utils.all_syn_functions import get_depth_df_all
from queries.coverage_queries import fetch_syn_all_types


def plot_mi1_t4_alignment(
) -> None:
    """
    Create plots showing mi1-t4 alignment.

    It generates interactive plots in the data path named, e.g. `Alignment_mi1_t4.pdf`.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    #reference hex ids
    hex_ids, _ = find_neuropil_hex_coords(roi_str='ME(R)')
    hex1_vals_empty = hex_ids['hex1_id'].values
    hex2_vals_empty = hex_ids['hex2_id'].values

    t4_df = load_hexed_body_ids(roi_str='LOP(R)')
    t4_df = t4_df[['hex1_id','hex2_id','T4a','T4b','T4c','T4d']].drop_duplicates()
    t4a_df = t4_df[t4_df['T4a'].isna()]
    t4b_df = t4_df[t4_df['T4b'].isna()]
    t4c_df = t4_df[t4_df['T4c'].isna()]
    t4d_df = t4_df[t4_df['T4d'].isna()]
    t40_df = t4_df[
        (t4_df['T4a'].isna()) \
      & (t4_df['T4b'].isna()) \
      & (t4_df['T4c'].isna()) \
      & (t4_df['T4d'].isna())
    ]

    # plotting parameters
    dotsize = 15
    symbol_number = 15
    tot_max = np.multiply([hex_ids['hex1_id'].max() + hex_ids['hex2_id'].max()],  1)
    tot_min = np.multiply([hex_ids['hex1_id'].min() - hex_ids['hex2_id'].max()],  1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=-1*(hex1_vals_empty - hex2_vals_empty)
      , y=(hex1_vals_empty + hex2_vals_empty)
      , mode='markers'
      , marker_symbol=symbol_number
      , marker={
            'size': dotsize
          , 'color': 'gainsboro'
          , 'line': {'width': 1, 'color': 'white'}
        }
      , showlegend=False))
    fig.add_trace(go.Scatter(
        x=-1*(t4a_df['hex1_id'].values - t4a_df['hex2_id'].values)
      , y=(t4a_df['hex1_id'].values + t4a_df['hex2_id'].values)
      , mode='markers'
      , marker_symbol=symbol_number
      , marker={
           'size': dotsize
          , 'color': 'red'
          , 'line': {'width': 0.5, 'color': 'white'}
        }
      , showlegend=False)
    )
    fig.add_trace(go.Scatter(
        x=-1*(t4b_df['hex1_id'].values - t4b_df['hex2_id'].values)
      , y=(t4b_df['hex1_id'].values + t4b_df['hex2_id'].values)
      , mode='markers'
      , marker_symbol=symbol_number
      , marker={
            'size': dotsize
          , 'color': 'blue'
          , 'line': {'width': 0.5, 'color': 'white'}
        }
      , showlegend=False)
    )
    fig.add_trace(go.Scatter(
        x=-1*(t4c_df['hex1_id'].values - t4c_df['hex2_id'].values)
      , y=(t4c_df['hex1_id'].values + t4c_df['hex2_id'].values)
      , mode='markers'
      , marker_symbol=symbol_number
      , marker={
            'size': dotsize
          , 'color': 'green'
          , 'line': {'width': 0.5, 'color': 'white'}
        }
      , showlegend=False)
    )
    fig.add_trace(go.Scatter(
        x=-1*(t4d_df['hex1_id'].values - t4d_df['hex2_id'].values)
      , y=(t4d_df['hex1_id'].values + t4d_df['hex2_id'].values)
      , mode='markers'
      , marker_symbol=symbol_number
      , marker={
           'size': dotsize
          , 'color': 'orange'
          , 'line': {'width': 0.5, 'color': 'white'}
        }
      , showlegend=False)
    )
    fig.add_trace(go.Scatter(
        x=-1*(t40_df['hex1_id'].values - t40_df['hex2_id'].values)
      , y=(t40_df['hex1_id'].values + t40_df['hex2_id'].values)
      , mode='markers'
      , marker_symbol=symbol_number
      , marker={
            'size': dotsize
          , 'color': 'dimgrey'
          , 'line': {'width': 0.5, 'color': 'white'}
        }
      , showlegend=False)
    )
    fig.update_layout(
        yaxis_range=[tot_min , tot_max + tot_max/10]
      , xaxis_range=[tot_min, tot_max + tot_max/10]
      , height=660
      , width=625
      , paper_bgcolor='rgba(255,255,255,255)'
      , plot_bgcolor='rgba(255,255,255,255)'
    )
    fig.update_xaxes(showgrid=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, showticklabels=False)
    fig.write_image(get_data_path() / 'Alignment_mi1_t4.pdf')


def plot_pin_assignment(
    rois:list[str]=None
) -> None:
    """
    Create plots showing synapse assignments to columns (for specific cell-types depending on the
    neuropil).
    It generates svg and pdf files named, e.g. `ME_column_assignment.svg`.

    Parameters
    ----------
    rois : list[str], default=None
        If `rois` is None, it uses ['ME(R)', 'LO(R)', 'LOP(R)'].

    Returns
    -------
    None
    """
    assert rois is None or set(rois) <= set(['ME(R)', 'LO(R)', 'LOP(R)']), \
        "ROIs are not from the list of ME(R), LO(R),  or LOP(R)"

    if rois is None:
        rois = ['ME(R)', 'LO(R)', 'LOP(R)']

    # set formatting parameters
    colors_dict = {'spline': 'gray', 'straight': 'red'}
    style = {
        'font_type': 'arial'
      , 'markerlinecolor': 'black'
      , 'linecolor': 'black'
    }
    sizing = {
        'fig_width': 160 # units = mm, max 180
      , 'fig_height': 60 # units = mm, max 170
      , 'fig_margin': 0
      , 'fsize_ticks_pt': 6
      , 'fsize_title_pt': 7
      , 'ticklen': 2
      , 'tickwidth': 1
      , 'axislinewidth': 0.6
      , 'markerlinewidth': 0.5
      , 'markersize': 2
    }

    pixelsperinch = 72
    pixelspermm = pixelsperinch / 25.4
    effective_width = (sizing["fig_width"] - sizing["fig_margin"]) * pixelspermm
    effective_height = (sizing["fig_height"] - sizing["fig_margin"]) * pixelspermm
    fsize_ticks_px = sizing["fsize_ticks_pt"] * (1/72) * pixelsperinch
    fsize_title_px = sizing["fsize_title_pt"] * (1/72) * pixelsperinch

    pio.kaleido.scope.mathjax = None

    #compute assignment
    for roi_str in rois:
        # pick cell-type to do column assignment
        match roi_str:
            case 'ME(R)':
                target_types_list = [
                    'L1', 'L2', 'L3', 'L5'
                  , 'Mi1', 'Mi4', 'Mi9'
                  , 'C2', 'C3'
                  , 'Tm1', 'Tm2', 'Tm9', 'Tm20'
                  , 'T1', 'Tm4', 'TmY5a'
                ]
            case 'LO(R)':
                target_types_list = ['Tm1', 'Tm2', 'Tm9', 'Tm20', 'Tm4', 'TmY5a']
            case 'LOP(R)':
                target_types_list = [
                    'T4a', 'T4b', 'T4c', 'T4d'
                  , 'T5a', 'T5b', 'T5c', 'T5d'
                  , 'TmY5a'
                ]
        #collect assignments from all cell-types
        frac_all_df = pd.DataFrame()
        for target_type in target_types_list:

            syn_df = fetch_synapses(NC(type=target_type), SC(rois=roi_str))

            #assign using bent columns
            hex_df = find_hex_ids(syn_df, roi_str=roi_str)
            syn_df['col_id1'] = hex_df['col_id'].values
            #assign using straight columns
            if roi_str=='ME(R)':
                hex_df2 = find_straight_hex_ids(syn_df, roi_str=roi_str, suffix='_old')
            else:
                hex_df2 = find_straight_hex_ids(syn_df, roi_str=roi_str)
            syn_df['col_id2'] = hex_df2['col_id'].values
            #count synapse assigment per column
            count_df = syn_df\
                .groupby(['bodyId','col_id1'])['col_id1']\
                .count()\
                .to_frame(name='count1')\
                .reset_index()
            frac_df = \
                (count_df.groupby('bodyId')['count1'].max() \
                    / count_df.groupby('bodyId')['count1'].sum())\
                .to_frame(name='frac')\
                .reset_index()
            frac_df['method'] = 'spline'
            count_df2 = syn_df\
                .groupby(['bodyId','col_id2'])['col_id2']\
                .count()\
                .to_frame(name='count2')\
                .reset_index()
            frac_df2 = \
                (count_df2.groupby('bodyId')['count2'].max() \
                    / count_df2.groupby('bodyId')['count2'].sum())\
                .to_frame(name='frac')\
                .reset_index()
            frac_df2['method'] = 'straight'
            frac_df = pd.concat([frac_df, frac_df2])
            frac_df['type'] = target_type
            if frac_all_df.empty:
                frac_all_df = frac_df
            else:
                frac_all_df = pd.concat([frac_all_df, frac_df])

        #plot
        fig = go.Figure()
        for method in frac_all_df['method'].unique():
            df_plot = frac_all_df[frac_all_df['method']==method]
            fig.add_trace(
                go.Box(
                    x=df_plot['type']
                  , y=df_plot['frac']
                  , notched=True
                  , fillcolor=colors_dict[method]
                  , line={'color': style['markerlinecolor'], 'width': sizing['markerlinewidth']}
                  , marker={'size': sizing['markersize'], 'opacity': 0.5}
                  , name=method
                  , showlegend=False
                )
            )
        fig.update_layout(
            boxmode='group'
          , boxgroupgap=0.1
          , boxgap=0.1
          , yaxis_range=[0,1]
          , height=effective_height
          , width=effective_width
          , margin={
                'l': 20
              , 'r': 20
              , 'b': 20
              , 't': 20
              , 'pad': 5
            }
          , paper_bgcolor='rgba(255,255,255,255)'
          , plot_bgcolor='rgba(255,255,255,0)'
          , font={'size': fsize_title_px, 'family': style['font_type']}
        )
        fig.update_xaxes(
            showline=True
          , showticklabels = True
          , showgrid=False
          , linewidth=sizing['axislinewidth']
          , linecolor='black'
          , tickfont={
                'size': fsize_ticks_px
              , 'family': style['font_type']
              , 'color': style['linecolor']
            }
          , ticks='outside'
          , ticklen=sizing['ticklen']
          , tickwidth=sizing["tickwidth"]
          , tickangle=0
          , mirror=True
        )
        fig.update_yaxes(
            showline=True
          , showticklabels = True
          , showgrid=False
          , linewidth=sizing['axislinewidth']
          , linecolor='black'
          , tickfont={
                'size': fsize_ticks_px
              , 'family': style['font_type']
              , 'color': style['linecolor']
            }
          , ticks='outside'
          , ticklen=sizing['ticklen']
          , tickwidth=sizing["tickwidth"]
          , mirror=True
          , tickvals=[0, 0.5, 1]
          , ticktext=['0', '.5', '1']
        )
        fig.write_image(get_data_path() / f"{roi_str[:-3]}_column_assignment.svg")
        fig.write_image(get_data_path() / f"{roi_str[:-3]}_column_assignment.pdf")


def plot_all_syn(
    side:str='both'
  , syn_type:str='post'
) -> go.Figure:
    """
    Create a hexagonal eyemap heatplot of the number of synapses per column in
    the three main optic lobe regions.

    Parameters
    ----------
    side : str, default = 'both'
        Neurons from which hemisphere to include. Can only be 'L', 'R', 'R-dominant' or 'both'.
    syn_type : str, default = 'post'
        Synapse type to include. Can only be 'pre', 'post' or 'both.
    """
    project_root = Path(find_dotenv()).parent
    cachedir = project_root / "cache" / "columns" / "pickle_files"
    cachedir.mkdir(parents=True, exist_ok=True)
    syn_file = cachedir / f"syn_per_column_all_types_{side}_{syn_type}.pickle"

    if syn_file.is_file():
        df = pd.read_pickle(syn_file)
    else:
        print(f"{syn_file} file does not exist. Generating it now.")
        df = fetch_syn_all_types(side=side, syn_type=syn_type)
        df.to_pickle(syn_file)

    df = hex_from_col(df)

    # formatting parameters
    style = {
        'font_type': "arial"
      , 'markerlinecolor': "rgba(0,0,0,0)"
      , 'linecolor': "black"
    }
    sizing = {
        'fig_width': 750    # units = mm
      , 'fig_height': 210   # units = mm
      , 'fig_margin': 0
      , 'fsize_ticks_pt': 35
      , 'fsize_title_pt': 35
      , 'markersize': 17
      , 'ticklen': 15
      , 'tickwidth': 5
      , 'axislinewidth': 4
      , 'markerlinewidth': 0.9
      , 'cbar_thickness': 30
      , 'cbar_len': 0.75
    }

    cs, _ = find_cmax_across_all_neuropils(df, thresh_val=1)

    plot_specs = {
        'filename': f'HEX_all_syn_{side}_{syn_type}'
      , 'cmax_cells': 0
      , 'cmax_syn': cs
      , 'export_type': "pdf"
      , 'cbar_title_x': 1.17
      , 'cbar_title_y': 0.06
      , 'save_path': project_root / "results" / "cov_compl"
    }

    return plot_per_col(
        df
      , style
      , sizing
      , plot_specs
      , plot_type="synapses"
      , cmap_type='cont'
      , trim=False
    )


def plot_synapses_per_depth(
    style=dict,
    sizing=dict,
    plot_specs=dict,
) -> None:
    """
    Makes a 1x3 subplot of the number of synapses per depth in ME, LO and LOP (left to right)
    using the plotting parameters specified in 'style', 'sizing' and 'plot_specs'.

    Parameters
    ----------
    style : dict
        dict with styling parameters
    sizing : dict
        dict with sizing parameters
    plot_specs : dict
        dict with plot specs

    Returns
    -------
    fig : go.Figure
      figure of the number of pre and post synapses per depth
    """
    project_root = Path(find_dotenv()).parent
    cache_dir = project_root / "cache" / "columns" / "pickle_files"
    syn_depth_file = cache_dir / "depth_all.pickle"

    cache_dir.mkdir(parents=True, exist_ok=True)

    if syn_depth_file.is_file():
        syn_df_all = pd.read_pickle(syn_depth_file)
    else:
        print(f"{syn_depth_file} file does not exist. Generating it now.")
        syn_df_all = get_depth_df_all()
        syn_df_all.to_pickle(syn_depth_file)

    # styling
    pre_color = OL_COLOR.OL_SYNAPSES.hex[1]
    post_color = OL_COLOR.OL_SYNAPSES.hex[0]

    pixelsperinch = 72
    pixelspermm = pixelsperinch / 25.4
    effective_width = (sizing["fig_width"]-sizing["fig_margin"]) * pixelspermm
    effective_height = (sizing["fig_height"]-sizing["fig_margin"]) * pixelspermm
    fsize_ticks_px = sizing["fsize_ticks_pt"]*(1/72) * pixelsperinch
    fsize_title_px = sizing["fsize_title_pt"]*(1/72) * pixelsperinch

    # initiate figure with subplots
    fig = make_subplots(
        rows=1
      , cols=3
      , shared_yaxes=True
      , horizontal_spacing=0.1
    )

    fig.update_layout(
        height=effective_height
      , width=effective_width
      , margin={
            'l': 20
          , 'r': 20
          , 'b': 20
          , 't': 20
          , 'pad': 5
        }
      , paper_bgcolor='rgba(255,255,255,255)'
      , plot_bgcolor='rgba(255,255,255,0)'
      , font={
            'size': fsize_title_px
          , 'family': style['font_type']
          , 'color': style['linecolor']
        }
      , yaxis={
            'showline': True
          , 'title': {
                'text': 'Depth'
              , 'font': {
                    'family': style['font_type']
                  , 'size': fsize_title_px
                }
            }
          , 'showticklabels': True
          , 'ticks': 'outside'
          , 'tickfont': {
                'size': fsize_ticks_px
              , 'family': style['font_type']
              , 'color': style['linecolor']
            }
          , 'autorange': 'reversed'
          , 'title_standoff': 5
          , 'anchor': 'free'
          , 'range': [-0.03, 1.03]
          , 'tickvals': [0, 0.5, 1]
          , 'ticktext': ['1', '0.5', '0']
        }
      , yaxis2={
            'showline': False
          , 'title': ""
          , 'showticklabels': False
          , 'tickvals': []
        }
      , yaxis3={
            'showline': False
          , 'title': ""
          , 'showticklabels': False
          , 'tickvals': []
        }
      , xaxis1={
            'range': [0, 450000]
        }
      , xaxis2={
            'range': [0, 450000]
          , 'title': {
                'text': '# synaptic connections'
              , 'font': {'family': style['font_type'], 'size': fsize_title_px}
            }
          , 'title_standoff': 12
          , 'tickfont': {
                'size': fsize_ticks_px
              , 'family': style['font_type']
              , 'color': style['linecolor']
            }
        }
      , xaxis3={'range': [0, 450000]}
    )

    fig.update_annotations(font={'family': style['font_type'], 'size': fsize_title_px * 1.5})

    for col_idx, roi_str in enumerate(['ME(R)', 'LO(R)', 'LOP(R)'], start=1):

        #no. of synapses per depth
        bdry_val = load_layer_thre(roi_str=roi_str)
        syn_df = syn_df_all
        syn_df = syn_df[syn_df['roi']==roi_str].reset_index(drop=True)
        _, bin_centers = load_depth_bins(roi_str=roi_str, samp=2)
        depth_df_pre = syn_df[syn_df['type']=='pre']
        depth_df_post = syn_df[syn_df['type']=='post']
        count_pre_all = depth_df_pre['syn_sum'].values
        count_post_all = depth_df_post['syn_sum'].values
        count_pre = []
        count_post = []
        # bin depths subsampled by two, so sum neighbouring bin counts
        for i in range(0, len(count_pre_all)-1, 2):
            count_pre.append(count_pre_all[i] + count_pre_all[i+1])
            count_post.append(count_post_all[i] + count_post_all[i+1])

        # modified for plotting
        mod_bin_centers = np.append(np.insert(bin_centers, 0, -0.01), 1.01)
        mod_count_pre = np.append(np.insert(count_pre, 0, 0), 0)
        mod_count_post = np.append(np.insert(count_post, 0, 0), 0)

        fig.add_trace(go.Scatter(
            x=mod_count_pre
          , y=mod_bin_centers
          , showlegend=False
          , mode='lines'
          , name='pre'
          , line={'color': pre_color, 'width': sizing['markerlinewidth']})
          , row=1
          , col=col_idx
        )

        fig.add_trace(go.Scatter(
            x=mod_count_post
          , y=mod_bin_centers
          , showlegend=False
          , mode='lines'
          , name='post'
          , line={'color': post_color, 'width': sizing['markerlinewidth']})
          , row=1
          , col=col_idx
        )

        for x0 in bdry_val:
            fig.add_hline(
                y=x0
              , line_width=sizing['markerlinewidth'] - 0.5
              , layer='below'
              , line_color="lightgrey"
              , row=1, col=col_idx
            )

        fig.update_xaxes(
            showline=True
          , showticklabels=True
          , showgrid=False
          , linewidth=sizing['axislinewidth']
          , linecolor='black'
          , tickfont={
                'size': fsize_ticks_px
              , 'family': style['font_type']
              , 'color': style['linecolor']
            }
          , ticks='outside'
          , ticklen=sizing['ticklen']
          , tickwidth=sizing["tickwidth"]
          , tickangle=90
          , mirror=False
        )

        fig.update_yaxes(
            showgrid=False
          , linewidth=sizing['axislinewidth']
          , linecolor='black'
          , mirror=False
          , ticklen=sizing['ticklen']
          , tickwidth=sizing["tickwidth"]
        )

    fig.write_image(get_data_path() / f"SUBPLOT_syn_per_depth.{plot_specs['export_type']}")

    return fig


def find_max_pin_length():
    """
    Find the maximum value of pin length across neuropils.
    """
    maxvals=[]
    for roi in ['ME(R)', 'LO(R)', 'LOP(R)']:
        _, _, n_bins, pins = load_pins(roi_str=roi)
        pins = pins.astype(float)
        pins_length = \
            (8 / 1000) * np.sqrt((np.diff(pins.reshape((-1,n_bins,3)),axis=1)**2).sum(2))\
            .sum(1)
        max_len = max(pins_length)
        maxvals.append(max_len)
    max_length = max(maxvals)
    return max_length


def find_max_pin_deviation():
    """
    Find the maximum value of pin deviation across neuropils.
    """
    maxvals=[]
    for roi in ['ME(R)', 'LO(R)', 'LOP(R)']:
        col_ids, _, n_bins, pins = load_pins(roi_str=roi)
        pins = pins.astype(float)
        pin_interp = np.linspace(0,1,n_bins)
        pins_straight = pins.copy()
        for j in range(col_ids.shape[0]):
            pins_straight[j*n_bins:(j+1)*n_bins] = \
                (1 - pin_interp[:, np.newaxis])*pins[j*n_bins][np.newaxis,:] \
              + pin_interp[:, np.newaxis]*pins[(j+1)*n_bins-1][np.newaxis,:]
        pins_length = (8  / 1000)\
            * np.sqrt((np.diff(pins.reshape((-1,n_bins,3)),axis=1)**2).sum(2))\
            .sum(1)
        pins_diff = (8 / 1000)\
            * np.sqrt(((pins-pins_straight)**2).sum(1) ).reshape((-1,n_bins))\
            .sum(1)
        max_dev = max(pins_diff / pins_length)
        maxvals.append(max_dev)
    max_deviation = max(maxvals)
    return max_deviation


def find_max_pin_volume():
    """
    Find the maximum value of pin volume across neuropils.
    """
    maxvals=[]
    for roi in ['ME(R)', 'LO(R)', 'LOP(R)']:
        roi_fn = get_data_path('cache') / f'{roi[:-3]}_ZYX_columns.npy.gz'
        if not roi_fn.is_file():
            voxelize_col_and_lay(rois=[roi],layers=False)
        with gzip.GzipFile(roi_fn, "rb") as f:
            zyx_col = np.load(f)
        columns_list = list(np.unique(zyx_col)[1:])
        voxel_ct = np.array([ (zyx_col==col).sum() for col in columns_list ])
        voxel_area = ( (8*2**6/1000)**3 )*voxel_ct
        max_vol = max(voxel_area)
        maxvals.append(max_vol)
    max_volume = max(maxvals)
    return max_volume


def plot_pin_length_subplot(
    style: dict
  ,  sizing: dict
  ,  plot_specs: dict
)-> go.Figure:
    """
    Makes a 1x3 subplot of the length of pins in ME, LO and LOP (left to right)
    using the plotting parameters specified in 'style', 'sizing' and 'plot_specs'.

    Parameters
    ----------
    style : dict
        Dict with styling parameters.
    sizing : dict
        Dict with sizing parameters.
    plot_specs : dict
        Dict with plot specs.

    Returns
    -------
    fig : go.Figure
        Output figure containing the heatmap plot of the length of the column pins in each neuropil
    """
    pio.kaleido.scope.mathjax = None
    # specs
    symbol_number = 15

    # styling
    cmap = Colormap("reds_5").to_plotly()

    # 96 for png, 72 for svg and pdf
    if plot_specs['export_type'] in ['svg', 'pdf']:
        pixelsperinch = 72 
    else:
        pixelsperinch = 96
    pixelspermm = pixelsperinch / 25.4
    effective_width = (sizing['fig_width'] - sizing['fig_margin']) * pixelspermm
    effective_height = (sizing['fig_height'] - sizing['fig_margin']) * pixelspermm
    fsize_ticks_px = sizing['fsize_ticks_pt'] * (1/72) * pixelsperinch
    fsize_title_px = sizing['fsize_title_pt'] * (1/72) * pixelsperinch

    #reference hex ids
    hex_ids_ref, _ = find_neuropil_hex_coords('ME(R)')
    hex1_vals_empty = hex_ids_ref['hex1_id'].values
    hex2_vals_empty = hex_ids_ref['hex2_id'].values

    # initiate plot
    fig = make_subplots(rows=1, cols=3)
    fig.update_layout(
        autosize=False
      , height=effective_height
      , width=effective_width
      , margin={
            'l': 0
          , 'r': 0
          , 'b': 0
          , 't': 0
          , 'pad': 0
        }
      , paper_bgcolor='rgba(255,255,255,255)'
      , plot_bgcolor='rgba(255,255,255,255)'
    )
    fig.update_xaxes(showgrid=False, showticklabels=False, showline=False)
    fig.update_yaxes(showgrid=False, showticklabels=False, showline=False)

    for col_idx, roi_str in enumerate(['ME(R)', 'LO(R)', 'LOP(R)'], start=1):
        #compute and plot pin length
        col_ids, hex_ids, n_bins, pins = load_pins(roi_str=roi_str)
        pins = pins.astype(float)
        pin_interp = np.linspace(0,1,n_bins)
        pins_straight = pins.copy()
        for j in range(col_ids.shape[0]):
            pins_straight[j*n_bins:(j+1)*n_bins] = \
                (1 - pin_interp[:, np.newaxis])*pins[j*n_bins][np.newaxis,:] \
              + pin_interp[:, np.newaxis]*pins[(j+1)*n_bins-1][np.newaxis,:]
        pins_length = (8 / 1000)\
            * np.sqrt((np.diff(pins.reshape((-1,n_bins,3)),axis=1)**2).sum(2))\
            .sum(1)
        hex1_vals = hex_ids['hex1_id'].values
        hex2_vals = hex_ids['hex2_id'].values
        color_quantity = pins_length

        fig.add_trace(
            go.Scatter(
                x=-1 * (hex1_vals_empty - hex2_vals_empty)
              , y=(hex1_vals_empty + hex2_vals_empty)
              , mode='markers'
              , marker_symbol = symbol_number
              , marker={
                    'size': sizing['markersize']
                  , 'color': 'lightgrey'
                  , 'line': {
                        'width': sizing['markerlinewidth']
                      , 'color': style['markerlinecolor']
                    }
                }
              , showlegend=False
            )
          ,  row=1, col=col_idx
        )

        if roi_str == 'ME(R)':
            fig.add_trace(
                go.Scatter(
                    x=-1 * (hex1_vals - hex2_vals)
                  , y=(hex1_vals + hex2_vals)
                  , mode='markers'
                  , marker_symbol=symbol_number
                  , marker={
                        'cmin': 0
                      , 'cmax': plot_specs['cmax']
                      , 'size': sizing['markersize']
                      , 'color': color_quantity
                      , 'line': {
                            'width': sizing['markerlinewidth']
                          , 'color': style['markerlinecolor']
                        }
                      , 'colorbar': {
                            'x': -0.15
                          , 'y': 0.5
                          , 'orientation': 'v'
                          , 'outlinecolor': style['linecolor']
                          , 'outlinewidth': sizing['axislinewidth']
                          , 'thickness': sizing['cbar_thickness']
                          , 'len': sizing['cbar_len']
                          , 'title': {
                                'font': {
                                    'family': style['font_type']
                                  , 'size': fsize_title_px
                                  , 'color': style['linecolor']
                                }
                              , 'side': "right"
                              , 'text': "column length (µm)"
                            }
                          , 'ticklen': sizing['ticklen']
                          , 'tickwidth': sizing["tickwidth"]
                          , 'tickcolor': style['linecolor']
                          , 'tickmode': "array"
                          , 'tickvals': [plot_specs['cmax'], 60, 40, 20, 0]
                          , 'ticktext': [str(int(plot_specs['cmax'])), '60', '40', '20', '0']
                          , 'tickformat': "s"
                          , 'tickfont': {
                                'size': fsize_ticks_px
                              , 'family': style['font_type']
                              , 'color': style['linecolor']
                            }
                        }
                      , 'colorscale': cmap
                    }
                  , showlegend=False
                )
              , row=1, col=col_idx
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=-1 * (hex1_vals - hex2_vals)
                  , y=(hex1_vals + hex2_vals)
                  , mode='markers'
                  , marker_symbol=symbol_number
                  , marker={
                        'cmin': 0
                      , 'cmax': plot_specs['cmax']
                      , 'size': sizing['markersize']
                      , 'color': color_quantity
                      , 'line': {
                            'width': sizing['markerlinewidth']
                          , 'color': style['markerlinecolor']
                        }
                      , 'colorscale': cmap
                    }
                  , showlegend=False
                )
              , row=1, col=col_idx
            )

        fig.update_layout(
            height=effective_height
          , width=effective_width
          , paper_bgcolor='rgba(255,255,255,255)'
          , plot_bgcolor='rgba(255,255,255,255)'
        )
        fig.update_xaxes(showgrid=False, showticklabels=False, showline=False)
        fig.update_yaxes(showgrid=False, showticklabels=False, showline=False)

    # Save the image
    pio.write_image(
        fig
      , get_data_path() /  f"Subplot_pin_length_per_col.{plot_specs['export_type']}"
      , width=effective_width
      , height=effective_height
    )
    return fig


def plot_pin_deviation_subplot(
    style: dict
  , sizing: dict
  , plot_specs: dict
)-> go.Figure:
    """
    Makes a 1x3 subplot of the deviation of pins in ME, LO and LOP (left ot right)
    using the plotting parameters specified in 'style', 'sizing' and 'plot_specs'.

    Parameters
    ----------
    style : dict
        dict with styling parameters
    sizing : dict
        dict with sizing parameters
    plot_specs : dict
        dict with plot specs

    Returns
    -------
    fig : go.Figure
        Output figure containing the heatmap plot of the deviation of the column
        pins in each neuropil
    """
    pio.kaleido.scope.mathjax = None
    # specs
    symbol_number = 15

    # styling
    cmap = Colormap("reds_5").to_plotly()

    if plot_specs['export_type'] in ['svg', 'pdf']:
        pixelsperinch = 72 #96 for png, 72 for svg and pdf
    else:
        pixelsperinch = 96
    pixelspermm = pixelsperinch / 25.4
    effective_width = (sizing['fig_width'] - sizing['fig_margin']) * pixelspermm
    effective_height = (sizing['fig_height'] - sizing['fig_margin']) * pixelspermm
    fsize_ticks_px = sizing['fsize_ticks_pt'] * (1/72) * pixelsperinch
    fsize_title_px = sizing['fsize_title_pt'] * (1/72) * pixelsperinch

    #reference hex ids
    column_hex_ids, _ = find_neuropil_hex_coords(roi_str='ME(R)')
    hex1_vals_empty = column_hex_ids['hex1_id'].values
    hex2_vals_empty = column_hex_ids['hex2_id'].values

    # initiate plot
    fig = make_subplots(rows=1, cols=3, subplot_titles=("ME", "LO", "LOP"))
    fig.update_layout(
        autosize=False
      , height=effective_height
      , width=effective_width
      , margin={
            'l': 0
          , 'r': 0
          , 'b': 0
          , 't': 0
          , 'pad': 0
        }
      , paper_bgcolor='rgba(255,255,255,255)'
      , plot_bgcolor='rgba(255,255,255,255)'
    )
    fig.update_xaxes(showgrid=False, showticklabels=False, showline=False)
    fig.update_yaxes(showgrid=False, showticklabels=False, showline=False)

    for col_idx, roi_str in enumerate(['ME(R)', 'LO(R)', 'LOP(R)'], start=1):
        #compute and plot pin length
        col_ids, hex_ids, n_bins, pins = load_pins(roi_str=roi_str)
        pins = pins.astype(float)
        pin_interp = np.linspace(0,1,n_bins)
        pins_straight = pins.copy()
        for j in range(col_ids.shape[0]):
            pins_straight[j*n_bins:(j+1)*n_bins] = \
                (1 - pin_interp[:, np.newaxis])*pins[j*n_bins][np.newaxis,:] \
              + pin_interp[:, np.newaxis]*pins[(j+1)*n_bins-1][np.newaxis,:]
        pins_length = (8 / 1000)\
            * np.sqrt((np.diff(pins.reshape((-1,n_bins,3)), axis=1) **2).sum(2))\
            .sum(1)
        pins_diff = (8 / 1000)\
            * np.sqrt(((pins-pins_straight)**2).sum(1))\
            .reshape((-1,n_bins)).sum(1)
        color_quantity = pins_diff / pins_length
        hex1_vals = hex_ids['hex1_id'].values
        hex2_vals = hex_ids['hex2_id'].values

        fig.add_trace(
            go.Scatter(
                x=-1*(hex1_vals_empty - hex2_vals_empty)
              , y=(hex1_vals_empty + hex2_vals_empty)
              , mode='markers'
              , marker_symbol = symbol_number
              , marker={
                    'size': sizing['markersize']
                  , 'color': 'lightgrey'
                  , 'line': {
                        'width': sizing['markerlinewidth']
                      , 'color': style['markerlinecolor']
                    }
                }
              , showlegend=False
            )
          , row=1, col=col_idx
        )

        if roi_str == 'ME(R)':
            fig.add_trace(
                go.Scatter(
                    x=-1 * (hex1_vals - hex2_vals)
                  , y=(hex1_vals + hex2_vals)
                  , mode='markers'
                  , marker_symbol=symbol_number
                  , marker={
                        'cmin': 0
                      , 'cmax': plot_specs['cmax']
                      , 'size': sizing['markersize']
                      , 'color': color_quantity
                      , 'line': {
                            'width': sizing['markerlinewidth']
                          , 'color': style['markerlinecolor']
                        }
                      , 'colorbar': {
                            'x': -0.15
                          , 'y': 0.5
                          , 'orientation': 'v'
                          , 'outlinecolor': style['linecolor']
                          , 'outlinewidth': sizing['axislinewidth']
                          , 'thickness': sizing['cbar_thickness']
                          , 'len': sizing['cbar_len']
                          , 'title': {
                                'font': {
                                    'family': style['font_type']
                                  , 'size': fsize_title_px
                                  , 'color': style['linecolor']
                                }
                              , 'side': "right"
                              , 'text': "col deviation / col length"
                            }
                          , 'ticklen': sizing['ticklen']
                          , 'tickwidth': sizing["tickwidth"]
                          , 'tickcolor': style['linecolor']
                          , 'tickmode': "array"
                          , 'tickvals': [plot_specs['cmax'], 8, 6,4,2,0]
                          , 'ticktext': [str(int(plot_specs['cmax'])), '8', '6', '4', '2', '0']
                          , 'tickfont': {
                                'size': fsize_ticks_px
                              , 'family': style['font_type']
                              , 'color': style['linecolor']
                            }
                        }
                      , 'colorscale': cmap
                    }
                  , showlegend=False
                )
              , row=1, col=col_idx
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=-1 * (hex1_vals - hex2_vals)
                  , y=(hex1_vals + hex2_vals)
                  , mode='markers'
                  , marker_symbol=symbol_number
                  , marker={
                        'cmin': 0
                      , 'cmax': plot_specs['cmax']
                      , 'size': sizing['markersize']
                      , 'color': color_quantity
                      , 'line': {
                            'width': sizing['markerlinewidth']
                          , 'color': style['markerlinecolor']
                        }
                      , 'colorscale': cmap
                    }
                  , showlegend=False
                )
              , row=1, col=col_idx
            )

        fig.update_layout(
            height=effective_height
          , width=effective_width
          , paper_bgcolor='rgba(255,255,255,255)'
          , plot_bgcolor='rgba(255,255,255,255)'
        )
        fig.update_xaxes(showgrid=False, showticklabels=False, showline=False, visible=False)
        fig.update_yaxes(showgrid=False, showticklabels=False, showline=False, visible=False)

    # Save the image
    pio.write_image(
        fig
      , get_data_path() /  f"Subplot_pin_deviation_per_col.{plot_specs['export_type']}"
      , width=effective_width
      , height=effective_height
    )
    return fig


def plot_pin_volume_subplot(
    style: dict
  , sizing: dict
  , plot_specs: dict
) -> go.Figure:
    """
    Makes a 1x3 subplot of the volume of pins in ME, LO and LOP (left ot right)
    using the plotting parameters specified in 'style', 'sizing' and 'plot_specs'.

    Parameters
    ----------
    style : dict
        dict with styling parameters
    sizing : dict
        dict with sizing parameters
    plot_specs : dict
        dict with plot specs

    Returns
    -------
    fig : go.Figure
        Output figure containing the heatmap plot of the volume of the column pins in each neuropil
    """
    pio.kaleido.scope.mathjax = None

    # specs
    symbol_number = 15

    # styling
    cmap = Colormap("reds_5").to_plotly()

    if plot_specs['export_type'] in ['svg', 'pdf']:
        pixelsperinch = 72 # 96 for png, 72 for svg and pdf
    else:
        pixelsperinch = 96
    effective_width = (sizing['fig_width'] - sizing['fig_margin']) * pixelsperinch / 25.4
    effective_height = (sizing['fig_height'] - sizing['fig_margin']) * pixelsperinch / 25.4
    fsize_ticks_px = sizing['fsize_ticks_pt'] * (1/72) * pixelsperinch
    fsize_title_px = sizing['fsize_title_pt'] * (1/72) * pixelsperinch

    #reference hex ids
    column_hex_ids, _ = find_neuropil_hex_coords(roi_str='ME(R)')
    hex1_vals_empty = column_hex_ids['hex1_id'].values
    hex2_vals_empty = column_hex_ids['hex2_id'].values

    # initiate plot
    fig = make_subplots(rows=1, cols=3, subplot_titles=("ME", "LO", "LOP"))
    fig.update_layout(
        autosize=False
      , height=effective_height
      , width=effective_width
      , margin={
            'l': 0
          , 'r': 0
          , 'b': 0
          , 't': 0
          , 'pad': 0
        }
      , paper_bgcolor='rgba(255,255,255,255)'
      , plot_bgcolor='rgba(255,255,255,255)'
    )
    fig.update_xaxes(showgrid=False, showticklabels=False, showline=False)
    fig.update_yaxes(showgrid=False, showticklabels=False, showline=False)

    for col_idx, roi_str in enumerate(['ME(R)', 'LO(R)', 'LOP(R)'], start=1):
        #compute and plot pin volume
        roi_fn = get_data_path('cache') / f'{roi_str[:-3]}_ZYX_columns.npy.gz'
        if not roi_fn.is_file():
            voxelize_col_and_lay(rois=[roi_str],layers=False)
        with gzip.GzipFile(roi_fn, "rb") as f:
            zyx_col = np.load(f)
        columns_list = list(np.unique(zyx_col)[1:])
        voxel_ct = np.array([ (zyx_col==col).sum() for col in columns_list ])
        voxel_area = ((8 * 2**6 / 1000)**3) * voxel_ct
        hex1_vals = np.array([col // 100 for col in columns_list])
        hex2_vals = np.array([col % 100 for col in columns_list])

        fig.add_trace(
            go.Scatter(
                x=-1*(hex1_vals_empty - hex2_vals_empty)
              , y=(hex1_vals_empty + hex2_vals_empty)
              , mode='markers'
              , marker_symbol=symbol_number
              , marker={
                    'size': sizing['markersize']
                  , 'color': 'lightgrey'
                  , 'line': {
                        'width': sizing['markerlinewidth']
                      , 'color': style['markerlinecolor']
                    }
                }
              , showlegend=False
            )
          , row=1, col=col_idx
        )

        if roi_str == 'ME(R)':
            fig.add_trace(
                go.Scatter(
                    x=-1 * (hex1_vals - hex2_vals)
                  , y=(hex1_vals + hex2_vals)
                  , mode='markers'
                  , marker_symbol=symbol_number
                  , marker={
                        'cmin': 0
                      , 'cmax': plot_specs['cmax']
                      , 'size': sizing['markersize']
                      , 'color': voxel_area
                      , 'line': {
                            'width': sizing['markerlinewidth']
                          , 'color': style['markerlinecolor']
                        }
                      , 'colorbar': {
                            'x': -0.15
                          , 'y': 0.5
                          , 'orientation': 'v'
                          , 'outlinecolor': style['linecolor']
                          , 'outlinewidth': sizing['axislinewidth']
                          , 'thickness': sizing['cbar_thickness']
                          , 'len': sizing['cbar_len']
                          , 'title': {
                                'font': {
                                    'family': style['font_type']
                                  , 'size': fsize_title_px
                                  , 'color': style['linecolor']
                                }
                              , 'side': "right"
                              , 'text': "volume (µm\N{SUPERSCRIPT THREE})"
                            }
                          , 'ticklen': sizing['ticklen']
                          , 'tickwidth': sizing['tickwidth']
                          , 'tickmode': 'array'
                          , 'tickvals': [plot_specs['cmax'], 4000, 3000, 2000, 1000, 0]
                          , 'ticktext': [
                                str(int(plot_specs['cmax'])), '4000', '3000', '2000', '1000', '0'
                            ]
                          , 'tickcolor': style['linecolor']
                          , 'tickformat': "s"
                          , 'tickfont': {
                                'size': fsize_ticks_px
                              , 'family': style['font_type']
                              , 'color': style['linecolor']
                            }
                        }
                      , 'colorscale': cmap
                    }
                  , showlegend=False
                )
              , row=1, col=col_idx
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=-1 * (hex1_vals - hex2_vals)
                  , y=(hex1_vals + hex2_vals)
                  , mode='markers'
                  , marker_symbol=symbol_number
                  , marker={
                        'cmin': 0
                      , 'cmax': plot_specs['cmax']
                      , 'size': sizing['markersize']
                      , 'color': voxel_area
                      , 'line': {
                            'width': sizing['markerlinewidth']
                          , 'color': style['markerlinecolor']
                        }
                      , 'colorscale': cmap
                    }
                  , showlegend=False
                )
              , row=1, col=col_idx
            )

        fig.update_layout(
            height=effective_height
          , width=effective_width
          , paper_bgcolor='rgba(255,255,255,255)'
          , plot_bgcolor='rgba(255,255,255,255)'
        )
        fig.update_xaxes(showgrid=False, showticklabels=False, showline=False, visible=False)
        fig.update_yaxes(showgrid=False, showticklabels=False, showline=False, visible=False)

    # Save the image
    pio.write_image(
        fig
      , get_data_path() /  f"Subplot_volume_per_col.{plot_specs['export_type']}"
      , width=effective_width
      , height=effective_height
    )
    return fig
