import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from neuprint import NeuronCriteria as NC, SynapseCriteria as SC, fetch_synapses

from utils.ROI_calculus import \
    find_depth, find_hex_ids, find_straight_hex_ids\
  , load_depth_bins, load_layer_thre, _get_data_path
from utils.ROI_columns import load_hexed_body_ids


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
    data_path = _get_data_path('cache')
    column_df = pd.read_pickle(
        data_path / 'ME_col_center_pins.pickle')[['hex1_id','hex2_id']]
    hex1_vals_empty = column_df['hex1_id'].values
    hex2_vals_empty = column_df['hex2_id'].values

    t4_df = load_hexed_body_ids(roi_str='LOP(R)')
    t4_df = t4_df[['hex1_id','hex2_id','T4a','T4b','T4c','T4d']].drop_duplicates()
    t4_df = column_df.merge(t4_df, 'left', on=['hex1_id','hex2_id'])
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

    #plotting parameters
    w = 625
    h = 660
    dotsize = 15
    symbol_number = 15
    mul_fac = 1
    tot_max = np.multiply([column_df['hex1_id'].max() + column_df['hex2_id'].max()],  mul_fac)
    tot_min = np.multiply([column_df['hex1_id'].min() - column_df['hex2_id'].max()],  mul_fac)

    pio.kaleido.scope.mathjax = None

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
          , 'line': {'width':0.5, 'color': 'white'}
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
      , height=h
      , width=w
      , paper_bgcolor='rgba(255,255,255,255)'
      , plot_bgcolor='rgba(255,255,255,255)'
    )
    fig.update_xaxes(showgrid=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, showticklabels=False)
    fig.write_image(_get_data_path() / 'mi1_t4_alignment.pdf')


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
      , 'markerlinecolor':'black'
       , 'linecolor':'black'
    }
    sizing = {
        'fig_width':160 # units = mm, max 180
      , 'fig_height':60 # units = mm, max 170
      , 'fig_margin':0
      , 'fsize_ticks_pt':6
      , 'fsize_title_pt':7
      , 'ticklen':2
      , 'tickwidth':1
      , 'axislinewidth':0.6
      , 'markerlinewidth':0.5
      , 'markersize':2
    }

    # get sizing values
    font_type = style["font_type"]
    markerlinecolor = style["markerlinecolor"]
    linecolor = style["linecolor"]
    ticklen = sizing["ticklen"]
    tickwidth = sizing["tickwidth"]
    axislinewidth = sizing["axislinewidth"]
    markerlinewidth = sizing["markerlinewidth"]
    markersize = sizing["markersize"]

    pixelsperinch = 72
    pixelspermm = pixelsperinch / 25.4
    w = (sizing["fig_width"]-sizing["fig_margin"])*pixelspermm
    h = (sizing["fig_height"]-sizing["fig_margin"])*pixelspermm
    fsize_ticks_px = sizing["fsize_ticks_pt"]*(1/72)*pixelsperinch
    fsize_title_px = sizing["fsize_title_pt"]*(1/72)*pixelsperinch

    pio.kaleido.scope.mathjax = None

    #compute assignment
    for roi_str in rois:
        #pick cell-type to do column assignment
        if roi_str=='ME(R)':
            target_types_list = [
                'L1', 'L2', 'L3', 'L5'
              , 'Mi1', 'Mi4', 'Mi9'
              , 'C2', 'C3'
              , 'Tm1', 'Tm2', 'Tm9', 'Tm20'
              , 'T1', 'Tm4', 'TmY5a'
            ]
        elif roi_str=='LO(R)':
            target_types_list = ['Tm1', 'Tm2', 'Tm9', 'Tm20', 'Tm4', 'TmY5a']
        elif roi_str=='LOP(R)':
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
            frac_df = pd.concat([frac_df,frac_df2])
            frac_df['type'] = target_type
            if frac_all_df.empty:
                frac_all_df = frac_df
            else:
                frac_all_df = pd.concat([frac_all_df,frac_df])

        #plot
        fig=go.Figure()
        for _, method in enumerate(frac_all_df['method'].unique()):
            df_plot=frac_all_df[frac_all_df['method']==method]
            fig.add_trace(
                go.Box(
                    x=df_plot['type']
                  , y=df_plot['frac']
                  , notched=True
                  , fillcolor=colors_dict[method]
                  , line={'color': markerlinecolor, 'width': markerlinewidth}
                  , marker={"size": markersize, "opacity": 0.5}
                  , name=method
                  , showlegend=False
                )
                         )
        fig.update_layout(
              boxmode='group'
            , boxgroupgap=0.1
            , boxgap=0.1
            , yaxis_range=[0,1]
            , height=h
            , width=w
            , margin={
                  'l':20
                , 'r':20
                , 'b':20
                , 't':20
                , 'pad':5
              }
            , paper_bgcolor='rgba(255,255,255,255)'
            , plot_bgcolor='rgba(255,255,255,0)'
            , font={'size': fsize_title_px, 'family': font_type}
        )
        fig.update_xaxes(
              showline=True
            , showticklabels = True
            , showgrid=False
            , linewidth=axislinewidth
            , linecolor='black'
            , tickfont={
                  'size': fsize_ticks_px
                , 'family': font_type
                , 'color': linecolor
              }
            , ticks='outside'
            , ticklen=ticklen
            , tickwidth=tickwidth
            , tickangle=0
            , mirror=True
          )
        fig.update_yaxes(
              showline=True
            , showticklabels = True
            , showgrid=False
            , linewidth=axislinewidth
            , linecolor='black'
            , tickfont={
                  'size': fsize_ticks_px
                , 'family': font_type
                , 'color': linecolor
              }
            , ticks='outside'
            , ticklen=ticklen
            , tickwidth=tickwidth
            , mirror=True
            , tickvals=[0,0.5,1]
            , ticktext=['0','.5','1']
        )
        fig.write_image(_get_data_path() / f"{roi_str[:-3]}_column_assignment.svg")
        fig.write_image(_get_data_path() / f"{roi_str[:-3]}_column_assignment.pdf")


def plot_all_syn(
    rois:list[str]=None
) -> None:
    """
    Create plots using all synapses of neuropils.

    It generates files in the data path named, e.g. `ME_all_syn_per_col.pdf`.

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

    #reference hex ids
    data_path = _get_data_path('cache')
    column_df = pd.read_pickle(
        data_path / 'ME_col_center_pins.pickle')[['hex1_id','hex2_id']]
    hex1_vals_empty = column_df['hex1_id'].values
    hex2_vals_empty = column_df['hex2_id'].values

    #plotting parameters
    w = 625
    h = 700
    dotsize = 15
    symbol_number = 15
    mul_fac = 1
    tot_max = np.multiply([column_df['hex1_id'].max() + column_df['hex2_id'].max()],  mul_fac)
    tot_min = np.multiply([column_df['hex1_id'].min() - column_df['hex2_id'].max()],  mul_fac)
    w2 = 400
    h2 = 700
    linewidth2 = 2

    if rois is None:
        rois = ['ME(R)', 'LO(R)', 'LOP(R)']

    pio.kaleido.scope.mathjax = None

    for roi_str in rois:

        syn_df = fetch_synapses(None, SC(rois=roi_str))

        #no. of synapses per columns
        hex_df = find_hex_ids(syn_df, roi_str=roi_str)
        count_df_tot = hex_df\
            .groupby(['hex1_id','hex2_id'])['col_id']\
            .count()\
            .to_frame(name='count')\
            .reset_index()
        hex1_vals = count_df_tot['hex1_id'].values
        hex2_vals = count_df_tot['hex2_id'].values
        color_quantity = count_df_tot['count'].values

        fig = go.Figure()
        # Add grey hexagons for columns that are not present in the neuropil
        fig.add_trace(go.Scatter(
            x=-1*(hex1_vals_empty - hex2_vals_empty)
          , y=(hex1_vals_empty + hex2_vals_empty)
          , mode='markers'
          , marker_symbol=symbol_number
          , marker={
                'size': dotsize
              , 'color': 'lightgrey'
              , 'line': {'width': 1, 'color': 'white'}
            }
          , showlegend=False))
        fig.add_trace(go.Scatter(
            x=-1*(hex1_vals - hex2_vals)
          , y=(hex1_vals + hex2_vals)
          , mode='markers'
          , marker_symbol=symbol_number
          , marker={
                'cmin': 0
              , 'size': dotsize
              , 'color': color_quantity
              , 'line': {'width': 0.5, 'color': 'white'}
              , 'colorbar' : {
                    "x": 0.5
                  , "y": -0.2
                  , 'orientation': 'h'
                  , "outlinecolor": 'black'
                  , "outlinewidth": 1.5
                  , "thickness": 18
                  , "titleside": 'bottom'
                  , "len": 0.9
                  , "title": f'No. of synapses in {roi_str[:-3]}'
                  , "title_font_size": 25
                  , 'ticklen' : 8
                  , 'tickwidth': 1.5
                  , 'tickfont' : {'size': 22}
                }
              , 'colorscale': "Inferno_r"
            }
          , showlegend=False)
        )
        fig.update_layout(
            yaxis_range=[tot_min , tot_max + tot_max/10]
          , xaxis_range=[tot_min, tot_max + tot_max/10]
          , height=h
          , width=w
          , paper_bgcolor='rgba(255,255,255,255)'
          , plot_bgcolor='rgba(255,255,255,255)'
        )
        fig.update_xaxes(showgrid=False, showticklabels=False)
        fig.update_yaxes(showgrid=False, showticklabels=False)
        fig.write_image(data_path / f"{roi_str[:-3]}_all_syn_per_col.pdf")
        fig.write_image(data_path / f"{roi_str[:-3]}_all_syn_per_col.svg")

        #no. of synapses per depth
        bdry_val = load_layer_thre(roi_str=roi_str)
        depth_df = find_depth(syn_df, roi_str=roi_str, samp=2)
        bin_edges, bin_centers = load_depth_bins(roi_str=roi_str, samp=2)
        depth_df_pre = depth_df[syn_df['type']=='pre']
        depth_df_post = depth_df[syn_df['type']=='post']
        count_pre, _ = np.histogram(depth_df_pre['depth'].values, bins=bin_edges, density=False)
        count_post, _ = np.histogram(depth_df_post['depth'].values, bins=bin_edges, density=False)

        fig = go.Figure()
        for x0 in bdry_val:
            fig.add_hline(y=x0, line_width=1, line_color="lightgrey")
        fig.add_trace(go.Scatter(
            x=count_pre
          , y=bin_centers
          , mode='lines', name='pre'
          , line={'color': 'rgba(7,177,210,1.0)', 'width': linewidth2}
        ))
        fig.add_trace(go.Scatter(
            x=count_post
          , y=bin_centers
          , mode='lines', name='post'
          , line={'color': 'rgba(253,198,43,1.0)', 'width': linewidth2}
        ))

        fig.update_layout(
            yaxis_range=[bdry_val[0] , bdry_val[-1]]
          , xaxis_range=[tot_min, tot_max + tot_max/5]
          , height=h2
          , width=w2
          , paper_bgcolor='rgba(255,255,255,255)'
          , plot_bgcolor='rgba(255,255,255,255)'
          , xaxis={'title': f'No. of synapses in {roi_str[:-3]}'}
          , yaxis={'title': 'Depth'}
          , font={'size': 25}
        )
        fig.update_xaxes(
            showline=True
          , linewidth=1
          , linecolor='black'
          , mirror=True
        )
        fig.update_yaxes(
            showline=True
          , linewidth=1
          , linecolor='black'
          , mirror=True
          , autorange="reversed"
        )
        fig.write_image(data_path / f"{roi_str[:-3]}_all_syn_per_depth.pdf")
        fig.write_image(data_path / f"{roi_str[:-3]}_all_syn_per_depth.svg")
