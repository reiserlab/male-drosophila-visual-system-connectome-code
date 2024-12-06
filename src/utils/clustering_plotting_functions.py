import numpy as np
import pandas as pd

from neuprint import NeuronCriteria as NC, SynapseCriteria as SC

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from utils.ol_color import OL_COLOR

from utils.clustering_functions import \
   make_in_and_output_df \
  , get_row_linkage \
  , cluster_dict_from_linkage \
  , make_count_table \
  , cluster_with_type_names \
  , get_combined_synapses_with_stdev \

from utils.clustering_functions import add_two_colors_to_df, add_three_colors_to_df

# global parameters
CLUSTER1 = 1
CLUSTER2 = 2
CLUSTER3 = 3

def plot_only_spatialmap(
    synapses_for_plotting_colored:pd.DataFrame
  , color_legend_mapping:dict
  , marker_size_sf:int
  , type_name:str
  , style:dict
  , sizing:dict
  , plot_specs:dict
) -> go.Figure:
    """
    This function plots the spatial map along the two principal components of the locations
    of the synapses across the eye.

    Parameters
    ----------
    synapses_for_plotting_colored : pd.DataFrame
        dataframe containing all of the connectivity information for the different clusters
    color_legend_mapping : dict
        dictionary that instructs the color mapping of the data points to the legend
    marker_size_sf : int
        scaling factor of the marker size: different spatial maps are scaled differentially
        depending on how crowded or sparse the synapses are
    type_name : str
        grouped name for both the clusters together
    style : dict
        dict containing the values of the fixed styling formatting variables
    sizing : dict
        dict containing the values of the size formatting variables
    plot_specs : dict
        dict containing the values of the formatting variables relevant to the specific plot

    Returns
    -------
    fig : go.Figure
        formatted plotly scatterplot
    """
    export_type=style['export_type']
    font_type=style['font_type']
    markerlinecolor=style['markerlinecolor']
    linecolor = style['linecolor']
    opacity_spatialmap = style['opacity_spatialmap']

    # get sizing values
    markersize = sizing['markersize']
    axislinewidth = sizing['axislinewidth']
    markerlinewidth = sizing['markerlinewidth']

    pixelsperinch = 96
    pixelspermm = pixelsperinch/25.4
    fig_width = (sizing['fig_width'] - sizing['fig_margin'])*pixelspermm
    fig_height = (sizing['fig_height'] - sizing['fig_margin'])*pixelspermm
    fsize_ticks_px = sizing['fsize_ticks_pt']*(1/72)*pixelsperinch
    fsize_title_px = sizing['fsize_title_pt']*(1/72)*pixelsperinch

    fsize_ticks_px = 5
    fsize_title_px = 6

    fig = go.Figure()

    ## spatial map
    for color in synapses_for_plotting_colored['color'].unique():
        filtered_df = synapses_for_plotting_colored[
            synapses_for_plotting_colored['color'] == color
        ]
        trace = go.Scatter(
            x=-filtered_df['Y']
          , y=filtered_df['X']
          , name=color_legend_mapping[color]
          , hovertext=filtered_df['bodyId']
          , opacity=opacity_spatialmap
          , showlegend=True
          , mode='markers'
          , marker={
                'size':marker_size_sf * markersize
              , 'color': color
              , 'line': {
                    'width': markerlinewidth
                  , 'color': markerlinecolor
                }
            }
        )
        fig.add_trace(trace)

    fig.update_xaxes(
        title={
            'font': {
                'size': fsize_title_px
              , 'family': font_type
              , 'color': linecolor
            }
          , 'text':'PC1'
        }
      , ticks='outside'
      , ticklen=0
      , tickwidth=0
      , tickfont={
            'family': font_type
          , 'size': fsize_ticks_px
          , 'color': linecolor
          }
      , range=plot_specs['range_x']
      , showgrid=False
      , showline=False
      , linewidth=axislinewidth
      , linecolor=linecolor
      , showticklabels=False
      , visible=False
    )

    fig.update_yaxes(
        title={
            'font': {
                'size': fsize_title_px
              , 'family': font_type
              , 'color': linecolor
            }
          , 'text':'PC2'
        }
      , ticks='outside'
      , ticklen=0
      , tickwidth=0
      , tickfont={
            'family': font_type
          , 'size': fsize_ticks_px
          , 'color': linecolor
          }
      , range=plot_specs['range_y']
      , showgrid=False
      , showline=False
      , linewidth=axislinewidth
      , linecolor=linecolor
      , showticklabels=False
      , visible=False
    )

    fig.add_shape(
        type='line'
      , x0=plot_specs['range_x'][0]
      , y0=0
      , x1=plot_specs['range_x'][1]
      , y1=0
      , layer="below"
      , line={'color': linecolor, 'width': axislinewidth}
    )


    fig.add_shape(
        type='line'
      , x0=0
      , y0=plot_specs['range_y'][0]
      , x1=0
      , y1=plot_specs['range_y'][1]
      , layer="below"
      , line={'color': 'black', 'width': axislinewidth}
    )

    fig.update_layout(
        autosize=False
      , width=fig_width
      , height=fig_height
      , legend={
            'x': -0.2, 'y': -0.23
          , 'bgcolor': 'rgba(0,0,0,0)'
          , 'font': {'size': 5}
        }  # Set the border color
      , paper_bgcolor='rgba(255,255,255,1)'
      , plot_bgcolor='rgba(255,255,255,1)'
    )

    fig.update_annotations(font={'family': font_type, 'size': 6})

    save_path = plot_specs['save_path']
    save_path.mkdir(parents=True, exist_ok=True)

    pio.write_image(
        fig
      , save_path / f"{type_name}.{export_type}"
      , width=fig_width
      , height=fig_height
    )

    return fig


def plot_spatialmap_two_clusters(
    main_conn_celltypes_new:list
  , synapses_for_plotting_colored:pd.DataFrame
  , color_legend_mapping:dict
  , marker_size_sf:int
  , top_connection_clusters_df:pd.DataFrame
  , clusters_bids:dict
  , type_names: list
  , type_name:str
  , rois:list
  , colors:list
  , order:str
  , style:dict
  , sizing:dict
  , plot_specs:dict
) -> go.Figure:
    """
    This function plots the spatial map along the two principal components of the locations of
    the synapses across the eye as well as the distribution of #connections made by the two
    clusters of a cell type with some example cell types

    Parameters
    ----------
    main_conn_celltypes_new : list
        list of two input and two output celltypes that connect to the bodyIds within both
        clusters of the given celltype
    synapses_for_plotting_colored : pd.DataFrame
        dataframe containing all of the connectivity information for the different clusters
    color_legend_mapping : dict
        dictionary that instructs the color mapping of the data points to the legend
    marker_size_sf : int
        scaling factor of the marker size: different spatial maps are scaled differentially
        depending on how crowded or sparse the synapses are
    top_connection_clusters_df: pd.dataframe
        dataframe containing the connectivity information of the chosen example cell types with
        the clusters of the cell type under consideration
    clusters_bids : dict
        dictionary containing the mapping between bodyIds and cluster identity
    type_names : str
        names of existing clusters or clusters that the celltype needs to be divided into
    type_name : str
        grouped name for both the clusters together
    rois : list
        list of rois within which the synapses of the given celltype are located
    colors : list
        list of colors for the two clusters
    order : str
        refers to the cluster order in which plots need to be made
    style : dict
        dict containing the values of the fixed styling formatting variables
    sizing : dict
        dict containing the values of the size formatting variables
    plot_specs : dict
        dict containing the values of the formatting variables relevant to the specific plot

    Returns
    -------
    fig : go.Figure
        formatted plotly scatterplot
    """
    assert order in ['straight', 'reversed']\
      , f"only 'straight' or 'reversed' order supported, not '{order}'"

    export_type = style['export_type']
    font_type = style['font_type']
    markerlinecolor = style['markerlinecolor']
    linecolor = style['linecolor']
    opacity = style['opacity']
    opacity_spatialmap = style['opacity_spatialmap']
    jitter_extent = style['jitter_extent']
    x_centers = style['x_centers']
    x_deviation = style['x_deviation']

    # get sizing values
    markersize = sizing['markersize']
    ticklen = sizing['ticklen']
    tickwidth = sizing['tickwidth']
    axislinewidth = sizing['axislinewidth']
    markerlinewidth = sizing['markerlinewidth']
    markerlinewidth_spatialmap = sizing['markerlinewidth_spatialmap']

    pixelsperinch = 96
    pixelspermm = pixelsperinch/25.4
    fig_width = (sizing['fig_width'] - sizing['fig_margin']) * pixelspermm
    fig_height = (sizing['fig_height'] - sizing['fig_margin']) * pixelspermm
    fsize_ticks_px = sizing['fsize_ticks_pt'] * (1/72) * pixelsperinch
    fsize_title_px = sizing['fsize_title_pt'] * (1/72) * pixelsperinch

    fsize_ticks_px = 5
    fsize_title_px = 6

    fig = make_subplots(
        rows=2, cols=4
      , specs=[[{"colspan": 2, "rowspan": 2}, None, {}, {}]
      , [None, None, {}, {}]]
      , print_grid=True
      , subplot_titles=(
        '',  # Empty for the second column (since it's part of the first large subplot)
        f"input <br>{main_conn_celltypes_new[0]}",  # Row=0, Col=3 (col=2 in zero index)
        f"output <br>{main_conn_celltypes_new[2]}",  # Row=0, Col=4 (col=3 in zero index)
        f"{main_conn_celltypes_new[1]}",  # Row=1, Col=3 (col=2 in zero index)
        f"{main_conn_celltypes_new[3]}"   # Row=1, Col=4 (col=3 in zero index)
      )
      , column_widths=[0.3, 0.3, 0.2, 0.2]
      , row_heights=[0.5, 0.5]
      , horizontal_spacing = 0.09
      , vertical_spacing = 0.13
    )

    ## spatial map
    for color in synapses_for_plotting_colored['color'].unique():
        filtered_df = synapses_for_plotting_colored[
            synapses_for_plotting_colored['color'] == color
        ]
        if rois==['LO(R)']:
            filtered_df['Y'] = -filtered_df['Y']
        trace = go.Scatter(
            x=-filtered_df['Y']
          , y=filtered_df['X']
          , name=color_legend_mapping[color]
          , hovertext = filtered_df['bodyId']
          , opacity=opacity_spatialmap
          , showlegend=True
          , mode='markers'
          , marker={
                'size':marker_size_sf*markersize
              , 'color': color
              , 'line': {'width':markerlinewidth_spatialmap, 'color': markerlinecolor}
            }
        )
        fig.add_trace(trace, row=1, col=1)

    fig.update_xaxes(
        title={
            'font': {
                'size': fsize_title_px
              , 'family': font_type
              , 'color': linecolor
            }
          , 'text': 'PC1'
        }
      , ticks='outside'
      , ticklen=0
      , tickwidth=0
      , tickfont={
            'family': font_type
          , 'size': fsize_ticks_px
          , 'color': linecolor
        }
      , range=plot_specs['range_x']
      , showgrid=False
      , showline=False
      , linewidth=axislinewidth
      , linecolor=linecolor
      , showticklabels=False
      , visible=False
      , row=1, col=1
    )

    fig.update_yaxes(
        title={
            'font': {
                'size': fsize_title_px
              , 'family': font_type
              , 'color': linecolor
            }
          , 'text':'PC2'
        }
      , ticks='outside'
      , ticklen=0
      , tickwidth=0
      , tickfont={
            'family': font_type
          , 'size': fsize_ticks_px
          , 'color': linecolor
          }
      , range=plot_specs['range_y']
      , showgrid=False
      , showline=False
      , linewidth=axislinewidth
      , linecolor=linecolor
      , showticklabels=False
      , visible=False
      , row=1, col=1
    )


    fig.add_shape(
        type='line'
      , x0=plot_specs['range_x'][0]
      , y0=0
      , x1=plot_specs['range_x'][1]
      , y1=0
      , layer="below"
      , line={'color': linecolor, 'width': axislinewidth}
      , row=1, col=1
    )

    fig.add_shape(
        type='line'
      , x0=0
      , y0=plot_specs['range_y'][0]
      , x1=0
      , y1=plot_specs['range_y'][1]
      , layer="below"
      , line={'color': 'black', 'width': axislinewidth}
      , row=1, col=1
    )

    # cluster boxplot 1

    ## cluster order changes based on order being straight or reversed
    if order=='straight':
        cluster_a = 1
        celltypes_a = 'top_conn1_C1'
        color_temp_a = colors[0]
        label_a = type_names[0]
        cluster_b = 2
        celltypes_b = 'top_conn1_C2'
        color_temp_b = colors[1]
        label_b = type_names[1]
    else :
        cluster_a = 2
        celltypes_a = 'top_conn1_C2'
        color_temp_a = colors[1]
        label_a = type_names[1]
        cluster_b = 1
        celltypes_b = 'top_conn1_C1'
        color_temp_b = colors[0]
        label_b = type_names[0]

    data = top_connection_clusters_df.loc[clusters_bids[cluster_a],celltypes_a].values
    x_jitter = np.random.uniform(-jitter_extent, jitter_extent, size=len(data))
    max_y11 = max(data)
    x_center = x_centers[0]

    fig.add_trace(
        go.Scatter(
            x=np.full(len(data), x_center) + x_jitter  # Center x at 1 with jitter
          , y=data
          , name=label_a
          , showlegend=False
          , opacity=opacity
          , mode='markers'
          , marker={
                'size': markersize
              , 'color': color_temp_a
              , 'line': {
                    'width': markerlinewidth
                  , 'color': markerlinecolor
                }
            }
        )
      , row=1, col=3
    )

    fig.add_shape(
        type='line'
      , x0=x_center-x_deviation
      , y0=np.median(data)
      , x1=x_center+x_deviation
      , y1=np.median(data)
      , opacity=opacity
      , line={'color': color_temp_a, 'width': 1}
      , row=1, col=3
    )

    data = top_connection_clusters_df.loc[clusters_bids[cluster_b],celltypes_b].values
    x_jitter = np.random.uniform(-jitter_extent, jitter_extent, size=len(data))
    max_y12 = max(data)
    x_center = x_centers[1]

    fig.add_trace(
        go.Scatter(
            x=np.full(len(data), x_center) + x_jitter  # Center x at 1 with jitter
          , y=data
          , name=label_b
          , showlegend=False
          , opacity=opacity
          , mode='markers'
          , marker={
                'size': markersize
              , 'color': color_temp_b
              , 'line': {
                    'width': markerlinewidth
                  , 'color': markerlinecolor
                }
            }
        )
      , row=1, col=3
    )

    fig.add_shape(
        type='line'
      , x0=x_center-x_deviation
      , y0=np.median(data)
      , x1=x_center+x_deviation
      , y1=np.median(data)
      , opacity=opacity
      , line={'color': color_temp_b, 'width': 1}
      , row=1, col=3
    )

    # determining range of x and y axis:
    y1_max = max(max_y11,max_y12)
    y1_max = y1_max+15

    fig.update_xaxes(
        title={
            'font': {
                'size': fsize_title_px
              , 'family': font_type
              , 'color': linecolor
            }
        }
      , ticks='outside'
      , ticklen=ticklen
      , tickwidth=tickwidth
      , tickfont={
            'family': font_type
          , 'size': fsize_ticks_px
          , 'color': linecolor
          }
      , tickvals=[x_centers[0],x_centers[1]]
      , showticklabels=False
      , showgrid=False
      , showline=True
      , linewidth=axislinewidth
      , linecolor=linecolor
      , scaleratio=1
      , row=1, col=3
    )

    fig.update_yaxes(
        title={
            'font': {
                'size': fsize_title_px
              , 'family': font_type
              , 'color': linecolor
            }
          , 'text': '# connections'
          , 'standoff': 0
        }
      , ticks='outside'
      , ticklen=ticklen
      , tickwidth=tickwidth
      , tickfont={
          'family': font_type
          ,'size': fsize_ticks_px
          ,'color': linecolor
          }
      , tickvals=[0, y1_max]
      , ticktext=[0, y1_max]
      , showline=True
      , linewidth=axislinewidth
      , linecolor=linecolor
      , range=[-5,y1_max]
      , row=1, col=3
    )

    # cluster histogram 2
    ## cluster order changes based on order being straight or reversed
    if order=='straight':
        celltypes_a = 'top_conn2_C1'
        celltypes_b = 'top_conn2_C2'
    elif order=='reversed':
        celltypes_a = 'top_conn2_C2'
        celltypes_b = 'top_conn2_C1'

    data = top_connection_clusters_df.loc[clusters_bids[cluster_a],celltypes_a].values
    x_jitter = np.random.uniform(-jitter_extent, jitter_extent, size=len(data))
    max_y21 = max(data)
    x_center = x_centers[0]

    fig.add_trace(
        go.Scatter(
            x=np.full(len(data), x_center) + x_jitter  # Center x at 1 with jitter
          , y=data
          , name=label_a
          , showlegend=False
          , opacity=opacity
          , mode='markers'
          , marker={
                'size': markersize
              , 'color': color_temp_a
              , 'line': {
                    'width': markerlinewidth
                  , 'color': markerlinecolor
                }
            }
        )
      , row=2, col=3
    )

    fig.add_shape(
        type='line'
      , x0=x_center-x_deviation
      , y0=np.median(data)
      , x1=x_center+x_deviation
      , y1=np.median(data)
      , opacity=opacity
      , line={'color': color_temp_a, 'width': 1}
      , row=2, col=3)

    data = top_connection_clusters_df.loc[clusters_bids[cluster_b],celltypes_b].values
    x_jitter = np.random.uniform(-jitter_extent, jitter_extent, size=len(data))
    max_y22 = max(data)
    x_center = x_centers[1]

    fig.add_trace(
        go.Scatter(
            x=np.full(len(data), x_center) + x_jitter  # Center x at 1 with jitter
          , y=data
          , name=label_b
          , showlegend=False
          , opacity=opacity
          , mode='markers'
          , marker={
                'size': markersize
              , 'color': color_temp_b
              , 'line': {
                    'width': markerlinewidth
                  , 'color': markerlinecolor
                }
            }
        )
      , row=2, col=3
    )

    fig.add_shape(
        type='line'
      , x0=x_center-x_deviation
      , y0=np.median(data)
      , x1=x_center+x_deviation
      , y1=np.median(data)
      , opacity=opacity
      , line={'color': color_temp_b, 'width': 1}
      , row=2, col=3
    )

    # determining range of x and y axis:
    y2_max = max(max_y21,max_y22)
    y2_max = y2_max+15

    fig.update_xaxes(
        title={
            'font': {
                'size': fsize_title_px
              , 'family': font_type
              , 'color': linecolor
            }
        }
      , ticks='outside'
      , ticklen=ticklen
      , tickwidth=tickwidth
      , tickfont={
          'family': font_type
          ,'size': fsize_ticks_px
          ,'color': linecolor
          }
      , tickvals=[x_centers[0],x_centers[1]]
      , showticklabels=False
      , showgrid=False
      , showline=True
      , linewidth=axislinewidth
      , linecolor=linecolor
      , scaleratio=1
      , row=2, col=3
    )

    fig.update_yaxes(
        title={
            'font': {
                'size': fsize_title_px
              , 'family': font_type
              , 'color': linecolor
            }
          , 'text': '# connections'
          , 'standoff': 0
        }
      , ticks='outside'
      , ticklen=ticklen
      , tickwidth=tickwidth
      , tickfont={
          'family': font_type
          ,'size': fsize_ticks_px
          ,'color': linecolor
          }
      , tickvals=[0, y2_max]
      , ticktext=[0, y2_max]
      , showline=True
      , linewidth=axislinewidth
      , linecolor=linecolor
      , range=[-5, y2_max]
      , row=2, col=3
    )

    # cluster histogram 3
    ## cluster order changes based on order being straight or reversed
    if order=='straight':
        celltypes_a = 'top_conn3_C1'
        celltypes_b = 'top_conn3_C2'
    elif order=='reversed':
        celltypes_a = 'top_conn3_C2'
        celltypes_b = 'top_conn3_C1'

    data = top_connection_clusters_df.loc[clusters_bids[cluster_a],celltypes_a].values
    x_jitter = np.random.uniform(-jitter_extent, jitter_extent, size=len(data))
    max_y31 = max(data)
    x_center = x_centers[0]

    fig.add_trace(
        go.Scatter(
            x=np.full(len(data), x_center) + x_jitter  # Center x at 1 with jitter
          , y=data
          , name=label_a
          , showlegend=False
          , opacity=opacity
          , mode='markers'
          , marker={
                'size': markersize
              , 'color': color_temp_a
              , 'line': {
                    'width': markerlinewidth
                  , 'color': markerlinecolor
                }
            }
        )
      , row=1, col=4
    )

    fig.add_shape(
        type='line'
      , x0=x_center-x_deviation
      , y0=np.median(data)
      , x1=x_center+x_deviation
      , y1=np.median(data)
      , opacity=opacity
      , line={'color': color_temp_a, 'width': 1}
      , row=1, col=4)

    data = top_connection_clusters_df.loc[clusters_bids[cluster_b],celltypes_b].values
    x_jitter = np.random.uniform(-jitter_extent, jitter_extent, size=len(data))
    max_y32 = max(data)
    x_center = x_centers[1]

    fig.add_trace(
        go.Scatter(
            x=np.full(len(data), x_center) + x_jitter  # Center x at 1 with jitter
          , y=data
          , name=label_b
          , showlegend=False
          , opacity=opacity
          , mode='markers'
          , marker={
                'size': markersize
              , 'color': color_temp_b
              , 'line': {
                    'width': markerlinewidth
                  , 'color': markerlinecolor
                }
            }
        )
      , row=1, col=4
    )

    fig.add_shape(
        type='line'
      , x0=x_center - x_deviation
      , y0=np.median(data)
      , x1=x_center + x_deviation
      , y1=np.median(data)
      , opacity=opacity
      , line={'color': color_temp_b, 'width': 1}
      , row=1, col=4
    )

    # determining range of x and y axis:
    y3_max = max(max_y31,max_y32)
    y3_max = y3_max+15

    fig.update_xaxes(
        title={
            'font': {
                'size': fsize_title_px
              , 'family': font_type
              , 'color': linecolor
            }
        }
      , ticks='outside'
      , ticklen=ticklen
      , tickwidth=tickwidth
      , tickfont={
            'family': font_type
          , 'size': fsize_ticks_px
          , 'color': linecolor
        }
      , tickvals=[x_centers[0],x_centers[1]]
      , showgrid=False
      , showline=True
      , linewidth=axislinewidth
      , linecolor=linecolor
      , scaleratio=1
      , showticklabels=False
      , row=1, col=4
    )

    fig.update_yaxes(
        title={
            'font': {
                'size': fsize_title_px
              , 'family': font_type
              , 'color': linecolor
            }
        }
      , ticks='outside'
      , ticklen=ticklen
      , tickwidth=tickwidth
      , tickfont={
            'family': font_type
          , 'size': fsize_ticks_px
          , 'color': linecolor
        }
      , tickvals=[0, y3_max]
      , ticktext=[0, y3_max]
      , showline=True
      , linewidth=axislinewidth
      , linecolor=linecolor
      , range=[-5, y3_max]
      , row=1, col=4
    )

    # cluster histogram 4
    ## cluster order changes based on order being straight or reversed
    if order=='straight':
        celltypes_a = 'top_conn4_C1'
        celltypes_b = 'top_conn4_C2'
    elif order=='reversed':
        celltypes_a = 'top_conn4_C2'
        celltypes_b = 'top_conn4_C1'

    data = top_connection_clusters_df.loc[clusters_bids[cluster_a],celltypes_a].values
    x_jitter = np.random.uniform(-jitter_extent, jitter_extent, size=len(data))
    max_y41 = max(data)
    x_center = x_centers[0]

    fig.add_trace(
        go.Scatter(
            x=np.full(len(data), x_center) + x_jitter  # Center x at 1 with jitter
          , y=data
          , name=label_a
          , showlegend=False
          , opacity=opacity
          , mode='markers'
          , marker={
                'size': markersize
              , 'color': color_temp_a
              , 'line': {
                    'width': markerlinewidth
                  , 'color': markerlinecolor
                }
          }
        )
      , row=2, col=4
    )

    fig.add_shape(
        type='line'
      , x0=x_center-x_deviation
      , y0=np.median(data)
      , x1=x_center+x_deviation
      , y1=np.median(data)
      , opacity=opacity
      , line={'color': color_temp_a, 'width': 1}
      , row=2, col=4)

    data = top_connection_clusters_df.loc[clusters_bids[cluster_b],celltypes_b].values
    x_jitter = np.random.uniform(-jitter_extent, jitter_extent, size=len(data))
    max_y42 = max(data)
    x_center = x_centers[1]

    fig.add_trace(
        go.Scatter(
            x=np.full(len(data), x_center) + x_jitter  # Center x at 1 with jitter
          , y=data
          , name=label_b
          , showlegend=False
          , opacity=opacity
          , mode='markers'
          , marker={
                'size': markersize
              , 'color': color_temp_b
              , 'line': {
                    'width': markerlinewidth
                  , 'color': markerlinecolor
                }
            }
        )
      , row=2, col=4)

    fig.add_shape(
        type='line'
      , x0=x_center-x_deviation
      , y0=np.median(data)
      , x1=x_center+x_deviation
      , y1=np.median(data)
      , opacity=opacity
      , line={'color': color_temp_b, 'width': 1}
      , row=2, col=4)

    # determining range of x and y axis:
    y4_max = max(max_y41,max_y42)
    y4_max = y4_max+15

    fig.update_xaxes(
        title={
            'font': {
                'size': fsize_title_px
              , 'family': font_type
              , 'color': linecolor
            }
        }
      , ticks='outside'
      , ticklen=ticklen
      , tickwidth=tickwidth
      , tickfont={
            'family': font_type
          , 'size': fsize_ticks_px
          , 'color': linecolor
        }
      , tickvals=[x_centers[0], x_centers[1]]
      , showticklabels=False
      , showgrid=False
      , showline=True
      , linewidth=axislinewidth
      , linecolor=linecolor
      , scaleratio=1
      , row=2, col=4
    )

    fig.update_yaxes(
        title={
            'font': {
                'size': fsize_title_px
              , 'family': font_type
              , 'color': linecolor
            }
        }
      , ticks='outside'
      , ticklen=ticklen
      , tickwidth=tickwidth
      , tickfont={
          'family': font_type
          ,'size': fsize_ticks_px
          ,'color': linecolor
          }
      , tickvals=[0, y4_max]
      , ticktext=[0, y4_max]
      , showline=True
      , linewidth=axislinewidth
      , linecolor=linecolor
      , range=[-5, y4_max]
      , row=2, col=4
    )

    fig.update_layout(
        autosize=False
      , width=fig_width
      , height=fig_height
      , legend={
            'x': -0.2
          , 'y': -0.23
          , 'bgcolor': 'rgba(0,0,0,0)'
          , 'font': {'size':5}
        }  # Set the border color
      , paper_bgcolor='rgba(255,255,255,1)'
      , plot_bgcolor='rgba(255,255,255,1)'
    )

    fig.update_annotations(font={'family': font_type, 'size': 6})

    save_path = plot_specs['save_path']
    save_path.mkdir(parents=True, exist_ok=True)

    pio.write_image(
        fig
      , save_path / f"{type_name}.{export_type}"
      , width=fig_width
      , height=fig_height
    )

    return fig


def plot_spatialmap_three_clusters(
    main_conn_celltypes_new:list
  , synapses_for_plotting_colored:pd.DataFrame
  , color_legend_mapping:dict
  , marker_size_sf:int
  , top_connection_clusters_df:pd.DataFrame
  , clusters_bids:dict
  , type_names: list
  , type_name: str
  , rois:list
  , colors:list
  , style:dict
  , sizing:dict
  , plot_specs:dict
) -> go.Figure:
    """
    This function plots the spatial map along the two principal components of the locations of
    the synapses across the eye as well as the distribution of #connections made by the three
    clusters of a cell type with some example cell types

    Parameters
    ----------
    main_conn_celltypes_new: list
        list of two input and two output celltypes that connect to the bodyIds within all three
        clusters of the given celltype
    synapses_for_plotting_colored : pd.DataFrame
        dataframe containing all of the connectivity information for the different clusters
    color_legend_mapping : dict
        dictionary that instructs the color mapping of the data points to the legend
    marker_size_sf : int
        scaling factor of the marker size: different spatial maps are scaled differentially
        depending on how crowded or sparse the synapses are
    top_connection_clusters_df: pd.dataframe
        dataframe containing the connectivity information of the chosen example cell types with
        the clusters of the cell type under consideration
    clusters_bids: dict
        dictionary containing the mapping between bodyIds and cluster identity
    type_names: str
        names of existing clusters or clusters that the celltype needs to be divided into
    type_name : str
        grouped name for all three clusters together
    rois : list
        list of rois within which the synapses of the given celltype are located
    colors: list
        list of colors for the three clusters
    style : dict
        dict containing the values of the fixed styling formatting variables
    sizing : dict
        dict containing the values of the size formatting variables
    plot_specs : dict
        dict containing the values of the formatting variables relevant to the specific plot

    Returns
    -------
        fig : go.Figure
            formatted plotly scatterplot
    """

    export_type = style['export_type']
    font_type = style['font_type']
    markerlinecolor = style['markerlinecolor']
    linecolor = style['linecolor']
    opacity = style['opacity']
    opacity_spatialmap = style['opacity_spatialmap']
    jitter_extent_3c = style['jitter_extent_3C']
    x_centers = style['x_centers_three_clusters']
    x_deviation = style['x_deviation_three_clusters']

    # get sizing values
    markersize = sizing['markersize']
    ticklen = sizing['ticklen']
    tickwidth = sizing['tickwidth']
    axislinewidth = sizing['axislinewidth']
    markerlinewidth = sizing['markerlinewidth']
    markerlinewidth_spatialmap = sizing['markerlinewidth_spatialmap']

    pixelsperinch = 96
    pixelspermm = pixelsperinch / 25.4
    fig_width = (sizing['fig_width'] - sizing['fig_margin']) * pixelspermm
    fig_height = (sizing['fig_height'] - sizing['fig_margin']) * pixelspermm
    fsize_ticks_px = sizing['fsize_ticks_pt'] * (1/72) * pixelsperinch
    fsize_title_px = sizing['fsize_title_pt'] * (1/72) * pixelsperinch

    fsize_ticks_px = 5
    fsize_title_px = 6

    fig = make_subplots(
        rows=2, cols=4
      , specs=[[{"colspan": 2, "rowspan": 2}, None, {}, {}]
      , [None, None, {}, {}]]
      , print_grid=True
      , subplot_titles=(
            ''
          , f"input <br>{main_conn_celltypes_new[0]}"
          , f"output <br>{main_conn_celltypes_new[2]}"
          , main_conn_celltypes_new[1]
          , main_conn_celltypes_new[3]
        )
      , column_widths=[0.3, 0.3, 0.2, 0.2]
      , row_heights=[0.5, 0.5]
      , horizontal_spacing=0.09
      , vertical_spacing=0.13
    )

    ## spatial map
    for color in synapses_for_plotting_colored['color'].unique():
        filtered_df = synapses_for_plotting_colored[
            synapses_for_plotting_colored['color'] == color
        ]
        if rois==['LO(R)']:
            filtered_df['Y'] = -filtered_df['Y']
        trace = go.Scatter(
            x=-filtered_df['Y']
          , y=filtered_df['X']
          , name=color_legend_mapping[color]
          , hovertext = filtered_df['bodyId']
          , opacity=opacity_spatialmap
          , showlegend=True
          , mode='markers'
          , marker={
                'size':marker_size_sf*markersize
              , 'color': color
              , 'line': {
                    'width': markerlinewidth_spatialmap
                  , 'color': markerlinecolor
                }
            }
        )
        fig.add_trace(trace, row=1, col=1)

    fig.update_xaxes(
        title={
            'font': {
                'size': fsize_title_px
              , 'family': font_type
              , 'color': linecolor
            }
        }
      , ticks='outside'
      , ticklen=0
      , tickwidth=0
      , tickfont={
          'family': font_type
          , 'size': fsize_ticks_px
          ,'color': linecolor
          }
      , range=plot_specs['range_x']
      , showgrid=False
      , showline=False
      , linewidth=axislinewidth
      , linecolor=linecolor
      , showticklabels=False
      , visible=False
      , row=1, col=1
    )

    fig.update_yaxes(
        title={
            'font': {
                'size': fsize_title_px
              , 'family': font_type
              , 'color': linecolor
            }
        }
      , ticks='outside'
      , ticklen=0
      , tickwidth=0
      , tickfont={
            'family': font_type
          , 'size': fsize_ticks_px
          , 'color': linecolor
        }
      , range=plot_specs['range_y']
      , showgrid=False
      , showline=False
      , linewidth=axislinewidth
      , linecolor=linecolor
      , showticklabels=False
      , visible=False
      , row=1, col=1
    )


    fig.add_shape(
        type='line'
      , x0=plot_specs['range_x'][0]
      , y0=0
      , x1=plot_specs['range_x'][1]
      , y1=0
      , layer="below"
      , line={'color': linecolor, 'width':axislinewidth}
      , row=1, col=1
    )


    fig.add_shape(
        type='line'
      , x0=0
      , y0=plot_specs['range_y'][0]
      , x1=0
      , y1=plot_specs['range_y'][1]
      , layer="below"
      , line={'color': 'black', 'width': axislinewidth}
      , row=1, col=1
    )

    ## cluster boxplot 1

    data = top_connection_clusters_df.loc[clusters_bids[CLUSTER1],'top_conn1_C1'].values
    x_jitter = np.random.uniform(-jitter_extent_3c, jitter_extent_3c, size=len(data))
    max_y11 = max(data)
    x_center = x_centers[0]

    fig.add_trace(
        go.Scatter(
            x=np.full(len(data), x_center) + x_jitter  # Center x at 1 with jitter
          , y=data
          , name=type_names[0]
          , showlegend=False
          , opacity=opacity
          , mode='markers'
          , marker={
                'size': markersize
              , 'color': colors[0]
              , 'line': {
                    'width': markerlinewidth
                  , 'color': markerlinecolor
                }
            }
        )
      , row=1, col=3
    )

    fig.add_shape(
        type='line'
      , x0=x_center - x_deviation
      , y0=np.median(data)
      , x1=x_center + x_deviation
      , y1=np.median(data)
      , opacity=opacity
      , line={'color': colors[0], 'width': 1}
      , row=1, col=3
    )

    data = top_connection_clusters_df.loc[clusters_bids[CLUSTER3],'top_conn1_C3'].values
    x_jitter = np.random.uniform(-jitter_extent_3c, jitter_extent_3c, size=len(data))
    max_y12 = max(data)
    x_center = x_centers[1]

    fig.add_trace(
        go.Scatter(
            x=np.full(len(data), x_center) + x_jitter  # Center x at 1 with jitter
          , y=data
          , name=type_names[2]
          , showlegend=False
          , opacity=opacity
          , mode='markers'
          , marker={
                'size': markersize
              , 'color': colors[2]
              , 'line': {
                    'width': markerlinewidth
                  , 'color': markerlinecolor
                }
            }
        )
      , row=1, col=3
    )

    fig.add_shape(
        type='line'
      , x0=x_center-x_deviation
      , y0=np.median(data)
      , x1=x_center+x_deviation
      , y1=np.median(data)
      , opacity=opacity
      , line={'color': colors[2], 'width': 1}
      , row=1, col=3
    )

    data = top_connection_clusters_df.loc[clusters_bids[CLUSTER2],'top_conn1_C2'].values
    x_jitter = np.random.uniform(-jitter_extent_3c, jitter_extent_3c, size=len(data))
    max_y13 = max(data)
    x_center = x_centers[2]

    fig.add_trace(
        go.Scatter(
            x=np.full(len(data), x_center) + x_jitter  # Center x at 1 with jitter
          , y=data
          , name=type_names[1]
          , showlegend=False
          , opacity=opacity
          , mode='markers'
          , marker={
                'size': markersize
              , 'color': colors[1]
              , 'line': {
                    'width': markerlinewidth
                  , 'color': markerlinecolor
                }
            }
        )
      , row=1, col=3)

    fig.add_shape(
        type='line'
      , x0=x_center-x_deviation
      , y0=np.median(data)
      , x1=x_center+x_deviation
      , y1=np.median(data)
      , opacity=opacity
      , line={'color': colors[1], 'width': 1}
      , row=1, col=3
    )

    # determining range of x and y axis:
    y1_max = max(max_y11,max_y12,max_y13)
    y1_max = y1_max+15

    fig.update_xaxes(
        title={
            'font': {
                'size': fsize_title_px
              , 'family': font_type
              , 'color': linecolor
            }
        }
      , ticks='outside'
      , ticklen=ticklen
      , tickwidth=tickwidth
      , tickfont={
            'family': font_type
          , 'size': fsize_ticks_px
          , 'color': linecolor
        }
      , tickvals=[x_centers[0], x_centers[1], x_centers[2]]
      , showticklabels=False
      , showgrid=False
      , showline=True
      , linewidth=axislinewidth
      , linecolor=linecolor
      , scaleratio=1
      , row=1, col=3
    )

    fig.update_yaxes(
        title={
            'font': {
                'size': fsize_title_px
              , 'family': font_type
              , 'color': linecolor
            }
          , 'text': '# connections'
          , 'standoff': 0
        }
      , ticks='outside'
      , ticklen=ticklen
      , tickwidth=tickwidth
      , tickfont={
            'family': font_type
          , 'size': fsize_ticks_px
          , 'color': linecolor
        }
      , tickvals=[0, y1_max]
      , ticktext=[0, y1_max]
      , showline=True
      , linewidth=axislinewidth
      , linecolor=linecolor
      , range=[-5,y1_max]
      , row=1, col=3
    )

    # cluster histogram 2

    data = top_connection_clusters_df.loc[clusters_bids[CLUSTER1],'top_conn2_C1'].values
    x_jitter = np.random.uniform(-jitter_extent_3c, jitter_extent_3c, size=len(data))
    max_y21 = max(data)
    x_center = x_centers[0]

    fig.add_trace(
        go.Scatter(
            x=np.full(len(data), x_center) + x_jitter  # Center x at 1 with jitter
          , y=data
          , name=type_names[0]
          , showlegend=False
          , opacity=opacity
          , mode='markers'
          , marker={
                'size':markersize
              , 'color': colors[0]
              , 'line': {
                    'width': markerlinewidth
                  , 'color': markerlinecolor
                }
          }
        )
      , row=2, col=3
    )

    fig.add_shape(
        type='line'
      , x0=x_center-x_deviation
      , y0=np.median(data)
      , x1=x_center+x_deviation
      , y1=np.median(data)
      , opacity=opacity
      , line={'color': colors[0], 'width': 1}
      , row=2, col=3
    )

    data = top_connection_clusters_df.loc[clusters_bids[CLUSTER3],'top_conn2_C3'].values
    x_jitter = np.random.uniform(-jitter_extent_3c, jitter_extent_3c, size=len(data))
    max_y22 = max(data)
    x_center = x_centers[1]

    fig.add_trace(
        go.Scatter(
            x=np.full(len(data), x_center) + x_jitter  # Center x at 1 with jitter
          , y=data
          , name=type_names[2]
          , showlegend=False
          , opacity=opacity
          , mode='markers'
          , marker={
                'size': markersize
              , 'color': colors[2]
              , 'line': {
                    'width': markerlinewidth
                  , 'color': markerlinecolor
                }
            }
        )
      , row=2, col=3
    )

    fig.add_shape(
        type='line'
      , x0=x_center-x_deviation
      , y0=np.median(data)
      , x1=x_center+x_deviation
      , y1=np.median(data)
      , opacity=opacity
      , line={'color': colors[2], 'width': 1}
      , row=2, col=3
    )

    data = top_connection_clusters_df.loc[clusters_bids[CLUSTER2],'top_conn2_C2'].values
    x_jitter = np.random.uniform(-jitter_extent_3c, jitter_extent_3c, size=len(data))
    max_y23 = max(data)
    x_center = x_centers[2]

    fig.add_trace(
        go.Scatter(
            x=np.full(len(data), x_center) + x_jitter  # Center x at 1 with jitter
          , y=data
          , name=type_names[1]
          , showlegend=False
          , opacity=opacity
          , mode='markers'
          , marker={
                'size': markersize
              , 'color': colors[1]
              , 'line': {
                    'width': markerlinewidth
                  , 'color': markerlinecolor
                }
            }
        )
      , row=2, col=3
    )

    fig.add_shape(
        type='line'
      , x0=x_center-x_deviation
      , y0=np.median(data)
      , x1=x_center+x_deviation
      , y1=np.median(data)
      , opacity=opacity
      , line={'color': colors[1], 'width': 1}
      , row=2, col=3
    )

    # determining range of x and y axis:
    y2_max = max(max_y21,max_y22,max_y23)
    y2_max = y2_max+15

    fig.update_xaxes(
        title={
            'font': {
                'size': fsize_title_px
              , 'family': font_type
              , 'color': linecolor
            }
        }
      , ticks='outside'
      , ticklen=ticklen
      , tickwidth=tickwidth
      , tickfont={
            'family': font_type
          , 'size': fsize_ticks_px
          , 'color': linecolor
        }
      , tickvals=[x_centers[0], x_centers[1], x_centers[2]]
      , showticklabels=False
      , showgrid=False
      , showline=True
      , linewidth=axislinewidth
      , linecolor=linecolor
      , scaleratio=1
      , row=2, col=3
    )

    fig.update_yaxes(
        title={
            'font': {
                'size': fsize_title_px
              , 'family': font_type
              , 'color': linecolor
            }
          , 'text': '# connections'
          , 'standoff': 0
        }
      , ticks='outside'
      , ticklen=ticklen
      , tickwidth=tickwidth
      , tickfont={
            'family': font_type
          , 'size': fsize_ticks_px
          , 'color': linecolor
        }
      , tickvals=[0, y2_max]
      , ticktext=[0, y2_max]
      , showline=True
      , linewidth=axislinewidth
      , linecolor=linecolor
      , range=[-5,y2_max]
      , row=2, col=3
    )

    # cluster histogram 3

    data = top_connection_clusters_df.loc[clusters_bids[CLUSTER1],'top_conn3_C1'].values
    x_jitter = np.random.uniform(-jitter_extent_3c, jitter_extent_3c, size=len(data))
    max_y31 = max(data)
    x_center = x_centers[0]

    fig.add_trace(
        go.Scatter(
            x=np.full(len(data), x_center) + x_jitter  # Center x at 1 with jitter
          , y=data
          , name=type_names[0]
          , showlegend=False
          , opacity=opacity
          , mode='markers'
          , marker={
                'size': markersize
              , 'color': colors[0]
              , 'line': {
                    'width': markerlinewidth
                  , 'color': markerlinecolor
                }
            }
        )
      , row=1, col=4
    )

    fig.add_shape(
        type='line'
      , x0=x_center-x_deviation
      , y0=np.median(data)
      , x1=x_center+x_deviation
      , y1=np.median(data)
      , opacity=opacity
      , line={'color': colors[0], 'width': 1}
      , row=1, col=4
    )

    data = top_connection_clusters_df.loc[clusters_bids[CLUSTER3],'top_conn3_C3'].values
    x_jitter = np.random.uniform(-jitter_extent_3c, jitter_extent_3c, size=len(data))
    max_y32 = max(data)
    x_center = x_centers[1]

    fig.add_trace(
        go.Scatter(
            x=np.full(len(data), x_center) + x_jitter  # Center x at 1 with jitter
          , y=data
          , name=type_names[2]
          , showlegend=False
          , opacity=opacity
          , mode='markers'
          , marker={
                'size': markersize
              , 'color': colors[2]
              , 'line': {
                    'width': markerlinewidth
                  , 'color': markerlinecolor
                }
            }
        )
      , row=1, col=4
    )

    fig.add_shape(
        type='line'
      , x0=x_center-x_deviation
      , y0=np.median(data)
      , x1=x_center+x_deviation
      , y1=np.median(data)
      , opacity=opacity
      , line={'color': colors[2], 'width': 1}
      , row=1, col=4
    )

    data = top_connection_clusters_df.loc[clusters_bids[CLUSTER2],'top_conn3_C2'].values
    x_jitter = np.random.uniform(-jitter_extent_3c, jitter_extent_3c, size=len(data))
    max_y33 = max(data)
    x_center = x_centers[2]

    fig.add_trace(
        go.Scatter(
            x=np.full(len(data), x_center) + x_jitter  # Center x at 1 with jitter
          , y=data
          , name=type_names[1]
          , showlegend=False
          , opacity=opacity
          , mode='markers'
          , marker={
                'size': markersize
              , 'color': colors[1]
              , 'line': {
                    'width': markerlinewidth
                  , 'color': markerlinecolor
                }
            }
        )
      , row=1, col=4
    )

    fig.add_shape(
        type='line'
      , x0=x_center-x_deviation
      , y0=np.median(data)
      , x1=x_center+x_deviation
      , y1=np.median(data)
      , opacity=opacity
      , line={'color': colors[1], 'width': 1}
      , row=1, col=4
    )

    # determining range of x and y axis:
    y3_max = max(max_y31,max_y32,max_y33)
    y3_max = y3_max+15

    fig.update_xaxes(
        title={
            'font': {
                'size': fsize_title_px
              , 'family': font_type
              , 'color': linecolor
            }
        }
      , ticks='outside'
      , ticklen=ticklen
      , tickwidth=tickwidth
      , tickfont={
            'family': font_type
          , 'size': fsize_ticks_px
          , 'color': linecolor
        }
      , tickvals=[x_centers[0], x_centers[1], x_centers[2]]
      , showgrid=False
      , showline=True
      , linewidth=axislinewidth
      , linecolor=linecolor
      , scaleratio=1
      , showticklabels=False
      , row=1, col=4
    )

    fig.update_yaxes(
        title={
            'font': {
                'size': fsize_title_px
              , 'family': font_type
              , 'color': linecolor
            }
        }
      , ticks='outside'
      , ticklen=ticklen
      , tickwidth=tickwidth
      , tickfont={
            'family': font_type
          , 'size': fsize_ticks_px
          , 'color': linecolor
          }
      , tickvals=[0, y3_max]
      , ticktext=[0, y3_max]
      , showline=True
      , linewidth=axislinewidth
      , linecolor=linecolor
      , range=[-5, y3_max]
      , row=1, col=4
    )

    # cluster histogram 4

    data = top_connection_clusters_df.loc[clusters_bids[CLUSTER1],'top_conn4_C1'].values
    x_jitter = np.random.uniform(-jitter_extent_3c, jitter_extent_3c, size=len(data))
    max_y41 = max(data)
    x_center = x_centers[0]

    fig.add_trace(
        go.Scatter(
            x=np.full(len(data), x_center) + x_jitter  # Center x at 1 with jitter
          , y=data
          , name=type_names[0]
          , showlegend=False
          , opacity=opacity
          , mode='markers'
          , marker={
                'size': markersize
              , 'color': colors[0]
              , 'line': {
                    'width': markerlinewidth
                  , 'color': markerlinecolor
                }
            }
        )
      , row=2, col=4
    )

    fig.add_shape(
        type='line'
      , x0=x_center-x_deviation
      , y0=np.median(data)
      , x1=x_center+x_deviation
      , y1=np.median(data)
      , opacity=opacity
      , line={'color': colors[0], 'width': 1}
      , row=2, col=4
    )

    data = top_connection_clusters_df.loc[clusters_bids[CLUSTER3],'top_conn4_C3'].values
    x_jitter = np.random.uniform(-jitter_extent_3c, jitter_extent_3c, size=len(data))
    max_y42 = max(data)
    x_center = x_centers[1]

    fig.add_trace(
        go.Scatter(
            x=np.full(len(data), x_center) + x_jitter  # Center x at 1 with jitter
          , y=data
          , name=type_names[2]
          , showlegend=False
          , opacity=opacity
          , mode='markers'
          , marker={
                'size': markersize
              , 'color': colors[2]
              , 'line': {
                    'width': markerlinewidth
                  , 'color': markerlinecolor
                }
            }
        )
      , row=2, col=4
    )

    fig.add_shape(
        type='line'
      , x0=x_center-x_deviation
      , y0=np.median(data)
      , x1=x_center+x_deviation
      , y1=np.median(data)
      , opacity=opacity
      , line={'color': colors[2], 'width': 1}
      , row=2, col=4
    )

    data = top_connection_clusters_df\
        .loc[clusters_bids[CLUSTER2],'top_conn4_C2']\
        .values
    x_jitter = np\
        .random\
        .uniform(-jitter_extent_3c, jitter_extent_3c, size=len(data))
    max_y43 = max(data)
    x_center = x_centers[2]

    fig.add_trace(
        go.Scatter(
            x=np.full(len(data), x_center) + x_jitter  # Center x at 1 with jitter
          , y=data
          , name=type_names[1]
          , showlegend=False
          , opacity=opacity
          , mode='markers'
          , marker={
                'size': markersize
              , 'color': colors[1]
              , 'line': {
                    'width': markerlinewidth
                  , 'color': markerlinecolor
                }
            }
        )
      , row=2, col=4
    )

    fig.add_shape(
        type='line'
      , x0=x_center-x_deviation
      , y0=np.median(data)
      , x1=x_center+x_deviation
      , y1=np.median(data)
      , opacity=opacity
      , line={
            'color': colors[1]
          , 'width': 1
        }
      , row=2, col=4
    )

    # determining range of x and y axis:
    y4_max = max(max_y41,max_y42,max_y43)
    y4_max = y4_max+15

    fig.update_xaxes(
        title={
            'font': {
                'size': fsize_title_px
              , 'family': font_type
              , 'color': linecolor
            }
        }
      , ticks='outside'
      , ticklen=ticklen
      , tickwidth=tickwidth
      , tickfont={
          'family': font_type
          ,'size': fsize_ticks_px
          ,'color': linecolor
          }
      , tickvals=[x_centers[0], x_centers[1], x_centers[2]]
      , showticklabels=False
      , showgrid=False
      , showline=True
      , linewidth=axislinewidth
      , linecolor=linecolor
      , scaleratio=1
      , row=2, col=4
    )

    fig.update_yaxes(
        title={
            'font': {
                'size': fsize_title_px
              , 'family': font_type
              , 'color': linecolor
            }
        }
      , ticks='outside'
      , ticklen=ticklen
      , tickwidth=tickwidth
      , tickfont={
            'family': font_type
          ,'size': fsize_ticks_px
          ,'color': linecolor
          }
      , tickvals=[0, y4_max]
      , ticktext=[0, y4_max]
      , showline=True
      , linewidth=axislinewidth
      , linecolor=linecolor
      , range=[-5,y4_max]
      , row=2, col=4
    )

    fig.update_layout(
        autosize=False
      , width=fig_width
      , height=fig_height
      , legend={
            'x': -0.2
          , 'y': -0.3
          , 'bgcolor': 'rgba(0,0,0,0)'
          , 'font': {'size': 5}
        }  # Set the border color
      , paper_bgcolor='rgba(255,255,255,1)'
      , plot_bgcolor='rgba(255,255,255,1)'
    )

    fig.update_annotations(font={'family': font_type, 'size': 6})

    save_path = plot_specs['save_path']
    save_path.mkdir(parents=True, exist_ok=True)

    pio.write_image(
        fig
      , save_path / f"{type_name}.{export_type}"
      , width=fig_width
      , height=fig_height
    )

    return fig


def make_spatialmap_two_clusters_fig(
    rois: list
  , pca_roi: object
  , type_names: list
  , type_name: str
  , input_df: pd.DataFrame
  , output_df: pd.DataFrame
  , bid_type: dict
  , exclude_from_clustering: list
  , fragment_type_dict: dict
  , main_conn_celltypes : list
  , marker_size_sf:int
  , order: str
  , style:dict
  , sizing:dict
  , plot_specs:dict
):
    """
    Plot spatial map and a distribution of #connections for the clusters

    Parameters
    ----------
    rois : str
        optic lobe brain region that the celltypes belong to.
    pca_roi : sklearn.decomposition.PCA
       pca associated with the brain region (ex. pca_medulla vs pca_lobula)
    type_names : str
        list of cell types that need to be clustered
    type_name : str
        a common name that can be appended to all the cell types within the "type_names" list
    input_df: pd.dataFrame
        connectivity dataframe that contains all the inputs to all the cell types
    output_df: pd.dataFrame
        connectivity dataframe that contains all the output to all the cell types
    bid_type: dict
        dictionary containing the mapping between bodyIds and types.
    exclude_from_clustering: list
        all the cell types with an 'unclear' in its name
    fragment_type_dict: dict
        dictionary that maps the cell type fragment to the cell type
    main_conn_celltypes: list
        list of the main (top or major) cell types that connect to the list of cells that belong
        to "type_names"
    marker_size_sf: int
        marker size scaling factor in order to scale marker size differently for different cell
        types depending on number of cells per type, number of clusters, etc (purely for
        visualization purposes)
    order: str
        cluster order in which plots need to be made
    style: dict
        dictionary that contains styling parameters for plots
    sizing: dict
        dictionary with sizing parameters for plots
    plot_specs: dict
        dictionary for specifics for plotting and saving data

    Returns
    -------
    fig : go.Figure
        formatted spatial map + clusters figure
    """
    type_selection = list(set(
        [
            cell_type for cell_type\
                in bid_type.values()\
                if (cell_type in type_names)
        ]
    ))
    cell_list = [
      bodyId for bodyId in bid_type.keys()\
          if  bid_type[bodyId] in type_selection
    ]
    types_to_exclude = exclude_from_clustering

    connection_table = (make_in_and_output_df(
        input_df
      , output_df
      , bid_type
      , types_to_exclude=types_to_exclude
      , fragment_type_dict=fragment_type_dict
      , bids_to_use=cell_list
    ))

    number_of_clusters = 2
    row_linkage = get_row_linkage(
        connection_table
      , metric='cosine'
      , linkage_method='ward'
    )
    clusters_bids = cluster_dict_from_linkage(
        row_linkage
      , connection_table
      , threshold=number_of_clusters
      , criterion='maxclust'
    )
    clusters_cell_types = cluster_with_type_names(
        clusters_bids
      , bid_type
    )

    cells_per_cluster_by_type = make_count_table(clusters_cell_types)
    print(cells_per_cluster_by_type)

    cell_list1 = clusters_bids[1]
    bids_to_exclude = []  #list of relevant false merges
    neuron_criteria = NC(bodyId=cell_list1)
    synapse_criteria = SC(rois=rois, primary_only=False)
    combined_synapses = get_combined_synapses_with_stdev(
        neuron_criteria
      , synapse_criteria=synapse_criteria
      , pca=pca_roi
      , bids_to_exclude=bids_to_exclude
    )

    cell_list2 = clusters_bids[2]
    neuron_criteria = NC(bodyId=cell_list2)
    combined_synapses2 = get_combined_synapses_with_stdev(
        neuron_criteria
      , synapse_criteria=synapse_criteria
      , pca=pca_roi
      , bids_to_exclude=bids_to_exclude
    )

    combined_synapses['type'] = type_name + "-1"
    combined_synapses2['type'] = type_name+"-2"
    synapses_for_plotting = pd.concat([combined_synapses, combined_synapses2])
    synapses_for_plotting_2 = synapses_for_plotting.copy()
    synapses_for_plotting_2['type'] = [type_name] * len(synapses_for_plotting_2)
    synapses_for_plotting_2['facet'] = ['all'] * len(synapses_for_plotting_2)
    synapses_for_plotting['facet'] = [type_name + ' split'] * len(synapses_for_plotting)

    print('synapse dataframe generated')

    # Input
    top_conn1_c1 = connection_table.loc[clusters_bids[CLUSTER1], main_conn_celltypes[0]]
    top_conn1_c2 = connection_table.loc[clusters_bids[CLUSTER2], main_conn_celltypes[0]]

    top_conn2_c1 = connection_table.loc[clusters_bids[CLUSTER1], main_conn_celltypes[1]]
    top_conn2_c2 = connection_table.loc[clusters_bids[CLUSTER2], main_conn_celltypes[1]]

    # Output
    top_conn3_c1 = connection_table.loc[clusters_bids[CLUSTER1], main_conn_celltypes[2]]
    top_conn3_c2 = connection_table.loc[clusters_bids[CLUSTER2], main_conn_celltypes[2]]

    top_conn4_c1 = connection_table.loc[clusters_bids[CLUSTER1], main_conn_celltypes[3]]
    top_conn4_c2 = connection_table.loc[clusters_bids[CLUSTER2], main_conn_celltypes[3]]

    data = {
        'top_conn1_C1': top_conn1_c1
      , 'top_conn1_C2': top_conn1_c2
      , 'top_conn2_C1': top_conn2_c1
      , 'top_conn2_C2': top_conn2_c2
      , 'top_conn3_C1': top_conn3_c1
      , 'top_conn3_C2': top_conn3_c2
      , 'top_conn4_C1': top_conn4_c1
      , 'top_conn4_C2': top_conn4_c2
    }
    top_connection_clusters_df = pd.DataFrame(data)
    # top_connection_clusters_df.fillna(0, inplace=True)
    # cluster dataframe generated

    main_conn_celltypes_new = main_conn_celltypes
    main_conn_celltypes_new[0] = main_conn_celltypes[0].replace('-IN', '')
    main_conn_celltypes_new[1] = main_conn_celltypes[1].replace('-IN', '')
    main_conn_celltypes_new[2] = main_conn_celltypes[2].replace('-OUT', '')
    main_conn_celltypes_new[3] = main_conn_celltypes[3].replace('-OUT', '')

    if len(type_names)==1:
        if type_names==['LC14a-1']:
            type_names=[type_names[0] + '_R', type_names[0] + '_L']
        else:
            type_names=[type_names[0] + '-1', type_names[0] + '-2']

    if type_name=='Mi4_Mi9':
        c1_color = OL_COLOR.MAGENTA_AND_GREEN.rgb[0][1]
        c2_color = OL_COLOR.MAGENTA_AND_GREEN.rgb[1][1]
        colors = [c1_color, c2_color]
    elif type_name in ['Pm2', 'Cm']:
        c1_color = OL_COLOR.OL_TYPES.rgb[2][1]
        c2_color = OL_COLOR.OL_TYPES.rgb[1][1]
        colors = [c1_color, c2_color]
    else:
        c1_color = OL_COLOR.OL_TYPES.rgb[1][1]
        c2_color = OL_COLOR.OL_TYPES.rgb[2][1]
        colors = [c1_color, c2_color]

    synapses_for_plotting_new = synapses_for_plotting.reset_index()
    synapses_for_plotting_colored = add_two_colors_to_df(
        df=synapses_for_plotting_new
      , type_names=type_names
      , cluster_bid=clusters_bids
      , colors=colors
    )
    color_legend_mapping = dict(zip(
        synapses_for_plotting_colored['color'].unique()
      , synapses_for_plotting_colored['label'].unique()
    ))

    if type_name=='Tm5':
        fig = plot_only_spatialmap(
            synapses_for_plotting_colored
          , color_legend_mapping
          , marker_size_sf
          , type_name
          , style
          , sizing
          , plot_specs
        )
    else:
        fig = plot_spatialmap_two_clusters(
            main_conn_celltypes_new
          , synapses_for_plotting_colored
          , color_legend_mapping
          , marker_size_sf
          , top_connection_clusters_df
          , clusters_bids
          , type_names
          , type_name
          , rois
          , colors
          , order
          , style
          , sizing
          , plot_specs
        )
    return fig


def make_spatialmap_three_clusters_fig(
    rois: list
  , pca_roi: object
  , type_names: list
  , type_name: str
  , input_df: pd.DataFrame
  , output_df: pd.DataFrame
  , bid_type: dict
  , exclude_from_clustering: list
  , fragment_type_dict: dict
  , main_conn_celltypes: list
  , marker_size_sf:int
  , style:dict
  , sizing:dict
  , plot_specs:dict
) -> go.Figure:
    """
    Plot spatial map and a distribution of #connections for the clusters

    Parameters
    ----------
    rois : str
        optic lobe brain region that the celltypes belong to.
    pca_roi : sklearn.decomposition._pca.PCA
       pca associated with the brain region (ex. pca_medulla vs pca_lobula)
    type_names : str
        list of cell types that need to be clustered
    type_name : str
        a common name that can be appended to all the cell types within the "type_names" list
    input_df: pd.dataFrame
        connectivity dataframe that contains all the inputs to all the cell types
    output_df: pd.dataFrame
        connectivity dataframe that contains all the output to all the cell types
    bid_type: dict
        dictionary containing the mapping between bodyIds and types.
    exclude_from_clustering: list
        all the cell types with an 'unclear' in its name
    fragment_type_dict: dict
        dictionary that maps the cell type fragment to the cell type
    main_conn_celltypes: list
        list of the main (top or major) cell types that connect to the list of cells that belong
        to "type_names"
    marker_size_sf: int
        marker size scaling factor in order to scale marker size differently for different cell
        types depending on number of cells per type, number of clusters, etc (purely for
        visualization purposes)
    style: dict
        dictionary that contains styling parameters for plots
    sizing: dict
        dictionary with sizing parameters for plots
    plot_specs: dict
        dictionary for specifics for plotting and saving data

    Returns
    -------
    fig : go.Figure
        formatted spatial map + clusters figure

    """
    type_selection = list(set(
        [
          cell_type for cell_type\
              in bid_type.values()\
              if cell_type in type_names
        ]
    ))
    cell_list = [bodyId for bodyId in bid_type.keys() if  bid_type[bodyId] in type_selection]
    types_to_exclude = exclude_from_clustering

    connection_table = (make_in_and_output_df(
        input_df
      , output_df
      , bid_type
      , types_to_exclude=types_to_exclude
      , fragment_type_dict=fragment_type_dict
      , bids_to_use=cell_list
    ))

    number_of_clusters = 3
    row_linkage = get_row_linkage(
        connection_table
      , metric='cosine'
      , linkage_method='ward'
    )
    clusters_bids = cluster_dict_from_linkage(
        row_linkage
      , connection_table
      , threshold=number_of_clusters
      , criterion='maxclust'
    )
    clusters_cell_types = cluster_with_type_names(clusters_bids, bid_type)

    cells_per_cluster_by_type = make_count_table(clusters_cell_types)
    print(cells_per_cluster_by_type)

    cell_list1 = clusters_bids[1]
    bids_to_exclude = []  #list of relevant false merges
    neuron_criteria = NC(bodyId=cell_list1)
    synapse_criteria = SC(rois=rois, primary_only=False)
    combined_synapses = get_combined_synapses_with_stdev(
        neuron_criteria\
      , synapse_criteria=synapse_criteria
      , pca=pca_roi
      , bids_to_exclude=bids_to_exclude
    )

    cell_list2 = clusters_bids[2]
    neuron_criteria = NC(bodyId=cell_list2)
    combined_synapses2 =get_combined_synapses_with_stdev(
        neuron_criteria
      , synapse_criteria=synapse_criteria
      , pca=pca_roi
      , bids_to_exclude=bids_to_exclude
    )

    cell_list3 = clusters_bids[3]
    neuron_criteria = NC(bodyId=cell_list3)
    combined_synapses3 = get_combined_synapses_with_stdev(
        neuron_criteria
      , synapse_criteria=synapse_criteria
      , pca=pca_roi
      , bids_to_exclude=bids_to_exclude
    )

    combined_synapses['type'] = f"{type_name}-1"
    combined_synapses2['type'] = f"{type_name}-2"
    combined_synapses3['type'] = f"{type_name}-3"
    synapses_for_plotting = pd.concat([combined_synapses,combined_synapses2,combined_synapses3])
    synapses_for_plotting_2 = synapses_for_plotting.copy()
    synapses_for_plotting_2['type'] = [type_name]*len(synapses_for_plotting_2)
    synapses_for_plotting_2['facet'] = ['all']*len(synapses_for_plotting_2)
    synapses_for_plotting['facet'] = [type_name+' split'] * len(synapses_for_plotting)

    print('synapse dataframe generated')

    # Input
    top_conn1_c1 = connection_table.loc[clusters_bids[CLUSTER1], main_conn_celltypes[0]]
    top_conn1_c2 = connection_table.loc[clusters_bids[CLUSTER2], main_conn_celltypes[0]]
    top_conn1_c3 = connection_table.loc[clusters_bids[CLUSTER3], main_conn_celltypes[0]]

    top_conn2_c1 = connection_table.loc[clusters_bids[CLUSTER1], main_conn_celltypes[1]]
    top_conn2_c2 = connection_table.loc[clusters_bids[CLUSTER2], main_conn_celltypes[1]]
    top_conn2_c3 = connection_table.loc[clusters_bids[CLUSTER3], main_conn_celltypes[1]]

    # Output
    top_conn3_c1 = connection_table.loc[clusters_bids[CLUSTER1], main_conn_celltypes[2]]
    top_conn3_c2 = connection_table.loc[clusters_bids[CLUSTER2], main_conn_celltypes[2]]
    top_conn3_c3 = connection_table.loc[clusters_bids[CLUSTER3], main_conn_celltypes[2]]

    top_conn4_c1 = connection_table.loc[clusters_bids[CLUSTER1], main_conn_celltypes[3]]
    top_conn4_c2 = connection_table.loc[clusters_bids[CLUSTER2], main_conn_celltypes[3]]
    top_conn4_c3 = connection_table.loc[clusters_bids[CLUSTER3], main_conn_celltypes[3]]

    data = {
        'top_conn1_C1': top_conn1_c1
      , 'top_conn1_C2': top_conn1_c2
      , 'top_conn1_C3': top_conn1_c3
      , 'top_conn2_C1': top_conn2_c1
      , 'top_conn2_C2': top_conn2_c2
      , 'top_conn2_C3': top_conn2_c3
      , 'top_conn3_C1': top_conn3_c1
      , 'top_conn3_C2': top_conn3_c2
      , 'top_conn3_C3': top_conn3_c3
      , 'top_conn4_C1': top_conn4_c1
      , 'top_conn4_C2': top_conn4_c2
      , 'top_conn4_C3': top_conn4_c3
    }

    top_connection_clusters_df = pd.DataFrame(data)
    # cluster dataframe generated

    main_conn_celltypes_new = main_conn_celltypes
    main_conn_celltypes_new[0] = main_conn_celltypes[0].replace('-IN','')
    main_conn_celltypes_new[1] = main_conn_celltypes[1].replace('-IN','')
    main_conn_celltypes_new[2] = main_conn_celltypes[2].replace('-OUT','')
    main_conn_celltypes_new[3] = main_conn_celltypes[3].replace('-OUT','')

    c1_color = OL_COLOR.OL_TYPES.rgb[1][1]
    c2_color = OL_COLOR.OL_TYPES.rgb[3][1]
    c3_color = OL_COLOR.OL_TYPES.rgb[2][1]
    colors = [c1_color, c2_color, c3_color]

    synapses_for_plotting_new = synapses_for_plotting.reset_index()
    synapses_for_plotting_colored = add_three_colors_to_df(
        synapses_for_plotting_new
      , type_names
      , clusters_bids
      , colors
    )
    color_legend_mapping = dict(zip(
        synapses_for_plotting_colored['color'].unique()
      , synapses_for_plotting_colored['label'].unique()
    ))

    fig = plot_spatialmap_three_clusters(
        main_conn_celltypes_new
      , synapses_for_plotting_colored
      , color_legend_mapping
      , marker_size_sf
      , top_connection_clusters_df
      , clusters_bids
      , type_names
      , type_name
      , rois
      , colors
      , style
      , sizing
      , plot_specs
    )

    return fig
