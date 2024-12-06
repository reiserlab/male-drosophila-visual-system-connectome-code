import os
from pathlib import Path
from dotenv import find_dotenv

import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from utils.ol_color import OL_COLOR
from utils.overall_summary_queries import add_color_group
from utils.overall_summary_table_plotting_functions import \
    plot_group_summary_table\
  , plot_neuropil_group_table\
  , plot_neuropil_group_celltype_table\
  , plot_neuropil_group_cell_table


def plot_ncells_nsyn_linked(
    df:pd.DataFrame
  , xval:str
  , yval1:str
  , yval2:str
  , yval3:str
  , yval4:str
  , yval5:str
  , style:dict
  , sizing:dict
  , plot_specs:dict
) -> go.Figure:
    """
    This function plots
    (a) number of cells per cell type for the top n cell types sorted by total number of
        connections
    (b) number of pre and post connections for each of the n cell types
    (c) cumulative fraction of number of cells and number of connections

    Parameters
    ----------
    df : pd.DataFrame
        dataframe containing all of the quantified values
        contains columns 'group' and 'color' which assigns data points order and color
    xval : str
        column of 'df' to be shown on the x-axis
    yval1 : str
        column of 'df' that corresponds to number of cells per cell type/cell instance
    yval2 : str
        column of 'df' that corresponds to number of downstream connections per
        cell type/cell instance
    yval3 : str
        column of 'df' that corresponds to number of upstream connections per
        cell type/cell instance
    yval4 : str
        column of 'df' that corresponds to cumulative fraction of number of cells per
        cell type/cell instance
    yval5 : str
        column of 'df' that corresponds to cumulative fraction of number of connections per
        cell type/cell instance
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

    # preparing the dataframe
    df = df.sort_values(by='updown',ascending=False)
    total_ncell_sum = df['n_cells'].sum()
    total_nsyn_sum = df['updown'].sum()
    df = df.reset_index()

    # get styling values
    export_type=style['export_type']
    font_type=style['font_type']
    markerlinecolor=style['markerlinecolor']
    linecolor = style['linecolor']
    opacity = style['opacity']

    # get sizing values
    ticklen = sizing['ticklen']
    tickwidth = sizing['tickwidth']
    axislinewidth = sizing['axislinewidth']
    markerlinewidth=sizing['markerlinewidth']
    markersize=sizing['markersize']
    if export_type =='svg':
        pixelsperinch = 72
    else:
        pixelsperinch = 96
    pixelspermm = pixelsperinch/25.4
    w = (sizing['fig_width'] - sizing['fig_margin'])*pixelspermm
    h = (sizing['fig_height'] - sizing['fig_margin'])*pixelspermm
    fsize_ticks_px = sizing['fsize_ticks_pt']*(1/72)*pixelsperinch
    fsize_title_px = sizing['fsize_title_pt']*(1/72)*pixelsperinch

    # dictionary of colors for the main groups
    color_legend_mapping = dict(zip(df['color'].unique(), df['main_group'].unique()))

    fig = make_subplots(
        3, 1
      , shared_xaxes=True
      , vertical_spacing=0.05
      , row_heights=[0.3,0.5,0.2]
    )
    fsize_ticks_px_x = 2

    # preparing the dataframe: to contain only the first n cell types of each group or
    # without grouping
    n_celltypes_to_plot = plot_specs['n_celltypes_to_plot']
    # grouping = 1 means the plots are grouped based on main groups, but there are overall
    # number of celltypes to be shown
    if plot_specs['grouping'] == 1:
        df = df[0:n_celltypes_to_plot]
        frames = []
        for color in df['color'].unique():
            filtered_df = df[df['color']==color]
            frames.append(filtered_df)
        df = pd.concat(frames)
    # grouping = 2 means the plots are not shown by groups
    elif plot_specs['grouping'] == 2:
        df = df[0:n_celltypes_to_plot]
    else:
        print('no grouping option specified')

    # 1. number of cells per celltype vs celltype

    for color in df['color'].unique():
        filtered_df = df[df['color'] == color]
        trace = go.Scatter(
            x=filtered_df[xval]
          , y=np.log10(filtered_df[yval1])
          , name=color_legend_mapping[color]
          , hovertext=filtered_df[xval]
          , opacity=opacity
          , showlegend=True
          , mode='markers'
          , marker={
                'size': markersize
              , 'color': color
              , 'line': {
                    'width': markerlinewidth
                  , 'color': markerlinecolor
                }
            }
        )
        fig.add_trace(trace, row=1, col=1)

    ncells_log = np.log10(df[yval1])
    ncells_log_max = max(ncells_log)

     # Draw the x zeroline
    fig.add_shape(
        type='line'
      , x0=-10
      , y0=0
      , x1=n_celltypes_to_plot
      , y1=0
      , line={'color': 'black', 'width': 1}
      , row=1, col=1
    )

     # Draw lines through y axis at equal intervals
    fig.add_shape(
        type='line'
      , x0=-10
      , y0=1
      , x1=n_celltypes_to_plot
      , y1=1
      , line={'color': 'black', 'width': 1}
      , row=1, col=1
    )

    fig.add_shape(
        type='line'
      , x0=-10
      , y0=2
      , x1=n_celltypes_to_plot
      , y1=2
      , line={'color': 'black', 'width': 1}
      , row=1, col=1
    )

    fig.add_shape(
        type='line'
      , x0=-10
      , y0=3
      , x1=n_celltypes_to_plot
      , y1=3
      , line={'color': 'black', 'width': 1}
      , row=1, col=1
    )

    fig.add_shape(
        type='line'
      , x0=-10
      , y0=4
      , x1=n_celltypes_to_plot
      , y1=4
      , line={'color': 'black', 'width': 1}
      , row=1, col=1
    )

    fig.update_yaxes(
        title={
            'font': {
                'size': fsize_title_px
              , 'family': font_type
              , 'color': linecolor
            }
          , 'text': '# of cells/type'
        }
      , ticks='outside'
      , ticklen=ticklen
      , tickwidth=tickwidth
      , tickfont={
            'family': font_type
          , 'size': fsize_ticks_px
          , 'color': linecolor
          }
      , tickvals=plot_specs['tickvals_y1']
      , ticktext=plot_specs['ticktext_y1']
      , showgrid=False
      , showline=True
      , linewidth=axislinewidth
      , linecolor=linecolor
      , scaleratio = 1
      , anchor="free"
      , side="left"
      , overlaying="y"
      , range=[-1, np.ceil(ncells_log_max)]
      , row=1, col=1
    )

    # 2. number of synapses

    for color in df['color'].unique():
        filtered_df = df[df['color'] == color]
        trace = go.Scatter(
            x=filtered_df[xval]
          , y=filtered_df[yval2]
          , hovertext = filtered_df[xval]
          , hoverinfo='text'
          , opacity = opacity
          , showlegend=False
          , mode='markers'
          , marker={
                'size': 0.5 * markersize
              , 'color': color
              , 'line': {
                    'width': 0.5 * markerlinewidth
                  , 'color': color
                }
            }
        )
        fig.add_trace(trace, row=2, col=1)

    for color in df['color'].unique():
        filtered_df = df[df['color'] == color]
        trace = go.Scatter(
            x=filtered_df[xval]
          , y=-filtered_df[yval3]
          , hovertext=filtered_df[xval]
          , hoverinfo='text'
          , opacity=opacity
          , showlegend=False
          , mode='markers'
          , marker={
                'size': 0.5 * markersize
              , 'color': color
              , 'line': {
                    'width': 0.5 * markerlinewidth
                  , 'color': color
                }
            }
        )
        fig.add_trace(trace, row=2, col=1)

    for _, row in df.iterrows():
        # Extend the line alove the zero line
        fig.add_shape(
            type='line'
          , x0=row[xval]
          , y0=0
          , x1=row[xval]
          , y1=row[yval2]
          , line={'color': row['color'], 'width': axislinewidth}
          , row=2, col=1
        )

        # Extend the line below the zero line
        fig.add_shape(
            type='line'
          , x0=row[xval]
          , y0=0
          , x1=row[xval]
          , y1=-row[yval3]
          , line={'color': row['color'], 'width': axislinewidth}
          , row=2, col=1
        )

    # Draw the x zeroline
    fig.add_shape(
        type='line'
      , x0=-1
      , y0=0
      , x1=n_celltypes_to_plot
      , y1=0
      , line={'color': 'black', 'width': axislinewidth}
      , row=2, col=1
    )

    fig.update_yaxes(
        title={
            'font':{
                'size': fsize_title_px
              , 'family': font_type
              , 'color': linecolor
            }
        }
      , ticks='outside'
      , ticklen=ticklen
      , tickwidth=tickwidth
      , tickfont={
            'family':font_type
          , 'size':fsize_ticks_px
          , 'color' : linecolor
        }
      , showgrid = False
      , showline=True
      , linewidth=axislinewidth
      , linecolor=linecolor
      , scaleratio = 1
      , anchor="free"
      , side="left"
      , overlaying="y"
      , range=plot_specs['range_y2']
      , tickvals=plot_specs['tickvals_y2']
      , row=2, col=1
    )
    fig.add_annotation(
        text='input'
      , xref='paper', yref='y'
      , x=-4, y=500000
      , showarrow=False
      , font={'size': 9}
      , textangle=270
      , row=2, col=1
    )
    fig.add_annotation(
        text='output'
      , xref='paper', yref='y'
      , x=-4, y=-500000
      , showarrow=False
      , font={'size': 9}
      , textangle=270
      , row=2, col=1
    )

    # 3. cumulative fraction of cells represented by shown celltypes

    df['cell_frac'] = df['n_cells']/total_ncell_sum
    df['cum_cell'] = df['cell_frac'].cumsum()

    trace = go.Scatter(
        name='# cells'
      , x=df[xval]
      , y=df[yval4]
      , showlegend=True
      , mode='lines'
      , fill='tozeroy'
      , fillcolor='rgba(0, 0, 0, 0.2)'
      , line={'color': 'black'}
    )
    fig.add_trace(trace, row=3, col=1)

    # cumulative fraction of synapses
    df['syn_frac'] = df['updown']/total_nsyn_sum
    df['cum_syn'] = df['syn_frac'].cumsum()

    trace = go.Scatter(
        name='# synaptic connections'
      , x=df[xval]
      , y=df[yval5]
      , showlegend=True
      , mode='lines'
      , fill='tozeroy'
      , fillcolor='rgba(255, 0, 0, 0.2)'
      , line={'color': 'red'}
    )
    fig.add_trace(trace, row=3, col=1)

    fig.update_yaxes(
        title={
            'font': {
                'size': fsize_title_px
              , 'family': font_type
              , 'color': linecolor
            }
          , 'text': 'cumulative fraction'
        }
      , ticks='outside'
      , ticklen=ticklen
      , tickwidth=tickwidth
      , tickfont={
            'family': font_type
          , 'size': fsize_ticks_px
          , 'color' : linecolor
        }
      , showgrid=False
      , showline=True
      , linewidth=axislinewidth
      , linecolor=linecolor
      , scaleratio=1
      , anchor="free"
      , side="left"
      , overlaying="y"
      , range=plot_specs['range_y4']
      , tickvals=plot_specs['tickvals_y4']
      , row=3, col=1
    )

    # Draw the x zeroline
    fig.add_shape(
        type='line'
      , x0=-10
      , y0=0
      , x1=n_celltypes_to_plot
      , y1=0
      , line={'color': 'black', 'width': 1}
      , row=3, col=1
    )

    # Draw lines through the cumulative fraction plot at equal intervals
    fig.add_shape(
        type='line'
      , x0=-10
      , y0=0.25
      , x1=n_celltypes_to_plot
      , y1=0.25
      , line={'color': 'black', 'width': 1}
      , row=3, col=1
    )

    # Draw the x zeroline
    fig.add_shape(
        type='line'
      , x0=-10
      , y0=0.5
      , x1=n_celltypes_to_plot
      , y1=0.5
      , line={'color': 'black', 'width': 1}
      , row=3, col=1
    )

    # Draw the x zeroline
    fig.add_shape(
        type='line'
      , x0=-10
      , y0=0.75
      , x1=n_celltypes_to_plot
      , y1=0.75
      , line={'color': 'black', 'width': 1}
      , row=3, col=1
    )

    # Draw the x zeroline
    fig.add_shape(
        type='line'
      , x0=-10
      , y0=1
      , x1=n_celltypes_to_plot
      , y1=1
      , line={'color': 'black', 'width': 1}
      , row=3, col=1
    )

    fig.update_xaxes(
        title={
            'font':{
                'size': fsize_ticks_px_x
              , 'family': font_type
              , 'color': linecolor
            }
          , 'text': 'celltype'
        }
      , ticks='outside'
      , ticklen=ticklen
      , tickwidth=tickwidth
      , tickfont={
          'family':font_type
          , 'size':fsize_ticks_px_x
          ,'color' : linecolor
          }
      , showgrid = False
      , showline=True
      , linewidth=axislinewidth
      , linecolor=linecolor
      , row=3, col=1
    )

    fig.update_layout(
        autosize = False
      , width = w
      , height = h
      , margin={
            'l':w//6
          , 'r':w//12
          , 'b':h//15
          , 't':h//12
          , 'pad':8
      }
      , showlegend = True
      , paper_bgcolor='rgba(255,255,255,1)'
      , plot_bgcolor='rgba(255,255,255,1)'
    )

    save_path = plot_specs['save_path']
    export_type = plot_specs['export_type']
    plot_name = plot_specs['plot_name']
    save_path.mkdir(parents=True, exist_ok=True)

    fig.write_html(save_path / f"{plot_name}.html")

    pio.write_image(
        fig
      , save_path / f"{plot_name}.{export_type}"
    )
    return fig


def plot_summary_scatterplots(
    df:pd.DataFrame
  , xval:str
  , yval:str
  , star_neurons:list
  , style:dict
  , sizing:dict
  , plot_specs:dict
) -> go.Figure:
    """
    Plot scatterplot - certain neuron types are highlighted.

    Parameters
    ----------
    df : pd.DataFrame
        dataframe containing all of the quantified values
        contains columns 'group' and 'color' which assigns data points order and color
    xval : str
        column of 'df' to be shown on the x-axis
    yval : str
        column of 'df' to be shown on the y-axis
    star_neurons : list
        list of 'star neurons' to plot on top
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

    # get styling values
    export_type=style['export_type']
    font_type=style['font_type']
    markerlinecolor=style['markerlinecolor']
    linecolor = style['linecolor']
    opacity = style['opacity']

    # get sizing values
    ticklen = sizing['ticklen']
    tickwidth = sizing['tickwidth']
    axislinewidth = sizing['axislinewidth']
    markerlinewidth=sizing['markerlinewidth']
    markersize=sizing['markersize']
    if export_type =='svg':
        pixelsperinch = 72
    else:
        pixelsperinch = 96
    pixelspermm = pixelsperinch/25.4
    w = (sizing['fig_width'] - sizing['fig_margin']) * pixelspermm
    h = (sizing['fig_height'] - sizing['fig_margin']) * pixelspermm
    fsize_ticks_px = sizing['fsize_ticks_pt'] * (1/72) * pixelsperinch
    fsize_title_px = sizing['fsize_title_pt'] * (1/72) * pixelsperinch
    fsize_ticks_px = 3

    df['markersize'] = markersize
    df['markersize'] = df['markersize'].astype('float64')
    df['markerlinewidth'] = markerlinewidth

    if star_neurons:
        df.loc[df['type'].isin(star_neurons), 'markersize'] = markersize * 2
        df.loc[df['type'].isin(star_neurons), 'markerlinewidth'] = markerlinewidth * 7
        df.loc[df['type'].isin(star_neurons), 'group'] = 6

    fig = go.Figure()
    yval_max = df[yval].max()

    if yval == 'n_pre_median_conn_cells':
        # 1X
        fig.add_shape(
            type='line'
          , x0=1
          , y0=1
          , x1=yval_max
          , y1=yval_max
          , layer="below"
          , line={
                'color': 'grey'
              , 'width': 0.8
              , 'dash': 'solid'
            }
        )

        # # 5X
        fig.add_shape(
            type='line'
          , x0=1
          , y0=5
          , x1=yval_max / 5
          , y1=yval_max
          , layer="below"
          , line={
                'color': 'grey'
              , 'width':0.4
              , 'dash': 'solid'
            }
        )

        # # 0.2X
        fig.add_shape(
            type='line'
          , x0=1
          , y0=0.2
          , x1=yval_max / 0.2
          , y1=yval_max
          , layer="below"
          , line={
                'color': 'grey'
              , 'width': 0.4
              , 'dash': 'solid'
            }
        )

    gp_values = [5, 4, 3, 2, 1, 6] # order of plotting groups, group 6: star neurons
    for grp in gp_values:

        df_gp = df[df['group'] == grp]

        fig.add_trace(
            go.Scatter(
                x = df_gp[xval]
              , y = df_gp[yval]
              , hovertext = df_gp['type']
              , hoverinfo = 'text'
              , opacity = opacity
              , mode='markers'
              , marker={
                    'size': df_gp['markersize']
                  , 'color': df_gp['color']
                  , 'line': {
                        'width': df_gp['markerlinewidth']
                      , 'color': markerlinecolor
                    }
                }
            )
        )

    if plot_specs['log_x'] is True:
        typex = "log"
        tickformx = ".1r"
    else:
        typex = "-"
        tickformx = ""

    if plot_specs['log_y'] is True:
        typey = "log"
        tickformy = ".1r"
    else:
        typey = "-"
        tickformy = ""

    fig.update_xaxes(
        title={
            'font':{
                'size': fsize_title_px
                , 'family': font_type
                , 'color': linecolor
                }
            , 'text':plot_specs['xlabel']
            }
        , title_standoff = (h//4)/4
        , ticks='outside'
        , ticklen=ticklen
        , tickwidth=tickwidth
        , tickfont={
            'family': font_type
            , 'size': fsize_ticks_px
            , 'color': linecolor
            }
        , tickformat=tickformx
        , tickcolor='black'
        , tickangle=90
        , type=typex
        , showgrid=False
        , showline=True
        , linewidth=axislinewidth
        , linecolor = linecolor
        , range=plot_specs['range_x']
        , tickvals = plot_specs['tickvals_x']
        )

    fig.update_yaxes(
        title={
            'font':{
                'size': fsize_title_px
                , 'family': font_type
                , 'color': linecolor
                }
            ,'text':plot_specs['ylabel']
            }
        , title_standoff = (w//5)/5
        , ticks='outside'
        , tickcolor='black'
        , ticklen=ticklen
        , tickwidth=tickwidth
        , tickfont={
            'family': font_type
            , 'size': fsize_ticks_px
            , 'color': linecolor
            }
        , tickformat=tickformy
        , type=typey
        , showgrid=False
        , showline=True
        , linewidth=axislinewidth
        , linecolor = linecolor
        , range = plot_specs['range_y']
        , tickvals = plot_specs['tickvals_y']
        )

    fig.update_layout(
        autosize=False
        , width=w
        , height=h
        , margin={
            'l':w//8,
            'r':0,
            'b':h//4,
            't':0,
            'pad':w//30,
        }
        , showlegend=False
        , legend={'bgcolor': 'rgba(0,0,0,0)'}
        , paper_bgcolor='rgba(255,255,255,1)'
        , plot_bgcolor='rgba(255,255,255,1)'
        )

    save_path = plot_specs['save_path']
    save_path.mkdir(parents=True, exist_ok=True)

    fig.write_html(save_path / f"{xval}_versus_{yval}.html")

    pio.write_image(
        fig
      , save_path / f"{xval}_versus_{yval}.svg"
      , width=w
      , height=h
    )

    return fig


def make_circles_celltype_groups(
    types:pd.DataFrame
  , ref_circle_areas:list
  , plot_specs:dict
  , style: dict
  , sizing: dict
) -> go.Figure:
    """
    This function plots the #celltypes, #cells and #upstream and #downstream connections
    aggregated by celltype groups, (ONIN, ONCN, VPN, VCN, other)

    Parameters
    ----------
    types : pd.DataFrame
        dataframe containing all of the quantified values ,
        contains column 'color' which assigns data points color
    ref_circle_areas : list
        factor by which circles in each column have to be scaled
    plot_specs: dict
        plotting specifications
    style : dict
        dict containing the values of the fixed styling formatting variables
    sizing : dict
        dict containing the values of the size formatting variables

    Returns
    -------
    fig : go.Figure
        formatted plotly scatterplot
    """
    cache_fn = Path(find_dotenv()).parent / "cache" / "summary_plots" / "celltype_groups.pickle"
    if cache_fn.is_file():
        celltype_groups_df = pd.read_pickle(cache_fn)
    else:
        fig, celltype_groups_df = plot_group_summary_table(
            neuron_list=types
          , style=style
          , sizing=sizing
        )

    colors = pd.DataFrame(
    data={
        'groups': OL_COLOR.OL_TYPES.map.keys()
      , 'color': OL_COLOR.OL_TYPES.map.values()
    }
    )
    colors['groups'] = ['ONIN', 'ONCN', 'VPN', 'VCN', 'other']
    merge_df = pd.merge(celltype_groups_df, colors, on='groups')
    df = merge_df.drop(['groups'], axis=1)

    circle_areas = {
        'n_celltypes': ref_circle_areas[0]
      , 'n_cells': ref_circle_areas[1]
      , 'n_upstream': ref_circle_areas[2]
      , 'n_downstream': ref_circle_areas[3]
    }

    df_data = df[['n_celltypes', 'n_cells', 'n_upstream', 'n_downstream']]

    # Calculate scaling factors based on the maximum value in each column
    scaling_factors = {
        col: circle_areas[col] / df_data[col].max() for col in df_data.columns
    }

    # Create subplots for each circle
    fig = make_subplots(
        rows=df_data.shape[0]
      , cols=df_data.shape[1]
      , subplot_titles=df_data.columns
      , horizontal_spacing=0.1
    )
    df_matrix = np.zeros((df_data.shape[0], df_data.shape[1]))

    # Add traces for each circle
    for i in range(df_data.shape[0]):
        for j, col in enumerate(df_data.columns):
            circle_area = df_data.loc[i, col] * scaling_factors[col]
            circle_radius = np.sqrt(circle_area / np.pi)
            df_matrix[i,j] = circle_area
            fig.add_trace(
                go.Scatter(
                    x=[0], y=[0]
                  , mode="markers"
                  , marker={
                        'size': 2*circle_radius
                      , 'color': df['color'].loc[i]
                    }
                )
              , row=i+1, col=j+1
            )

    # Update axes
    for i in range(df_data.shape[0]):
        for j in range(df_data.shape[1]):
            fig.update_xaxes(
                ticklen=0
              , tickwidth=0
              , showgrid = False
              , showline=False
              , showticklabels=False
              , visible = False
            )

        fig.update_yaxes(
            ticklen=0
          , tickwidth=0
          , showgrid = False
          , showline=False
          , showticklabels=False
          , visible = False
        )

    fig.update_layout(
        autosize=False
      , width=500
      , height=500
      , paper_bgcolor='rgba(255,255,255,1)'
      , plot_bgcolor='rgba(255,255,255,1)'
      , showlegend = False
    )

    save_path = plot_specs['save_path']
    export_type = plot_specs['export_type']
    plot_name = plot_specs['plot_name']

    save_path.mkdir(parents=True, exist_ok=True)

    pio.write_image(
        fig
      , save_path / f"{plot_name}.{export_type}"
    )

    return fig


def make_neuropil_celltype_groups_panel(
    types:pd.DataFrame
  , threshold: int
  , ref_circle_areas:list
  , plot_specs:dict
  , style: dict
  , sizing: dict
) -> go.Figure:
    """
    This function plots the #celltypes, #cells and #synapses aggregated by brain regions. It also
    plots pie charts that show the breakdown of cell type groups (OLIN/OLCN/VPN/VCN) for cell
    types and cells (6 x 3)


    Parameters
    ----------
    types : pd.DataFrame
        dataframe containing all of the quantified values ,
        contains column 'color' which assigns data points color
    threshold: threshold fraction of synapses that needs to be crossed within a brain region
        to be assigned to that brain region
    ref_circle_areas : list
        factor by which circles in each column have to be scaled
    plot_specs : dict
        plotting specifications
    style : dict
        dict containing the values of the fixed styling formatting variables
    sizing : dict
        dict containing the values of the size formatting variables

    Returns
    -------
    fig : go.Figure
        formatted plotly scatterplot
    """
    cache_fn1 = Path(find_dotenv()).parent / "cache" / "summary_plots" / "neuropil_groups.pickle"
    if cache_fn1.is_file():
        neuropil_df = pd.read_pickle(cache_fn1)
    else:
        df = add_color_group(types)
        df = df.reset_index()
        fig, neuropil_df = plot_neuropil_group_table(df, threshold, style, sizing)

    colors = pd.DataFrame(
    data={
        'neuropil': OL_COLOR.OL_NEUROPIL.map.keys()
      , 'color': OL_COLOR.OL_NEUROPIL.map.values()
    }
    )

    neuropil_df = pd.merge(neuropil_df, colors, on='neuropil')
    neuropil_df = neuropil_df.drop(['neuropil'], axis=1)

    cache_fn2 = Path(find_dotenv()).parent / "cache" \
        / "summary_plots" / "neuropil_groups_celltypes.pickle"
    if cache_fn2.is_file():
        neuropil_groups_celltypes_df = pd.read_pickle(cache_fn2)
    else:
        df = add_color_group(types)
        df = df.reset_index()
        fig, neuropil_groups_celltypes_df = plot_neuropil_group_celltype_table(
            df=df
          , threshold=threshold
          , style=style
          , sizing=sizing
        )

    cache_fn3 = Path(find_dotenv()).parent / "cache" \
        / "summary_plots" / "neuropil_groups_cells.pickle"
    if cache_fn3.is_file():
        neuropil_groups_cells_df = pd.read_pickle(cache_fn3)
    else:
        df = add_color_group(types)
        df = df.reset_index()
        fig, neuropil_groups_cells_df = plot_neuropil_group_cell_table(
            df=df
          , threshold=threshold
          , style=style
          , sizing=sizing
        )

    circle_areas = {
        'n_celltypes': ref_circle_areas[0]
      , 'n_cells': ref_circle_areas[1]
      , 'n_upstream': ref_circle_areas[2]
      , 'n_downstream': ref_circle_areas[3]
    }

    neuropil_df_data = neuropil_df[['n_celltypes', 'n_cells', 'n_upstream', 'n_downstream']]
    # Calculate scaling factors based on the maximum value in each column
    scaling_factors = {
        col: circle_areas[col] / neuropil_df_data[col].max() for col in neuropil_df_data.columns
    }

    # Create subplots for each circle/pie chart
    fig = make_subplots(
        rows=neuropil_df_data.shape[0]
      , cols=6
      , subplot_titles=[
            "cell types", " "
          , "cells", " "
          , "up", "down"
        ]
      , specs=[
            [
                {'type':'scatter'}, {'type':'pie'}
              , {'type':'scatter'}, {'type':'pie'}
              , {'type':'scatter'}, {'type':'scatter'}
            ]
          , [
                {'type':'scatter'}, {'type':'pie'}
              , {'type':'scatter'}, {'type':'pie'}
              , {'type':'scatter'}, {'type':'scatter'}
            ]
          , [
                {'type':'scatter'}, {'type':'pie'}
              , {'type':'scatter'}, {'type':'pie'}
              , {'type':'scatter'}, {'type':'scatter'}
            ]
          , [
                {'type':'scatter'}, {'type':'pie'}
              , {'type':'scatter'}, {'type':'pie'}
              , {'type':'scatter'}, {'type':'scatter'}
            ]
          , [
                {'type':'scatter'}, {'type':'pie'}
              , {'type':'scatter'}, {'type':'pie'}
              , {'type':'scatter'},{'type':'scatter'}
            ]
        ]
    )

    # Add traces for each circle
    for i in range(neuropil_df_data.shape[0]):
        for j, col in enumerate(neuropil_df_data.columns[:-1]):
            circle_area = neuropil_df_data.loc[i, col] * scaling_factors[col]
            circle_radius = np.sqrt(circle_area / np.pi)
            fig.add_trace(
                go.Scatter(
                    x=[0], y=[0]
                  , mode="markers"
                  , marker={
                        'size': 2 * circle_radius
                      , 'color': neuropil_df['color'].loc[i]
                    }
                )
              , row=i+1, col=2*j+1
            )

    # last column of downstream connections
    for i in range(neuropil_df_data.shape[0]):
        circle_area = neuropil_df_data.loc[i, 'n_downstream'] * scaling_factors['n_downstream']
        circle_radius = np.sqrt(circle_area / np.pi)
        fig.add_trace(
            go.Scatter(
                x=[0], y=[0]
              , mode="markers"
              , marker={
                    'size': 2*circle_radius
                  , 'color': neuropil_df['color'].loc[i]
                }
            )
          , row=i+1, col=6
        )

    # pie charts
    colors = [
        OL_COLOR.OL_TYPES.rgb[0][1]
      , OL_COLOR.OL_TYPES.rgb[1][1]
      , OL_COLOR.OL_TYPES.rgb[2][1]
      , OL_COLOR.OL_TYPES.rgb[3][1]
      , OL_COLOR.OL_TYPES.rgb[4][1]
    ]

    # pie charts in the 2nd and 4th columns:
    for m in range(neuropil_groups_celltypes_df.shape[0]):
        pie_values = neuropil_groups_celltypes_df.loc[m].values[1:6]
        fig.add_trace(
            go.Pie(
                values=pie_values
              , marker={'colors': colors}
            )
          , row=m+1, col=2
        )
        fig.update_traces(
            hoverinfo='label+percent'
          , textinfo='none'
          , textfont_size=40
          , row=m+1, col=2
        )

    for n in range(neuropil_groups_cells_df.shape[0]):
        pie_values = neuropil_groups_cells_df.loc[n].values[1:6]
        fig.add_trace(
            go.Pie(
                values=pie_values
              , marker={'colors': colors}
            )
          , row=n+1, col=4
        )
        fig.update_traces(
            hoverinfo='label+percent'
          , textinfo='none'
          , textfont_size=40
          , row=n+1, col=4
        )

    # Update axes
    for i in range(neuropil_df_data.shape[0]):
        for j in range(neuropil_df_data.shape[1]):
            fig.update_xaxes(
                ticklen=0
              , tickwidth=0
              , showgrid = False
              , showline=False
              , showticklabels=False
              , visible = False
            )

            fig.update_yaxes(
                ticklen=0
              , tickwidth=0
              , showgrid = False
              , showline=False
              , showticklabels=False
              , visible = False
            )

    fig.update_layout(
        autosize=False
      , width=500
      , height=500
      , paper_bgcolor='rgba(255,255,255,1)'
      , plot_bgcolor='rgba(255,255,255,1)'
      , showlegend = False
    )

    save_path = plot_specs['save_path']
    export_type = plot_specs['export_type']
    plot_name = plot_specs['plot_name']

    save_path.mkdir(parents=True, exist_ok=True)
    pio.write_image(
        fig
      , save_path / f"{plot_name}.{export_type}"
    )

    return fig


def make_connectivity_sufficiency_scatter(
    df:pd.DataFrame
  , xval:str
  , yval1:str
  , yval2:str
  , yval3:str
  , style:dict
  , sizing:dict
  , plot_specs:dict
) -> go.Figure:
    """
    plots the fraction of cell types that can be distinguished vs number of top connections
    considered

    Parameters
    ----------
    df : pd.DataFrame
        dataframe containing all of the quantified values
        contains columns 'group' and 'color' which assigns data points order and color
    xval : str
        column of 'df' to be shown on the x-axis
    yval1, yval2, yval3 : str
        columns of 'df' to be shown on the y-axis
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
    #slightly shifted xvals
    df['n_conn_shifted_pre'] = df['n_connections']+0.1
    df['n_conn_shifted_post'] = df['n_connections']+0.2

    # get styling values
    export_type=style['export_type']
    font_type=style['font_type']
    linecolor = style['linecolor']

    # get sizing values
    ticklen = sizing['ticklen']
    tickwidth = sizing['tickwidth']
    axislinewidth = sizing['axislinewidth']
    markerlinewidth=sizing['markerlinewidth']
    markersize=sizing['markersize']
    if export_type =='svg':
        pixelsperinch = 72
    else:
        pixelsperinch = 96
    pixelspermm = pixelsperinch/25.4
    w = (sizing['fig_width'] - sizing['fig_margin'])*pixelspermm
    h = (sizing['fig_height'] - sizing['fig_margin'])*pixelspermm
    fsize_ticks_px = sizing['fsize_ticks_pt']*(1/72)*pixelsperinch
    fsize_title_px = sizing['fsize_title_pt']*(1/72)*pixelsperinch

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            name='both'
          , x = df[xval]
          , y = df[yval1]
          , hovertext = df[yval1]
          , hoverinfo = 'text'
          , opacity = 0.9
          , mode='markers+lines'
          , marker={
                'size':markersize
              , 'color': 'black'
              , 'line': {
                    'width':markerlinewidth
                  , 'color': 'black'
                }
            }
        )
    )

    fig.add_trace(
        go.Scatter(
            name='output'
          , x = df[xval]
          , y = df[yval2]
          , hovertext = df[yval2]
          , hoverinfo = 'text'
          , opacity = 0.7
          , mode='markers+lines'
          , marker={
                'size':markersize
              , 'color': OL_COLOR.OL_SYNAPSES.rgb[1][1]
              , 'line': {
                    'width':markerlinewidth
                  , 'color': OL_COLOR.OL_SYNAPSES.rgb[1][1]
                }
            }
        )
    )

    fig.add_trace(
        go.Scatter(
            name='input'
          , x = df[xval]
          , y = df[yval3]
          , hovertext = df[yval3]
          , hoverinfo = 'text'
          , opacity = 0.7
          , mode='markers+lines'
          , marker={
                'size':markersize
              , 'color': OL_COLOR.OL_SYNAPSES.rgb[0][1]
              , 'line': {
                    'width':markerlinewidth
                  , 'color': OL_COLOR.OL_SYNAPSES.rgb[0][1]
                }
            }
        )
    )

    fig.update_layout(
        autosize=False
        , width=w
        , height=h
        , margin={
            'l': w//8
          , 'r': 0 #w//12,
          , 'b': h//4
          , 't': 0 #h//12,
          , 'pad': w//30,
        }
        , showlegend = True
        , legend={
              'x': 0.5, 'y':0.2
            , 'bgcolor': 'rgba(0,0,0,0)'
            , 'font': {'size': 5}
          }  # Set the border color
        , paper_bgcolor='rgba(255,255,255,1)'
        , plot_bgcolor='rgba(255,255,255,1)'
        )

    fig.update_xaxes(
        title={
            'font':{
                'size': fsize_title_px
              , 'family': font_type
              , 'color': linecolor
            }
          , 'text':'number of top connections considered'
        }
      , title_standoff = (h//4)/4
      , ticks='outside'
      , ticklen=ticklen
      , tickwidth=tickwidth
      , tickfont={
            'family': font_type
          , 'size': fsize_ticks_px
          , 'color': linecolor
        }
      , tickcolor='black'
      , showgrid=False
      , showline=True
      , linewidth=axislinewidth
      , linecolor = linecolor
      , range=plot_specs['range_x']
      , tickvals = plot_specs['tickvals_x']
    )

    fig.update_yaxes(
        title={
            'font': {
                'size': fsize_title_px
              , 'family': font_type
              , 'color': linecolor
            }
          , 'text':'fraction of unique connections'
        }
      , title_standoff = 5
      , ticks='outside'
      , tickcolor='black'
      , ticklen=ticklen
      , tickwidth=tickwidth
      , tickfont={
            'family': font_type
          , 'size': fsize_ticks_px
          , 'color': linecolor
        }
      , showgrid=False
      , showline=True
      , linewidth=axislinewidth
      , linecolor = linecolor
      , range = plot_specs['range_y']
      , tickvals = plot_specs['tickvals_y']
    )

    save_path = plot_specs['save_path']
    export_type = plot_specs['export_type']
    plot_name = plot_specs['plot_name']
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    fig.write_html( save_path / f"{plot_name}.html")

    pio.write_image(
        fig
      , save_path / f"{plot_name}.{export_type}"
    )

    return fig
