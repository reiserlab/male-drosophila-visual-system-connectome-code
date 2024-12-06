import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd


def plot_chiasm_connectivity(
    df:pd.DataFrame
  , xval:str
  , yval1:str
  , yval2:str
  , style:dict
  , sizing:dict
  , plot_specs:dict
) -> go.Figure:
    """
    This function plots the number of non primary connections for each cell type pair with
    synapses in non primary rois

    Parameters
    ----------
    df : pd.DataFrame
         dataframe containing number of connections in all rois and in non-primary rois for each
         celltype pair
    xval : str
          column of the dataframe that contains the x axis of the plot: cell type pairs
    yval1 : str
          column of the dataframe that contains the y axis of the plot: # non-primary connections
    yval2 : str
          column of the dataframe that contains the annotation: chiasm fraction
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
    df = df.sort_values(by=yval1, ascending=False).reset_index()

    # get styling values
    export_type=style['export_type']

    # get sizing values
    if export_type =='svg':
        pixelsperinch = 72
    else:
        pixelsperinch = 96
    pixelspermm = pixelsperinch / 25.4
    figure_width = (sizing['fig_width'] - sizing['fig_margin']) * pixelspermm
    figure_height = (sizing['fig_height'] - sizing['fig_margin']) * pixelspermm
    fsize_ticks_px = sizing['fsize_ticks_pt'] * (1/72) * pixelsperinch
    fsize_title_px = sizing['fsize_title_pt'] * (1/72) * pixelsperinch

    fig = go.Figure()

    trace = go.Scatter(
        x=df[xval]
      , y=df[yval1]
      , hovertext = df[xval]
      , hoverinfo='text'
      , opacity=style['opacity']
      , showlegend=False
      , mode='markers'
      , marker={
            'size': sizing['markersize']
          , 'color': 'black'
          , 'line':{
                'width': sizing['markerlinewidth']
              , 'color': df['color_pre']
            }
        }
    )
    fig.add_trace(trace)

    for _, row in df.iterrows():
        # Extend the line alove the zero line
        fig.add_shape(
            type='line'
          , x0=row[xval]
          , y0=0
          , x1=row[xval]
          , y1=row[yval1]
          , line={
                'color': row['color_pre']
              , 'width': sizing['axislinewidth']
            }
        )

    fig.update_yaxes(
        title={
            'font': {
                'size': fsize_title_px
              , 'family': style['font_type']
              , 'color': style['linecolor']
            }
          , 'text': '# non primary connections'
        }
      , ticks='outside'
      , ticklen=sizing['ticklen']
      , tickwidth=sizing['tickwidth']
      , tickfont={
            'family': style['font_type']
          , 'size': fsize_ticks_px
          , 'color': style['linecolor']
        }
      , showgrid=False
      , showline=True
      , linewidth=sizing['axislinewidth']
      , linecolor=style['linecolor']
      , scaleratio=1
      , anchor="free"
      , side="left"
      , overlaying="y"
      , range=plot_specs['range_y']
      , tickvals=plot_specs['tickvals_y']
    )

    fig.update_xaxes(
        title={
            'font':{
                'size': fsize_ticks_px
              , 'family': style['font_type']
              , 'color': style['linecolor']
            }
          , 'text':'cell type pairs'
        }
      , ticks='outside'
      , ticklen=sizing['ticklen']
      , tickwidth=sizing['tickwidth']
      , tickangle=270
      , tickfont={
            'family': style['font_type']
          , 'size': fsize_ticks_px
          , 'color': style['linecolor']
        }
      , showgrid=False
      , showline=True
      , linewidth=sizing['axislinewidth']
      , linecolor=style['linecolor']
    )

    annotations = []
    for _, row in df.iterrows():
        annotations.append({
            'x': row[xval], 'y': row[yval1]
          , 'text': str(row[yval2])
          , 'showarrow':False
          , 'font':{
                'size': 5  # Adjust font size as needed
            }
          , 'xshift': 0    # Adjust xshift to position annotations correctly
          , 'yshift': 10   # Adjust yshift to position annotations correctly
          , 'textangle': -90
        })

    fig.update_layout(
        autosize=False
      , width=figure_width
      , height=figure_height
      , margin={
            'l': figure_width // 6
          , 'r': figure_width // 12
          , 'b': figure_height // 15
          , 't': figure_height // 12
          , 'pad': 8
        }
      , showlegend=True
      , paper_bgcolor='rgba(255, 255, 255, 1)'
      , plot_bgcolor='rgba(255, 255, 255, 1)'
      , annotations=annotations
    )

    save_path = plot_specs['save_path']
    plot_name = plot_specs['plot_name']
    save_path.mkdir(parents=True, exist_ok=True)

    fig.write_html(save_path / f"{plot_name}.html")

    pio.write_image(
        fig
      , save_path / f"{plot_name}.{plot_specs['export_type']}"
    )

    return fig
