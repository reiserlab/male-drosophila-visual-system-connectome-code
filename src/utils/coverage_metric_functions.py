""" Helper functions for plotting coverage metrics """
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from utils.ol_color import OL_COLOR
from utils.scatterplot_functions import load_and_process_df, get_axis_labels

def plot_coverage_metric_histogram(
    style:dict
  , sizing:dict
  , plot_specs:dict
  , metric:str
):
    """
    Generate 3x1 subplot of histograms of coverage metric values per type per optic lobe region.

    Parameters
    ----------
    style : dict
        Dictionary of parameters defining to the style of the returned plot.
        font_type : str
            Font type used for tick and axis labels
        markerlinecolor : str
            Marker line color
        linecolor : str
            Line color
    sizing : dict
        Dictionary of parameters related to the size of the returned plot.
            fig_width : int
                Width of returned figure (mm)
            fig_height : int
                Height of returned figure (mm)
            fig_margin : int or float
                Figure margin (mm)
            fsize_ticks_pt : int or float
                Tick font size (pt)
            fsize_title_pt : int or float
                Title font size (pt)
            markersize : int or float
                Marker size
            ticklen : int or float
                Tick length
            tickwidth : int or float
                Tick width
            axislinewidth : int or float
                Axis line width
            markerlinewidth : int or float
                Marker line width
    plot_specs : dict
        Dictionary of parameters unique to the returned plot.
            log_x : str
                X-axis type. Value can be 'linear' or 'log'
            log_y : str
                Y-axis type. Value can be 'linear' or 'log'
            range_x : tuple containing integers or float values.
                X-axis range. For log axes consult plotly documentation.
            range_y : tuple containing integers or float values.
                Y-axis range. For log axes consult plotly documentation.
            save_path : Path
                path to save the figure
            tickvals_y : list of integers or floats
                Y-axis tick values
            ticktext_y : list of str
                Y-axis tick text values
            tickvals_x : list of integers or floats
                X-axis tick values
            x_bin_start : int or float
                Lowest x value for bins
            x_bin_end : int or float
                Highest x value for bins
            x_bin_width : int or float
                X value bin width
            export_type : str
                Plot export type. Typically 'pdf' or 'svg'
    metric : str
        Variable to be plotted as a histogram from the columns within the dataframe returned
        from the function `load_and_process_df()`.

    Returns
    -------
    fig: go.Figure
        Plotly histogram plot.
    """
    pio.kaleido.scope.mathjax = None

    # saving parameters
    plot_specs['save_path'].mkdir(parents=True, exist_ok=True)

    # sizing of the figure and font
    if plot_specs["export_type"] in ["svg"]:
        pixelsperinch = 72
    else:
        pixelsperinch = 96
    pixelspermm = pixelsperinch / 25.4
    w = (sizing["fig_width"] - sizing["fig_margin"]) * pixelspermm
    h = (sizing["fig_height"] - sizing["fig_margin"]) * pixelspermm
    fsize_ticks_px =sizing["fsize_ticks_pt"] * (1 / 72) * pixelsperinch
    fsize_title_px =sizing["fsize_title_pt"] * (1 / 72) * pixelsperinch

    region_colors = {
        'ME(R)': {'color': OL_COLOR.OL_NEUROPIL.hex[0],}
      , 'LO(R)': {'color': OL_COLOR.OL_NEUROPIL.hex[1],}
      , 'LOP(R)': {'color': OL_COLOR.OL_NEUROPIL.hex[2]}
    }

    for roi in ['ME(R)', 'LO(R)', 'LOP(R)']:
        df = load_and_process_df(roi_str=roi, npl_syn_thresh=0.05, n_syn_thresh=50)
        match roi:
            case 'ME(R)':
                x_me = df[metric]
            case 'LO(R)':
                x_lo = df[metric]
            case _: # 'LOP(R)'
                x_lop = df[metric]

    label = get_axis_labels(metric, label_type='hover')

    # Initiate plot
    fig = make_subplots(
        rows=3
      , cols=1
      , horizontal_spacing=0
      , vertical_spacing=0.08
    )

    fig.add_trace(
        go.Histogram(
            x=x_me
          , xbins={
                'start': plot_specs['x_bin_start']
              , 'end': plot_specs['x_bin_end']
              , 'size': plot_specs['x_bin_width']
            }
          , marker_color=region_colors['ME(R)']['color']
          , histnorm=""
        )
      ,  row=1, col=1
    )

    fig.add_trace(
        go.Histogram(
            x=x_lo
          , xbins={
                'start': plot_specs['x_bin_start']
              , 'end': plot_specs['x_bin_end']
              , 'size': plot_specs['x_bin_width']
            }
          , marker_color=region_colors['LO(R)']['color']
          , histnorm=""
        )
      , row=2, col=1
    )

    fig.add_trace(
        go.Histogram(
            x=x_lop
          , xbins={
                'start': plot_specs['x_bin_start']
              , 'end': plot_specs['x_bin_end']
              , 'size': plot_specs['x_bin_width']
            }
          , marker_color=region_colors['LOP(R)']['color']
          , histnorm=""
        )
      , row=3, col=1
    )

    fig.update_traces(opacity=0.75)

    fig.update_layout(
        autosize=False
      , width=w
      , height=h
      , margin={
            'l':w//10
          , 'r':w//20
          , 'b':h//10
          , 't':h//20
          , 'pad':w//30
        }
      , showlegend=False
      , paper_bgcolor='rgba(255,255,255,1)'
      , plot_bgcolor='rgba(255,255,255,1)'
    )

    fig.update_xaxes(
        title={
            'font':{
                'size': fsize_title_px
              , 'family': style['font_type']
              , 'color': style['linecolor']
            }
          , 'text': f"{label}"
        }
      , tickfont={
            'family': style['font_type']
          , 'size': fsize_ticks_px
          , 'color': style['linecolor']
        }
      , showgrid=False
      , showline=True
      , ticks='outside'
      , ticklen=sizing['ticklen']
      , tickwidth=sizing['tickwidth']
      , type=plot_specs['log_x']
      , linecolor=style['linecolor']
      , range=plot_specs['range_x']
      , tickvals=plot_specs['tickvals_x']
      , row=3, col=1
    )

    fig.update_xaxes(
        showline=True
      , linecolor=style['linecolor']
      , tickvals=[]
      , range=plot_specs['range_x']
      , type=plot_specs['log_x']
      , row=1, col=1
    )

    fig.update_xaxes(
        showline=True
      , linecolor=style['linecolor']
      , tickvals=[]
      , range=plot_specs['range_x']
      , type=plot_specs['log_x']
      , row=2, col=1
    )

    fig.update_yaxes(
        tickfont={
            'family': style['font_type']
          , 'size': fsize_ticks_px
          , 'color': style['linecolor']
        }
      , showgrid=False
      , showline=True
      , linecolor=style['linecolor']
      , ticks='outside'
      , ticklen=sizing['ticklen']
      , tickwidth=sizing['tickwidth']
      , type=plot_specs['log_y']
      , tickvals=plot_specs['tickvals_y']
      , ticktext=plot_specs['ticktext_y']
      , range=plot_specs['range_y']
    )

    fig.update_yaxes(
        tickfont={
            'family': style['font_type']
          , 'size': fsize_ticks_px
          , 'color': style['linecolor']
        }
      , title={
            'font': {
                'size': fsize_title_px
              , 'family': style['font_type']
              , 'color': style['linecolor']
            }
          , 'text': 'number of types'
        }
      , showgrid=False
      , showline=True
      , linecolor=style['linecolor']
      , ticks='outside'
      , ticklen=sizing['ticklen']
      , tickwidth=sizing['tickwidth']
      , type=plot_specs['log_y']
      , tickvals=plot_specs['tickvals_y']
      , ticktext=plot_specs['ticktext_y']
      , range=plot_specs['range_y']
      , row=2, col=1
    )

    # add median per region
    fig.add_shape(
        type='line'
      , x0=x_me.median()
      , y0=0
      , x1=x_me.median()
      , y1=1000
      , layer="above"
      , line={
            'color': style['linecolor']
          , 'width': sizing['axislinewidth']
        }
      , row=1, col=1
    )

    fig.add_shape(
        type='line'
      , x0=x_lo.median()
      , y0=0
      , x1=x_lo.median()
      , y1=1000
      , layer="above"
      , line={
            'color': style['linecolor']
          , 'width': sizing['axislinewidth']
        }
      , row=2, col=1
    )

    fig.add_shape(
        type='line'
      , x0=x_lop.median()
      , y0=0
      , x1=x_lop.median()
      , y1=1000
      , layer="above"
      , line={
            'color': style['linecolor']
          , 'width': sizing['axislinewidth']
        }
      , row=3, col=1
    )

    pio.write_image(
        fig
      , plot_specs['save_path'] / f"{metric}_histogram.{plot_specs['export_type']}"
      , width=w
      , height=h
    )

    return fig
