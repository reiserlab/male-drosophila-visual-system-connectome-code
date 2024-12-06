from pathlib import Path
from dotenv import find_dotenv

import plotly.graph_objects as go
import plotly.io as pio

from cmap import Colormap
import pandas as pd
import numpy as np

from utils.column_features_helper_functions import find_neuropil_hex_coords
from utils.ol_types import OLTypes
from utils.ol_color import OL_COLOR
from utils.completion_metrics import fetch_cxn_df
from utils.metric_functions import get_metrics_df


def load_and_process_df(
    roi_str:str
  , npl_syn_thresh:float=0.05
  , n_syn_thresh:int=50) -> pd.DataFrame:
    """
    Load and process the dataframe with information about spatial coverage features for cell types
    within a region of interest.

    Parameters
    ----------
    roi_str : str
        Optic lobe region of interest.
    npl_syn_thresh : float, default=0.05
        Threshold value for the percentage of all synapses, from all neurons of a type
        that lie within an optic lobe region for that type to be included in the types
        shown for that region's scatterplot. 0.05 would be 5% of all synapses.
    n_syn_thresh : int, default=50
        Threshold value for the total number of synapses (after trimming) from all neurons
        of a type that lie within an optic lobe region for that type to be included in the
        types shown for that region's scatterplot.

    Returns
    -------
    df : pd.DataFrame
        Dataframe with coverage and completeness metrics per neuron instance type.
    """
    cachedir = Path(find_dotenv()).parent / "cache" / "complete_metrics"
    metric_file = cachedir / "complete_metrics.pickle"

    if metric_file.is_file():
        with metric_file.open("rb") as metric_fh:
            metrics_df = pd.read_pickle(metric_fh)
    else:
        print(
            f"{metric_file} file does not exist. Running the required scripts to generate it."
        )
        metrics_df = get_metrics_df()

    colors = OL_COLOR.OL_TYPES.hex + ["#D3D3D3"]
    types_neuropil = find_types_thresh_syn(
        metrics_df, thresh_val=npl_syn_thresh, roi_str=roi_str
    )
    df = metrics_df[metrics_df["roi"] == roi_str]
    # add columns with color and group information
    df = add_color_group(df, colors)
    # remove cell types that have less than 'npl_syn_thresh' of their total synpases
    # in the chosen neuropil
    df = df[df["instance"].isin(types_neuropil)]
    # remove cell types with < 50 synapses in the chosen neuropil
    df = df[df["n_syn_trim"] > n_syn_thresh]

    return df


def get_axis_labels(
    val:str
  , label_type:str) -> str:
    """
    Return axis label from column name in dataframe

    Parameters
    ----------
    val : str
        String of column name.
    label_type : {"axis", "hover"}
        Whether the label returned is going to be used as an axis title ('axis')
        or for the text within a hover text box in the interactive scatterplots ('hover').

    Returns
    -------
    label : str
        Modified string to use.
    """
    assert label_type in ["axis", "hover"]\
      , f"Unsupported label_type '{label_type}' can only be 'axis' or 'hover' "
    match label_type:
        case "axis":
            axis_dict = {
                "population_size": "population size (# cells/type)"
              , "cols_covered_pop": "population columns innervated"
              , "coverage_factor_trim": "coverage factor (# cells/col)"
              , "synaptic_coverage_factor": "syn coverage factor (# syn/col)"
              , "cell_size_cols": "cell size (# cols/cell)"
              , "area_completeness": "region completeness (area)"
              , "area_covered_pop": "population area (columns)"
              , "instance": "Type"
              , "group": "Group"
            }
            label = axis_dict.get(val, "")
        case _:  # hover
            hover_dict = {
                "population_size": "pop size"
              , "cols_covered_pop": "col compl"
              , "coverage_factor_trim": "cov factor"
              , "synaptic_coverage_factor": "syn cov factor "
              , "cell_size_cols": "cell size"
              , "area_completeness": "area compl"
              , "area_covered_pop": "area pop"
              , "instance": "type"
              , "group": "group"
            }
            label = hover_dict.get(val, "")
    return label


def add_color_group(
    df:pd.DataFrame
  , colors:list) -> pd.DataFrame:
    """
    Add color and group columns to dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with coverage and completeness metrics per neuron instance type.
    colors : list
        List of 5 colors to use.

    Returns
    -------
    df : pd.DataFrame
        Same df returned with 'color' and 'group' columns.
    """
    types = OLTypes()
    color_dict = {
        "OL_intrinsic": {"group": 1, "color": colors[0]}
      , "OL_connecting": {"group": 2, "color": colors[1]}
      , "VPN": {"group": 3, "color": colors[2]}
      , "VCN": {"group": 4, "color": colors[3]}
      , "other": {"group": 5, "color": colors[4]}
    }

    df = df.copy()
    df["type_n"] = df["instance"].str[:-2]
    df["main_group"] = df["type_n"].apply(types.get_main_group)
    df["group"] = df["main_group"].map(lambda x: color_dict[x]["group"])
    df["color"] = df["main_group"].map(lambda x: color_dict[x]["color"])
    df.drop(columns=["type_n", "main_group"], inplace=True)

    return df


def find_types_thresh_syn(
    df:pd.DataFrame
  , thresh_val:float
  , roi_str:str) -> list[str]:
    """
    Return a list of instance types in 'df' that have the more than 'thresh_val'
    of all their synapses in the region ('roi_str').

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with coverage and completeness metrics per neuron instance type.
    thresh_val : float
        Threshold value for the percentage of all synapses, from all neurons of a type
        that lie within an optic lobe region for that type to be included in the types
        shown for that regions scatterplot. 0.05 would be 5% of all synapses.
    roi_str : str
        Region of interest within the optic lobe.

    Returns
    -------
    types_roi : list
        List of instance types.
    """
    assert roi_str in ["ME(R)", "LO(R)", "LOP(R)"], f"unknown ROI '{roi_str}'."
    df = df.fillna(0)
    df_me = df[df["roi"] == "ME(R)"]
    df_lo = df[df["roi"] == "LO(R)"]
    df_lop = df[df["roi"] == "LOP(R)"]

    # total number of synapses per cell type across all regions
    df_grouped = df.groupby("instance")["n_syn_trim"].sum()

    for cell_type, total_syn in df_grouped.items():
        me_syn = df_me.loc[df_me["instance"] == cell_type, "n_syn_trim"].sum()
        lo_syn = df_lo.loc[df_lo["instance"] == cell_type, "n_syn_trim"].sum()
        lop_syn = df_lop.loc[df_lop["instance"] == cell_type, "n_syn_trim"].sum()

        in_me = in_lo = in_lop = 0
        if total_syn > 0:
            if me_syn / total_syn > thresh_val:
                in_me = 1
            if lo_syn / total_syn > thresh_val:
                in_lo = 1
            if lop_syn / total_syn > thresh_val:
                in_lop = 1

        df.loc[df["instance"] == cell_type, "in_ME"] = in_me
        df.loc[df["instance"] == cell_type, "in_LO"] = in_lo
        df.loc[df["instance"] == cell_type, "in_LOP"] = in_lop

    # Extract cell types satisfying condition for the specified neuropil
    types_roi = df[df[f"in_{roi_str[:-3]}"] == 1]["instance"]

    return types_roi


def make_scatterplot_with_star_cells(
    xval:str
  , yval:str
  , roi_str:str
  , style:dict
  , sizing:dict
  , plot_specs:dict
  , star_neurons:list
  , save_plot:bool=True) -> go.Figure:
    """
    Plot scatterplot with 'star_neuron' types highlighted.

    Parameters
    ----------
    xval : str
       Column of 'df' to be shown on the x-axis.
    yval : str
        Column of 'df' to be shown on the y-axis.
    roi_str : str
        Optic lobe region of interest to plot the data from.
    style : dict
        Dict containing the values of the fixed styling formatting variables.
    sizing : dict
        Dict containing the values of the size formatting variables.
    plot_specs : dict
        Dict containing the values of the formatting variables relevant to the specific plot.
    star_neurons : list
        List of 'star neurons' to highlight in the plot.
    save_plot : bool, default = True
        Whether to save the plot or only return it.

    Returns
    -------
    fig : go.Figure
        Formatted plotly scatterplot.
    """
    pio.kaleido.scope.mathjax = None

    assert plot_specs["colorscale"] in ["red", "group"]\
      , f"Colorscale must be 'red' or 'group', not '{plot_specs['colorscale']}'"

    # load the data
    df = load_and_process_df(roi_str=roi_str)
    if xval == "instance":
        df = df.sort_values(yval)

    df["markersize"] = float(sizing["markersize"])
    df["markerlinewidth"] = sizing["markerlinewidth"]
    df["text"] = ""

    if star_neurons:
        df.loc[df["instance"].isin(star_neurons), "markersize"] = (
            sizing["markersize"] * 1.75
        )
        df.loc[df["instance"].isin(star_neurons), "markerlinewidth"] = (
            sizing["markerlinewidth"] * 7
        )
        df.loc[df["instance"].isin(star_neurons), "group"] = 6

    # get sizing values
    if style["export_type"] == "svg":
        pixelsperinch = 72
    else:
        pixelsperinch = 96
    pixelspermm = pixelsperinch / 25.4
    plot_width = (sizing["fig_width"] - sizing["fig_margin"]) * pixelspermm
    plot_height = (sizing["fig_height"] - sizing["fig_margin"]) * pixelspermm
    fsize_ticks_px = sizing["fsize_ticks_pt"] * (1 / 72) * pixelsperinch
    fsize_title_px = sizing["fsize_title_pt"] * (1 / 72) * pixelsperinch

    cmap = Colormap("reds_5").to_plotly()  # colorscale == 'red'; default
    if plot_specs["colorscale"] == "group":
        cmap = [
            [0, "rgb(200, 200, 200)"]
          , [0.1666, "rgb(200, 200, 200)"]
          , [0.1666, "rgb(2, 158, 115)"]
          , [0.333, "rgb(2, 158, 115)"]
          , [0.333, "rgb(213, 94, 0)"]
          , [0.5, "rgb(213, 94, 0)"]
          , [0.5, "rgb(1,115,178)"]
          , [0.666, "rgb(1,115,178)"]
          , [0.6666, "rgb(222,143,5)"]
          , [0.833, "rgb(222,143,5)"]
          , [0.833, "rgb(204, 120, 188)"]
          , [1.0, "rgb(204, 120, 188)"]
        ]

    # set the maximum
    _, n_cols_region = find_neuropil_hex_coords(roi_str)

    fig = go.Figure()

    match yval:
        case "coverage_factor_trim":
            fig.add_shape(
                type="line"
              , y0=1
              , x0=1
              , x1=n_cols_region
              , y1=1
              , layer="below"
              , line={"color": "grey", "width": 0.8, "dash": "solid"}
            )

        case "cell_size_cols":
            for multipl in [0.2, 0.5, 1, 2, 5]:
                if multipl < 0.5 or multipl > 2:
                    line_width = 0.25
                elif multipl != 1:
                    line_width = 0.4
                else:
                    line_width = 0.8

                fig.add_shape(
                    type="line"
                  , x0=1
                  , y0=n_cols_region * multipl
                  , x1=n_cols_region * multipl
                  , y1=1
                  , layer="below"
                  , line={"color": "grey", "width": line_width, "dash": "solid"}
                )

    # add groups in order to ensure star neurons are plotted last
    gp_values = df["group"].sort_values(ascending=True).unique()
    for grp in gp_values:

        df_gp = df[df["group"] == grp]

        fig.add_trace(
            go.Scatter(
                x=df_gp[xval]
              , y=df_gp[yval]
              , text=[df_gp["text"]]
              , textposition="top center"
              , customdata=np.stack(
                    (df_gp["instance"], df_gp[plot_specs["color_factor"]]), axis=-1
                )
              , hovertemplate=plot_specs["hover_template"]
              , opacity=0.95
              , mode="markers"
              , marker={
                    "cmax": plot_specs["cmax"]
                  , "cmin": plot_specs["cmin"]
                  , "size": df_gp["markersize"]
                  , "color": df_gp[plot_specs["color_factor"]]
                  , "line": {
                        "width": df_gp["markerlinewidth"]
                      , "color": style["markerlinecolor"]
                    },
                    "colorscale": cmap,
                    "colorbar": {
                        "orientation": "v"
                      , "outlinecolor": style["linecolor"]
                      , "outlinewidth": sizing["axislinewidth"]
                      , "thickness": sizing["cbar_thickness"]
                      , "len": sizing["cbar_length"]
                      , "ticklen": sizing["cbar_tick_length"]
                      , "tickcolor": "black"
                      , "tickwidth": sizing["cbar_tick_width"]
                      , "tickmode": "array"
                      , "tickvals": plot_specs["tickvals"]
                      , "ticktext": plot_specs["ticktext"]
                      , "tickfont": {"size": fsize_ticks_px, "color": "black"}
                      , "title": {
                            "font": {
                                "family": style["font_type"]
                              , "size": fsize_title_px
                              , "color": "black"
                            },
                            "side": "right"
                          , "text": plot_specs["cbar_title"]
                        }
                    }
                }
            )
        )

    xlabel = get_axis_labels(xval, label_type="axis")
    ylabel = get_axis_labels(yval, label_type="axis")

    if plot_specs["log_x"]:
        typex = "log"
        tickformx = ".1r"
    else:
        typex = "-"
        tickformx = ""

    if plot_specs["log_y"]:
        typey = "log"
        tickformy = ".1r"
    else:
        typey = "-"
        tickformy = ""

    fig.update_layout(
        autosize=False,
        width=plot_width,
        height=plot_height,
        margin={
            "l": plot_width // 8,
            "r": 0,
            "b": plot_height // 4,
            "t": 0,
            "pad": plot_width // 30,
        },
        showlegend=False,
        paper_bgcolor="rgba(255,255,255,1)",
        plot_bgcolor="rgba(255,255,255,1)",
    )

    fig.update_xaxes(
        title={
            "font": {
                "size": fsize_title_px,
                "family": style["font_type"],
                "color": style["linecolor"],
            },
            "text": xlabel,
        },
        title_standoff=(plot_height // 4) / 4,
        ticks="outside",
        ticklen=sizing["ticklen"],
        tickwidth=sizing["tickwidth"],
        tickfont={
            "family": style["font_type"],
            "size": fsize_ticks_px,
            "color": style["linecolor"],
        },
        tickformat=tickformx,
        tickcolor="black",
        type=typex,
        showgrid=False,
        showline=True,
        linewidth=sizing["axislinewidth"],
        linecolor=style["linecolor"],
        range=plot_specs["range_x"],
    )

    match xval:
        case "instance":
            fig.update_xaxes(title="", tickvals=[], visible=False)
        case "coverage_factor_trim":
            fig.update_xaxes(tickvals=[1, 2, 4, 8, 12])
        case "cell_size_cols":
            fig.update_xaxes(tickvals=[1, 10, 100, 1000])
        case "area_covered_pop":
            fig.update_xaxes(tickvals=[0, 200, 400, 600, 800, 1000])
        case "cols_covered_pop":
            fig.update_xaxes(tickvals=[0, 200, 400, 600, 800, 1000])
        case "population_size":
            fig.update_xaxes(tickvals=[1, 10, 100, 1000])

    fig.update_yaxes(
        title={
            "font": {
                "size": fsize_title_px
              , "family": style["font_type"]
              , "color": style["linecolor"]
            }
          , "text": f"{ylabel}"
        }
      , title_standoff=(plot_width // 5) / 5
      , ticks="outside"
      , tickcolor="black"
      , ticklen=sizing["ticklen"]
      , tickwidth=sizing["tickwidth"]
      , tickfont={
            "size": fsize_ticks_px
          , "family": style["font_type"]
          , "color": "black"
        }
      , tickformat=tickformy
      , type=typey
      , showgrid=False
      , showline=True
      , linewidth=sizing["axislinewidth"]
      , linecolor=style["linecolor"]
      , scaleanchor="x"
      , scaleratio=1
      , anchor="free"
      , side="left"
      , overlaying="y"
      , range=plot_specs["range_y"]
    )

    match yval:
        case "coverage_factor_trim":
            fig.update_yaxes(tickvals=[1, 2, 4, 8, 12], scaleanchor=False)
        case "cell_size_cols":
            fig.update_yaxes(tickvals=[1, 10, 100, 1000])
        case "area_covered_pop":
            fig.update_yaxes(tickvals=[0, 200, 400, 600, 800, 1000])
        case "population_size":
            fig.update_yaxes(tickvals=[1, 10, 100, 1000])

    if save_plot:
        save_path = plot_specs["save_path"]
        assert isinstance(save_path, Path)\
          , f"save_path needs to be 'Path', not '{type(save_path)}'"
        save_path.mkdir(parents=True, exist_ok=True)

        basename = (
            save_path
            / f"{roi_str[:-3]}_{xval}_versus_{yval}_{sizing['fsize_title_pt']}pt_"
            f"w{sizing['fig_width']}_h{sizing['fig_height']}_cscale-{plot_specs['color_factor']}"
        )

        if style["export_type"] == "html":
            fig.write_html(f"{basename}.html")
        else:
            pio.write_image(
                fig,
                f"{basename}.{style['export_type']}",
                width=plot_width,
                height=plot_height,
            )
    return fig


def generate_plot_specs(
    x_val:str
  , y_val:str
  , colorscale:str
  , roi_str:str
  , save_path:str=None) -> dict:
    """
    Generate the 'plot_specs' for scatterplots.

    Parameters
    ----------
    x_val : str
        Column of 'df' to be shown on the x-axis.
    y_val : str
        Column of 'df' to be shown on the y-axis.
    colorscale : str
        Column of 'df' to be used as the colorscale.
    roi_str : str
        Optic lobe region of interest.
    save_path : str
        Path to save scatterplot.

    Returns
    -------
    plot_specs : dict
        Dictionary of plot specifications to use for different x and y values.
    """
    if save_path is None:
        save_path = (
            Path(find_dotenv()).parent / "results" / "html_pages" / "scatterplots"
        )

    _, graph_lims = find_neuropil_hex_coords(roi_str=roi_str)
    n_types_region = find_n_types_region(roi_str=roi_str)
    xlabel = get_axis_labels(x_val, label_type="hover")
    ylabel = get_axis_labels(y_val, label_type="hover")
    clabel = get_axis_labels(colorscale, label_type="hover")

    specs_dict = {
        "population_size": {
            "log_x": True
          , "range_x": [-0.3, 3.3]
          , "log_y": True
          , "range_y": [-0.3, 3.3]
          , "cmax": 500
          , "cmin": 0
          , "colorscale": "red"
          , "tickvals": [0, 100, 200, 300, 400, 500]
          , "ticktext": ["0", "100", "200", "300", "400", "500"]
          , "cbar_title": "population size"
          , "cbar_title_scaling": 1
          , "d3": ",.0f"
        },
        "cols_covered_pop": {
            "log_x": False
          , "range_x": [-10, graph_lims * 1.05]
          , "log_y": False
          , "range_y": [-10, graph_lims * 1.05]
          , "cmax": 500
          , "cmin": 0
          , "colorscale": "red"
          , "tickvals": [0, 100, 200, 300, 400, 500]
          , "ticktext": ["0", "100", "200", "300", "400", "500"]
          , "cbar_title": "population columns innervated"
          , "cbar_title_scaling": 1
          , "d3": ",.0f"
        },
        "coverage_factor_trim": {
            "log_x": True
          , "range_x": [-0.1, 1.25]
          , "log_y": True
          , "range_y": [-0.1, 1.25]
          , "cmax": 5
          , "cmin": 1
          , "colorscale": "red"
          , "tickvals": [1, 2, 3, 4, 5]
          , "ticktext": ["1", "2", "3", "4", "5"]
          , "cbar_title": "coverage factor"
          , "cbar_title_scaling": 0.7
          , "d3": ",.2f"
        },
        "synaptic_coverage_factor": {
            "log_x": True
          , "range_x": [-0.3, 3.3]
          , "log_y": True
          , "range_y": [-0.1, 3.3]
          , "cmax": 20000
          , "cmin": 0
          , "colorscale": "red"
          , "tickvals": [0, 10000, 20000, 30000]
          , "ticktext": ["0", "10k", "20k", "30k"]
          , "cbar_title": "# syn"
          , "cbar_title_scaling": 1
          , "d3": ",.0f"
        },
        "cell_size_cols": {
            "log_x": True
          , "range_x": [-0.1, 3.3]
          , "log_y": True
          , "range_y": [-0.1, 3.3]
          , "cmax": 800
          , "cmin": 0
          , "colorscale": "red"
          , "tickvals": [0, 200, 400, 600, 800]
          , "ticktext": ["0", "200", "400", "600", "800"]
          , "cbar_title": "cell size (cols/cell)"
          , "cbar_title_scaling": 1
          , "d3": ",.0f"
        },
        "area_completeness": {
            "log_x": False
          , "range_x": [-0.1, 1.1]
          , "log_y": False
          , "range_y": [-0.1, 1.1]
          , "cmax": 1
          , "cmin": 0
          , "colorscale": "red"
          , "tickvals": [0, 1]
          , "ticktext": ["0", "1"]
          , "cbar_title": "population area (# columns)"
          , "cbar_title_scaling": 1
          , "d3": ",.2f"
        },
        "area_covered_pop": {
            "log_x": False
          , "range_x": [0, graph_lims * 1.05]
          , "log_y": False
          , "range_y": [0, graph_lims * 1.05]
          , "cmax": 900
          , "cmin": 0
          , "colorscale": "red"
          , "tickvals": [0, 200, 400, 600, 800]
          , "ticktext": ["0", "200", "400", "600", "800"]
          , "cbar_title": "population area (# columns)"
          , "cbar_title_scaling": 1
          , "d3": ",.0f"
        },
        "instance": {  # only x/ y
            "log_x": False
          , "range_x": [-10, n_types_region + 10]
          , "log_y": False
          , "range_y": [-10, n_types_region + 10]
          , "d3": ""
        },
        "group": {  # only colorscale
            "colorscale": "group"
          , "cmax": 1
          , "cmin": 0
          , "tickvals": [0.0833, 0.25, 0.4166, 0.5833, 0.75, 0.9166]
          , "ticktext": ["L_hemi", "intrinsic", "connecting", "VPN", "VCN", "other"]
          , "cbar_title": "Group"
          , "d3": ""
          , "cbar_title_scaling": 1
        }
    }

    if colorscale == "group":
        h_temp = (
            f"<b>%{{customdata[0]}}</b><br>{ylabel}: %{{y:{specs_dict[y_val]['d3']}}}"
            "<br><extra></extra>"
        )
    else:
        h_temp = (
            f"<b>%{{customdata[0]}}</b><br>{xlabel}: %{{x:{specs_dict[x_val]['d3']}}}"
            f"<br>{ylabel}: %{{y:{specs_dict[y_val]['d3']}}}"
            f"<br>{clabel}: %{{customdata[1]:{specs_dict[colorscale]['d3']}}}<br><extra></extra>"
        )

    plot_specs = {
        "log_x": specs_dict[x_val]["log_x"]
      , "log_y": specs_dict[y_val]["log_y"]
      , "range_x": specs_dict[x_val]["range_x"]
      , "range_y": specs_dict[y_val]["range_y"]
      , "save_path": save_path
      , "color_factor": colorscale
      , "cmax": specs_dict[colorscale]["cmax"]
      , "cmin": specs_dict[colorscale]["cmin"]
      , "colorscale": specs_dict[colorscale]["colorscale"]
      , "tickvals": specs_dict[colorscale]["tickvals"]
      , "ticktext": specs_dict[colorscale]["ticktext"]
      , "cbar_title": specs_dict[colorscale]["cbar_title"]
      , "hover_template": h_temp
      , "cbar_title_scaling": specs_dict[colorscale]["cbar_title_scaling"]
    }

    if colorscale == "group":
        plot_specs["color_factor"] = "color"

    return plot_specs


def find_n_types_region(roi_str:str) -> int:
    """
    Find the number of cell types within each ROI that are used in the scatterplots.

    Parameters
    ----------
    roi_str : str
        Optic lobe region.

    Returns
    -------
    n_types : int
        Number of instance types within the chosen optic lobe region.
    """
    df = load_and_process_df(roi_str=roi_str)
    n_types = len(df)

    return n_types


def make_covcompl_scatterplot(
    x_val:str
  , y_val:str
  , colorscale:str
  , roi_str:str
  , star_instances:list
  , export_type:str
  , save_path:Path=None
  , save_plot:bool=True
) -> go.Figure:
    """
    Generate plotly scatterplot.

    Parameters
    ----------
    x_val : str
        Column of 'df' to be shown on the x-axis.
    y_val : str
        Column of 'df' to be shown on the y-axis.
    colorscale : str
        Column of 'df' to be used as the colorscale.
    roi_str : str
        Optic lobe region.
    star_instances : list
        List of 'star instances' types to be highlighted in the scatterplots.
    export_type : str
        Export type of fig. Typically only 'pdf' or 'html'.
    save_path : Path, default=None
        Path to save the figure if 'save_plot'=True.
    save_plot : bool, default=True
        If 'True', saves the plot else only returns the plot.

    Returns
    -------
    fig : go.Figure
        Formatted plotly scatterplot.
    """
    assert export_type in [
        "pdf",
        "html",
    ], f"Can only export 'pdf' or 'html', not f{export_type}"
    assert roi_str in ["ME(R)", "LO(R)", "LOP(R)"], f"unsupported roi_str '{roi_str}'."

    if save_path is None:
        save_path = (
            Path(find_dotenv()).parent / "results" / "cov_compl" / "scatterplots"
        )

    style = {
        "export_type": "html",
        "font_type": "arial",
        "markerlinecolor": "black",
        "linecolor": "black",
    }

    match export_type:
        case "pdf":
            sizing = {
                "fig_width": 48         # units = mm
              , "fig_height": 35        # units = mm
              , "fig_margin": 0
              , "fsize_ticks_pt": 5
              , "fsize_title_pt": 5
              , "markersize": 2
              , "ticklen": 2
              , "tickwidth": 0.7
              , "axislinewidth": 0.65
              , "markerlinewidth": 0.07
              , "cbar_thickness": 2.5
              , "cbar_length": 1.1
              , "cbar_tick_length": 2.25
              , "cbar_tick_width": 0.75
              , "cbar_title_x": 1.28
              , "cbar_title_y": -0.23
            }
            if colorscale == "group":
                sizing["fig_width"] = 53
                sizing["cbar_title_x"] = 1.35
                sizing["cbar_title_y"] = -0.15
        case _:  # html
            sizing = {
                "fig_width": 240        # units = mm
              , "fig_height": 175       # units = mm
              , "fig_margin": 0
              , "fsize_ticks_pt": 18
              , "fsize_title_pt": 20
              , "markersize": 9.5
              , "ticklen": 10
              , "tickwidth": 3
              , "axislinewidth": 2.85
              , "markerlinewidth": 0.35
              , "cbar_thickness": 8
              , "cbar_length": 0.95
              , "cbar_tick_length": 4
              , "cbar_tick_width": 2
              , "cbar_title_x": 1.09
              , "cbar_title_y": -0.2
            }
            if colorscale == "group":
                sizing["fig_width"] = 265
                sizing["cbar_title_x"] = 1.15
                sizing["cbar_title_y"] = -0.1

    # function - take in x and y values and give back plot_specs
    plot_specs = generate_plot_specs(
        x_val, y_val, colorscale, roi_str, save_path=save_path
    )

    fig = make_scatterplot_with_star_cells(
        xval=x_val
      , yval=y_val
      , roi_str=roi_str
      , style=style
      , sizing=sizing
      , plot_specs=plot_specs
      , star_neurons=star_instances
      , save_plot=save_plot
    )

    return fig


def make_completeness_scatter(
    roi_str:str
  , style:dict
  , sizing:dict
  , plot_specs:dict
  , save_fig:bool=True) -> go.Figure:
    """
    Create scatterplots from the 'cxn_df' data frame, comparing the completeness of the
    connectome data per neuropil column. Requires the existence of the 'cxn_df' data frame
    that is generated by the function 'fetch_cxns_per_col' in 'src/utils/completion_metrics'.

    Parameters
    ----------
    roi_str : str
        Neuprint ROI, can only be ME(R), LO(R), LOP(R).
    style : dict
        Dict containing the values of the fixed styling formatting variables.
    sizing : dict
        Dict containing the values of the size formatting variables.
    plot_specs : dict
        Dict containing the values of the formatting variables relevant to the specific plot.
    save_fig : bool, default=True
        If True, saves the generated figure to a file. If False, the figure is displayed
        but not saved.

    Returns
    -------
    fig : go.Figure

    """
    fname = Path(find_dotenv()).parent / 'cache' / 'completeness' / f'cxn_df_{roi_str[:-3]}.pickle'

    if fname.is_file():
        df = pd.read_pickle(fname)
    else:
        df = fetch_cxn_df(roi_str)

    fig = go.Figure()

    fig.update_layout(
        width=sizing["fig_width"]
      , height=sizing["fig_width"]
      , autosize=False
      , showlegend=False
      , margin={
            "l": sizing["fig_width"] // 8
          , "r": 0
          , "b": sizing["fig_height"] // 4
          , "t": 0
          , "pad": 0
        }
      , font={
            "family": style["font_type"]
          , "size": sizing["fsize_ticks_pt"]
          , "color": style["linecolor"]
        }
      , paper_bgcolor="rgba(255,255,255,1)"
      , plot_bgcolor="rgba(255,255,255,1)"
    )

    fig.update_xaxes(
        range=plot_specs["xlim"]
      , showgrid=False
      , showticklabels=True
      , showline=True
      , visible=True
      , linecolor=style["linecolor"]
      , tickmode="auto"
      , tickcolor="black"
      , ticklen=sizing["ticklen"]
      , tickwidth=sizing["tickwidth"]
      , ticks="outside"
      , linewidth=sizing["linewidth"]
      , tickformat=".2s"
      , title={"text": plot_specs["xaxis_title"]}
    )

    fig.update_yaxes(
        range=plot_specs["ylim"]
      , showgrid=False
      , showticklabels=True
      , showline=True
      , visible=True
      , linecolor=style["linecolor"]
      , tickmode="auto"
      , tickcolor="black"
      , ticklen=sizing["ticklen"]
      , tickwidth=sizing["tickwidth"]
      , ticks="outside"
      , linewidth=sizing["linewidth"]
      , tickformat=".2s"
      , anchor="x"
      , side="left"
      , title={"text": plot_specs["yaxis_title"]}
    )

    fig.add_trace(
        go.Scatter(
            x=df[plot_specs["xval"]] + df[plot_specs["yval"]]
          , y=df[plot_specs["yval"]]
          , mode="markers"
          , marker={
                "color": "lightgrey"
              , "size": sizing["markersize"]
              , "line": {
                    "width": sizing["markerlinewidth"]
                  , "color": style["linecolor"]
                }
            }
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[0, plot_specs["xlim"][1]]
          , y=[
                0,
                plot_specs["xlim"][1] * df[f"{plot_specs['xval'][:-4]}_ratio"].median()
            ]
          , mode="lines"
          , line={'width': 1, 'color': "black"}
        )
    )

    if save_fig:
        save_path = plot_specs["save_path"]
        save_path.mkdir(parents=True, exist_ok=True)
        fig.write_image(
            save_path / f"{plot_specs['filename']}.{plot_specs['export_type']}"
        )

    return fig
