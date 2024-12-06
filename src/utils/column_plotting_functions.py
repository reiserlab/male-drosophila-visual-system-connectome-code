"""
Plotting functions for generating the hexagonal 'eyemap' heatmap plots.
"""
from pathlib import Path
from dotenv import find_dotenv
import plotly.graph_objects as go
import plotly.io as pio

from plotly.subplots import make_subplots
from cmap import Colormap, Color
import pandas as pd
import numpy as np
from utils.column_features_helper_functions import find_neuropil_hex_coords


def plot_per_col(
    df:pd.DataFrame
  , style:dict
  , sizing:dict
  , plot_specs:dict
  , plot_type:str
  , cmap_type:str="discrete"
  , trim:bool=False
  , save_fig:bool=True
) -> go.Figure:
    """
    Generate a hexagonal heat map plot of the number of cells or the number of synapses per
      column in each of the three main optic lobe neuropils, ME(R), LO(R) and LOP(R).

    Parameters
    ----------
    df : pd.DataFrame
        Pandas dataframe that must include the columns:
            roi : str
                Optic lobe region of interest, options include 'ME(R)', 'LO(R)' or 'LOP(R)'.
            hex1_id : int
                Hex 1 coordinates of optic lobe columns.
            hex2_id : int
                Hex 2 coordinates of optic lobe columns.
        and one of either:
            n_syn : int
                The number of synapses within that column.
            n_cells : int
                The number of cells within that column.
    style : dict
        Dictionary of parameters defining to the style of the returned plot.
        font_type : str
            Font type used for tick and axis labels.
        markerlinecolor : str
            Marker line color.
        linecolor : str
            Line color.
    sizing : dict
        dictionary of parameters related to the size of the returned plot
            fig_width : int
                Width of returned figure (mm).
            fig_height : int
                Height of returned figure (mm).
            fig_margin : int or float
                Figure margin (mm).
            fsize_ticks_pt : int or float
                Tick font size (pt).
            fsize_title_pt : int or float
                Title font size (pt).
            markersize : int or float
                Marker size.
            ticklen : int or float
                Tick length.
            tickwidth : int or float
                Tick width.
            axislinewidth : int or float
                Axis line width.
            markerlinewidth : int or float
                Marker line width.
            cbar_thickness : int or float
                Thickness of the colorbar.
            cbar_len : int or float
                Length of the colorbar.
    plot_specs : dict
        Dictionary of parameters unique to the returned plot.
            filename : str
                File name to use if saving the figure.
            cmax_cells : int
                Maximum number of cells to use for the shared colorbar.
            cmax_syn : int
                Maximum number of synapses to use for the shared colorbar.
            export_type : str
                Plot export type. Typically 'pdf' or 'svg'.
    plot_type : str
        Type of hexagonal heatmap to generate, can be 'cells' or 'synapses'.
    cmap_type : str
        Type of colormap to use, can be 'discrete' or 'cont'.
    trim : bool, default = False
        Type of data used. If True, trimmed data is being used, else the raw data.
    save_fig : bool, default = True
        If True will save the returned figure at the location `plot_specs['save_path']`.

    Returns
    -------
    fig : go.Figure
        Plotly 1x3 subplot containing hexagonal heatmaps for the number of cells or synapses
        per column in the Medulla, Lobula and Lobula plate.
    """
    assert plot_type in [
        "synapses"
      , "cells"
    ], f"plot_type must be 'synapses' or 'cells', not '{plot_type}'"

    assert cmap_type in [
        "discrete"
      , "cont"
    ], f"cmap_type must be 'discrete' or 'cont', not '{cmap_type}'"

    if plot_type == "synapses":
        if save_fig:
            fig_save_path = Path(plot_specs["save_path"], "syn_per_col")
            fig_save_path.mkdir(parents=True, exist_ok=True)
        col_name = "n_syn"
        maxval = plot_specs["cmax_syn"]
        cbar_title = "Number of synapses"
    else: # cells
        if save_fig:
            fig_save_path = Path(plot_specs["save_path"], "cells_per_col")
            fig_save_path.mkdir(parents=True, exist_ok=True)
        col_name = "n_cells"
        maxval = plot_specs["cmax_cells"]
        cbar_title = "Number of cells"

    # saving parameters
    filename = plot_specs["filename"]

    # sizing of the figure and font
    if plot_specs["export_type"] in ["svg", "pdf"]:
        pixelsperinch = 72
    else:
        pixelsperinch = 96
    pixelspermm = pixelsperinch / 25.4
    area_width = (sizing["fig_width"] - sizing["fig_margin"]) * pixelspermm
    area_height = (sizing["fig_height"] - sizing["fig_margin"]) * pixelspermm
    fsize_ticks_px = sizing["fsize_ticks_pt"] * (1 / 72) * pixelsperinch
    fsize_title_px = sizing["fsize_title_pt"] * (1 / 72) * pixelsperinch

    # initiate plot
    fig = make_subplots(rows=1, cols=3, subplot_titles=("ME", "LO", "LOP"))
    fig.update_layout(
        autosize=False
      , height=area_height
      , width=area_width
      , margin={
            "l": 0
          , "r": 0
          , "b": 0
          , "t": 0
          , "pad": 0
        }
      , paper_bgcolor="rgba(255,255,255,255)"
      , plot_bgcolor="rgba(255,255,255,255)"
    )
    fig.update_xaxes(
        showgrid=False, showticklabels=False, showline=False, visible=False
    )
    fig.update_yaxes(
        showgrid=False, showticklabels=False, showline=False, visible=False
    )

    symbol_number = 15

    if maxval == 0:
        tickvals = [1.25]
        tickvals_text = ["0"]

        if cmap_type == "discrete":
            _, _, cmap = ticks_from_max(1)
        else: # "cont"
            cmap = Colormap("reds_5").to_plotly()

        for col_idx, roi_str in enumerate(["ME(R)", "LO(R)", "LOP(R)"], start=1):

            # Hex coordinates of medulla columns that do not exist in LO / LOP
            if roi_str in ["LO(R)", "LOP(R)"]:
                col_hex_ids_empty, _ = find_neuropil_hex_coords("ME(R)")
                hex1_vals_empty = col_hex_ids_empty["hex1_id"]
                hex2_vals_empty = col_hex_ids_empty["hex2_id"]

            # Hex coordinates of the columns in the chosen neuropil
            col_hex_ids, _ = find_neuropil_hex_coords(roi_str)
            hex1_vals = col_hex_ids["hex1_id"]
            hex2_vals = col_hex_ids["hex2_id"]

            col_hex_ids["count"] = 0

            if roi_str in ["LO(R)", "LOP(R)"]:
                # Add grey hexagons for columns that are not present in the neuropil
                fig.add_trace(
                    go.Scatter(
                        x=-1 * (hex1_vals_empty - hex2_vals_empty)
                      , y=hex1_vals_empty + hex2_vals_empty
                      , mode="markers"
                      , marker_symbol=symbol_number
                      , marker={
                            "size": sizing["markersize"]
                          , "color": "lightgrey"
                          , "line": {
                                "width": sizing["markerlinewidth"]
                              , "color": style["markerlinecolor"]
                            }
                        }
                      , showlegend=False
                    )
                  , row=1
                  , col=col_idx
                )

            if roi_str == "LOP(R)":  # Add empty white 'background' hexagons with cbar
                fig.add_trace(
                    go.Scatter(
                        x=-1 * (hex1_vals - hex2_vals)
                      , y=(hex1_vals + hex2_vals)
                      , mode="markers"
                      , marker_symbol=symbol_number
                      , marker={
                            "size": sizing["markersize"]
                          , "color": "white"
                          , "cmin": 0
                          , "cmax": 0
                          , "line": {
                                "width": sizing["markerlinewidth"]
                              , "color": "lightgrey"
                            }
                          , "colorbar": {
                                "orientation": "v"
                              , "outlinecolor": style["linecolor"]
                              , "outlinewidth": sizing["axislinewidth"]
                              , "thickness": sizing["cbar_thickness"]
                              , "len": sizing["cbar_len"]
                              , "tickmode": "array"
                              , "tickvals": tickvals
                              , "ticktext": tickvals_text
                              , "ticklen": sizing["ticklen"]
                              , "tickwidth": sizing["tickwidth"]
                              , "tickcolor": style["linecolor"]
                              , "tickfont": {
                                    "size": fsize_ticks_px
                                  , "family": style["font_type"]
                                  , "color": style["linecolor"]
                                }
                              , "title": {
                                    "font": {
                                        "family": style["font_type"]
                                      , "size": fsize_title_px
                                      , "color": style["linecolor"]
                                    }
                                  , "side": "right"
                                  , "text": cbar_title
                                }
                            }
                          , "colorscale": cmap
                        }
                      , showlegend=False
                    )
                  , row=1
                  , col=col_idx
                )
            else:  # Add empty white 'background' hexagons without cbar for ME and LO
                fig.add_trace(
                    go.Scatter(
                        x=-1 * (hex1_vals - hex2_vals)
                      , y=hex1_vals + hex2_vals
                      , mode="markers"
                      , marker_symbol=symbol_number
                      , marker={
                            "size": sizing["markersize"]
                          , "color": "white"
                          , "cmin": 0
                          , "cmax": 0
                          , "line": {
                                "width": sizing["markerlinewidth"]
                              , "color": "lightgrey"
                            }
                          , "colorscale": cmap
                        }
                      , showlegend=False
                    )
                  , row=1
                  , col=col_idx
                )
    else:
        if cmap_type == "discrete":
            tickvals, tickvals_text, cmap = ticks_from_max(maxval)
        else:  # "cont"
            tickvals = [maxval, 10000, 20000, 0]
            tickvals_text = [f"{t //1000:,}K" for t in tickvals[0:3]]
            cmap = Colormap("reds_5").to_plotly()

        for col_idx, roi_str in enumerate(["ME(R)", "LO(R)", "LOP(R)"], start=1):
            syn_col_df = df[df["roi"] == roi_str]

            # Hex coordinates of medulla columns that do not exist in LO / LOP
            if roi_str in ["LO(R)", "LOP(R)"]:
                col_hex_ids_empty, _ = find_neuropil_hex_coords("ME(R)")
                hex1_vals_empty = col_hex_ids_empty["hex1_id"]
                hex2_vals_empty = col_hex_ids_empty["hex2_id"]

            # Hex coordinates of the columns in the chosen neuropil
            col_hex_ids, _ = find_neuropil_hex_coords(roi_str)
            hex1_vals = col_hex_ids["hex1_id"]
            hex2_vals = col_hex_ids["hex2_id"]

            if syn_col_df.empty:  # If there are no cells/synapses in that neuropil
                col_hex_ids["count"] = 0
                if roi_str in ["LO(R)", "LOP(R)"]:
                    # Add grey hexagons for columns that are not present in the neuropil
                    fig.add_trace(
                        go.Scatter(
                            x=-1 * (hex1_vals_empty - hex2_vals_empty)
                          , y=hex1_vals_empty + hex2_vals_empty
                          , mode="markers"
                          , marker_symbol=symbol_number
                          , marker={
                                "size": sizing["markersize"]
                              , "color": "lightgrey"
                              , "line": {
                                    "width": sizing["markerlinewidth"]
                                  , "color": style["markerlinecolor"]
                                }
                            }
                          , showlegend=False
                        )
                      , row=1
                      , col=col_idx
                    )

                if (
                    roi_str == "LOP(R)"
                ):  # Add empty white 'background' hexagons with cbar
                    fig.add_trace(
                        go.Scatter(
                            x=-1 * (hex1_vals - hex2_vals)
                          , y=hex1_vals + hex2_vals
                          , mode="markers"
                          , marker_symbol=symbol_number
                          , marker={
                                "size": sizing["markersize"]
                              , "color": "white"
                              , "cmin": 0
                              , "cmax": maxval
                              , "line": {
                                    "width": sizing["markerlinewidth"]
                                  , "color": "lightgrey"
                                }
                              , "colorbar": {
                                    "orientation": "v"
                                  , "outlinecolor": style["linecolor"]
                                  , "outlinewidth": sizing["axislinewidth"]
                                  , "thickness": sizing["cbar_thickness"]
                                  , "len": sizing["cbar_len"]
                                  , "tickmode": "array"
                                  , "tickvals": tickvals
                                  , "ticktext": tickvals_text
                                  , "ticklen": sizing["ticklen"]
                                  , "tickwidth": sizing["tickwidth"]
                                  , "tickcolor": style["linecolor"]
                                  , "tickfont": {
                                        "size": fsize_ticks_px
                                      , "family": style["font_type"]
                                      , "color": style["linecolor"]
                                    }
                                  , "title": {
                                        "font": {
                                            "family": style["font_type"]
                                          , "size": fsize_title_px
                                          , "color": style["linecolor"]
                                        }
                                      , "side": "right"
                                      , "text": cbar_title
                                    }
                                }
                              , "colorscale": cmap
                            }
                          , showlegend=False
                        )
                      , row=1
                      , col=col_idx
                    )
                else:  # Add empty white 'background' hexagons without cbar for ME and LO
                    fig.add_trace(
                        go.Scatter(
                            x=-1 * (hex1_vals - hex2_vals)
                          , y=hex1_vals + hex2_vals
                          , mode="markers"
                          , marker_symbol=symbol_number
                          , marker={
                                "size": sizing["markersize"]
                              , "color": "white"
                              , "cmin": 0
                              , "cmax": maxval
                              , "line": {
                                    "width": sizing["markerlinewidth"]
                                  , "color": "lightgrey"
                                }
                              , "colorscale": cmap
                            }
                          , showlegend=False
                        )
                      , row=1
                      , col=col_idx
                    )
            else:
                if roi_str in ["LO(R)", "LOP(R)"]:
                    # Add grey hexagons for columns that are not present in the neuropil
                    fig.add_trace(
                        go.Scatter(
                            x=-1 * (hex1_vals_empty - hex2_vals_empty)
                          , y=hex1_vals_empty + hex2_vals_empty
                          , mode="markers"
                          , marker_symbol=symbol_number
                          , marker={
                                "size": sizing["markersize"]
                              , "color": "lightgrey"
                              , "line": {
                                    "width": sizing["markerlinewidth"]
                                  , "color": style["markerlinecolor"]
                                }
                            }
                          , showlegend=False
                        )
                      , row=1
                      , col=col_idx
                    )

                # Add empty white 'background' hexagons - all neuropils
                fig.add_trace(
                    go.Scatter(
                        x=-1 * (hex1_vals - hex2_vals)
                      , y=hex1_vals + hex2_vals
                      , mode="markers"
                      , marker_symbol=symbol_number
                      , marker={
                            "size": sizing["markersize"]
                          , "color": "white"
                          , "line": {
                                "width": sizing["markerlinewidth"]
                              , "color": "lightgrey"
                            }
                        }
                      , showlegend=False
                    )
                  , row=1
                  , col=col_idx
                )

                if roi_str == "LOP(R)":
                    fig.add_trace(
                        go.Scatter(
                            x=-1 * (syn_col_df["hex1_id"] - syn_col_df["hex2_id"])
                          , y=(syn_col_df["hex1_id"] + syn_col_df["hex2_id"])
                          , mode="markers"
                          , marker_symbol=symbol_number
                          , marker={
                                "cmin": 0
                              , "cmax": maxval
                              , "size": sizing["markersize"]
                              , "color": syn_col_df[f"{col_name}"]
                              , "line": {
                                    "width": sizing["markerlinewidth"]
                                  , "color": style["markerlinecolor"]
                                }
                              , "colorbar": {
                                    "orientation": "v"
                                  , "outlinecolor": style["linecolor"]
                                  , "outlinewidth": sizing["axislinewidth"]
                                  , "thickness": sizing["cbar_thickness"]
                                  , "len": sizing["cbar_len"]
                                  , "tickmode": "array"
                                  , "tickvals": tickvals
                                  , "ticktext": tickvals_text
                                  , "ticklen": sizing["ticklen"]
                                  , "tickwidth": sizing["tickwidth"]
                                  , "tickcolor": style["linecolor"]
                                  , "tickfont": {
                                        "size": fsize_ticks_px
                                      , "family": style["font_type"]
                                      , "color": style["linecolor"]
                                    }
                                  , "title": {
                                        "font": {
                                            "family": style["font_type"]
                                          , "size": fsize_title_px
                                          , "color": style["linecolor"]
                                        }
                                      , "side": "right"
                                      , "text": cbar_title
                                    }
                                }
                              , "colorscale": cmap
                            }
                          , showlegend=False
                        )
                      , row=1
                      , col=col_idx
                    )
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=-1 * (syn_col_df["hex1_id"] - syn_col_df["hex2_id"])
                          , y=syn_col_df["hex1_id"] + syn_col_df["hex2_id"]
                          , mode="markers"
                          , marker_symbol=symbol_number
                          , marker={
                                "cmin": 0
                              , "cmax": maxval
                              , "size": sizing["markersize"]
                              , "color": syn_col_df[f"{col_name}"]
                              , "line": {
                                    "width": sizing["markerlinewidth"]
                                  , "color": style["markerlinecolor"]
                                }
                              , "colorscale": cmap
                            }
                          , showlegend=False
                        )
                      , row=1
                      , col=col_idx
                    )

    # add ME, LO, LOP labels
    x_pos_dict = {"ME": 0.008, "LO": 0.385, "LOP": 0.755}
    for region in ["ME", "LO", "LOP"]:
        fig.add_annotation(
            x=x_pos_dict[region]
          , y=0.97
          , xref="paper"
          , yref="paper"
          , text=region
          , showarrow=False
          , font={"size": 40, "color": "lightgrey", "family": "arial"}
        )

    fig.update_coloraxes(colorbar_tickformat=".2s")

    if save_fig:
        # Save the image
        pio.kaleido.scope.mathjax = None
        pio.write_image(
            fig
          , fig_save_path / f"{filename}_{'trim' if trim else 'raw'}.{plot_specs['export_type']}"
          , width=area_width
          , height=area_height
        )
    return fig


def plot_per_col_subplot(
    df1:pd.DataFrame
  , df2:pd.DataFrame
  , style:dict
  , sizing:dict
  , plot_specs:dict
  , save_fig:bool=True
):
    """
    Generate a 2x3 hexagonal heatmap subplot, with a continuous colormap,
     of the number of cells per column for two distinct cell types.

    Parameters
    ----------
    df1 : pd.DataFrame
        Pandas dataframe containing the per column data for the first cell
         type that will occupy the first row of the subplot.
        This dataframe must include the columns:
            roi : str
                Optic lobe region of interest, options include 'ME(R)', 'LO(R)' or 'LOP(R)'.
            hex1_id : int
                Hex 1 coordinates of optic lobe columns.
            hex2_id : int
                Hex 2 coordinates of optic lobe columns.
            n_cells : int
                The number of cells within that column.
    df2 : pd.DataFrame
        Pandas dataframe containing the per column data for the second cell
         type that will occupy the second row of the subplot.
        This dataframe must include the columns:
            roi : str
                Optic lobe region of interest, options include 'ME(R)', 'LO(R)' or 'LOP(R)'.
            hex1_id : int
                Hex 1 coordinates of optic lobe columns.
            hex2_id : int
                Hex 2 coordinates of optic lobe columns.
            n_cells : int
                The number of cells within that column.
    style : dict
        Dictionary of parameters defining to the style of the returned plot.
        font_type : str
            Font type used for tick and axis labels.
        markerlinecolor : str
            Marker line color.
        linecolor : str
            Line color.
    sizing : dict
        dictionary of parameters related to the size of the returned plot
            fig_width : int
                Width of returned figure (mm).
            fig_height : int
                Height of returned figure (mm).
            fig_margin : int or float
                Figure margin (mm).
            fsize_ticks_pt : int or float
                Tick font size (pt).
            fsize_title_pt : int or float
                Title font size (pt).
            markersize : int or float
                Marker size.
            ticklen : int or float
                Tick length.
            tickwidth : int or float
                Tick width.
            axislinewidth : int or float
                Axis line width.
            markerlinewidth : int or float
                Marker line width.
            cbar_thickness : int or float
                Thickness of the colorbar.
            cbar_len : int or float
                Length of the colorbar.
    plot_specs : dict
        Dictionary of parameters unique to the returned plot.
            filename : str
                File name to use if saving the figure.
            cmax_cells : int
                Maximum number of cells to use for the shared colorbar.
            cmax_syn : int
                Maximum number of synapses to use for the shared colorbar.
            export_type : str
                Plot export type. Typically 'pdf' or 'svg'.
    save_fig : bool, default = True
        If True will save the returned figure at the location `plot_specs['save_path']`.

    Returns
    -------
    fig : go.Figure
        Plotly 2x3 subplot containing hexagonal heatmaps for the number of cells
         per column in the Medulla, Lobula and Lobula plate for two cell types.
    """
    fig_save_path = Path(find_dotenv()).parent / "results" / "cov_compl" / "subplots"
    col_name = "n_cells"
    maxval = plot_specs["cmax_cells"]

    # saving parameters
    fig_save_path.mkdir(parents=True, exist_ok=True)
    filename = plot_specs["filename"]

    # sizing of the figure and font
    if plot_specs["export_type"] in ["svg"]:
        pixelsperinch = 72
    else:
        pixelsperinch = 96
    pixelspermm = pixelsperinch / 25.4
    area_width = (sizing["fig_width"] - sizing["fig_margin"]) * pixelspermm
    area_height = (sizing["fig_height"] - sizing["fig_margin"]) * pixelspermm
    fsize_ticks_px = sizing["fsize_ticks_pt"] * (1 / 72) * pixelsperinch
    fsize_title_px = sizing["fsize_title_pt"] * (1 / 72) * pixelsperinch

    # initiate plot
    fig = make_subplots(rows=2, cols=3, horizontal_spacing=0, vertical_spacing=0)
    fig.update_layout(
        autosize=False
      , height=area_height
      , width=area_width
      , margin={
            "l": 0
          , "r": 0
          , "b": 0
          , "t": 0
          , "pad": 0
        }
      , paper_bgcolor="rgba(255,255,255,255)"
      , plot_bgcolor="rgba(255,255,255,255)"
    )
    fig.update_xaxes(
        showgrid=False, showticklabels=False, showline=False, visible=False
    )
    fig.update_yaxes(
        showgrid=False,
        showticklabels=False,
        showline=False,
        visible=False,
    )

    symbol_number = 15
    tickvals, tickvals_text, cmap = ticks_from_max(maxval)

    for row_idx, df in enumerate([df1, df2], start=1):
        for col_idx, roi_str in enumerate(["ME(R)", "LO(R)", "LOP(R)"], start=1):
            syn_col_df = df[df["roi"] == roi_str]

            # Hex coordinates of medulla columns that do not exist in LO / LOP
            if roi_str in ["LO(R)", "LOP(R)"]:
                col_hex_ids_empty, _ = find_neuropil_hex_coords("ME(R)")
                hex1_vals_empty = col_hex_ids_empty["hex1_id"]
                hex2_vals_empty = col_hex_ids_empty["hex2_id"]

            # Hex coordinates of the columns in the chosen neuropil
            col_hex_ids, _ = find_neuropil_hex_coords(roi_str)
            hex1_vals = col_hex_ids["hex1_id"]
            hex2_vals = col_hex_ids["hex2_id"]

            if syn_col_df.empty:  # If there are no cells/synapses in that neuropil

                col_hex_ids["count"] = 0

                if roi_str in ["LO(R)", "LOP(R)"]:
                    # Add grey hexagons for columns that are not present in the neuropil
                    fig.add_trace(
                        go.Scatter(
                            x=-1 * (hex1_vals_empty - hex2_vals_empty)
                          , y=hex1_vals_empty + hex2_vals_empty
                          , mode="markers"
                          , marker_symbol=symbol_number
                          , marker={
                                "size": sizing["markersize"]
                              , "color": "lightgrey"
                              , "line": {
                                    "width": sizing["markerlinewidth"]
                                  , "color": style["markerlinecolor"]
                                }
                            }
                          , showlegend=False
                        )
                      , row=row_idx
                      , col=col_idx
                    )

                if roi_str == "LOP(R)":  # Add empty white 'background' hexagons with cbar
                    fig.add_trace(
                        go.Scatter(
                            x=-1 * (hex1_vals - hex2_vals)
                          , y=hex1_vals + hex2_vals\
                          , mode="markers"
                          , marker_symbol=symbol_number
                          , marker={
                                "size": sizing["markersize"]
                              , "color": "white"
                              , "cmin": 0
                              , "cmax": maxval
                              , "line": {
                                    "width": sizing["markerlinewidth"]
                                  , "color": "lightgrey"
                                }
                              , "colorbar": {
                                    "orientation": "v"
                                  , "x": 0.98
                                  , "y": 0.58
                                  , "outlinecolor": style["linecolor"]
                                  , "outlinewidth": sizing["axislinewidth"]
                                  , "thickness": sizing["cbar_thickness"]
                                  , "len": sizing["cbar_len"]
                                  , "tickmode": "array"
                                  , "tickvals": tickvals
                                  , "ticktext": tickvals_text
                                  , "ticklen": sizing["ticklen"]
                                  , "tickwidth": sizing["tickwidth"]
                                  , "tickcolor": style["linecolor"]
                                  , "tickfont": {
                                        "size": fsize_ticks_px
                                      , "family": style["font_type"]
                                      , "color": style["linecolor"]
                                    }
                                }
                              , "colorscale": cmap
                            }
                          , showlegend=False
                        )
                      , row=row_idx
                      , col=col_idx
                    )
                else:  # Add empty white 'background' hexagons without cbar for ME and LO
                    fig.add_trace(
                        go.Scatter(
                            x=-1 * (hex1_vals - hex2_vals)
                          , y=hex1_vals + hex2_vals
                          , mode="markers"
                          , marker_symbol=symbol_number
                          , marker={
                                "size": sizing["markersize"]
                              , "color": "white"
                              , "cmin": 0
                              , "cmax": maxval
                              , "line": {
                                    "width": sizing["markerlinewidth"]
                                  , "color": "lightgrey"
                                }
                              , "colorscale": cmap
                            }
                          , showlegend=False
                        )
                      , row=row_idx
                      , col=col_idx
                    )
            else:
                if roi_str in ["LO(R)", "LOP(R)"]:
                    # Add grey hexagons for columns that are not present in the neuropil
                    fig.add_trace(
                        go.Scatter(
                            x=-1 * (hex1_vals_empty - hex2_vals_empty)
                          , y=hex1_vals_empty + hex2_vals_empty
                          , mode="markers"
                          , marker_symbol=symbol_number
                          , marker={
                                "size": sizing["markersize"]
                              , "color": "lightgrey"
                              , "line": {
                                    "width": sizing["markerlinewidth"]
                                  , "color": style["markerlinecolor"]
                                }
                            }
                          , showlegend=False
                        )
                      , row=row_idx
                      , col=col_idx
                    )

                # Add empty white 'background' hexagons - all neuropils
                fig.add_trace(
                    go.Scatter(
                        x=-1 * (hex1_vals - hex2_vals)
                      , y=hex1_vals + hex2_vals
                      , mode="markers"
                      , marker_symbol=symbol_number
                      , marker={
                            "size": sizing["markersize"]
                          , "color": "white"
                          , "line": {
                                "width": sizing["markerlinewidth"]
                              , "color": "lightgrey"
                            }
                        }
                      , showlegend=False
                    )
                  , row=row_idx
                  , col=col_idx
                )

                if roi_str == "LOP(R)":
                    fig.add_trace(
                        go.Scatter(
                            x=-1 * (syn_col_df["hex1_id"] - syn_col_df["hex2_id"])
                          , y=syn_col_df["hex1_id"] + syn_col_df["hex2_id"]
                          , mode="markers"
                          , marker_symbol=symbol_number
                          , marker={
                                "cmin": 0
                              , "cmax": maxval
                              , "size": sizing["markersize"]
                              , "color": syn_col_df[f"{col_name}"]
                              , "line": {
                                    "width": sizing["markerlinewidth"]
                                  , "color": style["markerlinecolor"]
                                }
                              , "colorbar": {
                                    "orientation": "v"
                                  , "x": 0.98
                                  , "y": 0.58
                                  , "outlinecolor": style["linecolor"]
                                  , "outlinewidth": sizing["axislinewidth"]
                                  , "thickness": sizing["cbar_thickness"]
                                  , "len": sizing["cbar_len"]
                                  , "tickmode": "array"
                                  , "tickvals": tickvals
                                  , "ticktext": tickvals_text
                                  , "ticklen": sizing["ticklen"]
                                  , "tickwidth": sizing["tickwidth"]
                                  , "tickcolor": style["linecolor"]
                                  , "tickfont": {
                                        "size": fsize_ticks_px
                                      , "family": style["font_type"]
                                      , "color": style["linecolor"]
                                    }
                                }
                              , "colorscale": cmap
                            }
                          , showlegend=False
                        )
                      , row=row_idx
                      , col=col_idx
                    )
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=-1 * (syn_col_df["hex1_id"] - syn_col_df["hex2_id"])
                          , y=syn_col_df["hex1_id"] + syn_col_df["hex2_id"]
                          , mode="markers"
                          , marker_symbol=symbol_number
                          , marker={
                                "cmin": 0
                              , "cmax": maxval
                              , "size": sizing["markersize"]
                              , "color": syn_col_df[f"{col_name}"]
                              , "line": {
                                    "width": sizing["markerlinewidth"]
                                  , "color": style["markerlinecolor"]
                                }
                              , "colorscale": cmap
                            }
                          , showlegend=False
                        )
                      , row=row_idx
                      , col=col_idx
                    )

    fig.add_annotation(
        x=plot_specs["cbar_title_x"]
      , y=plot_specs["cbar_title_y"]
      , xref="paper"
      , yref="paper"
      , text=plot_specs["cbar_title"]
      , showarrow=False
      , font={"family": style["font_type"], "size": fsize_title_px, "color": "black"}
    )

    if save_fig:
        # Save the image
        pio.kaleido.scope.mathjax = None
        pio.write_image(
            fig
          , fig_save_path / f"{filename}_{sizing['fig_width']}w"\
            f"_{sizing['fig_height']}h.{plot_specs['export_type']}"
          , width=area_width
          , height=area_height
        )

    return fig


def ticks_from_max(maxval:int) -> tuple[np.array, int, list]:
    """
    Generates the colormap and tick values for the column-based heatmap 
    plots based on the maximum value in the dataset. All colormaps generated 
    are discrete colormaps with a maximum of 5 red values. 
    
    If 'maxval' <= 5 then the generated colormap consists of 'maxval' red
    colors plus white for 0. Ticks are placed in the middle of each color
    block on the colormap. 

    If 'maxval' > 5, then the range between 0.001 and maxval is split into
    5 groups and ticks are placed at the edge of the colorblocks to represent
    the range of values represented by that color. Each color represents values
    from the lower tick value, up to and including the upper tick value. Zero 
    is always represented as white.

    See `src.tests.test_colormap_generation.ipynb` for a method to test the 
    generated colormap for a given value of 'maxval'. 

    Parameters
    ----------
    maxval : int
        The maximum value to be used for the colormap.

    Returns
    -------
    tickvals : np.array
        Position of the colorbar tick values.
    tickvals_text :  int
        Labels of the colorbar tick values.
    cmap : list
        Colormap to be used in the heatmap.
    """
    cmapp = Colormap("reds_5").lut()

    # Assert that x is an integer
    assert isinstance(maxval, int), f"Expected an integer, got {type(maxval).__name__}."
    assert maxval >= 0, f"number must be unsigned integer (>=0), not {maxval}."

    # Create the colormap based on the maxval.
    if 1 < maxval <= 5: # 2…5
        v = []
        for idx in range(0, maxval, 1):
            v.append(Color(cmapp[idx]).hex)
        vv = v[0:maxval:1]
        frac = 1 / (maxval + 1)
        a = np.repeat(np.linspace(frac, 1, len(vv) + 1), 2)[1:]
        b = np.insert(a, 0, [0, frac])
        c = np.repeat(vv, 2)
        d = np.insert(c, 0, ["#FFFFFF", "#FFFFFF"])
        nx = maxval + 1
    elif maxval <= 1: # 0, 1
        v = [Color(cmapp[1]).hex]
        vv = v
        frac = 1 / (maxval + 1)
        a = np.repeat(np.linspace(frac, 1, len(vv) + 1), 2)[1:]
        b = np.insert(a, 0, [0, frac])
        c = np.repeat(vv, 2)
        d = np.insert(c, 0, ["#FFFFFF", "#FFFFFF"])
        nx = 2
    else: # > 5
        v = []
        for idx in range(0, 5, 1):
            v.append(Color(cmapp[idx]).hex)
        vv = v[0:5:1]
        a = np.repeat(np.linspace(0.001, 1, len(vv) + 1), 2)[1:]
        b = np.insert(a, 0, [0, 0.001])
        c = np.repeat(vv, 2)
        d = np.insert(c, 0, ["#FFFFFF", "#FFFFFF"])
        nx = 6

    cmap = list(zip(b, d))

    tickvals = np.linspace(0, maxval, nx)

    # Where to position the tick marks:
    # If maxval between 1 and 5, shift the ticks to the middle of the colorblock,
    # else if maxval > 5 keep them at the edges of the colorblock.
    # If maxval = 0, then the colormap is just white.
    if 0 < maxval <= 5:  # 1…5
        f2 = maxval / (maxval + 1)
        tickvals_text = tickvals
        tickvals = tickvals * f2
        tickinterval = (tickvals[1] - tickvals[0]) / 2
        tickvals = tickvals + tickinterval
    elif maxval >= 5:  # >5
        tickvals_text = np.floor(tickvals)
    elif maxval == 0: # 0
        tickvals = [0.5]
        tickvals_text = [0]

    return tickvals, tickvals_text, cmap


def plot_per_col_simple(
    df:pd.DataFrame
  , roi_str:str
  , style:dict
  , sizing:dict
  , plot_specs:dict
  , max_val:int=None
  , save_fig:bool=True
) -> go.Figure:
    """
    Generate a hexagonal plot of a data frame column value per ROI column in ONE neuropil.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to include the columns 'hex1_id', 'hex2_id', and a column that contains
          numerical data to plot per colum which is specified in plot_specs['column_to_plot']
    roi_str : str
        Neuprint ROI, can only be ME(R), LO(R), LOP(R).
    style : dict
        Dict containing the values of the fixed styling formatting variables.
    sizing : dict
        Dict containing the values of the size formatting variables.
    plot_specs : dict
        Dict containing the values of the formatting variables relevant to the specific plot.
    max_val : int, default=None
        Maximum colormap value. If max_val=None, the value is the maximum value
        in df[plot_specs['column_to_plot']].
    save_fig : bool, default=True
        If True, saves the generated figure to a file. If False, the figure is displayed
        but not saved.

    Returns
    -------
    fig : go.Figure

    """
    assert isinstance(roi_str, str) and roi_str in [
        "ME(R)"
      , "LO(R)"
      , "LOP(R)"
    ], f"ROI must be one of 'ME(R)', 'LO(R)', 'LOP(R)', but is actually '{roi_str}'"

    col_name = plot_specs["column_to_plot"]
    fig_save_path = Path(plot_specs["save_path"], col_name)

    filename = plot_specs["filename"]

    # sizing of the figure and font
    if plot_specs["export_type"] in ["svg", "pdf"]:
        pixelsperinch = 72
    else:
        pixelsperinch = 96
    pixelspermm = pixelsperinch / 25.4
    area_width = (sizing["fig_width"] - sizing["fig_margin"]) * pixelspermm
    area_height = (sizing["fig_height"] - sizing["fig_margin"]) * pixelspermm
    fsize_ticks_px = sizing["fsize_ticks_pt"] * (1 / 72) * pixelsperinch
    fsize_title_px = sizing["fsize_title_pt"] * (1 / 72) * pixelsperinch

    # initiate plot
    fig = go.Figure()

    fig.update_layout(
        autosize=False
      , height=area_height
      , width=area_width
      , margin={"l": 0, "r": 0, "b": 0, "t": 0, "pad": 0}
      , paper_bgcolor="rgba(255,255,255,255)"
      , plot_bgcolor="rgba(255,255,255,255)"
    )
    fig.update_xaxes(
        showgrid=False, showticklabels=False, showline=False, visible=False
    )
    fig.update_yaxes(
        showgrid=False, showticklabels=False, showline=False, visible=False
    )
    # Symbol number to choose to plot hexagons
    symbol_number = 15

    # Find the maximum value and create colormap
    if max_val is None:
        max_val = max(df[col_name])

    if "cmap" in plot_specs:
        cmap = plot_specs["cmap"]
        tickvals = plot_specs["tickvals"]
        tickvals_text = plot_specs["tickvals_text"]
    else:
        tickvals, tickvals_text, cmap = ticks_from_max(max_val)

    # Hex coordinates of the columns in the chosen neuropil
    col_hex_ids, _ = find_neuropil_hex_coords(roi_str)
    hex1_vals = col_hex_ids["hex1_id"]
    hex2_vals = col_hex_ids["hex2_id"]

    # Hex coordinates of medulla columns that do not exist in LO / LOP
    if roi_str in ["LO(R)", "LOP(R)"]:
        col_hex_ids_empty, _ = find_neuropil_hex_coords("ME(R)")
        hex1_vals_empty = col_hex_ids_empty["hex1_id"]
        hex2_vals_empty = col_hex_ids_empty["hex2_id"]

    if roi_str in ["LO(R)", "LOP(R)"]:
        # Add grey hexagons for columns that are not present in the neuropil
        fig.add_trace(
            go.Scatter(
                x=-1 * (hex1_vals_empty - hex2_vals_empty)
              , y=(hex1_vals_empty + hex2_vals_empty)
              , mode="markers"
              , marker_symbol=symbol_number
              , marker={
                    "size": sizing["markersize"]
                  , "color": "lightgrey"
                  , "line": {
                        "width": sizing["markerlinewidth"]
                      , "color": style["markerlinecolor"]
                    }
                }
              , showlegend=False
            )
        )

    # Add empty white 'background' hexagons - all neuropils
    fig.add_trace(
        go.Scatter(
            x=-1 * (hex1_vals - hex2_vals)
          , y=hex1_vals + hex2_vals
          , mode="markers"
          , marker_symbol=symbol_number
          , marker={
                "size": sizing["markersize"]
              , "color": "white"
              , "line": {"width": sizing["markerlinewidth"], "color": "lightgrey"}
            }
          , showlegend=False
        )
    )

    # Add data from chosen column
    fig.add_trace(
        go.Scatter(
            x=-1 * (df["hex1_id"] - df["hex2_id"])
          , y=df["hex1_id"] + df["hex2_id"]
          , mode="markers"
          , marker_symbol=symbol_number
          , marker={
                "cmin": 0
              , "cmax": max_val
              , "size": sizing["markersize"]
              , "color": df[f"{col_name}"]
              , "line": {
                    "width": sizing["markerlinewidth"]
                  , "color": style["markerlinecolor"]
                }
              , "colorbar": {
                    "orientation": "v"
                  , "outlinecolor": style["linecolor"]
                  , "outlinewidth": sizing["axislinewidth"]
                  , "thickness": sizing["cbar_thickness"]
                  , "len": sizing["cbar_len"]
                  , "tickmode": "array"
                  , "tickvals": tickvals
                  , "ticktext": tickvals_text
                  , "ticklen": sizing["ticklen"]
                  , "tickwidth": sizing["tickwidth"]
                  , "tickcolor": style["linecolor"]
                  , "tickfont": {
                        "size": fsize_ticks_px
                      , "family": style["font_type"]
                      , "color": style["linecolor"]
                    }
                  , "title": {
                        "font": {
                            "family": style["font_type"]
                          , "size": fsize_title_px
                          , "color": style["linecolor"]
                        }
                      , "side": "right"
                    }
                }
              , "colorscale": cmap
            }
          , showlegend=False
        )
    )

    if save_fig:
        fig_save_path.mkdir(parents=True, exist_ok=True)
        # Save the image
        pio.kaleido.scope.mathjax = None
        pio.write_image(
            fig
          , fig_save_path / f"{filename}.{plot_specs['export_type']}"
          , width=area_width
          , height=area_height
        )
    return fig
