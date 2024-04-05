"""
Functions used for generating plots in the notebook
`src/column_features/column_features_analysis.ipynb`
"""

import os
from pathlib import Path

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
  , trim:bool=False
  , save_fig:bool=True
) -> go.Figure:
    """
    Plot synapses per column.

    plot_type = 'synapses' or 'cells'
    """

    assert plot_type in ['synapses', 'cells'],\
        f"plot_type must be 'synapses' or 'cells', not '{plot_type}'"

    if trim is False:
        trim_string = "raw"
    elif trim is True:
        trim_string = "trim"

    if plot_type == "synapses":
        if save_fig:
            fig_save_path = Path(plot_specs['save_path'],'syn_per_col')
        col_name = "n_syn"
        maxval = plot_specs["cmax_syn"]
        cbar_title = "Number of synapses"
    elif plot_type == "cells":
        if save_fig:
            fig_save_path = Path(plot_specs['save_path'],'cells_per_col')
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
    w = (sizing["fig_width"] - sizing["fig_margin"]) * pixelspermm
    h = (sizing["fig_height"] - sizing["fig_margin"]) * pixelspermm
    fsize_ticks_px = sizing["fsize_ticks_pt"] * (1 / 72) * pixelsperinch
    fsize_title_px = sizing["fsize_title_pt"] * (1 / 72) * pixelsperinch

    # initiate plot
    fig = make_subplots(rows=1, cols=3, subplot_titles=("ME", "LO", "LOP"))
    fig.update_layout(
        autosize=False,
        height=h,
        width=w,
        margin={
            "l": 0,  # w//15,
            "r": 0,  # w//4,
            "b": 0,  # h//10,
            "t": 0,  # h//10,
            "pad": 0,  # 8
        },
        paper_bgcolor="rgba(255,255,255,255)",
        plot_bgcolor="rgba(255,255,255,255)",
    )
    fig.update_xaxes(
        showgrid=False, showticklabels=False, showline=False, visible=False
    )
    fig.update_yaxes(
        showgrid=False, showticklabels=False, showline=False, visible=False
    )
    col_idx = 1
    symbol_number = 15
   
    if maxval == 0:
        tickvals = [1.25]
        tickvals_text = ['0']
        _, _, cmap = ticks_from_max(1)
        for roi_str in ["ME(R)", "LO(R)", "LOP(R)"]:

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
                        x=-1 * (hex1_vals_empty - hex2_vals_empty),
                        y=(hex1_vals_empty + hex2_vals_empty),
                        mode="markers",
                        marker_symbol=symbol_number,
                        marker={
                            "size": sizing["markersize"],
                            "color": "lightgrey",
                            "line": {
                                "width": sizing["markerlinewidth"],
                                "color": style["markerlinecolor"],
                            },
                        },
                        showlegend=False,
                    ),
                    row=1,
                    col=col_idx,
                )

            if roi_str == "LOP(R)":  # Add empty white 'background' hexagons with cbar
                fig.add_trace(
                    go.Scatter(
                        x=-1 * (hex1_vals - hex2_vals),
                        y=(hex1_vals + hex2_vals),
                        mode="markers",
                        marker_symbol=symbol_number,
                        marker={
                            "size": sizing["markersize"],
                            "color": "white",
                            "cmin": 0,
                            "cmax": 0,
                            "line": {
                                "width": sizing["markerlinewidth"],
                                "color": "lightgrey",
                            },
                            "colorbar": {
                                "orientation": "v",
                                "outlinecolor": style["linecolor"],
                                "outlinewidth": sizing["axislinewidth"],
                                "thickness": sizing["cbar_thickness"],
                                "len": sizing["cbar_len"],
                                "tickmode": "array",
                                "tickvals": tickvals,
                                "ticktext": tickvals_text,
                                "ticklen": sizing["ticklen"],
                                "tickwidth": sizing["tickwidth"],
                                "tickcolor": style["linecolor"],
                                "tickfont": {
                                    "size": fsize_ticks_px,
                                    "family": style["font_type"],
                                    "color": style["linecolor"],
                                },
                                "title": {
                                    "font": {
                                        "family": style["font_type"],
                                        "size": fsize_title_px,
                                        "color": style["linecolor"],
                                    },
                                    "side": "right",
                                    "text": f"{cbar_title}",
                                },
                            },
                            "colorscale": cmap,
                        },
                        showlegend=False,
                    ),
                    row=1,
                    col=col_idx,
                )
            else:  # Add empty white 'background' hexagons without cbar for ME and LO
                fig.add_trace(
                    go.Scatter(
                        x=-1 * (hex1_vals - hex2_vals),
                        y=(hex1_vals + hex2_vals),
                        mode="markers",
                        marker_symbol=symbol_number,
                        marker={
                            "size": sizing["markersize"],
                            "color": "white",
                            "cmin": 0,
                            "cmax": 0,
                            "line": {
                                "width": sizing["markerlinewidth"],
                                "color": "lightgrey",
                            },
                            "colorscale": cmap,
                        },
                        showlegend=False,
                    ),
                    row=1,
                    col=col_idx,
                )
            col_idx = col_idx+1
    else:
        tickvals, tickvals_text, cmap = ticks_from_max(maxval)

        for roi_str in ["ME(R)", "LO(R)", "LOP(R)"]:
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
                          , showlegend=False,
                        )
                      , row=1
                      , col=col_idx
                    )

                if roi_str == "LOP(R)":  # Add empty white 'background' hexagons with cbar
                    fig.add_trace(
                        go.Scatter(
                            x=-1 * (hex1_vals - hex2_vals),
                            y=(hex1_vals + hex2_vals),
                            mode="markers",
                            marker_symbol=symbol_number,
                            marker={
                                "size": sizing["markersize"],
                                "color": "white",
                                "cmin": 0,
                                "cmax": maxval,
                                "line": {
                                    "width": sizing["markerlinewidth"],
                                    "color": "lightgrey",
                                },
                                "colorbar": {
                                    "orientation": "v",
                                    "outlinecolor": style["linecolor"],
                                    "outlinewidth": sizing["axislinewidth"],
                                    "thickness": sizing["cbar_thickness"],
                                    "len": sizing["cbar_len"],
                                    "tickmode": "array",
                                    "tickvals": tickvals,
                                    "ticktext": tickvals_text,
                                    "ticklen": sizing["ticklen"],
                                    "tickwidth": sizing["tickwidth"],
                                    "tickcolor": style["linecolor"],
                                    "tickfont": {
                                        "size": fsize_ticks_px,
                                        "family": style["font_type"],
                                        "color": style["linecolor"],
                                    },
                                    "title": {
                                        "font": {
                                            "family": style["font_type"],
                                            "size": fsize_title_px,
                                            "color": style["linecolor"],
                                        },
                                        "side": "right",
                                        "text": f"{cbar_title}",
                                    },
                                },
                                "colorscale": cmap,
                            },
                            showlegend=False,
                        ),
                        row=1,
                        col=col_idx,
                    )
                else:  # Add empty white 'background' hexagons without cbar for ME and LO
                    fig.add_trace(
                        go.Scatter(
                            x=-1 * (hex1_vals - hex2_vals),
                            y=(hex1_vals + hex2_vals),
                            mode="markers",
                            marker_symbol=symbol_number,
                            marker={
                                "size": sizing["markersize"],
                                "color": "white",
                                "cmin": 0,
                                "cmax": maxval,
                                "line": {
                                    "width": sizing["markerlinewidth"],
                                    "color": "lightgrey",
                                },
                                "colorscale": cmap,
                            },
                            showlegend=False,
                        ),
                        row=1,
                        col=col_idx,
                    )
            else:
                if roi_str in ["LO(R)", "LOP(R)"]:
                    # Add grey hexagons for columns that are not present in the neuropil
                    fig.add_trace(
                        go.Scatter(
                            x=-1 * (hex1_vals_empty - hex2_vals_empty),
                            y=(hex1_vals_empty + hex2_vals_empty),
                            mode="markers",
                            marker_symbol=symbol_number,
                            marker={
                                "size": sizing["markersize"],
                                "color": "lightgrey",
                                "line": {
                                    "width": sizing["markerlinewidth"],
                                    "color": style["markerlinecolor"],
                                },
                            },
                            showlegend=False,
                        ),
                        row=1,
                        col=col_idx,
                    )

                # Add empty white 'background' hexagons - all neuropils
                fig.add_trace(
                    go.Scatter(
                        x=-1 * (hex1_vals - hex2_vals),
                        y=(hex1_vals + hex2_vals),
                        mode="markers",
                        marker_symbol=symbol_number,
                        marker={
                            "size": sizing["markersize"],
                            "color": "white",
                            "line": {
                                "width": sizing["markerlinewidth"],
                                "color": "lightgrey",
                            },
                        },
                        showlegend=False,
                    ),
                    row=1,
                    col=col_idx,
                )

                if roi_str == "LOP(R)":
                    fig.add_trace(
                        go.Scatter(
                            x=-1 * (syn_col_df["hex1_id"] - syn_col_df["hex2_id"]),
                            y=(syn_col_df["hex1_id"] + syn_col_df["hex2_id"]),
                            mode="markers",
                            marker_symbol=symbol_number,
                            marker={
                                "cmin": 0,
                                "cmax": maxval,
                                "size": sizing["markersize"],
                                "color": syn_col_df[f"{col_name}"],
                                "line": {
                                    "width": sizing["markerlinewidth"],
                                    "color": style["markerlinecolor"],
                                },
                                "colorbar": {
                                    "orientation": "v",
                                    "outlinecolor": style["linecolor"],
                                    "outlinewidth": sizing["axislinewidth"],
                                    "thickness": sizing["cbar_thickness"],
                                    "len": sizing["cbar_len"],
                                    "tickmode": "array",
                                    "tickvals": tickvals,
                                    "ticktext": tickvals_text,
                                    "ticklen": sizing["ticklen"],
                                    "tickwidth": sizing["tickwidth"],
                                    "tickcolor": style["linecolor"],
                                    "tickfont": {
                                        "size": fsize_ticks_px,
                                        "family": style["font_type"],
                                        "color": style["linecolor"],
                                    },
                                    "title": {
                                        "font": {
                                            "family": style["font_type"],
                                            "size": fsize_title_px,
                                            "color": style["linecolor"],
                                        },
                                        "side": "right",
                                        "text": f"{cbar_title}",
                                    },
                                },
                                "colorscale": cmap,
                            },
                            showlegend=False,
                        ),
                        row=1,
                        col=col_idx,
                    )
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=-1 * (syn_col_df["hex1_id"] - syn_col_df["hex2_id"]),
                            y=(syn_col_df["hex1_id"] + syn_col_df["hex2_id"]),
                            mode="markers",
                            marker_symbol=symbol_number,
                            marker={
                                "cmin": 0,
                                "cmax": maxval,
                                "size": sizing["markersize"],
                                "color": syn_col_df[f"{col_name}"],
                                "line": {
                                    "width": sizing["markerlinewidth"],
                                    "color": style["markerlinecolor"],
                                },
                                "colorscale": cmap,
                            },
                            showlegend=False,
                        ),
                        row=1,
                        col=col_idx,
                    )
            col_idx = col_idx + 1

    # add ME, LO, LOP labels
    x_pos_dict={'ME': 0.008, 'LO': 0.385, 'LOP': 0.755}
    for region in ['ME', 'LO', 'LOP']:
        fig.add_annotation(x=x_pos_dict[region],
                        y=0.97,
                        xref = 'paper',
                        yref = 'paper',
                        text=region,
                        showarrow=False,
                        font={'size': 40,
                                'color': 'lightgrey',
                                'family': 'arial'
                                }
                                )

    if save_fig:
        if not os.path.exists(fig_save_path):
            os.makedirs(fig_save_path)
        # Save the image
        pio.kaleido.scope.mathjax = None
        pio.write_image(
            fig,
            os.path.join(
                fig_save_path, f"{filename}_{trim_string}.{plot_specs['export_type']}"
            ),
            width=w,
            height=h,
        )
    return fig


def ticks_from_max(maxval: int):
    """
    Generates the colormap and tick values for the column-based heatmap plots based on the maximum
    value in the dataset. A discretized colormap with a maximum of 5 colors is used for the
    heatmaps. If 'maxval' is less than 5 then only 'maxval' number of colors are used. 
    If maxval <= 5, then 5 colors are used and ticks are placed in the centre of the colors on the
    colorbar, else values between 0.001 and maxval are split into 5 groups and ticks are placed at
    the edge of the colorblocks to represent the range of values represented by that color. Values
    represented by the color range from above the tick value on the left side, up to and including 
    the tick value on the right side.

    Parameters
    ----------
    maxval : int
        The maximum value in the dataset. This would be the maximum number of synapses or 
        cells per column depending on whether coverage or completeness is being analysed.

    Returns
    -------
    tickvals : np.array
        position of the colorbar tick values.
    tickvals_text :  int
        labels of the colorbar tick values.
    cmap : list
        colormap to be used in the heatmap.
    """
    cmapp = Colormap("reds_5").lut()

    # # Create new colormap
    if 1 < maxval <= 5:
        v = []
        for idx in range(0, maxval, 1):
            v.append(Color(cmapp[idx]).hex)
        vv = v[0:maxval:1]
        frac = 1 / (maxval + 1)
        a = np.repeat(np.linspace(frac, 1, len(vv) + 1), 2)[1:]
        b = np.insert(a, 0, [0, frac])
        c = np.repeat(vv, 2)
        d = np.insert(c, 0, ["#FFFFFF", "#FFFFFF"])
    elif maxval <= 1:
        v = [Color(cmapp[1]).hex]
        vv = v
        frac = 1 / (maxval + 1)
        a = np.repeat(np.linspace(frac, 1, len(vv) + 1), 2)[1:]
        b = np.insert(a, 0, [0, frac])
        c = np.repeat(vv, 2)
        d = np.insert(c, 0, ["#FFFFFF", "#FFFFFF"])
    else:
        v = []
        for idx in range(0, 5, 1):
            v.append(Color(cmapp[idx]).hex)
        vv = v[0:5:1]
        a = np.repeat(np.linspace(0.001, 1, len(vv) + 1), 2)[1:]
        b = np.insert(a, 0, [0, 0.001])
        c = np.repeat(vv, 2)
        d = np.insert(c, 0, ["#FFFFFF", "#FFFFFF"])

    cmap = list(zip(b, d))

    # Number of ticks to generate
    if maxval > 5:
        nx = 6
    elif 1 < maxval <= 5:
        nx = maxval + 1
    elif maxval == 1:
        nx = 2

    tickvals = np.linspace(0, maxval, nx)

    # shift ticks to the middle of the colorblock if each color block
    # represents one value, else keep them at the edges of the colorblock.
    if 0 < maxval <= 5:
        f2 = maxval / (maxval + 1)
        tickvals_text = tickvals
        tickvals = tickvals * f2
        tickinterval = (tickvals[1] - tickvals[0]) / 2
        tickvals = tickvals + tickinterval
    elif maxval >= 5:  # else put the ticks at the boundaries
        tickvals_text = np.floor(tickvals)  # .astype(int)
    elif maxval == 0:
        tickvals = [0.5]
        tickvals_text = "0"

    return tickvals, tickvals_text, cmap
