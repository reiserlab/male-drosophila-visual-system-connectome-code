"""
Functions to generate quality control plots to check data completeness
"""

from pathlib import Path

from dotenv import find_dotenv
from cmap import Colormap

from utils.column_plotting_functions import plot_per_col_simple
from utils.scatterplot_functions import make_completeness_scatter
from utils.completion_metrics import fetch_cxn_df


def generate_completeness_plots(roi_str: str):
    """
    Function to generate hexagonal heatmap and scatterplots of the synapse
      / connection completeness per column for the neuropil 'roi_str'.

    Parameters
    ----------
    roi_str : str
        Optic lobe region of interest for which to generate plots.
    """
    assert isinstance(roi_str, str) and roi_str in ["ME(R)", "LO(R)", "LOP(R)"]\
      , f"ROI must be one of 'ME(R)', 'LO(R)', 'LOP(R)', but is actually '{roi_str}'"

    plots = []

    cxn_df = fetch_cxn_df(roi_str=roi_str)

    # Style and sizing for hexagonal heatmap plots:
    style_hex = {
        "font_type": "arial"
      , "markerlinecolor": "rgba(0,0,0,0)"      # transparent
      , "linecolor": "black"
    }

    sizing_hex = {
        "fig_width": 240                        # units = mm
      , "fig_height": 210                       # units = mm
      , "fig_margin": 0
      , "fsize_ticks_pt": 20
      , "fsize_title_pt": 20
      , "markersize": 21
      , "ticklen": 10
      , "tickwidth": 3
      , "axislinewidth": 3
      , "markerlinewidth": 0.9
      , "cbar_thickness": 20
      , "cbar_len": 0.75
      , "cbar_title": "Completeness (captured/all)"
    }

    save_path = Path(find_dotenv()).parent / 'results' / 'completeness'

    for col_to_plot in ["n_pre_ratio", "n_post_ratio", "n_up_ratio"]:

        plot_specs = {
            "filename": f"{col_to_plot}_captured-to-all_ratio_{roi_str[:-3]}"
          , "export_type": "png"
          , "column_to_plot": col_to_plot
          , "save_path": save_path
          , "cmap": Colormap("reds_5").to_plotly()
          , "tickvals": [0, 0.2, 0.4, 0.6, 0.8, 1]
          , "tickvals_text": ["0", "0.2", "0.4", "0.6", "0.8", "1"]
        }

        plot_per_col_simple(
            df=cxn_df
          , roi_str=roi_str
          , style=style_hex
          , sizing=sizing_hex
          , plot_specs=plot_specs
          , max_val=1
          , save_fig=True
        )

    # Style and sizing for scatter plots:
    sizing_scatter = {
        "fig_width": 240                        # units = mm
      , "fig_height": 240                       # units = mm
      , "fig_margin": 0
      , "fsize_ticks_pt": 12
      , "fsize_title_pt": 13
      , "markersize": 6
      , "ticklen": 5
      , "tickwidth": 1.8
      , "linewidth": 1.8
      , "markerlinewidth": 0.8
    }

    for xval in ["n_up_seg", "n_pre_seg", "n_post_seg"]:

        yval = f"{xval[:-3]}neu"

        if "up" in xval:
            xtitle = "Total connections"
            ytitle = "Captured connections"
        else:
            xtitle = "Total synapses"
            ytitle = "Captured synapses"

        if "pre" in xval:
            if roi_str == "ME(R)":
                x_lim = 8000
                y_lim = x_lim
            else:
                x_lim = 7000
                y_lim = x_lim
        else:
            match roi_str:
                case "ME(R)":
                    x_lim = 50000
                    y_lim = 30000
                case "LO(R)":
                    x_lim = 35000
                    y_lim = 20000
                case _: # "LOP(R)"
                    x_lim = 25000
                    y_lim = 20000

        plot_specs = {
            "filename": f"{roi_str[:-3]}_scatter_x-totalsyn_y-captured_medianline_{xval[:-4]}"
          , "export_type": "pdf"
          , "xval": xval
          , "yval": yval
          , "xlim": [0, x_lim]
          , "ylim": [0, y_lim]
          , "xaxis_title": xtitle
          , "yaxis_title": ytitle
          , "save_path": save_path / 'scatter'
        }

        plots.append(
            make_completeness_scatter(
                roi_str=roi_str
              , style=style_hex
              , sizing=sizing_scatter
              , plot_specs=plot_specs
              , save_fig=True
            )
        )
    return plots
