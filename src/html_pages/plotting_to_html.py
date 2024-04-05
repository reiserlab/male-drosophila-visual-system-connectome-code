import pickle
import warnings
from pathlib import Path
import navis

from dotenv import find_dotenv
import plotly.graph_objects as go

from utils.plotter import get_mesh, get_roi
from utils.ol_types import OLTypes
from utils.helper import slugify
from utils.ol_color import OL_COLOR


def plot_rois(
    fig
  , roi_list:list
  , resample_precision:float=1.0
):
    """
    Add ROI traces to a given figure.

    Parameters:
    ----------
    fig : go.Figure
        The Plotly figure object to which ROI traces are added.
    roi_list : list
        A list of ROI names to be plotted.
    resample_precision : float
        resampling precision with % of mesh faces to keep (between 0…1)
    """
    for roi in roi_list:
        roi_data = get_roi(roi, ignore_cache=True)
        navis.simplify_mesh(roi_data, F=resample_precision, inplace=True)
        f_roi = navis.plot3d(roi_data, inline=False)
        if f_roi.data:
            fig.add_trace(f_roi.data[0])


def get_dynamic_plot(
    instance
  , resample_precision:float=1.0
  , ignore_cache:bool=False
) -> go.Figure:
    """
    Generates a 3D plot for a given cell type and its associated star neuron, then saves the plot
    as an HTML file.

    Parameters:
    ----------
    instance : str
        The name or identifier of the cell type to be plotted. This is used to name the output
        HTML file and in debugging messages.
    resample_precision : float
        resampling precision with % of mesh faces to keep (between 0…1)
    ignore_cache : bool
        if true, regenerate the dynamic plot from scatch, even if the simplifications
        were calculated before. If false (default), use a cached version of the simplified
        mesh to generate the dynamic plot.

    Returns:
    --------
    fig : go.Figure
        Representation of the figure.

    """

    assert 0 < resample_precision <= 1, "Resampling must be >0 and <=1"

    cachedir = Path(find_dotenv()).parent / "cache" / "html_pages" / "three_d"
    cachedir.mkdir(parents=True, exist_ok=True)
    fig_fn = cachedir / f"{slugify(instance)}_{slugify(resample_precision)}.pickle"
    if not ignore_cache and fig_fn.is_file():
        with fig_fn.open('rb') as fig_fh:
            fig = pickle.load(fig_fh)
    else:
        olt = OLTypes()
        star_neuron = olt.get_star(instance_str=instance)

        mesh = get_mesh(body_id=star_neuron)

        navis.downsample_neuron(mesh, downsampling_factor=1.0/resample_precision, inplace=True)
        fig = go.Figure()

        # colr = (227/255, 66/255, 52/255, 0.7)  # Convert to 0-1 scale for RGB
        m_g = olt.get_main_group(instance[:-2])
        clr_map = {'OL_intrinsic':0, 'OL_connecting':1, 'VPN':2, 'VCN':3, 'other':4}
        colr = OL_COLOR.OL_TYPES.rgba[clr_map[m_g]][:3]


        # Plot the region ROIs
        roi_list = ['ME(R)', 'LO(R)', 'LOP(R)']
        plot_rois(
            fig=fig
          , roi_list=roi_list
          , resample_precision=0.05
        )

        # Plot the cell
        fig_mesh = navis.plot3d(mesh, inline=False, color=colr)

        if fig_mesh.data:
            cell_type = instance[:-2]
            trace = fig_mesh.data[0]
            trace.name = cell_type + f" ({instance[-1]})"
            fig.add_trace(trace)

        # Update the layout:
        fig.update_layout(
            margin={'l': 0, 'r': 0, 'b': 10, 't': 0}
          , showlegend=True
          , legend={'x':0, 'y': 1, 'font': {'color': "black"}}
          , scene = {
                "xaxis" : {
                    "showgrid": False
                  , "gridcolor": 'rgba(0,0,0,0)'
                  , "showbackground": False
                  , "visible": False
                }
              , "yaxis" : {
                    "showgrid": False
                  , "gridcolor": 'rgba(0,0,0,0)'
                  , "showbackground": False
                  , "visible": False
                }
              , "zaxis" : {
                    "showgrid":False
                  , "gridcolor": 'rgba(0,0,0,0)'
                  , "showbackground":False
                  , "visible":False
                }
              , 'aspectmode': 'data'
              , 'camera': {
                    'center': {
                        'x': -2.220446049250313e-16
                      , 'y': 2.220446049250313e-16
                      , 'z': 6.661338147750939e-16
                    }
                  , 'eye': {
                        'x': 0.7151943936701185
                      , 'y': 1.4171877426035864
                      , 'z': 0.14149842193542092
                    }
                  , 'projection': {'type': 'perspective'}
                  , 'up': {
                        'x': -0.7078481105324922
                      , 'y': 0.41433972841506905
                      , 'z': -0.5720783529137489
                    }
                }
            }
          , paper_bgcolor='white'
          , height=400
          , width=600
        )
        with fig_fn.open('wb') as fig_fh:
            pickle.dump(fig, fig_fh)

    #debugging
    if not fig.data:
        warnings.warn(f"Empty figure for {instance}.")

    return fig
