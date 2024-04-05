"""
Function to generate spatial coverage plots for webpages
"""
import pickle
import warnings
from pathlib import Path
from dotenv import find_dotenv

import pandas as pd
import plotly.graph_objects as go

from utils.column_features_functions import\
    find_cmax_across_all_neuropils, hex_from_col, cov_compl_calc
from utils.column_plotting_functions import plot_per_col
from utils.hex_plot_config import HexPlotConfig
from utils.helper import slugify
from utils.trim_helper import TrimHelper

from queries.coverage_queries import fetch_cells_synapses_per_col
from queries.completeness import fetch_ol_types_and_instances


def plot_cells_per_column(
    instance:str
  , trim_helper:TrimHelper=None
) -> go.Figure:
    """
    Plot the "cells per column" figure.

    This function uses an internal cache in the `coverage_cells` directory.

    This function was initially extracted from make_spatial_coverage_plots_for_webpages

    Parameters
    ----------
    instance : str
        cell instance name, e.g. 'TmY5a_R'
    trim_helper : TrimHelper
        Object caching access to the trimmed and raw data frames. If None, it's created
        internally.
    
    Returns
    -------
    fig_synapses : go.Figure
        Cells per column figure.
    """
    export_type = 'png' # this defines the internal resolution of the plot

    cachedir = Path(find_dotenv()).parent / "cache" / "coverage_cells"
    cachedir.mkdir(parents=True, exist_ok=True)

    if trim_helper is None:
        trim_helper = TrimHelper(instance)

    plot_fn = cachedir / f"{slugify(instance)}_{export_type}.pickle"

    if plot_fn.is_file():
        with plot_fn.open('rb') as plot_fh:
            fig_cells = pickle.load(plot_fh)
    else:
        cell_type = instance[:-2]

        trim_df = trim_helper.trim_df

        cfg = HexPlotConfig()

        cc = 0
        cs = 0
        if trim_df.empty:
            print(f'empty cell plot for {cell_type}')
        else:
            cs, cc = find_cmax_across_all_neuropils(
                trim_df, thresh_val=0.98
            )  # thresh_val can be None or float

        plot_specs = {
            "filename": instance,
            "cmax_cells": cc,
            "cmax_syn": cs,
            "export_type": export_type
        }

        fig_cells = plot_per_col(
            trim_df
          , cfg.style
          , cfg.sizing
          , plot_specs
          , plot_type="cells"
          , trim=True
          , save_fig=False
        )

        fig_cells.layout.title = {
            'text': instance,
            'font': {'color': 'lightgrey', 'family': 'arial'},
            'y':0.98,
            'x':0.01,
            'xanchor': 'left',
            'yanchor': 'top'
        }

        with plot_fn.open('wb') as plot_fh:
            pickle.dump(fig_cells, plot_fh)

    return fig_cells


def plot_synapses_per_column(instance:str) -> go.Figure:
    """
    Plot the "synapse per column" figure.

    This function uses an internal cache in the `coverage_synapses` directory.

    This function was initially extracted from make_spatial_coverage_plots_for_webpages

    Parameters
    ----------
    instance : str
        cell instance name, e.g. 'TmY5a_R'
    
    Returns
    -------
    fig_synapses : go.Figure
        Synapses per column figure.
    """
    export_type = 'png' # This defines the internal resolution of the plot

    cachedir = Path(find_dotenv()).parent / "cache" / "coverage_synapses"
    cachedir.mkdir(parents=True, exist_ok=True)

    plot_fn = cachedir / f"{slugify(instance)}_{export_type}.pickle"

    cfg = HexPlotConfig()

    if plot_fn.is_file():
        with plot_fn.open('rb') as plot_fh:
            fig_synapses = pickle.load(plot_fh)
    else:
        df = fetch_cells_synapses_per_col(
            cell_instance=instance
          , roi_str=["ME(R)", "LO(R)", "LOP(R)", "AME(R)", "LA(R)"]
        )
        cc = 0
        cs = 0
        if df.empty:
            warnings.warn(f'empty synapse plot for {instance}')
        else:
            cs, cc = find_cmax_across_all_neuropils(
                df, thresh_val=0.98
            )  # thresh_val can be None or float
            df = hex_from_col(df)

        plot_specs = {
            "filename": instance,
            "cmax_cells": cc,
            "cmax_syn": cs,
            "export_type": export_type,
        }

        fig_synapses = plot_per_col(
            df
          , cfg.style
          , cfg.sizing
          , plot_specs
          , plot_type="synapses"
          , trim=False
          , save_fig=False
        )

        fig_synapses.layout.title = {
            'text': instance,
            'font': {'color': 'lightgrey', 'family': 'arial'},
            'y':0.98,
            'x':0.01,
            'xanchor': 'left',
            'yanchor': 'top'
        }

        with plot_fn.open('wb') as plot_fh:
            pickle.dump(fig_synapses, plot_fh)

    return fig_synapses


def get_completeness_metrics(
    instance:str
  , trim_helper:TrimHelper=None
) -> pd.DataFrame:
    """
    Store the completeness metrics in a file

    This function was initially extracted from make_spatial_coverage_plots_for_webpages

    Parameters
    ----------
    instance : str
        name of a cell instance, e.g. 'TmY5a_R'
    trim_helper : TrimHelper
        Object caching access to the trimmed and raw data frames. If None, it's created
        internally without the benefit of caching.
    
    Returns
    -------
    quant_df : pd.DataFrame
        Quantification
    """

    cachedir = Path(find_dotenv()).parent / "cache" / "complete_metrics"
    cachedir.mkdir(parents=True, exist_ok=True)

    if trim_helper is None:
        trim_helper = TrimHelper(instance)

    metric_fn = cachedir / f"{slugify(instance)}.pickle"

    if metric_fn.is_file():
        with metric_fn.open('rb') as metric_fh:
            quant_df = pickle.load(metric_fh)
    else:
        df = fetch_cells_synapses_per_col(
            cell_instance=instance
          , roi_str=["ME(R)", "LO(R)", "LOP(R)", "AME(R)", "LA(R)"]
        )
        df = hex_from_col(df)

        named_df = fetch_ol_types_and_instances(side='both')
        pop_size = named_df[named_df['instance']==instance]['count']\
            .values\
            .astype(int)

        quant_df = cov_compl_calc(
            df
          , trim_df=trim_helper.trim_df
          , size_df=trim_helper.size_df
          , size_df_raw=trim_helper.size_df_raw
          , n_cells=pop_size
          , cell_type=instance
        )
        with metric_fn.open('wb') as metric_fh:
            pickle.dump(quant_df, metric_fh)
    return quant_df


def make_spatial_coverage_plots_for_webpages(
    instance:str
) -> tuple[go.Figure, go.Figure, pd.DataFrame]:
    """
    Generates two spatial coverage plots and one dataframe per cell type to be used when making
    the webpages.

    Parameters
    ----------
    instance : str
        cell type instance

    Returns
    -------
    fig_synapses: fig
        figure of the number of synapses per column for the cell type
    fig_cells: fig
        figure of the number of cells per column for the cell type
    quant_df : pd.DataFrame
        dataframe containing the coverage and completeness metrics
    """

    trim_helper = TrimHelper(instance)
    
    # 1 - make 'synapses per column' plot from raw data
    fig_synapses = plot_synapses_per_column(instance=instance)

    #  2 - make 'cells per column' plot from trimmed data.
    fig_cells  = plot_cells_per_column(instance=instance, trim_helper=trim_helper)

    # 3 - Calculate the coverage / completeness metrics
    quant_df = get_completeness_metrics(instance=instance, trim_helper=trim_helper)

    return fig_synapses, fig_cells, quant_df
