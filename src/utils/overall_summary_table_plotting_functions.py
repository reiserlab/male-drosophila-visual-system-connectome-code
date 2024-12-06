from pathlib import Path
from dotenv import find_dotenv

import plotly.graph_objects as go
import pandas as pd

from utils.ol_color import OL_COLOR

from utils.make_overall_summary import\
    get_celltypes_groups_df\
  , get_neuropil_df\
  , get_neuropil_groups_celltypes_df\
  , get_neuropil_groups_cells_df


def plot_group_summary_table(
    neuron_list:pd.DataFrame
  , style:dict
  , sizing:dict
) -> tuple[go.Figure, pd.DataFrame]:
    """
    get counts (of cell types, cells, and input / output synapses) for the 5 main groups
    (ONIN, ONCN, VPN, VCN, other) and makes a color-coded table. Counts are from
    `get_celltypes_groups_df`.

    Parameters
    ----------
    neuron_list : pd.DataFrame
        list of neurons. 'instance' and 'main_groups' are needed for this function to work.
    style : dict
        dict containing the values of the fixed styling formatting variables
    sizing : dict
        dict containing the values of the size formatting variables

    Returns
    -------
    fig: go.Figure
        figure of a color coded table displaying the dataframe
    celltype_groups_df : pd.DataFrame
        dataframe containing counts for the 5 cell type groups
    """

    celltype_groups_df = get_celltypes_groups_df(
        neuron_list=neuron_list
      , correct_counts=False
    )

    colors = pd.DataFrame(
        data={
            'main_group': OL_COLOR.OL_TYPES.map.keys()
          , 'color': OL_COLOR.OL_TYPES.map.values()
        }
    )
    # FIXME: This depends on ordering from OLCOLOR:
    colors['main_group'] = ['ONIN', 'ONCN', 'VPN', 'VCN', 'other']

    celltype_groups_df = celltype_groups_df\
        .reindex(["OL_intrinsic", "OL_connecting", "VPN", "VCN", "other"])\
        .rename(index={'OL_intrinsic': 'ONIN', 'OL_connecting': 'ONCN'})\
        .join(colors.set_index('main_group'))\
        .reset_index()\
        .rename(columns={'main_group': 'groups'})
    cols_to_show = ["groups", "n_celltypes", "n_cells", "n_upstream", "n_downstream"]
    text_color = celltype_groups_df["color"].to_list()

    fig = go.Figure(data=[
        go.Table(
            header={
                'values': [
                    "<b>groups<b>"
                  , "<b> #celltypes</b>"
                  , "<b> #cells</b>"
                  , "<b> #inputconn</b>"
                  , "<b> #outputconn</b>"
                ]
              , 'line_color': style['linecolor']
              , 'fill_color': style['fillcolor']
              , 'align': 'center'
              , 'font': {
                    'family': style['font_type']
                  , 'color': style['linecolor']
                  , 'size': 12
                }
            }
          , cells={
                'values': celltype_groups_df[cols_to_show].values.T
              , 'line_color': style['linecolor']
              , 'fill_color': style['fillcolor']
              , 'align': 'center'
              , 'font': {
                    'family': style['font_type']
                  , 'color': text_color
                  , 'size': 11
                }
            }
        )
    ])
    celltype_groups_df = celltype_groups_df[cols_to_show]

    fig.update_layout(width=sizing['fig_width'], height=sizing['fig_height'])

    cache_fn = Path(find_dotenv()).parent / "cache" / "summary_plots" / "celltype_groups.pickle"
    cache_fn.parent.mkdir(parents=True, exist_ok=True)
    celltype_groups_df.to_pickle(cache_fn)

    return fig, celltype_groups_df


def plot_neuropil_group_table(
    df:pd.DataFrame
  , threshold:float
  , style:dict
  , sizing:dict
) -> tuple[go.Figure, pd.DataFrame]:
    """
    get counts (of cell types, cells, and synapses) for the 5 brain regions (LA, ME, LO, LOP, AME)
    and make a color-coded table along with corrected numbers of a different threshold for LA
    and AME.

    Parameters
    ----------
    df : pd.DataFrame
        list of neurons. 'instance' and 'main_groups' are needed for this function to work.
    threshold : float
        threshold fraction of synapses that needs to be crossed within a brain region
        to be assigned to that brain region
    style : dict
        dict containing the values of the fixed styling formatting variables
    sizing : dict
        dict containing the values of the size formatting variables

    Returns
    -------
    fig : go.Figure
        figure of a color coded table displaying the dataframe
    neuropil_df : pd.DataFrame
        dataframe containing counts for the 5 main brain regions
    """

    neuropil_df = get_neuropil_df(df, threshold)

    neuropil_df = neuropil_df.reset_index()
    neuropil_df['roi'] = neuropil_df['roi']\
        .replace({
            'LA(R)': 'LA'
          , 'ME(R)': 'ME'
          , 'LO(R)': 'LO'
          , 'LOP(R)': 'LOP'
          , 'AME(R)': 'AME'
        })

    colors = pd.DataFrame(
        data={
            'roi': OL_COLOR.OL_NEUROPIL.map.keys()
          , 'color': OL_COLOR.OL_NEUROPIL.map.values()
        }
    )

    neuropil_df = pd\
        .merge(neuropil_df, colors, on='roi')\
        .reset_index()\
        .rename(columns={'roi': 'neuropil'})

    cols_to_show = ["neuropil", "n_celltypes", "n_cells", "n_upstream", "n_downstream"]
    text_color = neuropil_df["color"].to_list()

    fig = go.Figure(
        data=[
            go.Table(
                header={
                    'values': [
                        "<b>neuropil<b>"
                      , "<b> #celltypes</b>"
                      , "<b> #cells</b>"
                      , "<b> #inputconn</b>"
                      , "<b> #outputconn</b>"
                    ]
                  , 'line_color': style['linecolor']
                  , 'fill_color': style['fillcolor']
                  , 'align': 'center'
                  , 'font': {
                        'family': style['font_type']
                      , 'color': style['linecolor']
                      , 'size': 12
                    }
                }
              , cells={
                    'values': neuropil_df[cols_to_show].values.T
                  , 'line_color':style['linecolor']
                  , 'fill_color': style['fillcolor']
                  , 'align': 'center'
                  , 'font': {
                        'family': style['font_type']
                      , 'color': text_color
                      , 'size': 11
                    }
                }
            )
        ]
    )
    neuropil_df = neuropil_df[cols_to_show]

    fig.update_layout(
        width=sizing['fig_width']
      , height=sizing['fig_height']
    )

    cache_fn = Path(find_dotenv()).parent / "cache" / "summary_plots" / "neuropil_groups.pickle"
    cache_fn.parent.mkdir(parents=True, exist_ok=True)
    neuropil_df.to_pickle(cache_fn)

    return fig, neuropil_df


def plot_neuropil_group_celltype_table(
    df:pd.DataFrame
  , threshold:int
  , style:dict
  , sizing:dict
) -> tuple[go.Figure, pd.DataFrame]:
    """
    get counts of cell types for the 5 brain regions aggregated at cell type level and make a
    color-coded table along with corrected numbers of a different threshold for LA and AME

    Parameters
    ----------
    df : pd.DataFrame
        list of neurons. 'instance' and 'main_groups' are needed for this function to work.
    threshold : int
        threshold fraction of synapses that needs to be crossed within a brain region
        to be assigned to that brain region
    style : dict
        dict containing the values of the fixed styling formatting variables
    sizing : dict
        dict containing the values of the size formatting variables

    Returns
    -------
    fig : go.Figure
        figure of a color coded table displaying the dataframe
    neuropil_df : pd.DataFrame
        dataframe containing counts of cell types for the 5 main brain regions aggregated at
        cell type level
    """
    neuropil_groups_celltypes_df = get_neuropil_groups_celltypes_df(df, threshold)

    neuropil_groups_celltypes_df = neuropil_groups_celltypes_df.reset_index()
    # reordering columns
    neuropil_groups_celltypes_df = neuropil_groups_celltypes_df[
        ['roi','OL_intrinsic', 'OL_connecting', 'VPN', 'VCN', 'other']
    ]

    # renaming columns
    neuropil_groups_celltypes_df = neuropil_groups_celltypes_df\
        .rename(columns={'OL_intrinsic': 'ONIN', 'OL_connecting': 'ONCN'})

    neuropil_groups_celltypes_df['roi'] = neuropil_groups_celltypes_df['roi']\
        .replace({
            'LA(R)': 'LA'
          , 'ME(R)': 'ME'
          , 'LO(R)': 'LO'
          , 'LOP(R)': 'LOP'
          , 'AME(R)': 'AME'
        })

    colors = pd.DataFrame(
        data={
            'roi': OL_COLOR.OL_NEUROPIL.map.keys()
          , 'color': OL_COLOR.OL_NEUROPIL.map.values()
        }
    )

    neuropil_groups_celltypes_df = pd\
        .merge(neuropil_groups_celltypes_df, colors, on='roi')
    neuropil_groups_celltypes_df = neuropil_groups_celltypes_df\
        .reset_index()\
        .rename(columns={'roi': 'neuropil'})

    cols_to_show = ["neuropil", "ONIN", "ONCN", "VPN", "VCN", "other"]
    text_color = neuropil_groups_celltypes_df["color"].to_list()

    fig = go.Figure(
        data=[
            go.Table(
                header={
                    'values': [
                        "<b>neuropil<b>"
                      , "<b> ONIN</b>"
                      , "<b> ONCN</b>"
                      , "<b> VPN</b>"
                      , "<b> VCN</b>"
                      , "<b> other</b>"
                    ]
                  , 'line_color': style['linecolor']
                  , 'fill_color': style['fillcolor']
                  , 'align': 'center'
                  , 'font': {
                        'family': style['font_type']
                      , 'color': style['linecolor']
                      , 'size': 12
                    }
                }
              , cells={
                    'values': neuropil_groups_celltypes_df[cols_to_show].values.T
                  , 'line_color': style['linecolor']
                  , 'fill_color': style['fillcolor']
                  , 'align': 'center'
                  , 'font': {
                        'family': style['font_type']
                      , 'color': text_color
                      , 'size': 11
                    }
                }
            )
        ]
    )
    neuropil_groups_celltypes_df = neuropil_groups_celltypes_df[cols_to_show]

    fig.update_layout(
        width=sizing['fig_width']
      , height=sizing['fig_height']
    )

    cache_fn = Path(find_dotenv()).parent / "cache"\
        / "summary_plots" / "neuropil_groups_celltypes.pickle"
    cache_fn.parent.mkdir(parents=True, exist_ok=True)
    neuropil_groups_celltypes_df.to_pickle(cache_fn)

    return fig, neuropil_groups_celltypes_df


def plot_neuropil_group_cell_table(
    df:pd.DataFrame
  , threshold:int
  , style:dict
  , sizing:dict
) -> tuple[go.Figure, pd.DataFrame]:
    """
    get counts of cells for the 5 brain regions aggregated at cell level and make a
    color-coded table along with corrected numbers of a different threshold for LA and AME

    Parameters
    ----------
    df : pd.DataFrame
        list of neurons. 'instance' and 'main_groups' are needed for this function to work.
    threshold: int
        threshold fraction of synapses that needs to be crossed within a brain region
        to be assigned to that brain region
    style : dict
        dict containing the values of the fixed styling formatting variables
    sizing : dict
        dict containing the values of the size formatting variables

    Returns
    -------
    fig : go.Figure
        figure of a color coded table displaying the dataframe
    neuropil_df : pd.DataFrame
        dataframe containing counts of cells for the 5 brain regions aggregated at cell level
    """
    neuropil_groups_cells_df = get_neuropil_groups_cells_df(df, threshold)

    neuropil_groups_cells_df = neuropil_groups_cells_df.reset_index()

    # reordering columns
    neuropil_groups_cells_df = neuropil_groups_cells_df[
        ['roi','OL_intrinsic', 'OL_connecting', 'VPN', 'VCN', 'other']
    ]

    # renaming columns
    neuropil_groups_cells_df = neuropil_groups_cells_df\
        .rename(columns={'OL_intrinsic': 'ONIN', 'OL_connecting': 'ONCN'})

    neuropil_groups_cells_df['roi'] = neuropil_groups_cells_df['roi']\
        .replace({
            'LA(R)': 'LA'
          , 'ME(R)': 'ME'
          , 'LO(R)': 'LO'
          , 'LOP(R)': 'LOP'
          , 'AME(R)': 'AME'
        })

    colors = pd.DataFrame(
        data={
            'roi': OL_COLOR.OL_NEUROPIL.map.keys()
          , 'color': OL_COLOR.OL_NEUROPIL.map.values()
        }
    )

    neuropil_groups_cells_df = pd.merge(neuropil_groups_cells_df,colors,on='roi')
    neuropil_groups_cells_df = neuropil_groups_cells_df\
        .reset_index()\
        .rename(columns={'roi': 'neuropil'})

    cols_to_show = ["neuropil", "ONIN", "ONCN", "VPN", "VCN", "other"]
    text_color = neuropil_groups_cells_df["color"].to_list()

    fig = go.Figure(
        data=[
            go.Table(
                header={
                    'values': [
                        "<b>neuropil<b>"
                      , "<b> ONIN</b>"
                      , "<b> ONCN</b>"
                      , "<b> VPN</b>"
                      , "<b> VCN</b>"
                      , "<b> other</b>"
                    ]
                  , 'line_color': style['linecolor']
                  , 'fill_color': style['fillcolor']
                  , 'align': 'center'
                  , 'font': {
                        'family': style['font_type']
                      , 'color': style['linecolor']
                      , 'size': 12
                    }
                }
              , cells={
                    'values': neuropil_groups_cells_df[cols_to_show].values.T
                  , 'line_color': style['linecolor']
                  , 'fill_color': style['fillcolor']
                  , 'align': 'center'
                  , 'font': {
                        'family': style['font_type']
                      , 'color': text_color
                      , 'size': 11
                    }
                }
            )
        ]
    )
    neuropil_groups_cells_df = neuropil_groups_cells_df[cols_to_show]

    fig.update_layout(
        width=sizing['fig_width']
      , height=sizing['fig_height']
    )

    cache_fn = Path(find_dotenv()).parent / "cache"\
        / "summary_plots" / "neuropil_groups_cells.pickle"
    cache_fn.parent.mkdir(parents=True, exist_ok=True)
    neuropil_groups_cells_df.to_pickle(cache_fn)

    return fig, neuropil_groups_cells_df
