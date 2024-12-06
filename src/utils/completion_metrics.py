from time import sleep
from random import random

import concurrent.futures
from pathlib import Path
from textwrap import dedent

from dotenv import find_dotenv
import pandas as pd
from neuprint import fetch_all_rois, fetch_custom
from utils import olc_client

from utils.column_features_helper_functions import find_neuropil_hex_coords


def get_upstream_downstream_connections(roi_str:list=None) -> pd.DataFrame:
    """
    This function calculates the total number of upstream and downstream connections made by all
    the segments (~10M) within the 5 optic lobe regions

    Parameters
    ---------
    roi_str: list
        The list of rois that correspond to the 5 primary optic lobe regions default
        roi_str = ['LA(R)', 'ME(R)', 'LO(R)', 'LOP(R)', 'AME(R)']

    Returns
    -------
    result : pd.DataFrame
        Dataframe with the total number of upstream and downstream connections within optic lobe
        regions
    """
    if isinstance(roi_str, str):
        roi_str = [roi_str]
    all_roi = ["LA(R)", "ME(R)", "LO(R)", "LOP(R)", "AME(R)"]
    if roi_str is None:
        roi_str = all_roi

    assert set(roi_str) <= set(all_roi), f"ROIs must be a subset of {all_roi}"

    result = pd.DataFrame()
    for roi in roi_str:
        cql = dedent(f"""
            MATCH (n:Segment)
            WITH n, '{roi}' as roi, apoc.convert.fromJsonMap(n.roiInfo) as roiInfo
            RETURN
                count(n) as count,
                sum(roiInfo[roi]['upstream']) as olr_upstream,
                sum(roiInfo[roi]['downstream']) as olr_downstream
        """)
        r0 = fetch_custom(cql)
        result = result.add(r0, fill_value=0)

    print(f"Total number of upstream connections is {int(result['olr_upstream'][0])}")
    print(
        f"Total number of downstream connections is {int(result['olr_downstream'][0])}"
    )

    return result


def get_completion_metrics(roi_str:list=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function calculates the completion metrics for pre and post synapses. For both cases,
    the completion percentage is defined as the ratio of number of pre(post) of defined neurons
    to the number of pre(post) of segments

    Parameters
    ----------
    roi_str : list
        The list of rois that correspond to the 5 primary optic lobe regions
        default roi_str = ['LA(R)', 'ME(R)', 'LO(R)', 'LOP(R)', 'AME(R)']

    Returns
    -------
    result_neurons : pd.DataFrame
        completion metrics for pre and post synapses for Neurons
    result_segments : pd.DataFrame
        completion metrics for pre and post synapses for Segments
    """
    if isinstance(roi_str, str):
        roi_str = [roi_str]
    all_roi = ["LA(R)", "ME(R)", "LO(R)", "LOP(R)", "AME(R)"]
    if roi_str is None:
        roi_str = all_roi

    assert set(roi_str) <= set(all_roi), f"ROIs must be a subset of {all_roi}"

    result_neurons = pd.DataFrame()
    result_segments = pd.DataFrame()

    for roi in roi_str:
        cql = dedent(f"""
            MATCH (n:Neuron)
            WITH n, '{roi}' as roi, apoc.convert.fromJsonMap(n.roiInfo) as roiInfo
            RETURN
                count(n) as count,
                sum(roiInfo[roi]['pre']) as olr_pre,
                sum(roiInfo[roi]['post']) as olr_post
        """)
        r0n = fetch_custom(cql)
        result_neurons = result_neurons.add(r0n, fill_value=0)
        cql = dedent(f"""
            MATCH (n:Segment)
            WITH n, '{roi}' as roi, apoc.convert.fromJsonMap(n.roiInfo) as roiInfo
            RETURN
                count(n) as count,
                sum(roiInfo[roi]['pre']) as olr_pre,
                sum(roiInfo[roi]['post']) as olr_post
        """)
        r0s = fetch_custom(cql)
        result_segments = result_segments.add(r0s, fill_value=0)

        pre_syn_completion = (
            result_neurons["olr_pre"][0] / result_segments["olr_pre"][0]
        ) * 100
        print(f"The completion percentage for pre synapses is {pre_syn_completion}")

        post_syn_completion = (
            result_neurons["olr_post"][0] / result_segments["olr_post"][0]
        ) * 100
        print(f"The completion percentage for post synapses is {post_syn_completion}")

        return result_neurons, result_segments


def fetch_cxns_per_col(
    col_hex_ids:pd.DataFrame
  , seg_or_neu:str
  , verbose:bool=False) -> pd.DataFrame:
    """
    Given a data frame `col_hex_ids` with a column `column`, that contains the ROIs for every
    column within a given optic lobe neuropil (e.g. `LOP_R_col_01_07`), fetch all of the
    pre/post synapses and up/downstream connections within that column. If `seg_or_neu` is
    `segments` then only find the connections / synapses from unassigned segments. Otehrwise,
    if `seg_or_neu` is `neurons` then only find the connections / synapses from segments that
    have been assigned a neuron type.

    This function fetches the data about the number of synapses / connections from the database,
    adds these values as columns to the appropriate row of a copied version of `col_hex_ids`
    and returns it.

    Parameters
    ----------
    col_hex_ids: pd.DataFrame
        Data frame that contains the columns 'hex1_id', 'hex2_id' and 'column'.
        The 'column' col contains strings in the ROI form required by the cypher query.
        For example, 'ME_R_col_18_28'.
    seg_or_neu : str
        String to choose between finding connections for unassigned segments
          ("segments") or assigned neurons ("neurons")

    Returns
    -------
    col_hex_ids: pd.dataframe
        Returns the same data frame, col_hex_ids, but with additional columns 'n_pre', 'n_post',
        'n_up', 'n_down'.
    """
    assert seg_or_neu in ["segments", "neurons"]\
      , f"seg_or_neu must be 'segments' or 'neurons', not '{seg_or_neu}'"

    if seg_or_neu == "segments":
        call_fun = fetch_col_for_segment
        if verbose:
            print('segments')
    else:
        call_fun = fetch_col_for_neuron
        if verbose:
            print('neurons')

    mydf = pd.DataFrame()
    with concurrent.futures.ProcessPoolExecutor(max_workers=48) as executor:
        for r0 in executor.map(call_fun, col_hex_ids['column'].to_list()):
            mydf = pd.concat([mydf, r0])

    col_hex_ids_copy = pd.merge(col_hex_ids, mydf, how='left', on='column')

    return col_hex_ids_copy


def fetch_col_for_segment(column:str) -> pd.DataFrame:
    """
    Alias to `fetch_info_for_column` for Segments.
    """
    return fetch_info_for_column(column, seg_or_neu='Segment')


def fetch_col_for_neuron(column:str) -> pd.DataFrame:
    """
    Alias to `fetch_info_for_column` for Neurons.
    """
    return fetch_info_for_column(column, seg_or_neu='Neuron')


def fetch_info_for_column(
    column:str
  , seg_or_neu:str
  , verbose:bool=False) -> pd.DataFrame:
    """
    Fetch the connection information for the column (pre, post, upstream, and downstream).
    Depending on `seg_or_neu`.

    Paramters
    ---------
    column : str
        name of the column, e.g. 'ME_R_col_18_18'
    seg_or_neu: str, {'Segment', 'Neuron'}

    Returns
    -------
    df : pd.DataFrame
        Dataframe with the connections from that column.
        column : str
            name of the column
        pre : int
            number of pre synaptic connections
        post : int
            number of post synatic connections
        upstream : int
            number of upstream connections
        downstream : int
            number of downstream connections
    """
    olc_client.connect(verbose=False)
    assert seg_or_neu in ['Segment', 'Neuron'], "Must be Segment or Neuron"
    fname = Path(find_dotenv()).parent / 'cache' / 'completeness'\
        / f'cxn_df_{column}_{seg_or_neu}.pickle'
    if fname.is_file():
        return pd.read_pickle(fname)
    fname.parent.mkdir(parents=True, exist_ok=True)
    type_str = ""
    if seg_or_neu == "Segment":
        type_str = "and m.type is null"
    cql = dedent(f"""\
        MATCH (m:{seg_or_neu})
        WHERE m.{column} {type_str}
        WITH m, '{column}' as roi, apoc.convert.fromJsonMap(m.roiInfo) as roiInfo
        RETURN
            '{column}' as column,
            sum(roiInfo[roi]['pre']) as n_pre,
            sum(roiInfo[roi]['post']) as n_post,
            sum(roiInfo[roi]['upstream']) as n_up,
            sum(roiInfo[roi]['downstream']) as n_down
    """)
    for attempt in range(10):
        sleep(random())
        if verbose:
            print(column)
        try:
            col_info = fetch_custom(cql)
            col_info.to_pickle(fname)
            return col_info
        except Exception as e:
            if attempt < (5 - 1):
                print(f"Retry {attempt}: {cql}")
            else:
                print(f"Retry {attempt} failed")
                raise e
    return None


def fetch_cxn_df(roi_str:str) -> pd.DataFrame:
    """
    Generate dataframe containing information about the number of pre/post
    synapses and up/downstream connections from all neurons or all non-neuronal
    segments within a optic lobe neuropil column. This data is used to determine
    the completeness of the connectome within the different neuropils.

    Parameters
    ----------
    roi_str : str
        Optic lobe region of interest for which to generate plots.
    """
    assert isinstance(roi_str, str) and roi_str in ["ME(R)", "LO(R)", "LOP(R)"]\
      , f"ROI must be one of 'ME(R)', 'LO(R)', 'LOP(R)', but is actually '{roi_str}'"

    # Check if the cache file exists, and if not generate it.
    fname = Path(find_dotenv()).parent / 'cache' / 'completeness' / f'cxn_df_{roi_str[:-3]}.pickle'

    if not fname.is_file():
        print(f"'{fname}' does not exist."
            " Generating this now.")

        # Extract all roi names from the database
        all_rois = fetch_all_rois()

        # Look for all column rois in the neuropil
        col_str = f"{roi_str[:-3]}_R_col"

        # col_hex_ids is a dataframe with the columns 'hex1_id' and 'hex2_id' needed for plotting.
        col_hex_ids, _ = find_neuropil_hex_coords(roi_str)
        # Add a column with the relevant column roi string
        col_hex_ids["column"] = pd.Series(
            [i for i in all_rois if col_str in i], name="column"
        )
        col_hex_ids_neu = col_hex_ids.copy()
        # Extract data from all non-neuronal segments
        seg_cxn_df = fetch_cxns_per_col(col_hex_ids, seg_or_neu="segments")
        # Extract data from all identified neurons
        neu_cxn_df = fetch_cxns_per_col(col_hex_ids_neu, seg_or_neu="neurons")

        # Merge these two dataframes together
        cxn_df = pd.merge(
            neu_cxn_df,
            seg_cxn_df,
            on=["hex1_id", "hex2_id", "column"],
            suffixes=("_neu", "_seg"),
        )

        # Calculate the ratio of 'captured' synapses and connections against all
        cxn_df["n_pre_ratio"] = cxn_df["n_pre_neu"] / (
            cxn_df["n_pre_seg"] + cxn_df["n_pre_neu"]
        )
        cxn_df["n_post_ratio"] = cxn_df["n_post_neu"] / (
            cxn_df["n_post_seg"] + cxn_df["n_post_neu"]
        )
        cxn_df["n_up_ratio"] = cxn_df["n_up_neu"] / (
            cxn_df["n_up_seg"] + cxn_df["n_up_neu"]
        )
        cxn_df["n_down_ratio"] = cxn_df["n_down_neu"] / (
            cxn_df["n_down_seg"] + cxn_df["n_down_neu"]
        )
        cxn_df["roi"] = roi_str
        # save the cxn_df for each roi separately
        fname.parent.mkdir(parents=True, exist_ok=True)
        cxn_df.to_pickle(fname)
    else:
        cxn_df = pd.read_pickle(fname)

    return cxn_df
