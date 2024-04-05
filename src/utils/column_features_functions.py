"""
Functions used in the notebook `src/column_features/column_features_analysis.ipynb`
"""

from pathlib import Path
import os
from dotenv import find_dotenv

import pandas as pd
import numpy as np
import kneed
from scipy.spatial import ConvexHull

from neuprint import fetch_custom

from queries.coverage_queries import fetch_syn_per_col_for_instance

from utils.neuron_bag import NeuronBag
from utils.column_features_helper_functions import find_neuropil_hex_coords


def hex_from_col(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coverts string entry in df of the hex column coordinates into separate columns
    in the df of 'hex1_id' and 'hex2_id'.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe. Must have column named 'column'

    Returns
    -------
    df : pd.DataFrame
        input df with additional `hex1_id` and `hex2_id` columns

    """
    assert (
        "column" in df.columns
    ), "DataFrame must contain 'column' as a '<HEX1>_<HEX2>' string"
    df = df.assign(
        hex1_id=lambda x: [int(i[0]) for i in x["column"].str.split("_")],
        hex2_id=lambda x: [int(i[1]) for i in x["column"].str.split("_")],
    )
    return df


def find_cmax_across_all_neuropils(
    df:pd.DataFrame
  , thresh_val:float=0.98
) -> [int, int]:
    """
    For a particular cell type, find the maximum number of cells and synapses
    per column in ME(R), LO(R), and LOP(R) and output the maximum of these values
    to be used to set 'cmax' when plotting the spatial coverage heatmaps.

    Parameters
    ----------
    df : pd.DataFrame
        query output. Contains the columns 'column', 'roi', 'cells_per_column', 
        'synpases_per_column' and 'cell_body_ids'
    thresh_val : int
         default = .98 - value of the 98th quantile

    Returns
    -------
    cs : int
        Maximum values of '# of synapses per column' across all three neuropils for
        this cell type.
    cc : int
        Maximum values of '# of cells per column' across all three neuropils for
        this cell type.
    """
    cc = 0
    cs = 0

    if "n_syn" in df.columns:
        cs = df["n_syn"].max()
    if "n_cells" in df.columns:
        cc = df["n_cells"].max()

    if isinstance(thresh_val, float):
        if "n_cells" in df.columns:
            cc = df["n_cells"].quantile(thresh_val)
        if "n_hex_from_colsyn" in df.columns:
            cs = df["n_syn"].quantile(thresh_val)

    return int(cs), int(cc)


def get_trim_df(
    cell_instance:str
  , roi_str:str
  , syn_type:str='all'
  , cumsum_min:float=0.775
  , cumsum_fix:float=0.999
) -> pd.DataFrame:
    """
    Determine the number of cells from a cell type that innervate each column after trimming off
    the "outlier" synapses.
    For each neuron, find a lower threshold on the number synapses in a column in order to retain
    those synapses. This threshold equals the number of synapses in the `rank_thre`'th largest
    column, where `rank_thre` is computed from cell-type information. Namely, we use an elbow
    method on the median synapse count per column vs. rank of column (the column with the
    largest number of synapses has rank 1, the column with the second largest number of
    synapses has rank 2 etc.).

    Parameters
    ----------
    cell_instance : str
        instance name
    roi : str
        neuprint ROI, can only be ME(R), LO(R), LOP(R)
    syn_type : str, default='all'
        synapse type to use. Possible options are 'pre', 'post' or 'all'
    cumsum_min : float, default=0.775
        minimum fraction of cumulative sum of synapses in columns after trimming
        if knee finder gives a lower fraction then we find the rank with fraction cumsum_fix.
        the value 0.775 was obtained as a dip in the bimodal distribution of the cumsum fraction
        from the knee finder for cell-types in ME with at least 1000 synapses.
    cumsum_fix : float, default=0.999
        a fixed fraction of cumulative sum of synapses in columns that is used if
        knee finder gives a lower fraction than cumsum_min.
        the value 0.999 is such that almost no synapses get trimmed off for large cells except
        for those in columns with very few synapses.

    Returns
    -------
    trim_df : pd.DataFrame
        One row per single column of a single bodyId. Dataframe is sorted by 'synapse_perc' and
        'bodyId'
        column : str
            column name in the form '39_20'
        roi : str
            neuropil region
        bodyId : int
            neuron's bodyId
        synapse_count : int
            number of synapses assigned to that bodyId in that column
        synapse_perc : float
            fraction of that bodyId's synapses within the column
        cum_sum : float
            cumulative sum of 'synapse_perc' per bodyId
        rank : float
            Rank of each column starting with the column with the highest fraction of bodyId's
            total synapses

    size_df : pd.DataFrame
        One row per bodyID
        bodyId : int
            bodyId of neuron
        n_cols : int
            number of columns innervated per neuron of the chosen cell type
        roi : str
            neuropil region of interest

    df2 : pd.DataFrame
        One row per column
        column : str
            column name in the form '39_20'
        n_cells : int
            number of cells that innervate the column after trimming
        n_syn : int
            number of synapses within the column after trimming
        hex1_id : int
            hex1_id value of the column
        hex2_id : int
            hex2_id value of the column
        roi : str
            neuropil region
        issue : bool
            1 if unable to form a knee during the trimming process.

    n_cells : int
        number of cells of that cell type

    """
    # fetch bodyIds of neurons from cell type
    neuronbag = NeuronBag(cell_instance=cell_instance, side=None)
    n_cells = neuronbag.size
    trim_df = pd.DataFrame()
    issue = 0

    trim_df = fetch_syn_per_col_for_instance(
        neuron_instance=cell_instance
      , roi_str=roi_str
      , syn_type=syn_type
    )

    if not trim_df.empty:

        trim_df['cum_sum'] = trim_df\
            .groupby('bodyId')['synapse_perc']\
            .cumsum()
        trim_df['rank'] = trim_df\
            .groupby('bodyId')['synapse_count']\
            .rank(ascending=False, method="first")
        raw_df = trim_df.copy()
        # for whole df of all cells.
        cumsum_per_rank_df = (
            trim_df.groupby(["bodyId", "rank"])["cum_sum"].first().unstack(-1, 1)
        )
        n_pts = int(trim_df["rank"].max())
        x_val = np.zeros(n_pts + 1)
        y_val = np.zeros(n_pts + 1)
        x_val[1:] = np.linspace(1, n_pts, n_pts)
        y_val[1:] = cumsum_per_rank_df.median(0)
        kneedle = kneed.KneeLocator(
            x_val, y_val, S=1.0, curve="concave", direction="increasing"
        )
        rank_thre = kneedle.knee
        if (rank_thre is None) or (y_val[int(rank_thre)] < cumsum_min):
            issue = 1
            # print(f"{cell_type} in {roi}")
            find_rank = np.where(y_val >= cumsum_fix)[0]
            if find_rank.shape[0] == 0:
                rank_thre = y_val.size - 1
            else:
                rank_thre = find_rank[0]
        # only keep columns that have over the threshold % of synapses in them.
        count_thre_df = trim_df[trim_df['rank'] <= rank_thre]\
            .groupby('bodyId')['synapse_count']\
            .last()\
            .to_frame()\
            .rename(columns={'synapse_count': 'count_thre'})\
            .reset_index()
        
        trim_df = trim_df.merge(count_thre_df, on='bodyId')
        trim_df = trim_df[trim_df['synapse_count'] >= trim_df['count_thre']]
        # coverage factor - n_cells per col
        df2 = (
            trim_df.groupby("column")["bodyId"]
            .nunique()
            .to_frame()
            .reset_index()
            .rename(columns={"bodyId": "n_cells"})
        )
        # n_syn per col
        df2["n_syn"] = (
            trim_df.groupby("column")["synapse_count"]
            .sum()
            .to_frame()
            .reset_index()["synapse_count"]
        )
        # used for plotting with the trimmed data
        df2 = hex_from_col(df2)
        df2["roi"] = roi_str
        # issue = 1 means that the cell did not fulfil the thresh criteria for the knee
        df2["issue"] = issue
        # size - n_cols per cell.
        size_df = trim_df.groupby(["roi", "bodyId"])\
            .nunique()\
            .reset_index()\
            .loc[:, ["bodyId", "column", "roi"]]\
            .rename(columns={"column": "n_cols"})
        # size - n_cols per cell - raw data - untrimmed
        size_df_raw = raw_df.groupby(["roi", "bodyId"])\
            .nunique()\
            .reset_index()\
            .loc[:, ["bodyId", "column", "roi"]]\
            .rename(columns={"column": "n_cols"})
    else:
        trim_df = pd.DataFrame()
        size_df = pd.DataFrame(data={'roi': [roi_str]})
        size_df_raw = pd.DataFrame(data={'roi': [roi_str]})
        df2 = pd.DataFrame(data={'roi': [roi_str]})
        n_cells = 0

    return trim_df, size_df, size_df_raw, df2, n_cells


def get_trim_df_all_rois(
    cell_instance:str
  , syn_type:str='all'
) -> pd.DataFrame:
    """
    Combines pd.DataFrames from all three neuropil regions after trimming.

    Parameters
    ----------
    cell_type : str
        name of cell type
    syn_type : str, default='all'
        type of synapses. Can be 'post', 'pre', and 'all'

    Returns
    -------
    df : pd.DataFrame
        One row per column
        column : str
            column name in the form '39_20'
        n_cells : int
            number of cells that innervate the column after trimming
        n_syn : int
            number of synapses within the column after trimming
        hex1_id : int
            hex1_id value of the column
        hex2_id : int
            hex2_id value of the column
        roi : str
            neuropil region
        issue : bool
            1 if unable to form a knee during the trimming process.

    size_df : pd.DataFrame
        One row per bodyID
        bodyId : int
            bodyId of neuron
        n_cols : int
            number of columns innervated per neuron of the chosen cell type
        roi : str
            neuropil region of interest

    n_cells : int
        number of cells of that cell type
    """

    _, size_me, size_me_raw, df_me, _ = get_trim_df(
        cell_instance=cell_instance
      , roi_str="ME(R)"
      , syn_type=syn_type
    )
    if not isinstance(df_me, pd.DataFrame):
        df_me = pd.DataFrame()

    _, size_lo, size_lo_raw, df_lo, _ = get_trim_df(
        cell_instance=cell_instance
      , roi_str="LO(R)"
      , syn_type=syn_type
    )
    if not isinstance(df_lo, pd.DataFrame):
        df_lo = pd.DataFrame()

    _, size_lop, size_lop_raw, df_lop, _ = get_trim_df(
        cell_instance=cell_instance
      , roi_str="LOP(R)"
      , syn_type=syn_type
    )
    if not isinstance(df_lop, pd.DataFrame):
        df_lop = pd.DataFrame()
    df = pd.concat([df_me, df_lo, df_lop])
    size_df = pd.concat([size_me, size_lo, size_lop])
    size_df_raw = pd.concat([size_me_raw, size_lo_raw, size_lop_raw])
    return df, size_df, size_df_raw


def cov_compl_calc(
    df:pd.DataFrame
  , trim_df:pd.DataFrame
  , size_df:pd.DataFrame
  , size_df_raw: pd.DataFrame
  , n_cells:int
  , cell_type:str
) -> pd.DataFrame:
    """
    Generates a dataframe with each column containing the value of a different coverage or 
    completeness quantification.

    Parameters
    ----------
    df : pd.DataFrame
        One row per column - raw data
        column : str
            column name in the form '39_20'
        roi : str
            neuropil region
        n_cells : int
            number of cells that innervate the column
        n_syn : int
            number of synapses within the column
        cell_body_ids : list
            list of bodyIds of the cells innervating the column
    trim_df : pd.DataFrame
        One row per column - trimmed data
        column : str
            column name in the form '39_20'
        n_cells : int
            number of cells that innervate the column after trimming
        n_syn : int
            number of synapses within the column after trimming
        hex1_id : int
            hex1_id value of the column
        hex2_id : int
            hex2_id value of the column
        roi : str
            neuropil region
        issue : bool
            1 if unable to form a knee during the trimming process.
    size_df : pd.DataFrame
        One row per bodyID - trimmed data
        bodyId : int
            bodyId of neuron
        n_cols : int
            number of columns innervated per neuron of the chosen cell type
        roi : str
            neuropil region of interest
    size_df_raw : pd.DataFrame
        One row per bodyID - raw data
        bodyId : int
            bodyId of neuron
        n_cols : int
            number of columns innervated per neuron of the chosen cell type
        roi : str
            neuropil region of interest
    n_cells : int
        number of cells of the instance
    cell_type : str
        cell type of choice

    Returns
    -------
    quant_df_all : pd.DataFrame
        One dataframe per cell type for all three optic lobe regions.
    
        cell_type : str
            cell type (instance)
        roi : str
            neuropil region
        cols_covered_pop : int
            number of columns covered by all cells of cell type - raw data
        col_completeness : float
            proportion of all columns in neuropil that are innervated by cells from the cell type
        coverage_factor : float
            mean value of the number of cells per column across all columns occupied - raw data
        synaptic_coverage_factor : float
            median number of synapses per column across all columns - raw data
        coverage_factor_trim
            mean value of the number of cells per column across all columns occupied - trimmed data
        synaptic_coverage_factor_trim
            mean value of the number of synapses per column across all columns occupied - trimmed
            data
        n_syn_total
            number of all synapses from all cells of this cell type in this roi
        n_syn_trim
            number of all synapses from all cells of this cell type in this roi - trimmed data
        population_size
            number of cells in the cell type
        cell_size_cols
            median number of columns spanned per cell of cell type - trimmed data
        area_covered_pop
            area covered by convex hull around all columns innervated by all cells of cell types,
              using the hex coordinates of the columns - raw data
        area_completeness
            the area covered by all cells as a proportion of the total roi area - raw data
    """
    quant_df_all = pd.DataFrame()

    for roi_str in ["ME(R)", "LO(R)", "LOP(R)"]:

        cql = f"""
            MATCH (n:Neuron)-[e:ConnectsTo]->(m:Neuron)
            WHERE n.instance='{cell_type}'
            WITH DISTINCT e
            WITH apoc.convert.fromJsonMap(e.roiInfo) as ri
            RETURN sum(ri['{roi_str}'].post) as num_conn
        """
        o_con = fetch_custom(cql)
        [num_output_conn] = o_con.iloc[0].to_list()

        cql = f"""
            MATCH (n:Neuron)-[e:ConnectsTo]->(m:Neuron)
            WHERE n.instance='{cell_type}'
            with distinct n
            WITH apoc.convert.fromJsonMap(n.roiInfo) as nri
            RETURN 
                sum(nri['{roi_str}'].pre) as num_pre
              , sum(nri['{roi_str}'].post) as num_post
        """
        o_syn = fetch_custom(cql)

        [n_syn_pre, n_syn_post] = o_syn.iloc[0].to_list()

        # Get only the data from the region of interest
        roi_df = df[df["roi"] == roi_str]
        sz_df = size_df[size_df["roi"] == roi_str]
        sz_df_raw = size_df_raw[size_df_raw["roi"] == roi_str]
        roi_trim_df = trim_df[trim_df["roi"] == roi_str]

        # all columns in region - i.e ME(R)
        col_hex_ids, n_cols_region = find_neuropil_hex_coords(roi_str)
        edge_cells = get_edge_cell_types(roi_str=roi_str)

        data = {}
        quant_df = pd.DataFrame()

        if not roi_df.empty:
            cell_string = cell_type.replace("/", "-")
            data["cell_type"] = cell_string
            data["roi"] = roi_str
            data["cols_covered_pop"] = roi_df["column"].nunique()
            data["col_completeness"] = data["cols_covered_pop"] / n_cols_region
            # coverage_factor
            data["coverage_factor"] = roi_df["n_cells"].mean()
            data["synaptic_coverage_factor"] = roi_df["n_syn"].median()  # median for raw data
            data["n_pre"] = n_syn_pre
            data["n_post"] = n_syn_post
            data["n_output_conn"] = num_output_conn
            data["n_syn_total"] = roi_df["n_syn"].sum()

            if roi_trim_df.empty:
                data["coverage_factor_trim"] = np.nan
                data["synaptic_coverage_factor_trim"] = np.nan
                data["n_syn_trim"] = np.nan
                data["population_size"] = np.nan
                data["cell_size_cols"] = np.nan
                data["cell_size_cols_raw"] = np.nan
                data["area_covered_pop"] = np.nan
                data["area_completeness"] = np.nan
            else:
                data["coverage_factor_trim"] = roi_trim_df["n_cells"].mean()
                data["synaptic_coverage_factor_trim"] = roi_trim_df["n_syn"].mean()

                # total number of synapses
                data["n_syn_trim"] = roi_trim_df["n_syn"].sum()
                data["population_size"] = n_cells
                # need to find the average size of each neuron in a cell type in columns.
                data["cell_size_cols"] = sz_df["n_cols"].median()
                data["cell_size_cols_raw"] = sz_df_raw["n_cols"].median()

                if data["cols_covered_pop"] > 3:
                    # function to find maximum area - convex hull of neuropil
                    max_area_neuropil = calc_convex_hull_col_area(col_hex_ids)
                    # completeness_factor_area
                    area_cols = calc_convex_hull_col_area(roi_df)
                    # if the cell type is densely around the edge
                    # don't find area but use columns occupied.
                    if cell_type in edge_cells:
                        area_cols = data["cols_covered_pop"]
                    data["area_covered_pop"] = area_cols

                    if area_cols != np.nan:
                        # convex hull area of occ cols/ total area of neuropil
                        data["area_completeness"] = (
                            data["area_covered_pop"] / max_area_neuropil
                        )
                    else:
                        data["area_completeness"] = np.nan
                else:
                    data["area_covered_pop"] = np.nan
                    data["area_completeness"] = np.nan

            quant_df = pd.DataFrame(data, index=[0])

        quant_df_all = pd.concat([quant_df_all, quant_df])

    return quant_df_all


def solve(coordinates):
    """
    Check if points lie on a straight line.

    Parameters
    ----------
    coordinates : np.array
        coordinates in the style [(5, 5),(8, 8),(9, 9)].
    
    Returns
    -------
    solve : bool
        True if points are on a straight line. False if not.
    """
    (x0, y0), (x1, y1) = coordinates[0], coordinates[1]
    for i in range(2, len(coordinates)):
        x, y = coordinates[i]
        if (x0 - x1) * (y1 - y) != (x1 - x) * (y0 - y1):
            return False
    return True


def calc_convex_hull_col_area(roi_df: pd.DataFrame):
    """
    Calculates the area of the 2D convex hull of the columns covered in the roi.
    Uses the hex coordinates of the columns covered by synapses of all cells of the cell type.

    Parameters
    ----------
    roi_df : pd.DataFrame
        DataFrame with 'hex1_id', 'hex2_id' columns

    Returns
    -------
    col_area: float
        Surface area of the convex hull
    """
    df = roi_df[["hex1_id", "hex2_id"]].drop_duplicates()
    if df.shape[0] > 3:
        coords = df.to_numpy()
        straight_line = solve(coords)
        if straight_line:
            col_area = np.nan
        else:
            hull = ConvexHull(coords)
            col_area = hull.volume  # see docs - when shape is 2D volume calcs area
    else:
        col_area = np.nan
    return col_area


def make_metrics_df() -> pd.DataFrame:
    """
    Check if metrics_df file exists with combined metric information 
    from all neuron instances in ME(R), LO(R) and LOP(R)
    
    Returns
    -------
    metrics_df : pd.DataFrame 
        One dataframe for all cell types. 
        Each row contains coverage metrics values for cells of one instance type 
        in one of the optic lobe regions.
        cell_type : str
            cell type (instance)
        roi : str
            neuropil region
        cols_covered_pop : int
            number of columns covered by all cells of cell type - raw data
        col_completeness : float
            proportion of all columns in neuropil that are innervated by cells from the cell type
        coverage_factor : float
            mean value of the number of cells per column across all columns occupied - raw data
        synaptic_coverage_factor : float
            median number of synapses per column across all columns - raw data
        coverage_factor_trim
            mean value of the number of cells per column across all columns occupied - trimmed data
        synaptic_coverage_factor_trim
            mean value of the number of synapses per column across all columns occupied - trimmed
            data
        n_syn_total
            number of all synapses from all cells of this cell type in this roi
        n_syn_trim
            number of all synapses from all cells of this cell type in this roi - trimmed data
        population_size
            number of cells in the cell type
        cell_size_cols
            median number of columns spanned per cell of cell type - trimmed data
        area_covered_pop
            area covered by convex hull around all columns innervated by all cells of cell types,
              using the hex coordinates of the columns - raw data
        area_completeness
            the area covered by all cells as a proportion of the total roi area - raw data
    """
    cachedir = Path(find_dotenv()).parent / "cache" / "complete_metrics"
    metric_file = cachedir / "complete_metrics.pickle"

    if metric_file.is_file():
        with metric_file.open('rb') as metric_fh:
            metrics_df = pd.read_pickle(metric_fh)
    else:
        data_frames = []
        metrics_df = pd.DataFrame
        for filename in os.listdir(cachedir):
            if filename.endswith(".pickle"):
                file_path = os.path.join(cachedir, filename)
                df = pd.read_pickle(file_path)
                data_frames.append(df)
            metrics_df = pd.concat(data_frames, axis=0, ignore_index=True)
        with metric_file.open('wb') as metric_fh:
            metrics_df.to_pickle(metric_fh)
    return metrics_df


def get_edge_cell_types(roi_str: str):
    """
    Get list of manually curated cell types that are predominantly at the edge of the region.
    For these cells, do not find the area covered using the convex hull mechanism.
    Instead, use the number of innervated columns, before trimming, as the columnar 'area' 
    covered by the cell type.

    Parameters
    ----------
    roi_str : str
        optic lobe region

    Returns
    -------
    edge_cells : list 
        list of instances for which finding the convex hull does not give a good approximation
         of the area covered. For these types the number of columns occupied, before trimming,
         is used as a proxy for the area covered.
    """
    if roi_str == "ME(R)":
        edge_cells = [
            "aMe2_R",
            "aMe10_R",
            "Ascending_TBD_1_R",
            "Cm-DRA_R",
            "Dm-DRA1_R",
            "Dm-DRA2_R",
            "MeLo11_R",
            "MeTu4f_R",
            "MeVP15_R",
            "MeVP20_R",
            "MeVPMe8_R",
            "Mi16_R",
            "R7d_R",
            "R8d_R",
            "TmY19b_R",
            "LC14b_L",
        ]
    elif roi_str == "LO(R)":
        edge_cells = [
            "aMeVP_TBD_1_R",
            "LC14a-1_L",
            "LC14a-1_R",
            "LC14a-2_L",
            "LC14a-2_R",
            "LC14b_L",
            "LC14b_R",
            "LC31a_R",
            "Li37_R",
            "LoVC17_R",
            "LoVP12_R",
            "LoVP26_R",
            "LoVP49_R",
            "LoVP92_R",
            "LPT31_R",
            "LT80_R",
            "LT81_R",
            "TmY19b_R",
        ]
    elif roi_str == "LOP(R)":
        edge_cells = ["LPT31_R", "LPT100_R", "LC14b_L",]
    return edge_cells
