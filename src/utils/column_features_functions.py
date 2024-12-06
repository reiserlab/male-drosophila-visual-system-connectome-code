"""
Functions used in the notebook `src/column_features/column_features_analysis.ipynb`
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
from neuprint import fetch_custom
from dotenv import find_dotenv

from utils.column_features_helper_functions import find_neuropil_hex_coords


def hex_from_col(df:pd.DataFrame) -> pd.DataFrame:
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
    assert ("column" in df.columns)\
        , "DataFrame must contain 'column' as a '<HEX1>_<HEX2>' string"
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
        Maximum values of 'n_syn' to use across all three rois.
    cc : int
        Maximum values of 'n_cells' to use across all three rois.
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


def cov_compl_calc(
    df:pd.DataFrame
  , trim_df:pd.DataFrame
  , size_df:pd.DataFrame
  , size_df_raw: pd.DataFrame
  , n_cells:int
  , instance_type:str
) -> pd.DataFrame:
    """
    Generates a dataframe with each column containing the value of a different coverage or
    completeness quantification.

    Parameters
    ----------
    df : pd.DataFrame
        One row per column - raw data
        column : str
            Column name in the form '39_20'.
        roi : str
            Optic lobe region of interest.
        n_cells : int
            Number of cells that innervate the column.
        n_syn : int
            Number of synapses within the column.
        cell_body_ids : list
            List of bodyIds of the cells innervating the column.
    trim_df : pd.DataFrame
        One row per column - trimmed data
        column : str
            Column name in the form '39_20'.
        n_cells : int
            Number of cells that innervate the column after trimming.
        n_syn : int
            Number of synapses within the column after trimming.
        hex1_id : int
            Hex1_id value of the column.
        hex2_id : int
            Hex2_id value of the column.
        roi : str
            Optic lobe region of interest.
        issue : bool
            1 if unable to form a knee during the trimming process.
    size_df : pd.DataFrame
        One row per bodyID - trimmed data
        bodyId : int
            BodyId of neuron.
        n_cols : int
            Number of columns innervated per neuron of the chosen cell type.
        roi : str
            Optic lobe region of interest.
    size_df_raw : pd.DataFrame
        One row per bodyID - raw data
        bodyId : int
            BodyId of the neuron.
        n_cols : int
            Number of columns innervated per neuron of the chosen cell type.
        roi : str
            Optic lobe region of interest.
    n_cells : int
        Number of cells of the instance.
    instance_type : str
        instance type.

    Returns
    -------
    quant_df_all : pd.DataFrame
        One dataframe per cell type for all three optic lobe regions.

        instance : str
            cell type (instance)
        roi : str
            Optic lobe region of interest.
        cols_covered_pop : int
            Number of columns covered by all cells of cell type - raw data.
        col_completeness : float
            Proportion of all columns in neuropil that are innervated by cells from the cell type.
        coverage_factor : float
            Mean value of the number of cells per column across all columns occupied - raw data.
        synaptic_coverage_factor : float
            Median number of synapses per column across all columns - raw data.
        coverage_factor_trim
            Mean value of the number of cells per column across all columns occupied - trimmed
            data.
        synaptic_coverage_factor_trim
            Mean value of the number of synapses per column across all columns occupied - trimmed
            data.
        n_syn_total
            Number of all synapses from all cells of this cell type in this roi.
        n_syn_trim
            Number of all synapses from all cells of this cell type in this roi - trimmed data.
        population_size
            Total number of cells of the cell type.
        cell_size_cols
            Median number of columns spanned per cell of cell type - trimmed data.
        area_covered_pop
            Area covered by convex hull around all columns innervated by all cells of cell types,
            using the hex coordinates of the columns - raw data.
        area_completeness
            The area covered by all cells as a proportion of the total roi area - raw data.
    """
    quant_df_all = pd.DataFrame()

    for roi_str in ["ME(R)", "LO(R)", "LOP(R)"]:

        cql = f"""
            MATCH (n:Neuron)-[e:ConnectsTo]->(m:Neuron)
            WHERE n.instance='{instance_type}'
            WITH DISTINCT e
            WITH apoc.convert.fromJsonMap(e.roiInfo) as ri
            RETURN sum(ri['{roi_str}'].post) as num_conn
        """
        o_con = fetch_custom(cql)
        [num_output_conn] = o_con.iloc[0].to_list()

        cql = f"""
            MATCH (n:Neuron)-[e:ConnectsTo]->(m:Neuron)
            WHERE n.instance='{instance_type}'
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
            instance_string = instance_type.replace("/", "-")
            data["instance"] = instance_string
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
                    if instance_type in edge_cells:
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
        DataFrame with 'hex1_id', 'hex2_id' columns.

    Returns
    -------
    col_area: float
        Surface area of the convex hull.
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
        instance : str
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


def get_edge_cell_types(roi_str: str) -> pd.DataFrame:
    """
    Get list of manually curated cell types that are predominantly at the edge of the region.
    Reads this list from an Excel file in the 'params' folder.

    Parameters
    ----------
    roi_str : str
        Optic lobe region.

    Returns
    -------
    edge_cells : list
        List of instances that are considered on the edge for the specified optic lobe region.
    """
    # Define the path to the Excel file
    file_path = Path(find_dotenv()).parent / "params" / "Edge_cell_types.xlsx"

    assert file_path.is_file()\
      , f"Parameter file for edge cells is missing. Make sure {file_path} exists."
    # Load the spreadsheet
    df = pd.read_excel(file_path)

    # Extract the edge cells for the specific region
    assert roi_str in df.columns\
      , f"Neuropil {roi_str} does not exist in Edge cell type list {file_path}"
    edge_cells = df[roi_str].dropna()

    return edge_cells
