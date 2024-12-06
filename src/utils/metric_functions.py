"""
Functions used when accessing or generating the 'completeness_metrics.pickle' file that contains
a dataframe with spatial coverage and completness metrics for all instance types within the optic
lobe.
"""

from pathlib import Path
import concurrent.futures
from dotenv import find_dotenv
import pandas as pd

from utils.trim_helper import TrimHelper
from utils.column_features_functions import hex_from_col, cov_compl_calc
from utils.helper import slugify
from queries.completeness import fetch_ol_types_and_instances
from queries.coverage_queries import fetch_cells_synapses_per_col


def get_metrics_df(
    verbose:bool=False) -> pd.DataFrame:
    """
    Check if 'complete_metrics.pickle' file exists with combined spatial coverage and completeness
    metric information from all neuron instances in ME(R), LO(R) and LOP(R). If not, generate this
    file.

    Parameters
    ----------
    verbose : bool, default=False
        print verbose output

    Returns
    -------
    metrics_df : pd.DataFrame
        One dataframe containing the coverage and completeness metrics for all instance types
         within the optic lobe. Each row contains coverage metrics values for cells of one
         instance type in one of the optic lobe regions. Individual cell types will have between
         one and three rows in the data frame, depending on which of the neuropils of the optic
         lobe ('ME(R)', 'LO(R)' and 'LOP(R)') they innervate.

        instance : str
            Cell type (instance).
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
        coverage_factor_trim : float
            Mean value of the number of cells per column across all columns occupied - trimmed
            data.
        synaptic_coverage_factor_trim : float
            Mean value of the number of synapses per column across all columns occupied - trimmed
            data.
        n_syn_total : int
            Number of all synapses from all cells of this cell type in this roi.
        n_syn_trim : int
            Number of all synapses from all cells of this cell type in this roi - trimmed data.
        population_size : int
            Total number of cells in the cell type.
        cell_size_cols : int
            Median number of columns spanned per cell of cell type - trimmed data.
        area_covered_pop : int
            Area covered by convex hull around all columns innervated by all cells of cell types,
            using the hex coordinates of the columns - raw data.
        area_completeness : float
            The area covered by all cells as a proportion of the total roi area - raw data.
    """
    cachedir = Path(find_dotenv()).parent / "cache" / "complete_metrics"
    metric_file = cachedir / "complete_metrics.pickle"

    if metric_file.is_file():
        with metric_file.open("rb") as metric_fh:
            metrics_df = pd.read_pickle(metric_fh)
    else:  # If the file 'complete_metrics.pickle' does not exist, generate it.
        data_frames = []
        metrics_df = pd.DataFrame

        # generate a list of all of the cell type instances in the optic lobe
        types = fetch_ol_types_and_instances(side="both")
        all_cell_types = types["instance"]

        # generate a pickle file containing the metrics data frame for each cell type instance.
        # These data frames will contain separate rows for the synapses of that instance type in
        # the ME(R), LO(R) and LOP(R) if necessary.
        with concurrent.futures.ProcessPoolExecutor(max_workers=48) as executor:
            for instance_df in executor.map(get_completeness_metrics, all_cell_types.to_list()):
                data_frames.append(instance_df)

        # Concatenate all instance data frames into a single data frame
        metrics_df = pd.concat(data_frames, axis=0, ignore_index=True)

        # Save the combined data frame as a pickle file
        with metric_file.open("wb") as metric_fh:
            metrics_df.to_pickle(metric_fh)

    if verbose:
        print(
            f"The 'metrics_df' data frame contains {len(metrics_df)} "\
            f"rows with about {metrics_df['instance'].nunique()} unique "\
            "instance types."
        )

    return metrics_df


def get_completeness_metrics(
    instance:str
  , trim_helper:TrimHelper=None) -> pd.DataFrame:
    """
    Checks for the existence of a pickle file containing the coverage metrics for an individual
    'instance' type. If the pickle file exists, it loads and returns the dataframe, else the
    pickle file is generated for that 'instance' type.

    Parameters
    ----------
    instance : str
        name of a cell instance, e.g. 'TmY5a_R'
    trim_helper : TrimHelper
        Object caching access to the trimmed and raw data frames. If None, it's created
        internally without the benefit of caching.

    Returns
    -------
    metric_df : pd.DataFrame
        Data frame containing the coverage and completeness metrics for a given 'instance'. The
        data frame contains one row for each neuropil (e.g. 'ME(R)', 'LO(R)' or 'LOP(R)') that
        the instance type occupies.

        instance : str
            Cell type (instance).
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
            Total number of cells in the cell type.
        cell_size_cols
            Median number of columns spanned per cell of cell type - trimmed data.
        area_covered_pop
            Area covered by convex hull around all columns innervated by all cells of cell types,
            using the hex coordinates of the columns - raw data.
        area_completeness
            The area covered by all cells as a proportion of the total roi area - raw data.
    """

    cachedir = Path(find_dotenv()).parent / "cache" / "complete_metrics"
    cachedir.mkdir(parents=True, exist_ok=True)

    if trim_helper is None:
        trim_helper = TrimHelper(instance)

    metric_fn = cachedir / f"{slugify(instance)}.pickle"

    if metric_fn.is_file():
        with metric_fn.open('rb') as metric_fh:
            metric_df = pd.read_pickle(metric_fh)
    else:
        df = fetch_cells_synapses_per_col(
            cell_instance=instance
          , roi_str=["ME(R)", "LO(R)", "LOP(R)", "AME(R)", "LA(R)"]
        )
        df = hex_from_col(df)

        named_df = fetch_ol_types_and_instances(side="both")
        pop_size = named_df[named_df["instance"] == instance]["count"]\
            .values\
            .astype(int)

        metric_df = cov_compl_calc(
            df
          , trim_df=trim_helper.trim_df
          , size_df=trim_helper.size_df
          , size_df_raw=trim_helper.size_df_raw
          , n_cells=pop_size
          , instance_type=instance
        )
        with metric_fn.open('wb') as metric_fh:
            metric_df.to_pickle(metric_fh)

    return metric_df
