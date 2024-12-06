from abc import ABC
import pandas as pd
import numpy as np

import kneed

from utils.column_features_functions import hex_from_col
from queries.coverage_queries import fetch_syn_per_col_for_instance

class TrimHelper(ABC):

    def __init__(
        self
      , instance:str
    ):
        self.__instance = instance
        self.__trim = None
        self.__size = None
        self.__size_raw = None

    @property
    def trim_df(self):
        if self.__trim is None:
            self.__calc_trim()
        return self.__trim

    @property
    def size_df(self):
        if self.__size is None:
            self.__calc_trim()
        return self.__size

    @property
    def size_df_raw(self):
        if self.__size_raw is None:
            self.__calc_trim()
        return self.__size_raw

    def __calc_trim(self):
        trim_df, size_df, size_df_raw = self.__get_trim_df_all_rois()
        self.__trim = trim_df
        self.__size = size_df
        self.__size_raw = size_df_raw


    def __get_trim_df_all_rois(
        self
      , syn_type:str='all'
    ) -> pd.DataFrame:
        """
        Combines pd.DataFrames from all three neuropil regions after trimming.

        Parameters
        ----------
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

        size_me, size_me_raw, df_me = self.__get_trim_df(
            roi_str="ME(R)"
          , syn_type=syn_type
        )
        if not isinstance(df_me, pd.DataFrame):
            df_me = pd.DataFrame()

        size_lo, size_lo_raw, df_lo = self.__get_trim_df(
            roi_str="LO(R)"
          , syn_type=syn_type
        )
        if not isinstance(df_lo, pd.DataFrame):
            df_lo = pd.DataFrame()

        size_lop, size_lop_raw, df_lop = self.__get_trim_df(
            roi_str="LOP(R)"
          , syn_type=syn_type
        )
        if not isinstance(df_lop, pd.DataFrame):
            df_lop = pd.DataFrame()

        df = pd.concat([df_me, df_lo, df_lop])
        size_df = pd.concat([size_me, size_lo, size_lop])
        size_df_raw = pd.concat([size_me_raw, size_lo_raw, size_lop_raw])
        return df, size_df, size_df_raw


    def __get_trim_df(
        self
      , roi_str:str
      , syn_type:str='all'
      , cumsum_min:float=0.775
      , cumsum_fix:float=0.999
    ) -> pd.DataFrame:
        """
        Determine the number of cells from a cell type that innervate each column after trimming
        off the "outlier" synapses.
        For each neuron, find a lower threshold on the number synapses in a column in order to
        retain those synapses. This threshold equals the number of synapses in the `rank_thre`'th
        largest column, where `rank_thre` is computed from cell-type information. Namely, we use
        an elbow method on the median synapse count per column vs. rank of column (the column with
        the largest number of synapses has rank 1, the column with the second largest number of
        synapses has rank 2 etc.).

        Parameters
        ----------
        roi_str : str
            neuprint ROI, can only be ME(R), LO(R), LOP(R)
        syn_type : str, default='all'
            synapse type to use. Possible options are 'pre', 'post' or 'all'
        cumsum_min : float, default=0.775
            minimum fraction of cumulative sum of synapses in columns after trimming
            if knee finder gives a lower fraction then we find the rank with fraction cumsum_fix.
            the value 0.775 was obtained as a dip in the bimodal distribution of the cumsum
            fraction from the knee finder for cell-types in ME with at least 1000 synapses.
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
        assert isinstance(roi_str, str) and roi_str in ["ME(R)", "LO(R)", "LOP(R)"]\
          , f"ROI must be one of 'ME(R)', 'LO(R)', 'LOP(R)', but is actually '{roi_str}'"

        # fetch bodyIds of neurons from cell type
        trim_df = pd.DataFrame()
        issue = 0

        trim_df = fetch_syn_per_col_for_instance(
            neuron_instance=self.__instance
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
            size_df = trim_df\
                .groupby(["roi", "bodyId"])\
                .nunique()\
                .reset_index()\
                .loc[:, ["bodyId", "column", "roi"]]\
                .rename(columns={"column": "n_cols"})
            # size - n_cols per cell - raw data - untrimmed
            size_df_raw = raw_df\
                .groupby(["roi", "bodyId"])\
                .nunique()\
                .reset_index()\
                .loc[:, ["bodyId", "column", "roi"]]\
                .rename(columns={"column": "n_cols"})
        else:
            size_df = pd.DataFrame(data={'roi': [roi_str]})
            size_df_raw = pd.DataFrame(data={'roi': [roi_str]})
            df2 = pd.DataFrame(data={'roi': [roi_str]})

        return size_df, size_df_raw, df2
