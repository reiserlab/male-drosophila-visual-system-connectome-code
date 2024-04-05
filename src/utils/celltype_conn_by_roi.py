from abc import ABC
from functools import reduce

import numpy as np
import pandas as pd

from neuprint import fetch_neurons, fetch_synapse_connections\
  , merge_neuron_properties, fetch_all_rois, fetch_mean_synapses
from neuprint import NeuronCriteria as NC, SynapseCriteria as SC

from utils.helper import slugify

class CelltypeConnByRoi(ABC):
    """
    This object is used to fetch all the input and output synapses from a target cell type in a
    specific ROI.
    Basic stats are then calculated at the neuron and cell type level for all the inputs/outputs
    """

    def __init__(
        self
      , target_celltype:str
      , roi:str | list
      , hemisphere:str=None
    ):
        all_rois = fetch_all_rois()
        self.__target_celltype = target_celltype
        if (roi == 'ALL') or \
            (isinstance(roi, str) and roi in all_rois) or \
            (isinstance(roi, list) and all([ct in all_rois for ct in roi])):
            self.__roi = roi
        else:
            raise ValueError(
                "given roi not valid - should be from list in `fetch_all_rois()` or ALL")
        assert hemisphere is None or hemisphere in ['R', 'L'],\
            f"Only L and R are allowed as hemispheres, not {hemisphere}"

        # used in fetch synapses methods
        self.__input_syn_conf_thresh = 0.5
        self.__output_syn_conf_thresh = 0.5

        # store hemisphere
        self.__hemisphere = hemisphere

        # used in calculating cell type stats
        self.__input_conn_min_thresh = 1
        self.__output_conn_min_thresh = 1

        # used when fetching output synapses (since sometimes the fetch can timeout)
        self.input_min_tot_w = 1
        self.input_batch_siz = 100
        self.output_min_tot_w = 1
        self.output_batch_siz = 100

        self.__input_synapses:pd.DataFrame = None
        self.__output_synapses:pd.DataFrame = None
        self.__input_neurons_w_stats:pd.DataFrame = None
        self.__output_neurons_w_stats:pd.DataFrame = None
        self.__input_celltypes_w_stats:pd.DataFrame = None
        self.__output_celltypes_w_stats:pd.DataFrame = None


    def get_target_celltype(self) -> str:
        """
        get the target celltype

        Returns
        -------
        celltype : str
            Celltype for this object
        """
        return self.__target_celltype

    @property
    def instance(self) -> str:
        return f"{self.get_target_celltype()}_{self.hemisphere}"

    @property
    def hemisphere(self) -> str:
        """
        get the target celltype's hemisphere

        Returns
        -------
        hemisphere : str
            hemisphere for the target celltype
        """
        return self.__hemisphere


    def file_id(self) -> str:
        """
        get a unique file ID

        Returns
        -------
        file_name : str
            unique filename for this object
        """
        roilist = self.__roi
        if isinstance(roilist, list):
            roilist = '_'.join(roilist)
        filename = f"{self.__target_celltype}_{roilist}"
        if self.hemisphere:
            filename += f"_{self.hemisphere}"
        return slugify(filename)


    def get_roi(self) -> str:
        """
        Get the ROI

        Returns
        -------
        roi : str
            one of ME(R), LO(R), LOP(R)
        """
        return self.__roi


    def set_input_syn_conf_thresh(self, conf_thresh:float):
        """
        Set the threshold for input synapses

        Parameters
        ----------
        conf_thresh : float
            Lower threshold for the input synapse confidence
        """
        assert (conf_thresh is None) or ((conf_thresh >=0) & (conf_thresh < 1)), \
            f"conf_thresh should be None or between 0 and 1, not {conf_thresh}"
        if self.__input_syn_conf_thresh != conf_thresh:
            self.__input_synapses = None
            self.__input_neurons_w_stats = None
        self.__input_syn_conf_thresh = conf_thresh


    def set_output_syn_conf_thresh(self, conf_thresh:float):
        """
        Set the threshold for the output synapses

        Parameters
        ----------
        conf_thresh : float
            lower threshold for output synapse confidence
        """
        assert (conf_thresh is None) or ((conf_thresh >=0) & (conf_thresh < 1)), \
            f"conf_thresh should be None or between 0 and 1, not {conf_thresh}"
        if self.__output_syn_conf_thresh != conf_thresh:
            self.__output_synapses = None
            self.__output_neurons_w_stats = None
        self.__output_syn_conf_thresh = conf_thresh


    def set_input_conn_min_thresh(self, conn_min:int):
        """
        Set the threshold of minimal number of connections for the input

        Parameter
        ---------
        conn_min : int
            Minimum number of connections considered for the output
        """
        assert conn_min >=1, 'conn_min should be at least 1'
        if self.__input_conn_min_thresh != conn_min:
            self.__input_celltypes_w_stats = None
        self.__input_conn_min_thresh = conn_min


    def set_output_conn_min_thresh(self, conn_min:int):
        """
        Set the threshold of minimal number of connections for the output

        Parameter
        ---------
        conn_min : int
            Minimum number of connections considered for the output
        """
        assert conn_min >=1, 'conn_min should be at least 1'
        if self.__output_conn_min_thresh != conn_min:
            self.__output_celltypes_w_stats = None
        self.__output_conn_min_thresh = conn_min


    def get_input_synapses(self) -> pd.DataFrame:
        """
        Get all input sunapses. If they haven't been fetched before, this
          method automagically fetches the synapses via
          `__fetch_input_synapses()`

        Returns
        -------
        self.__input_synapses: pd.DataFrame

            Data frame that results from neuPrint 'fetch_synapse_connections' with 'type' and
            'soma location' added for both pre and post synapses
        """
        if not isinstance(self.__input_synapses, pd.DataFrame):
            self.__fetch_input_synapses()
        return self.__input_synapses


    def get_output_synapses(self) -> pd.DataFrame:
        """
        Get all output synapses. If they haven't been fetched before, this
          method automagically fetches the synapses via
          `__fetch_output_synapses()`

        Returns
        -------
        self.__output_synapses: pd.DataFrame
            Data frame that results from neuPrint 'fetch_synapse_connections'
              with 'type' and 'soma location' added for both pre and post
              synapses
        """
        if not isinstance(self.__output_synapses, pd.DataFrame):
            self.__fetch_output_synapses()
        return self.__output_synapses


    def get_input_neurons_w_stats(self) -> pd.DataFrame:
        """
        Returns
        -------
        self.__input_neurons_w_stats : pd.DataFrame
            Uses the 'input_synapses' data frame to calculate different stats at the neuron level

            Data frame detailing connections to the above target type with the following columns
            bodyID_post : str
                BodyId of relevant target neurons
            bodyId_pre : str
                BodyId of relevant inputs into target neuron
            type_post/pre : str
                Neuron type of the corresponding post/pre bodyID neuron
            somaLocation_post : list
                soma coordinates from the post cell
            x/y/z_post : float
                mean position of all input synapses onto post cell within the given rio_pre
            syn_count : int
                Number of pre to post synapses from the corresponding neurons
            tot_syn_per_pre (tot_syn_per_preNroi) : int
                Number of input synapses from all the same pre_type neurons to this specific
                  bodyID_post neuron (and this roi)
            tot_syn (tot_syn_per_roi) : int
                Number of input synapses from all the neurons (including 'None') to this specific
                  bodyId (and roi) from all that were fetched
            frac_tot_pre (frac_tot_preNroi) : float
                Fraction of all pre_type neurons as this bodyID_pre to this bodyID post
                  (including 'None') (in this roi)
            frac_tot_pre_roi_tot : float
                Fraction of all pre_type neurons as this bodyID_pre to this bodyID post in this ROI
                divided by tot synapses (bot just ROI synapses)
            frac_inp_pre_type : float
                Fraction of this specific pre_type neuron from all the same pre_type neurons to
                  this bodyId_post from all that were fetched
            tot_num_inp : int
                total number of input cells to a specific post cell
            tot_num_syn : int
                total number of synapses to a specific post cell
                  (excluding synapses from None types)
            rank_first/dense : float
                Two different rank calculations for frac_inp_pre_type (see rank() for description)
        """
        if not isinstance(self.__input_neurons_w_stats, pd.DataFrame):
            self.__calc_input_neurons_w_stats()
        return self.__input_neurons_w_stats


    def get_output_neurons_w_stats(self) -> pd.DataFrame:
        """
        Returns
        -------
        __output_neurons_w_stats : pd.DataFrame
            Data frame detailing connections to the above target type with the following columns
            bodyID_post : str
                BodyId of relevant target neurons
            bodyId_pre : str
                BodyId of relevant inputs into target neuron
            type_post/pre : str
                Neuron type of the corresponding post/pre bodyID neuron
            syn_count : int
                Number of pre to post synapses from the corresponding neurons
            tot_syn_per_post (tot_syn_per_postNroi) : float
                Number of output synapses to all the same post_type neurons from this specific
                  bodyID_pre neuron (and this roi)
            tot_syn (tot_syn_per_roi) : int
                Number of output synapses to all the neurons (including 'None') from this
                  specific bodyId (and roi) from all that were fetched
            frac_tot_post (frac_tot_postNroi) : float
                total syn per post divided by total syn (for the roi)
            frac_tot_post_roi_tot : float
                total syn per post divided in the ROI by overall total syn for the bodyId_pre cell
                from all that were fetched
            frac_out_post_type : float
                Fraction of this specific post_type neuron from all the same post_type neurons of
                  this bodyId_pre
            tot_num_sym : int
                total number of synapses from a specific pre-cell
            tot_num_out : int
                total number of post cells a specific pre cell outputs to
            rank_first/dense : float
                Two different rank calculations for frac_inp_pre_type (see rank() for description)
        """
        if not isinstance(self.__output_neurons_w_stats, pd.DataFrame):
            self.__calc_output_neurons_w_stats()
        return self.__output_neurons_w_stats


    def get_input_celltypes_w_stats(self) -> pd.DataFrame:
        """
        Returns
        -------
        self.__input_celltypes_w_stats : pd.Dataframe
            Data frame with by connected celltype statistics. Includes the following columns:

            target : str
                target cell type that is the also the target for all the different connected
                  cell types.
                Note! use target and source here for networkx use later
            type_pre : str
                names for the relevant connected cell types
            mean/med/std_tot_conn : Float
                mean/median/standard deviation of total number of connected cells from a particular
                  celltype to an individual target cell type
            mean/med/std_tot_syn : float
                mean/median/standard deviation of total number of synapses from a particular
                  celltype to an individual target cell type
            type_counter : int
                number of target cells that are connected to the specific connected cell type
            type_frac : float
                fraction of target cells that are connected to the specific connected cell type
            frac_tot_pre : float
                median fraction for the specific connected cell type out of total inputs
                Note! does not add to 1, since calculated per target type
            tot_syn_per_pre : float
                Same as med_tot_syn but is calculated before (in `calc_input_neurons_w_stats()`)
                  and does not take conn_min_thresh into account
        """
        if not isinstance(self.__input_celltypes_w_stats, pd.DataFrame):
            self.__calc_input_celltypes_w_stats()
        return self.__input_celltypes_w_stats


    def get_output_celltypes_w_stats(self) -> pd.DataFrame:
        """
        Returns
        -------
        self.__output_celltypes_w_stats: pd.Dataframe
            Data frame with by connected celltype statistics. Includes the following columns:

            source : str
                target cell type that is the source for all the different connected cell types.
                Note! use target and source here for networkx use later
            type_post : str
                names for the relevant connected cell types
            mean/med/std_tot_conn : Float
                mean/median/standard deviation of total number of connected cells from a particular
                  celltype to an individual target cell type
            mean/med/std_tot_syn : float
                mean/median/standard deviation of total number of synapses from a particular
                  celltype to an individual target cell type
            type_counter : int
                number of target (labelled source here) cells that are connected to the specific
                  connected cell type
            type_frac : float
                fraction of target (labelled source here) cells that are connected to the specific
                  connected cell type
            frac_tot_post : float
                median fraction for the specific connected cell type out of total outputs
                Note! does not add to 1, since calculated per target type
            tot_syn_per_post : float
                Same as med_tot_syn but is calculated before (in 'calc_output_neurons_w_stats()')
                  and does not take conn_min_thresh into account
        """
        if not isinstance(self.__output_celltypes_w_stats, pd.DataFrame):
            self.__calc_output_celltypes_w_stats()
        return self.__output_celltypes_w_stats


    def show_celltype_conn_by_roi_obj(self):
        """
        Print input synapses, output synapses, and input / output neurons
          with stats.
        """
        print(f"celltype: {self.get_target_celltype()}")
        print(f"hemisphere: {self.hemisphere}")
        print(f"ROI: {self.get_roi()}")
        print("Parameters:")
        print(f"input_syn_conf_thresh:{self.__input_syn_conf_thresh}")
        print(f"output_syn_conf_thresh:{self.__output_syn_conf_thresh}")
        print(f"input_conn_min_thresh:{self.__input_conn_min_thresh}")
        print(f"output_conn_min_thresh:{self.__output_conn_min_thresh}")
        print(f"output_min_tot_w:{self.output_min_tot_w}")
        print(f"output_batch_siz:{self.output_batch_siz}", )

        self.__print_df_by_getter('get_input_synapses')
        self.__print_df_by_getter('get_output_synapses')
        self.__print_df_by_getter('get_input_neurons_w_stats')
        self.__print_df_by_getter('get_output_neurons_w_stats')
        self.__print_df_by_getter('get_input_celltypes_w_stats')
        self.__print_df_by_getter('get_output_celltypes_w_stats')


    def __print_df_by_getter(self, getter_name:str):
        tmp_get = getattr(self, getter_name) # TODO: check if function exists
                                             #       before calling it
        tmp_df = tmp_get()
        print(f"{getter_name} df size: {tmp_df.size}")
        tmp_df.info(verbose=False)


    def __fetch_input_synapses(self):
        """
        This function fetches all the synaptic inputs to the target cell type in the specified ROI.

        This function fetches all the target_type neurons that are in roi_pre and looks for all
            their input types (also limited to inputs in roi_pre). The function removes all
            synapses with confidence below a specified threshold, and removes all entries
            with a `None` as their `pre_type`
        """
        syn_conf_thresh = self.__input_syn_conf_thresh

        if self.__input_synapses is None:

            if self.__input_neurons_w_stats is not None:
                print('input syanpses fetched again - deleted input_syapses_w_stats')
                self.__input_neurons_w_stats = None

            if self.hemisphere:
                neuron_criteria_post = NC(
                    instance=f"^.*\\b?{self.__target_celltype}(\\b|_).*{self.hemisphere}$"
                  , regex=True
                )
            else:
                neuron_criteria_post  = NC(
                    type=self.__target_celltype,
                    regex=True) # regex to deal with all types of T4 and T5 (T4.* or T4[a-d])
            neuron_df, _ = fetch_neurons(neuron_criteria_post)

            print(f"synapse connection fetched with {syn_conf_thresh} confidence threshold")
            if self.__roi == 'ALL':
                syn_criteria = SC(confidence=syn_conf_thresh)
            else:
                syn_criteria = SC(rois=self.__roi, confidence=syn_conf_thresh)
            syn_df = fetch_synapse_connections(
                None
              , neuron_criteria_post
              , syn_criteria
              , min_total_weight=self.input_min_tot_w
              , batch_size=self.input_batch_siz)

            nid_pre = pd.unique(syn_df['bodyId_pre']).astype(int)
            neuron_criteria = NC(bodyId=nid_pre)
            neuron_pre_df , _ = fetch_neurons(neuron_criteria)
            # adds type to the merged df
            comb_df = pd.concat([
                    neuron_df[['bodyId', 'type', 'somaLocation']],
                    neuron_pre_df[['bodyId', 'type', 'somaLocation']]])\
                .drop_duplicates(subset='bodyId')\
                .reset_index(drop=True)
            rel_syn_df = merge_neuron_properties(comb_df, syn_df, ['type', 'somaLocation'])

            # correcting cases where one ROI is present but the other is missing
            #   (probably at the edge of the ROI)
            post_miss_pre_pres_ind = rel_syn_df[
                    (rel_syn_df['roi_pre'].notnull()) & (rel_syn_df['roi_post'].isna())
                ].index
            rel_syn_df.loc[post_miss_pre_pres_ind, 'roi_post'] = rel_syn_df.\
                loc[post_miss_pre_pres_ind, 'roi_pre']
            post_pres_pre_miss_ind = rel_syn_df[
                    (rel_syn_df['roi_post'].notnull()) & (rel_syn_df['roi_pre'].isna())
                ].index
            rel_syn_df.loc[post_pres_pre_miss_ind, 'roi_pre'] = rel_syn_df\
                .loc[post_pres_pre_miss_ind, 'roi_post']

            self.__input_synapses = rel_syn_df


    def get_input_per_roi(self):
        """
        initially produced by a figure aux function (`num_df`)

        Returns
        -------
        num_df : pd.DataFrame
            target : str
                name of the target celltype
            roi : str
                ROI
            num_cells : int
                count of cells
        """
        syn_conf_thresh = self.__input_syn_conf_thresh

        if self.hemisphere:
            neuron_criteria_post = NC(
                instance=f"^.*\\b?{self.__target_celltype}(\\b|_).*{self.hemisphere}$"
              , regex=True
            )
        else:
            neuron_criteria_post  = NC(
                type=self.__target_celltype
              , regex=True
            )
        syn_criteria = SC(rois=self.__roi, confidence=syn_conf_thresh)
        mean_synapses = fetch_mean_synapses(neuron_criteria_post, syn_criteria)
        num_df = mean_synapses.groupby(['roi'])['bodyId'].nunique().to_frame()
        num_df['target'] = self.__target_celltype
        num_df = num_df.reset_index().rename(columns={'bodyId':'num_cells'})
        return num_df.loc[:,['target', 'roi', 'num_cells']]


    def __fetch_output_synapses(self):
        """
        This function fetches all the synaptic output to the target cell type in the specified ROI.

        This function fetches all the target_type neurons that are in roi_pre and looks for all
        Their output types. The function removes all synapses with confidence
        below a specified threshold, and removes all entries with a 'None' as their pre_type
        """
        syn_conf_thresh = self.__output_syn_conf_thresh

        if self.__output_synapses is None:

            if self.__output_neurons_w_stats is not None:
                print('output syanpses fetched again - deleted output_syapses_w_stats')
                self.__output_neurons_w_stats = None

            if self.hemisphere:
                neuron_criteria_pre = NC(
                    instance=f"^.*\\b?{self.__target_celltype}(\\b|_).*{self.hemisphere}$"
                  , regex=True
                )
            else:
                neuron_criteria_pre  = NC(
                    type=self.__target_celltype,
                    regex=True
                )
            neuron_df, _ = fetch_neurons(neuron_criteria_pre)

            print(f"synapse connection fetched with {syn_conf_thresh} confidence threshold")

            if self.__roi == 'ALL':
                syn_criteria = SC(confidence=syn_conf_thresh)
            else:
                syn_criteria = SC(rois=self.__roi, confidence=syn_conf_thresh)
            syn_df = fetch_synapse_connections(
                neuron_criteria_pre
              , None
              , syn_criteria
              , min_total_weight=self.output_min_tot_w
              , batch_size=self.output_batch_siz)

            nid_post = pd.unique(syn_df['bodyId_post']).astype(int)
            neuron_criteria = NC(bodyId=nid_post)
            neuron_post_df , _ = fetch_neurons(neuron_criteria)
            comb_df = pd.concat([
                    neuron_df[['bodyId', 'type', 'somaLocation']],
                    neuron_post_df[['bodyId', 'type', 'somaLocation']]])\
                .drop_duplicates(subset='bodyId')\
                .reset_index(drop=True)
            rel_syn_df = merge_neuron_properties(comb_df, syn_df, ['type', 'somaLocation'])

            rel_syn_df = rel_syn_df[ ~rel_syn_df.round(
                {'x_pre': 2, 'y_pre': 2, 'z_pre': 2})\
                [['x_pre','y_pre','z_pre']]\
                .duplicated() ] #JUDITH
            # correcting cases where one ROI is present but the other is missing
            #    (probably at the edge of the ROI)
            post_miss_pre_pres_ind = rel_syn_df[
                    (rel_syn_df['roi_pre'].notnull()) & (rel_syn_df['roi_post'].isna())
                ].index
            rel_syn_df.loc[post_miss_pre_pres_ind, 'roi_post'] = rel_syn_df\
                .loc[post_miss_pre_pres_ind, 'roi_pre']
            post_pres_pre_miss_ind = rel_syn_df[
                    (rel_syn_df['roi_post'].notnull()) & (rel_syn_df['roi_pre'].isna())
                ].index
            rel_syn_df.loc[post_pres_pre_miss_ind, 'roi_pre'] = rel_syn_df\
                .loc[post_pres_pre_miss_ind, 'roi_post']

            self.__output_synapses = rel_syn_df


    def __calc_input_neurons_w_stats(self):
        """
        This function calculates basic stats on all the synaptic inputs to the target cell type in
          the specified ROI.

        if 'input_neurons' is empty it fetches it and calculates the result

        Returns
        -------
        self.__input_neurons_w_stats : pd.DataFrame
            Uses the 'input_synapses' data frame to calculate different stats at the neuron level

            Data frame detailing connections to the above target type with the following columns
            bodyID_post : str
                BodyId of relevant target neurons
            bodyId_pre : str
                BodyId of relevant inputs into target neuron
            type_post/pre : str
                Neuron type of the corresponding post/pre bodyID neuron
            somaLocation_post : list
                soma coordinates from the post cell
            x/y/z_post : float
                mean position of all input synapses onto post cell within the given rio_pre
            syn_count : int
                Number of pre to post synapses from the corresponding neurons
            tot_syn_per_pre (tot_syn_per_preNroi) : int
                Number of input synapses from all the same pre_type neurons to this specific
                  bodyID_post neuron (and this roi)
            tot_syn (tot_syn_per_roi) : int
                Number of input synapses from all the neurons (including 'None') to this specific
                  bodyId (and roi) from all that were fetched
            frac_tot_pre (frac_tot_preNroi) : float
                Fraction of all pre_type neurons as this bodyID_pre to this bodyID post
                  (including 'None') (in this roi)
            frac_tot_pre_roi_tot : float
                Fraction of all pre_type neurons as this bodyID_pre to this bodyID post in this ROI
                divided by tot synapses (bot just ROI synapses)
            frac_inp_pre_type : float
                Fraction of this specific pre_type neuron from all the same pre_type neurons to
                  this bodyId_post from all that were fetched
            tot_num_inp : int
                total number of input cells to a specific post cell
            tot_num_syn : int
                total number of synapses to a specific post cell
                  (excluding synapses from None types)
            rank_first/dense : float
                Two different rank calculations for frac_inp_pre_type (see rank() for description)
        """

        if self.__input_neurons_w_stats is None:

            if self.__input_synapses is None:
                self.__fetch_input_synapses()

            if self.__input_celltypes_w_stats is not None:
                print('input neurons calculated again - deleted input_celltypes_w_stats')
                self.__input_celltypes_w_stats = None

            rel_syn_df = self.__input_synapses
            # total synapses and synapse center of mass calculated before removing None types
            tot_inp_df = rel_syn_df\
                .groupby(['bodyId_post'])\
                .size()\
                .reset_index(name='tot_syn')

            tot_inp_df_proi = rel_syn_df\
                .groupby(['roi_post','bodyId_post'])\
                .size()\
                .reset_index(name='tot_syn_per_roi')\
                .rename(columns={'roi_post':'roi'})

            syn_com_df = rel_syn_df\
                .groupby(['roi_post', 'bodyId_post'])[['x_post', 'y_post', 'z_post']]\
                .mean()\
                .reset_index()\
                .rename(columns={'roi_post':'roi'})

            syn_com_df.drop_duplicates(inplace=True)

            criteria2 = rel_syn_df['type_pre'].notnull()
            rel_syn_df = rel_syn_df[criteria2]

            soma_loc_df = rel_syn_df\
                .groupby('bodyId_post')['somaLocation_post']\
                .first()

            tot_pre_post_syn_df = rel_syn_df\
                .groupby(['bodyId_pre', 'bodyId_post', 'type_pre', 'type_post', 'roi_post'])\
                .size()\
                .reset_index(name='syn_count')\
                .rename(columns={'roi_post':'roi'})
            tot_pre_post_syn_df = tot_pre_post_syn_df\
                .merge(soma_loc_df, on='bodyId_post')
            tot_pre_post_syn_df = tot_pre_post_syn_df\
                .merge(syn_com_df, on=['roi', 'bodyId_post'])
            tot_pre_post_syn_df['num_pre_per_post'] = tot_pre_post_syn_df\
                .groupby(['bodyId_post', 'type_pre'], sort=False)['syn_count']\
                .transform('count')
            tot_pre_post_syn_df['tot_syn_per_pre'] = tot_pre_post_syn_df\
                .groupby(['bodyId_post', 'type_pre'], sort=False)['syn_count']\
                .transform('sum')
            tot_pre_post_syn_df['tot_syn_per_preNroi'] = tot_pre_post_syn_df\
                .groupby(['roi','bodyId_post', 'type_pre'], sort=False)['syn_count']\
                .transform('sum')
            tot_pre_post_syn_df['tot_conn_per_preNroi'] = tot_pre_post_syn_df\
                .groupby(['roi','bodyId_post', 'type_pre'], sort=False)['syn_count']\
                .transform('count')
            tot_pre_post_syn_df = tot_pre_post_syn_df\
                .merge(tot_inp_df, 'left', on='bodyId_post')
            tot_pre_post_syn_df = tot_pre_post_syn_df\
                .merge(tot_inp_df_proi, on=['roi','bodyId_post'])
            temp_col = tot_pre_post_syn_df['tot_syn_per_pre']\
                .div(tot_pre_post_syn_df['tot_syn'].values)
            temp_col_proi = tot_pre_post_syn_df['tot_syn_per_preNroi']\
                .div(tot_pre_post_syn_df['tot_syn_per_roi'].values)
            temp_col_roitot = tot_pre_post_syn_df['tot_syn_per_preNroi']\
                .div(tot_pre_post_syn_df['tot_syn'].values)
            tot_pre_post_syn_df.insert(
                len(tot_pre_post_syn_df.columns)-1
              , 'frac_tot_pre'
              , temp_col
            )
            tot_pre_post_syn_df.insert(
                len(tot_pre_post_syn_df.columns)-1
              , 'frac_tot_pre_by_roi'
              , temp_col_proi
            )
            tot_pre_post_syn_df.insert(
                len(tot_pre_post_syn_df.columns)-1
              , 'frac_tot_pre_roi_tot'
              , temp_col_roitot
            )
            temp_col2 = tot_pre_post_syn_df['syn_count']\
                .div(tot_pre_post_syn_df['tot_syn_per_pre'].values)
            temp_loc = tot_pre_post_syn_df.columns.get_loc('tot_syn_per_pre')+1
            tot_pre_post_syn_df.insert(temp_loc, 'frac_inp_pre_type', temp_col2)

            tot_pre_post_syn_df['tot_num_inp'] = tot_pre_post_syn_df\
                .groupby(['bodyId_post'])['syn_count']\
                .transform('count')
            tot_pre_post_syn_df['tot_num_inp_by_roi'] = tot_pre_post_syn_df\
                .groupby(['roi','bodyId_post'])['syn_count']\
                .transform('count')
            tot_pre_post_syn_df['tot_num_syn'] = tot_pre_post_syn_df\
                .groupby(['bodyId_post'])['syn_count']\
                .transform('sum')
            tot_pre_post_syn_df['tot_num_syn_by_roi'] = tot_pre_post_syn_df\
                .groupby(['roi', 'bodyId_post'])['syn_count']\
                .transform('sum')

            # sorting the df just so it would be readable if printed for a single post_cell
            tot_pre_post_syn_df = tot_pre_post_syn_df\
                .sort_values(
                    by=['bodyId_post', 'type_pre', 'syn_count'],
                    ascending=[True, True, False])\
                .reset_index(drop=True)

            # adding to calculations of rank
            tot_pre_post_syn_df['rank_first'] = tot_pre_post_syn_df\
                .groupby(['bodyId_post', 'type_pre'])['syn_count']\
                .rank('first', ascending=False)
            tot_pre_post_syn_df['rank_dense'] = tot_pre_post_syn_df\
                .groupby(['bodyId_post', 'type_pre'])['syn_count']\
                .rank('dense', ascending=False)

            self.__input_neurons_w_stats = tot_pre_post_syn_df


    def __calc_output_neurons_w_stats(self):
        """
        This function calculates basic stats on all the synaptic outputs to the target cell type
          in the specified ROI.

        if 'output_neurons' is empty it fetches it and calculates the result

        Returns
        -------
        __output_neurons_w_stats : pd.DataFrame
            Data frame detailing connections to the above target type with the following columns
            bodyID_post : str
                BodyId of relevant target neurons
            bodyId_pre : str
                BodyId of relevant inputs into target neuron
            type_post/pre : str
                Neuron type of the corresponding post/pre bodyID neuron
            syn_count : int
                Number of pre to post synapses from the corresponding neurons
            tot_syn_per_post (tot_syn_per_postNroi) : float
                Number of output synapses to all the same post_type neurons from this specific
                  bodyID_pre neuron (and this roi)
            tot_syn (tot_syn_per_roi) : int
                Number of output synapses to all the neurons (including 'None') from this
                  specific bodyId (and roi) from all that were fetched
            frac_tot_post (frac_tot_postNroi) : float
                total syn per post divided by total syn (for the roi)
            frac_tot_post_roi_tot : float
                total syn per post divided in the ROI by overall total syn for the bodyId_pre cell
                from all that were fetched
            frac_out_post_type : float
                Fraction of this specific post_type neuron from all the same post_type neurons of
                  this bodyId_pre
            tot_num_sym : int
                total number of synapses from a specific pre-cell
            tot_num_out : int
                total number of post cells a specific pre cell outputs to
            rank_first/dense : float
                Two different rank calculations for frac_inp_pre_type (see rank() for description)
        """

        if self.__output_neurons_w_stats is None:

            if self.__output_synapses is None:
                self.__fetch_output_synapses()

            if self.__output_celltypes_w_stats is not None:
                print('output neurons calculated again - deleted output_celltypes_w_stats')
                self.__output_celltypes_w_stats = None

            rel_syn_df = self.__output_synapses
            # total synapses from a cell before removing None bodyId_posts with None as type
            tot_out_df = rel_syn_df\
                .groupby(['bodyId_pre'])\
                .size().\
                reset_index(name='tot_syn')

            tot_out_df_proi = rel_syn_df\
                .groupby(['roi_pre','bodyId_pre'])\
                .size()\
                .reset_index(name='tot_syn_per_roi')\
                .rename(columns={'roi_pre':'roi'})

            syn_com_df = rel_syn_df\
                .groupby(['roi_pre', 'bodyId_pre'])[['x_post', 'y_post', 'z_post']]\
                .mean()\
                .reset_index()\
                .rename(columns={'roi_pre':'roi'})

            # removing None pre_type cells
            criteria2 = rel_syn_df['type_post'].notnull()
            rel_syn_df = rel_syn_df[criteria2]

            soma_loc_df = rel_syn_df\
                .groupby('bodyId_pre')['somaLocation_pre']\
                .first()

            tot_pre_post_syn_df = rel_syn_df\
                .groupby(['bodyId_pre', 'bodyId_post', 'type_pre', 'type_post','roi_pre'])\
                .size()\
                .reset_index(name='syn_count')\
                .rename(columns={'roi_pre':'roi'})
            tot_pre_post_syn_df = tot_pre_post_syn_df\
                .merge(soma_loc_df, on='bodyId_pre')
            tot_pre_post_syn_df = tot_pre_post_syn_df\
                .merge(syn_com_df, on=['roi', 'bodyId_pre'])
            tot_pre_post_syn_df['num_post_per_pre'] = tot_pre_post_syn_df\
                .groupby(['bodyId_pre', 'type_post'])['syn_count']\
                .transform('count')
            tot_pre_post_syn_df['tot_syn_per_post'] = tot_pre_post_syn_df\
                .groupby(['bodyId_pre', 'type_post'])['syn_count']\
                .transform('sum')
            tot_pre_post_syn_df['tot_syn_per_postNroi'] = tot_pre_post_syn_df\
                .groupby(['roi','bodyId_pre', 'type_post'], sort=False)['syn_count']\
                .transform('sum')
            tot_pre_post_syn_df['tot_conn_per_postNroi'] = tot_pre_post_syn_df\
                .groupby(['roi','bodyId_pre', 'type_post'], sort=False)['syn_count']\
                .transform('count')
            tot_pre_post_syn_df = tot_pre_post_syn_df\
                .merge(tot_out_df, 'left', on='bodyId_pre')
            tot_pre_post_syn_df = tot_pre_post_syn_df\
                .merge(tot_out_df_proi, 'left', on=['roi','bodyId_pre'])
            temp_col = tot_pre_post_syn_df['tot_syn_per_post']\
                .div(tot_pre_post_syn_df['tot_syn'].values)
            temp_col_proi = tot_pre_post_syn_df['tot_syn_per_postNroi']\
                .div(tot_pre_post_syn_df['tot_syn_per_roi'].values)
            temp_col_roi_tot = tot_pre_post_syn_df['tot_syn_per_postNroi']\
                .div(tot_pre_post_syn_df['tot_syn'].values)
            tot_pre_post_syn_df\
                .insert(
                    len(tot_pre_post_syn_df.columns)-1
                  , 'frac_tot_post'
                  , temp_col
                )
            tot_pre_post_syn_df\
                .insert(
                    len(tot_pre_post_syn_df.columns)-1
                  , 'frac_tot_post_by_roi'
                  , temp_col_proi
                )
            tot_pre_post_syn_df\
                .insert(
                    len(tot_pre_post_syn_df.columns)-1
                  , 'frac_tot_post_roi_tot'
                  , temp_col_roi_tot
                )
            temp_col2 = tot_pre_post_syn_df['syn_count']\
                .div(tot_pre_post_syn_df['tot_syn_per_post'].values)
            temp_loc = tot_pre_post_syn_df.columns.get_loc('tot_syn_per_post')+1
            tot_pre_post_syn_df.insert(temp_loc, 'frac_out_post_type', temp_col2)

            # adding stats on total number of cells per post, and total number of syn per post
            tot_pre_post_syn_df['tot_num_out'] = tot_pre_post_syn_df\
                .groupby(['bodyId_pre'])['syn_count']\
                .transform('count')
            tot_pre_post_syn_df['tot_num_out_by_roi'] = tot_pre_post_syn_df\
                .groupby(['roi', 'bodyId_pre'])['syn_count']\
                .transform('count')
            tot_pre_post_syn_df['tot_num_syn'] = tot_pre_post_syn_df\
                .groupby(['bodyId_pre'])['syn_count']\
                .transform('sum')
            tot_pre_post_syn_df['tot_num_syn_by_roi'] = tot_pre_post_syn_df\
                .groupby(['roi', 'bodyId_pre'])['syn_count']\
                .transform('sum')

            # sorting the df just so it would be readable if printed for a single pre_cell
            tot_pre_post_syn_df = tot_pre_post_syn_df\
                .sort_values(
                    by=['bodyId_pre', 'type_post', 'syn_count'],
                    ascending=[True, True, False])\
                .reset_index(drop=True)

            # adding to calculations of rank
            tot_pre_post_syn_df['rank_first'] = tot_pre_post_syn_df\
                .groupby(['bodyId_pre', 'type_post'])['syn_count']\
                .rank('first', ascending=False)
            tot_pre_post_syn_df['rank_dense'] = tot_pre_post_syn_df\
                .groupby(['bodyId_pre', 'type_post'])['syn_count']\
                .rank('dense', ascending=False)

            self.__output_neurons_w_stats = tot_pre_post_syn_df


    def __calc_input_celltypes_w_stats(self):
        """
        Uses the input_neurons_w_stats dataframe to calculate some stats at the
            input cell type level
        """
        conn_min_thresh = self.__input_conn_min_thresh

        const_df = self.__get_celltype_consistency('input')
        print(f"cell type connections calculated with {conn_min_thresh} threshold")
        ct_stats_df = self.__get_celltype_stats('input', conn_min_thresh=conn_min_thresh)

        self.__input_celltypes_w_stats = ct_stats_df\
            .merge(const_df, how='outer', on='type_pre')\
            .sort_values(by=['med_tot_syn'], ascending=False)\
            .reset_index()


    def __calc_output_celltypes_w_stats(self):
        """
        Uses the output_neurons_w_stats dataframe to calculate some stats at the output
          cell type level
        """

        conn_min_thresh = self.__output_conn_min_thresh

        const_df = self.__get_celltype_consistency('output')
        print(f"cell type connections calculated with {conn_min_thresh} threshold")
        ct_stats_df = self.__get_celltype_stats(
            'output'
          , conn_min_thresh=conn_min_thresh)

        self.__output_celltypes_w_stats = ct_stats_df\
            .merge(const_df, how='outer', on='type_post')\
            .sort_values(by=['med_tot_syn'], ascending=False)\
            .reset_index()


    def __get_celltype_consistency(
        self
      , conn_dir:str
    ) -> pd.DataFrame:
        assert conn_dir in ['input', 'output'], f"only input/output allowed, not {conn_dir}"

        if conn_dir == 'input':
            dir_type = 'pre'
            body_id_type = 'bodyId_post'
            conn_neu_df = self.get_input_neurons_w_stats()
        else:
            dir_type = 'post'
            body_id_type = 'bodyId_pre'
            conn_neu_df = self.get_output_neurons_w_stats()

        conn_type = 'type_' + dir_type
        frac_type = 'frac_tot_' + dir_type
        tot_syn_type = 'tot_syn_per_' + dir_type
        tot_conn_type = 'tot_conn_per_' + dir_type

        # calculating connection consistency

        all_types = conn_neu_df.groupby(['roi'])[conn_type].unique()
        temp = conn_neu_df\
            .groupby(['roi', body_id_type])[conn_type]\
            .apply(lambda x: set(x))
        type_counter = {}
        for temp_roi in temp.index.get_level_values(0).unique():
            type_counter[temp_roi] = np.array([
                temp\
                    .loc[temp_roi]\
                    .apply(lambda x: ntype in x)\
                    .sum() for ntype in all_types.loc[temp_roi]
            ])

        type_count_df = pd.DataFrame(data = all_types)\
            .reset_index()\
            .explode(conn_type, ignore_index=True)
        temp_count_df = pd.DataFrame(
                data={
                    'roi':type_counter.keys()
                  , 'type_counter':type_counter.values()
                }
            )\
            .explode('type_counter', ignore_index=True)
        type_count_df = type_count_df.join(temp_count_df['type_counter'])
        type_count_df['type_frac'] = \
            type_count_df['type_counter'] / temp.index.get_level_values(1).nunique()

        # reducing it to by type connections (to not over count)
        crit = conn_neu_df['rank_dense'] == 1
        rank1_df = conn_neu_df[crit]

        med_df_nondiv = rank1_df\
            .groupby(conn_type)[frac_type]\
            .median()\
            .reset_index()
        med_df = rank1_df\
            .groupby(['roi', conn_type])[frac_type + '_roi_tot']\
            .median()\
            .reset_index()
        med_syn_df = rank1_df\
            .groupby(['roi', conn_type])[tot_syn_type + 'Nroi']\
            .median()\
            .reset_index()
        med_conn_df = rank1_df\
            .groupby(['roi', conn_type])[tot_conn_type + 'Nroi']\
            .median()\
            .reset_index()

        type_count_df = type_count_df\
            .sort_values(by='type_frac', ascending=False)\
            .reset_index(drop=True)

        type_count_df = type_count_df\
            .merge(med_df_nondiv, on=conn_type)
        type_count_df = type_count_df\
            .merge(med_df, on=['roi', conn_type])
        type_count_df = type_count_df\
            .merge(med_syn_df, on=['roi', conn_type])
        type_count_df = type_count_df\
            .merge(med_conn_df, on=['roi', conn_type])

        return type_count_df


    def __get_celltype_stats(
        self,
        conn_dir:str,
        conn_min_thresh:int=1
    ) -> pd.DataFrame:
        """
        Internal function for calculating celltype statistics
        (number of celltype neurons per target and number of syn from celltype per target)
        """
        assert conn_dir in ['input', 'output'], f"only input/output allowed, not {conn_dir}"

        if conn_dir == 'input':
            conn_type = 'type_pre'
            tar_name = 'target'
            tar_body_id = 'bodyId_post'
            conn_neu_df = self.get_input_neurons_w_stats()
        else:
            conn_type = 'type_post'
            tar_name = 'source'
            tar_body_id = 'bodyId_pre'
            conn_neu_df = self.get_output_neurons_w_stats()

        if conn_neu_df is None:
            raise ValueError('neurons_w_stats df does exist run calc_XXX_neurons_w_stats first')

        rel_col_mame = 'syn_count'
        temp_df = conn_neu_df[conn_neu_df[rel_col_mame] >= conn_min_thresh].copy()

        count_name = 'tot_conn'
        sum_name = 'tot_syn'

        ct_num_df = temp_df\
            .groupby([tar_body_id, conn_type])[rel_col_mame]\
            .count()\
            .reset_index(drop=False, name=count_name)
        ct_syn_df = temp_df\
            .groupby([tar_body_id, conn_type])[rel_col_mame]\
            .sum()\
            .reset_index(drop=False, name=sum_name)

        cnt_df1 = ct_num_df\
            .groupby([conn_type])[count_name]\
            .mean()\
            .reset_index(name = 'mean_' + count_name)
        cnt_df2 = ct_num_df\
            .groupby([conn_type])[count_name]\
            .median()\
            .reset_index(name = 'med_' + count_name)
        cnt_df3 = ct_num_df\
            .groupby([conn_type])[count_name]\
            .std()\
            .reset_index(name='std_' + count_name)

        syn_df1 = ct_syn_df\
            .groupby([conn_type])[sum_name]\
            .mean()\
            .reset_index(name = 'mean_' + sum_name)
        syn_df2 = ct_syn_df\
            .groupby([conn_type])[sum_name]\
            .median()\
            .reset_index(name = 'med_' + sum_name)
        syn_df3 = ct_syn_df\
            .groupby([conn_type])[sum_name]\
            .std()\
            .reset_index(name='std_' + sum_name)

        dfs = [cnt_df1,cnt_df2, cnt_df3, syn_df1, syn_df2, syn_df3]

        comb_dfs = reduce(lambda left,right: pd.merge(left,right,how='outer', on=conn_type), dfs)
        tar_col = [self.__target_celltype] * comb_dfs.shape[0]
        comb_dfs.insert(0, tar_name, tar_col)

        return comb_dfs