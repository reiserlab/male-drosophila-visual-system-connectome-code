"""
Part of the Optic Lobe Connectome code release
"""
from abc import ABC
import warnings
from pathlib import Path
import re

import pandas as pd

from dotenv import find_dotenv

from queries.completeness import fetch_ol_types_and_instances
from utils.helper import num_expand

class OLTypes(ABC):

    """
    Class to access all neuron types in the optic lobe data set.

    Parameters
    ----------
    include_tbd : bool, default=True
        keep types ending on "TBD1" and "TBD_1"
    include_placeholder : bool, default=False
        keep types ending on "_unclear" and with a bracket ")"
    """

    def __init__(
        self
      , include_tbd:bool=True
      , include_placeholder:bool=False
    ):
        self.__column_list = [
            'type', 'hemisphere', 'instance', 'star_neuron'
          , 'main_groups', 'figure_group', 'figure_subgroup'
          , 'slice_width', 'fb_view', 'ol_view'
        ]
        self.__with_tbd = include_tbd
        self.__with_placeholder = include_placeholder
        self.__pfile = None


    def get_star(self, type_str:str=None, instance_str:str=None) -> int:
        """
        Get the star neuron for one specific neuron type. "Star Neurons" are hand curated
        neurons that are a good representation of the neuron type.

        Parmeters
        ---------
        type_str : str
            Name of the neuron type

        Returns
        -------
        star : int
            body ID of the "star" neuron.
        """
        assert type_str is None or instance_str is None,\
            "either use type or instance, not both"
        if type_str:
            my_star = self.__fl[self.__fl['type']==type_str]\
                .loc[:, 'star_neuron']\
                .to_numpy()
        elif instance_str:
            my_star = self.__fl[self.__fl['instance']==instance_str]\
                .loc[:, 'star_neuron']\
                .to_numpy()
        if len(my_star) == 0:
            return None
        if len(my_star)!=1:
            warnings.warn(f"unexpected number of star neurons for {type_str}")
        return my_star[0]

    def is_bilateral(self, type_str:str) -> bool:
        """
        Check if neurons exist in more than one hemisphere.

        Parameters
        ----------
        type_str : str
            Type of the neuron

        Returns
        -------
        is_bilateral : bool
            True if this type has instances in more than 1 hemisphere. False if not.
        """
        count = self.__fl[self.__fl['type']==type_str]['hemisphere'].nunique()
        if count == 0:
            warnings.warn(f"type {type_str} doesn't exist")
        if count >= 2:
            return True
        return False
    
    def get_main_group(self, type_str:str) -> bool:
        """
        Find and return the main group for the specified cell type

        Parameters
        ----------
        type_str : str
            type of the neuron

        Returns
        -------
        main_group : str
            One of [None, 'OL_intrinsic', 'OL_connecting', 'VCN', 'VPN', 'other']
        """
        m_r = self.__fl[self.__fl['type']==type_str]
        if len(m_r) == 0:
            return None
        main_groups = ['OL_intrinsic', 'OL_connecting', 'VCN', 'VPN', 'other']
        for m_g in main_groups:
            if m_r['main_groups'].values[0].startswith(m_g):
                return m_g
        return None

    def get_neuron_list(
        self
      , primary_classification:str|list[str]=None
      , side:str="R-dominant"
    ):
        """
        Get list of neurons

        Parameters
        ----------
        primary_classification : str, default=None
            Can be 'OL', 'OL_intrinsic', 'OL_connecting', 'VCN', 'VPN', 'other', 'non-OL'.
            None means no filter = all neurons
            'OL' is an alias to 'OL_intrinsic' and 'OL_connecting', 'non-OL' is an alias
            to 'VCN', 'VPN', and 'other'.
        side : {'R', 'L', 'R-dominant', 'both'}, default='R-dominant'
            'R' means all neurons that have their cellbody on the right side, 'L' means that their
            cellbody is on the left side, 'R-dominant' chooses the neurons that have their
            'dominant features' in the right hemisphere, and 'both' means to get both sides
            (if available).
        """
        if isinstance(primary_classification, str):
            primary_classification = [primary_classification]
        allowed_main_groups = ['OL_intrinsic', 'OL_connecting', 'VCN', 'VPN', 'other']
        allowed_classes = ['OL', 'non-OL'] + allowed_main_groups
        assert primary_classification is None \
            or set(primary_classification) < set(allowed_classes), \
            f"wrong primary classification {primary_classification}"
        assert side in ["R", "L", "R-dominant", "both"],\
            f"Unsupported side '{side}', only 'R', 'L', 'R-dominant' or 'both' are allowed"

        grp = pd.DataFrame()

        if primary_classification:
            if 'OL_intrinsic' in primary_classification and 'OL' in primary_classification:
                primary_classification.remove('OL_intrinsic')
            if 'OL_connecting' in primary_classification and 'OL' in primary_classification:
                primary_classification.remove('OL_connecting')
            if 'non-OL' in primary_classification:
                primary_classification.extend(['VCN', 'VPN', 'other'])
                primary_classification.remove('non-OL')
            primary_classification = list(set(primary_classification))

            for p_c in primary_classification:
                grp = pd.concat([grp, self.__fl[self.__fl['main_groups'].str.startswith(p_c)]])
        else:
            grp = self.__fl
        all_in = fetch_ol_types_and_instances(side=side)
        all_in['hemisphere'] = all_in['instance'].str[-1:]

        grp['main_groups'] = grp['main_groups']\
            .apply(
                lambda x: self.__maingroup_searcher(search_str=x, search_list=allowed_main_groups)
            )
        grp = grp\
            .set_index(['type', 'hemisphere'])\
            .drop('instance', axis=1)\
            .join(all_in.set_index(['type', 'hemisphere']), how='inner')\
            .reset_index()
        grp['tmp_sort_index'] = grp['instance'].apply(num_expand)
        grp = grp.sort_values(by='tmp_sort_index')
        grp = grp\
            .loc[:, self.__column_list]\
            .reset_index(drop=True)
        return grp


    @property
    def __fl(self) -> pd.DataFrame:
        if self.__pfile is None:
            self.__load_primary_from_file()
            self.__merge_stars()
        return self.__pfile


    def __load_primary_from_file(
        self
    ):
        primary_table_fn = Path(find_dotenv()).parent / 'params' / "Primary_cell_type_table.xlsx"
        required_columns = [
            'type', 'OL_or_OL_and_CB'
          , 'main_groups', 'notes_main_groups'
          , 'figure_group', 'figure_subgroup'
        ]
        if not primary_table_fn.is_file():
            self.__pfile = pd.DataFrame(columns=required_columns)
            return
        pt_df = pd.read_excel(primary_table_fn)
        if not set(required_columns).issubset(pt_df.columns):
            warnings.warn(
                "There are columns missing in the table, "
                f"ignoring the whole file {primary_table_fn}")

        if not self.__with_placeholder:
            pt_df = pt_df[~pt_df.loc[:, 'type'].str.endswith("_unclear")]
            pt_df = pt_df[~pt_df.loc[:, 'type'].str.endswith(")")]
            pt_df = pt_df[~pt_df.loc[:, 'type'].str.contains("Pm7_Li28")]

        if not self.__with_tbd:
            pt_df = pt_df[~pt_df.loc[:, 'type'].str.endswith("_TBD1")]
            pt_df = pt_df[~pt_df.loc[:, 'type'].str.endswith("_TBD_1")]

        self.__pfile = pt_df


    def __merge_stars(
        self
    ):
        params_p = Path(find_dotenv()).parent / 'params'
        star_list = params_p.glob('[!~.]*_stars.xlsx')
        all_stars = pd.DataFrame()
        for sl in star_list:
            #FIXME: find hemisphere based on existence of instance (or not)
            new_stars = pd.read_excel(sl)
            if 'instance' in new_stars.columns:
                # default values for empty fields
                new_stars['instance'] = new_stars['instance'].fillna('R')
                new_stars['hemisphere'] = new_stars['instance'].str[-1:]
            else:
                new_stars['hemisphere'] = 'R'
            all_stars = pd.concat([all_stars, new_stars])
        all_stars['star_neuron'] = all_stars['star_neuron'].astype('Int64')
        self.__pfile = self.__pfile.merge(all_stars, how='left', on='type')


    def __maingroup_searcher(
        self
      , search_str:str
      , search_list:list
    ) -> str:
        search_pattern = '|'.join(search_list)
        search_obj = re.search(search_pattern, search_str)
        if search_obj :
            return_str = search_str[search_obj.start(): search_obj.end()]
        else:
            return_str = 'NA'
            warnings.warn(f"I found an unknown main group {search_str}, calling it 'NA'")
        return return_str
