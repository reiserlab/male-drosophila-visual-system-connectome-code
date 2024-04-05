""" Neuron Bag class """
from abc import ABC
import warnings

import pandas as pd

from neuprint import fetch_neurons, fetch_custom, NeuronCriteria as NC

from utils.ol_types import OLTypes
from utils.ol_neuron import OLNeuron

class NeuronBag(ABC):
    """
    NeuronBag is a collection of Neurons defined by their cell type.
    """

    def __init__(
        self
      , cell_type:str | list[str]=None
      , cell_instance:str | list[str]=None
      , rois:str | list[str]=['ME(R)', 'LO(R)', 'LOP(R)', 'AME(R)', 'LA(R)']
      , side:str='R-dominant'
    ):
        """
        Initialize the bag of neurons.

        Parameters
        ----------
        cell_type : str
            name of the cell type, e.g. 'LC4'. Either cell_type OR cell_instance are allowed, not
            both.
        cell_instance : str
            name of the cell instance, e.g. 'LC4_R'. Either cell_type OR cell_instance are allowed,
            not both.
        rois : str | list[str], default=['ME(R)', 'LO(R)', 'LOP(R)', 'AME(R)', 'LA(R)']
            Only get neurons that innervate the listed ROI(s).
        side : str, default='R-dominant'
            options include, 'R', 'L', 'R-dominant' or 'both'.
            If cell_instance is given, this must be set to None. For cell_type, it defines what
            instances of the type to use.
            'R' means all neurons that have their cellbody on the right side, 'L' means that their
            cellbody is on the left side, 'R-dominant' chooses the neurons that have their 
            'dominant features' in the right hemisphere, and 'both' means to get both sides 
            (if available).
            For most analysis that works on one side, the 'R-dominant' is probably the best choice.
            There will be a 'L-dominant' once the other side is proof-read. 'both' returns the
            types that are present on either side and counts their total. If you know what you are
            doing and there is a reason to diverge, you can choose 'R' or 'L'.
        """

        if side:
            assert side in ['L', 'R', 'R-dominant', 'both'], f"side can only be 'L', 'R', 'R-dominant' or 'both' not {side}"
        self.__is_ordered = False

        if not cell_type and not cell_instance:
            warnings.warn("This type of bag is not yet supported.")
            self._bids = []
            return

        assert cell_type or cell_instance,\
            "Only support type or instance, not both."

        if cell_instance:
            assert side is None,\
                f"If instance is used, side has to be None, not '{side}'"

        if cell_type:
            my_nc = NC(type=cell_type, rois=rois, roi_req='any')
        elif cell_instance:
            my_nc = NC(instance=cell_instance, rois=rois, roi_req='any')

        n_df, _ = fetch_neurons(my_nc)

        if side:
            match side:
                case "L":
                    n_df = n_df[n_df.loc[:,'instance'].str.match(r'.*_L$')]
                case "R":
                    n_df = n_df[n_df.loc[:,'instance'].str.match(r'.*_R$')]
                case "R-dominant":
                    n_instances = n_df['instance'].nunique()
                    if n_instances > 1:
                        # if both L and R instances exist, choose the R instance
                        n_df = n_df[n_df.loc[:,'instance'].str.match(r'.*_R$')]
                case "both":
                    n_df = n_df.copy()

        self.__cell_types = n_df['type'].unique().tolist()
        self._bids=n_df['bodyId'].to_numpy()
        if len(self._bids) == 0:
            warnings.warn(f"No neurons found for cell type '{cell_type}'")


    def sort_by_distance_to_hex(
        self
      , neuropil:str
      , hex1_id:int
      , hex2_id:int
    ) -> None:
        """
        Sort the neurons in the bag by distance from the specified column

        Parameters
        ----------
        neuropil: str
            neuropil, for example 'ME(R)'
        hex1_id : int
            hex1 ID of column
        hex2_id : int
            hex2 ID of column
        """
        self._bids = self.__find_neurons_close_to_hex(
            neuropil
          , hex1_id=hex1_id
          , hex2_id=hex2_id
        )
        self.__is_ordered = True


    def sort_by_distance_to_star(self):
        """
        Sort the neurons in the bag by the distance from the star neuron. This
        assumes, that all neurons in the bag are of the same type.
        """
        self._bids = self.__find_neurons_close_to_star()
        self.__is_ordered = True


    def get_body_ids(self
      , cell_count=10
    ) -> list[int]:
        """
        get a list of body IDs.

        If you want the whole list, you could use size attribute by calling
        `bag.get_body_ids(cell_count=bag.size())`
        """
        return self._bids[:cell_count]


    def get_hex_ids(self) -> pd.DataFrame:
        """
        return hex ID for all neurons in the bag.

        Note: this relies on the `NeuronBag.get_hex_id()` method, which is currently
            not very precise.

        Returns
        -------
        ret : pd.DataFrame
            bodyID : int
                body ID of neuron
            ROI : str
                neuropil
            hex1 : int
                hex1_id for neuron
            hex2 : int
                hex2_id for neuron
        """
        ret = None
        for bid in self._bids:
            neuron = OLNeuron(body_id=bid)
            nhx = neuron.get_hex_id()
            nhx['bodyID'] = bid
            ret = pd.concat([ret, nhx])
        return ret


    @property
    def is_sorted(self)->bool:
        """
        Check if bag is sorted.
        """
        return self.__is_ordered

    @property
    def hemispheres(self)->list[str]:
        return self.__hemisphere

    @property
    def size(self)->int:
        """
        The size of the bag (how many neurons are in there).

        Use it as `my_neuron_bag.size` to find out how big your bag is (not `my_neuron_bag.size()`)
        """
        return len(self._bids)


    @property
    def first_item(self) -> int:
        if self.size>0:
            return self._bids[0]
        return None


    def __find_neurons_close_to_hex(
        self
      , neuropil:str
      , hex1_id:int
      , hex2_id:int
    ) -> list[int]:
        if len(self._bids) == 0:
            return self._bids

        hemisphere = neuropil[-2]
        assert hemisphere.upper() in ['L', 'R'], \
            "Make sure to provide the neuropil in the format 'ME(R)' or 'LO(R)'…"

        all_neurons = pd.DataFrame()

        for bid in self._bids:
            oln = OLNeuron(bid)
            tmp_l = oln.get_roi_hex_id(roi_str=neuropil)
            tmp_l['bid'] = bid
            all_neurons = pd.concat([all_neurons, tmp_l])

        all_neurons['hex_distance'] = (
            abs(hex1_id - all_neurons['hex1_id'])
          + abs(hex1_id + hex2_id - all_neurons['hex1_id'] - all_neurons['hex2_id'])
          + abs(hex2_id - all_neurons['hex2_id'])
        ) // 2
        return all_neurons.sort_values('hex_distance')['bid'].to_numpy()


    def __find_neurons_close_to_star(
        self
    ) -> list[int]:
        ctype = self.__cell_types
        if isinstance(self.__cell_types, list) and len(self.__cell_types)>1:
            warnings.warn("bag cannot be sorted since it contains more than one type")
            return self._bids
        elif isinstance(self.__cell_types, list):
            ctype = self.__cell_types[0]
        oltype = OLTypes()
        star_id = oltype.get_star(ctype)
        if not star_id:
            warnings.warn(f"couldn't find a star for {ctype}.")
            return self._bids
        cql = f"""
            MATCH (star:Neuron)-[:Contains]->(:SynapseSet)-[:Contains]->(star_s:Synapse)
            WHERE star.bodyId = {star_id}
            WITH
                point({{
                    x: avg(star_s.location.x)
                  , y: avg(star_s.location.y)
                  , z: avg(star_s.location.z)}}) as star_com
            MATCH (n:Neuron)-[:Contains]->(:SynapseSet)-[:Contains]->(ns:Synapse)
            WHERE n.type='{ctype}'
            WITH n
              , star_com
              , point({{
                    x: avg(ns.location.x)
                  , y: avg(ns.location.y)
                  , z: avg(ns.location.z)}}) as np

            RETURN n.bodyId as bodyId
              , point.distance(star_com, np) as star_distance
            ORDER BY star_distance
            """
        named_df = fetch_custom(cql)
        new_bids = named_df['bodyId'].to_numpy()
        if len(self._bids) != len(new_bids):
            warnings.warn("The number of neurons changed by reordering them. Bug?")
        return new_bids
