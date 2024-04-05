from abc import ABC

from .plot_slicer import PlotSlicer

class PlotNeuron(ABC):

    """
    Representation of the "Neuron" inside the configuration.
    """

    def __init__(
        self
      , name:str
      , neuron_dict:dict
    ):
        self.__name = name

        if tmp_bid := neuron_dict.get("body_ids"):
            if isinstance(tmp_bid, float):
                tmp_bid = int(tmp_bid)
            if isinstance(tmp_bid, int):
                tmp_bid = [tmp_bid]
            assert isinstance(tmp_bid, list), "body_ids must be number or list of numbers"
            self.__bids = tmp_bid
        if tmp_col := neuron_dict.get("body_color"):
            if isinstance(tmp_col, list) and isinstance(tmp_col[0], (int, float)):
                tmp_col = [tmp_col]
            for iter_col in tmp_col:
                assert len(iter_col)==4, "colors gotta have 4 numbers"
                for num in iter_col:
                    assert 0<=num<=1, "color values have to be 0…1"
            while len(tmp_col) < len(self.__bids):
                tmp_col.append(tmp_col[-1])
            self.__colors = tmp_col
        self.__slicer = []
        if tmp_slc := neuron_dict.get("slicer"):
            if isinstance(tmp_slc, dict):
                tmp_slc = [tmp_slc]
            for slc in tmp_slc:
                self.__slicer.append(PlotSlicer(slc))



    @property
    def name(self):
        """
        get the name

        Returns
        -------
        name : str
            name of the "neuron"
        """
        return self.__name

    @property
    def bids(self) -> list:
        """
        List of body IDs for this "neuron"

        Returns
        -------
        bids : list[int]
            list of body IDs
        """
        return self.__bids

    @property
    def colors(self) -> list:
        """
        Returns the color for this neuron

        Returns
        -------
        color : list[float]
            4-item color list (R,G,B,A)
        """
        return self.__colors

    @property
    def slicers(self) -> list:
        """
        Return a list of slicers, if they exist in the config

        Returns
        -------
        slicer : list[PlotSlicer]
            A list of slicers
        """
        return self.__slicer
