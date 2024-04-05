from abc import ABC
import numpy as np

class PlotSlicer(ABC):

    """
    Representation of the slicer
    """

    def __init__(
        self
      , slice_dict:dict
    ):

        self.__roi = None
        self.__keep = "intersection"
        self.__apply = True

        if tmp_roi := slice_dict.get("roi"):
            if isinstance(tmp_roi, str):
                tmp_roi = [tmp_roi]
            assert isinstance(tmp_roi, list),\
                "ROI has to be a list of (existing) brain regions"
            self.__roi = tmp_roi
        if tmp_loc := slice_dict.get("location"):
            assert len(tmp_loc)==3, "location needs 3 coordinates"
            self.__location = tmp_loc
        if tmp_rot := slice_dict.get("rotation"):
            assert len(tmp_rot)==3, "rotation needs 3 angles"
            self.__rotation = np.radians(tmp_rot).tolist()
        if tmp_thick := slice_dict.get("thickness"):
            if isinstance(tmp_thick, (int, float)):
                tmp_thick = [tmp_thick]
            assert len(tmp_thick) ==1, "only 1 thickness allowed"
            self.__box = [80000, 80000, tmp_thick[0]]
        if tmp_siz := slice_dict.get("size"):
            assert not tmp_thick, "can't use thickness and size"
            assert len(tmp_siz)==3, "size needs 3 coordinates"
            self.__box = tmp_siz
        if tmp_keep := slice_dict.get("keep"):
            assert tmp_keep in ['intersection', 'difference'],\
                f"only difference or intersection allowed, not {tmp_keep}"
            self.__keep = tmp_keep
        tmp_apply = slice_dict.get("apply")
        if isinstance(tmp_apply, bool):
            self.__apply = tmp_apply


    @property
    def is_named(self) -> bool:
        """
        Is it a named slicer or a box?

        Returns
        -------
        is_named : bool
            True if named slice (e.g. 'ME(R)'), False if box
        """
        if self.__roi:
            return True
        return False

    @property
    def roi(self) -> list:
        """
        If the slice is 'named', this returns the name of the slicer

        Returns
        -------
        name : list[str]
            The name of an ROI to be used as a slicer
        """
        return self.__roi
    
    @property
    def location(self) -> list:
        """
        For box slicers, this is the location (x,y,z)

        Returns
        -------
        location : list[float]
            location of the box slicer
        """
        return self.__location

    @property
    def rotation(self) -> list:
        """
        For box slixers, this is the rotation

        Returns
        -------
        rotation : list[float]
            Rotation in degrees
        """
        return self.__rotation

    @property
    def box(self) -> list:
        """
        Size of the box slicer

        Returns
        -------
        box : list[float]
            Size of the box slicer
        """
        return self.__box

    @property
    def keep(self) -> str:
        """
        Define what to keep

        Returns
        -------
        keep : {'intersection', 'difference'}
            Either keep the intersection between object (neuron) and slicer, 
            or everything outside of the intersection
        """
        return self.__keep

    @property
    def apply(self) -> bool:
        """
        True, if slicers should be applied (non-destructive modifiers in blender)

        Returns
        -------
        apply : bool
            If true, then don't apply the modifier
        """
        return self.__apply