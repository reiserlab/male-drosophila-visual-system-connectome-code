from abc import ABC
import re

import numpy as np

class PlotRoi(ABC):

    def __init__(
        self
      , name:str
      , roi_dict:dict
    ):
        self.__name = name

        # Set defaults
        self.__outline = False
        self.__visible = True

        self.__slice = None
        self.__scale = 1.0

        self.__is_slice = False
        self.__is_flat = False
        
        # Parse dict
        fn = re.sub(r"[\(\)]", "_", name)
        fn = re.sub(r"[^\w\s-]", "", fn)
        self.__filename = f"roi.{fn}"

        if tmp_sl := roi_dict.get("slice"):
            assert len(tmp_sl)==3, "slice needs 3 coordinates"
            assert not self.__is_flat, "can only do slicing or flattening, not both"
            self.__slice = tmp_sl
            self.__is_slice = True

            if tmp_rot := roi_dict.get("rotation"):
                assert len(tmp_rot)==3, "rotation needs 3 coordinates"
                self.__rotation = np.radians(tmp_rot).tolist()
            
            assert roi_dict.get("size") or roi_dict.get("thickness"), "only size or thickness allowed, not both"
            self.__box = [1, 1, 1]
            if tmp_sz := roi_dict.get("size"):
                assert len(tmp_sz)==3, "size needs 3 lengths"
                self.__box = tmp_sz
            if tmp_thk := roi_dict.get("thickness"):
                assert tmp_thk > 0, "size must be >0"
                self.__box = [40000, 40000, tmp_thk]
        
        if tmp_fl := roi_dict.get("flat"):
            assert len(tmp_fl)==3, "flatten needs 3 coordinates"
            assert not self.__is_slice, "can only do flatting or slicing, not both"
            self.__slice = tmp_fl
            self.__is_flat = True

            if tmp_rot := roi_dict.get("rotation"):
                assert len(tmp_rot)==3, "rotation needs 3 coordinates"
                self.__rotation = np.radians(tmp_rot).tolist()
        
        if tmp_col := roi_dict.get("color"):
            assert len(tmp_col)==4, "color needs 4 values"
            self.__color = tmp_col
        else:
            self.__color = [0, 0, 0, 1]

        if tmp_scale := roi_dict.get("scale"):
            
            self.__scale = tmp_scale
        
        tmp_out = roi_dict.get("outline")
        if tmp_out:
            assert isinstance(tmp_out, (bool, list)), "outline needs to be true/false or a list"
            if isinstance(tmp_out, list) and len(tmp_out)==4:
                self.__outline = tmp_out
            elif isinstance(tmp_out, bool) and tmp_out is True:
                self.__outline = [0.0, 0.0, 0.0, 1.0]
            else:
                self.__outline = False
        tmp_vis = roi_dict.get("visible")
        if isinstance(tmp_vis, bool):
            self.__visible = tmp_vis


    @property
    def location(self):
        if self.__slice:
            return self.__slice
        return None

    @property
    def rotation(self):
        if self.__slice:
            return self.__rotation
        return None

    @property
    def box(self):
        if self.__slice:
            return self.__box
        return None

    @property
    def filename(self):
        return self.__filename

    @property
    def oname(self):
        return self.__name

    @property
    def name(self):
        return self.__name

    @property
    def color(self):
        return self.__color

    @property
    def scale(self):
        return self.__scale

    @property
    def has_outline(self):
        if isinstance(self.__outline, list):
            return True
        return False
    
    @property
    def outline_color(self):
        if self.has_outline:
            return self.__outline

    @property
    def is_visible(self):
        return self.__visible

    @property
    def is_sliced(self):
        return self.__is_slice

    @property
    def is_flat(self):
        return self.__is_flat
