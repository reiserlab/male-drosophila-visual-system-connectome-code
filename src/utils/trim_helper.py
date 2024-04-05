from abc import ABC

from utils.column_features_functions import get_trim_df_all_rois

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
        trim_df, size_df, size_df_raw = get_trim_df_all_rois(
            cell_instance=self.__instance
        )
        self.__trim = trim_df
        self.__size = size_df
        self.__size_raw = size_df_raw