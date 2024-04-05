from abc import ABC
import re
from utils.ol_types import OLTypes
from utils.helper import slugify

class OLInstance(ABC):

    """
    Helper class to simplify access to properties of a cell instance.

    Parameters
    ----------
    instance : str
        name of an instance
    olt : OLTypes
        If an OLTypes object is provided at initialization, it improves performance
        for access to some of the properties.
    """

    def __init__(
        self
      , instance:str=None
      , olt:OLTypes=None
    ):
        self.__name_html = None

        self.__instance = instance
        self.__olt = olt


    @property
    def name(self) -> str:
        """
        Name of the instance

        Returns
        -------
        name : str
            Instance name, for example Mi1_R
        """
        return self.__instance

    @property
    def name_html(self) -> str:
        """
        HTML representation of the instance. It makes the hemisphere smaller (and gray),
        adds a mouse over title for it.

        Returns
        -------
        name_html : str
            HTML representation of the instance
        """
        if self.__name_html is None:
            inst_rgx = re.compile('(.*)_([LR])$')
            inst_mtc = inst_rgx.match(self.__instance)
            if inst_mtc:
                abbrv = {
                    'L': 'Cell body in left hemisphere'
                  , 'R': 'Cell body in right hemisphere'
                }
                self.__name_html = f'{inst_mtc.group(1)}&#8239;'\
                    f'<span class="txt_hemisphere" title="{abbrv[inst_mtc.group(2)]}">({inst_mtc.group(2)})</span>'
        return self.__name_html

    @property
    def slug(self) -> str:
        """
        Slugified name. Useful for file names

        Returns
        -------
        slug : str
            A files system safe representation of the instance.
        """
        return slugify(self.name, to_lower=False)

    @property
    def type(self) -> str:
        """
        Type name associated with the instance

        Returns
        -------
        type : str
            cell type name
        """
        return self.name[:-2]

    @property
    def main_group(self) -> str:
        """
        Get the main group the Instance belongs to.

        Returns
        -------
        main_group : str
            one of ['OL_intrinsic', 'OL_connecting', 'VPN', 'VCN', 'other']
        """
        return self.olt.get_main_group(type_str=self.type)

    @property
    def main_group_name(self) -> str:
        """
        Return a readable name of the main group. For example, 'OL_intrinsic' becomes
        'Optic Lobe Intrinsic Neurons'.

        Returns
        -------
        main_group_name : str
            long name of the main group
        """
        full_group_names = {
            'OL_intrinsic': 'Optic Lobe Intrinsic Neurons'
          , 'OL_connecting': 'Optic Lobe Connecting Neurons'
          , 'VPN': 'Visual Projection Neurons'
          , 'VCN': 'Visual Centrifugal Neurons'
          , 'other': 'Other'
        }
        return full_group_names[self.main_group]

    @property
    def olt(self) -> OLTypes:
        """
        Get an OLTypes object. Mostly used internally.

        Returns
        -------
        olt : OLTypes
            OLTypes object
        """
        if self.__olt is None:
            self.__olt = OLTypes()
        return self.__olt

    @property
    def resample_precision(self) -> float:
        """
        Instance specific resample rate based on the file sizes for the dynamic plots. Larger 
        neurons are resampled at a worse rate.

        TODO: move this to a `/params/*` file
        """
        rtn = 0.2
        if self.name in [ # 0.2 resampling >= 75MB
            'OA-AL2i1_R', 'DNp27_L', 'Li32_R', 'MeVC11_L', 'MeVPOL1_L', 'MeVPOL1_R', 'Li33_R'
          , 'Pm12_R', 'Li38_L', 'Cm34_R', 'Cm31b_R', 'LPi4b_R', 'LoVCLo3_L', 'MeVC25_R', 'LT33_L'
          , 'LoVCLo3_R', 'LT56_R', 'MeVC1_L']:
          rtn = 0.05
        elif self.name in [
            'MeVCMe1_R', 'MeVCMe1_L', 'LPi12_R', 'CT1_L', 'DCH_L', 'DNpe053_L', 'DNpe053_R'
          , 'LoVC16_R', 'OA-AL2i2_R',  'DNp30_R', 'DNp30_L', 'VCH_L', 'OLVC5_R', 'aMe17e_R'
          , 'H2_R', 'LT11_R', 'MeVPLp1_R', 'Pm11_R', 'OA-AL2i3_R', 'LT58_R', 'Li16_R'
          , 'Pm13_R', 'MeVPLp1_L']:
          rtn = 0.1
        return rtn
