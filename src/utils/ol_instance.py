from abc import ABC
import re
from madvisc.utils.ol_types import OLTypes
from madvisc.utils.helper import slugify

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
        'Optic Neuropil Intrinsic Neurons'.

        Returns
        -------
        main_group_name : str
            long name of the main group
        """
        full_group_names = {
            'OL_intrinsic': 'Optic Neuropil Intrinsic Neurons'
          , 'OL_connecting': 'Optic Neuropil Connecting Neurons'
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

        Returns
        -------
        sample_rate : float
            downsampling rate for navis

        TODO: move this to a `/params/*` file
        """
        rtn = 0.08
        if self.name in [
            'Cm31a_R', 'Cm31b_R', 'Cm35_R', 'DNp27_L', 'DNp27_R', 'LPi2b_R', 'LPi4b_R'
          , 'LT1b_R', 'LT33_L', 'LT56_R', 'LT79_R', 'Li33_R', 'Li38_L', 'LoVCLo3_L'
          , 'LoVCLo3_R', 'MeVC11_L', 'MeVC1_L', 'MeVC23_R', 'MeVC25_R', 'MeVPOL1_L'
          , 'MeVPOL1_R', 'Mi19_R', 'OA-AL2i1_R', 'Pm12_R']:
            rtn = 0.02
        elif self.name in [
            '5-HTPMPV03_L', '5-HTPMPV03_R', 'Cm34_R', 'DCH_L', 'DNp30_L', 'DNp30_R'
          , 'DNpe053_L', 'DNpe053_R', 'H2_R', 'LPi12_R', 'LT11_R', 'LT58_R', 'Li16_R', 'Li32_R'
          , 'LoVC16_R', 'MeVCMe1_L', 'MeVCMe1_R', 'MeVPLp1_L', 'MeVPLp1_R', 'OA-AL2i2_R'
          , 'OA-AL2i3_R', 'OLVC5_R', 'Pm11_R', 'Pm13_R', 'VCH_L', 'aMe17a_R', 'aMe17e_R']:
            rtn = 0.01
        elif self.name in [
            'Am1_R', 'Li39_L', 'LPi21_R' ]:
            rtn = 0.005
        elif self.name in [
            'CT1_L']:
            rtn = 0.0007
        return rtn
