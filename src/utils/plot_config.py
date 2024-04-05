from abc import ABC

import json
import warnings
from pathlib import Path

from .plot_roi import PlotRoi
from .plot_neuron import PlotNeuron

class PlotConfig(ABC):
    """
    Simple interface to the config files used to store gallery views.

    Currently the config files should look like this (upper case are placeholders)

    ```json
    {
        "group" : {
            "group_name" : "NAME OF THE GROUP",
            "SOME NAME : {
                "body_ids": [1, 2, 3, 4, 5],
                "body_color:" : (0.3, 0, 1, 3)
            }
        }
    }
    ```
    """

    def __init__(
        self
      , config_filename: str | Path
    ):
        """
        Initialize interface ot the config file.

        Parameters
        ----------
        config_filename : str | Path
            name of the config file.
        """

        assert isinstance(config_filename, (str, Path)), \
            f"config_filename must be str or Path, but it is {type(config_filename)}"
        config_fn = config_filename
        if isinstance(config_filename, str):
            config_fn = Path(config_filename)

        if not config_fn.is_file():
            warnings.warn(f"The plot config {config_fn} doesn't exist.")

        with open(config_fn, mode='r', encoding='utf-8') as read_file:
            self.__data = json.load(read_file)

        if not self.__data['plot']:
            warnings.warn(f"Invalid JSON file for PlotConfig at {config_fn}, 'plot' is missing")
        if not self.__data['plot']['group_name']:
            warnings.warn(f"Invalid JSON file for PlotConfig at {config_fn}, the plot's 'group_name' missing")


    @property
    def name(self) -> str:
        """
        The name of the plotting group

        Returns
        -------
        group_name : str
            name of the group from the config file.
        """
        return self.__data['plot']['group_name']


    @property
    def directory(self) -> str:
        """
        Directory specified in the config

        Returns
        -------
        directory : str
        """
        return self.__data['plot']['directory']


    @property
    def basename(self) -> str:
        """
        Base file name

        Returns
        -------
        name : str
            {plot type}.{name}
        """
        tpe = self.__data['plot']['plot_type']
        tpe = tpe.replace(" ", "_")
        nme = self.name
        nme = nme.replace(" ", "_")
        return f"{tpe}.{nme}"


    @property
    def bids(self) -> list[int]:
        """
        Get all the body IDs from the config file.

        Returns
        -------
        bids : list[int]
            list of body IDs
        """
        bids = []
        for _, values in self.__data['plot']['neuron_types'].items():
            if isinstance(values, dict) and values.get("body_ids"):
                bodyids = values.get("body_ids")
                if isinstance(bodyids, list):
                    bids.extend(bodyids)
                else:
                    bids.append(bodyids)
        bids = list(set(bids))
        return bids


    @property
    def bid_dict(self) -> dict:
        """
        Body ID and colors

        Returns
        -------
        bid : dict
            bid : tuple
                body ID + tuple
        """
        bid_dict = {}
        for _, values in self.__data['plot']['neuron_types'].items():
            if isinstance(values, dict) and values.get("body_ids"):
                bids = values.get("body_ids")
                clrs = values.get("body_color")
                if not isinstance(clrs[0], list):
                    clrs = [clrs]
                prev_clr = clrs[0]
                for idx, bid in enumerate(bids):
                    if idx + 1 > len(clrs):
                        clr = prev_clr
                    else:
                        clr = clrs[idx]
                    prev_clr = clr
                    bid_dict[bid] = clr
        return bid_dict


    @property
    def max_slice(self) -> float:
        """
        Find and return the thickest slice

        Returns
        -------
        max_slice : float
            Return the thickness of the thickest slice
        """
        max_slice = 0
        for neuron in self.neurons:
            for slc in neuron.slicers:
                if not slc.is_named:
                    max_slice = max(max_slice, slc.box[2])
        return max_slice


    @property
    def text_dict(self) -> dict:
        """
        Text dictionary

        Returns
        -------
        txt : dict
            plot title and names
        """
        txt_dict = {}
        for grp_name, values in self.__data['plot']['neuron_types'].items():
            color = values.get('body_color')
            has_text = values.get('text')
            if has_text:
                text = values.get('text').get('text_string')
                text_pos = values.get('text').get('paper_position')
                text_align = values.get('text').get('text_align', 'l')
                txt_dict[grp_name] = {
                    'text': text
                  , 'color': color
                  , 'pos': text_pos
                  , 'align': text_align
                }

        txt_dict['title'] = {
            'text': self.__data['plot']['title']
          , 'color': [0,0,0,1]
          , 'pos': [0.50, 0.02]
          , 'align': 'c'
        }
        
        for key, values in self.__data['plot'].items():
            if isinstance(values, dict) and key == 'slice_indicator':
                txt_dict['slice_indicator'] = {
                    'text': self.__data['plot']['slice_indicator']['text']
                  , 'color': [0.7,0.7,0.7,1]
                  , 'pos': [0.05, 0.78]
                  , 'align': 'c'
                }
        return txt_dict


    @property
    def scalebar(self) -> dict:
        """
        Reads the scalebar definition from the configuration file.

        Returns
        -------
        ret : dict
            scalebar definition with `type` and `length` keys.
        """
        ret = {}
        for key, values in self.__data['plot'].items():
            if isinstance(values, dict) and key == 'scalebar':
                sb_type = values.get('type')
                assert sb_type is None or sb_type in ['-', 'L'],\
                    f"Scalebar type must be '-' or 'L', not '{sb_type}'"
                if sb_type:
                    ret['type'] = sb_type
                sb_len = int(values.get('length'))
                assert sb_len is None or isinstance(sb_len, (int, float)),\
                    f"scalebar length must be a number in µm, not {sb_len}"
                if sb_len:
                    ret['length'] = sb_len
                
                sb_loc = values.get('location')

                if sb_loc:
                    ret['location'] = sb_loc
                
                sb_txt_loc = values.get('text_location')

                if sb_txt_loc:
                    ret['text_location'] = sb_txt_loc              
        return ret


    @property
    def camera(self) -> dict:
        """
        Camera location

        Returns
        -------
        camera : dict
            camera with location + rotation
        """
        ret = None
        for key, values in self.__data['plot'].items():
            if isinstance(values, dict) and key == 'camera':
                ret = values
        return ret


    @property
    def rois(self) -> list:
        """
        Get a list of ROIs mentioned in the configuration

        Returns
        -------
        rois : list[PlotRoi]
            list of ROIs
        """
        roi_list = []
        for key, values in self.__data['plot']['rois'].items():
            roi_list.append(PlotRoi(key, values))
        return roi_list


    @property
    def neurons(self) -> list:
        """
        Get a list of neurons mentioned in the config

        Returns
        -------
        neuron_list : list[PlotNeuron]
            All neurons and neuron groups from the configuration.
        """
        neuron_list = []
        for grp_name, values in self.__data['plot']['neuron_types'].items():
            neuron_list.append(PlotNeuron(grp_name, values))
        return neuron_list
