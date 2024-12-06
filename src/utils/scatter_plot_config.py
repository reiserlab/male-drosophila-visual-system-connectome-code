from abc import ABC
from pathlib import Path
from dotenv import find_dotenv

class ScatterConfig(ABC):
    """
    Object to retrieve plot specifications for the coverage metric scatterplots.

    Parameters
    ----------
    roi_str : str
        region of interest for which to plot the coverage metric data
    """

    def __init__(
        self
      , roi_str:str
    ):
        assert isinstance(roi_str, str), "roi_str must be a string"
        assert (roi_str in ["ME(R)", "LO(R)", "LOP(R)"])\
          , f"roi_str must be 'ME(R)', 'LO(R)' or 'LOP(R)' not '{roi_str}' "
        self.__roi = roi_str


    @property
    def style(self) -> dict:
        """
        get the 'style' parameters dictionary

        Returns
        -------
        style : dict
            dictionary with scatter plot styling information
        """
        style = {
            'export_type': 'pdf',
            'font_type': 'arial',
            'markerlinecolor': 'black',
            'linecolor': 'black',
        }
        return style


    @property
    def sizing(self) -> dict:
        """
        get the 'sizing' parameters dictionary

        Returns
        -------
        sizing : dict
            dictionary with size information for plot
        """
        sizing = {
            'fig_width': 48
          , 'fig_height': 35
          , 'fig_margin': 0
          , 'fsize_ticks_pt': 5
          , 'fsize_title_pt': 5
          , 'markersize': 2.8
          , 'ticklen': 2
          , 'tickwidth': 0.7
          , 'axislinewidth': 0.65
          , 'markerlinewidth': 0.07
          , 'cbar_thickness': 2
          , 'cbar_length': 1.1
          , 'cbar_tick_length': 2.25
          , 'cbar_tick_width': 0.75
        }
        return sizing


    @property
    def plot_specs(self) -> dict:
        """
        get the 'plot_specs' parameters dictionary
        
        Returns
        -------
        specs : dict
            other plot specifications
        """

        plot_specs = {
            'save_path': Path(find_dotenv()).parent / 'results' / 'cov_compl' / 'scatterplots'
          , 'color_factor': 'coverage_factor_trim'
          , 'cmax': 5
          , 'cmin': 1
          , 'colorscale': 'red'
          , 'tickvals': [1, 2, 3, 4, 5]
          , 'ticktext': ['1', '2', '3', '4', '5']
          , 'cbar_title': 'coverage factor'
          , 'hover_template': ''
          , 'cbar_title_scaling': 1
        }
        return plot_specs


    @property
    def star_neurons(self) -> list[str]:
        """
        get a list of the 'star_neurons' to be highlighted in the scatterplot
        for the given region

        Returns
        -------
        celltype_highlights : list[str]
            list of cell types that should be highlighted
        """
        star_neurons = None
        if self.__roi == "ME(R)":
            star_neurons = ['Dm4_R', 'Dm20_R', 'l-LNv_R', 'MeVP10_R']
        elif self.__roi == "LO(R)":
            star_neurons = ['T2_R', 'Tm2_R', 'MeVPLo1_L', 'LC40_R']
        elif self.__roi == "LOP(R)":
            star_neurons = ['LPLC2_R', 'LPLC4_R', 'OLVC3_L', 'LPT31_R']
        return star_neurons
