from abc import ABC

class HexPlotConfig(ABC):

    def __init__(self):
        pass

    @property
    def style(self):
        style = {
            "font_type": "arial",
            "markerlinecolor": "rgba(0,0,0,0)", #transparent
            "linecolor": "black",
        }
        return style

    @property
    def sizing(self):
        sizing = {
            "fig_width": 750,  # units = mm, max 180
            "fig_height": 210,  # units = mm, max 170
            "fig_margin": 0,
            "fsize_ticks_pt": 35,
            "fsize_title_pt": 35,
            "markersize": 21,
            "ticklen": 15,
            "tickwidth": 5,
            "axislinewidth": 4,
            "markerlinewidth": 0.9,
            "cbar_thickness": 30,
            "cbar_len": 0.75,
        }
        return sizing