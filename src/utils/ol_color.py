""" part of utils """
from enum import Enum
from cmap import Colormap

class OL_COLOR(Enum):

    """
    Define frequently used colos. See `src/python-bootcamp/ol_colors_example.ipynb`
    for how to use it.
    """

    # optic lobe groups
    OL_TYPES = (
        'ol_types',
        ['#029e73', '#d55e00', '#0173b2', '#de8f05', '#cc78bc'] ,
        ['intrinsic', 'connecting', 'VPN', 'VCN', 'OL/CB other'],
        False
    )

    OL_DARK_TYPES = (
        'ol_dark_types',
        ['#01523c', '#893c00', '#014266', '#935f03', '#943883'] ,
        ['intrinsic', 'connecting', 'VPN', 'VCN', 'OL/CB other'],
        False
    )

    OL_LIGHT_TYPES = (
        'ol_light_types',
        ['#5dfed0', '#ffc599', '#85d4fe', '#fdc35e', '#e3b5db'] ,
        ['intrinsic', 'connecting', 'VPN', 'VCN', 'OL/CB other'],
        False
    )

    # optic lobe groups: sequence
    OL_IN_SEQ = (
        'ol_in_seq',
        ['#01523c', '#029e73', '#02dea0', '#5dfed0', '#d6fff3'] ,
        ['darkest', 'dark', 'mid', 'light', 'lightest'],
        True
    )

    OL_CONN_SEQ = (
        'ol_conn_seq',
        ['#893c00', '#d55e00', '#ff8b33', '#ffc599', '#ffe7d6'] ,
        ['darkest', 'dark', 'mid', 'light', 'lightest'],
        True
    )

    OL_VPN_SEQ = (
        'ol_vpn_seq',
        ['#014266', '#0173b2', '#20b0fe', '#85d4fe', '#d6f1ff'] ,
        ['darkest', 'dark', 'mid', 'light', 'lightest'],
        True
    )

    OL_VCN_SEQ = (
        'ol_vcn_seq',
        ['#4f3202', '#935f03', '#de8f05', '#fdc35e', '#fce7c5'] ,
        ['darkest', 'dark', 'mid', 'light', 'lightest'],
        True
    )

    OL_CB_OTHER_SEQ = (
        'ol_cb_other_seq',
        ['#592250', '#943883', '#cc78bc', '#e3b5db', '#f3e2f1'] ,
        ['darkest', 'dark', 'mid', 'light', 'lightest'],
        True
    )

    # optic lobe neuropils
    OL_NEUROPIL = (
        'ol_neuropil',
        ['#5E184D', '#BB4A5F', '#F16E65', '#598FCF', '#8EBD78'] ,
        ['ME', 'LO', 'LOP', 'AME', 'LA'],
        False
    )
    #neuropil layers-original colors
    OL_NEUROPIL_LAYERS= (
        'ol_neuropil',
        ['#EEEEEE', '#FAFAFA', '#BABABA', '#FAF7FA', '#F0E3EB', '#AC8BAF', '#FDF9FA', '#F8EAEC', '#E59DAD', '#FFFBFB', '#FFF0EF', '#FE9DAD'] ,
        ['gray_odd_layer', 'gray_even_layer', 'gray_outline', 'ME_odd_layer', 'ME_even_layer', 'ME_outline', 'LO_odd_layer', 'LO_even_layer', 'LO_outline', 'LOP_odd_layer', 'LOP_even_layer', 'LOP_outline'],
        False
    )

    #darker
    OL_NEUROPIL_DARKER= (
        'ol_neuropil',
        ['#D0D0D0', '#E5E5E5', '#646464', '#FAF7FA', '#F0E3EB', '#AC8BAF', '#FDF9FA', '#F8EAEC', '#E59DAD', '#FFFBFB', '#FFF0EF', '#FE9DAD'] ,
        ['gray_odd_layer', 'gray_even_layer', 'gray_outline', 'ME_odd_layer', 'ME_even_layer', 'ME_outline', 'LO_odd_layer', 'LO_even_layer', 'LO_outline', 'LOP_odd_layer', 'LOP_even_layer', 'LOP_outline'],
        False
    )

    #darker more contrast
    OL_NEUROPIL_HIGH_CONTRAST= (
        'ol_neuropil',
        ['#D0D0D0', '#FAFAFA', '#646464', '#FAF7FA', '#F0E3EB', '#AC8BAF', '#FDF9FA', '#F8EAEC', '#E59DAD', '#FFFBFB', '#FFF0EF', '#FE9DAD'] ,
        ['gray_odd_layer', 'gray_even_layer', 'gray_outline', 'ME_odd_layer', 'ME_even_layer', 'ME_outline', 'LO_odd_layer', 'LO_even_layer', 'LO_outline', 'LOP_odd_layer', 'LOP_even_layer', 'LOP_outline'],
        False
    )

    #just outline-no shading
    OL_NEUROPIL_NO_SHADING= (
        'ol_neuropil',
        ['#FFFFFF', '#FFFFFF', '#787878', '#FAF7FA', '#F0E3EB', '#AC8BAF', '#FDF9FA', '#F8EAEC', '#E59DAD', '#FFFBFB', '#FFF0EF', '#FE9DAD'] ,
        ['gray_odd_layer', 'gray_even_layer', 'gray_outline', 'ME_odd_layer', 'ME_even_layer', 'ME_outline', 'LO_odd_layer', 'LO_even_layer', 'LO_outline', 'LOP_odd_layer', 'LOP_even_layer', 'LOP_outline'],
        False
    )

    # optic lobe synapses
    OL_SYNAPSES = (
        'test',
        ['#FEC72B', '#06B1D2'] ,
        ['post', 'pre'],
        False
    )

    PALE_AND_YELLOW = (
        'test',
        ['#C28E1D', '#943883', '#000000'] ,
        ['yellow', 'pale', 'black'],
        False
    )
    SIX_MORE_COLORS = (
        'test',
        ['#C28E1D', '#943883', '#000000', '#2071E3', '#F16E65', '#694766'] ,
        ['yellow', 'pale', 'black', 'Red', 'Blue', 'lilac'],
        False
    )

    MY_COLOR = (
        'ol_neuropil',
        ['#030303', '#2071E3', '#F16E65', '#4F4F4F'] ,
        ['Black', 'Blue', 'Red', 'Gray'],
        False
    )

    MAGENTA_AND_GREEN = (
        'ol_neuropil',
        ['#D100D1', '#00D100'] ,
        ['Magenta', 'Green'],
        False
    )

    HEATMAP = (
        'ol_heatmap',
        ['#FFFFFF', '#FEE5D9', '#FCAE91', '#FB6A4A', '#DE2D26', '#A50F15'],
        ['0','1','2','3','4','5'],
        False
    )

    NT = (
        'ol_nt  ',
        ["#EE672D", "#09A64D", "#1F4695", "#000000", "#59626E", "#979DA5", "#C9D0D9"],
        ['ACh', 'Glu', 'GABA', 'His', 'Dop', 'OA', '5HT'],
         False
     )


    def hex(self, which):
        return self.hex[which]

    def __get_dict(self, colors, colornames):
        """ Helper function to create a dictionary of the color names """
        color_output = {}
        for idx, col in enumerate(colornames):
            color_output[col] = colors[idx]
        return color_output

    def __get_nrgb(self, colornames):
        rgbastr = [color.rgba_string for color in self.cmap.iter_colors()]
        color_output = {}
        for idx, col in enumerate(rgbastr):
            name = colornames[idx]
            color_output[name] = col
        return color_output

    def __init__(self, name, colors, colornames,  interpolation):
        self.cmap = Colormap(colors, name=name, interpolation=interpolation)
        self.rgb = self.cmap.to_plotly()
        self.nrgb = self.__get_nrgb(colornames)
        self.hex = [color.hex for color in self.cmap.iter_colors()]
        self.map = self.__get_dict(colors, colornames)

    @property
    def rgba(self):
        return [list(color) for color in self.cmap.iter_colors()]
