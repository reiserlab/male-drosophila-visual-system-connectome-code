"""
Helper script to identify lists for gallery / movie / summary figures.
"""

from pathlib import Path
import pandas as pd
from dotenv import find_dotenv

from utils.ol_color import OL_COLOR

def get_one_off_params():
    """
    wrapper function to get a list of parameters in a dictionary.

    Returns:
    --------
    plots : dict
        Plotting parameters for one-off figures.
    """
    plots = {
        "Fig2_fischbach_style_plot_Kit_order": {
            'columnar_list': [
                'Mi1', 'Mi4', 'Mi9', 'C2', 'C3', 'L1', 'L2', 'L3', 'L5', 'T1'
              , 'Tm1', 'Tm2', 'Tm4', 'Tm9', 'Tm20'
            ]
          , 'list_bids_to_plot': []
          , 'hex_assign': [24, 12, -1, 1]
          , 'text_placement': [0.78, 0.95, 0,-0.06]
          , 'replace': {'name': 'Mi1', 'bid': [38620]}
          , 'directory': "fig_2"
          , 'body_color_list':
                OL_COLOR.OL_TYPES.rgba + OL_COLOR.OL_DARK_TYPES.rgba
              + OL_COLOR.OL_LIGHT_TYPES.rgba
          , 'body_color_order': [0, 6, 2, 13, 4, 5, 1, 7, 3, 9, 12, 10]
          , 'color_by':'type'
          , 'n_vis':{}
          , 'view': 'Equator_slice'
        }
      , "Fig2_fischbach_style_plot": {
            'columnar_list': [
                'L1', 'L2', 'L3', 'L5', 'Mi1', 'Mi4', 'Mi9', 'C2', 'C3'
              , 'Tm1', 'Tm2', 'Tm4', 'Tm9', 'Tm20', 'T1'
            ]
          , 'list_bids_to_plot':[]
          , 'hex_assign':[24, 12, -1, 1]
          , 'text_placement':[0.78, 0.95, 0, -0.06]
          , 'replace':{}
          , 'directory': "fig_2"
          , 'body_color_list':
                OL_COLOR.OL_TYPES.rgba + OL_COLOR.OL_DARK_TYPES.rgba
              + OL_COLOR.OL_LIGHT_TYPES.rgba
          , 'body_color_order': [0, 6, 2, 13, 4, 5, 1, 7, 3, 9, 12, 10]
          , 'color_by':'type'
          , 'n_vis':{}
          , 'view': 'Equator_slice'
        }
      , "Fig2_fischbach_style_plot_inverted": {
            'columnar_list': [
                'T1', 'Tm20', 'Tm9', 'Tm4', 'Tm2', 'Tm1', 'C3', 'C2'
              , 'Mi9', 'Mi4', 'Mi1', 'L5', 'L3', 'L2', 'L1'
            ]
          , 'list_bids_to_plot':[]
          , 'hex_assign':[24, 12,-1, 1]
          , 'text_placement':[0.78, 0.95, 0,-0.06]
          , 'replace':{}
          , 'directory': "fig_2"
          , 'body_color_list':
                OL_COLOR.OL_TYPES.rgba + OL_COLOR.OL_DARK_TYPES.rgba
              + OL_COLOR.OL_LIGHT_TYPES.rgba
          , 'body_color_order': [0, 6, 2, 13, 4, 5, 1, 7, 3, 9, 12, 10]
          , 'color_by':'type'
          , 'n_vis':{}
          , 'view': 'Equator_slice'
        }
      , "Fig2_Mi9_Mi4": {
            'columnar_list': ['Mi9', 'Mi4']
          , 'list_bids_to_plot':[]
          , 'hex_assign': [17, 19,-1, 1]
          , 'text_placement': [0.78, 0.95, 0,-0.06]
          , 'replace': {}
          , 'directory': "fig_2"
          , 'body_color_list': OL_COLOR.MAGENTA_AND_GREEN.rgba
          , 'body_color_order': [1, 0]
          , 'color_by':'type'
          , 'n_vis': {}
          , 'view': 'Equator_slice'
        }
      , "Fig2S_Tm5a_Tm5b_Tm29": {
            'columnar_list': ['Tm5a', 'Tm5b', 'Tm29']
          , 'list_bids_to_plot': [557255, 132379, 69794]
          , 'hex_assign': []
          , 'text_placement':[0.78, 0.95, 0,-0.06]
          , 'replace': {}
          , 'directory': "fig_2S"
          , 'body_color_list': OL_COLOR.PALE_AND_YELLOW.rgba
          , 'body_color_order': [0, 1, 2]
          , 'color_by':'type'
          , 'n_vis': {}
          , 'view': 'Equator_slice'
        }
      , "Fig4_TmY_Group1": {
            'columnar_list': ['TmY3', 'TmY4', 'TmY5a', 'TmY9a']
          , 'list_bids_to_plot': [42010, 91611, 43032, 52346]
          , 'hex_assign': []
          , 'text_placement':[0.78, 0.95, 0,-0.06]
          , 'replace': {}
          , 'directory': "fig_4"
          , 'body_color_list':
                OL_COLOR.OL_TYPES.rgba + OL_COLOR.OL_DARK_TYPES.rgba
              + OL_COLOR.SIX_MORE_COLORS.rgba
          , 'body_color_order': [0, 3, 6, 9, 12, 1, 4, 7, 10, 13, 2, 5, 8, 11, 14]
          , 'color_by':'type'
          , 'n_vis': {}
          , 'view': 'Equator_slice'
        }
      , "Fig4_TmY_Group2": {
            'columnar_list': ['TmY9b', 'TmY10', 'TmY13', 'TmY14']
          , 'list_bids_to_plot':[55648, 47625, 57663, 21807]
          , 'hex_assign': []
          , 'text_placement':[0.78, 0.95, 0,-0.06]
          , 'replace': {}
          , 'directory': "fig_4"
          , 'body_color_list':
                OL_COLOR.OL_TYPES.rgba + OL_COLOR.OL_DARK_TYPES.rgba
              + OL_COLOR.SIX_MORE_COLORS.rgba
          , 'body_color_order': [0, 3, 6, 9, 12, 1, 4, 7, 10, 13, 2, 5, 8, 11, 14]
          , 'color_by':'type'
          , 'n_vis': {}
          , 'view': 'Equator_slice'
        }

        , "Fig1_Mi1": {
            'columnar_list': ['Mi1']
          , 'list_bids_to_plot':[
                77952, 49351, 35683, 33110, 57398, 35888, 58862, 34189, 36252, 34057
              , 35840, 36954, 36911, 47967, 39727, 41399, 45664, 79752
            ]
          , 'hex_assign':[]
          , 'text_placement':[0.05, 0.95, 0, 0]
          , 'replace':{}
          , 'directory': "fig_1"
          , 'body_color_list': OL_COLOR.OL_IN_SEQ.rgba
          , 'body_color_order': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
          , 'color_by': 'bid'
          , 'n_vis': {}
          , 'view': 'Equator_slice'
        }
        , "Fig1_TmY5a": {
            'columnar_list': ['TmY5a']
          , 'list_bids_to_plot':[
                65773, 61841, 74821, 55509, 54914, 53500, 59398, 76285, 46155, 76233, 52887, 53380
              , 51229, 53274, 59774, 53751, 60813, 80862, 103422
            ]
          , 'hex_assign':[]
          , 'text_placement':[0.05, 0.95, 0, 0]
          , 'replace':{}
          , 'directory': "fig_1"
          , 'body_color_list': OL_COLOR.OL_CONN_SEQ.rgba
          , 'body_color_order': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
          , 'color_by': 'bid'
          , 'n_vis': {}
          , 'view': 'Equator_slice'
        }
        , "Fig1_LC17": {
            'columnar_list': ['LC17']
          , 'list_bids_to_plot':[
                25742, 24409, 28207, 46140, 32917, 29219, 34645, 54277, 40641, 52895, 28605
            ]
          , 'hex_assign':[]
          , 'text_placement':[0.05, 0.95, 0, 0]
          , 'replace':{}
          , 'directory': "fig_1"
          , 'body_color_list': OL_COLOR.OL_VPN_SEQ.rgba
          , 'body_color_order': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
          , 'color_by': 'bid'
          , 'n_vis': {}
          , 'view': 'Equator_slice'
        }
          , "Fig1_LoVC16": {
            'columnar_list': ['LoVC16']
          , 'list_bids_to_plot':[10029, 10053]
          , 'hex_assign':[]
          , 'text_placement':[0.05, 0.95, 0, 0]
          , 'replace':{}
          , 'directory': "fig_1"
          , 'body_color_list': OL_COLOR.OL_VCN_SEQ.rgba
          , 'body_color_order': [1, 2]
          , 'color_by': 'bid'
          , 'n_vis': {}
          , 'view': 'Equator_slice'
        }
      , "Fig4_TmY_Group3": {
            'columnar_list': ['TmY15', 'TmY16', 'TmY17', 'TmY18']
          , 'list_bids_to_plot':[26151, 23353, 39768, 92012]
          , 'hex_assign': []
          , 'text_placement':[0.78, 0.95, 0,-0.06]
          , 'replace': {}
          , 'directory': "fig_4"
          , 'body_color_list':
                OL_COLOR.OL_TYPES.rgba + OL_COLOR.OL_DARK_TYPES.rgba
              + OL_COLOR.SIX_MORE_COLORS.rgba
          , 'body_color_order': [0, 3, 6, 9, 12, 1, 4, 7, 10, 13, 2, 5, 8, 11, 14]
          , 'color_by': 'type'
          , 'n_vis': {}
          , 'view': 'Equator_slice'
        }
      , "Fig4_TmY_Group4": {
            'columnar_list': ['TmY19a', 'TmY19b', 'TmY20', 'TmY21']
          , 'list_bids_to_plot':[17884, 19702, 49775, 34293]
          , 'hex_assign': []
          , 'text_placement':[0.78, 0.95, 0,-0.06]
          , 'replace': {}
          , 'directory': "fig_4"
          , 'body_color_list':
                OL_COLOR.OL_TYPES.rgba + OL_COLOR.OL_DARK_TYPES.rgba
              + OL_COLOR.SIX_MORE_COLORS.rgba
          , 'body_color_order': [0, 3, 6, 9, 12, 1, 4, 7, 10, 13, 2, 5, 8, 11, 14]
          , 'color_by': 'type'
          , 'n_vis': {}
          , 'view': 'Equator_slice'
        }
      , "Fig2_Pm2": {
            'columnar_list': ['Pm2a','Pm2b']
          , 'list_bids_to_plot': [[21674, 21511],[21497, 23746]]
          , 'hex_assign': []
          , 'text_placement': [0.10, 0.45,0.20, -0.21]
          #, 'text_placement': [0.03, 0.03,0.20, 0]
          , 'replace': {}
          , 'directory': "fig_2"
          , 'body_color_list': OL_COLOR.OL_TYPES.rgba
          , 'body_color_order': [1, 2]
          , 'color_by': 'type'
          , 'n_vis': {'npil':'ME','lay':9}
          , 'view': 'medulla_face_on'
        }
      , "Fig2f_MeVP10": {
            'columnar_list': ['MeVP10']
          , 'list_bids_to_plot':[
                32594, 32929, 33757, 34435, 34502
              , 34639, 35766, 37671, 41321
              , 41409, 41848, 42525, 42911, 46146
              , 46537, 47163, 49326, 50050, 51239, 52515
              , 52891, 52986, 53284, 53311, 54358, 55952, 58332
              , 60113, 64162, 64658, 66093, 67628, 68046
            ]
          , 'hex_assign':[]
          , 'text_placement':[0.05, 0.95, 0, 0]
          , 'replace':{}
          , 'directory': "fig_5"
          , 'body_color_list':
                OL_COLOR.OL_TYPES.rgba + OL_COLOR.OL_DARK_TYPES.rgba
              + OL_COLOR.SIX_MORE_COLORS.rgba
          , 'body_color_order': [0, 3, 6, 9, 12, 1, 4, 7, 10, 13, 2, 5, 8, 11, 14]
          , 'color_by': 'bid'
          , 'n_vis': {'npil':'ME','lay':6}
          , 'view': 'medulla_face_on'
        }
      , "Fig2f_Dm4": {
            'columnar_list': ['Dm4']
          , 'list_bids_to_plot':[
                16333,  16413,  16678,  16726,  16991,  17145,  17235,  17345
              , 17636,  17806,  17913,  17999,  18006,  18412,  18467,  18530
              , 18541,  18833,  19111,  19131,  19370,  19383,  19430,  19435
              , 19568,  19734,  19782,  19795,  20035,  20198,  20220,  20439
              , 20519,  20660,  20732,  20813,  21139,  21424,  21585,  21765
              , 21985,  22178,  22363,  22680,  23036,  27208,  40923, 203515
            ]
          , 'hex_assign':[]
          , 'text_placement':[0.05, 0.95, 0, 0]
          , 'replace':{}
          , 'directory': "fig_5"
          , 'body_color_list':
                OL_COLOR.OL_TYPES.rgba + OL_COLOR.OL_DARK_TYPES.rgba
              + OL_COLOR.SIX_MORE_COLORS.rgba
          , 'body_color_order': [0, 3, 6, 9, 12, 1, 4, 7, 10, 13, 2, 5, 8, 11, 14]
          , 'color_by': 'bid'
          , 'n_vis': {'npil':'ME','lay':3}
          , 'view': 'medulla_face_on'
        }
      , "Fig2f_Dm20": {
            'columnar_list': ['Dm20']
          , 'list_bids_to_plot':[
                18858, 19123, 19401, 19528, 19797, 19876, 20027, 20114, 20169
              , 20342, 20483, 20528, 20701, 20857, 20948, 20952, 21030, 21202
              , 21265, 21493, 21620, 22016, 22042, 22044, 22262, 22334, 22411
              , 22483, 22488, 22500, 22687, 22786, 22865, 23044, 23131, 23534
              , 23579, 23767, 24230, 24383, 25603, 25653, 26982, 28124, 28323
              , 28970, 30713, 33246
            ]
          , 'hex_assign':[]
          , 'text_placement':[0.05, 0.95, 0, 0]
          , 'replace':{}
          , 'directory': "fig_5"
          , 'body_color_list':
                OL_COLOR.OL_TYPES.rgba + OL_COLOR.OL_DARK_TYPES.rgba
              + OL_COLOR.SIX_MORE_COLORS.rgba
          , 'body_color_order': [0, 3, 6, 9, 12, 1, 4, 7, 10, 13, 2, 5, 8, 11, 14]
          , 'color_by': 'bid'
          , 'n_vis':{'npil':'ME','lay':3}
          , 'view': 'medulla_face_on'
        }
      , "Fig2f_l-LNv": {
           'columnar_list': ['l-LNv']
          , 'list_bids_to_plot':[10870, 11114, 11212, 11715]
          , 'hex_assign':[]
          , 'text_placement':[0.05, 0.95, 0, 0]
          , 'replace':{}
          , 'directory': "fig_5"
          , 'body_color_list':
                OL_COLOR.OL_TYPES.rgba + OL_COLOR.OL_DARK_TYPES.rgba
              + OL_COLOR.SIX_MORE_COLORS.rgba
          , 'body_color_order': [0, 3, 6, 9, 12, 1, 4, 7, 10, 13, 2, 5, 8, 11, 14]
          , 'color_by': 'bid'
          , 'n_vis': {'npil':'ME','lay':1}
          , 'view': 'medulla_face_on'
        }
      , "Fig5_Pathway_1": {
            'columnar_list': ['Tm1', 'Tm2', 'TmY4', 'TmY9a', 'TmY9b','LC11','LC15']
          , 'list_bids_to_plot':[48193, 45468, 56703, 62719, 62635, 24953, 31391]
          , 'hex_assign':[]
          , 'text_placement':[0.78, 0.95, 0,-0.06]
          , 'replace':{}
          , 'directory': "fig_5"
          , 'body_color_list':
                OL_COLOR.PALE_AND_YELLOW.rgba + OL_COLOR.OL_CONN_SEQ.rgba
              + OL_COLOR.OL_IN_SEQ.rgba + OL_COLOR.OL_VPN_SEQ.rgba
          , 'body_color_order': [3, 6, 4, 0, 5, 13, 14]
          , 'color_by': 'type'
          , 'n_vis': {}
          , 'view': 'Equator_slice'
        }
      , "Fig5_Pathway_2": {
            'columnar_list':['L2', 'L3','Dm3a', 'Dm3b','Dm3c', 'Li16']
          , 'list_bids_to_plot':[33893, 94403, 114083, 116753, 136002, 11394]
          , 'hex_assign':[]
          , 'text_placement':[0.78, 0.95, 0,-0.06]
          , 'replace':{}
          , 'directory': "fig_5"
          , 'body_color_list':
                OL_COLOR.MY_COLOR.rgba + OL_COLOR.OL_CONN_SEQ.rgba
              + OL_COLOR.OL_IN_SEQ.rgba + OL_COLOR.OL_VPN_SEQ.rgba
          , 'body_color_order': [0, 3, 10, 11, 12, 9]
          , 'color_by': 'type'
          , 'n_vis': {}
          , 'view': 'Equator_slice'
        }
    }
    return plots


def get_rend_params(
    camera_or_slice:str
  , view_type:str
) -> dict:
    """
    Helper function to return variables from a spreadsheet in `params/`

    Parameters
    ----------
    camera_or_slice : {'camera', 'slice'}
        get parameters either for the camer or the slicer
    view_type : str
        must be one of the views defined in the Rendering_parameters.xlsx
        (e.g. Dorsal_view)

    Returns
    -------
    param_dict : dict
        simple dictionary with locations, rotations, and other parameters

    """

    assert camera_or_slice in ['camera', 'slice', 'scalebar'], \
        f"Only 'camera', 'slice', or 'scalebar' are allowed, not {camera_or_slice}"
    rendering_parameters = pd.read_excel(
        Path(find_dotenv()).parent / 'params' / 'Rendering_parameters.xlsx'
    )
    rendering_parameters = rendering_parameters.set_index('type_of_view')

    if camera_or_slice=='slice':
        param_dict = {
            'loc': {
                'x': rendering_parameters.loc[view_type,'slice_loc_x']
              , 'y': rendering_parameters.loc[view_type,'slice_loc_y']
              , 'z': rendering_parameters.loc[view_type,'slice_loc_z']
            }
          , 'rot': {
                'x': rendering_parameters.loc[view_type,'slice_rot_x']
              , 'y': rendering_parameters.loc[view_type,'slice_rot_y']
              , 'z': rendering_parameters.loc[view_type,'slice_rot_z']
            }
        }
    if camera_or_slice=='camera':
        param_dict = {
            'loc': {
                'x': rendering_parameters.loc[view_type,'cam_loc_x']
              , 'y': rendering_parameters.loc[view_type,'cam_loc_y']
              , 'z': rendering_parameters.loc[view_type,'cam_loc_z']
            }
          , 'rot': {
                'x': rendering_parameters.loc[view_type,'cam_rot_x']
              , 'y': rendering_parameters.loc[view_type,'cam_rot_y']
              , 'z': rendering_parameters.loc[view_type,'cam_rot_z']
            }
          , 'res': {
                'x':rendering_parameters.loc[view_type,'cam_res_x']
              , 'y':rendering_parameters.loc[view_type,'cam_res_y']
            }
          , 'ortho': rendering_parameters.loc[view_type,'cam_ortho']
        }
    if camera_or_slice=='scalebar':
       param_dict = {
          'loc': {
              'x': rendering_parameters.loc[view_type,'scalebar_loc_x']
            , 'y': rendering_parameters.loc[view_type,'scalebar_loc_y']
            , 'z': rendering_parameters.loc[view_type,'scalebar_loc_z']
          }
          ,'txt_loc': {
              'x': rendering_parameters.loc[view_type,'sb_text_loc_x']
            , 'y': rendering_parameters.loc[view_type,'sb_text_loc_y']
            , 'z': rendering_parameters.loc[view_type,'sb_text_loc_z']
          }
      }
    return param_dict
