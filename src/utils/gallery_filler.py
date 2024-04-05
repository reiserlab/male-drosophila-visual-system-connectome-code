from pathlib import Path
import jinja2
import pandas as pd
import numpy as np
from dotenv import find_dotenv

from utils.rend_params import get_rend_params
from utils.ol_types import OLTypes
from utils.ol_color import OL_COLOR
from utils.hex_hex import hex_to_bids

def generate_gallery_json(
    type_of_plot:str
  , type_or_group:str
  , title:str
  , description: str
  , view:str
  , list_of_ids:dict
  , camera:dict
  , slicer:dict
  , scalebar:dict
  , n_vis:dict
  , neuropil_color:list
  , directory:str
  , template:str="gallery-descriptions.json.jinja"
  , list_of_rois:dict=None
) -> None:
    """
    Generate the gallery or group plot JSON file using a specific template.

    It generates a file <neuron_type.json> inside the `src/movies/` directory.

    Parameters
    ----------
    type_of_plot : str
        Type of plot, either 'Group' or 'Gallery'
    type_or_group : str
        name of the group or neuron type to be plotted
    title : str
        Title of group plot to be printed
        on plot--can be left empty ("")
    list_of_ids : dict
        dictionary with type name
    n_vis: dict
        which layer meshes to show by inputing neuropil ('ME','LO', or 'LOP') and layer number; stucture n_vis{'npil':*,'lay':*}
    camera : dict
        gets camera rotation, location etc from "Rendering_Parameters.xlsx" based on view type
    slicer : dict
        gets slicer rotation, location etc from "Rendering_Parameters.xlsx" based on view type
    template : str, default="gallery-descriptions.json.jinja"
        name of the template to use (must be a Jinja2 template)
    description : str
        Is it a type, group, or star neuron
    view : str
        camera rotation, location and ortho scale in blender
        appropriate to type of plot(`whole_brain`, `half_brain`, or `ol_intrinsic`)
        indicated in template
    list_of_ids : dict
        dictionary with type name
    list_of_rois : dict
        dictionary with additional ROIs
    camera : dict
        gets camera rotation, location etc from "Rendering_Parameters.xlsx" based on view type
    slicer : dict
        gets slicer rotation, location etc from "Rendering_Parameters.xlsx" based on view type
    n_vis: dict
        which layer meshes to show by inputing neuropil ('ME','LO', or 'LOP') and layer number;
        stucture n_vis{'npil':*,'lay':*}
    neuropil_color : list
        assigns color to neuropil_layers; use new OL_COLOR.OL_NEUROPIL_LAYERS.rgba
        from utils/ol_color.py
    directory : str
        Name of folder in "results/galleries" that results will be sent to
    template : str, default="gallery-descriptions.json.jinja"
        name of the template to use (must be a Jinja2 template)
    """

    assert type_of_plot in ['Full-Brain', 'Optic-Lobe'], \
        f"type of plot has the unexpected value {type_of_plot}"

    if list_of_rois is None:
        list_of_rois = {}

    allowed_views = [
        'whole_brain', 'half_brain', 'ol_intrinsic', 'ol_intrinsic_closeup'\
      , 'Equator_slice', 'Dorsal_slice', 'Ventral_slice', 'AME_slice'\
      , 'Lamina_slice', 'Lamina_equator_slice','Lamina_dorsal_slice'\
      , 'medulla_face_on'
    ]
    assert view in allowed_views, \
        f"view has to be one of {allowed_views} - not {view}"

    template_path = Path(find_dotenv()).parent / "src" / "gallery_generation"
    environment = jinja2.Environment(
        loader=jinja2.FileSystemLoader(template_path)
    )
    template = environment.get_template(template)

    gallery_list = template.render(
        type_of_plot=type_of_plot
      , description=description
      , type_or_group=type_or_group
      , title=title
      , view=view
      , list_of_ids=list_of_ids
      , list_of_rois=list_of_rois
      , camera=camera
      , slicer=slicer
      , scalebar=scalebar
      , n_vis=n_vis
      , neuropil_color=neuropil_color
      , directory=directory
    )

    gallery_path = Path(find_dotenv()).parent / "results" / "gallery-descriptions"
    gallery_path.mkdir(parents=True, exist_ok=True)
    gallery_fn = gallery_path / f"{type_of_plot}_{description}_{type_or_group}.json"

    with open(gallery_fn, mode='w', encoding="utf-8") as gallery_json:
        gallery_json.write(gallery_list)


def generate_one_off(
    plot_name:str
  , columnar_list:list
  , list_bids_to_plot:list
  , hex_assign:list
  , text_placement:list
  , replace:dict
  , body_color_list:list
  , body_color_order:list
  , color_by:str
  , n_vis:dict
  , neuropil_color:list
  , directory:str
  , the_view:str
) -> None:
    """
    Parameters
    ----------
    plot_name : str
        self explanatory
    columnar list : list
        Cell types to be plotted
    list_bids_to_plot : list
        hand curated body ids in same order as columnar list.
        Left empty [] if using "hex assign"
    hex_assign : list
        [hex1 for first neuron type, hex2 for first neuron type
            , increment for hex# change for hex1, increment for hex# change for hex2]
        left empty[] if curated by hand(using "list_bids_to_plot")
    text_placement : list
        [text x_coordinate for first neuron type, text y_coordinate for first neuron type
          , increment for changing x_coordinate, increment for changing y_coordinate]
    replace : dict
        replaces a single body id if necessary
    body_color_list : list
        color of the neurons
    body_color_order : list
        order of color assignment
    color_by: str
        can either color by type of neuron or by body id. string='type' or 'bid'
    n_vis: dict
        which layer meshes to show by inputing neuropil ('ME','LO', or 'LOP') and layer number;
        stucture n_vis{'npil':*,'lay':*}
    neuropil_color : list
        assigns color to neuropil_layers; use new OL_COLOR.OL_NEUROPIL_LAYERS.rgba
        from utils/ol_color.py
    directory : str
        folder output end up in
    view : str
        view used; used for getting camera and slice parameters from "Rendering_parameters.xlsx"
    """
    olt = OLTypes()
    neuron_list = olt.get_neuron_list()
    one_off_list = neuron_list[neuron_list['type']\
        .isin(columnar_list)]\
        .set_index('type')\
        .reindex(columnar_list)\
        .reset_index()

    if color_by=='type':
        res = []
        for idx, i_type in enumerate(columnar_list):
            if len(list_bids_to_plot)==0:
                h1 = hex_assign[0] + (hex_assign[2]*idx)
                h2 = hex_assign[1] + (hex_assign[3]*idx)
                res.append({
                    'type': i_type
                  , 'the_new_star': hex_to_bids((h1, h2), n_types=[i_type], return_type='list')[0]
                  , 'h1': h1
                  , 'h2': h2
                })
            else:
                res.append({
                    'type': i_type
                  , 'the_new_star': list_bids_to_plot[idx]
                })
        one_off_list = one_off_list.merge(pd.DataFrame(res), how='left')


    if color_by=='bid':
        if len(list_bids_to_plot) > len(one_off_list):
            one_off_list = pd.DataFrame(
                np.repeat(
                    one_off_list
                  , len(list_bids_to_plot)
                  , axis=0
                )
              , columns=one_off_list.columns
            )
        for idx, bid in enumerate(list_bids_to_plot):
            one_off_list.loc[idx, 'the_new_star'] = bid
        one_off_list = one_off_list[one_off_list['the_new_star'].notnull()]
        one_off_list['the_new_star'] = one_off_list['the_new_star'].astype('Int64')

    if len(replace)!=0:
        one_off_list\
            .loc[one_off_list.loc[:, 'type'] == replace.get('name'), 'the_new_star']\
          = replace.get('bid')

    body_color_list[:] = [body_color_list[idx] for idx in body_color_order]

    neuropil_color = [
        OL_COLOR.OL_NEUROPIL_LAYERS.rgba[0], OL_COLOR.OL_NEUROPIL_LAYERS.rgba[1]
      , OL_COLOR.OL_NEUROPIL_LAYERS.rgba[2], OL_COLOR.OL_NEUROPIL_LAYERS.rgba[0]
      , OL_COLOR.OL_NEUROPIL_LAYERS.rgba[1], OL_COLOR.OL_NEUROPIL_LAYERS.rgba[2]
      , OL_COLOR.OL_NEUROPIL_LAYERS.rgba[0], OL_COLOR.OL_NEUROPIL_LAYERS.rgba[1]
      , OL_COLOR.OL_NEUROPIL_LAYERS.rgba[2]
    ]

    fischbach_dict = {}

    txt_pos_x = text_placement[0]
    txt_pos_y = text_placement[1]

    one_off_list['grp_count'] = one_off_list.groupby('type').cumcount()

    for idx, row in one_off_list.iterrows():
        the_type = row['type']
        if row['grp_count']>0:
            the_type = f"{row['type']}_{row['grp_count']}"

        the_star = [row['the_new_star']]
        if isinstance(row['the_new_star'], list):
            the_star = row['the_new_star']

        body_id_dict = {
            'type': the_type
          , 'body_ids': the_star
          , 'body_color': body_color_list[idx % len(body_color_list)]
          , 'text_position': [txt_pos_x, txt_pos_y]
          , 'text_align': 'l'
          , 'number_of_cells': 1
          , 'slice_width': 12000
        }

        if color_by=='type':
            fischbach_dict[row['type']] = body_id_dict
        elif color_by=='bid':
            fischbach_dict[row['the_new_star']] = body_id_dict
        txt_pos_x = txt_pos_x + text_placement[2]
        txt_pos_y = txt_pos_y + text_placement[3]

    the_title = ""
    if len(t_item := one_off_list.groupby('type')['type'].unique().tolist())==1 :
        the_title = t_item[0][0]

    camera_dict = get_rend_params('camera', the_view)
    slicer_dict = get_rend_params('slice', the_view)
    scale_bar_dict = get_rend_params('scalebar', the_view)

    generate_gallery_json(
        type_of_plot="Optic-Lobe"
      , description="OLi"
      , type_or_group=plot_name
      , title=the_title
      , camera=camera_dict
      , slicer=slicer_dict
      , scalebar=scale_bar_dict
      , list_of_ids=fischbach_dict
      , n_vis=n_vis
      , neuropil_color=neuropil_color
      , view=the_view
      , directory=directory
      , template="gallery-descriptions.json.jinja"
    )

    print(f"Json generation done for {plot_name}")
