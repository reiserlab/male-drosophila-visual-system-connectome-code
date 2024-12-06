import json

from pathlib import Path
import jinja2
from dotenv import find_dotenv

from utils.ol_neuron import OLNeuron


def generate_movie_json(
    neuron_type:str
  , sorted_body_ids:list[int]
  , template:str="MEi.json.jinja"
  , is_general_template=False
  , number_of_neighbors:int=10
  , innervation_threshold:list[int|float, int|float]=[0.0, 0.01]
  , movie_group:str='vpn_vcn'
  , ignore_missing:bool=True
) -> None:
    """
    Generate the movie JSON file using a specific template.

    It generates a file <neuron_type.json> inside the src/movies/ directory.

    Parameters
    ----------
    neuron_type : str
        Type of the neuron
    sorted_body_ids : list[int]
        List of body IDs where the first element is the "star" of the show, the next n are the
        neighbors, followed by the rest of the neurons.
    template : str, default = "MEi.json.jinja"
        name of the template to use (must be a Jinja2 template)
    number_of_neighbors:
        allow a different number of neighbors
    ignore_missing : bool, default=True
        ignore body IDs that are known to not have meshes.
    """

    template_path = Path(find_dotenv()).parent / "src" / "movies"
    environment = jinja2.Environment(loader=jinja2.FileSystemLoader(template_path))
    template = environment.get_template(template)


    ignore_fn = Path(find_dotenv()).parent / 'params' / 'ignore_render.json'

    if ignore_missing and ignore_fn.is_file():
        with open(ignore_fn, encoding="utf-8") as json_data:
            data = json.load(json_data)

        ignore_list = []
        for _, v_a in data['ignore'].items():
            ignore_list.extend(v_a)
        sorted_body_ids = [x for x in sorted_body_ids if x not in ignore_list]

    central_neuron = OLNeuron(sorted_body_ids[0])
    column_rois, layer_rois = central_neuron.innervated_rois(
        column_threshold=innervation_threshold[0]
      , layer_threshold=innervation_threshold[1]
    )

    me_columns = []
    lo_columns = []
    lop_columns = []
    for column in column_rois:
        if column.startswith("ME_"):
            me_columns.append(column)
        elif column.startswith("LO_"):
            lo_columns.append(column)
        elif column.startswith("LOP_"):
            lop_columns.append(column)

    layers_list = [
        'ME_R_layer_01', 'ME_R_layer_02', 'ME_R_layer_03', 'ME_R_layer_04'
      , 'ME_R_layer_05', 'ME_R_layer_06', 'ME_R_layer_07', 'ME_R_layer_08'
      , 'ME_R_layer_09', 'ME_R_layer_10'
      , 'LO_R_layer_1', 'LO_R_layer_2', 'LO_R_layer_3', 'LO_R_layer_4'
      , 'LO_R_layer_5', 'LO_R_layer_6', 'LO_R_layer_7'
      , 'LOP_R_layer_1','LOP_R_layer_2', 'LOP_R_layer_3', 'LOP_R_layer_4'
    ]

    me_layers_innervated = []
    me_layers_noninnervated = []
    lo_layers_innervated = []
    lo_layers_noninnervated = []
    lop_layers_innervated = []
    lop_layers_noninnervated = []

    for layer in layers_list:
        if layer.startswith("ME_"):
            if layer in layer_rois:
                me_layers_innervated.append(layer)
            else:
                me_layers_noninnervated.append(layer)
        elif layer.startswith("LO_"):
            if layer in layer_rois:
                lo_layers_innervated.append(layer)
            else:
                lo_layers_noninnervated.append(layer)
        elif layer.startswith("LOP_"):
            if layer in layer_rois:
                lop_layers_innervated.append(layer)
            else:
                lop_layers_noninnervated.append(layer)

    match movie_group:
        case 'OL_intrinsic':
            if me_layers_innervated:
                the_movie_type = 'MEi'
            elif lo_layers_innervated:
                the_movie_type = 'LOi'
            elif lop_layers_innervated:
                the_movie_type = 'LOPi'
            else:
                the_movie_type = 'MEi'
        case 'OL_connecting':
            #if layers in ME and LO
            if me_layers_innervated and lo_layers_innervated and not lop_layers_innervated:
                the_movie_type = 'OL_connecting'
            #if layers in ME and LOP
            elif me_layers_innervated and not lo_layers_innervated and lop_layers_innervated:
                the_movie_type = 'OL_connecting'
            #if layers in LO and LOP
            elif not me_layers_innervated and lo_layers_innervated and lop_layers_innervated:
                the_movie_type = 'OL_connecting'
            #if only in medulla(special case for C2,C3 etc. Medulla +Lamina)
            elif not lo_layers_innervated and not lop_layers_innervated and  me_layers_innervated:
                the_movie_type = 'MEi'
            else:
                the_movie_type = 'OL_connecting'
        case 'VPN' | 'VCN' | 'other':
            if me_layers_innervated and not lo_layers_innervated and not lop_layers_innervated:
                the_movie_type = 'MEi'
            elif not me_layers_innervated and lo_layers_innervated and not lop_layers_innervated:
                the_movie_type = 'LOi'
            elif not me_layers_innervated and not lo_layers_innervated and lop_layers_innervated:
                the_movie_type = 'LOPi'
            else:
                the_movie_type = 'OL_connecting'
        case _:
            the_movie_type='OL_connecting'

    pre_count, post_count = central_neuron.synapse_count()

    movie_description = template.render(
        neuron_type=neuron_type
      , star_neuron=sorted_body_ids[0]
      , neighbor_neurons=sorted_body_ids[1:number_of_neighbors+1]
      , other_neurons=sorted_body_ids[number_of_neighbors+1:]
      , neuron_count=len(sorted_body_ids)
      , pre_count=pre_count
      , post_count=post_count
      , ME_columns=me_columns
      , LO_columns=lo_columns
      , LOP_columns=lop_columns
      , ME_layers_innervated=me_layers_innervated
      , ME_layers_noninnervated=me_layers_noninnervated
      , LO_layers_innervated=lo_layers_innervated
      , LO_layers_noninnervated=lo_layers_noninnervated
      , LOP_layers_innervated=lop_layers_innervated
      , LOP_layers_noninnervated=lop_layers_noninnervated
      , innervated_layers=layer_rois
      , number_of_columns=len(column_rois)
      , movie_type=the_movie_type
      , movie_group=movie_group
    )

    movie_path = Path(find_dotenv()).parent / "results" / "movie-descriptions"
    movie_path.mkdir(parents=True, exist_ok=True)
    if is_general_template:
        movie_fn = movie_path / f"{neuron_type}_general.json"
    else:
        movie_fn = movie_path / f"{neuron_type}.json"

    with open(movie_fn, mode="w", encoding="utf-8") as movie_json:
        movie_json.write(movie_description)

def generate_tiling_movie_json(
    neuron_type: str
  , sorted_body_ids: list[int]
  , text_dict: dict
  , params: dict
  , template: str = "MEi_tiling.json.jinja"
  , number_of_neighbors: int = 10
  , innervation_threshold: list[int|float, int|float] = [0, 0]
) -> None:
    """
    Generate the tiling movie JSON file using a specific template.
    It generates a file <neuron_type.json> inside the src/movies directory.

    Parameters
    ----------
    neuron_type : str
        Type of the neuron
    sorted_body_ids : list[int]
        List of body IDs where the first element is the "star" of the show, the next n are the
        neighbors, followed by the rest of the neurons.
    text_dict : dict
        dict containing the text strings to be displayed in the movies
    params : dict
        dict of specific movie parameters to be used for rendering the json
    template : str, default = "MEi_tiling.json.jinja"
        name of the template to use (must be a Jinja2 template)
    number_of_neighbors : int
        allow a different number of neighbors
    innervation_threshold : tuple
        tuple containing threshold for including different columns and layers
    """
    template_path = Path(find_dotenv()).parent / "src" / "movies"
    environment = jinja2.Environment(loader=jinja2.FileSystemLoader(template_path))
    template = environment.get_template(template)

    central_neuron = OLNeuron(sorted_body_ids[0])

    # Not needed for tiling movie but blender seems to need these parameters
    column_rois, layer_rois= central_neuron.innervated_rois(
        column_threshold=innervation_threshold[0]
      , layer_threshold=innervation_threshold[1]
    )
    number_of_columns=len(column_rois)
    pre_count, post_count = central_neuron.synapse_count()

    movie_description = template.render(
        neuron_type=neuron_type
      , star_neuron=sorted_body_ids[0]
      , neighbor_neurons=sorted_body_ids[1:number_of_neighbors+1]
      , other_neurons=sorted_body_ids[number_of_neighbors+1:]
      , neuron_count=len(sorted_body_ids)
      , pre_count=pre_count
      , post_count=post_count
      , innervated_columns=column_rois
      , innervated_layers=layer_rois
      , number_of_columns=number_of_columns
      , text_dict=text_dict
      , params=params
    )

    movie_path = Path(find_dotenv()).parent / "results" / "movie-descriptions"
    movie_path.mkdir(parents=True, exist_ok=True)
    movie_fn = movie_path / f"{neuron_type}.json"

    with open(movie_fn, mode="w", encoding="utf-8") as movie_json:
        movie_json.write(movie_description)
