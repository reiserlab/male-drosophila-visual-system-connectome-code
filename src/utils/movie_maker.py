from pathlib import Path
import jinja2
import pandas as pd
from dotenv import find_dotenv
from utils.ol_neuron import OLNeuron

def create_template_filename(row:pd.Series):
    """
    Helper function to generate a filename based on a panda row

    Parameters
    ----------
    row : pd.Series
        from a list of OL intrinisc neuron types with at least 1 required column:
        Group1 : str
            Must contain 'ME_INTRINSIC', 'LO_INTRINSIC', or 'LOP_INTRINSIC' for specific templates

    Returns
    -------
    template_name : str
        name of the jinja template to be used
    """
    mapping = {
        'ME_INTRINSIC': 'MEi'
      , 'LO_INTRINSIC': 'LOi'
      , 'LOP_INTRINSIC': 'LOPi'
    }
    neuropil = mapping.get(row['Group1'])
    extension = 'json.jinja'
    if neuropil:
        return f"{neuropil}.{extension}"
    return None

def generate_movie_json(
    neuron_type:str
  , sorted_body_ids:list[int]
  , template:str = "MEi.json.jinja"
  , number_of_neighbors:int=10
  , innervation_threshold:list[int|float, int|float]=[0.01, 0.05]
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
    """

   # assert len(sorted_body_ids)>number_of_neighbors+1, "reconsider your number of neighbors"

    template_path = Path(find_dotenv()).parent / "src" / "movies"
    environment = jinja2.Environment(loader=jinja2.FileSystemLoader(template_path))
    template = environment.get_template(template)

    central_neuron = OLNeuron(sorted_body_ids[0])
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
    )

    movie_path = Path(find_dotenv()).parent / "results" / "movie-descriptions"
    movie_path.mkdir(parents=True, exist_ok=True)
    movie_fn = movie_path / f"{neuron_type}.json"

    with open(movie_fn, mode="w", encoding="utf-8") as movie_json:
        movie_json.write(movie_description)
