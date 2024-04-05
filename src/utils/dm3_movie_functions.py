"""
Functions for generating the 'striped' Dm3 videos in 'src/movies/generate_json_Dm3.ipynb'
"""
import sys
from pathlib import Path
import jinja2
import pandas as pd
import numpy as np
from cmap import Color


from dotenv import load_dotenv, find_dotenv

load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath("src")))
print(f"Project root directory: {PROJECT_ROOT}")

from utils.ol_neuron import OLNeuron
from utils.neuron_bag import NeuronBag


def get_body_id_by_hex(
    cell_type: str
) -> pd.Series:
    """
    Creates a series containing the bodyIds of all the Dm3 neurons of a particular type ordered by
    their hex1/hex2 position. Neurons are grouped according to this position and this determines
    the order in which they are presented in the movie.

    Parameters
    ----------
    cell_type : str
        name of the cell type

    Returns
    -------
    ids_df : pd.Series
        series with bodyIds of all the neurons from the 'cell_type' ordered
        based on their hex1/hex2 position and relative to [18, 18].
    """
    allowed_ct = ['Dm3a', 'Dm3b', 'Dm3c']
    assert cell_type in allowed_ct,\
        f"cell_type can only be one of {allowed_ct}, not {cell_type}"

    a_bag = NeuronBag(cell_type=cell_type)

    # sort all bodyIDs by their distance to [18,18]
    a_bag.sort_by_distance_to_hex(neuropil="ME(R)", hex1_id=18, hex2_id=18)

    # get the hex coordinates of all bodyIDs
    hex_df = a_bag.get_hex_ids()
    hex_df = hex_df[hex_df["ROI"] == "ME(R)"]

    # Specify grouping variable based on cell type
    if cell_type == "Dm3a":
        grp = "hex1_id"
        val = 18
    elif cell_type == "Dm3b":
        grp = "hex2_id"
        val = 18
    elif cell_type == "Dm3c":
        hex_df["hex3_id"] = hex_df["hex1_id"] - hex_df["hex2_id"]
        grp = "hex3_id"
        val = -2

    # group the bodyIDs by their 'grp'
    hex_grouped = hex_df\
        .groupby(grp)\
        .bodyID.apply(list)\
        .drop_duplicates()\
        .reset_index()
    grps_below = abs(val - (hex_df[grp]).min())
    grps_above = abs((hex_df[grp]).max() - val)

    # get 'group' number of the order in which the cells will be presented
    group_id = pd.Series(
        [*range(grps_below + 1, 0, -1)] + [0] + [*range(1, grps_above + 2, 1)]
    )
    hex_grouped["group"] = group_id
    hex_grouped.drop(columns={grp})
    # combine bodyIDs based on presentation 'group'
    ids_df = hex_grouped\
        .explode("bodyID")\
        .groupby("group")["bodyID"]\
        .apply(list)

    return ids_df


def get_group_colors(
    color:str
  , stripes:bool
  , n_groups:int
) -> list:
    """
    get hexcode colors to be used for each group of neurons in the dm3 images/videos

    Parameters
    ----------
    color : str
        defines colormap to be used to color the outer groups of Dm3 cells.
    stripes : bool
        If `True` alternate hex column groups of cells are white / the chosen 'color'. 
        If `False` all of the groups are colored.
    n_groups : int
        the number of groups of neurons

    Returns
    -------
    group_colors : list
        hexcodes of the 'n_groups' colors to be used to color the groups of neurons
    """
    col_hex = Color(color).hex
    group_colors = [col_hex for i in range(n_groups)]
    column_vals = range(n_groups - 1)

    # if stripes=True, make every other column white
    if stripes:
        for idx in list(column_vals[1::2]):
            group_colors[idx] = "#8F8F8F"

    return group_colors


def generate_movie_description(
    cell_type:str
  , df:pd.Series
  , template:str="Dm3-template.json.jinja"
  , number_of_neighbors:int=5
  , color:str="blue"
  , stripes:bool=True
) -> None:
    """
    Generate the movie JSON file using the Dm3 template.

    It generates a file <ME_tiling_neuron_type_color.json> inside the src/movies/ directory.

    Parameters
    ----------
    cell_type : str
        name of the cell type
    ids_df : dataframe
        List of body IDs with each column of the dataframe representing groups of neurons that 
        would be shown sequentially. First column contains the bodyIds in the central (hex1_id=18)
        with the first element corresponding to the star neuron (18,18), the next 
        "number_of_neighbors" elements being the closest neighbors and the remaining elements
        being the rest of the bodyIds in (hex1_id=18). The remaining columns of the dataframe
        contain the bodyIds of neurons in the flanking columns of hex1_id=18.
    template : str, default = "Dm3-template.tmpl"
        name of the template to use (must be a Jinja2 template)
    number_of_neighbors:
        allow a different number of neighbors in the hex1_id=18 column.
    color: str
        defines colormap to be used to color the outer groups of Dm3 cells.
    stripes: bool
        If 'True' alternate hex column groups of cells are white / the chosen 'color'. 
        If 'False' all of the groups are colored.
    """

    central_hex_column = df[0]
    assert (len(central_hex_column) > number_of_neighbors),\
        "reconsider your number of neighbors"

    template_path = f"{PROJECT_ROOT}/src/movies"
    environment = jinja2.Environment(loader=jinja2.FileSystemLoader(template_path))
    template = environment.get_template(template)

    # star neuron
    central_neuron = OLNeuron(body_id=central_hex_column[0])

    neighbor_neurons = central_hex_column[1 : number_of_neighbors + 1]
    other_neurons = central_hex_column[number_of_neighbors + 1 :]

    n_groups = len(df)
    # additional hex columns
    group_neurons = df[1:]

    # if there are more than 17 groups then group all bodyIds from groups >17 into 17
    if n_groups > 18:
        df[17] = df[17:].explode().values
        df = df[:18]
        n_groups = len(df)

    group_colors = get_group_colors(color, stripes, n_groups)

    column_rois, layer_rois = central_neuron.innervated_rois()
    pre_count, post_count = central_neuron.synapse_count()

    movie_description = template.render(
        neuron_type=cell_type
      , star_neuron=central_neuron.get_body_id()
      , neighbor_neurons=neighbor_neurons
      , other_neurons=other_neurons
      , group_neurons=group_neurons
      , group_colors=group_colors
      , neuron_count=np.sum(df.count())
      , pre_count=pre_count
      , post_count=post_count
      , innervated_columns=column_rois
      , innervated_layers=layer_rois
    )

    movie_fn = (
        Path(find_dotenv()).parent
      / "results"
      / "movie-descriptions"
      / f"ME_tiling_{cell_type}_{color}.json"
    )

    movie_fn.parent.mkdir(parents=True, exist_ok=True)
    with open(movie_fn, mode="w", encoding="utf-8") as movie_json:
        movie_json.write(movie_description)
