""" Functions for generation of analysis page of webpages """

from datetime import datetime
from pathlib import Path

from dotenv import find_dotenv
import pandas as pd
import jinja2

from neuprint import fetch_meta
from utils.scatterplot_functions import make_covcompl_scatterplot
from utils.ol_types import OLTypes


def get_meta_data() -> dict:
    """
    Fetch meta data from neuprint to use for footer of webpages.

    Returns
    -------
    meta : dict
        Metadata from the neuprint database

    """
    meta = fetch_meta()
    return meta


def get_last_database_edit() -> str:
    """
    Fetch the last database edit and convert it to ISO-8601 format.

    Returns
    -------
    last_database_edit : str
        timestamp formatted as string of last database entry
    """
    meta = fetch_meta()
    last_database_edit = datetime\
        .fromisoformat(meta['lastDatabaseEdit'])\
        .replace(tzinfo=None)\
        .isoformat(timespec='minutes')
    return last_database_edit


def get_formatted_now() -> str:
    """
    Format the current date in ISO-8601 format.

    Returns
    -------
    formatted_date : str
        current date
    """
    formatted_date = datetime.now().date().isoformat()
    return formatted_date


def render_and_save_templates(
    template_name:str
  , data_dict:dict
  , output_filename:str
):
    """
    Render jinja template and save resulting html page

    Parameters
    ----------
    template_name : str
        name of jinja template to be used
    data_dict : dict
        dictionary with information that will be used to fill template
    output_filename : str
        name of output file
    """
    # Assuming the templates are in the current directory for simplicity
    environment = jinja2.Environment(loader=jinja2.FileSystemLoader('.'))

    # Load the template
    template = environment.get_template(template_name)

    # Render the template with the dynamically passed data
    rendered_template = template.render(**data_dict)

    # Save the rendered template to an HTML file
    with open(output_filename, "w", encoding='UTF-8') as file:
        file.write(rendered_template)


def create_scatter_html(
    roi_str:str
  , scatter_list:list[dict]
  , output_path:str
) -> None:
    """
    Create html page with multiple scatterplots for a single optic lobe region.

    Parameters
    ----------
    roi_str : str
        optic lobe region of interest
    scatter_list : list[dict]
        xval : str
            column of dataframe in 'complete_metrics.pickle' to plot on x-axis
        yval : str
            column of dataframe in 'complete_metrics.pickle' to plot on y-axis
        colorscale : str
            column of dataframe in 'complete_metrics.pickle' to use as color scale for the markers
    output_path : str
        path to save the scatterplot html pages
    """
    assert isinstance(roi_str, str), f"roi_str must be str, not '{type(roi_str)}'."
    assert isinstance(scatter_list, list)\
      , f"scatter_list must be list, not '{type(scatter_list)}'."
    assert roi_str in ['ME(R)', 'LO(R)', 'LOP(R)'], f"unsupported roi_str '{roi_str}'."
    match roi_str:
        case 'ME(R)':
            star_instances = ['Dm4_R', 'Dm20_R', 'l-LNv_R', 'MeVP10_R']
        case 'LO(R)':
            star_instances = ['T2_R', 'Tm2_R', 'MeVPLo1_L', 'LC40_R']
        case _: # LOP(R)
            star_instances = ['LPLC2_R', 'LPLC4_R', 'OLVC3_L', 'LPT31_R']

    for idx, plot_data in enumerate(scatter_list):
        fig = make_covcompl_scatterplot(
            x_val=plot_data['xval']
          , y_val=plot_data['yval']
          , colorscale = plot_data['colorscale']
          , roi_str = roi_str
          , star_instances=star_instances
          , export_type='html'
          , save_plot=False
        )
        scatter_list[idx]['fig'] = fig.to_html(full_html=False)

    # metadata
    meta = get_meta_data()
    last_database_edit = get_last_database_edit()
    formatted_date = get_formatted_now()

    # all data dict
    scatter_data_dict = {
        'roi_str': roi_str
      , 'scatter_dict': scatter_list
      , 'meta': meta
      , 'formattedDate': formatted_date
      , 'lastDataBaseEdit': last_database_edit
    }

    # render and save
    render_and_save_templates(
        "scatterplots-page.html.jinja"
      , scatter_data_dict
      , output_path / f"scatterplots-{roi_str[:-3]}.html"
    )


def create_all_scatter_html() -> None:
    """
    Create interactive scatterplot html pages for ME(R), LO(R) and LOP(R)
    """
    output_path = Path(find_dotenv()).parent / "results" / "html_pages"  / "scatterplots"
    output_path.mkdir(parents=True, exist_ok=True)

    scatter_list = [
        {'xval': 'population_size', 'yval': 'cell_size_cols'
          , 'colorscale': 'coverage_factor_trim'}
      , {'xval': 'population_size', 'yval': 'cell_size_cols'
          , 'colorscale': 'area_completeness'}
      , {'xval': 'population_size', 'yval': 'coverage_factor_trim'
          , 'colorscale': 'cell_size_cols'}
      , {'xval': 'population_size', 'yval': 'coverage_factor_trim'
          , 'colorscale': 'area_completeness'}
      , {'xval': 'cols_covered_pop', 'yval': 'area_covered_pop'
          , 'colorscale': 'coverage_factor_trim'}
      , {'xval': 'cols_covered_pop', 'yval': 'area_covered_pop'
          , 'colorscale': 'cell_size_cols'}
    ]
    for roi_str in ['ME(R)', 'LO(R)', 'LOP(R)']:
        create_scatter_html(roi_str=roi_str, scatter_list=scatter_list, output_path=output_path)


def get_youtube_link(cell_type:str) -> str:
    """
    Find the corresponding youtube link for each cell type's webpage.

    Parameters
    ----------
    cell_type : str
        cell type name

    Returns
    -------
    youtube_link : str
        part of link to the cell type's corresponding YouTube video.
        Only returns the part of the path after "https://www.youtube.com/embed/'
    """
    movie_params = Path(find_dotenv()).parent / "params" / "Movie_curation.xlsx"

    olt = OLTypes()
    cell_type_list = olt.get_neuron_list(side='both')
    types_df = cell_type_list[['type', 'instance']]

    assert movie_params.is_file()\
      , f"Movie parameter file is missing at '{movie_params}'"
    df = pd.read_excel(movie_params, dtype={'Youtube URL': str})

    for index, _ in df.iterrows():
        str_to_use = df.loc[index, 'cell type'][:-4]
        df.loc[index, 'type'] = str_to_use
        df2 = df.merge(
            types_df
          , how='outer'
          , on='type'
        )
    if pd.isna(df2.loc[df2['type']==cell_type, 'Youtube URL']).iloc[0]:
        youtube_url = df2.loc[df2['type']=='coming ', 'Youtube URL']
    else:
        youtube_url = df2.loc[df2['type']==cell_type, 'Youtube URL']
    youtube_link = youtube_url.iloc[0].split('/')[-1]

    return youtube_link
