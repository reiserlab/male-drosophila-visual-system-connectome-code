""" Functions for generation of analysis page of webpages """
import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
from neuprint import fetch_meta
import datetime
from datetime import datetime
import jinja2
from utils.scatterplot_functions import make_covcompl_scatterplot


def get_meta_data():
    """
    Fetch meta data from neuprint to use for footer of webpages.

    Returns
    -------
    meta : dict
        Metadata from the neuprint database
    last_database_edit : str
        timestamp formatted as string of last database entry
    formatted_date : str
        current date
    """

    # all meta data
    meta = fetch_meta()

    # time of last database edit
    timestamp_str = meta['lastDatabaseEdit']
    timestamp = datetime.fromisoformat(timestamp_str.rstrip("Z"))
    last_database_edit = timestamp.strftime('%Y-%m-%d %H:%M:%S')

    # date when page was generated
    today_date = datetime.now().date()
    formatted_date = today_date.strftime("%Y-%m-%d")

    return meta, last_database_edit, formatted_date


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
    with open(output_filename, "w") as file:
        file.write(rendered_template)


def create_scatter_html(
    roi_str:str
  , scatter_dict:dict
  , output_path:str
):
    """
    Create html page with multiple scatterplots for a single optic lobe region.

    Parameters
    ----------
    roi_str : str
        optic lobe region of interest

    scatter_dict : dict
        xval : str
            column of dataframe in 'complete_metrics.pickle' to plot on x-axis
        yval : str
            column of dataframe in 'complete_metrics.pickle' to plot on y-axis
        colorscale : str
            column of dataframe in 'complete_metrics.pickle' to use as color scale for the markers

    output_path : str
        path to save the scatterplot html pages
    """
    if roi_str == 'ME(R)':
        star_instances = ['Dm4_R', 'Dm20_R', 'l-LNv_R', 'MeVP10_R']
    elif roi_str == 'LO(R)':
        star_instances = ['T2_R', 'Tm2_R', 'MeVPLo1_L', 'LC40_R']
    elif roi_str == 'LOP(R)':
        star_instances = ['LPLC2_R', 'LPLC4_R', 'OLVC3_L', 'LPT31_R']

    for idx in scatter_dict:
        plot_data = scatter_dict[idx]
        fig = make_covcompl_scatterplot(
            x_val=plot_data['xval']
          , y_val=plot_data['yval']
          , colorscale = plot_data['colorscale']
          , roi_str = roi_str
          , star_instances=star_instances
          , export_type='html'
          , save_plot=False
        )
        scatter_dict[idx]['fig'] = fig.to_html(full_html=False)

    # metadata
    meta, last_database_edit, formatted_date = get_meta_data()

    # all data dict
    scatter_data_dict = {
        'roi_str': roi_str
      , 'scatter_dict': scatter_dict
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

def create_all_scatter_html():
    """
    Create interactive scatterplot html pages for ME(R), LO(R) and LOP(R)
    """
    PROJECT_ROOT = Path(find_dotenv()).parent
    output_path = PROJECT_ROOT / "results" / "html_pages"  / "scatterplots"
    os.makedirs(output_path, exist_ok=True)

    scatter_dict = {
        0 : {'xval': 'population_size', 'yval': 'cell_size_cols', 'colorscale': 'coverage_factor_trim'},
        1 : {'xval': 'population_size', 'yval': 'cell_size_cols', 'colorscale': 'area_completeness'},
        2 : {'xval': 'population_size', 'yval': 'coverage_factor_trim', 'colorscale': 'cell_size_cols'},
        3 : {'xval': 'population_size', 'yval': 'coverage_factor_trim', 'colorscale': 'area_completeness'},
        4 : {'xval': 'cols_covered_pop', 'yval': 'area_covered_pop', 'colorscale': 'coverage_factor_trim'},
        5 : {'xval': 'cols_covered_pop', 'yval': 'area_covered_pop', 'colorscale': 'cell_size_cols'},
    }
    for roi_str in ['ME(R)', 'LO(R)', 'LOP(R)']:
        create_scatter_html(roi_str=roi_str, scatter_dict=scatter_dict, output_path=output_path)
