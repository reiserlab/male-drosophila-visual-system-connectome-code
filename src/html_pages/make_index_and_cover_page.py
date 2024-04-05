# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# Import optic lobe components
import sys
from pathlib import Path
import jinja2
import datetime
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath('src')))

from utils import olc_client
c = olc_client.connect(verbose=True)

from neuprint import NeuronCriteria as NC
from neuprint import fetch_neurons
# from patterns import fetch_and_organize_neuron_data

from queries.completeness import fetch_ol_types_and_instances
from utils.ol_color import OL_COLOR 
from html_pages.webpage_functions import get_meta_data


# %%
# Fetch unique instances and their types
neuron_names = fetch_ol_types_and_instances(side='both', client=c)

# # Determine types with multiple instances
# type_counts = neuron_names['type'].value_counts()
# multiple_instances = type_counts[type_counts > 1].index.tolist()

# # Initialize list for available tags
available_tags = []

# Iterate over rows in the DataFrame
for index, row in neuron_names.iterrows():
    # Determine filename based on presence of multiple instances
    link_to_instance = row['instance']
    filename = row['type'] + f" ({link_to_instance[-1]})"
    tag = {"value": filename, "url": f"{link_to_instance}.html"}

    # Add tag to available_tags if not already present
    if tag not in available_tags:
        available_tags.append(tag)

# %%
from utils.ol_types import OLTypes
olt = OLTypes()
mylist = olt.get_neuron_list(side='both')

# %%
# Get main groups and html-tags
# main_groups_dict, available_tags = fetch_and_organize_neuron_data(neuron_names)

# Define the mapping from abbreviations to full names
full_group_names = {
    'OL_intrinsic': 'Optic Lobe Intrinsic Neurons',
    'OL_connecting': 'Optic Lobe Connecting Neurons',
    'VPN': 'Visual Projection Neurons',
    'VCN': 'Visual Centrifugal Neurons',
    'other': 'Other'
}

color_mapping_groups = {
        'OL_intrinsic': OL_COLOR.OL_TYPES.hex[0], 
        'OL_connecting': OL_COLOR.OL_TYPES.hex[1],  
        'VPN': OL_COLOR.OL_TYPES.hex[2],  
        'VCN': OL_COLOR.OL_TYPES.hex[3],  
        'other': OL_COLOR.OL_TYPES.hex[4],  
    }

# %%
# Fetch meta to the footer
meta, lastDataBaseEdit, formattedDate = get_meta_data()


# %%
def render_and_save_templates(template_name, data_dict, output_filename):
    # Assuming the templates are in the current directory for simplicity
    environment = jinja2.Environment(loader=jinja2.FileSystemLoader('.'))

    # Load the template
    template = environment.get_template(template_name)

    # Render the template with the dynamically passed data
    rendered_template = template.render(**data_dict)

    # Save the rendered template to an HTML file
    with open(output_filename, "w") as file:
        file.write(rendered_template)

output_path = Path(PROJECT_ROOT, 'results',  'html_pages')
# Data for the index page
index_data_dict = {
    'mylist': mylist,
    'full_group_names': full_group_names,
    'meta': meta,
    'formattedDate' : formattedDate,
    'lastDataBaseEdit' : lastDataBaseEdit,
    'color_mapping_groups' : color_mapping_groups
}

render_and_save_templates(
    "cell_types.html.jinja"
  , index_data_dict
  , output_path / "cell_types.html"
)


# %%

# Data for the cover page
cover_data_dict = {
    'available_tags': available_tags, 
    'meta': meta, 
    'lastDataBaseEdit' : lastDataBaseEdit,
    'formattedDate' : formattedDate
}
render_and_save_templates(
    "index.html.jinja"
  , cover_data_dict
  , output_path / "index.html")

# %%
render_and_save_templates(
    "webpages_glossary.html.jinja"
  , {}
  , output_path / "webpages_glossary.html"
)
