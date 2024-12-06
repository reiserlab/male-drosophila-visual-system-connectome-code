# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: default
#     language: python
#     name: python3
# ---

# %%
# Import optic lobe components
import sys
from pathlib import Path
import jinja2

from dotenv import load_dotenv, find_dotenv
load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath('src')))


from queries.completeness import fetch_ol_types_and_instances
from html_pages.webpage_functions import \
    get_meta_data\
  , get_last_database_edit\
  , get_formatted_now\
  , render_and_save_templates
from utils.ol_color import OL_COLOR

from utils import olc_client
c = olc_client.connect(verbose=True)


# %%
# Fetch unique instances and their types
neuron_names = fetch_ol_types_and_instances(side='both')

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
# Define the mapping from abbreviations to full names
full_group_names = {
    'OL_intrinsic': 'Optic Neuropil Intrinsic Neurons'
  , 'OL_connecting': 'Optic Neuropil Connecting Neurons'
  , 'VPN': 'Visual Projection Neurons'
  , 'VCN': 'Visual Centrifugal Neurons'
  , 'other': 'Other'
}

color_mapping_groups = {
    'OL_intrinsic': OL_COLOR.OL_TYPES.hex[0] 
  , 'OL_connecting': OL_COLOR.OL_TYPES.hex[1]
  , 'VPN': OL_COLOR.OL_TYPES.hex[2]
  , 'VCN': OL_COLOR.OL_TYPES.hex[3]
  , 'other': OL_COLOR.OL_TYPES.hex[4]
}

# %%
# Fetch meta to the footer
meta = get_meta_data()
lastDataBaseEdit = get_last_database_edit()
formattedDate = get_formatted_now()

# %%
output_path = PROJECT_ROOT / 'results' / 'html_pages'

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
    'available_tags': available_tags
  , 'meta': meta
  , 'lastDataBaseEdit' : lastDataBaseEdit
  , 'formattedDate' : formattedDate
}
render_and_save_templates(
    "index.html.jinja"
  , cover_data_dict
  , output_path / "index.html"
)

# %%
render_and_save_templates(
    "webpages_glossary.html.jinja"
  , {}
  , output_path / "webpages_glossary.html"
)
