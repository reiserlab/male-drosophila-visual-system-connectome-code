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

# %% [markdown]
# ## Fill and save the template

# %%
import sys
import os
import math
from pathlib import Path
import pandas as pd

from dotenv import load_dotenv, find_dotenv
load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath('src')))
print(f"Project root directory: {PROJECT_ROOT}")

from utils.neuron_bag import NeuronBag
from utils.gallery_filler import generate_gallery_json
from utils.rend_params import get_rend_params
from utils import olc_client
from utils.ol_types import OLTypes
from utils.ol_color import OL_COLOR

c = olc_client.connect(verbose=True)

# %%
olt = OLTypes()
oli_list = olt.get_neuron_list()

# %%
#excludes uncurated neurons
oli_list = oli_list[oli_list['star_neuron'].notnull()]
oli_list = oli_list[oli_list['Slice_width'].notnull()]

# %%
"""
Generate  JSON files for OL neurons determined by Aljoscha using "gallery-descriptions.json.jinja" 
template and function from `utils/gallery_filler.py`
"""

# TODO: fix color scheme with 
neuropil_color = [
    OL_COLOR.OL_NEUROPIL_LAYERS.rgba[3], OL_COLOR.OL_NEUROPIL_LAYERS.rgba[4]
  , OL_COLOR.OL_NEUROPIL_LAYERS.rgba[5], OL_COLOR.OL_NEUROPIL_LAYERS.rgba[6]
  , OL_COLOR.OL_NEUROPIL_LAYERS.rgba[7], OL_COLOR.OL_NEUROPIL_LAYERS.rgba[8]
  , OL_COLOR.OL_NEUROPIL_LAYERS.rgba[9], OL_COLOR.OL_NEUROPIL_LAYERS.rgba[10]
  , OL_COLOR.OL_NEUROPIL_LAYERS.rgba[11]
]

iter_counter = 0
for idx, row in oli_list.reset_index().sample(frac=1).iterrows():

    iter_counter += 1
    txt_pos = 0.95
    
    a_bag = NeuronBag(cell_type=row['type'])

    body_id = a_bag.first_item
    if isinstance(row['star_neuron'], int):
        body_id = row['star_neuron']

    camera_dict = get_rend_params('camera',row['ol_view'])
    slicer_dict = get_rend_params('slice',row['ol_view'])
   
    gallery_dict = {}
    body_id_dict = {
        'type':row['type']
      , 'body_ids': [body_id]
      , 'body_color': [0,0,0,1]
      , 'text_position': [0.03, txt_pos]
      , 'text_align': 'l'
      , 'number_of_cells': a_bag.size
      , 'slice_width':row['Slice_width'] 
    }
    
    gallery_dict[row['type']] = body_id_dict
    
    if row['main_groups']=='OL_intrinsic' or row['main_groups']=='OL_connecting' or row['main_groups']=='OL_intrinsic*':
        the_directory='ol_gallery_plots' 
    else: 
        the_directory='vpn_vcn_gallery_plots'  

    generate_gallery_json(
        type_of_plot="Optic-Lobe"
      , description = "OLi"
      , type_or_group=row['type']
      , title=""
      , list_of_ids= gallery_dict
      , neuropil_color=neuropil_color
      , camera=camera_dict
      , slicer=slicer_dict
      , view=row['ol_view']
      , directory=the_directory
      , template="gallery-descriptions.json.jinja"
    )
    print(f"Json generation done for {row['type']}")
    
    # Stop if number of iterations exceed the environment variable `GALLERY_EXAMPLES`
    stop_after = os.environ.get('GALLERY_EXAMPLES')
    if stop_after:
        if stop_after := int(stop_after):
            if stop_after <= iter_counter:
                break
        else:
            break

# %%
