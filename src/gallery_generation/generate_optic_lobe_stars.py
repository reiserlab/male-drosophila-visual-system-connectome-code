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

# %% [markdown]
# ## Fill and save the template

# %%
import sys
import os
import warnings
import math
from pathlib import Path

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
oli_list = olt.get_neuron_list(side='both')

# %%
"""
Generate  JSON files for OL neurons determined by Aljoscha using "gallery-descriptions.json.jinja" 
template and function from `utils/gallery_filler.py`
"""

neuropil_color = [
    OL_COLOR.OL_NEUROPIL_LAYERS.rgba[3], OL_COLOR.OL_NEUROPIL_LAYERS.rgba[4]
  , OL_COLOR.OL_NEUROPIL_LAYERS.rgba[5], OL_COLOR.OL_NEUROPIL_LAYERS.rgba[6]
  , OL_COLOR.OL_NEUROPIL_LAYERS.rgba[7], OL_COLOR.OL_NEUROPIL_LAYERS.rgba[8]
  , OL_COLOR.OL_NEUROPIL_LAYERS.rgba[9], OL_COLOR.OL_NEUROPIL_LAYERS.rgba[10]
  , OL_COLOR.OL_NEUROPIL_LAYERS.rgba[11]
]

iter_counter = 0
for idx, row in oli_list.reset_index().sample(frac=1).iterrows():

    # celltypes with slice_width empty
    if math.isnan(row['slice_width']):
        warnings.warn(f"No slice width, skipping {row['type']}_{row['hemisphere']}")
        continue

    iter_counter += 1
    txt_pos = 0.85
    
    # if the type has more than one instance, include the hemisphere in the gallery name
    gallery_name = (
        f"{row['type']} ({row['hemisphere']})"
        if olt.is_bilateral(type_str=row["type"])
        else row["type"]
    )
    
    a_bag = NeuronBag(cell_type=row['type'], side=row["hemisphere"])

    body_id = a_bag.first_item
    if isinstance(row['star_neuron'], int):
        body_id = row['star_neuron']

    camera_dict = get_rend_params('camera', row['ol_view'])
    slicer_dict = get_rend_params('slice', row['ol_view'])
    scalebar_dict = get_rend_params('scalebar', row['ol_view'])
   
    gallery_dict = {}
    body_id_dict = {
        'type': gallery_name
      , 'body_ids': [body_id]
      , 'body_color': [0.2,0.2,0.2,1]
      , 'text_position': [0.03, txt_pos]
      , 'text_align': 'l'
      , 'number_of_cells': a_bag.size
      , 'slice_width': row['slice_width'] 
    }
   
    gallery_dict[row['type']] = body_id_dict
    
    if row['main_groups'] in ['OL_intrinsic', 'OL_connecting']:
        the_directory='ol_gallery_plots' 
    else: 
        the_directory='vpn_vcn_gallery_plots'  

    generate_gallery_json(
        type_of_plot="Optic-Lobe"
      , description = "Gallery"
      , type_or_group=f"{row['type']}_{row['hemisphere']}"
      , title=""
      , list_of_ids=gallery_dict
      , neuropil_color=neuropil_color
      , camera=camera_dict
      , slicer=slicer_dict
      , scalebar=scalebar_dict
      , view=row['ol_view']
      , n_vis={}
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
