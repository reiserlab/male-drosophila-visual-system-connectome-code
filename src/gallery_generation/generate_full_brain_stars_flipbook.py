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
from utils.ol_color import OL_COLOR
from utils.ol_types import OLTypes

c = olc_client.connect(verbose=True)

# %%
olt = OLTypes()
oli_list = olt.get_neuron_list()

# %%
for idx, row in oli_list.reset_index().sample(frac=1).iterrows():
    all_cell_dict = {}
    txt_pos = 0.92

    star_bids = {}
    non_star_bids = {}

    a_bag = NeuronBag(cell_type=row['type'])
    body_id = a_bag.get_body_ids(1)[0]
    if isinstance(row['star_neuron'], int):
        body_id = row['star_neuron']

    sorted_body_ids = a_bag.get_body_ids(a_bag.size)
    non_star_list = [bid for bid in sorted_body_ids.tolist() if bid!=body_id]

    camera_dict = get_rend_params('camera', 'whole_brain')
    scalebar_dict = get_rend_params('scalebar', 'whole_brain')

    match row['main_groups']:
        case 'OL_intrinsic':
            the_directory = 'flipbook_OL_intrinsic'
            the_color = OL_COLOR.OL_IN_SEQ.rgba[1]
            the_title = 'Optic Neuropil Intrinsic Neurons (ONIN)'
        case 'OL_connecting':
            the_directory = 'flipbook_OL_connecting'
            the_color = OL_COLOR.OL_CONN_SEQ.rgba[1]
            the_title = 'Optic Neuropil Connecting Neurons (ONCN)'
        case 'VPN':
            the_directory = 'flipbook_VPN'
            the_color = OL_COLOR.OL_VPN_SEQ.rgba[1]
            the_title = 'Visual Projection Neurons (VPN)'
        case 'VCN':
            the_directory = 'flipbook_VCN'
            the_color = OL_COLOR.OL_VCN_SEQ.rgba[1]
            the_title = 'Visual Centrifugal Neurons (VCN)'
        case 'other':
            the_directory = 'flipbook_other'
            the_color = OL_COLOR.OL_CB_OTHER_SEQ.rgba[1]
            the_title = 'Other Visual Neurons'

    non_star_type = {}
    star_bids = {
        'type': f"{row['type']}"
      , 'body_ids': [body_id]
      , 'body_color': the_color
      , 'text_position': [0.03, txt_pos]
      , 'text_align': 'l'
      , 'number_of_cells': a_bag.size
      , 'slice_width': 0
    }

    non_star_bids =  {
        'type': f"{row['type']}out"
      , 'body_ids': non_star_list
      , 'body_color': [0.8, 0.8, 0.8, 1]
      , 'text_position': [-5, -5]
      , 'text_align': 'l'
      , 'number_of_cells': a_bag.size
      , 'slice_width': 0
   }
    non_star_type['rest'] = non_star_bids
    non_star_type[row['type']] = star_bids

    all_types = {}
    all_types[row['type']] = star_bids

    generate_gallery_json(
        type_of_plot="Full-Brain"
      , description="flipbook"
      , type_or_group=f"{row['type']}_all"
      , title=the_title
      , view='whole_brain'
      , list_of_ids=non_star_type
      , neuropil_color=[]
      , camera=camera_dict
      , slicer={}
      , scalebar=scalebar_dict
      , n_vis={}
      , directory=the_directory
      , template="gallery-descriptions.json.jinja"
    )

    generate_gallery_json(
        type_of_plot="Full-Brain"
      , description="flipbook"
      , type_or_group=f"{row['type']}"
      , title=the_title
      , view='whole_brain'
      , list_of_ids=all_types
      , neuropil_color=[]
      , camera=camera_dict
      , slicer={}
      , scalebar=scalebar_dict
      , n_vis={}
      , directory=the_directory
      , template="gallery-descriptions.json.jinja"
    )
    print(f"Json generation done for {row['type']}")

    # Stop if number of iterations exceed the environment variable `GALLERY_EXAMPLES`
    stop_after = os.environ.get('GALLERY_EXAMPLES')
    if stop_after:
        if stop_after := int(stop_after):
            if stop_after <= idx + 1:
                break
        else:
            break

# %%
