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
from utils.ol_types import OLTypes

c = olc_client.connect(verbose=True)

# %%
olt = OLTypes()
oli_list = olt.get_neuron_list()
aME_list = ['HBeyelet', 'aMe6a', 'aMe6b', 'aMe6c', 'aMe3', 'aMe1', 'aMe2', '5thsLNv_LNd6', 'CL125', 's-LNv'
  , 'DN1a', 'SLP249', 'SLP250', 'SMP217', 'aMe22', 'aMe23', 'l-LNv', 'aMe15', 'MeVPaMe1', 'MeVPaMe2'
  , 'aMe5', 'aMe10', 'aMe12', 'MeVP63', 'Lat1', 'Lat2', 'aMe4']

groups_by_types = oli_list[oli_list['type']\
    .isin(aME_list)]\
    .set_index('type')\
    .reindex(aME_list)\
    .reset_index()

# %%
neuropil_color = []

for idx, row in groups_by_types.reset_index().sample(frac=1).iterrows():
    vpn_dict = {}
    txt_pos = 0.92

    body_id_dict = {}

    a_bag = NeuronBag(cell_type=row['type'], side='R-dominant')

    sorted_body_ids = a_bag.get_body_ids(a_bag.size)
    body_id_list = sorted_body_ids.tolist()

    camera_dict = get_rend_params('camera', row['fb_view'])
    slicer_dict = {}
    scalebar_dict = get_rend_params('scalebar', row['fb_view'])

    group_dict = {}
    body_id_dict = {
        'type': row['type']
      , 'body_ids': body_id_list
      , 'body_color': [0,0,0,1]
      , 'text_position': [0.03, txt_pos]
      , 'text_align': 'l'
      , 'number_of_cells': len(sorted_body_ids)
      , 'slice_width': 0
    }

    roi_dict = {
      "AME(R)": {
          "flat": [46000, 30000, 27500]
        , "rotation": [-180, 0, -2]
        , "color":[0.76,0.80,0.92,1.0]
      }
    }

    group_dict[row['type']] = body_id_dict

    generate_gallery_json(
        type_of_plot="Full-Brain"
      , description = "type"
      , type_or_group=row['type']
      , title=""
      , view='whole_brain'
      , list_of_ids=group_dict
      , list_of_rois=roi_dict
      , neuropil_color=neuropil_color
      , camera=camera_dict
      , slicer=slicer_dict
      , scalebar=scalebar_dict
      , n_vis={}
      , directory='group-of-one'
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
