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
from utils.ol_color import OL_COLOR
from utils.ol_types import OLTypes

c = olc_client.connect(verbose=True)

# %%
olt = OLTypes()
oli_list = olt.get_neuron_list(primary_classification=['VPN','VCN','other'])

# %%
#excludes uncurated neurons
oli_list = oli_list[oli_list['star_neuron'].notnull()]
groups_by_types = oli_list[oli_list['Slice_width'].notnull()]

# %%
"""
GROUPS OF ONE TYPE

Only VPNs and VCNs are included in this type of plot because it is not a useful
plot for optic lobe intrinsic neurons. For the most part, when optic lobe intrinsic neurons are 
plotted in this way, they simply fill one or more of the optic lobe neuropils completely. 
The primary purpose of this type of plot was so that people could see how individual neuron types project to the central brain.

Generate JSON files for making Groups of all neurons by type
"""
neuropil_color = []

iter_counter = 0
for idx, row in groups_by_types.reset_index().sample(frac=1).iterrows():
    iter_counter += 1
    vpn_dict = {}
    txt_pos = 0.92

    body_id_dict = {}

    a_bag = NeuronBag(cell_type=row['type'], side='R-dominant')

    sorted_body_ids = a_bag.get_body_ids(a_bag.size)
    body_id_list=sorted_body_ids.tolist()

    camera_dict = get_rend_params('camera',row['fb_view'])
    slicer_dict = {}

    group_dict = {}
    body_id_dict = {
        'type':row['type']
      , 'body_ids': body_id_list
      , 'body_color': [0,0,0,1]
      , 'text_position': [0.03, txt_pos]
      , 'text_align': 'l'
      , 'number_of_cells': len(sorted_body_ids) 
    }
  
    group_dict[row['type']] = body_id_dict   

    generate_gallery_json(
        type_of_plot="Full-Brain"
      , description = "type"
      , type_or_group=row['type']
      , title=""
      , view='whole_brain'
      , list_of_ids=group_dict
      , neuropil_color=neuropil_color
      , camera=camera_dict
      , slicer=slicer_dict
      , directory='group-of-one'
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
