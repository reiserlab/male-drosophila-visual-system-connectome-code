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
groups_by_types = olt.get_neuron_list(primary_classification=['VPN', 'VCN', 'other'])

# %%
"""
Generate  JSON files for VPN, VCN, and other groups determined by Aljoscha using "gallery-descriptions.json.jinja" template 
and function from "utils/gallery_filler.py
"""
neuropil_color=[]
color_list = OL_COLOR.OL_TYPES.rgba \
    + OL_COLOR.OL_DARK_TYPES.rgba \
    + OL_COLOR.OL_LIGHT_TYPES.rgba
#order = [0,6,12, 3,9,10, 1,7,13, 4,5,11, 2,8,14] # 3 colors
#order = [0,6,2,8,4,5,1,7,3,9] # 2 colors
order = [0,6,2,13,4,5,1,7,3,9,12,10] # 2 colors
color_list[:] = [color_list[idx] for idx in order]

iter_counter = 0
for name, n_types_in_group in groups_by_types.groupby('figure_group'):
    iter_counter += 1
    group_dict = {}
    txt_pos_y = 0.92

    if n_types_in_group[n_types_in_group['fb_view'].isin(['whole_brain'])].empty==True:
        the_view='half_brain'
        txt_pos_x=0.03
        txt_align = 'l'
    else:
        the_view='whole_brain'
        txt_pos_x=0.97
        txt_align = 'r'

    camera_dict=get_rend_params('camera',the_view)
    slicer_dict ={}

    for idx, row in n_types_in_group.sort_values(['type'],  key=lambda col: col.str.lower()).reset_index().iterrows():
        txt_pos_y = txt_pos_y - 0.06

        body_id_dict = {}
         
        a_bag = NeuronBag(cell_type=row['type'], side='R-dominant')

        sorted_body_ids = a_bag.get_body_ids(a_bag.size)
        body_id_list=sorted_body_ids.tolist()
        
        gallery_dict = {}
        body_id_dict = {
            'type':row['type']
          , 'body_ids': body_id_list
          , 'body_color': color_list[idx % len(color_list)]
          , 'text_position': [txt_pos_x, txt_pos_y]
          , 'text_align': txt_align
          , 'number_of_cells': len(sorted_body_ids)  
        }
  
        group_dict[row['type']] = body_id_dict   
        
        if row['main_groups']=='VPN':
             the_directory='vpn_group_plots' 
        elif row['main_groups']=='VCN':  
             the_directory='vcn_group_plots'
        else:  
            the_directory='other_neuron_group_plots'
        
        generate_gallery_json(
            type_of_plot="Full-Brain"
          , description = "Group"
          , type_or_group=row['figure_group']
          , title=row['figure_group'].replace("_"," ")
          , view=the_view
          , list_of_ids=group_dict
          , neuropil_color=neuropil_color
          , camera=camera_dict
          , slicer=slicer_dict
          , directory=the_directory
          , template="gallery-descriptions.json.jinja"
        )
    print(f"Json generation done for {row['figure_group']}")

    # Stop if number of iterations exceed the environment variable `GALLERY_EXAMPLES`
    stop_after = os.environ.get('GALLERY_EXAMPLES')
    if stop_after:
        if stop_after := int(stop_after):
            if stop_after <= iter_counter:
                break
        else:
            break

# %%
