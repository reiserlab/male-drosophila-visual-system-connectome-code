# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: default
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Fill and save the template

# %%
import os
from dotenv import load_dotenv
from madvisc.utils.neuron_bag import NeuronBag
from madvisc.utils.gallery_filler import generate_gallery_json
from madvisc.utils.rend_params import get_rend_params
from madvisc.utils import olc_client
from madvisc.utils.ol_types import OLTypes

load_dotenv()
c = olc_client.connect(verbose=True)

# %%
olt = OLTypes()
oli_list = olt.get_neuron_list()

# %%
# Configuraiton of the plot

r78_lists=[
    {
        'types': ['R7d', 'Cm-DRA', 'Dm-DRA1', 'MeTu2a', 'MeVP15', 'MeVP31']
      , 'color': [0.7, 0.23, 0.93, 1]
    }
  , {
        'types': ['R8d', 'Dm-DRA2']
      , 'color': [0, 0.47, 0.93, 1]
    }
  , {
        'types': ['Cm22', 'MeTu2b', 'MeVP39', 'Cm18']
      , 'color': [0, 0, 0, 1]
    }
  , {
        'types': ['MeVPMe10']
      , 'color': [[0.4, 0.2, 0.2, 1], [0.6, 0.6, 0.7, 1]]
    }
]

# %%
"""
GROUPS OF ONE TYPE

Color coded dorsal rim neuron types plotted in the "Group-of-one" style.
Used for Figure ED 11.

Generate JSON files for making Groups of all neurons by type
"""

for subgroup in r78_lists:
    one_off_list = oli_list[oli_list['type'].isin(subgroup['types'])]
    the_body_color = subgroup['color']
    neuropil_color = []

    for iter_counter, row in one_off_list.reset_index().sample(frac=1).iterrows():
        a_bag = NeuronBag(cell_type=row['type'], side='R-dominant')

        sorted_body_ids = a_bag.get_body_ids(a_bag.size)
        body_id_list = sorted_body_ids.tolist()

        camera_dict = get_rend_params('camera', row['fb_view'])
        scalebar_dict = get_rend_params('scalebar', row['fb_view'])

        group_dict = {}
        body_id_dict = {
            'type': row['type']
          , 'body_ids': body_id_list
          , 'body_color': the_body_color
          , 'text_position': [0.03, 0.92]
          , 'text_align': 'l'
          , 'number_of_cells': len(sorted_body_ids)
          , 'slice_width': 0
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
          , slicer={}
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
                if stop_after <= iter_counter + 1:
                    break
            else:
                break
